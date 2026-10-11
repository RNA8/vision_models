#!/usr/bin/env python3
"""
finetune.py — fine-tune a pretrained timm model (or one of our own runs) on a GPU-resident dataset,
optionally with knowledge distillation from a large teacher. Edge-model track: find the smallest
model that gets near-SOTA on CIFAR-100 etc. using external (ImageNet) pretraining.

Same machinery as train_vit.py (imported from it): whole uint8 dataset on the GPU, GPU crop/flip,
RandAugment, MixUp/CutMix, label smoothing, bf16 autocast, fused AdamW, warmup+cosine, torch.compile,
checkpoint/resume, wandb, results.json. New here:
  * --model <timm name>: timm.create_model(name, pretrained=True, num_classes=N, img_size=R) (img_size /
    drop_path_rate are dropped for archs that do not take them). Archs whose pretrained weights do not
    load at img_size R (resolution-dependent attention-bias tables: EfficientFormerV2, EfficientViT-M)
    are built at R and get the 224 weights with the bias tables bicubic-resized.
  * --img-size R: augmentation runs at the native resolution (32 for CIFAR), then the batch is upsampled
    on the GPU (F.interpolate bilinear, antialias) to R and normalized with the model's own
    pretrained_cfg mean/std.
  * --lr-decay: layer-wise lr decay (timm param_groups_layer_decay / group_matcher).
  * --init-ckpt runs/<name>: start from one of our train_vit.py / finetune.py runs (architecture from its
    results.json, weights from model.pt; head re-initialised, abs pos-embed resampled to the new grid).
  * --teacher <run dir | timm name> --kd {soft,hard} --kd-alpha --kd-tau [--teacher-img-size]: teacher
    logits on the same augmented (mixed) batch, no_grad, bf16. soft = Hinton KL * tau^2; hard = DeiT-style
    CE on the teacher argmax. loss = (1 - alpha) * CE(targets) + alpha * KD.
  * --ema D: EMA of weights (timm ModelEmaV3), evaluated alongside the raw weights, saved as model_ema.pt.

results.json also stores `model_spec` (how to rebuild the network) so a finished run can be used as a
--teacher, as an --init-ckpt, or by edge_bench.py.

Examples:
  python finetune.py --model deit_tiny_patch16_224.fb_in1k --img-size 224 --epochs 2 --fake-data   # smoke
  python finetune.py --model vit_base_patch16_224.augreg2_in21k_ft_in1k --dataset cifar100 --img-size 224 \
      --epochs 30 --lr 2.5e-4 --lr-decay 0.75 --run-name ft_c100_vitb_224
  python finetune.py --model tiny_vit_5m_224.dist_in22k_ft_in1k --dataset cifar100 --img-size 128 \
      --teacher runs/ft_c100_vitb_224 --teacher-img-size 224 --kd soft --kd-alpha 0.5 --kd-tau 2
"""
import argparse
import importlib
import json
import math
import os
import re
import time
import uuid
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
from train_vit import (load_data, mix_batch, rand_augment, random_crop_flip, random_erasing,  # noqa: E402
                       make_schedule, smooth_one_hot)


# --------------------------------------------------------------------------- args
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Fine-tune / distill pretrained models on a GPU-resident dataset")
    p.add_argument("--model", default=None,
                   help="timm model name (pretrained weights). Not needed with --init-ckpt")
    p.add_argument("--init-ckpt", default=None,
                   help="start from our run: runs/<name> (or .../model.pt); head reset, pos-embed resampled")
    p.add_argument("--no-pretrained", action="store_true", help="timm model from scratch (control)")
    p.add_argument("--dataset", default="cifar100", help="module name in vit_data/")
    p.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    p.add_argument("--train-subset", type=int, default=0, help="use only the first N train images (tests)")
    p.add_argument("--test-subset", type=int, default=0, help="use only the first N test images (tests)")
    p.add_argument("--img-size", type=int, default=224, help="model input resolution (GPU upsampling)")
    p.add_argument("--patch-size", type=int, default=0,
                   help="--init-ckpt only (our ViTs): switch the patch size (FlexiViT): PI-resize the patch "
                        "embedding, bilinear-resample the pos-embed to the new grid; 0 = keep the ckpt's")
    p.add_argument("--interp", default="bilinear", choices=["bilinear", "bicubic"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4, help="peak LR (no automatic batch-size scaling)")
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--lr-decay", type=float, default=0.0,
                   help="layer-wise lr decay (e.g. 0.65-0.85); 0 = off. Uses timm's group_matcher: deep "
                        "hierarchical/conv nets get many groups, so their lowest scale can get tiny (see --lr-decay-min)")
    p.add_argument("--lr-decay-min", type=float, default=0.0, help="floor on the layer-decay lr scale")
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--mixup", type=float, default=0.8, help="MixUp alpha; 0 disables")
    p.add_argument("--cutmix", type=float, default=1.0, help="CutMix alpha; 0 disables")
    p.add_argument("--mix-prob", type=float, default=1.0)
    p.add_argument("--mix-off-epoch", type=int, default=0)
    p.add_argument("--reprob", type=float, default=0.0, help="random erasing prob. (at --img-size)")
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--no-aug", action="store_true", help="disable random crop + flip")
    p.add_argument("--randaug", action="store_true")
    p.add_argument("--randaug-n", type=int, default=2)
    p.add_argument("--randaug-m", type=float, default=9)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay of weights (e.g. 0.9998); 0 = off")
    p.add_argument("--ema-warmup", action="store_true", help="timm EMA warmup (decay ramps up early on)")
    g = p.add_argument_group("knowledge distillation")
    g.add_argument("--teacher", default=None, help="run dir (finetune.py/train_vit.py) or timm name with N classes")
    g.add_argument("--teacher-img-size", type=int, default=0, help="0 = the teacher's own fine-tune size")
    g.add_argument("--kd", default="soft", choices=["soft", "hard"])
    g.add_argument("--kd-alpha", type=float, default=0.5, help="weight of the KD term")
    g.add_argument("--kd-tau", type=float, default=1.0, help="softmax temperature (soft KD)")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-bs", type=int, default=500)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--cudagraphs", action="store_true", help="torch.compile mode=reduce-overhead")
    p.add_argument("--fake-data", action="store_true", help="random tensors, for smoke tests")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-entity", default="softeye")
    p.add_argument("--wandb-project", default="vision_transformers")
    p.add_argument("--run-name", default=None)
    p.add_argument("--out-dir", default=os.path.join(HERE, "runs"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every-steps", type=int, default=0)
    p.add_argument("--ckpt-every", type=int, default=5)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--keep-ckpt", action="store_true")
    p.add_argument("--max-steps", type=int, default=0, help="stop each epoch after N steps (smoke tests)")
    args = p.parse_args(argv)
    if not args.model and not args.init_ckpt:
        p.error("need --model or --init-ckpt")
    return args


# --------------------------------------------------------------------------- model building
def _resize_bias_tables(sd, model):
    """Copy `sd` into `model`, bicubic-resizing tensors whose shape differs only because they are
    (heads, n*n) / (n*n, heads) relative-position bias tables of a different window/resolution.
    Non-persistent index buffers are the model's own. Returns the list of resized keys."""
    own = model.state_dict()
    out, resized = {}, []
    for k, v in sd.items():
        if k not in own:
            continue
        t = own[k]
        if v.shape == t.shape:
            out[k] = v
            continue
        if v.ndim == 2 and t.ndim == 2 and v.is_floating_point():
            heads_first = v.shape[0] == t.shape[0]
            a, b = (v, t) if heads_first else (v.T, t.T)
            n0, n1 = math.isqrt(a.shape[1]), math.isqrt(b.shape[1])
            if n0 * n0 == a.shape[1] and n1 * n1 == b.shape[1]:
                r = F.interpolate(a.float().reshape(1, a.shape[0], n0, n0), size=(n1, n1), mode="bicubic",
                                  align_corners=False).reshape(a.shape[0], n1 * n1).to(v.dtype)
                out[k] = r if heads_first else r.T
                resized.append(k)
                continue
        raise RuntimeError(f"cannot adapt pretrained {k}: {tuple(v.shape)} -> {tuple(t.shape)}")
    missing = [k for k in own if k not in out and not k.endswith("attention_bias_idxs")]
    model.load_state_dict(out, strict=False)
    if missing:
        print(f"note: {len(missing)} keys keep their init: {missing[:6]}")
    return resized


def create_timm(name, num_classes, img_size, drop_path=0.0, pretrained=True, check=True):
    """timm.create_model with img_size/drop_path_rate, dropping kwargs an arch does not accept.
    Returns (model, create_kwargs) where create_kwargs rebuild the same shapes with pretrained=False."""
    import timm

    kw = dict(num_classes=num_classes, img_size=img_size)
    if drop_path > 0:
        kw["drop_path_rate"] = drop_path
    while True:
        try:
            m = timm.create_model(name, pretrained=pretrained, **kw)
            break
        except TypeError as e:
            bad = re.search(r"unexpected keyword argument '(\w+)'", str(e))
            if bad and bad.group(1) in kw and bad.group(1) != "num_classes":
                kw.pop(bad.group(1))
                continue
            raise
        except RuntimeError as e:  # pretrained state_dict does not fit at this img_size
            if not (pretrained and "size mismatch" in str(e) and "img_size" in kw):
                raise
            m = timm.create_model(name, pretrained=False, **kw)
            src = timm.create_model(name, pretrained=True, **{k: v for k, v in kw.items() if k != "img_size"})
            resized = _resize_bias_tables(src.state_dict(), m)
            print(f"note: {name} built at img_size={img_size}; resized {len(resized)} bias tables from 224")
            del src
            break
    if check:  # catch archs that ignore img_size and cannot run at this resolution
        m.eval()
        with torch.no_grad():
            m(torch.zeros(1, 3, img_size, img_size))
        m.train()
    return m, kw


def _read_run(path):
    """runs/<name> or runs/<name>/model.pt -> (run_dir, results dict)."""
    d = path if os.path.isdir(path) else os.path.dirname(path)
    with open(os.path.join(d, "results.json")) as f:
        return d, json.load(f)


def _run_weights(d, res, prefer_ema=True):
    pt = os.path.join(d, "model.pt")
    ema_pt = os.path.join(d, "model_ema.pt")
    if prefer_ema and os.path.exists(ema_pt) and (res.get("final_ema_val_acc") or -1) > (res.get("final_val_acc") or 0):
        pt = ema_pt
    return torch.load(pt, map_location="cpu", weights_only=True), pt


def spec_from_run(d, res):
    """model_spec of a finished run (finetune.py stores it; train_vit.py runs are reconstructed)."""
    if "model_spec" in res:
        return res["model_spec"]
    a = res["args"]  # train_vit.py run
    ds = importlib.import_module(f"vit_data.{a['dataset']}")
    if a.get("fake_data"):
        img = 32
    else:
        img = {"imagenet64": 64, "tiny_imagenet": 64}.get(a["dataset"], 32)
    return {"kind": "ours", "train_args": a, "img_size": img, "num_classes": ds.NUM_CLASSES,
            "mean": list(ds.MEAN), "std": list(ds.STD)}


def build_from_spec(spec, num_classes=None, img_size=None, drop_path=0.0):
    """Rebuild a network (random init) from a model_spec, optionally with a new head / input size."""
    num_classes = num_classes or spec["num_classes"]
    img_size = img_size or spec["img_size"]
    if spec["kind"] == "timm":
        kw = dict(spec["create_kwargs"], num_classes=num_classes)
        if "img_size" in kw:
            kw["img_size"] = img_size
        if "drop_path_rate" in kw or drop_path > 0:
            kw["drop_path_rate"] = drop_path
        import timm

        try:
            return timm.create_model(spec["name"], pretrained=False, **kw)
        except TypeError:
            kw.pop("drop_path_rate", None)
            return timm.create_model(spec["name"], pretrained=False, **kw)
    from train_vit import build_model

    a = Namespace(**spec["train_args"])
    a.drop_path = drop_path
    return build_model(a, num_classes, img_size)


def adapt_state_dict(sd, model):
    """Load `sd` into `model` for transfer: drop tensors that no longer fit (classifier head), resample
    absolute pos-embeds to the new token grid. Returns the dropped keys."""
    from timm.layers import resample_abs_pos_embed

    own = model.state_dict()
    out, dropped = {}, []
    for k, v in sd.items():
        if k not in own:
            dropped.append(k)
            continue
        if v.shape != own[k].shape:
            if k.endswith("patch_embed.proj.weight") and v.ndim == 4:  # FlexiViT PI-resize
                from timm.layers import resample_patch_embed

                v = resample_patch_embed(v, list(model.patch_embed.patch_size), interpolation="bilinear")
            elif k.endswith("pos_embed") and v.ndim == 3:
                npt = 0 if getattr(model, "no_embed_class", False) else getattr(model, "num_prefix_tokens", 1)
                gs = model.patch_embed.grid_size
                v = resample_abs_pos_embed(v, new_size=list(gs), num_prefix_tokens=npt)
            else:
                dropped.append(k)
                continue
        out[k] = v
    missing = model.load_state_dict(out, strict=False).missing_keys
    print(f"init-ckpt: loaded {len(out)} tensors; dropped {dropped}; re-initialised {missing}")
    return dropped


# timm archs whose freshly initialised classifier must be zeroed when fine-tuning from pretrained weights.
# MobileNetV4-Hybrid-Medium's pre-logit features are large (rms ~2.5, max ~65 on CIFAR at 128/224 px vs ~0.5 for
# MNv4-Conv-Small), so timm's default head init (uniform, std ~0.058) gives logits with std ~5 and an initial loss
# ~13; the backward pass then explodes into the backbone (grad norm ~7e4, mostly conv_stem) and the pretrained
# features are wrecked in the first steps. ec100_mnv4_hm_128 collapsed to chance (4.4%) in compiled bf16, and on
# a 20k subset at lr 5e-4 compiled bf16, compiled fp32, eager bf16 and compiled bf16 with MultiQueryAttention2d
# in fp32 ALL collapsed (val 1.5-2.5%), so it is not a precision problem. Zero head -> initial loss ln(100),
# grad norm 8. Prefix-matched, so other archs keep the head init the rest of the sweep used.
ZERO_HEAD = ("mobilenetv4_hybrid_",)


def load_model(ref, num_classes=None, img_size=None, pretrained=True, drop_path=0.0, transfer=False, patch_size=0):
    """Model reference -> (model, spec). `ref` is a run dir (or its model.pt) or a timm name.
    transfer=True: new head/img_size (fine-tuning start); else the run's exact network (teacher/bench)."""
    if os.path.exists(ref):
        d, res = _read_run(ref)
        spec = dict(spec_from_run(d, res))
        sd, pt = _run_weights(d, res, prefer_ema=not transfer)
        if transfer:
            if patch_size:
                spec["train_args"] = dict(spec["train_args"], patch_size=patch_size)
            m = build_from_spec(spec, num_classes, img_size, drop_path)
            adapt_state_dict(sd, m)
            spec.update(num_classes=num_classes or spec["num_classes"], img_size=img_size or spec["img_size"])
        else:
            m = build_from_spec(spec, drop_path=drop_path)
            m.load_state_dict(sd)
        spec["source"] = os.path.relpath(pt, HERE)
        print(f"loaded {pt}")
        return m, spec
    m, kw = create_timm(ref, num_classes or 1000, img_size or 224, drop_path, pretrained=pretrained)
    if pretrained and (num_classes or 1000) != 1000 and ref.startswith(ZERO_HEAD):
        head = m.get_classifier()
        torch.nn.init.zeros_(head.weight)
        if head.bias is not None:
            torch.nn.init.zeros_(head.bias)
        print(f"note: zero-initialised the new {num_classes}-class head of {ref} (see ZERO_HEAD)")
    cfg = m.pretrained_cfg
    spec = {"kind": "timm", "name": ref, "create_kwargs": {k: v for k, v in kw.items() if k != "drop_path_rate"},
            "img_size": img_size or 224, "num_classes": num_classes or 1000,
            "mean": list(cfg.get("mean", (0.485, 0.456, 0.406))), "std": list(cfg.get("std", (0.229, 0.224, 0.225)))}
    return m, spec


def num_params(m):
    return sum(p.numel() for p in m.parameters())


# Modules run in fp32 (autocast off) during bf16 training. EfficientFormerV2's attention (q@k + bias ->
# talking-head conv -> softmax -> talking-head conv) is unstable in bf16 under torch.compile: fine-tuning
# efficientformerv2_s0 on CIFAR-100 collapsed to chance (loss ~ln 100, val ~5%) at 128 AND 224 px, while eager
# bf16, compiled fp32, and compiled bf16 with only these modules in fp32 all train normally (verified on a
# 20k-image subset at lr 1e-3; BN-in-fp32 alone did not help). Cost: 5 low-resolution attention layers.
FP32_MODULES = {"Attention2d", "Attention2dDownsample"}  # timm.models.efficientformer_v2
_FP32_CLS = {}


def fp32_modules(model, names=FP32_MODULES):
    """Swap the class of modules named in `names` to a subclass whose forward runs in fp32 with autocast
    disabled (a class swap, not a patched bound method, so deepcopy for EMA keeps it). Returns the count."""
    n = 0
    for mod in model.modules():
        cls = type(mod)
        if cls.__name__ in names and not getattr(cls, "_fp32", False):
            if cls not in _FP32_CLS:
                def forward(self, *a, _base=cls, **k):
                    with torch.autocast("cuda", enabled=False), torch.autocast("cpu", enabled=False):
                        return _base.forward(self, *[t.float() if torch.is_tensor(t) else t for t in a], **k)
                _FP32_CLS[cls] = type(cls.__name__, (cls,), {"forward": forward, "_fp32": True,
                                                             "__module__": cls.__module__})
            mod.__class__ = _FP32_CLS[cls]
            n += 1
    return n


# --------------------------------------------------------------------------- optim
def make_optimizer(model, args, fused):
    """AdamW, no weight decay on 1-d params / no_weight_decay(); optional layer-wise lr decay. Each group's
    lr is pre-multiplied by its lr_scale so the LambdaLR warmup+cosine (train_vit.make_schedule) applies."""
    from timm.optim._param_groups import param_groups_layer_decay, param_groups_weight_decay

    skip = set(getattr(model, "no_weight_decay", lambda: set())())
    if args.lr_decay > 0:
        try:
            groups = param_groups_layer_decay(model, args.weight_decay, skip, layer_decay=args.lr_decay,
                                              min_scale=args.lr_decay_min)
        except Exception as e:  # no group_matcher on this model
            print(f"WARNING: layer decay unsupported ({type(e).__name__}: {e}); using a flat lr")
            groups = param_groups_weight_decay(model, args.weight_decay, skip)
    else:
        groups = param_groups_weight_decay(model, args.weight_decay, skip)
    for g in groups:
        g["lr"] = args.lr * g.get("lr_scale", 1.0)
    if args.lr_decay > 0:
        scales = sorted({round(g.get("lr_scale", 1.0), 4) for g in groups})
        print(f"layer-wise lr decay {args.lr_decay}: {len(scales)} distinct scales, min {scales[0]} max {scales[-1]}")
    return torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.999), fused=fused)


# --------------------------------------------------------------------------- data -> model input
def to_input(x01, size, mean, std, interp="bilinear"):
    """[0,1] float batch at native res -> model input at `size`, normalized with the model's mean/std."""
    if x01.shape[-1] != size:
        x01 = F.interpolate(x01, size=(size, size), mode=interp, align_corners=False,
                            antialias=size < x01.shape[-1] or interp == "bilinear")
    return (x01 - mean) / std


@torch.no_grad()
def evaluate(model, x, y, size, mean, std, amp, bs=500, interp="bilinear"):
    was = model.training
    model.eval()
    correct = torch.zeros((), device=x.device)
    for k in range(0, x.size(0), bs):
        with amp():
            logits = model(to_input(x[k:k + bs].float().div_(255.0), size, mean, std, interp))
        correct += (logits.argmax(1) == y[k:k + bs]).sum()
    model.train(was)
    return 100.0 * correct.item() / x.size(0)


def kd_loss(s_logits, t_logits, mode, tau):
    s = s_logits.float()
    t = t_logits.float()
    if mode == "hard":
        return F.cross_entropy(s, t.argmax(1))
    return F.kl_div(F.log_softmax(s / tau, -1), F.log_softmax(t / tau, -1), reduction="batchmean",
                    log_target=True) * tau * tau


# --------------------------------------------------------------------------- main
def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: no CUDA device; running on CPU (smoke tests only).")

    def amp():
        return torch.autocast(device_type=device, dtype=torch.bfloat16)

    base = args.model or os.path.basename(os.path.normpath(args.init_ckpt))
    run_name = args.run_name or f"ft_{args.dataset}_{base.split('.')[0]}_{args.img_size}_e{args.epochs}"
    out_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    xtr, ytr, xte, yte, ds = load_data(args, device)
    if args.train_subset:
        xtr, ytr = xtr[:args.train_subset], ytr[:args.train_subset]
    if args.test_subset:
        xte, yte = xte[:args.test_subset], yte[:args.test_subset]
    num_classes, native = ds.NUM_CLASSES, xtr.size(-1)
    R = args.img_size

    if args.init_ckpt:
        model, spec = load_model(args.init_ckpt, num_classes, R, drop_path=args.drop_path, transfer=True,
                                 patch_size=args.patch_size)
    else:
        model, spec = load_model(args.model, num_classes, R, pretrained=not args.no_pretrained,
                                 drop_path=args.drop_path)
    model = model.to(device)
    mean = torch.tensor(spec["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(spec["std"], device=device).view(1, 3, 1, 1)
    n_params = num_params(model)

    teacher = None
    if args.teacher:
        teacher, tspec = load_model(args.teacher)
        if tspec["num_classes"] != num_classes:
            raise SystemExit(f"teacher has {tspec['num_classes']} classes, dataset {num_classes}: "
                             "fine-tune the teacher first (finetune.py) and pass its run dir")
        teacher = teacher.to(device).eval().requires_grad_(False)
        tR = args.teacher_img_size or tspec["img_size"]
        tmean = torch.tensor(tspec["mean"], device=device).view(1, 3, 1, 1)
        tstd = torch.tensor(tspec["std"], device=device).view(1, 3, 1, 1)
        t_acc = evaluate(teacher, xte, yte, tR, tmean, tstd, amp, args.eval_bs, args.interp)
        print(f"teacher {args.teacher}: {num_params(teacher) / 1e6:.2f}M params, {tR}px, val acc {t_acc:.2f}")
        t_fwd = teacher if args.no_compile else torch.compile(teacher)

    n_train = xtr.size(0)
    bs = min(args.batch_size, n_train)
    steps_per_epoch = n_train // bs
    if args.max_steps:
        steps_per_epoch = min(steps_per_epoch, args.max_steps)
    clean_idx = torch.randperm(n_train, generator=torch.Generator().manual_seed(0))[:10000].to(device)
    print(f"Model: {spec.get('name', args.init_ckpt)}  params: {n_params / 1e6:.2f}M  input {native}->{R}px  "
          f"mean/std {spec['mean']}/{spec['std']}  train images: {n_train}  batch: {bs}  steps/epoch: {steps_per_epoch}")

    opt = make_optimizer(model, args, fused=(device == "cuda"))
    sched = make_schedule(opt, args, steps_per_epoch)
    n_fp32 = fp32_modules(model)  # before compile and the EMA copy; see FP32_MODULES
    if n_fp32:
        print(f"running {n_fp32} modules in fp32 under bf16 autocast: {sorted(FP32_MODULES)}")
    train_model = model if args.no_compile else torch.compile(
        model, mode="reduce-overhead" if args.cudagraphs else None)
    ema = None
    if args.ema > 0:
        from timm.utils import ModelEmaV3

        ema = ModelEmaV3(model, decay=args.ema, use_warmup=args.ema_warmup)

    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    history, best_val, best_ema, start_epoch, elapsed, wb_id = [], 0.0, 0.0, 1, 0.0, None
    if os.path.exists(ckpt_path) and not args.no_resume:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        if ema is not None and ck.get("ema") is not None:
            ema.load_state_dict(ck["ema"])
        history, best_val, best_ema = ck["history"], ck["best_val"], ck.get("best_ema", 0.0)
        elapsed, wb_id, start_epoch = ck["elapsed"], ck["wandb_id"], ck["epoch"] + 1
        torch.set_rng_state(ck["torch_rng"])
        if device == "cuda" and ck["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        rng.bit_generator.state = ck["np_rng"]
        print(f"Resumed from {ckpt_path}: continuing at epoch {start_epoch}/{args.epochs}")

    wb = None
    if args.wandb:
        import wandb

        wb_id = wb_id or uuid.uuid4().hex[:8]
        wb = wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=run_name,
                        config=dict(vars(args), params_M=n_params / 1e6, model_spec=spec), dir=out_dir,
                        id=wb_id, resume="allow", tags=["edge-finetune"])

    def save_ckpt(epoch):
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
              "ema": ema.state_dict() if ema is not None else None,
              "epoch": epoch, "history": history, "best_val": best_val, "best_ema": best_ema, "wandb_id": wb_id,
              "elapsed": elapsed + time.time() - t_start, "torch_rng": torch.get_rng_state(),
              "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else None,
              "np_rng": rng.bit_generator.state}
        torch.save(ck, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)

    t_start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        train_model.train()
        perm = torch.randperm(n_train, device=device)
        loss_sum = torch.zeros((), device=device)
        kd_sum = torch.zeros((), device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for k in range(steps_per_epoch):
            idx = perm[k * bs:(k + 1) * bs]
            x = xtr[idx].float().div_(255.0)
            if not args.no_aug:
                x = random_crop_flip(x, pad=native // 8)
            if args.randaug:
                x = rand_augment(x, args.randaug_n, args.randaug_m / 10.0)
            t = smooth_one_hot(ytr[idx], num_classes, args.label_smoothing)
            if not (0 < args.mix_off_epoch <= epoch):
                x, t = mix_batch(x, t, args, rng)  # mixing at native res == mixing after upsampling
            xs = to_input(x, R, mean, std, args.interp)
            if args.reprob > 0:
                xs = random_erasing(xs, args.reprob)
            with amp():
                logits = train_model(xs)
            loss = torch.sum(-t * F.log_softmax(logits.float(), dim=-1), dim=-1).mean()
            if teacher is not None:
                with torch.no_grad(), amp():
                    t_logits = t_fwd(to_input(x, tR, tmean, tstd, args.interp))
                kd = kd_loss(logits, t_logits, args.kd, args.kd_tau)
                loss = (1 - args.kd_alpha) * loss + args.kd_alpha * kd
                kd_sum += kd.detach()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            if ema is not None:
                ema.update(model, step=(epoch - 1) * steps_per_epoch + k)
            loss_sum += loss.detach()
            gstep = (epoch - 1) * steps_per_epoch + k + 1
            if wb and args.log_every_steps > 0 and gstep % args.log_every_steps == 0:
                wb.log({"step_loss": loss.item(), "step_lr": sched.get_last_lr()[0]}, step=gstep)

        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        train_loss = loss_sum.item() / steps_per_epoch
        ips = steps_per_epoch * bs / dt
        lr_now = sched.get_last_lr()[0]
        rec = {"epoch": epoch, "train_loss": train_loss, "lr": lr_now, "epoch_sec": dt, "img_per_sec": ips}
        if teacher is not None:
            rec["kd_loss"] = kd_sum.item() / steps_per_epoch
        line = f"ep {epoch:4d}/{args.epochs} | loss {train_loss:.4f} | lr {lr_now:.2e} | {dt:6.2f}s | {ips:7.0f} img/s"
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val = evaluate(model, xte, yte, R, mean, std, amp, args.eval_bs, args.interp)
            tr_acc = evaluate(model, xtr[clean_idx], ytr[clean_idx], R, mean, std, amp, args.eval_bs, args.interp)
            best_val = max(best_val, val)
            rec.update(val_acc=val, train_acc_clean=tr_acc)
            line += f" | train acc {tr_acc:6.2f} | val acc {val:6.2f}"
            if ema is not None:
                ev = evaluate(ema.module, xte, yte, R, mean, std, amp, args.eval_bs, args.interp)
                best_ema = max(best_ema, ev)
                rec["ema_val_acc"] = ev
                line += f" | ema val {ev:6.2f}"
        print(line, flush=True)
        history.append(rec)
        if wb:
            wb.log(rec, step=epoch * steps_per_epoch if args.log_every_steps > 0 else epoch)
        if (args.ckpt_every > 0 and epoch % args.ckpt_every == 0) or epoch == args.epochs:
            save_ckpt(epoch)

    total_min = (elapsed + time.time() - t_start) / 60
    final = history[-1]
    steady = history[1:] or history
    spec = dict(spec, num_classes=num_classes, img_size=R)
    summary = {
        "run_name": run_name,
        "args": vars(args),
        "model_spec": spec,
        "params_M": n_params / 1e6,
        "total_minutes": total_min,
        "epoch_sec_steady": float(np.median([h["epoch_sec"] for h in steady])),
        "img_per_sec_steady": float(np.median([h["img_per_sec"] for h in steady])),
        "final_val_acc": final.get("val_acc"),
        "final_train_acc_clean": final.get("train_acc_clean"),
        "best_val_acc": best_val,
        "final_ema_val_acc": final.get("ema_val_acc"),
        "best_ema_val_acc": best_ema if ema is not None else None,
        "teacher": args.teacher,
        "history": history,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    if ema is not None:
        torch.save(ema.module.state_dict(), os.path.join(out_dir, "model_ema.pt"))
    if not args.keep_ckpt and os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    msg = f"\nDone in {total_min:.1f} min. final val {final.get('val_acc'):.2f}% | best val {best_val:.2f}%"
    if ema is not None:
        msg += f" | final EMA val {final.get('ema_val_acc'):.2f}% (best {best_ema:.2f}%)"
    print(msg)
    print(f"Steady state: {summary['epoch_sec_steady']:.2f} s/epoch, {summary['img_per_sec_steady']:,.0f} img/s")
    print(f"Saved results.json and model.pt to {out_dir}")
    if wb:
        wb.summary.update({"best_val_acc": best_val, "best_ema_val_acc": best_ema, "total_minutes": total_min})
        wb.finish()


if __name__ == "__main__":
    main()
