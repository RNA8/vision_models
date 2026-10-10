#!/usr/bin/env python3
"""
train_vit.py — train a Vision Transformer on CIFAR-10/100, fast, on one GPU.

Why it is fast:
  * The whole dataset (uint8; 150 MB for CIFAR) lives on the GPU. No DataLoader, no CPU workers.
  * Augmentation (random crop + flip, MixUp/CutMix) runs on the GPU as batched tensor ops.
  * bf16 autocast, TF32 matmuls, torch.compile, fused AdamW, large batches.

The model comes from timm, so there is nothing to implement yourself.

Quick start:
  python train_vit.py --epochs 2                 # smoke test (downloads CIFAR-10)
  python train_vit.py                            # full recipe, ViT-Tiny, 100 epochs
  python train_vit.py --model small --epochs 200 # bigger model, longer run
  python train_vit.py --model resnet18           # CNN control
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))

VIT_CFGS = {
    "tiny": dict(embed_dim=192, depth=12, num_heads=3),
    "small": dict(embed_dim=384, depth=12, num_heads=6),
    "base": dict(embed_dim=768, depth=12, num_heads=12),
}


# --------------------------------------------------------------------------- args
def parse_args():
    p = argparse.ArgumentParser(description="Fast ViT training on CIFAR")
    p.add_argument("--dataset", default="cifar10", help="module name in vit_data/")
    p.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    p.add_argument("--model", default="tiny", choices=list(VIT_CFGS) + ["resnet18"])
    p.add_argument("--patch-size", type=int, default=4, help="ViT patch size: (img_size/p)^2 tokens")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-3, help="peak LR (for batch 1024)")
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--mixup", type=float, default=0.8, help="MixUp alpha; 0 disables")
    p.add_argument("--cutmix", type=float, default=1.0, help="CutMix alpha; 0 disables")
    p.add_argument("--mix-prob", type=float, default=1.0, help="prob. a batch gets mixed")
    p.add_argument("--drop-path", type=float, default=0.1, help="stochastic depth rate")
    p.add_argument("--no-aug", action="store_true", help="disable random crop + flip")
    p.add_argument("--randaug", action="store_true", help="enable GPU RandAugment")
    p.add_argument("--randaug-n", type=int, default=2, help="RandAugment ops per image")
    p.add_argument("--randaug-m", type=float, default=9, help="RandAugment magnitude (0-10)")
    p.add_argument("--grad-clip", type=float, default=1.0, help="0 disables")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--fake-data", action="store_true", help="random tensors, for smoke tests")
    p.add_argument("--plot", action="store_true", help="save position-embedding similarity plot")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-entity", default="softeye")
    p.add_argument("--wandb-project", default="vision_transformers")
    p.add_argument("--run-name", default=None)
    p.add_argument("--out-dir", default=os.path.join(HERE, "runs"))
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# --------------------------------------------------------------------------- data
def load_data(args, device):
    """Return train/test images as uint8 (N, C, H, W) tensors on `device`, labels, and the
    dataset module (NUM_CLASSES, MEAN, STD). Datasets are plug-in modules in vit_data/."""
    import importlib

    ds = importlib.import_module(f"vit_data.{args.dataset}")
    if args.fake_data:
        g = torch.Generator().manual_seed(0)
        xtr = torch.randint(0, 256, (4096, 32, 32, 3), dtype=torch.uint8, generator=g)
        ytr = torch.randint(0, ds.NUM_CLASSES, (4096,), generator=g)
        xte = torch.randint(0, 256, (1024, 32, 32, 3), dtype=torch.uint8, generator=g)
        yte = torch.randint(0, ds.NUM_CLASSES, (1024,), generator=g)
    else:
        xtr, ytr, xte, yte = ds.load(args.data_dir)

    def to_dev(x):  # NHWC uint8 -> NCHW uint8 on device
        return x.permute(0, 3, 1, 2).contiguous().to(device)

    return to_dev(xtr), ytr.to(device), to_dev(xte), yte.to(device), ds


def normalize(x_uint8, mean, std):
    return (x_uint8.float().div_(255.0) - mean) / std


def random_crop_flip(x, pad=4):
    """Per-sample random crop (reflect padding) + horizontal flip, fully on the GPU."""
    B, C, H, W = x.shape
    dev = x.device
    xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    i = torch.randint(0, 2 * pad + 1, (B,), device=dev)
    j = torch.randint(0, 2 * pad + 1, (B,), device=dev)
    rows = (i[:, None] + torch.arange(H, device=dev))[:, None, :, None]  # B,1,H,1
    cols = (j[:, None] + torch.arange(W, device=dev))[:, None, None, :]  # B,1,1,W
    b = torch.arange(B, device=dev)[:, None, None, None]
    c = torch.arange(C, device=dev)[None, :, None, None]
    out = xp[b, c, rows, cols]
    flip = torch.rand(B, device=dev) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(3), out)


# RandAugment op set and magnitudes follow timm's "rand-m9-n2" (rotate 30 deg, shear 0.3,
# translate 45%, enhance factor 1 +/- 0.9 at magnitude 10).
RA_OPS = ["identity", "autocontrast", "equalize", "rotate", "solarize", "color", "posterize",
          "contrast", "brightness", "sharpness", "shear_x", "shear_y", "translate_x", "translate_y"]
SMOOTH_KERNEL = torch.tensor([[1.0, 1.0, 1.0], [1.0, 5.0, 1.0], [1.0, 1.0, 1.0]]) / 13.0


def _gray(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def _equalize(x):
    """Per-image, per-channel histogram equalization of a [0,1] batch (batched, no loops)."""
    B, C, H, W = x.shape
    q = (x * 255).round().long().clamp_(0, 255).view(B, C, H * W)
    hist = torch.zeros(B, C, 256, device=x.device).scatter_add_(2, q, torch.ones_like(q, dtype=x.dtype))
    cdf = hist.cumsum(-1)
    cdf_min = cdf.gather(2, q.amin(-1, keepdim=True))  # count at the darkest present value
    denom = H * W - cdf_min
    out = (cdf.gather(2, q) - cdf_min) / denom.clamp_min(1)
    return torch.where(denom > 0, out, q.to(x.dtype) / 255).view(B, C, H, W)  # flat channel: unchanged


def rand_augment(x, n, mag):
    """Batched RandAugment on a float [0,1] batch: each image gets `n` random ops at magnitude
    `mag` (0-1). Every op is computed for the whole batch and picked per image with torch.where,
    so there are no per-image loops and no host syncs."""
    B, dev = x.size(0), x.device
    kernel = SMOOTH_KERNEL.to(dev).expand(3, 1, 3, 3)
    for _ in range(n):
        op = torch.randint(0, len(RA_OPS), (B,), device=dev)
        is_op = {name: op == i for i, name in enumerate(RA_OPS)}
        sign = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0)

        # geometric ops: one per-image affine warp (identity for images without one), gray fill
        ang = sign * math.radians(30 * mag) * is_op["rotate"]
        theta = torch.zeros(B, 2, 3, device=dev)
        theta[:, 0, 0], theta[:, 0, 1] = torch.cos(ang), -torch.sin(ang)
        theta[:, 1, 0], theta[:, 1, 1] = torch.sin(ang), torch.cos(ang)
        theta[:, 0, 1] += sign * 0.3 * mag * is_op["shear_x"]
        theta[:, 1, 0] += sign * 0.3 * mag * is_op["shear_y"]
        theta[:, 0, 2] = sign * 2 * 0.45 * mag * is_op["translate_x"]  # grid coords span 2
        theta[:, 1, 2] = sign * 2 * 0.45 * mag * is_op["translate_y"]
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        x = F.grid_sample(x - 0.5, grid, mode="bilinear", padding_mode="zeros", align_corners=False) + 0.5

        # pixel ops
        f = (1.0 + sign * 0.9 * mag).view(B, 1, 1, 1)  # enhance factor
        g = _gray(x)
        lo, hi = x.amin((2, 3), keepdim=True), x.amax((2, 3), keepdim=True)
        blur = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), kernel, groups=3)
        shift = 2.0 ** round(4 * mag)  # posterize: drop round(4*mag) of 8 bits
        candidates = {
            "autocontrast": torch.where(hi > lo, (x - lo) / (hi - lo).clamp_min(1e-5), x),
            "equalize": _equalize(x),
            "solarize": torch.where(x >= 1.0 - mag, 1.0 - x, x),
            "color": g + f * (x - g),
            "posterize": torch.floor(x * 255 / shift) * shift / 255,
            "contrast": g.mean((2, 3), keepdim=True) + f * (x - g.mean((2, 3), keepdim=True)),
            "brightness": x * f,
            "sharpness": blur + f * (x - blur),
        }
        for name, val in candidates.items():
            x = torch.where(is_op[name].view(B, 1, 1, 1), val, x)
        x = x.clamp(0.0, 1.0)
    return x


def smooth_one_hot(y, num_classes, eps):
    off = eps / num_classes
    t = torch.full((y.size(0), num_classes), off, device=y.device)
    t.scatter_(1, y[:, None], 1.0 - eps + off)
    return t


def mix_batch(x, t, args, rng):
    """Batch-level MixUp or CutMix (50/50 when both are on). Returns mixed images and soft targets."""
    if args.mixup <= 0 and args.cutmix <= 0:
        return x, t
    if rng.random() > args.mix_prob:
        return x, t
    use_cutmix = args.cutmix > 0 and (args.mixup <= 0 or rng.random() < 0.5)
    perm = torch.randperm(x.size(0), device=x.device)
    if use_cutmix:
        lam = rng.beta(args.cutmix, args.cutmix)
        H, W = x.shape[2:]
        cut = math.sqrt(1.0 - lam)
        ch, cw = int(H * cut), int(W * cut)
        cy, cx = rng.integers(H), rng.integers(W)
        top, bot = max(cy - ch // 2, 0), min(cy + ch // 2, H)
        lft, rgt = max(cx - cw // 2, 0), min(cx + cw // 2, W)
        x = x.clone()
        x[:, :, top:bot, lft:rgt] = x[perm, :, top:bot, lft:rgt]
        lam = 1.0 - (bot - top) * (rgt - lft) / (H * W)  # actual pasted area
    else:
        lam = rng.beta(args.mixup, args.mixup)
        x = lam * x + (1.0 - lam) * x[perm]
    return x, lam * t + (1.0 - lam) * t[perm]


# --------------------------------------------------------------------------- model
def build_model(args, num_classes, img_size=32):
    if args.model == "resnet18":
        import timm

        m = timm.create_model("resnet18", num_classes=num_classes, drop_path_rate=args.drop_path)
        # CIFAR stem: 3x3 conv, stride 1, no max-pool (the ImageNet stem throws away 32x32 detail)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        return m
    from timm.models.vision_transformer import VisionTransformer

    return VisionTransformer(
        img_size=img_size,
        patch_size=args.patch_size,
        num_classes=num_classes,
        drop_path_rate=args.drop_path,
        **VIT_CFGS[args.model],
    )


def make_optimizer(model, args, fused):
    skip = set(getattr(model, "no_weight_decay", lambda: set())())  # pos_embed, cls_token
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or n in skip else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.999), fused=fused)


def make_schedule(opt, args, steps_per_epoch):
    warm = max(1, args.warmup_epochs * steps_per_epoch)
    total = args.epochs * steps_per_epoch
    floor = args.min_lr / args.lr

    def f(step):
        if step < warm:
            return (step + 1) / warm
        prog = min(1.0, (step - warm) / max(1, total - warm))
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(opt, f)


# --------------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model, x, y, mean, std, amp, bs=2048):
    model.eval()
    correct = torch.zeros((), device=x.device)
    for k in range(0, x.size(0), bs):
        with amp():
            logits = model(normalize(x[k : k + bs], mean, std))
        correct += (logits.argmax(1) == y[k : k + bs]).sum()
    model.train()
    return 100.0 * correct.item() / x.size(0)


def save_pos_embed_plot(model, path):
    pe = getattr(model, "pos_embed", None)
    if pe is None:
        print("No pos_embed on this model; skipping plot.")
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pe = pe.detach().float()[0]
    pe = pe[getattr(model, "num_prefix_tokens", 1):]  # drop CLS
    n = pe.size(0)
    g = int(math.isqrt(n))
    sim = F.normalize(pe, dim=-1) @ F.normalize(pe, dim=-1).T  # n x n
    sim = sim.reshape(g, g, g, g).cpu().numpy()
    fig, axes = plt.subplots(g, g, figsize=(g * 1.1, g * 1.1))
    for r in range(g):
        for c in range(g):
            ax = axes[r, c]
            ax.imshow(sim[r, c], vmin=-1, vmax=1, cmap="viridis")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Cosine similarity of each patch's position embedding to all others")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"Saved position-embedding plot to {path}")


# --------------------------------------------------------------------------- main
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: no CUDA device found; running on CPU (fine for --fake-data smoke tests only).")

    def amp():
        return torch.autocast(device_type=device, dtype=torch.bfloat16)

    run_name = args.run_name or f"{args.model}_p{args.patch_size}_{args.dataset}_e{args.epochs}"
    out_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    xtr, ytr, xte, yte, ds = load_data(args, device)
    num_classes, mean_, std_ = ds.NUM_CLASSES, ds.MEAN, ds.STD
    img_size = xtr.size(-1)
    mean = torch.tensor(mean_, device=device).view(1, 3, 1, 1)
    std = torch.tensor(std_, device=device).view(1, 3, 1, 1)
    n_train = xtr.size(0)
    bs = min(args.batch_size, n_train)
    clean_idx = torch.randperm(n_train, generator=torch.Generator().manual_seed(0))[:10000].to(device)
    steps_per_epoch = n_train // bs  # drop the last partial batch: keeps shapes static for compile

    model = build_model(args, num_classes, img_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}  params: {n_params / 1e6:.2f}M  train images: {n_train}  "
          f"batch: {bs}  steps/epoch: {steps_per_epoch}")

    opt = make_optimizer(model, args, fused=(device == "cuda"))
    sched = make_schedule(opt, args, steps_per_epoch)
    train_model = model if args.no_compile else torch.compile(model)

    wb = None
    if args.wandb:
        import wandb

        wb = wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=run_name,
                        config=vars(args), dir=out_dir)

    history, best_val = [], 0.0
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_model.train()
        perm = torch.randperm(n_train, device=device)
        loss_sum = torch.zeros((), device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        for k in range(steps_per_epoch):
            idx = perm[k * bs : (k + 1) * bs]
            x = xtr[idx].float().div_(255.0)
            if not args.no_aug:
                x = random_crop_flip(x, pad=img_size // 8)
            if args.randaug:
                x = rand_augment(x, args.randaug_n, args.randaug_m / 10.0)
            x = (x - mean) / std
            y = ytr[idx]
            t = smooth_one_hot(y, num_classes, args.label_smoothing)
            x, t = mix_batch(x, t, args, rng)

            with amp():
                logits = train_model(x)
            loss = torch.sum(-t * F.log_softmax(logits.float(), dim=-1), dim=-1).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            loss_sum += loss.detach()

        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        train_loss = loss_sum.item() / steps_per_epoch
        ips = steps_per_epoch * bs / dt
        lr_now = sched.get_last_lr()[0]
        rec = {"epoch": epoch, "train_loss": train_loss, "lr": lr_now, "epoch_sec": dt, "img_per_sec": ips}

        line = (f"ep {epoch:4d}/{args.epochs} | loss {train_loss:.4f} | lr {lr_now:.2e} | "
                f"{dt:6.2f}s | {ips:9.0f} img/s")
        if epoch == 1 and not args.no_compile:
            line += "  (includes torch.compile warm-up)"

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val = evaluate(model, xte, yte, mean, std, amp)
            # clean (no-aug) accuracy on 10k training images: shows the train/val gap
            tr_acc = evaluate(model, xtr[clean_idx], ytr[clean_idx], mean, std, amp)
            best_val = max(best_val, val)
            rec.update(val_acc=val, train_acc_clean=tr_acc)
            line += f" | train acc {tr_acc:6.2f} | val acc {val:6.2f}"
        print(line, flush=True)
        history.append(rec)
        if wb:
            wb.log(rec, step=epoch)

    total_min = (time.time() - t_start) / 60
    final = history[-1]
    steady = history[1:] or history  # epoch 1 includes the torch.compile warm-up
    epoch_sec_steady = float(np.median([h["epoch_sec"] for h in steady]))
    img_per_sec_steady = float(np.median([h["img_per_sec"] for h in steady]))
    summary = {
        "run_name": run_name,
        "args": vars(args),
        "params_M": n_params / 1e6,
        "total_minutes": total_min,
        "epoch_sec_steady": epoch_sec_steady,
        "img_per_sec_steady": img_per_sec_steady,
        "final_val_acc": final.get("val_acc"),
        "final_train_acc_clean": final.get("train_acc_clean"),
        "best_val_acc": best_val,
        "history": history,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    if args.plot:
        save_pos_embed_plot(model, os.path.join(out_dir, "pos_embed_similarity.png"))

    print(f"\nDone in {total_min:.1f} min. final val {final.get('val_acc'):.2f}% | "
          f"best val {best_val:.2f}% | clean train {final.get('train_acc_clean'):.2f}%")
    print(f"Steady state (excl. epoch 1): {epoch_sec_steady:.2f} s/epoch, {img_per_sec_steady:,.0f} img/s")
    print(f"Saved results.json and model.pt to {out_dir}")
    if wb:
        wb.summary.update({"best_val_acc": best_val, "total_minutes": total_min})
        wb.finish()


if __name__ == "__main__":
    main()
