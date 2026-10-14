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
  python train_vit.py --model slvit --batch-size 128 --lr 3e-3 --warmup-epochs 10 --randaug \
      --repeat-aug 3 --reprob 0.25 --init impulse    # SL-ViT-style recipe (see run_baseline_fix.sh)
"""
import argparse
import json
import math
import os
import time
import uuid

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
from models.recurrent_vit import RD_PRESETS  # recurrent-depth ViT presets (rd_*), see models/recurrent_vit.py

VIT_CFGS = {
    "tiny": dict(embed_dim=192, depth=12, num_heads=3),
    "small": dict(embed_dim=384, depth=12, num_heads=6),
    "base": dict(embed_dim=768, depth=12, num_heads=12),
    # SL-ViT's plain-ViT baseline (arXiv 2112.13492, 93.58% C10) = the best shape in Gani et al. Tab. 15
    "slvit": dict(embed_dim=192, depth=9, num_heads=12, mlp_ratio=2),
}


# --------------------------------------------------------------------------- args
def parse_args():
    p = argparse.ArgumentParser(description="Fast ViT training on CIFAR")
    p.add_argument("--dataset", default="cifar10", help="module name in vit_data/")
    p.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    p.add_argument("--model", default="tiny", choices=list(VIT_CFGS) + ["resnet18"] + list(RD_PRESETS))
    p.add_argument("--patch-size", type=int, default=4, help="ViT patch size: (img_size/p)^2 tokens")
    p.add_argument("--embed-dim", type=int, default=None, help="override VIT_CFGS[model] width")
    p.add_argument("--depth", type=int, default=None, help="override VIT_CFGS[model] depth")
    p.add_argument("--heads", type=int, default=None, help="override VIT_CFGS[model] num_heads")
    p.add_argument("--mlp-ratio", type=float, default=None, help="override VIT_CFGS[model] mlp_ratio (timm default 4)")
    p.add_argument("--pool", default="cls", choices=["cls", "seqpool"],
                   help="cls = class token; seqpool = CCT attention pooling over the patch tokens")
    p.add_argument("--lsa", action="store_true",
                   help="Locality Self-Attention (SL-ViT): learnable per-head temperature + diagonal mask")
    p.add_argument("--spt", action="store_true",
                   help="Shifted Patch Tokenization (SL-ViT, arXiv 2112.13492 Eq. 6): concat 4 diagonal half-patch shifts, "
                        "then patchify, flatten, LayerNorm, Linear")
    p.add_argument("--multiscale", action="store_true",
                   help="multiscale input tokens: patchify the image, then 2x-downscaled copies with the same patch "
                        "size until one patch is left, and feed all tokens to one transformer (32px, p4: 64+16+4+1)")
    p.add_argument("--ms-proj", default="shared", choices=["shared", "per-scale"],
                   help="multiscale: one patch projection for every scale, or one per scale")
    p.add_argument("--ms-pos", default="learned", choices=["learned", "pooled"],
                   help="multiscale coarse-scale position embeddings: learned = own parameters per token (initialized "
                        "as the avg-pooled fine-grid pos-embed); pooled = always the avg-pooled fine pos-embed plus a "
                        "learned per-scale embedding")
    p.add_argument("--ms-down", default="avg", choices=["avg", "bilinear"],
                   help="multiscale 2x downscale: avg-pool 2x2, or antialiased bilinear")
    p.add_argument("--ms-pyramid", default="gauss", choices=["gauss", "laplacian"],
                   help="multiscale: gauss = each level is the downscaled image; laplacian = each level but the "
                        "coarsest encodes the residual from the upsampled next-coarser level")
    p.add_argument("--reg-tokens", type=int, default=0,
                   help="extra learnable register tokens (Darcet et al. 2309.16588; not pooled by seqpool)")
    p.add_argument("--init", default="default", choices=["default", "mimetic", "impulse"],
                   help="attention init: mimetic (Trockman & Kolter 2023) or impulse (Zheng et al. 2025)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-3,
                   help="peak LR, used exactly as given: no automatic batch-size scaling (2e-3 was tuned for batch 1024)")
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--mixup", type=float, default=0.8, help="MixUp alpha; 0 disables")
    p.add_argument("--cutmix", type=float, default=1.0, help="CutMix alpha; 0 disables")
    p.add_argument("--mix-prob", type=float, default=1.0, help="prob. a batch gets mixed")
    p.add_argument("--mix-off-epoch", type=int, default=0,
                   help="no MixUp/CutMix from this epoch on (CCT: 175 of 300); 0 = never switch off")
    p.add_argument("--repeat-aug", type=int, default=1,
                   help="repeated augmentation: each batch = bs/K unique images x K independent augmentations. "
                        "An epoch still covers every training image once, so steps/epoch = n_train // (bs/K), K x more")
    p.add_argument("--reprob", type=float, default=0.0,
                   help="random-erasing prob. per sample (timm 'pixel' mode: area 2-33%%, aspect 0.3-3.3, noise fill)")
    p.add_argument("--drop-path", type=float, default=0.1, help="stochastic depth rate")
    p.add_argument("--no-aug", action="store_true", help="disable random crop + flip")
    p.add_argument("--randaug", action="store_true", help="enable GPU RandAugment")
    p.add_argument("--randaug-n", type=int, default=2, help="RandAugment ops per image")
    p.add_argument("--randaug-m", type=float, default=9, help="RandAugment magnitude (0-10)")
    p.add_argument("--grad-clip", type=float, default=1.0, help="0 disables")
    p.add_argument("--moe", default=None, help="moe.py spec applied after build_model, e.g. mlp:e=4,last=4")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--cudagraphs", action="store_true",
                   help="compile with CUDA graphs (mode=reduce-overhead): ~1.6-3x faster for small ViTs, "
                        "which are kernel-launch bound; same math")
    p.add_argument("--fake-data", action="store_true", help="random tensors, for smoke tests")
    p.add_argument("--plot", action="store_true", help="save position-embedding similarity plot")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-entity", default="softeye")
    p.add_argument("--wandb-project", default="vision_transformers")
    p.add_argument("--run-name", default=None)
    p.add_argument("--out-dir", default=os.path.join(HERE, "runs"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--holdout", type=int, default=0,
                   help="hold out N train images (fixed split, independent of --seed) as the validation set used during "
                        "training; the test set is scored once at the end as final_test_acc")
    p.add_argument("--r18-stem", default="cifar", choices=["cifar", "stride2"],
                   help="ResNet-18 stem: cifar = 3x3 s1; stride2 = 3x3 s2 (for 64x64 inputs), both no max-pool")
    p.add_argument("--eval-bs", type=int, default=2048)
    p.add_argument("--log-every-steps", type=int, default=0, help="wandb train loss/lr every N steps; 0 = off")
    p.add_argument("--ckpt-every", type=int, default=10, help="save runs/<name>/ckpt.pt every N epochs")
    p.add_argument("--no-resume", action="store_true", help="ignore an existing ckpt.pt")
    p.add_argument("--keep-ckpt", action="store_true", help="keep ckpt.pt after the run finishes")
    # recurrent-depth ViT (--model rd_*); None = use the preset's value. See models/recurrent_vit.py.
    g = p.add_argument_group("recurrent depth (rd_* models)")
    g.add_argument("--rd-r-mean", type=int, default=None, help="mean training recurrence r")
    g.add_argument("--rd-k-backprop", type=int, default=None, help="backprop through last k loops (default r_mean)")
    g.add_argument("--rd-depth-dist", default=None, choices=["poisson_lognormal", "uniform", "fixed"])
    g.add_argument("--rd-state-init", default=None, choices=["normal", "zeros"])
    g.add_argument("--rd-inject", default=None, choices=["concat", "add", "none"], help="input re-injection")
    g.add_argument("--rd-outer-norm", default=None, choices=["ln", "rms", "none"], help="norm on the state per loop")
    g.add_argument("--rd-coda-every-step", action="store_true",
                   help="deep supervision: decode every loop in training and average the logits")
    g.add_argument("--rd-lora-rank", type=int, default=None, help="per-iteration LoRA rank on the core's qkv/proj/MLP")
    g.add_argument("--rd-sandwich", action="store_true", help="sandwich norm in the core (RMSNorm after attn/MLP sums)")
    g.add_argument("--qk-norm", action="store_true", help="QK-norm in attention (rd_* models)")
    g.add_argument("--eval-r", type=int, default=None, help="recurrence used by evaluate() (default r_mean)")
    return p.parse_args()


# --------------------------------------------------------------------------- data
def load_data(args, device, with_stream=False):
    """Return train/test images as uint8 (N, C, H, W) tensors on `device`, labels, and the
    dataset module (NUM_CLASSES, MEAN, STD). Datasets are plug-in modules in vit_data/.
    Streaming datasets (module flag STREAMING, e.g. in21k64: too big for the GPU) have a Stream object
    (vit_data/in21k64.py) that yields train batches; here xtr/ytr are then only a fixed 10k-image train subset
    (for the clean-train-accuracy metric); with_stream=True appends the Stream (or None) as a 6th return value."""
    import importlib

    ds = importlib.import_module(f"vit_data.{args.dataset}")
    if args.fake_data:
        g = torch.Generator().manual_seed(0)
        xtr = torch.randint(0, 256, (4096, 32, 32, 3), dtype=torch.uint8, generator=g)
        ytr = torch.randint(0, ds.NUM_CLASSES, (4096,), generator=g)
        xte = torch.randint(0, 256, (1024, 32, 32, 3), dtype=torch.uint8, generator=g)
        yte = torch.randint(0, ds.NUM_CLASSES, (1024,), generator=g)
    elif getattr(ds, "STREAMING", False):
        st = ds.Stream(args.data_dir, device, args.seed)
        out = (st.xclean, st.yclean, st.xte, st.yte, ds)
        return out + (st,) if with_stream else out
    else:
        xtr, ytr, xte, yte = ds.load(args.data_dir)

    def to_dev(x):  # NHWC uint8 -> NCHW uint8 on device
        return x.permute(0, 3, 1, 2).contiguous().to(device)

    out = (to_dev(xtr), ytr.to(device), to_dev(xte), yte.to(device), ds)
    return out + (None,) if with_stream else out


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


def random_erasing(x, p, min_area=0.02, max_area=1 / 3, min_aspect=0.3, tries=10):
    """Batched timm RandomErasing(mode='pixel') on a normalized batch: with prob. p a sample gets one
    rectangle of area U(min_area, max_area) and log-uniform aspect in [min_aspect, 1/min_aspect],
    filled with per-pixel N(0, 1) noise. Like timm, up to `tries` boxes are drawn and the first that
    fits is used (none fits -> no erase). All on the GPU, no host syncs."""
    B, C, H, W = x.shape
    dev = x.device
    area = torch.empty(B, tries, device=dev).uniform_(min_area, max_area) * (H * W)
    log_r = torch.empty(B, tries, device=dev).uniform_(math.log(min_aspect), -math.log(min_aspect))
    h = torch.sqrt(area * log_r.exp()).round().long()
    w = torch.sqrt(area / log_r.exp()).round().long()
    fits = (h < H) & (w < W)
    first = fits.int().argmax(1, keepdim=True)  # first box that fits (0 if none; masked out below)
    h, w = h.gather(1, first)[:, 0], w.gather(1, first)[:, 0]
    top = (torch.rand(B, device=dev) * (H - h + 1)).long()  # randint(0, H - h) inclusive
    lft = (torch.rand(B, device=dev) * (W - w + 1)).long()
    erase = (torch.rand(B, device=dev) < p) & fits.any(1)
    r = torch.arange(H, device=dev)[None, :, None]
    c = torch.arange(W, device=dev)[None, None, :]
    m = (r >= top[:, None, None]) & (r < (top + h)[:, None, None]) \
        & (c >= lft[:, None, None]) & (c < (lft + w)[:, None, None]) & erase[:, None, None]
    return torch.where(m[:, None], torch.randn_like(x), x)


# --------------------------------------------------------------------------- model
class SeqPool(nn.Module):
    """CCT sequence pooling (arXiv 2104.05704, Eq. 2-4): w = softmax(Linear(d, 1)(x_L)) over tokens,
    z = w^T x_L. Plugged into timm's `attn_pool` slot, so it sees the final-LayerNorm output with the
    class token already dropped (timm pool_include_prefix=False). The class token is kept in the
    sequence (it still attends and is attended to), it is just not pooled or read by the head."""

    def __init__(self, dim):
        super().__init__()
        self.attention_pool = nn.Linear(dim, 1)
        nn.init.trunc_normal_(self.attention_pool.weight, std=0.02)  # CCT init
        nn.init.zeros_(self.attention_pool.bias)

    def forward(self, x):  # (B, N, D) -> (B, D)
        w = self.attention_pool(x).softmax(dim=1)  # (B, N, 1)
        return (w * x).sum(dim=1)


def _lsa_forward(self, x, attn_mask=None, is_causal=False):
    """Forward of a timm Attention subclass (built in build_model).
    Locality Self-Attention (SL-ViT, arXiv 2112.13492, Eq. 5-7 / official code): logits =
    (q k^T) * tau_h with a learnable per-head temperature tau_h (init head_dim^-0.5, replacing the
    fixed scale), and the diagonal masked (official code sets it to -987654321; a boolean mask
    passed to SDPA is the same as -inf), so no token attends to itself. CLS is masked too, as in the
    official code. Stays on F.scaled_dot_product_attention (mask -> mem-efficient/math kernel)."""
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)
    q = q * self.lsa_temperature.view(1, -1, 1, 1).to(q.dtype)
    keep = ~torch.eye(N, dtype=torch.bool, device=x.device)  # True = may attend
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=keep, scale=1.0,
                                       dropout_p=self.attn_drop.p if self.training else 0.0)
    x = self.norm(x.transpose(1, 2).reshape(B, N, self.attn_dim))
    return self.proj_drop(self.proj(x))


def _ortho_like(d, alpha, beta, sign, dist="normal"):
    """Mimetic factorization (Trockman & Kolter, arXiv 2305.09828, Sec. 4): sample
    A = alpha * Z + sign * beta * I with Z ~ N(0, I/d) (Eq. 1-2), SVD A = U S V^T (Eq. 3, 5), and return
    L = U S^1/2, R = S^1/2 V^T so that L @ R = A exactly. (Eq. 4 as printed, W_V = U S, W_proj = V S^1/2,
    does not multiply back to A; the symmetric split is the one in Zheng et al.'s released code.)
    dist='uniform' = U(+-sqrt(3/d)), same variance, as in Zheng et al.'s code for V/proj."""
    if dist == "normal":
        Z = torch.randn(d, d, dtype=torch.float64) / math.sqrt(d)
    else:
        Z = (torch.rand(d, d, dtype=torch.float64) * 2 - 1) * math.sqrt(3.0 / d)
    U, S, Vh = torch.linalg.svd(alpha * Z + sign * beta * torch.eye(d, dtype=torch.float64))
    return U * S.sqrt(), S.sqrt()[:, None] * Vh


def _impulse_qk(heads, grid, rank, pe, kernel=3, scale=1000.0):
    """Impulse init (Zheng et al., arXiv 2505.19985, Sec. 4 / Alg. 1; ported from the official
    timm fork, github.com/osiriszjq/structured_initialization, `impulse_init`). Per head:
      M~ = scale * H_impulse + noise          (Alg. 1: alpha*H + beta*Z; code: 1000*H + 3*TN(0, 0.6) off-support)
      X~ = LayerNorm(P),  P_inv = (X~^T X~)^+ X~^T   (pseudo-inverse truncated to `rank` singular values)
      M^ = P_inv M~ P_inv^T,  M^ = U S V^T,  Q~ = U[:, :r] S^1/2,  K~ = V[:, :r] S^1/2
      Q = Q~ / ||Q~||_F,  K = K~ / ||K~||_F   (the code's gamma is applied by the caller)
    H_impulse is the zero-padded (non-wrapping) 3x3 single-tap convolution on the token grid; head i
    gets tap k[i % 9] with the code's order k = [4,1,5,3,7,0,2,6,8] (centre first). Returns Q, K as
    (heads, D, r) so that q = x Q[h], k = x K[h] and Q[h] K[h]^T ~ M^."""
    order = [4, 1, 5, 3, 7, 0, 2, 6, 8]
    n = grid * grid
    Hm = torch.zeros(heads, n, n, dtype=torch.float64)
    ii = torch.arange(grid)
    for h in range(heads):
        dy, dx = order[h % kernel**2] // kernel - kernel // 2, order[h % kernel**2] % kernel - kernel // 2
        r, c = torch.meshgrid(ii, ii, indexing="ij")
        ok = (r + dy >= 0) & (r + dy < grid) & (c + dx >= 0) & (c + dx < grid)
        src = (r * grid + c)[ok]
        Hm[h, src, ((r + dy) * grid + (c + dx))[ok]] = 1.0
    noise = nn.init.trunc_normal_(torch.empty_like(Hm), std=0.6)
    M = scale * Hm + 3.0 * noise * (Hm == 0)
    X = F.layer_norm(pe.double(), [pe.size(-1)])  # (N, D)
    U, S, Vh = torch.linalg.svd(X.T @ X)
    s_inv = torch.zeros_like(S)
    s_inv[:rank] = 1.0 / S[:rank]
    P_inv = (Vh.T * s_inv) @ U.T @ X.T  # (D, N)
    U, S, Vh = torch.linalg.svd(P_inv @ M @ P_inv.T)  # batched over heads
    s = S[:, :rank].sqrt()
    Q = U[:, :, :rank] * s[:, None]
    K = Vh.transpose(1, 2)[:, :, :rank] * s[:, None]
    Q = Q / Q.norm(dim=(1, 2), keepdim=True)
    K = K / K.norm(dim=(1, 2), keepdim=True)
    return Q, K


@torch.no_grad()
def structured_init(model, mode):
    """Overwrite the attention weights of a timm ViT in place. timm's qkv weight is (3D, D), rows
    ordered [q | k | v], head-major, applied as x @ W^T; so for q = x Wq_h we write Wq_h^T into rows."""
    D = model.embed_dim
    if mode == "impulse":
        # official code: pos_embed *= 18, Q/K *= 7 (its norm_p, norm_q defaults); P excludes CLS
        model.pos_embed.mul_(18.0)
        pe = model.pos_embed[0, model.num_prefix_tokens:].detach().cpu()
        grid = math.isqrt(pe.size(0))
    for blk in model.blocks:
        a = blk.attn
        H, hd = a.num_heads, a.head_dim
        W = a.qkv.weight
        if mode == "mimetic":
            # W_Q W_K^T ~ alpha1 Z1 + beta1 I (Eq. 1), W_Q/W_K = first hd columns of U S^1/2, V S^1/2
            # (Eq. 6-7, i.e. the rank-hd truncation), Z resampled per head ("we resample Z2 for each head",
            # Eq. 5 labelling); alpha1 = beta1 = 0.7 (Sec. 5.1)
            for h in range(H):
                L, R = _ortho_like(D, 0.7, 0.7, +1)
                W[h * hd:(h + 1) * hd] = L[:, :hd].T.float()
                W[D + h * hd:D + (h + 1) * hd] = R[:hd].float()
            dist = "normal"
        else:
            Q, K = _impulse_qk(H, grid, hd, pe)
            for h in range(H):
                W[h * hd:(h + 1) * hd] = 7.0 * Q[h].T.float()
                W[D + h * hd:D + (h + 1) * hd] = 7.0 * K[h].T.float()
            dist = "uniform"
        # W_V W_proj ~ alpha2 Z2 - beta2 I (Eq. 2), alpha2 = beta2 = 0.4, one Z2 across heads (as in the
        # official impulse code, which uses the same mimetic V/proj init): x W_V W_proj = x (L R)
        L, R = _ortho_like(D, 0.4, 0.4, -1, dist)
        W[2 * D:] = L.T.float()
        a.proj.weight.copy_(R.T.float())  # biases stay at timm's zero init


class SPTProj(nn.Module):
    """Shifted Patch Tokenization (Lee et al., arXiv 2112.13492, Eq. 6): S(x) = LN(P([x s1 s2 s3 s4])) E_S.
    Each image is shifted by half the patch size in the four diagonal directions (zero fill, cropped back to
    the input size), concatenated with the input on channels, split into non-overlapping patches, flattened,
    layer-normalized and linearly projected. Drop-in for timm PatchEmbed.proj: (B, C, H, W) -> (B, D, H/p, W/p)."""

    def __init__(self, in_chans, embed_dim, patch_size):
        super().__init__()
        self.p, self.s = patch_size, patch_size // 2
        d_in = 5 * in_chans * patch_size * patch_size
        self.norm = nn.LayerNorm(d_in)
        self.proj = nn.Linear(d_in, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        s, p = self.s, self.p
        xp = F.pad(x, (s, s, s, s))
        shifts = [xp[:, :, s + dy:s + dy + H, s + dx:s + dx + W] for dx, dy in ((-s, -s), (s, -s), (-s, s), (s, s))]
        x = torch.cat([x] + shifts, dim=1)                       # (B, 5C, H, W)
        x = F.unfold(x, kernel_size=p, stride=p).transpose(1, 2)  # (B, N, 5C*p*p)
        x = self.proj(self.norm(x))                              # (B, N, D)
        return x.transpose(1, 2).reshape(B, -1, H // p, W // p)


class MultiscaleEmbed(nn.Module):
    """Multiscale input tokens (--multiscale). The (normalized) image x_0 is patchified; x_{s+1} = 2x-downscaled x_s
    is patchified with the same patch size, until the grid is 1x1. Tokens of all scales are concatenated
    fine-to-coarse: (B, sum_s (g/2^s)^2, D), e.g. 64+16+4+1 = 85 for 32 px, patch 4. Replaces timm's patch_embed;
    `proj0` (Conv2d or SPTProj, (B,C,H,W)->(B,D,h,w)) embeds scale 0, and the other scales too if shared.
    Laplacian pyramid: level s < S-1 gets x_s - up(x_{s+1}) (the coarsest level is the plain image).
    Position embeddings of the coarse scales live here (coarse_pos); timm's pos_embed keeps CLS + fine grid."""

    def __init__(self, proj0, make_proj, grid, patch_size, shared, down, pyramid, pos_mode, fine_pe):
        super().__init__()
        self.n_scales = int(math.log2(grid)) + 1
        assert 2 ** (self.n_scales - 1) == grid, f"--multiscale needs a power-of-2 token grid, got {grid}"
        self.grid, self.down, self.pyramid, self.pos_mode = grid, down, pyramid, pos_mode
        self.projs = nn.ModuleList([proj0] + ([] if shared else [make_proj() for _ in range(self.n_scales - 1)]))
        self.shared = shared
        # timm-PatchEmbed-like attributes, for code that inspects them
        self.grid_size, self.patch_size = (grid, grid), (patch_size, patch_size)
        self.num_patches = sum((grid >> s) ** 2 for s in range(self.n_scales))
        if pos_mode == "learned":
            self.pos_coarse = nn.Parameter(self._pool_pe(fine_pe.detach()).clone())
        else:
            self.scale_embed = nn.Parameter(torch.zeros(self.n_scales - 1, fine_pe.size(-1)))

    def _pool_pe(self, fine_pe):  # (1, g*g, D) -> (1, sum_{s>=1} (g/2^s)^2, D), avg-pooled per scale
        D, g = fine_pe.size(-1), self.grid
        pe = fine_pe.transpose(1, 2).reshape(1, D, g, g)
        return torch.cat([F.avg_pool2d(pe, 2 ** s).flatten(2) for s in range(1, self.n_scales)], 2).transpose(1, 2)

    def coarse_pos(self, fine_pe):
        if self.pos_mode == "learned":
            return self.pos_coarse
        sizes = [(self.grid >> s) ** 2 for s in range(1, self.n_scales)]
        se = torch.cat([self.scale_embed[i].expand(n, -1) for i, n in enumerate(sizes)], 0)
        return self._pool_pe(fine_pe) + se[None]

    def _down(self, x):
        if self.down == "avg":
            return F.avg_pool2d(x, 2)
        return F.interpolate(x, scale_factor=0.5, mode="bilinear", antialias=True, align_corners=False)

    def _up(self, x):
        if self.down == "avg":  # nearest: up(avg_pool(x)) is the exact blockwise mean, so the residual sums to 0
            return F.interpolate(x, scale_factor=2, mode="nearest")
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x):
        levels = [x]
        for _ in range(self.n_scales - 1):
            levels.append(self._down(levels[-1]))
        if self.pyramid == "laplacian":
            levels = [levels[s] - self._up(levels[s + 1]) for s in range(self.n_scales - 1)] + [levels[-1]]
        toks = [self.projs[0 if self.shared else s](lv).flatten(2) for s, lv in enumerate(levels)]
        return torch.cat(toks, 2).transpose(1, 2)  # (B, N, D)


def _ms_pos_embed(self, x):
    """VisionTransformer._pos_embed for --multiscale: CLS + all-scale tokens, plus [pos_embed | coarse pos]."""
    pe = self.pos_embed
    pe = torch.cat([pe, self.patch_embed.coarse_pos(pe[:, self.num_prefix_tokens:])], 1)
    x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], 1)
    return self.pos_drop(x + pe)


def apply_multiscale(m, args, img_size):
    """Turn a built timm ViT (after structured_init / SPT) into the multiscale-token ViT, in place."""
    from timm.models.vision_transformer import VisionTransformer

    assert m.num_prefix_tokens == 1 and not m.no_embed_class, "--multiscale: CLS-only, timm pos_embed layout"
    p, D = args.patch_size, m.embed_dim
    spt = getattr(args, "spt", False)

    def make_proj():
        return SPTProj(3, D, p) if spt else nn.Conv2d(3, D, p, p)

    m.patch_embed = MultiscaleEmbed(m.patch_embed.proj, make_proj, img_size // p, p,
                                    shared=args.ms_proj == "shared", down=args.ms_down, pyramid=args.ms_pyramid,
                                    pos_mode=args.ms_pos, fine_pe=m.pos_embed[:, m.num_prefix_tokens:])
    nwd = VisionTransformer.no_weight_decay(m) | {"patch_embed.pos_coarse", "patch_embed.scale_embed"}
    m.__class__ = type("MultiscaleViT", (type(m),), {"_pos_embed": _ms_pos_embed, "no_weight_decay": lambda self: nwd})


def apply_lsa(m):
    """Switch every timm Attention in `m` to LSA (same weights): learnable per-head temperature
    initialized to head_dim^-0.5 plus diagonal masking. Works for the timm ViT and RecurrentViT."""
    from timm.layers import Attention

    lsa_cls = type("LSAAttention", (Attention,), {"forward": _lsa_forward})
    for a in m.modules():
        if type(a) is Attention:
            a.__class__ = lsa_cls
            a.lsa_temperature = nn.Parameter(torch.full((a.num_heads,), a.head_dim ** -0.5))


def build_model(args, num_classes, img_size=32):
    if args.model.startswith("rd_"):
        from models.recurrent_vit import build_recurrent_vit

        m = build_recurrent_vit(
            args.model, num_classes, img_size, args.patch_size, args.drop_path,
            r_mean=args.rd_r_mean, k_backprop=args.rd_k_backprop, depth_dist=args.rd_depth_dist,
            state_init=args.rd_state_init, inject=args.rd_inject, outer_norm=args.rd_outer_norm,
            qk_norm=args.qk_norm or None, r_eval=args.eval_r,
            coda_every_step=args.rd_coda_every_step or None,
            lora_rank=args.rd_lora_rank, sandwich=args.rd_sandwich or None)  # train mode samples r inside forward()
        if args.init != "default":
            print(f"note: --init {args.init} is only implemented for the standard ViT; ignored for {args.model}")
        if args.lsa:
            apply_lsa(m)
        if args.pool == "seqpool":
            m.attn_pool = SeqPool(m.embed_dim)  # decode() pools the patch tokens instead of reading CLS
        return m
    if args.model == "resnet18":
        import timm

        m = timm.create_model("resnet18", num_classes=num_classes, drop_path_rate=args.drop_path)
        # CIFAR stem: 3x3 conv, stride 1, no max-pool (the ImageNet stem throws away 32x32 detail).
        # stride2: same but stride 2, so 64x64 inputs give the same feature-map cost as 32x32 with stride 1.
        stride = 2 if args.r18_stem == "stride2" else 1
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=stride, padding=1, bias=False)
        m.maxpool = nn.Identity()
        return m
    from timm.layers import Attention
    from timm.models.vision_transformer import VisionTransformer

    cfg = dict(VIT_CFGS[args.model])
    for k, v in [("embed_dim", args.embed_dim), ("depth", args.depth), ("num_heads", args.heads),
                 ("mlp_ratio", args.mlp_ratio)]:
        if v is not None:
            cfg[k] = v
    if getattr(args, "reg_tokens", 0):
        cfg["reg_tokens"] = args.reg_tokens
    m = VisionTransformer(
        img_size=img_size,
        patch_size=args.patch_size,
        num_classes=num_classes,
        drop_path_rate=args.drop_path,
        **cfg,
    )
    # everything below only runs for non-default flags, so default runs consume the same RNG stream
    if args.init != "default":
        structured_init(m, args.init)
    if getattr(args, "spt", False):  # callers like finetune.py build their own Namespace
        m.patch_embed.proj = SPTProj(3, m.embed_dim, args.patch_size)
    if getattr(args, "multiscale", False):
        apply_multiscale(m, args, img_size)
    if args.lsa:
        apply_lsa(m)
    if args.pool == "seqpool":
        m.attn_pool = SeqPool(m.embed_dim)  # timm's pool() runs attn_pool on x[:, num_prefix_tokens:]
    return m


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

    xtr, ytr, xte, yte, ds, stream = load_data(args, device, with_stream=True)
    xtest = ytest = None
    if args.holdout:  # train on N_train - holdout images, validate on the held-out ones, test once at the end
        assert stream is None, "--holdout needs an in-memory dataset"
        perm = torch.randperm(xtr.size(0), generator=torch.Generator().manual_seed(1234)).to(device)
        xtest, ytest = xte, yte
        xte, yte = xtr[perm[:args.holdout]], ytr[perm[:args.holdout]]
        xtr, ytr = xtr[perm[args.holdout:]], ytr[perm[args.holdout:]]
        print(f"holdout: train {xtr.size(0)}, val (held-out train) {xte.size(0)}, test {xtest.size(0)} scored at the end")
    num_classes, mean_, std_ = ds.NUM_CLASSES, ds.MEAN, ds.STD
    img_size = xtr.size(-1)
    mean = torch.tensor(mean_, device=device).view(1, 3, 1, 1)
    std = torch.tensor(std_, device=device).view(1, 3, 1, 1)
    n_train = stream.n if stream else xtr.size(0)
    bs = min(args.batch_size, n_train)
    clean_idx = torch.randperm(xtr.size(0), generator=torch.Generator().manual_seed(0))[:10000].to(device)
    # Repeated augmentation (DeiT): a batch is ubs = bs/K unique images, each repeated K times and
    # augmented independently. An epoch is one pass over the unique images, so it has K x the steps
    # (and K x the samples) of a K=1 epoch; the batch shape stays (bs, ...). Note: SL-ViT's "--ra 3"
    # (official code: RASampler(repetitions=1, len_factor=3)) also makes an epoch 3x longer, but
    # without in-batch repeats; this definition matches its step count per epoch.
    K = args.repeat_aug
    assert K >= 1 and bs % K == 0, f"--batch-size {bs} must be divisible by --repeat-aug {K}"
    ubs = bs // K
    steps_per_epoch = n_train // ubs  # drop the last partial batch: keeps shapes static for compile

    model = build_model(args, num_classes, img_size)
    if args.moe:
        import moe

        print("MoE conversion:", args.moe, moe.apply(model, args.moe, args.seed))
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}  params: {n_params / 1e6:.2f}M  train images: {n_train}  "
          f"batch: {bs}" + (f" ({ubs} unique x {K})" if K > 1 else "") + f"  steps/epoch: {steps_per_epoch}")

    opt = make_optimizer(model, args, fused=(device == "cuda"))
    sched = make_schedule(opt, args, steps_per_epoch)
    if args.model.startswith("rd_") and not args.no_compile:
        # variable r would recompile a whole-model graph per r: compile the pieces, loop in Python
        train_model = model.compile_parts(*(("reduce-overhead", ("core",)) if args.cudagraphs else ()))
        if args.cudagraphs:
            print("NOTE: rd_* model with --cudagraphs: only the core step is compiled (CUDA graphs, experimental)")
    else:
        train_model = model if args.no_compile else torch.compile(
            model, mode="reduce-overhead" if args.cudagraphs else None)

    # resume: ckpt.pt holds everything needed to continue exactly (model, opt, sched, RNGs, history)
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    history, best_val, start_epoch, elapsed, wb_id = [], 0.0, 1, 0.0, None
    if os.path.exists(ckpt_path) and not args.no_resume:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        history, best_val, elapsed, wb_id = ck["history"], ck["best_val"], ck["elapsed"], ck["wandb_id"]
        start_epoch = ck["epoch"] + 1
        torch.set_rng_state(ck["torch_rng"])
        if device == "cuda" and ck["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        rng.bit_generator.state = ck["np_rng"]
        print(f"Resumed from {ckpt_path}: continuing at epoch {start_epoch}/{args.epochs}")

    wb = None
    if args.wandb:
        import wandb

        wb_id = wb_id or uuid.uuid4().hex[:8]  # stable id so a resumed run continues the same wandb run
        wb = wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=run_name,
                        config=vars(args), dir=out_dir, id=wb_id, resume="allow")

    def save_ckpt(epoch):
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
              "epoch": epoch, "history": history, "best_val": best_val, "wandb_id": wb_id,
              "elapsed": elapsed + time.time() - t_start, "torch_rng": torch.get_rng_state(),
              "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else None,
              "np_rng": rng.bit_generator.state}
        torch.save(ck, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)

    t_start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        train_model.train()
        if stream:  # host->GPU streamed uint8 batches (background threads); restarts cleanly on resume
            batches = stream.epoch(epoch, ubs)
        else:
            perm = torch.randperm(n_train, device=device)
        loss_sum = torch.zeros((), device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        for k in range(steps_per_epoch):
            if stream:
                xb, y = next(batches)
            else:
                idx = perm[k * ubs : (k + 1) * ubs]
                xb, y = xtr[idx], ytr[idx]
            if K > 1:  # K consecutive copies of each image
                xb, y = xb.repeat_interleave(K, dim=0), y.repeat_interleave(K)
            x = xb.float().div_(255.0)
            if not args.no_aug:
                x = random_crop_flip(x, pad=img_size // 8)
            if args.randaug:
                x = rand_augment(x, args.randaug_n, args.randaug_m / 10.0)
            x = (x - mean) / std
            if args.reprob > 0:
                x = random_erasing(x, args.reprob)
            t = smooth_one_hot(y, num_classes, args.label_smoothing)
            if not (0 < args.mix_off_epoch <= epoch):
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
            gstep = (epoch - 1) * steps_per_epoch + k + 1
            if wb and args.log_every_steps > 0 and gstep % args.log_every_steps == 0:
                wb.log({"step_loss": loss.item(), "step_lr": sched.get_last_lr()[0]}, step=gstep)  # only sync here

        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        train_loss = loss_sum.item() / steps_per_epoch
        ips = steps_per_epoch * bs / dt
        lr_now = sched.get_last_lr()[0]
        rec = {"epoch": epoch, "train_loss": train_loss, "lr": lr_now, "epoch_sec": dt, "img_per_sec": ips}
        if args.moe:
            rec.update(moe.epoch_stats(model))

        line = (f"ep {epoch:4d}/{args.epochs} | loss {train_loss:.4f} | lr {lr_now:.2e} | "
                f"{dt:6.2f}s | {ips:9.0f} img/s")
        if stream:
            line += f" | loader wait {stream.wait_s:.1f}s"
            stream.wait_s = 0.0
        if epoch == 1 and not args.no_compile:
            line += "  (includes torch.compile warm-up)"

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val = evaluate(model, xte, yte, mean, std, amp, args.eval_bs)
            # clean (no-aug) accuracy on 10k training images: shows the train/val gap
            tr_acc = evaluate(model, xtr[clean_idx], ytr[clean_idx], mean, std, amp, args.eval_bs)
            best_val = max(best_val, val)
            rec.update(val_acc=val, train_acc_clean=tr_acc)
            line += f" | train acc {tr_acc:6.2f} | val acc {val:6.2f}"
        print(line, flush=True)
        history.append(rec)
        if wb:
            # with step logging on, wandb's step axis is the global step; else the epoch
            wb.log(rec, step=epoch * steps_per_epoch if args.log_every_steps > 0 else epoch)
        if (args.ckpt_every > 0 and epoch % args.ckpt_every == 0) or epoch == args.epochs:
            save_ckpt(epoch)

    total_min = (elapsed + time.time() - t_start) / 60
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
    if xtest is not None:
        summary["final_test_acc"] = evaluate(model, xtest, ytest, mean, std, amp, args.eval_bs)
        print(f"final test acc (scored once): {summary['final_test_acc']:.2f}%")
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    if not args.keep_ckpt and os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    if args.moe:
        moe.finalize(model, lambda xb: model((xb.float().div(255.0) - mean) / std), xte, out_dir, img_size,
                     extra={"moe_spec": args.moe, "best_val": best_val, "final_val": final.get("val_acc")})
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
