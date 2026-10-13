#!/usr/bin/env python3
"""
qat_int8.py - static INT8 for the 2.75M SL-ViT(+SPT+LSA+SeqPool) CIFAR-100 model: PTQ baselines, QAT, ONNX QDQ export.

Modes
  ort_ptq   reproduce the int8_audit.py static PTQ baseline on this checkpoint (ORT quantize_static, QDQ, per-channel
            int8 weights, int8 asymmetric activations; full default op set and MatMul/Gemm-only; MinMax/Percentile/Entropy),
            plus the audit's additive op-type ablation. Uses int8_audit.py's functions so numbers are comparable.
  qat       our own explicit quantizer placement (every tensor a fully-integer NPU would hold in int8, see QViT below),
            calibration-initialised scales (PTQ = --epochs 0), then QAT: LSQ/LSQ+ fake-quant (learned step size and
            offset, straight-through estimator), KD from the FP32 model, light augmentation. Exports an ONNX QDQ model
            (int8 weight initializers + DequantizeLinear, int32 bias, QuantizeLinear/DequantizeLinear on every quantized
            activation) and evaluates it with ONNX Runtime CPU on the full 10k test set + batch-1 latency.

Quantizer groups (--groups; default = all = "full static"):
  input    the normalised input image
  mm       inputs of every Linear/MatMul: LN outputs (SPT-LN, ln1, ln2, final norm), attention output (proj input),
           GELU output (fc1->fc2), q, k, v, softmax probs (attn@v), seqpool weights, pooled feature (head input)
  w        all Linear weights (per-output-channel symmetric int8), cls token + pos_embed (per-tensor int8); biases int32
  linout   outputs of every Linear (incl. GELU input = fc1 output, branch outputs before the residual add, logits)
  softmax  softmax inputs (q k^T logits, seqpool logits)
  residual the residual stream after every Add (incl. the pos-embed add)
Ops that compute in float with int8 inputs/outputs (QDQ "DQ -> op -> Q"): LayerNorm, Softmax(+diagonal mask), GELU.
--a16 <groups>: those activation groups use int16 instead of int8 (mixed precision).

Commands used for research/results_qat_int8.md (source env.sh first; ORT parts run on CPU):
  python qat_int8.py --mode ort_ptq                                  # ORT baseline + op-type ablations
  python qat_int8.py --mode ort_ptq --mask-ablation-only             # + 'full minus mask-Add/attn-Softmax'
  python qat_int8.py --mode fp32_explicit                            # FP32 ONNX of the same explicit graph
  python qat_int8.py --mode qat --epochs 0 --no-onnx --run-name ptq_full_mse [--groups ...] [--a16 residual]  # PTQ ablations
  python qat_int8.py --mode qat --epochs 0  --run-name qat_ptq_full_s0                     # PTQ + ONNX + ORT
  python qat_int8.py --mode qat --epochs 30 --run-name qat_full_e30                        # QAT, crop/flip KD
  python qat_int8.py --mode qat --epochs 10 --aug strong --run-name qat_full_strong_e10_s1 \
      --ckpt-run runs/c100_spt_slvit_1000ep_s1                                             # QAT, strong-aug KD
  python qat_int8.py --mode bench --bench fp32=runs/qat_int8_work/fp32.onnx qat=runs/qat_int8_work/qat_full_e30.onnx ...
"""
import argparse
import json
import math
import os
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ALL_GROUPS = ["input", "mm", "w", "linout", "softmax", "residual"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["ort_ptq", "qat", "bench", "fp32_explicit"], default="qat")
    p.add_argument("--ckpt-run", default=os.path.join(HERE, "runs", "c100_spt_slvit_1000ep"))
    p.add_argument("--run-name", default="qat_int8")
    p.add_argument("--out-dir", default=os.path.join(HERE, "runs"))
    p.add_argument("--work", default=os.path.join(HERE, "runs", "qat_int8_work"), help="onnx files (not committed)")
    # quantizer config
    p.add_argument("--groups", nargs="+", default=ALL_GROUPS)
    p.add_argument("--a16", nargs="*", default=[], help="activation groups kept at int16")
    p.add_argument("--calib", default="mse", choices=["minmax", "percentile", "mse"])
    p.add_argument("--pct", type=float, default=99.99)
    p.add_argument("--calib-n", type=int, default=512)
    p.add_argument("--no-learn-scales", action="store_true")
    # training
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-5, help="weights")
    p.add_argument("--qlr", type=float, default=1e-4, help="quantizer scales / offsets")
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--kd", type=float, default=1.0, help="weight of KL(teacher||student); 1-kd on CE")
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--aug", default="cropflip", choices=["none", "cropflip", "strong"],
                   help="strong = crop/flip + RandAugment(2, 9) + mixup/cutmix (KD targets come from the teacher on the same input)")
    p.add_argument("--freeze-scales-epoch", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=5)
    # onnx / ort
    p.add_argument("--no-onnx", action="store_true")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--lat-threads", type=int, default=4)
    p.add_argument("--keep-onnx", action="store_true")
    # ort_ptq
    p.add_argument("--ort-calibs", nargs="+", default=["minmax", "percentile", "entropy"])
    p.add_argument("--bench", nargs="*", default=[], help="bench mode: tag=path.onnx ...")
    p.add_argument("--bench-rounds", type=int, default=5)
    p.add_argument("--mask-ablation-only", action="store_true",
                   help="ort_ptq: only run the mask-path exclusion variants and merge them into the existing json")
    p.add_argument("--ort-additive", nargs="*", default=["Add", "Softmax", "LayerNormalization", "Mul", "Erf", "Div", "Where"])
    return p.parse_args()


# --------------------------------------------------------------------------- model loading
def load_fp32(run):
    import train_vit as T
    a = json.load(open(os.path.join(run, "results.json")))["args"]
    ns = argparse.Namespace(**a)
    ns.drop_path = 0.0
    torch.manual_seed(0)
    m = T.build_model(ns, 100, 32)
    m.load_state_dict(torch.load(os.path.join(run, "model.pt"), map_location="cpu"))
    return m.eval()


def load_np():
    import vit_data.cifar100 as C
    d = np.load(os.path.join(HERE, "data", "cifar100_uint8.npz"))
    return d["xtrain"], d["ytrain"], d["xtest"], d["ytest"], C.MEAN, C.STD


# --------------------------------------------------------------------------- quantizers
def round_ste(x):
    return (x.round() - x).detach() + x


def grad_scale(x, g):
    return (x - x * g).detach() + x * g


class _QDQ(torch.autograd.Function):
    """Export-time activation fake quant -> ONNX QuantizeLinear + DequantizeLinear (int16: com.microsoft domain)."""

    @staticmethod
    def forward(ctx, x, scale, zp, bits):
        qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        return (torch.clamp(torch.round(x / scale) + zp, qmin, qmax) - zp) * scale

    @staticmethod
    def symbolic(g, x, scale, zp, bits):
        s = g.op("Constant", value_t=torch.tensor(scale, dtype=torch.float32))
        z = g.op("Constant", value_t=torch.tensor(zp, dtype=torch.int8 if bits == 8 else torch.int16))
        dom = "" if bits == 8 else "com.microsoft::"
        return g.op(dom + "DequantizeLinear", g.op(dom + "QuantizeLinear", x, s, z), s, z)


class _DQ(torch.autograd.Function):
    """Export-time dequantization of an integer constant (int8 weights, axis = channel; int32 bias)."""

    @staticmethod
    def forward(ctx, q, scale, axis):
        shape = [1] * q.dim()
        if scale.numel() > 1:
            shape[axis] = -1
        return q.float() * scale.view(shape)

    @staticmethod
    def symbolic(g, q, scale, axis):
        zdt = torch.int8 if q.type().scalarType() == "Char" else torch.int32
        n = scale.type().sizes()[0] if scale.type().sizes() else 1
        z = g.op("Constant", value_t=torch.zeros(n, dtype=zdt) if n > 1 else torch.tensor(0, dtype=zdt))
        if n > 1:
            return g.op("DequantizeLinear", q, scale, z, axis_i=axis)
        return g.op("DequantizeLinear", q, scale, z)


class ActQ(nn.Module):
    """Per-tensor asymmetric fake quantizer with learnable step size s and offset z (LSQ / LSQ+)."""

    def __init__(self, name, group, bits=8):
        super().__init__()
        self.name, self.group, self.bits = name, group, bits
        self.qmin, self.qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.zp = nn.Parameter(torch.tensor(0.0))
        self.mode = "off"  # off | calib | quant | export
        self.samples = []

    def forward(self, x):
        if self.mode == "off":
            return x
        if self.mode == "calib":
            f = x.detach().float().flatten()
            if f.numel() > 200_000:
                f = f[torch.randint(0, f.numel(), (200_000,), device=f.device)]
            self.samples.append(f)
            return x
        if self.mode == "export":
            return _QDQ.apply(x, float(self.scale.abs()), int(self.zp.round().clamp(self.qmin, self.qmax)), self.bits)
        g = 1.0 / math.sqrt(x[0].numel() * self.qmax)
        s = grad_scale(self.scale.abs() + 1e-12, g)
        z = round_ste(grad_scale(self.zp, g)).clamp(self.qmin, self.qmax)
        q = torch.clamp(round_ste(x / s) + z, self.qmin, self.qmax)
        return (q - z) * s

    @torch.no_grad()
    def calibrate(self, method, pct):
        v = torch.cat(self.samples).float()
        self.samples = []
        lo, hi = v.min(), v.max()
        if method == "percentile":
            k = v[torch.randperm(v.numel(), device=v.device)[:1_000_000]]
            lo, hi = torch.quantile(k, 1 - pct / 100), torch.quantile(k, pct / 100)
        lo, hi = min(float(lo), 0.0), max(float(hi), 0.0)  # 0 must be representable (zero padding)
        if method == "mse" and self.bits == 8:
            best = None
            for frac in np.linspace(0.05, 1.0, 40):
                for lfrac in ([frac] if lo < 0 else [1.0]):
                    l, h = lo * lfrac, hi * frac
                    s, z = self._sz(l, h)
                    err = float((((torch.clamp(torch.round(v / s) + z, self.qmin, self.qmax) - z) * s - v) ** 2).mean())
                    if best is None or err < best[0]:
                        best = (err, l, h)
            lo, hi = best[1], best[2]
        s, z = self._sz(lo, hi)
        self.scale.fill_(s)
        self.zp.fill_(z)

    def _sz(self, lo, hi):
        s = max((hi - lo) / (self.qmax - self.qmin), 1e-8)
        z = float(np.clip(round(self.qmin - lo / s), self.qmin, self.qmax))
        return s, z


class QLinear(nn.Module):
    """Linear with per-output-channel symmetric int8 weights (LSQ step size) and bias rounded to the int32 grid
    s_in * s_w (Jacob et al. 2018). x_q is the (already fake-quantized) input; its quantizer gives s_in."""

    def __init__(self, lin, wq_on, name):
        super().__init__()
        self.weight, self.bias, self.name = lin.weight, lin.bias, name
        self.wq_on = wq_on
        self.qmax = 127
        self.w_scale = nn.Parameter(lin.weight.detach().abs().amax(1).clamp_min(1e-8) / 127)
        self.mode = "off"

    @torch.no_grad()
    def prepare_export(self, in_q):
        s = self.w_scale.detach().abs().clamp_min(1e-12).float().clone()
        self._wq = torch.clamp(torch.round(self.weight.detach() / s[:, None]), -127, 127).to(torch.int8).t().contiguous()
        self._ws, self._bq, self._bf = s, None, None
        if self.bias is not None:
            if in_q is not None and in_q.on:
                self._bs = (s * in_q.scale.detach().abs()).float().clone()
                self._bq = torch.round(self.bias.detach() / self._bs).to(torch.int32)
            else:
                self._bf = self.bias.detach().clone()

    @torch.no_grad()
    def calibrate_mse(self):
        w = self.weight.detach()
        amax = w.abs().amax(1, keepdim=True).clamp_min(1e-8)
        best_err, best_s = None, None
        for frac in np.linspace(0.5, 1.0, 26):
            s = amax * frac / 127
            err = ((torch.clamp(torch.round(w / s), -127, 127) * s - w) ** 2).mean(1)
            if best_err is None:
                best_err, best_s = err, s.squeeze(1).clone()
            else:
                better = err < best_err
                best_err = torch.where(better, err, best_err)
                best_s = torch.where(better, s.squeeze(1), best_s)
        self.w_scale.copy_(best_s)

    def forward(self, x, in_q):
        if self.mode == "off" or not self.wq_on:
            return F.linear(x, self.weight, self.bias)
        quant_in = in_q is not None and in_q.mode in ("quant", "export")
        if self.mode == "export":  # integer tensors precomputed by prepare_export() -> traced as int initializers
            y = torch.matmul(x, _DQ.apply(self._wq, self._ws, 1))  # W^T (in, out), per-column scale, axis 1
            if self.bias is not None:
                y = y + (_DQ.apply(self._bq, self._bs, 0) if self._bq is not None else self._bf)
            return y
        g = 1.0 / math.sqrt(self.weight.shape[1] * self.qmax)
        s = grad_scale(self.w_scale.abs() + 1e-12, g)[:, None]
        w = torch.clamp(round_ste(self.weight / s), -127, 127) * s
        b = self.bias
        if b is not None and quant_in:
            sb = s[:, 0].detach() * in_q.scale.detach().abs()
            b = round_ste(b / sb) * sb
        return F.linear(x, w, b)


class ConstQ(nn.Module):
    """Per-tensor symmetric int8 quantization of a constant tensor (cls token, pos embedding)."""

    def __init__(self, p, on):
        super().__init__()
        self.p, self.on = p, on
        self.scale = nn.Parameter(p.detach().abs().max() / 127)
        self.mode = "off"

    @torch.no_grad()
    def prepare_export(self):
        self._s = self.scale.detach().abs().view(1).clone()
        self._q = torch.clamp(torch.round(self.p.detach() / self._s), -127, 127).to(torch.int8)

    def forward(self):
        if self.mode == "off" or not self.on:
            return self.p
        if self.mode == "export":
            return _DQ.apply(self._q, self._s, 0)
        s = self.scale.abs() + 1e-12
        return torch.clamp(round_ste(self.p / s), -127, 127) * s


# --------------------------------------------------------------------------- quantized model
class QViT(nn.Module):
    """Explicit forward of the timm VisionTransformer (SPT patch embed, LSA attention, SeqPool head) with a fake
    quantizer on every activation tensor of a fully-int8 deployment. The LSA temperature is folded into the q rows
    of qkv (exact). Parameters are shared with `m` (a private copy)."""

    def __init__(self, m, groups, a16=()):
        super().__init__()
        self.m = m
        self.groups = set(groups)
        self.D, self.H = m.embed_dim, m.blocks[0].attn.num_heads
        self.hd = self.D // self.H
        with torch.no_grad():  # fold the per-head LSA temperature into W_q, b_q
            for b in m.blocks:
                a = b.attn
                t = a.lsa_temperature.detach().repeat_interleave(self.hd)
                a.qkv.weight[:self.D] *= t[:, None]
                a.qkv.bias[:self.D] *= t
                a.lsa_temperature.fill_(1.0)
        self.aq = nn.ModuleDict()
        self.lin = nn.ModuleDict()
        self.a16 = set(a16)

        def A(name, group):
            q = ActQ(name, group, 16 if group in self.a16 else 8)
            q.on = group in self.groups
            self.aq[name] = q
            return q
        won = "w" in self.groups
        self.qin = A("input", "input")
        pe = m.patch_embed.proj
        self.spt_ln = A("spt_ln", "mm")
        self.lin["spt"] = QLinear(pe.proj, won, "spt")
        self.spt_out = A("spt_out", "linout")
        self.cls_q, self.pos_q = ConstQ(m.cls_token, won), ConstQ(m.pos_embed, won)
        self.res0 = A("res0", "residual")
        for i, b in enumerate(m.blocks):
            for n, g in [("ln1", "mm"), ("q", "mm"), ("k", "mm"), ("v", "mm"), ("logit", "softmax"), ("prob", "mm"),
                         ("ao", "mm"), ("po", "linout"), ("r1", "residual"), ("ln2", "mm"), ("h", "linout"),
                         ("g", "mm"), ("fo", "linout"), ("r2", "residual")]:
                A(f"b{i}_{n}", g)
            self.lin[f"b{i}_qkv"] = QLinear(b.attn.qkv, won, f"b{i}_qkv")
            self.lin[f"b{i}_proj"] = QLinear(b.attn.proj, won, f"b{i}_proj")
            self.lin[f"b{i}_fc1"] = QLinear(b.mlp.fc1, won, f"b{i}_fc1")
            self.lin[f"b{i}_fc2"] = QLinear(b.mlp.fc2, won, f"b{i}_fc2")
        A("norm", "mm")
        A("sp_logit", "softmax")
        A("sp_w", "mm")
        A("pooled", "mm")
        A("logits", "linout")
        self.lin["sp"] = QLinear(m.attn_pool.attention_pool, won, "sp")
        self.lin["head"] = QLinear(m.head, won, "head")
        self.mask_value = -1e4  # finite (ORT-friendly); exp(-1e4) == 0 in fp32, same as -inf

    def set_mode(self, mode):
        for q in self.aq.values():
            q.mode = mode if q.on else "off"
        for q in list(self.lin.values()) + [self.cls_q, self.pos_q]:
            q.mode = mode if mode != "calib" else "off"

    def prepare_export(self):
        ins = {"spt": "spt_ln", "sp": "norm", "head": "pooled"}
        for n, l in self.lin.items():
            i = ins.get(n) or n.split("_")[0] + "_" + {"qkv": "ln1", "proj": "ao", "fc1": "ln2", "fc2": "g"}[n.split("_")[1]]
            l.prepare_export(self.aq[i])
        self.cls_q.prepare_export()
        self.pos_q.prepare_export()

    def Q(self, name, x):
        return self.aq[name](x)

    def lin_(self, name, x, in_name):
        return self.lin[name](x, self.aq[in_name] if self.aq[in_name].on else None)

    def forward(self, x):
        m = self.m
        x = self.Q("input", x)
        B, C, Hh, Ww = x.shape
        pe = m.patch_embed.proj
        s, p = pe.s, pe.p
        xp = F.pad(x, (s, s, s, s))
        sh = [xp[:, :, s + dy:s + dy + Hh, s + dx:s + dx + Ww] for dx, dy in ((-s, -s), (s, -s), (-s, s), (s, s))]
        x = torch.cat([x] + sh, dim=1)  # (B, 5C, H, W)
        gh, gw = Hh // p, Ww // p
        x = x.reshape(B, 5 * C, gh, p, gw, p).permute(0, 2, 4, 1, 3, 5).reshape(B, gh * gw, 5 * C * p * p)  # == unfold
        x = self.Q("spt_ln", pe.norm(x))
        x = self.Q("spt_out", self.lin_("spt", x, "spt_ln"))
        x = torch.cat([self.cls_q().expand(B, -1, -1), x], dim=1) + self.pos_q()
        x = self.Q("res0", x)
        N = x.shape[1]
        i_ = torch.arange(N, device=x.device)
        diag = (i_[:, None] == i_[None, :])
        for i, b in enumerate(m.blocks):
            P = f"b{i}_"
            h = self.Q(P + "ln1", b.norm1(x))
            qkv = self.lin_(P + "qkv", h, P + "ln1").reshape(B, N, 3, self.H, self.hd).permute(2, 0, 3, 1, 4)
            q, k, v = self.Q(P + "q", qkv[0]), self.Q(P + "k", qkv[1]), self.Q(P + "v", qkv[2])
            att = self.Q(P + "logit", q @ k.transpose(-2, -1))
            att = att.masked_fill(diag, self.mask_value).softmax(-1)
            att = self.Q(P + "prob", att)
            o = self.Q(P + "ao", (att @ v).transpose(1, 2).reshape(B, N, self.D))
            o = self.Q(P + "po", self.lin_(P + "proj", o, P + "ao"))
            x = self.Q(P + "r1", x + o)
            h = self.Q(P + "ln2", b.norm2(x))
            h = self.Q(P + "h", self.lin_(P + "fc1", h, P + "ln2"))
            h = self.Q(P + "g", F.gelu(h))
            h = self.Q(P + "fo", self.lin_(P + "fc2", h, P + "g"))
            x = self.Q(P + "r2", x + h)
        x = self.Q("norm", m.norm(x))[:, m.num_prefix_tokens:]
        w = self.Q("sp_logit", self.lin_("sp", x, "norm")).softmax(1)
        w = self.Q("sp_w", w)
        z = self.Q("pooled", (w.transpose(1, 2) @ x).squeeze(1))
        return self.Q("logits", self.lin_("head", z, "pooled"))

    def float_ops(self):
        off = sorted({q.group for q in self.aq.values() if not q.on})
        return off


# --------------------------------------------------------------------------- data / eval
def normalize(xu8, mean, std):
    return (xu8.float().div_(255.0) - mean) / std


@torch.no_grad()
def evaluate(model, xte, yte, mean, std, bs=1000):
    model.eval()
    c = 0
    for k in range(0, len(xte), bs):
        c += int((model(normalize(xte[k:k + bs], mean, std)).argmax(1) == yte[k:k + bs]).sum())
    return 100.0 * c / len(xte)


def crop_flip(x, pad=4):
    B, C, H, W = x.shape
    xp = F.pad(x, (pad,) * 4, mode="reflect")
    i = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    j = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    r = (i[:, None] + torch.arange(H, device=x.device))[:, None, :, None].expand(B, C, H, W + 2 * pad)
    xr = xp.gather(2, r)
    c = (j[:, None] + torch.arange(W, device=x.device))[:, None, None, :].expand(B, C, H, W)
    x = xr.gather(3, c)
    flip = torch.rand(B, device=x.device) < 0.5
    return torch.where(flip[:, None, None, None], x.flip(3), x)


def strong_aug(xu8):
    """RandAugment (train_vit.rand_augment, n=2, m=9) then per-batch mixup (alpha .8) or cutmix (alpha 1), 50/50.
    Returns float [0,1]. Labels are not needed: the loss is KD against the FP32 teacher on the same images."""
    import train_vit as T
    x = T.rand_augment(xu8.float().div_(255.0), 2, 0.9)
    perm = torch.randperm(x.size(0), device=x.device)
    if np.random.rand() < 0.5:
        lam = float(np.random.beta(0.8, 0.8))
        return lam * x + (1 - lam) * x[perm]
    lam = float(np.random.beta(1.0, 1.0))
    H, W = x.shape[-2:]
    rh, rw = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
    cy, cx = np.random.randint(H), np.random.randint(W)
    y0, y1, x0, x1 = max(cy - rh // 2, 0), min(cy + rh // 2, H), max(cx - rw // 2, 0), min(cx + rw // 2, W)
    x = x.clone()
    x[:, :, y0:y1, x0:x1] = x[perm][:, :, y0:y1, x0:x1]
    return x


@torch.no_grad()
def predict(model, xte, mean, std, bs=1000):
    model.eval()
    return torch.cat([model(normalize(xte[k:k + bs], mean, std)).argmax(1) for k in range(0, len(xte), bs)]).cpu().numpy()


def ort_predict(path, xte_np, mean, std, threads, bs=500):
    import int8_audit as IA
    sess = IA.session(path, threads)
    return np.concatenate([sess.run(None, {"input": IA.prep(xte_np[k:k + bs], 32, mean, std)})[0].argmax(1)
                           for k in range(0, len(xte_np), bs)])


# --------------------------------------------------------------------------- ORT helpers
def ort_eval(path, xte, yte, mean, std, args):
    import int8_audit as IA
    return {"acc": IA.accuracy(path, xte, yte, 32, mean, std, args.threads, 500),
            "lat_ms": IA.latency(path, 32, args.lat_threads, warmup=20, runs=300), "MB": IA.size_mb(path)}


def export_qdq(qm, path):
    qm = qm.cpu().eval()
    qm.set_mode("export")
    qm.prepare_export()
    x = torch.randn(1, 3, 32, 32)
    if True:  # no torch.no_grad(): under no_grad the TorchScript tracer inlines autograd.Functions (symbolic ignored)
        torch.onnx.export(qm, (x,), path, opset_version=17, input_names=["input"], output_names=["logits"],
                          dynamo=False, do_constant_folding=True,
                          dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, autograd_inlining=False)
    qm.set_mode("quant")


# --------------------------------------------------------------------------- modes
def _mask_ablation(rec, pp, calib_x, xte, yte, mean, std, args):
    """Full default static op set, but leave float only the LSA diagonal-mask Add (logits + Where(mask, 0, -3e4))
    and the attention Softmax that consumes it. Tests whether the full-PTQ collapse is the mask artifact."""
    import int8_audit as IA
    import onnx
    g = onnx.load(pp).graph
    prod = {o: n for n in g.node for o in n.output}
    mask_add = [prod[n.input[0]].name for n in g.node if n.op_type == "Softmax" and "attn_pool" not in n.name]
    attn_sm = [n.name for n in g.node if n.op_type == "Softmax" and "attn_pool" not in n.name]
    for cm in args.ort_calibs:
        for tag, ex in [("full_minus_maskAdd_attnSoftmax", mask_add + attn_sm)]:
            qp = os.path.join(args.work, f"ptq_{tag}_{cm}.onnx")
            IA.static_quant(pp, qp, calib_x, cm, exclude_names=ex)
            rec["variants"][f"static_{tag}_{cm}"] = ort_eval(qp, xte, yte, mean, std, args) | {"ops": IA.op_hist(qp)}
            print(tag, cm, rec["variants"][f"static_{tag}_{cm}"]["acc"], flush=True)


def mode_ort_ptq(args):
    import int8_audit as IA
    if args.mask_ablation_only:
        _, _, xte, yte, mean, std = load_np()
        _, _, xtr = IA.load_data(0)
        out = os.path.join(HERE, "research", "qat_int8_ort_ptq.json")
        rec = json.load(open(out))
        _mask_ablation(rec, os.path.join(args.work, "fp32_pp.onnx"), IA.prep(xtr, 32, mean, std), xte, yte, mean, std, args)
        json.dump(rec, open(out + ".tmp", "w"), indent=1)
        os.replace(out + ".tmp", out)
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(args.threads)
    _, _, xte, yte, mean, std = load_np()
    _, _, xtr = IA.load_data(0)  # same 512 fixed-seed calibration images as the audit
    os.makedirs(args.work, exist_ok=True)
    m = load_fp32(args.ckpt_run)
    fp = os.path.join(args.work, "fp32.onnx")
    IA.export_fp32(m, 32, fp)
    with torch.no_grad():
        xp = torch.from_numpy(IA.prep(xte[:8], 32, mean, std))
        parity = float(np.abs(IA.session(fp, 1).run(None, {"input": xp.numpy()})[0] - m(xp).numpy()).max())
    rec = {"ckpt": args.ckpt_run, "onnx_parity_maxdiff": parity, "fp32": ort_eval(fp, xte, yte, mean, std, args),
           "ops_fp32": IA.op_hist(fp)}
    print("fp32", rec["fp32"], flush=True)
    pp = os.path.join(args.work, "fp32_pp.onnx")
    IA.preprocess_onnx(fp, pp)
    calib_x = IA.prep(xtr, 32, mean, std)
    v = {}
    dp = os.path.join(args.work, "dyn.onnx")
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(fp, dp, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])
    v["dynamic_matmul_gemm"] = ort_eval(dp, xte, yte, mean, std, args)
    print("dynamic", v["dynamic_matmul_gemm"], flush=True)
    for cm in args.ort_calibs:
        for tag, ops in [("full", None), ("matmul_gemm_only", ["MatMul", "Gemm"])]:
            qp = os.path.join(args.work, f"ptq_{tag}_{cm}.onnx")
            IA.static_quant(pp, qp, calib_x, cm, op_types=ops)
            v[f"static_{tag}_{cm}"] = ort_eval(qp, xte, yte, mean, std, args) | {"ops": IA.op_hist(qp)}
            print(tag, cm, {k: v[f"static_{tag}_{cm}"][k] for k in ("acc", "lat_ms", "MB")}, flush=True)
    rec["variants"] = v
    _mask_ablation(rec, pp, calib_x, xte, yte, mean, std, args)
    best = max([k for k in v if k.startswith("static_matmul")], key=lambda k: v[k]["acc"])
    cm = best.rsplit("_", 1)[1]
    ab = {}
    for t in args.ort_additive:
        qp = os.path.join(args.work, f"ab_{t}.onnx")
        IA.static_quant(pp, qp, calib_x, cm, op_types=["MatMul", "Gemm", t])
        ab[f"matmul_gemm+{t}"] = ort_eval(qp, xte, yte, mean, std, args)
        print("additive", t, ab[f"matmul_gemm+{t}"], flush=True)
        os.remove(qp)
    for t in ["Add", "Softmax"]:
        qp = os.path.join(args.work, f"abx_{t}.onnx")
        IA.static_quant(pp, qp, calib_x, cm, exclude_types=(t,))
        ab[f"full_minus_{t}"] = ort_eval(qp, xte, yte, mean, std, args)
        print("full minus", t, ab[f"full_minus_{t}"], flush=True)
        os.remove(qp)
    qp = os.path.join(args.work, "abx_add_softmax.onnx")
    IA.static_quant(pp, qp, calib_x, cm, exclude_types=("Add", "Softmax"))
    ab["full_minus_Add_Softmax"] = ort_eval(qp, xte, yte, mean, std, args)
    print("full minus add+softmax", ab["full_minus_Add_Softmax"], flush=True)
    rec["ablation"] = {"calib": cm, "results": ab}
    rec["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out = os.path.join(HERE, "research", "qat_int8_ort_ptq.json")
    json.dump(rec, open(out + ".tmp", "w"), indent=1)
    os.replace(out + ".tmp", out)
    print("wrote", out)


def mode_qat(args):
    import copy
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xtr_np, ytr_np, xte_np, yte_np, MEAN, STD = load_np()
    to = lambda a: torch.from_numpy(a).permute(0, 3, 1, 2).contiguous().to(dev)
    xtr, xte = to(xtr_np), to(xte_np)
    ytr, yte = torch.from_numpy(ytr_np).long().to(dev), torch.from_numpy(yte_np).long().to(dev)
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    teacher = load_fp32(args.ckpt_run).to(dev)
    for p_ in teacher.parameters():
        p_.requires_grad_(False)
    qm = QViT(copy.deepcopy(teacher), args.groups, args.a16).to(dev)
    rec = {"run_name": args.run_name, "args": vars(args), "ckpt": args.ckpt_run, "float_groups": qm.float_ops(),
           "n_act_quantizers_on": sum(q.on for q in qm.aq.values())}
    rec["fp32_acc_teacher"] = evaluate(teacher, xte, yte, mean, std)
    t_pred = predict(teacher, xte, mean, std)
    qm.set_mode("off")
    rec["fp32_acc_qvit_off"] = evaluate(qm, xte, yte, mean, std)  # parity of our explicit forward
    print("fp32", rec["fp32_acc_teacher"], "explicit forward", rec["fp32_acc_qvit_off"], flush=True)
    # calibration: same 512 fixed-seed train images as int8_audit (np RandomState(0))
    ci = np.random.RandomState(0).choice(len(xtr_np), 512, replace=False)[:args.calib_n]
    qm.set_mode("calib")
    with torch.no_grad():
        for k in range(0, len(ci), 128):
            qm(normalize(xtr[ci[k:k + 128]], mean, std))
    with torch.no_grad():
        for q in qm.aq.values():
            if q.on:
                q.calibrate(args.calib, args.pct)
        if args.calib == "mse":
            for l in qm.lin.values():
                l.calibrate_mse()
    qm.set_mode("quant")
    rec["ptq_fakequant_acc"] = evaluate(qm, xte, yte, mean, std)
    rec["ptq_fakequant_agree_fp32"] = 100.0 * float((predict(qm, xte, mean, std) == t_pred).mean())
    print("PTQ (fake-quant, our placement)", rec["ptq_fakequant_acc"], flush=True)
    hist = []
    if args.epochs > 0:
        qparams = [p_ for n, p_ in qm.named_parameters() if n.endswith(("scale", "zp", "w_scale"))]
        qids = {id(p_) for p_ in qparams}
        wparams = [p_ for p_ in qm.parameters() if id(p_) not in qids and p_.requires_grad]
        if args.no_learn_scales:
            for p_ in qparams:
                p_.requires_grad_(False)
        opt = torch.optim.AdamW([{"params": wparams, "lr": args.lr, "weight_decay": args.wd},
                                 {"params": qparams, "lr": args.qlr, "weight_decay": 0.0}])
        steps_ep = len(xtr) // args.bs
        total = steps_ep * args.epochs
        base = [g["lr"] for g in opt.param_groups]
        step = 0
        for ep in range(args.epochs):
            if ep == args.freeze_scales_epoch:
                for p_ in qparams:
                    p_.requires_grad_(False)
            qm.train()
            perm = torch.randperm(len(xtr), device=dev)
            t0, tl = time.time(), 0.0
            for it in range(steps_ep):
                idx = perm[it * args.bs:(it + 1) * args.bs]
                xb = xtr[idx]
                if args.aug in ("cropflip", "strong"):
                    xb = crop_flip(xb)
                if args.aug == "strong":
                    xb = strong_aug(xb)
                    xb = (xb - mean) / std
                else:
                    xb = normalize(xb, mean, std)
                f = 0.5 * (1 + math.cos(math.pi * step / total))
                for g_, b_ in zip(opt.param_groups, base):
                    g_["lr"] = b_ * f
                out = qm(xb)
                with torch.no_grad():
                    tlog = teacher(xb)
                loss = args.kd * F.kl_div(F.log_softmax(out, -1), F.log_softmax(tlog, -1), log_target=True,
                                          reduction="batchmean")
                if args.kd < 1:
                    loss = loss + (1 - args.kd) * F.cross_entropy(out, ytr[idx], label_smoothing=args.label_smoothing)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(qm.parameters(), 1.0)
                opt.step()
                tl += float(loss)
                step += 1
            e = {"epoch": ep + 1, "loss": tl / steps_ep, "sec": time.time() - t0}
            if (ep + 1) % args.eval_every == 0 or ep + 1 == args.epochs:
                e["acc"] = evaluate(qm, xte, yte, mean, std)
            hist.append(e)
            print(e, flush=True)
    rec["history"] = hist
    rec["qat_fakequant_acc"] = evaluate(qm, xte, yte, mean, std)
    rec["qat_fakequant_agree_fp32"] = 100.0 * float((predict(qm, xte, mean, std) == t_pred).mean())
    print("final fake-quant", rec["qat_fakequant_acc"], flush=True)
    out_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(qm.state_dict(), os.path.join(out_dir, "qvit.pt"))
    rec["scales"] = {n: {"scale": float(q.scale.abs()), "zp": int(q.zp.round()), "bits": q.bits}
                     for n, q in qm.aq.items() if q.on}
    if not args.no_onnx:
        os.makedirs(args.work, exist_ok=True)
        path = os.path.join(args.work, f"{args.run_name}.onnx")
        export_qdq(qm, path)
        with torch.no_grad():  # parity: ORT vs fake-quant torch on CPU
            import int8_audit as IA
            xs = torch.from_numpy(IA.prep(xte_np[:256], 32, MEAN, STD))
            ref = qm(xs).numpy()
            o = IA.session(path, 4).run(None, {"input": xs.numpy()})[0]
            rec["onnx_vs_fakequant_maxdiff"] = float(np.abs(o - ref).max())
            rec["onnx_vs_fakequant_argmax_agree"] = float((o.argmax(1) == ref.argmax(1)).mean())
        torch.set_num_threads(args.threads)
        rec["ort_int8"] = ort_eval(path, xte_np, yte_np, MEAN, STD, args)
        rec["ort_ops"] = IA.op_hist(path)
        rec["ort_int8"]["agree_fp32"] = 100.0 * float((ort_predict(path, xte_np, MEAN, STD, args.threads) == t_pred).mean())
        print("ORT int8", rec["ort_int8"], "parity", rec["onnx_vs_fakequant_maxdiff"], flush=True)
        if not args.keep_onnx and "full" not in args.run_name and "best" not in args.run_name:
            pass  # onnx files live in --work (gitignored runs/), kept for inspection
    rec["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    pth = os.path.join(out_dir, "results.json")
    json.dump(rec, open(pth + ".tmp", "w"), indent=1)
    os.replace(pth + ".tmp", pth)
    print("wrote", pth)


def mode_bench(args):
    """Interleaved batch-1 CPU latency of several ONNX files (rounds x median-of-300), to reduce load drift between
    models on the shared box. Also records the ORT-optimised op histogram (which QDQ groups got fused to int8 kernels)."""
    import int8_audit as IA
    import onnxruntime as ort
    res = {}
    paths = dict(t.split("=", 1) for t in args.bench)
    for tag, pth in paths.items():
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        so.optimized_model_filepath = os.path.join(args.work, f"bench_{tag}_opt.onnx")
        ort.InferenceSession(pth, so, providers=["CPUExecutionProvider"])
        res[tag] = {"MB": IA.size_mb(pth), "lat_ms": [], "optimized_ops": IA.op_hist(so.optimized_model_filepath)}
    for r in range(args.bench_rounds):
        for tag, pth in paths.items():
            res[tag]["lat_ms"].append(IA.latency(pth, 32, args.lat_threads, warmup=20, runs=300))
        print(r, {t: round(v["lat_ms"][-1], 3) for t, v in res.items()}, flush=True)
    for v in res.values():
        v["lat_ms_median"] = statistics.median(v["lat_ms"])
    res["_meta"] = {"threads": args.lat_threads, "loadavg": os.getloadavg(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    out = os.path.join(HERE, "research", "qat_int8_latency.json")
    json.dump(res, open(out + ".tmp", "w"), indent=1)
    os.replace(out + ".tmp", out)
    print(json.dumps({t: (v.get("lat_ms_median"), v.get("MB")) for t, v in res.items() if t != "_meta"}))


def mode_fp32_explicit(args):
    """FP32 ONNX of the same explicit forward (QViT with all quantizers off) = latency control for the QDQ export,
    whose graph is simpler than the timm/SDPA export (no IsNaN/Not/Range mask plumbing)."""
    import copy
    import int8_audit as IA
    m = load_fp32(args.ckpt_run)
    qm = QViT(copy.deepcopy(m), args.groups).eval()
    qm.set_mode("off")
    os.makedirs(args.work, exist_ok=True)
    path = os.path.join(args.work, "fp32_explicit.onnx")
    torch.onnx.export(qm, (torch.randn(1, 3, 32, 32),), path, opset_version=17, input_names=["input"],
                      output_names=["logits"], dynamo=False, do_constant_folding=True,
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    _, _, xte, yte, mean, std = load_np()
    r = ort_eval(path, xte, yte, mean, std, args)
    out = os.path.join(HERE, "research", "qat_int8_ort_ptq.json")
    rec = json.load(open(out))
    rec["fp32_explicit"] = r
    json.dump(rec, open(out + ".tmp", "w"), indent=1)
    os.replace(out + ".tmp", out)
    print("fp32 explicit", r)


if __name__ == "__main__":
    a = parse_args()
    {"ort_ptq": mode_ort_ptq, "qat": mode_qat, "bench": mode_bench, "fp32_explicit": mode_fp32_explicit}[a.mode](a)
