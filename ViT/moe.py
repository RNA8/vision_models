"""
moe.py -- conditional-parameter ("mixture of experts") conversions for the edge track.

  apply(model, spec, **kw) -> info dict       in-place conversion, spec strings:
    "cc:k=4,stages=3"                CondConv/DynamicConv experts (per-image softmax(fc(GAP(x))/tau) mix of K kernels)
                                     in every ConvNorm conv of the given RepViT stages (timm repvit); experts are
                                     copies of the pretrained kernel + noise*std noise (upcycling)
    "mlp:e=4,last=4"                 per-image top-1 routed MLP experts (+ shared expert, each half the dense hidden
                                     width => active params/MACs ~ dense) in the last N timm ViT blocks, from scratch
    "upcycle:e=4,last=4"             copy each pretrained MLP of the last N blocks into E full-size experts (top-1,
                                     no shared expert, small random router), forward gate = 1 (straight-through)
  set_tau(model, tau), epoch_stats(model, reset), aux gradient is injected inside the forward through an identity
  autograd Function (AuxAttach), so no change to the training loop's loss is needed (torch.compile friendly).
  profile(model, x, ...) -> active/total params, MACs (FlopCounterMode, batch 1, routed path), CPU latency.
  finalize(model, fwd, xte, out_dir, extra) -> runs/<name>/moe_stats.json (load fractions/entropy per layer, profile)
"""
import copy
import json
import math
import os
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- aux-loss injection
class AuxAttach(torch.autograd.Function):
    """forward: identity on x (aux is only a graph input); backward: gradient `w` into aux, i.e. adds w*aux to the loss."""

    @staticmethod
    def forward(ctx, x, aux, w):
        ctx.w = w
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g, torch.full((), ctx.w, device=g.device, dtype=g.dtype), None


# --------------------------------------------------------------------------- CondConv
class CondConv2d(nn.Module):
    def __init__(self, conv: nn.Conv2d, K=4, noise=0.05, tau=30.0):
        super().__init__()
        assert conv.bias is None and conv.padding_mode == "zeros"
        self.K, self.stride, self.padding, self.dilation, self.groups = K, conv.stride, conv.padding, conv.dilation, conv.groups
        self.in_ch, self.out_ch = conv.in_channels, conv.out_channels
        w = conv.weight.data
        std = w.std()
        self.weight = nn.Parameter(w.unsqueeze(0).repeat(K, *([1] * w.ndim)) + noise * std * torch.randn(K, *w.shape))
        self.fc = nn.Linear(self.in_ch, K)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)
        self.register_buffer("tau", torch.tensor(float(tau)))
        self.register_buffer("count", torch.zeros(K))  # sum of routing probs (soft load), reset by epoch_stats
        self.register_buffer("hard", torch.zeros(K))   # argmax counts
        self.register_buffer("n", torch.zeros(()))
        self.pw = w.shape[-1] == 1 and w.shape[-2] == 1 and conv.groups == 1 and conv.stride == (1, 1)

    def route(self, x):
        g = self.fc(x.float().mean((2, 3)))  # (B, K)
        r = F.softmax(g / self.tau, dim=-1)
        with torch.no_grad():
            self.count.add_(r.detach().sum(0))
            self.hard.add_(F.one_hot(r.argmax(-1), self.K).float().sum(0))
            self.n.add_(r.shape[0])
        return r

    def forward(self, x):
        B, C, H, W = x.shape
        r = self.route(x).to(x.dtype if not torch.is_autocast_enabled() else torch.bfloat16)
        wk = self.weight
        if torch.is_autocast_enabled():
            wk = wk.to(torch.bfloat16)
        if self.pw:
            wb = torch.einsum("bk,koi->boi", r, wk[:, :, :, 0, 0])
            return torch.bmm(wb, x.reshape(B, C, H * W)).view(B, self.out_ch, H, W)
        wb = torch.einsum("bk,k...->b...", r, wk).reshape(B * self.out_ch, *wk.shape[2:])
        y = F.conv2d(x.reshape(1, B * C, H, W), wb, None, self.stride, self.padding, self.dilation, self.groups * B)
        return y.view(B, self.out_ch, *y.shape[-2:])

    def extra_repr(self):
        return f"K={self.K}, {self.in_ch}->{self.out_ch}, groups={self.groups}, stride={self.stride}"


# --------------------------------------------------------------------------- routed MLP experts
class MoEMlp(nn.Module):
    """Per-image top-1 routed experts (+ optional shared expert). Forward value of the gate is 1
    (straight-through on p_top) so an upcycled copy starts equal to the dense model."""

    def __init__(self, dim, hidden, E, shared_hidden=0, aux_w=0.01, act=None, dense=None):
        super().__init__()
        self.E, self.aux_w, self.dim, self.hidden = E, aux_w, dim, hidden
        self.act = act or nn.GELU()
        self.router = nn.Linear(dim, E)
        nn.init.normal_(self.router.weight, std=0.02)
        nn.init.zeros_(self.router.bias)
        if dense is not None:  # upcycle: copies of the pretrained MLP
            w1, b1, w2, b2 = (t.data for t in (dense.fc1.weight, dense.fc1.bias, dense.fc2.weight, dense.fc2.bias))
            self.w1 = nn.Parameter(w1.unsqueeze(0).repeat(E, 1, 1).clone())
            self.b1 = nn.Parameter(b1.unsqueeze(0).repeat(E, 1).clone())
            self.w2 = nn.Parameter(w2.unsqueeze(0).repeat(E, 1, 1).clone())
            self.b2 = nn.Parameter(b2.unsqueeze(0).repeat(E, 1).clone())
        else:
            self.w1 = nn.Parameter(nn.init.trunc_normal_(torch.empty(E, hidden, dim), std=0.02))
            self.b1 = nn.Parameter(torch.zeros(E, hidden))
            self.w2 = nn.Parameter(nn.init.trunc_normal_(torch.empty(E, dim, hidden), std=0.02))
            self.b2 = nn.Parameter(torch.zeros(E, dim))
        self.shared = None
        if shared_hidden:
            self.shared = nn.Sequential(nn.Linear(dim, shared_hidden), nn.GELU(), nn.Linear(shared_hidden, dim))
            for m in self.shared:
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    nn.init.zeros_(m.bias)
        self.register_buffer("count", torch.zeros(E))
        self.register_buffer("n", torch.zeros(()))

    def expert(self, e, x):
        return F.linear(self.act(F.linear(x, self.w1[e], self.b1[e])), self.w2[e], self.b2[e])

    def forward(self, x):
        B = x.shape[0]
        p = F.softmax(self.router(x.mean(1)).float(), dim=-1)  # (B, E)
        idx = p.argmax(-1)
        onehot = F.one_hot(idx, self.E).float()
        f = onehot.mean(0)
        aux = self.E * (f * p.mean(0)).sum()
        with torch.no_grad():
            self.count.add_(onehot.sum(0))
            self.n.add_(B)
        if self.training:
            gate = onehot * (1.0 + p - p.detach())  # forward value = one-hot
            out = 0
            for e in range(self.E):
                out = out + gate[:, e].view(B, 1, 1).to(x.dtype) * self.expert(e, x)
        elif B == 1:
            out = self.expert(int(idx[0]), x)
        else:
            out = torch.zeros_like(x)
            for e in range(self.E):
                sel = (idx == e).nonzero().squeeze(1)
                if sel.numel():
                    out.index_add_(0, sel, self.expert(e, x[sel]).to(out.dtype))
        if self.shared is not None:
            out = out + self.shared(x)
        return AuxAttach.apply(out, aux, self.aux_w) if self.training else out


# --------------------------------------------------------------------------- apply / controls
def _parse(spec):
    kind, _, rest = spec.partition(":")
    kw = {}
    for it in filter(None, rest.split(",")):
        k, v = it.split("=")
        kw[k] = v
    return kind, kw


def apply(model, spec, seed=0):
    torch.manual_seed(seed + 1234)  # do not disturb data RNG semantics more than necessary
    kind, kw = _parse(spec)
    n0 = sum(p.numel() for p in model.parameters())
    if kind == "cc":
        K = int(kw.get("k", 4))
        stages = [int(s) for s in kw.get("stages", "3").split("+")]
        noise, tau = float(kw.get("noise", 0.05)), float(kw.get("tau", 30))
        from timm.models.repvit import ConvNorm
        n = 0
        for si in stages:
            for mod in list(model.stages[si].modules()):
                if isinstance(mod, ConvNorm) and isinstance(mod.c, nn.Conv2d) and mod.c.bias is None:
                    mod.c = CondConv2d(mod.c, K, noise, tau)
                    n += 1
        info = dict(kind=kind, K=K, stages=stages, layers=n)
    elif kind in ("mlp", "upcycle"):
        E, last = int(kw.get("e", 4)), int(kw.get("last", 4))
        aux = float(kw.get("aux", 0.01))
        blocks = model.blocks
        for blk in list(blocks)[-last:]:
            d = blk.mlp
            D, H = d.fc1.in_features, d.fc1.out_features
            if kind == "mlp":
                half = H // 2
                blk.mlp = MoEMlp(D, half, E, shared_hidden=half, aux_w=aux)
            else:
                blk.mlp = MoEMlp(D, H, E, shared_hidden=0, aux_w=aux, dense=d)
        info = dict(kind=kind, E=E, last=last, aux=aux)
    else:
        raise ValueError(spec)
    info["total_params"] = sum(p.numel() for p in model.parameters())
    info["dense_params"] = n0
    return info


def set_tau(model, tau):
    for m in model.modules():
        if isinstance(m, CondConv2d):
            m.tau.fill_(tau)


def tau_at(epoch, tau0=30.0, anneal_epochs=10):
    """DynamicConv schedule: linear tau0 -> 1 over the first `anneal_epochs` epochs (epoch is 1-based, at epoch start)."""
    return max(1.0, tau0 - (tau0 - 1.0) * (epoch - 1) / max(1, anneal_epochs))


def _layers(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, (CondConv2d, MoEMlp))]


def load_entropy(model, reset=False, hard=True):
    """Mean normalised entropy (1 = perfectly balanced) and min-expert-share over routed layers, from the counters."""
    ents, mins, per = [], [], {}
    for name, m in _layers(model):
        c = (m.hard if isinstance(m, CondConv2d) and hard else m.count).clone().float()
        K = c.numel()
        if c.sum() > 0:
            q = c / c.sum()
            h = float(-(q * (q + 1e-12).log()).sum() / math.log(K))
            ents.append(h)
            mins.append(float(q.min()))
            per[name] = [round(float(v), 4) for v in q]
        if reset:
            m.count.zero_()
            if isinstance(m, CondConv2d):
                m.hard.zero_()
            m.n.zero_()
    if not ents:
        return {}
    return {"moe_entropy": sum(ents) / len(ents), "moe_min_share": sum(mins) / len(mins)}, per


def epoch_stats(model, reset=True):
    r = load_entropy(model, reset)
    return r[0] if r else {}


# --------------------------------------------------------------------------- profile
def param_counts(model):
    """total, active (parameters touched/used per image after kernel aggregation / top-1 routing)."""
    total = sum(p.numel() for p in model.parameters())
    active = total
    for _, m in _layers(model):
        if isinstance(m, CondConv2d):
            active -= (m.K - 1) * m.weight[0].numel()
        else:
            active -= (m.E - 1) * (m.w1[0].numel() + m.b1[0].numel() + m.w2[0].numel() + m.b2[0].numel())
    return total, active


@torch.no_grad()
def macs(model, x):
    from torch.utils.flop_counter import FlopCounterMode

    model.eval()
    with FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops() / 2


@torch.no_grad()
def cpu_latency(model, x, threads=4, runs=50, warmup=10):
    prev = torch.get_num_threads()
    torch.set_num_threads(threads)
    model.eval()
    for _ in range(warmup):
        model(x)
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        model(x)
        ts.append((time.perf_counter() - t) * 1e3)
    torch.set_num_threads(prev)
    return statistics.median(ts)


def profile(model, img_size, runs=30):
    m = copy.deepcopy(model).float().cpu().eval()
    x = torch.randn(1, 3, img_size, img_size)
    total, active = param_counts(m)
    # routing at random-init statistics: MACs of the routed path do not depend on which expert is picked
    return {"params_total": total, "params_active": active, "macs_active": macs(m, x),
            "cpu_ms_bs1_t4_pytorch_eager": cpu_latency(m, x, 4, runs), "cpu_ms_bs1_t1_pytorch_eager": cpu_latency(m, x, 1, runs)}


@torch.no_grad()
def finalize(model, fwd, xte, out_dir, img_size, extra=None, bs=500):
    """Reset counters, pass the test set once (eval path), dump per-layer routing load + profile."""
    model.eval()
    load_entropy(model, reset=True)
    for i in range(0, xte.size(0), bs):
        fwd(xte[i:i + bs])
    r = load_entropy(model, reset=False)
    stats = dict(extra or {})
    if r:
        stats.update(r[0])
        stats["load_per_layer"] = r[1]
    stats.update(profile(model, img_size))
    with open(os.path.join(out_dir, "moe_stats.json.tmp"), "w") as f:
        json.dump(stats, f, indent=2)
    os.replace(os.path.join(out_dir, "moe_stats.json.tmp"), os.path.join(out_dir, "moe_stats.json"))
    print("moe_stats:", {k: v for k, v in stats.items() if k != "load_per_layer"})
    return stats
