"""
recurrent_vit.py: a recurrent-depth Vision Transformer (prelude -> looped core -> coda).

The design follows Geiping et al., "Scaling up Test-Time Compute with Latent Reasoning: A
Recurrent Depth Approach" (arXiv 2502.05171), adapted to image classification:

    e   = InjectNorm(Prelude(PatchEmbed(x) + pos, CLS))           # computed once
    s_0 = TruncNormal(0, sigma^2) or 0                             # random state init (Geiping: sigma^2 = 2/5)
    s_t = OuterNorm(Core(Adapter([s_{t-1}, e])))   for t = 1..r    # "recall": e is re-injected every loop
    y   = Head(Norm(Coda(s_r))[CLS])

* Input injection ("recall"). `concat` uses a linear adapter R^{2d} -> R^d, as in Geiping (who find
  concat beats addition at scale). `add` uses s + e. `none` is the ablation without recall: then s_0 = e
  and s_t = Core(s_{t-1}), i.e. a plain weight-shared looped ViT. 2604.15259 reports that extrapolating
  to more loops than were trained needs both recall and an outer normalization, which motivates both.
* Outer normalization. A LayerNorm or RMSNorm on the state after every loop (Geiping's n_c after the core).
  It keeps ||s_t|| bounded however many times the core is applied. `none` disables it (ablation).
* Training depth. r is sampled per batch: Geiping's Poisson-lognormal
      tau ~ N(log(r_mean - 1) - sigma^2/2, sigma),   r = 1 + Poisson(exp(tau))
  (we use r_mean - 1 inside the log so that E[r] = r_mean exactly; Geiping writes log(r_bar), giving
  E[r] = r_bar + 1), or uniform on {1, ..., 2 r_mean - 1}, or fixed. r is clipped at r_max (default 4 r_mean).
* Truncated backprop. Only the last k iterations (default k = r_mean; Geiping uses k = 8 with r_bar = 32)
  run with autograd; earlier ones run under no_grad, so memory and backward cost do not grow with r.
  The prelude still gets gradients through the injected e in the last k loops.
* The coda runs only once, on the final state, in training (unless coda_every_step=True). At inference
  `rollout()` / `forward(return_all=True)` decode every intermediate state for depth sweeps and adaptive exit.
* QK-norm (timm Block(qk_norm=True)) is an option for attention-logit stability in the looped core.
* Inject norm. The prelude output is LayerNorm'ed before injection so that e and the (normalized) state
  have the same scale; without it, s + e would be dominated by whichever has the larger norm.

torch.compile / CUDA graphs. A variable r breaks static graphs: compiling the whole model unrolls the
Python loop with r baked in, so every new r is a recompile (and torch's recompile limit is 8), and a
CUDA graph captures one fixed r. `compile_parts()` instead compiles the pieces separately: the embed +
prelude, ONE core step, and the coda + head. The loop over r stays in Python, and each core call is a
static-shape compiled graph (one for the no_grad iterations, one for the grad ones, one for eval). Pass
parts=("core",) to compile only the core step. mode="reduce-overhead" (CUDA graphs) on the core is
experimental: cudagraph trees record one graph per position in the call sequence, so it should work but
is untested here (only CPU tests were run).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import PatchEmbed, trunc_normal_
from timm.models.vision_transformer import Block

# Presets. ViT-Tiny is 12 layers of dim 192; each rd_ preset matches that 12-layer compute at r = r_mean
# (P + C * r_mean + D = 12), with fewer unique parameters. Parameter counts are for CIFAR-10.
RD_PRESETS = {
    # 8 unique blocks (~3.7M params), 2 + 4*2 + 2 = 12 layers at r_mean = 2
    "rd_tiny": dict(embed_dim=192, num_heads=3, n_prelude=2, n_core=4, n_coda=2, r_mean=2),
    # 6 unique blocks (~2.8M params), 2 + 2*4 + 2 = 12 layers at r_mean = 4. The main pilot model.
    "rd_tiny4": dict(embed_dim=192, num_heads=3, n_prelude=2, n_core=2, n_coda=2, r_mean=4),
    # 5 unique blocks (~2.3M params), single-block core (reViT / bViT style), 2 + 1*8 + 2 = 12 at r_mean = 8
    "rd_tiny8": dict(embed_dim=192, num_heads=3, n_prelude=2, n_core=1, n_coda=2, r_mean=8),
    # ViT-Small width, 2 + 2*4 + 2 = 12 layers at r_mean = 4
    "rd_small4": dict(embed_dim=384, num_heads=6, n_prelude=2, n_core=2, n_coda=2, r_mean=4),
    # Parameter-matched NON-recurrent baseline for rd_tiny4: 6 unique blocks run once (r = 1, no
    # injection, no outer norm). It is a standard 6-layer pre-norm ViT-Tiny except for one LayerNorm
    # (the inject norm) on the residual stream after layer 2.
    "rd_flat6": dict(embed_dim=192, num_heads=3, n_prelude=2, n_core=2, n_coda=2, r_mean=1,
                     depth_dist="fixed", inject="none", outer_norm="none", state_init="zeros"),
}

DEPTH_DISTS = ("poisson_lognormal", "uniform", "fixed")


def sample_depth(r_mean, dist="poisson_lognormal", sigma=0.5, r_max=None, generator=None):
    """Sample a recurrence count r >= 1 with E[r] = r_mean (before clipping at r_max). Runs on the CPU,
    so it never syncs the GPU."""
    r_mean = int(r_mean)
    if r_mean <= 1 or dist == "fixed":
        r = r_mean
    elif dist == "uniform":
        r = int(torch.randint(1, 2 * r_mean, (1,), generator=generator))
    elif dist == "poisson_lognormal":
        mu = math.log(r_mean - 1) - sigma**2 / 2  # E[exp(tau)] = r_mean - 1
        tau = mu + sigma * float(torch.randn((), generator=generator))
        r = 1 + int(torch.poisson(torch.tensor([math.exp(tau)]), generator=generator))
    else:
        raise ValueError(f"unknown depth distribution {dist!r}; choose from {DEPTH_DISTS}")
    r_max = r_max or 4 * r_mean
    return max(1, min(r, r_max))


def _norm(kind, dim):
    if kind == "ln":
        return nn.LayerNorm(dim, eps=1e-6)
    if kind == "rms":
        return nn.RMSNorm(dim, eps=1e-6)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm {kind!r}; choose ln, rms or none")


class RecurrentViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10, embed_dim=192, num_heads=3,
                 mlp_ratio=4.0, n_prelude=2, n_core=2, n_coda=2, r_mean=4, k_backprop=None,
                 depth_dist="poisson_lognormal", depth_sigma=0.5, r_max=None, state_init="normal",
                 state_std=math.sqrt(2 / 5), inject="concat", outer_norm="ln", qk_norm=False,
                 drop_path=0.0, coda_every_step=False, r_eval=None):
        super().__init__()
        assert n_core >= 1, "the core needs at least one block"
        assert inject in ("concat", "add", "none"), inject
        assert state_init in ("normal", "zeros"), state_init
        assert depth_dist in DEPTH_DISTS, depth_dist
        self.num_classes, self.embed_dim = num_classes, embed_dim
        self.mlp_ratio = mlp_ratio
        self.n_prelude, self.n_core, self.n_coda = n_prelude, n_core, n_coda
        self.r_mean = int(r_mean)
        self.k_backprop = self.r_mean if k_backprop is None else int(k_backprop)
        self.depth_dist, self.depth_sigma, self.r_max = depth_dist, depth_sigma, r_max
        self.state_init, self.state_std, self.inject = state_init, state_std, inject
        self.coda_every_step = coda_every_step
        self.r_eval = int(r_eval) if r_eval else self.r_mean  # depth used by forward() in eval mode
        self.num_prefix_tokens = 1  # CLS (used by train_vit.save_pos_embed_plot)

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        dpr = torch.linspace(0, drop_path, n_prelude + n_core + n_coda).tolist()
        mk = lambda i: Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_norm=qk_norm,
                             drop_path=dpr[i], norm_layer=nn.LayerNorm)
        self.prelude = nn.ModuleList([mk(i) for i in range(n_prelude)])
        self.core = nn.ModuleList([mk(n_prelude + i) for i in range(n_core)])
        self.coda = nn.ModuleList([mk(n_prelude + n_core + i) for i in range(n_coda)])
        self.inject_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.adapter = nn.Linear(2 * embed_dim, embed_dim) if inject == "concat" else None
        self.outer_norm = _norm(outer_norm, embed_dim)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)
        self.attn_pool = None  # optional pooling over patch tokens (e.g. SeqPool); None = read CLS

        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        self.apply(self._init_weights)
        # the compiled-or-eager callables used by forward(); compile_parts() swaps them
        self._embed_fn, self._core_fn, self._decode_fn = self.embed, self.core_step, self.decode

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    # ------------------------------------------------------------------ pieces
    def embed(self, x):
        """Patch embed + CLS + pos embed + prelude -> injected input e (B, N, d)."""
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1) + self.pos_embed
        for blk in self.prelude:
            x = blk(x)
        return self.inject_norm(x)

    def init_state(self, e):
        if self.inject == "none":
            return e  # no recall: the loop starts from the prelude output and never sees it again
        if self.state_init == "zeros":
            return torch.zeros_like(e)
        s = torch.randn_like(e) * self.state_std  # Geiping: truncated at 3 sigma
        return s.clamp_(-3 * self.state_std, 3 * self.state_std)

    def core_step(self, s, e):
        """One recurrence: s <- OuterNorm(Core(inject(s, e)))."""
        if self.inject == "concat":
            s = self.adapter(torch.cat([s, e], dim=-1))
        elif self.inject == "add":
            s = s + e
        for blk in self.core:
            s = blk(s)
        return self.outer_norm(s)

    def decode(self, s):
        """Coda + final norm + head on CLS (or attn_pool over the patch tokens) -> logits (B, C)."""
        for blk in self.coda:
            s = blk(s)
        s = self.norm(s)
        return self.head(s[:, 0] if self.attn_pool is None else self.attn_pool(s[:, 1:]))

    def sample_r(self):
        return sample_depth(self.r_mean, self.depth_dist, self.depth_sigma, self.r_max)

    # ------------------------------------------------------------------ forward
    def forward(self, x, r=None, return_all=False):
        """Train mode: r is sampled per call (unless given) and only the last k_backprop iterations
        carry gradients. Eval mode: r defaults to self.r_eval (= r_mean unless overridden).
        return_all=True returns per-iteration logits of shape (r, B, C)."""
        if return_all:
            return self.rollout(x, r)["logits"]
        if r is None:
            r = self.sample_r() if self.training else self.r_eval
        e = self._embed_fn(x)
        s = self.init_state(e)
        n_nograd = max(0, r - self.k_backprop) if (self.training and torch.is_grad_enabled()) else 0
        if n_nograd:
            with torch.no_grad():
                for _ in range(n_nograd):
                    s = self._core_fn(s, e)
        outs = []
        for _ in range(r - n_nograd):
            s = self._core_fn(s, e)
            if self.coda_every_step and self.training:
                outs.append(self._decode_fn(s))
        if outs:  # average of per-step logits: a deep-supervision variant (off by default)
            return torch.stack(outs).mean(0)
        return self._decode_fn(s)

    def rollout(self, x, r=None):
        """Run r iterations and decode every intermediate state (meant for eval; honours grad mode).
        Returns dict(logits=(r, B, C), delta=(r, B)) with delta_t = ||s_t - s_{t-1}|| / ||s_t||
        over all tokens of an image (delta_1 compares against s_0)."""
        r = r or self.r_eval
        e = self._embed_fn(x)
        s = self.init_state(e)
        logits, delta = [], []
        for _ in range(r):
            s_new = self._core_fn(s, e)
            d = (s_new - s).float().flatten(1).norm(dim=1) / s_new.float().flatten(1).norm(dim=1).clamp_min(1e-12)
            s = s_new
            delta.append(d)
            logits.append(self._decode_fn(s))
        return {"logits": torch.stack(logits), "delta": torch.stack(delta)}

    # ------------------------------------------------------------------ compile
    def compile_parts(self, mode=None, parts=("embed", "core", "decode")):
        """Compile the pieces separately so a variable r never triggers a recompile: the Python loop
        calls a static-shape compiled core step r times. See the module docstring."""
        kw = dict(mode=mode, dynamic=False)
        if "embed" in parts:
            self._embed_fn = torch.compile(self.embed, **kw)
        if "core" in parts:
            core = torch.compile(self.core_step, **kw)
            # CUDA graphs reuse their output buffer on every replay, and the loop feeds each output back
            # in; clone so iteration t+1 doesn't overwrite the state iteration t returned.
            self._core_fn = (lambda s, e: core(s, e).clone()) if mode == "reduce-overhead" else core
        if "decode" in parts:
            self._decode_fn = torch.compile(self.decode, **kw)
        return self

    # ------------------------------------------------------------------ cost
    def flops(self, r, coda_evals=1):
        """Analytic forward FLOPs per image (2 x multiply-accumulates; matmuls only, norms/softmax
        ignored) for r core iterations and `coda_evals` coda+head evaluations (r for an exit rule that
        decodes every step)."""
        d, n = self.embed_dim, self.num_patches + 1
        p = self.patch_embed.patch_size
        blk = (4 + 2 * self.mlp_ratio) * n * d * d + 2 * n * n * d  # qkv+proj+mlp, QK^T and AV
        patch = self.num_patches * d * self.patch_embed.proj.in_channels * p[0] * p[1]
        adapter = n * 2 * d * d if self.inject == "concat" else 0
        head = d * self.num_classes
        macs = patch + self.n_prelude * blk + r * (self.n_core * blk + adapter) \
            + coda_evals * (self.n_coda * blk + head)
        return 2 * macs


def build_recurrent_vit(name, num_classes, img_size=32, patch_size=4, drop_path=0.0, **overrides):
    """Build a preset from RD_PRESETS; any override that is not None replaces the preset value."""
    if name not in RD_PRESETS:
        raise ValueError(f"unknown recurrent preset {name!r}; choose from {list(RD_PRESETS)}")
    cfg = dict(RD_PRESETS[name])
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return RecurrentViT(img_size=img_size, patch_size=patch_size, num_classes=num_classes,
                        drop_path=drop_path, **cfg)
