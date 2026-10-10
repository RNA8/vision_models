#!/usr/bin/env python3
"""CPU tests for models/recurrent_vit.py. Run: CUDA_VISIBLE_DEVICES="" python test_recurrent_vit.py
(or with pytest)."""
import itertools

import numpy as np
import torch

from models.recurrent_vit import RD_PRESETS, RecurrentViT, build_recurrent_vit, sample_depth


def small(**kw):
    cfg = dict(img_size=16, patch_size=4, num_classes=10, embed_dim=32, num_heads=2, n_prelude=1,
               n_core=2, n_coda=1, r_mean=3)
    cfg.update(kw)
    return RecurrentViT(**cfg)


def x_batch(b=4):
    return torch.randn(b, 3, 16, 16)


def test_forward_backward_all_modes():
    # every value of every option appears, across inject x outer-norm (9 configs)
    for i, (inject, outer) in enumerate(itertools.product(["concat", "add", "none"], ["ln", "rms", "none"])):
        init = ["normal", "zeros"][i % 2]
        dist = ["poisson_lognormal", "uniform", "fixed"][i % 3]
        qk, coda_every = bool(i % 2), i in (4, 8)
        torch.manual_seed(0)
        m = small(inject=inject, outer_norm=outer, state_init=init, depth_dist=dist, qk_norm=qk,
                  drop_path=0.1, coda_every_step=coda_every)
        m.train()
        out = m(x_batch())
        assert out.shape == (4, 10)
        out.float().pow(2).mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        m.eval()
        with torch.no_grad():
            assert m(x_batch()).shape == (4, 10)
            assert m(x_batch(), r=7).shape == (4, 10)


def core_grad_norm(m):
    gs = [p.grad for p in m.core.parameters()]
    if all(g is None for g in gs):
        return None
    return sum(float(g.norm()) for g in gs if g is not None)


def test_truncated_backprop():
    torch.manual_seed(0)
    m = small(state_init="zeros")
    m.train()
    x = x_batch()
    # k = 0: every core iteration under no_grad -> core grads are None; coda/head still get grads
    m.k_backprop = 0
    m(x, r=5).sum().backward()
    assert core_grad_norm(m) is None
    assert m.adapter.weight.grad is None and m.prelude[0].attn.qkv.weight.grad is None
    assert m.head.weight.grad is not None and m.coda[0].mlp.fc1.weight.grad is not None
    # k = 2 of r = 5: the gradient equals the one from detaching the state after iteration 3
    m.zero_grad(set_to_none=True)
    m.k_backprop = 2
    torch.manual_seed(1)
    m(x, r=5).sum().backward()
    g_trunc = [p.grad.clone() for p in m.core.parameters()]
    m.zero_grad(set_to_none=True)
    m.k_backprop = 10**6  # full backprop, detach manually
    e = m.embed(x)
    s = m.init_state(e)
    for i in range(5):
        s = m.core_step(s, e)
        if i == 2:
            s = s.detach()
    m.decode(s).sum().backward()
    g_ref = [p.grad.clone() for p in m.core.parameters()]
    for a, b in zip(g_trunc, g_ref):
        assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max()
    # and differs from full backprop through all 5 iterations
    m.zero_grad(set_to_none=True)
    m(x, r=5).sum().backward()
    g_full = [p.grad.clone() for p in m.core.parameters()]
    assert any(not torch.allclose(a, b, atol=1e-6) for a, b in zip(g_trunc, g_full))
    # in eval / no_grad the truncation is irrelevant
    m.eval()
    m.k_backprop = 0
    with torch.no_grad():
        m(x, r=5)


def test_return_all_and_rollout():
    torch.manual_seed(0)
    m = small(state_init="zeros").eval()
    x = x_batch()
    with torch.no_grad():
        allr = m(x, r=6, return_all=True)
        assert allr.shape == (6, 4, 10)
        for r in (1, 3, 6):  # step r of the rollout == forward with r iterations (deterministic init)
            assert torch.allclose(allr[r - 1], m(x, r=r), atol=1e-5)
        ro = m.rollout(x, 6)
        assert ro["delta"].shape == (6, 4) and torch.isfinite(ro["delta"]).all()
        assert m(x).shape == (4, 10) and m.r_eval == 3


def test_depth_sampling_mean():
    g = torch.Generator().manual_seed(0)
    for r_mean in (2, 4, 8):
        for dist in ("poisson_lognormal", "uniform"):
            rs = np.array([sample_depth(r_mean, dist, generator=g) for _ in range(20000)])
            assert rs.min() >= 1
            assert abs(rs.mean() - r_mean) < 0.05 * r_mean, (dist, r_mean, rs.mean())
            if dist == "uniform":
                assert rs.max() == 2 * r_mean - 1
        assert sample_depth(r_mean, "fixed") == r_mean
    assert sample_depth(1, "poisson_lognormal") == 1


def test_presets_and_flops():
    for name in RD_PRESETS:
        m = build_recurrent_vit(name, 10)
        n = sum(p.numel() for p in m.parameters()) / 1e6
        layers = m.n_prelude + m.n_core * m.r_mean + m.n_coda
        print(f"  {name:10s} params {n:5.2f}M  layers at r_mean: {layers:2d}  "
              f"GFLOPs@r_mean {m.flops(m.r_mean) / 1e9:.3f}")
        assert layers in (6, 12)
    from timm.models.vision_transformer import VisionTransformer
    vit = VisionTransformer(img_size=32, patch_size=4, num_classes=10, embed_dim=192, depth=12, num_heads=3)
    print(f"  vit-tiny   params {sum(p.numel() for p in vit.parameters()) / 1e6:5.2f}M")
    m = build_recurrent_vit("rd_tiny4", 10)
    assert m.flops(8) > m.flops(4) > m.flops(1)


def test_compile_parts_cpu():
    torch.manual_seed(0)
    m = small(state_init="zeros")
    ref = m.eval()(x_batch(), r=4)
    m.compile_parts()
    with torch.no_grad():
        out = m(x_batch(), r=4)
    m.train()
    m(x_batch(), r=5).sum().backward()  # no_grad prefix + grad suffix graphs
    assert out.shape == ref.shape


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(name)
            fn()
            print("  ok")
    print("all tests passed")
