#!/usr/bin/env python3
"""MACs (analytic, matmuls only, per 32x32 image, CIFAR-100) and params of the C2 variants vs flat models."""
import json, sys
from models.recurrent_vit import build_recurrent_vit
V = {
 "flat6 (rd_flat6)": ("rd_flat6", {}), "flat20 (equal-MACs)": ("rd_flat20", {}),
 "r8 add (baseline)": ("rd_tiny4", dict(r_mean=8, inject="add")),
 "r8 add + LoRA8": ("rd_tiny4", dict(r_mean=8, inject="add", lora_rank=8)),
 "r8 add + LoRA32": ("rd_tiny4", dict(r_mean=8, inject="add", lora_rank=32)),
 "r8 add + sandwich": ("rd_tiny4", dict(r_mean=8, inject="add", sandwich=True)),
 "r8 concat": ("rd_tiny4", dict(r_mean=8, inject="concat")),
 "r8 concat + sandwich": ("rd_tiny4", dict(r_mean=8, inject="concat", sandwich=True)),
}
out = {}
for k, (name, kw) in V.items():
    m = build_recurrent_vit(name, 100, **kw)
    r = m.r_mean if name == "rd_tiny4" else 1
    out[k] = dict(params_M=sum(p.numel() for p in m.parameters()) / 1e6, mmacs=m.flops(r) / 2e6)
    print(f"{k:24s} params {out[k]['params_M']:.3f}M  MACs {out[k]['mmacs']:.1f}M")
json.dump(out, open("research/c2_macs.json", "w"), indent=1)
