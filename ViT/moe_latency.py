"""Interleaved CPU latency (PyTorch eager, bs1, 4 threads): dense vs MoE, min over 7 rounds of the round-median (robust to a shared CPU)."""
import sys, os, torch, timm, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moe
from argparse import Namespace
from train_vit import build_model
def vit(patch, img, spec=None):
    a = Namespace(model="tiny", patch_size=patch, embed_dim=None, depth=None, heads=None, mlp_ratio=None, pool="seqpool", lsa=True, init="default", drop_path=0.0)
    m = build_model(a, 100, img)
    if spec: moe.apply(m, spec)
    return m.eval()
def rv(size, spec=None):
    m = timm.create_model("repvit_m0_9", num_classes=100)
    if spec: moe.apply(m, spec)
    return m.eval()
cfgs = {"vit32_dense": (vit(4, 32), 32), "vit32_e4_last4": (vit(4, 32, "mlp:e=4,last=4"), 32), "vit32_e8_last4": (vit(4, 32, "mlp:e=8,last=4"), 32),
        "vit64p8_dense": (vit(8, 64), 64), "vit64p8_upcycle_e4": (vit(8, 64, "upcycle:e=4,last=4"), 64),
        "repvit128_dense_unfused": (rv(128), 128), "repvit128_cc_k4": (rv(128, "cc:k=4,stages=3"), 128), "repvit128_cc_k8": (rv(128, "cc:k=8,stages=3"), 128),
        "repvit224_dense_unfused": (rv(224), 224), "repvit224_cc_k4": (rv(224, "cc:k=4,stages=3"), 224)}
res = {k: [] for k in cfgs}
for r in range(7):
    for k, (m, s) in cfgs.items():
        res[k].append(moe.cpu_latency(m, torch.randn(1, 3, s, s), 4, runs=30, warmup=5))
out = {k: round(min(v), 2) for k, v in res.items()}
print(json.dumps(out, indent=1)); json.dump(out, open("runs/moe_latency.json", "w"), indent=1)
