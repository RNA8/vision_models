"""Collect MoE runs into research/results_moe.md rows (reads runs/*/results.json, moe_stats.json, edge_bench.json)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "runs")
sys.path.insert(0, HERE)
import torch, moe


def J(p):
    p = os.path.join(R, p)
    return json.load(open(p)) if os.path.exists(p) else None


def dense_vit_prof(img, patch, depth_args=None):
    from train_vit import build_model
    from argparse import Namespace
    a = Namespace(model="tiny", patch_size=patch, embed_dim=None, depth=None, heads=None, mlp_ratio=None, pool="seqpool", lsa=True,
                  init="default", drop_path=0.0)
    m = build_model(a, 100, img).eval()
    x = torch.randn(1, 3, img, img)
    return sum(p.numel() for p in m.parameters()), moe.macs(m, x), moe.cpu_latency(m, x, 4)


rows = []
vit_dense = dense_vit_prof(32, 4)
vit64_dense = dense_vit_prof(64, 8)
def row(name, kind, dense=None):
    r = J(f"{name}/results.json")
    if r is None:
        return
    s = J(f"{name}/moe_stats.json")
    eb = J(f"{name}/edge_bench.json"); eb = eb[0] if isinstance(eb, list) else eb
    ema = r.get("best_ema_val_acc")
    if s:
        par = (s["params_active"], s["params_total"], s["macs_active"], s["cpu_ms_bs1_t4_pytorch_eager"], s.get("moe_entropy"))
    elif dense:
        par = (dense[0], dense[0], dense[1], dense[2], None)
    elif eb:
        par = (eb["params_M"] * 1e6, eb["params_M"] * 1e6, eb["GMACs"] * 1e9, None, None)
    else:
        par = (r["params_M"] * 1e6,) * 2 + (None, None, None)
    f = lambda v, d=2, sc=1: "-" if v is None else f"{v / sc:.{d}f}"
    rows.append(f"| {name} | {r['final_val_acc']:.2f} | {f(ema)} | {par[0]/1e6:.2f} | {par[1]/1e6:.2f} | {f(par[2],0,1e6)} | {f(par[3],1)} | {f(par[4],3)} | {r['total_minutes']:.1f} |")


for n, d in [("ec100_repvit_m0_9_128", None), ("ec100_moe_repvit_k4_128", None), ("ec100_moe_repvit_k8_128", None),
             ("ec100_repvit_m0_9_224", None), ("ec100_moe_repvit_k4_224", None), ("ec100_moe_repvit_k8_224", None), ("ec100_repvit_m0_9_224_s1", None), ("ec100_moe_repvit_k4_224_s1", None),
             ("c100_tinyshape_bf3b", vit_dense), ("c100_moe_vit_e4", None), ("c100_moe_vit_e8", None),
             ("c100_dense_vit_150ep", vit_dense), ("c100_moe_vit_e4_150ep", None), ("c100_moe_vit_e8_150ep", None),
             ("ec100_in64pre_tiny_64", vit64_dense), ("ec100_dense_in64pre_tiny_64_s1", vit64_dense), ("ec100_upcyc_in64pre_tiny_e4_64", None),
             ("ec100_upcyc_in64pre_tiny_e4_64_s1", None), ("ec100_dense_in64pre_tiny_64_60ep", vit64_dense),
             ("ec100_upcyc_in64pre_tiny_e4_64_60ep", None)]:
    row(n, None, d)
print("| run | val (final) | best EMA val | active params M | total params M | active MACs M | CPU ms bs1 t4 (PyTorch eager) | expert-load entropy (1=balanced) | GPU min (shared GPU) |")
print("|---|---|---|---|---|---|---|---|---|")
print("\n".join(rows))
