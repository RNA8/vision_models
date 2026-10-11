"""Latency/MACs/params of dense RepViT vs a CondConv-converted one (PyTorch eager CPU bs1; ONNX attempt for the MoE)."""
import sys, json, copy, torch, timm, tempfile, os, time, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moe
torch.set_num_threads(4)
size, K = int(sys.argv[1]), int(sys.argv[2])
x = torch.randn(1, 3, size, size)
out = {"size": size, "K": K}
dense = timm.create_model("repvit_m0_9", num_classes=100).eval()
out["dense_unfused"] = {"macs": moe.macs(dense, x), "params": sum(p.numel() for p in dense.parameters()),
                        "t4_ms": moe.cpu_latency(dense, x, 4), "t1_ms": moe.cpu_latency(dense, x, 1)}
from timm.utils.model import reparameterize_model
fused = reparameterize_model(copy.deepcopy(dense)).eval()
out["dense_reparam"] = {"macs": moe.macs(fused, x), "params": sum(p.numel() for p in fused.parameters()),
                        "t4_ms": moe.cpu_latency(fused, x, 4), "t1_ms": moe.cpu_latency(fused, x, 1)}
m = timm.create_model("repvit_m0_9", num_classes=100)
moe.apply(m, f"cc:k={K},stages=3"); m.eval()
tot, act = moe.param_counts(m)
out["moe_unfused"] = {"macs": moe.macs(m, x), "params_total": tot, "params_active": act,
                      "t4_ms": moe.cpu_latency(m, x, 4), "t1_ms": moe.cpu_latency(m, x, 1)}
try:
    import onnxruntime as ort
    p = tempfile.mktemp(suffix=".onnx")
    torch.onnx.export(m, x, p, opset_version=17, dynamo=False)
    res = {}
    for th in (1, 4):
        so = ort.SessionOptions(); so.intra_op_num_threads = th
        s = ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])
        xn = x.numpy()
        for _ in range(10): s.run(None, {s.get_inputs()[0].name: xn})
        ts = []
        for _ in range(50):
            t = time.perf_counter(); s.run(None, {s.get_inputs()[0].name: xn}); ts.append((time.perf_counter() - t) * 1e3)
        res[f"ort_fp32_t{th}_ms"] = statistics.median(ts)
    ref = m(x).detach().numpy(); got = s.run(None, {s.get_inputs()[0].name: x.numpy()})[0]
    res["max_abs_diff"] = float(abs(ref - got).max()); res["onnx_MB"] = os.path.getsize(p) / 1e6
    out["moe_onnx"] = res
except Exception as e:
    out["moe_onnx"] = f"FAILED: {type(e).__name__}: {str(e)[:200]}"
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"runs/moe_bench_cc_k{K}_{size}.json"), "w"), indent=1)
