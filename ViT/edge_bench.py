#!/usr/bin/env python3
"""
edge_bench.py — edge-budget profile of a model at a given input size.

For each model (timm name or one of our run dirs) and input size it reports:
  * params (M), MACs (G, batch 1; torch.utils.flop_counter.FlopCounterMode counts matmul/conv/SDPA-type ops,
    MACs = FLOPs / 2; elementwise ops/norms/softmax are not counted, as in fvcore/thop conventions)
  * peak activation memory at batch 1: measured on the GPU (max_memory_allocated minus weights) when CUDA is
    available, and always an allocator-independent estimate (peak bytes of live op outputs, CPU, fp32)
  * GPU latency at batch 1 (median of N, CUDA events) and throughput at batch 256, fp32 and bf16 autocast
  * edge proxy: ONNX export (opset 17) + onnxruntime CPU latency at batch 1 with 1 and 4 threads, fp32 and
    int8 dynamic-quantized (onnxruntime.quantization.quantize_dynamic on MatMul/Gemm: weights int8, activations
    quantized on the fly; convs stay fp32 because ORT CPU has no ConvInteger kernel), plus .onnx file sizes
Models with train-time re-parameterisable branches (RepViT, FastViT, MobileOne, EfficientViT-M BN fusion,
...) are fused first with timm.utils.model.reparameterize_model (--no-reparam to disable).
Weights do not matter for these numbers: timm names are built with pretrained=False (no downloads).

Output: one JSON record per (model, size) on stdout, appended to runs/edge_bench.jsonl (--jsonl).
GPU numbers are only meaningful when the GPU is otherwise idle (record carries gpu_util_at_start).

  python edge_bench.py deit_tiny_patch16_224 repvit_m0_9 --img-size 224 128
  python edge_bench.py runs/ft_c100_tiny_vit_5m_128 --no-gpu          # a finetune.py run at its own size
"""
import argparse
import json
import os
import statistics
import subprocess
import tempfile
import time
import weakref

import torch
from torch.utils._python_dispatch import TorchDispatchMode

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description="Edge-budget benchmark: params, MACs, memory, GPU + ONNX/ORT CPU latency")
    p.add_argument("models", nargs="+", help="timm model names or run dirs (finetune.py / train_vit.py)")
    p.add_argument("--img-size", type=int, nargs="+", default=None,
                   help="input sizes (default: the run's own size, or 224 for timm names)")
    p.add_argument("--num-classes", type=int, default=100, help="head size for timm names")
    p.add_argument("--runs", type=int, default=50, help="timed runs (median reported)")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--threads", type=int, nargs="+", default=[1, 4], help="onnxruntime intra-op threads")
    p.add_argument("--tput-bs", type=int, default=256, help="GPU throughput batch size")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--no-onnx", action="store_true")
    p.add_argument("--no-int8", action="store_true")
    p.add_argument("--no-reparam", action="store_true")
    p.add_argument("--keep-onnx", default=None, help="directory to keep the exported .onnx files in")
    p.add_argument("--jsonl", default=os.path.join(HERE, "runs", "edge_bench.jsonl"), help="'' = do not append")
    p.add_argument("--out", default=None, help="also write all records as a JSON list here")
    return p.parse_args()


# --------------------------------------------------------------------------- model
def get_model(ref, img_size, num_classes):
    from finetune import load_model

    if os.path.exists(ref):
        m, spec = load_model(ref)  # the run's exact network and weights
        if img_size and img_size != spec["img_size"]:
            m, spec = load_model(ref, spec["num_classes"], img_size, transfer=True)  # pos-embed resampled
    else:
        m, spec = load_model(ref, num_classes, img_size or 224, pretrained=False)
    return m.eval(), spec


# --------------------------------------------------------------------------- measurements
def count_macs(model, x):
    from torch.utils.flop_counter import FlopCounterMode

    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops() / 2


class _LiveBytes(TorchDispatchMode):
    """Peak bytes of live op outputs during a no-grad forward: every op output storage is counted once
    while any tensor viewing it is alive (weakref finalizers). Weights and the input are not counted."""

    def __init__(self):
        super().__init__()
        self.live, self.refs, self.cur, self.peak = {}, {}, 0, 0
        self.ignore = set()

    def _free(self, key):
        self.refs[key] -= 1
        if self.refs[key] == 0:
            self.cur -= self.live.pop(key)
            del self.refs[key]

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in (out if isinstance(out, (tuple, list)) else [out]):
            if not isinstance(t, torch.Tensor):
                continue
            st = t.untyped_storage()
            key = st.data_ptr()
            if key in self.ignore or st.nbytes() == 0:
                continue
            if key not in self.live:
                self.live[key], self.refs[key] = st.nbytes(), 0
                self.cur += st.nbytes()
                self.peak = max(self.peak, self.cur)
            self.refs[key] += 1
            weakref.finalize(t, self._free, key)
        return out


def act_mem_estimate(model, x):
    mode = _LiveBytes()
    mode.ignore = {p.untyped_storage().data_ptr() for p in list(model.parameters()) + list(model.buffers())}
    mode.ignore.add(x.untyped_storage().data_ptr())
    with torch.no_grad(), mode:
        y = model(x)
    del y
    return mode.peak


def gpu_util():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.split()[0])
    except Exception:
        return None


def gpu_bench(model, size, args):
    dev = "cuda"
    m = model.to(dev)
    res = {}
    x1 = torch.randn(1, 3, size, size, device=dev)
    with torch.inference_mode():
        m(x1)  # first call: cuDNN/cuBLAS workspaces, cached attention biases; not activations
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        m(x1)
    torch.cuda.synchronize()
    res["gpu_peak_act_MB_bs1_fp32"] = (torch.cuda.max_memory_allocated() - base) / 2**20
    for prec in ("fp32", "bf16"):
        ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if prec == "bf16" else torch.no_grad
        with torch.inference_mode(), ctx():
            for _ in range(args.warmup):
                m(x1)
            times = []
            for _ in range(args.runs):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                m(x1)
                b.record()
                torch.cuda.synchronize()
                times.append(a.elapsed_time(b))
            res[f"gpu_lat_ms_bs1_{prec}"] = statistics.median(times)
            try:
                xb = torch.randn(args.tput_bs, 3, size, size, device=dev)
                for _ in range(3):
                    m(xb)
                torch.cuda.synchronize()
                n = max(5, args.runs // 5)
                t0 = time.perf_counter()
                for _ in range(n):
                    m(xb)
                torch.cuda.synchronize()
                res[f"gpu_tput_img_s_bs{args.tput_bs}_{prec}"] = n * args.tput_bs / (time.perf_counter() - t0)
                del xb
            except torch.OutOfMemoryError:
                res[f"gpu_tput_img_s_bs{args.tput_bs}_{prec}"] = None
    model.cpu()
    torch.cuda.empty_cache()
    return res


def onnx_export(model, size, path, opset):
    x = torch.randn(1, 3, size, size)
    errs = []
    for dynamo in (False, True):
        try:
            torch.onnx.export(model, (x,), path, opset_version=opset, input_names=["input"],
                              output_names=["logits"], dynamo=dynamo, **({} if dynamo else {"do_constant_folding": True}))
            return "dynamo" if dynamo else "torchscript", None
        except Exception as e:
            errs.append(f"{'dynamo' if dynamo else 'torchscript'}: {type(e).__name__}: {str(e)[:200]}")
    return None, " | ".join(errs)


def ort_latency(path, size, threads, args):
    import numpy as np
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 3, size, size).astype(np.float32)
    name = sess.get_inputs()[0].name
    for _ in range(args.warmup):
        sess.run(None, {name: x})
    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times), sess.run(None, {name: x})[0]


def onnx_bench(model, size, args, tag):
    import numpy as np

    res = {}
    d = args.keep_onnx or tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{tag}_{size}.onnx")
    exporter, err = onnx_export(model, size, path, args.opset)
    if exporter is None:
        return {"onnx_error": err}
    res["onnx_exporter"] = exporter
    res["onnx_MB"] = os.path.getsize(path) / 2**20
    if os.path.exists(path + ".data"):
        res["onnx_MB"] += os.path.getsize(path + ".data") / 2**20
    with torch.no_grad():
        xp = torch.randn(1, 3, size, size, generator=torch.Generator().manual_seed(0))
        ref = model(xp).numpy()
    for th in args.threads:
        try:
            lat, out = ort_latency(path, size, th, args)
            res[f"ort_lat_ms_fp32_t{th}"] = lat
        except Exception as e:
            res["ort_error"] = f"{type(e).__name__}: {str(e)[:200]}"
            break
    try:  # export/runtime parity check (random input)
        import onnxruntime as ort

        s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        o = s.run(None, {s.get_inputs()[0].name: xp.numpy()})[0]
        res["onnx_max_abs_diff"] = float(np.abs(o - ref).max())
        res["onnx_ref_abs_max"] = float(np.abs(ref).max())
    except Exception:
        pass
    if not args.no_int8:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            qpath = path.replace(".onnx", "_int8.onnx")
            # MatMul/Gemm only: ORT's CPU EP has no ConvInteger kernel, so conv layers stay fp32
            quantize_dynamic(path, qpath, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])
            res["onnx_int8_MB"] = os.path.getsize(qpath) / 2**20
            for th in args.threads:
                res[f"ort_lat_ms_int8_t{th}"], _ = ort_latency(qpath, size, th, args)
        except Exception as e:
            res["int8_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    if not args.keep_onnx:
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))
        os.rmdir(d)
    return res


def bench_one(ref, size, args):
    t0 = time.time()
    model, spec = get_model(ref, size, args.num_classes)
    size = size or spec["img_size"]
    rec = {"model": ref, "img_size": size, "num_classes": spec["num_classes"],
           "time": time.strftime("%Y-%m-%d %H:%M:%S"), "torch": torch.__version__}
    rec["params_M"] = sum(p.numel() for p in model.parameters()) / 1e6
    if not args.no_reparam:
        from timm.utils.model import reparameterize_model

        model = reparameterize_model(model)
        rec["params_M_deploy"] = sum(p.numel() for p in model.parameters()) / 1e6
    model = model.float().eval()
    x = torch.randn(1, 3, size, size)
    rec["GMACs"] = count_macs(model, x) / 1e9
    rec["act_MB_est_bs1_fp32"] = act_mem_estimate(model, x) / 2**20
    rec["weights_MB_fp32"] = sum(p.numel() * 4 for p in model.parameters()) / 2**20
    if torch.cuda.is_available() and not args.no_gpu:
        rec["gpu"] = torch.cuda.get_device_name(0)
        rec["gpu_util_at_start"] = gpu_util()
        rec.update(gpu_bench(model, size, args))
    if not args.no_onnx:
        tag = os.path.basename(os.path.normpath(ref)).replace("/", "_")
        rec.update(onnx_bench(model, size, args, tag))
    rec["bench_sec"] = time.time() - t0
    return rec


def main():
    args = parse_args()
    if torch.cuda.is_available() and not args.no_gpu:
        torch.backends.cuda.matmul.allow_tf32 = False  # honest fp32 latency
    recs = []
    for ref in args.models:
        for size in (args.img_size or [None]):
            try:
                rec = bench_one(ref, size, args)
            except Exception as e:
                rec = {"model": ref, "img_size": size, "error": f"{type(e).__name__}: {str(e)[:300]}"}
            print(json.dumps(rec), flush=True)
            recs.append(rec)
            if args.jsonl:
                os.makedirs(os.path.dirname(args.jsonl), exist_ok=True)
                with open(args.jsonl, "a") as f:
                    f.write(json.dumps(rec) + "\n")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(recs, f, indent=2)


if __name__ == "__main__":
    main()
