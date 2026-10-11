#!/usr/bin/env python3
"""
int8_audit.py - INT8 accuracy / latency audit of fine-tuned CIFAR-100 edge models (CPU only).

Per model: load the run's reported checkpoint (finetune.load_model: EMA iff it was the better one), fuse
re-parameterisable branches, export FP32 ONNX (dynamic batch), then with ONNX Runtime CPU:
  fp32                      full 10k CIFAR-100 test top-1
  static QDQ int8 (PTQ)     per-channel weights, 512 train images for calibration (fixed seed), calibration
                            MinMax / Percentile(99.999) / Entropy; same preprocessing as training (resize with
                            F.interpolate bilinear(+antialias), mean/std)
  dynamic int8              quantize_dynamic on MatMul/Gemm (what edge_bench.py does; convs stay fp32)
Latency: batch 1, intra-op threads=4, 10 warmup, median of 100 runs. File size = .onnx size.
Optional ablations (--ablate): re-quantize with op types excluded (Softmax/LayerNorm/Gelu/Add/Mul/MatMul...).

  CUDA_VISIBLE_DEVICES= python int8_audit.py --models repvit_m0_9_224 tinyvit_5m_224
"""
import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_VIT = "/home/ubuntu/workspace/vision_models/ViT"  # runs/ and data/ are not tracked in worktrees
sys.path.insert(0, HERE)
sys.path.insert(0, MAIN_VIT)

DEFAULT_MODELS = ["repvit_m0_9_224", "repvit_m0_9_128", "kd_repvit_m0_9_128", "tinyvit_5m_224", "tinyvit_5m_128",
                  "fastvit_t8_224", "effformv2_s0_224", "iformer_t_224", "mnv4_cs_224", "deit_t_224",
                  "in64pre_tiny_64_kd"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="ec100_<name> run dirs")
    p.add_argument("--runs-dir", default=os.path.join(MAIN_VIT, "runs"))
    p.add_argument("--work", default=os.path.join(MAIN_VIT, "runs", "int8_audit"))
    p.add_argument("--out", default=os.path.join(HERE, "research", "int8_audit_results.jsonl"))
    p.add_argument("--threads", type=int, default=8, help="ORT threads for accuracy eval / calibration")
    p.add_argument("--lat-threads", type=int, default=4)
    p.add_argument("--calib", type=int, default=512)
    p.add_argument("--calibs", nargs="+", default=["minmax", "percentile", "entropy"])
    p.add_argument("--ablate", action="store_true", help="op-type exclusion ablations on each model's best calib")
    p.add_argument("--additive", nargs="*", default=None,
                   help="with --ablate: ONLY run 'Conv+Gemm+MatMul plus this one op type' variants")
    p.add_argument("--ablate-calib", default=None)
    p.add_argument("--exclude-types", nargs="*", default=[], help="op types left in fp32 in the main static variants")
    p.add_argument("--op-types", nargs="*", default=None, help="restrict static quantization to these op types")
    p.add_argument("--no-reparam", action="store_true", help="export the multi-branch (train-time) network")
    p.add_argument("--bs", type=int, default=250)
    p.add_argument("--test-subset", type=int, default=0)
    return p.parse_args()


def load_data(n_test):
    d = np.load(os.path.join(MAIN_VIT, "data", "cifar100_uint8.npz"))
    xte, yte = d["xtest"], d["ytest"]
    if n_test:
        xte, yte = xte[:n_test], yte[:n_test]
    rng = np.random.RandomState(0)
    ci = rng.choice(len(d["xtrain"]), 512, replace=False)
    return xte, yte, d["xtrain"][ci]


def prep(xu8, size, mean, std):
    x = torch.from_numpy(xu8).permute(0, 3, 1, 2).float().div_(255.0)
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=size < x.shape[-1] or True)
    m = torch.tensor(mean).view(1, 3, 1, 1)
    s = torch.tensor(std).view(1, 3, 1, 1)
    return ((x - m) / s).numpy().astype(np.float32)


def session(path, threads):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def accuracy(path, xte, yte, size, mean, std, threads, bs):
    sess = session(path, threads)
    name = sess.get_inputs()[0].name
    correct = 0
    for k in range(0, len(xte), bs):
        out = sess.run(None, {name: prep(xte[k:k + bs], size, mean, std)})[0]
        correct += int((out.argmax(1) == yte[k:k + bs]).sum())
    return 100.0 * correct / len(xte)


def latency(path, size, threads, warmup=10, runs=100):
    sess = session(path, threads)
    name = sess.get_inputs()[0].name
    x = np.random.RandomState(0).randn(1, 3, size, size).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {name: x})
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def size_mb(p):
    s = os.path.getsize(p)
    if os.path.exists(p + ".data"):
        s += os.path.getsize(p + ".data")
    return s / 2**20


def export_fp32(model, size, path):
    x = torch.randn(1, 3, size, size)
    eye = torch.eye

    def eye_export(n, *a, **k):  # ORT CPU has no bool EyeLike kernel (our LSA diagonal mask uses torch.eye(bool))
        if k.get("dtype") == torch.bool and not a:
            i = torch.arange(n, device=k.get("device"))
            return i[:, None] == i[None, :]
        return eye(n, *a, **k)
    torch.eye = eye_export
    try:
        torch.onnx.export(model, (x,), path, opset_version=17, input_names=["input"], output_names=["logits"],
                          dynamo=False, do_constant_folding=True,
                          dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    finally:
        torch.eye = eye
    import onnx
    from onnx import numpy_helper
    m = onnx.load(path)  # inline external data if any, keep a single file
    nfix = 0
    for n in m.graph.node:  # attention-mask -inf constants make calibration ranges infinite: use a large finite value
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    arr = numpy_helper.to_array(a.t)
                    if arr.dtype == np.float32 and np.isneginf(arr).any():
                        a.t.CopyFrom(numpy_helper.from_array(np.where(np.isneginf(arr), -3e4, arr).astype(np.float32)))
                        nfix += 1
    for i in m.graph.initializer:
        arr = numpy_helper.to_array(i)
        if arr.dtype == np.float32 and np.isneginf(arr).any():
            i.CopyFrom(numpy_helper.from_array(np.where(np.isneginf(arr), -3e4, arr).astype(np.float32), i.name))
            nfix += 1
    print("replaced -inf constants:", nfix, flush=True)
    onnx.save(m, path)


def make_reader(path, calib_x):
    from onnxruntime.quantization import CalibrationDataReader
    name = session(path, 1).get_inputs()[0].name

    class R(CalibrationDataReader):
        def __init__(self):
            self.i = 0

        def get_next(self):
            if self.i >= len(calib_x):
                return None
            b = calib_x[self.i:self.i + 1]  # batch 1 per calibration step
            self.i += 1
            return {name: b}

        def rewind(self):
            self.i = 0
    return R()


def _patch_hist_calibrator(chunk=16):
    """ORT's HistogramCalibrater keeps the outputs of EVERY calibration sample (all tensors) in RAM before
    building histograms (>100 GB for 512 samples of a 224px transformer). Feed the collector in chunks
    instead; HistogramCollector.collect merges histograms across calls."""
    from onnxruntime.quantization import calibrate as C

    def collect_data(self, data_reader):
        names_in = {a.name for a in self.infer_session.get_inputs()}
        out_names = [a.name for a in self.infer_session.get_outputs()]
        if not self.collector:
            self.collector = C.HistogramCollector(method=self.method, symmetric=self.symmetric,
                                                  num_bins=self.num_bins, num_quantized_bins=self.num_quantized_bins,
                                                  percentile=self.percentile, scenario=self.scenario)
        done = False
        while not done:
            merged = {}
            for _ in range(chunk):
                inputs = data_reader.get_next()
                if not inputs:
                    done = True
                    break
                outs = self.infer_session.run(None, inputs)
                for n, o in zip(out_names, outs):
                    if n in self.tensors_to_calibrate:
                        merged.setdefault(n, []).append(o.copy() if n in names_in else o)
            if merged:
                self.collector.collect(merged)
    C.HistogramCalibrater.collect_data = collect_data


def static_quant(src, dst, calib_x, method, op_types=None, exclude_types=(), exclude_names=()):
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat, QuantType, quantize_static)
    import onnx
    _patch_hist_calibrator()
    cm = {"minmax": CalibrationMethod.MinMax, "percentile": CalibrationMethod.Percentile,
          "entropy": CalibrationMethod.Entropy}[method]
    extra = {"ActivationSymmetric": False, "WeightSymmetric": True}
    if method == "percentile":
        extra["CalibPercentile"] = 99.999
    exclude = list(exclude_names)
    if exclude_types:
        g = onnx.load(src).graph
        exclude = [n.name for n in g.node if n.op_type in exclude_types]
    quantize_static(src, dst, make_reader(src, calib_x), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, per_channel=True,
                    calibrate_method=cm, op_types_to_quantize=op_types, nodes_to_exclude=exclude,
                    extra_options=extra)


def preprocess_onnx(src, dst):
    from onnxruntime.quantization.shape_inference import quant_pre_process
    try:
        quant_pre_process(src, dst, skip_symbolic_shape=False)
    except Exception:
        quant_pre_process(src, dst, skip_symbolic_shape=True)


def op_hist(path):
    import onnx
    return dict(Counter(n.op_type for n in onnx.load(path, load_external_data=False).graph.node))


def audit_model(name, args, xte, yte, xtr):
    from finetune import load_model
    from timm.utils.model import reparameterize_model
    t0 = time.time()
    run = os.path.join(args.runs_dir, f"ec100_{name}")
    res = json.load(open(os.path.join(run, "results.json")))
    model, spec = load_model(run)
    size = spec["img_size"]
    mean, std = spec["mean"], spec["std"]
    try:
        if args.no_reparam:
            raise RuntimeError('--no-reparam')
        model = reparameterize_model(model)
    except Exception as e:
        print("reparam skipped:", e)
    model = model.float().eval()
    os.makedirs(args.work, exist_ok=True)
    base = os.path.join(args.work, name + ("_norep" if args.no_reparam else ""))
    fp = base + "_fp32.onnx"
    export_fp32(model, size, fp)
    rec = {"model": name, "img_size": size, "params_M": res.get("params_M"),
           "reported_acc": res["final_ema_val_acc"] if spec.get("source", "").endswith("ema.pt") else res["final_val_acc"],
           "ckpt": spec.get("source"), "ops_fp32": op_hist(fp)}
    # torch parity
    with torch.no_grad():
        xp = torch.from_numpy(prep(xte[:8], size, mean, std))
        ref = model(xp).numpy()
    o = session(fp, 1).run(None, {"input": xp.numpy()})[0]
    rec["onnx_parity_maxdiff"] = float(np.abs(o - ref).max())
    del model
    acc = lambda p: accuracy(p, xte, yte, size, mean, std, args.threads, args.bs)
    rec["fp32_acc"] = acc(fp)
    rec["fp32_lat_ms"] = latency(fp, size, args.lat_threads)
    rec["fp32_MB"] = size_mb(fp)
    print(name, "fp32", rec["fp32_acc"], rec["fp32_lat_ms"], flush=True)
    calib_x = prep(xtr[:args.calib], size, mean, std)
    variants = {}

    def record(tag, path):
        v = {"acc": acc(path), "lat_ms": latency(path, size, args.lat_threads), "MB": size_mb(path)}
        variants[tag] = v
        print(name, tag, v, flush=True)

    # dynamic
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        dp = base + "_dyn.onnx"
        quantize_dynamic(fp, dp, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])
        record("dynamic", dp)
    except Exception as e:
        variants["dynamic"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    # static
    pp = base + "_pp.onnx"
    try:
        preprocess_onnx(fp, pp)
    except Exception as e:
        pp = fp
        rec["preprocess_error"] = str(e)[:200]
    for cm in args.calibs:
        qp = f"{base}_qdq_{cm}.onnx"
        try:
            static_quant(pp, qp, calib_x, cm, op_types=args.op_types, exclude_types=tuple(args.exclude_types))
            record(f"static_{cm}", qp)
            os.path.exists(qp) and variants[f"static_{cm}"].update(ops=op_hist(qp))
        except Exception as e:
            variants[f"static_{cm}"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            print(name, cm, "ERR", e, flush=True)
    rec["variants"] = variants
    rec["exclude_types"] = list(args.exclude_types)
    rec["op_types"] = args.op_types
    good = {k: v for k, v in variants.items() if k.startswith("static_") and "acc" in v}
    if good:
        rec["best_static"] = max(good, key=lambda k: good[k]["acc"])
    if args.ablate and good:
        cm = args.ablate_calib or rec["best_static"].split("_")[1]
        ab = {}
        base_ops = rec["variants"][f"static_{cm}"].get("ops", {})
        ab_defs = {"conv_gemm_matmul_only": dict(op_types=["Conv", "Gemm", "MatMul"]),
                   "conv_gemm_only": dict(op_types=["Conv", "Gemm"]),
                   "conv_only": dict(op_types=["Conv"]),
                   "matmul_gemm_only": dict(op_types=["Gemm", "MatMul"])}
        for t in ["Softmax", "Add", "Mul", "MatMul", "Gemm", "Conv", "Gelu", "Sigmoid", "Relu", "HardSwish", "Resize",
                  "AveragePool", "GlobalAveragePool", "Concat", "ReduceMean"]:
            ab_defs[f"excl_{t}"] = dict(exclude_types=(t,))
        import onnx
        convs = [n for n in onnx.load(pp).graph.node if n.op_type == "Conv"]
        inits = {i.name: i for i in onnx.load(pp).graph.initializer}

        def groups(n):
            return next((a.i for a in n.attribute if a.name == "group"), 1)
        dw = [n.name for n in convs if groups(n) > 1]
        pwc = [n.name for n in convs if groups(n) == 1]
        ab_defs["excl_depthwise_convs"] = dict(exclude_names=dw)
        ab_defs["excl_dense_convs"] = dict(exclude_names=pwc)
        k = max(1, len(convs) // 4)
        for qi in range(4):
            ab_defs[f"excl_conv_quarter{qi + 1}"] = dict(exclude_names=[n.name for n in convs[qi * k:(qi + 1) * k if qi < 3 else None]])
        ab_defs["excl_first_conv"] = dict(exclude_names=[convs[0].name])
        rec["n_conv"], rec["n_depthwise"] = len(convs), len(dw)
        if args.additive is not None:
            ab_defs = {f"conv_gemm_matmul+{t}": dict(op_types=["Conv", "Gemm", "MatMul", t]) for t in args.additive}
        for tag, kw in ab_defs.items():
            qp = f"{base}_ab_{tag}.onnx"
            try:
                static_quant(pp, qp, calib_x, cm, **kw)
                nq = op_hist(qp).get("QuantizeLinear", 0)
                if tag.startswith("excl_") and "names" not in str(kw.keys()) and nq == base_ops.get("QuantizeLinear"):
                    continue  # op type not present / no effect
                ab[tag] = {"acc": acc(qp), "lat_ms": latency(qp, size, args.lat_threads), "MB": size_mb(qp), "n_QL": nq}
                print(name, "ablate", tag, ab[tag], flush=True)
            except Exception as e:
                ab[tag] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            if os.path.exists(qp):
                os.remove(qp)
        rec["ablation"] = {"calib": cm, "results": ab}
    rec["audit_sec"] = time.time() - t0
    return rec


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    xte, yte, xtr = load_data(args.test_subset)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for name in args.models:
        try:
            rec = audit_model(name, args, xte, yte, xtr)
        except Exception as e:
            import traceback
            traceback.print_exc()
            rec = {"model": name, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        rec["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
