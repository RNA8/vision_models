import json
last={}
for l in open("research/int8_audit_results.jsonl"):
    r=json.loads(l)
    if "fp32_acc" in r: last[r["model"]]=r
order=["repvit_m0_9_224","repvit_m0_9_128","kd_repvit_m0_9_128","tinyvit_5m_224","tinyvit_5m_128","fastvit_t8_224","effformv2_s0_224","iformer_t_224","mnv4_cs_224","deit_t_224","in64pre_tiny_64_kd"]
print("| model | res | FP32 acc | dyn INT8 acc (MatMul/Gemm) | static MinMax | static Pctl | static Entropy | best static Δ | lat ms FP32 / dyn / best-static | size MB FP32 / dyn / static |")
print("|---|---|---|---|---|---|---|---|---|---|")
for m in order:
    r=last[m];v=r["variants"];f=r["fp32_acc"]
    g=lambda k:v[k]["acc"] if k in v and "acc" in v[k] else float("nan")
    b=r["best_static"];bv=v[b]
    print(f"| {m} | {r['img_size']} | {f:.2f} | {g('dynamic'):.2f} ({g('dynamic')-f:+.2f}) | {g('static_minmax'):.2f} | {g('static_percentile'):.2f} | {g('static_entropy'):.2f} | {bv['acc']-f:+.2f} ({b[7:]}) | {r['fp32_lat_ms']:.1f} / {v['dynamic']['lat_ms']:.1f} / {bv['lat_ms']:.1f} | {r['fp32_MB']:.1f} / {v['dynamic']['MB']:.1f} / {bv['MB']:.1f} |")
