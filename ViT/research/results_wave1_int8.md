# Wave 1 / A4: INT8 accuracy and latency audit (ONNX Runtime CPU, 2026-10-11)

Code: `ViT/int8_audit.py` (table: `ViT/int8_table.py`). Raw records: `ViT/research/int8_*.jsonl`.
Models: our fine-tuned CIFAR-100 runs `ec100_*` (reported checkpoint via `finetune.load_model`: EMA iff it was better), fused
(`reparameterize_model`), exported FP32 ONNX (opset 17, dynamic batch), ORT 1.23 CPU. Accuracy: full 10k CIFAR-100 test set, same
preprocessing as training (bilinear+antialias resize, model mean/std). FP32 ORT accuracy matches the training-time number to <=0.15 pt.

Variants
- **dynamic**: `quantize_dynamic` MatMul/Gemm, int8 weights, activations quantized on the fly (convs stay FP32; ORT CPU has no ConvInteger).
- **static**: `quantize_static`, QDQ format, per-channel int8 weights, int8 (QInt8, asymmetric) activations, ORT default op set
  (Conv, MatMul, Gemm, Add, Mul, Softmax, ... ), 512 fixed-seed CIFAR-100 train images, calibration MinMax / Percentile(99.999) / Entropy.
  (ORT's histogram calibrators keep every sample's tensors in RAM, >100 GB for 512 samples at 224 px; the script patches them to feed
  the collector in chunks of 16. Histograms merge across chunks.)
- Latency: single image, 4 intra-op threads, 10 warmup + median of 100 runs, on a **shared Xeon 8480+ (other jobs running: load avg 14-40)
  so latency is noisy (easily +-20%, and ms numbers for different models were taken at different times)**. Size = .onnx file.

## Main table (accuracy in %, top-1 on 10k test)

| model | res | FP32 acc | dyn INT8 acc (MatMul/Gemm) | static MinMax | static Pctl | static Entropy | best static Δ | lat ms FP32 / dyn / best-static | size MB FP32 / dyn / static |
|---|---|---|---|---|---|---|---|---|---|
| repvit_m0_9_224 | 224 | 88.49 | 88.55 (+0.06) | 21.02 | 37.63 | 31.26 | -50.86 (percentile) | 6.7 / 6.6 / 20.8 | 18.1 / 18.0 / 5.3 |
| repvit_m0_9_128 | 128 | 87.33 | 87.32 (-0.01) | 43.36 | 50.05 | 48.32 | -37.28 (percentile) | 3.1 / 3.1 / 8.6 | 18.1 / 18.0 / 5.3 |
| kd_repvit_m0_9_128 | 128 | 87.87 | 87.82 (-0.05) | 39.75 | 56.69 | 53.11 | -31.18 (percentile) | 3.1 / 3.1 / 8.5 | 18.1 / 18.0 / 5.3 |
| tinyvit_5m_224 | 224 | 88.25 | 88.08 (-0.17) | 59.01 | 68.75 | 54.96 | -19.50 (percentile) | 14.8 / 13.0 / 35.2 | 20.0 / 6.6 / 6.9 |
| tinyvit_5m_128 | 128 | 86.27 | 86.16 (-0.11) | 64.60 | 74.91 | 74.23 | -11.36 (percentile) | 7.5 / 6.9 / 15.8 | 20.2 / 6.9 / 7.0 |
| fastvit_t8_224 | 224 | 85.53 | 85.53 (+0.00) | 29.80 | 39.32 | 35.61 | -46.21 (percentile) | 6.4 / 6.6 / 31.8 | 12.6 / 12.4 / 3.6 |
| effformv2_s0_224 | 224 | 85.30 | 85.27 (-0.03) | 37.22 | 68.65 | 36.57 | -16.65 (percentile) | 7.2 / 7.8 / 23.6 | 12.6 / 12.5 / 4.1 |
| iformer_t_224 | 224 | 85.08 | 85.06 (-0.02) | 69.82 | 71.68 | 70.41 | -13.40 (percentile) | 5.7 / 5.3 / 17.2 | 10.2 / 10.2 / 3.2 |
| mnv4_cs_224 | 224 | 80.15 | 80.10 (-0.05) | 78.86 | 79.15 | 79.60 | -0.55 (entropy) | 1.3 / 1.4 / 2.5 | 10.0 / 9.6 / 2.8 |
| deit_t_224 | 224 | 84.47 | 84.16 (-0.31) | 82.95 | 83.59 | 83.41 | -0.88 (percentile) | 11.5 / 7.9 / 20.0 | 21.3 / 6.1 / 5.9 |
| in64pre_tiny_64_kd | 64 | 81.46 | 81.48 (+0.02) | 32.03 | 31.59 | 30.87 | -49.43 (minmax) | 5.3 / 4.9 / 9.1 | 20.8 / 5.6 / 5.8 |

Delta is static (best calibration, chosen on the test set, so slightly optimistic) minus FP32. Dynamic delta is in parentheses in its column.
Reported FP32 numbers in training logs: see `results.json` of each run; ORT FP32 here is within 0.15 pt.

## Findings

1. **Dynamic INT8 (MatMul/Gemm only) is accuracy-free** (|delta| <= 0.31 pt for all 11 models; our IN-64-pretrained ViT +0.02) and shrinks the
   size of transformer models ~3x (TinyViT 20.0 -> 6.6 MB, DeiT-T 21.3 -> 6.1 MB, our ViT 20.8 -> 5.6 MB). But for conv-dominated models
   (RepViT, FastViT, EffFormerV2, iFormer, MNv4) almost all weights are in convs and stay FP32, so there is no size win (RepViT 18.1 -> 18.0 MB)
   and no latency win. Latency gains are modest and inside the noise except for the ViTs (DeiT-T 11.5 -> 7.9 ms, our ViT 5.3 -> 4.9, TinyViT-224 14.8 -> 13.0).
2. **Naive static INT8 PTQ (QDQ, per-channel, 512 images) fails for most of our edge models**: RepViT-M0.9 -37 to -51 pt, FastViT-T8 -46, our ViT -49,
   TinyViT-5M -11 (128 px) / -20 (224 px), iFormer-T -13, EffFormerV2-S0 -17 (only with Percentile; MinMax/Entropy -48/-49).
   Only MobileNetV4-Conv-S (-0.55, entropy) and DeiT-T (-0.88, percentile) are usable. Percentile is best for all models except MNv4
   (entropy) and our ViT (all three equally bad, MinMax marginally best); MinMax is worst in nearly all cases. This reproduces and extends the
   EffFormerV2 PTQ-drop warning in NEXT_STEPS (76.2 -> 68.2 on ImageNet there; 85.3 -> 68.7 here).
3. **Static INT8 was slower than FP32 on this CPU in our ORT QDQ setup** (e.g. RepViT-224 6.7 -> 20.8 ms, TinyViT-224 14.8 -> 35 ms, MNv4 1.3 -> 2.1-2.5 ms,
   DeiT-T 11.5 -> 20 ms) despite a 3-4x smaller file. QDQ graphs are only partly fused into QLinear kernels; the many Q/DQ nodes around
   non-fused ops (Add, Mul, Softmax, attention reshapes) cost more than the int8 GEMMs save. Not an accuracy-vs-latency tradeoff worth taking as is.
   Latency here is not a statement about a real NPU/mobile int8 runtime, which would need its own op-set-aware export.
4. **Where the drop comes from (ablations; calibration Percentile for RepViT, MinMax for our ViT)**
   - *Our IN-64 ViT* (31-32% static): quantizing **only Conv/Gemm/MatMul (incl. the attention matmuls) gives 81.55 (+0.09 vs FP32)**. Adding op types
     one at a time to that set (additive test): +Mul 81.55, +Concat 81.62, +Where 81.55, +LayerNorm 81.46, +Erf 81.58, +Div 81.59, +Transpose 81.55
     all fine; **+Add 31.92 and +Softmax 32.31 are the two culprits**. Residual-stream Add: the ViT residual carries a few large-magnitude
     channels, and per-tensor int8 on it destroys them (standard ViT quantization issue; plausible, not separately verified with per-channel stats).
     Softmax: the LSA/diagonal mask is exported as a Where with -inf; ORT cannot calibrate infinite ranges, so for this audit the -inf constants are replaced by -3e4
     (finite) before quantizing (parity with PyTorch preserved), but then the Softmax *input* range spans [-3e4, x] and the scale is useless. This part is an
     artifact of masked attention under PTQ; a fix is to keep Softmax (and the masked Add) in FP32.
     Single-op-type exclusions from the full default set (excl_Softmax, excl_Add, ...) do not recover accuracy because both culprits are present together.
     (Our ViT's export needed one workaround: `torch.eye(bool)` -> arange comparison, since ORT CPU has no bool EyeLike kernel.)
   - *RepViT-M0.9 @128* (worst conv model; 50.05 static-percentile vs 87.33 FP32): quantizing only Conv+Gemm+MatMul: 49.65, only Conv: 49.76,
     only Gemm/MatMul (= head): 87.38; excluding the first conv: 49.9; excluding depthwise convs: 41.6; excluding any one quarter of the 103 convs: 45-48.
     Quantizing everything except Conv (activations on Add/Mul only): 35.2. No single layer or group of layers is responsible: error accumulates over ~100
     BN-fused convs with per-tensor int8 activations. Not caused by re-parameterization either: the unfused multi-branch network is *worse* (1-2.4%).
     Restricting the op set does not help conv models: with Conv/Gemm/MatMul only, iFormer 72.2, EffFormerV2 68.3, FastViT 42.8, MNv4 77.4 (worse than default 79.6).
     Restricting helps only the pure-ViT models (our ViT 81.51, DeiT-T 84.28 vs 83.59 default, TinyViT-128 75.05 vs 74.91, TinyViT-224 72.28 vs 68.75 default; all still well below FP32 for TinyViT).
5. **QAT (task step 4): not run.** Every static model that fails would need torch.ao FX QAT plus a quantized-ONNX export that ORT executes in int8, and the
   post-QAT latency of the current ORT QDQ path is worse than FP32 anyway; the rational next step is per-model: (a) ViTs: keep Add/Softmax/LN in FP32 (Conv/Gemm/MatMul int8 recovers
   accuracy for our ViT and DeiT-T), (b) RepViT/FastViT: QAT or a better activation quantizer (per-channel/percentile-clipped, or mixed precision), measured on the target
   runtime (TFLite/QNN/CoreML), not ORT-x86. Gated GPU QAT was not started.

## Caveats
- Single seed/calibration subset (512 images, seed 0); no confidence intervals. Differences < ~0.3 pt (10k test: s.e. ~0.35 pt) are not meaningful.
- Best calibration is selected on the test set (optimistic by a small margin; the 3 calibration choices differ by up to 30 pt for the failing models, which dominates).
- Latency noisy (shared CPU, 4 threads, other jobs). ORT x86 QDQ is not an edge-NPU proxy; size is the more reliable number.
- TinyViT 128 px and 224 px use the same weights family but different fine-tuned checkpoints; KD-RepViT-128 is the soft-KD run.
