# Static INT8 (PTQ + QAT) for the 2.75M SL-ViT (SPT + LSA + SeqPool), CIFAR-100 (2026-10-13)

Code: `ViT/qat_int8.py` (reuses `ViT/int8_audit.py` for the ORT PTQ baseline, accuracy and latency, so the numbers are comparable with
`results_wave1_int8.md`). Raw results: `ViT/research/qat_int8_ort_ptq.json` (ORT PTQ + ablations), `ViT/research/qat_int8_latency.json`
(interleaved latency bench), `ViT/runs/{ptq_*,qat_*}/results.json` (our quantizer placement: PTQ ablations and QAT runs; incl. every
learned scale/zero-point).

Model: `runs/c100_spt_slvit_1000ep/model.pt` (seed 0; 9 blocks, d=192, 12 heads, MLP ratio 2, patch 4, SPT, LSA, SeqPool; 2.75M params;
reported FP32 80.66, 80.64 in our eval). Seeds `_s1`, `_s2` used as a robustness check. Test = full 10k CIFAR-100 test set.

## TL;DR

- **The full static-PTQ collapse (80.64 -> 11%) is not a ViT-quantization problem; it is an export artifact of the LSA diagonal mask.**
  The timm/SDPA export adds `Where(mask, 0, -inf)` to the attention logits (the audit replaces -inf by -3e4 so ORT can calibrate);
  ORT then quantizes that Add output and the Softmax input per-tensor over a range of [-3e4, x], so every real logit lands in one
  int8 bin. Leaving only those 9 mask-Adds + 9 attention Softmaxes un-quantized (everything else in ORT's default set, incl. the residual
  Adds, LayerNorm, GELU/Erf, Mul, seqpool softmax, quantized) gives **80.47** (Percentile). Additive ablation: MatMul/Gemm + `Add` = 11.75,
  MatMul/Gemm + `Softmax` = 12.16, every other op type ~80.3-80.8.
- **With a deployment-correct placement (quantize the q k^T logits *before* the mask; mask applied inside the softmax), full static INT8 is
  already within ~0.3-0.5 pt by plain PTQ** (seed 0: 80.31 fake-quant / 80.26 ORT vs 80.64; seed 1: 80.71 ORT vs 81.21).
  135 per-tensor int8 activation quantizers (every tensor between ops, incl. residual stream, LN outputs, softmax in/out, GELU in/out,
  logits) + per-channel int8 weights + int32 biases.
- **QAT (LSQ/LSQ+ learned scales and offsets, KD from the FP32 model, 10-30 epochs, lr 2e-5)**: best ORT INT8 **80.53 (-0.11 vs 80.64)**
  (seed 0, 30 epochs). QAT does not move top-1 beyond the test-set noise (all seed-0 variants lie in 80.2-80.7, s.e. of one accuracy
  ~0.4 pt), but it raises top-1 *agreement* with the FP32 model (3-seed mean, RandAug+mixup KD, 10 ep: 95.50% PTQ -> 96.31% QAT). **3-seed mean, ORT INT8: FP32 80.91, full static PTQ 80.50 (-0.41),
  full static QAT 80.65 (-0.26)**; worst single seed QAT -0.37 (seed 0), all three within the 0.5 pt goal.
- **What stays float**: nothing is stored or passed between ops in float. LayerNorm, Softmax (+ the diagonal mask) and GELU(erf) *compute*
  in float from int8 inputs to int8 outputs (QDQ "DQ -> op -> Q", the standard ORT/QNN representation); biases are int32. Mixed precision
  is *not needed* to reach the goal; int16 residual stream (A16) is the one place where it measurably helps PTQ (80.31 -> 80.66).
- **Size**: 2.99 MB INT8 ONNX vs 10.55 MB FP32 (3.5x). **Latency** (x86 Xeon, ORT 1.23, batch 1, 4 threads, shared box, noisy): our
  QAT QDQ model 3.38 ms vs 2.71 ms FP32 (same explicit graph) / 3.48 ms FP32 (timm export); ORT's own static QDQ models are 5.6-6.8 ms.
  INT8 is not a latency win on this CPU (as in the wave-1 audit); it is a size/NPU-compatibility win.

## Main table (seed 0, top-1 % on 10k test; ORT CPU unless marked)

| variant | what is int8 | top-1 | delta vs FP32 | agree w/ FP32 | size MB | lat ms (b=1, 4 thr) |
|---|---|---|---|---|---|---|
| FP32 (timm export, audit tooling) | - | 80.64 | 0 | 100 | 10.62 | 3.48 |
| FP32 (our explicit graph) | - | 80.64 | 0 | 100 | 10.55 | 2.71 |
| dynamic INT8 (MatMul/Gemm) | weights; acts quantized on the fly | 80.62 | -0.02 | - | 2.94 | 3.02 |
| ORT static PTQ, full default op set (MinMax / Pctl / Entropy) | all ORT default ops | 11.31 / 10.86 / 10.81 | -69 | - | 3.09 | 6.82 |
| ORT static PTQ, MatMul/Gemm only (MinMax / Pctl / Entropy) | Linear + attention matmuls | 80.66 / 80.57 / 80.83 | +0.02 / -0.07 / +0.19 | - | 3.02 | 5.64 |
| ORT static PTQ, full minus mask-Add + attn-Softmax (MinMax / Pctl / Entropy) | all default ops except 9 mask Adds, 9 attn Softmax | 79.96 / 80.47 / 80.03 | -0.68 / -0.17 / -0.61 | - | 3.08 | 6.47 |
| ORT static PTQ, full minus all Add + all Softmax (Entropy) | | 80.26 | -0.38 | - | 3.11 | - |
| **ours, full static PTQ** (MSE-calibrated, 512 imgs) | everything (135 act. quantizers, see below) | 80.26 (fake-quant 80.31) | -0.38 | 94.77 | 2.99 | 3.74* |
| ours, full static QAT, 10 ep, crop/flip KD | everything | 80.36 (fq 80.41) | -0.28 | - | 2.99 | 3.84* |
| **ours, full static QAT, 30 ep, crop/flip KD** | everything | **80.53** (fq 80.47) | **-0.11** | - | 2.99 | **3.38** |
| ours, full static QAT, 10 ep, RandAug+mixup/cutmix KD | everything | 80.27 (fq 80.20) | -0.37 | 95.75 (fq 95.83) | 2.99 | 3.74* |

Model files: `runs/qat_int8_work/<run>.onnx` (not committed; regenerate with the commands in the script docstring).

Latency without `*`: interleaved bench (`--mode bench`: all models alternated, 7 rounds x median of 300 batch-1 runs, median over rounds,
load avg 9-12 from other jobs). `*`: single measurement at the end of that run (different load; same 2.99 MB graph structure as the
benched QAT model, so expect ~3.4 ms). "agree" = % of test images where the INT8 top-1 class equals the FP32 top-1 class (paired, less
noisy than accuracy; recorded only for runs started after it was added).
fq = PyTorch fake-quant accuracy of the same model on GPU.

### Seeds (robustness check; same recipe, no tuning)

| seed | FP32 (our eval) | full static PTQ, ORT (agree) | full static QAT 10 ep strong-aug KD, ORT (agree) | QAT delta |
|---|---|---|---|---|
| 0 | 80.64 | 80.26 (94.77) | 80.27 (95.75) | -0.37 |
| 1 | 81.21 | 80.71 (96.46) | 80.98 (96.94) | -0.23 |
| 2 | 80.88 | 80.52 (95.28) | 80.71 (96.24) | -0.17 |
| mean | 80.91 | 80.50 (95.50), -0.41 | **80.65 (96.31), -0.26** | |

QAT improves over PTQ on 3/3 seeds in agreement (+0.5 to +1.0 pt) and on 3/3 in top-1 (+0.01, +0.27, +0.19); the top-1 gains are
individually inside the noise. Seed 0 with 30-epoch crop/flip KD reached 80.53 (-0.11), the best single INT8 number.

## Our quantizer placement ("full static")

Per transformer block (x9), each arrow is an int8 tensor (per-tensor, asymmetric, learned scale s and zero-point z):
`x_res(int8) -> LN1 -> [ln1] -> qkv (W int8 per-out-channel, b int32) -> [q] [k] [v] -> q k^T -> [logit] -> mask diag + softmax -> [prob]
-> prob @ v -> [ao] -> proj -> [po] -> Add -> [r1] -> LN2 -> [ln2] -> fc1 -> [h] -> GELU -> [g] -> fc2 -> [fo] -> Add -> [r2]`.
Stem/head: `[input] -> SPT shifts/concat/patchify (exact on int8) -> LN -> [spt_ln] -> proj -> [spt_out] -> concat cls (int8 const) + pos
(int8 const) -> [res0]` and `final LN -> [norm] -> seqpool linear -> [sp_logit] -> softmax -> [sp_w] -> w^T x -> [pooled] -> head -> [logits]`.
135 activation quantizers in total. The LSA per-head temperature is folded exactly into the q rows of W_qkv / b_qkv, so there is no extra
Mul. The diagonal mask is applied after the int8 logit quantizer with a finite -1e4 (identical to -inf after exp in fp32), i.e. inside the
float softmax; on an integer NPU this is a masked-softmax (or set-to-qmin) and does not need its own quantization range.

Groups used for the ablation (`--groups`): `input`; `w` (weights, cls/pos constants; biases follow as int32 when the input is quantized);
`mm` (all inputs of Linear/MatMul: LN outputs, q, k, v, softmax probs, attention output, GELU output, seqpool weights, pooled feature);
`linout` (outputs of Linear layers: fc1 output = GELU input, branch outputs before the residual Add, logits); `softmax` (softmax inputs);
`residual` (the residual stream after every Add).

### PTQ ablation with our placement (seed 0, PyTorch fake-quant, GPU; MSE calibration unless noted)

| quantized groups | float groups | top-1 |
|---|---|---|
| weights only (`w`) | all activations | 80.62 |
| `mm`+`w` (MatMul/Gemm-only analogue) | input, linout, softmax, residual | 80.71 |
| `mm`+`w`+`input` | linout, softmax, residual | 80.66 |
| all activations, float weights | w | 80.31 |
| full, minus `residual` | residual | 80.49 |
| full, minus `softmax` | softmax inputs | 80.21 |
| full, minus `linout` | linout | 80.46 |
| **full** (MSE / MinMax / Percentile 99.99) | none | **80.31 / 80.41 / 79.93** |
| full, residual stream int16 (A16W8 there) | none (int16 residual) | 80.66 |
| full, residual + linout int16 | none | 80.55 |

Reading: no single group costs more than ~0.3 pt; the int8 residual stream is the most likely contributor (A16 residual recovers
exactly FP32 accuracy under PTQ; "minus residual" 80.49), consistent with Bondarenko et al. (outliers in residual sums), but the effect is at
the edge of the test-set noise here (residual-stream ranges are modest for this small CIFAR model: last block |r| range ~ +-25 with
scale 0.2). The dominant failure in ORT's PTQ is the mask artifact, not any of these groups.

## Method and exact settings

- **Toolchain choice.** PyTorch's `torch.ao.quantization` is being removed in favour of torchao PT2E (docs, 2.14: "We are centralizing
  all quantization related development to torchao"; "We plan to delete torch.ao.quantization in 2.10 ..."), torchao is not installed in
  our venv, and PT2E's documented lowering paths are Inductor and ExecuTorch ("Currently there are two typical lowering paths, 1.
  torch.compile through inductor lowering 2. ExecuTorch through delegation"), not ONNX. Because we need an ORT-evaluable static INT8
  artifact (and NPU toolchains such as QNN consume QDQ ONNX), we use the third option: **ONNX-side QDQ with our own fake-quant QAT in
  PyTorch** (ORT docs list "Quantization-Aware training (QAT) models ... exported from PyTorch" as a QDQ source). The fake-quant
  forward is written so its export is exactly the QDQ graph: activations via a custom `autograd.Function` whose ONNX symbolic emits
  `QuantizeLinear -> DequantizeLinear` (int16: `com.microsoft` domain), weights as int8 initializers + per-channel `DequantizeLinear`
  (axis 1 on W^T, ORT's MatMul convention), biases as int32 initializers with scale s_in * s_w (Jacob et al.). (Pitfalls fixed:
  the TorchScript exporter inlines autograd Functions under `torch.no_grad()` and with `autograd_inlining=True`.)
- **Quantizers.** Activations: per-tensor, asymmetric int8 [-128, 127], learnable step s and offset z (LSQ / LSQ+), gradient scale
  g = 1/sqrt(N * Q_P) (LSQ), STE through round. Weights: per-output-channel symmetric int8 [-127, 127], learnable step (LSQ).
  Init/calibration: 512 fixed-seed CIFAR-100 train images (the same RandomState(0) subset as int8_audit), activation ranges by MSE-optimal
  clipping (grid search of 40 shrink factors over the min/max range, minimising the quantization MSE on up to 200k sampled values per
  tensor per batch), weights by per-channel MSE over 26 clip ratios in [0.5, 1].
  Per-tensor activations are what integer GEMM kernels / NPUs support without re-scaling inside the reduction; per-channel weights are free.
- **QAT.** Start from the FP32 checkpoint (LSA temperature folded). AdamW, weights lr 2e-5, quantizer params lr 1e-4, wd 0, cosine to 0,
  batch 256, no warmup, grad-clip 1.0, no drop-path, loss = KL(teacher || student) with the frozen FP32 model as teacher (T=1, kd=1, no
  label loss). Augmentation: (a) crop(4, reflect)+flip; (b) "strong" = crop/flip + RandAugment(2, 9) (train_vit.rand_augment) + mixup
  (alpha 0.8) / cutmix (alpha 1), 50/50 per batch, KD only. Epochs 10 or 30. ~120 s/epoch on the shared H100 (4 other jobs, no compile).
- **Evaluation.** PyTorch fake-quant on GPU, then ONNX export (opset 17) and ORT 1.23 CPUExecutionProvider on the full 10k test set
  (same preprocessing as int8_audit). ORT fuses the graph into 37 `MatMulIntegerToFloat` (all Linear layers incl. SPT), 19 `QLinearMatMul`
  (q k^T, prob @ v, seqpool w^T x), 1 `QGemm` (head), 1 `QLinearSoftmax` (seqpool); LayerNorm, attention Softmax, Erf/GELU, residual
  Adds stay as DQ -> float op -> Q.
  Parity: our PyTorch export-mode forward equals the fake-quant forward bit-exactly; ORT vs PyTorch on the same int8 model agree on
  96.5-98.8% of argmaxes with max logit diff ~1-1.8. This is not a bug: the same fake-quant model in float64 vs float32 PyTorch also
  disagrees on 2.4% of argmaxes - 135 cascaded rounding steps make borderline values flip with tiny float-accumulation differences, and
  the flips are accuracy-neutral (ORT and PyTorch accuracies differ by <=0.07 pt in every run).
- **Latency.** `qat_int8.py --mode bench`: all ONNX files alternated for 7 rounds, each round median of 300 batch-1 runs after 20 warmup,
  4 intra-op threads, report the median over rounds. Box: Xeon 8480+, load average 9-12 from other jobs.

## Caveats

- Accuracy differences below ~0.4 pt are within test-set noise (10k images, binomial s.e. ~0.4 pt for one model; paired comparisons
  are tighter, which is why agreement with FP32 is reported). The "best" QAT number (80.53) is one run; QAT variants span 80.27-80.53 in
  ORT for seed 0 and we did not select hyper-parameters on a held-out set (none were tuned; the recipes were fixed a priori; the strong-aug
  recipe was added because crop/flip KD gives almost no training signal: the teacher fits the training set at 99.96%, KD loss ~1e-3).
- Calibration choice for the ORT rows is selected on the test set (as in the audit); differences between calibrators are small here.
- "Float compute inside LN/Softmax/GELU with int8 I/O" is what ORT and QDQ-consuming NPU compilers do; an integer-only target (no float
  unit) would need integer approximations (I-ViT's Shiftmax/ShiftGELU/I-LayerNorm) or LUTs, not evaluated here.
- The diagonal LSA mask must be implemented as a masked softmax (or by writing qmin into the diagonal) on the target; quantizing a
  -inf/-3e4 mask tensor per-tensor is what destroyed ORT's PTQ.
- x86 ORT latency is not an NPU proxy. The QDQ model is ~25% slower than our FP32 explicit graph on this CPU (it is 2x faster than ORT's
  own static-QDQ output because our graph has fewer unfused Q/DQ islands); size is the reliable number (3.5x smaller).
- One MatMul (seqpool Linear 192->1) is left unfused by ORT (its inputs are still int8-DQ'd; numerically identical).

## Sources (fetched 2026-10-13; quotes verbatim)

- FQ-ViT, Lin et al., IJCAI 2022, https://arxiv.org/abs/2111.13824 - "many of these difficulties arise because of serious inter-channel
  variation in LayerNorm inputs" (PTF); "observing an extreme non-uniform distribution in attention maps, we propose Log-Int-Softmax (LIS)";
  "Previous works [Liu et al., 2021b] chose to not quantize Softmax because the negligibility of calculation amount in Softmax, and
  quantization may lead to significant accuracy degradation."
- PTQ4ViT, Yuan et al., https://arxiv.org/abs/2111.12293 - "We find the problems in PTQ on vision transformers are special distributions of
  post-softmax and post-GELU activations and the inaccurate metric."; "The distribution of activations after softmax is very unbalanced, in
  which most values are very close to zero and only a few values are close to one."
- RepQ-ViT, Li et al., ICCV 2023, https://arxiv.org/abs/2212.08254 - "post-LayerNorm activations with severe inter-channel variation and
  post-Softmax activations with power-law features"; scales are reparameterized to layer-wise quantization "by adjusting the LayerNorm's
  affine factors and the next layer's weights".
- I-ViT, Li & Gu, ICCV 2023, https://arxiv.org/abs/2207.01405 - "to enable ViTs to perform the entire computational graph of inference with
  integer arithmetic and bit-shifting, and without any floating-point arithmetic"; "non-linear operations (e.g., Softmax, GELU, and
  LayerNorm) are approximated by the proposed light-weight integer-only arithmetic methods."
- LSQ, Esser et al., ICLR 2020, https://arxiv.org/abs/1902.08153 - "estimate and scale the task loss gradient at each weight and activation
  layer's quantizer step size, such that it can be learned in conjunction with other network parameters"; "for weight step size
  g = 1/√(N_W Q_P) and for activation step size g = 1/√(N_F Q_P)".
- LSQ+, Bhalgat et al., CVPR-W 2020, https://arxiv.org/abs/2004.09576 - "a general asymmetric quantization scheme with trainable scale and
  offset parameters that can learn to accommodate the negative activations."
- Jacob et al., https://arxiv.org/abs/1712.05877 - bias "quantization scale S_bias is the same as that of the accumulators, which is the
  product of the scales of the weights and of the input activations"; "quantizes both weights and activations as 8-bit integers, and just a
  few parameters (bias vectors) as 32-bit integers."
- Bondarenko et al., https://arxiv.org/abs/2109.12948 - "these activations contain structured outliers in the residual connections";
  "Applying per-tensor quantization for the FFN's residual sum is likely to cause a notable error because of the following trade-off between
  the range and the precision."
- ONNX Runtime quantization docs, https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html - QDQ "inserts
  DeQuantizeLinear(QuantizeLinear(tensor)) between the original operators"; "If neither post-training quantization method can meet your
  accuracy goal, you can try using quantization-aware training (QAT)"; "S8S8 with QDQ is the default setting and balances performance and
  accuracy." ORT QNN EP docs, https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html - "com.microsoft:QuantizeLinear
  Provides 16-bit integer quantization support"; mixed-precision QDQ "regions with different activation/weight quantization data types".
- PyTorch quantization docs (2.14), https://docs.pytorch.org/docs/stable/quantization.html - "We are centralizing all quantization related
  development to torchao"; torchao PT2E, https://docs.pytorch.org/ao/stable/pt2e_quantization/index.html - "Currently there are two typical
  lowering paths, 1. torch.compile through inductor lowering 2. ExecuTorch through delegation".
