# Edge efficiency and recurrent-depth ViTs: literature evidence (fetched 2026-10-11)

Rule: every number below was read from a page fetched in this session (URL given). Quotes are verbatim fragments (WebFetch limits quotes to <125 chars; table rows are the tool's reconstruction of the paper table). Anything not fetched is marked unverified. WebFetch summarises pages with a small model, so treat table rows as "as extracted"; re-check before putting in a paper.

Our baseline: recurrent-depth ViT (tied block, additive input re-injection, deep supervision) CIFAR-10 85.1 vs flat 6L 83.5; KL exit ~2.4x less compute, oracle exit 87.8; CIFAR-100 65.3 vs 62.9. RepViT-M0.9 88.3 @224 CIFAR-100 fine-tuned.

## 1. Looped / recurrent / weight-shared transformers

| Technique | Source | Quoted number | Setting | Relevance |
|---|---|---|---|---|
| bViT: one block reused every step (ViT) | https://arxiv.org/abs/2605.10661 (2026) | Table: ViT-S 22.0M 0.782 vs bViT-S 2.5M **0.681**; ViT-B 86.6M 0.789 vs bViT-B 8.6M 0.779; ViT-L 304.3M 0.808 vs bViT-L 14.6M 0.805. "gap ... shrinks as model width increases" | ImageNet-1K, same recipe/budget, 12 steps (S,B), 24 (L) | Closest prior art to us. Shows tied ViT loses 10 pts at width 384, so weight tying at small width is costly; our +1.6 over flat 6L is at equal-params (not equal-compute) which is a different claim. No input injection (x^t=F(x^{t-1})), RMSNorm pre-norm, no stochastic depth. Per-step embedding: 0.780 vs 0.779 (negligible). No dedicated early-exit experiment: only a stop-at-step-k curve (accuracy low until ~step 6) |
| bViT cheap adaptation | same | LoRA r8 on QV: 25K params, avg acc 0.867 vs ViT-B LoRA 0.871 (296K); zero-init per-step embedding tuning ~9K params 0.844 avg | transfer tasks | Per-loop-step tiny adapters are a viable cheap way to specialise a looped model to a target task |
| Mixture-of-Recursions (MoR) | https://arxiv.org/abs/2507.10524 (2025) | "MoR forms a new Pareto frontier"; Table 3 @16.5e18 FLOPs: Vanilla 315M 42.3 few-shot; Recursive Middle-Cycle 167M 42.6; MoR expert-choice 167M **43.1**, NLL 2.7511 vs 2.7824; "nearly 50% fewer parameters"; "up to a 2.06x speedup" | LM 135M-1.7B | Per-token routing of depth beats fixed recursion at equal FLOPs. Expert-choice 42.6 vs token-choice 40.0 few-shot (token-choice worse). Middle-Cycle sharing best. Direct analogue: per-patch-token depth routing in a looped ViT |
| Recurrent-depth LM (Geiping) | https://arxiv.org/abs/2502.05171 (2025) | "3.5 billion parameters and 800 billion tokens"; ARC-C 27.99 (r=4), 35.15 (r=8), 37.71 (r=16), 38.23 (r=32); exit when "KL divergence between successive steps falls below 5e-4"; MT-Bench 5.63 vs 5.56 with early exit | LM | Design recipe: adapter combining state + input embedding, "concatenation works best at scale" (ours: addition); sandwich RMSNorm "needed to train the recurrence at scale"; random init latent; random recurrence depth in training, backprop last 8 iterations. Our KL exit matches their criterion |
| Relaxed Recursive Transformers | https://arxiv.org/abs/2410.20672 (2024) | recursive Gemma 1B beats reduced-size baselines "by up to 13.5 percentage points (51.7% vs. 38.2%)"; relaxed rank-512 LoRA "58.4% vs. 58.6%" of original; theoretical throughput up to "2.66x"; first-loop output only 4.6 pts below final after training | LM uptraining | Per-loop LoRA recovers most of the gap (accuracy rises with rank); SVD init beats zero init by up to 6.5 pts; "Average" init best for relaxed. Strong candidate for us: per-iteration LoRA on tied block |
| MiniViT (weight multiplexing + transformations) | https://arxiv.org/abs/2204.07154 | DeiT-Ti 5M/72.2 -> 3M/72.8; DeiT-S 22M/79.9 -> 11M/80.7; DeiT-B 86M/81.8 -> 44M/83.2; Swin-T WS only 79.0 vs full MiniViT 81.4 (orig 81.2); "compress DeiT-B by 9.7 times from 86M to 9M" | ImageNet, pairs of layers shared | Plain sharing hurts (DeiT-S 78.3); fixing it needs per-layer transformations + weight distillation from the unshared model |
| Universal Transformer | https://arxiv.org/abs/1807.03819 | "dynamic per-position halting mechanism"; "0.9 BLEU improvement over Transformers" (WMT14 En-De) | NMT | Origin of per-position halting; old baseline |
| Mixture-of-Depths | https://arxiv.org/abs/2404.02258 | "can be upwards of 50% faster to step during post-training sampling" | LM, top-k token routing per block | Per-token compute budgets, static graph (hardware friendly) |
| Loop-ViT (visual reasoning) | https://arxiv.org/abs/2602.02156 (2026) | "18M-parameter model reaches 65.8%" on ARC-AGI-1 vs "73M-parameter ensembles"; parameter-free Dynamic Exit by predictive entropy | ARC-AGI, not natural-image classification | Another entropy-based exit; hybrid conv+attention loop block |
| PonderNet | https://arxiv.org/abs/2107.05407 | no numbers in abstract ("dramatically improves performance" on synthetic problem; matches SOTA on QA "using less compute") | parity/bAbI | Learned halting with geometric prior regulariser; unbiased gradients vs ACT |

Gap found: no 2024-2026 paper fetched that combines tied looped ViT + learned halting on natural-image classification (a search for it returned only bViT, Loop-ViT, DVLT). Novelty room exists, but bViT (May 2026) must be cited and differentiated.

## 2. Learned / better exit signals

| Technique | Source | Quoted number | Setting | Relevance |
|---|---|---|---|---|
| LGViT (local+global exit heads, self-distillation) | https://arxiv.org/abs/2308.00255 | "approximately 1.8" x speed-up (text truncated; "competitive performance") | 3 ViT backbones, 3 datasets | Exit-head design matters: shallow heads are weak, so use richer heads (conv/attn pooling), not linear |
| PCEE (calibrated exits) | https://arxiv.org/abs/2412.19325 | no numbers in abstract; uses "the average accuracy of samples with similar confidence levels from a held-out validation set" | MSDNet, ViT on CIFAR/ImageNet | Calibrating confidence to validation accuracy is cheap and gives controllable accuracy; baseline to beat |
| UAT (bandit-adapted thresholds) | https://arxiv.org/abs/2509.23666 | "speedup (1.70-2.10x) with a minimal performance drop (<2%)" | vision-language, text, classification | Online-adaptive threshold; no image-classification numbers in abstract |
| DART | https://arxiv.org/html/2603.12269 | (from search summary, not fetched in full) authors say extending to LeViT caused "accuracy loss of up to 17%" | early-exit DNNs | Warns that exits on transformers are fragile; unverified detail |
| A-ViT per-token halting | https://arxiv.org/abs/2112.07658 | DeiT-T 71.3 @1.2 GFLOPs -> A-ViT-T 71.0 @0.8, 3.4K vs 2.1K img/s; DeiT-S 78.9 -> 78.6 @3.6 GFLOPs; "+distillation" 72.4 (T), 80.7 (S); halting uses one existing embedding channel, "two shared scalars" | ImageNet, RTX GPU | Learned per-token halting with ~0 extra params; works on DeiT-T scale. The halting score could be used on loop iterations |
| Recurrent-LM KL exit | (Geiping, above) | KL < 5e-4; MT-Bench 5.63 vs 5.56 | LM | Our current signal; evidence it is workable but not optimal |

Note: no fetched source shows a learned halting head beating confidence/KL on ViT classification with numbers. Our oracle gap (87.8 vs 85.1) is the opportunity, and we would be producing that evidence ourselves.

## 3. Token reduction (small ViTs)

| Technique | Source | Quoted number | Setting | Relevance |
|---|---|---|---|---|
| ToMe | https://arxiv.org/html/2210.09461 | Trained DeiT-S r=16: 79.96 -> 79.13 (-0.83), 4.61 -> 2.30 GFLOPs, 2.00x; trained DeiT-Ti r=16: 71.84 -> 70.74 (-1.09), 1.26 -> 0.63 GFLOPs; off-the-shelf AugReg ViT-Ti r=13: 75.50 -> 73.04 (-2.46) at 1.56x | ImageNet-1k | Merging works training-free; small models lose more (Ti -2.5 off-the-shelf). For CIFAR (65 tokens at patch 4) token counts are already low, so gains smaller |
| DynamicViT | https://arxiv.org/abs/2106.02034 | "pruning 66% of the input tokens ... reduces 31%~37% FLOPs"; "throughput by over 40%"; accuracy drop "within 0.5%" | DeiT/LV-ViT | Needs learned predictors + distillation; ragged shapes hurt edge export |
| A-ViT | (see 2) | 0.8 vs 1.2 GFLOPs on DeiT-T | ImageNet | Token-level adaptivity; data-dependent shapes are awkward in ONNX |

Edge caveat (our inference, not sourced): dynamic token counts change tensor shapes per sample; for CPU ONNX proxy, masking does not save compute, so use merge to a fixed r (ToMe) or batch-1 dynamic shapes.

## 4. Reparameterization and quantization

| Technique | Source | Quoted number | Setting | Relevance |
|---|---|---|---|---|
| RepViT | https://arxiv.org/abs/2307.09283 | "over 80% top-1 accuracy with 1.0 ms latency on an iPhone 12" | ImageNet, iPhone 12 | Our baseline family |
| FastViT / RepMixer | https://arxiv.org/abs/2303.14189 | "4.2% better Top-1 accuracy on ImageNet than MobileOne" at similar latency; RepMixer "removing skip-connections" at inference | mobile | Reparam to remove skip/branches = lower memory access |
| GhostNetV3 | https://arxiv.org/html/2404.11202 | 1.3x: 8.9M, 269 MFLOPs, 79.1 vs GhostNetV2 76.9 at identical latency; 1.0x 77.1 vs 75.3; BEiTV2-B teacher; N=3 rep branches | ImageNet, mobile CPU | Gain comes purely from training (rep branches + KD + other tricks), no latency cost. Transferable to our edge track |
| PTQ4ViT | https://arxiv.org/abs/2111.12293 | "less than 0.5% drop at 8-bit"; prior PTQ ">1% accuracy drop even in 8-bit" | ImageNet | 8-bit PTQ is basically solved for ViT |
| RepQ-ViT | https://arxiv.org/html/2212.08254 | Table 1 W4/A4: DeiT-T 57.43 (FP 72.21), DeiT-S 69.03 (FP 79.85); PTQ4ViT 36.96/34.08; W6/A6 DeiT-T 70.76, DeiT-S 78.90 | ImageNet PTQ | 4-bit activations still cost 15 pts on DeiT-T; 6-bit ~1.5 pts. Attention softmax and post-LayerNorm are the hard parts |
| FQ-ViT | https://arxiv.org/pdf/2111.13824 | "first to achieve lossless accuracy degradation (~1%) on fully quantized vision transformers"; LayerNorm input inter-channel variation is main cause | full int8 incl. LN/Softmax | Int8 sensitivity of LN/softmax. Looped blocks reuse one set of weights, so one set of scales must fit all iterations: a risk (our inference) |
| Q-ViT | https://deepai.org/publication/q-vit-fully-differentiable-quantization-for-vision-transformer | search snippet only: DeiT-Tiny improves SOTA uniform quantization "by 1.5%" (bit-width unverified) | QAT | Not verified; do not cite numbers |

## 5. 2024-2026 lightweight backbones

Latency columns are NOT comparable across rows from different papers (different devices).

| Model | Source | Params / MACs | Top-1 | Latency / throughput as reported |
|---|---|---|---|---|
| SHViT-S1 / S4 | https://arxiv.org/html/2401.16456v1 (CVPR'24) | 6.3M/241M; 16.5M/986M (256px) | 72.8; 79.4 | S4 iPhone12 1.6 ms; CPU-ONNX 509 img/s; "3.3x, 8.1x and 2.4x faster than MobileViTv2x1.0 on GPU, CPU and iPhone 12 ... 1.3% more accurate" (search summary of abstract) |
| EfficientViM-M1..M4 | https://arxiv.org/html/2411.15241 | M1 6.7M/239M; M4 19.6M/1111M | 72.9; 79.4 (73.5/79.6 at 450 ep) | GPU 20,731 / 8,170 img/s. Abstract: "up to a 0.7% performance improvement over the second-best model SHViT with faster speed" |
| MobileMamba | https://arxiv.org/html/2411.15941v1 (CVPR'25) | T2 8.8M/255M; B1 17.1M/1080M (256px) | 73.6 (76.9 w/ recipe); 79.9 (82.2 w/ recipe) | GPU 21071 / 6986 img/s; abstract-level: "up to 83.6% on Top-1" |
| LowFormer-B0 / B1 | https://arxiv.org/html/2409.03460 | 14.1M/944M; 17.9M/1410M | 78.4; 79.9 | mobile GPU 20/34 ms, ARM CPU 0.835/1.186 s; B0 ~2x GPU throughput vs MobileOne-S2 with +1% top-1; RepViT-M1.5 81.2 at ARM 2.315 s |
| RepNeXt-M1..M5 | https://arxiv.org/html/2406.16004 | M1 4.8M/0.8G; M2 6.5M/1.1G; M4 13.3M/2.3G | 78.8; 80.1; 82.3 | iPhone12 0.86/1.00/1.48 ms vs RepViT-M0.9 5.1M/0.8G 78.7 @0.89 ms: essentially a tie with RepViT at small sizes |
| iFormer | https://arxiv.org/abs/2501.15369 (ICLR'25) | not fetched | 80.4 | "1.10 ms on an iPhone 13"; "surpassing ... MobileNetV4 under similar latency" |
| CPUBone-B0 | https://arxiv.org/html/2603.26425 (2026) | 10.4M/519M | 77.6 | Pi5 24.2 ms, Pixel7 13.8 ms, Intel 6.9 ms vs RepViT-M0.9 (5.1M/816M, 77.4) 40.8/35.1/10.2 ms. Lesson: fewer MACs != lower CPU latency; memory-access/parallelism matter |
| MobileNetV4 | https://arxiv.org/abs/2404.10518 | not fetched | MNv4-Hybrid-Large 87% (with large distillation recipe) | "Pixel 8 EdgeTPU runtime of just 3.8ms"; Mobile MQA "39% speedup" |
| GhostNetV3-1.0x | https://arxiv.org/html/2404.11202 | 6.1M/167M | 77.1 | CPU 7.81 ms, mobile 14.69 ms (same as V2) |
| FastViT-T12 | (RepNeXt table) | 6.8M/1.4G | 80.3 (256px) | iPhone12 1.33 ms |

Takeaways: at our <=6M budget, RepViT-M0.9 (78.7 @0.89 ms) is already on the frontier; RepNeXt-M1 ties it; GhostNetV3-1.0x (6.1M, 167 MFLOPs, 77.1) is much lower-MAC; CPUBone is the one that reports CPU-latency wins (B0 77.6 at 6.9 ms Intel vs RepViT-M0.9 77.4 at 10.2 ms), and is the most relevant for our ONNX-CPU proxy. Not tested by us yet: SHViT-S1, CPUBone-B0, GhostNetV3, EfficientViM-M1 (M1 is 72.9, weaker).

## Ranked experiments (1x H100; estimates are our own, not sourced)

1. **Learned halting head for our looped ViT (publishable core).** Train per-iteration exit head (small MLP on pooled state + KL-to-previous-step + entropy features) with a PonderNet-style geometric prior or a "will-the-next-step-fix-the-error" target; compare against KL, entropy, PCEE-calibrated and oracle (87.8) at matched average compute on CIFAR-10/100 and a 100-class ImageNet subset. Success = closes >50% of the 2.7 pt oracle gap. Cost: ~1-2 GPU-h per CIFAR run, ~10 GPU-h total for sweeps. Novel vs bViT (no exit study) and Loop-ViT (parameter-free entropy only).
2. **Per-iteration LoRA / step adapters + concatenation injection + sandwich norm ablation.** Evidence: Relaxed Recursive (LoRA rank up -> near full model; SVD init +6.5), Geiping (concat injection, sandwich norm), bViT (step embedding no gain at B, 9K-param tuning helps in transfer). Also test whether the +1.6 pt win persists at equal MACs vs flat. ~5 GPU-h (CIFAR, 6 variants x 3 seeds).
3. **Per-token depth routing (MoR-style) in the looped ViT with expert-choice top-k per iteration.** MoR: expert-choice 42.6 vs token-choice 40.0. Fixed-capacity top-k keeps static shapes, so it ONNX-exports with gather, unlike A-ViT. Report MACs and CPU-ONNX latency. ~8-15 GPU-h.
4. **Edge track: RepViT-M0.9 / RepNeXt-M1 / CPUBone-B0 / SHViT-S1 / GhostNetV3 fine-tuned on CIFAR-100 at 128 and 224, measuring CPU-ONNX latency and int8 PTQ accuracy.** Fills the "not tested" backbones with a latency-vs-accuracy Pareto; CPUBone claims 1.5x lower Intel latency than RepViT-M0.9. Add a BEiTv2/DINOv2-teacher KD arm (GhostNetV3 gained +1.8 at equal latency from training alone). ~15 GPU-h, mostly fine-tuning.
5. **Int8 sensitivity study for looped vs flat ViTs (shared scales across iterations) plus ToMe on the loop.** Measure PTQ (RepQ-ViT-style) and fixed-r token merging inside later iterations only. Hypothesis (unverified): shared weights see drifting activation statistics across iterations, hurting per-tensor int8; per-iteration scale/BN fixes it. Publishable as "quantizing recurrent-depth ViTs." ~6 GPU-h.

Must-cite and differentiate: bViT (2605.10661), Loop-ViT (2602.02156), MoR, Relaxed Recursive, MiniViT.
