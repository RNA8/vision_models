# MoE experiments on CIFAR-100 (edge track), 2026-10-11

Code: `ViT/moe.py` (CondConv, per-image top-1 routed MLP experts + shared expert, sparse upcycling), `--moe <spec>` in `finetune.py` / `train_vit.py`,
queues `run_moe*.sh`, table generator `make_results_moe.py`, CPU benches `moe_bench.py` (ONNX export works), `moe_latency.py`.
All numbers below are read from `runs/<name>/results.json` and `moe_stats.json`; single seed 0 unless a `_s1` row says otherwise. Seed noise in this repo is
about 0.6 pt (dense IN-64 fine-tune: 81.17 vs 81.81 for seeds 0/1), so differences under ~0.6 pt are not evidence.

Setup notes. Active params = parameters used per image after kernel aggregation / top-1 routing; MACs from `FlopCounterMode` on the routed batch-1 path
(for CondConv this includes kernel aggregation and the router). GPU was shared with up to 3 other jobs, so "GPU min" is inflated and noisy. Latency is
PyTorch eager CPU, batch 1, 4 threads, interleaved min-of-7-rounds-medians (`runs/moe_latency.json`), because the shared CPU made single measurements vary 2x; ONNX/ORT was not used for the
table (export of the aggregated-kernel conv does work, `runs/moe_bench_cc_k*_*.json`: e.g. K=4@128 ORT fp32 t1/t4 15.5/11.4 ms, but with BN and RepVGG branches unfused it is not comparable to edge_bench's fused dense 8.5/3.7 ms).
Expert-load entropy = mean over routed layers of the normalised entropy of per-expert argmax share on the test set (1 = uniform; CondConv entropy is of the argmax of soft weights, so it is low (0.4-0.5) while the soft mixture is still used).
Router temperature: CondConv softmax tau 30 -> 1 linearly over 10 of 30 epochs (DynamicConv schedule); expert kernels = pretrained kernel + 0.05*std noise (upcycle); router weights N(0, 0.02).
MLP-MoE: top-1 per image from mean-pooled tokens, shared expert + routed experts each half the dense MLP width (active = dense), last 4 of 12 blocks, load-balance aux 0.01 (injected through an autograd identity so the training loop is unchanged), gate forward value 1 with straight-through gradient on p_top.
Upcycle: each dense MLP of the last 4 blocks copied to 4 full-size experts, top-1, no shared expert, router N(0,0.02) (not exactly zero: a zero router makes top-1 tie to expert 0).
Smoke tests: initial loss 4.61 (ln 100 = 4.605) for CondConv (starts equal to the pretrained model) and 4.57 for MLP-MoE; expert loads non-collapsed (min share 0.09-0.12 after 2 epochs).

## Table
| run | val (final) | best EMA val | active params M | total params M | active MACs M | CPU ms bs1 t4 (PyTorch eager) | expert-load entropy (1=balanced) | GPU min (shared GPU) |
|---|---|---|---|---|---|---|---|---|
| ec100_repvit_m0_9_128 | 87.42 | 87.39 | 4.80 | 4.80 | 266 | - | - | 12.5 |
| ec100_moe_repvit_k4_128 | 87.18 | 87.39 | 4.82 | 10.82 | 275 | 18.7 | 0.467 | 13.6 |
| ec100_moe_repvit_k8_128 | 87.31 | 87.42 | 4.84 | 18.85 | 283 | 27.3 | 0.475 | 15.9 |
| ec100_repvit_m0_9_224 | 88.33 | 88.60 | 4.80 | 4.80 | 815 | - | - | 29.6 |
| ec100_moe_repvit_k4_224 | 88.77 | 88.88 | 4.82 | 10.82 | 825 | 35.7 | 0.399 | 32.0 |
| c100_tinyshape_bf3b | 75.16 | - | 5.38 | 5.38 | 346 | 6.3 | - | 57.0 |
| c100_moe_vit_e4 | 74.68 | - | 5.38 | 7.16 | 346 | 6.7 | 0.995 | 75.3 |
| c100_dense_vit_150ep | 71.43 | - | 5.38 | 5.38 | 346 | 6.3 | - | 27.3 |
| c100_moe_vit_e4_150ep | 70.36 | - | 5.38 | 7.16 | 346 | 7.7 | 0.982 | 34.7 |
| c100_moe_vit_e8_150ep | 69.71 | - | 5.39 | 9.53 | 346 | 9.9 | 0.907 | 44.1 |
| ec100_in64pre_tiny_64 | 81.17 | 81.12 | 5.41 | 5.41 | 347 | 8.0 | - | 5.8 |
| ec100_dense_in64pre_tiny_64_s1 | 81.81 | 81.59 | 5.41 | 5.41 | 347 | 8.0 | - | 7.5 |
| ec100_upcyc_in64pre_tiny_e4_64 | 81.56 | 81.29 | 5.41 | 8.96 | 347 | 9.2 | 0.996 | 8.1 |
| ec100_upcyc_in64pre_tiny_e4_64_s1 | 80.62 | 80.59 | 5.41 | 8.96 | 347 | 6.9 | 0.987 | 7.9 |
| ec100_dense_in64pre_tiny_64_60ep | 83.06 | 83.01 | 5.41 | 5.41 | 347 | 8.0 | - | 11.4 |
| ec100_upcyc_in64pre_tiny_e4_64_60ep | 83.12 | 83.08 | 5.41 | 8.96 | 347 | 9.3 | 0.994 | 16.2 |

Latency (PyTorch eager CPU, bs1, 4 threads, ms, interleaved): ViT-5.4M 32px: dense 5.21, MoE E4 5.88, E8 5.85 (+13%); IN-64 ViT p8 64px: dense 5.15, upcycled E4 5.63 (+9%);
RepViT-M0.9 unfused (train-form, BN/RepVGG not folded for either): 128px dense 12.88, CondConv K4 15.40 (+20%), K8 16.00 (+24%); 224px dense 18.75, K4 21.36 (+14%).
The latency column in the table is a single non-interleaved measurement taken under load and is NOT reliable; use the line above.
Memory: total params in the table are what must be resident (CondConv K4 10.8M, K8 18.8M vs 4.8M dense; MLP-MoE E4 7.2M, E8 9.5M; upcycle 9.0M vs 5.4M).

## Conclusions (equal active compute vs dense; single seeds, noise ~0.6 pt)

1. CondConv/DynamicConv experts in RepViT-M0.9 stage 3 (all ConvNorm convs, K experts): no reliable gain. 128px: K4 87.18 (EMA 87.39), K8 87.31 (EMA 87.42) vs dense 87.42 (EMA 87.39).
   224px K4: 88.77 (EMA 88.88) vs dense 88.33 (EMA 88.5, best 88.6): +0.4 pt, inside seed noise (seed-1 controls for dense and K4 at 224 are queued as
   `ec100_repvit_m0_9_224_s1` / `ec100_moe_repvit_k4_224_s1`; if they have finished their rows appear in the table when `make_results_moe.py` is re-run). Cost: 2.3x (K4) / 3.9x (K8) total params,
   +3% (K4) to +6% (K8) MACs, +14-24% unfused CPU latency. At 4.8M active params this is not a good deal on this evidence; the +0.3 to +0.6 transfer gain in the literature (ViMoE, +0.3 on C100) is the same size as our noise.
2. Per-image MLP-MoE in the 5.4M ViT from scratch is worse than dense, not better. 150 ep: dense 71.43, E4 70.36 (-1.07), E8 69.71 (-1.72; more experts = worse, the opposite of the Mobile V-MoE trend).
   300 ep: dense 75.16 final / 75.37 best vs E4 74.68 / 74.91 (-0.5; 300-ep E8 was not run because 150 ep already shows E8 < E4 < dense). Routing was balanced (entropy 0.91-0.99), so this is not collapse: with 50k images and mixup/cutmix the extra
   total params are not used productively, and splitting the MLP into half-width experts loses capacity per image (shared half + routed half). Training is also ~1.3-1.6x slower (dense-masked compute of all experts). Mobile V-MoE's gains were
   obtained against weak ImageNet baselines; with the well-tuned bf3b recipe there is nothing to gain. By the stop rule in moe_vision.md (needs >= +1 pt) MoE should be dropped for the from-scratch track.
3. Sparse upcycling of the IN-64 pretrained ViT (last 4 blocks, E=4, +3.6M total params): 30 ep 81.56 (seed 0) / 80.62 (seed 1), mean 81.09, vs dense 81.17 (seed 0) / 81.81 (seed 1), mean 81.49: no gain. With 60 ep:
   upcycled 83.12 vs dense 83.06 (equal). The extra-compute control matters more than the MoE: dense 30 -> 60 ep gives +1.9 pt, which no MoE variant matched. Router load stays balanced (entropy 0.99) because the experts remain near-identical copies; i.e. they did not specialise within 30-60 fine-tuning epochs.
4. Net: at <=6M active parameters on CIFAR-100, none of the three MoE variants beats dense at equal active compute outside seed noise, and all add total-parameter/memory cost (up to 3.9x) and latency. Not recommended for the edge model; if revisited, the only
   untested lever with precedent is pretraining with MoE on a large corpus (Mobile V-MoE style), not CIFAR-scale training or fine-tuning.
