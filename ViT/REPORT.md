# ViT training report

_Generated 2026-10-10 18:42 from `/home/ubuntu/workspace/vision_models/ViT/runs`; 24 runs included._

Gap = clean train acc minus val acc (points). Val acc shows final epoch, best in parentheses when different.

## Compute summary

| section | runs done | pending | GPU minutes | cost |
|---|---|---|---|---|
| CIFAR-10 | 16 | 8 | 121.9 | $6.68 |
| CIFAR-100 | 8 | 0 | 63.7 | $3.49 |
| Tiny-ImageNet | 0 | 8 | 0.0 | $0.00 |
| ImageNet-64 | 0 | 4 | 0.0 | $0.00 |
| **Total (sum of run-minutes)** | 24 | 20 | **185.6** (3.09 h) | $10.18 at $3.29/h (upper bound) |

**Wall-clock (estimated, counting in-progress runs up to now):** 223 min (3.71 h) from the earliest run start (2026-10-10 15:00) to the latest run end (2026-10-10 18:42); **GPU cost on wall-clock: $12.21** at $3.29/h.
Sum of completed run-minutes is 186 min. Completed runs rarely overlapped (peak 1), so the span exceeds the sum of run-minutes by idle gaps and startup/eval time (the sum will overstate wall-clock once the concurrent queues complete); 3 runs currently in progress (24 min elapsed so far; peak concurrency now 3) are counted in the span but not in the sum; time with at least one run active is 200 min ($10.99). Start = run-directory creation time (falls back to results.json mtime minus total_minutes), end = results.json mtime (in-progress runs end at report time). Per-run minutes and img/s of overlapping runs are inflated by GPU sharing.

## CIFAR-10

| run | epochs | params (M) | clean train acc | val acc | train-val gap | minutes | img/s | what this step taught |
|---|---|---|---|---|---|---|---|---|
| t1_plain | 100 | 5.36 | 100.00 | 82.90 | 17.1 | 4.3 | 21,861 | baseline (crop+flip only) |
| t2_ls_wd | 100 | 5.36 | 99.95 | 83.07 (83.11) | 16.9 | 2.6 | 35,280 | +label smoothing+weight decay: val +0.2 pts, gap shrinks 17→17 |
| t3_randaug | 100 | 5.36 | 93.21 | 85.96 | 7.2 | 3.0 | 29,440 | +RandAugment: val +2.9 pts, gap shrinks 17→7 |
| t4_mix | 100 | 5.36 | 72.89 | 70.92 | 2.0 | 3.0 | 29,650 | +MixUp/CutMix: val -15.0 pts, gap shrinks 7→2 |
| t5_droppath | 100 | 5.36 | 74.53 | 71.97 | 2.6 | 3.6 | 29,233 | +drop-path: val +1.0 pts, gap grows 2→3 |
| t6_full_300ep | 300 | 5.36 | 92.80 | 87.96 | 4.8 | 12.9 | 19,535 | epochs 100->300: val +16.0 pts, gap grows 3→5 |
| r18_full | 100 | 11.17 | 95.86 | 93.16 | 2.7 | 3.0 | 31,982 | vs matching ViT (t5_droppath): val +21.2 pts (better), gap 3 vs 3, 0.8x the time; conv inductive bias helps at this budget |
| t5b_mix_half | 100 | 5.36 | 81.98 | 79.08 | 2.9 | 4.4 | 19,504 | vs t5_droppath: mix-prob 1.0->0.5: val +7.1 pts, gap 3->3, img/s x0.67. t5 underfit (train acc ~ val acc, both low) so MixUp/CutMix was too strong for a 100-epoch budget; applying it to only half the batches makes the task easier |
| t5c_bs256 | 100 | 5.36 | 84.74 | 82.09 | 2.6 | 16.5 | 5,198 | vs t5_droppath: batch 1024->256, lr 0.002->0.001: val +10.1 pts, gap 3->3, img/s x0.18. 4x more optimizer steps for the same 100 epochs (lr scaled 2e-3->1e-3); steps, not regularization, were the limit. img/s is lower because this run was not CUDA-graphed |
| s_small_200ep | 200 | 21.34 | 59.50 | 58.03 (58.13) | 1.5 | 11.0 | 16,073 | ViT-Small vs Tiny (t5_droppath): val -13.9 pts, 21.3M vs 5.4M params, 200 vs 100 epochs |
| s_small_200ep_lr1e-3 | 200 | 21.34 | 82.44 | 78.64 (78.97) | 3.8 | 10.6 | 16,117 | vs s_small_200ep: lr 0.002->0.001: val +20.6 pts, gap 1->4, img/s x1.00. the lr 2e-3 Small run stalled (loss plateau ~2.1); halving the LR trains it, so that failure was optimization, not capacity |
| p2_tiny | 100 | 5.39 | 77.83 | 75.01 | 2.8 | 9.8 | 9,143 | vs t5_droppath: patch 4->2: val +3.0 pts, gap 3->3, img/s x0.31. patch 2 = (32/2)^2 = 256 tokens vs 64; same heavy (underfitting) recipe, so this shows what the extra tokens buy when steps are fixed; img/s was measured with other queues on the GPU, so treat it as indicative |
| p8_tiny | 100 | 5.38 | 58.63 | 57.38 | 1.2 | 4.5 | 20,565 | vs t5_droppath: patch 4->8: val -14.6 pts, gap 3->1, img/s x0.70. patch 8 = (32/8)^2 = 16 tokens vs 64; same heavy (underfitting) recipe; cheaper but loses spatial detail; img/s was measured with other queues on the GPU, so treat it as indicative |
| bf1_slvit | 100 | 2.70 | 78.21 | 75.84 | 2.4 | 10.7 | 8,163 | vs t5c_bs256: model tiny->slvit, batch 256->128, lr 0.001->0.003, warmup 5->10: val -6.2 pts, gap 3->2, img/s x1.57. SL-ViT shape (9 layers, 12 heads, MLP 2, 2.7M params) with the SL-ViT recipe (batch 128, lr 3e-3, warmup 10, reprob 0.25), compared with the best same-budget Tiny run |
| bf1b_tiny_samerecipe | 100 | 5.36 | 48.77 | 48.59 (48.87) | 0.2 | 11.8 | 7,092 | vs bf1_slvit: model slvit->tiny: val -27.2 pts, gap 2->0, img/s x0.87. shape effect: identical SL-ViT recipe, but our 12-layer / 3-head / MLP-4 Tiny; train acc is also low, so it is an optimization stall, not overfitting |
| bf2_slvit_init | 100 | 2.70 | 49.62 | 49.33 (49.36) | 0.3 | 10.3 | 8,299 | vs bf1_slvit: init default->impulse: val -26.5 pts, gap 2->0, img/s x1.02. init effect: identical to bf1 but impulse init; train acc is also low, so training stalled (suspect lr 3e-3 with this init; bf1_slvit_lr* will tell) |
| bf3_slvit_300ep | pending | pending | pending | pending | pending | pending | pending | running (epoch 88/300, log updated 0 min ago); result pending |
| bf4_slvit_300ep_mixhalf | pending | pending | pending | pending | pending | pending | pending | not started |
| bf1_slvit_lr1e-3 | pending | pending | pending | pending | pending | pending | pending | running (epoch 55/100, log updated 0 min ago); result pending |
| bf1_slvit_lr5e-4 | pending | pending | pending | pending | pending | pending | pending | not started |
| rd_pilot_a | pending | pending | pending | pending | pending | pending | pending | running (epoch 39/100, log updated 0 min ago); result pending |
| rd_pilot_b | pending | pending | pending | pending | pending | pending | pending | not started |
| rd_pilot_noinject | pending | pending | pending | pending | pending | pending | pending | not started |
| rd_flat6 | pending | pending | pending | pending | pending | pending | pending | not started |

Subtotal: **121.9 GPU min** (2.03 h), **$6.68**

### Comparison with published results

Reference rows are from `REFERENCES.md` (verified 2026-10-10; epochs/batch differ from ours, so this is a rough yardstick, not a controlled comparison).

| model | params | setting | acc (%) | our best ViT minus ref | our best CNN minus ref | type | source |
|---|---|---|---|---|---|---|---|
| **ours: t6_full_300ep** (tiny, 300 ep, batch 1024) | 5.4M | this repo | 87.96 |  |  | ViT | this repo |
| **ours: r18_full** (resnet18, 100 ep, batch 1024) | 11.2M | this repo | 93.16 |  |  | CNN | this repo |
| ViT-Tiny 192/12/3 p4, plain init (mimetic-init paper baseline) | ~5.4M | 100 ep, batch 512, RandAug+Cutout | 88.43 | -0.5 | +4.7 | ViT | [arXiv 2305.09828 Tab. 7](https://arxiv.org/html/2305.09828) |
| DeiT-T patch 2 (NesT authors) | 5.3M | n/s | 88.39 | -0.4 | +4.8 | ViT | [arXiv 2105.12723 Tab. A3](https://arxiv.org/html/2105.12723) |
| ViT-Lite-7/4, no aug / no tuning | 3.72M | 300 ep | 83.59 | +4.4 | +9.6 | ViT | [arXiv 2104.05704 Tab. 8](https://arxiv.org/html/2104.05704v4) |
| ViT-12/16 | 85.6M | 300 ep, batch 128 | 83.04 | +4.9 | +10.1 | ViT | [arXiv 2104.05704 Tab. 1](https://arxiv.org/html/2104.05704v4) |
| omihub777/ViT-CIFAR (7L, dim 384) | 6.3M | 200 ep, batch 128 | 90.92 | -3.0 | +2.2 | ViT | [GitHub](https://github.com/omihub777/ViT-CIFAR) |
| ViT-Lite-7/4, full DeiT-style recipe | 3.72M | 300 ep, batch 128 | 93.57 | -5.6 | -0.4 | ViT | [arXiv 2104.05704 Tab. 1](https://arxiv.org/html/2104.05704v4) |
| ViT baseline (depth 9, dim 192), SPT/LSA paper | 2.8M | 100 ep, batch 128 | 93.58 | -5.6 | -0.4 | ViT | [arXiv 2112.13492 Tab. 2](https://arxiv.org/html/2112.13492v1) |
| SL-ViT (ViT + SPT + LSA) | 2.9M | 100 ep, batch 128 | 94.53 | -6.6 | -1.4 | ViT (locality) | [arXiv 2112.13492 Tab. 2](https://arxiv.org/html/2112.13492v1) |
| CCT-7/3x1 (conv tokenizer, hybrid) | 3.76M | 300 ep, batch 128 | 96.53 | -8.6 | -3.4 | hybrid | [arXiv 2104.05704 Tab. 1](https://arxiv.org/html/2104.05704v4) |
| ResNet-110 (He et al.) | 1.7M | ~164 ep, batch 128 | 93.57 | -5.6 | -0.4 | CNN | [arXiv 1512.03385 Tab. 6](https://ar5iv.labs.arxiv.org/html/1512.03385) |
| ResNet18 (kuangliu/pytorch-cifar) | 11M | n/s | 93.02 | -5.1 | +0.1 | CNN | [GitHub](https://github.com/kuangliu/pytorch-cifar) |
| ResNet18 (Cutout paper), crop+flip | 11M | 200 ep, batch 128 | 95.28 | -7.3 | -2.1 | CNN | [arXiv 1708.04552](https://ar5iv.labs.arxiv.org/html/1708.04552) |
| WRN-28-10 | 36.5M | 200 ep | 96.00 | -8.0 | -2.8 | CNN | [arXiv 1605.07146](https://ar5iv.labs.arxiv.org/html/1605.07146) |
| airbench94 (Keller Jordan) | small CNN | 3.29 s on 1x A100 | 94.00 | -6.0 | -0.8 | CNN | [arXiv 2404.00498](https://arxiv.org/abs/2404.00498) |

![Ablation ladder: train (dashed) vs validation (solid) accuracy and gap](report_assets/acc_curves.png)

![Follow-up runs: t5 vs t5b/t5c, ViT-Small LR fix, patch 2/8, ResNet-18](report_assets/followups_acc_curves.png)

![Baseline-fix runs: SL-ViT shape/recipe (bf1) vs Tiny shape (bf1b) vs impulse init (bf2), with t5c for reference](report_assets/bf_acc_curves.png)

![loss curves](report_assets/loss_curves.png)

![throughput](report_assets/throughput.png)

![pos embed similarity](report_assets/pos_embed_similarity.png)

![cls attention](report_assets/cls_attention.png)

![randaug check](report_assets/randaug_check.png)

![randaug per op](report_assets/randaug_per_op.png)

## CIFAR-100

| run | epochs | params (M) | clean train acc | val acc | train-val gap | minutes | img/s | what this step taught |
|---|---|---|---|---|---|---|---|---|
| c100_t1_plain | 100 | 5.38 | 99.97 | 57.33 | 42.6 | 5.5 | 16,505 | baseline (crop+flip only) |
| c100_t2_ls_wd | 100 | 5.38 | 99.98 | 56.79 (56.89) | 43.2 | 5.2 | 16,557 | +label smoothing+weight decay: val -0.5 pts, gap grows 43→43 |
| c100_t3_randaug | 100 | 5.38 | 98.04 | 61.84 (61.88) | 36.2 | 5.8 | 14,714 | +RandAugment: val +5.1 pts, gap shrinks 43→36 |
| c100_t4_mix | 100 | 5.38 | 65.81 | 56.91 | 8.9 | 4.7 | 22,091 | +MixUp/CutMix: val -4.9 pts, gap shrinks 36→9 |
| c100_t5_droppath | 100 | 5.38 | 65.76 | 57.31 | 8.5 | 8.2 | 12,038 | +drop-path: val +0.4 pts, gap shrinks 9→8 |
| c100_t6_full_300ep | 300 | 5.38 | 95.38 | 70.49 | 24.9 | 16.7 | 14,250 | epochs 100->300: val +13.2 pts, gap grows 8→25 |
| c100_r18_full | 100 | 11.22 | 84.00 | 73.22 (73.30) | 10.8 | 3.6 | 27,102 | vs matching ViT (c100_t5_droppath): val +15.9 pts (better), gap 11 vs 8, 0.4x the time; conv inductive bias helps at this budget |
| c100_s_small_200ep | 200 | 21.38 | 95.25 | 68.01 (68.32) | 27.2 | 13.9 | 12,947 | ViT-Small vs Tiny (c100_t5_droppath): val +10.7 pts, 21.4M vs 5.4M params, 200 vs 100 epochs |

Subtotal: **63.7 GPU min** (1.06 h), **$3.49**

### Comparison with published results

Rows tagged [verified] are from `REFERENCES.md` (CIFAR-100 section, re-checked against the source pages). The same-shape ViT (68.09) and L-ViT (74.87) rows were quoted by a research agent in `RESEARCH_DIRECTIONS.md` and are not independently re-checked. No verified published CIFAR-100 ResNet-18 number is recorded, so the CNN comparison is our own r18_full only.

| model | params | setting | acc (%) | our best ViT minus ref | our best CNN minus ref | type | source |
|---|---|---|---|---|---|---|---|
| **ours: c100_t6_full_300ep** (tiny, 300 ep, batch 256) | 5.4M | this repo | 70.49 |  |  | ViT | this repo |
| **ours: c100_r18_full** (resnet18, 100 ep, batch 256) | 11.2M | this repo | 73.22 |  |  | CNN | this repo |
| ViT 12L/3h/MLP4 (same shape as ours), Gani et al. recipe [not re-checked] | n/s | n/s | 68.09 | +2.4 | +5.1 | ViT | [arXiv 2210.07240 Tab. 15 (quoted by research agent, RESEARCH_DIRECTIONS.md)](https://arxiv.org/abs/2210.07240) |
| L-ViT (+LSA), SL-ViT paper ablation [not re-checked] | n/s | n/s | 74.87 | -4.4 | -1.7 | ViT (locality) | [arXiv 2112.13492 (quoted by research agent, RESEARCH_DIRECTIONS.md)](https://arxiv.org/abs/2112.13492) |
| ViTRM (Vision Tiny Recursion Model) [verified] | ~3.6M | n/s | 72.10 | -1.6 | +1.1 | ViT (recurrent) | [arXiv 2603.19503 (REFERENCES.md)](https://arxiv.org/abs/2603.19503) |
| ViT baseline (depth 9, dim 192), SPT/LSA paper [verified] | 2.8M | 100 ep, batch 128 | 73.81 | -3.3 | -0.6 | ViT | [arXiv 2112.13492 Tab. 2 (REFERENCES.md)](https://arxiv.org/html/2112.13492v1) |
| SL-ViT (ViT + SPT + LSA) [verified] | 2.9M | 100 ep, batch 128 | 76.92 | -6.4 | -3.7 | ViT (locality) | [arXiv 2112.13492 Tab. 2 (REFERENCES.md)](https://arxiv.org/html/2112.13492v1) |
| ViT-Lite-7/4 [verified] | 3.72M | 300 ep, batch 128 | 73.94 | -3.5 | -0.7 | ViT | [arXiv 2104.05704 Tab. 1 (REFERENCES.md)](https://arxiv.org/html/2104.05704v4) |
| ViT + SSL pretraining on CIFAR (Gani et al.) [verified] | 2.8M | 200 SSL + 100 ep | 79.15 | -8.7 | -5.9 | ViT, 2-stage | [arXiv 2210.07240 Tab. 3 (REFERENCES.md)](https://arxiv.org/html/2210.07240) |
| CCT-7/3x1 (conv tokenizer, hybrid) [verified] | 3.76M | 300 ep, batch 128 | 80.92 | -10.4 | -7.7 | hybrid | [arXiv 2104.05704 Tab. 1 (REFERENCES.md)](https://arxiv.org/html/2104.05704v4) |
| CCT-7/3x1, long schedule [verified] | 3.76M | 5000 ep | 82.87 | -12.4 | -9.7 | hybrid | [arXiv 2104.05704 Tab. 2 (REFERENCES.md)](https://arxiv.org/html/2104.05704v4) |

![Ablation ladder: train (dashed) vs validation (solid) accuracy and gap](report_assets/c100_acc_curves.png)

![c100 loss curves](report_assets/c100_loss_curves.png)

![c100 throughput](report_assets/c100_throughput.png)

![c100 pos embed similarity](report_assets/c100_pos_embed_similarity.png)

![c100 cls attention](report_assets/c100_cls_attention.png)

## Tiny-ImageNet

_No runs yet; all rows pending._

| run | epochs | params (M) | clean train acc | val acc | train-val gap | minutes | img/s | what this step taught |
|---|---|---|---|---|---|---|---|---|
| tin_t1_plain | pending | pending | pending | pending | pending | pending | pending | not started |
| tin_t2_ls_wd | pending | pending | pending | pending | pending | pending | pending | not started |
| tin_t3_randaug | pending | pending | pending | pending | pending | pending | pending | not started |
| tin_t4_mix | pending | pending | pending | pending | pending | pending | pending | not started |
| tin_t5_droppath | pending | pending | pending | pending | pending | pending | pending | not started |
| tin_r18_full | pending | pending | pending | pending | pending | pending | pending | not started |
| tin_s_small_200ep | pending | pending | pending | pending | pending | pending | pending | not started |
| tin_p4_tiny | pending | pending | pending | pending | pending | pending | pending | not started |

Subtotal: **0.0 GPU min** (0.00 h), **$0.00**

## ImageNet-64

_No runs yet; all rows pending._

| run | epochs | params (M) | clean train acc | val acc | train-val gap | minutes | img/s | what this step taught |
|---|---|---|---|---|---|---|---|---|
| in64_t1_plain | pending | pending | pending | pending | pending | pending | pending | not started |
| in64_t5_full | pending | pending | pending | pending | pending | pending | pending | not started |
| in64_r18_full | pending | pending | pending | pending | pending | pending | pending | not started |
| in64_s_small_full | pending | pending | pending | pending | pending | pending | pending | not started |

Subtotal: **0.0 GPU min** (0.00 h), **$0.00**

## What I learned

All numbers below are single-seed runs from `runs/*/results.json`; differences of about 1 point are within what we could not separate from noise. Published numbers are from `REFERENCES.md` (verified) and `RESEARCH_DIRECTIONS.md`. Claims about runs that have not finished are marked **pending**.

### Why plain ViT overfits

- CIFAR-10 `t1_plain` (ViT-Tiny, crop+flip only) reaches 100.0% clean train accuracy but only 82.9% val, a 17.1-point gap.
- CIFAR-100 is much worse: `c100_t1_plain` hits 99.97% train and 57.33% val, a 42.6-point gap. Same 50k images but 100 classes means about 500 per class, so memorizing is easier than learning features.
- A plain ViT has no convolutional locality or translation prior, so it has to learn these from 50k small images. The ResNet-18 with the heavy recipe sits at a 2.7-point gap on CIFAR-10 and 10.8 on CIFAR-100.
- The learned position embeddings and attention maps (CIFAR-100 `c100_t6_full_300ep`) show the model does recover locality: nearby patches have similar position embeddings and CLS attention lands on the object. It learns this from data instead of having it built in.
- Even after 300 epochs of the full recipe, CIFAR-100 still has a 24.9-point gap (95.4 train / 70.5 val), so more regularization or more data is still needed there.

### Which regularizers mattered

- RandAugment was the clear win: +2.9 points on CIFAR-10 (83.1 to 86.0, gap 17 to 7) and +5.1 on CIFAR-100 (56.8 to 61.8).
- Label smoothing + weight decay were negligible: +0.2 on CIFAR-10 and -0.5 on CIFAR-100, with no gap change. We could not tell them apart from noise.
- MixUp/CutMix hurt at a 100-epoch budget: -15.0 points on CIFAR-10 (86.0 to 70.9) and -4.9 on CIFAR-100, while the gap collapsed (train acc 72.9%): the model underfits instead of overfitting.
- The underfit is a budget issue, not a verdict against mixing. Applying it to half the batches (`t5b_mix_half`) recovers +7.1 points (71.97 to 79.08), and 300 epochs (`t6_full_300ep`) reaches 87.96 on CIFAR-10 and 70.49 on CIFAR-100.
- Drop-path 0.1 gave +1.0 (CIFAR-10) and +0.4 (CIFAR-100), too small to claim an effect.

### Batch size / steps

- `t5c_bs256` is `t5_droppath` with batch 1024 to 256 and lr 2e-3 to 1e-3, i.e. 4x more optimizer steps for the same 100 epochs. Val goes 71.97 to 82.09 (+10.1) and train acc 74.5 to 84.7: the model was step-starved, not over-regularized.
- That step count is expensive here: 16.5 min vs 3.6 min, mostly because this run was not CUDA-graphed (5.2k img/s).
- The ViT-Small lr 2e-3 run (`s_small_200ep`) stalled at loss ~2.1 and 58.0% val. Halving lr to 1e-3 (`s_small_200ep_lr1e-3`) gives 78.64 (+20.6), so that failure was a learning-rate problem, not a capacity limit.
- Even fixed, Small (21M params, 200 ep) is below Tiny at batch 256 (82.09), and on CIFAR-100 Small (68.01) is below Tiny at 300 epochs (70.49). More parameters did not help at this data size.
- The baseline-fix runs support this: with batch 128 and lr 3e-3, `bf1b_tiny_samerecipe` and `bf2_slvit_init` both sat at ~49% train accuracy (stalled, not overfit). The LR check (`bf1_slvit_lr1e-3`, `lr5e-4`) is **pending**, so the lr 3e-3 diagnosis is not confirmed.

### ViT vs ResNet-18

- ResNet-18 with the same heavy recipe gets 93.16% on CIFAR-10 in 3.0 min, versus 71.97% for the same-budget ViT-Tiny (+21.2) and 87.96% for the best ViT-Tiny run (300 epochs, 12.9 min). On CIFAR-100: 73.22% vs 57.31% (same budget) and 70.49% (300 epochs).
- The gap narrows on the harder dataset: the best ViT is 5.2 points behind on CIFAR-10 and 2.7 on CIFAR-100.
- ResNet-18 also generalizes better per unit of fitting: gap 2.7 on CIFAR-10 at 95.9% train acc, whereas the ViT only reaches comparable train accuracy after 300 epochs.
- Our ResNet-18 is a plain 100-epoch run at batch 1024, so it is not tuned; see the published comparison below.

### Patch size / sequence length tradeoff

- All three patch runs use the heavy recipe at 100 epochs and batch 1024, which underfits (train acc 77.8 / 74.5 / 58.6 for p2 / p4 / p8, gaps 1 to 3). The ranking therefore mostly shows "more tokens means more compute per step", not generalization.
- Val accuracy: p2 (256 tokens) 75.01, p4 (64 tokens, `t5_droppath`) 71.97, p8 (16 tokens) 57.38. Going p4 to p2 is +3.0 points; p4 to p8 costs -14.6.
- Throughput: p2 9.1k img/s vs 29.2k for p4 (0.31x). p8 measured 20.6k img/s, which is slower than p4 despite 4x fewer tokens. That is a measurement artifact (these runs shared the GPU with other jobs, and small models are launch-bound), so treat the img/s column as indicative only.
- A fair patch comparison needs the non-underfitting recipe (e.g. `t5b` or `t5c` settings); we have not run it.

### How we compare to published results

- Our best CIFAR-10 ViT-Tiny (87.96%, 300 epochs) is within 0.5 points of the closest matched published baseline (88.43%, same 192/12/3 p4 shape, 100 epochs, Trockman & Kolter). It is 5.6 points below tuned pure ViTs (ViT-Lite-7/4 93.57, SPT-paper ViT 93.58, SL-ViT 94.53), which are shallower, use batch 128, and use tuned lr and wd.
- Our recipe port of those results has not matched them yet: `bf1_slvit` (their shape and recipe, 100 ep) got 75.84%. Part of the gap is step count: their repeated-aug sampler makes "100 epochs" about 117k steps versus our 39k. The 300-epoch `bf3_slvit_300ep` is **pending**.
- Our ResNet-18 (93.16%) matches kuangliu's 93.02% but is 2.1 points under the Cutout-paper 95.28% (200 epochs, batch 128).
- CIFAR-100 (verified in `REFERENCES.md`): our 70.49% is 1.6 below ViTRM (72.1), 3.3 below the SPT-paper plain ViT (73.81), 3.4 below ViT-Lite-7/4 (73.94), 6.4 below SL-ViT (76.92), and well below SSL-pretrained ViT (79.15) and CCT-7/3x1 (80.92; 82.87 at 5000 epochs). Our 73.22% ResNet-18 is level with the plain small ViTs. The same-shape 12L/3h ViT at 68.09 (Gani Tab. 15) is quoted by a research agent and not re-checked.
- Results above ~96% under 10M params in the references use convolutional tokenizers (CCT 96.53), self-supervised pretraining, or very long schedules (CCT 98.00 at 5000 epochs).

### Throughput engineering

- CUDA graphs (`--cudagraphs`, `torch.compile` reduce-overhead) give 2.7x, measured. A 10-epoch real CIFAR-10 run (batch 256, ViT-Tiny, RandAugment) took 9.45 s/epoch without vs 3.48 s/epoch with graphs, with the same accuracy within noise (val 53.2% vs 53.0%). `bench_step.py` (the reproduction script) gives 49.3 vs 16.6-17.5 ms/step at batch 256 and 50.6 vs 32.1 ms/step at batch 1024 (1.6x), so the win is largest at small batch, where small ViTs are kernel-launch bound.
- The un-graphed `t5c_bs256` ran at 9.6 s/epoch (5.2k img/s) for the same reason; graphed CIFAR-100 batch-256 runs take about 3 to 4 s/epoch.
- Parallel queues were measured: with 3 concurrent jobs, per-job epoch time rose 9 to 55% (bf3 6.1 to 6.65 s; recurrent pilot ~5.6 to 8.8 s) for roughly 2x the total throughput, at 9 GB of 80 GB memory and 226 W of 350 W. Small launch-bound models leave the GPU underused, so parallel queues are worth it. The report gives both the sum of run-minutes and an estimated wall-clock span (see Compute summary); per-run img/s of overlapping runs is lower than solo.

### Scaling to bigger datasets

- **Pending.** Tiny-ImageNet and ImageNet-64 runs have not been executed, so we make no claim about how the ViT-vs-ResNet picture changes with scale. On CIFAR-10 to CIFAR-100 the gap shrank from 5.2 to 2.7 points, but that is one step on one axis.
