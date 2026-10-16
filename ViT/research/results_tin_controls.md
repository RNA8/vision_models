# Tiny-ImageNet CNN controls

Question: is our 2.9M pure ViT (SL-ViT + SPT + LSA + SeqPool, patch 8, 64 px) really ahead of a properly tuned CNN of
the same size trained the same way? Queue: `run_wave15_tin.sh` (screen, first run) and `run_wave17_tin.sh` (rest;
reordered on 2026-10-15 so the param-matched width-32 ResNet-18 starts first).

CNN recipe: ResNet-18 with the CIFAR/Tiny-ImageNet stem (3x3 stride-1 conv, no max-pool), SGD + Nesterov 0.9, batch
128, warmup + cosine, no grad clipping, and the ViT's augmentation/regularization stack (RandAugment m9, MixUp 0.8 /
CutMix 1.0 switched off at 58.3% of training, label smoothing 0.1, random erasing 0.25, stochastic depth 0.1).

## lr / wd screen (ResNet-18 width 64, 11.27M, 100 epochs, 10k held-out images from train; val set untouched)

| lr | wd | Held-out top-1 | Minutes |
|---|---|---|---|
| 0.1 | 1e-4 | 66.63 | 60 |
| **0.1** | **5e-4** | **67.35** (selected) | 61 |
| 0.2 | 1e-4 | 67.29 | 61 |
| 0.2 | 5e-4 | 67.03 | 61 |

The four settings span 0.7 points, so the CNN recipe is not sensitive to them. Note that 100 CNN epochs on
held-out data already reach ~67, about where the ViT ends after 1000 epochs on the real val set (66.26); the splits
differ, so this is only a hint. The 1000-epoch runs below settle it.

## 1000-epoch runs (lr 0.1, wd 5e-4; ~2x the ViT's GPU time per epoch)

| Run | Width | Params | Top-1 (val) | GPU minutes |
|---|---|---|---|---|
| `tin_r18w32_sgd_1000ep` | 32 | 2.86M | **66.52** (best 66.67) | 657 |
| `tin_r18_sgd_1000ep` | 64 | 11.27M | **69.84** | 659 |
| `tin_r18w56_sgd_1000ep` | 56 | 8.65M | **69.81** | 764 |
| `tin_r18w32_kd_ens2_fm_1000ep` (width 32, distilled from the 2-ViT ensemble, 68.98) | 32 | 2.86M | **67.18** | 750 |
| ViT reference: `tin_spt_slvit_p8_1000ep{,_s1}` | | 2.9M | 66.26 (2 seeds) | |
| ViT distilled from a copy of itself: `tin_kd_ba_fm_1000ep{,_s1}` | | 2.9M | 67.99 (2 seeds) | |
| ViT distilled from 2 copies of itself: `tin_kd_ens2_fm_1000ep{,_s1}` | | 2.9M | 69.48 ± 0.26 (2 seeds: 69.66 / 69.29) | 657 / 648 |

GPU minutes are wall-clock with 4 jobs sharing one H100, so they depend on what else was running; the CNN runs
shared the GPU with each other (~39 s/epoch), the ViT reference runs with lighter neighbours (~25-26 s/epoch).
Treat them as rough cost, not a speed comparison.

## Verdict (2026-10-15 22:00 UTC)

- **Same size, no teacher: level.** ResNet-18 at 2.86M gets 66.52 (1 seed); our 2.9M pure ViT gets 66.26 (2 seeds:
  66.16, 66.36). The 0.26 gap is inside seed noise. With the same augmentation/regularization stack and a tuned
  optimizer, the recipe-trained ViT matches a convnet of its size on 64 px Tiny-ImageNet, but does not beat it.
- **The ViT's lead comes from distillation, not architecture.** Distilled from its own 2-model ensemble the ViT
  reaches 69.48 (2 seeds), +3.0 over the same-size ResNet. That comparison is not like-for-like (the ResNet had no teacher),
  so the fair follow-up distilled the 2.86M ResNet from the same ViT ensemble (`tin_r18w32_kd_ens2_fm_1000ep`):
  **67.18**, only +0.66 from the teacher against +3.2 for the ViT student. With the same teacher the ViT student is
  2.3 ahead. Caveat: the teacher is a ViT, which may favour a ViT student; CIFAR-100 shows the same pattern
  (ResNet +0.95, ViT +1.86).
- **A standard ResNet-18 (11.27M) is the best model here: 69.84** (1 seed), above our best distilled ViT (69.48, 2 seeds)
  at ~4x the parameters, and above the published ResNet-18 + AdAutoMix (69.19, arXiv 2312.11954) with a plain
  recipe. So "best Tiny-ImageNet result under 10M parameters" claims for the ViT must say "pure transformer" and
  "at 2.9M"; a 11M convnet trained the same way still edges it.
- **Under 10M, a convnet trained our way ties our best ViT.** ResNet-18 at width 56 (8.65M) gets 69.81, level with
  the distilled 2.9M ViT (69.48, 2 seeds; within noise) at 3x the parameters. The ViT is the better
  accuracy-per-parameter model only once it is distilled; per FLOP the comparison is closer, since the CIFAR-stem
  ResNet runs full 64x64 resolution in its first stage.

## CIFAR-100 counterpart (2026-10-15 22:30 UTC, `run_wave18.sh`)

Same CNN recipe (lr 0.1 / wd 5e-4 from the Tiny-ImageNet screen, not re-screened on CIFAR-100), 1000 epochs, no teacher.

| Run | Params | Top-1 | Minutes |
|---|---|---|---|
| `c100_r18w32_sgd_1000ep` (ResNet-18 width 32) | 2.83M | **78.59** (1 seed) | 158 |
| ViT reference: `c100_spt_slvit_1000ep{,_s1,_s2}` | 2.75M | 80.91 ± 0.26 (3 seeds) | |
| `c100_r18w32_kd_ens3_fm_1000ep` (ResNet-18 width 32, distilled from the same ViT ensemble, 83.27) | 2.83M | **79.54** (1 seed) | 275 |
| ViT distilled from its 3-seed ensemble | 2.75M | 82.77 ± 0.16 (3 seeds) | |

On CIFAR-100 at 32 px the recipe-trained ViT is **+2.3 ahead** of a same-size ResNet-18 with no teacher on either
side, unlike Tiny-ImageNet where they tie. The CNN hyperparameters came from the Tiny-ImageNet screen; a CIFAR-100
screen run afterwards (below) picked the same setting, so the control is tuned. The distilled-CNN counterpart (`c100_r18w32_kd_ens3_fm_1000ep`, same ViT
teacher) is queued in `run_wave20.sh`.

**Distilled-student result (2026-10-16 05:05 UTC):** with the same ViT teacher and the same function-matching setup,
the ResNet student reaches 79.54 (+0.95 from distillation), the ViT student 82.77 (+1.86). The ViT stays 3.2 ahead.
Caveat: the teacher is a ViT, and students that share the teacher's architecture usually distill better, so this
measures "best student for this teacher" as much as architecture.

### CIFAR-100 lr / wd screen (`run_wave22.sh`, 100 ep, 5k held-out train images, ResNet-18 width 32)

| lr | wd 5e-4 | wd 1e-3 |
|---|---|---|
| 0.05 | 72.14 | 72.44 |
| 0.1 | **73.42** | 72.50 |
| 0.2 | 73.20 | 70.96 |

Best is lr 0.1 / wd 5e-4, the setting already used for `c100_r18w32_sgd_1000ep` (78.59) and its distilled twin
(79.54), so no rerun was needed and the CIFAR-100 CNN control stands as a tuned baseline.
