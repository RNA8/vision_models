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
| `tin_r18w56_sgd_1000ep` | 56 | 8.65M | running | |
| ViT reference: `tin_spt_slvit_p8_1000ep{,_s1}` | | 2.9M | 66.26 (2 seeds) | |
| ViT distilled from a copy of itself: `tin_kd_ba_fm_1000ep{,_s1}` | | 2.9M | 67.99 (2 seeds) | |
| ViT distilled from 2 copies of itself: `tin_kd_ens2_fm_1000ep` | | 2.9M | 69.66 (1 seed) | 657 |

GPU minutes are wall-clock with 4 jobs sharing one H100, so they depend on what else was running; the CNN runs
shared the GPU with each other (~39 s/epoch), the ViT reference runs with lighter neighbours (~25-26 s/epoch).
Treat them as rough cost, not a speed comparison.

## Verdict (2026-10-15 20:00 UTC; width-56 still running)

- **Same size, no teacher: level.** ResNet-18 at 2.86M gets 66.52 (1 seed); our 2.9M pure ViT gets 66.26 (2 seeds:
  66.16, 66.36). The 0.26 gap is inside seed noise. With the same augmentation/regularization stack and a tuned
  optimizer, the recipe-trained ViT matches a convnet of its size on 64 px Tiny-ImageNet, but does not beat it.
- **The ViT's lead comes from distillation, not architecture.** Distilled from its own 2-model ensemble the ViT
  reaches 69.66, +3.1 over the same-size ResNet. That comparison is not like-for-like (the ResNet had no teacher),
  so the fair follow-up is to distill the 2.86M ResNet from the same ViT ensemble (queued as
  `tin_r18w32_kd_ens2_fm_1000ep`, `run_wave19.sh`).
- **A standard ResNet-18 (11.27M) is the best model here: 69.84** (1 seed), above our best distilled ViT (69.66)
  at ~4x the parameters, and above the published ResNet-18 + AdAutoMix (69.19, arXiv 2312.11954) with a plain
  recipe. So "best Tiny-ImageNet result under 10M parameters" claims for the ViT must say "pure transformer" and
  "at 2.9M"; a 11M convnet trained the same way still edges it.
