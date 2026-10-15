# Tiny-ImageNet CNN controls (in progress)

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
| `tin_r18w32_sgd_1000ep` | 32 | 2.86M | running (started 2026-10-15 08:56 UTC) | |
| `tin_r18_sgd_1000ep` | 64 | 11.27M | running | |
| `tin_r18w56_sgd_1000ep` | 56 | 8.65M | running | |
| ViT reference: `tin_spt_slvit_p8_1000ep{,_s1}` | | 2.9M | 66.26 (2 seeds) | |
| ViT distilled from a copy of itself: `tin_kd_ba_fm_1000ep{,_s1}` | | 2.9M | 67.99 (2 seeds) | |
