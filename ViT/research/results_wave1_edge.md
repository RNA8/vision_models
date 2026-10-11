# Wave-1 edge results (A1 function-matching KD, A2 KD loss swap, A3 resolution)

All CIFAR-100, ImageNet-pretrained timm students fine-tuned with `finetune.py`; teacher `runs/ec100_T_vitb_ar2_224` (ViT-B/16, 93.43 @224).
Numbers are copied from each run's `results.json` / `edge_bench.json`. Single seed. "GPU min" is wall-clock `total_minutes` on a GPU shared with 3 other jobs
(so not comparable across rows; alone, a 224 KD epoch is ~21 s). EMA = EMA(0.999) weights. Val = final-epoch top-1.

| run | student | val top-1 | EMA val | params (M) | GMACs | epochs | GPU min | recipe |
|---|---|---|---|---|---|---|---|---|
| ec100_repvit_m0_9_224 (ref) | RepViT-M0.9 | 88.33 | 88.50 | 4.80 | 0.815 | 30 | 30 | CE, no KD |
| ec100_kd_repvit_m0_9_128 (ref) | RepViT-M0.9 @128 | 87.81 | 87.82 | 4.80 | - | 30 | 34 | soft KD a0.5 t2 |
| ec100_kd_tinyvit_5m_224 (ref) | TinyViT-5M | 88.31 | 88.15 | 5.10 | 1.255 | 30 | 63 | soft KD a0.5 t2 |
| ec100_fm_repvit_224_e30 | RepViT-M0.9 | 88.48 | 88.40 | 4.80 | 0.815 | 30 | 24 | FM (KD only, mixup 1.0) |
| ec100_fm_repvit_224_e200 | RepViT-M0.9 | **89.93** | 89.88 | 4.80 | 0.815 | 200 | 263 | FM |
| ec100_fm_tinyvit5m_224_e200 | TinyViT-5M (dist_in22k_ft_in1k) | **90.32** | 90.29 | 5.10 | 1.255 | 200 | 293 | FM |
| ec100_kd60_soft_repvit_224 | RepViT-M0.9 | 89.03 | 89.05 | 4.80 | 0.815 | 60 | 85 | soft KD a0.5 t2 |
| ec100_kd60_dist_repvit_224 | RepViT-M0.9 | 89.34 | 89.46 | 4.80 | 0.815 | 60 | 67 | DIST b=g=2 t1, a0.5 |
| ec100_kd60_lsd_repvit_224 | RepViT-M0.9 | 89.38 | 89.48 | 4.80 | 0.815 | 60 | 26* | logit-std t2, a0.5 |
| ec100_repvit_m0_9_256 | RepViT-M0.9 @256 | 88.63 | 88.61 | 4.80 | 1.065 | 30 | 37 | CE, no KD |
| ec100_repvit_m0_9_288 | RepViT-M0.9 @288 | 88.37 | 88.31 | 4.80 | 1.348 | 30 | 44 | CE, no KD |

*lsd ran when the GPU was less contended; its wall time is not a cost difference.

FixRes-style test-resolution sweep of the 224-trained RepViT (no training, `eval_res.py`, EMA weights): @224 88.50, @256 88.06, @288 85.73, @320 82.03.

## Setup details
- Consistency check: `finetune.py` already feeds the teacher the exact same augmented, mixed batch (crop/flip -> RandAug -> mixup/cutmix at 32 px, then each
  network upsamples to its own resolution). No fix needed. Random erasing is applied only to the student input, and was off in all runs here.
- FM recipe: `--kd soft --kd-alpha 1 --kd-tau 1` (no hard-label term), `--mixup 1.0 --cutmix 0 --mix-prob 1`, RandAug, drop-path 0.1, AdamW lr 5e-4, wd 0.05, EMA 0.999, warmup 5 ep (200 ep) / 3 ep (30 ep).
  Not Beyer's Inception-crop/mixup at 224; our views are 32 px CIFAR crops upsampled.
- TinyViT tag: the earlier run already used `tiny_vit_5m_224.dist_in22k_ft_in1k` (22k-distilled); no change.
- Loss implementations in `finetune.py` (`--kd dist|lsd`), formulas from fetched papers: DIST (arXiv 2205.10536: row-wise and column-wise Pearson distance on softmax(z/t); paper CIFAR-100/ImageNet weights b=g=2; the paper's own text did not state DIST's temperature, I used t=1, unverified);
  logit standardization (arXiv 2403.01427 Alg. 1: per-sample z-score of logits, base temperature t shared, KL*t^2; t=2 as in their ablation). Both used here with a=0.5 (0.5 CE + 0.5 KD), unlike LSD's paper (CE weight 0.1) and DIST's (CE weight 1), so they are not paper-exact.

## Conclusions
1. Schedule length with function matching is the lever: RepViT-M0.9 88.33 (30 ep CE) -> 89.93 (200 ep FM), +1.6; TinyViT-5M 88.31 (30 ep soft KD) -> 90.32 (200 ep FM), +2.0. The 30-ep FM control gives 88.48 (+0.15 vs CE, within seed noise), so the gain comes from the schedule, not the loss/mixup change alone. We did not run 200-ep CE-only, so how much of the gain needs the teacher (vs. just longer training) is not isolated.
2. Gap to the 93.43 teacher is still 3.1 (TinyViT) to 3.5 (RepViT) points; curves were still rising slowly at 200 ep (RepViT 89.9, TinyViT 90.3 vs ~89.0 at 60 ep soft KD).
3. Loss swap at 60 ep (RepViT): soft 89.03 -> DIST 89.34 (+0.31) / LSD 89.38 (+0.35); EMA 89.05 -> 89.46 / 89.48. Differences are about 0.3 pt on single seeds (CIFAR-100 test noise is ~0.2-0.3), so suggestive but not conclusive; DIST and LSD are indistinguishable. 60 ep soft KD (89.03) already beats 30 ep soft-KD-style results (88.3), again pointing at schedule.
4. Resolution: fine-tuning at 256 gives 88.63 (+0.30 vs 224, +30% MACs), 288 gives 88.37 (no gain, +65% MACs). Raising only the test resolution of the 224 model hurts (88.06 @256, 85.73 @288), i.e. no FixRes benefit without retraining here. Not a worthwhile lever at this budget.
5. Practical pick: TinyViT-5M 200-ep FM is best (90.32) but costs 1.26 GMACs; RepViT-M0.9 at 0.815 GMACs gets 89.93. Combining DIST/LSD with the FM long schedule is untested.
