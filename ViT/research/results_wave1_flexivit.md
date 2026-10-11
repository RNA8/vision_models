# Wave 1, B1: FlexiViT patch switch (IN-64 patch-8 -> patch 4)

Code: `finetune.py --patch-size N` (with `--init-ckpt`) PI-resizes `patch_embed.proj.weight` with
`timm.layers.resample_patch_embed` (bilinear, antialias, pseudo-inverse = FlexiViT PI-resize) and bilinearly
resamples `pos_embed` to the new grid (existing `adapt_state_dict` path). LSA temperatures and SeqPool are
grid-independent. The saved model spec records the new patch size, so edge_bench / KD teacher loading work.
Tests: `test_flexivit.py`. Launch: `run_flexivit.sh`. Recipe identical to `ec100_in64pre_tiny_64`
(30 ep, lr 5e-4, bs 256, RandAug, mixup/cutmix, EMA 0.999, IN-64 init `runs/in64pre_tiny_p8`); KD run uses the same
teacher/settings as `ec100_in64pre_tiny_64_kd` (`ec100_T_vitb_ar2_224`, soft, alpha 0.5, tau 2, teacher 224px).
Single seed (0).

| run | input | patch | tokens | params (M) | GMACs | CIFAR-100 top-1 | EMA | train min* |
|---|---|---|---|---|---|---|---|---|
| ec100_in64pre_tiny_64 (baseline) | 64 | 8 | 64 | 5.41 | 0.347 | 81.17 | 81.12 | 5.8 |
| ec100_in64pre_tiny_64_kd (baseline) | 64 | 8 | 64 | 5.41 | 0.347 | 81.47 | 81.25 | 5.9 |
| ec100_in64pre_tiny_p4_32 | 32 | 4 | 64 | 5.38 | 0.346 | 80.60 | 80.49 | 7.2 |
| ec100_in64pre_tiny_p4_64 | 64 | 4 | 256 | 5.42 | 1.367 | **83.78** | 83.74 | 18.1 |
| ec100_in64pre_tiny_p4_64_kd | 64 | 4 | 256 | 5.42 | 1.367 | **84.08** | 83.93 | 29.1 |

*wall-clock on a shared GPU (2-4 concurrent jobs), only indicative. Params/GMACs from `edge_bench.py --no-gpu`
(`runs/edge_bench.jsonl`; batch 1, matmul/SDPA FLOPs/2).

## Sanity checks (`python test_flexivit.py`)
- Token grids: p8@64 -> 8x8 = 64 tokens, p4@64 -> 16x16 = 256, p4@32 -> 8x8 = 64.
- PI identity: for a 4x4 patch xi, w8 . pinv(B) xi == w4 . xi to 3e-6 (weights scale ~2), i.e. the patch embedding is
  exactly preserved on the pseudo-inverse-upsampled input.
- Zero-training transfer of the fine-tuned patch-8 net (81.17 at 64px): the PI-resized patch-4 net on the original 32px
  images gets 64.71 with the trained head (70.4% argmax agreement with the p8 model); on 64px inputs it gets 28.55
  (expected: finer patches at 2x the pixel scale are a different scale; this is why the fine-tune is needed).
  So the resize is approximately function-preserving only at matched content scale, as in FlexiViT; fine-tuning does the rest.

## Conclusions
- Patch 4 at 64px (4x the tokens, 3.9x the MACs) gives +2.6 over patch 8 with plain CE (83.78 vs 81.17) and
  +2.6 with KD (84.08 vs 81.47). The KD gain (+0.3) matches the patch-8 KD gain (+0.3), so the two are additive and small.
- Patch 4 at native 32px costs the same as patch 8 at 64px (64 tokens, 0.346 vs 0.347 GMACs, same params) and is
  0.6 lower (80.60 vs 81.17), with no upsampling needed. At equal compute, p8@64 is slightly better; p4@32 is the
  cheaper pipeline (no resize, 4x less input data) and is within ~0.5 pt.
- Cost/benefit of the +2.6: 3.9x MACs. Compute-for-accuracy is consistent with FlexiViT (finer grid = better at higher compute).
- Caveats: single seed (seed noise on CIFAR-100 is typically ~0.2-0.3, so the p4_32 vs p8_64 gap and KD gaps
  are within noise; the +2.6 is not); IN-64 pretraining stayed at patch 8, so these are transfer-only gains.
  Not tried: randomised-patch-size pretraining, fine-tuning p4 for more epochs, patch 2.
