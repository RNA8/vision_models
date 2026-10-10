# ViT GPU schedule (1x H100 PCIe 80GB, $3.29/hr)

Drafted 2026-10-10. Chains run strictly back to back, in this order: CIFAR-10 ablations, CIFAR-10 scale-up,
CIFAR-100, Tiny-ImageNet, ImageNet-64. Clock starts at the CIFAR-10 chain start in `runs/chain_cifar10.log`
(15:00:11 UTC, 2026-10-10). ETAs assume zero idle gaps between chains.

**Total: ~13.7 h wall, ~$45 (range 10.0-17.5 h, $33-$57). ETA 2026-10-11 ~04:43 UTC.**
Not included: the ImageNet-64 cache build (download ~150 GB + decode, needs HF token), failures/reruns.

## Method
- Measured: ViT-Tiny p4 CIFAR-10 = 2.2 s/epoch steady (~22.4k img/s; `smoke_real/results.json` 2.30 s, chain log 2.1-2.3 s).
  Eval every 5 epochs plus startup/compile is small: from `smoke_real` (2 epochs, eval every epoch) and the live
  t1_plain timing, overhead is ~5% of epoch time plus ~0.5 min fixed per run. Run minutes = `epochs * s/ep * 1.05 / 60 + 0.5`
  (gives 4.3 min for t1_plain, matching the live log).
- Model for everything else: train FLOPs/img = `6 * P_nonembed * T + 12 * L * T^2 * d`, `P_nonembed = 12 L d^2`, `T = (img/p)^2 + 1`,
  L = 12, d = 192 (tiny) / 384 (small). Tiny p4 CIFAR = 2.19 GFLOP/img; calibrated to 2.2 s per 50k images
  (effective ~49 TFLOP/s, so well under peak and small-dim bound).
- Utilization factor g (point estimate): small 1.3x better than tiny; tiny p2 / 257 tokens 1.15x; tiny p8 on CIFAR (17 tokens) 0.6x
  (launch/overhead bound); all others 1.0. Range column uses best case (small/long-seq up to ~1.3x faster)
  to worst case (pure FLOP scaling, no utilization gain).
- ResNet-18: 0.56 GFLOP fwd (x3 train) at 32x32, x4 at 64x64 with the CIFAR stem, times a 1.4 memory-bound/BN penalty
  (not measured: no r18 result exists yet; range 0.7x-1.6x).
- Runs with RandAugment (+mixup/cutmix) get +10% s/epoch as an assumed GPU-aug cost (only t1_plain, which has no aug, is measured).
- Imagenet/Tiny-ImageNet per-image cost matches CIFAR p4 (65 tokens); cost scales with image count (100k = 2x, 1.281M = 25.6x).

## Chain 1: CIFAR-10 ablations (50k imgs) - 0.69 h, $2.3
| run | model | patch | tokens | epochs | s/epoch | min | source | ETA (UTC) |
|---|---|---|---|---|---|---|---|---|
| t1_plain | tiny | 4 | 65 | 100 | 2.20 | 4 | measured (live) | 15:04 |
| t2_ls_wd | tiny | 4 | 65 | 100 | 2.20 | 4 | est | 15:08 |
| t3_randaug | tiny | 4 | 65 | 100 | 2.42 | 5 | est | 15:13 |
| t4_mix | tiny | 4 | 65 | 100 | 2.42 | 5 | est | 15:18 |
| t5_droppath | tiny | 4 | 65 | 100 | 2.42 | 5 | est | 15:23 |
| t6_full_300ep | tiny | 4 | 65 | 300 | 2.42 | 13 | est | 15:36 |
| r18_full | resnet18 | - | - | 100 | 2.60 | 5 | est (range 4-8) | 15:41 |

## Chain 2: CIFAR-10 scale-up - 0.76 h, $2.5 (cum 1.4 h)
| run | model | patch | tokens | epochs | s/epoch | min | source | ETA |
|---|---|---|---|---|---|---|---|---|
| s_small_200ep | small | 4 | 65 | 200 | 7.25 | 26 | est (20-30) | 16:07 |
| p8_tiny | tiny | 8 | 17 | 100 | 1.01 | 2 | est | 16:09 |
| p2_tiny | tiny | 2 | 257 | 100 | 9.63 | 17 | est (13-20) | 16:26 |

## Chain 3: CIFAR-100 (50k imgs, same cost as CIFAR-10) - 1.12 h, $3.7 (cum 2.6 h)
| run | model | patch | tokens | epochs | s/epoch | min | source | ETA |
|---|---|---|---|---|---|---|---|---|
| c100_t1_plain | tiny | 4 | 65 | 100 | 2.20 | 4 | est (= C10 measured) | 16:31 |
| c100_t2_ls_wd | tiny | 4 | 65 | 100 | 2.20 | 4 | est | 16:35 |
| c100_t3_randaug | tiny | 4 | 65 | 100 | 2.42 | 5 | est | 16:40 |
| c100_t4_mix | tiny | 4 | 65 | 100 | 2.42 | 5 | est | 16:45 |
| c100_t5_droppath | tiny | 4 | 65 | 100 | 2.42 | 5 | est | 16:49 |
| c100_t6_full_300ep | tiny | 4 | 65 | 300 | 2.42 | 13 | est | 17:02 |
| c100_r18_full | resnet18 | - | - | 100 | 2.60 | 5 | est | 17:08 |
| c100_s_small_200ep | small | 4 | 65 | 200 | 7.25 | 26 | est | 17:33 |

## Chain 4: Tiny-ImageNet (100k imgs, 64x64) - 3.75 h (2.7-4.7), $12.3 (cum 6.3 h)
| run | model | patch | tokens | epochs | s/epoch | min | source | ETA |
|---|---|---|---|---|---|---|---|---|
| tin_t1_plain | tiny | 8 | 65 | 100 | 7.33 | 13 | est | 17:47 |
| tin_t2_ls_wd | tiny | 8 | 65 | 100 | 7.33 | 13 | est | 18:00 |
| tin_t3_randaug | tiny | 8 | 65 | 100 | 8.07 | 15 | est | 18:15 |
| tin_t4_mix | tiny | 8 | 65 | 100 | 8.07 | 15 | est | 18:29 |
| tin_t5_droppath | tiny | 8 | 65 | 100 | 8.07 | 15 | est | 18:44 |
| tin_r18_full | resnet18 (CIFAR stem) | - | - | 100 | 20.8 | 37 | est (26-59) | 19:21 |
| tin_s_small_200ep | small | 8 | 65 | 200 | 22.2 | 78 | est (60-90) | 20:39 |
| tin_p4_tiny | tiny | 4 | 257 | 100 | 22.2 | 39 | est (30-45) | 21:18 |

## Chain 5: ImageNet-64 (1.281M imgs) - 7.41 h (5.4-9.8), $24.4 (cum 13.7 h)
| run | model | patch | tokens | epochs | s/epoch | min | source | ETA |
|---|---|---|---|---|---|---|---|---|
| in64_t1_plain | tiny | 8 | 65 | 30 | 94 | 50 | est | 22:08 |
| in64_t5_full | tiny | 8 | 65 | 30 | 103 | 55 | est | 23:03 |
| in64_r18_full | resnet18 (CIFAR stem) | - | - | 30 | 267 | 141 | est (98-225) | 01:23 (10-11) |
| in64_s_small_full | small | 8 | 65 | 40 | 284 | 199 | est (153-229) | 04:43 (10-11) |

## Summary
| chain | hours | cost | cumulative h |
|---|---|---|---|
| 1 CIFAR-10 ablations | 0.69 | $2.3 | 0.7 |
| 2 CIFAR-10 scale-up | 0.76 | $2.5 | 1.4 |
| 3 CIFAR-100 | 1.12 | $3.7 | 2.6 |
| 4 Tiny-ImageNet | 3.75 | $12.3 | 6.3 |
| 5 ImageNet-64 | 7.41 | $24.4 | 13.7 |
| **total** | **13.7** (10.0-17.5) | **$45** ($33-57) | |

## Top 3 cost drivers and cheaper alternatives
Chains 4-5 are 81% of the bill; the three biggest runs total 418 min (~7 h, $23, half the total).

1. **in64_s_small_full**: 199 min, $10.9. Alternatives: 30 epochs instead of 40 saves ~50 min ($2.7); 20 epochs saves ~100 min ($5.5).
2. **in64_r18_full**: 141 min, $7.7. Alternative: `--r18-stem stride2` (4x fewer activations at 64x64, roughly the 32x32 cost) ~36 min, saves ~105 min ($5.8);
   or 20 epochs with the CIFAR stem saves ~47 min ($2.6). The stem changes the architecture, so note it in the control's label.
3. **tin_s_small_200ep**: 78 min, $4.3. Alternative: 100 epochs saves ~39 min ($2.1). (Runner-up: tin_r18_full, 37 min; stride2 saves ~27 min, $1.5.)

Largest single saving: stride2 on both 64x64 ResNets (~132 min, $7.2) plus Small ImageNet-64 at 30 epochs (~50 min, $2.7) cuts the total by ~3 h to ~10.7 h (~$35).

## Caveats
- Only t1_plain is measured; re-fit after chain 1 finishes (t3-t5 give the real RandAugment/mixup overhead, r18_full the real ResNet number)
  and after s_small_200ep (the real small-model utilization factor). Estimates for chains 4-5 inherit those uncertainties.
- ImageNet-64 uint8 data is ~15.7 GB; assumes it fits in GPU memory / loads without a data-loader bottleneck.
