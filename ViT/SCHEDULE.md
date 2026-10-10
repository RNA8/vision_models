# ViT GPU schedule (1x H100 PCIe 80GB, $3.29/hr)

Refreshed 2026-10-10. CIFAR-10 ablations, CIFAR-10 scale-up and CIFAR-100 are done (results in `runs/*/results.json`).
Remaining chains, in order: **Tiny-ImageNet (`run_tiny_imagenet.sh`), then ImageNet-64 (`run_imagenet64.sh`)**.
Both use batch 256, sqrt-scaled LR (Tiny 1e-3, Small 5e-4), `--cudagraphs`, ResNet-18 with `--r18-stem stride2`,
and checkpoint every 10 epochs (crash-resumable). `--lsa` / `--pool seqpool` / SL-ViT are NOT included yet
(baseline-fix experiments still running; ladders stay comparable with CIFAR-10/100).

**Totals (standalone GPU): ~5.1 h, ~$16.6. With ~3 queues sharing the GPU: ~2.3 h of packed wall-clock (~$7.6) at best;
running the two chains concurrently gives ~4.5 h (~$14.8) because ImageNet-64 is the critical path (see below).**
Not included: the ImageNet-64 cache build (download ~150 GB + decode, needs HF token), failures/reruns.

## Method (calibrated from measurements)
Measured `epoch_sec_steady` for 50k 32x32 images, batch 256 + CUDA graphs (c100_* runs):
| run | s/epoch | note |
|---|---|---|
| c100_t1_plain / t2_ls_wd (Tiny p4, 65 tok) | 3.02 | no aug |
| c100_t3_randaug | 3.39 | +12% for RandAugment |
| c100_t6_full_300ep | 3.50 | full recipe |
| c100_s_small_200ep | 3.86 | Small only 1.1-1.3x Tiny (launch bound) |
| c100_r18_full (CIFAR stem, 32x32) | 1.84 | stride2 at 64x64 costs about the same per image |
(c100_t4/t5 show 2.3 / 4.1 s: contended by other queues, so not used for calibration.)

- Run minutes = `epochs * s/ep / 60 + 0.5` (fixed startup/compile; c100_t1: 5.04 min train, 5.49 total).
- Tiny-ImageNet: 100k images = 2.0x CIFAR; Tiny p8 on 64x64 = 64 patches + cls = 65 tokens, same per-image cost as CIFAR p4.
  Tiny plain 6.1 s/ep, with RandAug/mix 6.8 s/ep; Small 7.7; r18 stride2 3.7.
- Tiny p4 on 64x64 = 257 tokens: no CUDA-graph measurement exists; the 257-token CIFAR p2 run was 2.4x p4 at batch 1024, assume
  3x with graphs => ~20 s/ep (range 27-45 min).
- ImageNet-64: 1.281M images = 25.6x CIFAR; adds ~3% for eval every 2 epochs and step logging.
- Concurrency: we run ~3 queues on the GPU at once. Each job is ~1.5x slower, total throughput ~2.2x. Standalone minutes
  below are per-job GPU-alone times; concurrent wall-clock per job = x1.5.

## Chain 4: Tiny-ImageNet (100k imgs, 64x64) - 2.07 h standalone, 3.1 h at 1.5x
| run | model | tokens | epochs | s/epoch | min alone | min at 1.5x | source |
|---|---|---|---|---|---|---|---|
| tin_t1_plain | tiny p8 | 65 | 100 | 6.1 | 10.6 | 16 | scaled from c100_t1 |
| tin_t2_ls_wd | tiny p8 | 65 | 100 | 6.1 | 10.6 | 16 | scaled |
| tin_t3_randaug | tiny p8 | 65 | 100 | 6.8 | 11.8 | 18 | scaled from c100_t3 |
| tin_t4_mix | tiny p8 | 65 | 100 | 6.8 | 11.8 | 18 | scaled |
| tin_t5_droppath | tiny p8 | 65 | 100 | 6.8 | 11.8 | 18 | scaled |
| tin_r18_full | resnet18 stride2 | - | 100 | 3.7 | 6.6 | 10 | scaled from c100_r18 |
| tin_s_small_200ep | small p8 | 65 | 200 | 7.7 | 26.2 | 39 | scaled from c100_small |
| tin_p4_tiny | tiny p4 | 257 | 100 | 20 | 34.5 (27-45) | 52 | extrapolated |
| **total** | | | | | **124 (2.07 h)** | **186 (3.1 h)** | |

## Chain 5: ImageNet-64 (1.281M imgs) - 2.98 h standalone, 4.5 h at 1.5x
| run | model | tokens | epochs | s/epoch | min alone | min at 1.5x | source |
|---|---|---|---|---|---|---|---|
| in64_t1_plain | tiny p8 | 65 | 30 | 80 | 40.5 | 61 | scaled from c100_t1 |
| in64_t5_full | tiny p8 | 65 | 30 | 90 | 46 | 69 | scaled from c100_t6 |
| in64_r18_full | resnet18 stride2 | - | 30 | 49 | 24.5 | 37 | scaled from c100_r18 |
| in64_s_small_full | small p8 | 65 | 40 | 102 | 68 | 102 | scaled from c100_small |
| **total** | | | | | **179 (2.98 h)** | **269 (4.5 h)** | |

## Summary
| chain | alone h | cost alone | wall h at 1.5x (concurrent) |
|---|---|---|---|
| 4 Tiny-ImageNet | 2.07 | $6.8 | 3.1 |
| 5 ImageNet-64 | 2.98 | $9.8 | 4.5 |
| **total** | **5.05** | **$16.6** | |

Concurrency-aware wall-clock (3 queues active, 2.2x aggregate throughput, $3.29/hr of wall time):
- Perfectly packed across 3 queues: 5.05 / 2.2 = **2.3 h, ~$7.6** (only reachable if the work is split so all 3 queues stay busy to the end).
- Both chains launched concurrently (the two remaining chains plus one other queue): Tiny-ImageNet finishes in ~3.1 h, ImageNet-64 in
  ~4.5 h, so the GPU is billed **~4.5 h, ~$14.8**; after Tiny-ImageNet ends, ImageNet-64 speeds up (fewer neighbours), so 3.5-4.5 h
  (~$11.5-14.8) is realistic.
- Strictly back to back with no other queues: 5.05 h, $16.6.
- To approach the packed number, run `in64_s_small_full` (the longest single job, 68 min alone) as its own queue rather than last in the chain.

## Caveats
- Tiny-ImageNet/ImageNet-64 are scaled from CIFAR measurements; re-fit after tin_t1_plain / tin_t3 finish (data loading and 64x64 RandAug cost
  are not measured). tin_p4_tiny is the least certain (no CUDA-graph measurement at 257 tokens).
- ImageNet-64 uint8 data is ~15.7 GB; assumes it fits in GPU memory / loads without a data-loader bottleneck.
- Estimates apply to the baseline recipe; adding --lsa / seqpool later changes per-epoch cost (bf1_slvit measured ~6.1 s/ep vs 7.0 s/ep for the matching Tiny while under contention).
