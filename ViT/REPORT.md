# ViT training report

_Generated 2026-10-10 15:05 from `/home/ubuntu/workspace/vision_models/ViT/runs`; 1 runs included._

Gap = clean train acc minus val acc (points). Val acc shows final epoch, best in parentheses when different.

## Compute summary

| section | runs done | pending | GPU minutes | cost |
|---|---|---|---|---|
| CIFAR-10 | 1 | 9 | 4.3 | $0.23 |
| CIFAR-100 | 0 | 8 | 0.0 | $0.00 |
| Tiny-ImageNet | 0 | 8 | 0.0 | $0.00 |
| ImageNet-64 | 0 | 4 | 0.0 | $0.00 |
| **Total** | 1 | 29 | **4.3** (0.07 h) | **$0.23** at $3.29/h |

## CIFAR-10

| run | epochs | params (M) | clean train acc | val acc | train-val gap | minutes | img/s | what this step taught |
|---|---|---|---|---|---|---|---|---|
| t1_plain | 100 | 5.36 | 100.00 | 82.90 | 17.1 | 4.3 | 21,861 | baseline (crop+flip only) |
| t2_ls_wd | pending | pending | pending | pending | pending | pending | pending |  |
| t3_randaug | pending | pending | pending | pending | pending | pending | pending |  |
| t4_mix | pending | pending | pending | pending | pending | pending | pending |  |
| t5_droppath | pending | pending | pending | pending | pending | pending | pending |  |
| t6_full_300ep | pending | pending | pending | pending | pending | pending | pending |  |
| r18_full | pending | pending | pending | pending | pending | pending | pending |  |
| s_small_200ep | pending | pending | pending | pending | pending | pending | pending |  |
| p8_tiny | pending | pending | pending | pending | pending | pending | pending |  |
| p2_tiny | pending | pending | pending | pending | pending | pending | pending |  |

Subtotal: **4.3 GPU min** (0.07 h), **$0.23**

![acc curves](report_assets/acc_curves.png)

![cls attention](report_assets/cls_attention.png)

![loss curves](report_assets/loss_curves.png)

![pos embed similarity](report_assets/pos_embed_similarity.png)

![randaug check](report_assets/randaug_check.png)

![randaug per op](report_assets/randaug_per_op.png)

![throughput](report_assets/throughput.png)

## CIFAR-100

_No runs yet; all rows pending._

| run | epochs | params (M) | clean train acc | val acc | train-val gap | minutes | img/s | what this step taught |
|---|---|---|---|---|---|---|---|---|
| c100_t1_plain | pending | pending | pending | pending | pending | pending | pending |  |
| c100_t2_ls_wd | pending | pending | pending | pending | pending | pending | pending |  |
| c100_t3_randaug | pending | pending | pending | pending | pending | pending | pending |  |
| c100_t4_mix | pending | pending | pending | pending | pending | pending | pending |  |
| c100_t5_droppath | pending | pending | pending | pending | pending | pending | pending |  |
| c100_t6_full_300ep | pending | pending | pending | pending | pending | pending | pending |  |
| c100_r18_full | pending | pending | pending | pending | pending | pending | pending |  |
| c100_s_small_200ep | pending | pending | pending | pending | pending | pending | pending |  |

Subtotal: **0.0 GPU min** (0.00 h), **$0.00**

## Tiny-ImageNet

_No runs yet; all rows pending._

| run | epochs | params (M) | clean train acc | val acc | train-val gap | minutes | img/s | what this step taught |
|---|---|---|---|---|---|---|---|---|
| tin_t1_plain | pending | pending | pending | pending | pending | pending | pending |  |
| tin_t2_ls_wd | pending | pending | pending | pending | pending | pending | pending |  |
| tin_t3_randaug | pending | pending | pending | pending | pending | pending | pending |  |
| tin_t4_mix | pending | pending | pending | pending | pending | pending | pending |  |
| tin_t5_droppath | pending | pending | pending | pending | pending | pending | pending |  |
| tin_r18_full | pending | pending | pending | pending | pending | pending | pending |  |
| tin_s_small_200ep | pending | pending | pending | pending | pending | pending | pending |  |
| tin_p4_tiny | pending | pending | pending | pending | pending | pending | pending |  |

Subtotal: **0.0 GPU min** (0.00 h), **$0.00**

## ImageNet-64

_No runs yet; all rows pending._

| run | epochs | params (M) | clean train acc | val acc | train-val gap | minutes | img/s | what this step taught |
|---|---|---|---|---|---|---|---|---|
| in64_t1_plain | pending | pending | pending | pending | pending | pending | pending |  |
| in64_t5_full | pending | pending | pending | pending | pending | pending | pending |  |
| in64_r18_full | pending | pending | pending | pending | pending | pending | pending |  |
| in64_s_small_full | pending | pending | pending | pending | pending | pending | pending |  |

Subtotal: **0.0 GPU min** (0.00 h), **$0.00**

## What I learned

## Why plain ViT overfits

TODO: one line on the train/val gap of t1_plain and why (no inductive bias, small data).

## Which regularizers mattered most

TODO: one line ranking label smoothing/WD, RandAugment, MixUp/CutMix, drop-path by val gain and gap reduction.

## ViT vs ResNet-18

TODO: one line comparing r18_full to the matching ViT run (accuracy, time, gap).

## Patch size / sequence length tradeoff

TODO: one line on tokens vs accuracy vs img/s for patch 2/4/8.

## Scaling to bigger datasets

TODO: one line on how the ViT-vs-ResNet picture changes from CIFAR to Tiny-ImageNet to ImageNet-64.
