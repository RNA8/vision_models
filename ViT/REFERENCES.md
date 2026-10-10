# Published CIFAR-10 reference results (verified 2026-10-10)

Every number below was taken from a source fetched on 2026-10-10. The quoted evidence is the table row or sentence in that source.
The ViT-Lite-7/4, CCT-7/3x1, mimetic-init and Cutout-ResNet-18 rows were re-checked against the page text.
All results are 32x32 and from scratch (no extra data) unless noted. Accuracy is 100 − error where the paper reports error.

## Small ViTs trained from scratch (the comparison that matters for our ViT-Tiny)

| Model | Params | Epochs (batch) | Pure / hybrid | Acc | Source | Evidence |
|---|---|---|---|---|---|---|
| ViT-Tiny 192/12/3, p4, plain init (Trockman & Kolter, mimetic-init paper, baseline) | ~5.4M | 100 (512) | pure | **88.43** | [arXiv 2305.09828](https://arxiv.org/html/2305.09828) Tab. 7 | "4 32 88.43 90.47 2.03" |
| same + mimetic attention init | ~5.4M | 100 (512) | pure | 90.47 | same | same row |
| DeiT-T, patch 2 (trained by NesT authors) | 5.3M | n/s | pure | 88.39 | [arXiv 2105.12723](https://arxiv.org/html/2105.12723) Tab. A3 | "DeiT-T … 5.3M 158M 1905.3 88.39 67.52" |
| ViT-12/16 (CCT paper) | 85.6M | 300 (128) | pure | 83.04 | [arXiv 2104.05704](https://arxiv.org/html/2104.05704v4) Tab. 1 | "ViT-12/16 83.04 % … 85.63 M" |
| ViT-Lite-7/4, no aug / no tuning (CCT ablation) | 3.72M | 300 | pure | 83.59 | same, Tab. 8 | "ViT-Lite-7/4 CT ✗ ✗ ✗ ✗ 83.59 %" |
| omihub777/ViT-CIFAR (7L, dim 384) | 6.3M | 200 (128) | pure | 90.92 | [GitHub](https://github.com/omihub777/ViT-CIFAR) | "CIFAR-10 \| **90.92**" |
| ViT-Lite-7/4 (7L, dim 256, p4), full DeiT-style recipe | 3.72M | 300 (128) | pure | **93.57** | [arXiv 2104.05704](https://arxiv.org/html/2104.05704v4) Tab. 1 | "ViT-Lite-7/4 93.57 % … 3.72 M" |
| ViT baseline (depth 9, dim 192), SPT/LSA paper | 2.8M | 100 (128) | pure | 93.58 | [arXiv 2112.13492](https://arxiv.org/html/2112.13492v1) Tab. 2 | "ViT 8593 189.8 2.8 93.58" |
| CVT-7/4 (SeqPool instead of CLS) | 3.72M | 300 (128) | pure | 94.01 | arXiv 2104.05704 Tab. 1 | "CVT-7/4 94.01 % … 3.72 M" |
| SL-ViT (ViT + SPT + LSA) | 2.9M | 100 (128) | pure + locality | 94.53 | arXiv 2112.13492 Tab. 2 | "SL-ViT 7697 199.2 2.9 94.53" |
| NesT-T (local attention + conv aggregation) | 6.2M | n/s | hybrid | 96.04 | arXiv 2105.12723 Tab. A3 | "NesT-T 6.2M 187M 1616.9 96.04" |
| ViT + self-supervised pretraining on CIFAR itself (Gani et al.) | 2.8M | 200 SSL + 100 | pure, 2-stage | 96.41 | [arXiv 2210.07240](https://arxiv.org/html/2210.07240) Tab. 3 | "ViT (Ours) 2.8 63.36 96.41" |
| CCT-7/3x1 (conv tokenizer) | 3.76M | 300 (128) | hybrid | 96.53 | arXiv 2104.05704 Tab. 1 | "CCT-7/3×1 96.53 % … 3.76 M" |
| CCT-7/3x1, long schedule | 3.76M | 5000 | hybrid | **98.00** | same, Tab. 2 + abstract | "5000 Sinusoidal 98.00%" |

The CCT recipe (repo `configs/datasets/cifar10.yml`) uses batch 128, AdamW lr 5.5e-4, wd 0.06, rand-m9-mstd0.5-inc1, mixup 0.8 / cutmix 1.0
(mixup off after epoch 175), label smoothing 0.1 and 10 warmup epochs. The mimetic paper uses RandAugment + Cutout, batch 512, AdamW lr 3e-3,
wd 0.01 and 100 epochs.

## CNN baselines and overall state of the art

| Model | Params | Setting | Acc | Source | Evidence |
|---|---|---|---|---|---|
| ResNet-20 / 56 / 110 (He et al.) | 0.27 / 0.85 / 1.7M | ~164 ep, batch 128 | 91.25 / 93.03 / 93.57 | [arXiv 1512.03385](https://ar5iv.labs.arxiv.org/html/1512.03385) Tab. 6 | "ResNet 110 1.7M 6.43" (error) |
| ResNet18 (kuangliu/pytorch-cifar) | 11M | n/s | 93.02 | [GitHub](https://github.com/kuangliu/pytorch-cifar) | "ResNet18 \| 93.02%" |
| PreActResNet18 (kuangliu) | 11M | n/s | 95.11 | same | "PreActResNet18 \| 95.11%" |
| ResNet18 (Cutout paper) | 11M | 200 ep, batch 128, crop + flip | **95.28** (96.01 + Cutout) | [arXiv 1708.04552](https://ar5iv.labs.arxiv.org/html/1708.04552) | "we train for 200 epochs with batches of 128" |
| WRN-28-10 | 36.5M | 200 ep | 96.00 (97.3 + RandAugment) | [arXiv 1605.07146](https://ar5iv.labs.arxiv.org/html/1605.07146), [1909.13719](https://ar5iv.labs.arxiv.org/html/1909.13719) | "WRN 28-10 36.5M 4.00" |
| PyramidNet + ShakeDrop + AA/RA (+SAM) | n/s | long | 98.5–98.6 | [1805.09501](https://ar5iv.labs.arxiv.org/html/1805.09501), [2010.01412](https://ar5iv.labs.arxiv.org/html/2010.01412) | "PyramidNet+ShakeDrop … 1.5±0.1" |
| airbench94 / 95 / 96 (Keller Jordan) | small CNN | 3.29 s / 10.4 s / 46.3 s on 1x A100 | 94 / 95 / 96 | [arXiv 2404.00498](https://arxiv.org/abs/2404.00498) | "94% accuracy in 3.29 seconds … single NVIDIA A100" |
| ViT-H/14, JFT-300M pretrained (*not from scratch*) | ~632M | fine-tuned at 384px | 99.50 | [arXiv 2010.11929](https://ar5iv.labs.arxiv.org/html/2010.11929) Tab. 2 | "CIFAR-10 99.50± 0.06" |
| EfficientNet-L2 + SAM, pretrained (*not from scratch*) | n/s | fine-tuned | 99.70 | arXiv 2010.01412 Tab. 3 | "0.30±0.01" (error) |

The Papers with Code leaderboard is offline: it redirects to huggingface.co/papers/trending.

## Where our runs sit

- **Our ViT-Tiny is at the low end of the published range.** The closest matched baseline (same 192/12/3, p4, 32px) is 88.43% at 100 epochs. Our `t3_randaug` is 86.0% at 100 epochs, and our 300-epoch full recipe is 88.0%.
- **Supervised pure ViTs reach ~93.5–94%**: ViT-Lite-7/4, the SPT-paper ViT and CVT. They are shallower (7–9 layers), use batch 128 (8x the optimizer steps per epoch of our batch 1024), and use tuned lr and wd.
- **Results above ~96% under 10M params** need CNN-like inductive bias (CCT, NesT), self-supervised pretraining (Gani et al.), or very long schedules (CCT at 5000 epochs).
- **Our ResNet-18 (93.2% at 100 epochs)** matches kuangliu's 93.02% and is ~2 points under the tuned 95.28% bar (200 epochs, batch 128).
- **Speed:** airbench reaches 94% in ~3 s on an A100. Our recipe optimizes for teaching (ablations), not for record times.

## CIFAR-100 (verified by the main agent against the source pages, 2026-10-10)

All from scratch, 32x32, no extra data unless noted.

| Model | Params | Epochs | Acc | Source | Evidence |
|---|---|---|---|---|---|
| ViTRM (Vision Tiny Recursion Model) | ~3.6M | n/s | 72.1 | [arXiv 2603.19503](https://arxiv.org/abs/2603.19503) | "On CIFAR-100, ViTRM (72.1%)" |
| ViT baseline (depth 9, dim 192), SPT/LSA paper | 2.8M | 100 (128) | 73.81 | [arXiv 2112.13492](https://arxiv.org/html/2112.13492v1) Tab. 2 | "ViT 8593 189.8 2.8 93.58 73.81 ..." |
| SL-ViT (ViT + SPT + LSA) | 2.9M | 100 (128) | 76.92 | arXiv 2112.13492 Tab. 2 | "SL-ViT 7697 199.2 2.9 94.53 {76.92} ..." |
| ViT + self-supervised pretraining on CIFAR itself (Gani et al.) | 2.8M | 200 SSL + 100 | 79.15 | [arXiv 2210.07240](https://arxiv.org/html/2210.07240) Tab. 3 | "ViT (Ours) 2.8 63.36 96.41 79.15" |
| ViT-Lite-7/4 | 3.72M | 300 (128) | 73.94 | [arXiv 2104.05704](https://arxiv.org/html/2104.05704v4) Tab. 1 | "ViT-Lite-7/4 93.57 % 73.94 % 95.16 % 99.77 % 3.72 M" (C10, C100, …) |
| CCT-7/3x1 (hybrid) | 3.76M | 300 (128) | 80.92 | arXiv 2104.05704 Tab. 1 | "CCT-7/3×1 96.53 % 80.92 % 95.56 % 99.82 % 3.76 M" (C10, C100, …) |
| CCT-7/3x1, long schedule (hybrid) | 3.76M | 5000 | 82.87 | arXiv 2104.05704 Tab. 2 | "5000 Sinusoidal 98.00% 82.87%" |

Not independently re-checked (quoted by a research agent in `RESEARCH_DIRECTIONS.md`): Gani et al. Tab. 15 same-shape ViT 12L/3h/MLP4 = 68.09 (and 9L/12h/MLP2 = 73.81), and the SL-ViT-paper L-ViT (+LSA) ablation = 74.87.
