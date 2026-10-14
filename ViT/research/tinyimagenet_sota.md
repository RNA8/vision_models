# Tiny-ImageNet: best published from-scratch results for models with 10M parameters or fewer

Searched 2026-10-14. The question: is our 2.9M pure ViT (SL-ViT shape + SPT + LSA + SeqPool, patch 8, 64 px, 1000 epochs) the best published Tiny-ImageNet result at 10M parameters or fewer when trained only on Tiny-ImageNet?

- Our numbers: **66.26** (2 seeds: 66.16, 66.36), and **68.09** for the same model distilled from one 66.16 copy of itself (born-again, function matching, 1 seed).
- "From scratch" here means no ImageNet or other external data. Self-supervised pretraining on Tiny-ImageNet itself is allowed but flagged [SSL]. Distillation from a teacher trained only on Tiny-ImageNet is allowed but flagged [KD].
- Every number below was read from the paper's own table, using the PDF text or the arXiv HTML. Error rates were converted to accuracy (100 - error) where noted. Unless stated otherwise, "Tiny-ImageNet accuracy" means the official 10k validation set, which every paper here uses as its test set.

## Verdict

**68.09 is not the best published result at 10M parameters or fewer. 66.26 is further behind.**

- **Above both of our numbers:** one result. Zhang et al. ([arXiv 2407.19394](https://arxiv.org/abs/2407.19394), Table 3) report **70.47** for a 6.4M CaiT-xxs12 with a depth-wise-convolution shortcut, trained from scratch on Tiny-ImageNet for 300 epochs.
  - That is **+2.38** over 68.09 and **+4.21** over 66.26.
  - The comparison is not like-for-like in two ways. The model is a conv/attention hybrid, not a pure ViT. It also upsamples the 64 px images to **224 px** (patch 16, 196 tokens, 1.29 GFLOPs). Our model uses 64 px and 65 tokens, on the order of 0.2 GFLOPs: Lee et al. report 199.2 MFLOPs for the same SL-ViT shape.
  - Its 2-block variant reaches 68.76 (Table 4), which is also above 68.09.
  - The paper states no seed count, and it does not say whether it reports the final-epoch or the best-epoch accuracy.
- **Between 66.26 and 68.09:** three results. All three are below the distilled 68.09 but above the undistilled 66.26.
  - Gani et al. CaiT [SSL]: 67.46 (7.7M).
  - Lee et al. SL-CaiT: 67.18 (9.2M).
  - Lee et al. EfficientNet-B0: 66.79 (3.7M).
- **Just below 66.26:** DOT ([arXiv 2307.08436](https://arxiv.org/abs/2307.08436)) distils into a ShuffleNetV2 student [KD] and gets 66.21, a 3-run mean.
- **Pure ViT, no convolutions, at 64 px:**
  - Undistilled, 66.26 is the best result we found. The next best are Gani et al.'s 2.8M ViT at 63.36 [SSL] and SL-ViT at 61.07. Gani's CaiT at 67.46 is a pure transformer, but it is 2.7x larger and SSL-pretrained.
  - Distilled, 68.09 beats every pure-transformer result at 10M parameters or fewer. That includes SL-CaiT (67.18) and Gani's CaiT (67.46).
  - This narrower claim holds as far as our search goes.
- **Training length:** 66.26 uses 1000 epochs. Most competitors use 100 epochs (Lee, Gani's supervised stage) or 200-300 epochs (DOT, Zhang). Our margin over the 100-epoch SL-* numbers is partly a schedule effect.

**Defensible claim:** the best Tiny-ImageNet result we found for a pure ViT of 10M parameters or fewer, at native 64 px, trained only on Tiny-ImageNet: 66.26 without distillation, 68.09 with self-distillation. It is *not* the best result for all architectures of 10M or fewer: a 224 px conv-hybrid CaiT reaches 70.47.

## Table 1: 10M parameters or fewer, trained only on Tiny-ImageNet (ranked by top-1)

| # | Model | Params | Top-1 | Flags | Input | Epochs | Seeds | Source |
|---|---|---|---|---|---|---|---|---|
| 1 | CaiT-xxs12 w/o (LS, TH, PE) + DWConv shortcut (hybrid) | 6.4M | **70.47** | upsampled; hybrid | 224 (patch 16) | 300 | n/s | Zhang et al., [arXiv 2407.19394](https://arxiv.org/abs/2407.19394) Table 3 (Neurocomputing preprint); code: github.com/ZTX-100/Efficient_ViT_with_DW |
| 2 | Same, DWConv spanning 2 blocks | <6.4M | 68.76 | upsampled; hybrid | 224 | 300 | n/s | same, Table 4 |
| - | **Ours: SL-ViT + SPT + LSA + SeqPool, self-distilled** | 2.9M | **68.09** | [KD] from own Tiny-only copy | 64 (patch 8) | 1000 | 1 | this repo |
| 3 | Same as row 2, DWConv spanning 3 blocks | <6.4M | 67.61 | upsampled; hybrid | 224 | 300 | n/s | same, Table 4 |
| 4 | CaiT, DINO-style self-supervised init + supervised fine-tune (Gani et al.) | 7.7M | 67.46 | [SSL] on Tiny-ImageNet | 64 | 200 SSL + 100 sup. | n/s | Gani et al., BMVC 2022, [arXiv 2210.07240](https://arxiv.org/abs/2210.07240) Table 3 |
| 5 | SL-CaiT (SPT + LSA) | 9.2M | 67.18 | - | 64 | 100 | n/s | Lee et al., [arXiv 2112.13492](https://arxiv.org/abs/2112.13492) Table 2 |
| 6 | EfficientNet-B0 | 3.7M | 66.79 | - | 64 | 100 | n/s | Lee et al. Table 2 |
| - | **Ours: SL-ViT + SPT + LSA + SeqPool** | 2.9M | **66.26** | - | 64 (patch 8) | 1000 | 2 (66.16, 66.36) | this repo |
| 7 | ShuffleNetV2 student, DKD+DOT, ResNet-18 teacher (62.99) | ~1.5M (computed: 1.0x with 200-way head) | 66.21 | [KD]; teacher is 62.99 at 64 px, so Tiny-trained (the paper does not state it) | 64 | 200 | mean of 3 | Zhao et al. (DOT), ICCV 2023, [arXiv 2307.08436](https://arxiv.org/abs/2307.08436) Table 3(b) |
| 8 | Swin (Gani et al., SSL init) | 7.1M | 65.13 | [SSL] | 64 | 200 + 100 | n/s | Gani et al. Table 3 |
| 9 | SL-Swin w/ SeqPool | 10.2M (just over) | 64.95 | - | 64 | 100 | n/s | Lee et al. Table 2 |
| 10 | MobileNetV2 student, DKD+DOT | ~2.5M (computed) | 64.60 | [KD] | 64 | 200 | mean of 3 | DOT Table 3(b) |
| 11 | MMA-CCT-7/3x2 (conv tokenizer) | 3.86M | 64.41 | - | 64 | 300 | n/s | Konstantinidis et al., [arXiv 2207.08569](https://arxiv.org/abs/2207.08569) Table 2 |
| 12 | CaiT (baseline) | 9.1M | 64.37 | - | 64 | 100 | n/s | Lee et al. Table 2 |
| 13 | ViT-Tiny + DWConv shortcut | 5.6M | 64.10 | upsampled; hybrid | 224 | 300 | n/s | Zhang et al. Table 1 |
| 14 | ResNet8x4 student, CRLD-NormKD, ResNet32x4 teacher (64.30) | ~1.3M | 63.84 | [KD] | 64 | 240 | mean of 3 | Zhang et al. (CRLD), [arXiv 2412.16493](https://arxiv.org/abs/2412.16493) Tiny-ImageNet table |
| 15 | CMT-Ti | 8.7M | 63.75 | upsampled | 224 | 300 | n/s | MSCViT, [arXiv 2501.06040](https://arxiv.org/abs/2501.06040) Table 4 |
| 16 | ViT, SSL init (Gani et al.) | 2.8M | 63.36 | [SSL] | 64 | 200 + 100 | n/s | Gani et al. Table 3 |
| 17 | ResNet-110 | 1.7M | 62.96 | - | 64 | 100 | n/s | Lee et al. Table 2 |
| 18 | SL-PiT w/ SeqPool | 8.7M | 62.91 | - | 64 | 100 | n/s | Lee et al. Table 2 |
| 19 | DenseNet-121, AI-KD self-distillation | ~7.2M (computed) | 61.67 (error 38.33) | [KD] self | n/s | 300 | mean of 3 | Kim et al. (AI-KD), [arXiv 2211.10938](https://arxiv.org/abs/2211.10938) |
| 20 | SL-ViT | 2.9M | 61.07 | - | 64 | 100 | n/s | Lee et al. Table 2 |
| 21 | SATA-ViT | 2.8M | 58.77 | - | 64 | 100 | n/s | Chen et al., WACV 2023, [arXiv 2210.12333](https://arxiv.org/abs/2210.12333) Table 2 |
| 22 | WaveMix-Lite-160/13 | 6.9M | 54.76 | no augmentation | 64 | n/s | "best of three runs" (paper-wide statement) | Jeevan et al., [arXiv 2205.14375](https://arxiv.org/abs/2205.14375) Table 11 |
| 23 | ViT-Tiny, mimetic init | ~5.5M | 50.87 | - | 64 (patch 4) | 100 | n/s | Trockman & Kolter, [arXiv 2305.09828](https://arxiv.org/abs/2305.09828) Table 4 |

Notes on Table 1:
- n/s means not stated.
- **Computed parameter counts:**
  - ShuffleNetV2 1.0x is 2.3M with a 1000-way head, of which 1.02M is the head, so about 1.5M with 200 classes.
  - MobileNetV2 is 3.5M with a 1000-way head, of which 1.28M is the head, so about 2.5M.
  - DenseNet-121 is about 7.0M plus a 0.2M head.
- **Zhang et al. (2407.19394):** all CaiT/ViT/Swin rows train at 224 px, AdamW, 300 epochs, with Swin-style augmentation (AutoAugment, Mixup, CutMix, random erasing). The paper's quote: "All experiments are trained from scratch on each dataset without the assistance of an extra dataset."
- **Lee et al. (2112.13492), Table 2:** all models train 100 epochs at 64 px with the DeiT recipe. Their FLOPs: SL-ViT 199.2M, EfficientNet-B0 123.9M, SL-CaiT 623.3M.
- **Gani et al. (2210.07240), Table 3:** also lists the SL-* and DRLoc rows. The DRLoc (Liu et al., NeurIPS 2021) Tiny-ImageNet numbers there (ViT 42.33, Swin 48.66, CaiT 45.95) are Gani's re-runs. The DRLoc paper itself has no Tiny-ImageNet experiment; we checked its text.

## Table 2: 10-25M parameters, for context (trained only on Tiny-ImageNet)

| Model | Params | Top-1 | Input | Epochs | Seeds | Source |
|---|---|---|---|---|---|---|
| ResNeXt-50 (32x4d), AdAutoMix | ~23.4M (computed) | 72.89 | 64 | 400 | mean of 3, median of last 10 epochs | Qin et al., ICLR 2024, [arXiv 2312.11954](https://arxiv.org/abs/2312.11954) Table 1 |
| ResNeXt-50, SAMix + Decoupled Mixup | ~23.4M | 72.39 | 64 | 400 | mean of 3 | Liu et al. (DM), [arXiv 2203.10761](https://arxiv.org/abs/2203.10761) Table A2 |
| MSCViT-S (hybrid) | 14.0M | 72.11 | 224 | 300 | n/s | [arXiv 2501.06040](https://arxiv.org/abs/2501.06040) Table 4 |
| BiFormer-T | 12.7M | 71.07 | 224 | 300 | n/s | same |
| CMT-XS | 14.4M | 70.96 | 224 | 300 | n/s | same |
| CaiT-xxs24 + DWConv (hybrid) | 11.8M | 70.64 | 224 | 300 | n/s | [arXiv 2407.19394](https://arxiv.org/abs/2407.19394) Table 3 |
| Shunted-T | 11.1M | 69.61 | 224 | 300 | n/s | [arXiv 2501.06040](https://arxiv.org/abs/2501.06040) Table 4 |
| ResNet-18 (CIFAR stem), AdAutoMix | ~11.2M | 69.19 | 64 | 400 | mean of 3 | [arXiv 2312.11954](https://arxiv.org/abs/2312.11954) Table 1 |
| ResNet-18, SAMix + DM | ~11.2M | 69.16 | 64 | 400 | mean of 3 | [arXiv 2203.10761](https://arxiv.org/abs/2203.10761) Table A2 |
| ResNet-18, SAMix | ~11.2M | 68.89 | 64 | 400 | mean of 3 | Li et al. (SAMix), [arXiv 2111.15454](https://arxiv.org/abs/2111.15454) Table 3 |
| ResNet-18, AutoMix | ~11.2M | 67.33 | 64 | 400 | mean of 3 | same (and the AutoMix paper, [arXiv 2103.13027](https://arxiv.org/abs/2103.13027)) |
| UPANets64 | 24.4M | 67.67 | n/s | 100 | n/s | Tseng et al., [arXiv 2103.08640](https://arxiv.org/abs/2103.08640) Table 6 |
| ViT-Small + DWConv | 21.8M | 66.86 | 224 | 300 | n/s | [arXiv 2407.19394](https://arxiv.org/abs/2407.19394) Table 1 |
| ResNet-18, PuzzleMix / CutMix / vanilla (OpenMixup re-runs) | ~11.2M | 65.81 / 65.53 / 61.68 | 64 | 400 | mean of 3 | [arXiv 2111.15454](https://arxiv.org/abs/2111.15454) Table 3 |
| PreActResNet-18 (w=1), CutMix | 11.2M | 65.09 | 64 | 1200 | n/s | Ramé et al. (MixMo), ICCV 2021, [arXiv 2103.06132](https://arxiv.org/abs/2103.06132) Table 5 |
| PreActResNet-18, Co-Mixup / PuzzleMix / vanilla | 11.2M | 64.15 / 63.48 / 56.60 (errors 35.85 / 36.52 / 43.40) | 64 | 1200 | n/s | Kim et al. (Co-Mixup), ICLR 2021, [arXiv 2102.03065](https://arxiv.org/abs/2102.03065) Table 1 |
| ResNet-18 (dataset-distillation "full data" reference) | ~11.2M | 59.6 | n/s | n/s | n/s | FocusDD, [arXiv 2501.06405](https://arxiv.org/abs/2501.06405) Table 14 (same table: MobileNetV2 45.2, EfficientNet-B0 41.6) |
| PreActResNet-18, CS-KD self-distillation | 11.2M | 58.32 (error 41.68) | n/s | 300 | mean of 3 | [arXiv 2211.10938](https://arxiv.org/abs/2211.10938) |
| ResNet-18, Zipf's LS / BAN (born-again) / vanilla | 11.2M | 59.25 / 58.65 / 56.41 | **32 px** (downsampled) | 200 | 5 seeds | Liang et al., [arXiv 2207.12980](https://arxiv.org/abs/2207.12980) Table 2 |

Parameter computation for ResNeXt-50 (32x4d): 25.0M with a 1000-way head, of which 2.05M is the head, so about 23.4M with 200 classes.

**Larger than 25M, or not from scratch:** listed only so nobody cites them as competitors by mistake.
- PreActResNet-18-3 + Cut-MixMo: 70.24 at 100.5M params (MixMo Table 5).
- Swin-T + DWConv: 71.96 at 27.7M, 224 px.
- SDMAE: 72.24, a ViT-Base (~86M) MAE-pretrained on Tiny-ImageNet at 224 px ([arXiv 2212.05677](https://arxiv.org/abs/2212.05677) Table 1).
- Astroformer: claims 92.98 "without extra data", but with hundreds of millions of parameters at 224 eval ([arXiv 2304.05350](https://arxiv.org/abs/2304.05350) Table 3).
- The Papers with Code leaderboard top entries (91-92%) are ImageNet-pretrained, even where the leaderboard marks them "no extra data":
  - Swin-L 91.35 ([arXiv 2205.10660](https://arxiv.org/abs/2205.10660)): fine-tuned at 384 px from ImageNet-21k checkpoints.
  - DeiT-B + OCD 92.0 and PUGD ViT-B/DeiT-B 90.7-91.0: ImageNet-pretrained backbones.
  - WaveMixLite 77.47: the WaveMix paper's Table 7 marks its Tiny-ImageNet 77.49 model "pre-trained on ImageNet-1k". Its from-scratch 64 px model gets 64.51.
  - We checked the leaderboard through the Wayback Machine's archive of paperswithcode.com/sota/image-classification-on-tiny-imagenet-1 and -2 (June 2024). The from-scratch entries there are already covered above: SAMix, AutoMix, DM, MixMo, UPANets, PuzzleMix, and MMA-CCT.

## Caveats

1. **Schedule length:**
   - Our 66.26 uses 1000 epochs, against 100 for Lee et al. (EfficientNet-B0, SL-CaiT), 100 supervised + 200 SSL for Gani et al., and 200-400 for the KD and mixup papers.
   - The CNN mixup papers use 400-1200 epochs, but at 11M+ parameters.
   - None of the 10M-or-fewer competitors were retrained at 1000 epochs. EfficientNet-B0 or SL-CaiT at 1000 epochs could plausibly close the gap or pass us, but no published number shows that either way.
2. **Seeds:**
   - 68.09 is a single seed. 66.26 is a 2-seed mean with a 0.2-point spread.
   - Most competitors do not state seeds. DOT, CRLD, the mixup papers, and AI-KD report means of 3 runs.
   - With 10k validation images, one standard error from sampling alone is about 0.47 points. So 67.18-67.46 vs 68.09 is a 0.6-0.9 point margin, which is real but not large.
3. **Resolution and compute:**
   - The only result above 68.09 (70.47) runs at 224 px with 196 tokens, about 6x our FLOPs at inference.
   - The MSCViT, CMT and BiFormer rows are also 224 px.
   - At native 64 px, the best 10M-or-fewer result we found is our 68.09 (distilled) or Gani's 67.46 (undistilled, SSL).
4. **Validation set used as test set:**
   - Every paper here, and our runs, report on the official 10k val split, because Tiny-ImageNet's test labels are not public.
   - Mixup papers report the median of the last 10 epochs. Some ViT papers report the final epoch.
   - None report a held-out split for model selection.
   - If our hyperparameters were tuned on the val set, our numbers carry the same small optimistic bias as everyone else's.
5. **Distillation flags:**
   - 68.09 uses a Tiny-only teacher (a copy of itself). DOT and CRLD rows also use Tiny-trained teachers, inferred from teacher accuracies of 62.99 and 64.30.
   - Comparing 68.09 against the undistilled rows is not like-for-like. The fair undistilled comparison is 66.26 vs 67.46 / 67.18 / 66.79, and on that we are 4th among 10M-or-fewer results.
6. **Search coverage:**
   - Covered: small-data ViT papers (SPT/LSA, Gani, DRLoc, SATA, DHVT, locality guidance, GvT, mimetic init, DWConv-ViT, MSCViT, MMA, SDMAE, MAE-small-data), mixup papers (Co-Mixup, MixMo, AutoMix, SAMix, DM, AdAutoMix), KD/self-KD papers (DOT, CRLD, AI-KD, Zipf's LS with BAN/CS-KD/PS-KD), WaveMix, UPANets, Astroformer, dataset-distillation baselines (FocusDD), and the archived PwC leaderboards.
   - DHVT, locality guidance (2207.10026), GvT (2404.04924) and DRLoc report no Tiny-ImageNet numbers.
   - Not exhaustively covered:
     - SAM-variant optimizer papers. The ones found report ResNet-18/56 on Tiny-ImageNet in the 48-60% range.
     - Journal-only hybrid-ViT papers (e.g. Add-ViT, Springer 2024), whose tables we could not access.
     - Chinese-language venues.
   - A 224 px conv-hybrid result above 70.47 at 10M parameters or fewer could exist in a journal we did not reach.
