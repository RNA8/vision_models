# Improving a 2.7M-param ViT trained from scratch on CIFAR: literature review

Compiled 2026-10-11. Baseline being improved: SL-ViT shape (9 layers, dim 192, 12 heads, MLP ratio 2, patch 4, 64 tokens), LSA, SeqPool, mimetic-style init, AdamW, 300 epochs, strong augmentation. Results: **CIFAR-10 93.37 / CIFAR-100 75.69**.

**How the numbers were gathered.** Every number below comes from a page fetched in this session (arXiv abstract or ar5iv full text). The WebFetch tool returns extracted table cells and sentences, and I quote those as returned. Table rows were sometimes rebuilt from flattened text by the extractor, so check them against the PDF before citing externally. Anything without a fetched source is marked **unverified**. In the gain column, "abs" means a change in percentage points.

**Caveat that affects almost every row.** Most published gains are measured against *weak* baselines. Examples: DeiT-Ti at 65.08 on CIFAR-100, vanilla ViT-Tiny at 60.94, and ViT at 73.81 after 100 epochs. Our model is already at 75.69, so expect smaller deltas than the ones published.

---

## 1. Ranked table (from-scratch and same-data techniques, plus external pretraining)

Ranking criterion: expected gain for *our* model divided by cost, using the published evidence closest to our scale.

| # | Technique | Stays pure ViT? | Published gain (dataset, model size) | Cost | Source |
|---|---|---|---|---|---|
| 1 | **Shifted Patch Tokenization (SPT)** | Yes at inference; tokenizer only (concatenates shifted copies before the patch linear layer) | CIFAR-100: ViT 73.81 → S-ViT (SPT only) 76.29 (**+2.48**). L-ViT (LSA only) 74.87 → SL-ViT 76.92 (**+2.05**, the delta that applies to us). Tiny-ImageNet: L-ViT 58.50 → SL-ViT 61.07. ViT has 2.8M params, 100 epochs | Very low: a wider patch-embedding input, about the same FLOPs | Lee et al. 2021, arXiv 2112.13492 (Tables 2, 5) |
| 2 | **External supervised pretraining (ImageNet-1k / 21k), then CIFAR-100 fine-tune** | Yes | CIFAR-100 Ti/16 (~5.7M): **IN-1k 300ep 84.0**, IN-21k 30ep 85.5, IN-21k 300ep 87.2. S/16: 86.9 / 89.4 / 90.8 (VTAB results, resolution not stated). DeiT-Tiny supervised init 85.8 vs random init 42.7 (fine-tuned at **224×224**) | High at 224px. The low-res (32/64px) variant is **unverified**: no paper found | Steiner et al. 2021 arXiv 2106.10270 (Table 5); Wang et al. 2022 arXiv 2205.14443 (Table 4) |
| 3 | **Self-supervised pretraining on CIFAR itself (Gani et al.)** | Yes | CIFAR-100: ViT 2.8M scratch 73.81 → **79.15 (+5.34)**. CIFAR-10: 93.58 → 96.41. Tiny-ImageNet: 57.07 → 63.36. 200 epochs SSL plus 100 epochs fine-tune | Moderate: about 2× total epochs | Gani et al. 2022 arXiv 2210.07240 (Table 3) |
| 4 | **MAE pretraining on CIFAR at native resolution** | Yes | CIFAR-100: Mae-ViT-C100 **78.27** (3.64M, 36×36 input, patch 3, mask 0.75, **~3700-epoch** pretrain plus 300 fine-tune). mmLiT-S **79.15** (2.38M). Without MAE (100 epochs): mLiT-S 60.58 → mmLiT-S 78.18 | Very high: 4000–8000 pretrain epochs | Tan 2024 arXiv 2402.03752 (Table 3); arXiv 2407.17783 (Tables 7, 8) |
| 5 | **Distillation from a CNN teacher trained on the same data** | Yes. DeiT-hard adds a dist token; LG adds nothing at inference | CIFAR-100, DeiT-Ti (300 epochs), teacher ResNet-56 at 70.43 (32px): baseline 65.08 → DeiT-Soft 66.92 (+1.84), **DeiT-Hard 73.25 (+8.17)**, **Locality-Guidance 78.15 (+13.07)**. On ImageNet, DeiT-Ti 72.2 → hard 74.3 → class+dist 74.5 | Moderate: train a teacher, then one teacher forward per step | Li et al. 2022 arXiv 2207.10026 (Tables 1, 4); Touvron et al. arXiv 2012.12877 (Table 3) |
| 6 | **Longer schedule** | Yes | CIFAR-100, CCT-7/3×1 (3.76M): 300ep 80.92 → **1500ep 82.72 (+1.80)** → 5000ep 82.87 (+0.15 more). CIFAR-10: 96.53 → 97.48 → 98.00. ConViT: "After 1000 epochs, the ConViTs shows clear signs of overfitting" | Linear in epochs (5× for 1500) | Hassani et al. arXiv 2104.05704 (Table 2); d'Ascoli et al. arXiv 2103.10697 |
| 7 | **Smaller patch (patch 2, 256 tokens)** | Yes | CIFAR-100, DHVT-T: patch 4 (6.0M, 0.4 GFLOPs) 80.93 → patch 2 (5.8M, 1.4 GFLOPs) **83.54 (+2.61)**. DHVT-S: 82.91 → 85.68. CCT ablation (no aug): ViT-Lite-7/8 55.69 vs 7/4 58.43 | About 3.5–4× FLOPs (attention is quadratic in tokens) | Lu et al. 2022 arXiv 2210.05958 (Table 4); CCT Table 8 |
| 8 | **Weight averaging (EWA / EMA)** | Yes | CIFAR-100: EWA on ViT-XT+SL (SPT+LSA) 77.04 → **77.94 (+0.90)**; ViT-S 72.61 → 74.33. EMA on ImageNet DeiT-B: 81.8 → 81.9, with "small gains, which vanish after fine-tuning". EMA for a small ViT on CIFAR: **unverified** | Negligible | arXiv 2308.06093 (Table 2a); DeiT Table 8 |
| 9 | **SAM / GSAM** | Yes | ImageNet (no strong aug): ViT-S/16 74.4 → SAM 78.1 → **GSAM 79.5**; ViT-B/16 74.6 → 79.9 → 81.0. With aug: ViT-B/16 AUG 79.6 → SAM+AUG 81.5 (+1.9). "around 2x computational cost per update". Small ViT on CIFAR: **unverified** | 2× per step | Chen et al. arXiv 2106.01548; Zhuang et al. arXiv 2203.08065 (Table 1) |
| 10 | **Dense relative localization auxiliary loss (drloc)** | Yes (auxiliary head only during training) | CIFAR-100 at 100 epochs: T2T-ViT-14 65.16 → 68.03 (+2.87); CvT-13 73.50 → 74.51 (+1.01); ViT-B/16 59.67 → 61.42 (+1.75) | Low | Liu et al. 2021 arXiv 2106.03746 (Tables 4, 9) |
| 11 | **Mimetic / structured (impulse) attention init** (already used) | Yes | CIFAR-100, ViT-Tiny patch 2, 100 epochs: 60.94 → **67.33 (+6.39)**. CIFAR-10: 86.07 → 90.78. Structured init at 200 epochs, CIFAR-100: trunc-normal 66.50, mimetic 70.40, Imp.-5 70.46, Imp.-3 68.81 | Free | Trockman & Kolter arXiv 2305.09828 (Tables 1, 4); Zheng et al. arXiv 2404.01139 (Table 2) |
| 12 | **"Better plain ViT" recipe items (Beyer 2022)** | Yes | ImageNet ViT-S/16, 300 epochs (deltas computed from the table): GAP → [cls] −1.8; sincos2d → learned −0.4; batch 1024 → 4096 −1.4; no RandAug+MixUp −6.3; MLP head → linear −0.2 | Free | Beyer et al. arXiv 2205.01580 (Table 1) |
| 13 | **Repeated augmentation** | Yes | ImageNet DeiT-B: removing it drops 81.8 → 76.5. Small-data / CIFAR effect: **unverified** | Low, but fewer unique images per epoch | DeiT Table 8 |
| 14 | **Convolutional stem** (Xiao et al.) | **No** (conv stem) | ImageNet: "∼1-2% top-1". 1GF top-1 error 27.7 → 24.7 at 400 epochs. Also much better optimizer and lr/wd stability | Low | Xiao et al. arXiv 2106.14881 |
| 15 | **Locality add-ons: ConViT GPSA, LocalViT DW-conv FFN** | ConViT: pure attention with gated positional heads. LocalViT: **No** | ConViT-S on 10% of ImageNet: DeiT 48.0 vs ConViT 59.6; ConViT-Ti 73.1 vs DeiT-Ti 72.2. LocalViT: "+2.6%" over DeiT-T on ImageNet | Low | arXiv 2103.10697 (Tables 1, 2); arXiv 2104.05707 abstract |
| 16 | **3-Augment (DeiT III)** | Yes | ImageNet 400 epochs, vs RandAugment: ViT-S 80.7 → 80.4 (−0.3); ViT-B 82.7 → 83.1 (+0.4) | Free | Touvron et al. arXiv 2204.07118 (Table 4) |
| 17 | **Bootstrapping ViTs** (CNN-agent co-training) | Mostly (weight-sharing variant) | CIFAR-100 ViT-S (6.3M, 240 epochs): 61.25 → 76.19 (joint) / 75.50 (shared) | Moderate | arXiv 2112.03552 (Table 1) |
| 18 | **Registers** | Yes | ImageNet linear probe: DeiT-III 84.7 → 84.7; DINOv2 84.3 → 84.8. The paper reports no small-model or small-data results. Gain for us: **unverified, likely ~0** | Free | Darcet et al. arXiv 2309.16588 (Table 2) |
| 19 | **Muon optimizer** | Yes | ViTs on ImageNet-100 / Pl@ntNet: "Muon consistently outperforms AdamW", "recipe-dependent" (no numbers in the abstract). A claimed DeiT-S CIFAR-100 result (Muon 77.38 vs AdamW 72.15) seen in search results could **not** be found in the fetched paper → **unverified** | About the same as AdamW | arXiv 2605.24770 abstract |
| 20 | nGPT / normalized transformer, Differential attention, μP | Yes | No vision or small-data results found → **unverified** | n/a | nGPT arXiv 2410.01131 reports LM-only results ("factor of 4 to 20" fewer steps, LM) |
| 21 | Stochastic depth tuning | Yes | No small-ViT CIFAR sweep found → **unverified**. On ImageNet DeiT-B, removing it caused a training failure (03.4*) | Free | DeiT Table 8 |

### Reference points: SOTA-ish from-scratch CIFAR-100 at similar scale (to calibrate ambitions)
- CCT-7/3×1 (3.76M, conv tokenizer, not pure): 80.92 at 300 epochs, 82.72 at 1500 epochs (arXiv 2104.05704).
- ViT-Lite-7/4 (3.72M, pure ViT): 73.94. CVT-7/4 (SeqPool): 76.49 (arXiv 2104.05704 Table 1).
- DHVT-T (6M, hybrid): 80.93 (patch 4) and 83.54 (patch 2) at 300 epochs, "best out of five runs" (arXiv 2210.05958).
- SL-ViT (2.8M): 76.92 at 100 epochs (arXiv 2112.13492). Gani SSL ViT (2.8M): 79.15 (arXiv 2210.07240).
- Our 75.69 at 300 epochs **without SPT** sits between L-ViT (74.87) and SL-ViT (76.92), which strongly suggests SPT is the missing piece.

---

## 2. Pretrain-then-fine-tune recipes (external data now allowed)

| Recipe | Published number | Notes / source |
|---|---|---|
| Supervised IN-1k (300ep) → CIFAR-100, ViT-Ti/16 | **84.0** | Steiner et al. arXiv 2106.10270, Table 5 (VTAB). Resolution not stated in the fetched text |
| Supervised IN-21k (300ep, AugReg) → CIFAR-100, ViT-Ti/16 | **87.2** (30ep IN-21k: 85.5) | Same source. R+Ti/16: 81.9 / 82.7 / 85.3 |
| Supervised IN-1k / IN-21k → CIFAR-100, ViT-S/16 | 86.9 / 90.8 | Same source |
| DeiT-Tiny supervised init → CIFAR-100 (fine-tuned at 224) | **85.8** vs random init 42.7. MAE-Tiny 78.9. D-MAE-Tiny (distilled from MAE-Base) 85.0 (+6.1 over MAE-Tiny) | Wang et al. arXiv 2205.14443, Tables 4, 5; "we resize images to 224 × 224" |
| DeiT-B (IN-1k) → CIFAR-100 | 90.8; distilled DeiT-B rows 91.3 / 91.4. ViT-B/16 (IN-1k) 87.1 | DeiT arXiv 2012.12877, Table 7 (fine-tune 224 / ↑384) |
| TinyViT-5M: IN-1k 79.1 → with IN-21k distillation pretraining (CLIP-ViT-L/14 teacher) 80.7 on ImageNet | No CIFAR-100 number for 5M | arXiv 2207.10666 Table 4 |
| **Downsampled ImageNet (32/64px) pretraining → CIFAR at 32px, small ViT** | **No published number found → unverified.** The Chrabaszcz et al. paper reports only WRN on ImageNet64x64 (WRN-36-5: 32.34% top-1 error) and contains **no** CIFAR transfer result | arXiv 1707.08819 Table 1 |
| Fine-tune at 224 vs 32/64 | No controlled comparison for a ~3M ViT found → **unverified**. All 84–91% CIFAR-100 transfer numbers above use 224/384 inputs (≥196 tokens), which is roughly 3–50× our compute per image | |
| Large pretrained teacher → ~3M ViT student on CIFAR-100 at 32px | **No direct published result found → unverified.** Closest: D-MAE-Tiny above (224px). TVT (arXiv 2311.14337): searched ViT on CIFAR-100, vanilla 68.82 → distilled 78.80 (ConvNet teacher, details not stated). A search-summary claim about DeiT-Tiny CIFAR-100 KD from DeiT III-S was checked and is actually an **ImageNet** result (74.42 → KD 76.01 → WTTM 77.03; arXiv 2402.11148 Table 13) | |

**What this means for us.** Supervised ImageNet pretraining moves Ti/16 CIFAR-100 from about 43% (scratch, 224px, plain ViT) to 84–87%. That is the largest lever in the literature, but every measured point is at 224px. Our cached ImageNet-64 (1.28M images) is the cheap analogue: pretrain the *same* 2.7M architecture at 64px with patch 8 (64 tokens, identical sequence length, so the compute per image matches CIFAR training), or at 32px with patch 4 after downsampling. Then fine-tune at 32px with patch 4, either keeping 64 tokens by resizing the patch-embed kernel 8→4 or by fine-tuning at 64px with patch 8 on upsampled CIFAR. The expected gain is **unverified**, because no paper measures this exact setup.

---

## 3. Recommended next 5 experiments (CIFAR-100, ordered by expected gain/cost)

1. **Add SPT to the current recipe (300 epochs).** The published Δ for LSA→SL is +2.05 at 100 epochs on the identical 2.8M shape. Cost is about 1.0× a run. Expect roughly 76.5–78 (the expected range is my estimate, unverified). SPT also becomes the base architecture for experiments 2–5.
2. **Supervised ImageNet-64 pretraining → CIFAR fine-tune** of the SPT+LSA ViT. Pretrain about 30–100 epochs at 64px/patch 8 (64 tokens), then fine-tune about 50–100 epochs at 32px/patch 4 with a lower lr (e.g. 1e-4) and lighter augmentation. Run both the "resize patch kernel" and "upsample CIFAR to 64" fine-tune variants. Published analogue: IN-1k Ti/16 → CIFAR-100 **84.0** (224px). The low-res gain is unverified but likely the biggest available jump. Cost is about 1.28M images/epoch, roughly 25× a CIFAR epoch per pretrain epoch, which fits the ~40 min/epoch budget.
3. **Hard distillation (DeiT-style dist token, or locality guidance) from a strong CNN teacher.** Train or fine-tune it on CIFAR-100. An ImageNet-pretrained ConvNeXt/ResNet fine-tuned on CIFAR-100 is allowed now. Same-data evidence: +8.17 (hard) and +13.07 (LG) on DeiT-Ti 65.08, with a ResNet-56 teacher at 70.43. Note that the teacher must beat our 75.69 student, so a ResNet-56 is too weak; use a WRN-28-10 or a pretrained CNN. Cost is about 1.3–1.5× per step plus teacher training. This combines with experiment 2.
4. **Weight EMA plus a longer schedule (300 → 1000 epochs).** CCT gained +1.80 from 300→1500 epochs, and weight averaging gave +0.90 on an SL-ViT at CIFAR-100. EMA is free to add, so add it to every run from now on and log both online and EMA accuracy. Cost is about 3.3× a run. Skip this if experiment 2 succeeds, because pretraining usually replaces long scratch schedules (unverified).
5. **SAM/GSAM (ρ≈0.05–0.1) on the best of the above, or patch 2 as the alternative.** SAM gives +3.7 to +5.3 on ImageNet ViTs without strong aug and +1.9 with aug (ViT-B). Small-ViT CIFAR evidence is unverified. Cost is 2×. Patch 2 (DHVT: +2.61) costs about 4× and changes the 64-token design, so try it only if the token budget is negotiable.

Deprioritized: registers (no small-data evidence), 3-Augment (±0.4 on ImageNet), Muon / nGPT / Diff-attention / μP (no verified small-data ViT numbers), conv stem and LocalViT (break pure ViT, ImageNet-only evidence), and MAE-on-CIFAR (needs thousands of epochs; Gani-style SSL is cheaper if external data were disallowed).

---

## 4. Sources (all fetched 2026-10-10/11)
- SPT/LSA: https://arxiv.org/abs/2112.13492, https://ar5iv.labs.arxiv.org/html/2112.13492
- Gani et al. SSL on small data: https://ar5iv.labs.arxiv.org/html/2210.07240
- CCT: https://ar5iv.labs.arxiv.org/html/2104.05704
- Locality guidance: https://ar5iv.labs.arxiv.org/html/2207.10026
- Mimetic init: https://ar5iv.labs.arxiv.org/html/2305.09828
- Structured init: https://ar5iv.labs.arxiv.org/html/2404.01139
- DeiT: https://ar5iv.labs.arxiv.org/html/2012.12877
- Better plain ViT: https://ar5iv.labs.arxiv.org/html/2205.01580
- Early convolutions: https://ar5iv.labs.arxiv.org/html/2106.14881
- DeiT III: https://ar5iv.labs.arxiv.org/html/2204.07118
- Registers: https://ar5iv.labs.arxiv.org/html/2309.16588
- SAM for ViT: https://ar5iv.labs.arxiv.org/html/2106.01548 ; GSAM: https://ar5iv.labs.arxiv.org/html/2203.08065
- ConViT: https://ar5iv.labs.arxiv.org/html/2103.10697 ; LocalViT: https://arxiv.org/abs/2104.05707
- Bootstrapping ViTs: https://ar5iv.labs.arxiv.org/html/2112.03552
- drloc: https://ar5iv.labs.arxiv.org/html/2106.03746
- DHVT: https://ar5iv.labs.arxiv.org/html/2210.05958
- EWA: https://ar5iv.labs.arxiv.org/html/2308.06093
- MAE lightweight ViT: https://ar5iv.labs.arxiv.org/html/2402.03752 ; mmLiT: https://ar5iv.labs.arxiv.org/html/2407.17783
- AugReg / How to train your ViT: https://ar5iv.labs.arxiv.org/html/2106.10270
- Lightweight SSL ViTs (MAE-Tiny, D-MAE): https://ar5iv.labs.arxiv.org/html/2205.14443
- Downsampled ImageNet: https://ar5iv.labs.arxiv.org/html/1707.08819
- TinyViT: https://ar5iv.labs.arxiv.org/html/2207.10666
- TVT: https://ar5iv.labs.arxiv.org/html/2311.14337 ; WTTM: https://ar5iv.labs.arxiv.org/html/2402.11148
- Muon in ViTs: https://arxiv.org/abs/2605.24770 ; OmniOpt (claim not located): https://arxiv.org/html/2607.04033
