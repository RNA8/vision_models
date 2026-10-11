# CIFAR-100: SOTA and the smallest model within 99% of it

Research date: 2026-10-11. Every number below was read from a source fetched in this session: arXiv HTML or ar5iv pages, a CVF supplement PDF, and the Hugging Face model API. Each number has its URL and a verbatim table row or sentence, copied from the HTML-to-text dump (LaTeX such as `\pm`/`\mathbf` is shown as it appears there). Anything I could not verify is marked **unverified**. Papers with Code is offline, so leaderboard claims are limited to what these papers report.

Question: *"What is the smallest model that gets within 99% of SOTA on CIFAR-100?"* The answer depends on whether extra data is allowed, so there are two regimes:

- **(A)** CIFAR-100 only, trained from scratch. Distillation from a teacher trained on CIFAR counts as no extra data.
- **(B)** Extra-data pretraining (ImageNet-1k/21k, JFT, CLIP/LVD), then fine-tuning or a linear probe. **Per the 2026-10-11 update, (B) is now the primary target.**

Our context: we have a 2.7M-param pure ViT (9L, d=192, LSA, SeqPool) that reaches **75.69%** from scratch in 300 epochs.

---

## TL;DR

| Regime | SOTA (source) | 99% threshold | Smallest model ≥ threshold | Params | Acc |
|---|---|---|---|---|---|
| A: from scratch | **89.90%**: PyramidNet-272 (α=200) + ShakeDrop + AA + CutMix + ASAM (ASAM, arXiv 2102.11600) | **89.00%** | PyramidNet-272 α=200 + ShakeDrop (same family; SAM+AA = 89.7, SGD+AA = 89.3–89.4) | **26.0M** | 89.3–89.9 |
| B: pretrained + fine-tune | **96.08%**: EfficientNet-L2 + SAM fine-tune at 475px, NoisyStudent/JFT pretraining (SAM, arXiv 2010.01412) | **95.12%** | EfficientNet-L2 without SAM (95.93%). No fetched model under 480M params reaches 95.12 | **480M** | 95.93 |

Regime (A) has a hard floor: nothing under about 26M params published ≥ 89.0% from scratch. The best results under 5M are CCT-7/3x1 (3.76M) at 80.92 (300 ep) and 82.87 (5000 ep), which is 90.0% and 92.2% of SOTA.

Regime (B) has the same floor at 99% of SOTA, because only EffNet-L2-scale models (480M) clear 95.1. The thresholds are much easier to reach at 90–95% of SOTA:

- **≥ 90% of 96.08 (86.47)**: **EfficientNet-B0, 4M, 88.1%** (ImageNet-1k pretraining). Also MobileNetV3, 4.2M, 88.5% (IN-21k-P, 224px); DeiT-Ti, 5.5M, 86.5–86.8% (224px); ViT-Ti/16 augreg IN-21k, 5.8M, 87.2% (protocol caveat below).
- **≥ 95% (91.28)**: **EfficientNetV2-S, 24M, 91.5%** (IN-1k). ViT-B/16, 86M, reaches 91.67 with IN-21k and about 94.2 with the IN-21k-P recipe.
- A **frozen** DINOv2 ViT-S/14 (about 21M, param count unverified) gets **87.5%** with only a linear probe.

Our 75.69% is 84.2% of regime-A SOTA and 91.3% of CCT's 82.87.

---

## Regime A: from scratch on CIFAR-100 (no extra data)

### A.1 SOTA candidates

1. **ASAM, PyramidNet-272: 89.90 ± 0.13**. Source: https://arxiv.org/html/2102.11600 (Kwon et al., ICML 2021), Table 3:
   - "Table 3: Maximum test accuracies for SGD, SAM and ASAM on CIFAR-100 dataset."
   - "PyramidNet-272† | 88.91_{\pm 0.12} | 89.36_{\pm 0.20} | \mathbf{89.90}_{\pm 0.13} |" (SGD | SAM | ASAM)
   - Setup: "For PyramidNet-272 (Han et al., 2017), we additionally apply some latest techniques: AutoAugment (Cubuk et al., 2019), CutMix (Yun et al., 2019) and ShakeDrop (Yamada et al., 2019). … The number of epochs is set to 900 for SAM and ASAM and 1800 for SGD."
   - Same table: "WRN-28-10 | 81.56_{\pm 0.13} | 83.42_{\pm 0.04} | \mathbf{83.68}_{\pm 0.12} |", "ResNeXt29-32x4d | 79.76… | 81.48… | \mathbf{82.30}…", "WRN-28-2 | 75.28… | 77.25… | \mathbf{77.54}…", "ResNet-56 | 73.12… | 75.16… | \mathbf{75.86}…", "ResNet-20 | 69.76… | 71.06… | \mathbf{71.40}…"
2. **SAM, PyramidNet+ShakeDrop + AutoAugment: error 10.3 (89.7%)**. Source: https://arxiv.org/html/2010.01412 (Foret et al., ICLR 2021), Table 1, CIFAR-100 columns (SAM | SGD):
   - "PyramidNet+ShakeDrop | AA | … | \textbf{10.3}_{\pm 0.1} | 10.6_{\pm 0.1} |"
   - "WRN-28-10 (1800 epochs) | AA | … | \textbf{12.8}_{\pm 0.2} | 16.1_{\pm 0.2} |" (87.2%)
   - "Shake-Shake (26 2x96d) | AA | 12.8±0.1 | 14.1±0.2" (87.2%)
   - "PyramidNet | AA | … | \textbf{11.6}_{\pm 0.1} | 14.6_{\pm 0.1} |"
3. **AutoAugment, PyramidNet+ShakeDrop: error 10.7 (89.3%)**. Source: https://arxiv.org/html/1805.09501, Table 2: "| PyramidNet+ShakeDrop [65] | 14.0 | 12.2 | \mathbf{10.7\pm 0.2} |" (Baseline | Cutout | AA). The text says: "we achieve the state-of-art result on this dataset, beating the previous record of 12.19% error rate by ShakeDrop regularization".
4. I searched for later from-scratch records above 90% and found none. Two searches turned up SAM variants, but on PyramidNet-110 at about 85%. **SOTA (A) = 89.90 (ASAM). 99% threshold = 0.99 × 89.90 = 89.00.** Using SAM's 89.7 instead gives a threshold of 88.80, and the answer does not change.

**PyramidNet-272 α=200 parameter count: 26.0M.** Source: https://arxiv.org/html/1610.02915 (PyramidNet paper), Table 4: "PyramidNet (bottleneck, \alpha=200 ) | 26.0M | 864 | 272 | 5,005MB | 3.31 \pm 0.08 | 16.35 \pm 0.24 |". The ShakeDrop paper (https://arxiv.org/html/1802.02375) uses the same "PyramidNet-272 \alpha 200" with "| ShakeDrop | 3.08 | 14.96 |". I am assuming the SAM paper's "PyramidNet+ShakeDrop" is this 272/α200 model; this is consistent with the AutoAugment paper but **not stated verbatim** in the SAM paper.

**Smallest model ≥ 89.00: PyramidNet-272 α=200 + ShakeDrop, 26.0M** (89.9 ASAM / 89.7 SAM / 89.4 SGD+AA). Every other fetched result ≥ 85% is also ≥ 26M: WRN-28-10 is 36.5M ("Wide ResNet (width \times 10 ) [34] | 36.5M | … | 20.50"), and DenseNet-BC k=40 is 25.6M at 82.82% ("DenseNet-BC ( k=40 ) [9] | 25.6M | … | 17.18"). A search for "89% CIFAR-100 from scratch under 10M params" returned nothing.

### A.2 Small-model frontier (from scratch, CIFAR-100 only; KD with a CIFAR-trained teacher allowed)

| Params | Model | CIFAR-100 | Recipe | Source |
|---|---|---|---|---|
| 0.28M | CCT-2/3x2 | 66.93 | 300 ep | CCT, see below |
| 0.28M | ResNet-20 (ASAM) | 71.40 | ASAM | 2102.11600 |
| 0.47M | ResNet32 + DKD (T: ResNet110) | 74.11 | KD | DKD |
| 0.57M | WRN-40-1 + SimKD (T: WRN-40-2) | 75.56 | KD | SimKD |
| 0.70M | WRN-16-2 + SimKD (T: ResNet-32x4) | 77.17 | KD | SimKD |
| 0.70M | MobileNetV2/0.5 | 56.32 | 300 ep | CCT |
| 0.81M | MobileNetV2 (w0.5) + SimKD (T: WRN-40-2) | 70.71 | KD | SimKD |
| 0.86M | ResNet56 | 74.81 (CCT) / 75.86 (ASAM) / 76.36 (SL-ViT recipe) | | |
| 0.95M | ShuffleNetV1 + SimKD (T: ResNet-32x4) | 77.18 | KD | SimKD |
| 1.23M | ResNet8x4 + SimKD (T: ResNet-32x4) | **78.08** | KD | SimKD |
| 1.36M | ShuffleNetV2 + SimKD (T: ResNet-32x4) | **78.39** | KD | SimKD |
| 1.7M | ResNet110 | 76.63 (CCT) / **79.86** (SL-ViT recipe) | | |
| 1.7M | PyramidNet-110 α=48 | 76.88 (23.12 err) | | PyramidNet |
| ~1.97M+ | airbench96 | 79.27 (no cutout), 79.76 (cutout), with TTA | | airbench; params **unverified** (1.97M is airbench94) |
| 2.24M | MobileNetV2/1.0 | 63.69 | | CCT |
| 2.7M | **ours: ViT 9L d192 LSA SeqPool** | **75.69** | 300 ep | internal |
| 2.8–2.9M | ViT / SL-ViT | 73.81 / 76.92 | | SL-ViT |
| 3.23M | CCT-6/3x1 | 79.40 | 300 ep | CCT |
| 3.72M | CVT-7/4 | 76.49 | | CCT |
| 3.76M | **CCT-7/3x1** | **80.92** (300 ep), **82.87** (5000 ep) | | CCT |
| 3.8M | PyramidNet-110 α=84 | 79.34 (20.66 err) | | PyramidNet |
| 3.85M | CCT-7/3x2 | 77.72 | | CCT |
| 7.4M | ResNet-32x4 (teacher) | 79.42 | | DKD / SimKD |
| 8.7M | WRN-40-4 | 77.11 (22.89 err) | | PyramidNet table |
| 9.2M | SL-CaiT | **80.32** | | SL-ViT |
| 10.2–10.3M | ResNet-1001 pre-act | 77.29 (22.71 err) | | |
| ~26M | PyramidNet-272 + ShakeDrop | 85.0 (no AA), 89.3–89.9 (AA/SAM/ASAM) | | see A.1 |
| 36.5M | WRN-28-10 | 83.68 (ASAM), 87.2 (SAM+AA, 1800 ep) | | |

The 5–15M band is thin. The best published from-scratch results there are SL-CaiT (9.2M) at 80.32 and SL-Swin (10.2M) at 79.99. I did not find a 5–15M model above 83% from scratch. The ASAM WRN-28-10 result (83.68) is 36.5M.

Sources and verbatim quotes for A.2:

- **CCT** (Hassani et al.), https://arxiv.org/abs/2104.05704 (ar5iv HTML). Table 1, "Top-1 validation accuracy comparisons" (columns C-10 | C-100 | Fashion | MNIST | # Params | MACs):
  - "CCT-7/3×1 | 96.53\% | 80.92\% | … | 3.76 M | 1.19 G |"
  - "CCT-7/3×1⋆ | 98.00% | 82.72% | - | - | 3.76 M | 1.19 G |"
  - "CCT-7/3×2 | 95.04\% | 77.72\% | … | 3.85 M | 0.29 G |"
  - "CCT-2/3×2 | 89.75\% | 66.93\% | … | 0.28 M | 0.04 G |"
  - "ResNet56[16] | 94.63\% | 74.81\% | … | 0.85 M |"
  - "ResNet110[16] | 95.08\% | 76.63\% | … | 1.73 M |"
  - "MobileNetV2/0.5 | 84.78\% | 56.32\% | … | 0.70 M |"
  - "ViT-Lite-7/4 | 93.57\% | 73.94\% | … | 3.72 M |"
  - "CVT-7/4 | 94.01\% | 76.49\% | … | 3.72 M |"
  - Table 2 row: "5000 | Sinusoidal | 98.00% | 82.87% |"
  - Text: "82.87% accuracy on CIFAR-100 when trained for 5000 epochs".
  - These values came from the WebFetch summary, not the HTML dump: CCT-6/3x1 79.40 (3.23M), CCT-6/3x2 77.14 (3.33M), CCT-4/3x2 71.51 (0.48M), MobileNetV2/1.0 63.69 (2.24M), MobileNetV2/2.0 67.44 (8.72M), ResNet18 66.46 (11.18M). The ResNet18 row was also seen verbatim: "ResNet18 | 90.27\% | 66.46\% | … | 11.18 M |".
- **SL-ViT** (Lee et al., "Vision Transformer for Small-Size Datasets"), https://arxiv.org/html/2112.13492. Table 2 (THROUGHPUT | FLOPs | PARAMS(M) | CIFAR10 | CIFAR100 | …):
  - "SL-ViT | 7697 | 199.2 | 2.9 | 94.53 | {76.92} |"
  - "ViT | 8593 | 189.8 | 2.8 | 93.58 | 73.81 |"
  - "ResNet 56 | 4295 | 506.2 | 0.9 | 95.70 | 76.36 |"
  - "ResNet 110 | 2143 | 1020.0 | 1.7 | \bf 96.37 | 79.86 |"
  - "EfficientNet B0 | 4078 | 123.9 | 3.7 | 94.66 | 76.04 |"
  - "SL-CaiT | 2967 | 623.3 | 9.2 | 95.81 | \bf 80.32 |"
  - "SL-Swin w/ \mathrm{S}_{pool} | 5711 | 284.9 | 10.2 | 95.93 | 79.99 |"
  - "SL-PiT w/ \mathrm{S}_{pool} | 5981 | 322.9 | 8.7 | 95.88 | 79.00 |"
  - Config: "the depth was set to 9, the hidden dimension was set to 192, and the number of heads was set to 12."
- **PyramidNet** Table 4 (CIFAR-100 error), https://arxiv.org/html/1610.02915:
  - "PyramidNet ( \alpha=48 ) | 1.7M | 64 | 110 | 655MB | 4.58 \pm 0.06 | 23.12 \pm 0.04 |"
  - "PyramidNet ( \alpha=84 ) | 3.8M | … | 20.66 \pm 0.40 |"
  - "Wide ResNet (width \times 4 ) [34] | 8.7M | 256 | 40 | 775MB | 4.97 | 22.89 |"
  - "Pre-activation ResNet [8] | 10.2M | 64 | 1001 | 2,921MB | 4.62 | 22.71 |"
- **DKD** (Zhao et al., CVPR 2022), https://arxiv.org/html/2203.08679. Table 6 (same architecture), student rows (ResNet20 | ResNet32 | ResNet8×4 | WRN-16-2 | WRN-40-1 | VGG8):
  - Student baselines: "69.06 | 71.14 | 72.50 | 73.26 | 71.98 | 70.36 |"
  - "DKD | 71.97 | 74.11 | 76.32 | 76.24 | 74.81 | 74.68 |"
  - "CRD[33] | 71.16 | 73.48 | 75.51 | 75.48 | 74.14 | 73.94 |"
  - "ReviewKD[1] | 71.89 | 73.89 | 75.63 | 76.12 | 75.09 | 74.84 |"
  - Table 7 (different architectures; ShuffleNet-V1 | ShuffleNet-V1 | MobileNet-V2 | MobileNet-V2 | ShuffleNet-V2): "DKD | 76.45 | 76.70 | 69.71 | 70.35 | 77.07 |", "ReviewKD[1] | 77.45 | 77.14 | 70.37 | 69.89 | 77.78 |"
- **SimKD** (Chen et al., CVPR 2022), https://arxiv.org/html/2203.14001:
  - Table 1 students (WRN-40-1 | ResNet-8x4 | ResNet-110 | ResNet-116 | VGG-8 | ResNet-8x4 | ShuffleNetV2): "SimKD | 75.56 \pm 0.27 | 78.08 \pm 0.15 | 77.82 \pm 0.15 | 77.90 \pm 0.11 | 75.76 \pm 0.12 | 76.75 \pm 0.23 | 78.39 \pm 0.27 |"
  - Teachers: "WRN-40-2 | ResNet-32x4 | ResNet-110x2 | ResNet-110x2 | ResNet-32x4 | WRN-40-2 | ResNet-32x4 |"
  - Table 2 students (ShuffleNetV1 | WRN-16-2 | ShuffleNetV2 | MobileNetV2 | MobileNetV2x2 | WRN-40-2 | ShuffleNetV2x1.5): "SimKD | 77.18 \pm 0.26 | 77.17 \pm 0.32 | 78.25 \pm 0.24 | 70.71 \pm 0.41 | 75.43 \pm 0.26 | 79.29 \pm 0.11 | 79.54 \pm 0.26 |"
  - Params for ShuffleNetV2x1.5 and MobileNetV2x2: **unverified**.
- **KD-benchmark param counts** (RepDistiller/CRD model zoo), from the SuperMix CVPR'21 supplement: https://openaccess.thecvf.com/content/CVPR2021/supplemental/Dabouei_SuperMix_Supervising_the_CVPR_2021_supplemental.pdf
  - "TNP: WRN-40-2 = 2255156, WRN-40-1 = 569780, WRN-16-2 = 703284."
  - "TNP: ResNet110 = 1736564, ResNet56 = 861620, ResNet32 = 472756, ResNet20 = 278324"
  - "TNP: ResNet32x4 = 7433860, ResNet8x4 = 1233540."
  - "VGG13 = 9462180, VGG8 = 3965028."
  - "MobileNetV2: The original MobileNetV2 [8] with the width factor of 0.5. TNP: 812836."
  - "ShuffleNetV1 = 949258, ShuffleNetV2 = 1355528."
- **airbench** (Jordan, 2024), https://arxiv.org/html/2404.00498v2. Table 5:
  - "CIFAR-100 | Yes | No | 77.54% | 79.27% |"
  - "CIFAR-100 | Yes | Yes | 78.04% | 79.76% |" (ResNet-18 | airbench96)
  - The paper notes "airbench96 uses test-time augmentation whereas standard ResNet-18 training does not."
  - "We train a convolutional network with a total of 1.97 million parameters". That figure describes airbench94; **the airbench96 param count is unverified** (it is a wider net).
- **RandAugment** (WRN-28-2, 1.5M, param count unverified), https://arxiv.org/html/1909.13719. Table 2: "Wide-ResNet-28-2 | 75.4 | - | - | 78.5 | 78.3 |" (baseline | PBA | FastAA | AA | RA).

---

## Regime B: extra-data pretraining allowed (PRIMARY)

### B.1 Overall SOTA

- **EfficientNet-L2 + SAM: 3.92 ± 0.06 error, so 96.08%.** Source: https://arxiv.org/html/2010.01412, Table 3, CIFAR-100 row:
  - "CIFAR-100 | \textbf{7.44}_{\pm 0.06} | 7.68_{\pm 0.06} | 7.83 (BiT-M) | \textbf{3.92}_{\pm 0.06} | 4.07_{\pm 0.08} | 6.49 (BiT-L) |"
  - Columns are EffNet-b7+SAM | EffNet-b7 | prev SOTA (IN only) | EffNet-L2+SAM | EffNet-L2 | prev SOTA.
  - Setup: "EfficientNet-L2 (pretrained on ImageNet plus unlabeled JFT; input resolution 475)".
  - EfficientNet-L2 without SAM: 4.07 error, so **95.93%**.
- **EfficientNet-L2 params: 480M.** Source: https://arxiv.org/html/1911.04252 (Noisy Student): "Noisy Student Training (EfficientNet-L2) | 480M | 300M unlabeled images from JFT | 88.4% | 98.7% |".
- **ViT-H/14 (JFT-300M): 94.55 ± 0.04** at fine-tune resolution 384. Source: https://arxiv.org/html/2010.11929, Table 2: "CIFAR-100 | \mathbf{94.55}{\scriptstyle\,\pm\,0.04} | 93.90… | 93.25… | 93.51… | - |". The columns are ViT-H/14 JFT | ViT-L/16 JFT | ViT-L/16 I21k | BiT-L R152x4 | NoisyStudent L2.
  - Table 6: "ViT-H/14 | 14 | 88.08 | 90.36 | 99.50 | 94.71 |…"
  - Params: "ViT-Huge | 32 | 1280 | 5120 | 16 | 632M |", "ViT-Large | 24 | 1024 | 4096 | 16 | 307M |"
  - Table 5 caption: "Models are fine-tuned at 384 resolution."
- **BiT-L (R152x4, JFT): 93.51.** BiT (https://arxiv.org/html/1912.11370) also documents its resolution: "We resize input images with area smaller than 96\times 96 pixels to 160\times 160 pixels, and then take a random crop of 128\times 128 pixels." So BiT fine-tunes CIFAR at 128px.
- I searched for anything above 96.08 (ViT-G, EVA, DINOv2 fine-tune, and so on) and found nothing verifiable. **SOTA (B) = 96.08%. 99% threshold = 0.99 × 96.08 = 95.12%.**
- **Smallest model ≥ 95.12: EfficientNet-L2 (480M), 95.93 plain / 96.08 with SAM.** ViT-H/14 (632M) at 94.55–94.71 falls short (98.4–98.6% of SOTA). No fetched model under 480M reaches 95.12.

### B.2 Small and medium pretrained models with published CIFAR-100 transfer numbers

FT = full fine-tune. LP = linear probe on frozen features.

| Params | Model | Pretrain | CIFAR-100 | Mode | Input res | Source |
|---|---|---|---|---|---|---|
| 2.2M | MobileNetV2 1.0 | IN-1k | **80.8** | FT | 224 (native) | Kornblith 1805.08974 |
| 3.2M | MobileNet v1 | IN-1k | 82.3 | FT | 224 | Kornblith |
| 4M | **EfficientNet-B0** | IN-1k | **88.1** | FT | **unverified** (paper does not state it) | EfficientNet 1905.11946 |
| 4.2M | NASNet-A Mobile | IN-1k | 83.9 | FT | 224 | Kornblith |
| 4.2M ("only 4.2 M") | **MobileNetV3 (large)** | IN-1k / **IN-21k-P** | 86.7 / **88.5** | FT | **224** | Ridnik 2104.10972 |
| 4.3M | MobileNetV2 1.4 | IN-1k | 82.5 | FT | 224 | Kornblith |
| 5.5M | MobileNetV3-Large (torchvision, weaker recipe) | IN-1k | 79.51 | FT | 256 | 2406.05612 |
| 5.5M | **DeiT-Tiny** | IN-1k | **86.54** | FT | 224 | FTerViT 2605.21171 |
| 5.5–5.7M | **DeiT-Tiny** | IN-1k | **86.78** | FT, 1000 ep | 224 | ToE 2404.00672 |
| 5.8M | **ViT-Ti/16 (AugReg)** | IN-21k (300 ep) | **87.2** (IN-1k-pretrained: 84.0) | VTAB transfer, see caveat | 224 (or 384) | How to train your ViT 2106.10270 |
| — | OFA-595 | IN-1k / IN-21k-P | 88.3 / 90.3 | FT | 224 | Ridnik |
| 7.4M | ShuffleNetV2 2.0x | IN-1k | 83.32 | FT | 256 | 2406.05612 |
| 21M | TinyViT-21M | IN-21k + CLIP-distill | **86.6** | **LP** | 224 | TinyViT 2207.10666 |
| ~21M (unverified) | **DINOv2 ViT-S/14** (distilled from g/14) | LVD-142M | **87.5** | **LP (frozen)** | 224 | DINOv2 2304.07193 |
| 21.5M | EfficientNetV2-S (torchvision recipe) | IN-1k | 83.20 | FT | 256 | 2406.05612 |
| 22.1M | **DeiT-Small** | IN-1k | **90.15** | FT | 224 | ToE |
| 22.2M | **ViT-S/16 (AugReg)** | IN-21k (300 ep) | **90.8** | VTAB transfer | 224/384 | 2106.10270 |
| **24M** | **EfficientNetV2-S** | IN-1k | **91.5 ± 0.11** | FT | unverified | EfficientNetV2 2104.00298 |
| 25M | ResNet-50 | IN-1k / IN-21k-P | 86.8 / 90.3 | FT | 224 | Ridnik |
| 28.6M | ConvNeXt-Tiny (torchvision) | IN-1k | 82.60 | FT | 256 | 2406.05612 |
| — | TResNet-M | IN-1k / IN-21k-P | 89.5 / 91.7 | FT | 224 | Ridnik |
| 55M | EfficientNetV2-M | IN-1k | 92.2 | FT | | 2104.00298 |
| 64–66M | EfficientNet-B7 | IN-1k | 91.7 (92.56 with SAM) | FT | | EffNet / SAM |
| 86M | DeiT-B | IN-1k | 90.8 (91.3 distilled; 91.4 at 384) | FT | 224/384 | DeiT |
| 86M | ViT-B/16 | IN-21k | 91.67 (Ridnik recipe: 94.2, model identity partly unverified) | FT | 384 | ViT / Ridnik |
| ~86M (unverified) | DINOv2 ViT-B/14 | LVD-142M | 91.3 | LP | | DINOv2 |
| ~150M (unverified) | CLIP ViT-B/16 | WIT-400M | 83.1 | LP | 224 | CLIP |
| 121M | EfficientNetV2-L | IN-1k | 92.3 | FT | | 2104.00298 |
| 307M | ViT-L/16 | JFT / IN-21k | 94.04 / 93.25–93.44 | FT | 384 | ViT |
| ~300M (unverified) | DINOv2 ViT-L/14 | LVD-142M | 93.4 | LP | | DINOv2 |
| 632M | ViT-H/14 | JFT | 94.55 / 94.71 | FT | 384 | ViT |
| ~1.1B (unverified) | DINOv2 ViT-g/14 | LVD-142M | 94.4 | LP | | DINOv2 |
| 480M | EfficientNet-L2 (NS, JFT) | JFT + IN | 95.93 / **96.08 (SAM)** | FT | **475** | SAM |

**Thresholds for regime B (SOTA 96.08):**

| Target | Threshold | Smallest model that meets it |
|---|---|---|
| 99% | 95.12 | EffNet-L2, 480M (no smaller verified) |
| 98% | 94.16 | ViT-H/14, 632M (94.55). ViT-L/16 JFT, 307M, reaches 94.04 and just misses. EffNet-L2 is smaller than ViT-H, so the answer is still 480M |
| 95% | 91.28 | **EfficientNetV2-S, 24M (91.5)** |
| 90% | 86.47 | **EfficientNet-B0, 4M (88.1)**. Then MobileNetV3, 4.2M (88.5), and DeiT-Ti, 5.5M (86.5–86.8) |
| (ref) | — | Frozen DINOv2 ViT-S/14 + linear head: 87.5 (91% of SOTA) without any backbone fine-tuning |

### B.3 Verbatim quotes for B.2

- **EfficientNet** (Tan & Le 2019), https://arxiv.org/html/1905.11946, Table 5: "CIFAR-100 | NASNet-A | 87.5% | 85M | EfficientNet-B0 | 88.1% | 4M (21x) | Gpipe | 91.3% | 556M | EfficientNet-B7 | 91.7% | 64M (8.7x) |". The fine-tune resolution is not stated in the fetched text (**unverified**).
- **EfficientNetV2** (Tan & Le 2021), https://arxiv.org/html/2104.00298. Table 8 caption: "All models are pretrained on ImageNet ILSVRC2012 and finetuned on downstream datasets."
  - "EfficientNetV2-S | 24M | 83.2 | 98.7 \pm 0.04 | 91.5 \pm 0.11 |…"
  - "EfficientNetV2-M | 55M | 85.1 | 99.0 … | 92.2 \pm 0.08 |"
  - "EfficientNetV2-L | 121M | 85.7 | 99.1 … | 92.3 \pm 0.13 |"
  - "DeiT-B (ViT+regularization) (Touvron et al., 2021) | 86M | 81.8 | 99.1 | 90.8 |"
- **Kornblith et al., "Do Better ImageNet Models Transfer Better?"**, https://arxiv.org/html/1805.08974. Appendix Table I.1, "Fine-tuned" block (Food | CIFAR10 | CIFAR100 | …):
  - "MobileNet v2 | 86.2 | 95.74 | 80.8 |"
  - "MobileNet v2 (1.4) | 87.7 | 96.13 | 82.5 |"
  - "MobileNet v1 | 87.1 | 96.15 | 82.3 |"
  - "NASNet-A Mobile | 85.5 | 96.83 | 83.9 |"
  - "NASNet-A Large | 88.9 | 98.04 | 87.7 |"
  - "Trained from random initialization" block: "MobileNet v2 | 80.9 | 93.68 | 75.2 |"
  - Params and native resolution: "MobileNet v2 [68] | 2.2M | 1280 | 224 |", "MobileNet v2 (1.4) [68] | 4.3M | 1792 | 224 |", "MobileNet v1 [35] | 3.2M | 1024 | 224 |", "NASNet-A Mobile [92] | 4.2M | 1056 | 224 |"
- **ImageNet-21K Pretraining for the Masses** (Ridnik et al., NeurIPS'21 D&B), https://arxiv.org/html/2104.10972:
  - Table 4 (MobileNetV3 | OFA595 | ResNet50 | TResNet-M | TResNet-L; 1K | 21K each): "CIFAR100(1) | 86.7 | 88.5 | 88.3 | 90.3 | 86.8 | 90.3 | 89.5 | 91.7 | 90.2 | 92.5 |"
  - "MobileNetV3 backbone, for example, has only 4.2 M parameters"
  - "All results are reported for input resolution 224 ."
  - Table 3 ("we compare downstream results when using the official ImageNet-21K weights, and when using weights from semantic softmax pretraining"): "CIFAR 100(1) | 92.7 | 94.2 | | 85.5 | 92.3 |". The text names "ViT [14] and Mixer [53]", so the first pair is ViT-B/16 and the second is Mixer-B/16. The column headers did not render, so this mapping is **unverified**.
- **DeiT** (Touvron et al.), https://arxiv.org/html/2012.12877, Table 7:
  - "DeiT-B | 81.8 | 99.1 | 90.8 |…"
  - "DeiT-B \uparrow 384 | 84.4 | 99.2 | 91.4 |" (distilled row)
  - "ViT-B/16 [15] | 77.9 | 98.1 | 87.1 |" (IN-1k pretrained ViT)
  - DeiT reports no Ti/S CIFAR numbers.
- **DeiT-Ti/S fine-tunes** (ToE, Huang et al. CVPR'24), https://arxiv.org/html/2404.00672, Tab. 5:
  - "DeiT-tiny | Baseline [4] | 1.0x | Baseline [4] | 1.0x | 98.07 | 86.78 |"
  - "DeiT-small | Baseline [4] | 1.0x | Baseline [4] | 1.0x | 98.93 | 90.15 |"
  - Setup: "we fine-tune DeiT for 1,000 epochs … The input image size of 32\times 32 are resized to 224\times 224 ."
- **FTerViT** (2026), https://arxiv.org/html/2605.21171:
  - "DeiT-Tiny | CIFAR-100 | 224\times 224 | 86.54% |" (FP32 column)
  - "DeiT-Tiny [2] (5.5M params, 22.9 MB FP32) and DeiT-Small [3] (22.1M params…)"
- **How to train your ViT (AugReg)**, https://arxiv.org/html/2106.10270:
  - Table 1: "ViT-Ti (39) | 12 | 192 | 768 | 3 | 5.8M |", "ViT-S (39) | 12 | 384 | 1536 | 6 | 22.2M |"
  - Table 5 ("Detailed VTAB results"; column order Caltech101 | CIFAR-100 | DTD | …):
    - ImageNet-21k (300ep): "| Ti/16 | 93.7 | 87.2 | 73.1 |…" and "| S/16 | 95.2 | 90.8 | 77.8 |…"
    - ImageNet-1k (300ep): "| Ti/16 | 92.7 | 84.0 |…"
  - Fine-tune resolution: "Fine-tuning was done both at the original resolution (224), as well as at a higher resolution (384)".
  - **Caveat:** the appendix says "full VTAB" with sweeps over "{500,2500} steps". I could not confirm whether these CIFAR-100 numbers use the VTAB-1k subset or the full training set, so treat them as indicative.
- **TinyViT**, https://arxiv.org/html/2207.10666:
  - Table 5, "Performance of TinyViT-21M w/ and w/o pretraining for linear probe…", row "| 3 | IN-21k | 96.9 | 86.6 |" (C-10 | C-100). Row 2 (IN-21k without distillation): 84.7. Row 0 (IN-1k): 75.2.
  - Sizes: "TinyViT-5M (ours) | 80.7 | 95.6 | 5.4 |", "TinyViT-11M (ours) | 83.2 | 96.5 | 11 |"
  - **No CIFAR-100 fine-tune numbers are published for TinyViT-5M/11M** (only "more than 10% gains over CIFAR-100" for the 21M linear probe).
- **DINOv2**, https://arxiv.org/html/2304.07193, Table 8, "Linear evaluation of frozen features on fine-grained benchmarks" (Food | C10 | C100 | …):
  - "DINOv2 | ViT-S/14 | 89.1 | 97.7 | 87.5 |"
  - "ViT-B/14 | 92.8 | 98.7 | 91.3 |"
  - "ViT-L/14 | 94.3 | 99.3 | 93.4 |"
  - "ViT-g/14 | 94.7 | 99.5 | 94.4 |"
  - Param counts were not quoted from the fetched text (**unverified**: S about 21M, B about 86M, L about 300M, g about 1.1B).
- **CLIP**, https://arxiv.org/html/2103.00020, Table 10, linear probe (Food101 | CIFAR10 | CIFAR100 | …):
  - "CLIP-ViT | B/32 | | 88.8 | 95.1 | 80.5 |"
  - "B/16 | | 92.8 | 96.2 | 83.1 |"
  - "L/14 | | 95.2 | 98.0 | 87.5 |"
  - "EfficientNet | B0 | | 74.3 | 92.5 | 76.5 |"
  - "EfficientNet Noisy Student | B0 | | 78.1 | 94.0 | 78.6 |"
  - "L2-475 | | 91.6 | 99.0 | 91.0 |"
  - "BiT-M | R50x1 | | 83.3 | 94.9 | 82.2 |"
  - "ViT | … L/16 | | 87.4 | 97.9 | 89.0 |"
- **Torchvision-weights study** (2406.05612, a weaker recipe, useful as a lower bound), https://arxiv.org/html/2406.05612:
  - "All images were resized to 256\times 256 for our experiments"
  - "MobileNetV3-large | … | 95.92 | 79.51 |"
  - "ShuffleNetV2 2.0\times | … | 96.30 | 83.32 |"
  - "EfficientNetV2-S | … | 96.42 | 83.20 |"
  - "ConvNeXt-Tiny | … | 96.48 | 82.60 |"
  - Params: "MobileNetV3-Large [16] | 5.5 |", "ShuffleNetV2 [19] 2.0\times | 7.4 |"
  - The same model can land anywhere from 79.5 to 88.5 depending on the recipe and pretraining (compare MobileNetV3 here with Ridnik's 86.7/88.5).

**Not found / unverified:** published CIFAR-100 fine-tune numbers for TinyViT-5M/11M, ConvNeXt-Atto/Femto/Pico, MobileViT, EfficientFormer, MobileNetV4, and DINOv2/CLIP small models. I also found no published fine-tune of a pretrained model at **native 32×32** resolution. Every transfer number above used 128–475px inputs, which multiplies compute by roughly 16–200× compared with 32px.

### B.4 Weights availability

I checked each of these with `https://huggingface.co/api/models/<id>` on 2026-10-11, and all returned HTTP 200:

- `timm/vit_tiny_patch16_224.augreg_in21k`, `timm/vit_tiny_patch16_384.augreg_in21k_ft_in1k`, `timm/vit_small_patch16_224.augreg_in21k`
- `timm/tiny_vit_5m_224.dist_in22k`, `timm/tiny_vit_11m_224.dist_in22k`, `timm/tiny_vit_21m_224.dist_in22k`
- `timm/deit_tiny_patch16_224.fb_in1k`, `timm/deit_small_patch16_224.fb_in1k`
- `timm/efficientnet_b0.ra_in1k`, `timm/tf_efficientnet_b0.ns_jft_in1k`, `timm/tf_efficientnetv2_s.in21k`, `timm/tf_efficientnet_l2.ns_jft_in1k`
- `timm/mobilenetv3_large_100.miil_in21k` (the Ridnik IN-21k-P MobileNetV3), `timm/mobilenetv2_100.ra_in1k`, `timm/mobilenetv4_conv_small.e2400_r224_in1k`
- `timm/convnext_atto.d2_in1k`, `timm/convnext_nano.in12k`
- `facebook/dinov2-small`, `timm/vit_small_patch14_dinov2.lvd142m`
- `google/vit-huge-patch14-224-in21k`, `openai/clip-vit-base-patch32`

So every small model in the table has public weights. A CIFAR-100 fine-tuned EffNet-L2+SAM checkpoint was not checked.

---

## Implications for our project

1. **Regime A (from scratch):** "99% of SOTA" means ≥ 89.0, which needs a 26M PyramidNet + ShakeDrop + AutoAugment + SAM over 900–1800 epochs. It is not reachable at 2–4M by any published result. At our size, the realistic frontier is CCT-7/3x1 (3.76M): 80.9 at 300 ep, 82.9 at 5000 ep. With KD from a CIFAR teacher, the frontier is about 78 at 1.2–1.4M (SimKD).
2. **Regime B (pretraining allowed):** "99% of SOTA" means ≥ 95.1, which needs EffNet-L2 (480M at 475px). This is also out of reach for small models. The practical targets are:
   - **About 87–88.5% at about 4–6M params**: EfficientNet-B0, MobileNetV3 IN-21k-P, DeiT-Ti, ViT-Ti/16 AugReg IN-21k. All are fine-tuned at 224px, and all have public timm weights.
   - **About 90–91.5% at 22–24M**: DeiT-S, ViT-S/16 IN-21k, EfficientNetV2-S.
   - These correspond to 90% and 95% of SOTA.
3. The pretrained numbers all come from **224–475px inputs**. Fine-tuning at 32px (our pipeline) will score lower, by an amount I did not find published. The cost of upsampling depends on the architecture (my arithmetic, not from a source):
   - **CNNs:** 224² is 49× the pixels of 32², so FLOPs go up roughly 49×.
   - **Patch-16 ViT-Ti at 224:** 196 tokens vs our 64 tokens (patch 4 at 32px), at the same width of 192. That is about 3× the tokens and roughly 4× the FLOPs, before counting depth 12 vs 9.

   Measure the accuracy drop at 32px before committing.
