# Edge-model KD and transfer techniques: quantified evidence (fetched 2026-10-11)

Rule: every number below came from a page fetched this session; the quote is the fetch tool's text for that page
(WebFetch summarises with a small model, so table rows may be re-ordered; verify against the PDF before external use).
"unverified" = not found in a fetched source. Our setup: CIFAR-100, ViT-B/16 teacher 93.43 @224, best <=6M student
RepViT-M0.9 88.33 @224 (87.81 @128 with soft KD), TinyViT-5M 88.24, KD gain only +0.07..+0.7 at 30 epochs.

## 1. Function matching / patient + consistent teacher (Beyer et al. 2022, arXiv 2106.05237)
URL: https://arxiv.org/html/2106.05237 (abstract: https://arxiv.org/abs/2106.05237)

| Technique | Quoted number | Setting | Expected gain for us |
|---|---|---|---|
| Long schedule, Pets | "82.75% / 88.01% / 90.98% / 91.06%" at 300 / 1,000 / 10,000 / 30,000 epochs; teacher 91.03 | ResNet-50 @128px student, BiT teacher | Student needs ~30x the epochs to match the teacher; at 1000 ep it is within ~3 pts |
| Long schedule, Flowers102 | 77.51 (1k ep) -> 92.83 (10k) -> 96.93 (1M); teacher 97.82; "no overfitting even when we optimize for a 1M epochs" | same | Small fine-tune sets keep gaining from epochs |
| Long schedule, Food101 | 83.29 (100 ep) -> 86.64 (1000) -> 87.20 (3000); teacher 86.24 | same | student can exceed teacher |
| Consistent vs inconsistent | "all inconsistent distillation settings plateau at a lower score"; fixed-teacher overfits on Flowers | Flowers102 | Teacher must see the SAME augmented crop as the student (on-the-fly teacher forward) |
| ImageNet | FM 80.30 (300 ep), 81.54 (1,200), 82.18 (4,800), 82.31 (9,600); fixed teacher 76.95 at 9,600 | ResNet-50, BiT-M-R152x2 | Mixup/function-matching with consistent views is the key; our 30-ep fine-tune is ~1/30 to 1/100 of the schedules where gains appear |

Caveat: these are from-scratch students (ImageNet-pretrained init is not used), so the epoch counts overstate what a pretrained student needs. Gain from 30 -> ~200-300 epochs for a pretrained student is unverified.

## 2. Fine-tuning-time tricks

| Technique | Source | Quoted number | Setting | Expected gain for us |
|---|---|---|---|---|
| SAM fine-tuning | arXiv 2010.01412, https://arxiv.org/html/2010.01412 | CIFAR-100 error EfficientNet-b7: 7.44 (SAM) vs 7.68 (no SAM); L2: 3.92 vs 4.07. "uniformly improves performance relative to finetuning without SAM" | ImageNet-pretrained, 5k steps | ~+0.15-0.25 pt, but costs 2x compute; small |
| Model soups | arXiv 2203.05482, https://arxiv.org/html/2203.05482 | "greedy soup improves on over the best model in the hyperparameter sweep by 0.7 and 0.5 percentage points" (CLIP/ALIGN ImageNet); ViT-G/14 90.94 vs 90.78 | large models, hyperparameter-sweep soups | +0.2-0.7 pt; no CIFAR-100 or tiny-model number found; free at inference |
| FixRes (train/test res) | arXiv 1906.06423 | "a computationally cheap fine-tuning of the network at the test resolution"; ResNet-50 trained at 128 77.1 vs trained at 224 79.8 (ImageNet) | ImageNet | Test-time res >= train res with brief fine-tune; cheap; we already see +2 from 128->224 |
| Layer-wise lr decay | Search only (no numeric ablation found). BEiT recipes use 0.65 (Base) / 0.75 (Large); co-training paper uses 0.7..0.85 for ViT-S..H | ViT fine-tuning | **unverified gain**; mostly relevant to ViTs (TinyViT), little to RepViT |
| Embedding-layer freezing / SGD | arXiv 2211.09359 | "freeze that embedding layer, which lets SGD ... perform slightly better than AdamW" (distribution-shift sets, not CIFAR-100) | ViT/ConvNeXt | not directly applicable |
| EfficientNetV2 progressive resizing | arXiv 2104.00298 | "Progressive resizing rises from 84.3% to 85.1%" | ImageNet training | pretraining-side; not a fine-tune trick for us |
| BN handling, lower lr for convnets | no fetched quantitative source | - | unverified |

## 3. Better source checkpoints

| Checkpoint | Source | Quoted number | Setting | Expected gain for us |
|---|---|---|---|---|
| TinyViT-5M with IN-21k distillation | arXiv 2207.10666, https://arxiv.org/html/2207.10666 | IN-1k top-1 79.1 -> 80.7 (5.4M, 1.3 GMACs); DeiT-Ti 72.2 scratch / 73.0 21k pretrain / 74.4 21k distilled | ImageNet-1k | Check we use the 21k-distilled weights (timm tag `dist_in22k_ft_in1k`; tag name not verified here) |
| TinyViT-21M transfer (linear probe) | same, Table 5 | CIFAR-100: IN-1k 75.2 (no KD) / 74.5 (KD); IN-21k 84.7 (no KD) / **86.6 (KD)** | frozen features | 21k data is worth ~+10 pts on CIFAR-100 for frozen features, distillation another +1.9; no 5M transfer row published |
| EfficientNetV2 | arXiv 2104.00298 | CIFAR-100 fine-tune: V2-S 91.5, V2-M 92.2, V2-L 92.3, B7 91.7; ViT-B/16 87.1, DeiT-B 90.8 | ImageNet-pretrained (21k used for the V2 pretrain? the fetch said "no ImageNet21k" - ambiguous) | Shows convnets beat ViTs on CIFAR-100 at equal pretraining; V2-S is 24M, too large. B0-sized V2 CIFAR number: not found |
| DINOv2 small (frozen) | arXiv 2304.07193 | CIFAR-100 linear: ViT-S/14 87.5, ViT-B/14 91.3; "The distilled model outperforms the one trained from scratch on all 12 benchmarks" (ViT-L case) | linear probe, frozen | ViT-S/14 is ~21M params, >6M budget (param count not in fetched text); fine-tuning would exceed 87.5 but exceeds budget |
| MobileCLIP-S0 | arXiv 2311.17049 | IN zero-shot 67.8 (DataCompDR-1B); image encoder 11.4M | zero-shot | Image tower 11.4M > 6M budget; no CIFAR-100 number fetched |
| TinyCLIP ViT-8M/16 | arXiv 2309.12314 (via search summary) | "zero-shot top-1 accuracy of 41.1% on ImageNet" (YFCC-15M) | zero-shot | Weak; not a useful source |
| MobileNetV4 distilled | arXiv 2404.10518 | Conv-S 73.8 (3.8M, no KD) -> 75.5 (distilled recipe); Conv-M 79.9 -> 82.7 | ImageNet; Conv-L 85.9 with 2000-epoch distillation | MNv4-Conv-S (3.8M, 0.2G MACs) is a viable cheaper student; CIFAR-100 number unavailable |
| RepViT-M0.9 | arXiv 2307.09283 | 77.4 (no KD, 300 ep) vs 78.7 (KD, 300) vs 79.1 (450 ep) | ImageNet, RegNetY-16GF teacher | The checkpoint we use is already KD-pretrained; the pretraining KD gain is +1.3 |
| FastViT-T8 | arXiv 2303.14189 | 75.6 -> 76.7 with distillation (3.6M) | ImageNet | candidate extra student, +1.1 from pretrain KD |

## 4. Logit vs feature / relation KD

| Technique | Source | Quoted number | Setting | Expected gain for us |
|---|---|---|---|---|
| DKD | arXiv 2203.08679 | CIFAR-100 ResNet32x4->ResNet8x4: KD 73.33, DKD 76.32 (+2.99); ->ShuffleNet-V1 74.07 vs 76.45 (+2.38); ImageNet R34->R18 70.66 vs 71.70 | from scratch, CIFAR resolution | Gains are on from-scratch students with weak baselines; expect <=+0.5 for us |
| DIST | arXiv 2205.10536 (ar5iv fetch) | CIFAR-100 row "KD 73.54 70.66 73.33 67.35 74.07 74.45" vs "DIST 74.73 71.75 76.31 68.66 76.34 77.35"; ImageNet ResNet-50 student, Swin-L teacher: baseline 78.5, KD 80.0, DIST 80.2; R18 student R50 teacher: baseline 73.4, KD 72.6, DIST 74.5 | strong training strategy | Key observation: plain KD can HURT with a stronger teacher (72.6 < 73.4), DIST does not; relation-based loss suits a big teacher-small student gap like ours. Estimated +0.3-1.0 |
| Logit standardization | arXiv 2403.01427 | CIFAR-100 KD 73.33 -> 76.62 (+3.29); ImageNet ResNet50->MobileNet-V1 70.50 -> 72.18 (+1.68); "setting the temperature as the weighted standard deviation of logit" | no ViT teacher results in main text | Zero-cost change; teacher logit scale (ViT-B augreg) differs from student, so likely helps; +0.2-0.8 est. |
| ViTKD (feature) | arXiv 2209.02432 | DeiT-Tiny (74.42): logit KD 75.01, ViTKD 75.40, ViTKD+NKD 76.18 (DeiT-S teacher); with DeiT III-S teacher 76.01 / 76.06 / 77.78 | ViT->ViT only, ImageNet | Needs same-family student (TinyViT/DeiT-Ti); no CNN student results. Feature+logit adds ~+0.7-1.1 over logit alone |
| DeiT dist token / MobileViT etc. | not fetched | - | unverified |

## 5. Quantization impact

| Model | Source | Quoted number | Setting | Notes |
|---|---|---|---|---|
| MobileNet-v1/v2 | arXiv 1806.08342 | v2: FP32 0.719, PTQ per-layer 0.001, PTQ per-channel 0.698, QAT per-channel 0.711 | INT8 W8A8, ImageNet | Per-channel PTQ costs ~2 pts; per-tensor can collapse; QAT recovers to ~-0.8 |
| EfficientFormerV2-S0 (3.5M) | arXiv 2303.12557 | FP32 76.2; W8A8 EasyQuant 68.21, PTQ4ViT 68.40, RepQ-ViT 40.02, Q-HyViT 74.69 | PTQ, ImageNet | Naive PTQ loses 8-36 pts on tiny hybrids; specialised PTQ -1.5 |
| MobileViTv1-xxs/xs/s | same | FP32 69.0/74.8/78.4; Q-HyViT W8A8 68.20/74.31/77.92 | PTQ | Special PTQ -0.5..-0.8 |
| RepVGG family (RepViT parent) | search summary of arXiv 2212.01593 | "more than 20% top-1 accuracy drop on ImageNet" with naive INT8; QARepVGG within 2% | PTQ | Reparameterised convs are quantization-fragile; RepViT itself: no int8 numbers found (the RepViT paper has none) |
| TinyViT, FastViT, MobileNetV4 | arXiv 2207.10666 / 2303.14189 / 2404.10518 | FastViT paper: no quantization content; MNv4: "quantized to INT8 for mobile CPUs, Hexagon, and EdgeTPUs" but no accuracy | - | **No fetched INT8 accuracy numbers.** Plan to measure ourselves |

## Ranked experiments for our setup (1xH100, GPU-time estimates)

Estimates assume 224px student epochs of 5-45 s and a ViT-B/16 teacher forward of ~15-20 s/epoch on 50k images (my estimate, not measured).

1. **Long, consistent-teacher function-matching KD** (teacher sees the same mixup/cutmix+crop view on the fly, soft KD, no hard label, 200-300 epochs) on RepViT-M0.9 and TinyViT-5M. Evidence: Beyer Pets 82.75 -> 91.06, Flowers 77.5 -> 96.9 with epochs; gains scale with epochs, and our 30-epoch run sits at the bottom of that curve. Cost: ~25-60 s/epoch x 200 = 1.5-3.5 h per model. Highest upside (could close 1-3 pts, uncertain).
2. **Swap the KD loss: DIST / logit standardization / DKD at 30-60 epochs.** Evidence: DIST ImageNet baseline 73.4, KD 72.6 vs DIST 74.5 (KD hurts with a strong teacher); LS +1.68 on R50->MobileNet. Cost: ~15-30 min per run, 3 losses x 2 models ~2 h. Trivial code (precompute teacher logits is not valid with augmentation; use on-the-fly).
3. **Resolution/FixRes + stronger source checkpoint:** fine-tune at 256-288 with test at same res (or train 224, test 256 after short fine-tune); also try MNv4-Conv-S/M and FastViT-T8 as sources (3.8M / 3.6M) and confirm TinyViT-5M uses the 21k-distilled weights. Evidence: FixRes, 128 -> 224 already +2 for us. Cost: ~0.5-1 h.
4. **Weight averaging: 3-5 seed / hyperparameter model soup + EMA + SAM(rho small).** Evidence: soups +0.5-0.7 at large scale, SAM +0.15-0.25 on CIFAR-100. Cost: SAM 2x compute; 5 x 30-epoch runs ~1-2 h. Expected +0.2-0.7.
5. **ViT-family feature+logit KD (ViTKD-style) for TinyViT-5M, plus INT8 PTQ/QAT audit** of the best model (per-channel PTQ, then QAT 3-5 epochs). Evidence: ViTKD adds ~+0.7-1.1 over logit KD for ViT students; naive PTQ on tiny hybrids loses 8-36 pts (EfficientFormerV2-S0), QAT recovers to within ~1 pt for MobileNets. Cost: ViTKD ~1 h; quant ~1 h. 

Honest bottom line: no source in this search shows a 4-5 pt gap closing for a <=6M model on CIFAR-100 by KD at short schedules. The best-evidenced lever is schedule length (item 1); the 93-vs-88 gap likely also reflects capacity, and the larger frozen-feature gains in the literature (DINOv2 ViT-B/14 91.3, EfficientNetV2-M 92.2) all come from >20M-param models.
