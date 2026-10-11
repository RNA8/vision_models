# Edge / mobile ViT and hybrid architectures: research notes (fetched 2026-10-10/11)

Rule: every number below came from a page fetched in this session (URL given). "Quote" is a verbatim
sentence or a table row as extracted from that page. Many arXiv HTML tables come out flattened, so the
row text is the fetch tool's reconstruction of cell values in column order. Anything not fetched is
marked **unverified**.

IN-1k = ImageNet-1K top-1. "no-KD" = trained without knowledge distillation; "KD" = with distillation.

---

## 1. Per-architecture notes

### MobileViT v1 (Apple, ICLR'22): https://arxiv.org/abs/2110.02178 , https://arxiv.org/html/2110.02178
- Idea: MobileNetV2 blocks plus a MobileViT block that unfolds the feature map into patches, applies a transformer, and folds it back ("transformers as convolutions").
- Numbers (Fig. 5b): XXS 1.3M / 69.0; XS 2.3M / 74.8; S 5.6M / 78.4. Quote: "MobileViT achieves top-1 accuracy of 78.4% with about 6 million parameters".
- Training: "Unlike most ViT-based models, MobileViT models train with basic augmentation and are less sensitive to L2 regularization". No-KD.
- CIFAR: none.
- Weights: `timm/mobilevit_xxs.cvnets_in1k`, `mobilevit_xs.cvnets_in1k`, `mobilevit_s.cvnets_in1k` (HF API listing https://huggingface.co/api/models?author=timm&search=mobilevit).

### MobileViT v2 (2022): https://arxiv.org/abs/2206.02680 , https://arxiv.org/html/2206.02680
- Idea: "a separable self-attention method with linear complexity, i.e. O(k)".
- Table 6 (256px): "MobileViTv2-0.50" 1.4M, 0.5G FLOPs, 70.2; "MobileViTv2-0.75" 2.9M, 1.0G, 75.6; "MobileViTv2-1.00" 4.9M, 1.8G, 78.1. iPhone12 latency: 0.5 = 1.6 ms, 1.0 = 3.4 ms.
- Abstract: "With about three million parameters, MobileViTv2 achieves a top-1 accuracy of 75.6%".
- CIFAR: none. Weights: `timm/mobilevitv2_050/075/100/...cvnets_in1k`.

### MobileViT v3 (2022): https://arxiv.org/html/2209.15159
- Idea: simpler fusion block (1x1 conv fusion of local and global features, input added back).
- Table 2 rows: XXS (batch 384) "289 | 1.2 | 70.98"; XS (384) "927 | 2.5 | 76.7"; 0.5 "481 | 1.4 | 72.33"; 0.75 "1064 | 3.0 | 76.55" (FLOPs M | params M | top-1).
- CIFAR: none. timm weights: not found in the timm mobilevit listing (unverified elsewhere).

### EdgeNeXt (ECCVW'22): https://arxiv.org/abs/2206.10589 , https://github.com/mmaaz60/EdgeNeXt
- Idea: "split depth-wise transpose attention (STDA) encoder that splits input tensors into multiple channel groups". It combines DWConv with channel-wise (transposed) attention.
- Model zoo: edgenext_xx_small 71.23, 1.33M, 261M MAdds; x_small 74.96, 2.34M, 538M; small 79.41, 5.59M, 1.26G; small_usi 81.07 (USI recipe, i.e. KD-based).
- CIFAR: none. Weights: `timm/edgenext_xx_small.in1k`, `edgenext_x_small.in1k`, `edgenext_small.usi_in1k`.

### EfficientFormer (NeurIPS'22): https://arxiv.org/abs/2206.01191 ; V2 (ICCV'23): https://arxiv.org/html/2212.08059 , https://github.com/snap-research/EfficientFormer
- V1: "EfficientFormer-L1, achieves 79.2% top-1 accuracy on ImageNet-1K with only 1.6 ms inference latency on iPhone 12".
- V2 ideas: a 3x3 DWConv inside the FFN ("unified FFN"), stride attention (Q/K/V at 1/32 resolution), dual-path attention downsampling, DWConv on V, and talking heads.
- V2 rows: "EfficientFormerV2-S0 75.7 / 76.2 3.5M 0.40B 0.9ms"; "S1 79.0 / 79.7 6.1M 0.65B 1.1ms".
- KD: yes. "we use RegNetY-16GF with 82.9% top-1 accuracy as the teacher model for hard distillation."
- CIFAR: none. Weights: `timm/efficientformerv2_s0.snap_dist_in1k`, `..._s1...`, `efficientformer_l1.snap_dist_in1k`.

### MobileOne (CVPR'23): https://arxiv.org/html/2206.04040
- Idea: train-time multi-branch overparameterized convs that are re-parameterized into a single conv at inference. Pure CNN.
- Table 9: "MobileOne-S0 71.4 275 2.1 ... 0.79"; "S1 75.9 825 4.8 ... 0.89" (top-1, FLOPs M, params M, ..., iPhone12 ms). Trivial overparam branches give "about 0.5%" for S0.
- CIFAR: none. Weights: `timm/mobileone_s0..s4.apple_in1k`.

### FastViT (ICCV'23): https://arxiv.org/html/2303.14189 , https://github.com/apple/ml-fastvit
- Idea: RepMixer, a reparameterized DWConv token mixer with no skip connection at inference, in the early stages. Attention only in the last stage (SA variants). Train-time overparameterization and large kernels.
- Table 5: T8 3.6M, 0.7 GFLOPs, 0.8 ms, 75.6; T12 6.8M, 1.4G, 1.2 ms, 79.1. GitHub: T8 76.2 (no-KD) / 77.2 (KD).
- CIFAR: none. Weights: `timm/fastvit_t8.apple_in1k`, `fastvit_t8.apple_dist_in1k`, etc.

### RepViT (CVPR'24): https://github.com/THU-MIG/RepViT
- Idea: "revisit the efficient design of lightweight CNNs" by porting ViT macro/micro design (separate token/channel mixers, reparameterized DWConv) into MobileNetV3. Pure CNN.
- Rows: M0.9 78.7/79.1, 5.1M, 0.8G MACs, 0.9 ms; M1.0 80.0/80.3, 6.8M, 1.1G, 1.0 ms. KD: checkpoints are named "distill", and timm names are `repvit_m0_9.dist_300e_in1k`.
- CIFAR: none.

### EfficientViT (MSRA, CVPR'23, cascaded group attention): https://arxiv.org/html/2305.07027
- Idea: "a sandwich layout, i.e., using a single memory-bound MHSA between efficient FFN layers", plus "a cascaded group attention module feeding attention heads with different splits of the full feature". Motivation: "attention maps share high similarities across heads".
- Table 2 (300 epochs): M0 63.2 / 79M / 2.3M; M1 68.4 / 167M / 3.0M; M2 70.8 / 201M / 4.2M; M3 73.4 / 263M / 6.9M; M4 74.3 / 299M / 8.8M; M5 77.1 / 522M / 12.4M (top-1 / FLOPs / params).
- **CIFAR transfer (Table 4)**: "EfficientViT-M5 10621 77.1 98.0 86.4 89.7 97.1 92.0", which gives CIFAR-10 98.0 and **CIFAR-100 86.4** (ImageNet-pretrained, fine-tuned; resolution not captured, presumably 224, unverified).
- Weights: `timm/efficientvit_m0..m5.r224_in1k`.

### EfficientViT (MIT Han lab, ICCV'23): https://arxiv.org/html/2205.14756
- Idea: "substituting the inefficient softmax attention with lightweight ReLU linear attention", plus small-kernel convs to form multi-scale tokens.
- The fetched table only had B2+: "EfficientViT-B2 (r256) | 82.7 | 96.1 | 24M | 2.1G". B0/B1 numbers are **unverified**.
- Weights: `timm/efficientvit_b0.r224_in1k`, `efficientvit_b1.r224_in1k`, etc. CIFAR: none.

### SHViT (CVPR'24, single-head attention): https://arxiv.org/html/2401.16456
- Idea: a 16x16 large-stride patchify stem, and single-head attention on a partial channel slice (ratio "1 / 4.67"). Finding: "there is a lot of redundancy both at the feature and prediction levels" across heads.
- Table 2 (no-KD): S1 241M / 6.3M / 72.8; S2 366M / 11.4M / 75.2; S3 601M / 14.2M / 77.4; S4 (256px) 986M / 16.5M / 79.4. KD (DeiT recipe): S1 74.0 ... S4 80.2. S4 runs at 1.6 ms on iPhone 12.
- CIFAR: none. Weights: `timm/shvit_s1..s4.in1k`.

### MobileNetV4 (ECCV'24): https://arxiv.org/html/2404.10518
- Idea: the UIB block, plus "Mobile MQA", a multi-query attention with shared K/V and K/V downsampling: "achieves over a 39% inference speedup on mobile accelerators".
- Table 6: "MNv4-Conv-S 73.8 3.8 0.2 2.4 0.7 ..." (top-1, params M, GMACs, Pixel6 CPU ms, Pixel8 EdgeTPU ms); "MNv4-Hybrid-M 80.7 10.5 1.2 14.3 1.5"; "MNv4-Hybrid-L 83.4 35.9 7.2 87.6 3.8". The 87% headline uses KD from EfficientNet-L2 with JFT.
- Smallest hybrid is 10.5M, which is above the budget. CIFAR: none. Weights: `timm/mobilenetv4_conv_small.e2400_r224_in1k`, `mobilenetv4_hybrid_medium.e500_r224_in1k`, etc.

### StarNet (CVPR'24): https://arxiv.org/html/2403.19967
- Idea: the "star operation" (element-wise multiplication of two linear branches) acts as an implicit high-dimensional feature map. Pure CNN.
- Rows (top-1 | params M | FLOPs M | mobile ms): "StarNet-S1 | 73.5 | 2.9 | 425 | 0.7"; "S2 | 74.8 | 3.7 | 547 | 0.7"; "StarNet-100 | 64.3 | 1.04 | 187.1". No KD ("distillation" is listed as future work).
- CIFAR: none. Weights: `timm/starnet_s1..s4.in1k`.

### EMO (ICCV'23): https://arxiv.org/html/2301.01146
- Idea: the iRMB block, an inverted residual whose middle is windowed MHSA plus DWConv, in a single-block-type network.
- Table 7: EMO-1M 1.3M / 261M / 71.5; EMO-2M 2.3M / 439M / 75.1; EMO-5M 5.1M / 903M / 78.4. No KD ("without extra datasets and pre-trained models").
- CIFAR: none. Weights: GitHub checkpoints (resources/EMO_1M/net.pth etc., https://github.com/zhangzjn/EMO). No timm name verified.

### CAS-ViT (2024): https://arxiv.org/html/2408.03703
- Idea: "Convolutional Additive Token Mixer (CATM) employing underlying spatial and channel attention". It uses additive, not softmax, Q-K interaction.
- Table II: "CAS-ViT-XS 3.20 560 ... 78.3/78.7"; "CAS-ViT-S 5.76 932 ... 80.9/81.1" (params M, FLOPs M, top-1 300/450 epochs).
- **KD: yes**, RegNetY-16GF teacher (hard distillation).
- CIFAR: none. Weights: via the repo's MODEL_ZOO.md (individual links unverified).

### LowFormer (WACV'25): https://arxiv.org/html/2409.03460
- Idea: hardware-efficient macro design, with attention run "on half the resolution" (downsample, attend, upsample).
- Smallest: "LowFormer-B0 14.1 944 5988 0.30 224 78.4" (params M, MACs M, GPU throughput, GPU latency ms, res, top-1). Above budget. CIFAR: none.

### iFormer (ICLR'25): https://arxiv.org/html/2501.15369
- Idea: a ConvNeXt-like net streamlined for mobile. "In the earlier, high-resolution stages, we utilize fast convolution". Later stages use "single-head modulation self-attention (SHMA)".
- Table 3 (no KD, 300 epochs; iPhone 13 latency): T 2.9M / 0.53 GMACs / 0.60 ms / 74.1; S 6.5M / 1.09 / 0.85 / 78.8; M 8.9M / 1.64 / 1.10 / 80.4 (KD 81.1).
- CIFAR: none. Weights: `timm/iformer_t.in1k`, `iformer_s.in1k`, `iformer_m.in1k`, `iformer_m_distilled.in1k`, ...

### TinyViT (ECCV'22): https://arxiv.org/html/2207.10666
- Idea: hierarchical hybrid (MBConv early, windowed attention later), with fast pretraining distillation from big IN-21k teachers using stored sparse logits.
- TinyViT-5M: 5.4M params, 1.3G MACs. IN-1k from scratch 79.1; IN-21k distill then IN-1k 80.7. Rows: "TinyViT-5M (ours) 79.1 94.8 5.4 1.3 ..." / "80.7 95.6 5.4 1.3". The fetch tool assigned the rows by order, so treat which row is which as slightly uncertain.
- CIFAR: only TinyViT-21M linear-probe (CIFAR-100 86.6 with IN-21k KD). Not fine-tune, not 5M.
- Weights: `timm/tiny_vit_5m_224.in1k`, `tiny_vit_5m_224.dist_in22k`, `tiny_vit_5m_224.dist_in22k_ft_in1k`.

### LeViT (ICCV'21): https://github.com/facebookresearch/LeViT
- Idea: conv stem (4 conv layers, 16x downsample), shrinking attention stages, attention bias instead of positional embedding, Hardswish.
- LeViT-128S: "76.6", "305M", "7.8M". KD: "trained with distllation". Smallest is 7.8M, above budget.
- Weights: `timm/levit_128s.fb_dist_in1k`, `levit_conv_128s.fb_dist_in1k`.

### MobileMamba (CVPR'25): https://arxiv.org/html/2411.15941
- Idea: three-stage net whose MRFFI module splits channels among a Mamba global branch, multi-kernel DWConv, and identity.
- Smallest is T2: 255M FLOPs, 8.8M params, 192px, 73.6 (no-KD) / 76.9 († = KD + 1000 epochs). Above budget.
- CIFAR: none. Weights: only a third-party HF repo (`Lewandofski/MobileMamba`). No official timm name found.

### EfficientViM (CVPR'25, Mamba-lite): https://github.com/mlvlab/EfficientViM
- Idea: HSM-SSD, channel mixing inside compressed SSM hidden states.
- M1: 72.9 (300 epochs) / 73.5 (450 epochs), 6.7M, 239M FLOPs; KD 74.6. CIFAR: none.

### RepNeXt (2024): https://arxiv.org/abs/2406.16004
- Idea: multi-scale CNN with "serial and parallel structural reparameterization (SRP)".
- Only the abstract was fetched: M4 "matches RepViT-M1.5's 82.3% accuracy" and runs "within 1.5ms on an iPhone 12". Small-variant numbers **unverified**.

### RecNeXt / RecConv (2024-25): https://arxiv.org/html/2412.19628v3
- Idea: recursive small-kernel convs give an effective receptive field of k x 2^l at near-constant FLOPs. The A-series uses linear attention.
- Table 7: "M0 2.5 0.4 1.0 189 750 73.2 / 74.7*"; "A0 2.8 0.4 1.4 177 4891 73.6 / 75.0*" (params M, GFLOPs, NPU ms, CPU ms, throughput, top-1 no-KD / KD).
- CIFAR: none.

### LSNet (CVPR'25, THU-MIG): https://github.com/jameslahm/lsnet
- Idea: "See Large, Focus Small". LS convolution combines large-kernel DWConv perception with small-kernel dynamic aggregation.
- LSNet-T "74.9 / 76.1*", 11.4M, 0.3G (\* = KD). Above budget. Weights: `hf_hub:jameslahm/lsnet_t` (and `_distill`).

### TinyNeXt (ICCV'25): https://iccv.thecvf.com/virtual/2025/poster/2639 (via search summary; the PDF fetch was 403)
- Idea: "Lean Single-Head Self-Attention" plus stage-specific macro design for MCUs.
- "TinyNeXt-T achieves ... 71.5% top-1 accuracy with only 1.0M parameters". Other numbers **unverified**.

### Others found, numbers unverified
- LOLViT (2508.01385): Fast Window Attention with ReLU softmax approximation. Only "5x" speed vs MobileViT-X is given.
- CPUBone (2603.26425, 2026): grouped convs and smaller kernels for CPU. No numbers fetched.
- No 2026 paper was found with verified numbers beating iFormer/MNv4 at <=6M.

---

## 2. Small-data / CIFAR evidence (directly relevant to our 3M CIFAR ViT)

- **SL-ViT (SPT + LSA)**, https://ar5iv.labs.arxiv.org/html/2112.13492, Table 2 (from scratch, ~3M params, the closest analogue to our model):
  "ViT 8593 189.8 2.8 93.58 73.81 ..." vs "SL-ViT 7697 199.2 2.9 94.53 76.92 ..." (params 2.8M vs 2.9M; CIFAR-10 / CIFAR-100).
  Ablation (CIFAR-100 / Tiny-IN): "L-ViT 74.87 58.50" (LSA only, +1.06); "S-ViT 76.29 60.67" (SPT only, +2.48).
- **CCT** (conv tokenizer + sequence pooling), https://github.com/SHI-Labs/Compact-Transformers: CCT-7/3x1, no pretraining, CIFAR-100 "80.92%" (300 epochs), "82.72%" (1500 epochs), "82.87%" (5000 epochs). Abstract (https://arxiv.org/abs/2104.05704): "98% accuracy when training from scratch on CIFAR-10 with only 3.7M parameters".
- **Conv stem** (Xiao et al.), https://arxiv.org/abs/2106.14881: "dramatically increases optimization stability" and "improves peak performance (by ~1-2% top-1 accuracy on ImageNet-1k)" "while maintaining flops and runtime".
- **Pretrained to CIFAR-100 fine-tune:**
  - DeiT-Tiny (5.5-5.7M, pure ViT), https://arxiv.org/html/2404.00672: "Baseline [4] ... 98.07 86.78" (CIFAR-10 / CIFAR-100), fine-tuned from IN-1k; "32×32 are resized to 224×224".
  - FTerViT (https://arxiv.org/html/2605.21171) Table 6: "DeiT-Tiny (pretrained) | 5.5 | 22.9 | 97.52% | 86.54%" at 224x224.
  - EfficientViT-M5 (12.4M): CIFAR-100 86.4 (see above).
  - No verified CIFAR-100 fine-tune numbers were found for any <=6M edge hybrid (MobileViT, EfficientFormerV2-S0, FastViT-T8, TinyViT-5M, iFormer-T, StarNet). We would need to run these ourselves.
  - Weights: `timm/deit_tiny_patch16_224.fb_in1k`, `deit_tiny_distilled_patch16_224.fb_in1k`, `vit_tiny_patch16_224.augreg_in21k(_ft_in1k)`.

---

## 3. Transplant analysis for a ~3M-param ViT, 32x32 CIFAR-100, patch-4 (64 tokens), from scratch

Key observation: most edge-model tricks (MQA, single-head, cascaded group attention, linear/separable attention, stride attention, reparameterization, large-stride stems) **optimize latency and memory traffic at high token counts**. At 64 tokens, attention cost is negligible, so these give us speed we don't need, and they usually cost a little accuracy. The ideas that help on small data are the ones that **inject locality inductive bias**.

### Ranking A: from scratch, ~3M params, expected CIFAR-100 gain
1. **Conv / overlapping tokenizer stem** (CCT, LeViT, SPT, Xiao et al.). It replaces the non-overlapping 4x4 patchify with 2-3 small 3x3 convs (+BN/GELU) down to 8x8, or uses SPT. Evidence: SPT alone gives +2.5 on CIFAR-100 at 2.8M (73.81 to 76.29), and CCT reaches 80.9 at 300 epochs. Expected gain is the largest and the cost is ~zero params. *Status:* "ViT with conv stem". Most papers still call this a ViT (the CCT/LeViT convention), but strictly it is a mild hybrid. SPT (shift + concat + linear) keeps it **pure ViT**.
2. **DWConv inside the MLP** (local FFN; EfficientFormerV2 "unified FFN", EMO iRMB, also used in iFormer/EdgeNeXt). It adds a 3x3 DWConv on the 8x8 token grid between FC1 and FC2. The param cost is tiny (9 x hidden). It gives every block locality and acts as an implicit positional encoding. Gain magnitude on CIFAR is **unverified** from fetched sources (expectation: +1-3). *Status:* **hybrid**.
3. **Locality-biased attention** (LSA: learnable temperature + diagonal masking; EfficientFormerV2 DWConv-on-V and talking heads). LSA alone gives +1.06 on CIFAR-100 (73.81 to 74.87), and the gains stack with SPT (76.92). *Status:* LSA and talking heads are **pure ViT**. DWConv-on-V is hybrid.
4. **Conv token mixer in early blocks, attention later** (iFormer, FastViT RepMixer, MobileViT, EdgeNeXt). For example, the first 2 of ~7-9 blocks use DWConv mixers. This is the structure behind the best <=3M IN-1k nets (iFormer-T 74.1 no-KD at 2.9M), but with only 64 tokens and one resolution, the extra gain over items 1+2 is likely smaller. *Status:* **hybrid**.
5. **Train-time structural reparameterization** (MobileOne/RepViT/FastViT/RepNeXt). This means multi-branch BN'd convs (3x3 + 1x1 + identity) in the stem/DWConvs, merged after training. It is free at inference. MobileOne reports ~+0.5 for its smallest model on IN-1k, and it only applies where convs exist. *Status:* inherits from 1/2/4.

Not recommended for our setting:
- MQA / Mobile MQA, single-head (SHViT/SHMA), cascaded group attention: these save compute that is cheap at N=64. A possible exception is the SHViT/CGA observation of head redundancy, which suggests trying fewer heads (e.g. 2-4 at dim ~192) and re-spending the saved params. That is a cheap ablation, not a top idea.
- Linear/separable/ReLU attention (MobileViTv2, EfficientViT-MIT): approximations that only pay off at large N.
- Hierarchical stages / token downsampling (8x8 to 4x4): this reduces tokens further and would need patch-2 (256 tokens) to be useful. It is possibly useful only paired with a conv stem at higher token counts. Uncertain.

### Ranking B: pretrain then fine-tune on CIFAR-100, <=6M params
Ordering uses IN-1k accuracy and weight availability as a proxy for transfer, which is a hypothesis. Only DeiT-Ti has a verified CIFAR-100 fine-tune number. Caveat: these hierarchical nets downsample 32x, so CIFAR must be upsampled (typically to 224, as in the DeiT-Ti numbers). At native 32px their final feature map is 1x1, so expect a large drop.
1. **TinyViT-5M** (`tiny_vit_5m_224.dist_in22k(_ft_in1k)`; 5.4M; IN-1k 80.7 with IN-21k distill). Highest verified IN-1k accuracy under 6M with IN-21k-level pretraining. Hybrid.
2. **DeiT-Tiny / ViT-Ti augreg-in21k** (5.7M; CIFAR-100 86.5-86.8 @224 verified). This is the only **pure ViT** option, the only verified CIFAR-100 number, and has native patch tokens so it is easy to adapt to patch-4 @32 via pos-embed interpolation. The best choice if the model must stay "pure ViT".
3. **iFormer-T / iFormer-S** (`iformer_t.in1k` 2.9M/74.1; `iformer_s.in1k` 6.5M/78.8, slightly over budget). The best 2025 hybrid at ~3M. Hybrid.
4. **EfficientFormerV2-S0** (`efficientformerv2_s0.snap_dist_in1k`, 3.5M, 75.7 KD) / **FastViT-T8** (`fastvit_t8.apple_dist_in1k`, 3.6M, 77.2 KD). Hybrid.
5. **MobileViTv2-0.75** (2.9M, 75.6) / **MobileViT-XS** (2.3M, 74.8). Hybrid. Also CAS-ViT-XS (3.2M, 78.3 KD), but its weights are only on GitHub and the links are unverified.
