# Pretraining a 3-6M-param ViT at low resolution: literature evidence (fetched 2026-10-11)

Rules: every number was returned by WebFetch/WebSearch this session. The fetch tool summarises pages, so "quote" = the tool's extracted sentence/table row (not guaranteed character-exact; check PDF before external citing). Items marked **not found** were searched for and no source was found. Rows marked (prior doc) come from sibling files in this folder, fetched in earlier sessions.

Our reference: 5.4M ViT-Tiny shape, scratch CIFAR-100 75.4-75.7; IN-64 sup. pretrain (30 ep) -> CIFAR-100@64px 81.2-81.5.

## 1. Self-supervised pretraining for tiny ViTs

| Technique | Source | Quoted number | Setting | Expected gain for us |
|---|---|---|---|---|
| MAE-lite recipe, ViT-Tiny (5.7M) | Wang et al., arXiv 2205.14443, https://ar5iv.labs.arxiv.org/html/2205.14443 | "MAE-Tiny reaches 78.0%" IN-1k vs DeiT-Tiny scratch 75.8; MAE-Tiny 400 ep "took about 23 hours ... 8xV100" | 224px, IN-1k, 400 ep | Pretraining cost is high; IN-1k top-1 gain small |
| Same paper, CIFAR-100 transfer (Table 4) | same | DeiT-Tiny (supervised init) 85.8, MoCov3-Tiny 83.9, **MAE-Tiny 78.9**, random init 42.7 | fine-tune at 224px | KEY: for tiny ViTs, MAE-pretrained init transfers WORSE than supervised (-6.9). Do not prefer plain MAE over supervised for CIFAR-100 |
| D-MAE (attention distillation from MAE-Base into Tiny) | same, Table 5 | CIFAR-100 85.0 (+6.1 over MAE-Tiny 78.9); Flowers 95.2 (+9.4); Aircraft 79.2 (+14.6); IN-1k 78.4 (+0.4) | MAE-Base teacher | Closes the gap to supervised on CIFAR-100; big on fine-grained. Needs a pretrained MAE-Base teacher at our resolution |
| Layer finding | same | "lower layers of the pre-trained models matter more than higher ones if sufficient downstream data is provided"; "higher layers matter in data-insufficient downstream tasks" | | Small downstream sets (Pets/Flowers) need good high-layer features, so SSL-only pretraining is risky there |
| DINO ViT-S/16 | Caron et al. 2104.14294, https://arxiv.org/html/2104.14294 | IN-1k linear 77.0 / k-NN 74.5; CIFAR-100 fine-tune supervised 89.5 vs DINO 90.5 | ViT-S (22M), 224px | +1.0 over supervised at ViT-S scale; no ViT-Ti/low-res evidence. Cost 2x+ multi-crop: not a priority |
| SSL on CIFAR itself (Gani et al.) | prior doc, arXiv 2210.07240 | CIFAR-100 73.81 -> 79.15 (+5.34), 2.8M ViT, 200 ep SSL + 100 ep FT | CIFAR only | Cheap, but ours (supervised IN-64 -> 81.3) already beats it |
| MAE on CIFAR (36px, patch 3) | prior doc | Mae-ViT-C100 78.27 (3.64M), ~3700 ep pretrain | CIFAR only | Too costly, below our 81.3 |
| SSL on ImageNet-32/64 for ViT -> CIFAR-100 | **not found** (two searches) | - | - | Open niche; no published target |

## 2. Supervised pretraining on downsampled ImageNet

| Technique | Source | Quoted number | Setting | Expected gain |
|---|---|---|---|---|
| WRN on ImageNet64x64 | Chrabaszcz et al. 1707.08819, https://arxiv.org/html/1707.08819 | WRN-36-2 6.2M params: top-1 error 39.55% (=60.45% acc); WRN-36-1 1.6M: 49.79% err; WRN-36-5 37.6M: 32.34% err | "up to a total budget of 40 epochs" | Our ViT 56.8% (5.4M, 30 ep) vs 60.45% for 6.2M WRN: CNN is ~3.6 pt better per param at 64px. Our ResNet-18 51.2 is weaker than their WRN-36-2 |
| WRN on ImageNet32x32 | same | WRN-28-2 1.6M: 56.92% err; WRN-28-5 9.5M: 45.36% err | 40 ep | 32px IN accuracy is ~15-20 pt lower than 64px for similar params: 64px is the right pretraining res |
| ImageNet-1k/21k supervised then CIFAR-100, Ti/16 (5.7M) | Steiner et al. 2106.10270, https://arxiv.org/html/2106.10270 | CIFAR-100 (Table 5): IN-1k 300 ep 84.0; IN-21k 30 ep 85.5; IN-21k 300 ep 87.2 | 224/384px FT | Full-res ceiling for a ViT-Ti: ~84-87. Our 81.3 at 64px px is within 3-6 pt |
| IN-1k pretrain, CIFAR-100 FT, DHVT-T (6.2M) | 2210.05958, https://arxiv.org/html/2210.05958 | "DHVT-T ... 76.5 IN-1k, 100 FT ep, 224 -> 86.73"; CeiT-T 88.4; DHVT-T scratch CIFAR-100 80.93 (patch 4, 6.0M), 83.54 (patch 2, 5.8M) | | Gives a scratch target: 83.5 for 6M at patch 2. CeiT-T (conv-ish) 88.4 at 224px is the ≤6M pretrained target |
| ViT on ImageNet-64 top-1 | **not found** | - | - | Our 56.8 vs ViT-S 57.9 is unreferenced; only WRN baselines exist |

## 3. Distillation

| Technique | Source | Quote | Gain |
|---|---|---|---|
| DeiT hard KD, convnet teacher | Touvron 2012.12877, https://arxiv.org/html/2012.12877 | DeiT-Ti 72.2 (5M) -> 74.5 (6M, distilled token); "hard distillation reaches 83.0% ... compared to the soft distillation accuracy of 81.8%" (DeiT-B); "a convnet teacher gives better performance than using a transformer" | +2.3 IN-1k at Ti scale. Downstream CIFAR-100 transfer gain only +0.5 (DeiT-B 90.8 -> 91.3) |
| TinyViT saved-logit distillation | Wu et al. 2207.10666, https://arxiv.org/html/2207.10666 | "The logits of large teacher models are sparsified and stored in disk in advance"; TinyViT-21M IN-1k 83.1 scratch vs 84.8 with IN-21k+distill (+1.7; 90 ep pretrain); linear probe CIFAR-100: IN-1k 75.2 / IN-1k+distill 74.5 / IN-21k 84.7 / IN-21k+distill 86.6 | Gain is +1.9 on CIFAR-100 for IN-21k data, negative with IN-1k. No 5M ablation numbers in paper (fetch tool said none given) |
| D-MAE | see section 1 | +6.1 CIFAR-100 over MAE | applies only to SSL route |

Implication: KD gains for a ~5M ViT are ~+2 on in-domain IN accuracy; transfer gain unclear (+0.5 to +1.9). Saved-logit trick makes it cheap (teacher forward once, offline).

## 4. Resolution / patch / pos-embed

| Technique | Source | Quote | Implication |
|---|---|---|---|
| FixRes: fine-tune at test res | Touvron 1906.06423, https://arxiv.org/html/1906.06423 | ResNet-50 train 128 / test 224: 75.7 no-adapt -> 77.1 with FT; 160/224: 77.3 -> 77.6; 224/384: 77.7 -> 78.9 | Short FT at the target res gives +0.3-1.4 (CNN). For us: pretrain at 64px then FT at target 64 (already matched); if downstream is 32px upsample consistently |
| FlexiViT | Beyer et al. 2212.08013, https://arxiv.org/html/2212.08013 | "the patch size provides a simple and effective lever to change the compute and predictive performance"; "standard ViT models perform well only at the patch size that they have been trained at"; 81.8% at 8x8 grid -> 85.3% at finer grid "at no additional training cost"; PI-resize patch embeddings, bilinear resample of pos-embed | Pretrain once at patch 8 on 64px (64 tokens), then FT at patch 4 (256 tokens) with PI-resize: ~+3.5 on IN-1k in their case. Direct candidate (Exp 1) |
| Patch size from scratch CIFAR-100 | 2210.05958 above | DHVT-T 80.93 (p4) -> 83.54 (p2) | +2.6 for 4x tokens (~3.5x FLOPs); finer patches help at low-res |
| Higher FT res for pretrained ViT | Steiner; Touvron | fine-tune at 224 and 384 both used; DeiT-B CIFAR-100 90.8 (224) vs 90.8 (384) | No gain from res beyond 224 for CIFAR; do not chase it |

## 5. Targets: <=6M transformer on CIFAR-100 with external pretraining

| Model | Params | CIFAR-100 | Source / setting |
|---|---|---|---|
| ViT-Ti/16 IN-21k 300 ep | ~5.7M | 87.2 | Steiner, 224px |
| ViT-Ti/16 IN-1k 300 ep | ~5.7M | 84.0 | Steiner |
| DeiT-Tiny supervised init | 5.7M | 85.8 | Wang et al. Table 4 |
| CeiT-T | ~6M | 88.4 | 2210.05958 Table 7 (224px, 100 FT ep) |
| DHVT-T | 6.2M | 86.73 | same |
| MAE-Tiny / D-MAE-Tiny | 5.7M | 78.9 / 85.0 | Wang et al. |
| Ours (64px, IN-64 pretrain) | 5.4M | 81.2-81.5 | project |
All published pretrained numbers use 224px+ pretraining and finetune (far more compute). **No <=6M low-res (32/64px) pretrained ViT CIFAR-100 result was found.** Target for the project: 84-87 (ViT-Ti full-res), i.e. a +3 to +6 gap from our 81.3.

## Ranked experiments (1x H100; baseline = ~1.5-2 h per 30-ep IN-64 run when shared)

1. **Longer/stronger supervised IN-64 pretrain** (e.g. 100 ep, light aug, label smoothing; maybe patch 4 at 64px=256 tokens). Evidence: WRN gets 60.45% at 40 ep with 6.2M; Steiner shows IN-1k 300 ep->IN-21k both help (+1.2 to +3.2). Est. 6-8 h for 100 ep patch 8; patch 4 pretrain ~4x FLOPs (~20+ h). Expect IN-64 +3-5, CIFAR-100 +1-2.
2. **FlexiViT-style patch-size switch**: pretrain at patch 8, fine-tune at patch 4 on CIFAR-100@64px with PI-resize patch embedding and bilinear pos-embed resample; or randomise patch size during pretrain. Reuse existing checkpoint: ~1-2 h per FT run. Evidence +3.5 IN-1k (FlexiViT), +2.6 scratch CIFAR-100 (DHVT p4->p2). Highest value per hour.
3. **Offline-logit KD from a CNN teacher during IN-64 pretraining** (DeiT hard KD / TinyViT saved logits; teacher e.g. a WRN-36-2 at 64px, ~60% acc, or a stronger IN-1k model on 64px inputs). Teacher logit pass ~1-2 h once, student 30 ep ~1.5-2 h. Evidence: DeiT-Ti +2.3 IN-1k; transfer gain uncertain (+0.5 to +1.9).
4. **Pretrain on larger/cleaner data at 64px**: IN-21k downsampled subset (evidence: TinyViT linear-probe CIFAR-100 75.2 -> 84.7 going IN-1k -> IN-21k at 21M; Steiner +1.5 Ti). Needs data; ~10-20 h for ~10 ep. Highest ceiling, highest cost, data availability unchecked.
5. **D-MAE-style distillation or SSL (MAE/DINO) on IN-64**: evidence says plain MAE for a tiny ViT is worse than supervised (78.9 vs 85.8) and needs distillation to catch up (85.0); gain only on fine-grained (Flowers +9.4, Aircraft +14.6 over MAE). Only worth it after 1-3, and for Pets/Flowers. ~8-12 h incl. teacher.

Caveats: IN-1k+distill linear-probe CIFAR-100 was lower than plain (74.5 vs 75.2) in TinyViT, so KD benefits on transfer are not guaranteed. GPU-time estimates are extrapolations from our 1.5-2 h/30 ep run, not from sources.
