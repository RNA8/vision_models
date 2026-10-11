# Next experiments: combined ranking (2026-10-11)

Synthesised from `small_vit_pretraining.md`, `edge_kd_transfer.md` and `edge_efficiency_recurrent.md`.
Every quoted number and its source is in those files. The GPU-time figures are estimates.

## Track A: edge Pareto (pretrained, distilled; target is closing 88.3 -> ~93 at <=6M)
1. **Long, consistent function-matching KD** (Beyer et al. 2022). The teacher runs on the same augmented crop, for 200-300 ep, into RepViT-M0.9 and TinyViT-5M (21k-distilled weights). Our 30-ep KD sits at the bottom of their curve. ~2-3.5 h per model.
2. **KD loss swap**: DIST / logit-standardisation vs plain KL at 60 ep. Plain KD can hurt with a strong teacher (R18: 73.4 -> 72.6, DIST 74.5). ~2 h.
3. **Resolution 256-288 + FixRes test-res bump**. ~1 h.
4. **INT8 audit** (per-channel PTQ, short QAT) for RepViT/TinyViT/FastViT/EffFormerV2/MNv4. Naive PTQ drops EffFormerV2-S0 from 76.2 to 68.2. ~2 h, mostly CPU.
5. **Newer backbones**: RepNeXt-M1, CPUBone-B0, SHViT-S1, GhostNetV3, on the same pipeline. ~3-4 h.

## Track B: our own small ViT (pretrain -> transfer; 81.5 now, published ViT-Ti 84-87 @224)
1. **FlexiViT patch switch**: fine-tune the IN-64 patch-8 checkpoint at patch 4 (PI-resize), ~1-2 h.
2. **Longer IN-64 pretraining** (100 ep). A WRN of similar size gets ~60.5 vs our 56.8. ~6-8 h.
3. **CNN-teacher KD during IN-64 pretraining** (offline logits). ~3-4 h.

## Track C: recurrent depth (publishable; closest prior art bViT 2605.10661 has no injection and no exit study)
1. **Learned halting head** vs KL / entropy / calibrated exits vs the 87.8 oracle. ~10 h.
2. **Per-loop LoRA + concat injection + sandwich norm** at equal MACs. ~5 h.
3. **MoR-style per-token depth routing**. ~8-15 h.

Suggested first wave (fits in ~4 concurrent GPU slots, ~4-5 h wall): A1 (RepViT + TinyViT), A2, B1, C2; A4 on CPU in parallel.
