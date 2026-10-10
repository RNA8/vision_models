# Closing the gap to SOTA, and a publishable direction (researched 2026-10-10)

Sources were fetched by research agents. The novelty-critical papers (reViT, ViTRM, bViT, structured init) were re-checked for title, date and the quotes below.

## Part 1: why our ViT-Tiny is at 88%, and what closes the gap

**Main finding: the gap is mostly recipe plus model shape, not the ViT itself.**

| Lever | Evidence | Expected gain | Cost |
|---|---|---|---|
| **Recipe: batch 128, lr 3e-3, wd 0.05, warmup 10, AutoAugment/RandAugment, Repeated Augment ×3** | SL-ViT paper ([2112.13492](https://arxiv.org/abs/2112.13492)) Tab. 2: plain ViT 2.8M, **93.58%** in 100 ep at batch 128 (**correction:** their official code uses `RASampler(len_factor=3)`, so each "epoch" is 3 passes over the data and "100 ep" is ~117k steps; our 300 ep at batch 1024 is ~14.6k). DeiT: disabling repeated aug costs 81.8→76.5 on ImageNet | large | ~20 min / "100 ep" (×3 passes) |
| **Shape: depth 9, 12 heads (head dim 16), MLP ratio 2** (vs our 12 layers, 3 heads, MLP 4) | Gani et al. ([2210.07240](https://arxiv.org/abs/2210.07240)) Tab. 15, same recipe on C100: 12L/3h/mlp4 → 68.09 vs 9L/12h/mlp2 → 73.81 (+5.7). ViT-Lite-7/4: 7 layers, MLP ratio 2, 93.57% | +2–6 | free |
| **Structured / impulse init** | Zheng et al., "Structured Initialization for ViTs" ([2505.19985](https://arxiv.org/abs/2505.19985)), ViT-Tiny CIFAR-10 300 ep: default 92.29 / mimetic 93.50 / impulse 94.67. Mimetic ([2305.09828](https://arxiv.org/abs/2305.09828)): 88.43 → 90.47 at 100 ep | +1–2 | free |
| SeqPool instead of CLS | CCT ([2104.05704](https://arxiv.org/abs/2104.05704)) Tab. 1: ViT-Lite-7/4 93.57 → CVT-7/4 94.01 | +0.4 | free |
| LSA (learnable temperature + diagonal mask) | SL-ViT C100 ablation: 73.81 → 74.87 (L-ViT) | +1 | ~1% |
| Mixup off for the last ~40% of epochs | CCT official config: mixup_off_epoch 175 of 300 (not ablated in the paper) | ? | free |
| DeiT hard distillation from a CNN teacher | DeiT ([2012.12877](https://arxiv.org/abs/2012.12877)): CIFAR-10 DeiT-B 97.5 → 98.5 with distillation. Tiny ViT C100 at 224px: +8.2 ([2207.10026](https://arxiv.org/abs/2207.10026)). airbench96 is a 96% teacher in ~30 s | +1–3 | +30–50% / epoch |
| Conv tokenizer (hybrid) | CCT-7/3×2 95.04, CCT-7/3×1 96.53 (300 ep), 98.00 (5000 ep) | to 95–96.5 | 20–75 min |
| Self-supervised pretraining on CIFAR | Gani et al.: 2.8M ViT 93.58 → 96.41 (200 SSL + 100 sup. ep) | +3 | ~45 min |
| Muon / SAM / EMA / registers / QK-norm | No consistent small-ViT CIFAR gain found (e.g. Polar Express [2505.16932]: "do not show a consistent advantage") | low | — |

**Plan to reach ≥93% (pure ViT):**
- R1: SL-ViT recipe and shape, 100 ep with `--repeat-aug 3` (= ~117k steps), ~20 min. Run a twin with our 12/3/4 shape to isolate the shape effect.
- R2: R1 + impulse/mimetic init.
- R3: best of R1/R2 at 300 ep, with SeqPool + LSA and mixup off at 175.

**Then ≥95%:** R4 distillation from an airbench96 teacher, or R5/R6 CCT hybrids.

## Part 2: a publishable direction (recurrent depth)

### What already exists (verified)

- **reViT**, "One Block, Multiple Depths: Recurrent ViTs with Depth-Programmed Experts" ([2610.12448](https://arxiv.org/abs/2610.12448), **2026-10-08**). ImageNet only. It reuses one block and trains with elastic depth, but *"a global depth is selected at inference"*: there is no per-image depth.
- **ViTRM**, "Vision Tiny Recursion Model" ([2603.19503](https://arxiv.org/abs/2603.19503), 2026-03). Trained from scratch on CIFAR: **93.1% on C10** and 72.1% on C100 at ~3.6M params. Halting is decided per batch, and at inference it runs a fixed number of steps.
- **bViT** ([2605.10661](https://arxiv.org/abs/2605.10661), 2026-05). Single-block recurrence on ImageNet. Narrow recurrent models lose badly (bViT-S 68.1 vs ViT-S 78.2). No adaptive exit and no depth extrapolation.
- **RD-ViT** ([2605.03999](https://arxiv.org/abs/2605.03999)). The Geiping recurrent-depth design applied to cardiac MRI *segmentation*.
- **Other related work:**
  - LoopViT: ARC puzzles.
  - SR-ViT: soft mixture-of-recursions on ImageNet.
  - Older weight sharing: SReT, MiniViT.
  - Token halting: A-ViT.
- **Foundations:**
  - Geiping et al. recurrent depth ([2502.05171](https://arxiv.org/abs/2502.05171)).
  - Mixture-of-Recursions ([2507.10524](https://arxiv.org/abs/2507.10524)).
  - Relaxed Recursive Transformers ([2410.20672](https://arxiv.org/abs/2410.20672)).
- **Cautions:**
  - [2609.35748]: looped LMs gain accuracy faster with test-time compute but underperform at matched compute.
  - [2604.15259]: extrapolating to more loops needs recall (input re-injection) plus outer normalization.
  - [2502.10954]: overthinking, where extra iterations can make predictions worse.

### Ranked ideas

1. **Recurrent-depth ViT with test-time depth scaling and per-image adaptive exit, validated against image difficulty.** This is the novel combination. No fetched paper shows a classification ViT that keeps improving when run past its training depth *and* stops per image.
   - Architecture: a prelude, a shared core looped r times with the input re-injected each loop, and a coda. QK-norm and state normalization keep it stable.
   - Training: r is sampled randomly each step, with gradients through only the last k loops (truncated backprop).
   - Exit rule: parameter-free, based on how much the prediction changes between loops (KL or entropy).
   - Difficulty measures: CIFAR-10H human label uncertainty, C-scores, CIFAR-10-C severity, and OOD sets.
   - Figures:
     - accuracy vs effective FLOPs for r = 1…32;
     - accuracy beyond the training depth;
     - exit step vs difficulty, with Spearman correlation against CIFAR-10H;
     - FLOPs saved at equal accuracy.
   - Est. ~12–15 H100-hours (3 seeds, C10/C100, ablations).
   - Risks:
     - looped models may lose at matched compute;
     - narrow recurrent models are weak;
     - reViT is one step away from per-image depth, so the publication window is short.
2. **Which stabilizers make a vision looped model extrapolate in depth.** Candidates are recall plus outer normalization, an nGPT-style normalized core, per-loop LoRA and Muon. Can be a paper on its own or an analysis section of #1 (~6 GPU-h).
3. **Loop count as a free uncertainty / OOD signal.** Compare convergence speed against max-softmax probability for detecting CIFAR-10-C and SVHN. Reuses the #1 models (~1 GPU-h).
4. **Muon plus speedrun techniques for small ViTs.** No small-ViT CIFAR study exists, but the idea is incremental (~10 GPU-h).

**Recommendation:** first fix the baseline to ≥93% (Part 1, R1–R3), because reviewers will compare against ViTRM's 93.1%. Then pursue #1, with #2 and #3 as its analysis sections. Our fast pipeline (~3 s/epoch, every component a CLI flag) is well suited to the many short runs this needs.
