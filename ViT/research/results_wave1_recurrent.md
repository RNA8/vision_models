# Wave 1, experiment C2: recurrent-depth ViT ablations (CIFAR-100, 100 epochs)

All runs: `rd_tiny4` preset, r_mean 8, k_backprop 4, LSA + SeqPool, batch 256, lr 1e-3, RandAugment, seed 0, 1 seed each
(recipe identical to `runs/c100_rd_r8_add`; launch script `run_c100_c2.sh`, MACs `c2_macs.py` -> `c2_macs.json`).
Val = final-epoch top-1 on the 10k CIFAR-100 test set, evaluated at r = r_mean = 8. Params are the full model
(incl. SeqPool and CIFAR-100 head). MACs are analytic (matmuls only, norms/softmax ignored) per 32x32 image at r = 8.
Train minutes are wall-clock on a GPU shared with 2-4 other jobs, so they are NOT a speed measurement
(use the MACs column; e.g. baseline took 26.1 min in its first run and 24.1 min in the rerun).

| run | val | params (M) | MACs (M) | train min |
|---|---|---|---|---|
| c100_rd_flat6 (flat 6 blocks, existing) | 62.94 | 2.71 | 182.9 | 12.3 |
| c100_rd_c2_flat20 (flat 20 blocks, equal MACs) | 63.41 | 8.94 | 608.1 | 25.7 |
| c100_rd_r8_add (baseline, existing) | 65.28 | 2.71 | 608.1 | 26.1 |
| c100_rd_c2_base_rerun (baseline, new code) | 65.05 | 2.71 | 608.1 | 24.1 |
| c100_rd_c2_lora8 (add + per-iteration LoRA r=8) | 65.15 | 3.11 (+0.39, +14.5%) | 633.7 (+4.2%) | 29.0 |
| c100_rd_c2_lora32 (add + per-iteration LoRA r=32) | **65.38** | 4.28 (+1.57, +58%) | 710.4 (+16.8%) | 27.9 |
| c100_rd_c2_sandwich_add (add + sandwich norm) | 64.70 | 2.71 (+1.5K) | 608.1 | 24.6 |
| c100_rd_c2_concat (concat injection, no sandwich) | 60.24 | 2.79 (+0.07) | 646.5 (+6.3%) | 22.6 |
| c100_rd_c2_sandwich_concat (concat + sandwich) | 62.21 | 2.79 | 646.5 | 23.3 |

Noise: the same baseline config gave 65.28 and 65.05 (code path for the baseline is unchanged; test asserts zero-init LoRA
reproduces the tied model exactly). Treat differences below ~0.3-0.4 as noise; we have one seed per variant.

## Implementation
* `models/recurrent_vit.py`: `lora_rank` (LoRA on the core blocks' qkv, proj, fc1, fc2; one A/B pair per iteration t < r_mean = 8,
  later iterations reuse the last; A ~ N(0, 1/in), B = 0; scale 1) and `sandwich` (RMSNorm). Flags `--rd-lora-rank`, `--rd-sandwich`.
  The iteration index is passed as a (1,) device tensor so the compiled / CUDA-graph core step stays static.
  LoRA overhead per core block and iteration is rank x 3072 params (qkv 768, proj 384, fc1 960, fc2 960 per rank).
* Sandwich placement, from the Geiping et al. (arXiv 2502.05171, Sec. 3.2, fetched 2026-10-11) equations: n1 on the layer input before
  attention, attention output added to the residual and the sum normalised by n2; n3 before the MLP; MLP output added to the
  residual and the sum normalised by n4; RMSNorm throughout. Here n1, n3 are the existing pre-norm LayerNorms; n2, n4 are new
  RMSNorms (weight 1) in the two core blocks only (prelude/coda stay pre-norm), in addition to the per-loop outer norm that the baseline has.
* `rd_flat20` preset: 2 + 16 + 2 distinct blocks, no loop; the equal-MACs flat control.
* Tests: `test_recurrent_vit.py` (zero-init LoRA equals tied model with/without LSA, per-iteration LoRA gradients, param count
  formula, equal-MACs preset, sandwich+LoRA modes, compile_parts).

## Test-time depth (`eval_recurrent.py`, CIFAR-100, r = 1..32)
| | acc@8 | acc@32 | KL exit tau=5e-4 | largest tau with no loss | oracle | state delta at t=8 |
|---|---|---|---|---|---|---|
| base_rerun | 64.96 | 65.09 | 64.96 @ 5.67 iters | tau 5.6e-3: 64.98 @ 3.68 iters | 69.80 @ 10.5 | 0.0098 |
| lora32 | 65.35 | 65.23 | 65.34 @ 8.05 iters | tau 1e-3: 65.38 @ 6.93 iters | 70.19 @ 10.4 | 0.0553 |

LoRA32 slightly raises accuracy but the state stays "moving" longer, so the KL exit saves much less compute (6.9 vs 3.7 mean iterations
at iso-accuracy): the per-iteration adapters make iterations less redundant, which trades away the adaptive-exit benefit. The oracle
gap (~4.5 pts) is the same for both. (CIFAR-10-C/H analyses are CIFAR-10 only and were skipped.)

## Conclusions
1. **Per-iteration LoRA does not help at this scale.** r=8: 65.15, r=32: 65.38 vs 65.05-65.28 baseline, i.e. within noise, for +14.5% / +58% params
   and +4% / +17% MACs. Train accuracy did not rise either (72.9 / 72.8 vs 72.9-73.4). Unlike in Relaxed Recursive Transformers (uptraining
   a pretrained LM), we train from scratch where the tied block is not the bottleneck. Not recommended; r=32 also reduces exit savings.
2. **Sandwich norm does not help with add injection** (64.70, -0.35 vs the mean baseline of 65.17, borderline/noise) and costs no MACs.
3. **Concat injection is much worse than add in our regime** (60.24, -4.9), confirming the earlier CIFAR-10 finding (2.4-4.5 pts) and
   the gap is larger here. Sandwich norm recovers +2.0 of it (62.21), which supports the Geiping et al. claim that sandwich norm is what makes
   the concat recurrence train, but it still lands 3 pts below additive injection and below the flat 6-block control (62.94 vs 62.21),
   so concat should be dropped for 100-epoch CIFAR-scale training. (An untested guess: the zero/normal state init and the adapter's tiny
   init std 0.02 slow concat down at this short schedule.)
4. **Equal compute check.** The flat 20-block model with the same MACs as the r=8 loop (608M) gets 63.41, i.e. 1.8-2.0 pts below the recurrent model
   (65.05-65.28) while having 3.3x the parameters (8.9M vs 2.7M); the 6-block flat control (183M MACs) gets 62.94. Hence the looped model
   wins at equal MACs and equal params, and flat depth beyond 6 blocks buys only +0.5 here (overfitting-limited CIFAR-100). Caveat: one seed;
   the flat20 recipe is the recurrent recipe unchanged (not tuned for a 20-layer net).
5. Best variant by val is lora32 (65.38) but it is not distinguishable from the baseline; the recommended recurrent config stays
   additive injection, no sandwich, no LoRA (65.05-65.28 at 2.71M params, 608M MACs).
