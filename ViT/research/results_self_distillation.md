# Results: teacher distillation and self-distillation for the 2.75M pure ViT on CIFAR-100

Plan and literature: `research/self_distillation.md` (E1 = ensemble function matching, E4 = EMA self-KD).
Queues: `run_wave10_selfkd.sh` (300-ep screens), `run_wave10b_kd1000.sh` (1000-ep confirms).

**Status (2026-10-14 06:40 UTC):** all 300-ep screens are done, and the 1000-ep FM seed 0 is done (**82.88**).
1000-ep seed 1 is running.

## Setup

- **Student:** SL-ViT shape + SPT + LSA + SeqPool + impulse init, 2.75M params, bf3b recipe (batch 128, AdamW lr 5e-4,
  wd 0.05, 30 warmup, RandAugment m9, LS 0.1, drop-path 0.1, random erasing 0.25, MixUp 0.8 / CutMix 1.0).
- **Teacher (E1):** the 3 seeds `c100_spt_slvit_1000ep{,_s1,_s2}`. Ensemble = mean of the members' softmax
  probabilities (at tau). Its test accuracy, measured by the run itself in bf16, is **83.27** (the plan measured 83.23
  in fp32). The teacher costs 3 x 2.75M = 8.26M params **at train time only**. The deployed student is still the
  same 2.75M pure ViT: no conv, no extra data, no extra inference cost.
- **Consistent views:** the teacher runs on exactly the student's input tensor, after crop/flip, RandAugment,
  normalization, random erasing and MixUp/CutMix.
- **KD loss:** `finetune.kd_loss` (soft = KL(p_t || p_s) * tau^2, tau = 1). Total loss =
  (1 - a) * CE(mixed label-smoothed target) + a * KD.
- **Mix schedule:** the E1 arms keep MixUp/CutMix on for the whole run (`--mix-off-epoch 0`). The bf3b baseline
  switches mixing off at 58.3% of epochs. The `c100_mixon_300ep` control isolates that change.

## 300-epoch screens (CIFAR-100 test top-1, final epoch)

| Run | Teacher | a | Mix | Final | Best | Clean train | Delta vs baseline | s/epoch* |
|---|---|---|---|---|---|---|---|---|
| `c100_spt_slvit_300ep` (baseline) | none | - | off at 175 | 76.70 | 76.74 | 98.51 | - | 10.4 |
| `c100_mixon_300ep` (control) | none | - | on all run | 77.06 | 77.06 | 92.13 | +0.36 | 9.9 |
| `c100_kd_ens3_soft_a05_300ep` (E1-A) | ens3 | 0.5 | on all run | **80.28** | 80.48 | 96.29 | **+3.58** | 20.0 |
| `c100_kd_ens3_fm_300ep` (E1-B, function matching) | ens3 | 1.0 (labels off) | on all run | **80.55** | 80.55 | 97.29 | **+3.85** | 20.0 |
| `c100_selfkd_ema_300ep` (E4) | EMA of student, decay 0.999 | 0 -> 0.8 (PS-KD ramp) | off at 175 | 77.13 (EMA weights 77.31) | 77.31 | 96.92 | +0.43 (EMA +0.61) | 13.2 |

\*Steady-state s/epoch while sharing the H100 with 3 other jobs, so it is noisy. Use the same-time measurement below
for overhead.

Learning curves (test acc at epoch 50 / 100 / 150 / 200 / 250 / 300):
- baseline: 45.4 / 61.7 / 68.9 / 74.6 / 76.3 / 76.7
- E1-A (a = 0.5): 51.4 / 67.2 / 73.9 / 77.8 / 79.8 / 80.3
- E1-B (FM): 53.6 / 69.3 / 75.5 / 78.4 / 80.3 / 80.6
- mix-on control: identical to the baseline through epoch 170 (same seed and same recipe up to mix-off), then
  73.1 / 76.2 / 77.1 at 200 / 250 / 300
- E4 self-KD student (EMA weights): 44.3 (45.5) / 62.1 (63.9) / 69.4 (71.2) / 75.3 (75.9) / 76.8 (77.0) / 77.1 (77.3)

Teacher-student top-1 agreement on the training batch (mixed views) at the end: 0.68 (a = 0.5) and 0.71 (FM).

## Step-time overhead (measured back to back, same GPU load, 4-epoch smoke runs, 2026-10-13)

| Config | s/epoch | vs plain |
|---|---|---|
| plain, `--cudagraphs` | 12.9 | 1.00x |
| ens3 teacher, `--cudagraphs` (teacher fn compiled reduce-overhead) | 23.8 | 1.85x |
| ens3 teacher, no CUDA graphs (default compile) | 32.2 | 2.50x |
| EMA self-KD, `--cudagraphs` | 19.6 | 1.52x |

CUDA graphs stay on with a teacher: they are 26% faster than without. Three teacher forwards cost about one student
forward + backward, which matches about 1.85x.

## Honest reading

- **KD from the ensemble is a large, real effect.** At 300 ep both arms gain **+3.6 to +3.9** over the baseline and
  **+3.2 to +3.5** over the mix-on control (77.06). Seed noise is about 0.3 to 0.6, so this is about 6 to 10 sigma.
  A 300-ep KD student (80.3 to 80.6) nearly matches the 1000-ep baseline (80.91 ± 0.26).
- **The mix schedule is not the source of the gain.** Keeping MixUp/CutMix on all run without a teacher gives 77.06 vs
  76.70, which is within noise.
- **FM (labels off) vs a = 0.5:** +0.27, within seed noise. FM was chosen for the 1000-ep confirm because it is the
  simpler, published recipe (Beyer et al. 2106.05237) and it led at every eval epoch.
- **E4, EMA self-KD:** 77.13 vs 76.70, +0.43, borderline within noise. The EMA weights themselves score 77.31 (+0.61);
  that is mostly a weight-averaging effect, not distillation. This is below the plan's +0.5 student bar and far below
  E1, so it was **not** promoted to 1000 ep.
- **Compute caveat:** a KD epoch costs about 1.85x (teacher forwards), so a 300-ep KD run is about the compute of a
  ~550-ep baseline run. The teachers cost 3 x 1000 epochs, which is sunk: they were trained anyway as seed repeats.
  Even per unit of student compute, KD beats the baseline curve (76.7 at 300 ep, 80.9 at 1000 ep).
- **Deployment:** the teacher ensemble (3 x 2.75M = 8.26M params) exists only at train time. The deployed model is
  the same 2.75M pure ViT (no conv, no extra data) with unchanged inference cost.

## 1000-epoch confirm (`run_wave10b_kd1000.sh`, FM arm)

| Run | Final | Best | Clean train | s/epoch | Minutes |
|---|---|---|---|---|---|
| `c100_kd_ens3_fm_1000ep` (seed 0) | **82.88** | 82.99 | 99.82 | 19.5 | 328 |
| `c100_kd_ens3_fm_1000ep_s1` (seed 1) | running | | | | |
| baseline 3-seed mean (`c100_spt_slvit_1000ep{,_s1,_s2}`) | 80.91 ± 0.26 | | | 10.4 | |
| teacher: 3-seed ensemble (prob avg) | 83.27 | | | | |
| CCT-7/3x1 (arXiv 2104.05704, conv tokenizer) | 82.72 (1500 ep) / 82.87 (5000 ep) | | | | |

Seed 0 test accuracy along the run (KD vs baseline seed 0): 640 ep 81.38 vs 78.93, 800 ep 82.25 vs 79.55,
1000 ep 82.88 vs 80.66. The student ends 0.4 below its teacher ensemble, consistent with function matching closing
most of the gap on long schedules.

## Not run

- **E2 (CIFAR-only ResNet-18 teacher + DeiT hard KD):** skipped. `train_vit.py` only has AdamW (`make_optimizer`),
  and the plan's CNN recipe needs SGD 0.1 + nesterov. `--kd hard` is implemented and smoke-tested, so the hard-KD
  screen only needs a properly trained CNN teacher.
