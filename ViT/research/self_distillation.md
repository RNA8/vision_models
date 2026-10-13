# Can (self-)distillation close the pure-ViT vs CCT gap on CIFAR-100? (researched 2026-10-13)

Question: can our best pure ViT on CIFAR-100 catch the CCT conv-tokenizer hybrid using self-distillation or another
CIFAR-only teacher method? Constraints: no extra data and no convolutions in the student.

- **Our student:** 2.75M SL-ViT with SPT, LSA, SeqPool and impulse init, trained with the bf3b recipe.
  It scores 76.70 at 300 ep, 80.91 ± 0.26 at 1000 ep (3 seeds) and 80.98 at 2000 ep.
- **Target:** CCT-7/3x1 (arXiv 2104.05704) scores 80.92 at 300 ep, 82.72 at 1500 ep and 82.87 at 5000 ep.
- **Gap:** about 1.8 to 1.9 points at long schedules.

Rules:
- Every literature number below comes from a page or PDF fetched this session (2026-10-13). Function-matching
  numbers come from `research/edge_kd_transfer.md` (fetched 2026-10-11).
- WebFetch summarizes pages with a small model. Tables taken from PDFs were extracted locally with pypdf and read directly.
- **unverified** = my estimate, not a published number.

## 0. TL;DR

1. **We already have a pure-ViT teacher that beats CCT.** I evaluated the saved checkpoints on CPU (test set,
   fp32, no TTA; script described in §5):

   | Ensemble (prob average) | CIFAR-100 test top-1 |
   |---|---|
   | single models (s0 / s1 / s2 / 2000ep) | 80.64 / 81.21 / 80.88 / 80.91 |
   | 2 models (range over pairs) | 82.45 to 82.89 |
   | 3 seeds at 1000 ep | **83.23** |
   | 3 seeds at 1000 ep + 2000-ep run | **83.57** |

   The 4-model ViT ensemble (11M params total) beats CCT's 82.87 at 5000 ep. Distilling it into one 2.75M student
   ("born-again ensemble" or function matching) keeps both teacher and student pure ViT. Nothing touches extra data or
   convolutions. This is the cleanest experiment and ranks first.
2. **Distillation helps most when the student is weak.** Big ViT-on-small-data KD gains (+5 to +13 points) come from weak
   students at 65 to 77%. Examples: locality guidance, DeiT-LT, Co-advise. Our student is already at 80.9 and above
   every published pure-ViT CIFAR-100 number under 6M at patch 4. Realistic KD gains for us are about +0.5 to +1.5
   (**unverified**), which is enough to reach 81.5 to 82.5 but not obviously enough to beat 82.7.
3. **Self-distillation without a separate teacher gives CNNs +0.6 to +3 points on CIFAR-100.** This covers PS-KD, DLB,
   BYOT and Snapshot Distillation. None of these papers reports a ViT result on CIFAR-100. BYOT and path-ensemble
   self-distillation need auxiliary heads or extra paths.
4. **DINO-style self-supervised pretext (Gani et al.) is not worth it for us.** Their pure ViT reaches 79.15 after 200 SSL
   + 100 supervised epochs. Our plain 1000-ep supervised run already gets 80.91.
5. **A non-KD finding that matters more than any KD method:** a pure ViT-T/2 (5.6M params, patch 2, 256 tokens) trained
   from scratch on CIFAR-100 alone for 2400 ep reaches **83.6**. A ViT-T/1 on pixels reaches 85.1 (arXiv 2406.09415).
   Both beat CCT's 82.87 with no conv and no extra data, but at about 4x or more our per-image compute. The 82.7 gap
   may be mostly a token-count gap, not missing knowledge. Run a patch-2 control (E5) alongside the KD runs.
6. **The existing ResNet-18 teacher is too weak to use.** `runs/c100_r18_full` scored 73.22 after 100 ep with the ViT
   recipe (AdamW 1e-3, MixUp/CutMix, clean train accuracy 84.0, so underfit). BAN Table 3 shows that a teacher clearly
   weaker than the student hurts it (WRN-28-2 teacher at 74.7 → DenseNet student 81.5, against an 82.3 self-taught
   baseline). Any CNN teacher must be retrained to about 80% or more first.

## 1. Evidence table

| Method | Paper (fetched) | Setup | Reported gain | Pure-ViT student? No extra data? | Relevance to us |
|---|---|---|---|---|---|
| Born-Again Networks | arXiv 1805.04770, Tables 2–3 (PDF) | Same-arch teacher→student, CIFAR-100 | WRN-28-10: 19.08→18.25 err (+0.83). DenseNet-80-80: 17.16→16.36 (+0.80). Ensemble of BAN generations 15.14. **Cross-arch: WRN-28-10 teacher (80.92) → DenseNet-90-60 student 83.21 vs DenseNet-90-60 self-BAN 83.31 / baseline 82.31.** Weak teacher WRN-28-2 (74.68) → DenseNet-90-60 81.51, below its 82.31 baseline | Yes and yes (self-teacher) | Same-size teacher gives about +0.8. The teacher must be about as strong as the student. |
| Ensemble → single (born-again ensemble) | BAN "Ens*2/Ens*3" plus our own ensemble eval | 3–4 ViT seeds | Our ensemble 83.23–83.57 (measured). No published ViT distillation-from-own-ensemble number found | Yes and yes | **Best teacher available (rank 1).** |
| Function matching (patient + consistent) | arXiv 2106.05237, via `research/edge_kd_transfer.md` | Teacher and student see the **same** mixup/aggressive-aug crop, long schedules | "all inconsistent distillation settings plateau at a lower score". Student nearly matches the teacher only after 1k–10k epochs (Pets: 82.75 / 88.01 / 90.98 at 300 / 1k / 10k ep, teacher 91.03). ImageNet FM 80.30 (300 ep) → 82.31 (9600 ep) | Student any; teachers were BiT (extra data) in the paper | Gives the **recipe**: consistent views, mixup kept on, long schedule. Teacher quality is our own choice. |
| PS-KD (progressive self-KD) | arXiv 2006.12000 | Teacher = own model from epoch t−1. α_t = α_T·t/T, α_T = 0.8. CNNs, CIFAR-100 | Top-1 err: PreAct-R18 24.18→20.82 (+3.4). R101 20.75→19.43. DenseNet-121 20.05→18.73. PyramidNet+CutMix+ShakeDrop 14.07→13.59 (+0.5). "ViT results: none" | Yes and yes | Cheap. Gain shrinks with strong augmentation: +0.5 on top of CutMix/ShakeDrop vs +1.3 on plain PyramidNet. |
| DLB (self-distillation from last mini-batch) | arXiv 2203.16172, Table 2 | Half of each batch is repeated next step and the step-t−1 predictions are distilled. τ = 3, α = 1. CNNs | CIFAR-100 err: WRN20-8 22.42→20.79. ResNet-110 23.64→21.82. VGG-16 26.37→23.88 ("0.83% to 2.50%"). PS-KD on the same table: 21.26 / 22.85 / 25.95 | Yes and yes | Cheapest. Teacher signal comes from a different random augmentation of the same image (inconsistent view). |
| Snapshot Distillation | arXiv 1812.00123, Table 3 (PDF) | 4 cyclic-LR mini-generations; the previous snapshot teaches with T = 2 or 3 | DenseNet190 best-epoch err 16.69 (BL) → 16.06 (SD, +0.63). Plain-aug CNNs | Yes and yes | Small gain. Cyclic LR clashes with our cosine plus mix-off schedule. |
| Be Your Own Teacher (deep supervision) | arXiv 1905.08094, Table 1 | Shallow auxiliary classifiers learn from the deepest one | CIFAR-100 top-1: ResNet18 77.09→78.64. WRN20-8 79.76→80.92. ResNeXt29-8 81.29→81.51 (+0.22). PyramidNet101-240 81.12→82.30. Average "+2.65%" | Yes (aux heads dropped at test) | Needs aux SeqPool heads after blocks 3/6. Gains shrink for strong baselines. |
| Path-ensemble self-distillation | arXiv 2308.06548 | ViT seen as an ensemble of paths; deeper path teaches shallower (Δ = 2) | ImageNet: DeiT-S 79.8→80.2 (logit) →81.0 (logit + hidden + path). DeiT-B 81.8→82.7. No CIFAR-100 self-distillation result | Yes and yes | Only evidence of self-distillation working for ViTs; ImageNet only, +0.4 to +1.2. |
| DeiT hard-label distillation with a convnet teacher | arXiv 2012.12877, Tables 2–3 (ImageNet) | DeiT-B student, RegNetY teachers | No distillation 81.8. Soft 81.8. **Hard 83.0**. Dist token 83.1. Class + dist 83.4. A RegNetY-4GF teacher at 80.0 still gives a student at 82.7. "Hard distillation significantly outperforms soft distillation for transformers". "using a convnet teacher gives better performance than using a transformer" | Student pure; **teacher is a CNN (training aid only)** | Weaker teacher than student still helps through inductive bias. ImageNet only. No CIFAR-only evidence in DeiT. |
| Co-advise / CivT | arXiv 2106.12378 (PDF) | Small conv and involution teachers (RegNetY-600M 73.5, RedNet-26 73.0) teach DeiT-Ti on ImageNet | DeiT-Ti 72.2, DeiT-Ti-KD 74.5, CivT-Ti 74.9. "the accuracy of the teacher model is not the most important factor ... when the accuracy of teacher model is sufficiently large, improving teacher accuracy won't result in the improvement of student". Teacher *type* matters more | Student pure; CNN/INN teacher | Supports a diverse-architecture teacher (CNN) over a stronger same-type teacher. ImageNet only. |
| Locality guidance (feature imitation of a CNN) | arXiv 2207.10026, Tables 1–2 | ResNet-56 teacher (0.86M, **70.43** on CIFAR-100, 32 px). VT at 224 px, 300 ep. L = L_cls + β·‖proj(VT tokens) − CNN stage features‖² over 3 stages. β = 2.5 best for DeiT-Ti | CIFAR-100: DeiT-Ti 65.08→78.15. T2T-7 69.37→78.35. PiT-Ti 73.58→78.48. ConViT-Ti 75.32→78.95. **PVTv2-B0 77.44→79.30 (+1.86)**. Student ends **above** its 70.4 teacher | Student pure; CNN teacher | Gain shrinks fast as baseline locality improves (+13 → +1.9). Our SPT+LSA+impulse student already has strong locality. Expected small. |
| Logit KD, ResNet-56 → ViTs on CIFAR-100 | CIST arXiv 2605.20357, Table 7 (PDF) | Same protocol as locality guidance. Teacher ResNet-56 | DeiT-Ti 65.08 → KD 73.25 → KD + CIST 77.04. PiT-Ti 73.58 → 75.47 → 77.83. T2T-7 69.37 → 74.15 → 74.53 | Student pure; CNN teacher | Logit KD from a weak CNN still lifts weak ViTs by 2 to 8 points. Students stay below 78. |
| DeiT-LT (OOD distillation) | arXiv 2404.02900 | DeiT-B from scratch on CIFAR-LT, 1200 ep. ResNet-32 teacher **fed the strongly augmented student view** | C100-LT ρ = 100: DeiT 31.3 → +OOD distill 48.9 → +DRW 54.5 → +SAM teacher 55.6 | Student pure; CNN teacher | Long-tailed only. Shows that distilling on the student's strongly augmented view, where a weak teacher is "OOD", is the key ingredient. |
| Bootstrapping ViTs (agent CNN) | arXiv 2112.03552, repo README (via search, not verified against the paper) | Agent CNN built from the ViT's own weights, trained jointly | CIFAR-100 ViT-B: 53.07 scratch → 77.83 | Student pure at inference | Weak baseline. Below our 80.9. |
| DINO-style SSL pretext then supervised | Gani et al. arXiv 2210.07240, Table 3 | EMA teacher with 2 global (32 px) + 8 local (16 px) views. 200 SSL + 100 supervised ep. 4×V100 | ViT 2.8M: 73.81 scratch → **79.15**. SL-ViT 2.9M row 76.92. Swin 7.1M 80.95. CaiT 7.7M 80.79 | Yes and yes | Their gain is measured against 100-ep baselines. Our supervised 1000-ep run already beats 79.15. **Not recommended.** |
| MAE on CIFAR only | arXiv 2402.03752, Table 3 | Mae-ViT 3.64M, 36 px, patch 3, ~3700 ep MAE + 300 ep fine-tune | CIFAR-100 78.27 | Yes and yes | Below our 80.9. Not recommended. |
| **Small patches, no KD (control)** | **arXiv 2406.09415, Table 3a, App. A** | ViT-T (12 layers, d 192, 5.6M) at 32 px, AdamW lr 4e-3, bs 1024, wd 0.3, drop-path 0.1, 2400 ep, CIFAR-100 only | **ViT-T/2 83.6. ViT-T/1 85.1. ViT-S/2 83.7. ViT-S/1 86.4.** With CIFAR-only MAE pretraining: ViT-T/2 85.7, T/1 86.0 | **Yes and yes; above CCT** | 256 tokens vs our 64 means about 4x or more per-image FLOPs, so the edge budget is the real cost. |
| Conv-hybrid references (not eligible) | MSCViT arXiv 2501.06040 | Conv stem + depth-wise convs | MSCViT-T 3.8M 80.11. XS 7.8M 83.44 (300 ep) | No (convs) | Context only. |

Pure ViTs above 81% on CIFAR-100, under about 6M params, with no extra data: the only ones I found are ViT-T/2 (83.6)
and ViT-T/1 (85.1) from arXiv 2406.09415. Both use 2400 epochs and 256 or 1024 tokens. At patch 4 (64 tokens), our
80.91 / 81.17 (seed 1) is the best number I know of. Searches for KD- or self-distillation-based pure-ViT results
above 81% on balanced CIFAR-100 found none.

### Is a CNN teacher allowed under "pure ViT"?
The student stays a pure ViT at inference: no conv weights, no conv ops. A CNN teacher is used only during training,
and DeiT, Co-advise and locality guidance all report this as a pure transformer. It does inject conv inductive bias
through the loss, though, so a "pure-ViT" claim from it is weaker than one from a ViT-only teacher. **Flag it in any
write-up.** The born-again-ensemble plan (E1) avoids the issue entirely. CCT itself as a teacher (conv tokenizer) falls
in the same flagged category.

## 2. Diagnostics on our own teacher (CPU, measured 2026-10-13)

I ran the 3-seed 1000-ep ensemble on 4,000 random **train** images:

| View | Ens top-1 vs label | Mean max-prob | Entropy (nats) | Notes |
|---|---|---|---|---|
| clean | 99.9 | 0.916 | 0.664 | Basically the label-smoothed label; little dark knowledge |
| crop + RandAug m9 + erasing 0.25 | 96.0 | 0.855 | 0.881 | |
| + MixUp/CutMix | 50.8 (vs. dominant label) | 0.698 | 1.502 | Only 0.657 of the teacher's mass falls on the two mixed classes, so the teacher disagrees with the mix target by about 34% |

Implications:
- Because the teachers memorized the train set, KD on clean or lightly augmented views is nearly label smoothing.
  The signal is in the aggressive or mixed views.
- So the teacher must be evaluated **on exactly the student's augmented and mixed batch** (consistent views, as in
  Beyer and DeiT-LT), and MixUp/CutMix should probably **stay on for the whole run** instead of switching off at 58%.
  In function matching, the mixed image's target comes from the teacher, so the label-mixing artifacts that motivated
  mix-off no longer apply.
- Optional upgrade (**unverified** benefit): teachers that never saw the image. Cross-fitted teachers (for example,
  K = 2 halves) give genuinely soft clean-view targets but cost 2 more teacher trainings.
  `runs/c100_spt_slvit_1000ep_holdout` (held-out split) is one such model, but it is a single fold.

## 3. Ranked experiment plan

All runs:
- **Student:** the current `bf3b` slvit config with `--spt --lsa --pool seqpool --init impulse` on CIFAR-100.
- **Screen:** run each idea at 300 ep first (baseline 76.70; KD gains are usually larger at short schedules), then
  confirm the best one or two at 1000 ep (baseline 80.91 ± 0.26). Seed noise is ±0.26, so a 1000-ep claim of
  "beats 81.5" needs 2 seeds.
- **Cost basis:** 10 s/epoch for the student while sharing the H100 four ways. All teacher-overhead figures below are
  **estimates**.

| Rank | Run | Teacher | Loss and settings | Views | Epochs | Expected (source) | GPU cost (shared) |
|---|---|---|---|---|---|---|---|
| **1** | **E1: ensemble function matching** (`c100_fm_vitens3`) | 3 pure-ViT seeds (1000 ep s0/s1/s2, **83.23**). Option: add the 2000-ep run (83.57) | KL(p_t ‖ p_s) at τ = 1. Variant A: α = 0.5 with label CE (LS 0.1). Variant B (Beyer FM): α = 1, labels off. Teacher = mean of member softmaxes, bf16, eval mode | Teacher sees the **same** crop+RA+erase+mix batch. **Keep MixUp/CutMix on all run** (`--mix-off-epoch 0`) | 300 screen (A and B), then 1000 | 1000 ep: **81.8–82.5** (**unverified**). Basis: BAN same-size +0.8; ensemble distillation; FM closes most of the student-teacher gap only with long schedules (2106.05237). Teacher sits 2.3–2.7 points above the student | 3 teacher forwards ≈ +1x step cost, so about 20 s/epoch. 300 ep ≈ 1.7 h, 1000 ep ≈ 5.5 h. No teacher training needed |
| **2** | **E2: strong CIFAR-only CNN teacher, DeiT-style hard + soft distillation** (`c100_kd_r18long`) | First retrain ResNet-18 CIFAR-stem properly: SGD 0.1, momentum 0.9, nesterov, wd 5e-4, bs 128, 300 ep cosine, crop+flip+CutMix, LS 0.1. Target ≥ 79–80 (BYOT baseline at plain aug is 77.09, Table 1; with CutMix **unverified**). If it ends below 79, use WRN-28-10 (BAN: 80.92) | DeiT hard: ½·CE(s, y) + ½·CE(s, argmax t). Second arm: soft KL at τ = 1 (or `lsd`), α = 0.5 | Consistent views as E1, mix on all run | 300 screen, then 1000 | +0.3 to +1.0 over baseline (**unverified**). Supported by DeiT (CNN teacher better through inductive bias; hard > soft, +1.2 on ImageNet DeiT-B), Co-advise (teacher type > accuracy) and BAN cross-arch (+0.9 when teacher ≈ student). Risk: hard labels lose dark knowledge; BAN weak-teacher case shows harm below about 79 | Teacher training: R18 about 10–20 min at our 1.8 s/epoch R18 throughput. WRN-28-10 about 1–2 h (**unverified**). KD run about +20–30% step time, so 1000 ep ≈ 3.7 h |
| **3** | **E3: mixed-architecture ensemble teacher** (`c100_fm_vitens3_r18`) | E1's 3 ViTs + E2's CNN (prob average, or 0.75 ViT / 0.25 CNN) | As E1 (soft, τ = 1, α = 0.5 or 1) | As E1 | 1000 (only if E1 and E2 both help at 300) | Best guess **82–82.7** (**unverified**). Diversity from a different inductive bias (Co-advise) should raise ensemble accuracy above 83.2 and add bias the student lacks | About 6 h |
| **4** | **E4: online self-distillation, no separate teacher** (`c100_pskd_ema`) | EMA of the student (decay 0.999 per step, about 2.6 epochs at 390 steps/epoch), or previous-epoch weights (PS-KD) | Soft target = (1 − α_t)·mix-target + α_t·p_teacher, with α_t = 0.8·t/T (PS-KD). Teacher forward on the same batch | Consistent views | 300 screen, then 1000 if > +0.5 | +0.3 to +1.0 (**unverified**). CNN analogues are PS-KD +0.5 (on CutMix+ShakeDrop) to +3.4, DLB +0.8 to +2.5, path-ensemble ViT +0.4 (ImageNet). Our student already uses LS and heavy aug, so expect the low end | +1 forward ≈ +33%, so 1000 ep ≈ 4 h. Zero teacher cost. The DLB variant (τ = 3, α = 1, reuse last batch's logits) costs almost nothing but uses inconsistent views |
| **5** | **E5: control, no KD, patch 2** (`c100_spt_slvit_p2_1000ep`) | none | bf3b. Note 2406.09415 used lr 4e-3, bs 1024, wd 0.3 and **2400 ep** | Standard | 1000 (2400 if promising) | Their ViT-T/2 (5.6M): **83.6** at 2400 ep. Our 2.75M SL-ViT at p2 / 1000 ep: maybe 82–83 (**unverified**) | 256 tokens: about 4–5x per-epoch time (attention cost grows with tokens²), roughly 45 s/epoch, so 1000 ep ≈ 12 h. Inference MACs about 4x or more: **check against the edge budget** before treating this as a winner |

Not recommended:
- **DINO or MAE SSL pretext:** the best published CIFAR-only numbers (79.15 and 78.27) are below our supervised
  baseline. Costs 2–10x.
- **Locality-guidance feature imitation:** gains collapse to +1.9 for an already-local baseline (PVTv2), and our
  SPT+LSA+impulse student is already local. Needs a projector and multi-scale feature hooks. Revisit only if E2 shows
  CNN bias helps.
- **Snapshot distillation:** cyclic LR conflicts with our schedule, and the gain is +0.6.
- **BYOT:** needs auxiliary heads, and its gain on strong baselines is +0.2 to +1.2. A cheap later add-on to E4.

Suggested order (GPU budget about 1 day):
1. Launch E1-A, E1-B and E4 at 300 ep and train the E2 teacher, all at once (about 2 h).
2. Run E2 at 300 ep.
3. Run the best two at 1000 ep with 2 seeds, plus E5 in parallel (about 12 h).

**Decision rule:** a method counts as closing the gap if the 1000-ep two-seed mean is at least 82.3. That is within
seed noise of CCT's 82.72 at 1500 ep.

## 4. Minimal code change in `ViT/train_vit.py`

Do this after the other agent's edit lands; I did not touch the file. Reuse `kd_loss` from `finetune.py`
(soft / hard / dist / lsd already implemented and tested there).

```text
--teacher RUN[,RUN...]   run dirs under runs/; build each with build_model(Namespace(**results.json["args"]))
                         and load model.pt (strip "_orig_mod."). eval(), requires_grad_(False), bf16 autocast.
                         Works for ViT or resnet18 runs (same normalize(mean,std) pipeline, verified on CPU).
--teacher-weights w1,..  optional ensemble weights (default uniform); ensemble = weighted mean of softmax(logits/tau)
--kd {soft,hard,lsd,dist}  --kd-alpha A  --kd-tau T
--kd-labels-off          function matching: loss = KD only (Beyer); equivalent to --kd-alpha 1
--self-kd {ema,prev_epoch}  --self-kd-decay 0.999  --self-kd-alpha-final 0.8 (PS-KD linear ramp)
```

Train-step change, about 15 lines. Insert it right after `mix_batch` so the teacher sees the final student input
(consistent view):

```python
with torch.no_grad(), amp():
    p_t = sum(w * F.softmax(tm(x).float() / args.kd_tau, -1) for w, tm in zip(tw, teachers))
loss_ce = torch.sum(-t * F.log_softmax(logits.float(), -1), -1).mean()
loss_kd = F.kl_div(F.log_softmax(logits.float() / args.kd_tau, -1), p_t, reduction="batchmean") * args.kd_tau**2
loss = (1 - a) * loss_ce + a * loss_kd          # a = kd_alpha, or PS-KD ramp a_t = alpha_final * epoch / epochs
```

For `hard`, use `CE(logits, p_t.argmax(1))`. For a mixed batch this is the teacher's argmax on the mixed image, as in
DeiT.

CUDA-graph and GPU-resident notes:
- Teachers are frozen, fixed-shape, forward-only modules. Wrap each in `torch.compile(mode="reduce-overhead")` as
  well, or compile one `teacher_ens(x)` function. All data stays on the GPU, and the 3 ViT teachers together are
  about 33 MB of bf16 weights.
- For the EMA teacher, keep a deep copy and update it after `opt.step()` with
  `torch._foreach_lerp_(ema_params, params, 1 - decay)`. This is graph-safe and has no host sync. For PS-KD
  "previous epoch", copy the weights at the start of each epoch with `torch._foreach_copy_`.
- `--mix-off-epoch 0` is required for E1 and E3. When KD is on, log `kd_loss` and
  `agree = (argmax s == argmax t)` like finetune.py's `rec["kd_loss"]`.
- Smoke test after the edit: `--fake-data --epochs 2 --teacher c100_spt_slvit_1000ep --kd soft --cudagraphs`, plus
  the existing no-teacher path (per the shared-code smoke-test rule).

## 5. Reproducibility of the local numbers

- **Ensemble accuracy:** a CPU-only script (scratchpad, not committed) imports `build_model` and `normalize` from
  `ViT/train_vit.py` and loads `runs/<run>/model.pt` with `strict=True` (all keys matched). It evaluates the 10k
  CIFAR-100 test images in fp32 and averages softmax probabilities.
  - Single-model fp32 numbers (80.64 / 81.21 / 80.88 / 80.91) are within 0.3 of the logged bf16 `final_val_acc`
    (80.66 / 81.17 / 80.91 / 80.98).
- **Teacher softness:** the same script uses `random_crop_flip`, `rand_augment(n=2, m=0.9)`, `random_erasing(0.25)`
  and `mix_batch(mixup 0.8, cutmix 1.0)` on 4,000 train images (seed 0).

## 6. Sources (all fetched 2026-10-13 unless noted)

- BAN: https://arxiv.org/abs/1805.04770 (PDF Tables 2–3)
- PS-KD: https://arxiv.org/html/2006.12000
- DLB: https://arxiv.org/html/2203.16172
- Snapshot Distillation: https://arxiv.org/pdf/1812.00123 (Table 3)
- BYOT: https://arxiv.org/html/1905.08094
- Path-ensemble ViT self-distillation: https://arxiv.org/html/2308.06548
- DeiT: https://arxiv.org/html/2012.12877
- Co-advise / CivT: https://arxiv.org/pdf/2106.12378
- Locality guidance: https://arxiv.org/html/2207.10026
- CIST (ResNet-56→ViT logit KD on CIFAR-100): https://arxiv.org/pdf/2605.20357 (Table 7)
- DeiT-LT: https://arxiv.org/html/2404.02900
- Bootstrapping ViTs: https://arxiv.org/abs/2112.03552 (numbers from https://github.com/zhfeing/bootstrapping-vits-pytorch,
  via search snippet, **not verified**)
- Gani et al.: https://arxiv.org/html/2210.07240
- Mae-ViT small data: https://arxiv.org/html/2402.03752
- Pixel / ViT-T/2 from scratch: https://arxiv.org/html/2406.09415
- MSCViT: https://arxiv.org/html/2501.06040v2
- Function matching: https://arxiv.org/html/2106.05237 (numbers from `research/edge_kd_transfer.md`, fetched 2026-10-11)
- CCT: https://arxiv.org/html/2104.05704 (numbers from `research/long_runs.md`, fetched 2026-10-11)
