All numbers below are single-seed runs from `runs/*/results.json`; differences of about 1 point are within what we could not separate from noise. Published numbers are from `REFERENCES.md` (verified) and `RESEARCH_DIRECTIONS.md`. Claims about runs that have not finished are marked **pending**.

### Why plain ViT overfits

- CIFAR-10 `t1_plain` (ViT-Tiny, crop+flip only) reaches 100.0% clean train accuracy but only 82.9% val, a 17.1-point gap.
- CIFAR-100 is much worse: `c100_t1_plain` hits 99.97% train and 57.33% val, a 42.6-point gap. Same 50k images but 100 classes means about 500 per class, so memorizing is easier than learning features.
- A plain ViT has no convolutional locality or translation prior, so it has to learn these from 50k small images. The ResNet-18 with the heavy recipe sits at a 2.7-point gap on CIFAR-10 and 10.8 on CIFAR-100.
- The learned position embeddings and attention maps (CIFAR-100 `c100_t6_full_300ep`) show the model does recover locality: nearby patches have similar position embeddings and CLS attention lands on the object. It learns this from data instead of having it built in.
- Even after 300 epochs of the full recipe, CIFAR-100 still has a 24.9-point gap (95.4 train / 70.5 val), so more regularization or more data is still needed there.

### Which regularizers mattered

- RandAugment was the clear win: +2.9 points on CIFAR-10 (83.1 to 86.0, gap 17 to 7) and +5.1 on CIFAR-100 (56.8 to 61.8).
- Label smoothing + weight decay were negligible: +0.2 on CIFAR-10 and -0.5 on CIFAR-100, with no gap change. We could not tell them apart from noise.
- MixUp/CutMix hurt at a 100-epoch budget: -15.0 points on CIFAR-10 (86.0 to 70.9) and -4.9 on CIFAR-100, while the gap collapsed (train acc 72.9%): the model underfits instead of overfitting.
- The underfit is a budget issue, not a verdict against mixing. Applying it to half the batches (`t5b_mix_half`) recovers +7.1 points (71.97 to 79.08), and 300 epochs (`t6_full_300ep`) reaches 87.96 on CIFAR-10 and 70.49 on CIFAR-100.
- Drop-path 0.1 gave +1.0 (CIFAR-10) and +0.4 (CIFAR-100), too small to claim an effect.

### Batch size / steps

- `t5c_bs256` is `t5_droppath` with batch 1024 to 256 and lr 2e-3 to 1e-3, i.e. 4x more optimizer steps for the same 100 epochs. Val goes 71.97 to 82.09 (+10.1) and train acc 74.5 to 84.7: the model was step-starved, not over-regularized.
- That step count is expensive here: 16.5 min vs 3.6 min, mostly because this run was not CUDA-graphed (5.2k img/s).
- The ViT-Small lr 2e-3 run (`s_small_200ep`) stalled at loss ~2.1 and 58.0% val. Halving lr to 1e-3 (`s_small_200ep_lr1e-3`) gives 78.64 (+20.6), so that failure was a learning-rate problem, not a capacity limit.
- Even fixed, Small (21M params, 200 ep) is below Tiny at batch 256 (82.09), and on CIFAR-100 Small (68.01) is below Tiny at 300 epochs (70.49). More parameters did not help at this data size.
- The baseline-fix runs support this: with batch 128 and lr 3e-3, `bf1b_tiny_samerecipe` and `bf2_slvit_init` both sat at ~49% train accuracy (stalled, not overfit). The LR check (`bf1_slvit_lr1e-3`, `lr5e-4`) is **pending**, so the lr 3e-3 diagnosis is not confirmed.

### ViT vs ResNet-18

- ResNet-18 with the same heavy recipe gets 93.16% on CIFAR-10 in 3.0 min, versus 71.97% for the same-budget ViT-Tiny (+21.2) and 87.96% for the best ViT-Tiny run (300 epochs, 12.9 min). On CIFAR-100: 73.22% vs 57.31% (same budget) and 70.49% (300 epochs).
- The gap narrows on the harder dataset: the best ViT is 5.2 points behind on CIFAR-10 and 2.7 on CIFAR-100.
- ResNet-18 also generalizes better per unit of fitting: gap 2.7 on CIFAR-10 at 95.9% train acc, whereas the ViT only reaches comparable train accuracy after 300 epochs.
- Our ResNet-18 is a plain 100-epoch run at batch 1024, so it is not tuned; see the published comparison below.

### Patch size / sequence length tradeoff

- All three patch runs use the heavy recipe at 100 epochs and batch 1024, which underfits (train acc 77.8 / 74.5 / 58.6 for p2 / p4 / p8, gaps 1 to 3). The ranking therefore mostly shows "more tokens means more compute per step", not generalization.
- Val accuracy: p2 (256 tokens) 75.01, p4 (64 tokens, `t5_droppath`) 71.97, p8 (16 tokens) 57.38. Going p4 to p2 is +3.0 points; p4 to p8 costs -14.6.
- Throughput: p2 9.1k img/s vs 29.2k for p4 (0.31x). p8 measured 20.6k img/s, which is slower than p4 despite 4x fewer tokens. That is a measurement artifact (these runs shared the GPU with other jobs, and small models are launch-bound), so treat the img/s column as indicative only.
- A fair patch comparison needs the non-underfitting recipe (e.g. `t5b` or `t5c` settings); we have not run it.

### How we compare to published results

- Our best CIFAR-10 ViT-Tiny (87.96%, 300 epochs) is within 0.5 points of the closest matched published baseline (88.43%, same 192/12/3 p4 shape, 100 epochs, Trockman & Kolter). It is 5.6 points below tuned pure ViTs (ViT-Lite-7/4 93.57, SPT-paper ViT 93.58, SL-ViT 94.53), which are shallower, use batch 128, and use tuned lr and wd.
- Our recipe port of those results has not matched them yet: `bf1_slvit` (their shape and recipe, 100 ep) got 75.84%. Part of the gap is step count: their repeated-aug sampler makes "100 epochs" about 117k steps versus our 39k. The 300-epoch `bf3_slvit_300ep` is **pending**.
- Our ResNet-18 (93.16%) matches kuangliu's 93.02% but is 2.1 points under the Cutout-paper 95.28% (200 epochs, batch 128).
- CIFAR-100 (verified in `REFERENCES.md`): our 70.49% is 1.6 below ViTRM (72.1), 3.3 below the SPT-paper plain ViT (73.81), 3.4 below ViT-Lite-7/4 (73.94), 6.4 below SL-ViT (76.92), and well below SSL-pretrained ViT (79.15) and CCT-7/3x1 (80.92; 82.87 at 5000 epochs). Our 73.22% ResNet-18 is level with the plain small ViTs. The same-shape 12L/3h ViT at 68.09 (Gani Tab. 15) is quoted by a research agent and not re-checked.
- Results above ~96% under 10M params in the references use convolutional tokenizers (CCT 96.53), self-supervised pretraining, or very long schedules (CCT 98.00 at 5000 epochs).

### Throughput engineering

- CUDA graphs (`--cudagraphs`, `torch.compile` reduce-overhead) give 2.7x, measured. A 10-epoch real CIFAR-10 run (batch 256, ViT-Tiny, RandAugment) took 9.45 s/epoch without vs 3.48 s/epoch with graphs, with the same accuracy within noise (val 53.2% vs 53.0%). `bench_step.py` (the reproduction script) gives 49.3 vs 16.6-17.5 ms/step at batch 256 and 50.6 vs 32.1 ms/step at batch 1024 (1.6x), so the win is largest at small batch, where small ViTs are kernel-launch bound.
- The un-graphed `t5c_bs256` ran at 9.6 s/epoch (5.2k img/s) for the same reason; graphed CIFAR-100 batch-256 runs take about 3 to 4 s/epoch.
- Parallel queues were measured: with 3 concurrent jobs, per-job epoch time rose 9 to 55% (bf3 6.1 to 6.65 s; recurrent pilot ~5.6 to 8.8 s) for roughly 2x the total throughput, at 9 GB of 80 GB memory and 226 W of 350 W. Small launch-bound models leave the GPU underused, so parallel queues are worth it. The report gives both the sum of run-minutes and an estimated wall-clock span (see Compute summary); per-run img/s of overlapping runs is lower than solo.

### Scaling to bigger datasets

- **Pending.** Tiny-ImageNet and ImageNet-64 runs have not been executed, so we make no claim about how the ViT-vs-ResNet picture changes with scale. On CIFAR-10 to CIFAR-100 the gap shrank from 5.2 to 2.7 points, but that is one step on one axis.
