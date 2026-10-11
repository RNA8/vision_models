# B4: ImageNet-21K-P pretraining of our ViT-Tiny (64 px) -> CIFAR-100

Goal: same 5.6M-param ViT-Tiny (patch 8 @ 64 px = 64 tokens, LSA + SeqPool + impulse init) pretrained on ImageNet-21K-P
(winter21, 10,450 classes, 11,060,239 train images, 64x64) vs. ImageNet-1k at 64 px (`in64pre_tiny_p8`, 56.83% IN-64 top-1),
then the same four CIFAR-100 fine-tunes. IN-64 references: 81.17 / 81.47 (KD) / 83.78 (p4) / 84.08 (p4+KD).

## What the literature says (fetched 2026-10-11; quotes are from the ar5iv HTML renderings, as returned by the fetch tool)

**Ridnik et al., "ImageNet-21K Pretraining for the Masses", arXiv 2104.10972** (https://ar5iv.labs.arxiv.org/html/2104.10972, https://arxiv.org/abs/2104.10972)
- Optimizer/schedule: "using an Adam optimizer with learning rate of 3e-4 and one-cycle policy."
- Epochs: with ImageNet-1K init, 80 epochs "are enough for achieving strong pretrain results on ImageNet-21K-P"; random init needed 140 epochs (TResNet-M, 224 px; not a ViT).
- Single-label loss: "we increase the label-smooth factor from the common value of 0.1 to 0.2." Regularisation: "RandAugment, Cutout, Label-smoothing and True-weight-decay."
- Semantic softmax (one softmax per hierarchy level, 11 levels): Table 2 gains over single-label softmax, e.g. ImageNet-1K 81.1->81.4, CIFAR-100 89.5->90.4 (downstream, 224 px, TResNet-M).
- Not given: batch size, upstream weight decay. They did not train ViTs themselves.

**Steiner et al., "How to train your ViT?", arXiv 2106.10270** (https://ar5iv.labs.arxiv.org/html/2106.10270, https://arxiv.org/abs/2106.10270)
- "On the smaller ImageNet-1k dataset we trained for 300 epochs, and for 30 and 300 epochs on the ImageNet-21k dataset."
- Adam, batch 4096, cosine with linear warmup of 10k steps, peak lr 0.001 ("the learning-rate which peaks at 0.001"), decoupled weight decay 0.1 or 0.03, "gradients were clipped at global norm 1".
- At 30 epochs: "any kind of AugReg hurts performance for all but the largest models"; only at 300 epochs "AugReg helps more models"; "one needs to be careful not to apply AugReg to a model that is too small, or when pre-training for too short."
- "we observe consistent improvements on ImageNet-21k dataset with 10x longer schedule."
Caveat: the fetch tool summarises pages with a small model; numbers above were returned as quotes/paraphrase, not read by us from the PDF
(the PDFs could not be text-extracted here). Treat them as pointers to verify before citing in a paper.

## Recipe chosen (`run_in21k.sh`)
Same as the IN-64 pretrain (`run_next.sh pre`), which follows the same logic as Steiner's 30-epoch finding (a short schedule underfits a
tiny model, so no mixup/cutmix/drop-path): batch 256, AdamW lr 1e-3 (cosine to 1e-5), wd 0.05, grad-clip 1, RandAugment(2,9) + crop/flip,
plain softmax with label smoothing 0.1 (Ridnik use 0.2 for single-label; with 10,450 classes 0.1 is kept to stay comparable to
the IN-64 run, noted as an untested alternative), warmup 1 epoch (43k steps; epoch is 8.6x longer than IN-1k), 30 epochs = ~332M samples,
~1.3M steps. Head: 10,450-way linear on SeqPool. Semantic softmax not used: it needs 11 heads/hierarchy weights (small cost, but changes the
comparison; `.sem.npy` is in the shards so it is an easy follow-up).

## Cost / measurements
- Loader alone (no model): ~28k img/s at batch 256 (2000 batches incl. cold-disk windows), process RSS peak 2.6 GB; page cache grows toward the dataset size (136 GB) as epochs proceed.
- In training (smoke, 20 shards, 5 jobs sharing the GPU): 3.9k img/s with "loader wait" 0.3 s per 60 s epoch, i.e. GPU-bound, loader not a bottleneck.
- Reference `in64pre_tiny_p8`: 12,999 img/s steady when sharing -> 11.06M / 13k = ~14 min/epoch -> 30 epochs ~7.1 h (alone it should be faster; under
  the heavy 5-job contention measured in the smoke it would be ~3x slower). Kept at 30 epochs (< 12 h). Fine-tunes: ~15-25 min each.
- Smoke (20 shards, 4 ep, lr 1e-3): loss 9.198 -> 9.023 -> 8.797 -> 8.616 (ln 10450 = 9.25), resume from ckpt verified.
- Params with the 10,450-class head: 7.41M (5.58M with 1000 classes; the head accounts for the difference and is dropped for CIFAR-100).
