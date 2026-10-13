# vision_models: small Vision Transformers, trained fast and measured honestly

How far can a **tiny Vision Transformer** go on small datasets, and what is the best image classifier you can fit in an
**edge budget (≤ 6M parameters)**? This repo holds the training code, all launch scripts, about 190 per-run results files and the queue logs, and the
write-ups for a two-day research program on one H100.

The step-by-step educational report is [`ViT/DASHBOARD.html`](ViT/DASHBOARD.html). It covers every experiment, what changed, why, the
result, and how it compares with published numbers, with sources. Download it and open it in a browser; GitHub does not render HTML.
Raw per-run tables are in [`ViT/REPORT.md`](ViT/REPORT.md).

## Headline results

Top-1 accuracy on the official test set. "3 seeds" means mean ± standard deviation over 3 runs; everything else is a single run.

| Setting | Model | Params | Result | Best comparable published result we found |
|---|---|---|---|---|
| CIFAR-100, **pure ViT** (no convolutions), from scratch, no extra data | SL-ViT shape + SPT + LSA + SeqPool, 1000 epochs | 2.75M | **80.91 ± 0.26** (3 seeds) | Pure ViT: 79.15 ([Gani et al.](https://arxiv.org/abs/2210.07240), two-stage). Conv-tokenizer hybrid [CCT](https://arxiv.org/abs/2104.05704): 80.92 at 300 ep, 82.72 at 1500 ep |
| CIFAR-10, pure ViT, from scratch | ViT-Tiny shape + SPT + LSA + SeqPool, 1000 epochs | 5.4M | **96.87 ± 0.04** with SPT (3 seeds); 96.54 ± 0.13 without SPT. The 2.75M SL-ViT shape + SPT gives **96.75 ± 0.26** (3 seeds), level with the 5.4M model at half the parameters | Pure ViT: 96.41 (Gani et al.); [SL-ViT](https://arxiv.org/abs/2112.13492): 94.53; CCT: 96.53 at 300 ep |
| CIFAR-100, edge model, pretrained | TinyViT-5M, 200-epoch distillation from a ViT-B teacher | 5.1M | **90.40** (2 seeds: 90.32, 90.47; 400 epochs gives 90.47, so 200 is enough); the same schedule without a teacher (RepViT) gives 87.70, so the teacher matters | Similar-size pretrained models: MobileNetV3-L 88.5 ([Ridnik et al.](https://arxiv.org/abs/2104.10972)), EfficientNet-B0 88.1 ([Tan & Le](https://arxiv.org/abs/1905.11946)) |
| CIFAR-100, edge model, pretrained | RepViT-M0.9, same distillation | 4.8M | 89.85 (2 seeds: 89.93, 89.76; 400 epochs: 89.70; 0.82 GMACs); 88.63 at 128 px (0.27 GMACs) | as above |
| CIFAR-100, pure ViT, **64 px input**, our own ImageNet-21k pretraining | ViT-Tiny, patch 4, fine-tune with distillation | 5.4M | 88.14 at 200 epochs (120 epochs: 87.71, 2 seeds; 60 epochs: 86.89, 2 seeds) | n/a |

This is **not** state of the art on CIFAR-100. The best published results we found are 96.1% with a 480M-parameter model and JFT-300M pretraining, and 89.9% from scratch with a
26M-parameter CNN. The claim is narrower: these are the best results we found for models of this size and type.

## What actually mattered (pure ViT from scratch)

Each change was a one-variable ablation of the recipe in [`ViT/train_vit.py`](ViT/train_vit.py):

| Change | Gain (points) |
|---|---|
| More optimizer steps at the same epochs (batch 1024 → 256; CUDA graphs keep it fast) | +10 |
| Learning rate scaled down for width (ViT-Small) | +20.6 |
| 300 → 1000 epochs, same recipe (CIFAR-100, 3-seed means) | +3.3 |
| Locality Self-Attention, `--lsa` ([Lee et al.](https://arxiv.org/abs/2112.13492)) (CIFAR-10) | +2.6 |
| Shifted Patch Tokenization, `--spt` (same paper) (CIFAR-100, 3-seed means at 1000 ep) | +1.9 |
| Impulse attention init, `--init impulse` ([Zheng et al.](https://arxiv.org/abs/2505.19985)) (CIFAR-10) | +0.7 |
| Sequence pooling instead of the CLS token, `--pool seqpool` ([CCT](https://arxiv.org/abs/2104.05704)) (CIFAR-10) | +0.3 |

## Negative results (kept on purpose)

- **Recurrent depth** (looping a shared block): led a flat model by about 2 points at equal compute at 100 epochs, and the lead disappeared at 500
  epochs (77.44 vs 77.86). It speeds up convergence but does not reach a better optimum. Details: [`ViT/research/results_wave1_recurrent.md`](ViT/research/results_wave1_recurrent.md).
- **Mixture-of-experts at ≤ 6M active parameters** (CondConv, routed MLP-MoE, sparse upcycling): no gain over dense outside seed noise. A dense
  model trained twice as long gained +1.9, more than any MoE variant. Details: [`ViT/research/results_moe.md`](ViT/research/results_moe.md).
- **Static INT8 post-training quantization** (ONNX Runtime defaults) collapsed several of the best FP32 edge models, for example RepViT-M0.9 88.5 → 37.6.
  Dynamic INT8 was lossless on all 11 models. Details: [`ViT/research/results_wave1_int8.md`](ViT/research/results_wave1_int8.md).

## Caveats

- **The recipes were chosen while watching the CIFAR test set**, as in most CIFAR papers. To check that this did not inflate the numbers, the
  CIFAR-100 headline recipe was rerun on 45k training images with a fixed 5k held-out split (`--holdout 5000`, [`ViT/run_holdout.sh`](ViT/run_holdout.sh)),
  and the test set was scored only once at the end. Held-out val was 79.88 and test was 79.89, so recipe selection did not overfit the test set. The run is about
  1 point below the 50k-image runs because it trains on 10% less data. The same check on CIFAR-10 gave held-out 96.80 vs test 96.80.
- Edge numbers are single seeds unless marked; repeat runs on this setup vary by about 0.3–0.6 points.
- Published comparisons come from the papers linked above. Our search was thorough but not exhaustive, so "best we found" means exactly that.

## Reproduce

Hardware: 1× H100 (80 GB). Software: Python 3.10, PyTorch 2.11 (CUDA 12.8), timm 1.0.30. See [`requirements.txt`](requirements.txt).

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
source ViT/env.sh            # keeps all caches inside the repo
cd ViT

# CIFAR-100 headline: 2.75M pure ViT, ~3 h on one H100 (CIFAR downloads automatically from the Hugging Face mirror)
../.venv/bin/python train_vit.py --run-name c100_spt_slvit_1000ep --dataset cifar100 --model slvit --spt \
  --batch-size 128 --lr 5e-4 --weight-decay 0.05 --epochs 1000 --warmup-epochs 30 --randaug --label-smoothing 0.1 \
  --drop-path 0.1 --reprob 0.25 --init impulse --mix-off-epoch 583 --pool seqpool --lsa --eval-every 10 --cudagraphs

# CIFAR-10 headline: same recipe, ViT-Tiny shape
../.venv/bin/python train_vit.py --run-name c10_tinyshape_bf3b_1000ep --dataset cifar10 --model tiny \
  --batch-size 128 --lr 5e-4 --weight-decay 0.05 --epochs 1000 --warmup-epochs 30 --randaug --label-smoothing 0.1 \
  --drop-path 0.1 --reprob 0.25 --init impulse --mix-off-epoch 583 --pool seqpool --lsa --eval-every 10 --cudagraphs
```

Edge models and distillation use [`ViT/finetune.py`](ViT/finetune.py) on top of timm checkpoints:

- the ViT-B/16 teacher: 93.43% on CIFAR-100 after 20 epochs at 224 px;
- the 200-epoch function-matching students: [`ViT/run_edge_kd.sh`](ViT/run_edge_kd.sh);
- ImageNet-21k-P pretraining and its fine-tunes: [`ViT/run_in21k.sh`](ViT/run_in21k.sh). This needs gated Hugging Face access to the dataset.

Every experiment's exact command is in a `ViT/run_*.sh` script, and every run's full configuration and history is in `ViT/runs/<run>/results.json`.

## Repository map

| Path | What |
|---|---|
| `ViT/train_vit.py` | From-scratch training. GPU-resident data, GPU augmentation (RandAugment, MixUp/CutMix, random erasing), bf16 + `torch.compile` + CUDA graphs. Every recipe component is a flag |
| `ViT/finetune.py` | Fine-tuning and distillation (soft, hard, DIST, logit-standardized, function matching) of timm and our own checkpoints |
| `ViT/models/recurrent_vit.py`, `ViT/moe.py` | Recurrent-depth ViT; MoE conversions (CondConv, routed MLP-MoE, upcycling) |
| `ViT/vit_data/` | Dataset plug-ins: CIFAR-10/100, Tiny-ImageNet, ImageNet-64, streaming ImageNet-21k-P at 64 px |
| `ViT/edge_bench.py`, `ViT/int8_audit.py` | CPU latency / MACs benchmark; ONNX INT8 accuracy audit |
| `ViT/research/` | Literature notes and per-wave result write-ups |
| `ViT/runs/<run>/` | `results.json` (full config + per-epoch history) for every run; console logs and weights are not committed |
| `ViT/make_dashboard.py`, `ViT/make_report.py` | Generate `DASHBOARD.html` and `REPORT.md` / `REPORT.html` from the run directories |

Compute: about 175 runs, roughly 49 h of wall-clock time on one H100 PCIe (about $160 at $3.29/h).
