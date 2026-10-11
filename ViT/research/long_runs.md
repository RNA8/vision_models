# Long-schedule runs (500-1000 epochs), queued 2026-10-11

Queue: `ViT/run_long.sh` (tmux window `long`). It waits until no `train_vit.py`/`finetune.py` process and no other
`bash run_*.sh` queue is alive (checked every 5 min), then runs up to 4 concurrently. All single seed 0.

## Why long schedules (sources fetched 2026-10-11)
- Compact Convolutional Transformers, arXiv 2104.05704 (https://arxiv.org/html/2104.05704, Table 2), CCT-7/3x1, top-1:
  300 ep 96.53 / 80.92 (CIFAR-10 / CIFAR-100); 1500 ep 97.48 / 82.72; 5000 ep 98.00 / 82.87. Quote: "this model can get as high as
  98% accuracy on CIFAR-10, and 82.87% accuracy on CIFAR-100". So +1.8 on C100 from 300 -> 1500 ep, saturating by ~1500
  (+0.15 more at 5000). This is the closest published evidence for a ~3M-param transformer on CIFAR; it motivates 1000 ep for our ViTs.
  Caveat: CCT has a conv tokenizer, and the paper's own Table 1 and text disagree slightly on which number is the 1500-ep one.
- Beyer et al., "Knowledge distillation: a good teacher is patient and consistent", arXiv 2106.05237
  (https://arxiv.org/html/2106.05237, appendix Table 4), ResNet-50 student: Pets 82.75 (300 ep) -> 88.01 (1,000) -> 90.98 (10,000);
  Flowers102 77.51 (1,000 ep) -> 92.83 (10,000); Food101 86.64 at 1,000 ep vs teacher 86.24. Quote: "there is no overfitting even when we
  optimize for a 1M epochs". Abstract: "obtain a state-of-the-art ResNet-50 model for ImageNet, which achieves 82.8% top-1 accuracy."
  Motivates L6 (small data + function-matching KD keeps improving with epochs). These are from-scratch students, so a pretrained RepViT
  may gain less; L6 is gated on the 200-ep result for that reason.
- A web search found no controlled 300-vs-1000-epoch ViT-on-CIFAR-100 study beyond CCT; one hobby CIFAR-10 repo saw no gain past ~250 ep,
  so the gain is not guaranteed (recipe-dependent). See also `small_vit_improvements.md` row 6 and item 4 (longer schedule + EMA).

## Runs (order = start order; the first four start together)
| # | name | change vs reference | reference result (short) |
|---|------|--------------------|-------------------------|
| L1 | c100_tinyshape_bf3b_1000ep | `--epochs 1000 --mix-off-epoch 583`, warmup 30 | c100_tinyshape_bf3b 300 ep: 75.16 (best 75.37) |
| L4 | c100_rd_r8_add_500ep | `--epochs 500 --warmup-epochs 25` (warmup 5 x5) | c100_rd_r8_add 100 ep: 65.28 |
| L5 | c100_rd_flat6_500ep | same, param-matched flat control | c100_rd_flat6 100 ep: 62.94 |
| L2 | c100_bf3b_slvit_1000ep | as L1 with `--model slvit` | c100_bf3b_slvit_300ep_lr5e-4: 75.69 |
| L3 | c10_tinyshape_bf3b_1000ep | as L1 on CIFAR-10 | abl_tiny_shape 300 ep: 94.59 +/- 0.05 (3 seeds) |
| L6 | ec100_fm_repvit_128_e1000 | only if ec100_fm_repvit_224_e200 final_val_acc > 88.63 (= 88.33 + 0.3); `--img-size 128 --epochs 1000 --warmup-epochs 10` | ec100_repvit_m0_9_224 (30 ep): 88.33 |

All use `--eval-every 10` (logs half the size of the 5-ep default); recurrent-depth runs keep batch 256, lr 1e-3, `--lsa --pool seqpool`,
no mix-off (as in the 100-ep runs). Exact commands: `DRY=1 bash ViT/run_long.sh`. Every flag was checked against `--help` of
`train_vit.py` / `finetune.py` (no unknown flags). The L6 command is the e200 command from `run_edge_kd.sh` with the changes above.

## Robustness
- Both `train_vit.py` (`--ckpt-every 10`, default) and `finetune.py` save `runs/<name>/ckpt.pt` and resume automatically (model, optimizer,
  scheduler, RNG, history, stable wandb id). A crash therefore costs at most ~10 epochs: re-run `bash run_long.sh` (finished runs are
  skipped). Do not pass `--no-resume`. No refactor was needed.
- Single-GPU sharing: 4 jobs at once slow each other. Epoch times below are the steady-state seconds measured in runs that were already
  sharing the GPU, so they are the best available proxy.

## Wall-time estimates (from `total_minutes` / `epoch_sec_steady` of the reference runs, linear in epochs)
| run | ref min (epochs) | scaled alone-ish | under 4-way sharing (guess 1.3-1.8x) |
|-----|------------------|------------------|--------------------------------------|
| L1 | 57.0 (300) | 3.2 h | 4-6 h |
| L4 | 26.1 (100) | 2.2 h | 3-4 h |
| L5 | 12.3 (100) | 1.0 h | 1.5-2 h |
| L2 | 52.1 (300) | 2.9 h | 4-5 h |
| L3 | 78.7 (300) | 4.4 h | 6-8 h |
| L6 | 72.7 s/ep at 128 px with 224 px teacher (teacher-bound), 1000 ep | ~20 h at that load | 12-20 h (conditional; the longest job) |

Total GPU-time ~14 h without L6 (~34 h with it); wall clock after the wait (other queues need ~4-6 h more): ~8-10 h for L1-L5
(L3 starts when L5 frees a slot), L6 would extend this to ~1 day. Estimates are mine, not measured.
