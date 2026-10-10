#!/usr/bin/env bash
# ImageNet-1k @ 64x64 stage: ViT p8 (64 tokens) ladder + ResNet-18 control, back to back.
#   in64_t1_plain    ViT-Tiny p8, 30 ep, crop+flip only
#   in64_t5_full     ViT-Tiny p8, 30 ep, full recipe (LS, WD, mixup/cutmix, drop-path, RandAugment)
#   in64_r18_full    ResNet-18 (--r18-stem stride2), 30 ep, full recipe
#   in64_s_small_full ViT-Small p8, 40 ep, full recipe, LR 5e-4
# Needs data/imagenet64_uint8.npz (see vit_data/imagenet64.py: gated ImageNet, needs HF token). If the cache
# is missing this script builds it first (download ~150 GB + decode).
# A run is skipped if runs/<name>/results.json exists. Launch inside tmux:
#   tmux new -s in64 'bash ViT/run_imagenet64.sh'
# Lessons applied: batch 256 + sqrt-scaled LR (Tiny 1e-3, Small 5e-4 = half Tiny; Small at 2e-3 failed) because
# batch 1024 underfits small ViTs; --cudagraphs (2.7x faster; 1.28M imgs / 256 = 5000 steps/epoch is fine with graphs);
# r18 uses --r18-stem stride2 (same cost as 32x32; CIFAR stem would be 4x), so it differs from the CIFAR-stem
# controls; --eval-every 2 and --log-every-steps 500 (wandb train loss/lr every 500 steps).
# Checkpointing is on by default (--ckpt-every 10 epochs -> runs/<name>/ckpt.pt), so a crash mid-run resumes
# from the last checkpoint when this script is rerun (important for runs of 40 min - 1+ h).
# TODO (not yet): --lsa / --pool seqpool / SL-ViT recipe. Baseline-fix experiments testing them are still running
# and the ladders must stay comparable with the CIFAR-10/100 chains. Once validated, add them to COMMON/the ViT runs.
# Time estimates (standalone GPU; c100 measured ViT-Tiny 3.0 s/ep plain, 3.4 randaug, Small 3.86, r18 1.84 per 50k
# imgs; ImageNet-64 epoch = 25.6x that, same 65 tokens): t1_plain ~40 min, t5_full ~46 min, r18 stride2 ~25 min,
# s_small (40 ep) ~68 min => ~3.0 h standalone, ~4.5 h when sharing the GPU with 2 other queues. See SCHEDULE.md.
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
COMMON="--wandb --eval-every 2 --log-every-steps 500 --warmup-epochs 2 --dataset imagenet64 --patch-size 8 --batch-size 256 --lr 1e-3 --cudagraphs"

if [[ ! -f data/imagenet64_uint8.npz ]]; then
  mkdir -p data
  $PY -m vit_data.imagenet64 --prepare 2>&1 | tee -a data/imagenet64_download.log
  [[ -f data/imagenet64_uint8.npz ]] || { echo "!! imagenet64 cache missing, aborting"; exit 1; }
fi

run() {  # run <name> <args...>
  local name=$1; shift
  if [[ -f runs/$name/results.json ]]; then echo "== skip $name (done)"; return; fi
  mkdir -p runs/$name
  echo "== start $name: $*  ($(date))"
  $PY train_vit.py --run-name "$name" $COMMON "$@" 2>&1 | tee runs/$name/train.log
  [[ -f runs/$name/results.json ]] || echo "!! $name FAILED (see runs/$name/train.log)"
}

NOREG="--label-smoothing 0 --weight-decay 0 --mixup 0 --cutmix 0 --drop-path 0"
run in64_t1_plain      --model tiny --epochs 30 $NOREG
run in64_t5_full       --model tiny --epochs 30 --randaug
run in64_r18_full      --model resnet18 --epochs 30 --randaug --r18-stem stride2
run in64_s_small_full  --model small --epochs 40 --randaug --lr 5e-4
echo "== imagenet64 runs finished $(date)"
