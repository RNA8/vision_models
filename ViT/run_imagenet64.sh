#!/usr/bin/env bash
# ImageNet-1k @ 64x64 stage: ViT p8 (64 tokens) ladder + ResNet-18 control, back to back.
#   in64_t1_plain    ViT-Tiny p8, 30 ep, crop+flip only
#   in64_t5_full     ViT-Tiny p8, 30 ep, full recipe (LS, WD, mixup/cutmix, drop-path, RandAugment)
#   in64_r18_full    ResNet-18, 30 ep, full recipe
#   in64_s_small_full ViT-Small p8, 40 ep, full recipe
# Needs data/imagenet64_uint8.npz (see vit_data/imagenet64.py: gated ImageNet, needs HF token). If the cache
# is missing this script builds it first (download ~150 GB + decode).
# A run is skipped if runs/<name>/results.json exists. Launch inside tmux:
#   tmux new -s in64 'bash ViT/run_imagenet64.sh'
# Time estimates (1 ImageNet-64 epoch = 1.28M imgs = ~25x a CIFAR-10 epoch of 50k; same 64 tokens/img as
# CIFAR p4, so per-image cost matches): t1_plain and t5_full ~ 30 x 25 = ~750 CIFAR-tiny-epochs each;
# r18 ~ 750 CIFAR-r18-epochs; s_small ~ 40 x 25 x ~3 (Small is ~3-4x Tiny FLOPs) = ~3000+ CIFAR-tiny-epochs,
# i.e. the Small run is ~4-5x the wall time of a tiny run. Total ~ 8-9 tiny-run units.
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
COMMON="--wandb --eval-every 2 --warmup-epochs 2 --dataset imagenet64 --patch-size 8"

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
run in64_r18_full      --model resnet18 --epochs 30 --randaug
run in64_s_small_full  --model small --epochs 40 --randaug
echo "== imagenet64 runs finished $(date)"
