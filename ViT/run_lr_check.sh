#!/usr/bin/env bash
# LR check for the SL-ViT recipe: bf1_slvit (lr 3e-3, batch 128) reached only 75.8%, and
# bf1b/bf2 with the same LR ~49% (no fitting). Same recipe at lower peak LR. ~12 min each.
#   tmux new -s lr 'bash ViT/run_lr_check.sh'
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
COMMON="--wandb --eval-every 5 --dataset cifar10 --cudagraphs"
RECIPE="--batch-size 128 --weight-decay 0.05 --warmup-epochs 10 --randaug \
  --label-smoothing 0.1 --drop-path 0.1 --reprob 0.25"

run() {  # run <name> <args...>
  local name=$1; shift
  if [[ -f runs/$name/results.json ]]; then echo "== skip $name (done)"; return; fi
  mkdir -p runs/$name
  echo "== start $name: $*  ($(date))"
  $PY train_vit.py --run-name "$name" $COMMON "$@" 2>&1 | tee runs/$name/train.log
  [[ -f runs/$name/results.json ]] || echo "!! $name FAILED (see runs/$name/train.log)"
}

run bf1_slvit_lr1e-3  --model slvit $RECIPE --epochs 100 --lr 1e-3
run bf1_slvit_lr5e-4  --model slvit $RECIPE --epochs 100 --lr 5e-4
echo "== lr-check finished $(date)"
