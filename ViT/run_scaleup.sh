#!/usr/bin/env bash
# Scale-up: ViT-Small full recipe (200 ep) + patch-size comparison on ViT-Tiny (100 ep).
# Patch size 4 reuses the ablation run t5_droppath (identical config).
#   tmux new -s vit 'bash ViT/run_scaleup.sh'
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
COMMON="--wandb --eval-every 5 --randaug"

run() {  # run <name> <args...>
  local name=$1; shift
  if [[ -f runs/$name/results.json ]]; then echo "== skip $name (done)"; return; fi
  mkdir -p runs/$name
  echo "== start $name: $*  ($(date))"
  $PY train_vit.py --run-name "$name" $COMMON "$@" 2>&1 | tee runs/$name/train.log
  [[ -f runs/$name/results.json ]] || echo "!! $name FAILED (see runs/$name/train.log)"
}

run s_small_200ep  --model small --epochs 200
run p8_tiny        --model tiny --epochs 100 --patch-size 8
run p2_tiny        --model tiny --epochs 100 --patch-size 2
echo "== scale-up finished $(date)"
