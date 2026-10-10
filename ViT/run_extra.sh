#!/usr/bin/env bash
# Follow-up CIFAR-10 runs queued after the ablation/scale-up chain.
#   t5c_bs256:            full recipe at batch 256 (4x the optimizer steps per epoch), LR scaled
#                         2e-3 -> 1e-3 (sqrt rule). Tests whether t5's underfitting is a step-count issue.
#   s_small_200ep_lr1e-3: rerun of s_small_200ep, whose loss plateaued at ~2.1 (58% val) with peak
#                         LR 2e-3; DeiT scaling (5e-4 * bs/512) gives 1e-3 for ViT-Small at batch 1024.
#   tmux new -s vit 'bash ViT/run_extra.sh'
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

run t5c_bs256             --model tiny --epochs 100 --batch-size 256 --lr 1e-3
run s_small_200ep_lr1e-3  --model small --epochs 200 --lr 1e-3
echo "== extra finished $(date)"
