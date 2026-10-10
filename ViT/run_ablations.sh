#!/usr/bin/env bash
# ViT-Tiny ablation ladder on CIFAR-10 (one recipe change per run) + ResNet-18 control.
# Runs back to back; a run is skipped if its results.json exists, so rerunning this script
# after a failure only redoes the missing runs.  Launch inside tmux:
#   tmux new -s vit 'bash ViT/run_ablations.sh'
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
COMMON="--wandb --eval-every 5"

run() {  # run <name> <args...>
  local name=$1; shift
  if [[ -f runs/$name/results.json ]]; then echo "== skip $name (done)"; return; fi
  mkdir -p runs/$name
  echo "== start $name: $*  ($(date))"
  $PY train_vit.py --run-name "$name" $COMMON "$@" 2>&1 | tee runs/$name/train.log
  [[ -f runs/$name/results.json ]] || echo "!! $name FAILED (see runs/$name/train.log)"
}

NOREG="--label-smoothing 0 --weight-decay 0 --mixup 0 --cutmix 0 --drop-path 0"
run t1_plain       --model tiny --epochs 100 $NOREG
run t2_ls_wd       --model tiny --epochs 100 --mixup 0 --cutmix 0 --drop-path 0
run t3_randaug     --model tiny --epochs 100 --mixup 0 --cutmix 0 --drop-path 0 --randaug
run t4_mix         --model tiny --epochs 100 --drop-path 0 --randaug
run t5_droppath    --model tiny --epochs 100 --randaug
run t6_full_300ep  --model tiny --epochs 300 --randaug
run r18_full       --model resnet18 --epochs 100 --randaug
echo "== ablations finished $(date)"
