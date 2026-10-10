#!/usr/bin/env bash
# CIFAR-100 stage: the same ViT-Tiny ablation ladder as CIFAR-10 (one recipe change per run),
# a ResNet-18 control, the 300-epoch ViT-Tiny run and the 200-epoch ViT-Small run.
# No patch-size sweep. Run names are prefixed c100_. A run is skipped if its results.json
# exists, so rerunning after a failure only redoes the missing runs. Launch inside tmux:
#   tmux new -s c100 'bash ViT/run_cifar100.sh'
# Est. time: about the same as the CIFAR-10 ablations + the s_small run (same image count and
# size; only the 100-way head differs).
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
COMMON="--wandb --eval-every 5 --dataset cifar100"

run() {  # run <name> <args...>
  local name=$1; shift
  if [[ -f runs/$name/results.json ]]; then echo "== skip $name (done)"; return; fi
  mkdir -p runs/$name
  echo "== start $name: $*  ($(date))"
  $PY train_vit.py --run-name "$name" $COMMON "$@" 2>&1 | tee runs/$name/train.log
  [[ -f runs/$name/results.json ]] || echo "!! $name FAILED (see runs/$name/train.log)"
}

NOREG="--label-smoothing 0 --weight-decay 0 --mixup 0 --cutmix 0 --drop-path 0"
run c100_t1_plain       --model tiny --epochs 100 $NOREG
run c100_t2_ls_wd       --model tiny --epochs 100 --mixup 0 --cutmix 0 --drop-path 0
run c100_t3_randaug     --model tiny --epochs 100 --mixup 0 --cutmix 0 --drop-path 0 --randaug
run c100_t4_mix         --model tiny --epochs 100 --drop-path 0 --randaug
run c100_t5_droppath    --model tiny --epochs 100 --randaug
run c100_t6_full_300ep  --model tiny --epochs 300 --randaug
run c100_r18_full       --model resnet18 --epochs 100 --randaug
run c100_s_small_200ep  --model small --epochs 200 --randaug
echo "== cifar100 finished $(date)"
