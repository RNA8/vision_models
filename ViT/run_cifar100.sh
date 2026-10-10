#!/usr/bin/env bash
# CIFAR-100 stage: the same ViT-Tiny ablation ladder as CIFAR-10 (one recipe change per run),
# a ResNet-18 control, the 300-epoch ViT-Tiny run and the 200-epoch ViT-Small run.
# No patch-size sweep. Run names are prefixed c100_. A run is skipped if its results.json
# exists, so rerunning after a failure only redoes the missing runs. Launch inside tmux:
#   tmux new -s c100 'bash ViT/run_cifar100.sh'
# Est. time: ~70-80 min at batch 256 with CUDA graphs (Tiny ~6 min/100 ep, Small ~14 min/200 ep).
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
# Batch 256 (CIFAR-10: batch 1024 underfits; t5 72% vs t5c_bs256 82%), LR scaled by sqrt rule
# (Tiny 2e-3@1024 -> 1e-3@256; Small 1e-3@1024 -> 5e-4@256), CUDA graphs for speed (~3.5 s/epoch).
COMMON="--wandb --eval-every 5 --dataset cifar100 --batch-size 256 --lr 1e-3 --cudagraphs"

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
run c100_s_small_200ep  --model small --epochs 200 --randaug --lr 5e-4
echo "== cifar100 finished $(date)"
