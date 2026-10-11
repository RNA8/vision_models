#!/usr/bin/env bash
# C2: recurrent-depth ablations on CIFAR-100 (100 ep), same recipe as runs/c100_rd_r8_add.
#   bash ViT/run_c100_c2.sh a|b|c   (3 queues; each job waits for a free GPU slot, <4 jobs total)
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
COMMON="--wandb --eval-every 5 --dataset cifar100 --randaug --batch-size 256 --lr 1e-3 --epochs 100 --lsa --pool seqpool --cudagraphs"
RD="--model rd_tiny4 --rd-inject add --rd-r-mean 8 --rd-k-backprop 4"
REF=""; [[ -f runs/c100_r18_full/results.json ]] && REF="--ref-run-dir runs/c100_r18_full"

run() {  # run <variant> <args...>
  local name=c100_rd_c2_$1; shift
  if [[ ! -f runs/$name/results.json ]]; then
    until (( $(pgrep -fc 'python (train_vit|finetune)\.py') < 4 )); do sleep 60; done
    mkdir -p runs/$name
    echo "== start $name: $*  ($(date))"
    $PY train_vit.py --run-name "$name" $COMMON "$@" 2>&1 | tee runs/$name/train.log
  fi
  [[ -f runs/$name/results.json ]] || echo "!! $name FAILED (see runs/$name/train.log)"
}

case "${1:-a}" in
  a) run lora8 $RD --rd-lora-rank 8
     run sandwich_concat $RD --rd-inject concat --rd-sandwich ;;
  b) run sandwich_add $RD --rd-sandwich
     run lora32 $RD --rd-lora-rank 32 ;;
  c) run base_rerun $RD
     run concat $RD --rd-inject concat
     run flat20 --model rd_flat20 ;;
esac
echo "== c2 $1 finished $(date)"
