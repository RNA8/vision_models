#!/usr/bin/env bash
# Recurrent-depth ViT on CIFAR-100: the CIFAR-10 recurrent recipe unchanged (batch 256, lr 1e-3, 100 ep, RandAugment,
# LSA, SeqPool, add injection), plus the parameter-matched flat control. CIFAR-10 results: rd_deep_r8_add_ds 85.23,
# rd_deep_r8_add 84.42, rd_flat6 83.48. Two queues share the GPU; eval_recurrent.py (depth sweep, exits) after each.
#   bash ViT/run_c100_recurrent.sh a   # c100_rd_r8_add_ds                       ~35-45 min
#   bash ViT/run_c100_recurrent.sh b   # c100_rd_flat6, then c100_rd_r8_add      ~35-45 min
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
COMMON="--wandb --eval-every 5 --dataset cifar100 --randaug --batch-size 256 --lr 1e-3 --epochs 100 --lsa --pool seqpool --cudagraphs"
REF=""; [[ -f runs/c100_r18_full/results.json ]] && REF="--ref-run-dir runs/c100_r18_full"

run() {  # run <name> <args...>
  local name=$1; shift
  if [[ ! -f runs/$name/results.json ]]; then
    mkdir -p runs/$name
    echo "== start $name: $*  ($(date))"
    $PY train_vit.py --run-name "$name" $COMMON "$@" 2>&1 | tee runs/$name/train.log
  fi
  if [[ -f runs/$name/results.json && ! -f runs/$name/recurrent_eval.json ]]; then
    echo "== eval_recurrent $name  ($(date))"
    $PY eval_recurrent.py runs/$name $REF 2>&1 | tee runs/$name/eval_recurrent.log
  fi
  [[ -f runs/$name/results.json ]] || echo "!! $name FAILED (see runs/$name/train.log)"
}

case "${1:-a}" in
  a) run c100_rd_r8_add_ds --model rd_tiny4 --rd-inject add --rd-r-mean 8 --rd-k-backprop 4 --rd-coda-every-step ;;
  b) run c100_rd_flat6     --model rd_flat6
     run c100_rd_r8_add    --model rd_tiny4 --rd-inject add --rd-r-mean 8 --rd-k-backprop 4 ;;
esac
echo "== c100-recurrent $1 finished $(date)"
