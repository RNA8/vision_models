#!/usr/bin/env bash
# Deeper-loop recurrent ViTs. rd_pilot_a (r_mean 4) converged to a fixed point after ~3 loops: accuracy
# flat from r=3 to r=32, so extra test-time depth gave no gain. Train with more loops so that later
# iterations carry information. Same recipe as the pilot (batch 256, lr 1e-3, LSA, SeqPool). ~20-30 min
# each when sharing the GPU; eval_recurrent.py (depth sweep, exits, CIFAR-10H / CIFAR-10-C) after each.
#   rd_deep_r8:     2/2/2 blocks, r_mean 8, truncated backprop through the last 4 loops (Geiping-style)
#   rd_deep_r8_ds:  same + deep supervision (decode every loop, average logits) to make every loop useful
#   rd_deep_single: single-block core (2/1/2), r_mean 8 (rd_tiny8): deepest weight sharing
#   tmux new -s rd 'bash ViT/run_recurrent_deep.sh'
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
COMMON="--wandb --eval-every 5 --randaug --batch-size 256 --lr 1e-3 --epochs 100 --lsa --pool seqpool --cudagraphs"
REF=""; [[ -f runs/r18_full/results.json ]] && REF="--ref-run-dir runs/r18_full"

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

run rd_deep_r8      --model rd_tiny4 --rd-r-mean 8 --rd-k-backprop 4
run rd_deep_r8_ds   --model rd_tiny4 --rd-r-mean 8 --rd-k-backprop 4 --rd-coda-every-step
run rd_deep_single  --model rd_tiny8
echo "== recurrent-deep finished $(date)"
