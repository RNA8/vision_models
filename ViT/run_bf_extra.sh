#!/usr/bin/env bash
# Follow-up to run_baseline_fix.sh: bf3 with SL-ViT's lighter mixing (mix-prob 0.5, mixup and
# cutmix alpha 1.0; our default is mix-prob 1.0, mixup 0.8). bf1 underfit at 100 ep (74% val vs
# 77% clean train), pointing at too much regularization. Single change vs bf3. ~31 min.
#   tmux new -s vit 'bash ViT/run_bf_extra.sh'
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
COMMON="--wandb --eval-every 5 --dataset cifar10 --cudagraphs"
RECIPE="--batch-size 128 --lr 3e-3 --weight-decay 0.05 --warmup-epochs 10 --randaug \
  --label-smoothing 0.1 --drop-path 0.1 --reprob 0.25"

run() {  # run <name> <args...>
  local name=$1; shift
  if [[ -f runs/$name/results.json ]]; then echo "== skip $name (done)"; return; fi
  mkdir -p runs/$name
  echo "== start $name: $*  ($(date))"
  $PY train_vit.py --run-name "$name" $COMMON "$@" 2>&1 | tee runs/$name/train.log
  [[ -f runs/$name/results.json ]] || echo "!! $name FAILED (see runs/$name/train.log)"
}

run bf4_slvit_300ep_mixhalf  --model slvit $RECIPE --epochs 300 --warmup-epochs 30 --init impulse \
                             --mix-off-epoch 175 --pool seqpool --lsa --mix-prob 0.5 --mixup 1.0 --cutmix 1.0
echo "== bf-extra finished $(date)"
