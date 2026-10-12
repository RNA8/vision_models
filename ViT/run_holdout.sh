#!/bin/bash
# Held-out check of the two headline recipes (chosen while watching the CIFAR test set): train on 45k, select/monitor
# on a fixed 5k held-out train split, score the test set once at the end (final_test_acc). If held-out val and test
# agree, recipe selection did not overfit the test set.   DRY=1 bash run_holdout.sh
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
MAXJOBS=${MAXJOBS:-4}
DRY=${DRY:-0}
BF3B1000="--batch-size 128 --lr 5e-4 --weight-decay 0.05 --epochs 1000 --warmup-epochs 30 --randaug \
  --label-smoothing 0.1 --drop-path 0.1 --reprob 0.25 --init impulse --mix-off-epoch 583 --pool seqpool --lsa"
TV="--wandb --eval-every 10 --cudagraphs --holdout 5000"
gate() { sleep $((RANDOM % 30)); until (( $(ps -eo args= | grep -cE '^\S*python\S* (\S*/)?(train_vit|finetune)\.py') < MAXJOBS )); do sleep 60; done; }
done_() { [[ -f runs/$1/results.json ]]; }
launch() {
  local name=$1; shift
  if [[ $DRY == 1 ]]; then echo "DRY: $PY train_vit.py --run-name $name $*"; return; fi
  done_ $name && { echo "== skip $name (done)"; return; }
  gate; mkdir -p runs/$name; echo "== start $name ($(date))"
  ( $PY train_vit.py --run-name "$name" "$@" 2>&1 | tee -a runs/$name/train.log
    done_ $name || echo "!! $name FAILED" ) &
  sleep 120
}
launch c100_spt_slvit_1000ep_holdout    $TV --dataset cifar100 --model slvit --spt $BF3B1000
launch c10_spt_tinyshape_1000ep_holdout $TV --dataset cifar10  --model tiny  --spt $BF3B1000
[[ $DRY == 1 ]] || wait
echo "== run_holdout finished $(date)"
