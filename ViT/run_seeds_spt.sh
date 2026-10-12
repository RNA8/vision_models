#!/bin/bash
# Seeds for the two 1000-epoch headline results + SL-ViT's shifted patch tokenization (SPT, --spt).
# Starts after the ImageNet-21k queue (run_in21k.sh: pretrain + 4 CIFAR-100 fine-tunes) has exited.
#   DRY=1 bash run_seeds_spt.sh   # print the expanded commands
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
MAXJOBS=${MAXJOBS:-4}
DRY=${DRY:-0}

# same recipe as run_long.sh
BF3B1000="--batch-size 128 --lr 5e-4 --weight-decay 0.05 --epochs 1000 --warmup-epochs 30 --randaug \
  --label-smoothing 0.1 --drop-path 0.1 --reprob 0.25 --init impulse --mix-off-epoch 583 --pool seqpool --lsa"
BF3B300="--batch-size 128 --lr 5e-4 --weight-decay 0.05 --epochs 300 --warmup-epochs 30 --randaug \
  --label-smoothing 0.1 --drop-path 0.1 --reprob 0.25 --init impulse --mix-off-epoch 175 --pool seqpool --lsa"
TV="--wandb --eval-every 10 --cudagraphs"

in21k_alive() { ps -eo pid=,args= | grep -qE '^ *[0-9]+ (/bin/)?bash (\S*/)?run_in21k\.sh'; }
if [[ $DRY != 1 ]]; then
  echo "== waiting for run_in21k.sh to exit ($(date))"
  while in21k_alive; do sleep 300; done
  echo "== starting ($(date))"
fi
gate() { sleep $((RANDOM % 30)); until (( $(ps -eo args= | grep -cE '^\S*python\S* (\S*/)?(train_vit|finetune)\.py') < MAXJOBS )); do sleep 60; done; }
done_() { [[ -f runs/$1/results.json ]]; }
launch() {  # launch <name> <args...>
  local name=$1; shift
  if [[ $DRY == 1 ]]; then echo "DRY: $PY train_vit.py --run-name $name $*"; return; fi
  done_ $name && { echo "== skip $name (done)"; return; }
  gate; mkdir -p runs/$name; echo "== start $name ($(date))"
  ( $PY train_vit.py --run-name "$name" "$@" 2>&1 | tee -a runs/$name/train.log
    done_ $name || echo "!! $name FAILED" ) &
  sleep 120
}

launch c100_spt_slvit_1000ep          $TV --dataset cifar100 --model slvit --spt $BF3B1000
launch c100_bf3b_slvit_1000ep_s1      $TV --dataset cifar100 --model slvit --seed 1 $BF3B1000
launch c10_tinyshape_bf3b_1000ep_s1   $TV --dataset cifar10  --model tiny  --seed 1 $BF3B1000
launch c100_spt_slvit_300ep           $TV --dataset cifar100 --model slvit --spt $BF3B300
launch c100_bf3b_slvit_1000ep_s2      $TV --dataset cifar100 --model slvit --seed 2 $BF3B1000
launch c10_tinyshape_bf3b_1000ep_s2   $TV --dataset cifar10  --model tiny  --seed 2 $BF3B1000
[[ $DRY == 1 ]] || wait
echo "== run_seeds_spt finished $(date)"
