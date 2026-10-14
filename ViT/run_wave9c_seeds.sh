#!/bin/bash
# Wave 9c (2026-10-14): third seeds for wave 9. Seed noise at 300 ep is large (ms_shared s0 76.42, s1 77.48),
# so baseline, multiscale (shared) and the 21-register control each get seed 2.
#   DRY=1 bash run_wave9c_seeds.sh
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
MAXJOBS=${MAXJOBS:-4}
DRY=${DRY:-0}
BF3B300="--batch-size 128 --lr 5e-4 --weight-decay 0.05 --epochs 300 --warmup-epochs 30 --randaug \
  --label-smoothing 0.1 --drop-path 0.1 --reprob 0.25 --init impulse --mix-off-epoch 175 --pool seqpool --lsa"
TV="--wandb --eval-every 10 --cudagraphs"
C="--dataset cifar100 --model slvit --spt $BF3B300"

gate() { sleep $((RANDOM % 30)); until (( $(ps -eo args= | grep -cE '^\S*python\S* (\S*/)?(train_vit|finetune)\.py') < MAXJOBS )); do sleep 60; done; }
done_() { [[ -f runs/$1/results.json ]]; }
launch() {
  local name=$1 script=$2; shift 2
  if [[ $DRY == 1 ]]; then echo "DRY: $PY $script --run-name $name $*"; return; fi
  done_ $name && { echo "== skip $name (done)"; return; }
  gate; mkdir -p runs/$name; echo "== start $name ($(date))"
  ( $PY $script --run-name "$name" "$@" 2>&1 | tee -a runs/$name/train.log
    done_ $name || echo "!! $name FAILED" ) &
  sleep 120
}
launch c100_spt_slvit_300ep_s2     train_vit.py $TV $C --seed 2
launch c100_ms_shared_300ep_s2    train_vit.py $TV $C --multiscale --seed 2
launch c100_reg21_300ep_s2        train_vit.py $TV $C --reg-tokens 21 --seed 2
[[ $DRY == 1 ]] || wait
echo "== run_wave9c_seeds finished $(date)"
