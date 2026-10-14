#!/bin/bash
# Wave 13 (2026-10-14 12:50): push the self-distillation result (82.86, 2 seeds; teacher ens3 = 83.23).
# The student ends ~0.4 below its teacher, and function matching is known to keep paying with longer schedules
# (Beyer et al. 2106.05237), so test the two levers separately on CIFAR-100, same arm (FM, labels off, mix on):
#  1. stronger teacher: 4-member ensemble incl. the 2000-ep run (ensemble 83.57), 1000 ep;
#  2. longer schedule: 3-member teacher, 2000 ep.
#   DRY=1 bash run_wave13.sh
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
MAXJOBS=${MAXJOBS:-4}
DRY=${DRY:-0}
R=$PWD/runs
bf3b() { echo "--batch-size 128 --lr 5e-4 --weight-decay 0.05 --epochs $1 --warmup-epochs 30 --randaug \
  --label-smoothing 0.1 --drop-path 0.1 --reprob 0.25 --init impulse --mix-off-epoch $(( $1 * 583 / 1000 )) --pool seqpool --lsa"; }
TV="--wandb --eval-every 10 --cudagraphs"
C100="--dataset cifar100 --model slvit --spt"
FM="--mix-off-epoch 0 --kd soft --kd-tau 1 --kd-alpha 1.0"
ENS3="$R/c100_spt_slvit_1000ep,$R/c100_spt_slvit_1000ep_s1,$R/c100_spt_slvit_1000ep_s2"

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
launch c100_kd_ens4_fm_1000ep train_vit.py $TV $C100 $(bf3b 1000) $FM --teacher $ENS3,$R/c100_spt_slvit_2000ep
launch c100_kd_ens3_fm_2000ep train_vit.py $TV $C100 $(bf3b 2000) $FM --teacher $ENS3
[[ $DRY == 1 ]] || wait
echo "== run_wave13 finished $(date)"
