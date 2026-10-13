#!/bin/bash
# Wave 10b (2026-10-13 22:50). 1000-ep confirm of the best wave-10 KD arm (function matching, ens3 teacher,
# labels off, mix on all run; 300-ep screen 80.55 vs a=0.5 80.28 vs baseline 76.70), seeds 0 and 1.
# Success bar (research/self_distillation.md): 2-seed mean >= 82.3 (baseline 1000 ep 80.91 +- 0.26).
# Teacher = prob-average ensemble of the three 1000-ep seeds (83.23 test), train time only (3 x 2.75M);
# the deployed student stays 2.75M. Gated at 4 training processes; skips runs that already have results.json.
# Gated at 4 training processes; skips runs that already have results.json.
#   DRY=1 bash run_wave10b_kd1000.sh
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
MAXJOBS=${MAXJOBS:-4}
DRY=${DRY:-0}
bf3b() { echo "--batch-size 128 --lr 5e-4 --weight-decay 0.05 --epochs $1 --warmup-epochs 30 --randaug \
  --label-smoothing 0.1 --drop-path 0.1 --reprob 0.25 --init impulse --mix-off-epoch $(( $1 * 583 / 1000 )) --pool seqpool --lsa"; }
TV="--wandb --eval-every 10 --cudagraphs"
C100="--dataset cifar100 --model slvit --spt"
ENS3="--teacher c100_spt_slvit_1000ep,c100_spt_slvit_1000ep_s1,c100_spt_slvit_1000ep_s2"
MIXON="--mix-off-epoch 0"   # MixUp/CutMix for the whole run (overrides bf3b's mix-off at 58.3%)

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
FM="$MIXON $ENS3 --kd soft --kd-tau 1 --kd-alpha 1.0"
launch c100_kd_ens3_fm_1000ep    train_vit.py $TV $C100 $(bf3b 1000) $FM
launch c100_kd_ens3_fm_1000ep_s1 train_vit.py $TV $C100 $(bf3b 1000) $FM --seed 1
[[ $DRY == 1 ]] || wait
echo "== run_wave10b_kd1000 finished $(date)"
