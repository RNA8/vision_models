#!/bin/bash
# Wave 10 (2026-10-13). Teacher distillation / self-distillation screens for the 2.75M SL-ViT+SPT student on
# CIFAR-100 (plan: research/self_distillation.md, E1 and E4). 300-epoch screens, bf3b recipe; baseline
# c100_spt_slvit_300ep = 76.70. Teacher = prob-average ensemble of the three 1000-ep seeds (83.23 test).
# The teacher is used at train time only (3 x 2.75M); the deployed student stays 2.75M.
# Gated at 4 training processes; skips runs that already have results.json.
#   DRY=1 bash run_wave10_selfkd.sh
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
# E1-A: ensemble teacher, soft KL tau 1, alpha 0.5 (with LS-0.1 label CE), mix on all run
launch c100_kd_ens3_soft_a05_300ep train_vit.py $TV $C100 $(bf3b 300) $MIXON $ENS3 --kd soft --kd-tau 1 --kd-alpha 0.5
# E1-B: function matching (labels off)
launch c100_kd_ens3_fm_300ep       train_vit.py $TV $C100 $(bf3b 300) $MIXON $ENS3 --kd soft --kd-tau 1 --kd-alpha 1.0
# control: no teacher, mix on all run (isolates the mix-schedule change)
launch c100_mixon_300ep            train_vit.py $TV $C100 $(bf3b 300) $MIXON
# E4: self-KD, EMA-of-student teacher (decay 0.999/step), PS-KD alpha ramp 0 -> 0.8; bf3b mix-off kept
launch c100_selfkd_ema_300ep       train_vit.py $TV $C100 $(bf3b 300) --self-kd ema --self-kd-decay 0.999 --self-kd-alpha 0.8
[[ $DRY == 1 ]] || wait
echo "== run_wave10_selfkd finished $(date)"
