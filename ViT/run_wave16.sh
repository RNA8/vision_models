#!/bin/bash
# Wave 16 (2026-10-15 00:40): does longer IN-21k pretraining lift the CIFAR-100 fine-tune?
#  Waits for in21kpre_tiny_p8_e60 (60-ep IN-21k-P pretrain), then runs the same 200-ep KD fine-tune
#  as ec100_in21kpre_tiny_p4_64_kd_200ep (30-ep pretrain init: 88.14). One variable: pretraining length.
# Gate uses flock (shared runs/.gate.lock).
#   DRY=1 bash run_wave16.sh
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
MAXJOBS=${MAXJOBS:-4}
DRY=${DRY:-0}
LOCK=$PWD/runs/.gate.lock
TEACH=runs/ec100_T_vitb_ar2_224
FT200="--wandb --dataset cifar100 --epochs 200 --batch-size 256 --warmup-epochs 3 --weight-decay 0.05 --randaug --drop-path 0.1 --ema 0.999 --eval-every 5"
KD="--teacher $TEACH --teacher-img-size 224 --kd soft --kd-alpha 0.5 --kd-tau 2"
I21E60="--init-ckpt runs/in21kpre_tiny_p8_e60 --lr 5e-4 --img-size 64 --patch-size 4"

njobs() { ps -eo args= | grep -cE '^\S*python\S* (\S*/)?(train_vit|finetune)\.py'; }
done_() { [[ -f runs/$1/results.json ]]; }
launch() {
  local name=$1 script=$2; shift 2
  if [[ $DRY == 1 ]]; then echo "DRY: $PY $script --run-name $name $*"; return; fi
  done_ $name && { echo "== skip $name (done)"; return; }
  while true; do
    exec 9>"$LOCK"; flock 9
    if (( $(njobs) < MAXJOBS )); then
      mkdir -p runs/$name; echo "== start $name ($(date))"
      ( $PY $script --run-name "$name" "$@" 2>&1 | tee -a runs/$name/train.log
        done_ $name || echo "!! $name FAILED" ) &
      sleep 90; flock -u 9; exec 9>&-; break
    fi
    flock -u 9; exec 9>&-; sleep 60
  done
}

[[ $DRY == 1 ]] || until done_ in21kpre_tiny_p8_e60; do sleep 120; done
launch ec100_in21kpre_e60_tiny_p4_64_kd_200ep finetune.py $FT200 $I21E60 $KD
[[ $DRY == 1 ]] || wait
echo "== run_wave16 finished $(date)"
