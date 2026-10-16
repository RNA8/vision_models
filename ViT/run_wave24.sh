#!/bin/bash
# Wave 24 (2026-10-16 10:40): re-queue the two runs wave 23 skipped, now that the GPU is lightly loaded (runs are ~1.6x faster).
# Every launch is deadline-guarded: a run starts only if (now + its estimated minutes under load) < DEADLINE,
# so nothing is still training when backup + archive start (~14:00 UTC).
# Deadline 14:00 UTC (backup + archive afterwards, user back ~14:40). Estimates are for light load.
# Gate uses flock (shared runs/.gate.lock).
#   DRY=1 bash run_wave24.sh
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
MAXJOBS=${MAXJOBS:-4}
DRY=${DRY:-0}
R=$PWD/runs
LOCK=$PWD/runs/.gate.lock
bf3b() { echo "--batch-size 128 --lr 5e-4 --weight-decay 0.05 --epochs $1 --warmup-epochs 30 --randaug \
  --label-smoothing 0.1 --drop-path 0.1 --reprob 0.25 --init impulse --mix-off-epoch $(( $1 * 583 / 1000 )) --pool seqpool --lsa"; }
TV="--wandb --eval-every 10 --cudagraphs"
C100="--dataset cifar100 --model slvit --spt"
TIN="--dataset tiny_imagenet --patch-size 8 --model slvit --spt"
FM="--mix-off-epoch 0 --kd soft --kd-tau 1 --kd-alpha 1.0"
ENS3="$R/c100_spt_slvit_1000ep,$R/c100_spt_slvit_1000ep_s1,$R/c100_spt_slvit_1000ep_s2"

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
cnn() { echo "--opt sgd --momentum 0.9 --nesterov --batch-size 128 --lr $2 --weight-decay $3 --grad-clip 0 \
  --epochs $1 --warmup-epochs $4 --randaug --label-smoothing 0.1 --drop-path 0.1 --reprob 0.25 \
  --mix-off-epoch $(( $1 * 583 / 1000 ))"; }

DEADLINE=${DEADLINE:-$(date -u -d "2026-10-16 14:00" +%s)}
# guarded <name> <est_minutes> <script> <args...>
guarded() {
  local name=$1 est=$2; shift 2
  if [[ $DRY == 1 ]]; then echo "DRY ($est min): $name"; return; fi
  done_ $name && { echo "== skip $name (done)"; return; }
  while true; do
    exec 9>"$LOCK"; flock 9
    if (( $(njobs) < MAXJOBS )); then
      if (( $(date +%s) + est * 60 > DEADLINE )); then
        echo "== skip $name: would finish after the deadline ($(date -u))"; flock -u 9; exec 9>&-; return
      fi
      mkdir -p runs/$name; echo "== start $name ($(date))"
      ( $PY "$1" --run-name "$name" "${@:2}" 2>&1 | tee -a runs/$name/train.log
        done_ $name || echo "!! $name FAILED" ) &
      sleep 90; flock -u 9; exec 9>&-; return
    fi
    flock -u 9; exec 9>&-; sleep 60
  done
}
R18C="--model resnet18 --r18-stem cifar --r18-width 32"
guarded c100_kd_ens3_fm_d96_1000ep_s2 195 train_vit.py $TV $C100 --embed-dim 96 --heads 6 $(bf3b 1000) $FM --teacher $ENS3 --seed 2
guarded c100_r18w32_kd_ens3_fm_1000ep_s1 195 train_vit.py $TV --dataset cifar100 $R18C $(cnn 1000 0.1 5e-4 10) $FM --teacher $ENS3 --seed 1
[[ $DRY == 1 ]] || wait
echo "== run_wave24 finished $(date)"
