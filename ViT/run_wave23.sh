#!/bin/bash
# Wave 23 (2026-10-16 06:45): final fill before the instance is shut down (user back ~14:40 UTC).
# Every launch is deadline-guarded: a run starts only if (now + its estimated minutes under load) < DEADLINE,
# so nothing is still training when backup + archive start (~13:30 UTC). Priority order:
#  1. 3rd seed of the 1.25M distilled student (81.39 / 81.41)        ~280 min
#  2. 2nd seed of the CIFAR-100 ResNet-18 width-32 control (78.59)     ~170 min
#  3. CIFAR-10 ResNet-18 width-32 control (ViT 2.75M: 96.75, 3 seeds) ~170 min
#  4. 3rd seed of the 0.71M distilled student (78.38 / 79.36)          ~280 min
#  5. 2nd seed of the CIFAR-100 ResNet-18 student distilled from ViTs  ~280 min
# Gate uses flock (shared runs/.gate.lock).
#   DRY=1 bash run_wave23.sh
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

DEADLINE=${DEADLINE:-$(date -u -d "2026-10-16 13:30" +%s)}
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
guarded c100_kd_ens3_fm_d128_1000ep_s2 280 train_vit.py $TV $C100 --embed-dim 128 --heads 8 $(bf3b 1000) $FM --teacher $ENS3 --seed 2
guarded c100_r18w32_sgd_1000ep_s1 170 train_vit.py $TV --dataset cifar100 $R18C $(cnn 1000 0.1 5e-4 10) --seed 1
guarded c10_r18w32_sgd_1000ep 170 train_vit.py $TV --dataset cifar10 $R18C $(cnn 1000 0.1 5e-4 10)
guarded c100_kd_ens3_fm_d96_1000ep_s2 280 train_vit.py $TV $C100 --embed-dim 96 --heads 6 $(bf3b 1000) $FM --teacher $ENS3 --seed 2
guarded c100_r18w32_kd_ens3_fm_1000ep_s1 280 train_vit.py $TV --dataset cifar100 $R18C $(cnn 1000 0.1 5e-4 10) $FM --teacher $ENS3 --seed 1
[[ $DRY == 1 ]] || wait
echo "== run_wave23 finished $(date)"
