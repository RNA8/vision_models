#!/bin/bash
# Wave 21 (2026-10-16 00:40): keep the GPU full after wave 20.
#  1. CIFAR-100 lr/wd screen for the ResNet-18 width-32 control (100 ep, 5k held-out train images; val untouched).
#     Removes the caveat that c100_r18w32_sgd_1000ep (78.59) used hyperparameters screened on Tiny-ImageNet.
#     If a different config wins by > 0.3 on held-out, rerun the 1000-ep control with it (decided by hand afterwards).
#  2. 1.25M distilled student at 2000 epochs: distillation kept gaining past 1000 ep at 2.75M (+0.4); does the edge model too?
# Gate uses flock (shared runs/.gate.lock).
#   DRY=1 bash run_wave21.sh
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


for lr in 0.05 0.1 0.2; do for wd in 5e-4 1e-3; do
  launch c100_r18w32_screen_lr${lr}_wd${wd} train_vit.py $TV --dataset cifar100 --model resnet18 --r18-stem cifar --r18-width 32 $(cnn 100 $lr $wd 5) --holdout 5000
done; done
launch c100_kd_ens3_fm_d128_2000ep train_vit.py $TV $C100 --embed-dim 128 --heads 8 $(bf3b 2000) $FM --teacher $ENS3
[[ $DRY == 1 ]] || wait
echo "== run_wave21 finished $(date)"
