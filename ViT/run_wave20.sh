#!/bin/bash
# Wave 20 (2026-10-15 22:05): keep the GPU full; firm up the Tiny-ImageNet headline and add the CIFAR-100 CNN-student control.
#  1. Seed 1 of the Tiny-ImageNet 2-copy distillation (seed 0: 69.66; 1 seed is not enough for a headline, and an 8.65M
#     ResNet-18 sits at 69.81).
#  2. CIFAR-100 counterpart of wave 19: the ~2.8M ResNet-18 (width 32) distilled from the same 3-seed ViT ensemble used for
#     the 82.77 ViT student. Pairs with c100_r18w32_sgd_1000ep (no teacher). One variable: student architecture.
# Gate uses flock (shared runs/.gate.lock).
#   DRY=1 bash run_wave20.sh
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


launch tin_kd_ens2_fm_1000ep_s1 train_vit.py $TV $TIN $(bf3b 1000) $FM --teacher $R/tin_spt_slvit_p8_1000ep,$R/tin_spt_slvit_p8_1000ep_s1 --seed 1
launch c100_r18w32_kd_ens3_fm_1000ep train_vit.py $TV --dataset cifar100 --model resnet18 --r18-stem cifar --r18-width 32 $(cnn 1000 0.1 5e-4 10) $FM --teacher $ENS3
[[ $DRY == 1 ]] || wait
echo "== run_wave20 finished $(date)"
