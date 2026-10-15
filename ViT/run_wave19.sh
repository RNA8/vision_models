#!/bin/bash
# Wave 19 (2026-10-15 20:10): fair follow-up to the Tiny-ImageNet CNN control.
#  The 2.86M ResNet-18 (66.52) is level with the plain 2.9M ViT (66.26); the distilled ViT (69.66) used a teacher the
#  ResNet never had. Distill the same-size ResNet from the same 2-model ViT ensemble (68.98), same function-matching
#  setup (labels off, mixing on for the whole run), same CNN optimizer as tin_r18w32_sgd_1000ep. One variable: the
#  student architecture.
# Gate uses flock (shared runs/.gate.lock).
#   DRY=1 bash run_wave19.sh
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


launch tin_r18w32_kd_ens2_fm_1000ep train_vit.py $TV --dataset tiny_imagenet --model resnet18 --r18-stem cifar --r18-width 32 $(cnn 1000 0.1 5e-4 10) $FM --teacher $R/tin_spt_slvit_p8_1000ep,$R/tin_spt_slvit_p8_1000ep_s1
[[ $DRY == 1 ]] || wait
echo "== run_wave19 finished $(date)"
