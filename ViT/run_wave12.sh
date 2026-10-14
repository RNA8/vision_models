#!/bin/bash
# Wave 12 (2026-10-14 06:20): does self-distillation generalise beyond CIFAR-100, plus a Tiny-ImageNet seed.
#  1. CIFAR-10: 2.75M SL-ViT+SPT distilled from the ensemble of our 3 CIFAR-10 seeds (ensemble 97.28; seeds 96.75 +- 0.26),
#     function matching, mix on throughout (same arm that gave CIFAR-100 82.88 vs 80.91).
#  2. Tiny-ImageNet: born-again (single teacher = our 66.16 model), same arm. Tests whether a 1-teacher version helps.
#  3. Tiny-ImageNet 1000-ep seed 1 for the 66.16 headline.
# Smoke-tested 1 epoch each (teacher accuracies 97.28 / 66.16 reproduced). Gated at 4 jobs.
#   DRY=1 bash run_wave12.sh
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
MAXJOBS=${MAXJOBS:-4}
DRY=${DRY:-0}
R=$PWD/runs
bf3b() { echo "--batch-size 128 --lr 5e-4 --weight-decay 0.05 --epochs $1 --warmup-epochs 30 --randaug \
  --label-smoothing 0.1 --drop-path 0.1 --reprob 0.25 --init impulse --mix-off-epoch $(( $1 * 583 / 1000 )) --pool seqpool --lsa"; }
TV="--wandb --eval-every 10 --cudagraphs"
FM="--mix-off-epoch 0 --kd soft --kd-tau 1 --kd-alpha 1.0"
TIN="--dataset tiny_imagenet --patch-size 8 --model slvit --spt"

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
launch c10_kd_ens3_fm_1000ep train_vit.py $TV --dataset cifar10 --model slvit --spt $(bf3b 1000) $FM \
  --teacher $R/c10_spt_slvit_1000ep,$R/c10_spt_slvit_1000ep_s1,$R/c10_spt_slvit_1000ep_s2
launch tin_kd_ba_fm_1000ep train_vit.py $TV $TIN $(bf3b 1000) $FM --teacher $R/tin_spt_slvit_p8_1000ep
launch tin_spt_slvit_p8_1000ep_s1 train_vit.py $TV $TIN $(bf3b 1000) --seed 1
[[ $DRY == 1 ]] || wait
echo "== run_wave12 finished $(date)"
