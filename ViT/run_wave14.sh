#!/bin/bash
# Wave 14 (2026-10-14 19:10): edge Pareto for the self-distillation recipe + robustness of the headline.
#  1. Third seed of the CIFAR-100 headline (ens3 function matching; seeds 0/1: 82.88 / 82.85).
#  2. Smaller distilled students, same teacher, 1000 ep: width 128 (8 heads, ~1.2M) and width 96 (6 heads, ~0.7M).
#     Question: how much accuracy does the recipe keep at 1/2 and 1/4 of the 2.75M budget?
#  3. Tiny-ImageNet with a 2-model ensemble teacher (tin 1000-ep seeds 0+1), after seed 1 finishes.
# Gate uses flock so concurrent queue scripts can't both launch into the same free slot.
#   DRY=1 bash run_wave14.sh
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
launch c100_kd_ens3_fm_1000ep_s2 train_vit.py $TV $C100 $(bf3b 1000) $FM --teacher $ENS3 --seed 2
launch c100_kd_ens3_fm_d128_1000ep train_vit.py $TV $C100 --embed-dim 128 --heads 8 $(bf3b 1000) $FM --teacher $ENS3
launch c100_kd_ens3_fm_d96_1000ep  train_vit.py $TV $C100 --embed-dim 96 --heads 6 $(bf3b 1000) $FM --teacher $ENS3
if [[ $DRY != 1 ]]; then until done_ tin_spt_slvit_p8_1000ep_s1; do sleep 120; done; fi
launch tin_kd_ens2_fm_1000ep train_vit.py $TV $TIN $(bf3b 1000) $FM --teacher $R/tin_spt_slvit_p8_1000ep,$R/tin_spt_slvit_p8_1000ep_s1
[[ $DRY == 1 ]] || wait
echo "== run_wave14 finished $(date)"
