#!/bin/bash
# Wave 3 (2026-10-12 18:50): seeds for the new CIFAR-10 SPT result (96.86), a longer IN-21k fine-tune of the best
# 64 px pure ViT (IN-64: 30 -> 60 ep gave +1.9), and a seed for the 200-ep RepViT KD edge result.
#   DRY=1 bash run_wave3.sh
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
MAXJOBS=${MAXJOBS:-4}
DRY=${DRY:-0}
bf3b() { echo "--batch-size 128 --lr 5e-4 --weight-decay 0.05 --epochs $1 --warmup-epochs 30 --randaug \
  --label-smoothing 0.1 --drop-path 0.1 --reprob 0.25 --init impulse --mix-off-epoch $(( $1 * 583 / 1000 )) --pool seqpool --lsa"; }
TV="--wandb --eval-every 10 --cudagraphs"
TEACH=runs/ec100_T_vitb_ar2_224
# fine-tune recipe of run_in21k.sh, 60 epochs
FT21="--wandb --dataset cifar100 --epochs 60 --batch-size 256 --warmup-epochs 3 --weight-decay 0.05 --randaug --drop-path 0.1 --ema 0.999 --eval-every 5"
KD="--teacher $TEACH --teacher-img-size 224 --kd soft --kd-alpha 0.5 --kd-tau 2"
FTE="--wandb --dataset cifar100 --batch-size 256 --weight-decay 0.05 --randaug --drop-path 0.1 --ema 0.999 --eval-every 5"
FM="--teacher $TEACH --teacher-img-size 224 --kd soft --kd-alpha 1 --kd-tau 1 --mixup 1.0 --cutmix 0 --mix-prob 1.0"

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
launch ec100_in21kpre_tiny_p4_64_kd_60ep finetune.py $FT21 --init-ckpt runs/in21kpre_tiny_p8 --lr 5e-4 --img-size 64 --patch-size 4 $KD
launch c10_spt_tinyshape_1000ep_s1 train_vit.py $TV --dataset cifar10 --model tiny --spt --seed 1 $(bf3b 1000)
launch c10_spt_tinyshape_1000ep_s2 train_vit.py $TV --dataset cifar10 --model tiny --spt --seed 2 $(bf3b 1000)
launch ec100_fm_repvit_224_e200_s1 finetune.py $FTE --epochs 200 --warmup-epochs 5 --model repvit_m0_9.dist_450e_in1k --img-size 224 --lr 5e-4 --seed 1 $FM
[[ $DRY == 1 ]] || wait
echo "== run_wave3 finished $(date)"
