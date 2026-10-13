#!/bin/bash
# Wave 7 (2026-10-13 05:00, overnight fill). Gated at 4 jobs; runs in this order as slots free up:
#  1. IN-21k ViT KD fine-tune 200 ep (30/60/120 ep: 84.75 / 86.89 / 87.82, still climbing) and a seed of the 120-ep run;
#  2. seeds 1-2 for the 2.75M SL-ViT+SPT CIFAR-10 run (seed 0 running in wave 6);
#  3. 60-epoch IN-21k-P pretraining of ViT-Tiny (30 ep gave 32.4% val; does longer pretraining lift the fine-tune?).
#   DRY=1 bash run_wave7.sh
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
FT200="${FT21/--epochs 60/--epochs 200}"; FT120="${FT21/--epochs 60/--epochs 120}"
I21="--init-ckpt runs/in21kpre_tiny_p8 --lr 5e-4 --img-size 64 --patch-size 4"
launch ec100_in21kpre_tiny_p4_64_kd_200ep    finetune.py $FT200 $I21 $KD
launch ec100_in21kpre_tiny_p4_64_kd_120ep_s1 finetune.py $FT120 $I21 --seed 1 $KD
launch c10_spt_slvit_1000ep_s1 train_vit.py $TV --dataset cifar10 --model slvit --spt --seed 1 $(bf3b 1000)
launch c10_spt_slvit_1000ep_s2 train_vit.py $TV --dataset cifar10 --model slvit --spt --seed 2 $(bf3b 1000)
launch in21kpre_tiny_p8_e60 train_vit.py --dataset in21k64 --model tiny --patch-size 8 --batch-size 256 --lr 1e-3 --epochs 60 \
  --warmup-epochs 1 --randaug --mixup 0 --cutmix 0 --drop-path 0 --label-smoothing 0.1 --init impulse --pool seqpool --lsa \
  --eval-every 5 --ckpt-every 2 --cudagraphs --wandb
[[ $DRY == 1 ]] || wait
echo "== run_wave7 finished $(date)"
