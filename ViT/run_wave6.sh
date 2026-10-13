#!/bin/bash
# Wave 6 (2026-10-13 03:50). From-scratch length is saturated (C100 SPT 2000 ep 80.98 vs 1000 ep 80.91 mean), so:
#  1. the 2.75M SL-ViT+SPT shape on CIFAR-10 (the C10 headline uses the 5.4M shape);
#  2. ImageNet-21k-P pretraining of the 2.75M SL-ViT+SPT shape (same recipe as in21kpre_tiny_p8, 30 ep at 64 px), then the best
#     KD fine-tune at patch 8 / 128 px (256 tokens; the SPT projection is a Linear, so the conv PI-resize to patch 4 does not apply).
#   DRY=1 bash run_wave6.sh
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
MAXJOBS=${MAXJOBS:-4}
DRY=${DRY:-0}
BF3B1000="--batch-size 128 --lr 5e-4 --weight-decay 0.05 --epochs 1000 --warmup-epochs 30 --randaug \
  --label-smoothing 0.1 --drop-path 0.1 --reprob 0.25 --init impulse --mix-off-epoch 583 --pool seqpool --lsa"
TV="--wandb --eval-every 10 --cudagraphs"
PRE="--dataset in21k64 --model slvit --spt --patch-size 8 --batch-size 256 --lr 1e-3 --epochs 30 --warmup-epochs 1 --randaug
     --mixup 0 --cutmix 0 --drop-path 0 --label-smoothing 0.1 --init impulse --pool seqpool --lsa
     --eval-every 5 --ckpt-every 2 --cudagraphs --wandb"
FT21="--wandb --dataset cifar100 --epochs 60 --batch-size 256 --warmup-epochs 3 --weight-decay 0.05 --randaug --drop-path 0.1 --ema 0.999 --eval-every 5"
KD="--teacher runs/ec100_T_vitb_ar2_224 --teacher-img-size 224 --kd soft --kd-alpha 0.5 --kd-tau 2"
gate() { sleep $((RANDOM % 30)); until (( $(ps -eo args= | grep -cE '^\S*python\S* (\S*/)?(train_vit|finetune)\.py') < MAXJOBS )); do sleep 60; done; }
done_() { [[ -f runs/$1/results.json ]]; }
run1() {  # run1 <name> <script.py> <args...>  -- foreground (for chains)
  local name=$1 script=$2; shift 2
  if [[ $DRY == 1 ]]; then echo "DRY: $PY $script --run-name $name $*"; return; fi
  done_ $name && { echo "== skip $name (done)"; return; }
  gate; mkdir -p runs/$name; echo "== start $name ($(date))"
  $PY $script --run-name "$name" "$@" 2>&1 | tee -a runs/$name/train.log
  done_ $name || echo "!! $name FAILED"
}
( run1 c10_spt_slvit_1000ep train_vit.py $TV --dataset cifar10 --model slvit --spt $BF3B1000 ) &
[[ $DRY == 1 ]] || sleep 150
( run1 in21kpre_slvit_spt_p8 train_vit.py $PRE
  [[ $DRY == 1 ]] || done_ in21kpre_slvit_spt_p8 || exit 1
  run1 ec100_in21kpre_slvit_spt_p8_128_kd_60ep finetune.py $FT21 --init-ckpt runs/in21kpre_slvit_spt_p8 --lr 5e-4 --img-size 128 $KD ) &
wait
echo "== run_wave6 finished $(date)"
