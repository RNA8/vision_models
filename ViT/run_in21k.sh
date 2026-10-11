#!/usr/bin/env bash
# B4: pretrain our ViT-Tiny (p8, 64 tokens, LSA + SeqPool + impulse init, 5.6M params) on ImageNet-21K-P at 64 px
# (10,450 classes, 11.06M images, streamed from data/in21k64_shards by vit_data/in21k64.py), then the four CIFAR-100
# fine-tunes that mirror the IN-64 ones (ec100_in64pre_tiny_{64,64_kd,p4_64,p4_64_kd}: 81.17 / 81.47 / 83.78 / 84.08).
# Recipe = in64pre_tiny_p8 (run_next.sh pre) with warmup 1 ep (an epoch is 8.6x longer); rationale: research/in21k_pretrain.md.
#   tmux new-window -t vit -n in21k_pre 'cd .../ViT && bash run_in21k.sh 2>&1 | tee runs/run_in21k.out'
#   DRY=1 bash run_in21k.sh   # print commands, no waiting
# WAITS (poll 5 min) until the long-run queue's main runs have results.json, then gates on < MAXJOBS train/finetune procs.
# Resumable: train_vit.py resumes from runs/in21kpre_tiny_p8/ckpt.pt (every CKPT epochs) -> just re-run the script.
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
MAXJOBS=${MAXJOBS:-4}
DRY=${DRY:-0}
EPOCHS=${EPOCHS:-30}
NAME=in21kpre_tiny_p8

PRE="--dataset in21k64 --model tiny --patch-size 8 --batch-size 256 --lr 1e-3 --epochs $EPOCHS --warmup-epochs 1 --randaug
     --mixup 0 --cutmix 0 --drop-path 0 --label-smoothing 0.1 --init impulse --pool seqpool --lsa
     --eval-every 5 --ckpt-every 2 --cudagraphs --wandb"
FT="--wandb --dataset cifar100 --epochs 30 --batch-size 256 --warmup-epochs 3 --weight-decay 0.05 --randaug
    --drop-path 0.1 --ema 0.999 --eval-every 5"
TEACH=runs/ec100_T_vitb_ar2_224
KD="--teacher $TEACH --teacher-img-size 224 --kd soft --kd-alpha 0.5 --kd-tau 2"

WAIT_FOR="c100_tinyshape_bf3b_1000ep c100_rd_r8_add_500ep c100_rd_flat6_500ep c100_bf3b_slvit_1000ep c10_tinyshape_bf3b_1000ep"
done_() { [[ -f runs/$1/results.json ]]; }
all_waited() { for r in $WAIT_FOR; do done_ $r || return 1; done; }
gate() { sleep $((RANDOM % 30)); until (( $(ps -eo args= | grep -cE '^\S*python\S* (\S*/)?(train_vit|finetune)\.py') < MAXJOBS )); do sleep 60; done; }

if [[ $DRY != 1 ]]; then
  echo "== waiting for the long-run main runs ($WAIT_FOR) ($(date))"
  until all_waited; do sleep 300; done
  echo "== long runs done, starting ($(date))"
fi

train() {
  local name=$1; shift
  if [[ $DRY == 1 ]]; then echo "DRY: $PY train_vit.py --run-name $name $*"; return; fi
  done_ $name && { echo "== skip $name"; return; }
  gate; mkdir -p runs/$name; echo "== start $name ($(date))"
  $PY train_vit.py --run-name "$name" "$@" 2>&1 | tee -a runs/$name/train.log
  done_ $name || echo "!! $name FAILED"
}
ft() {
  local name=$1; shift
  if [[ $DRY == 1 ]]; then echo "DRY: $PY finetune.py --run-name $name $FT $*"; return; fi
  if ! done_ $name; then
    gate; mkdir -p runs/$name; echo "== start $name ($(date))"
    $PY finetune.py --run-name "$name" $FT "$@" 2>&1 | tee -a runs/$name/train.log
  fi
  if done_ $name && [[ ! -f runs/$name/edge_bench.json ]]; then
    CUDA_VISIBLE_DEVICES= $PY edge_bench.py runs/$name --no-gpu --out runs/$name/edge_bench.json 2>&1 | tail -n 1
  fi
  done_ $name || echo "!! $name FAILED"
}

train $NAME $PRE
[[ $DRY == 1 ]] || done_ $NAME || { echo "pretraining did not finish; no fine-tunes"; exit 1; }
I="--init-ckpt runs/$NAME --lr 5e-4"
ft ec100_in21kpre_tiny_64       $I --img-size 64
ft ec100_in21kpre_tiny_64_kd    $I --img-size 64 $KD
ft ec100_in21kpre_tiny_p4_64    $I --img-size 64 --patch-size 4
ft ec100_in21kpre_tiny_p4_64_kd $I --img-size 64 --patch-size 4 $KD
echo "== run_in21k finished $(date)"
