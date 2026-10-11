#!/usr/bin/env bash
# MoE experiments (see research/moe_vision.md). Usage: bash run_moe.sh <queue>
#   cc128  : CondConv RepViT-M0.9 K=4,8 @128   cc224 : best-K @224 (pass K)   vit : 150ep c100 ViT dense/E4/E8
#   up     : upcycle in64pre ViT E=4 + dense controls
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
gate() { until (( $(pgrep -fc 'python (train_vit|finetune)\.py') < 4 )); do sleep 60; done; }
run() {  # run <script> <name> args...
  local script=$1 name=$2; shift 2
  if [[ -f runs/$name/results.json ]]; then echo "== skip $name"; return; fi
  gate; mkdir -p runs/$name; echo "== start $name ($(date))"
  $PY $script --run-name $name --wandb "$@" 2>&1 | tee runs/$name/train.log
  [[ -f runs/$name/results.json ]] || echo "!! $name FAILED"
}
RV="--model repvit_m0_9.dist_450e_in1k --epochs 30 --randaug --ema 0.999 --eval-every 5"
BF3B="--dataset cifar100 --model tiny --patch-size 4 --pool seqpool --lsa --init impulse --batch-size 128 --lr 5e-4 --min-lr 1e-5 --weight-decay 0.05 --warmup-epochs 30 --label-smoothing 0.1 --drop-path 0.1 --reprob 0.25 --randaug --cudagraphs --eval-every 5"
case ${1:-} in
  up)  # upcycle the IN-64 pretrained ViT (recipe of ec100_in64pre_tiny_64), then dense controls with extra compute
       UP="--init-ckpt runs/in64pre_tiny_p8 --dataset cifar100 --img-size 64 --epochs 30 --randaug --ema 0.999 --eval-every 5"
       run finetune.py ec100_upcyc_in64pre_tiny_e4_64 $UP --moe upcycle:e=4,last=4
       run finetune.py ec100_dense_in64pre_tiny_64_60ep ${UP/--epochs 30/--epochs 60}
       run finetune.py ec100_upcyc_in64pre_tiny_e4_64_60ep ${UP/--epochs 30/--epochs 60} --moe upcycle:e=4,last=4 ;;
  vit300) V="$BF3B --epochs 300 --mix-off-epoch 175"
       for E in 4 8; do run train_vit.py c100_moe_vit_e${E} $V --moe mlp:e=$E,last=4; done ;;
  vit)   V150="$BF3B --epochs 150 --mix-off-epoch 88"; V150=${V150/--warmup-epochs 30/--warmup-epochs 15}
         for E in 4 8; do run train_vit.py c100_moe_vit_e${E}_150ep $V150 --moe mlp:e=$E,last=4; done
         run train_vit.py c100_dense_vit_150ep $V150 ;;
  *) echo "unknown queue"; exit 1 ;;
esac
