#!/bin/bash
# B1: FlexiViT patch switch. Fine-tune the IN-64 patch-8 ckpt at patch 4 (PI-resize), recipe = ec100_in64pre_tiny_64.
cd "$(dirname "$0")"; source env.sh
PY=../.venv/bin/python
COMMON="--init-ckpt runs/in64pre_tiny_p8 --patch-size 4 --dataset cifar100 --epochs 30 --lr 5e-4 --warmup-epochs 3 --ema 0.999 --randaug --wandb"
gate() { until (( $(pgrep -fc 'python (train_vit|finetune)\.py') < 4 )); do sleep 60; done; }
if [ "$1" = smoke ]; then
  $PY finetune.py $COMMON --img-size 32 --epochs 1 --max-steps 20 --no-compile --run-name smoke_flexivit --out-dir /tmp/smoke_flexivit; exit
fi
if [ "$1" = p4_64 ]; then gate; $PY finetune.py $COMMON --img-size 64 --run-name ec100_in64pre_tiny_p4_64; fi
if [ "$1" = p4_32 ]; then gate; $PY finetune.py $COMMON --img-size 32 --run-name ec100_in64pre_tiny_p4_32; fi
if [ "$1" = p4_64_kd ]; then gate; $PY finetune.py $COMMON --img-size 64 --teacher runs/ec100_T_vitb_ar2_224 --teacher-img-size 224 --kd soft --kd-alpha 0.5 --kd-tau 2.0 --run-name ec100_in64pre_tiny_p4_64_kd; fi
