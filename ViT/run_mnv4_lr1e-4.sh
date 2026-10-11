#!/usr/bin/env bash
# MobileNetV4-Hybrid-M at lr 1e-4. With the zero-init head (f09c9c1) it still diverged at lr 5e-4
# once warmup peaked (224: 6.15%, loss rising from epoch 3; 128: 74.27%), so the remaining problem is lr.
# Waits for a free GPU slot like run_next.sh.
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
FT="--wandb --dataset cifar100 --epochs 30 --batch-size 256 --warmup-epochs 3 --weight-decay 0.05 --randaug
    --drop-path 0.1 --ema 0.999 --eval-every 5 --lr 1e-4"
for R in 128 224; do
  name=ec100_mnv4_hm_${R}_lr1e-4
  [[ -f runs/$name/results.json ]] && continue
until (( $(pgrep -fc 'python (train_vit|finetune)\.py') < ${MAXJOBS:-4} )); do sleep 60; done
  mkdir -p runs/$name; echo "== start $name ($(date))"
  $PY finetune.py --run-name $name $FT --model mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k --img-size $R 2>&1 | tee runs/$name/train.log
  [[ -f runs/$name/results.json ]] && CUDA_VISIBLE_DEVICES= $PY edge_bench.py runs/$name --no-gpu --out runs/$name/edge_bench.json 2>&1 | tail -n 1
done
echo "== mnv4 rerun finished $(date)"
