#!/usr/bin/env bash
# Rerun the EfficientFormerV2-S0 fine-tunes that collapsed under compile + bf16 (fixed in finetune.py 41588c0:
# its attention modules now run in fp32). Waits for a free GPU slot like run_next.sh.
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
FT="--wandb --dataset cifar100 --epochs 30 --batch-size 256 --warmup-epochs 3 --weight-decay 0.05 --randaug
    --drop-path 0.1 --ema 0.999 --eval-every 5 --lr 5e-4"
for R in 128 224; do
  name=ec100_effformv2_s0_$R
  [[ -f runs/$name/results.json ]] && continue
  sleep $((RANDOM % 60)); until (( $(pgrep -fc 'python (train_vit|finetune)\.py') < ${MAXJOBS:-4} )); do sleep 60; done
  mkdir -p runs/$name; echo "== start $name ($(date))"
  $PY finetune.py --run-name $name $FT --model efficientformerv2_s0.snap_dist_in1k --img-size $R 2>&1 | tee runs/$name/train.log
  [[ -f runs/$name/results.json ]] && CUDA_VISIBLE_DEVICES= $PY edge_bench.py runs/$name --no-gpu --out runs/$name/edge_bench.json 2>&1 | tail -n 1
done
echo "== effformv2 rerun finished $(date)"
