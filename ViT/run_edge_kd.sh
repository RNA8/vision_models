#!/usr/bin/env bash
# Wave-1 edge experiments A1/A2/A3 (see research/NEXT_STEPS.md). Usage: bash run_edge_kd.sh {fm_repvit|fm_tiny|loss|res}
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
MAXJOBS=${MAXJOBS:-4}
FT="--wandb --dataset cifar100 --batch-size 256 --weight-decay 0.05 --randaug --drop-path 0.1 --ema 0.999 --eval-every 5"
TEACH=runs/ec100_T_vitb_ar2_224
KDB="--teacher $TEACH --teacher-img-size 224"
FM="$KDB --kd soft --kd-alpha 1 --kd-tau 1 --mixup 1.0 --cutmix 0 --mix-prob 1.0"   # function matching: KD-only, mixup, same view
RV=repvit_m0_9.dist_450e_in1k
TV=tiny_vit_5m_224.dist_in22k_ft_in1k
gate() { sleep $((RANDOM % 60)); until (( $(pgrep -fc 'python (train_vit|finetune)\.py') < MAXJOBS )); do sleep 60; done; }
done_() { [[ -f runs/$1/results.json ]]; }
ft() {  # ft <name> <epochs> <warmup> <args...>
  local name=$1 ep=$2 wu=$3; shift 3
  if ! done_ $name; then
    gate; mkdir -p runs/$name; echo "== start $name ($(date))"
    $PY finetune.py --run-name "$name" $FT --epochs $ep --warmup-epochs $wu "$@" 2>&1 | tee -a runs/$name/train.log
  fi
  if done_ $name && [[ ! -f runs/$name/edge_bench.json ]]; then
    CUDA_VISIBLE_DEVICES= $PY edge_bench.py runs/$name --no-gpu --out runs/$name/edge_bench.json 2>&1 | tail -n 1
  fi
  done_ $name || echo "!! $name FAILED"
}
case "$1" in
  fm_repvit)
    ft ec100_fm_repvit_224_e200 200 5 --model $RV --img-size 224 --lr 5e-4 $FM
    ft ec100_fm_repvit_224_e30  30  3 --model $RV --img-size 224 --lr 5e-4 $FM ;;
  fm_tiny)
    ft ec100_fm_tinyvit5m_224_e200 200 5 --model $TV --img-size 224 --lr 5e-4 $FM ;;
  loss)
    ft ec100_kd60_soft_repvit_224 60 3 --model $RV --img-size 224 --lr 5e-4 $KDB --kd soft --kd-alpha 0.5 --kd-tau 2
    ft ec100_kd60_dist_repvit_224 60 3 --model $RV --img-size 224 --lr 5e-4 $KDB --kd dist --kd-alpha 0.5 --kd-tau 1
    ft ec100_kd60_lsd_repvit_224  60 3 --model $RV --img-size 224 --lr 5e-4 $KDB --kd lsd --kd-alpha 0.5 --kd-tau 2 ;;
  res)
    for r in 256 288; do ft ec100_repvit_m0_9_$r 30 3 --model $RV --img-size $r --lr 5e-4; done
    $PY eval_res.py runs/ec100_repvit_m0_9_224 224 256 288 320 ;;
esac
echo "== edge_kd $1 finished $(date)"
