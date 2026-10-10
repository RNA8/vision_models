#!/usr/bin/env bash
# Multi-seed repeats and a per-component ablation of bf3b (CIFAR-10 93.37%, seed 0). Most gaps so far are <1 pt
# from single seeds, so: seeds 1 and 2 of bf3b, bf4 (lighter mixing, 93.44) and rd_deep_r8_add_ds (recurrent, 85.23).
# Ablation: bf3b with exactly one component removed (300 ep each, seed 0, compare to bf3b):
#   abl_no_lsa (plain attention), abl_cls (CLS token instead of SeqPool), abl_default_init (no impulse init),
#   abl_tiny_shape (timm ViT-Tiny 12L/192/3h/MLP4 instead of the SL-ViT 9L/192/12h/MLP2 shape).
# Four queues: s1 and a1 start now; s2 and a2 start when the CIFAR-100 recurrent queues free the GPU.
#   bash ViT/run_seeds_ablation.sh {s1|s2|a1|a2}
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
C10="--wandb --eval-every 5 --dataset cifar10 --cudagraphs"
BF3B="--model slvit --batch-size 128 --lr 5e-4 --weight-decay 0.05 --epochs 300 --warmup-epochs 30 --randaug \
  --label-smoothing 0.1 --drop-path 0.1 --reprob 0.25 --init impulse --mix-off-epoch 175 --pool seqpool --lsa"
BF4="$BF3B --mix-prob 0.5 --mixup 1.0 --cutmix 1.0"
RD="--randaug --batch-size 256 --lr 1e-3 --epochs 100 --lsa --pool seqpool \
  --model rd_tiny4 --rd-inject add --rd-r-mean 8 --rd-k-backprop 4 --rd-coda-every-step"
REF=""; [[ -f runs/r18_full/results.json ]] && REF="--ref-run-dir runs/r18_full"

run() {  # run <name> <args...>
  local name=$1; shift
  if [[ ! -f runs/$name/results.json ]]; then
    mkdir -p runs/$name
    echo "== start $name: $*  ($(date))"
    $PY train_vit.py --run-name "$name" $C10 "$@" 2>&1 | tee runs/$name/train.log
  fi
  if [[ $name == rd_* && -f runs/$name/results.json && ! -f runs/$name/recurrent_eval.json ]]; then
    $PY eval_recurrent.py runs/$name $REF 2>&1 | tee runs/$name/eval_recurrent.log
  fi
  [[ -f runs/$name/results.json ]] || echo "!! $name FAILED (see runs/$name/train.log)"
}

case "$1" in
  s1) for s in 1; do run bf3b_seed$s $BF3B --seed $s; run bf4_seed$s $BF4 --seed $s; run rd_deep_r8_add_ds_seed$s $RD --seed $s; done ;;
  s2) for s in 2; do run bf3b_seed$s $BF3B --seed $s; run bf4_seed$s $BF4 --seed $s; run rd_deep_r8_add_ds_seed$s $RD --seed $s; done ;;
  a1) run abl_no_lsa       ${BF3B/--lsa/}
      run abl_cls          ${BF3B/--pool seqpool/--pool cls} ;;
  a2) run abl_default_init ${BF3B/--init impulse/--init default}
      run abl_tiny_shape   ${BF3B/--model slvit/--model tiny} ;;
esac
echo "== seeds-ablation $1 finished $(date)"
