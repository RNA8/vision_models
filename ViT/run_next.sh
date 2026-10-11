#!/usr/bin/env bash
# Follow-ups queued 2026-10-11 after the bf3b ablation and the first edge results. Each queue waits for a free
# GPU slot (fewer than MAXJOBS train_vit.py/finetune.py processes) before every run, so they start as earlier jobs finish.
#   bash ViT/run_next.sh shape  # ViT-Tiny shape + bf3b recipe: CIFAR-100 300 ep, then CIFAR-10 seeds 1-2   (~4 h shared)
#   bash ViT/run_next.sh kd     # ViT-B teacher -> iFormer-T and FastViT-T8 at 128 px (KD)                  (~1 h)
#   bash ViT/run_next.sh pre    # our ViTs pretrained on ImageNet-64 (p8, 64 tokens, 30 ep), then CIFAR-100
#                               # fine-tunes at 64 px (32->64 upsample keeps 64 tokens), with and without KD   (~5-6 h)
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
MAXJOBS=${MAXJOBS:-4}
BF3B="--batch-size 128 --lr 5e-4 --weight-decay 0.05 --epochs 300 --warmup-epochs 30 --randaug \
  --label-smoothing 0.1 --drop-path 0.1 --reprob 0.25 --init impulse --mix-off-epoch 175 --pool seqpool --lsa"
FT="--wandb --dataset cifar100 --epochs 30 --batch-size 256 --warmup-epochs 3 --weight-decay 0.05 --randaug
    --drop-path 0.1 --ema 0.999 --eval-every 5"
TEACH=runs/ec100_T_vitb_ar2_224
KD="--teacher $TEACH --teacher-img-size 224 --kd soft --kd-alpha 0.5 --kd-tau 2"

gate() { sleep $((RANDOM % 60)); until (( $(pgrep -fc 'python (train_vit|finetune)\.py') < MAXJOBS )); do sleep 60; done; }
done_() { [[ -f runs/$1/results.json ]]; }
train() {  # train <name> <train_vit args...>
  local name=$1; shift; done_ $name && { echo "== skip $name"; return; }
  gate; mkdir -p runs/$name; echo "== start $name ($(date))"
  $PY train_vit.py --run-name "$name" --wandb --eval-every 5 --cudagraphs "$@" 2>&1 | tee runs/$name/train.log
  done_ $name || echo "!! $name FAILED"
}
ft() {  # ft <name> <finetune args...>
  local name=$1; shift
  if ! done_ $name; then
    gate; mkdir -p runs/$name; echo "== start $name ($(date))"
    $PY finetune.py --run-name "$name" $FT "$@" 2>&1 | tee runs/$name/train.log
  fi
  if done_ $name && [[ ! -f runs/$name/edge_bench.json ]]; then
    CUDA_VISIBLE_DEVICES= $PY edge_bench.py runs/$name --no-gpu --out runs/$name/edge_bench.json 2>&1 | tail -n 1
  fi
  done_ $name || echo "!! $name FAILED"
}

case "$1" in
  shape)
    train c100_tinyshape_bf3b --dataset cifar100 --model tiny $BF3B
    for s in 1 2; do train abl_tiny_shape_seed$s --dataset cifar10 --model tiny $BF3B --seed $s; done ;;
  kd)
    ft ec100_kd_iformer_t_128  --model iformer_t.in1k --img-size 128 --lr 5e-4 $KD
    ft ec100_kd_fastvit_t8_128 --model fastvit_t8.apple_dist_in1k --img-size 128 --lr 5e-4 $KD ;;
  pre)
    # ImageNet-64 at 30 ep underfits, so light regularization (no mixing / drop-path), as in in64_t1_plain.
    PRE="--dataset imagenet64 --patch-size 8 --batch-size 256 --lr 1e-3 --epochs 30 --warmup-epochs 2 --randaug
         --mixup 0 --cutmix 0 --drop-path 0 --label-smoothing 0.1 --init impulse --pool seqpool --lsa"
    for m in slvit tiny; do
      train in64pre_${m}_p8 --model $m $PRE
      done_ in64pre_${m}_p8 || continue
      ft ec100_in64pre_${m}_64    --init-ckpt runs/in64pre_${m}_p8 --img-size 64 --lr 5e-4
      ft ec100_in64pre_${m}_64_kd --init-ckpt runs/in64pre_${m}_p8 --img-size 64 --lr 5e-4 $KD
    done ;;
esac
echo "== next $1 finished $(date)"
