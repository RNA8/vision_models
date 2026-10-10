#!/usr/bin/env bash
# bf3b recipe on CIFAR-100: the CIFAR-10 pure-ViT that reached 93.37% (published pure ViT ~93.6%).
# SL-ViT shape (9L/12h/MLP2, 2.7M), batch 128, LR 5e-4, 300 ep (warmup 30), RandAugment, MixUp/CutMix
# off from epoch 175, random erasing 0.25, LS 0.1, WD 0.05, drop-path 0.1, LSA, SeqPool, impulse init.
# Published CIFAR-100 targets (verified): SPT-paper ViT 73.81, ViT-Lite-7/4 73.94, SL-ViT 76.92.
# Our current best CIFAR-100 ViT: 70.49 (c100_t6_full_300ep). ~40-45 min alone.
#   tmux new -s vit 'bash ViT/run_c100_bf3b.sh'
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
name=c100_bf3b_slvit_300ep_lr5e-4
if [[ -f runs/$name/results.json ]]; then echo "== skip $name (done)"; exit 0; fi
mkdir -p runs/$name
echo "== start $name  ($(date))"
$PY train_vit.py --run-name $name --wandb --eval-every 5 --dataset cifar100 --cudagraphs \
  --model slvit --batch-size 128 --lr 5e-4 --weight-decay 0.05 --epochs 300 --warmup-epochs 30 --randaug \
  --label-smoothing 0.1 --drop-path 0.1 --reprob 0.25 --init impulse --mix-off-epoch 175 --pool seqpool --lsa \
  2>&1 | tee runs/$name/train.log
[[ -f runs/$name/results.json ]] || echo "!! $name FAILED (see runs/$name/train.log)"
echo "== c100-bf3b finished $(date)"
