#!/usr/bin/env bash
# Baseline fix (RESEARCH_DIRECTIONS.md Part 1, R1-R3): SL-ViT recipe and shape, then init, then 300 ep.
#   bf1_slvit             --model slvit (192d, 9 layers, 12 heads, MLP 2), SL-ViT recipe: batch 128, lr 3e-3,
#                         wd 0.05, warmup 10, RandAugment, repeat-aug 3, LS 0.1, drop-path 0.1, reprob 0.25
#   bf1b_tiny_samerecipe  same recipe with --model tiny (12/3/4): isolates the shape effect
#   bf2_slvit_init        bf1 + --init impulse (Zheng et al. 2025)
#   bf3_slvit_300ep       bf2 at 300 ep + --mix-off-epoch 175 + --pool seqpool + --lsa
#
# Epoch length: --repeat-aug 3 makes each epoch one pass over the 50k unique images, each seen 3x, so
# 1170 steps/epoch at batch 128 (3x a plain epoch). That equals the step count of SL-ViT's own code,
# whose "--ra 3" is RASampler(repetitions=1, len_factor=3): 3x longer epochs, no in-batch repeats.
# So 100 ep here = ~117k steps, NOT ~39k.
#
# Estimated time, at ~4 s per 390-step epoch at batch 128 (65 tokens, CUDA graphs) -> ~12 s per epoch here:
#   bf1_slvit             100 ep  ~20 min  (9 layers, a bit faster than tiny)
#   bf1b_tiny_samerecipe  100 ep  ~22 min  (12 layers, MLP 4)
#   bf2_slvit_init        100 ep  ~20 min
#   bf3_slvit_300ep       300 ep  ~65 min  (+ LSA's masked attention can't use flash attention, maybe +5-10%)
#   total ~2.1 h
# MixUp/CutMix are the train_vit.py defaults (mixup 0.8, cutmix 1.0, mix-prob 1.0). SL-ViT used
# alpha 1.0 and mix-prob 0.5.
#   tmux new -s vit 'bash ViT/run_baseline_fix.sh'
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
COMMON="--wandb --eval-every 5 --dataset cifar10 --cudagraphs"
RECIPE="--batch-size 128 --lr 3e-3 --weight-decay 0.05 --warmup-epochs 10 --randaug --repeat-aug 3 \
  --label-smoothing 0.1 --drop-path 0.1 --reprob 0.25"

run() {  # run <name> <args...>
  local name=$1; shift
  if [[ -f runs/$name/results.json ]]; then echo "== skip $name (done)"; return; fi
  mkdir -p runs/$name
  echo "== start $name: $*  ($(date))"
  $PY train_vit.py --run-name "$name" $COMMON "$@" 2>&1 | tee runs/$name/train.log
  [[ -f runs/$name/results.json ]] || echo "!! $name FAILED (see runs/$name/train.log)"
}

run bf1_slvit             --model slvit $RECIPE --epochs 100
run bf1b_tiny_samerecipe  --model tiny  $RECIPE --epochs 100
run bf2_slvit_init        --model slvit $RECIPE --epochs 100 --init impulse
run bf3_slvit_300ep       --model slvit $RECIPE --epochs 300 --init impulse --mix-off-epoch 175 --pool seqpool --lsa
echo "== baseline fix finished $(date)"
