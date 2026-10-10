#!/usr/bin/env bash
# Recurrent-depth ViT pilot on CIFAR-10 (RESEARCH_DIRECTIONS.md Part 2, idea #1). 100 epochs each,
# full recipe (RandAugment + MixUp/CutMix + label smoothing + drop path 0.1), batch 256, LR 1e-3.
#   rd_pilot_a         rd_tiny4 (prelude 2, core 2 x r, coda 2; r_mean 4, k = 4, Poisson-lognormal r),
#                      concat injection + LayerNorm outer norm                         est. ~18 min
#   rd_pilot_b         same, additive injection                                         est. ~18 min
#   rd_pilot_noinject  same, NO injection (s_0 = prelude output, then s <- core(s)): the
#                      ablation for "recall is needed to extrapolate in depth" (2604.15259) est. ~18 min
#   rd_flat6           parameter-matched non-recurrent baseline: the same 6 unique blocks run once
#                      (a standard 6-layer ViT-Tiny up to one extra LayerNorm; train_vit.py has no
#                      --depth flag on this branch, so it is the rd_flat6 preset)       est. ~10 min
#   Compute-matched baseline: 12-layer ViT-Tiny, same recipe = existing runs/t5c_bs256 (82.09%); not rerun.
# rd_* models compile prelude / one core step / coda separately (variable r; no whole-model CUDA
# graphs), so they are slower per FLOP than --cudagraphs ViT-Tiny (~6 min/100 ep); the estimates assume
# t5c_bs256's 9.6 s/epoch without CUDA graphs. Total ~65 min + ~1-2 min eval per run.
# After each run, eval_recurrent.py writes the depth sweep (r = 1..32), adaptive-exit Pareto curves,
# difficulty correlation (vs runs/r18_full loss; CIFAR-10-C if data/CIFAR-10-C exists) and state
# convergence into runs/<name>/recurrent_eval.json + recurrent_*.png.
#   tmux new -s rd 'bash ViT/run_recurrent_pilot.sh'
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
COMMON="--wandb --eval-every 5 --randaug --batch-size 256 --lr 1e-3 --epochs 100"
REF=""; [[ -f runs/r18_full/model.pt ]] && REF="--ref-run-dir runs/r18_full"

run() {  # run <name> <args...>
  local name=$1; shift
  if [[ -f runs/$name/results.json ]]; then
    echo "== skip training $name (done)"
  else
    mkdir -p runs/$name
    echo "== start $name: $*  ($(date))"
    $PY train_vit.py --run-name "$name" $COMMON "$@" 2>&1 | tee runs/$name/train.log
    [[ -f runs/$name/results.json ]] || { echo "!! $name FAILED (see runs/$name/train.log)"; return; }
  fi
  if [[ ! -f runs/$name/recurrent_eval.json ]]; then
    echo "== eval_recurrent $name  ($(date))"
    $PY eval_recurrent.py runs/$name $REF 2>&1 | tee runs/$name/eval_recurrent.log
  fi
}

run rd_pilot_a         --model rd_tiny4 --rd-inject concat --rd-outer-norm ln
run rd_pilot_b         --model rd_tiny4 --rd-inject add    --rd-outer-norm ln
run rd_pilot_noinject  --model rd_tiny4 --rd-inject none   --rd-outer-norm ln
run rd_flat6           --model rd_flat6
echo "== recurrent pilot finished $(date)"
