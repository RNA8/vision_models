#!/bin/bash
# Wave 11 (2026-10-13 17:45): patch-2 control (plan E5 in research/self_distillation.md).
# Same 2.75M SL-ViT+SPT+LSA+SeqPool student and bf3b recipe, patch 2 at 32 px = 256 tokens (vs 64), no distillation.
# Tests whether the gap to CCT (82.72 at 1500 ep) is a token-count gap: arXiv 2406.09415 reports ViT-T/2 (5.6M) 83.6
# on CIFAR-100 at 2400 ep. ~4x inference MACs, so check against the edge budget. ~54 s/epoch shared -> ~15 h.
#   DRY=1 bash run_wave11_p2.sh
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
MAXJOBS=${MAXJOBS:-4}
DRY=${DRY:-0}
bf3b() { echo "--batch-size 128 --lr 5e-4 --weight-decay 0.05 --epochs $1 --warmup-epochs 30 --randaug \
  --label-smoothing 0.1 --drop-path 0.1 --reprob 0.25 --init impulse --mix-off-epoch $(( $1 * 583 / 1000 )) --pool seqpool --lsa"; }
TV="--wandb --eval-every 10 --cudagraphs"

gate() { sleep $((RANDOM % 30)); until (( $(ps -eo args= | grep -cE '^\S*python\S* (\S*/)?(train_vit|finetune)\.py') < MAXJOBS )); do sleep 60; done; }
done_() { [[ -f runs/$1/results.json ]]; }
launch() {
  local name=$1 script=$2; shift 2
  if [[ $DRY == 1 ]]; then echo "DRY: $PY $script --run-name $name $*"; return; fi
  done_ $name && { echo "== skip $name (done)"; return; }
  gate; mkdir -p runs/$name; echo "== start $name ($(date))"
  ( $PY $script --run-name "$name" "$@" 2>&1 | tee -a runs/$name/train.log
    done_ $name || echo "!! $name FAILED" ) &
  sleep 120
}
launch c100_spt_slvit_p2_1000ep train_vit.py $TV --dataset cifar100 --model slvit --spt --patch-size 2 $(bf3b 1000)
[[ $DRY == 1 ]] || wait
echo "== run_wave11 finished $(date)"
