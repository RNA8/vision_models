#!/bin/bash
# Wave 8 (2026-10-13 11:30). Tiny-ImageNet with the final pure-ViT recipe.
# Our only Tiny-ImageNet runs used the early baseline recipe (best ViT 49.55, ResNet-18 61.56);
# published pure ViTs from scratch: SL-ViT 61.07 (2112.13492), Gani et al. 63.36 (2210.07240, two-stage).
# 64 px, patch 8 (64 tokens, as SL-ViT). Starts only after wave 7 has launched its last job, then gated at 4 jobs.
#   DRY=1 bash run_wave8.sh
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
MAXJOBS=${MAXJOBS:-4}
DRY=${DRY:-0}
bf3b() { echo "--batch-size 128 --lr 5e-4 --weight-decay 0.05 --epochs $1 --warmup-epochs 30 --randaug \
  --label-smoothing 0.1 --drop-path 0.1 --reprob 0.25 --init impulse --mix-off-epoch $(( $1 * 583 / 1000 )) --pool seqpool --lsa"; }
TV="--wandb --eval-every 10 --cudagraphs"
TIN="--dataset tiny_imagenet --patch-size 8 --spt"

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
if [[ $DRY != 1 ]]; then
  until [[ -d runs/in21kpre_tiny_p8_e60 ]] || ! tmux list-windows -t vit 2>/dev/null | grep -q wave7; do sleep 120; done
  echo "== wave 7 fully launched ($(date))"
fi
launch tin_spt_slvit_p8_300ep    train_vit.py $TV $TIN --model slvit $(bf3b 300)
launch tin_spt_tinyshape_p8_300ep train_vit.py $TV $TIN --model tiny $(bf3b 300)
launch tin_spt_slvit_p8_1000ep   train_vit.py $TV $TIN --model slvit $(bf3b 1000)
[[ $DRY == 1 ]] || wait
echo "== run_wave8 finished $(date)"
