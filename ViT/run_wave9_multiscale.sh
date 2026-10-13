#!/bin/bash
# Wave 9 (2026-10-13): multiscale input tokens (--multiscale, see research/multiscale_vit.md).
# 32 px, patch 4: tokens of the image + 2x/4x/8x-downscaled copies, same patch size: 64+16+4+1 = 85 (+CLS).
# CIFAR-100, 300 ep, the exact recipe of c100_spt_slvit_300ep (76.70): SL-ViT shape + SPT + LSA + seqpool + impulse.
# Token-matched control: baseline + 21 register tokens (85 tokens, same attention cost, no multiscale input).
#   DRY=1 bash run_wave9_multiscale.sh
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
MAXJOBS=${MAXJOBS:-4}
DRY=${DRY:-0}
BF3B300="--batch-size 128 --lr 5e-4 --weight-decay 0.05 --epochs 300 --warmup-epochs 30 --randaug \
  --label-smoothing 0.1 --drop-path 0.1 --reprob 0.25 --init impulse --mix-off-epoch 175 --pool seqpool --lsa"
TV="--wandb --eval-every 10 --cudagraphs"
C="--dataset cifar100 --model slvit --spt $BF3B300"

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
launch c100_ms_shared_300ep    train_vit.py $TV $C --multiscale
launch c100_reg21_300ep        train_vit.py $TV $C --reg-tokens 21
launch c100_ms_perscale_300ep  train_vit.py $TV $C --multiscale --ms-proj per-scale
launch c100_ms_lap_300ep       train_vit.py $TV $C --multiscale --ms-pyramid laplacian
launch c100_ms_pooledpos_300ep train_vit.py $TV $C --multiscale --ms-pos pooled
launch c100_spt_slvit_300ep_s1 train_vit.py $TV $C --seed 1
[[ $DRY == 1 ]] || wait
echo "== run_wave9_multiscale finished $(date)"
