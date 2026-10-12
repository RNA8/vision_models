#!/bin/bash
# Wave 2 (2026-10-12): firm up the headline results and fill the open controls. Starts immediately; at most
# MAXJOBS concurrent train/finetune processes (shares the GPU with anything else running).
#   DRY=1 bash run_wave2.sh
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
MAXJOBS=${MAXJOBS:-4}
DRY=${DRY:-0}

bf3b() {  # bf3b <epochs> -> recipe of run_long.sh with warmup 30 and mix-off at 58% of training
  echo "--batch-size 128 --lr 5e-4 --weight-decay 0.05 --epochs $1 --warmup-epochs 30 --randaug \
  --label-smoothing 0.1 --drop-path 0.1 --reprob 0.25 --init impulse --mix-off-epoch $(( $1 * 583 / 1000 )) --pool seqpool --lsa"
}
TV="--wandb --eval-every 10 --cudagraphs"
FT="--wandb --dataset cifar100 --batch-size 256 --weight-decay 0.05 --randaug --drop-path 0.1 --ema 0.999 --eval-every 5"
KDB="--teacher runs/ec100_T_vitb_ar2_224 --teacher-img-size 224"
MIX="--mixup 1.0 --cutmix 0 --mix-prob 1.0"
RV=repvit_m0_9.dist_450e_in1k
TVIT=tiny_vit_5m_224.dist_in22k_ft_in1k

gate() { sleep $((RANDOM % 30)); until (( $(ps -eo args= | grep -cE '^\S*python\S* (\S*/)?(train_vit|finetune)\.py') < MAXJOBS )); do sleep 60; done; }
done_() { [[ -f runs/$1/results.json ]]; }
launch() {  # launch <name> <script.py> <args...>
  local name=$1 script=$2; shift 2
  if [[ $DRY == 1 ]]; then echo "DRY: $PY $script --run-name $name $*"; return; fi
  done_ $name && { echo "== skip $name (done)"; return; }
  gate; mkdir -p runs/$name; echo "== start $name ($(date))"
  ( $PY $script --run-name "$name" "$@" 2>&1 | tee -a runs/$name/train.log
    done_ $name || echo "!! $name FAILED" ) &
  sleep 120
}

# 1. seeds for the new best pure-ViT CIFAR-100 result (SPT+LSA+SeqPool, 1000 ep: 80.66)
launch c100_spt_slvit_1000ep_s1   train_vit.py $TV --dataset cifar100 --model slvit --spt --seed 1 $(bf3b 1000)
launch c100_spt_slvit_1000ep_s2   train_vit.py $TV --dataset cifar100 --model slvit --spt --seed 2 $(bf3b 1000)
# 2. SPT on the CIFAR-10 headline shape
launch c10_spt_tinyshape_1000ep   train_vit.py $TV --dataset cifar10  --model tiny  --spt $(bf3b 1000)
# 3. edge: seed for the best <=6M model, and the missing 200-ep no-teacher control for RepViT
launch ec100_fm_tinyvit5m_224_e200_s1 finetune.py $FT --epochs 200 --warmup-epochs 5 --model $TVIT --img-size 224 --lr 5e-4 --seed 1 $KDB --kd soft --kd-alpha 1 --kd-tau 1 $MIX
launch ec100_nokd_repvit_224_e200     finetune.py $FT --epochs 200 --warmup-epochs 5 --model $RV --img-size 224 --lr 5e-4 $MIX
# 4. SPT on the 5.4M shape for CIFAR-100, and a longer SL-ViT schedule (CCT keeps gaining to 1500+ ep)
launch c100_spt_tinyshape_1000ep  train_vit.py $TV --dataset cifar100 --model tiny  --spt $(bf3b 1000)
launch c100_spt_slvit_2000ep      train_vit.py $TV --dataset cifar100 --model slvit --spt $(bf3b 2000)
[[ $DRY == 1 ]] || wait
echo "== run_wave2 finished $(date)"
