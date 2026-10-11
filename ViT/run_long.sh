#!/usr/bin/env bash
# Long-schedule (500-1000 epoch) runs, single seed 0. WAITS until every other job/queue is finished, then runs up to
# MAXJOBS (4) concurrently. Rationale + time estimates: research/long_runs.md.
#   tmux new-window -t vit -n long 'cd .../ViT && bash run_long.sh 2>&1 | tee runs/run_long.out'
#   DRY=1 bash run_long.sh   # print the expanded commands (no waiting, no running)
# Both train_vit.py and finetune.py resume from runs/<name>/ckpt.pt, so after a crash just re-run this script.
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
MAXJOBS=${MAXJOBS:-4}
DRY=${DRY:-0}

# bf3b recipe (see run_next.sh / run_c100_bf3b.sh) with 1000 ep: warmup 30 kept, mix-off 175/300 -> 583
BF3B1000="--batch-size 128 --lr 5e-4 --weight-decay 0.05 --epochs 1000 --warmup-epochs 30 --randaug \
  --label-smoothing 0.1 --drop-path 0.1 --reprob 0.25 --init impulse --mix-off-epoch 583 --pool seqpool --lsa"
# recurrent-depth recipe (run_c100_recurrent.sh) at 500 ep: warmup 5 -> 25
RD500="--dataset cifar100 --randaug --batch-size 256 --lr 1e-3 --epochs 500 --warmup-epochs 25 --lsa --pool seqpool"
# function-matching KD (run_edge_kd.sh)
TEACH=runs/ec100_T_vitb_ar2_224
FM="--teacher $TEACH --teacher-img-size 224 --kd soft --kd-alpha 1 --kd-tau 1 --mixup 1.0 --cutmix 0 --mix-prob 1.0"
FT="--wandb --dataset cifar100 --batch-size 256 --weight-decay 0.05 --randaug --drop-path 0.1 --ema 0.999 --eval-every 10"
RV=repvit_m0_9.dist_450e_in1k

# --- waiting: no train/finetune process and no other run_*.sh queue (matched on the process's own argv[0..1], so the
#     "bash -c '...; read'" tmux wrappers and this script do not count) ---
others_alive() { ps -eo pid=,args= | grep -E '^ *[0-9]+ (/bin/)?bash (\S*/)?run_[A-Za-z0-9_]+\.sh' | grep -v 'run_long\.sh' | grep -q .; }
procs_alive() { ps -eo args= | grep -E '^\S*python\S* (\S*/)?(train_vit|finetune)\.py' | grep -q .; }
if [[ $DRY != 1 ]]; then
  echo "== waiting for all other jobs/queues to finish ($(date))"
  until ! procs_alive && ! others_alive; do sleep 300; done
  echo "== all clear, starting long runs ($(date))"
fi
gate() { sleep $((RANDOM % 30)); until (( $(ps -eo args= | grep -cE '^\S*python\S* (\S*/)?(train_vit|finetune)\.py') < MAXJOBS )); do sleep 60; done; }
done_() { [[ -f runs/$1/results.json ]]; }

launch() {  # launch <name> <script.py> <args...>  -- starts in background once a slot is free
  local name=$1 script=$2; shift 2
  if [[ $DRY == 1 ]]; then echo "DRY: $PY $script --run-name $name $*"; return; fi
  done_ $name && { echo "== skip $name (done)"; return; }
  gate; mkdir -p runs/$name; echo "== start $name ($(date))"
  ( $PY $script --run-name "$name" "$@" 2>&1 | tee -a runs/$name/train.log
    done_ $name || echo "!! $name FAILED" ) &
  sleep 120   # let the process appear before the next gate() counts
}
TV="--wandb --eval-every 10 --cudagraphs"

launch c100_tinyshape_bf3b_1000ep train_vit.py $TV --dataset cifar100 --model tiny $BF3B1000
launch c100_rd_r8_add_500ep       train_vit.py $TV $RD500 --model rd_tiny4 --rd-inject add --rd-r-mean 8 --rd-k-backprop 4
launch c100_rd_flat6_500ep        train_vit.py $TV $RD500 --model rd_flat6
launch c100_bf3b_slvit_1000ep     train_vit.py $TV --dataset cifar100 --model slvit $BF3B1000
launch c10_tinyshape_bf3b_1000ep  train_vit.py $TV --dataset cifar10 --model tiny $BF3B1000

# L6: only if the 200-ep function-matching run beat the 30-ep baseline (88.33) by > 0.3
if [[ $DRY == 1 ]]; then
  echo "DRY: (L6, conditional) $PY finetune.py --run-name ec100_fm_repvit_128_e1000 $FT --epochs 1000 --warmup-epochs 10 --model $RV --img-size 128 --lr 5e-4 $FM"
else
  R=runs/ec100_fm_repvit_224_e200/results.json
  if [[ -f $R ]] && $PY -c "import json,sys; d=json.load(open('$R')); print('e200 final',d['final_val_acc'],'ema',d.get('final_ema_val_acc')); sys.exit(0 if d['final_val_acc']>88.33+0.3 else 1)"; then
    launch ec100_fm_repvit_128_e1000 finetune.py $FT --epochs 1000 --warmup-epochs 10 --model $RV --img-size 128 --lr 5e-4 $FM
  else
    echo "== skip L6 ec100_fm_repvit_128_e1000: e200 missing or did not beat 88.33 by >0.3"
  fi
  wait
fi
echo "== run_long finished $(date)"
