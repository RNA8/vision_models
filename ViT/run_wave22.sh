#!/bin/bash
# Wave 22 (2026-10-16 03:00): wind-down queue, replaces wave 21 (stopped by the user before it launched anything;
# its 2000-ep 1.25M student is dropped to finish by ~11:00 UTC).
#  1. CIFAR-100 lr/wd screen for the ResNet-18 width-32 control (100 ep, 5k held-out train images; val untouched).
#  2. If the best held-out config beats the one used for c100_r18w32_sgd_1000ep (lr 0.1, wd 5e-4) by > 0.3,
#     rerun the 1000-ep control with it as c100_r18w32_sgd_1000ep_tuned. Otherwise the existing 78.59 stands.
# Gate uses flock (shared runs/.gate.lock).
#   DRY=1 bash run_wave22.sh
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
MAXJOBS=${MAXJOBS:-4}
DRY=${DRY:-0}
R=$PWD/runs
LOCK=$PWD/runs/.gate.lock
bf3b() { echo "--batch-size 128 --lr 5e-4 --weight-decay 0.05 --epochs $1 --warmup-epochs 30 --randaug \
  --label-smoothing 0.1 --drop-path 0.1 --reprob 0.25 --init impulse --mix-off-epoch $(( $1 * 583 / 1000 )) --pool seqpool --lsa"; }
TV="--wandb --eval-every 10 --cudagraphs"
C100="--dataset cifar100 --model slvit --spt"
TIN="--dataset tiny_imagenet --patch-size 8 --model slvit --spt"
FM="--mix-off-epoch 0 --kd soft --kd-tau 1 --kd-alpha 1.0"
ENS3="$R/c100_spt_slvit_1000ep,$R/c100_spt_slvit_1000ep_s1,$R/c100_spt_slvit_1000ep_s2"

njobs() { ps -eo args= | grep -cE '^\S*python\S* (\S*/)?(train_vit|finetune)\.py'; }
done_() { [[ -f runs/$1/results.json ]]; }
launch() {
  local name=$1 script=$2; shift 2
  if [[ $DRY == 1 ]]; then echo "DRY: $PY $script --run-name $name $*"; return; fi
  done_ $name && { echo "== skip $name (done)"; return; }
  while true; do
    exec 9>"$LOCK"; flock 9
    if (( $(njobs) < MAXJOBS )); then
      mkdir -p runs/$name; echo "== start $name ($(date))"
      ( $PY $script --run-name "$name" "$@" 2>&1 | tee -a runs/$name/train.log
        done_ $name || echo "!! $name FAILED" ) &
      sleep 90; flock -u 9; exec 9>&-; break
    fi
    flock -u 9; exec 9>&-; sleep 60
  done
}
cnn() { echo "--opt sgd --momentum 0.9 --nesterov --batch-size 128 --lr $2 --weight-decay $3 --grad-clip 0 \
  --epochs $1 --warmup-epochs $4 --randaug --label-smoothing 0.1 --drop-path 0.1 --reprob 0.25 \
  --mix-off-epoch $(( $1 * 583 / 1000 ))"; }

LAST_PID=; PIDS=()
launch_bg() { launch "$@"; [[ $DRY == 1 ]] || PIDS+=($!); }

SCREEN=()
for lr in 0.05 0.1 0.2; do for wd in 5e-4 1e-3; do
  n=c100_r18w32_screen_lr${lr}_wd${wd}; SCREEN+=($n)
  launch_bg $n train_vit.py $TV --dataset cifar100 --model resnet18 --r18-stem cifar --r18-width 32 $(cnn 100 $lr $wd 5) --holdout 5000
done; done
[[ $DRY == 1 ]] && exit 0
wait "${PIDS[@]}"
read LR WD GAIN < <($PY - "${SCREEN[@]}" <<'PYEOF'
import json, os, sys
res = {}
for n in sys.argv[1:]:
    f = f"runs/{n}/results.json"
    if os.path.exists(f):
        d = json.load(open(f)); a = d["args"]
        res[(a["lr"], a["weight_decay"])] = d["final_val_acc"]
        print(f"screen {n}: held-out {d['final_val_acc']:.2f}", file=sys.stderr)
best = max(res, key=res.get)
base = res.get((0.1, 5e-4), 0.0)
print(best[0], best[1], f"{res[best] - base:.2f}")
PYEOF
)
echo "== best held-out lr $LR wd $WD, gain over lr 0.1 / wd 5e-4: $GAIN"
if python3 -c "import sys; sys.exit(0 if float('$GAIN') > 0.3 else 1)"; then
  launch c100_r18w32_sgd_1000ep_tuned train_vit.py $TV --dataset cifar100 --model resnet18 --r18-stem cifar --r18-width 32 $(cnn 1000 $LR $WD 10)
  wait
else
  echo "== no rerun: existing control config is within 0.3 of the best"
fi
echo "== run_wave22 finished $(date)"
