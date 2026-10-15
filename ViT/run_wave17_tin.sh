#!/bin/bash
# Wave 17 (2026-10-15 03:35): reordered copy of wave 15 (user: "Run the width-32 ResNet first").
# Wave 15 was stopped after its first lr/wd screen; this finishes the screen, then runs the param-matched
# width-32 ResNet-18 before the 11.27M and 8.65M ones. Original wave-15 notes:
# Tiny-ImageNet controls for the claim "our 2.9M pure ViT is the best from-scratch
# Tiny-ImageNet model under 10M params".
#  1. Seed 1 of the born-again distillation run (tin_kd_ba_fm_1000ep: 68.09), same teacher (seed-0 ViT).
#  2. Properly tuned CNN controls at native 64x64: ResNet-18 with the CIFAR/Tiny-ImageNet stem
#     (3x3 stride-1 conv, no max-pool), SGD + Nesterov, warmup + cosine, the same augmentation/regularization
#     stack as the ViT (RandAugment m9, MixUp 0.8 / CutMix 1.0 off after 58.3% of training, label smoothing
#     0.1, random erasing 0.25, stochastic depth 0.1), no grad clipping, batch 128, 1000 epochs.
#     a. width 64 = standard ResNet-18 (11.27M)   b. width 32 (2.86M, param-matched to the 2.91M ViT)
#     c. width 56 (8.65M, the widest ResNet-18 under 10M)
#     lr / wd: 100-epoch screen on a 10k held-out split of train (never the val set), grid lr {0.1, 0.2} x
#     wd {1e-4, 5e-4}; the best held-out config is used for all 1000-epoch CNN runs.
# Gate uses flock so concurrent queue scripts can't both launch into the same free slot.
#   DRY=1 bash run_wave17_tin.sh
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
MAXJOBS=${MAXJOBS:-4}
DRY=${DRY:-0}
R=$PWD/runs
LOCK=$PWD/runs/.gate.lock
bf3b() { echo "--batch-size 128 --lr 5e-4 --weight-decay 0.05 --epochs $1 --warmup-epochs 30 --randaug \
  --label-smoothing 0.1 --drop-path 0.1 --reprob 0.25 --init impulse --mix-off-epoch $(( $1 * 583 / 1000 )) --pool seqpool --lsa"; }
# CNN recipe: $1 epochs, $2 lr, $3 wd, $4 warmup epochs
cnn() { echo "--opt sgd --momentum 0.9 --nesterov --batch-size 128 --lr $2 --weight-decay $3 --grad-clip 0 \
  --epochs $1 --warmup-epochs $4 --randaug --label-smoothing 0.1 --drop-path 0.1 --reprob 0.25 \
  --mix-off-epoch $(( $1 * 583 / 1000 ))"; }
TV="--wandb --eval-every 10 --cudagraphs"
TIN="--dataset tiny_imagenet --patch-size 8 --model slvit --spt"
R18="--dataset tiny_imagenet --model resnet18 --r18-stem cifar"
FM="--mix-off-epoch 0 --kd soft --kd-tau 1 --kd-alpha 1.0"

njobs() { ps -eo args= | grep -cE '^\S*python\S* (\S*/)?(train_vit|finetune)\.py'; }
done_() { [[ -f runs/$1/results.json ]]; }
LAST_PID=
launch() {
  local name=$1 script=$2; shift 2
  LAST_PID=
  if [[ $DRY == 1 ]]; then echo "DRY: $PY $script --run-name $name $*"; return; fi
  done_ $name && { echo "== skip $name (done)"; return; }
  while true; do
    exec 9>"$LOCK"; flock 9
    if (( $(njobs) < MAXJOBS )); then
      mkdir -p runs/$name; echo "== start $name ($(date))"
      ( $PY $script --run-name "$name" "$@" 2>&1 | tee -a runs/$name/train.log
        done_ $name || echo "!! $name FAILED" ) &
      LAST_PID=$!
      sleep 90; flock -u 9; exec 9>&-; break
    fi
    flock -u 9; exec 9>&-; sleep 60
  done
}


# 2. lr / wd screen (100 epochs, 10k held-out train images as validation)
SCREEN=()
PIDS=()
for lr in 0.1 0.2; do for wd in 1e-4 5e-4; do
  n=tin_r18_screen_lr${lr}_wd${wd}
  SCREEN+=($n)
  launch $n train_vit.py $TV $R18 $(cnn 100 $lr $wd 5) --holdout 10000
  [[ -n $LAST_PID ]] && PIDS+=($LAST_PID)
done; done
(( ${#PIDS[@]} )) && wait "${PIDS[@]}"
read LR WD < <($PY - "${SCREEN[@]}" <<'EOF'
import json, os, sys
best = None
for n in sys.argv[1:]:
    f = f"runs/{n}/results.json"
    if not os.path.exists(f):
        continue
    d = json.load(open(f)); a = d["args"]
    print(f"screen {n}: held-out val {d['final_val_acc']:.2f}", file=sys.stderr)
    if best is None or d["final_val_acc"] > best[0]:
        best = (d["final_val_acc"], a["lr"], a["weight_decay"])
print(*(best[1:] if best else (0.1, 5e-4)))
EOF
)
echo "== CNN lr $LR wd $WD (best held-out screen config)"

# 3. 1000-epoch CNN controls
launch tin_r18w32_sgd_1000ep  train_vit.py $TV $R18 --r18-width 32 $(cnn 1000 $LR $WD 10)
launch tin_r18_sgd_1000ep     train_vit.py $TV $R18 --r18-width 64 $(cnn 1000 $LR $WD 10)
launch tin_r18w56_sgd_1000ep  train_vit.py $TV $R18 --r18-width 56 $(cnn 1000 $LR $WD 10)
[[ $DRY == 1 ]] || wait
echo "== run_wave17_tin finished $(date)"
