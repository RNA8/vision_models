#!/usr/bin/env bash
# Tiny-ImageNet (64x64, 200 classes, 100k train / 10k val) ladder + scale-ups.
# Runs back to back; a run is skipped if its results.json exists, so rerunning this script
# after a failure only redoes the missing runs.  Launch inside tmux:
#   tmux new -s tin 'bash ViT/run_tiny_imagenet.sh'
#
# Runs (all --dataset tiny_imagenet, ViT-Tiny patch 8 = 64 tokens unless noted):
#   tin_t1_plain      100 ep  crop+flip only (no LS/WD/mixup/cutmix/drop-path)
#   tin_t2_ls_wd      100 ep  + label smoothing + weight decay
#   tin_t3_randaug    100 ep  + RandAugment
#   tin_t4_mix        100 ep  + MixUp/CutMix
#   tin_t5_droppath   100 ep  + drop-path (= full recipe)
#   tin_r18_full      100 ep  ResNet-18 (--r18-stem stride2), full recipe
#   tin_s_small_200ep 200 ep  ViT-Small p8, full recipe, LR 5e-4
#   tin_p4_tiny       100 ep  ViT-Tiny p4 (256 tokens), full recipe
#
# Lessons applied (from the CIFAR-10/100 chains):
#   - batch 256 with sqrt-scaled LR (Tiny 1e-3, Small 5e-4 = half of Tiny; Small at 2e-3 failed, 58%).
#     Batch 1024 underfits the small ViTs (CIFAR-10: t5 72% at bs1024 vs t5c_bs256 82%).
#   - --cudagraphs: ~2.7x faster, small ViTs are kernel-launch bound.
#   - ResNet-18 at 64x64 uses --r18-stem stride2 (same cost as 32x32; the CIFAR stem would be 4x).
#     This differs from the CIFAR-stem ResNet-18 in the CIFAR chains, so label it so in write-ups.
#   - Checkpointing is on by default (--ckpt-every 10, ckpt.pt in the run dir), so a crashed run
#     resumes from its last checkpoint when this script is rerun.
#   - TODO (not yet): --lsa / --pool seqpool / SL-ViT recipe. The baseline-fix experiments testing them are still
#     running and the ladders must stay comparable with the CIFAR-10/100 chains. Once validated, add them
#     to the ViT run lines (or to COMMON for ViT-only runs) below.
#
# Time estimates (standalone GPU, calibrated from c100_* runs: ViT-Tiny 3.0 s/ep plain, 3.4 s/ep randaug for 50k
# imgs at 65 tokens; Small 3.86 s/ep; r18 1.84 s/ep). Tiny-ImageNet = 2x images, Tiny p8 on 64x64 = same 65 tokens:
#   Tiny p8 ~6.1-6.8 s/ep (~11-12 min / 100 ep), r18 stride2 ~3.7 s/ep (~7 min), Small ~7.7 s/ep (~26 min / 200 ep),
#   Tiny p4 (257 tokens, ~3x per image) ~20 s/ep (~35 min).  Total ~2.1 h standalone; ~3.1 h when sharing the GPU
#   with two other queues (each job ~1.5x slower). See SCHEDULE.md.
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
COMMON="--wandb --eval-every 5 --dataset tiny_imagenet --batch-size 256 --lr 1e-3 --cudagraphs"

run() {  # run <name> <args...>
  local name=$1; shift
  if [[ -f runs/$name/results.json ]]; then echo "== skip $name (done)"; return; fi
  mkdir -p runs/$name
  echo "== start $name: $*  ($(date))"
  $PY train_vit.py --run-name "$name" $COMMON "$@" 2>&1 | tee runs/$name/train.log
  [[ -f runs/$name/results.json ]] || echo "!! $name FAILED (see runs/$name/train.log)"
}

NOREG="--label-smoothing 0 --weight-decay 0 --mixup 0 --cutmix 0 --drop-path 0"
P8="--patch-size 8"
run tin_t1_plain       --model tiny --epochs 100 $P8 $NOREG
run tin_t2_ls_wd       --model tiny --epochs 100 $P8 --mixup 0 --cutmix 0 --drop-path 0
run tin_t3_randaug     --model tiny --epochs 100 $P8 --mixup 0 --cutmix 0 --drop-path 0 --randaug
run tin_t4_mix         --model tiny --epochs 100 $P8 --drop-path 0 --randaug
run tin_t5_droppath    --model tiny --epochs 100 $P8 --randaug
run tin_r18_full       --model resnet18 --epochs 100 --randaug --r18-stem stride2
run tin_s_small_200ep  --model small --epochs 200 $P8 --randaug --lr 5e-4
run tin_p4_tiny        --model tiny --epochs 100 --patch-size 4 --randaug
echo "== tiny-imagenet runs finished $(date)"
