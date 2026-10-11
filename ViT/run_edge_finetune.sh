#!/usr/bin/env bash
# Edge track, sweep 1: fine-tune ImageNet-pretrained timm models (<= ~6M params, plus references) on CIFAR-100,
# at 224 and at 128 (32x32 images are upsampled on the GPU; hierarchical nets downsample 32x, so 32px input is
# useless), plus large fine-tuned teachers that serve as the KD teacher and the near-SOTA reference, then KD.
# After each run, edge_bench.py profiles the model at its input size on CPU (params, MACs, act. memory,
# ONNX/onnxruntime 1- and 4-thread latency, fp32 + int8) -> runs/<name>/edge_bench.json + runs/edge_bench.jsonl.
# GPU latency is skipped here (the GPU is shared by the queues); re-run edge_bench.py on an idle GPU for that.
#
# Every timm name below was checked against timm.list_models(pretrained=True) in timm 1.0.30, and every model was
# built + run forward at 128 and 224 (efficientformerv2 / efficientvit_m* get their attention-bias tables resized
# from the 224 checkpoint when built at 128; see finetune.create_timm).
#
# Recipe (first guess, same for all students): AdamW lr 5e-4 (bs 256), 30 epochs, 3 warmup, cosine to 1e-6,
# wd 0.05, label smoothing 0.1, RandAugment m9 n2, MixUp 0.8 / CutMix 1.0, drop-path 0.1 (where the arch has it),
# EMA 0.999 (horizon ~1k steps = 5 epochs; eval'd alongside raw weights). Plain ViTs (12 blocks) get layer-wise
# lr decay 0.75; hierarchical/conv nets use a flat lr (timm's group_matcher gives them 15-30 groups, so 0.75^depth
# would almost freeze the stem). One lr bracket on tiny_vit_5m@128 (queue c) checks the lr choice.
#
# Time estimates are ROUGH guesses for an otherwise idle H100 (not measured: the GPU was busy when this was
# written); queues sharing the GPU slow each other. Refine from runs/<name>/results.json epoch_sec_steady.
#   bash ViT/run_edge_finetune.sh t   # teachers: ViT-B/16 augreg2 @224 20 ep (~30-40 min), EVA02-S @224 30 ep (~20 min)
#   bash ViT/run_edge_finetune.sh a   # plain/hybrid ViT students @224, 7 runs        (~8-15 min each, ~1.5 h)
#   bash ViT/run_edge_finetune.sh b   # conv/mobile students @224, 9 runs              (~5-12 min each, ~1.3 h)
#   bash ViT/run_edge_finetune.sh c   # all 16 students @128 + lr bracket, 18 runs     (~3-6 min each, ~1.5 h)
#   bash ViT/run_edge_finetune.sh k   # KD from the best finished teacher, 6 runs (needs queue t); the teacher forward
#                                     # at 224 adds ~1/3 of its own training cost per epoch    (~2-3 h)
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh
PY=../.venv/bin/python
COMMON="--wandb --dataset cifar100 --epochs 30 --batch-size 256 --warmup-epochs 3 --weight-decay 0.05 --randaug
        --drop-path 0.1 --ema 0.999 --eval-every 5"
VIT="--lr 5e-4 --lr-decay 0.75"   # plain 12-block ViTs
HIER="--lr 5e-4"                  # hierarchical / conv / hybrid nets

run() {  # run <name> <args...>
  local name=$1; shift
  if [[ ! -f runs/$name/results.json ]]; then
    mkdir -p runs/$name
    echo "== start $name: $*  ($(date))"
    $PY finetune.py --run-name "$name" $COMMON "$@" 2>&1 | tee runs/$name/train.log
  fi
  if [[ -f runs/$name/results.json && ! -f runs/$name/edge_bench.json ]]; then
    echo "== edge_bench $name  ($(date))"
    CUDA_VISIBLE_DEVICES= $PY edge_bench.py runs/$name --no-gpu --out runs/$name/edge_bench.json \
      2>&1 | tail -n 1 | tee runs/$name/edge_bench.log
  fi
  [[ -f runs/$name/results.json ]] || echo "!! $name FAILED (see runs/$name/train.log)"
}

# students: <short name> <timm name> <optimizer flags>       params (M, 100-class head)
STUDENTS_A=(
  "deit_t       deit_tiny_patch16_224.fb_in1k                      VIT"   # 5.5  plain ViT, IN-1k
  "vit_t_ar     vit_tiny_patch16_224.augreg_in21k_ft_in1k          VIT"   # 5.5  plain ViT, IN-21k augreg
  "tinyvit_5m   tiny_vit_5m_224.dist_in22k_ft_in1k                 HIER"  # 5.1  hybrid, IN-22k distilled
  "iformer_t    iformer_t.in1k                                     HIER"  # 2.7  hybrid
  "effformv2_s0 efficientformerv2_s0.snap_dist_in1k                HIER"  # 3.3  hybrid
  "fastvit_t8   fastvit_t8.apple_dist_in1k                         HIER"  # 3.3  hybrid (reparam)
  "effvit_m1    efficientvit_m1.r224_in1k                          HIER"  # 2.8  hybrid (cascaded group attn)
)
STUDENTS_B=(
  "effvit_m0    efficientvit_m0.r224_in1k                          HIER"  # 2.2  hybrid
  "mvitv2_075   mobilevitv2_075.cvnets_in1k                        HIER"  # 2.5  hybrid
  "mvit_xs      mobilevit_xs.cvnets_in1k                           HIER"  # 2.0  hybrid
  "edgenext_xs  edgenext_x_small.in1k                              HIER"  # 2.2  hybrid
  "starnet_s1   starnet_s1.in1k                                    HIER"  # 2.7  conv
  "mnv4_cs      mobilenetv4_conv_small.e2400_r224_in1k             HIER"  # 2.6  conv
  "repvit_m0_9  repvit_m0_9.dist_450e_in1k                         HIER"  # 4.8  conv (reparam)
  "convnext_at  convnext_atto.d2_in1k                              HIER"  # 3.4  conv
  "mnv4_hm      mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k  HIER"  # 9.9  hybrid, over budget: reference
)

sweep() {  # sweep <img size> <student lines...>
  local R=$1; shift
  for line in "$@"; do
    read -r short name opt <<< "$line"
    local flags; [[ $opt == VIT ]] && flags=$VIT || flags=$HIER
    run "ec100_${short}_${R}" --model "$name" --img-size "$R" $flags
  done
}

best_teacher() {  # finished teacher run with the highest best_val_acc
  $PY - <<'EOF'
import json, os
best = None
for n in ("ec100_T_vitb_ar2_224", "ec100_T_eva02s_224"):
    p = f"runs/{n}/results.json"
    if os.path.exists(p):
        r = json.load(open(p))
        acc = max(r["best_val_acc"], r.get("best_ema_val_acc") or 0)
        if best is None or acc > best[0]:
            best = (acc, n)
print(f"runs/{best[1]}" if best else "")
EOF
}

case "${1:-a}" in
  t) run ec100_T_vitb_ar2_224 --model vit_base_patch16_224.augreg2_in21k_ft_in1k --img-size 224 \
       --epochs 20 --lr 2e-4 --lr-decay 0.75 --ema 0.9995
     run ec100_T_eva02s_224 --model eva02_small_patch14_336.mim_in22k_ft_in1k --img-size 224 \
       --lr 2.5e-4 --lr-decay 0.75 ;;
     # optional third teacher (CNN, different inductive bias): convnext_base.fb_in22k_ft_in1k @224, --lr 2e-4
  a) sweep 224 "${STUDENTS_A[@]}" ;;
  b) sweep 224 "${STUDENTS_B[@]}" ;;
  c) sweep 128 "${STUDENTS_A[@]}" "${STUDENTS_B[@]}"
     run ec100_tinyvit_5m_128_lr2.5e-4 --model tiny_vit_5m_224.dist_in22k_ft_in1k --img-size 128 --lr 2.5e-4
     run ec100_tinyvit_5m_128_lr1e-3   --model tiny_vit_5m_224.dist_in22k_ft_in1k --img-size 128 --lr 1e-3 ;;
  k) T=$(best_teacher)
     [[ -n $T ]] || { echo "!! no finished teacher run (queue t)"; exit 1; }
     echo "== KD teacher: $T"
     KD="--teacher $T --teacher-img-size 224 --kd soft --kd-alpha 0.5 --kd-tau 2"
     run ec100_kd_tinyvit_5m_224 --model tiny_vit_5m_224.dist_in22k_ft_in1k --img-size 224 $HIER $KD
     run ec100_kd_tinyvit_5m_128 --model tiny_vit_5m_224.dist_in22k_ft_in1k --img-size 128 $HIER $KD
     run ec100_kd_deit_t_224     --model deit_tiny_patch16_224.fb_in1k --img-size 224 $VIT $KD
     run ec100_kdhard_deit_t_224 --model deit_tiny_patch16_224.fb_in1k --img-size 224 $VIT \
       --teacher "$T" --teacher-img-size 224 --kd hard --kd-alpha 0.5   # DeiT-style hard distillation
     run ec100_kd_repvit_m0_9_128 --model repvit_m0_9.dist_450e_in1k --img-size 128 $HIER $KD
     run ec100_kd_effformv2_s0_128 --model efficientformerv2_s0.snap_dist_in1k --img-size 128 $HIER $KD ;;
esac
echo "== edge-finetune $1 finished $(date)"
