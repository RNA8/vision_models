#!/bin/bash
# Build the download archive before the instance is shut down.
#   bash make_archive.sh            -> ~/workspace/vision_models_archive_<date>.zip
#   DRY=1 bash make_archive.sh      -> print the file list and total size only
# Contents:
#   repo/vision_models.bundle   git bundle of every branch (clone with: git clone vision_models.bundle)
#   runs/<run>/...              results.json, train.log, *.json/*.jsonl/*.log/*.md/*.csv/*.png and wandb run files for ALL runs
#   runs/<run>/model*.pt        weights for the curated best runs below (all weights are also on the private HF repo
#                               whereisravi/vision_models_weights)
#   runs/qat_int8_work          QAT / static-INT8 ONNX models and logs
#   design-decisions/           private design-choices doc and LinkedIn drafts (not in the public repo)
#   MANIFEST.txt, SHA256SUMS
# Excluded: datasets (re-downloadable), resume checkpoints (ckpt.pt), .venv, files over 1 GB.
set -euo pipefail
cd "$(dirname "$0")"
WS=/home/ubuntu/workspace
OUT=${OUT:-$WS/vision_models_archive_$(date -u +%Y-%m-%d).zip}
STAGE=$(mktemp -d /tmp/claude-1000/archive.XXXXXX)
DRY=${DRY:-0}

BEST=(
  # CIFAR-100 pure ViT: teachers, distilled students (2.75M / 1.25M / 0.71M), long and patch-2 variants
  c100_spt_slvit_1000ep c100_spt_slvit_1000ep_s1 c100_spt_slvit_1000ep_s2 c100_spt_slvit_2000ep c100_spt_slvit_p2_1000ep
  c100_kd_ens3_fm_1000ep c100_kd_ens3_fm_1000ep_s1 c100_kd_ens3_fm_1000ep_s2 c100_kd_ens4_fm_1000ep
  c100_kd_ens3_fm_2000ep c100_kd_ens3_fm_2000ep_s1
  c100_kd_ens3_fm_d128_1000ep c100_kd_ens3_fm_d128_1000ep_s1 c100_kd_ens3_fm_d128_1000ep_s2
  c100_kd_ens3_fm_d96_1000ep c100_kd_ens3_fm_d96_1000ep_s1 c100_kd_ens3_fm_d96_1000ep_s2
  c100_rd_r8_add_500ep
  # CIFAR-10 pure ViT
  c10_spt_slvit_1000ep c10_spt_slvit_1000ep_s1 c10_spt_slvit_1000ep_s2 c10_kd_ens3_fm_1000ep
  c10_spt_tinyshape_1000ep c10_spt_tinyshape_1000ep_s1 c10_spt_tinyshape_1000ep_s2
  # Tiny-ImageNet pure ViT
  tin_spt_slvit_p8_1000ep tin_spt_slvit_p8_1000ep_s1 tin_kd_ba_fm_1000ep tin_kd_ba_fm_1000ep_s1
  tin_kd_ens2_fm_1000ep tin_kd_ens2_fm_1000ep_s1
  # ResNet-18 controls
  tin_r18w32_sgd_1000ep tin_r18w56_sgd_1000ep tin_r18_sgd_1000ep tin_r18w32_kd_ens2_fm_1000ep
  c100_r18w32_sgd_1000ep c100_r18w32_sgd_1000ep_s1 c100_r18w32_kd_ens3_fm_1000ep c100_r18w32_kd_ens3_fm_1000ep_s1
  c10_r18w32_sgd_1000ep
  # Our ImageNet-21k pretraining and its fine-tunes
  in21kpre_tiny_p8 in21kpre_tiny_p8_e60 in21kpre_slvit_spt_p8
  ec100_in21kpre_tiny_p4_64_kd_200ep ec100_in21kpre_slvit_spt_p8_128_kd_60ep
  # Edge models (ImageNet-pretrained timm) and the ViT-B teacher used to distill them
  ec100_fm_tinyvit5m_224_e200 ec100_fm_tinyvit5m_224_e200_s1 ec100_fm_repvit_224_e200 ec100_fm_repvit_224_e200_s1
  ec100_T_vitb_ar2_224
)

mkdir -p "$STAGE/runs"
# 1. small files for every run
( cd runs && find . -type f \( -name "*.json" -o -name "*.jsonl" -o -name "*.log" -o -name "*.md" -o -name "*.csv" -o -name "*.png" -o -name "*.txt" -o -name "*.out" -o -path "*/wandb/*" \) \
    -size -200M -print0 | xargs -0 -I{} cp --parents {} "$STAGE/runs/" )
# 2. weights of the curated runs (EMA weights too where they exist; teacher: EMA only)
missing=()
for r in "${BEST[@]}"; do
  [[ -d runs/$r ]] || { missing+=($r); continue; }
  mkdir -p "$STAGE/runs/$r"
  if [[ $r == ec100_T_vitb_ar2_224 ]]; then cp runs/$r/model_ema.pt "$STAGE/runs/$r/"; continue; fi
  for f in runs/$r/model.pt runs/$r/model_ema.pt; do [[ -f $f ]] && cp "$f" "$STAGE/runs/$r/"; done
done
# 3. QAT / INT8 work dir
cp -r runs/qat_int8_work "$STAGE/runs/"
# 4. git bundle of all branches + private docs
mkdir -p "$STAGE/repo" "$STAGE/design-decisions"
git -C .. bundle create "$STAGE/repo/vision_models.bundle" --all 2>/dev/null
cp -r $WS/design-decisions/. "$STAGE/design-decisions/"
# 5. drop anything over 1 GB (none expected) and write the manifest
find "$STAGE" -type f -size +1G -print -delete | sed 's/^/dropped (>1GB): /'
( cd "$STAGE" && {
  echo "vision_models archive, built $(date -u). Repo: github.com/RNA8/vision_models (bundle in repo/)."
  echo "All run weights: private HF repo whereisravi/vision_models_weights. Datasets not included (re-downloadable)."
  echo "Curated runs with weights: ${#BEST[@]} listed, missing: ${missing[*]:-none}"
  echo; find . -type f ! -name MANIFEST.txt ! -name SHA256SUMS -printf "%10s  %p\n" | sort -k2
} > MANIFEST.txt
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
du -sh "$STAGE"
[[ ${#missing[@]} -gt 0 ]] && echo "missing runs: ${missing[*]}"
if [[ $DRY == 1 ]]; then rm -rf "$STAGE"; exit 0; fi
rm -f "$OUT"
( cd "$STAGE" && zip -q -r "$OUT" . )
rm -rf "$STAGE"
ls -lh "$OUT"
unzip -tq "$OUT"
