#!/usr/bin/env bash
set -eo pipefail

# FP16 vs w8a8 smoke: in-calib and out-of-calib prompt comparisons.
# Usage:
#   bash scripts/compare_fp16_w8a8_dual.sh [gpu_id]
#   bash scripts/compare_fp16_w8a8_dual.sh [gpu_id] --skip-infer   # only extract frames

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${1:-1}"
SKIP_INFER=false
if [[ "${2:-}" == "--skip-infer" ]]; then
  SKIP_INFER=true
fi

OUT_DIR="${ROOT}/logs_smoke/compare_fp16_vs_w8a8"
PROMPT_FILE="${ROOT}/t2v/assets/texts/t2v_samples_10.txt"
W8A8_DIR="${ROOT}/logs_smoke/w8a8_ours_smoke/generated_videos_opensora"
FP16_DIR="${ROOT}/logs/fp16_inference/generated_videos"

# smoke calib used n_samples=2 -> prompt #0,#1 in-calib; #2+ out-of-calib
IN_CALIB_IDX=0
OUT_CALIB_IDX=2

mkdir -p "${OUT_DIR}"

run_out_calib_infer() {
  # batch_ids follows prompt index in the file; must use full list for correct text_embeds.
  local num_videos=$((OUT_CALIB_IDX + 1))

  echo "=== Generating w8a8 smoke videos 0..${OUT_CALIB_IDX} (need #${OUT_CALIB_IDX}, GPU ${GPU_ID}) ==="
  cd "${ROOT}"
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate qvdit

  CUDA_VISIBLE_DEVICES="${GPU_ID}" python t2v/scripts/quant_txt2video.py \
    ./t2v/configs/quant/opensora/16x512x512.py \
    --outdir ./logs_smoke/w8a8_ours_smoke \
    --ckpt_path ./logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth \
    --dataset_type opensora \
    --part_fp \
    --precompute_text_embeds ./t2v/utils_files/text_embeds.pth \
    --prompt_path "${PROMPT_FILE}" \
    --num_videos "${num_videos}"
}

make_compare() {
  local idx="$1"
  local tag="$2"
  local w8a8_path="$3"

  local fp16="${FP16_DIR}/sample_${idx}.mp4"
  if [[ ! -f "${fp16}" ]]; then
    echo "Missing FP16 video: ${fp16}"
    return 1
  fi
  if [[ ! -f "${w8a8_path}" ]]; then
    echo "Missing w8a8 video: ${w8a8_path}"
    return 1
  fi

  sed -n "$((idx + 1))p" "${PROMPT_FILE}" > "${OUT_DIR}/prompt_${tag}_${idx}.txt"

  local dur mid
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "${fp16}")
  mid=$(python3 -c "print(float('${dur}') / 2)")

  ffmpeg -y -ss "${mid}" -i "${fp16}" -frames:v 1 -q:v 2 "${OUT_DIR}/fp16_${tag}_${idx}.jpg" 2>/dev/null
  ffmpeg -y -ss "${mid}" -i "${w8a8_path}" -frames:v 1 -q:v 2 "${OUT_DIR}/w8a8_${tag}_${idx}.jpg" 2>/dev/null
  ffmpeg -y \
    -i "${OUT_DIR}/fp16_${tag}_${idx}.jpg" \
    -i "${OUT_DIR}/w8a8_${tag}_${idx}.jpg" \
    -filter_complex "[0:v][1:v]hstack=inputs=2[v]" \
    -map "[v]" "${OUT_DIR}/side_by_side_${tag}_${idx}.jpg" 2>/dev/null

  echo "  [${tag}] prompt #${idx}: ${OUT_DIR}/side_by_side_${tag}_${idx}.jpg"
}

if [[ "${SKIP_INFER}" != true ]]; then
  run_out_calib_infer
fi

echo "=== Building comparisons ==="
make_compare "${IN_CALIB_IDX}" "in_calib" "${W8A8_DIR}/sample_${IN_CALIB_IDX}.mp4"
make_compare "${OUT_CALIB_IDX}" "out_calib" "${W8A8_DIR}/sample_${OUT_CALIB_IDX}.mp4"

echo ""
echo "In-calib  (prompt #${IN_CALIB_IDX}): ${OUT_DIR}/side_by_side_in_calib_${IN_CALIB_IDX}.jpg"
echo "Out-calib (prompt #${OUT_CALIB_IDX}): ${OUT_DIR}/side_by_side_out_calib_${OUT_CALIB_IDX}.jpg"
