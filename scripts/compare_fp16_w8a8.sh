#!/usr/bin/env bash
set -eo pipefail

# Extract middle frame from FP16 vs w8a8 smoke videos for quick visual comparison.
# Usage: bash scripts/compare_fp16_w8a8.sh [sample_idx]

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IDX="${1:-0}"
OUT_DIR="${ROOT}/logs_smoke/compare_fp16_vs_w8a8"
FP16="${ROOT}/logs/fp16_inference/generated_videos/sample_${IDX}.mp4"
W8A8="${ROOT}/logs_smoke/w8a8_ours_smoke/generated_videos_opensora/sample_${IDX}.mp4"
PROMPT_FILE="${ROOT}/t2v/assets/texts/t2v_samples_10.txt"

mkdir -p "${OUT_DIR}"

if [[ ! -f "${FP16}" ]]; then
  echo "Missing FP16 video: ${FP16}"
  exit 1
fi
if [[ ! -f "${W8A8}" ]]; then
  echo "Missing w8a8 video: ${W8A8}"
  exit 1
fi

PROMPT="$(sed -n "$((IDX + 1))p" "${PROMPT_FILE}")"
echo "${PROMPT}" > "${OUT_DIR}/prompt_${IDX}.txt"

mid_frame() {
  local src="$1" dst="$2"
  local dur mid
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "${src}")
  mid=$(python3 -c "print(float('${dur}') / 2)")
  ffmpeg -y -ss "${mid}" -i "${src}" -frames:v 1 -q:v 2 "${dst}" 2>/dev/null
}

mid_frame "${FP16}" "${OUT_DIR}/fp16_frame_${IDX}.jpg"
mid_frame "${W8A8}" "${OUT_DIR}/w8a8_smoke_frame_${IDX}.jpg"

ffmpeg -y \
  -i "${OUT_DIR}/fp16_frame_${IDX}.jpg" \
  -i "${OUT_DIR}/w8a8_smoke_frame_${IDX}.jpg" \
  -filter_complex "[0:v][1:v]hstack=inputs=2[v]" \
  -map "[v]" "${OUT_DIR}/side_by_side_${IDX}.jpg" 2>/dev/null

echo "FP16:   ${FP16}"
echo "W8A8:   ${W8A8}"
echo "Frames: ${OUT_DIR}/side_by_side_${IDX}.jpg"
