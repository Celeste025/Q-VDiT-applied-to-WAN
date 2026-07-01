#!/usr/bin/env bash
set -eo pipefail

# VBench eval only for w8a8 subset-25 (videos must exist under w8a8_videos/).
# Usage: bash scripts/eval_w8a8_vbench_only.sh [gpu_id]

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${1:-7}"
EVAL_DIR="${ROOT}/logs/eval_vbench_subset25"
VIDEO_DIR="${EVAL_DIR}/w8a8_videos"
RESULT_DIR="${EVAL_DIR}/vbench_results_w8a8"
EVAL_LOG="${EVAL_DIR}/w8a8_vbench_eval.log"

video_count="$(find "${VIDEO_DIR}" -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l)"
if [[ "${video_count}" -lt 25 ]]; then
  echo "ERROR: expected 25 videos in ${VIDEO_DIR}, found ${video_count}" >&2
  echo "Run: bash scripts/run_vbench_w8a8_pipeline.sh <gpu>" >&2
  exit 1
fi

echo "=== VBench eval w8a8 (${video_count} videos, GPU ${GPU_ID}) ==="
bash "${ROOT}/scripts/eval_vbench_subset25.sh" "${VIDEO_DIR}" "${RESULT_DIR}" "${GPU_ID}" \
  2>&1 | tee "${EVAL_LOG}"

echo "=== Done. Results: ${RESULT_DIR}/aesthetic_quality_eval_results.json ==="
