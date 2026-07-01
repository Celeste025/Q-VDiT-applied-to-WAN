#!/usr/bin/env bash
set -eo pipefail

# VBench eval only for a quant subset (videos must exist under ${TAG}_videos/).
# Usage: bash scripts/eval_vbench_quant_only.sh <tag> [gpu_id]
# tag: w8a8 | w4a6 | w4a4 | w3a6

TAG="${1:?tag required (e.g. w4a6)}"
GPU_ID="${2:-7}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_DIR="${ROOT}/logs/eval_vbench_subset25"
VIDEO_DIR="${EVAL_DIR}/${TAG}_videos"
RESULT_DIR="${EVAL_DIR}/vbench_results_${TAG}"
EVAL_LOG="${EVAL_DIR}/${TAG}_vbench_eval.log"

video_count="$(find "${VIDEO_DIR}" -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l)"
if [[ "${video_count}" -lt 25 ]]; then
  echo "ERROR: expected 25 videos in ${VIDEO_DIR}, found ${video_count}" >&2
  exit 1
fi

echo "=== VBench eval ${TAG} (${video_count} videos, GPU ${GPU_ID}) ===" | tee "${EVAL_LOG}"
bash "${ROOT}/scripts/eval_vbench_subset25.sh" "${VIDEO_DIR}" "${RESULT_DIR}" "${GPU_ID}" \
  2>&1 | tee -a "${EVAL_LOG}"
echo "=== Done: ${RESULT_DIR}/aesthetic_quality_eval_results.json ==="
