#!/usr/bin/env bash
set -eo pipefail

# Re-run VBench eval only (videos must already exist).
# Usage: bash scripts/run_tmux_vbench_eval.sh [gpu_id] [session_name]

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${1:-7}"
SESSION="${2:-qvdit_vbench_eval}"
EVAL_DIR="${ROOT}/logs/eval_vbench_subset25"
VIDEO_DIR="${EVAL_DIR}/fp16_videos"
RESULT_DIR="${EVAL_DIR}/vbench_results"
EVAL_LOG="${RESULT_DIR}/eval.log"

mkdir -p "${EVAL_DIR}" "${RESULT_DIR}"

video_count="$(find "${VIDEO_DIR}" -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l)"
if [[ "${video_count}" -lt 1 ]]; then
  echo "ERROR: no videos in ${VIDEO_DIR}, run generation first" >&2
  exit 1
fi
echo "Found ${video_count} videos in ${VIDEO_DIR}"

tmux kill-session -t "${SESSION}" 2>/dev/null || true

tmux new-session -d -s "${SESSION}" -n "vbench_eval" \
  "bash '${ROOT}/scripts/eval_vbench_subset25.sh' '${VIDEO_DIR}' '${RESULT_DIR}' '${GPU_ID}' 2>&1 | tee '${EVAL_DIR}/vbench_eval_rerun.log'; bash -l"

echo "Started tmux session: ${SESSION} (VBench eval on GPU ${GPU_ID})"
echo "  attach: tmux attach -t ${SESSION}"
echo "  log: ${EVAL_DIR}/vbench_eval_rerun.log"
