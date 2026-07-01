#!/usr/bin/env bash
set -eo pipefail

# Background VBench eval for w8a8 subset-25 in tmux.
# Usage: bash scripts/run_tmux_w8a8_vbench_eval.sh [gpu_id] [session_name]

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${1:-7}"
SESSION="${2:-qvdit_w8a8_vbench_eval}"
EVAL_DIR="${ROOT}/logs/eval_vbench_subset25"
VIDEO_DIR="${EVAL_DIR}/w8a8_videos"
EVAL_LOG="${EVAL_DIR}/w8a8_vbench_eval_rerun.log"

video_count="$(find "${VIDEO_DIR}" -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l)"
if [[ "${video_count}" -lt 25 ]]; then
  echo "ERROR: expected 25 videos in ${VIDEO_DIR}, found ${video_count}" >&2
  exit 1
fi
echo "Found ${video_count} w8a8 videos in ${VIDEO_DIR}"

tmux kill-session -t "${SESSION}" 2>/dev/null || true

tmux new-session -d -s "${SESSION}" -n "w8a8_vbench" \
  "bash '${ROOT}/scripts/eval_w8a8_vbench_only.sh' '${GPU_ID}' 2>&1 | tee -a '${EVAL_LOG}'; bash -l"

echo "Started tmux session: ${SESSION} (w8a8 VBench on GPU ${GPU_ID})"
echo "  attach: tmux attach -t ${SESSION}"
echo "  log: ${EVAL_LOG}"
