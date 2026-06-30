#!/usr/bin/env bash
set -eo pipefail

# VBench subset-25: precompute embeds + FP16 generation + VBench eval
# Usage: bash scripts/run_tmux_vbench_fp16.sh [gpu_id] [session_name]

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${1:-7}"
SESSION="${2:-qvdit_vbench_fp16}"
EVAL_DIR="${ROOT}/logs/eval_vbench_subset25"
GEN_LOG="${EVAL_DIR}/fp16_generate.log"
EVAL_LOG="${EVAL_DIR}/vbench_eval.log"

mkdir -p "${EVAL_DIR}"

tmux kill-session -t "${SESSION}" 2>/dev/null || true

tmux new-session -d -s "${SESSION}" -n "vbench_fp16" \
  "bash '${ROOT}/scripts/run_vbench_fp16_pipeline.sh' ${GPU_ID} 2>&1 | tee '${EVAL_DIR}/pipeline.log'; bash -l"

echo "Started tmux session: ${SESSION} (GPU ${GPU_ID})"
echo "  attach: tmux attach -t ${SESSION}"
echo "  pipeline log: ${EVAL_DIR}/pipeline.log"
echo "  gen log: ${GEN_LOG}"
echo "  eval log: ${EVAL_LOG}"
