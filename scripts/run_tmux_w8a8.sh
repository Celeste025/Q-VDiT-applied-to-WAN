#!/usr/bin/env bash
set -eo pipefail

# Full w8a8_ours pipeline: calib (5000 iters) + quant inference (10 videos)
# Usage:
#   bash scripts/run_tmux_w8a8.sh [gpu_id] [session_name]
# Attach: tmux attach -t qvdit_w8a8

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${1:-0}"
SESSION="${2:-qvdit_w8a8}"
LOG_DIR="${ROOT}/logs_100steps/w8a8_ours"
CALIB_LOG="${LOG_DIR}/calib.log"
QUANT_LOG="${LOG_DIR}/quant_inference.log"

mkdir -p "${LOG_DIR}"

tmux kill-session -t "${SESSION}" 2>/dev/null || true

tmux new-session -d -s "${SESSION}" -n "w8a8_ours" \
  "cd '${ROOT}' && source \"$(conda info --base)/etc/profile.d/conda.sh\" && conda activate qvdit && \
   echo '=== [1/2] w8a8 calib (GPU ${GPU_ID}) ===' | tee '${CALIB_LOG}' && \
   bash t2v/shell_scripts/calib.sh ${GPU_ID} 2>&1 | tee -a '${CALIB_LOG}' && \
   echo '=== [2/2] w8a8 quant inference ===' | tee -a '${QUANT_LOG}' && \
   bash t2v/shell_scripts/quant_inference.sh ${GPU_ID} 2>&1 | tee -a '${QUANT_LOG}' && \
   echo '=== W8A8 DONE ===' | tee -a '${QUANT_LOG}' && \
   bash -l"

echo "Started tmux session: ${SESSION} (window: w8a8_ours)"
echo "  attach: tmux attach -t ${SESSION}"
echo "  calib log: ${CALIB_LOG}"
echo "  quant log: ${QUANT_LOG}"
