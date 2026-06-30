#!/usr/bin/env bash
set -eo pipefail

# Quick smoke: w8a8 calib + quant inference (2 prompts, 20 calib steps, 50 opt iters)
# Usage:
#   bash scripts/run_tmux_smoke.sh [gpu_id] [session_name]
# Attach: tmux attach -t qvdit_smoke

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${1:-0}"
SESSION="${2:-qvdit_smoke}"
LOG_DIR="${ROOT}/logs_smoke"
CALIB_LOG="${LOG_DIR}/calib_smoke.log"
QUANT_LOG="${LOG_DIR}/quant_smoke.log"

mkdir -p "${LOG_DIR}"

tmux kill-session -t "${SESSION}" 2>/dev/null || true

tmux new-session -d -s "${SESSION}" -n "qvdit_smoke" \
  "cd '${ROOT}' && source \"$(conda info --base)/etc/profile.d/conda.sh\" && conda activate qvdit && \
   echo '=== [1/2] calib smoke (GPU ${GPU_ID}) ===' | tee '${CALIB_LOG}' && \
   bash t2v/shell_scripts/calib_smoke.sh ${GPU_ID} 2>&1 | tee -a '${CALIB_LOG}' && \
   echo '=== [2/2] quant inference smoke ===' | tee -a '${QUANT_LOG}' && \
   bash t2v/shell_scripts/quant_inference_smoke.sh ${GPU_ID} 2>&1 | tee -a '${QUANT_LOG}' && \
   echo '=== SMOKE DONE ===' | tee -a '${QUANT_LOG}' && \
   bash -l"

echo "Started tmux session: ${SESSION} (window: qvdit_smoke)"
echo "  attach: tmux attach -t ${SESSION}"
echo "  calib log: ${CALIB_LOG}"
echo "  quant log: ${QUANT_LOG}"
