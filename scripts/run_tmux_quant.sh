#!/usr/bin/env bash
set -eo pipefail

# Launch quant pipeline in tmux
# Usage: bash scripts/run_tmux_quant.sh <exp_name> <gpu_id> [session_name]

EXP_NAME="${1:?exp name required}"
GPU_ID="${2:?gpu id required}"
SESSION="${3:-qvdit_${EXP_NAME}}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT}/logs_100steps/${EXP_NAME}"
PIPELINE_LOG="${LOG_DIR}/pipeline.log"

mkdir -p "${LOG_DIR}"

tmux kill-session -t "${SESSION}" 2>/dev/null || true

tmux new-session -d -s "${SESSION}" -n "${EXP_NAME}" \
  "bash '${ROOT}/scripts/run_quant_pipeline.sh' '${EXP_NAME}' '${GPU_ID}' 2>&1 | tee '${PIPELINE_LOG}'; bash -l"

echo "Started tmux session: ${SESSION} (${EXP_NAME} on GPU ${GPU_ID})"
echo "  attach: tmux attach -t ${SESSION}"
echo "  log: ${PIPELINE_LOG}"
