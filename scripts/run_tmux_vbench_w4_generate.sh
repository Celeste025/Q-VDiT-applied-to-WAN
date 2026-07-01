#!/usr/bin/env bash
set -eo pipefail

# Launch w4a6 + w4a4 VBench subset-25 video generation on separate GPUs (tmux).
# Usage: bash scripts/run_tmux_vbench_w4_generate.sh [w4a6_gpu] [w4a4_gpu]

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
W4A6_GPU="${1:-0}"
W4A4_GPU="${2:-1}"
EVAL_DIR="${ROOT}/logs/eval_vbench_subset25"

for exp in w4a6_ours w4a4_ours; do
  ckpt="${ROOT}/logs_100steps/${exp}/ckpt.pth"
  if [[ ! -f "${ckpt}" ]]; then
    echo "ERROR: missing ${ckpt}" >&2
    exit 1
  fi
done

launch() {
  local exp="$1"
  local gpu="$2"
  local session="$3"
  local log="${EVAL_DIR}/${exp/_ours/}_generate_tmux.log"
  tmux kill-session -t "${session}" 2>/dev/null || true
  tmux new-session -d -s "${session}" -n "gen" \
    "bash '${ROOT}/scripts/run_vbench_quant_generate.sh' '${exp}' '${gpu}' 2>&1 | tee '${log}'; bash -l"
  echo "Started ${session}: ${exp} on GPU ${gpu}"
  echo "  attach: tmux attach -t ${session}"
  echo "  log: ${log}"
}

launch w4a6_ours "${W4A6_GPU}" "qvdit_vbench_w4a6_gen"
launch w4a4_ours "${W4A4_GPU}" "qvdit_vbench_w4a4_gen"

echo ""
echo "Outputs (when done):"
echo "  w4a6: ${EVAL_DIR}/w4a6_videos/"
echo "  w4a4: ${EVAL_DIR}/w4a4_videos/"
