#!/usr/bin/env bash
set -eo pipefail

# Fix raw dirs, rename to VBench names, run VBench eval for one quant tag.
# Usage: bash scripts/postprocess_vbench_quant.sh <tag> [gpu_id]

TAG="${1:?tag required}"
GPU_ID="${2:-7}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_DIR="${ROOT}/logs/eval_vbench_subset25"
RAW_LEGACY="${EVAL_DIR}/${TAG}_videos_opensora_opensora"
RAW_CANON="${EVAL_DIR}/${TAG}_videos_opensora"

# Wait until 25 sample_*.mp4 exist (legacy or canonical raw dir)
echo "=== Waiting for 25 samples: ${TAG} ==="
while true; do
  n=0
  for d in "${RAW_LEGACY}" "${RAW_CANON}"; do
    if [[ -d "${d}" ]]; then
      c=$(find "${d}" -maxdepth 1 -name 'sample_*.mp4' 2>/dev/null | wc -l)
      n=$(( c > n ? c : n ))
    fi
  done
  echo "[${TAG}] samples: ${n}/25"
  if [[ "${n}" -ge 25 ]]; then
    break
  fi
  sleep 120
done

bash "${ROOT}/scripts/fix_vbench_raw_dirs.sh" "${TAG}"
python3 "${ROOT}/scripts/rename_vbench_subset_videos.py" --tag "${TAG}" --eval-dir "${EVAL_DIR}"
bash "${ROOT}/scripts/eval_vbench_quant_only.sh" "${TAG}" "${GPU_ID}"
python3 "${ROOT}/scripts/summarize_vbench_subset.py" >> "${EVAL_DIR}/summarize.log" 2>&1 || true
echo "=== ${TAG} postprocess DONE ==="
