#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIDEO_ROOT="${1:-${ROOT}/logs_wan/eval_vbench_subset25/w8a8_videos}"
OUTPUT_ROOT="${2:-${ROOT}/logs_wan/eval_vbench_subset25/vbench_results_w8a8}"
if [[ "${VIDEO_ROOT}" != /* ]]; then
  VIDEO_ROOT="${ROOT}/${VIDEO_ROOT#./}"
fi
if [[ "${OUTPUT_ROOT}" != /* ]]; then
  OUTPUT_ROOT="${ROOT}/${OUTPUT_ROOT#./}"
fi
SUBSET_JSON="${SUBSET_JSON:-${ROOT}/wan/assets/vbench_subset_25.json}"
if [[ "${SUBSET_JSON}" != /* ]]; then
  SUBSET_JSON="${ROOT}/${SUBSET_JSON#./}"
fi
VBENCH_DIR="${ROOT}/../ViDiT-Q-viditq/eval/video/Vbench"
GPU_ID="${3:-4}"
VBENCH_DIMENSIONS="${VBENCH_DIMENSIONS:-aesthetic_quality imaging_quality overall_consistency background_consistency subject_consistency dynamic_degree motion_smoothness scene}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HUGGINGFACE_HUB_ENDPOINT="${HF_ENDPOINT}"
export VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-${ROOT}/logs/vbench_cache}"
export PYTHONPATH="${VBENCH_DIR}:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_ROOT}" "${VBENCH_CACHE_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qvdit
cd "${VBENCH_DIR}"

echo "=== VBench eval Wan videos ${VIDEO_ROOT} (GPU ${GPU_ID}) ==="
CUDA_VISIBLE_DEVICES="${GPU_ID}" python evaluate.py \
  --output_path "${OUTPUT_ROOT}" \
  --full_json_dir "${SUBSET_JSON}" \
  --videos_path "${VIDEO_ROOT}" \
  --dimension \
    ${VBENCH_DIMENSIONS} \
  --load_ckpt_from_local True \
  2>&1 | tee "${OUTPUT_ROOT}/eval.log"

echo "=== Done. Results under ${OUTPUT_ROOT} ==="
