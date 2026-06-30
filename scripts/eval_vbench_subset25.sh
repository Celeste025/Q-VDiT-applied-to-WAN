#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIDEO_ROOT="${1:-${ROOT}/logs/eval_vbench_subset25/fp16_videos}"
OUTPUT_ROOT="${2:-${ROOT}/logs/eval_vbench_subset25/vbench_results}"
SUBSET_JSON="${ROOT}/logs/eval_vbench_subset25/vbench_subset_25.json"
VBENCH_DIR="${ROOT}/../ViDiT-Q-viditq/eval/video/Vbench"
GPU_ID="${3:-0}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HUGGINGFACE_HUB_ENDPOINT="${HF_ENDPOINT}"
export VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-${ROOT}/logs/vbench_cache}"
export PYTHONPATH="${VBENCH_DIR}:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_ROOT}" "${VBENCH_CACHE_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qvdit

cd "${VBENCH_DIR}"

echo "=== VBench eval on ${VIDEO_ROOT} (GPU ${GPU_ID}, env: qvdit) ==="
CUDA_VISIBLE_DEVICES="${GPU_ID}" python evaluate.py \
  --output_path "${OUTPUT_ROOT}" \
  --full_json_dir "${SUBSET_JSON}" \
  --videos_path "${VIDEO_ROOT}" \
  --dimension \
    aesthetic_quality \
    imaging_quality \
    overall_consistency \
    scene \
    background_consistency \
    subject_consistency \
    dynamic_degree \
    motion_smoothness \
  --load_ckpt_from_local True \
  2>&1 | tee "${OUTPUT_ROOT}/eval.log"

echo "=== Done. Results under ${OUTPUT_ROOT} ==="
