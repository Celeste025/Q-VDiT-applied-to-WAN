#!/usr/bin/env bash
set -eo pipefail

# L6 smoke: 2-prompt VBench at infer_smoke resolution
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${1:-4}"
MODEL="${WAN_MODEL_PATH:-${ROOT}/../DVDQuant_rep/pretrained_models/Wan2.1-T2V-1.3B-Diffusers}"
EVAL_DIR="${ROOT}/logs_wan/eval_vbench_subset2_smoke"
FP16_DIR="${EVAL_DIR}/fp16_videos"
W8A8_DIR="${EVAL_DIR}/w8a8_videos"
SUBSET_JSON="${ROOT}/wan/assets/vbench_subset_2.json"
INFER_CFG="${ROOT}/wan/configs/infer_smoke.yaml"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate wan-qvdit
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
cd "${ROOT}"
mkdir -p "${FP16_DIR}" "${W8A8_DIR}"

echo "=== Generate FP16 smoke VBench videos ==="
CUDA_VISIBLE_DEVICES="${GPU_ID}" python wan/scripts/fp_vbench_generate.py \
  --model-path "${MODEL}" \
  --save-dir "${FP16_DIR}" \
  --subset-json "${SUBSET_JSON}" \
  --infer-config "${INFER_CFG}"

echo "=== VBench FP16 ==="
VBENCH_DIMENSIONS="aesthetic_quality imaging_quality overall_consistency" \
  SUBSET_JSON="${SUBSET_JSON}" bash scripts/eval_vbench_wan_subset25.sh \
  "${FP16_DIR}" "${EVAL_DIR}/vbench_results_fp16" "${GPU_ID}"

echo "=== Generate w8a8 smoke VBench videos ==="
CUDA_VISIBLE_DEVICES="${GPU_ID}" python wan/scripts/vbench_generate.py \
  --model-path "${MODEL}" \
  --outdir "${W8A8_DIR}" \
  --quant-config "${ROOT}/wan/configs/quant/w8a8_ours_smoke.yaml" \
  --quant-ckpt "${ROOT}/logs_wan/w8a8_ours_smoke/ckpt.pth" \
  --subset-json "${SUBSET_JSON}" \
  --infer-config "${INFER_CFG}" \
  --part-fp

echo "=== VBench w8a8 ==="
VBENCH_DIMENSIONS="aesthetic_quality imaging_quality overall_consistency" \
  SUBSET_JSON="${SUBSET_JSON}" bash scripts/eval_vbench_wan_subset25.sh \
  "${W8A8_DIR}" "${EVAL_DIR}/vbench_results_w8a8" "${GPU_ID}"

python wan/utils/vbench_compare.py \
  --fp16-results "${EVAL_DIR}/vbench_results_fp16" \
  --quant-results "${EVAL_DIR}/vbench_results_w8a8" \
  --out "${EVAL_DIR}/fp16_vs_w8a8_smoke.md"

python -c "from pathlib import Path; from wan.utils.verify import gate_l6; gate_l6(Path('${EVAL_DIR}/vbench_results_fp16')); gate_l6(Path('${EVAL_DIR}/vbench_results_w8a8'))"
echo "=== L6 smoke VBench DONE ==="
