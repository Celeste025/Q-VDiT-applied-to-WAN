#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${1:-4}"
MODEL="${WAN_MODEL_PATH:-${ROOT}/../DVDQuant_rep/pretrained_models/Wan2.1-T2V-1.3B-Diffusers}"
EVAL_DIR="${ROOT}/logs_wan/eval_vbench_subset25"
FP16_DIR="${EVAL_DIR}/fp16_videos"
W8A8_DIR="${EVAL_DIR}/w8a8_videos"
W8A8_CKPT="${ROOT}/logs_wan/w8a8_ours/ckpt.pth"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate wan-qvdit
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
cd "${ROOT}"
mkdir -p "${FP16_DIR}" "${W8A8_DIR}"

if [[ ! -d "${FP16_DIR}" ]] || [[ -z "$(ls -A "${FP16_DIR}"/*.mp4 2>/dev/null)" ]]; then
  echo "=== Generate Wan FP16 VBench subset videos ==="
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python wan/scripts/fp_vbench_generate.py \
    --model-path "${MODEL}" \
    --save-dir "${FP16_DIR}" \
    --infer-config "${ROOT}/wan/configs/infer.yaml"
fi

echo "=== VBench FP16 ==="
bash scripts/eval_vbench_wan_subset25.sh "${FP16_DIR}" "${EVAL_DIR}/vbench_results_fp16" "${GPU_ID}"

if [[ -f "${W8A8_CKPT}" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python wan/scripts/vbench_generate.py \
    --model-path "${MODEL}" \
    --outdir "${W8A8_DIR}" \
    --quant-config "${ROOT}/wan/configs/quant/w8a8_ours.yaml" \
    --quant-ckpt "${W8A8_CKPT}" \
    --infer-config "${ROOT}/wan/configs/infer.yaml" \
    --part-fp
  bash scripts/eval_vbench_wan_subset25.sh "${W8A8_DIR}" "${EVAL_DIR}/vbench_results_w8a8" "${GPU_ID}"
fi

python -c "from pathlib import Path; from wan.utils.verify import gate_l6; gate_l6(Path('${EVAL_DIR}/vbench_results_fp16'))"
