#!/usr/bin/env bash
set -eo pipefail

# Wan Q-VDiT smoke pipeline (L1-L5 gates, smoke configs)
# Usage: bash scripts/run_wan_smoke.sh [gpu_id] [exp_name]

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${1:-4}"
EXP_NAME="${2:-w8a8_ours_smoke}"
MODEL="${WAN_MODEL_PATH:-${ROOT}/../DVDQuant_rep/pretrained_models/Wan2.1-T2V-1.3B-Diffusers}"
INFER_CFG="${ROOT}/wan/configs/infer_smoke.yaml"
Q_CFG="${ROOT}/wan/configs/quant/${EXP_NAME}.yaml"
CALIB_DIR="${ROOT}/logs_wan/calib_data_smoke"
OUTDIR="${ROOT}/logs_wan/${EXP_NAME}"

mkdir -p "${CALIB_DIR}" "${OUTDIR}" "${ROOT}/logs_wan"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate wan-qvdit
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
cd "${ROOT}"

echo "=== L0 gate ==="
python -c "from wan.utils.verify import gate_l0; from pathlib import Path; gate_l0(Path('${MODEL}'))"

echo "=== L1 FP inference ==="
CUDA_VISIBLE_DEVICES="${GPU_ID}" python wan/scripts/fp_inference.py \
  --model-path "${MODEL}" \
  --outdir "${ROOT}/logs_wan/fp16_baseline_smoke" \
  --infer-config "${INFER_CFG}" \
  --num-prompts 2 \
  --run-gate

echo "=== L2 QuantModel gate ==="
CUDA_VISIBLE_DEVICES="${GPU_ID}" python wan/scripts/verify_l2_gate.py --gpu 0

echo "=== L3 calib data ==="
CUDA_VISIBLE_DEVICES="${GPU_ID}" python wan/scripts/get_calib_data.py \
  --model-path "${MODEL}" \
  --outdir "${CALIB_DIR}" \
  --infer-config "${INFER_CFG}" \
  --num-prompts 2 \
  --run-gate

echo "=== L4 calib ==="
CUDA_VISIBLE_DEVICES="${GPU_ID}" python wan/scripts/calib.py \
  --calib_config "${Q_CFG}" \
  --calib_data "${CALIB_DIR}/calib_data.pt" \
  --outdir "${OUTDIR}" \
  --model-path "${MODEL}" \
  --gpu 0 \
  --part_fp \
  --dtype bfloat16

python -c "from wan.utils.verify import gate_l4; from pathlib import Path; gate_l4(Path('${OUTDIR}/ckpt.pth'))"

echo "=== L5 quant inference ==="
CUDA_VISIBLE_DEVICES="${GPU_ID}" python wan/scripts/quant_inference.py \
  --model-path "${MODEL}" \
  --outdir "${OUTDIR}" \
  --quant-config "${Q_CFG}" \
  --quant-ckpt "${OUTDIR}/ckpt.pth" \
  --infer-config "${INFER_CFG}" \
  --num-videos 2 \
  --part-fp \
  --run-gate

echo "=== Wan smoke pipeline DONE: ${EXP_NAME} ==="
