#!/usr/bin/env bash
set -eo pipefail

# Wan Q-VDiT full pipeline tuned for A100 40GB/80GB
# Usage: bash scripts/run_wan_pipeline_a100.sh <exp_name> [gpu_id]
# exp_name: w8a8_ours | w4a6_ours | w4a4_ours | w3a6_ours

EXP_NAME="${1:?exp name required (e.g. w8a8_ours)}"
GPU_ID="${2:-0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${WAN_MODEL_PATH:-${ROOT}/../DVDQuant_rep/pretrained_models/Wan2.1-T2V-1.3B-Diffusers}"
INFER_CFG="${ROOT}/wan/configs/infer.yaml"
Q_CFG="${ROOT}/wan/configs/quant/a100/${EXP_NAME}.yaml"
CALIB_DIR="${ROOT}/logs_wan/calib_data"
OUTDIR="${ROOT}/logs_wan/${EXP_NAME}_a100"
MP_DIR="${ROOT}/wan/configs/mixed_precision"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -f "${Q_CFG}" ]]; then
  echo "Missing A100 quant config: ${Q_CFG}" >&2
  exit 1
fi

MP_CALIB=()
MP_INFER=()
case "${EXP_NAME}" in
  w4a6_ours)
    MP_CALIB=(--time_mp_config_weight "${MP_DIR}/weight_4_mp.yaml" --time_mp_config_act "${MP_DIR}/act_6_mp.yaml")
    MP_INFER=(--time-mp-config-weight "${MP_DIR}/weight_4_mp.yaml" --time-mp-config-act "${MP_DIR}/act_6_mp.yaml")
    ;;
  w3a6_ours)
    MP_CALIB=(--time_mp_config_weight "${MP_DIR}/weight_3_mp.yaml" --time_mp_config_act "${MP_DIR}/act_6_mp.yaml")
    MP_INFER=(--time-mp-config-weight "${MP_DIR}/weight_3_mp.yaml" --time-mp-config-act "${MP_DIR}/act_6_mp.yaml")
    ;;
  w4a4_ours)
    MP_CALIB=(--time_mp_config_weight "${MP_DIR}/weight_4_mp.yaml" --time_mp_config_act "${MP_DIR}/act_4_mp.yaml")
    MP_INFER=(--time-mp-config-weight "${MP_DIR}/weight_4_mp.yaml" --time-mp-config-act "${MP_DIR}/act_4_mp.yaml")
    ;;
  w8a8_ours)
    ;;
  *)
    echo "Unknown EXP_NAME: ${EXP_NAME}" >&2
    exit 1
    ;;
esac

mkdir -p "${CALIB_DIR}" "${OUTDIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate wan-qvdit
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
cd "${ROOT}"

echo "=== A100 pipeline: ${EXP_NAME} on GPU ${GPU_ID} ==="
echo "=== quant config: ${Q_CFG} ==="
echo "=== PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF} ==="

echo "=== [1/3] calib data (480x832x81) ==="
CUDA_VISIBLE_DEVICES="${GPU_ID}" python wan/scripts/get_calib_data.py \
  --model-path "${MODEL}" \
  --outdir "${CALIB_DIR}" \
  --infer-config "${INFER_CFG}" \
  --num-prompts 10

echo "=== [2/3] calib ${EXP_NAME} (delta opt + grad checkpoint) ==="
CUDA_VISIBLE_DEVICES="${GPU_ID}" python wan/scripts/calib.py \
  --calib_config "${Q_CFG}" \
  --calib_data "${CALIB_DIR}/calib_data.pt" \
  --outdir "${OUTDIR}" \
  --model-path "${MODEL}" \
  --gpu 0 \
  --part_fp \
  --dtype bfloat16 \
  "${MP_CALIB[@]}"

echo "=== [3/3] quant inference ==="
CUDA_VISIBLE_DEVICES="${GPU_ID}" python wan/scripts/quant_inference.py \
  --model-path "${MODEL}" \
  --outdir "${OUTDIR}" \
  --quant-config "${Q_CFG}" \
  --quant-ckpt "${OUTDIR}/ckpt.pth" \
  --infer-config "${INFER_CFG}" \
  --num-videos 10 \
  --part-fp \
  "${MP_INFER[@]}"

echo "=== ${EXP_NAME} A100 DONE -> ${OUTDIR} ==="
