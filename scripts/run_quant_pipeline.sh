#!/usr/bin/env bash
set -eo pipefail

# Generic Q-VDiT quant pipeline: calib + quant inference
# Usage: bash scripts/run_quant_pipeline.sh <exp_name> [gpu_id]
# Example: bash scripts/run_quant_pipeline.sh w4a6_ours 1

EXP_NAME="${1:?exp name required}"
GPU_ID="${2:-1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CFG="./t2v/configs/quant/opensora/16x512x512.py"
Q_CFG="./t2v/configs/quant/opensora/${EXP_NAME}.yaml"
CKPT_PATH="./logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth"
CALIB_DATA_DIR="./logs_100steps/calib_data"
OUTDIR="./logs_100steps/${EXP_NAME}"
CALIB_LOG="${OUTDIR}/calib.log"
QUANT_LOG="${OUTDIR}/quant_inference.log"
MP_DIR="./t2v/configs/quant/opensora/mixed_precision"

MP_EXTRA=()
case "${EXP_NAME}" in
  w4a6_ours)
    MP_EXTRA=(--time_mp_config_weight "${MP_DIR}/weight_4_mp.yaml" --time_mp_config_act "${MP_DIR}/act_6_mp.yaml")
    ;;
  w3a6_ours)
    MP_EXTRA=(--time_mp_config_weight "${MP_DIR}/weight_3_mp.yaml" --time_mp_config_act "${MP_DIR}/act_6_mp.yaml")
    ;;
  w4a4_ours)
    MP_EXTRA=(--time_mp_config_weight "${MP_DIR}/weight_4_mp.yaml" --time_mp_config_act "${MP_DIR}/act_4_mp.yaml")
    ;;
  w8a8_ours)
    ;;
  *)
    echo "Unknown EXP_NAME: ${EXP_NAME}" >&2
    exit 1
    ;;
esac

mkdir -p "${OUTDIR}"
cd "${ROOT}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qvdit

echo "=== [1/2] ${EXP_NAME} calib (GPU ${GPU_ID}) ===" | tee "${CALIB_LOG}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" python t2v/scripts/calib.py "${CFG}" \
  --ckpt_path "${CKPT_PATH}" \
  --calib_config "${Q_CFG}" \
  --outdir "${OUTDIR}" \
  --calib_data "${CALIB_DATA_DIR}/calib_data.pt" \
  --part_fp \
  --precompute_text_embeds ./t2v/utils_files/text_embeds.pth \
  "${MP_EXTRA[@]}" 2>&1 | tee -a "${CALIB_LOG}"

echo "=== [2/2] ${EXP_NAME} quant inference (GPU ${GPU_ID}) ===" | tee "${QUANT_LOG}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" python t2v/scripts/quant_txt2video.py "${CFG}" \
  --outdir "${OUTDIR}" \
  --ckpt_path "${CKPT_PATH}" \
  --dataset_type opensora \
  --part_fp \
  --precompute_text_embeds ./t2v/utils_files/text_embeds.pth \
  --prompt_path t2v/assets/texts/t2v_samples_10.txt \
  "${MP_EXTRA[@]}" 2>&1 | tee -a "${QUANT_LOG}"

echo "=== ${EXP_NAME} DONE ===" | tee -a "${QUANT_LOG}"
