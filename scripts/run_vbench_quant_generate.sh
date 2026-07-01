#!/usr/bin/env bash
set -eo pipefail

# VBench subset-25: generate 25 quant videos + rename for VBench.
# Usage: bash scripts/run_vbench_quant_generate.sh <exp_name> [gpu_id]
# Example: bash scripts/run_vbench_quant_generate.sh w4a6_ours 0
#
# quant_txt2video appends "_opensora" to --save_dir automatically; pass
# ${TAG}_videos (no suffix) so output lands in ${TAG}_videos_opensora/.

EXP_NAME="${1:?exp name required (e.g. w4a6_ours)}"
GPU_ID="${2:-0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_DIR="${ROOT}/logs/eval_vbench_subset25"
OUTDIR="${ROOT}/logs_100steps/${EXP_NAME}"
PROMPT_FILE="${EVAL_DIR}/prompts.txt"
EMBED_PATH="${EVAL_DIR}/text_embeds.pth"
CFG="${ROOT}/t2v/configs/quant/opensora/16x512x512.py"
CKPT_PATH="${ROOT}/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth"
MP_DIR="${ROOT}/t2v/configs/quant/opensora/mixed_precision"

case "${EXP_NAME}" in
  w8a8_ours) TAG="w8a8" ;;
  w4a6_ours) TAG="w4a6" ;;
  w4a4_ours) TAG="w4a4" ;;
  w3a6_ours) TAG="w3a6" ;;
  *)
    echo "Unknown EXP_NAME: ${EXP_NAME}" >&2
    exit 1
    ;;
esac

SAVE_DIR="${EVAL_DIR}/${TAG}_videos"
RAW_DIR="${SAVE_DIR}_opensora"
VIDEO_DIR="${SAVE_DIR}"
GEN_LOG="${EVAL_DIR}/${TAG}_generate.log"

MP_EXTRA=()
case "${EXP_NAME}" in
  w4a6_ours)
    MP_EXTRA=(--time_mp_config_weight "${MP_DIR}/weight_4_mp.yaml" --time_mp_config_act "${MP_DIR}/act_6_mp.yaml")
    ;;
  w4a4_ours)
    MP_EXTRA=(--time_mp_config_weight "${MP_DIR}/weight_4_mp.yaml" --time_mp_config_act "${MP_DIR}/act_4_mp.yaml")
    ;;
  w3a6_ours)
    MP_EXTRA=(--time_mp_config_weight "${MP_DIR}/weight_3_mp.yaml" --time_mp_config_act "${MP_DIR}/act_6_mp.yaml")
    ;;
  w8a8_ours)
    ;;
esac

mkdir -p "${EVAL_DIR}" "${VIDEO_DIR}"

if [[ ! -f "${OUTDIR}/ckpt.pth" ]]; then
  echo "ERROR: missing ckpt: ${OUTDIR}/ckpt.pth" | tee "${GEN_LOG}"
  exit 1
fi

cd "${ROOT}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qvdit
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "=== [1/2] ${EXP_NAME} generate 25 VBench videos (GPU ${GPU_ID}) ===" | tee "${GEN_LOG}"
echo "save_dir arg: ${SAVE_DIR} -> writes to ${RAW_DIR}" | tee -a "${GEN_LOG}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" python t2v/scripts/quant_txt2video.py "${CFG}" \
  --outdir "${OUTDIR}" \
  --ckpt_path "${CKPT_PATH}" \
  --dataset_type opensora \
  --part_fp \
  --precompute_text_embeds "${EMBED_PATH}" \
  --prompt_path "${PROMPT_FILE}" \
  --save_dir "${SAVE_DIR}" \
  --num_videos 25 \
  "${MP_EXTRA[@]}" 2>&1 | tee -a "${GEN_LOG}"

echo "=== [2/2] Rename sample_*.mp4 -> {prompt}.mp4 for VBench ===" | tee -a "${GEN_LOG}"
python3 "${ROOT}/scripts/rename_vbench_subset_videos.py" \
  --tag "${TAG}" \
  --eval-dir "${EVAL_DIR}" 2>&1 | tee -a "${GEN_LOG}"

VIDEO_COUNT="$(find "${VIDEO_DIR}" -maxdepth 1 -name '*.mp4' | wc -l)"
echo "VBench-ready videos: ${VIDEO_COUNT}/25 in ${VIDEO_DIR}" | tee -a "${GEN_LOG}"
if [[ "${VIDEO_COUNT}" -lt 25 ]]; then
  echo "ERROR: expected 25 videos, got ${VIDEO_COUNT}" | tee -a "${GEN_LOG}"
  exit 1
fi

echo "=== VBENCH SUBSET25 ${TAG} GENERATE DONE ===" | tee -a "${GEN_LOG}"
