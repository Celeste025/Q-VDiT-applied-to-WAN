#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${1:-7}"
EVAL_DIR="${ROOT}/logs/eval_vbench_subset25"
GEN_LOG="${EVAL_DIR}/fp16_generate.log"
EVAL_LOG="${EVAL_DIR}/vbench_eval.log"
EMBED_PATH="${EVAL_DIR}/text_embeds.pth"
VIDEO_DIR="${EVAL_DIR}/fp16_videos"
RESULT_DIR="${EVAL_DIR}/vbench_results"

mkdir -p "${EVAL_DIR}" "${VIDEO_DIR}" "${RESULT_DIR}"

cd "${ROOT}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qvdit
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export VBENCH_CACHE_DIR="${ROOT}/logs/vbench_cache"

python3 "${ROOT}/scripts/select_vbench_subset_25.py" --root "${ROOT}"

echo "=== [0/3] Precompute text embeds (GPU ${GPU_ID}) ===" | tee "${GEN_LOG}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 "${ROOT}/scripts/precompute_vbench_text_embeds.py" \
  --gpu 0 \
  --out "${EMBED_PATH}" 2>&1 | tee -a "${GEN_LOG}"

echo "=== [1/3] FP16 generate 25 VBench videos (GPU ${GPU_ID}) ===" | tee -a "${GEN_LOG}"
export VBENCH_SUBSET_JSON="${EVAL_DIR}/vbench_subset_25.json"
CUDA_VISIBLE_DEVICES="${GPU_ID}" python t2v/scripts/vbench_generate.py \
  ./t2v/configs/opensora/inference/vbench_16x512x512.py \
  --ckpt_path ./logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth \
  --outdir "${EVAL_DIR}" \
  --save_dir "${VIDEO_DIR}" \
  --prompt_path "${EVAL_DIR}/prompts.txt" \
  --precompute_text_embeds "${EMBED_PATH}" \
  --sampler ddim 2>&1 | tee -a "${GEN_LOG}"

VIDEO_COUNT="$(find "${VIDEO_DIR}" -maxdepth 1 -name '*.mp4' | wc -l)"
echo "Generated videos: ${VIDEO_COUNT}/25" | tee -a "${GEN_LOG}"
if [[ "${VIDEO_COUNT}" -lt 1 ]]; then
  echo "ERROR: no videos generated, skip VBench eval" | tee -a "${EVAL_LOG}"
  exit 1
fi

echo "=== [2/3] VBench eval (GPU ${GPU_ID}) ===" | tee "${EVAL_LOG}"
bash scripts/eval_vbench_subset25.sh "${VIDEO_DIR}" "${RESULT_DIR}" "${GPU_ID}" 2>&1 | tee -a "${EVAL_LOG}"

echo "=== VBENCH SUBSET25 DONE (${VIDEO_COUNT} videos) ===" | tee -a "${EVAL_LOG}"
