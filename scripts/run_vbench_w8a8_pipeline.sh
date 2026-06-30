#!/usr/bin/env bash
set -eo pipefail

# VBench subset-25: w8a8 quant generate 25 videos + VBench eval
# Usage: bash scripts/run_vbench_w8a8_pipeline.sh [gpu_id]

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${1:-0}"
EVAL_DIR="${ROOT}/logs/eval_vbench_subset25"
W8A8_OUTDIR="${ROOT}/logs_100steps/w8a8_ours"
GEN_LOG="${EVAL_DIR}/w8a8_generate.log"
EVAL_LOG="${EVAL_DIR}/w8a8_vbench_eval.log"
PROMPT_FILE="${EVAL_DIR}/prompts.txt"
EMBED_PATH="${EVAL_DIR}/text_embeds.pth"
VIDEO_DIR="${EVAL_DIR}/w8a8_videos"
RAW_DIR="${VIDEO_DIR}_opensora"
RESULT_DIR="${EVAL_DIR}/vbench_results_w8a8"
CFG="${ROOT}/t2v/configs/quant/opensora/16x512x512.py"
CKPT_PATH="${ROOT}/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth"

mkdir -p "${EVAL_DIR}" "${VIDEO_DIR}" "${RESULT_DIR}"

cd "${ROOT}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qvdit
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export VBENCH_CACHE_DIR="${ROOT}/logs/vbench_cache"

if [[ ! -f "${W8A8_OUTDIR}/ckpt.pth" ]]; then
  echo "ERROR: missing w8a8 ckpt: ${W8A8_OUTDIR}/ckpt.pth" | tee "${GEN_LOG}"
  exit 1
fi

echo "=== [1/3] w8a8 generate 25 VBench videos (GPU ${GPU_ID}) ===" | tee "${GEN_LOG}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" python t2v/scripts/quant_txt2video.py "${CFG}" \
  --outdir "${W8A8_OUTDIR}" \
  --ckpt_path "${CKPT_PATH}" \
  --dataset_type opensora \
  --part_fp \
  --precompute_text_embeds "${EMBED_PATH}" \
  --prompt_path "${PROMPT_FILE}" \
  --save_dir "${VIDEO_DIR}" \
  --num_videos 25 2>&1 | tee -a "${GEN_LOG}"

echo "=== [2/3] Rename sample_*.mp4 -> {prompt}.mp4 for VBench ===" | tee -a "${GEN_LOG}"
python3 - <<'PY' | tee -a "${GEN_LOG}"
from pathlib import Path
import shutil

root = Path("/data/home/jinqiwen/workspace/video-distilation/Q-VDiT/logs/eval_vbench_subset25")
prompts = [line.strip() for line in (root / "prompts.txt").read_text().splitlines() if line.strip()]
raw_dir = root / "w8a8_videos_opensora"
out_dir = root / "w8a8_videos"
out_dir.mkdir(parents=True, exist_ok=True)

for i, prompt in enumerate(prompts):
    src = raw_dir / f"sample_{i}.mp4"
    dst = out_dir / f"{prompt}.mp4"
    if not src.is_file():
        raise SystemExit(f"missing generated video: {src}")
    shutil.copy2(src, dst)
    print(f"[ok] {dst.name}")

print(f"Renamed {len(prompts)} videos -> {out_dir}")
PY

VIDEO_COUNT="$(find "${VIDEO_DIR}" -maxdepth 1 -name '*.mp4' | wc -l)"
echo "VBench-ready videos: ${VIDEO_COUNT}/25" | tee -a "${GEN_LOG}"
if [[ "${VIDEO_COUNT}" -lt 25 ]]; then
  echo "ERROR: expected 25 videos, got ${VIDEO_COUNT}" | tee -a "${EVAL_LOG}"
  exit 1
fi

echo "=== [3/3] VBench eval w8a8 (GPU ${GPU_ID}) ===" | tee "${EVAL_LOG}"
bash scripts/eval_vbench_subset25.sh "${VIDEO_DIR}" "${RESULT_DIR}" "${GPU_ID}" 2>&1 | tee -a "${EVAL_LOG}"

echo "=== VBENCH SUBSET25 w8a8 DONE (${VIDEO_COUNT} videos) ===" | tee -a "${EVAL_LOG}"
