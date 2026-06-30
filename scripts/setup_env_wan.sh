#!/usr/bin/env bash
set -eo pipefail

# Wan2.1 Q-VDiT environment (isolated from OpenSora qvdit env)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${WAN_QVDIT_ENV:-wan-qvdit}"
WAN_MODEL="${WAN_MODEL_PATH:-${ROOT}/../DVDQuant_rep/pretrained_models/Wan2.1-T2V-1.3B-Diffusers}"

echo "[setup_env_wan] root: ${ROOT}"
echo "[setup_env_wan] env: ${ENV_NAME}"
echo "[setup_env_wan] model: ${WAN_MODEL}"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" python=3.10 -y
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

pip install -q "numpy<2" "setuptools<81"
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -q "diffusers>=0.33.0" "transformers>=4.40.0" accelerate omegaconf pyyaml safetensors \
  imageio imageio-ffmpeg tqdm einops decord ftfy regex scipy easydict

pip install -e "${ROOT}"

export WAN_MODEL_PATH="${WAN_MODEL}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HUGGINGFACE_HUB_ENDPOINT="${HF_ENDPOINT}"

python - <<'PY'
import json
from pathlib import Path
import os
from diffusers import WanPipeline
import qdiff

root = Path(os.environ.get("WAN_MODEL_PATH", ""))
assert root.is_dir(), f"Wan model not found: {root}"
cfg = json.loads((root / "transformer" / "config.json").read_text())
assert cfg["num_layers"] == 30
assert cfg["text_dim"] == 4096
assert cfg["in_channels"] == 16
print("L0 OK: WanPipeline import, qdiff import, transformer config validated")
PY

echo "[setup_env_wan] done. Use: conda activate ${ENV_NAME}"
echo "[setup_env_wan] Wan dev GPUs: prefer CUDA_VISIBLE_DEVICES=4,5,6,7"
