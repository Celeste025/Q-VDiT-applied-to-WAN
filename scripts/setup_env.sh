#!/usr/bin/env bash
set -eo pipefail

# Q-VDiT conda environment setup (no flash-attn)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${QVDIT_ENV:-qvdit}"

echo "[setup_env] project root: ${ROOT}"
echo "[setup_env] conda env: ${ENV_NAME}"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" python=3.10 -y
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y

pip install "numpy<2"
pip install "setuptools<81"
pip install colossalai==0.4.0 gdown mmengine pre-commit av wandb
pip install bitsandbytes==0.43.1
pip install diffusers==0.24.0 einops==0.3.0 omegaconf==2.1.1 transformers==4.36.2 \
  pytorch_lightning==1.4.2 imageio==2.9.0 imageio-ffmpeg==0.4.2 pandas==1.4.2 \
  Pillow==9.0.1 opencv_python_headless natsort==8.3.1 lmdb==1.3.0 kornia==0.6.9 \
  torchmetrics==0.6.0 torch-fidelity==0.3.0 invisible-watermark xformers==0.0.23
pip install "numpy<2"
pip install "setuptools<81"
pip install "huggingface_hub==0.23.5" timm decord ftfy

pip install -e "${ROOT}"
pip install -e "${ROOT}/t2v"

python -c "import torch; import xformers; import qdiff; import opensora; print('OK', torch.__version__, torch.cuda.is_available())"

echo "[setup_env] done. Activate with: conda activate ${ENV_NAME}"
