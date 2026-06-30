#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p logs/split_ckpt logs/vae_ckpt t2v/utils_files

echo "[download] HF mirror: ${HF_ENDPOINT}"

if [[ ! -f logs/split_ckpt/OpenSora-v1-HQ-16x512x512.pth ]]; then
  huggingface-cli download hpcai-tech/Open-Sora OpenSora-v1-HQ-16x512x512.pth \
    --local-dir logs/split_ckpt
else
  echo "[download] skip OpenSora ckpt (exists)"
fi

if [[ ! -f logs/vae_ckpt/config.json ]]; then
  huggingface-cli download stabilityai/sd-vae-ft-ema \
    --local-dir logs/vae_ckpt
else
  echo "[download] skip VAE (exists)"
fi

if [[ ! -f t2v/utils_files/text_embeds.pth ]] || [[ ! -s t2v/utils_files/text_embeds.pth ]]; then
  wget -O t2v/utils_files/text_embeds.pth \
    "https://raw.githubusercontent.com/thu-nics/ViDiT-Q/viditq_old/t2v/utils_files/text_embeds.pth" \
    || wget -O t2v/utils_files/text_embeds.pth \
    "https://ghproxy.net/https://github.com/thu-nics/ViDiT-Q/raw/viditq_old/t2v/utils_files/text_embeds.pth"
else
  echo "[download] skip text_embeds.pth (exists)"
fi

echo "[download] assets ready:"
ls -lh logs/split_ckpt/OpenSora-v1-HQ-16x512x512.pth
ls -lh logs/vae_ckpt/config.json
ls -lh t2v/utils_files/text_embeds.pth
