#!/usr/bin/env python3
"""Precompute T5 text embeddings for VBench subset prompts."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "t2v"))

from opensora.registry import MODELS, build_module
from opensora.utils.misc import to_torch_dtype


T5_FILES = [
    "config.json",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer_config.json",
    "pytorch_model.bin.index.json",
    "pytorch_model-00001-of-00002.bin",
    "pytorch_model-00002-of-00002.bin",
]


def ensure_t5(local_dir: Path) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    if all((local_dir / name).is_file() for name in T5_FILES[-2:]):
        print(f"[t5] reuse {local_dir}")
        return local_dir

    for incomplete in local_dir.glob("*.incomplete"):
        print(f"[t5] remove incomplete {incomplete}")
        incomplete.unlink(missing_ok=True)

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    print(f"[t5] downloading DeepFloyd/t5-v1_1-xxl -> {local_dir}")
    snapshot_download(
        repo_id="DeepFloyd/t5-v1_1-xxl",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return local_dir


def load_prompts(prompt_path: Path, subset_json: Optional[Path]) -> list[str]:
    if subset_json and subset_json.is_file():
        return [x["prompt_en"] for x in json.loads(subset_json.read_text())]
    return [line.strip() for line in prompt_path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-path", type=Path, default=ROOT / "logs/eval_vbench_subset25/prompts.txt")
    parser.add_argument("--subset-json", type=Path, default=ROOT / "logs/eval_vbench_subset25/vbench_subset_25.json")
    parser.add_argument("--ckpt-path", type=Path, default=ROOT / "logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth")
    parser.add_argument("--logs-dir", type=Path, default=ROOT / "logs")
    parser.add_argument("--out", type=Path, default=ROOT / "logs/eval_vbench_subset25/text_embeds.pth")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    prompts = load_prompts(args.prompt_path, args.subset_json)
    if not prompts:
        raise RuntimeError("No prompts found")

    t5_dir = ensure_t5(args.logs_dir / "t5-v1_1-xxl")
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype("fp16")

    input_size = (16, 512, 512)
    vae = build_module(
        dict(
            type="VideoAutoencoderKL",
            from_pretrained=str(args.logs_dir / "vae_ckpt"),
            micro_batch_size=128,
        ),
        MODELS,
    )
    latent_size = vae.get_latent_size(input_size)
    model = build_module(
        dict(
            type="STDiT-XL/2",
            space_scale=1.0,
            time_scale=1.0,
            enable_flashattn=False,
            enable_layernorm_kernel=False,
            from_pretrained=str(args.ckpt_path),
        ),
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        caption_channels=4096,
        model_max_length=120,
        dtype=dtype,
    )
    text_encoder = build_module(
        dict(
            type="t5",
            from_pretrained=str(args.logs_dir),
            local_cache=True,
            save_pretrained=str(t5_dir),
            model_max_length=120,
        ),
        MODELS,
        device=device,
        dtype=torch.float32,
    )
    text_encoder.y_embedder = model.y_embedder
    text_encoder = text_encoder  # keep reference alive

    ys = []
    masks = []
    with torch.no_grad():
        for prompt in prompts:
            enc = text_encoder.encode([prompt])
            null_y = text_encoder.null(1)
            # enc["y"] is on GPU; null() returns CPU tensors — stack on CPU before save.
            y_pair = torch.stack([enc["y"][0].cpu(), null_y[0].cpu()], dim=0)
            ys.append(y_pair)
            masks.append(enc["mask"][0].cpu())

    payload = {"y": torch.stack(ys, dim=0), "mask": torch.stack(masks, dim=0)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    print(f"[done] saved {len(prompts)} prompts -> {args.out}")
    print(f"  y: {payload['y'].shape}, mask: {payload['mask'].shape}")


if __name__ == "__main__":
    main()
