#!/usr/bin/env python3
"""Generate VBench-named videos ({prompt_en}.mp4) for FP16 / quant models."""

import json
import os
import sys
from pathlib import Path

import torch
from mmengine.runner import set_random_seed
from opensora.datasets import save_sample
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.misc import to_torch_dtype


def sanitize_filename(prompt: str) -> str:
    # VBench expects exact prompt_en + ".mp4"
    return f"{prompt}.mp4"


def main():
    cfg = parse_configs(training=False)
    subset_json = os.environ.get("VBENCH_SUBSET_JSON")
    if subset_json:
        prompts = [x["prompt_en"] for x in json.loads(Path(subset_json).read_text())]
    else:
        with open(cfg.prompt_path, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype(cfg.dtype)
    set_random_seed(seed=cfg.seed)

    scheduler = build_module(cfg.scheduler, SCHEDULERS)
    input_size = (cfg.num_frames, *cfg.image_size)
    vae = build_module(cfg.vae, MODELS)
    latent_size = vae.get_latent_size(input_size)
    model = build_module(
        cfg.model,
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        caption_channels=4096,
        model_max_length=cfg.text_encoder.model_max_length,
        dtype=dtype,
    )

    precompute = cfg.get("precompute_text_embeds", None)
    if precompute is not None:
        text_encoder = None
    else:
        text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
        text_encoder.y_embedder = model.y_embedder

    vae = vae.to(device, dtype).eval()
    model = model.to(device, dtype).eval()

    model_args = {}
    if precompute is not None:
        model_args["precompute_text_embeds"] = torch.load(precompute)

    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)
    model.timestep_wise_quant = False

    for i, prompt in enumerate(prompts):
        out_mp4 = os.path.join(save_dir, sanitize_filename(prompt))
        if os.path.isfile(out_mp4):
            print(f"[skip] exists: {out_mp4}")
            continue

        batch_prompts = [prompt]
        if precompute is not None:
            model_args["batch_ids"] = torch.arange(i, i + 1)

        samples = scheduler.sample(
            model,
            text_encoder,
            sampler_type=cfg.sampler,
            z_size=(vae.out_channels, *latent_size),
            prompts=batch_prompts,
            device=device,
            additional_args=model_args,
        )
        samples = vae.decode(samples.to(dtype))
        print(f"Prompt: {prompt}")
        save_path = out_mp4[:-4]  # save_sample adds .mp4
        save_sample(samples[0], fps=cfg.fps, save_path=save_path)


if __name__ == "__main__":
    main()
