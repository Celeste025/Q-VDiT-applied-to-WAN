"""Collect calibration trajectories from WanPipeline denoising."""

from __future__ import annotations

from typing import Any

import torch
from diffusers import WanPipeline
from tqdm import tqdm


def _align_num_frames(num_frames: int, temporal_factor: int) -> int:
    if num_frames % temporal_factor != 1:
        num_frames = num_frames // temporal_factor * temporal_factor + 1
    return max(num_frames, 1)


def collect_calib_trajectory(
    pipe: WanPipeline,
    prompts: list[str],
    *,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    device: str,
) -> dict[str, torch.Tensor]:
    """Return OpenSora-compatible calib tensors: [n_steps, batch, ...]."""
    num_frames = _align_num_frames(num_frames, pipe.vae_scale_factor_temporal)
    merged: dict[str, torch.Tensor | None] = {k: None for k in ("xs", "ts", "cond_emb", "mask")}

    for prompt_idx, prompt in enumerate(tqdm(prompts, desc="calib prompts")):
        generator = torch.Generator(device=device).manual_seed(seed + prompt_idx)
        cond, uncond = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt="",
            do_classifier_free_guidance=True,
            device="cpu",
        )
        cond = cond.to(device=device, dtype=pipe.transformer.dtype)
        uncond = uncond.to(device=device, dtype=pipe.transformer.dtype)

        latents = pipe.prepare_latents(
            1,
            pipe.transformer.config.in_channels,
            num_frames,
            height,
            width,
            pipe.transformer.dtype,
            device,
            generator,
            None,
        )
        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = pipe.scheduler.timesteps

        step_xs, step_ts, step_emb, step_mask = [], [], [], []
        for t in timesteps:
            x2 = torch.cat([latents, latents], dim=0)
            y2 = torch.cat([cond, uncond], dim=0)
            t2 = t.expand(2).to(device=device)
            mask2 = torch.ones(2, cond.shape[1], device=device, dtype=torch.long)

            step_xs.append(x2.detach().cpu())
            step_ts.append(t2.detach().cpu())
            step_emb.append(y2.detach().cpu())
            step_mask.append(mask2.detach().cpu())

            latent_in = latents.to(pipe.transformer.dtype)
            x2_gpu = torch.cat([latent_in, latent_in], dim=0)
            y2_gpu = torch.cat([cond, uncond], dim=0)
            t2_gpu = t.expand(2).to(device)
            with torch.inference_mode():
                noise_pred = pipe.transformer(
                    x2_gpu, t2_gpu, y2_gpu, return_dict=False
                )[0]
            noise_cond, noise_uncond = noise_pred[0:1], noise_pred[1:2]
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        cur = {
            "xs": torch.stack(step_xs, dim=0),
            "ts": torch.stack(step_ts, dim=0),
            "cond_emb": torch.stack(step_emb, dim=0),
            "mask": torch.stack(step_mask, dim=0),
        }
        for key in cur:
            if merged[key] is None:
                merged[key] = cur[key]
            else:
                merged[key] = torch.cat([merged[key], cur[key]], dim=1)

    return merged  # type: ignore[return-value]
