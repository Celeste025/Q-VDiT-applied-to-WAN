#!/usr/bin/env python3
"""Wan2.1 FP/BF16 baseline video generation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from diffusers.utils import export_to_video

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from wan.models.pipeline import build_pipeline, load_prompts
from wan.utils.config import WAN_ROOT, load_yaml
from wan.utils.verify import gate_l1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--infer-config", default=str(WAN_ROOT / "configs/infer_smoke.yaml"))
    parser.add_argument("--prompt-path", default=str(WAN_ROOT / "assets/prompts_calib_10.txt"))
    parser.add_argument("--num-prompts", type=int, default=2)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--run-gate", action="store_true")
    args = parser.parse_args()

    infer = load_yaml(args.infer_config)
    outdir = Path(args.outdir) / "generated_videos"
    outdir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.prompt_path, limit=args.num_prompts)
    pipe = build_pipeline(args.model_path, infer.get("dtype", "bfloat16"))

    timings = []
    for idx, prompt in enumerate(prompts):
        gen = torch.Generator(device="cuda").manual_seed(int(infer.get("seed", 42)) + idx)
        t0 = time.time()
        result = pipe(
            prompt=prompt,
            height=int(infer.height),
            width=int(infer.width),
            num_frames=int(infer.num_frames),
            num_inference_steps=int(infer.num_inference_steps),
            guidance_scale=float(infer.guidance_scale),
            generator=gen,
        )
        elapsed = time.time() - t0
        timings.append(elapsed)
        out_path = outdir / f"sample_{idx}.mp4"
        export_to_video(result.frames[0], str(out_path), fps=int(infer.get("fps", 16)))
        print(f"[{idx+1}/{len(prompts)}] {elapsed:.1f}s -> {out_path}")

    summary = {"count": len(prompts), "avg_seconds": sum(timings) / max(len(timings), 1)}
    (Path(args.outdir) / "manifest.json").write_text(json.dumps(summary, indent=2))

    if args.run_gate:
        gate_l1(outdir, min_videos=len(prompts), expect_frames=int(infer.num_frames))


if __name__ == "__main__":
    main()
