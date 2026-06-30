#!/usr/bin/env python3
"""Generate FP/BF16 VBench-named videos for Wan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from diffusers.utils import export_to_video

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from wan.models.pipeline import build_pipeline
from wan.utils.config import WAN_ROOT, infer_flow_shift, infer_guidance_scale, load_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--subset-json", default=str(WAN_ROOT / "assets/vbench_subset_25.json"))
    parser.add_argument("--infer-config", default=str(WAN_ROOT / "configs/infer.yaml"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    infer = load_yaml(args.infer_config)
    items = json.loads(Path(args.subset_json).read_text())
    if args.limit > 0:
        items = items[: args.limit]

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    pipe = build_pipeline(args.model_path, infer.get("dtype", "bfloat16"), flow_shift=infer_flow_shift(infer))

    for i, item in enumerate(items):
        prompt = item["prompt_en"]
        out = save_dir / f"{prompt}.mp4"
        if out.exists():
            print(f"[skip] {out}")
            continue
        gen = torch.Generator(device="cuda").manual_seed(int(infer.get("seed", 42)) + i)
        result = pipe(
            prompt=prompt,
            height=int(infer.height),
            width=int(infer.width),
            num_frames=int(infer.num_frames),
            num_inference_steps=int(infer.num_inference_steps),
            guidance_scale=infer_guidance_scale(infer),
            generator=gen,
        )
        export_to_video(result.frames[0], str(out), fps=int(infer.get("fps", 16)))
        print(f"[{i+1}/{len(items)}] -> {out}")


if __name__ == "__main__":
    main()
