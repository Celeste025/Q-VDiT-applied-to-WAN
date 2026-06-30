#!/usr/bin/env python3
"""Generate VBench-named videos with quantized Wan model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from wan.scripts.quant_inference import setup_quant_transformer
from wan.models.pipeline import build_pipeline
from wan.utils.config import WAN_ROOT, infer_flow_shift, infer_guidance_scale, load_yaml

from diffusers.utils import export_to_video
import torch


def sanitize_filename(prompt: str) -> str:
    return f"{prompt}.mp4"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--quant-config", required=True)
    parser.add_argument("--quant-ckpt", required=True)
    parser.add_argument("--subset-json", default=str(WAN_ROOT / "assets/vbench_subset_25.json"))
    parser.add_argument("--infer-config", default=str(WAN_ROOT / "configs/infer.yaml"))
    parser.add_argument("--part-fp", action="store_true")
    args = parser.parse_args()

    infer = load_yaml(args.infer_config)
    prompts = [x["prompt_en"] for x in json.loads(Path(args.subset_json).read_text())]
    save_dir = Path(args.outdir)
    save_dir.mkdir(parents=True, exist_ok=True)

    pipe = build_pipeline(args.model_path, infer.get("dtype", "bfloat16"), flow_shift=infer_flow_shift(infer))
    setup_quant_transformer(pipe, Path(args.quant_config), Path(args.quant_ckpt), args.part_fp)

    for i, prompt in enumerate(prompts):
        out_mp4 = save_dir / sanitize_filename(prompt)
        if out_mp4.exists():
            print(f"[skip] {out_mp4}")
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
        export_to_video(result.frames[0], str(out_mp4), fps=int(infer.get("fps", 16)))
        print(f"[{i+1}/{len(prompts)}] -> {out_mp4}")


if __name__ == "__main__":
    main()
