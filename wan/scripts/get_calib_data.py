#!/usr/bin/env python3
"""Collect Wan calibration trajectories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from wan.models.pipeline import build_pipeline, load_prompts
from wan.utils.calib_hooks import collect_calib_trajectory
from wan.utils.config import WAN_ROOT, load_yaml
from wan.utils.verify import gate_l3


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
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipe = build_pipeline(args.model_path, infer.get("dtype", "bfloat16"), cpu_offload=False)
    pipe.transformer.to(device)
    prompts = load_prompts(args.prompt_path, limit=args.num_prompts)

    data = collect_calib_trajectory(
        pipe,
        prompts,
        height=int(infer.height),
        width=int(infer.width),
        num_frames=int(infer.num_frames),
        num_inference_steps=int(infer.num_inference_steps),
        guidance_scale=float(infer.guidance_scale),
        seed=int(infer.get("seed", 42)),
        device=device,
    )
    out_path = outdir / "calib_data.pt"
    torch.save(data, out_path)
    print(f"Saved calib data -> {out_path}")

    if args.run_gate:
        gate_l3(out_path, n_samples=args.num_prompts, n_steps=int(infer.num_inference_steps))


if __name__ == "__main__":
    main()
