#!/usr/bin/env python3
"""Wan quantized text-to-video inference."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import yaml
from diffusers import WanTransformer3DModel
from diffusers.utils import export_to_video
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from qdiff.models.quant_model import QuantModel
from qdiff.utils import load_quant_params
from wan.models.pipeline import build_pipeline, load_prompts
from wan.models.wan_forward_adapter import WanForwardAdapter
from wan.utils.config import WAN_ROOT, load_yaml, parse_dtype
from wan.utils.verify import gate_l5


def setup_quant_transformer(pipe, quant_config_path: Path, ckpt_path: Path, part_fp: bool, mp_weight=None, mp_act=None):
    config = OmegaConf.load(str(quant_config_path))
    wq_params = config.quant.weight.quantizer
    aq_params = config.quant.activation.quantizer

    adapter = WanForwardAdapter(pipe.transformer, cfg_split=config.get("cfg_split", True))
    qnn = QuantModel(
        model=adapter,
        weight_quant_params=wq_params,
        act_quant_params=aq_params,
        model_type=config.model.model_type,
    )
    qnn.cfg_split = config.get("cfg_split", False)

    qnn.set_module_name_for_quantizer(module=qnn.model)
    if part_fp:
        with open(config.part_fp_list, "r") as f:
            fp_layer_list = [line.strip() for line in f if line.strip()]
        qnn.set_quant_state(True, True)
        qnn.set_layer_quant(
            model=qnn,
            module_name_list=fp_layer_list,
            quant_level="per_layer",
            weight_quant=False,
            act_quant=False,
            prefix="",
        )
    else:
        qnn.set_quant_state(True, True)

    if wq_params.n_bits <= 4 and mp_weight:
        with open(mp_weight, "r") as f:
            qnn.load_bitwidth_config(model=qnn, bit_config=yaml.safe_load(f), bit_type="weight")
    if aq_params.n_bits <= 6 and mp_act:
        with open(mp_act, "r") as f:
            qnn.load_bitwidth_config(model=qnn, bit_config=yaml.safe_load(f), bit_type="act")

    qnn.set_quant_init_done("weight")
    qnn.set_quant_init_done("activation")
    load_quant_params(qnn, str(ckpt_path), dtype=next(qnn.parameters()).dtype)
    qnn.eval()
    return qnn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--quant-config", required=True)
    parser.add_argument("--quant-ckpt", default=None)
    parser.add_argument("--infer-config", default=str(WAN_ROOT / "configs/infer_smoke.yaml"))
    parser.add_argument("--prompt-path", default=str(WAN_ROOT / "assets/prompts_calib_10.txt"))
    parser.add_argument("--num-videos", type=int, default=2)
    parser.add_argument("--part-fp", action="store_true")
    parser.add_argument("--time-mp-config-weight", default=None)
    parser.add_argument("--time-mp-config-act", default=None)
    parser.add_argument("--run-gate", action="store_true")
    args = parser.parse_args()

    infer = load_yaml(args.infer_config)
    outdir = Path(args.outdir)
    save_dir = outdir / "generated_videos"
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt = args.quant_ckpt or str(outdir / "ckpt.pth")
    pipe = build_pipeline(args.model_path, infer.get("dtype", "bfloat16"))
    setup_quant_transformer(
        pipe,
        Path(args.quant_config),
        Path(ckpt),
        args.part_fp,
        args.time_mp_config_weight,
        args.time_mp_config_act,
    )

    prompts = load_prompts(args.prompt_path, limit=args.num_videos)
    for idx, prompt in enumerate(prompts):
        gen = torch.Generator(device="cuda").manual_seed(int(infer.get("seed", 42)) + idx)
        result = pipe(
            prompt=prompt,
            height=int(infer.height),
            width=int(infer.width),
            num_frames=int(infer.num_frames),
            num_inference_steps=int(infer.num_inference_steps),
            guidance_scale=float(infer.guidance_scale),
            generator=gen,
        )
        out_path = save_dir / f"sample_{idx}.mp4"
        export_to_video(result.frames[0], str(out_path), fps=int(infer.get("fps", 16)))
        print(f"[{idx+1}/{len(prompts)}] -> {out_path}")

    if args.run_gate:
        gate_l5(save_dir, expected=len(prompts))


if __name__ == "__main__":
    main()
