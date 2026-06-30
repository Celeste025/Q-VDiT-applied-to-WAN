"""Wan config and path helpers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from omegaconf import OmegaConf

WAN_ROOT = Path(__file__).resolve().parents[1]
QVDIT_ROOT = WAN_ROOT.parent

# Wan2.1-T2V-1.3B native generate.py defaults at 480P (sample_guide_scale / sample_shift).
NATIVE_GUIDANCE_SCALE_1_3B = 6.0
NATIVE_FLOW_SHIFT_480P = 8.0

DEFAULT_MODEL = Path(
    os.environ.get(
        "WAN_MODEL_PATH",
        str(QVDIT_ROOT.parent / "DVDQuant_rep/pretrained_models/Wan2.1-T2V-1.3B-Diffusers"),
    )
)


def load_yaml(path: Path | str):
    return OmegaConf.load(str(path))


def infer_guidance_scale(infer) -> float:
    return float(infer.get("guidance_scale", NATIVE_GUIDANCE_SCALE_1_3B))


def infer_flow_shift(infer) -> float:
    return float(infer.get("flow_shift", NATIVE_FLOW_SHIFT_480P))


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--infer-config", type=str, default=str(WAN_ROOT / "configs/infer.yaml"))
    parser.add_argument("--prompt-path", type=str, default=str(WAN_ROOT / "assets/prompts_calib_10.txt"))
    parser.add_argument("--seed", type=int, default=42)
    return parser


def parse_dtype(name: str):
    import torch

    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
    }
    return mapping[name]
