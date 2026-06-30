"""Wan pipeline helpers."""

from __future__ import annotations

from pathlib import Path

import torch
from diffusers import UniPCMultistepScheduler, WanPipeline

from wan.models.wan_forward_adapter import WanForwardAdapter
from wan.utils.config import NATIVE_FLOW_SHIFT_480P, parse_dtype


def configure_flow_shift(pipe: WanPipeline, flow_shift: float) -> WanPipeline:
    """Apply Wan native sample_shift via UniPCMultistepScheduler.flow_shift."""
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config,
        flow_shift=float(flow_shift),
    )
    return pipe


def build_pipeline(
    model_path: Path | str,
    dtype_name: str = "bfloat16",
    device: str | None = None,
    cpu_offload: bool = True,
    flow_shift: float | None = NATIVE_FLOW_SHIFT_480P,
):
    dtype = parse_dtype(dtype_name)
    pipe = WanPipeline.from_pretrained(str(model_path), torch_dtype=dtype)
    if flow_shift is not None:
        configure_flow_shift(pipe, flow_shift)
    if device is not None:
        pipe.to(device)
    elif cpu_offload:
        pipe.enable_model_cpu_offload(gpu_id=0)
    return pipe


def build_transformer_adapter(pipe: WanPipeline, cfg_split: bool = True) -> WanForwardAdapter:
    adapter = WanForwardAdapter(pipe.transformer, cfg_split=cfg_split)
    adapter = adapter.to(dtype=pipe.transformer.dtype)
    return adapter


def load_prompts(path: Path | str, limit: int | None = None):
    lines = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if limit is not None:
        lines = lines[:limit]
    return lines
