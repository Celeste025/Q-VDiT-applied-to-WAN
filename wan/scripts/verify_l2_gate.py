#!/usr/bin/env python3
"""Run L2 verification gate for Wan QuantModel."""

import argparse
import sys
from pathlib import Path

import torch
from diffusers import WanTransformer3DModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf

from qdiff.models.quant_model import QuantModel
from wan.models.wan_forward_adapter import WanForwardAdapter
from wan.utils.config import DEFAULT_MODEL, parse_dtype
from wan.utils.verify import gate_l2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    dtype = parse_dtype("bfloat16")
    transformer = WanTransformer3DModel.from_pretrained(
        f"{args.model_path}/transformer", torch_dtype=dtype
    ).to(device)
    adapter = WanForwardAdapter(transformer, cfg_split=True).to(device)

    wq = OmegaConf.create({"n_bits": 8, "per_group": "channel", "scale_method": "grid_search_lp", "channel_wise": True, "round_mode": "nearest_ste", "sym": False})
    aq = OmegaConf.create({"n_bits": 8, "per_group": "token", "dynamic": True, "scale_method": "min_max", "round_mode": "nearest_ste", "running_stat": False, "sym": False})
    qnn = QuantModel(adapter, wq, aq, model_type="wan").to(device)

    fp_layers = [ln.strip() for ln in Path(ROOT / "wan/configs/quant/remain_fp.txt").read_text().splitlines() if ln.strip()]
    gate_l2(adapter, qnn, fp_layers)


if __name__ == "__main__":
    main()
