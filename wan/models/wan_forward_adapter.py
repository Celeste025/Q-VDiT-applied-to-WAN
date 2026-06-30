"""OpenSora-compatible forward wrapper for WanTransformer3DModel."""

from __future__ import annotations

import torch
import torch.nn as nn


class WanForwardAdapter(nn.Module):
    """Expose Wan transformer with QuantModel signature: forward(x, t, y, mask=None)."""

    def __init__(self, transformer: nn.Module, cfg_split: bool = True):
        super().__init__()
        self.transformer = transformer
        self.cfg_split = cfg_split
        self.in_channels = transformer.config.in_channels

    def forward(self, x, t, y, mask=None, **kwargs):
        del mask, kwargs
        if isinstance(t, torch.Tensor) and t.ndim == 0:
            t = t.unsqueeze(0)
        if isinstance(t, torch.Tensor) and t.numel() == 1 and x.shape[0] > 1:
            t = t.expand(x.shape[0])

        if self.cfg_split and x.shape[0] >= 2 and x.shape[0] % 2 == 0:
            half = x.shape[0] // 2
            half_x = x[:half]
            y = y.reshape(2, half, *y.shape[1:])
            t = t.reshape(2, half)
            y_cond, y_uncond = y[0], y[1]
            t_cond, t_uncond = t[0], t[1]
            out_cond = self.transformer(
                half_x, t_cond, y_cond, return_dict=False
            )[0]
            out_uncond = self.transformer(
                half_x, t_uncond, y_uncond, return_dict=False
            )[0]
            return torch.cat([out_cond, out_uncond], dim=0)

        return self.transformer(x, t, y, return_dict=False)[0]
