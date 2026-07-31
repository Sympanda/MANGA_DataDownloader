"""Pixel-level photometric baselines (1×1 only; no spatial context)."""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

PixelSEDVariant = Literal["linear", "mlp"]


def _assert_pointwise_convs(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Conv2d) and (m.kernel_size != (1, 1) or m.stride != (1, 1)):
            raise ValueError(
                f"Pixel SED models must use only 1×1 convolutions; found {m}"
            )


class PixelSEDRegressor(nn.Module):
    """
    Per-pixel ugriz → map regressor.

    ``linear``: 1×1 Conv 5→C_out (shared linear regression at every pixel).
    ``mlp``:    1×1 Conv MLP 5→H→H→C_out (nonlinear pixel-SED baseline).
    """

    def __init__(
        self,
        *,
        in_channels: int = 5,
        out_channels: int = 1,
        variant: PixelSEDVariant = "mlp",
        hidden_channels: int = 32,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.variant = variant
        act_cls: type[nn.Module]
        if activation == "gelu":
            act_cls = nn.GELU
        elif activation == "relu":
            act_cls = nn.ReLU
        else:
            raise ValueError(f"Unknown activation: {activation!r}")

        if variant == "linear":
            self.net = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        elif variant == "mlp":
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
                act_cls(),
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
                act_cls(),
                nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
            )
        else:
            raise ValueError(f"Unknown PixelSED variant: {variant!r}")
        _assert_pointwise_convs(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
