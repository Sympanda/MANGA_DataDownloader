"""Residual models that correct a frozen base map predictor."""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from src.models.pixel_sed import _assert_pointwise_convs

ResidualVariant = Literal["pixel", "local_cnn", "gaussian"]


class PixelResidualRegressor(nn.Module):
    """1×1 residual MLP on [ugriz, base_Hα] → residual correction."""

    def __init__(
        self,
        *,
        in_channels: int = 6,
        out_channels: int = 1,
        hidden_channels: int = 32,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        act = nn.GELU() if activation == "gelu" else nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            act,
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            type(act)(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )
        _assert_pointwise_convs(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LocalResidualCNN(nn.Module):
    """Shallow 3×3 CNN residual model (local neighbourhood, no encoder/decoder)."""

    def __init__(
        self,
        *,
        in_channels: int = 6,
        out_channels: int = 1,
        hidden_channels: int = 48,
        n_layers: int = 3,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        act_cls = nn.GELU if activation == "gelu" else nn.ReLU
        layers: list[nn.Module] = []
        ch = in_channels
        for i in range(n_layers - 1):
            layers.append(nn.Conv2d(ch, hidden_channels, kernel_size=3, padding=1))
            layers.append(act_cls())
            ch = hidden_channels
        layers.append(nn.Conv2d(ch, out_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPixelResidualRegressor(nn.Module):
    """1×1 residual MLP predicting residual mean and log-variance."""

    def __init__(
        self,
        *,
        in_channels: int = 6,
        out_channels: int = 1,
        hidden_channels: int = 32,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        act = nn.GELU() if activation == "gelu" else nn.ReLU(inplace=True)
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            act,
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            type(act)(),
        )
        self.mean_head = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.log_var_head = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        _assert_pointwise_convs(self)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.mean_head(h), self.log_var_head(h)


def build_residual_net(
    variant: ResidualVariant,
    *,
    in_channels: int,
    out_channels: int = 1,
    hidden_channels: int = 32,
) -> nn.Module:
    if variant == "pixel":
        return PixelResidualRegressor(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
        )
    if variant == "local_cnn":
        return LocalResidualCNN(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=max(hidden_channels, 48),
        )
    if variant == "gaussian":
        return GaussianPixelResidualRegressor(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
        )
    raise ValueError(f"Unknown residual variant: {variant!r}")
