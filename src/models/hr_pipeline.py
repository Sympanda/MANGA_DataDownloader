from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.config import HRProjectMode
from src.models.unet import ConvBlock, Down


class HREncoder(nn.Module):
    """Stride-down encoder for native-resolution survey cutouts (encoder spine only)."""

    def __init__(
        self,
        in_channels: int,
        *,
        base_channels: int = 64,
        n_down: int = 4,
        dropout: float = 0.0,
        norm: str = "gn",
    ) -> None:
        super().__init__()
        block_kw = {"dropout": dropout, "norm": norm, "residual": True}
        c = base_channels
        self.stem = ConvBlock(in_channels, c, **block_kw)
        self.downs = nn.ModuleList(
            [
                Down(c * (2**i), c * (2 ** (i + 1)), **block_kw)
                for i in range(n_down)
            ]
        )
        self._out_channels = c * (2**n_down)

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for down in self.downs:
            h = down(h)
        return h


class GridProjector(nn.Module):
    """Resize HR encoder features onto the Amara / target map grid."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        mode: HRProjectMode = "bilinear",
        dropout: float = 0.0,
        norm: str = "gn",
    ) -> None:
        super().__init__()
        block_kw = {"dropout": dropout, "norm": norm, "residual": True}
        if mode == "learned":
            self.refine = nn.Sequential(
                ConvBlock(in_channels, out_channels, **block_kw),
                ConvBlock(out_channels, out_channels, **block_kw),
            )
        else:
            self.refine = ConvBlock(in_channels, out_channels, **block_kw)

    def forward(self, x: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return self.refine(x)


class FootprintFusion(nn.Module):
    """Concatenate a 76×76 footprint mask channel and refine features on the target grid."""

    def __init__(self, in_channels: int, *, dropout: float = 0.0, norm: str = "gn") -> None:
        super().__init__()
        self.fuse = ConvBlock(in_channels + 1, in_channels, dropout=dropout, norm=norm, residual=True)

    def forward(self, features: torch.Tensor, footprint: torch.Tensor) -> torch.Tensor:
        if footprint.ndim == features.ndim - 1:
            footprint = footprint.unsqueeze(1)
        return self.fuse(torch.cat([features, footprint.float()], dim=1))
