from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.config import HRProjectMode
from src.models.unet import ConvBlock, Down


class HREncoder(nn.Module):
    """Stride-down encoder for native-resolution survey cutouts."""

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
        self.n_down = n_down
        self._base_channels = c
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

    @property
    def level_channels(self) -> list[int]:
        """Channel width at stem + each down level (n_down + 1 tensors)."""
        c = self._base_channels
        return [c * (2**i) for i in range(self.n_down + 1)]

    def forward_pyramid(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return multi-scale features [stem, level1, …, level_n] (shallow → deep)."""
        feats: list[torch.Tensor] = []
        h = self.stem(x)
        feats.append(h)
        for down in self.downs:
            h = down(h)
            feats.append(h)
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Deepest feature only (legacy hr_encoder bottleneck path)."""
        return self.forward_pyramid(x)[-1]


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


class HRLevelFusion(nn.Module):
    """Project an HR pyramid level onto a UNet feature and fuse by concat+conv."""

    def __init__(
        self,
        hr_channels: int,
        unet_channels: int,
        *,
        dropout: float = 0.0,
        norm: str = "gn",
    ) -> None:
        super().__init__()
        self.channel_proj = nn.Conv2d(hr_channels, unet_channels, kernel_size=1, bias=False)
        self.fuse = ConvBlock(
            unet_channels * 2,
            unet_channels,
            dropout=dropout,
            norm=norm,
            residual=True,
        )

    def forward(self, unet_feat: torch.Tensor, hr_feat: torch.Tensor) -> torch.Tensor:
        hr = F.interpolate(
            hr_feat,
            size=unet_feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        hr = self.channel_proj(hr)
        return self.fuse(torch.cat([unet_feat, hr], dim=1))


class FootprintFusion(nn.Module):
    """Concatenate a 76×76 footprint mask channel and refine features on the target grid."""

    def __init__(self, in_channels: int, *, dropout: float = 0.0, norm: str = "gn") -> None:
        super().__init__()
        self.fuse = ConvBlock(in_channels + 1, in_channels, dropout=dropout, norm=norm, residual=True)

    def forward(self, features: torch.Tensor, footprint: torch.Tensor) -> torch.Tensor:
        if footprint.ndim == features.ndim - 1:
            footprint = footprint.unsqueeze(1)
        return self.fuse(torch.cat([features, footprint.float()], dim=1))
