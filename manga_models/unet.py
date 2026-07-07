from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from manga_models.config import UpsampleMode


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    def __init__(
        self,
        up_channels: int,
        skip_channels: int,
        out_ch: int,
        *,
        upsample_mode: UpsampleMode = "bilinear",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.upsample_mode = upsample_mode
        in_ch = up_channels + skip_channels
        if upsample_mode == "transpose":
            # ConvTranspose2d(k=2,s=2) can produce checkerboard/grid artifacts; prefer bilinear.
            self.up = nn.ConvTranspose2d(up_channels, up_channels, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dy = skip.shape[-2] - x.shape[-2]
        dx = skip.shape[-1] - x.shape[-1]
        if dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetBackbone(nn.Module):
    """UNet for 76×76 inputs with optional transpose upsampling and deeper bottleneck."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        base_channels: int = 64,
        bottleneck_multiplier: int = 16,
        dropout: float = 0.0,
        upsample_mode: UpsampleMode = "transpose",
    ) -> None:
        super().__init__()
        c = base_channels
        m = bottleneck_multiplier
        up_kwargs = {"upsample_mode": upsample_mode, "dropout": dropout}

        self.inc = DoubleConv(in_channels, c, dropout=dropout)
        self.down1 = Down(c, c * 2, dropout=dropout)
        self.down2 = Down(c * 2, c * 4, dropout=dropout)
        self.down3 = Down(c * 4, c * 8, dropout=dropout)
        self.down4 = Down(c * 8, c * m, dropout=dropout)
        self.bottleneck_channels = c * m
        self.up1 = Up(c * m, c * 8, c * 4, **up_kwargs)
        self.up2 = Up(c * 4, c * 4, c * 2, **up_kwargs)
        self.up3 = Up(c * 2, c * 2, c, **up_kwargs)
        self.up4 = Up(c, c, c, **up_kwargs)
        self.outc = nn.Conv2d(c, out_channels, kernel_size=1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return x4, [x0, x1, x2, x3]

    def decode(self, x: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        x = self.up1(x, skips[3])
        x = self.up2(x, skips[2])
        x = self.up3(x, skips[1])
        x = self.up4(x, skips[0])
        return self.outc(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, skips = self.encode(x)
        return self.decode(bottleneck, skips)

    def forward_with_bottleneck_hook(
        self,
        x: torch.Tensor,
        bottleneck_fn,
    ) -> torch.Tensor:
        bottleneck, skips = self.encode(x)
        bottleneck = bottleneck_fn(bottleneck)
        return self.decode(bottleneck, skips)
