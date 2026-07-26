from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.config import UpsampleMode


def _align_to_ref(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    dy = ref.shape[-2] - x.shape[-2]
    dx = ref.shape[-1] - x.shape[-1]
    if dy or dx:
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
    if x.shape[-2:] != ref.shape[-2:]:
        x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
    return x


class SpatialUpsample2x(nn.Module):
    """2× spatial upsample: bilinear, transpose conv, or sub-pixel (pixel shuffle)."""

    def __init__(self, channels: int, mode: UpsampleMode) -> None:
        super().__init__()
        self.mode = mode
        if mode == "transpose":
            self.op: nn.Module | None = nn.ConvTranspose2d(channels, channels, 2, stride=2)
        elif mode == "pixel_shuffle":
            self.op = nn.Sequential(
                nn.Conv2d(channels, channels * 4, 3, padding=1, bias=False),
                nn.PixelShuffle(2),
            )
        else:
            self.op = None

    def forward(self, x: torch.Tensor, *, ref: torch.Tensor | None = None) -> torch.Tensor:
        if self.mode == "bilinear":
            if ref is not None:
                x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
            else:
                x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            return x
        x = self.op(x)  # type: ignore[misc]
        if ref is not None:
            x = _align_to_ref(x, ref)
        return x


def nested_upsample_channel_counts(base_channels: int, depth: int) -> set[int]:
    """Channel widths of UNet++ nodes X^{i+1,*} that get upsampled into level i."""
    c = base_channels
    return {c * (2 ** (i + 1)) for i in range(depth)}


class ConvBlock(nn.Module):
    """Double conv with optional residual connection and GroupNorm."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        dropout: float = 0.0,
        norm: str = "bn",
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.residual = residual
        Norm: type[nn.Module]
        if norm == "gn":
            Norm = lambda c: nn.GroupNorm(min(8, c), c)  # noqa: E731
        else:
            Norm = nn.BatchNorm2d

        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            Norm(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            Norm(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        self.net = nn.Sequential(*layers)
        if residual:
            self.proj: nn.Module = (
                nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            )
        else:
            self.proj = None  # type: ignore[assignment]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if self.residual:
            y = y + self.proj(x)
        return y


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, **block_kw) -> None:
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), ConvBlock(in_ch, out_ch, **block_kw))

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
        **block_kw,
    ) -> None:
        super().__init__()
        self.up = SpatialUpsample2x(up_channels, upsample_mode)
        self.conv = ConvBlock(up_channels + skip_channels, out_ch, **block_kw)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x, ref=skip)
        return self.conv(torch.cat([skip, x], dim=1))


class UNetBackbone(nn.Module):
    """Configurable UNet for 76×76 inputs."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        base_channels: int = 64,
        bottleneck_multiplier: int = 16,
        n_down: int = 4,
        dropout: float = 0.0,
        upsample_mode: UpsampleMode = "bilinear",
        norm: str = "bn",
        residual: bool = True,
    ) -> None:
        super().__init__()
        c = base_channels
        m = bottleneck_multiplier
        block_kw = {"dropout": dropout, "norm": norm, "residual": residual}

        self._base_channels = c
        self._bottleneck_multiplier = m
        self._n_down = n_down

        self.inc = ConvBlock(in_channels, c, **block_kw)
        downs: list[nn.Module] = []
        ch_in = c
        for i in range(n_down):
            ch_out = min(c * (2 ** (i + 1)), c * m)
            downs.append(Down(ch_in, ch_out, **block_kw))
            ch_in = ch_out
        self.downs = nn.ModuleList(downs)
        self.bottleneck_channels = ch_in

        ups: list[nn.Module] = []
        skip_channels = [c]
        for i in range(n_down):
            skip_channels.append(min(c * (2 ** i), c * (m // 2)))
        skip_channels = list(reversed(skip_channels))

        ch = ch_in
        for i in range(n_down):
            skip_ch = skip_channels[i]
            ch_out = skip_ch
            ups.append(Up(ch, skip_ch, ch_out, upsample_mode=upsample_mode, **block_kw))
            ch = ch_out
        self.ups = nn.ModuleList(ups)
        self.outc = nn.Conv2d(c, out_channels, kernel_size=1)

    @property
    def encoder_level_channels(self) -> list[int]:
        """Channel width after each encoder block (stem + each down), for multi-scale FiLM."""
        c = self._base_channels
        m = self._bottleneck_multiplier
        levels = [c]
        for i in range(self._n_down):
            levels.append(min(c * (2 ** (i + 1)), c * m))
        return levels

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skips = [self.inc(x)]
        h = skips[0]
        for down in self.downs:
            h = down(h)
            skips.append(h)
        bottleneck = skips.pop()
        return bottleneck, skips

    def decode(self, x: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        h = x
        for up, skip in zip(self.ups, reversed(skips)):
            h = up(h, skip)
        return self.outc(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, skips = self.encode(x)
        return self.decode(bottleneck, skips)

    def forward_with_stem_hook(self, x: torch.Tensor, stem_fn) -> torch.Tensor:
        h = stem_fn(self.inc(x))
        skips = [h]
        for down in self.downs:
            h = down(h)
            skips.append(h)
        bottleneck = skips.pop()
        return self.decode(bottleneck, skips)

    def forward_with_bottleneck_hook(self, x: torch.Tensor, bottleneck_fn) -> torch.Tensor:
        bottleneck, skips = self.encode(x)
        bottleneck = bottleneck_fn(bottleneck)
        return self.decode(bottleneck, skips)

    def forward_with_encoder_hooks(
        self,
        x: torch.Tensor,
        encoder_hooks: list,
        bottleneck_fn=None,
        *,
        level_fusions: list | None = None,
    ) -> torch.Tensor:
        """Apply hook after each encoder block (stem + downs); optional HR fusions."""
        fusions = level_fusions or []
        h = self.inc(x)
        if len(encoder_hooks) > 0 and encoder_hooks[0] is not None:
            h = encoder_hooks[0](h)
        if len(fusions) > 0 and fusions[0] is not None:
            h = fusions[0](h)
        skips = [h]
        for i, down in enumerate(self.downs):
            h = down(h)
            hook_idx = i + 1
            if hook_idx < len(encoder_hooks) and encoder_hooks[hook_idx] is not None:
                h = encoder_hooks[hook_idx](h)
            if hook_idx < len(fusions) and fusions[hook_idx] is not None:
                h = fusions[hook_idx](h)
            skips.append(h)
        bottleneck = skips.pop()
        if bottleneck_fn is not None:
            bottleneck = bottleneck_fn(bottleneck)
        return self.decode(bottleneck, skips)
