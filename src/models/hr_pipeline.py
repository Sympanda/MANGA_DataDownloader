from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.config import HRProjectMode
from src.models.unet import ConvBlock, Down


def normalized_xy_grid(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return ``(2, H, W)`` coords in ``[-1, 1]`` (x, y)."""
    ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=0)


def append_coord_channels(feat: torch.Tensor) -> torch.Tensor:
    """Concatenate normalised xy channels onto ``(B, C, H, W)``."""
    coords = normalized_xy_grid(
        feat.shape[-2],
        feat.shape[-1],
        device=feat.device,
        dtype=feat.dtype,
    )
    coords = coords.unsqueeze(0).expand(feat.shape[0], -1, -1, -1)
    return torch.cat([feat, coords], dim=1)


class CrossAttnHRBlock(nn.Module):
    """
    Cross-attend UNet features (queries) to HR morphology tokens (keys/values).

    HR stays a spatial token set — never resized onto the UNet grid and never
    concatenated. Positional coords are appended so attention can distinguish
    centre vs outer-disc morphology.
    """

    def __init__(
        self,
        unet_channels: int,
        hr_channels: int,
        *,
        num_heads: int = 4,
        dropout: float = 0.0,
        attn_dim: int | None = None,
    ) -> None:
        super().__init__()
        d = int(attn_dim) if attn_dim is not None else int(unet_channels)
        if d % int(num_heads) != 0:
            d = int(num_heads) * max(1, d // int(num_heads))
        self.attn_dim = d
        self.num_heads = int(num_heads)
        self.head_dim = d // self.num_heads
        self.scale = self.head_dim**-0.5

        # Explicit W_Q / W_K / W_V (channel-last Linear on tokens with +2 xy coords).
        self.query_proj = nn.Linear(unet_channels + 2, d, bias=False)
        self.key_proj = nn.Linear(hr_channels + 2, d, bias=False)
        self.value_proj = nn.Linear(hr_channels + 2, d, bias=False)
        self.out_proj = nn.Linear(d, unet_channels, bias=False)
        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.GroupNorm(min(8, unet_channels), unet_channels)

    def _tokens_with_coords(self, feat: torch.Tensor) -> torch.Tensor:
        """``(B, C, H, W)`` → ``(B, H*W, C+2)`` with normalised xy."""
        feat_c = append_coord_channels(feat)
        return feat_c.flatten(2).transpose(1, 2)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, N, D)`` → ``(B, heads, N, head_dim)``."""
        b, n, _ = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        unet_feat: torch.Tensor,
        hr_feat: torch.Tensor,
        *,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            unet_feat: ``(B, C_u, H, W)`` query grid (any H×W; not required to match HR).
            hr_feat: ``(B, C_hr, H_hr, W_hr)`` key/value spatial map (any size).
            return_attn: if True, also return averaged attention ``(B, H*W, N_hr)``.
        """
        b, _, h, w = unet_feat.shape
        q = self._reshape_heads(self.query_proj(self._tokens_with_coords(unet_feat)))
        k = self._reshape_heads(self.key_proj(self._tokens_with_coords(hr_feat)))
        v = self._reshape_heads(self.value_proj(self._tokens_with_coords(hr_feat)))

        # (B, heads, Hw, N_hr)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        ctx = torch.matmul(attn, v)  # (B, heads, Hw, head_dim)
        ctx = ctx.transpose(1, 2).contiguous().view(b, h * w, self.attn_dim)
        delta = self.out_proj(ctx).transpose(1, 2).reshape(b, -1, h, w)
        out = self.norm(unet_feat + delta)
        if return_attn:
            # Mean over heads for diagnostics / spatial tests.
            return out, attn.mean(dim=1)
        return out


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
