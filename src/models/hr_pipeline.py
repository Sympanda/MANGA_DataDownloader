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

    Default ``mode='local'``: each UNet location attends only to a ``window×window``
    HR neighbourhood gathered via ``unfold`` (no ``N_q × N_hr`` dense matrix).

    HR is projected to ``attn_dim`` *before* unfold/gather so window tensors are
    ``B×N×K×D`` rather than ``B×N×K×C_hr`` (C_hr can be 512+ and OOMs easily).
    """

    def __init__(
        self,
        unet_channels: int,
        hr_channels: int,
        *,
        num_heads: int = 4,
        dropout: float = 0.0,
        attn_dim: int | None = None,
        mode: str = "local",
        window: int = 7,
    ) -> None:
        super().__init__()
        d = int(attn_dim) if attn_dim is not None else int(unet_channels)
        if d % int(num_heads) != 0:
            d = int(num_heads) * max(1, d // int(num_heads))
        self.attn_dim = d
        self.num_heads = int(num_heads)
        self.head_dim = d // self.num_heads
        self.dropout_p = float(dropout)
        self.mode = str(mode)
        self.window = int(window)
        if self.mode == "local" and (self.window < 1 or self.window % 2 == 0):
            raise ValueError(f"local attention window must be odd and >= 1, got {self.window}")

        # Spatial 1×1 projections — applied before local gather to keep windows in D-dim.
        self.query_proj = nn.Conv2d(unet_channels + 2, d, kernel_size=1, bias=False)
        self.key_proj = nn.Conv2d(hr_channels + 2, d, kernel_size=1, bias=False)
        self.value_proj = nn.Conv2d(hr_channels + 2, d, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(d, unet_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(min(8, unet_channels), unet_channels)

    @property
    def local_token_count(self) -> int:
        return self.window * self.window

    def _map_query_to_hr_indices(
        self,
        h_u: int,
        w_u: int,
        h_hr: int,
        w_hr: int,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map each UNet pixel to the nearest HR feature index (shared FoV)."""
        if h_u == 1:
            ys = torch.zeros(1, device=device)
        else:
            ys = torch.linspace(0, h_hr - 1, h_u, device=device)
        if w_u == 1:
            xs = torch.zeros(1, device=device)
        else:
            xs = torch.linspace(0, w_hr - 1, w_u, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return yy.round().long().clamp(0, h_hr - 1), xx.round().long().clamp(0, w_hr - 1)

    def _query_hr_centers(
        self,
        h_u: int,
        w_u: int,
        h_hr: int,
        w_hr: int,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Flattened HR centers (N,) for each UNet query."""
        ih, iw = self._map_query_to_hr_indices(h_u, w_u, h_hr, w_hr, device=device)
        return ih.reshape(-1), iw.reshape(-1)

    def _window_offsets(self, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        win = self.window
        oy, ox = torch.meshgrid(
            torch.arange(win, device=device),
            torch.arange(win, device=device),
            indexing="ij",
        )
        return oy.reshape(-1), ox.reshape(-1)

    def _gather_windows_padded(
        self,
        feat: torch.Tensor,
        cy: torch.Tensor,
        cx: torch.Tensor,
        oy: torch.Tensor,
        ox: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gather local windows from ``feat`` (B,C,H,W) at centers (cy,cx).

        Returns ``(B, N, K, C)`` without materialising a full ``unfold`` over H×W
        (that intermediate is what OOMs at level-0 / ~196² HR).
        """
        b, c, h_hr, w_hr = feat.shape
        pad = self.window // 2
        padded = F.pad(feat, (pad, pad, pad, pad))
        hp, wp = padded.shape[-2:]
        # Matches F.unfold(..., padding=pad): window top-left at (cy, cx) in padded coords.
        ys = cy[:, None] + oy[None, :]
        xs = cx[:, None] + ox[None, :]
        lin = (ys * wp + xs).reshape(-1)
        flat = padded.reshape(b, c, hp * wp)
        gathered = flat.index_select(dim=2, index=lin)  # (B, C, N*K)
        n = cy.shape[0]
        k = self.local_token_count
        return gathered.view(b, c, n, k).permute(0, 2, 3, 1).contiguous()

    def _unfold_gather(self, feat: torch.Tensor, h_u: int, w_u: int) -> torch.Tensor:
        """Gather local windows for each UNet query → (B,N,K,C)."""
        _b, _c, h_hr, w_hr = feat.shape
        cy, cx = self._query_hr_centers(h_u, w_u, h_hr, w_hr, device=feat.device)
        oy, ox = self._window_offsets(device=feat.device)
        return self._gather_windows_padded(feat, cy, cx, oy, ox)

    def _gather_local_hr_windows(self, hr_feat: torch.Tensor, h_u: int, w_u: int) -> torch.Tensor:
        """Test helper: local HR patches with coords (pre K/V projection)."""
        return self._unfold_gather(append_coord_channels(hr_feat), h_u, w_u)

    def _local_cross_attn_hw(
        self,
        q: torch.Tensor,
        k_map: torch.Tensor,
        v_map: torch.Tensor,
        h_u: int,
        w_u: int,
        *,
        return_attn: bool,
        chunk_queries: int = 1024,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        b, heads, n, d = q.shape
        _bk, _ck, h_hr, w_hr = k_map.shape
        cy, cx = self._query_hr_centers(h_u, w_u, h_hr, w_hr, device=q.device)
        oy, ox = self._window_offsets(device=q.device)
        drop = self.dropout_p if self.training else 0.0
        scale = d**-0.5
        k_tok = self.local_token_count
        ctx_chunks: list[torch.Tensor] = []
        attn_chunks: list[torch.Tensor] = []

        for start in range(0, n, int(chunk_queries)):
            end = min(start + int(chunk_queries), n)
            k_loc = self._gather_windows_padded(k_map, cy[start:end], cx[start:end], oy, ox)
            v_loc = self._gather_windows_padded(v_map, cy[start:end], cx[start:end], oy, ox)
            nc = end - start
            # (B, heads, nc, K, d)
            k_ = k_loc.view(b, nc, k_tok, heads, d).permute(0, 3, 1, 2, 4)
            v_ = v_loc.view(b, nc, k_tok, heads, d).permute(0, 3, 1, 2, 4)
            q_ = q[:, :, start:end, :].unsqueeze(3)  # (B, heads, nc, 1, d)
            logits = torch.matmul(q_, k_.transpose(-1, -2)) * scale
            attn = torch.softmax(logits, dim=-1)
            if drop > 0:
                attn = F.dropout(attn, p=drop)
            ctx = torch.matmul(attn, v_).squeeze(3)  # (B, heads, nc, d)
            ctx_chunks.append(ctx)
            if return_attn:
                attn_chunks.append(attn.mean(dim=1).squeeze(2))  # (B, nc, K)

        ctx = torch.cat(ctx_chunks, dim=2)
        if return_attn:
            return ctx, torch.cat(attn_chunks, dim=1)
        return ctx, None

    def _sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        return_attn: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        drop = self.dropout_p if self.training else 0.0
        if return_attn:
            scale = q.shape[-1] ** -0.5
            logits = torch.matmul(q, k.transpose(-1, -2)) * scale
            attn = torch.softmax(logits, dim=-1)
            if drop > 0:
                attn = F.dropout(attn, p=drop)
            ctx = torch.matmul(attn, v)
            return ctx, attn.mean(dim=1)
        ctx = F.scaled_dot_product_attention(q, k, v, dropout_p=drop)
        return ctx, None

    def forward(
        self,
        unet_feat: torch.Tensor,
        hr_feat: torch.Tensor,
        *,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = unet_feat.shape
        n = h * w
        heads, d = self.num_heads, self.head_dim

        q_map = self.query_proj(append_coord_channels(unet_feat))
        q = q_map.flatten(2).transpose(1, 2).view(b, n, heads, d).transpose(1, 2)

        hr_c = append_coord_channels(hr_feat)
        if self.mode == "local":
            k_map = self.key_proj(hr_c)
            v_map = self.value_proj(hr_c)
            ctx, attn = self._local_cross_attn_hw(
                q, k_map, v_map, h, w, return_attn=return_attn
            )
        else:
            k = self.key_proj(hr_c).flatten(2).transpose(1, 2)
            v = self.value_proj(hr_c).flatten(2).transpose(1, 2)
            k = k.view(b, -1, heads, d).transpose(1, 2)
            v = v.view(b, -1, heads, d).transpose(1, 2)
            ctx, attn = self._sdpa(q, k, v, return_attn=return_attn)

        ctx = ctx.transpose(1, 2).contiguous().view(b, self.attn_dim, h, w)
        delta = self.out_proj(ctx)
        out = self.norm(unet_feat + delta)
        if return_attn:
            assert attn is not None
            return out, attn
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
