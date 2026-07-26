from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


SpectrumPooling = Literal["avg", "attention"]


class SpectrumEncoder(nn.Module):
    """
    1D CNN over spectrum channels → conditioning vector for FiLM.

    Input ``spectrum`` is ``(B, n_wave)`` (flux only) or ``(B, C, n_wave)`` with
    channels ordered as: flux [, wavelength] [, ivar].
    """

    def __init__(
        self,
        n_wave: int = 4563,
        out_dim: int = 384,
        *,
        in_channels: int = 1,
        pooling: SpectrumPooling = "attention",
    ) -> None:
        super().__init__()
        self.n_wave = n_wave
        self.pooling = pooling
        self.in_channels = in_channels
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.attn: nn.Module | None
        if pooling == "attention":
            self.attn = nn.Sequential(
                nn.Linear(256, 128),
                nn.Tanh(),
                nn.Linear(128, 1),
            )
        else:
            self.attn = None
        self.proj = nn.Sequential(
            nn.Linear(256, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        if spectrum.ndim == 2:
            spectrum = spectrum.unsqueeze(1)
        if spectrum.shape[1] != self.in_channels:
            raise ValueError(
                f"SpectrumEncoder expected {self.in_channels} channels, got {spectrum.shape[1]}"
            )
        feats = self.net(spectrum)  # (B, 256, L')
        if self.pooling == "avg" or self.attn is None:
            pooled = F.adaptive_avg_pool1d(feats, 1).flatten(1)
        else:
            tokens = feats.transpose(1, 2)  # (B, L', 256)
            scores = self.attn(tokens)  # (B, L', 1)
            weights = torch.softmax(scores, dim=1)
            pooled = (tokens * weights).sum(dim=1)
        return self.proj(pooled)


class FiLM2d(nn.Module):
    """Feature-wise linear modulation from a global conditioning vector."""

    def __init__(self, cond_dim: int, num_channels: int) -> None:
        super().__init__()
        self.to_gamma_beta = nn.Linear(cond_dim, 2 * num_channels)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class CoarseFineHead(nn.Module):
    """
    Coarse head predicts downsampled maps; fine head adds full-res residual.

    pred = coarse_up + detail_scale * residual
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        coarse_factor: int = 2,
        detail_scale_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.coarse_factor = coarse_factor
        self.coarse = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1),
        )
        self.fine = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1),
        )
        self.detail_scale = nn.Parameter(torch.tensor(float(detail_scale_init)))

    def forward(
        self,
        features: torch.Tensor,
        *,
        detail_scale_multiplier: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, w = features.shape[-2:]
        coarse_h, coarse_w = h // self.coarse_factor, w // self.coarse_factor
        coarse_in = F.adaptive_avg_pool2d(features, (coarse_h, coarse_w))
        coarse_out = self.coarse(coarse_in)
        coarse_up = F.interpolate(coarse_out, size=(h, w), mode="bilinear", align_corners=False)
        fine_in = torch.cat([features, coarse_up], dim=1)
        residual = self.fine(fine_in)
        scale = self.detail_scale * float(detail_scale_multiplier)
        maps = coarse_up + scale * residual
        return maps, coarse_up, residual
