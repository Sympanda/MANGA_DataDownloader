from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectrumEncoder(nn.Module):
    """1D CNN over resampled flux -> conditioning vector."""

    def __init__(self, n_wave: int = 4563, out_dim: int = 384) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(
            nn.Linear(256, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, flux: torch.Tensor) -> torch.Tensor:
        x = self.net(flux.unsqueeze(1)).flatten(1)
        return self.proj(x)


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
