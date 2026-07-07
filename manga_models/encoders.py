from __future__ import annotations

import torch
import torch.nn as nn


class SpectrumEncoder(nn.Module):
    """1D CNN over resampled flux -> conditioning vector."""

    def __init__(self, n_wave: int = 4563, out_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, flux: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        flux : (B, n_wave) raw resampled flux (NaNs should be zeroed by caller)
        """
        x = flux.unsqueeze(1)
        x = self.net(x).flatten(1)
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
