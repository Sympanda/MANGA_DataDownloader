"""Small conditional UNet for residual-map diffusion at 76×76."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(F.silu(emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return self.skip(x) + h


class CondResidualDiffusionUNet(nn.Module):
    """
    Small pixel-space noise predictor for residual maps.

    Input concat: [R_t, conditioning] where conditioning typically includes
    ugriz, frozen base prediction, and valid-region mask.
    """

    def __init__(
        self,
        *,
        cond_channels: int,
        residual_channels: int = 1,
        base_channels: int = 32,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        emb_dim: int = 128,
    ) -> None:
        super().__init__()
        self.residual_channels = residual_channels
        self.cond_channels = cond_channels
        in_ch = residual_channels + cond_channels

        self.emb_dim = emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )
        self.in_conv = nn.Conv2d(in_ch, base_channels, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = base_channels
        self.skip_channels: list[int] = []
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            self.down_blocks.append(ResidualBlock(ch, out_ch, emb_dim))
            self.skip_channels.append(out_ch)
            ch = out_ch
            if i < len(channel_mults) - 1:
                self.downsamples.append(nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1))
            else:
                self.downsamples.append(nn.Identity())

        self.mid = ResidualBlock(ch, ch, emb_dim)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            skip_ch = base_channels * mult
            self.up_blocks.append(ResidualBlock(ch + skip_ch, skip_ch, emb_dim))
            ch = skip_ch
            if i > 0:
                self.upsamples.append(nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1))
            else:
                self.upsamples.append(nn.Identity())

        self.out_conv = nn.Conv2d(ch, residual_channels, kernel_size=3, padding=1)

    def forward(
        self,
        residual_noisy: torch.Tensor,
        timesteps: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        if residual_noisy.shape[1] != self.residual_channels:
            raise ValueError(
                f"Expected residual channels {self.residual_channels}, got {residual_noisy.shape[1]}"
            )
        if cond.shape[1] != self.cond_channels:
            raise ValueError(
                f"Expected cond channels {self.cond_channels}, got {cond.shape[1]}"
            )
        emb = self.time_mlp(sinusoidal_timestep_embedding(timesteps, self.emb_dim))
        h = self.in_conv(torch.cat([residual_noisy, cond], dim=1))

        skips: list[torch.Tensor] = []
        for block, down in zip(self.down_blocks, self.downsamples):
            h = block(h, emb)
            skips.append(h)
            h = down(h)

        h = self.mid(h, emb)

        for block, up in zip(self.up_blocks, self.upsamples):
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = block(torch.cat([h, skip], dim=1), emb)
            h = up(h)

        return self.out_conv(h)


class ResidualDiffusionSchedule:
    """Linear / cosine β schedule for DDPM training and DDIM sampling."""

    def __init__(
        self,
        *,
        n_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        schedule: str = "linear",
    ) -> None:
        self.n_steps = int(n_steps)
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, self.n_steps)
        elif schedule == "cosine":
            s = 0.008
            steps = torch.arange(self.n_steps + 1, dtype=torch.float64)
            alphas_cumprod = torch.cos(((steps / self.n_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = (1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])).clamp(max=0.999).float()
        else:
            raise ValueError(f"Unknown schedule: {schedule!r}")
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register = {
            "betas": betas,
            "alphas": alphas,
            "alphas_cumprod": alphas_cumprod,
            "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
            "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
        }

    def to(self, device: torch.device) -> ResidualDiffusionSchedule:
        self.register = {k: v.to(device) for k, v in self.register.items()}
        return self

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.register["sqrt_alphas_cumprod"][t].view(-1, 1, 1, 1)
        a_om = self.register["sqrt_one_minus_alphas_cumprod"][t].view(-1, 1, 1, 1)
        return a * x0 + a_om * noise, noise

    @torch.no_grad()
    def ddim_sample(
        self,
        model: CondResidualDiffusionUNet,
        cond: torch.Tensor,
        *,
        steps: int = 50,
        eta: float = 0.0,
        generator: torch.Generator | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = cond.device
        b = cond.shape[0]
        x = torch.randn(
            (b, model.residual_channels, cond.shape[-2], cond.shape[-1]),
            device=device,
            generator=generator,
        )
        if mask is not None:
            x = x * mask

        step_ids = torch.linspace(self.n_steps - 1, 0, steps, device=device).long()
        alphas_cumprod = self.register["alphas_cumprod"]

        for i, t_val in enumerate(step_ids):
            t = torch.full((b,), int(t_val.item()), device=device, dtype=torch.long)
            eps = model(x, t, cond)
            a_t = alphas_cumprod[t].view(-1, 1, 1, 1)
            x0 = (x - torch.sqrt(1.0 - a_t) * eps) / torch.sqrt(a_t).clamp_min(1e-8)
            if mask is not None:
                x0 = x0 * mask
            if i == len(step_ids) - 1:
                x = x0
                break
            t_prev = int(step_ids[i + 1].item())
            a_prev = alphas_cumprod[t_prev].view(1, 1, 1, 1).to(dtype=a_t.dtype, device=device)
            sigma = (
                eta
                * torch.sqrt((1 - a_prev) / (1 - a_t).clamp_min(1e-8))
                * torch.sqrt((1 - a_t / a_prev.clamp_min(1e-8)).clamp_min(0.0))
            )
            dir_xt = torch.sqrt((1.0 - a_prev - sigma**2).clamp_min(0.0)) * eps
            if eta > 0:
                noise = torch.randn(x.shape, device=device, generator=generator)
            else:
                noise = 0.0
            x = torch.sqrt(a_prev) * x0 + dir_xt + sigma * noise
            if mask is not None:
                x = x * mask
        if mask is not None:
            x = x * mask
        return x
