"""Full-map conditional score / diffusion UNet (pixel-space, 76×76)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoders import FiLM2d, SpectrumEncoder
from src.models.residual_diffusion import sinusoidal_timestep_embedding


class MapScoreResBlock(nn.Module):
    """Residual block with time additive embedding + optional spectrum FiLM."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, *, cond_dim: int | None = None) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.film: FiLM2d | None
        if cond_dim is not None and cond_dim > 0:
            self.film = FiLM2d(cond_dim, out_ch)
        else:
            self.film = None

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        spec_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(F.silu(emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        h = self.skip(x) + h
        if self.film is not None and spec_cond is not None:
            h = self.film(h, spec_cond)
        return h


class SelfAttention2d(nn.Module):
    """Lightweight multi-head self-attention over H×W (for bottleneck only)."""

    def __init__(self, channels: int, *, num_heads: int = 4) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")
        self.num_heads = int(num_heads)
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        h_norm = self.norm(x)
        qkv = self.qkv(h_norm).reshape(b, 3, self.num_heads, c // self.num_heads, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        scale = (c // self.num_heads) ** -0.5
        attn = torch.softmax(torch.einsum("bhcn,bhcm->bhnm", q, k) * scale, dim=-1)
        out = torch.einsum("bhnm,bhcm->bhcn", attn, v).reshape(b, c, h, w)
        return x + self.proj(out)


class CondMapScoreUNet(nn.Module):
    """
    Pixel-space epsilon predictor for full Hα maps.

    Spatial conditioning is channel-concatenated with the noisy map.
    Spectrum conditioning is injected via FiLM into residual blocks.
    Optional self-attention is applied only at the bottleneck.
    """

    def __init__(
        self,
        *,
        cond_channels: int,
        map_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        emb_dim: int = 256,
        use_spectrum: bool = True,
        spectrum_n_wave: int = 4563,
        spectrum_in_channels: int = 1,
        spectrum_pooling: str = "attention",
        cond_dim: int = 128,
        use_bottleneck_attn: bool = False,
        attn_heads: int = 4,
    ) -> None:
        super().__init__()
        self.map_channels = map_channels
        self.cond_channels = cond_channels
        self.use_spectrum = use_spectrum
        self.emb_dim = emb_dim
        self.use_bottleneck_attn = bool(use_bottleneck_attn)
        in_ch = map_channels + cond_channels

        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )
        self.spectrum_encoder: SpectrumEncoder | None = None
        film_dim: int | None = None
        if use_spectrum:
            self.spectrum_encoder = SpectrumEncoder(
                n_wave=spectrum_n_wave,
                out_dim=cond_dim,
                in_channels=spectrum_in_channels,
                pooling=spectrum_pooling,  # type: ignore[arg-type]
            )
            film_dim = cond_dim

        self.in_conv = nn.Conv2d(in_ch, base_channels, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = base_channels
        self._skip_channels: list[int] = []
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(MapScoreResBlock(ch, out_ch, emb_dim, cond_dim=film_dim))
                ch = out_ch
                self._skip_channels.append(ch)
            if i < len(channel_mults) - 1:
                self.downsamples.append(nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1))
            else:
                self.downsamples.append(nn.Identity())

        self.mid1 = MapScoreResBlock(ch, ch, emb_dim, cond_dim=film_dim)
        self.mid_attn: SelfAttention2d | None
        if self.use_bottleneck_attn:
            self.mid_attn = SelfAttention2d(ch, num_heads=attn_heads)
        else:
            self.mid_attn = None
        self.mid2 = MapScoreResBlock(ch, ch, emb_dim, cond_dim=film_dim)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                skip_ch = self._skip_channels.pop()
                self.up_blocks.append(
                    MapScoreResBlock(ch + skip_ch, out_ch, emb_dim, cond_dim=film_dim)
                )
                ch = out_ch
            if i > 0:
                self.upsamples.append(nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1))
            else:
                self.upsamples.append(nn.Identity())

        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv2d(ch, map_channels, kernel_size=3, padding=1)

    def encode_spectrum(self, spectrum: torch.Tensor | None) -> torch.Tensor | None:
        if not self.use_spectrum or self.spectrum_encoder is None or spectrum is None:
            return None
        return self.spectrum_encoder(spectrum)

    def forward(
        self,
        noisy_map: torch.Tensor,
        timesteps: torch.Tensor,
        cond: torch.Tensor,
        spectrum: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noisy_map.shape[1] != self.map_channels:
            raise ValueError(
                f"Expected map channels {self.map_channels}, got {noisy_map.shape[1]}"
            )
        if cond.shape[1] != self.cond_channels:
            raise ValueError(
                f"Expected cond channels {self.cond_channels}, got {cond.shape[1]}"
            )
        emb = self.time_mlp(sinusoidal_timestep_embedding(timesteps, self.emb_dim))
        spec_cond = self.encode_spectrum(spectrum)

        h = self.in_conv(torch.cat([noisy_map, cond], dim=1))
        skips: list[torch.Tensor] = []
        n_levels = len(self.downsamples)
        blocks_per = len(self.down_blocks) // n_levels
        bi = 0
        for level in range(n_levels):
            for _ in range(blocks_per):
                h = self.down_blocks[bi](h, emb, spec_cond)
                bi += 1
                skips.append(h)
            h = self.downsamples[level](h)

        h = self.mid1(h, emb, spec_cond)
        if self.mid_attn is not None:
            h = self.mid_attn(h)
        h = self.mid2(h, emb, spec_cond)

        bi = 0
        for level in range(n_levels):
            for _ in range(blocks_per):
                skip = skips.pop()
                if h.shape[-2:] != skip.shape[-2:]:
                    h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
                h = self.up_blocks[bi](torch.cat([h, skip], dim=1), emb, spec_cond)
                bi += 1
            h = self.upsamples[level](h)

        return self.out_conv(F.silu(self.out_norm(h)))


@dataclass
class ScoreNormStats:
    """Per-channel mean/std for score-space maps (train-split, footprint pixels)."""

    mean: float
    std: float

    def to_dict(self) -> dict[str, float]:
        return {"mean": float(self.mean), "std": float(self.std)}

    @classmethod
    def from_dict(cls, d: dict) -> ScoreNormStats:
        return cls(mean=float(d["mean"]), std=float(d["std"]))

    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.mean) / max(self.std, 1e-6)

    def denormalize(self, y: torch.Tensor) -> torch.Tensor:
        return y * max(self.std, 1e-6) + self.mean


class MapDiffusionSchedule:
    """Linear / cosine schedule with DDIM sampling and SDEdit-style starts."""

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

    def to(self, device: torch.device) -> MapDiffusionSchedule:
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

    def t_from_fraction(self, frac: float) -> int:
        """Map frac in (0,1] to a discrete timestep index (higher = more noise)."""
        frac = float(np.clip(frac, 1.0 / self.n_steps, 1.0))
        return int(round(frac * (self.n_steps - 1)))

    @torch.no_grad()
    def ddim_sample(
        self,
        model: CondMapScoreUNet,
        cond: torch.Tensor,
        *,
        steps: int = 50,
        eta: float = 0.0,
        generator: torch.Generator | None = None,
        footprint_mask: torch.Tensor | None = None,
        x_init: torch.Tensor | None = None,
        t_start: int | None = None,
        spectrum: torch.Tensor | None = None,
        x0_clip: tuple[float, float] | None = (-10.0, 10.0),
    ) -> torch.Tensor:
        """
        DDIM reverse process over the footprint domain.

        - ``x_init is None`` → start from N(0,I) at T (direct generator).
        - ``x_init`` + ``t_start`` → SDEdit-style corrector start.
        Outside ``footprint_mask``, values are held at 0 (not label_mask).

        ``x0_clip`` clamps the predicted clean map in **score space** each step
        (prevents DDIM blow-ups that look like binary 0/1 after denorm+plot).
        """
        device = cond.device
        b = cond.shape[0]
        h, w = cond.shape[-2], cond.shape[-1]
        if footprint_mask is not None:
            if footprint_mask.ndim == 3:
                footprint_mask = footprint_mask.unsqueeze(1)
            fp = (footprint_mask > 0).to(dtype=cond.dtype)
        else:
            fp = None

        t0 = self.n_steps - 1 if t_start is None else int(t_start)
        t0 = max(0, min(t0, self.n_steps - 1))

        if x_init is None:
            x = torch.randn(
                (b, model.map_channels, h, w),
                device=device,
                generator=generator,
            )
        else:
            noise = torch.randn(
                x_init.shape, device=device, generator=generator, dtype=x_init.dtype
            )
            t_tensor = torch.full((b,), t0, device=device, dtype=torch.long)
            x, _ = self.q_sample(x_init, t_tensor, noise=noise)

        if fp is not None:
            x = x * fp

        if t0 == 0:
            return x if fp is None else x * fp

        # Inclusive schedule from t0 down to 0.
        step_ids = torch.linspace(t0, 0, steps, device=device).long().unique_consecutive()
        alphas_cumprod = self.register["alphas_cumprod"]

        for i, t_val in enumerate(step_ids):
            t = torch.full((b,), int(t_val.item()), device=device, dtype=torch.long)
            eps = model(x, t, cond, spectrum=spectrum)
            a_t = alphas_cumprod[t].view(-1, 1, 1, 1)
            x0 = (x - torch.sqrt(1.0 - a_t) * eps) / torch.sqrt(a_t).clamp_min(1e-8)
            if x0_clip is not None:
                x0 = x0.clamp(float(x0_clip[0]), float(x0_clip[1]))
            if fp is not None:
                x0 = x0 * fp
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
            if fp is not None:
                x = x * fp
        if fp is not None:
            x = x * fp
        return x

    @torch.no_grad()
    def ddim_inpaint(
        self,
        model: CondMapScoreUNet,
        cond: torch.Tensor,
        *,
        y0: torch.Tensor,
        known_mask: torch.Tensor,
        steps: int = 50,
        eta: float = 0.0,
        generator: torch.Generator | None = None,
        footprint_mask: torch.Tensor | None = None,
        spectrum: torch.Tensor | None = None,
        x0_clip: tuple[float, float] | None = (-10.0, 10.0),
    ) -> torch.Tensor:
        """
        RePaint-style DDIM: free pixels follow the model; known pixels are
        re-injected from ``q(y0, t)`` every reverse step (score-space ``y0``).
        """
        device = cond.device
        b = cond.shape[0]
        h, w = cond.shape[-2], cond.shape[-1]
        if footprint_mask is not None:
            if footprint_mask.ndim == 3:
                footprint_mask = footprint_mask.unsqueeze(1)
            fp = (footprint_mask > 0).to(dtype=cond.dtype)
        else:
            fp = torch.ones((b, 1, h, w), device=device, dtype=cond.dtype)
        if known_mask.ndim == 3:
            known_mask = known_mask.unsqueeze(1)
        known = ((known_mask > 0) & (fp > 0)).to(dtype=cond.dtype)
        free = (fp > 0).to(dtype=cond.dtype) * (1.0 - known)

        t0 = self.n_steps - 1
        x = torch.randn(
            (b, model.map_channels, h, w),
            device=device,
            generator=generator,
        )
        # Initialise known region at the noised truth for t0.
        t_tensor = torch.full((b,), t0, device=device, dtype=torch.long)
        x_known0, _ = self.q_sample(y0, t_tensor, noise=torch.randn_like(y0))
        x = known * x_known0 + free * x
        x = x * fp

        step_ids = torch.linspace(t0, 0, steps, device=device).long().unique_consecutive()
        alphas_cumprod = self.register["alphas_cumprod"]

        for i, t_val in enumerate(step_ids):
            t = torch.full((b,), int(t_val.item()), device=device, dtype=torch.long)
            eps = model(x, t, cond, spectrum=spectrum)
            a_t = alphas_cumprod[t].view(-1, 1, 1, 1)
            x0 = (x - torch.sqrt(1.0 - a_t) * eps) / torch.sqrt(a_t).clamp_min(1e-8)
            if x0_clip is not None:
                x0 = x0.clamp(float(x0_clip[0]), float(x0_clip[1]))
            x0 = x0 * fp

            if i == len(step_ids) - 1:
                # Final: hard paste known truth; free from model x0.
                x = known * y0 + free * x0
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
                noise = torch.zeros_like(x)
            x_u = torch.sqrt(a_prev) * x0 + dir_xt + sigma * noise

            # RePaint: resample known pixels from q(y0, t_prev).
            t_prev_b = torch.full((b,), t_prev, device=device, dtype=torch.long)
            x_k, _ = self.q_sample(y0, t_prev_b, noise=torch.randn_like(y0))
            x = known * x_k + free * x_u
            x = x * fp

        return x * fp


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = float(decay)
        self.shadow = {
            k: v.detach().clone() for k, v in model.state_dict().items() if v.is_floating_point()
        }

    def to(self, device: torch.device | str) -> EMA:
        self.shadow = {k: v.to(device) for k, v in self.shadow.items()}
        return self

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k not in self.shadow or not v.is_floating_point():
                continue
            if self.shadow[k].device != v.device:
                self.shadow[k] = self.shadow[k].to(device=v.device, dtype=v.dtype)
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: nn.Module) -> None:
        state = model.state_dict()
        for k, v in self.shadow.items():
            if k in state and state[k].shape == v.shape:
                if v.device != state[k].device:
                    v = v.to(state[k].device)
                state[k].copy_(v)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.clone() for k, v in state.items()}
