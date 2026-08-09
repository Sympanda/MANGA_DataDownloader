"""Shared batch → tensor preparation for imaging, spectrum, footprint, and targets.

All map models (UNet, pixel-SED, residual) must use these helpers so imaging
normalisation and target scaling stay identical across experiments.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class ImagingPrepConfig(Protocol):
    use_sdss: bool
    use_legacy: bool
    input_norm_mode: str
    imaging_asinh_scales: list[float] | None
    imaging_clamp_min: float | None
    imaging_clamp_max: float | None


@runtime_checkable
class HRPrepConfig(Protocol):
    use_hr_cross_attn: bool
    input_norm_mode: str
    hr_asinh_scales: list[float] | None
    imaging_clamp_min: float | None
    imaging_clamp_max: float | None


@runtime_checkable
class FootprintPrepConfig(Protocol):
    footprint_mode: str

    def uses_footprint_in_model(self) -> bool: ...


@runtime_checkable
class SpectrumPrepConfig(Protocol):
    use_spectrum: bool
    input_norm_mode: str
    spectrum_asinh_scale_fake: float | None
    spectrum_asinh_scale_real: float | None
    spectrum_use_wavelength: bool
    spectrum_use_ivar: bool
    spectrum_wave_min: float
    spectrum_wave_max: float


@runtime_checkable
class RedshiftPrepConfig(Protocol):
    use_redshift_cond: bool


@runtime_checkable
class TargetPrepConfig(Protocol):
    target_keys: tuple[str, ...]


def nan_to_num(t: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


def apply_asinh_imaging(
    x: torch.Tensor,
    scales: list[float],
) -> torch.Tensor:
    """Apply asinh(f / s_b) with per-channel scales. Raises if channel count mismatches."""
    if len(scales) != x.shape[1]:
        raise ValueError(
            f"asinh scale length {len(scales)} != imaging channels {x.shape[1]}"
        )
    scale_t = torch.tensor(scales, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return torch.asinh(x / scale_t)


def apply_imaging_clamp(
    x: torch.Tensor,
    clamp_min: float | None,
    clamp_max: float | None,
) -> torch.Tensor:
    if clamp_min is None and clamp_max is None:
        return x
    lo = clamp_min if clamp_min is not None else -float("inf")
    hi = clamp_max if clamp_max is not None else float("inf")
    return torch.clamp(x, min=lo, max=hi)


def prepare_imaging_input(batch: dict[str, object], config: ImagingPrepConfig) -> torch.Tensor:
    """Build (B, C, H, W) imaging tensor with shared asinh / clamp normalisation."""
    parts: list[torch.Tensor] = []
    inputs = batch.get("inputs", {})

    if config.use_sdss:
        parts.append(nan_to_num(inputs["sdss_imaging"].float()))  # type: ignore[index]
    if config.use_legacy:
        parts.append(nan_to_num(inputs["legacy_imaging"].float()))  # type: ignore[index]
    if not parts:
        raise ValueError("No spatial imaging found in batch.")
    x = torch.cat(parts, dim=1)
    if config.input_norm_mode == "asinh":
        if config.imaging_asinh_scales is None:
            raise ValueError("input_norm_mode='asinh' but imaging_asinh_scales is unset")
        x = apply_asinh_imaging(x, config.imaging_asinh_scales)
    elif config.input_norm_mode not in ("none", "asinh"):
        raise ValueError(f"Unknown input_norm_mode: {config.input_norm_mode!r}")
    return apply_imaging_clamp(x, config.imaging_clamp_min, config.imaging_clamp_max)


def prepare_hr_imaging_input(batch: dict[str, object], config: HRPrepConfig) -> torch.Tensor | None:
    """High-res morphology stream for cross-attention (not resized onto the UNet grid)."""
    if not config.use_hr_cross_attn:
        return None
    inputs = batch.get("inputs", {})
    if "hr_imaging" not in inputs:  # type: ignore[operator]
        raise KeyError("Batch missing inputs['hr_imaging'] required by use_hr_cross_attn")
    x = nan_to_num(inputs["hr_imaging"].float())  # type: ignore[index]
    if config.input_norm_mode == "asinh":
        if config.hr_asinh_scales is None:
            raise ValueError("input_norm_mode='asinh' but hr_asinh_scales is unset")
        x = apply_asinh_imaging(x, config.hr_asinh_scales)
    return apply_imaging_clamp(x, config.imaging_clamp_min, config.imaging_clamp_max)


def prepare_footprint_input(batch: dict[str, object], config: FootprintPrepConfig) -> torch.Tensor | None:
    if not config.uses_footprint_in_model():
        return None
    if config.footprint_mode == "loss_only":
        return None
    return batch["footprint_mask"].float()  # type: ignore[index]


def prepare_spatial_input(batch: dict[str, object], config: object) -> torch.Tensor:
    """Backwards-compatible alias: imaging stack, optionally with footprint channel."""
    x = prepare_imaging_input(batch, config)  # type: ignore[arg-type]
    spatial_pipeline = getattr(config, "spatial_pipeline", "symmetric")
    footprint_mode = getattr(config, "footprint_mode", "loss_only")
    if spatial_pipeline == "symmetric" and footprint_mode == "spatial_channel":
        footprint = prepare_footprint_input(batch, config)  # type: ignore[arg-type]
        if footprint is not None:
            if footprint.ndim == x.ndim - 1:
                footprint = footprint.unsqueeze(1)
            x = torch.cat([x, footprint], dim=1)
    return x


def prepare_spectrum_input(batch: dict[str, object], config: SpectrumPrepConfig) -> torch.Tensor | None:
    """
    Build spectrum tensor for SpectrumEncoder.

    Returns ``(B, C, n_wave)`` with channels: flux [, λ_norm] [, log1p(ivar)].
    """
    if not config.use_spectrum:
        return None
    inputs = batch.get("inputs", {})
    spec = inputs["spectrum"]  # type: ignore[index]
    flux = nan_to_num(spec["flux"].float())  # type: ignore[index]

    if config.input_norm_mode == "asinh":
        s_fake = config.spectrum_asinh_scale_fake
        s_real = config.spectrum_asinh_scale_real
        if s_fake is None or s_real is None:
            raise ValueError("input_norm_mode='asinh' but spectrum asinh scales are unset")
        is_real = spec.get("is_real_sdss_fiber")  # type: ignore[union-attr]
        if is_real is None:
            scale = float(s_fake)
            flux = torch.asinh(flux / scale)
        else:
            is_real_t = is_real.to(device=flux.device, dtype=torch.bool)  # type: ignore[union-attr]
            if is_real_t.ndim == 0:
                is_real_t = is_real_t.expand(flux.shape[0])
            s = torch.where(
                is_real_t,
                torch.full((), float(s_real), device=flux.device, dtype=flux.dtype),
                torch.full((), float(s_fake), device=flux.device, dtype=flux.dtype),
            ).view(-1, 1)
            flux = torch.asinh(flux / s)

    channels: list[torch.Tensor] = [flux]

    if config.spectrum_use_wavelength:
        wave = spec.get("wave")  # type: ignore[union-attr]
        if wave is None:
            b, n = flux.shape
            t = torch.linspace(0.0, 1.0, n, device=flux.device, dtype=flux.dtype)
            wave_norm = (2.0 * t - 1.0).unsqueeze(0).expand(b, -1)
        else:
            wave_t = nan_to_num(wave.float())
            lo = float(config.spectrum_wave_min)
            hi = float(config.spectrum_wave_max)
            wave_norm = 2.0 * (wave_t - lo) / max(hi - lo, 1e-6) - 1.0
            wave_norm = wave_norm.clamp(-1.0, 1.0)
        channels.append(wave_norm)

    if config.spectrum_use_ivar:
        ivar = spec.get("ivar")  # type: ignore[union-attr]
        if ivar is None:
            ivar_t = torch.ones_like(flux)
        else:
            ivar_t = nan_to_num(ivar.float()).clamp_min(0.0)
        channels.append(torch.log1p(ivar_t))

    return torch.stack(channels, dim=1)


def prepare_redshift_input(
    batch: dict[str, object],
    config: RedshiftPrepConfig,
) -> torch.Tensor | None:
    """Return ``(B,)`` redshifts when ``use_redshift_cond`` is enabled."""
    if not getattr(config, "use_redshift_cond", False):
        return None
    if "redshift" not in batch:
        raise KeyError("Batch missing 'redshift' required by use_redshift_cond")
    z = batch["redshift"]
    if not torch.is_tensor(z):
        z = torch.as_tensor(z, dtype=torch.float32)
    return z.float().reshape(-1)


def prepare_targets_and_masks(
    batch: dict[str, object],
    config: TargetPrepConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    keys = config.target_keys
    targets = torch.stack([batch["targets"][key].float() for key in keys], dim=1)  # type: ignore[index]
    masks = torch.stack([batch["target_loss_masks"][key].float() for key in keys], dim=1)  # type: ignore[index]
    return nan_to_num(targets), masks
