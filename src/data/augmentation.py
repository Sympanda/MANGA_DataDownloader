from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AugmentConfig:
    enabled: bool = True
    hflip: bool = True
    vflip: bool = True
    rot90: bool = True
    p: float = 0.5


def _sample_spatial_transform(cfg: AugmentConfig) -> tuple[int, bool, bool]:
    """Return (k_rot90, hflip, vflip) applied consistently across spatial tensors."""
    k = 0
    hflip = False
    vflip = False
    if cfg.rot90 and torch.rand(()) < cfg.p:
        k = int(torch.randint(1, 4, ()).item())
    if cfg.hflip and torch.rand(()) < cfg.p:
        hflip = True
    if cfg.vflip and torch.rand(()) < cfg.p:
        vflip = True
    return k, hflip, vflip


def _apply_spatial_transform(x: torch.Tensor, k: int, hflip: bool, vflip: bool) -> torch.Tensor:
    """Apply the same geometric transform to (C,H,W) or (H,W) tensors."""
    if hflip:
        x = torch.flip(x, dims=[-1])
    if vflip:
        x = torch.flip(x, dims=[-2])
    if k:
        x = torch.rot90(x, k=k, dims=[-2, -1])
    return x


def augment_spatial_sample(
    *,
    sdss: torch.Tensor | None = None,
    legacy: torch.Tensor | None = None,
    hr: torch.Tensor | None = None,
    footprint: torch.Tensor | None = None,
    targets: dict[str, torch.Tensor] | None = None,
    target_masks: dict[str, torch.Tensor] | None = None,
    cfg: AugmentConfig,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    dict[str, torch.Tensor] | None,
    dict[str, torch.Tensor] | None,
]:
    """
    Apply identical rotations/flips to all spatial inputs and target maps.
    Spectrum is not transformed.
    """
    if not cfg.enabled:
        return sdss, legacy, hr, footprint, targets, target_masks

    k, hflip, vflip = _sample_spatial_transform(cfg)
    if k == 0 and not hflip and not vflip:
        return sdss, legacy, hr, footprint, targets, target_masks

    if sdss is not None:
        sdss = _apply_spatial_transform(sdss, k, hflip, vflip)
    if legacy is not None:
        legacy = _apply_spatial_transform(legacy, k, hflip, vflip)
    if hr is not None:
        hr = _apply_spatial_transform(hr, k, hflip, vflip)
    if footprint is not None:
        footprint = _apply_spatial_transform(footprint, k, hflip, vflip)
    if targets is not None:
        targets = {key: _apply_spatial_transform(t, k, hflip, vflip) for key, t in targets.items()}
    if target_masks is not None:
        target_masks = {
            key: _apply_spatial_transform(m, k, hflip, vflip) for key, m in target_masks.items()
        }
    return sdss, legacy, hr, footprint, targets, target_masks
