from __future__ import annotations

import torch
import torch.nn as nn

from src.models.conditional_unet import ConditionalMapModel
from src.models.config import ModelConfig, effective_detail_scale_multiplier
from src.models.losses import (
    compose_map_losses,
    masked_charbonnier,
    masked_l1,
    masked_laplacian_loss,
    masked_mse,
    masked_pairwise_grad_loss,
)


def _nan_to_num(t: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


def prepare_imaging_input(batch: dict[str, object], config: ModelConfig) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    inputs = batch.get("inputs", {})

    if config.use_sdss:
        parts.append(_nan_to_num(inputs["sdss_imaging"].float()))  # type: ignore[index]
    if config.use_legacy:
        parts.append(_nan_to_num(inputs["legacy_imaging"].float()))  # type: ignore[index]
    if not parts:
        raise ValueError("No spatial imaging found in batch.")
    x = torch.cat(parts, dim=1)
    if config.imaging_clamp_min is not None or config.imaging_clamp_max is not None:
        lo = config.imaging_clamp_min if config.imaging_clamp_min is not None else -float("inf")
        hi = config.imaging_clamp_max if config.imaging_clamp_max is not None else float("inf")
        x = torch.clamp(x, min=lo, max=hi)
    return x


def prepare_footprint_input(batch: dict[str, object], config: ModelConfig) -> torch.Tensor | None:
    if not config.uses_footprint_in_model():
        return None
    if config.footprint_mode == "loss_only":
        return None
    return batch["footprint_mask"].float()  # type: ignore[index]


def prepare_spatial_input(batch: dict[str, object], config: ModelConfig) -> torch.Tensor:
    """Backwards-compatible alias: imaging stack, optionally with footprint channel."""
    x = prepare_imaging_input(batch, config)
    if config.spatial_pipeline == "symmetric" and config.footprint_mode == "spatial_channel":
        footprint = prepare_footprint_input(batch, config)
        if footprint is not None:
            if footprint.ndim == x.ndim - 1:
                footprint = footprint.unsqueeze(1)
            x = torch.cat([x, footprint], dim=1)
    return x


def prepare_spectrum_input(batch: dict[str, object], config: ModelConfig) -> torch.Tensor | None:
    """
    Build spectrum tensor for SpectrumEncoder.

    Returns ``(B, C, n_wave)`` with channels: flux [, λ_norm] [, log1p(ivar)].
    """
    if not config.use_spectrum:
        return None
    inputs = batch.get("inputs", {})
    spec = inputs["spectrum"]  # type: ignore[index]
    flux = _nan_to_num(spec["flux"].float())  # type: ignore[index]
    channels: list[torch.Tensor] = [flux]

    if config.spectrum_use_wavelength:
        wave = spec.get("wave")  # type: ignore[union-attr]
        if wave is None:
            b, n = flux.shape
            t = torch.linspace(0.0, 1.0, n, device=flux.device, dtype=flux.dtype)
            wave_norm = (2.0 * t - 1.0).unsqueeze(0).expand(b, -1)
        else:
            wave_t = _nan_to_num(wave.float())
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
            ivar_t = _nan_to_num(ivar.float()).clamp_min(0.0)
        channels.append(torch.log1p(ivar_t))

    return torch.stack(channels, dim=1)


def prepare_targets_and_masks(
    batch: dict[str, object],
    config: ModelConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    keys = config.target_keys
    targets = torch.stack([batch["targets"][key].float() for key in keys], dim=1)  # type: ignore[index]
    masks = torch.stack([batch["target_loss_masks"][key].float() for key in keys], dim=1)  # type: ignore[index]
    return _nan_to_num(targets), masks


_DS_LOSS_FN = {
    "l1": masked_l1,
    "mse": masked_mse,
    "charbonnier": masked_charbonnier,
    "grad": masked_pairwise_grad_loss,
    "laplacian": masked_laplacian_loss,
}


def add_deep_supervision_losses(
    loss_dict: dict[str, torch.Tensor],
    *,
    deep_maps: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    config: ModelConfig,
) -> dict[str, torch.Tensor]:
    """
    Add masked fidelity losses on UNet++ auxiliary DS heads (excludes deepest).

    deep_maps: (L, B, C, H, W) stacked predictions from shallow → deep.
    Deepest head already carries the full compose_map_losses term.
    """
    if deep_maps.ndim != 5 or deep_maps.shape[0] < 2:
        return loss_dict
    weights = config.resolved_deep_supervision_weights()
    loss_fn = _DS_LOSS_FN[config.deep_supervision_loss]
    total = loss_dict["loss"]
    n_aux = deep_maps.shape[0] - 1
    for i in range(n_aux):
        w = weights[i] if i < len(weights) else 0.0
        if w <= 0:
            continue
        aux_loss = loss_fn(deep_maps[i].float(), targets, masks)
        loss_dict[f"ds_{i}"] = aux_loss
        total = total + float(w) * aux_loss
    loss_dict["loss"] = total
    return loss_dict


class MapGenerator(nn.Module):
    """Wrapper: batch dict -> (pred_dict, loss_dict) for Trainer compatibility."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.model = ConditionalMapModel(config)

    def forward(
        self,
        batch: dict[str, object],
        *,
        epoch: int | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        x = prepare_imaging_input(batch, self.config)
        footprint = prepare_footprint_input(batch, self.config)
        spec = prepare_spectrum_input(batch, self.config)
        targets, masks = prepare_targets_and_masks(batch, self.config)

        detail_mult = effective_detail_scale_multiplier(self.config, epoch)
        pred_maps, aux = self.model(
            x,
            spectrum=spec,
            footprint=footprint,
            detail_scale_multiplier=detail_mult,
        )
        # Losses in fp32 — avoids AMP overflow (especially integration / grad terms).
        loss_dict = compose_map_losses(
            pred_maps.float(),
            targets.float(),
            masks.float(),
            losses=self.config.losses,
            loss_weights=self.config.loss_weights,
            loss_params=self.config.loss_params,
            target_keys=self.config.target_keys,
            residual=aux.get("residual"),
        )
        if self.config.deep_supervision and "deep_maps" in aux:
            loss_dict = add_deep_supervision_losses(
                loss_dict,
                deep_maps=aux["deep_maps"],
                targets=targets.float(),
                masks=masks.float(),
                config=self.config,
            )
        pred_dict = {"maps": pred_maps, "targets": targets, "masks": masks, **aux}
        return pred_dict, loss_dict

    @torch.no_grad()
    def predict(
        self,
        batch: dict[str, object],
        *,
        epoch: int | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self.eval()
        return self.forward(batch, epoch=epoch)
