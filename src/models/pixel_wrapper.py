"""Trainer-compatible wrapper for pixel-SED photometric baselines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from src.models.config import ModelConfig
from src.models.input_prep import prepare_imaging_input, prepare_targets_and_masks
from src.models.losses import compose_map_losses
from src.models.pixel_sed import PixelSEDRegressor, PixelSEDVariant

# Marker used by Trainer / eval dispatch.
USES_BATCH_FORWARD_EVAL = True


@dataclass
class PixelWrapperConfig:
    """Architecture knobs for pixel-SED models (prep/losses live on ModelConfig)."""

    variant: PixelSEDVariant = "mlp"
    hidden_channels: int = 32
    activation: str = "gelu"


class PixelMapGenerator(nn.Module):
    """
    Pixel-level ugriz → maps baseline.

    Does not use footprint / spectrum / spatial neighbourhood as features.
    Supervision uses the same Hα (or target_keys) loss masks as the UNet stack.
    """

    uses_batch_forward_eval = True

    def __init__(
        self,
        config: ModelConfig,
        *,
        variant: PixelSEDVariant = "mlp",
        hidden_channels: int = 32,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if config.imaging_resolution != "aligned":
            raise ValueError("Pixel SED baselines require imaging_resolution='aligned'")
        self.config = config
        self.variant = variant
        self.net = PixelSEDRegressor(
            in_channels=config.imaging_input_channels(),
            out_channels=config.n_target_maps,
            variant=variant,
            hidden_channels=hidden_channels,
            activation=activation,
        )

    def forward(
        self,
        batch: dict[str, object],
        *,
        epoch: int | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        del epoch  # unused; kept for Trainer signature parity
        x = prepare_imaging_input(batch, self.config)
        targets, masks = prepare_targets_and_masks(batch, self.config)
        pred_maps = self.net(x)
        loss_dict = compose_map_losses(
            pred_maps.float(),
            targets.float(),
            masks.float(),
            losses=self.config.losses,
            loss_weights=self.config.loss_weights,
            loss_params=self.config.loss_params,
            target_keys=self.config.target_keys,
        )
        pred_dict = {"maps": pred_maps, "targets": targets, "masks": masks}
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
