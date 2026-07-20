from __future__ import annotations

import torch
import torch.nn as nn

from src.models.conditional_unet import ConditionalMapModel
from src.models.config import ModelConfig, effective_detail_scale_multiplier
from src.models.losses import compose_map_losses
from src.models.wrapper import (
    prepare_footprint_input,
    prepare_imaging_input,
    prepare_spectrum_input,
    prepare_targets_and_masks,
)


class UncertaintyMapGenerator(nn.Module):
    """Heteroscedastic map model: μ + σ per target channel (gaussian output head)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.output_head != "gaussian":
            raise ValueError(
                f"UncertaintyMapGenerator requires output_head='gaussian', got {config.output_head!r}"
            )
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
            spectrum_flux=spec,
            footprint=footprint,
            detail_scale_multiplier=detail_mult,
        )
        log_var = aux["log_var"]
        loss_dict = compose_map_losses(
            pred_maps.float(),
            targets.float(),
            masks.float(),
            losses=self.config.losses,
            loss_weights=self.config.loss_weights,
            loss_params=self.config.loss_params,
            target_keys=self.config.target_keys,
            log_var=log_var.float(),
            residual=aux.get("residual"),
        )
        pred_dict = {
            "maps": pred_maps,
            "targets": targets,
            "masks": masks,
            "log_var": log_var,
            "sigma": aux["sigma"],
            **aux,
        }
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
