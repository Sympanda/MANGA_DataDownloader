from __future__ import annotations

import torch
import torch.nn as nn

from src.models.conditional_unet import ConditionalMapModel
from src.models.config import ModelConfig, effective_detail_scale_multiplier
from src.models.input_prep import (
    prepare_footprint_input,
    prepare_hr_imaging_input,
    prepare_imaging_input,
    prepare_redshift_input,
    prepare_spatial_input,
    prepare_spectrum_input,
    prepare_targets_and_masks,
)
from src.models.losses import (
    compose_map_losses,
    masked_charbonnier,
    masked_l1,
    masked_laplacian_loss,
    masked_mse,
    masked_pairwise_grad_loss,
)

# Re-export so existing imports from src.models.wrapper keep working.
__all__ = [
    "MapGenerator",
    "add_deep_supervision_losses",
    "prepare_footprint_input",
    "prepare_hr_imaging_input",
    "prepare_imaging_input",
    "prepare_redshift_input",
    "prepare_spatial_input",
    "prepare_spectrum_input",
    "prepare_targets_and_masks",
]


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
        x_hr = prepare_hr_imaging_input(batch, self.config)
        footprint = prepare_footprint_input(batch, self.config)
        spec = prepare_spectrum_input(batch, self.config)
        redshift = prepare_redshift_input(batch, self.config)
        targets, masks = prepare_targets_and_masks(batch, self.config)

        detail_mult = effective_detail_scale_multiplier(self.config, epoch)
        pred_maps, aux = self.model(
            x,
            spectrum=spec,
            footprint=footprint,
            x_hr=x_hr,
            redshift=redshift,
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
            log_var=aux.get("log_var"),
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
