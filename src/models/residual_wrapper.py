"""Trainer-compatible wrappers for frozen-base residual models."""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from src.models.base_loader import frozen_base_predict, load_frozen_base_map_generator
from src.models.config import ModelConfig
from src.models.input_prep import prepare_imaging_input, prepare_targets_and_masks
from src.models.losses import compose_map_losses, masked_gaussian_nll, masked_l1
from src.models.residual_models import (
    GaussianPixelResidualRegressor,
    LocalResidualCNN,
    PixelResidualRegressor,
    ResidualVariant,
    build_residual_net,
)
from src.models.wrapper import MapGenerator


class ResidualMapGenerator(nn.Module):
    """
    Frozen base UNet + residual corrector.

    residual_target = target - base_prediction  (scaled target space)
    final maps      = base_prediction + residual_prediction
    """

    uses_batch_forward_eval = True

    def __init__(
        self,
        config: ModelConfig,
        *,
        base_model: MapGenerator,
        base_channel_index: int | None,
        variant: ResidualVariant = "pixel",
        hidden_channels: int = 32,
        n_residual_samples: int = 32,
        gaussian_l1_weight: float = 0.1,
        min_log_var: float = -6.0,
        max_log_var: float = 6.0,
    ) -> None:
        super().__init__()
        self.config = config
        self.variant = variant
        self.base_channel_index = (
            None if base_channel_index is None else int(base_channel_index)
        )
        self.n_residual_samples = int(n_residual_samples)
        self.gaussian_l1_weight = float(gaussian_l1_weight)
        self.min_log_var = float(min_log_var)
        self.max_log_var = float(max_log_var)

        # Register base as a non-trainable submodule so .to(device) moves it,
        # but Trainer only optimises requires_grad parameters.
        self.base_model = base_model
        self.base_model.eval()
        self.base_model.requires_grad_(False)

        in_ch = config.imaging_input_channels() + config.n_target_maps
        self.residual_net = build_residual_net(
            variant,
            in_channels=in_ch,
            out_channels=config.n_target_maps,
            hidden_channels=hidden_channels,
        )

    @classmethod
    def from_base_run(
        cls,
        config: ModelConfig,
        *,
        base_run_dir: str,
        base_checkpoint: str | None = None,
        channel_key: str | None = "ha_flux",
        variant: ResidualVariant = "pixel",
        hidden_channels: int = 32,
        **kwargs,
    ) -> ResidualMapGenerator:
        base_model, _base_cfg, channel_index = load_frozen_base_map_generator(
            base_run_dir=base_run_dir,
            base_checkpoint=base_checkpoint,
            channel_key=channel_key,
        )
        return cls(
            config,
            base_model=base_model,
            base_channel_index=channel_index,
            variant=variant,
            hidden_channels=hidden_channels,
            **kwargs,
        )

    def _base_maps(self, batch: dict[str, object]) -> torch.Tensor:
        return frozen_base_predict(
            self.base_model,
            batch,
            channel_index=self.base_channel_index,
        )

    def forward(
        self,
        batch: dict[str, object],
        *,
        epoch: int | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        del epoch
        x = prepare_imaging_input(batch, self.config)
        targets, masks = prepare_targets_and_masks(batch, self.config)
        base_maps = self._base_maps(batch)
        residual_target = targets - base_maps
        # Zero invalid residual pixels so they cannot leak into the network target.
        residual_target = residual_target * masks

        residual_in = torch.cat([x, base_maps], dim=1)

        if self.variant == "gaussian":
            assert isinstance(self.residual_net, GaussianPixelResidualRegressor)
            mu_r, log_var = self.residual_net(residual_in)
            log_var = log_var.clamp(self.min_log_var, self.max_log_var)
            final_maps = base_maps + mu_r
            nll = masked_gaussian_nll(
                mu_r.float(),
                log_var.float(),
                residual_target.float(),
                masks.float(),
                min_log_var=self.min_log_var,
                max_log_var=self.max_log_var,
            )
            l1_r = masked_l1(mu_r.float(), residual_target.float(), masks.float())
            # Also report final-map fidelity for monitoring.
            final_l1 = masked_l1(final_maps.float(), targets.float(), masks.float())
            loss = nll + self.gaussian_l1_weight * l1_r
            loss_dict = {
                "loss": loss,
                "gaussian_nll": nll,
                "residual_l1": l1_r,
                "final_l1": final_l1,
            }
            sigma_r = torch.exp(0.5 * log_var)
            pred_dict = {
                "maps": final_maps,
                "base_maps": base_maps,
                "residual_target": residual_target,
                "residual_prediction": mu_r,
                "residual_sigma": sigma_r,
                "log_var": log_var,
                "predictive_mean": final_maps,
                "targets": targets,
                "masks": masks,
            }
            return pred_dict, loss_dict

        residual_pred = self.residual_net(residual_in)
        assert isinstance(residual_pred, torch.Tensor)
        final_maps = base_maps + residual_pred
        # Train against residual; also log/compose final-map fidelity metrics.
        residual_l1 = masked_l1(residual_pred.float(), residual_target.float(), masks.float())
        final_loss_dict = compose_map_losses(
            final_maps.float(),
            targets.float(),
            masks.float(),
            losses=self.config.losses,
            loss_weights=self.config.loss_weights,
            loss_params=self.config.loss_params,
            target_keys=self.config.target_keys,
        )
        loss_dict = {
            **final_loss_dict,
            "residual_l1": residual_l1,
            # Primary objective is residual L1; final compose terms are additive extras.
            "loss": residual_l1 + final_loss_dict["loss"],
        }
        pred_dict = {
            "maps": final_maps,
            "base_maps": base_maps,
            "residual_target": residual_target,
            "residual_prediction": residual_pred,
            "targets": targets,
            "masks": masks,
        }
        return pred_dict, loss_dict

    @torch.no_grad()
    def sample_gaussian(
        self,
        batch: dict[str, object],
        *,
        n_samples: int | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.variant != "gaussian":
            raise ValueError("sample_gaussian requires variant='gaussian'")
        pred_dict, _ = self.forward(batch)
        n = int(n_samples or self.n_residual_samples)
        mu_r = pred_dict["residual_prediction"]
        sigma_r = pred_dict["residual_sigma"]
        base = pred_dict["base_maps"]
        masks = pred_dict["masks"]
        eps = torch.randn(
            (n, *mu_r.shape),
            device=mu_r.device,
            dtype=mu_r.dtype,
            generator=generator,
        )
        residual_samples = mu_r.unsqueeze(0) + sigma_r.unsqueeze(0) * eps
        residual_samples = residual_samples * masks.unsqueeze(0)
        map_samples = base.unsqueeze(0) + residual_samples
        predictive_mean = map_samples.mean(dim=0)
        predictive_std = map_samples.std(dim=0, unbiased=False)
        q16 = torch.quantile(map_samples, 0.16, dim=0)
        q84 = torch.quantile(map_samples, 0.84, dim=0)
        return {
            **pred_dict,
            "samples": map_samples,
            "residual_samples": residual_samples,
            "predictive_mean": predictive_mean,
            "predictive_std": predictive_std,
            "percentile_16": q16,
            "percentile_84": q84,
        }

    @torch.no_grad()
    def predict(
        self,
        batch: dict[str, object],
        *,
        epoch: int | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self.eval()
        return self.forward(batch, epoch=epoch)


def assert_base_frozen(model: ResidualMapGenerator) -> None:
    """Acceptance helper: no base parameter receives gradients."""
    for p in model.base_model.parameters():
        if p.requires_grad:
            raise AssertionError("Base model parameter has requires_grad=True")
