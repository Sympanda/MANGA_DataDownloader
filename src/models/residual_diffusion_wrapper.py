"""Trainer-compatible wrapper for conditional residual diffusion."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.base_loader import frozen_base_predict, load_frozen_base_map_generator
from src.models.config import ModelConfig
from src.models.input_prep import prepare_imaging_input, prepare_targets_and_masks
from src.models.losses import masked_l1, masked_mse
from src.models.residual_diffusion import CondResidualDiffusionUNet, ResidualDiffusionSchedule
from src.models.wrapper import MapGenerator


class ResidualDiffusionMapGenerator(nn.Module):
    """
    Frozen base + conditional residual diffusion.

    Diffusion target is R = Y - Y_base (scaled space), not the full MaNGA map.
    """

    uses_batch_forward_eval = True

    def __init__(
        self,
        config: ModelConfig,
        *,
        base_model: MapGenerator,
        base_channel_index: int | None,
        diffusion_steps: int = 1000,
        ddim_steps: int = 50,
        n_samples: int = 32,
        base_channels: int = 32,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        schedule: str = "linear",
        use_footprint_cond: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.base_channel_index = (
            None if base_channel_index is None else int(base_channel_index)
        )
        self.ddim_steps = int(ddim_steps)
        self.n_samples = int(n_samples)
        self.use_footprint_cond = bool(use_footprint_cond)

        self.base_model = base_model
        self.base_model.eval()
        self.base_model.requires_grad_(False)

        # Cond: ugriz + base maps + 1 valid-region mask (+ optional footprint)
        cond_ch = config.imaging_input_channels() + config.n_target_maps + 1
        if use_footprint_cond:
            cond_ch += 1
        self.cond_channels = cond_ch
        self.denoiser = CondResidualDiffusionUNet(
            cond_channels=cond_ch,
            residual_channels=config.n_target_maps,
            base_channels=base_channels,
            channel_mults=channel_mults,
        )
        self.schedule = ResidualDiffusionSchedule(n_steps=diffusion_steps, schedule=schedule)

    @classmethod
    def from_base_run(
        cls,
        config: ModelConfig,
        *,
        base_run_dir: str,
        base_checkpoint: str | None = None,
        channel_key: str | None = "ha_flux",
        **kwargs,
    ) -> ResidualDiffusionMapGenerator:
        base_model, _cfg, channel_index = load_frozen_base_map_generator(
            base_run_dir=base_run_dir,
            base_checkpoint=base_checkpoint,
            channel_key=channel_key,
        )
        return cls(
            config,
            base_model=base_model,
            base_channel_index=channel_index,
            **kwargs,
        )

    def _move_schedule(self, device: torch.device) -> None:
        first = next(iter(self.schedule.register.values()))
        if first.device != device:
            self.schedule.to(device)

    def _conditioning(
        self,
        batch: dict[str, object],
        *,
        base_maps: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        x = prepare_imaging_input(batch, self.config)
        # Collapse per-target masks to one valid-region channel (matches cond_ch sizing).
        if masks.ndim == 4 and masks.shape[1] > 1:
            mask_cond = (masks > 0).any(dim=1, keepdim=True).to(dtype=x.dtype)
        elif masks.ndim == 3:
            mask_cond = (masks > 0).unsqueeze(1).to(dtype=x.dtype)
        else:
            mask_cond = masks.to(dtype=x.dtype)
        parts = [x, base_maps, mask_cond]
        if self.use_footprint_cond:
            fp = batch["footprint_mask"].float()  # type: ignore[index]
            if fp.ndim == x.ndim - 1:
                fp = fp.unsqueeze(1)
            parts.append(fp)
        return torch.cat(parts, dim=1)

    def forward(
        self,
        batch: dict[str, object],
        *,
        epoch: int | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        del epoch
        targets, masks = prepare_targets_and_masks(batch, self.config)
        base_maps = frozen_base_predict(
            self.base_model, batch, channel_index=self.base_channel_index
        )
        residual_target = (targets - base_maps) * masks
        cond = self._conditioning(batch, base_maps=base_maps, masks=masks)
        self._move_schedule(residual_target.device)

        b = residual_target.shape[0]
        t = torch.randint(0, self.schedule.n_steps, (b,), device=residual_target.device)
        noisy, noise = self.schedule.q_sample(residual_target, t)
        noisy = noisy * masks
        noise = noise * masks
        pred_noise = self.denoiser(noisy, t, cond)
        # Masked MSE on noise (invalid pixels forced to 0 on both sides).
        diff_loss = masked_mse(pred_noise.float(), noise.float(), masks.float())

        # Point estimate for Trainer metrics: one-step x0 from sampled t (noisy train view).
        # For eval, prefer sample().
        a = self.schedule.register["sqrt_alphas_cumprod"][t].view(-1, 1, 1, 1)
        a_om = self.schedule.register["sqrt_one_minus_alphas_cumprod"][t].view(-1, 1, 1, 1)
        x0_pred = ((noisy - a_om * pred_noise) / a.clamp_min(1e-8)) * masks
        final_maps = base_maps + x0_pred
        final_l1 = masked_l1(final_maps.float(), targets.float(), masks.float())

        loss_dict = {
            "loss": diff_loss,
            "diffusion_mse": diff_loss,
            "final_l1": final_l1,
        }
        pred_dict = {
            "maps": final_maps,
            "base_maps": base_maps,
            "residual_target": residual_target,
            "residual_prediction": x0_pred,
            "targets": targets,
            "masks": masks,
        }
        return pred_dict, loss_dict

    @torch.no_grad()
    def sample(
        self,
        batch: dict[str, object],
        *,
        n_samples: int | None = None,
        ddim_steps: int | None = None,
        seed: int | None = None,
    ) -> dict[str, torch.Tensor]:
        targets, masks = prepare_targets_and_masks(batch, self.config)
        base_maps = frozen_base_predict(
            self.base_model, batch, channel_index=self.base_channel_index
        )
        residual_target = (targets - base_maps) * masks
        cond = self._conditioning(batch, base_maps=base_maps, masks=masks)
        self._move_schedule(cond.device)

        n = int(n_samples or self.n_samples)
        steps = int(ddim_steps or self.ddim_steps)
        samples = []
        for k in range(n):
            gen = None
            if seed is not None:
                gen = torch.Generator(device=cond.device)
                gen.manual_seed(int(seed) + k)
            r_k = self.schedule.ddim_sample(
                self.denoiser,
                cond,
                steps=steps,
                eta=0.0,
                generator=gen,
                mask=masks,
            )
            samples.append(base_maps + r_k)
        map_samples = torch.stack(samples, dim=0)
        residual_samples = map_samples - base_maps.unsqueeze(0)
        predictive_mean = map_samples.mean(dim=0)
        predictive_median = map_samples.median(dim=0).values
        predictive_std = map_samples.std(dim=0, unbiased=False)
        q16 = torch.quantile(map_samples, 0.16, dim=0)
        q84 = torch.quantile(map_samples, 0.84, dim=0)
        return {
            "maps": predictive_mean,
            "base_maps": base_maps,
            "residual_target": residual_target,
            "residual_prediction": predictive_mean - base_maps,
            "targets": targets,
            "masks": masks,
            "samples": map_samples,
            "residual_samples": residual_samples,
            "predictive_mean": predictive_mean,
            "predictive_median": predictive_median,
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
