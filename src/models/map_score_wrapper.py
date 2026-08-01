"""Full-map score corrector and direct score generator wrappers."""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from src.models.base_loader import frozen_base_predict, load_frozen_base_map_generator
from src.models.config import ModelConfig
from src.models.input_prep import (
    prepare_imaging_input,
    prepare_spectrum_input,
    prepare_targets_and_masks,
)
from src.models.losses import masked_mse
from src.models.map_score import (
    CondMapScoreUNet,
    EMA,
    MapDiffusionSchedule,
    ScoreNormStats,
)
from src.models.wrapper import MapGenerator

ScoreMode = Literal["generator", "corrector"]


def _as_bchw_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 3:
        return mask.unsqueeze(1)
    return mask


class MapScoreModel(nn.Module):
    """
    Full-map epsilon-prediction diffusion model.

    Modes
    -----
    generator:
        Conditions on imaging + masks + spectrum. Never uses UNet map as network input.
        Frozen UNet (if provided) is only used to fill missing label pixels for a finite
        training tensor; those pixels are excluded from the score loss.
    corrector:
        Also conditions on the frozen UNet mean Hα prediction.
        Sampling starts from a noised base map (SDEdit).
    """

    uses_batch_forward_eval = False  # must use sample()-based score evaluator
    uses_score_sample_eval = True

    def __init__(
        self,
        config: ModelConfig,
        *,
        mode: ScoreMode,
        score_norm: ScoreNormStats,
        base_model: MapGenerator | None = None,
        base_channel_index: int | None = 0,
        diffusion_steps: int = 1000,
        ddim_steps: int = 50,
        n_samples: int = 16,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        schedule: str = "linear",
        ema_decay: float = 0.9999,
        t_start_frac: float = 0.25,
        receive_base_as_cond: bool | None = None,
    ) -> None:
        super().__init__()
        if config.n_target_maps != 1:
            raise ValueError("MapScoreModel currently supports a single target channel (Hα).")
        self.config = config
        self.mode: ScoreMode = mode
        self.register_buffer("_score_mean", torch.tensor(float(score_norm.mean)))
        self.register_buffer("_score_std", torch.tensor(float(max(score_norm.std, 1e-6))))
        self.base_channel_index = base_channel_index
        self.ddim_steps = int(ddim_steps)
        self.n_samples = int(n_samples)
        self.t_start_frac = float(t_start_frac)
        self.receive_base_as_cond = (
            (mode == "corrector") if receive_base_as_cond is None else bool(receive_base_as_cond)
        )

        self.base_model = base_model
        if self.base_model is not None:
            self.base_model.eval()
            self.base_model.requires_grad_(False)

        if self.receive_base_as_cond and self.base_model is None:
            raise ValueError("Corrector mode requires a frozen base_model")

        # Spatial cond: ugriz + footprint + label_mask [+ optional base map]
        cond_ch = config.imaging_input_channels() + 2
        if self.receive_base_as_cond:
            cond_ch += 1

        self.denoiser = CondMapScoreUNet(
            cond_channels=cond_ch,
            map_channels=1,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            use_spectrum=bool(config.use_spectrum),
            spectrum_n_wave=config.spectrum_n_wave,
            spectrum_in_channels=config.spectrum_input_channels() if config.use_spectrum else 1,
            spectrum_pooling=config.spectrum_pooling,
            cond_dim=config.cond_dim,
        )
        self.schedule = MapDiffusionSchedule(n_steps=diffusion_steps, schedule=schedule)
        self.ema = EMA(self.denoiser, decay=ema_decay)

    @property
    def score_norm(self) -> ScoreNormStats:
        return ScoreNormStats(
            mean=float(self._score_mean.item()),
            std=float(self._score_std.item()),
        )

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        for k, v in self.ema.state_dict().items():
            sd[f"_ema.{k}"] = v
        return sd

    def load_state_dict(self, state_dict, strict: bool = True):
        ema_sd = {}
        clean = {}
        for k, v in state_dict.items():
            if k.startswith("_ema."):
                ema_sd[k[len("_ema.") :]] = v
            else:
                clean[k] = v
        missing = super().load_state_dict(clean, strict=False)
        if ema_sd:
            self.ema.load_state_dict(ema_sd)
        if strict and missing.missing_keys:
            # Allow missing EMA keys on older checkpoints.
            real_missing = [k for k in missing.missing_keys if not k.startswith("_ema")]
            if real_missing:
                raise RuntimeError(f"Missing keys: {real_missing}")
        return missing

    @classmethod
    def build(
        cls,
        config: ModelConfig,
        *,
        mode: ScoreMode,
        score_norm: ScoreNormStats,
        base_run_dir: str | None = None,
        base_checkpoint: str | None = "best.pt",
        channel_key: str = "ha_flux",
        **kwargs,
    ) -> MapScoreModel:
        base_model = None
        channel_index: int | None = 0
        need_base = mode == "corrector" or base_run_dir is not None
        if need_base:
            if not base_run_dir:
                raise ValueError("base_run_dir is required for corrector / target filling")
            base_model, _cfg, channel_index = load_frozen_base_map_generator(
                base_run_dir=base_run_dir,
                base_checkpoint=base_checkpoint,
                channel_key=channel_key,
            )
        return cls(
            config,
            mode=mode,
            score_norm=score_norm,
            base_model=base_model,
            base_channel_index=channel_index,
            **kwargs,
        )

    def _move_schedule(self, device: torch.device) -> None:
        first = next(iter(self.schedule.register.values()))
        if first.device != device:
            self.schedule.to(device)

    def _base_ha(self, batch: dict[str, object]) -> torch.Tensor:
        if self.base_model is None:
            raise RuntimeError("No frozen base model available")
        return frozen_base_predict(
            self.base_model, batch, channel_index=self.base_channel_index
        )

    def _prepare_clean_map(
        self,
        batch: dict[str, object],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Returns
        -------
        y_score : score-normalised clean map (finite everywhere inside footprint)
        label_mask : where score loss is computed
        footprint : generation domain
        base_ha : frozen UNet Hα in scaled target space (or None)
        """
        targets, label_mask = prepare_targets_and_masks(batch, self.config)
        footprint = _as_bchw_mask(batch["footprint_mask"].float())  # type: ignore[index]
        label_mask = _as_bchw_mask(label_mask)

        base_ha = None
        if self.base_model is not None:
            base_ha = self._base_ha(batch)

        # Fill missing label pixels with base (never treat as physical zero).
        if base_ha is not None:
            fill = torch.where(label_mask > 0, targets, base_ha)
        else:
            # No base available: keep target where labeled; outside label but in
            # footprint leave as 0 only after score-norm of labeled stats — still
            # excluded from loss via label_mask. Prefer providing a base for ≥99%.
            fill = torch.where(label_mask > 0, targets, targets.new_zeros(targets.shape))

        # Outside footprint: keep 0 in score space after normalisation of fill.
        y_score = self.score_norm.normalize(fill) * footprint
        return y_score, label_mask, footprint, base_ha

    def _spatial_cond(
        self,
        batch: dict[str, object],
        *,
        footprint: torch.Tensor,
        label_mask: torch.Tensor,
        base_ha: torch.Tensor | None,
    ) -> torch.Tensor:
        x = prepare_imaging_input(batch, self.config)
        parts = [x, footprint, label_mask]
        if self.receive_base_as_cond:
            if base_ha is None:
                raise RuntimeError("Corrector conditioning requires base_ha")
            parts.append(self.score_norm.normalize(base_ha) * footprint)
        return torch.cat(parts, dim=1)

    def forward(
        self,
        batch: dict[str, object],
        *,
        epoch: int | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        del epoch
        y0, label_mask, footprint, base_ha = self._prepare_clean_map(batch)
        cond = self._spatial_cond(
            batch, footprint=footprint, label_mask=label_mask, base_ha=base_ha
        )
        spec = prepare_spectrum_input(batch, self.config)
        self._move_schedule(y0.device)

        b = y0.shape[0]
        t = torch.randint(0, self.schedule.n_steps, (b,), device=y0.device)
        noisy, noise = self.schedule.q_sample(y0, t)
        # Domain: only footprint carries signal; outside stays 0.
        noisy = noisy * footprint
        noise = noise * footprint

        pred_noise = self.denoiser(noisy, t, cond, spectrum=spec)
        # Score / epsilon loss only on reliable labels.
        loss = masked_mse(pred_noise.float(), noise.float(), label_mask.float())

        # Training-view x0 estimate (not a reverse-process sample).
        a = self.schedule.register["sqrt_alphas_cumprod"][t].view(-1, 1, 1, 1)
        a_om = self.schedule.register["sqrt_one_minus_alphas_cumprod"][t].view(-1, 1, 1, 1)
        x0_pred_score = ((noisy - a_om * pred_noise) / a.clamp_min(1e-8)) * footprint
        x0_pred = self.score_norm.denormalize(x0_pred_score)

        targets, _ = prepare_targets_and_masks(batch, self.config)
        pred_dict = {
            "maps": x0_pred,
            "targets": targets,
            "masks": label_mask,
            "footprint_mask": footprint,
            "label_mask": label_mask,
            "y_score": y0,
        }
        if base_ha is not None:
            pred_dict["base_maps"] = base_ha
        loss_dict = {"loss": loss, "diffusion_mse": loss}
        return pred_dict, loss_dict

    def update_ema(self) -> None:
        # Keep EMA shadows on the same device as the live denoiser.
        try:
            device = next(self.denoiser.parameters()).device
            self.ema.to(device)
        except StopIteration:
            pass
        self.ema.update(self.denoiser)

    @torch.no_grad()
    def sample(
        self,
        batch: dict[str, object],
        *,
        n_samples: int | None = None,
        ddim_steps: int | None = None,
        eta: float = 0.0,
        t_start_frac: float | None = None,
        seed: int | None = None,
        use_ema: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Run reverse-process sampling and return maps in scaled target space."""
        y0, label_mask, footprint, base_ha = self._prepare_clean_map(batch)
        cond = self._spatial_cond(
            batch, footprint=footprint, label_mask=label_mask, base_ha=base_ha
        )
        spec = prepare_spectrum_input(batch, self.config)
        self._move_schedule(cond.device)

        n = int(n_samples or self.n_samples)
        steps = int(ddim_steps or self.ddim_steps)
        # None → mode default: corrector uses self.t_start_frac; generator = full noise.
        if t_start_frac is None:
            frac = None if self.mode == "generator" else float(self.t_start_frac)
        else:
            frac = float(t_start_frac)

        backup = None
        if use_ema and self.ema.shadow:
            backup = {k: v.detach().clone() for k, v in self.denoiser.state_dict().items()}
            self.ema.copy_to(self.denoiser)

        samples_score = []
        try:
            for k in range(n):
                gen = None
                if seed is not None:
                    gen = torch.Generator(device=cond.device)
                    gen.manual_seed(int(seed) + k)
                if self.mode == "corrector":
                    if base_ha is None:
                        raise RuntimeError("Corrector sampling requires base_ha")
                    x_init = self.score_norm.normalize(base_ha) * footprint
                    t_start = self.schedule.t_from_fraction(float(frac if frac is not None else self.t_start_frac))
                    y_k = self.schedule.ddim_sample(
                        self.denoiser,
                        cond,
                        steps=steps,
                        eta=eta,
                        generator=gen,
                        footprint_mask=footprint,
                        x_init=x_init,
                        t_start=t_start,
                        spectrum=spec,
                    )
                else:
                    # Direct generator:
                    #   frac is None or >=1 → pure noise (full map generation)
                    #   0 < frac < 1 → noise the clean map to t and denoise
                    if frac is None or frac >= 1.0 - 1e-12:
                        y_k = self.schedule.ddim_sample(
                            self.denoiser,
                            cond,
                            steps=steps,
                            eta=eta,
                            generator=gen,
                            footprint_mask=footprint,
                            x_init=None,
                            t_start=None,
                            spectrum=spec,
                        )
                    else:
                        t_start = self.schedule.t_from_fraction(frac)
                        y_k = self.schedule.ddim_sample(
                            self.denoiser,
                            cond,
                            steps=steps,
                            eta=eta,
                            generator=gen,
                            footprint_mask=footprint,
                            x_init=y0,
                            t_start=t_start,
                            spectrum=spec,
                        )
                samples_score.append(y_k)
        finally:
            if backup is not None:
                self.denoiser.load_state_dict(backup, strict=False)

        stack_score = torch.stack(samples_score, dim=0)
        stack = self.score_norm.denormalize(stack_score)
        # Keep samples on footprint domain.
        stack = stack * footprint.unsqueeze(0)

        targets, _ = prepare_targets_and_masks(batch, self.config)
        mean = stack.mean(dim=0)
        median = stack.median(dim=0).values
        std = stack.std(dim=0, unbiased=False)
        q16 = torch.quantile(stack, 0.16, dim=0)
        q84 = torch.quantile(stack, 0.84, dim=0)

        out: dict[str, torch.Tensor] = {
            "maps": mean,
            "samples": stack,
            "predictive_mean": mean,
            "predictive_median": median,
            "predictive_std": std,
            "percentile_16": q16,
            "percentile_84": q84,
            "targets": targets,
            "masks": label_mask,
            "label_mask": label_mask,
            "footprint_mask": footprint,
        }
        if base_ha is not None:
            out["base_maps"] = base_ha
            out["residual_target"] = targets - base_ha
            out["residual_prediction"] = mean - base_ha
        return out

    def assert_generator_no_base_cond(self) -> None:
        if self.mode == "generator" and self.receive_base_as_cond:
            raise AssertionError("Direct generator must not receive frozen UNet as conditioning")

    def assert_corrector_has_base_cond(self) -> None:
        if self.mode == "corrector" and not self.receive_base_as_cond:
            raise AssertionError("Score corrector must receive frozen UNet mean as conditioning")
