"""Acceptance tests for full-map score generator and corrector."""
from __future__ import annotations

import unittest
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from src.models.map_score import CondMapScoreUNet, MapDiffusionSchedule, ScoreNormStats
from src.models.map_score_wrapper import MapScoreModel


@dataclass
class _Cfg:
    use_sdss: bool = True
    use_legacy: bool = False
    use_spectrum: bool = False
    use_hr_cross_attn: bool = False
    input_norm_mode: str = "none"
    imaging_asinh_scales: list[float] | None = None
    imaging_clamp_min: float | None = None
    imaging_clamp_max: float | None = None
    target_keys: tuple[str, ...] = ("ha_flux",)
    n_target_maps: int = 1
    n_sdss_bands: int = 5
    n_legacy_bands: int = 4
    footprint_mode: str = "loss_only"
    spectrum_n_wave: int = 64
    spectrum_pooling: str = "avg"
    spectrum_use_wavelength: bool = False
    spectrum_use_ivar: bool = False
    spectrum_asinh_scale_fake: float | None = None
    spectrum_asinh_scale_real: float | None = None
    spectrum_wave_min: float = 3622.0
    spectrum_wave_max: float = 10354.0
    cond_dim: int = 32
    losses: list[str] = field(default_factory=lambda: ["mse"])
    loss_weights: list[float] = field(default_factory=lambda: [1.0])
    loss_params: dict = field(default_factory=dict)

    def imaging_input_channels(self) -> int:
        return 5

    def spectrum_input_channels(self) -> int:
        return 1

    def uses_footprint_in_model(self) -> bool:
        return False


class _TinyBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _Cfg()
        self.bias = nn.Parameter(torch.tensor([0.4]))

    def forward(self, batch, *, epoch=None):
        del epoch
        x = batch["inputs"]["sdss_imaging"].float()
        b, _, h, w = x.shape
        maps = self.bias.view(1, 1, 1, 1).expand(b, 1, h, w)
        targets = batch["targets"]["ha_flux"].unsqueeze(1).float()
        masks = batch["target_loss_masks"]["ha_flux"].unsqueeze(1).float()
        return {"maps": maps, "targets": targets, "masks": masks}, {"loss": maps.new_tensor(0.0)}


def _batch(b: int = 2, h: int = 16, w: int = 16) -> dict:
    imaging = torch.randn(b, 5, h, w)
    target = torch.rand(b, h, w) * 0.8 + 0.1
    label = torch.ones(b, h, w)
    label[:, 0, 0] = 0.0  # one missing label pixel
    footprint = torch.ones(b, h, w)
    footprint[:, -1, -1] = 0.0
    return {
        "plateifu": [f"1-{i}" for i in range(b)],
        "inputs": {"sdss_imaging": imaging},
        "targets": {"ha_flux": target},
        "target_loss_masks": {"ha_flux": label},
        "footprint_mask": footprint,
    }


class MapScoreNetworkTests(unittest.TestCase):
    def test_unet_forward_shape(self) -> None:
        net = CondMapScoreUNet(
            cond_channels=7,
            map_channels=1,
            base_channels=16,
            channel_mults=(1, 2),
            num_res_blocks=1,
            use_spectrum=False,
            emb_dim=64,
        )
        y = torch.randn(2, 1, 16, 16)
        cond = torch.randn(2, 7, 16, 16)
        t = torch.randint(0, 10, (2,))
        out = net(y, t, cond)
        self.assertEqual(tuple(out.shape), (2, 1, 16, 16))


class GeneratorCorrectorContractTests(unittest.TestCase):
    def test_generator_never_receives_base_as_cond(self) -> None:
        cfg = _Cfg()
        norm = ScoreNormStats(mean=0.5, std=0.2)
        base = _TinyBase()
        model = MapScoreModel(
            cfg,  # type: ignore[arg-type]
            mode="generator",
            score_norm=norm,
            base_model=base,  # type: ignore[arg-type]
            base_channel_index=0,
            diffusion_steps=20,
            ddim_steps=4,
            n_samples=2,
            base_channels=16,
            channel_mults=(1, 2),
            num_res_blocks=1,
            receive_base_as_cond=False,
        )
        model.assert_generator_no_base_cond()
        # Cond channels: ugriz(5)+fp+label = 7, no base
        self.assertEqual(model.denoiser.cond_channels, 7)
        batch = _batch()
        pred, loss = model(batch)
        self.assertTrue(torch.isfinite(loss["loss"]))
        self.assertIn("footprint_mask", pred)
        self.assertIn("label_mask", pred)

    def test_corrector_receives_base_cond(self) -> None:
        cfg = _Cfg()
        norm = ScoreNormStats(mean=0.5, std=0.2)
        base = _TinyBase()
        model = MapScoreModel(
            cfg,  # type: ignore[arg-type]
            mode="corrector",
            score_norm=norm,
            base_model=base,  # type: ignore[arg-type]
            base_channel_index=0,
            diffusion_steps=20,
            ddim_steps=4,
            n_samples=2,
            base_channels=16,
            channel_mults=(1, 2),
            num_res_blocks=1,
        )
        model.assert_corrector_has_base_cond()
        self.assertEqual(model.denoiser.cond_channels, 8)  # + base

    def test_loss_only_on_label_mask(self) -> None:
        cfg = _Cfg()
        norm = ScoreNormStats(mean=0.5, std=0.2)
        base = _TinyBase()
        model = MapScoreModel(
            cfg,  # type: ignore[arg-type]
            mode="generator",
            score_norm=norm,
            base_model=base,  # type: ignore[arg-type]
            diffusion_steps=20,
            base_channels=16,
            channel_mults=(1, 2),
            num_res_blocks=1,
            receive_base_as_cond=False,
        )
        batch = _batch(b=1)
        # Poison a labeled-invalid pixel's target fill path via extreme imaging — loss uses label_mask.
        torch.manual_seed(0)
        _, loss_a = model(batch)
        batch["targets"]["ha_flux"][0, 0, 0] = 99.0  # invalid label pixel
        torch.manual_seed(0)
        _, loss_b = model(batch)
        self.assertAlmostEqual(float(loss_a["loss"]), float(loss_b["loss"]), places=5)

    def test_missing_targets_not_treated_as_zero_when_base_fills(self) -> None:
        cfg = _Cfg()
        norm = ScoreNormStats(mean=0.0, std=1.0)
        base = _TinyBase()
        model = MapScoreModel(
            cfg,  # type: ignore[arg-type]
            mode="generator",
            score_norm=norm,
            base_model=base,  # type: ignore[arg-type]
            diffusion_steps=20,
            base_channels=16,
            channel_mults=(1, 2),
            num_res_blocks=1,
            receive_base_as_cond=False,
        )
        batch = _batch(b=1)
        y, label, fp, base_ha = model._prepare_clean_map(batch)
        # Invalid label pixel should equal normalised base, not 0.
        self.assertIsNotNone(base_ha)
        expected = model.score_norm.normalize(base_ha)[0, 0, 0, 0] * fp[0, 0, 0, 0]
        self.assertTrue(torch.allclose(y[0, 0, 0, 0], expected))
        self.assertEqual(float(label[0, 0, 0, 0]), 0.0)

    def test_frozen_base_no_grad(self) -> None:
        cfg = _Cfg()
        norm = ScoreNormStats(mean=0.5, std=0.2)
        base = _TinyBase()
        model = MapScoreModel(
            cfg,  # type: ignore[arg-type]
            mode="corrector",
            score_norm=norm,
            base_model=base,  # type: ignore[arg-type]
            diffusion_steps=20,
            base_channels=16,
            channel_mults=(1, 2),
            num_res_blocks=1,
        )
        batch = _batch()
        _, loss = model(batch)
        loss["loss"].backward()
        for p in model.base_model.parameters():
            self.assertFalse(p.requires_grad)
            self.assertIsNone(p.grad)


class SamplingTests(unittest.TestCase):
    def test_generator_starts_from_noise_corrector_from_base(self) -> None:
        cfg = _Cfg()
        norm = ScoreNormStats(mean=0.5, std=0.2)
        base = _TinyBase()
        gen = MapScoreModel(
            cfg,  # type: ignore[arg-type]
            mode="generator",
            score_norm=norm,
            base_model=base,  # type: ignore[arg-type]
            diffusion_steps=20,
            ddim_steps=4,
            n_samples=2,
            base_channels=16,
            channel_mults=(1, 2),
            num_res_blocks=1,
            receive_base_as_cond=False,
        )
        corr = MapScoreModel(
            cfg,  # type: ignore[arg-type]
            mode="corrector",
            score_norm=norm,
            base_model=base,  # type: ignore[arg-type]
            diffusion_steps=20,
            ddim_steps=4,
            n_samples=2,
            base_channels=16,
            channel_mults=(1, 2),
            num_res_blocks=1,
            t_start_frac=0.5,
        )
        batch = _batch(b=1)
        out_g = gen.sample(batch, n_samples=2, ddim_steps=4, seed=0, use_ema=False)
        out_c = corr.sample(batch, n_samples=2, ddim_steps=4, seed=0, use_ema=False)
        self.assertEqual(tuple(out_g["samples"].shape), (2, 1, 1, 16, 16))
        self.assertEqual(tuple(out_c["samples"].shape), (2, 1, 1, 16, 16))
        self.assertIn("base_maps", out_c)
        # Sampling domain is footprint: corner outside footprint is ~0
        self.assertAlmostEqual(float(out_g["samples"][0, 0, 0, -1, -1]), 0.0, places=5)

    def test_fixed_seed_reproducible_and_seeds_differ(self) -> None:
        cfg = _Cfg()
        norm = ScoreNormStats(mean=0.5, std=0.2)
        base = _TinyBase()
        model = MapScoreModel(
            cfg,  # type: ignore[arg-type]
            mode="generator",
            score_norm=norm,
            base_model=base,  # type: ignore[arg-type]
            diffusion_steps=20,
            ddim_steps=4,
            base_channels=16,
            channel_mults=(1, 2),
            num_res_blocks=1,
            receive_base_as_cond=False,
        )
        batch = _batch(b=1)
        a = model.sample(batch, n_samples=2, ddim_steps=4, seed=11, use_ema=False)
        b = model.sample(batch, n_samples=2, ddim_steps=4, seed=11, use_ema=False)
        c = model.sample(batch, n_samples=2, ddim_steps=4, seed=99, use_ema=False)
        self.assertTrue(torch.allclose(a["samples"], b["samples"]))
        self.assertFalse(torch.allclose(a["samples"], c["samples"]))

    def test_eval_uses_sample_not_training_forward_contract(self) -> None:
        # Score models advertise sample-based eval.
        cfg = _Cfg()
        norm = ScoreNormStats(mean=0.5, std=0.2)
        model = MapScoreModel(
            cfg,  # type: ignore[arg-type]
            mode="generator",
            score_norm=norm,
            base_model=_TinyBase(),  # type: ignore[arg-type]
            diffusion_steps=10,
            base_channels=16,
            channel_mults=(1, 2),
            num_res_blocks=1,
            receive_base_as_cond=False,
        )
        self.assertTrue(model.uses_score_sample_eval)
        self.assertFalse(model.uses_batch_forward_eval)
        self.assertTrue(hasattr(model, "sample"))


class ScheduleTests(unittest.TestCase):
    def test_sdedit_start_timestep(self) -> None:
        sched = MapDiffusionSchedule(n_steps=1000)
        self.assertEqual(sched.t_from_fraction(1.0), 999)
        self.assertGreater(sched.t_from_fraction(0.5), sched.t_from_fraction(0.1))


if __name__ == "__main__":
    unittest.main()
