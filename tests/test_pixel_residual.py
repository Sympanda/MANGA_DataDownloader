"""Acceptance tests for pixel-SED baselines and frozen-base residual models."""
from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any
from unittest import mock

import torch
import torch.nn as nn

from src.data.augmentation import AugmentConfig, augment_spatial_sample
from src.data.splits import read_split_csv
from src.models.input_prep import prepare_imaging_input, prepare_targets_and_masks
from src.models.pixel_sed import PixelSEDRegressor
from src.models.pixel_wrapper import PixelMapGenerator
from src.models.residual_diffusion import CondResidualDiffusionUNet, ResidualDiffusionSchedule
from src.models.residual_diffusion_wrapper import ResidualDiffusionMapGenerator
from src.models.residual_models import (
    GaussianPixelResidualRegressor,
    LocalResidualCNN,
    PixelResidualRegressor,
)
from src.models.residual_wrapper import ResidualMapGenerator, assert_base_frozen


@dataclass
class _PrepCfg:
    use_sdss: bool = True
    use_legacy: bool = False
    input_norm_mode: str = "none"
    imaging_asinh_scales: list[float] | None = None
    imaging_clamp_min: float | None = None
    imaging_clamp_max: float | None = None
    target_keys: tuple[str, ...] = ("ha_flux",)
    footprint_mode: str = "loss_only"
    use_spectrum: bool = False
    use_hr_cross_attn: bool = False
    hr_asinh_scales: list[float] | None = None
    n_sdss_bands: int = 5
    n_legacy_bands: int = 4
    n_target_maps: int = 1
    imaging_resolution: str = "aligned"
    target_spatial_size: int = 76
    losses: list[str] | None = None
    loss_weights: list[float] | None = None
    loss_params: dict | None = None

    def __post_init__(self) -> None:
        if self.losses is None:
            self.losses = ["l1"]
        if self.loss_weights is None:
            self.loss_weights = [1.0]
        if self.loss_params is None:
            self.loss_params = {}

    def imaging_input_channels(self) -> int:
        return (self.n_sdss_bands if self.use_sdss else 0) + (
            self.n_legacy_bands if self.use_legacy else 0
        )

    def uses_footprint_in_model(self) -> bool:
        return False


class _TinyBase(nn.Module):
    """Stand-in MapGenerator: maps = mean(imaging) broadcast + bias."""

    def __init__(self) -> None:
        super().__init__()
        self.config = _PrepCfg()
        self.bias = nn.Parameter(torch.tensor([0.1]))

    def forward(self, batch, *, epoch=None):
        del epoch
        x = prepare_imaging_input(batch, self.config)
        targets, masks = prepare_targets_and_masks(batch, self.config)
        base = x.mean(dim=1, keepdim=True) * 0.0 + self.bias.view(1, 1, 1, 1)
        base = base.expand(-1, 1, x.shape[-2], x.shape[-1])
        return {"maps": base, "targets": targets, "masks": masks}, {
            "loss": base.new_tensor(0.0)
        }


def _synthetic_batch(b: int = 2, h: int = 76, w: int = 76) -> dict[str, Any]:
    imaging = torch.randn(b, 5, h, w)
    target = torch.rand(b, h, w)
    mask = torch.zeros(b, h, w)
    mask[:, 10:60, 10:60] = 1.0
    footprint = mask.clone()
    return {
        "plateifu": [f"1-{i}" for i in range(b)],
        "inputs": {"sdss_imaging": imaging},
        "targets": {"ha_flux": target},
        "target_loss_masks": {"ha_flux": mask},
        "footprint_mask": footprint,
    }


class PixelSEDTests(unittest.TestCase):
    def test_only_1x1_convs_linear_and_mlp(self) -> None:
        for variant in ("linear", "mlp"):
            net = PixelSEDRegressor(in_channels=5, out_channels=1, variant=variant)
            for m in net.modules():
                if isinstance(m, nn.Conv2d):
                    self.assertEqual(m.kernel_size, (1, 1))
                    self.assertEqual(m.stride, (1, 1))

    def test_output_shape_ha(self) -> None:
        cfg = _PrepCfg()
        model = PixelMapGenerator(cfg, variant="mlp")  # type: ignore[arg-type]
        batch = _synthetic_batch()
        pred, loss = model(batch)
        self.assertEqual(tuple(pred["maps"].shape), (2, 1, 76, 76))
        self.assertTrue(torch.isfinite(loss["loss"]))

    def test_invalid_pixels_ignored_in_loss(self) -> None:
        cfg = _PrepCfg()
        model = PixelMapGenerator(cfg, variant="linear")  # type: ignore[arg-type]
        batch = _synthetic_batch(b=1)
        # Poison invalid region in target — must not affect loss if masked.
        batch["targets"]["ha_flux"][0, 0, 0] = 1e6
        batch["target_loss_masks"]["ha_flux"][0, 0, 0] = 0.0
        _, loss_a = model(batch)
        batch["targets"]["ha_flux"][0, 0, 0] = -1e6
        _, loss_b = model(batch)
        self.assertAlmostEqual(float(loss_a["loss"]), float(loss_b["loss"]), places=5)


class SplitLeakageTests(unittest.TestCase):
    def test_split_csv_partitions_are_disjoint(self) -> None:
        path = "manga_sdss_fits/splits/default_split.csv"
        try:
            splits = read_split_csv(path)
        except FileNotFoundError:
            self.skipTest("split CSV not available")
        train, val, test = splits["train"], splits["val"], splits["test"]
        self.assertEqual(len(train & val), 0)
        self.assertEqual(len(train & test), 0)
        self.assertEqual(len(val & test), 0)
        self.assertGreater(len(train), 0)


class FrozenBaseTests(unittest.TestCase):
    def test_base_receives_no_gradients(self) -> None:
        cfg = _PrepCfg()
        base = _TinyBase()
        model = ResidualMapGenerator(
            cfg,  # type: ignore[arg-type]
            base_model=base,  # type: ignore[arg-type]
            base_channel_index=0,
            variant="pixel",
        )
        assert_base_frozen(model)
        batch = _synthetic_batch()
        pred, loss = model(batch)
        loss["loss"].backward()
        for p in model.base_model.parameters():
            self.assertFalse(p.requires_grad)
            self.assertIsNone(p.grad)
        # Residual net should have grads.
        self.assertTrue(any(p.grad is not None for p in model.residual_net.parameters()))

    def test_base_prediction_unchanged_after_residual_step(self) -> None:
        cfg = _PrepCfg()
        base = _TinyBase()
        model = ResidualMapGenerator(
            cfg,  # type: ignore[arg-type]
            base_model=base,  # type: ignore[arg-type]
            base_channel_index=0,
            variant="pixel",
        )
        batch = _synthetic_batch()
        with torch.inference_mode():
            before = model._base_maps(batch).clone()
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            _, loss = model(batch)
            loss["loss"].backward()
            opt.step()
        with torch.inference_mode():
            after = model._base_maps(batch)
        self.assertTrue(torch.allclose(before, after))

    def test_target_equals_base_plus_true_residual(self) -> None:
        cfg = _PrepCfg()
        base = _TinyBase()
        model = ResidualMapGenerator(
            cfg,  # type: ignore[arg-type]
            base_model=base,  # type: ignore[arg-type]
            base_channel_index=0,
            variant="pixel",
        )
        batch = _synthetic_batch()
        pred, _ = model(batch)
        recon = pred["base_maps"] + pred["residual_target"]
        m = pred["masks"] > 0
        self.assertTrue(torch.allclose(recon[m], pred["targets"][m], atol=1e-6))


class AugmentationSyncTests(unittest.TestCase):
    def test_spatial_aug_applies_identically(self) -> None:
        torch.manual_seed(0)
        sdss = torch.randn(5, 16, 16)
        target = {"ha_flux": torch.randn(16, 16)}
        mask = {"ha_flux": torch.ones(16, 16)}
        footprint = torch.ones(16, 16)
        cfg = AugmentConfig(enabled=True, hflip=True, vflip=True, rot90=True, p=1.0)
        # Force deterministic transforms by patching sampler.
        with mock.patch(
            "src.data.augmentation._sample_spatial_transform",
            return_value=(1, True, False),
        ):
            s2, _, _, fp2, t2, m2 = augment_spatial_sample(
                sdss=sdss.clone(),
                footprint=footprint.clone(),
                targets={k: v.clone() for k, v in target.items()},
                target_masks={k: v.clone() for k, v in mask.items()},
                cfg=cfg,
            )
        # Same rot90+hflip on all.
        expect_sdss = torch.rot90(torch.flip(sdss, dims=[-1]), k=1, dims=[-2, -1])
        expect_t = torch.rot90(torch.flip(target["ha_flux"], dims=[-1]), k=1, dims=[-2, -1])
        self.assertTrue(torch.allclose(s2, expect_sdss))
        self.assertTrue(torch.allclose(t2["ha_flux"], expect_t))
        self.assertTrue(torch.allclose(fp2, torch.rot90(torch.flip(footprint, dims=[-1]), 1, [-2, -1])))
        self.assertTrue(torch.allclose(m2["ha_flux"], expect_t * 0 + 1))


class ResidualModelTests(unittest.TestCase):
    def test_pixel_and_cnn_shapes(self) -> None:
        x = torch.randn(2, 6, 76, 76)
        pix = PixelResidualRegressor(in_channels=6, out_channels=1)
        cnn = LocalResidualCNN(in_channels=6, out_channels=1)
        self.assertEqual(tuple(pix(x).shape), (2, 1, 76, 76))
        self.assertEqual(tuple(cnn(x).shape), (2, 1, 76, 76))

    def test_gaussian_samples_shape_and_seed_variation(self) -> None:
        cfg = _PrepCfg()
        base = _TinyBase()
        model = ResidualMapGenerator(
            cfg,  # type: ignore[arg-type]
            base_model=base,  # type: ignore[arg-type]
            base_channel_index=0,
            variant="gaussian",
            n_residual_samples=8,
        )
        batch = _synthetic_batch(b=1)
        g1 = torch.Generator().manual_seed(123)
        g2 = torch.Generator().manual_seed(456)
        s1 = model.sample_gaussian(batch, n_samples=8, generator=g1)
        s2 = model.sample_gaussian(batch, n_samples=8, generator=g2)
        self.assertEqual(tuple(s1["samples"].shape), (8, 1, 1, 76, 76))
        self.assertFalse(torch.allclose(s1["samples"], s2["samples"]))

    def test_gaussian_invalid_pixels_zero_loss_contribution(self) -> None:
        cfg = _PrepCfg()
        base = _TinyBase()
        model = ResidualMapGenerator(
            cfg,  # type: ignore[arg-type]
            base_model=base,  # type: ignore[arg-type]
            base_channel_index=0,
            variant="gaussian",
        )
        batch = _synthetic_batch(b=1)
        batch["targets"]["ha_flux"][0, 0, 0] = 50.0
        batch["target_loss_masks"]["ha_flux"][0, 0, 0] = 0.0
        _, loss_a = model(batch)
        batch["targets"]["ha_flux"][0, 0, 0] = -50.0
        _, loss_b = model(batch)
        self.assertAlmostEqual(float(loss_a["loss"]), float(loss_b["loss"]), places=4)


class DiffusionTests(unittest.TestCase):
    def test_ddim_reproducible_with_fixed_seed_and_varies_otherwise(self) -> None:
        denoise = CondResidualDiffusionUNet(cond_channels=7, residual_channels=1, base_channels=16, channel_mults=(1, 2))
        sched = ResidualDiffusionSchedule(n_steps=20, schedule="linear").to(torch.device("cpu"))
        cond = torch.randn(1, 7, 16, 16)
        mask = torch.ones(1, 1, 16, 16)
        g1 = torch.Generator().manual_seed(7)
        g1b = torch.Generator().manual_seed(7)
        g2 = torch.Generator().manual_seed(99)
        a = sched.ddim_sample(denoise, cond, steps=5, generator=g1, mask=mask)
        b = sched.ddim_sample(denoise, cond, steps=5, generator=g1b, mask=mask)
        c = sched.ddim_sample(denoise, cond, steps=5, generator=g2, mask=mask)
        self.assertEqual(tuple(a.shape), (1, 1, 16, 16))
        self.assertTrue(torch.allclose(a, b))
        self.assertFalse(torch.allclose(a, c))

    def test_diffusion_wrapper_shapes_and_masking(self) -> None:
        cfg = _PrepCfg()
        base = _TinyBase()
        model = ResidualDiffusionMapGenerator(
            cfg,  # type: ignore[arg-type]
            base_model=base,  # type: ignore[arg-type]
            base_channel_index=0,
            diffusion_steps=20,
            ddim_steps=4,
            n_samples=3,
            base_channels=16,
            channel_mults=(1, 2),
        )
        batch = _synthetic_batch(b=1, h=16, w=16)
        # Resize footprint/targets already 16 from helper override
        pred, loss = model(batch)
        self.assertEqual(tuple(pred["maps"].shape), (1, 1, 16, 16))
        self.assertTrue(torch.isfinite(loss["loss"]))
        # Invalidate corner and ensure loss unchanged when poisoning it in residual path
        batch["targets"]["ha_flux"][0, 0, 0] = 99.0
        batch["target_loss_masks"]["ha_flux"][0, 0, 0] = 0.0
        torch.manual_seed(0)
        _, loss_a = model(batch)
        batch["targets"]["ha_flux"][0, 0, 0] = -99.0
        torch.manual_seed(0)
        _, loss_b = model(batch)
        self.assertAlmostEqual(float(loss_a["loss"]), float(loss_b["loss"]), places=5)

        out = model.sample(batch, n_samples=3, ddim_steps=4, seed=11)
        self.assertEqual(tuple(out["samples"].shape), (3, 1, 1, 16, 16))
        out2 = model.sample(batch, n_samples=3, ddim_steps=4, seed=11)
        self.assertTrue(torch.allclose(out["samples"], out2["samples"]))


class SharedNormTests(unittest.TestCase):
    def test_asinh_path_matches_manual(self) -> None:
        cfg = _PrepCfg(
            input_norm_mode="asinh",
            imaging_asinh_scales=[1.0, 2.0, 3.0, 4.0, 5.0],
        )
        batch = _synthetic_batch(b=1, h=8, w=8)
        x = prepare_imaging_input(batch, cfg)
        raw = batch["inputs"]["sdss_imaging"]
        scales = torch.tensor(cfg.imaging_asinh_scales).view(1, -1, 1, 1)
        expected = torch.asinh(raw / scales)
        self.assertTrue(torch.allclose(x, expected))


if __name__ == "__main__":
    unittest.main()
