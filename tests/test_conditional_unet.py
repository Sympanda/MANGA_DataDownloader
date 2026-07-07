"""Smoke tests for ConditionalMapUNet."""
from __future__ import annotations

import unittest

import torch

from manga_models.batch_utils import (
    masked_mse_loss_multichannel,
    prepare_spatial_input,
    prepare_spectrum_input,
    prepare_targets_and_masks,
)
from manga_models.conditional_unet import ConditionalMapUNet
from manga_models.config import ConditionalUNetConfig, MODEL_PRESETS


class ConditionalUNetTests(unittest.TestCase):
    def test_forward_sdss_spectrum_footprint(self) -> None:
        config = ConditionalUNetConfig(
            use_sdss=True,
            use_legacy=False,
            use_spectrum=True,
            use_footprint_mask=True,
            base_channels=16,
            cond_dim=64,
        )
        model = ConditionalMapUNet(config)
        batch_size = 2
        x = torch.randn(batch_size, config.input_channels(), 76, 76)
        spec = torch.randn(batch_size, config.spectrum_n_wave)
        out = model(x, spectrum_flux=spec)
        self.assertEqual(out.shape, (batch_size, config.n_target_maps, 76, 76))

    def test_batch_utils_shapes(self) -> None:
        config = ConditionalUNetConfig(use_sdss=True, use_spectrum=True, use_footprint_mask=True)
        batch = {
            "inputs": {
                "sdss_imaging": torch.randn(2, 5, 76, 76),
                "spectrum": {"flux": torch.randn(2, 4563)},
            },
            "targets": {k: torch.rand(2, 76, 76) for k in config.target_keys},
            "target_loss_masks": {k: torch.ones(2, 76, 76) for k in config.target_keys},
            "footprint_mask": torch.ones(2, 76, 76),
        }
        x = prepare_spatial_input(batch, config)
        self.assertEqual(x.shape, (2, 6, 76, 76))
        spec = prepare_spectrum_input(batch, config)
        assert spec is not None
        self.assertEqual(spec.shape, (2, 4563))
        targets, masks = prepare_targets_and_masks(batch, config)
        pred = torch.randn(2, 6, 76, 76)
        loss = masked_mse_loss_multichannel(pred, targets, masks)
        self.assertTrue(torch.isfinite(loss))

    def test_forward_medium_preset(self) -> None:
        preset = MODEL_PRESETS["medium"]
        config = ConditionalUNetConfig(
            use_sdss=True,
            use_legacy=False,
            use_spectrum=True,
            use_footprint_mask=True,
            **preset,
        )
        model = ConditionalMapUNet(config)
        x = torch.randn(2, config.input_channels(), 76, 76)
        spec = torch.randn(2, config.spectrum_n_wave)
        out = model(x, spectrum_flux=spec)
        self.assertEqual(out.shape, (2, config.n_target_maps, 76, 76))

    def test_combined_training_loss_finite(self) -> None:
        config = ConditionalUNetConfig(
            loss_mse_weight=0.5,
            loss_l1_weight=0.5,
            loss_grad_weight=0.1,
        )
        pred = torch.randn(2, 6, 76, 76)
        targets = torch.rand(2, 6, 76, 76)
        masks = torch.ones(2, 6, 76, 76)
        from manga_models.batch_utils import compute_map_training_loss

        loss = compute_map_training_loss(pred, targets, masks, config)
        self.assertTrue(torch.isfinite(loss))

    def test_masked_loss_ignores_nan_outside_mask(self) -> None:
        pred = torch.tensor([[[[0.5, 0.5]]]])
        target = torch.tensor([[[[float("nan"), 1.0]]]])
        mask = torch.tensor([[[[0.0, 1.0]]]])
        loss = masked_mse_loss_multichannel(pred, target, mask)
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(float(loss.item()), 0.25, places=5)


if __name__ == "__main__":
    unittest.main()
