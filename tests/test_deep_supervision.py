"""Tests for ConditionalMapModel FiLM sites and UNet++ deep supervision."""
from __future__ import annotations

import unittest

import torch

from src.models.conditional_unet import ConditionalMapModel
from src.models.config import ModelConfig
from src.models.wrapper import MapGenerator


def _batch(cfg: ModelConfig, batch_size: int = 2) -> dict:
    t = cfg.target_spatial_size
    return {
        "footprint_mask": torch.ones(batch_size, t, t),
        "inputs": {
            "sdss_imaging": torch.randn(batch_size, 5, t, t),
            "spectrum": {
                "flux": torch.randn(batch_size, cfg.spectrum_n_wave),
                "wave": torch.randn(batch_size, cfg.spectrum_n_wave),
                "ivar": torch.randn(batch_size, cfg.spectrum_n_wave),
            },
        },
        "targets": {k: torch.rand(batch_size, t, t) for k in cfg.target_keys},
        "target_loss_masks": {k: torch.ones(batch_size, t, t) for k in cfg.target_keys},
    }


class FilmAndDeepSupervisionTests(unittest.TestCase):
    def test_unetpp_bottleneck_film_hits_deep_encoder(self) -> None:
        cfg = ModelConfig(
            architecture="unetpp",
            film_injection="bottleneck",
            cond_dim=32,
            base_channels=16,
            n_down=3,
            deep_supervision=False,
            spectrum_use_wavelength=False,
            spectrum_use_ivar=False,
            spectrum_pooling="avg",
        )
        model = ConditionalMapModel(cfg)
        seen: dict[str, tuple[int, ...]] = {}
        orig = model.bottleneck_film.forward

        def spy(x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
            seen["shape"] = tuple(x.shape)
            return orig(x, cond)

        model.bottleneck_film.forward = spy  # type: ignore[method-assign]
        x = torch.randn(1, 5, 76, 76)
        fp = torch.ones(1, 76, 76)
        spec = torch.randn(1, cfg.spectrum_n_wave)
        with torch.no_grad():
            model(x, spectrum_flux=spec, footprint=fp)
        # Deepest spine for n_down=3, base=16 → 16*8=128 channels at ~9×9
        self.assertEqual(seen["shape"][1], cfg.base_channels * (2**cfg.n_down))
        self.assertLess(seen["shape"][-1], 76)

    def test_deep_supervision_forward_and_loss(self) -> None:
        cfg = ModelConfig(
            architecture="unetpp",
            output_head="single",
            film_injection="encoder",
            cond_dim=32,
            base_channels=16,
            n_down=3,
            deep_supervision=True,
            deep_supervision_loss="grad",
            deep_supervision_weights=[0.2, 0.4],
            losses=["l1", "grad", "laplacian"],
            loss_weights=[1.0, 1.0, 0.4],
        )
        wrap = MapGenerator(cfg)
        pred, loss = wrap(_batch(cfg))
        self.assertEqual(pred["maps"].shape, (2, cfg.n_target_maps, 76, 76))
        self.assertEqual(pred["deep_maps"].shape[0], cfg.n_down)
        self.assertIn("ds_0", loss)
        self.assertIn("ds_1", loss)
        self.assertTrue(torch.isfinite(loss["loss"]))

    def test_deep_supervision_rejects_coarse_fine(self) -> None:
        with self.assertRaises(ValueError):
            ModelConfig(
                architecture="unetpp",
                output_head="coarse_fine",
                deep_supervision=True,
            ).validate()

    def test_deep_supervision_requires_unetpp(self) -> None:
        with self.assertRaises(ValueError):
            ModelConfig(architecture="unet", deep_supervision=True).validate()


if __name__ == "__main__":
    unittest.main()
