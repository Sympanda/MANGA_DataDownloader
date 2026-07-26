"""Tests for aligned-76 + HR cross-attention architecture (not resize/concat)."""
from __future__ import annotations

import unittest

import torch

from src.models.config import ModelConfig
from src.models.hr_pipeline import CrossAttnHRBlock, HREncoder
from src.models.wrapper import MapGenerator


def _cfg(**kwargs) -> ModelConfig:
    base = dict(
        architecture="unetpp",
        output_head="single",
        imaging_resolution="aligned",
        spatial_pipeline="symmetric",
        footprint_mode="spatial_channel",
        use_hr_cross_attn=True,
        hr_survey="sdss",
        hr_cross_attn_levels=(0, 1),
        hr_encoder_n_down=2,
        film_injection="encoder",
        cond_dim=32,
        base_channels=8,
        n_down=3,
        deep_supervision=False,
        spectrum_pooling="attention",
        spectrum_use_wavelength=True,
        spectrum_use_ivar=True,
        losses=["l1"],
        loss_weights=[1.0],
    )
    base.update(kwargs)
    return ModelConfig(**base)


def _batch(cfg: ModelConfig, *, batch_size: int = 2, hr_size: int = 128) -> dict:
    t = cfg.target_spatial_size
    inputs: dict = {
        "sdss_imaging": torch.randn(batch_size, 5, 76, 76),
        "spectrum": {
            "flux": torch.randn(batch_size, cfg.spectrum_n_wave),
            "wave": torch.linspace(cfg.spectrum_wave_min, cfg.spectrum_wave_max, cfg.spectrum_n_wave)
            .unsqueeze(0)
            .expand(batch_size, -1),
            "ivar": torch.rand(batch_size, cfg.spectrum_n_wave) * 10,
        },
    }
    if cfg.use_hr_cross_attn:
        inputs["hr_imaging"] = torch.randn(batch_size, cfg.hr_imaging_channels(), hr_size, hr_size)
    return {
        "footprint_mask": torch.ones(batch_size, t, t),
        "inputs": inputs,
        "targets": {k: torch.rand(batch_size, t, t) for k in cfg.target_keys},
        "target_loss_masks": {k: torch.ones(batch_size, t, t) for k in cfg.target_keys},
    }


class HRCrossAttnArchitectureTests(unittest.TestCase):
    def test_shapes_aligned_in_maps_out(self) -> None:
        cfg = _cfg()
        wrap = MapGenerator(cfg)
        batch = _batch(cfg)
        self.assertEqual(batch["inputs"]["sdss_imaging"].shape[-2:], (76, 76))
        pred, loss = wrap(batch)
        self.assertEqual(pred["maps"].shape, (2, cfg.n_target_maps, 76, 76))
        self.assertTrue(torch.isfinite(loss["loss"]))

    def test_no_legacy_resize_concat_modules(self) -> None:
        wrap = MapGenerator(_cfg())
        self.assertIsNone(wrap.model.grid_projector)
        self.assertIsNone(wrap.model.hr_fusions)
        self.assertIsNotNone(wrap.model.hr_cross_encoder)
        self.assertIsNotNone(wrap.model.hr_cross_blocks)

    def test_gradients_through_hr_qkv_and_encoder(self) -> None:
        cfg = _cfg()
        wrap = MapGenerator(cfg)
        _, loss = wrap(_batch(cfg))
        loss["loss"].backward()

        enc = wrap.model.hr_cross_encoder
        assert enc is not None
        self.assertIsNotNone(enc.stem.net[0].weight.grad)
        self.assertGreater(float(enc.stem.net[0].weight.grad.abs().sum()), 0.0)

        assert wrap.model.hr_cross_blocks is not None
        for key, block in wrap.model.hr_cross_blocks.items():
            for name, module in (
                ("query_proj", block.query_proj),
                ("key_proj", block.key_proj),
                ("value_proj", block.value_proj),
                ("out_proj", block.out_proj),
            ):
                g = module.weight.grad
                self.assertIsNotNone(g, msg=f"level {key} {name} has no grad")
                self.assertGreater(float(g.abs().sum()), 0.0, msg=f"level {key} {name} dead")

    def test_spatial_hr_dependence_moves_attention(self) -> None:
        """A localised HR blob should shift where queries attend (not global pooling)."""
        block = CrossAttnHRBlock(unet_channels=8, hr_channels=8, num_heads=2)
        block.eval()
        unet = torch.zeros(1, 8, 16, 16)
        # Uniform UNet so attention is driven by HR+coords.
        unet[:, :, 8, 8] = 1.0

        def _hr_with_blob(y: int, x: int) -> torch.Tensor:
            hr = torch.zeros(1, 8, 32, 32)
            hr[:, :, y : y + 2, x : x + 2] = 5.0
            return hr

        with torch.no_grad():
            _, attn_a = block(unet, _hr_with_blob(4, 4), return_attn=True)
            _, attn_b = block(unet, _hr_with_blob(4, 28), return_attn=True)

        # Attention from centre query (index 8*16+8) over HR tokens.
        q_idx = 8 * 16 + 8
        peak_a = int(attn_a[0, q_idx].argmax().item())
        peak_b = int(attn_b[0, q_idx].argmax().item())
        ya, xa = divmod(peak_a, 32)
        yb, xb = divmod(peak_b, 32)
        self.assertNotEqual(peak_a, peak_b)
        self.assertLess(xa, xb)  # blob moved right → attention peak moves right
        self.assertLess(abs(ya - 4), 4)
        self.assertLess(abs(yb - 4), 4)

    def test_ablation_without_hr_cross_attn(self) -> None:
        """Disabled HR cross-attn → aligned 76 + spectrum FiLM only."""
        cfg = _cfg(use_hr_cross_attn=False)
        wrap = MapGenerator(cfg)
        self.assertIsNone(wrap.model.hr_cross_encoder)
        self.assertIsNone(wrap.model.hr_cross_blocks)
        self.assertIsNone(wrap.model.grid_projector)
        pred, loss = wrap(_batch(cfg))
        self.assertEqual(pred["maps"].shape, (2, cfg.n_target_maps, 76, 76))
        self.assertTrue(torch.isfinite(loss["loss"]))

    def test_hr_encoder_keeps_spatial_tokens(self) -> None:
        enc = HREncoder(5, base_channels=8, n_down=2, norm="gn")
        x = torch.randn(1, 5, 128, 128)
        feats = enc.forward_pyramid(x)
        deepest = feats[-1]
        self.assertEqual(deepest.ndim, 4)
        self.assertGreater(deepest.shape[-1] * deepest.shape[-2], 1)
        tokens = deepest.flatten(2).transpose(1, 2)
        self.assertEqual(tokens.shape[:2], (1, deepest.shape[-1] * deepest.shape[-2]))


if __name__ == "__main__":
    unittest.main()
