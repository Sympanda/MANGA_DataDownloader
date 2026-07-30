"""Tests for local HR cross-attention (memory-efficient, not dense N_q×N_hr)."""
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
        hr_cross_attn_levels=(1,),
        hr_attention_mode="local",
        hr_attention_window=7,
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


class LocalHRCrossAttnTests(unittest.TestCase):
    def test_shapes_aligned_in_maps_out(self) -> None:
        cfg = _cfg()
        wrap = MapGenerator(cfg)
        pred, loss = wrap(_batch(cfg))
        self.assertEqual(pred["maps"].shape, (2, cfg.n_target_maps, 76, 76))
        self.assertTrue(torch.isfinite(loss["loss"]))

    def test_attention_logits_depend_on_window_not_hr_area(self) -> None:
        block = CrossAttnHRBlock(
            unet_channels=8, hr_channels=8, num_heads=2, mode="local", window=7
        )
        block.eval()
        unet = torch.randn(1, 8, 16, 16)
        hr_small = torch.randn(1, 8, 24, 24)
        hr_large = torch.randn(1, 8, 64, 64)
        with torch.no_grad():
            _, attn_s = block(unet, hr_small, return_attn=True)
            _, attn_l = block(unet, hr_large, return_attn=True)
        k = 7 * 7
        self.assertEqual(attn_s.shape, (1, 16 * 16, k))
        self.assertEqual(attn_l.shape, (1, 16 * 16, k))
        self.assertEqual(block.local_token_count, k)

    def test_no_legacy_resize_concat_modules(self) -> None:
        wrap = MapGenerator(_cfg())
        self.assertIsNone(wrap.model.grid_projector)
        self.assertIsNone(wrap.model.hr_fusions)
        self.assertIsNotNone(wrap.model.hr_cross_encoder)
        assert wrap.model.hr_cross_blocks is not None
        for block in wrap.model.hr_cross_blocks.values():
            self.assertEqual(block.mode, "local")

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

    def test_spatial_hr_dependence_moves_with_blob(self) -> None:
        block = CrossAttnHRBlock(
            unet_channels=8, hr_channels=8, num_heads=2, mode="local", window=7
        )
        block.eval()
        unet = torch.zeros(1, 8, 16, 16)
        unet[:, :, 8, 8] = 1.0

        def _hr_blob(y: int, x: int) -> torch.Tensor:
            hr = torch.zeros(1, 8, 32, 32)
            hr[:, :, y : y + 2, x : x + 2] = 5.0
            return hr

        with torch.no_grad():
            out_a, _ = block(unet, _hr_blob(4, 4), return_attn=True)
            out_b, _ = block(unet, _hr_blob(4, 28), return_attn=True)
        # Feature response at the query should change when the local HR blob moves.
        self.assertGreater(float((out_a - out_b)[:, :, 8, 8].abs().sum()), 1e-5)

    def test_locality_far_blob_ignored(self) -> None:
        """HR energy outside the window must not affect a distant query."""
        block = CrossAttnHRBlock(
            unet_channels=4, hr_channels=4, num_heads=2, mode="local", window=3
        )
        block.eval()
        # GroupNorm couples spatial sites; replace so we can test locality of gather/attn alone.
        block.norm = torch.nn.Identity()
        with torch.no_grad():
            for proj in (block.query_proj, block.key_proj, block.value_proj, block.out_proj):
                proj.weight.zero_()
                eye = min(proj.weight.shape[0], proj.weight.shape[1])
                for i in range(eye):
                    proj.weight[i, i, 0, 0] = 1.0

        unet = torch.zeros(1, 4, 8, 8)
        unet[:, :, 1, 1] = 1.0
        hr_base = torch.zeros(1, 4, 32, 32)
        hr_far = hr_base.clone()
        hr_far[:, :, 28, 28] = 10.0

        q_idx = 1 * 8 + 1
        local_base = block._gather_local_hr_windows(hr_base, 8, 8)
        local_far = block._gather_local_hr_windows(hr_far, 8, 8)
        self.assertTrue(torch.allclose(local_base[:, q_idx], local_far[:, q_idx], atol=1e-6))

        with torch.no_grad():
            out_base = block(unet, hr_base)
            out_far = block(unet, hr_far)
        self.assertTrue(torch.allclose(out_base[:, :, 1, 1], out_far[:, :, 1, 1], atol=1e-5))

    def test_ablation_without_hr_cross_attn(self) -> None:
        cfg = _cfg(use_hr_cross_attn=False)
        wrap = MapGenerator(cfg)
        self.assertIsNone(wrap.model.hr_cross_encoder)
        pred, loss = wrap(_batch(cfg))
        self.assertEqual(pred["maps"].shape, (2, cfg.n_target_maps, 76, 76))
        self.assertTrue(torch.isfinite(loss["loss"]))

    def test_hr_encoder_keeps_spatial_tokens(self) -> None:
        enc = HREncoder(5, base_channels=8, n_down=2, norm="gn")
        deepest = enc.forward_pyramid(torch.randn(1, 5, 128, 128))[-1]
        self.assertGreater(deepest.shape[-1] * deepest.shape[-2], 1)


if __name__ == "__main__":
    unittest.main()
