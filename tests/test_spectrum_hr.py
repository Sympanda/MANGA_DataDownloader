"""Tests for spectrum attention encoder and HR multi-scale fusion."""
from __future__ import annotations

import unittest

import torch

from src.models.config import ModelConfig
from src.models.encoders import SpectrumEncoder
from src.models.hr_pipeline import CrossAttnHRBlock, HREncoder
from src.models.wrapper import MapGenerator, prepare_spectrum_input


def _batch(cfg: ModelConfig, *, img_size: int | None = None, batch_size: int = 2) -> dict:
    t = cfg.target_spatial_size
    img = img_size if img_size is not None else (76 if cfg.imaging_resolution == "aligned" else 196)
    inputs: dict = {
        "sdss_imaging": torch.randn(batch_size, 5, img, img),
        "spectrum": {
            "flux": torch.randn(batch_size, cfg.spectrum_n_wave),
            "wave": torch.linspace(cfg.spectrum_wave_min, cfg.spectrum_wave_max, cfg.spectrum_n_wave)
            .unsqueeze(0)
            .expand(batch_size, -1),
            "ivar": torch.rand(batch_size, cfg.spectrum_n_wave) * 10,
        },
    }
    if cfg.use_hr_cross_attn:
        inputs["hr_imaging"] = torch.randn(batch_size, cfg.hr_imaging_channels(), 128, 128)
    return {
        "footprint_mask": torch.ones(batch_size, t, t),
        "inputs": inputs,
        "targets": {k: torch.rand(batch_size, t, t) for k in cfg.target_keys},
        "target_loss_masks": {k: torch.ones(batch_size, t, t) for k in cfg.target_keys},
    }


class SpectrumEncoderTests(unittest.TestCase):
    def test_output_shape_attention(self) -> None:
        enc = SpectrumEncoder(n_wave=256, out_dim=64, in_channels=3, pooling="attention")
        x = torch.randn(2, 3, 256)
        y = enc(x)
        self.assertEqual(y.shape, (2, 64))

    def test_feature_shift_changes_conditioning(self) -> None:
        enc = SpectrumEncoder(n_wave=128, out_dim=32, in_channels=1, pooling="attention")
        enc.eval()
        base = torch.zeros(1, 1, 128)
        a = base.clone()
        a[0, 0, 20:25] = 1.0
        b = base.clone()
        b[0, 0, 90:95] = 1.0
        with torch.no_grad():
            ca, cb = enc(a), enc(b)
        self.assertGreater(float((ca - cb).abs().sum()), 1e-4)

    def test_attention_weights_finite(self) -> None:
        enc = SpectrumEncoder(n_wave=128, out_dim=32, in_channels=2, pooling="attention")
        x = torch.randn(2, 2, 128)
        y = enc(x)
        self.assertTrue(torch.isfinite(y).all())

    def test_prepare_spectrum_channels(self) -> None:
        cfg = ModelConfig(
            spectrum_use_wavelength=True,
            spectrum_use_ivar=True,
            spectrum_n_wave=64,
        )
        batch = _batch(cfg)
        # shrink spectrum length for this unit test
        for k in ("flux", "wave", "ivar"):
            batch["inputs"]["spectrum"][k] = batch["inputs"]["spectrum"][k][:, :64]
        cfg.spectrum_n_wave = 64
        spec = prepare_spectrum_input(batch, cfg)
        assert spec is not None
        self.assertEqual(spec.shape, (2, 3, 64))


class HRMultiscaleTests(unittest.TestCase):
    def test_pyramid_scales(self) -> None:
        enc = HREncoder(5, base_channels=8, n_down=3, norm="gn")
        x = torch.randn(1, 5, 128, 128)
        feats = enc.forward_pyramid(x)
        self.assertEqual(len(feats), 4)
        self.assertEqual(feats[0].shape[1], 8)
        self.assertLess(feats[-1].shape[-1], feats[0].shape[-1])

    def test_hr_multiscale_end_to_end(self) -> None:
        cfg = ModelConfig(
            architecture="unetpp",
            output_head="single",
            imaging_resolution="native",
            spatial_pipeline="hr_multiscale",
            footprint_mode="fusion_concat",
            film_injection="encoder",
            cond_dim=32,
            base_channels=8,
            n_down=3,
            deep_supervision=True,
            spectrum_pooling="attention",
            spectrum_use_wavelength=True,
            spectrum_use_ivar=True,
            losses=["l1"],
            loss_weights=[1.0],
        )
        wrap = MapGenerator(cfg)
        pred, loss = wrap(_batch(cfg, img_size=128))
        self.assertEqual(pred["maps"].shape, (2, cfg.n_target_maps, 76, 76))
        self.assertTrue(torch.isfinite(loss["loss"]))
        # HR fusion modules must get gradients
        loss["loss"].backward()
        assert wrap.model.hr_fusions is not None
        for i, fuse in enumerate(wrap.model.hr_fusions):
            w = fuse.channel_proj.weight
            self.assertIsNotNone(w.grad, msg=f"HR fusion level {i} has no grad")
            self.assertGreater(float(w.grad.abs().sum()), 0.0, msg=f"HR fusion level {i} dead")

    def test_removing_shallow_hr_changes_output(self) -> None:
        cfg = ModelConfig(
            architecture="unetpp",
            output_head="single",
            imaging_resolution="native",
            spatial_pipeline="hr_multiscale",
            footprint_mode="loss_only",
            use_footprint_mask=False,
            film_injection="none",
            use_spectrum=False,
            base_channels=8,
            n_down=3,
            deep_supervision=False,
            losses=["l1"],
            loss_weights=[1.0],
        )
        wrap = MapGenerator(cfg)
        wrap.eval()
        batch = _batch(cfg, img_size=96, batch_size=1)
        with torch.no_grad():
            base, _ = wrap(batch)
            assert wrap.model.hr_fusions is not None
            wrap.model.hr_fusions[0].channel_proj.weight.zero_()
            pert, _ = wrap(batch)
        self.assertGreater(float((pert["maps"] - base["maps"]).abs().sum()), 1e-5)


class HRCrossAttnTests(unittest.TestCase):
    def test_cross_attn_block_shape(self) -> None:
        block = CrossAttnHRBlock(unet_channels=16, hr_channels=32, num_heads=4)
        unet = torch.randn(2, 16, 38, 38)
        hr = torch.randn(2, 32, 24, 24)
        out = block(unet, hr)
        self.assertEqual(out.shape, unet.shape)
        self.assertTrue(torch.isfinite(out).all())

    def test_hr_cross_attn_end_to_end(self) -> None:
        cfg = ModelConfig(
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
            deep_supervision=True,
            spectrum_pooling="attention",
            spectrum_use_wavelength=True,
            spectrum_use_ivar=True,
            losses=["l1"],
            loss_weights=[1.0],
        )
        wrap = MapGenerator(cfg)
        pred, loss = wrap(_batch(cfg, img_size=76))
        self.assertEqual(pred["maps"].shape, (2, cfg.n_target_maps, 76, 76))
        self.assertTrue(torch.isfinite(loss["loss"]))
        loss["loss"].backward()
        assert wrap.model.hr_cross_blocks is not None
        for key, block in wrap.model.hr_cross_blocks.items():
            w = block.out_proj.weight
            self.assertIsNotNone(w.grad, msg=f"HR cross-attn level {key} has no grad")
            self.assertGreater(float(w.grad.abs().sum()), 0.0, msg=f"HR cross-attn level {key} dead")

    def test_disabling_hr_cross_attn_changes_output(self) -> None:
        cfg = ModelConfig(
            architecture="unetpp",
            output_head="single",
            imaging_resolution="aligned",
            spatial_pipeline="symmetric",
            footprint_mode="loss_only",
            use_footprint_mask=False,
            use_hr_cross_attn=True,
            hr_cross_attn_levels=(0,),
            hr_encoder_n_down=2,
            film_injection="none",
            use_spectrum=False,
            base_channels=8,
            n_down=3,
            deep_supervision=False,
            losses=["l1"],
            loss_weights=[1.0],
        )
        wrap = MapGenerator(cfg)
        wrap.eval()
        batch = _batch(cfg, img_size=76, batch_size=1)
        with torch.no_grad():
            base, _ = wrap(batch)
            assert wrap.model.hr_cross_blocks is not None
            wrap.model.hr_cross_blocks["0"].out_proj.weight.zero_()
            pert, _ = wrap(batch)
        self.assertGreater(float((pert["maps"] - base["maps"]).abs().sum()), 1e-5)


if __name__ == "__main__":
    unittest.main()
