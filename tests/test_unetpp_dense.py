"""Acceptance tests for dense UNet++ and ConvBlock residual projection."""
from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from src.models.config import ModelConfig
from src.models.conditional_unet import ConditionalMapModel
from src.models.unet import ConvBlock
from src.models.unetpp import UNetPPBackbone
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


class ConvBlockResidualTests(unittest.TestCase):
    def test_residual_projection_when_channels_differ(self) -> None:
        block = ConvBlock(8, 16, residual=True, norm="gn", dropout=0.0)
        self.assertTrue(block.residual)
        self.assertIsInstance(block.proj, nn.Conv2d)
        x = torch.randn(2, 8, 12, 12, requires_grad=True)
        y = block(x)
        self.assertEqual(y.shape, (2, 16, 12, 12))
        y.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        # Projection weights must receive gradients (previously dead).
        self.assertIsNotNone(block.proj.weight.grad)
        self.assertGreater(float(block.proj.weight.grad.abs().sum()), 0.0)

    def test_identity_residual_when_channels_match(self) -> None:
        block = ConvBlock(16, 16, residual=True, norm="gn", dropout=0.0)
        self.assertIsInstance(block.proj, nn.Identity)


class UNetPPDenseTests(unittest.TestCase):
    def _small_backbone(self, *, with_output_conv: bool = True) -> UNetPPBackbone:
        return UNetPPBackbone(
            in_channels=3,
            out_channels=2,
            base_channels=8,
            depth=3,
            dropout=0.0,
            upsample_mode="bilinear",
            norm="gn",
            with_output_conv=with_output_conv,
        )

    def test_output_shape(self) -> None:
        net = self._small_backbone()
        x = torch.randn(2, 3, 76, 76)
        y = net(x)
        self.assertEqual(tuple(y.shape), (2, 2, 76, 76))

    def test_all_nested_blocks_receive_gradients(self) -> None:
        net = self._small_backbone()
        x = torch.randn(2, 3, 32, 32)
        y = net(x)
        y.mean().backward()
        dead = []
        for name, module in net.nested.items():
            for pname, p in module.named_parameters():
                if p.grad is None or float(p.grad.abs().sum()) == 0.0:
                    dead.append(f"{name}.{pname}")
        self.assertEqual(dead, [], msg=f"Dead nested params: {dead}")

    def test_no_nested_dead_ends(self) -> None:
        """Perturbing every nested node must change the final output."""
        net = self._small_backbone()
        net.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            base = net(x).clone()

        for key in net.nested_node_keys():
            def hook(_module, _inp, out, key=key):
                return out + 0.25

            handle = net.nested[key].register_forward_hook(hook)
            with torch.no_grad():
                perturbed = net(x)
            handle.remove()
            delta = (perturbed - base).abs().sum().item()
            self.assertGreater(delta, 1e-5, msg=f"Node {key} does not affect output")

    def test_early_nested_node_affects_output(self) -> None:
        net = self._small_backbone()
        net.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            base = net(x).clone()

        def bump(_module, _inp, out):
            return out + 1.0

        handle = net.nested["x01"].register_forward_hook(bump)
        with torch.no_grad():
            perturbed = net(x)
        handle.remove()
        self.assertGreater(float((perturbed - base).abs().sum()), 1e-4)

    def test_film_encoder_hooks_still_work(self) -> None:
        cfg = ModelConfig(
            architecture="unetpp",
            output_head="single",
            film_injection="encoder",
            cond_dim=32,
            base_channels=8,
            n_down=3,
            deep_supervision=False,
            losses=["l1"],
            loss_weights=[1.0],
            spectrum_use_wavelength=False,
            spectrum_use_ivar=False,
            spectrum_pooling="avg",
        )
        model = ConditionalMapModel(cfg)
        seen: list[tuple[int, ...]] = []

        for film in model.encoder_film:
            orig = film.forward

            def spy(x, cond, orig=orig):
                seen.append(tuple(x.shape))
                return orig(x, cond)

            film.forward = spy  # type: ignore[method-assign]

        x = torch.randn(1, 5, 76, 76)
        with torch.no_grad():
            maps, _ = model(x, spectrum_flux=torch.randn(1, cfg.spectrum_n_wave), footprint=torch.ones(1, 76, 76))
        self.assertEqual(maps.shape, (1, cfg.n_target_maps, 76, 76))
        self.assertEqual(len(seen), cfg.n_down + 1)

    def test_deep_supervision_keys_are_full_resolution(self) -> None:
        net = self._small_backbone(with_output_conv=False)
        keys = net.deep_supervision_keys()
        self.assertEqual(keys, ["x01", "x02", "x03"])
        x = torch.randn(1, 3, 32, 32)
        nodes = net.forward_nodes(x)
        for key in keys:
            self.assertEqual(nodes[key].shape[-2:], (32, 32))

    def test_end_to_end_map_generator_shape(self) -> None:
        cfg = ModelConfig(
            architecture="unetpp",
            output_head="single",
            film_injection="encoder",
            cond_dim=32,
            base_channels=8,
            n_down=3,
            deep_supervision=True,
            losses=["l1"],
            loss_weights=[1.0],
        )
        wrap = MapGenerator(cfg)
        pred, loss = wrap(_batch(cfg))
        self.assertEqual(pred["maps"].shape, (2, cfg.n_target_maps, 76, 76))
        self.assertEqual(pred["deep_maps"].shape[0], cfg.n_down)
        self.assertTrue(torch.isfinite(loss["loss"]))


if __name__ == "__main__":
    unittest.main()
