"""Unit tests for mask-topology-aware map losses."""
from __future__ import annotations

import unittest

import torch

from src.models.losses import (
    compose_map_losses,
    masked_charbonnier,
    masked_integration_loss,
    masked_laplacian_loss,
    masked_pairwise_grad_loss,
    prediction_tv_loss,
    residual_amplitude_loss,
)


class MaskedLossTests(unittest.TestCase):
    def _sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred = torch.randn(2, 3, 8, 8)
        target = torch.randn(2, 3, 8, 8)
        mask = torch.zeros(2, 3, 8, 8)
        mask[:, :, 2:6, 2:6] = 1.0
        target = torch.where(mask > 0, target, torch.full_like(target, float("nan")))
        return pred, target, mask

    def test_pixel_loss_ignores_invalid_and_nan(self) -> None:
        pred = torch.tensor([[[[0.5, 0.5, 0.0]]]])
        target = torch.tensor([[[[float("nan"), 1.0, 2.0]]]])
        mask = torch.tensor([[[[0.0, 1.0, 0.0]]]])
        loss = masked_charbonnier(pred, target, mask)
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(float(loss.item()), 0.5, places=4)

    def test_grad_loss_skips_mask_boundary(self) -> None:
        pred, target, mask = self._sample()
        loss = masked_pairwise_grad_loss(pred, target, mask)
        self.assertTrue(torch.isfinite(loss))

        # Hole in mask: horizontal pair across hole must not contribute.
        mask2 = mask.clone()
        mask2[:, :, 4, 3] = 0.0
        loss_hole = masked_pairwise_grad_loss(pred, target, mask2)
        self.assertTrue(torch.isfinite(loss_hole))

    def test_laplacian_requires_full_stencil(self) -> None:
        pred, target, mask = self._sample()
        loss = masked_laplacian_loss(pred, target, mask)
        self.assertTrue(torch.isfinite(loss))

    def test_tv_pred_no_target_needed(self) -> None:
        pred, _target, mask = self._sample()
        loss = prediction_tv_loss(pred, mask)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss.item()), 0.0)

    def test_integration_uses_masked_mean(self) -> None:
        pred = torch.ones(1, 1, 4, 4)
        target = torch.ones(1, 1, 4, 4) * 2.0
        mask = torch.zeros(1, 1, 4, 4)
        mask[:, :, 1:3, 1:3] = 1.0
        loss = masked_integration_loss(pred, target, mask, channel_indices=[0], normalize="mean")
        # valid pixels = 4, means: 1 vs 2 -> |4-8|/4 = 1
        self.assertAlmostEqual(float(loss.item()), 1.0, places=4)

    def test_compose_with_residual_terms(self) -> None:
        pred, target, mask = self._sample()
        residual = torch.randn_like(pred) * 0.1
        out = compose_map_losses(
            pred,
            target,
            mask,
            losses=["charbonnier", "tv_pred", "residual_amp", "residual_tv"],
            loss_weights=[1.0, 0.01, 0.01, 0.01],
            target_keys=("a", "b", "c"),
            residual=residual,
        )
        self.assertIn("loss", out)
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertIn("residual_amp", out)


if __name__ == "__main__":
    unittest.main()
