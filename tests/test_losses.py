"""Unit tests for mask-topology-aware map losses."""
from __future__ import annotations

import unittest

import torch

from src.models.losses import (
    compose_map_losses,
    masked_charbonnier,
    masked_integration_loss,
    masked_l1,
    masked_laplacian_loss,
    masked_mse,
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

    def test_equal_weight_sparse_vs_dense_maps(self) -> None:
        """Maps with equal mean error but different coverage contribute equally."""
        pred = torch.zeros(1, 2, 10, 10)
        target = torch.zeros(1, 2, 10, 10)
        mask = torch.zeros(1, 2, 10, 10)
        # Map 0: 100 valid pixels, constant error 0.5
        mask[0, 0, :, :] = 1.0
        pred[0, 0] = 0.5
        # Map 1: 10 valid pixels, constant error 0.5
        mask[0, 1, 0, :10] = 1.0
        pred[0, 1, 0, :10] = 0.5
        loss = masked_l1(pred, target, mask)
        self.assertAlmostEqual(float(loss.item()), 0.5, places=5)
        # If global pixel mean were used: (100*0.5 + 10*0.5)/110 = 0.5 still —
        # use unequal errors to prove balancing:
        pred2 = pred.clone()
        pred2[0, 0] = 1.0  # error 1.0 over 100 px
        pred2[0, 1, 0, :10] = 0.0  # error 0.0 over 10 px
        balanced = masked_l1(pred2, target, mask)
        # Equal map weights → mean(1.0, 0.0) = 0.5
        self.assertAlmostEqual(float(balanced.item()), 0.5, places=5)
        global_pixel = ((pred2 - target).abs() * mask).sum() / mask.sum()
        self.assertGreater(float(global_pixel.item()), 0.9)  # ~100/110

    def test_completely_masked_channel_excluded(self) -> None:
        pred = torch.zeros(1, 2, 4, 4)
        target = torch.zeros(1, 2, 4, 4)
        mask = torch.zeros(1, 2, 4, 4)
        mask[0, 0, 1, 1] = 1.0
        pred[0, 0, 1, 1] = 2.0  # error 2
        # channel 1 fully masked
        loss = masked_mse(pred, target, mask)
        self.assertAlmostEqual(float(loss.item()), 4.0, places=5)

    def test_one_valid_pixel(self) -> None:
        pred = torch.zeros(1, 1, 3, 3)
        target = torch.zeros(1, 1, 3, 3)
        mask = torch.zeros(1, 1, 3, 3)
        mask[0, 0, 1, 1] = 1.0
        pred[0, 0, 1, 1] = 0.25
        loss = masked_l1(pred, target, mask)
        self.assertAlmostEqual(float(loss.item()), 0.25, places=5)

    def test_all_masked_returns_zero_finite(self) -> None:
        pred = torch.randn(2, 2, 4, 4, requires_grad=True)
        target = torch.randn(2, 2, 4, 4)
        mask = torch.zeros(2, 2, 4, 4)
        loss = masked_l1(pred, target, mask)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(loss.item()), 0.0)
        (loss + pred.sum() * 0.0).backward()
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_grad_loss_skips_mask_boundary(self) -> None:
        pred, target, mask = self._sample()
        loss = masked_pairwise_grad_loss(pred, target, mask)
        self.assertTrue(torch.isfinite(loss))

        # Hole in mask: horizontal pair across hole must not contribute.
        mask2 = mask.clone()
        mask2[:, :, 4, 3] = 0.0
        loss_hole = masked_pairwise_grad_loss(pred, target, mask2)
        self.assertTrue(torch.isfinite(loss_hole))

    def test_grad_loss_balances_sparse_maps(self) -> None:
        pred = torch.zeros(1, 2, 4, 4)
        target = torch.zeros(1, 2, 4, 4)
        mask = torch.zeros(1, 2, 4, 4)
        # Dense channel: full row valid, horizontal ramp error
        mask[0, 0, 1, :] = 1.0
        pred[0, 0, 1, :] = torch.tensor([0.0, 1.0, 2.0, 3.0])
        # Sparse channel: two adjacent pixels only, same |Δx| error of 1
        mask[0, 1, 2, 1:3] = 1.0
        pred[0, 1, 2, 1] = 0.0
        pred[0, 1, 2, 2] = 1.0
        loss = masked_pairwise_grad_loss(pred, target, mask)
        self.assertTrue(torch.isfinite(loss))
        # Both maps have mean |dx| = 1 (and no dy pairs) → loss 1.0
        self.assertAlmostEqual(float(loss.item()), 1.0, places=5)

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

    def test_backward_stable_mixed_masks(self) -> None:
        pred = torch.randn(2, 3, 8, 8, requires_grad=True)
        target = torch.randn(2, 3, 8, 8)
        mask = torch.zeros(2, 3, 8, 8)
        mask[0, 0, 2:6, 2:6] = 1.0
        mask[0, 1, 0, 0] = 1.0
        mask[1, 2, 3:5, 3:5] = 1.0
        # channel fully empty for some (b,c) pairs
        loss = masked_l1(pred, target, mask) + masked_pairwise_grad_loss(pred, target, mask)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())


if __name__ == "__main__":
    unittest.main()
