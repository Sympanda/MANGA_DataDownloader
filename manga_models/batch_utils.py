from __future__ import annotations

import torch
import torch.nn.functional as F

from manga_prep.targets.pipe3d_maps import AMARA_TARGET_KEYS
from manga_models.config import ConditionalUNetConfig

_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
_SOBEL_Y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])


def _nan_to_num(t: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


def prepare_spatial_input(
    batch: dict[str, object],
    config: ConditionalUNetConfig,
) -> torch.Tensor:
    """
    Build raw (unnormalized) spatial input tensor from a collated batch.

    Notebook percentile scaling is display-only; training uses raw aligned flux.
    """
    parts: list[torch.Tensor] = []

    inputs = batch.get("inputs", {})
    if config.use_sdss:
        if "sdss_imaging" not in inputs:
            raise KeyError("Batch missing sdss_imaging but config.use_sdss=True")
        parts.append(_nan_to_num(inputs["sdss_imaging"].float()))

    if config.use_legacy:
        if "legacy_imaging" not in inputs:
            raise KeyError("Batch missing legacy_imaging but config.use_legacy=True")
        parts.append(_nan_to_num(inputs["legacy_imaging"].float()))

    if config.use_footprint_mask:
        if "footprint_mask" not in batch:
            raise KeyError("Batch missing footprint_mask")
        fp = batch["footprint_mask"].float().unsqueeze(1)
        parts.append(fp)

    return torch.cat(parts, dim=1)


def prepare_spectrum_input(
    batch: dict[str, object],
    config: ConditionalUNetConfig,
) -> torch.Tensor | None:
    if not config.use_spectrum:
        return None
    inputs = batch.get("inputs", {})
    if "spectrum" not in inputs:
        raise KeyError("Batch missing spectrum but config.use_spectrum=True")
    return _nan_to_num(inputs["spectrum"]["flux"].float())


def prepare_targets_and_masks(
    batch: dict[str, object],
    config: ConditionalUNetConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    keys = config.target_keys
    targets = torch.stack([batch["targets"][key].float() for key in keys], dim=1)
    masks = torch.stack([batch["target_loss_masks"][key].float() for key in keys], dim=1)
    targets = _nan_to_num(targets)
    return targets, masks


def masked_mse_loss_multichannel(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """pred/target/mask: (B, C, H, W). Mask is per-channel."""
    mask = loss_mask.to(dtype=pred.dtype)
    if mask.ndim == pred.ndim - 1:
        mask = mask.unsqueeze(1)
    # Targets are NaN outside the analysis footprint; masked pixels must not
    # contribute (NaN * 0 is still NaN in PyTorch).
    diff2 = (pred - target) ** 2
    masked = torch.where(mask > 0, diff2, torch.zeros_like(diff2))
    return masked.sum() / mask.sum().clamp_min(eps)


def masked_l1_loss_multichannel(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """pred/target/mask: (B, C, H, W). Mask is per-channel."""
    mask = loss_mask.to(dtype=pred.dtype)
    if mask.ndim == pred.ndim - 1:
        mask = mask.unsqueeze(1)
    diff = (pred - target).abs()
    masked = torch.where(mask > 0, diff, torch.zeros_like(diff))
    return masked.sum() / mask.sum().clamp_min(eps)


def _sobel_gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Per-channel gradient magnitude, shape (B, C, H, W)."""
    b, c, _, _ = x.shape
    device, dtype = x.device, x.dtype
    kx = _SOBEL_X.to(device=device, dtype=dtype).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    ky = _SOBEL_Y.to(device=device, dtype=dtype).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    gx = F.conv2d(x, kx, padding=1, groups=c)
    gy = F.conv2d(x, ky, padding=1, groups=c)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def masked_gradient_loss_multichannel(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Match spatial gradients inside the loss mask (encourages sharper maps)."""
    mask = loss_mask.to(dtype=pred.dtype)
    if mask.ndim == pred.ndim - 1:
        mask = mask.unsqueeze(1)
    pred_g = _sobel_gradient_magnitude(pred)
    target_g = _sobel_gradient_magnitude(target)
    diff = (pred_g - target_g).abs()
    masked = torch.where(mask > 0, diff, torch.zeros_like(diff))
    return masked.sum() / mask.sum().clamp_min(eps)


def compute_map_training_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    config: ConditionalUNetConfig,
) -> torch.Tensor:
    """Combined masked loss: MSE + L1 + gradient terms (weights from config)."""
    total = pred.new_tensor(0.0)
    if config.loss_mse_weight > 0:
        total = total + config.loss_mse_weight * masked_mse_loss_multichannel(pred, target, loss_mask)
    if config.loss_l1_weight > 0:
        total = total + config.loss_l1_weight * masked_l1_loss_multichannel(pred, target, loss_mask)
    if config.loss_grad_weight > 0:
        total = total + config.loss_grad_weight * masked_gradient_loss_multichannel(pred, target, loss_mask)
    if config.loss_mse_weight <= 0 and config.loss_l1_weight <= 0 and config.loss_grad_weight <= 0:
        raise ValueError("At least one loss weight must be > 0")
    return total
