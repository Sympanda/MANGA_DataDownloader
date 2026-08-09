from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

# Flux map keys where spatial integration should be conserved.
FLUX_INTEGRATION_KEYS = ("ha_flux", "hbeta_flux", "oiii_5007_flux", "nii_6584_flux")

_LAPLACE = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])


def _ensure_bchw(mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Broadcast mask to (B, C, H, W) matching ref."""
    mask = mask.to(dtype=ref.dtype, device=ref.device)
    if mask.ndim == ref.ndim - 1:
        mask = mask.unsqueeze(1)
    if mask.shape[0] == 1 and ref.shape[0] > 1:
        mask = mask.expand(ref.shape[0], -1, -1, -1)
    if mask.shape[1] == 1 and ref.shape[1] > 1:
        mask = mask.expand(-1, ref.shape[1], -1, -1)
    return mask


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """
    Per-(batch, channel) masked spatial mean, then mean over active maps.

    Each galaxy×physical-map with ≥1 valid pixel contributes equally; empty
    (B, C) pairs are excluded entirely (not counted as zero loss).
    """
    m = (mask > 0).to(dtype=x.dtype)
    pixel_sum = (x * m).sum(dim=(-2, -1))
    valid_count = m.sum(dim=(-2, -1))
    per_map = pixel_sum / valid_count.clamp_min(eps)
    active = valid_count > 0
    if not bool(active.any()):
        return x.new_tensor(0.0)
    return per_map[active].mean()


def _per_map_masked_mean(
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (per_map_mean[B,C], active[B,C] bool)."""
    m = (mask > 0).to(dtype=x.dtype)
    valid_count = m.sum(dim=(-2, -1))
    per_map = (x * m).sum(dim=(-2, -1)) / valid_count.clamp_min(eps)
    return per_map, valid_count > 0


def pairwise_valid_masks(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
  Per-channel valid neighbour pairs for horizontal / vertical finite differences.

  valid_x[b,c,y,x] is True only if mask[b,c,y,x] and mask[b,c,y,x+1] are valid.
  valid_y[b,c,y,x] is True only if mask[b,c,y,x] and mask[b,c,y+1,x] are valid.
  """
    m = mask > 0
    valid_x = m[:, :, :, 1:] & m[:, :, :, :-1]
    valid_y = m[:, :, 1:, :] & m[:, :, :-1, :]
    return valid_x, valid_y


def _safe_target(target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replace invalid / NaN target pixels so they never poison masked reductions."""
    m = mask > 0
    clean = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.where(m, clean, torch.zeros_like(clean))


def masked_charbonnier(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    eps: float = 1e-3,
) -> torch.Tensor:
    mask = _ensure_bchw(loss_mask, pred)
    tgt = _safe_target(target, mask)
    diff = pred - tgt
    loss = torch.sqrt(diff * diff + eps * eps)
    return _masked_mean(loss, mask)


def masked_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    mask = _ensure_bchw(loss_mask, pred)
    tgt = _safe_target(target, mask)
    return _masked_mean((pred - tgt).abs(), mask)


def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    mask = _ensure_bchw(loss_mask, pred)
    tgt = _safe_target(target, mask)
    return _masked_mean((pred - tgt) ** 2, mask)


def masked_pairwise_grad_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Supervised gradient loss on valid horizontal/vertical neighbours only.

    X- and Y-pair counts are independent. Each (galaxy, map) gets
    mean(|Δx|) + mean(|Δy|) over its valid pairs, then those map losses
    are averaged equally.
    """
    mask = _ensure_bchw(loss_mask, pred)
    tgt = _safe_target(target, mask)
    valid_x, valid_y = pairwise_valid_masks(mask)

    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dx_true = tgt[:, :, :, 1:] - tgt[:, :, :, :-1]
    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dy_true = tgt[:, :, 1:, :] - tgt[:, :, :-1, :]

    loss_dx, active_x = _per_map_masked_mean((dx_pred - dx_true).abs(), valid_x)
    loss_dy, active_y = _per_map_masked_mean((dy_pred - dy_true).abs(), valid_y)
    combined = pred.new_zeros(loss_dx.shape)
    combined = torch.where(active_x, combined + loss_dx, combined)
    combined = torch.where(active_y, combined + loss_dy, combined)
    active = active_x | active_y
    if not bool(active.any()):
        return pred.new_tensor(0.0)
    return combined[active].mean()


def _per_channel_conv2d(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    b, c, _, _ = x.shape
    k = kernel.to(device=x.device, dtype=x.dtype).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    return F.conv2d(x, k, padding=1, groups=c)


def _laplacian_support_mask(mask: torch.Tensor) -> torch.Tensor:
    """True where centre pixel and full 3×3 neighbourhood are valid."""
    m = (_ensure_bchw(mask, mask) > 0).to(dtype=torch.float32)
    b, c, _, _ = m.shape
    kernel = torch.ones(1, 1, 3, 3, device=m.device, dtype=m.dtype)
    k = kernel.repeat(c, 1, 1, 1)
    support = F.conv2d(m, k, padding=1, groups=c)
    return support >= 9.0 - 1e-6


def masked_laplacian_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Laplacian match only where the full 3×3 stencil lies inside the valid mask."""
    mask = _ensure_bchw(loss_mask, pred)
    tgt = _safe_target(target, mask)
    valid = _laplacian_support_mask(mask)
    lap_p = _per_channel_conv2d(pred, _LAPLACE)
    lap_t = _per_channel_conv2d(tgt, _LAPLACE)
    return _masked_mean((lap_p - lap_t).abs(), valid)


def prediction_tv_loss(pred: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """
    Weak total-variation regulariser on the prediction (unsupervised).

    Penalises |Δpred| only across valid neighbour pairs, balanced per (B, C).
    """
    mask = _ensure_bchw(loss_mask, pred)
    valid_x, valid_y = pairwise_valid_masks(mask)
    tv_x = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs()
    tv_y = (pred[:, :, 1:, :] - pred[:, :, :-1, :]).abs()
    loss_x, active_x = _per_map_masked_mean(tv_x, valid_x)
    loss_y, active_y = _per_map_masked_mean(tv_y, valid_y)
    combined = pred.new_zeros(loss_x.shape)
    combined = torch.where(active_x, combined + loss_x, combined)
    combined = torch.where(active_y, combined + loss_y, combined)
    active = active_x | active_y
    if not bool(active.any()):
        return pred.new_tensor(0.0)
    return combined[active].mean()


def residual_amplitude_loss(residual: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """L1 penalty on the coarse/fine residual branch inside the valid mask."""
    mask = _ensure_bchw(loss_mask, residual)
    return _masked_mean(residual.abs(), mask)


def residual_tv_loss(residual: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Pairwise TV on the residual branch (valid neighbours only)."""
    return prediction_tv_loss(residual, loss_mask)


def masked_integration_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    channel_indices: list[int],
    normalize: str = "mean",
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Masked spatial-sum consistency per selected channel.

    Sums only valid pixels for that channel's mask. Normalisation keeps the term
    on a similar scale to pixel losses (avoids AMP / grad blow-ups from raw sums):

      mean         — |sum(pred) - sum(target)| / N_valid  (default, stable)
      relative_sum — |sum(pred) - sum(target)| / max(|sum(target)|, N_valid * eps)
      raw          — |sum(pred) - sum(target)|  (large; not recommended)
    """
    if not channel_indices:
        return pred.new_tensor(0.0)

    mask = _ensure_bchw(loss_mask, pred)
    losses = []
    for ch in channel_indices:
        m = mask[:, ch : ch + 1]
        tgt = _safe_target(target[:, ch : ch + 1], m)
        valid_n = m.sum(dim=(-2, -1)).clamp_min(0.0)
        p_sum = (pred[:, ch : ch + 1] * m).sum(dim=(-2, -1))
        t_sum = (tgt * m).sum(dim=(-2, -1))
        delta = (p_sum - t_sum).abs()
        if normalize == "raw":
            err = delta
        elif normalize == "relative_sum":
            ref = torch.maximum(t_sum.abs(), valid_n.clamp_min(1.0) * eps)
            err = delta / ref
        else:
            err = delta / valid_n.clamp_min(1.0)
        active = (valid_n > 0).to(dtype=err.dtype)
        losses.append((err * active).sum() / active.sum().clamp_min(1.0))
    return torch.stack(losses).mean()


def masked_gaussian_nll(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    min_log_var: float = -6.0,
    max_log_var: float = 6.0,
) -> torch.Tensor:
    """Masked Gaussian negative log-likelihood per valid spaxel (log_var = log σ²)."""
    mask = _ensure_bchw(loss_mask, mu)
    tgt = _safe_target(target, mask)
    log_var = log_var.clamp(min=min_log_var, max=max_log_var)
    inv_var = torch.exp(-log_var)
    sq_err = (mu - tgt) ** 2
    nll = 0.5 * (log_var + sq_err * inv_var)
    return _masked_mean(nll, mask)


def masked_fft_power_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Match 2D log power spectra after zeroing invalid pixels.

    Encourages high-frequency spatial power that pure L1 tends to wash out.
    Each (batch, channel) map with ≥1 valid pixel contributes equally.
    """
    mask = _ensure_bchw(loss_mask, pred)
    tgt = _safe_target(target, mask)
    m = (mask > 0).to(dtype=pred.dtype)
    pred_m = pred * m
    tgt_m = tgt * m

    fp = torch.fft.rfft2(pred_m)
    ft = torch.fft.rfft2(tgt_m)
    pp = (fp.real.square() + fp.imag.square()).clamp_min(eps)
    pt = (ft.real.square() + ft.imag.square()).clamp_min(eps)
    err = (pp.log() - pt.log()).abs()
    # Mean over frequency bins per map, then equal average over active maps.
    per_map = err.mean(dim=(-2, -1))
    active = m.sum(dim=(-2, -1)) > 0
    if not bool(active.any()):
        return pred.new_tensor(0.0)
    return per_map[active].mean()


LOSS_REGISTRY: dict[str, Callable] = {
    "charbonnier": masked_charbonnier,
    "l1": masked_l1,
    "mse": masked_mse,
    "grad": masked_pairwise_grad_loss,
    "laplacian": masked_laplacian_loss,
    "fft_power": masked_fft_power_loss,
    "tv_pred": prediction_tv_loss,
    "residual_amp": residual_amplitude_loss,
    "residual_tv": residual_tv_loss,
    "gaussian_nll": masked_gaussian_nll,
}


def compose_map_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    losses: list[str],
    loss_weights: list[float],
    loss_params: dict[str, dict] | None = None,
    target_keys: tuple[str, ...],
    integration_channel_keys: tuple[str, ...] = FLUX_INTEGRATION_KEYS,
    residual: torch.Tensor | None = None,
    log_var: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """
    Compose weighted map losses.

    Supervised terms (charbonnier, grad, laplacian, integration) use targets only
    on valid mask pixels. Regularisers (tv_pred, residual_*) never use targets
  outside the mask.
    """
    params = loss_params or {}
    out: dict[str, torch.Tensor] = {}
    total = pred.new_tensor(0.0)

    for name, weight in zip(losses, loss_weights):
        if weight <= 0:
            continue
        if name == "integration":
            ch_idx = [i for i, k in enumerate(target_keys) if k in integration_channel_keys]
            val = masked_integration_loss(
                pred,
                target,
                loss_mask,
                channel_indices=ch_idx,
                **params.get("integration", {}),
            )
        elif name == "gaussian_nll":
            if log_var is None:
                raise ValueError("gaussian_nll requires log_var from a gaussian output head")
            val = masked_gaussian_nll(
                pred,
                log_var,
                target,
                loss_mask,
                **params.get("gaussian_nll", {}),
            )
        elif name in ("tv_pred",):
            val = prediction_tv_loss(pred, loss_mask)
        elif name in ("residual_amp", "residual_tv"):
            if residual is None:
                continue
            fn = LOSS_REGISTRY[name]
            val = fn(residual, loss_mask, **params.get(name, {}))
        elif name in LOSS_REGISTRY:
            val = LOSS_REGISTRY[name](pred, target, loss_mask, **params.get(name, {}))
        else:
            raise ValueError(f"Unknown loss: {name!r}")
        out[name] = val
        total = total + float(weight) * val

    if total.item() == 0.0 and not out:
        raise ValueError("At least one loss weight must be > 0")
    out["loss"] = total
    return out
