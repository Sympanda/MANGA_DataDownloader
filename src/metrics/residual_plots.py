"""Batch-forward evaluation + residual diagnostic plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def _masked_stats(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    m = mask > 0
    if m.sum() == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "pearson": float("nan")}
    err = pred[m] - target[m]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    if pred[m].std() < 1e-12 or target[m].std() < 1e-12:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(pred[m], target[m])[0, 1])
    return {"mae": mae, "rmse": rmse, "pearson": pearson}


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = dict(batch)
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
    if "inputs" in batch:
        inputs = dict(batch["inputs"])
        for key in ("sdss_imaging", "legacy_imaging", "hr_imaging"):
            if key in inputs and torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].to(device)
        if "spectrum" in inputs:
            spec = dict(inputs["spectrum"])
            for sk in ("wave", "flux", "ivar"):
                if sk in spec and torch.is_tensor(spec[sk]):
                    spec[sk] = spec[sk].to(device)
            if "is_real_sdss_fiber" in spec and torch.is_tensor(spec["is_real_sdss_fiber"]):
                spec["is_real_sdss_fiber"] = spec["is_real_sdss_fiber"].to(device)
            inputs["spectrum"] = spec
        out["inputs"] = inputs
    if "targets" in batch:
        out["targets"] = {k: v.to(device) for k, v in batch["targets"].items()}
    if "target_loss_masks" in batch:
        out["target_loss_masks"] = {
            k: v.to(device) for k, v in batch["target_loss_masks"].items()
        }
    if "footprint_mask" in batch:
        out["footprint_mask"] = batch["footprint_mask"].to(device)
    return out


@torch.no_grad()
def evaluate_batch_forward_predictions(
    model,
    dataloader,
    *,
    device: torch.device,
    map_keys: tuple[str, ...],
    plots_dir: Path,
    split: str,
    max_plot: int = 8,
) -> list[dict[str, float | str]]:
    """Evaluate any wrapper that returns ``(pred_dict, loss_dict)`` from ``forward(batch)``.

    Pixel / point-estimate models use the same panel layout as UNet
    (``plot_map_prediction_panel``). Residual models use the residual diagnostic grid.
    """
    from src.metrics.plots import plot_map_prediction_panel

    model.eval()
    rows: list[dict[str, float | str]] = []
    plotted = 0
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    for batch in dataloader:
        batch = _move_batch_to_device(batch, device)
        plateifus = batch["plateifu"]
        pred_dict, _ = model(batch)
        pred = pred_dict["maps"]
        targets = pred_dict["targets"]
        masks = pred_dict["masks"]
        is_residual = "base_maps" in pred_dict

        for i, plateifu in enumerate(plateifus):
            per_map_mse = []
            for ch, key in enumerate(map_keys):
                m = masks[i, ch] > 0
                if m.sum() == 0:
                    per_map_mse.append(float("nan"))
                else:
                    per_map_mse.append(
                        float(((pred[i, ch][m] - targets[i, ch][m]) ** 2).mean().cpu())
                    )
            sample_mse = float(np.nanmean(per_map_mse))
            rows.append(
                {
                    "plateifu": str(plateifu),
                    "split": split,
                    "mse_all": sample_mse,
                    **{f"mse_{k}": float(v) for k, v in zip(map_keys, per_map_mse)},
                }
            )
            if plotted >= max_plot:
                continue

            out_path = plots_dir / f"{split}_{str(plateifu).replace('-', '_')}.png"
            if is_residual:
                plot_residual_diagnostic_panel(
                    plateifu=str(plateifu),
                    pred_dict={
                        k: (v[i] if torch.is_tensor(v) and v.ndim >= 3 else v)
                        for k, v in pred_dict.items()
                    },
                    map_keys=map_keys,
                    out_path=out_path,
                    # Only show uncertainty when the model actually produced it.
                    show_uncertainty=("residual_sigma" in pred_dict)
                    or ("predictive_std" in pred_dict),
                )
            else:
                sdss = None
                sdss_bands: tuple[str, ...] | None = None
                if getattr(model.config, "use_sdss", False) and "inputs" in batch:
                    sdss = batch["inputs"]["sdss_imaging"][i].detach().cpu().numpy()
                    raw_bands = batch["inputs"].get("sdss_imaging_bands")
                    if raw_bands is not None:
                        sdss_bands = tuple(str(b) for b in raw_bands)
                footprint = None
                if "footprint_mask" in batch:
                    footprint = batch["footprint_mask"][i].detach().cpu().numpy()
                plot_map_prediction_panel(
                    plateifu=str(plateifu),
                    sdss=sdss,
                    sdss_band_names=sdss_bands,
                    footprint_mask=footprint,
                    target=targets[i].detach().cpu().numpy(),
                    pred=pred[i].detach().cpu().numpy(),
                    mask=masks[i].detach().cpu().numpy(),
                    map_keys=map_keys,
                    out_path=out_path,
                )
            plotted += 1
    return rows


def plot_residual_diagnostic_panel(
    *,
    plateifu: str,
    pred_dict: dict[str, object],
    map_keys: tuple[str, ...],
    out_path: Path,
    sample_maps: np.ndarray | None = None,
    show_uncertainty: bool = False,
) -> None:
    """
    Diagnostic grid per map channel:

        Target | Base UNet | Final pred | True residual | Pred residual
        [optional: Uncertainty] [optional: Sample k ...]

    Target / Base / Final share one scale (same as UNet panels: 0–1).
    True / Pred residual share one symmetric divergent scale per channel.
    Uncertainty is off by default (enable for Gaussian / diffusion runs).
    """
    from src.metrics.plots import DIFF_VMIN, DIFF_VMAX, MAP_VMIN, MAP_VMAX

    def _np(key: str) -> np.ndarray | None:
        v = pred_dict.get(key)
        if v is None:
            return None
        if torch.is_tensor(v):
            return v.detach().float().cpu().numpy()
        return np.asarray(v)

    def _shared_resid_limit(*arrs: np.ndarray | None, m: np.ndarray | None) -> float:
        vals: list[float] = []
        for a in arrs:
            if a is None:
                continue
            show = np.where(m, a, np.nan) if m is not None else a
            finite = show[np.isfinite(show)]
            if finite.size == 0:
                continue
            vals.append(float(np.nanpercentile(np.abs(finite), 98)))
        vmax = max(vals) if vals else 0.15
        return max(vmax, 1e-4)

    target = _np("targets")
    pred = _np("maps")
    mask = _np("masks")
    base = _np("base_maps")
    resid_t = _np("residual_target")
    resid_p = _np("residual_prediction")
    unc = _np("predictive_std")
    if unc is None:
        unc = _np("residual_sigma")

    n_maps = len(map_keys)
    show_unc = bool(show_uncertainty and unc is not None)
    show_samples = sample_maps is not None and sample_maps.ndim == 4
    n_sample_cols = min(4, sample_maps.shape[0]) if show_samples else 0

    titles = ["Target", "Base UNet", "Final pred", "True residual", "Pred residual"]
    if show_unc:
        titles.append("Uncertainty")
    if show_samples:
        titles += [f"Sample {k}" for k in range(n_sample_cols)]

    ncols = len(titles)
    fig, axes = plt.subplots(n_maps, ncols, figsize=(2.4 * ncols, 2.5 * n_maps), squeeze=False)

    for ch, key in enumerate(map_keys):
        m = None if mask is None else (mask[ch] > 0)
        tgt_ch = None if target is None else target[ch]
        base_ch = None if base is None else base[ch]
        pred_ch = None if pred is None else pred[ch]
        rt_ch = None if resid_t is None else resid_t[ch]
        rp_ch = None if resid_p is None else resid_p[ch]
        resid_lim = _shared_resid_limit(rt_ch, rp_ch, m=m)
        # Prefer shared data-driven residual limit; fall back to UNet diff limits if tiny.
        if resid_lim < 1e-3:
            r_vmin, r_vmax = DIFF_VMIN, DIFF_VMAX
        else:
            r_vmin, r_vmax = -resid_lim, resid_lim

        panels: list[tuple[str, np.ndarray | None, str]] = [
            ("Target", tgt_ch, "map"),
            ("Base UNet", base_ch, "map"),
            ("Final pred", pred_ch, "map"),
            ("True residual", rt_ch, "resid"),
            ("Pred residual", rp_ch, "resid"),
        ]
        if show_unc:
            panels.append(("Uncertainty", None if unc is None else unc[ch], "unc"))
        if show_samples:
            for k in range(n_sample_cols):
                panels.append((f"Sample {k}", sample_maps[k, ch], "resid"))

        for j, (title, img, kind) in enumerate(panels):
            ax = axes[ch, j]
            ax.set_xticks([])
            ax.set_yticks([])
            if ch == 0:
                ax.set_title(title, fontsize=9)
            if j == 0:
                ax.set_ylabel(key, fontsize=9)
            if img is None:
                ax.set_facecolor("#ddd")
                continue
            show = img.astype(np.float32, copy=True)
            if m is not None and kind != "unc":
                show = np.where(m, show, np.nan)

            if kind == "map":
                im = ax.imshow(
                    show,
                    origin="lower",
                    cmap="viridis",
                    vmin=MAP_VMIN,
                    vmax=MAP_VMAX,
                )
            elif kind == "unc":
                u_hi = float(np.nanpercentile(show[np.isfinite(show)], 98)) if np.isfinite(show).any() else 0.15
                im = ax.imshow(
                    show,
                    origin="lower",
                    cmap="magma",
                    vmin=0.0,
                    vmax=max(u_hi, 1e-4),
                )
            else:
                im = ax.imshow(
                    show,
                    origin="lower",
                    cmap="coolwarm",
                    vmin=r_vmin,
                    vmax=r_vmax,
                )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(plateifu)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def coverage_within_percentile(
    samples: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    lo: float = 0.16,
    hi: float = 0.84,
) -> float:
    """Empirical fraction of valid pixels where target lies in [q_lo, q_hi] of samples."""
    m = mask > 0
    if m.sum() == 0:
        return float("nan")
    q_lo = np.quantile(samples, lo, axis=0)
    q_hi = np.quantile(samples, hi, axis=0)
    inside = (target >= q_lo) & (target <= q_hi)
    return float(inside[m].mean())
