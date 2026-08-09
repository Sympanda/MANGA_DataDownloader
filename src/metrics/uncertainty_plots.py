from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.metrics.plots import (
    DIFF_VMAX,
    DIFF_VMIN,
    MAP_VMAX,
    MAP_VMIN,
    SIGMA_VMAX,
    SIGMA_VMIN,
    _percentile_norm,
    write_metrics_csv,
)

__all__ = [
    "evaluate_uncertainty_predictions",
    "evaluate_ensemble_predictions",
    "write_ensemble_summary_plots",
    "write_metrics_csv",
    "plot_uncertainty_map_panel",
    "SIGMA_VMIN",
    "SIGMA_VMAX",
]


def _sigma_tot_from_components(sigma_epi: np.ndarray, sigma_ale: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(sigma_epi**2 + sigma_ale**2, 0.0))


def _coverage(errors: np.ndarray, sigma: np.ndarray, *, k: float) -> float:
    valid = np.isfinite(errors) & np.isfinite(sigma) & (sigma > 0)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.abs(errors[valid]) <= k * sigma[valid]))


def _gaussian_nll_sample(errors: np.ndarray, sigma: np.ndarray) -> float:
    valid = np.isfinite(errors) & np.isfinite(sigma) & (sigma > 0)
    if not np.any(valid):
        return float("nan")
    var = np.maximum(sigma[valid] ** 2, 1e-8)
    nll = 0.5 * (np.log(2 * math.pi * var) + (errors[valid] ** 2) / var)
    return float(np.mean(nll))


def _calibration_bins(errors: np.ndarray, sigma: np.ndarray, *, n_bins: int = 10) -> list[dict[str, float]]:
    valid = np.isfinite(errors) & np.isfinite(sigma) & (sigma > 0)
    if not np.any(valid):
        return []
    err = errors[valid]
    sig = sigma[valid]
    edges = np.quantile(sig, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    rows: list[dict[str, float]] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        m = (sig >= lo) & (sig < hi) if hi < edges[-1] else (sig >= lo) & (sig <= hi)
        if not np.any(m):
            continue
        rows.append(
            {
                "sigma_bin_lo": float(lo),
                "sigma_bin_hi": float(hi),
                "mean_sigma": float(np.mean(sig[m])),
                "rmse": float(np.sqrt(np.mean(err[m] ** 2))),
                "n_pixels": float(m.sum()),
            }
        )
    return rows


def plot_uncertainty_map_panel(
    *,
    plateifu: str,
    sdss: np.ndarray | None,
    sdss_band_names: tuple[str, ...] | None,
    footprint_mask: np.ndarray | None,
    target: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    map_keys: tuple[str, ...],
    out_path: Path,
    sigma_tot: np.ndarray | None = None,
    sigma_secondary: np.ndarray | None = None,
    secondary_label: str = "σ_epi",
    epoch: int | None = None,
) -> None:
    n_maps = len(map_keys)
    n_sdss = sdss.shape[0] if sdss is not None and sdss.ndim == 3 else 0
    n_col0 = n_sdss + (1 if footprint_mask is not None else 0)
    n_rows = max(n_maps, n_col0)

    # Columns: SDSS/fp | target | pred | diff | [σ_tot] | [σ_sec] | pred (full, unmasked)
    n_map_cols = 3 + int(sigma_tot is not None) + int(sigma_secondary is not None) + 1
    n_cols = 1 + n_map_cols
    col_tgt, col_pred, col_diff = 1, 2, 3
    col_next = 4
    col_sig_tot = col_sig_sec = col_full = None
    if sigma_tot is not None:
        col_sig_tot = col_next
        col_next += 1
    if sigma_secondary is not None:
        col_sig_sec = col_next
        col_next += 1
    col_full = col_next

    fig = plt.figure(figsize=(3.0 * n_cols, 3.0 * n_rows), dpi=150)
    gs = fig.add_gridspec(n_rows, n_cols, wspace=0.28, hspace=0.32)

    col0_row = 0
    if n_sdss > 0:
        bands = sdss_band_names or tuple(f"b{i}" for i in range(n_sdss))
        for b in range(n_sdss):
            ax = fig.add_subplot(gs[col0_row, 0])
            ax.imshow(_percentile_norm(sdss[b]), origin="lower", cmap="gray", vmin=0, vmax=1)
            label = bands[b] if b < len(bands) else f"b{b}"
            ax.set_title(f"SDSS {label}")
            ax.set_xticks([])
            ax.set_yticks([])
            col0_row += 1

    if footprint_mask is not None:
        ax = fig.add_subplot(gs[col0_row, 0])
        ax.imshow(footprint_mask.astype(np.float32), origin="lower", cmap="gray", vmin=0, vmax=1)
        ax.set_title("IFU footprint")
        ax.set_xticks([])
        ax.set_yticks([])

    for row in range(n_maps):
        key = map_keys[row]
        tgt = target[row]
        prd = pred[row]
        m = mask[row].astype(bool)
        diff = np.where(m, prd - tgt, np.nan)

        ax_tgt = fig.add_subplot(gs[row, col_tgt])
        im_tgt = ax_tgt.imshow(
            np.where(m, tgt, np.nan), origin="lower", cmap="viridis", vmin=MAP_VMIN, vmax=MAP_VMAX
        )
        ax_tgt.set_title(f"{key} target")
        ax_tgt.set_xticks([])
        ax_tgt.set_yticks([])
        fig.colorbar(im_tgt, ax=ax_tgt, fraction=0.046)

        ax_pred = fig.add_subplot(gs[row, col_pred])
        im_prd = ax_pred.imshow(
            np.where(m, prd, np.nan), origin="lower", cmap="viridis", vmin=MAP_VMIN, vmax=MAP_VMAX
        )
        ax_pred.set_title(f"{key} pred")
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])
        fig.colorbar(im_prd, ax=ax_pred, fraction=0.046)

        ax_diff = fig.add_subplot(gs[row, col_diff])
        im_diff = ax_diff.imshow(diff, origin="lower", cmap="coolwarm", vmin=DIFF_VMIN, vmax=DIFF_VMAX)
        ax_diff.set_title("diff")
        ax_diff.set_xticks([])
        ax_diff.set_yticks([])
        fig.colorbar(im_diff, ax=ax_diff, fraction=0.046)

        if col_sig_tot is not None and sigma_tot is not None:
            sig = sigma_tot[row]
            ax_sig = fig.add_subplot(gs[row, col_sig_tot])
            im_sig = ax_sig.imshow(
                np.where(m, sig, np.nan),
                origin="lower",
                cmap="magma",
                vmin=SIGMA_VMIN,
                vmax=SIGMA_VMAX,
            )
            ax_sig.set_title("σ_total")
            ax_sig.set_xticks([])
            ax_sig.set_yticks([])
            fig.colorbar(im_sig, ax=ax_sig, fraction=0.046)

        if col_sig_sec is not None and sigma_secondary is not None:
            sig2 = sigma_secondary[row]
            ax_sig2 = fig.add_subplot(gs[row, col_sig_sec])
            im_sig2 = ax_sig2.imshow(
                np.where(m, sig2, np.nan),
                origin="lower",
                cmap="cividis",
                vmin=SIGMA_VMIN,
                vmax=SIGMA_VMAX,
            )
            ax_sig2.set_title(secondary_label)
            ax_sig2.set_xticks([])
            ax_sig2.set_yticks([])
            fig.colorbar(im_sig2, ax=ax_sig2, fraction=0.046)

        ax_full = fig.add_subplot(gs[row, col_full])
        im_full = ax_full.imshow(prd, origin="lower", cmap="viridis", vmin=MAP_VMIN, vmax=MAP_VMAX)
        ax_full.set_title(f"{key} pred (full)")
        ax_full.set_xticks([])
        ax_full.set_yticks([])
        fig.colorbar(im_full, ax=ax_full, fraction=0.046)

    title = plateifu
    if epoch is not None:
        title += f"  epoch={epoch}"
    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _forward_uncertainty_batch(model, batch, device):
    from src.models.wrapper import (
        prepare_footprint_input,
        prepare_hr_imaging_input,
        prepare_imaging_input,
        prepare_redshift_input,
        prepare_spectrum_input,
        prepare_targets_and_masks,
    )

    x = prepare_imaging_input(batch, model.config).to(device)
    x_hr = prepare_hr_imaging_input(batch, model.config)
    if x_hr is not None:
        x_hr = x_hr.to(device)
    footprint = prepare_footprint_input(batch, model.config)
    if footprint is not None:
        footprint = footprint.to(device)
    spec = prepare_spectrum_input(batch, model.config)
    if spec is not None:
        spec = spec.to(device)
    redshift = prepare_redshift_input(batch, model.config)
    if redshift is not None:
        redshift = redshift.to(device)
    targets, masks = prepare_targets_and_masks(batch, model.config)
    targets = targets.to(device)
    masks = masks.to(device)
    pred, aux = model.model(
        x, spectrum_flux=spec, footprint=footprint, x_hr=x_hr, redshift=redshift
    )
    sigma = aux["sigma"]
    return pred, sigma, targets, masks, batch


@torch.no_grad()
def evaluate_uncertainty_predictions(
    model,
    dataloader,
    *,
    device: torch.device,
    map_keys: tuple[str, ...],
    plots_dir: Path,
    split: str,
    max_plot: int = 8,
) -> list[dict[str, float | str]]:
    model.eval()
    rows: list[dict[str, float | str]] = []
    plotted = 0

    for batch in dataloader:
        pred, sigma, targets, masks, batch = _forward_uncertainty_batch(model, batch, device)
        plateifus = batch["plateifu"]

        for i, plateifu in enumerate(plateifus):
            per_map_mse = []
            per_map_nll = []
            all_err = []
            all_sig = []
            for ch, key in enumerate(map_keys):
                m = masks[i, ch] > 0
                if m.sum() == 0:
                    per_map_mse.append(float("nan"))
                    per_map_nll.append(float("nan"))
                    continue
                err = (pred[i, ch][m] - targets[i, ch][m]).cpu().numpy()
                sig = sigma[i, ch][m].cpu().numpy()
                per_map_mse.append(float(np.mean(err**2)))
                per_map_nll.append(_gaussian_nll_sample(err, sig))
                all_err.append(err)
                all_sig.append(sig)

            err_cat = np.concatenate(all_err) if all_err else np.array([])
            sig_cat = np.concatenate(all_sig) if all_sig else np.array([])
            rows.append(
                {
                    "plateifu": str(plateifu),
                    "split": split,
                    "mse_all": float(np.nanmean(per_map_mse)),
                    "nll_all": _gaussian_nll_sample(err_cat, sig_cat),
                    "coverage_1sigma": _coverage(err_cat, sig_cat, k=1.0),
                    "coverage_2sigma": _coverage(err_cat, sig_cat, k=2.0),
                    **{f"mse_{k}": float(v) for k, v in zip(map_keys, per_map_mse)},
                    **{f"nll_{k}": float(v) for k, v in zip(map_keys, per_map_nll)},
                }
            )

            if max_plot <= 0 or plotted < max_plot:
                sdss = None
                sdss_bands: tuple[str, ...] | None = None
                if model.config.use_sdss and "inputs" in batch:
                    sdss = batch["inputs"]["sdss_imaging"][i].cpu().numpy()
                    raw_bands = batch["inputs"].get("sdss_imaging_bands")
                    if raw_bands is not None:
                        sdss_bands = tuple(str(b) for b in raw_bands)
                footprint = batch["footprint_mask"][i].cpu().numpy() if "footprint_mask" in batch else None
                sig_np = sigma[i].cpu().numpy()
                plot_uncertainty_map_panel(
                    plateifu=plateifu,
                    sdss=sdss,
                    sdss_band_names=sdss_bands,
                    footprint_mask=footprint,
                    target=targets[i].cpu().numpy(),
                    pred=pred[i].cpu().numpy(),
                    mask=masks[i].cpu().numpy(),
                    map_keys=map_keys,
                    out_path=plots_dir / f"{split}_{plateifu.replace('-', '_')}.png",
                    sigma_tot=sig_np,
                )
                plotted += 1

    return rows


def write_calibration_csv(rows: list[dict], path: str | Path) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_ensemble_summary_plots(
    rows: list[dict[str, float | str]],
    calib_rows: list[dict[str, float]],
    map_keys: tuple[str, ...],
    out_dir: Path,
) -> None:
    """Pooled test-set summary plots written alongside per-galaxy panels."""
    if not rows:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    cov1 = [float(r["coverage_1sigma"]) for r in rows if np.isfinite(float(r.get("coverage_1sigma", float("nan"))))]
    cov2 = [float(r["coverage_2sigma"]) for r in rows if np.isfinite(float(r.get("coverage_2sigma", float("nan"))))]
    if cov1:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), dpi=150)
        axes[0].hist(cov1, bins=30, color="#4C72B0", alpha=0.85, edgecolor="white")
        axes[0].axvline(0.68, color="gray", ls=":", lw=1.5, label="target 68%")
        axes[0].axvline(float(np.mean(cov1)), color="red", ls="--", lw=1.2, label=f"mean={np.mean(cov1):.2f}")
        axes[0].set_xlabel("Per-galaxy coverage @ 1σ")
        axes[0].set_title("Galaxy coverage (1σ)")
        axes[0].legend(fontsize=8)

        axes[1].hist(cov2, bins=30, color="#55A868", alpha=0.85, edgecolor="white")
        axes[1].axvline(0.95, color="gray", ls=":", lw=1.5, label="target 95%")
        axes[1].axvline(float(np.mean(cov2)), color="red", ls="--", lw=1.2, label=f"mean={np.mean(cov2):.2f}")
        axes[1].set_xlabel("Per-galaxy coverage @ 2σ")
        axes[1].set_title("Galaxy coverage (2σ)")
        axes[1].legend(fontsize=8)
        fig.suptitle(f"Ensemble test coverage (n={len(cov1)} galaxies)")
        fig.tight_layout()
        fig.savefig(out_dir / "summary_coverage_hist.png", bbox_inches="tight")
        plt.close(fig)

    mse_all = [float(r["mse_all"]) for r in rows if np.isfinite(float(r.get("mse_all", float("nan"))))]
    if mse_all:
        mse_all.sort()
        y = np.arange(1, len(mse_all) + 1) / len(mse_all)
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        ax.plot(mse_all, y, lw=2)
        ax.set_xlabel("Per-galaxy MSE (channels pooled)")
        ax.set_ylabel("CDF")
        ax.set_title("Ensemble test error CDF")
        ax.grid(alpha=0.3, linestyle="--")
        fig.tight_layout()
        fig.savefig(out_dir / "summary_mse_cdf.png", bbox_inches="tight")
        plt.close(fig)

    channel_mse = [
        [float(r.get(f"mse_{k}", float("nan"))) for r in rows if np.isfinite(float(r.get(f"mse_{k}", float("nan"))))]
        for k in map_keys
    ]
    if any(len(v) > 0 for v in channel_mse):
        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
        parts = ax.violinplot(channel_mse, showmeans=True, showmedians=True)
        for body in parts["bodies"]:
            body.set_alpha(0.7)
        ax.set_xticks(range(1, len(map_keys) + 1))
        ax.set_xticklabels(list(map_keys), rotation=25, ha="right")
        ax.set_ylabel("Per-galaxy MSE")
        ax.set_title("Ensemble test MSE by channel")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        fig.tight_layout()
        fig.savefig(out_dir / "summary_mse_by_channel.png", bbox_inches="tight")
        plt.close(fig)

    if calib_rows:
        sig = np.array([float(r["mean_sigma"]) for r in calib_rows])
        rmse = np.array([float(r["rmse"]) for r in calib_rows])
        w = np.array([float(r["n_pixels"]) for r in calib_rows])
        order = np.argsort(sig)
        sig, rmse, w = sig[order], rmse[order], w[order]
        fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=150)
        ax.scatter(sig, rmse, s=np.clip(w / w.max() * 120, 8, 120), alpha=0.55, c=sig, cmap="magma")
        lim = max(float(sig.max()), float(rmse.max()), SIGMA_VMAX) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1, label="RMSE = σ")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("Mean predicted σ (bin)")
        ax.set_ylabel("Observed RMSE (bin)")
        ax.set_title("σ vs RMSE (pooled spaxel bins)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, linestyle="--")
        fig.tight_layout()
        fig.savefig(out_dir / "summary_sigma_vs_rmse.png", bbox_inches="tight")
        plt.close(fig)


@torch.no_grad()
def evaluate_ensemble_predictions(
    models: list,
    dataloader,
    *,
    device: torch.device,
    map_keys: tuple[str, ...],
    plots_dir: Path,
    split: str,
    max_plot: int = 8,
    secondary_sigma: str = "epistemic",
) -> tuple[list[dict[str, float | str]], list[dict[str, float]]]:
    """Aggregate member μ/σ; plot σ_total and σ_epi (or σ_ale)."""
    if not models:
        raise ValueError("evaluate_ensemble_predictions requires at least one model")

    rows: list[dict[str, float | str]] = []
    calib_rows: list[dict[str, float]] = []
    plotted = 0
    secondary_label = "σ_epi" if secondary_sigma == "epistemic" else "σ_ale"

    for batch in dataloader:
        plateifus = batch["plateifu"]
        member_preds = []
        member_sigmas = []
        targets = masks = None

        for model in models:
            model.eval()
            pred, sigma, targets, masks, batch = _forward_uncertainty_batch(model, batch, device)
            member_preds.append(pred)
            member_sigmas.append(sigma)

        stacked_mu = torch.stack(member_preds, dim=0)
        stacked_sigma = torch.stack(member_sigmas, dim=0)
        mu_ens = stacked_mu.mean(dim=0)
        sigma_epi = stacked_mu.std(dim=0, unbiased=False)
        sigma_ale = stacked_sigma.mean(dim=0)
        sigma_tot = torch.sqrt(sigma_epi**2 + sigma_ale**2)

        for i, plateifu in enumerate(plateifus):
            per_map_mse = []
            per_map_cov1 = []
            all_err = []
            all_sig = []
            for ch, key in enumerate(map_keys):
                m = masks[i, ch] > 0
                if m.sum() == 0:
                    per_map_mse.append(float("nan"))
                    per_map_cov1.append(float("nan"))
                    continue
                err = (mu_ens[i, ch][m] - targets[i, ch][m]).cpu().numpy()
                sig = sigma_tot[i, ch][m].cpu().numpy()
                per_map_mse.append(float(np.mean(err**2)))
                per_map_cov1.append(_coverage(err, sig, k=1.0))
                all_err.append(err)
                all_sig.append(sig)

            err_cat = np.concatenate(all_err) if all_err else np.array([])
            sig_cat = np.concatenate(all_sig) if all_sig else np.array([])
            rows.append(
                {
                    "plateifu": str(plateifu),
                    "split": split,
                    "mse_all": float(np.nanmean(per_map_mse)),
                    "nll_all": _gaussian_nll_sample(err_cat, sig_cat),
                    "coverage_1sigma": _coverage(err_cat, sig_cat, k=1.0),
                    "coverage_2sigma": _coverage(err_cat, sig_cat, k=2.0),
                    "n_members": float(len(models)),
                    **{f"mse_{k}": float(v) for k, v in zip(map_keys, per_map_mse)},
                    **{f"coverage_1sigma_{k}": float(v) for k, v in zip(map_keys, per_map_cov1)},
                }
            )
            for bin_row in _calibration_bins(err_cat, sig_cat):
                calib_rows.append({"plateifu": str(plateifu), **bin_row})

            if max_plot <= 0 or plotted < max_plot:
                sdss = None
                sdss_bands: tuple[str, ...] | None = None
                if models[0].config.use_sdss and "inputs" in batch:
                    sdss = batch["inputs"]["sdss_imaging"][i].cpu().numpy()
                    raw_bands = batch["inputs"].get("sdss_imaging_bands")
                    if raw_bands is not None:
                        sdss_bands = tuple(str(b) for b in raw_bands)
                footprint = batch["footprint_mask"][i].cpu().numpy() if "footprint_mask" in batch else None
                sec = sigma_epi if secondary_sigma == "epistemic" else sigma_ale
                plot_uncertainty_map_panel(
                    plateifu=plateifu,
                    sdss=sdss,
                    sdss_band_names=sdss_bands,
                    footprint_mask=footprint,
                    target=targets[i].cpu().numpy(),
                    pred=mu_ens[i].cpu().numpy(),
                    mask=masks[i].cpu().numpy(),
                    map_keys=map_keys,
                    out_path=plots_dir / f"{split}_{plateifu.replace('-', '_')}.png",
                    sigma_tot=sigma_tot[i].cpu().numpy(),
                    sigma_secondary=sec[i].cpu().numpy(),
                    secondary_label=secondary_label,
                )
                plotted += 1

    write_ensemble_summary_plots(rows, calib_rows, map_keys, plots_dir)
    return rows, calib_rows
