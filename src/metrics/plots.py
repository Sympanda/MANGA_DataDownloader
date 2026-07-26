from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / no GUI — avoids silent crashes on Windows
import matplotlib.pyplot as plt
import numpy as np
import torch


def write_metrics_csv(rows: list[dict], path: str | Path) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_training_history(history: list[dict[str, float]], out_dir: str | Path) -> None:
    """Plot train/val loss curves from epoch history rows (no pandas)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not history:
        return

    epochs = [int(row["epoch"]) for row in history]
    columns = [k for k in history[0] if k != "epoch"]

    def _series(col: str) -> list[float]:
        return [float(row[col]) for row in history]

    if "train_loss" in history[0] and "val_loss" in history[0]:
        fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
        ax.plot(epochs, _series("train_loss"), label="Train", lw=1.5)
        ax.plot(epochs, _series("val_loss"), label="Val", lw=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total loss")
        ax.legend()
        ax.grid(alpha=0.3, linestyle="--")
        fig.tight_layout()
        fig.savefig(out_dir / "loss_total.png")
        plt.close(fig)

    if "lr" in history[0]:
        fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
        ax.plot(epochs, _series("lr"), lw=1.5, color="tab:green")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning rate")
        ax.grid(alpha=0.3, linestyle="--")
        fig.tight_layout()
        fig.savefig(out_dir / "lr_schedule.png")
        plt.close(fig)

    core = {"train_loss", "val_loss", "lr"}
    for col in sorted(columns):
        if not col.startswith("train_") or col in core:
            continue
        base = col[len("train_") :]
        val_col = f"val_{base}"
        if val_col not in history[0]:
            continue
        fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
        ax.plot(epochs, _series(col), label="Train", lw=1.5)
        ax.plot(epochs, _series(val_col), label="Val", lw=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(base)
        ax.legend()
        ax.grid(alpha=0.3, linestyle="--")
        fig.tight_layout()
        fig.savefig(out_dir / f"loss_{base}.png")
        plt.close(fig)


def _percentile_norm(x: np.ndarray, lo: float = 5, hi: float = 99) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0)
    pos = x[x > 0]
    if pos.size == 0:
        return np.zeros_like(x)
    p_lo, p_hi = np.percentile(pos, [lo, hi])
    return np.clip((x - p_lo) / max(p_hi - p_lo, 1e-6), 0, 1)


# Shared display limits for 0–1 scaled Pipe3D map targets (see manga_prep.targets.pipe3d_maps).
# Keep identical across point-estimate and uncertainty eval panels.
MAP_VMIN = 0.0
MAP_VMAX = 1.0
DIFF_VMIN = -0.15
DIFF_VMAX = 0.15
SIGMA_VMIN = 0.0
SIGMA_VMAX = 0.15


def plot_map_prediction_panel(
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
    epoch: int | None = None,
) -> None:
    n_maps = len(map_keys)
    n_sdss = sdss.shape[0] if sdss is not None and sdss.ndim == 3 else 0
    n_col0 = n_sdss + (1 if footprint_mask is not None else 0)
    n_rows = max(n_maps, n_col0)
    n_cols = 5  # SDSS + mask | target | pred (masked) | diff | pred (full)
    fig = plt.figure(figsize=(15, 3.2 * n_rows), dpi=150)
    gs = fig.add_gridspec(n_rows, n_cols, width_ratios=[1, 1, 1, 1, 1], wspace=0.25, hspace=0.3)

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

        ax_tgt = fig.add_subplot(gs[row, 1])
        im_tgt = ax_tgt.imshow(
            np.where(m, tgt, np.nan),
            origin="lower",
            cmap="viridis",
            vmin=MAP_VMIN,
            vmax=MAP_VMAX,
        )
        ax_tgt.set_title(f"{key} target")
        ax_tgt.set_xticks([])
        ax_tgt.set_yticks([])
        fig.colorbar(im_tgt, ax=ax_tgt, fraction=0.046)

        ax_pred = fig.add_subplot(gs[row, 2])
        im_prd = ax_pred.imshow(
            np.where(m, prd, np.nan),
            origin="lower",
            cmap="viridis",
            vmin=MAP_VMIN,
            vmax=MAP_VMAX,
        )
        ax_pred.set_title(f"{key} pred")
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])
        fig.colorbar(im_prd, ax=ax_pred, fraction=0.046)

        ax_diff = fig.add_subplot(gs[row, 3])
        im_diff = ax_diff.imshow(
            diff,
            origin="lower",
            cmap="coolwarm",
            vmin=DIFF_VMIN,
            vmax=DIFF_VMAX,
        )
        ax_diff.set_title("diff")
        ax_diff.set_xticks([])
        ax_diff.set_yticks([])
        fig.colorbar(im_diff, ax=ax_diff, fraction=0.046)

        ax_full = fig.add_subplot(gs[row, 4])
        im_map = ax_full.imshow(
            prd,
            origin="lower",
            cmap="viridis",
            vmin=MAP_VMIN,
            vmax=MAP_VMAX,
        )
        ax_full.set_title(f"{key} pred (full)")
        ax_full.set_xticks([])
        ax_full.set_yticks([])
        fig.colorbar(im_map, ax=ax_full, fraction=0.046)

    title = f"{plateifu}"
    if epoch is not None:
        title += f"  epoch={epoch}"
    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def evaluate_map_predictions(
    model,
    dataloader,
    *,
    device: torch.device,
    map_keys: tuple[str, ...],
    plots_dir: Path,
    split: str,
    max_plot: int = 8,
) -> list[dict[str, float | str]]:
    from src.models.wrapper import (
        prepare_footprint_input,
        prepare_hr_imaging_input,
        prepare_imaging_input,
        prepare_spectrum_input,
        prepare_targets_and_masks,
    )

    model.eval()
    rows = []
    plotted = 0

    for batch in dataloader:
        plateifus = batch["plateifu"]
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
        targets, masks = prepare_targets_and_masks(batch, model.config)
        targets = targets.to(device)
        masks = masks.to(device)

        pred, _aux = model.model(x, spectrum_flux=spec, footprint=footprint, x_hr=x_hr)

        for i, plateifu in enumerate(plateifus):
            per_map_mse = []
            for ch, key in enumerate(map_keys):
                m = masks[i, ch] > 0
                if m.sum() == 0:
                    per_map_mse.append(float("nan"))
                else:
                    per_map_mse.append(float(((pred[i, ch][m] - targets[i, ch][m]) ** 2).mean().cpu()))
            sample_mse = float(np.nanmean(per_map_mse))
            rows.append(
                {
                    "plateifu": str(plateifu),
                    "split": split,
                    "mse_all": sample_mse,
                    **{f"mse_{k}": float(v) for k, v in zip(map_keys, per_map_mse)},
                }
            )

            if plotted < max_plot:
                sdss = None
                sdss_bands: tuple[str, ...] | None = None
                if model.config.use_sdss and "inputs" in batch:
                    sdss = batch["inputs"]["sdss_imaging"][i].cpu().numpy()
                    raw_bands = batch["inputs"].get("sdss_imaging_bands")
                    if raw_bands is not None:
                        sdss_bands = tuple(str(b) for b in raw_bands)
                footprint = None
                if "footprint_mask" in batch:
                    footprint = batch["footprint_mask"][i].cpu().numpy()
                plot_map_prediction_panel(
                    plateifu=plateifu,
                    sdss=sdss,
                    sdss_band_names=sdss_bands,
                    footprint_mask=footprint,
                    target=targets[i].cpu().numpy(),
                    pred=pred[i].cpu().numpy(),
                    mask=masks[i].cpu().numpy(),
                    map_keys=map_keys,
                    out_path=plots_dir / f"{split}_{plateifu.replace('-', '_')}.png",
                )
                plotted += 1

    return rows
