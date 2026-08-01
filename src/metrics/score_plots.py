"""Score-model evaluation via reverse-process ``sample()`` (not training forward)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.metrics.plots import MAP_VMAX, MAP_VMIN
from src.metrics.residual_plots import (
    _masked_stats,
    _move_batch_to_device,
    plot_residual_diagnostic_panel,
)


@torch.no_grad()
def evaluate_score_samples(
    model,
    dataloader,
    *,
    device: torch.device,
    map_keys: tuple[str, ...],
    plots_dir: Path,
    split: str,
    max_plot: int = 4,
    n_samples: int = 8,
    ddim_steps: int = 50,
    t_start_frac: float | None = None,
    seed: int = 0,
    show_n_individual: int = 4,
    max_galaxies: int | None = None,
) -> list[dict[str, float | str]]:
    """
    Evaluate a MapScoreModel by calling ``model.sample(...)``.

    Do not use ``model(batch)`` here — that returns a random-timestep training view.
    Sampling is expensive; by default only the first ``max_galaxies`` galaxies are
    evaluated (defaults to ``max(max_plot, 16)``).
    """
    if not hasattr(model, "sample"):
        raise TypeError("evaluate_score_samples requires model.sample(...)")

    model.eval()
    rows: list[dict[str, float | str]] = []
    plotted = 0
    n_seen = 0
    limit = int(max_galaxies) if max_galaxies is not None else max(int(max_plot), 16)
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    for batch in dataloader:
        if n_seen >= limit:
            break
        batch = _move_batch_to_device(batch, device)
        plateifus = batch["plateifu"]
        # Trim batch if it would exceed the galaxy budget.
        take = min(len(plateifus), limit - n_seen)
        if take < len(plateifus):
            batch = _trim_batch(batch, take)
            plateifus = batch["plateifu"]

        out = model.sample(
            batch,
            n_samples=n_samples,
            ddim_steps=ddim_steps,
            t_start_frac=t_start_frac,
            seed=seed,
            use_ema=True,
        )
        pred = out["predictive_mean"]
        targets = out["targets"]
        masks = out["label_mask"] if "label_mask" in out else out["masks"]

        base_maps = out.get("base_maps")
        is_corrector = getattr(model, "mode", None) == "corrector"
        for i, plateifu in enumerate(plateifus):
            per_map = []
            resid_pearsons: list[float] = []
            resid_rmses: list[float] = []
            for ch, key in enumerate(map_keys):
                m = masks[i, ch] > 0
                if m.sum() == 0:
                    per_map.append(float("nan"))
                    resid_pearsons.append(float("nan"))
                    resid_rmses.append(float("nan"))
                else:
                    per_map.append(
                        float(((pred[i, ch][m] - targets[i, ch][m]) ** 2).mean().cpu())
                    )
                    if is_corrector and base_maps is not None:
                        true_r = (targets[i, ch] - base_maps[i, ch]).detach().cpu().numpy()
                        pred_r = (pred[i, ch] - base_maps[i, ch]).detach().cpu().numpy()
                        m_np = masks[i, ch].detach().cpu().numpy()
                        st = _masked_stats(pred_r, true_r, m_np)
                        resid_pearsons.append(st["pearson"])
                        resid_rmses.append(st["rmse"])
                    else:
                        resid_pearsons.append(float("nan"))
                        resid_rmses.append(float("nan"))
            row: dict[str, float | str] = {
                "plateifu": str(plateifu),
                "split": split,
                "mse_all": float(np.nanmean(per_map)),
                **{f"mse_{k}": float(v) for k, v in zip(map_keys, per_map)},
            }
            if is_corrector and base_maps is not None:
                row["resid_pearson_all"] = float(np.nanmean(resid_pearsons))
                row["resid_rmse_all"] = float(np.nanmean(resid_rmses))
                for k, p, r in zip(map_keys, resid_pearsons, resid_rmses):
                    row[f"resid_pearson_{k}"] = float(p)
                    row[f"resid_rmse_{k}"] = float(r)
            rows.append(row)
            if plotted < max_plot:
                sample_i = None
                if "samples" in out:
                    sample_i = out["samples"][:, i].detach().cpu().numpy()
                out_path = plots_dir / f"{split}_{str(plateifu).replace('-', '_')}.png"
                # Corrector = SDEdit residual path → residual diagnostic panels.
                # Generator = full-map samples from noise → absolute map panels.
                is_corrector = getattr(model, "mode", None) == "corrector"
                if is_corrector and base_maps is not None:
                    tgt_np = targets[i].detach().cpu().numpy()
                    pred_np = pred[i].detach().cpu().numpy()
                    mask_np = masks[i].detach().cpu().numpy()
                    base_np = base_maps[i].detach().cpu().numpy()
                    sample_resid = None
                    if sample_i is not None:
                        sample_resid = sample_i - base_np[None, ...]
                        sample_resid = sample_resid[:show_n_individual]
                    plot_residual_diagnostic_panel(
                        plateifu=str(plateifu),
                        pred_dict={
                            "targets": tgt_np,
                            "maps": pred_np,
                            "masks": mask_np,
                            "base_maps": base_np,
                            "residual_target": tgt_np - base_np,
                            "residual_prediction": pred_np - base_np,
                            "predictive_std": (
                                out["predictive_std"][i].detach().cpu().numpy()
                                if "predictive_std" in out
                                else None
                            ),
                        },
                        map_keys=map_keys,
                        out_path=out_path,
                        sample_maps=sample_resid,
                        show_uncertainty=True,
                    )
                    plot_residual_correlation_scatter(
                        plateifu=str(plateifu),
                        true_residual=tgt_np - base_np,
                        pred_residual=pred_np - base_np,
                        mask=mask_np,
                        map_keys=map_keys,
                        out_path=plots_dir
                        / f"{split}_{str(plateifu).replace('-', '_')}_resid_scatter.png",
                    )
                else:
                    # Optional Base UNet column is comparison-only; samples are
                    # absolute Ha maps (not residuals).
                    base_np = (
                        base_maps[i].detach().cpu().numpy()
                        if base_maps is not None
                        else None
                    )
                    plot_score_panel(
                        plateifu=str(plateifu),
                        target=targets[i].detach().cpu().numpy(),
                        pred_mean=pred[i].detach().cpu().numpy(),
                        mask=masks[i].detach().cpu().numpy(),
                        footprint=(
                            out["footprint_mask"][i].detach().cpu().numpy()
                            if "footprint_mask" in out
                            else None
                        ),
                        base=base_np,
                        pred_std=(
                            out["predictive_std"][i].detach().cpu().numpy()
                            if "predictive_std" in out
                            else None
                        ),
                        samples=sample_i,
                        map_keys=map_keys,
                        out_path=out_path,
                        show_n_individual=show_n_individual,
                    )
                plotted += 1
            n_seen += 1
            if n_seen >= limit:
                break
    return rows


def _trim_batch(batch: dict, n: int) -> dict:
    """Keep the first ``n`` galaxies in a collated batch (tensors + plateifu list)."""
    out: dict = {}
    for k, v in batch.items():
        if k == "plateifu":
            out[k] = list(v)[:n]
        elif torch.is_tensor(v):
            out[k] = v[:n]
        elif isinstance(v, dict):
            out[k] = _trim_batch(v, n)
        else:
            out[k] = v
    return out


def plot_residual_correlation_scatter(
    *,
    plateifu: str,
    true_residual: np.ndarray,
    pred_residual: np.ndarray,
    mask: np.ndarray,
    map_keys: tuple[str, ...],
    out_path: Path,
    max_points: int = 8000,
) -> None:
    """Per-channel scatter of true residual (target−base) vs score correction (mean−base)."""
    n_maps = len(map_keys)
    fig, axes = plt.subplots(1, n_maps, figsize=(4.2 * n_maps, 4.0), squeeze=False)
    rng = np.random.default_rng(0)
    for ch, key in enumerate(map_keys):
        ax = axes[0, ch]
        m = mask[ch] > 0
        x = true_residual[ch][m].astype(np.float64).ravel()
        y = pred_residual[ch][m].astype(np.float64).ravel()
        if x.size == 0:
            ax.set_facecolor("#ddd")
            ax.set_title(key)
            continue
        if x.size > max_points:
            idx = rng.choice(x.size, size=max_points, replace=False)
            x, y = x[idx], y[idx]
        ax.scatter(x, y, s=4, alpha=0.25, c="#1f77b4", linewidths=0)
        lim = float(np.nanpercentile(np.abs(np.concatenate([x, y])), 99))
        lim = max(lim, 1e-4)
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, alpha=0.7)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("True residual (target − base)")
        ax.set_ylabel("Score correction (mean − base)")
        if x.std() > 1e-12 and y.std() > 1e-12:
            r = float(np.corrcoef(x, y)[0, 1])
            ax.set_title(f"{key}  r={r:.3f}")
        else:
            ax.set_title(f"{key}  r=n/a")
        ax.axhline(0.0, color="0.5", lw=0.6)
        ax.axvline(0.0, color="0.5", lw=0.6)
    fig.suptitle(f"{plateifu} — residual correlation")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_score_panel(
    *,
    plateifu: str,
    target: np.ndarray,
    pred_mean: np.ndarray,
    mask: np.ndarray,
    map_keys: tuple[str, ...],
    out_path: Path,
    footprint: np.ndarray | None = None,
    base: np.ndarray | None = None,
    pred_std: np.ndarray | None = None,
    samples: np.ndarray | None = None,
    show_n_individual: int = 4,
) -> None:
    """Target | Base? | Mean | Std | samples… with shared map scales.

    Prefer ``plot_residual_diagnostic_panel`` when a frozen base is available
    (score corrector); this panel is mainly for the direct generator.
    """
    n_maps = len(map_keys)
    n_samp = 0 if samples is None else min(show_n_individual, samples.shape[0])
    titles = ["Target"]
    if base is not None:
        titles.append("Base UNet")
    titles += ["Sample mean", "Sample std"]
    titles += [f"Sample {k}" for k in range(n_samp)]
    ncols = len(titles)
    fig, axes = plt.subplots(n_maps, ncols, figsize=(2.3 * ncols, 2.5 * n_maps), squeeze=False)

    for ch, key in enumerate(map_keys):
        m = mask[ch] > 0
        panels: list[tuple[str, np.ndarray | None, str]] = [("Target", target[ch], "map")]
        if base is not None:
            panels.append(("Base UNet", base[ch], "map"))
        panels.append(("Sample mean", pred_mean[ch], "map"))
        panels.append(("Sample std", None if pred_std is None else pred_std[ch], "std"))
        for k in range(n_samp):
            panels.append((f"Sample {k}", samples[k, ch], "map"))

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
            if kind == "map":
                show = np.where(m, show, np.nan)
                im = ax.imshow(show, origin="lower", cmap="viridis", vmin=MAP_VMIN, vmax=MAP_VMAX)
            else:
                show = np.where(m, show, np.nan)
                hi = float(np.nanpercentile(show[np.isfinite(show)], 98)) if np.isfinite(show).any() else 0.15
                im = ax.imshow(show, origin="lower", cmap="magma", vmin=0.0, vmax=max(hi, 1e-4))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(plateifu)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
