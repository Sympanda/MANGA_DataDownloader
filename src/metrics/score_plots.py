"""Score-model evaluation via reverse-process ``sample()`` (not training forward)."""
from __future__ import annotations

from collections.abc import Sequence
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


def _normalize_t_fracs(
    t_start_frac: float | None,
    t_start_fracs: Sequence[float] | None,
) -> list[float | None]:
    """Resolve which start-noise fractions to sample.

    ``None`` entries mean full generation from pure noise (generator default).
    """
    if t_start_fracs is not None:
        out: list[float | None] = []
        for f in t_start_fracs:
            ff = float(f)
            out.append(None if ff >= 1.0 - 1e-12 else ff)
        return out if out else [t_start_frac]
    return [t_start_frac]


def _frac_label(frac: float | None) -> str:
    # t=1: pure generation. t<1: reverse from GT noised to that level (NOT fair gen).
    if frac is None:
        return "t=1.00 (from noise)"
    return f"t={float(frac):.2f} (GT+noise)"


def _frac_tag(frac: float | None) -> str:
    return "t1p00" if frac is None else f"t{float(frac):.2f}".replace(".", "p")


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
    t_start_fracs: Sequence[float] | None = None,
    seed: int = 0,
    show_n_individual: int = 4,
    max_galaxies: int | None = None,
    use_ema: bool = False,
) -> list[dict[str, float | str]]:
    """
    Evaluate a MapScoreModel by calling ``model.sample(...)``.

    Do not use ``model(batch)`` here — that returns a random-timestep training view.
    Sampling is expensive; by default only the first ``max_galaxies`` galaxies are
    evaluated (defaults to ``max(max_plot, 16)``).

    For the direct generator, pass ``t_start_fracs`` (e.g. ``[1.0, 0.5, 0.25, 0.1]``)
    to draw one panel row per start-noise level. ``t=1`` is pure-noise generation;
    ``t<1`` noises the clean map to that level then denoises.
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

    is_corrector = getattr(model, "mode", None) == "corrector"
    fracs = _normalize_t_fracs(t_start_frac, t_start_fracs)
    # Primary metrics use full-noise generation when present, else the first frac.
    primary_frac = None if any(f is None for f in fracs) else fracs[0]
    multi_t = (not is_corrector) and len(fracs) > 1

    for batch in dataloader:
        if n_seen >= limit:
            break
        batch = _move_batch_to_device(batch, device)
        plateifus = batch["plateifu"]
        take = min(len(plateifus), limit - n_seen)
        if take < len(plateifus):
            batch = _trim_batch(batch, take)
            plateifus = batch["plateifu"]

        # Always sample at the primary frac for metrics / default panels.
        out_primary = model.sample(
            batch,
            n_samples=n_samples,
            ddim_steps=ddim_steps,
            t_start_frac=primary_frac,
            seed=seed,
            use_ema=use_ema,
        )
        pred = out_primary["predictive_mean"]
        targets = out_primary["targets"]
        masks = out_primary["label_mask"] if "label_mask" in out_primary else out_primary["masks"]
        base_maps = out_primary.get("base_maps")

        # Extra fracs only for galaxies we will plot (DDIM is expensive).
        outs_by_frac: dict[str, dict[str, torch.Tensor]] = {
            _frac_tag(primary_frac): out_primary
        }
        if multi_t and plotted < max_plot:
            for frac in fracs:
                tag = _frac_tag(frac)
                if tag in outs_by_frac:
                    continue
                outs_by_frac[tag] = model.sample(
                    batch,
                    n_samples=n_samples,
                    ddim_steps=ddim_steps,
                    t_start_frac=frac,
                    seed=seed,
                    use_ema=use_ema,
                )

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
            # Per-t MSEs when multi-t sampling was run for this batch.
            if multi_t:
                for frac in fracs:
                    tag = _frac_tag(frac)
                    o = outs_by_frac.get(tag)
                    if o is None:
                        continue
                    p_t = o["predictive_mean"][i]
                    mses_t = []
                    for ch, _key in enumerate(map_keys):
                        m = masks[i, ch] > 0
                        if m.sum() == 0:
                            mses_t.append(float("nan"))
                        else:
                            mses_t.append(
                                float(((p_t[ch][m] - targets[i, ch][m]) ** 2).mean().cpu())
                            )
                    row[f"mse_all_{tag}"] = float(np.nanmean(mses_t))
            rows.append(row)

            if plotted < max_plot:
                out_path = plots_dir / f"{split}_{str(plateifu).replace('-', '_')}.png"
                if is_corrector and base_maps is not None:
                    sample_i = None
                    if "samples" in out_primary:
                        sample_i = out_primary["samples"][:, i].detach().cpu().numpy()
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
                                out_primary["predictive_std"][i].detach().cpu().numpy()
                                if "predictive_std" in out_primary
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
                elif multi_t:
                    row_data: list[dict[str, object]] = []
                    for frac in fracs:
                        o = outs_by_frac[_frac_tag(frac)]
                        sample_i = (
                            o["samples"][:, i].detach().cpu().numpy()
                            if "samples" in o
                            else None
                        )
                        row_data.append(
                            {
                                "frac": frac,
                                "pred_mean": o["predictive_mean"][i].detach().cpu().numpy(),
                                "pred_std": (
                                    o["predictive_std"][i].detach().cpu().numpy()
                                    if "predictive_std" in o
                                    else None
                                ),
                                "samples": sample_i,
                            }
                        )
                    plot_score_multi_t_panel(
                        plateifu=str(plateifu),
                        target=targets[i].detach().cpu().numpy(),
                        mask=masks[i].detach().cpu().numpy(),
                        map_keys=map_keys,
                        rows=row_data,
                        out_path=out_path,
                        base=(
                            base_maps[i].detach().cpu().numpy()
                            if base_maps is not None
                            else None
                        ),
                        footprint=(
                            out_primary["footprint_mask"][i].detach().cpu().numpy()
                            if "footprint_mask" in out_primary
                            else None
                        ),
                        show_n_individual=show_n_individual,
                        show_completion=bool(
                            getattr(model, "eval_show_completion", False)
                        ),
                    )
                else:
                    sample_i = None
                    if "samples" in out_primary:
                        sample_i = out_primary["samples"][:, i].detach().cpu().numpy()
                    plot_score_panel(
                        plateifu=str(plateifu),
                        target=targets[i].detach().cpu().numpy(),
                        pred_mean=pred[i].detach().cpu().numpy(),
                        mask=masks[i].detach().cpu().numpy(),
                        footprint=(
                            out_primary["footprint_mask"][i].detach().cpu().numpy()
                            if "footprint_mask" in out_primary
                            else None
                        ),
                        base=(
                            base_maps[i].detach().cpu().numpy()
                            if base_maps is not None
                            else None
                        ),
                        pred_std=(
                            out_primary["predictive_std"][i].detach().cpu().numpy()
                            if "predictive_std" in out_primary
                            else None
                        ),
                        samples=sample_i,
                        map_keys=map_keys,
                        out_path=out_path,
                        show_n_individual=show_n_individual,
                        t_label=_frac_label(primary_frac),
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


def plot_score_multi_t_panel(
    *,
    plateifu: str,
    target: np.ndarray,
    mask: np.ndarray,
    map_keys: tuple[str, ...],
    rows: list[dict[str, object]],
    out_path: Path,
    base: np.ndarray | None = None,
    footprint: np.ndarray | None = None,
    show_n_individual: int = 3,
    show_completion: bool = False,
) -> None:
    """
    One row per start-noise fraction.

    Columns: Target | Base? | Sample mean | Sample std | Sample 0… |
             [Labels] | Target−UNet | Target−mean
    Residual columns share one symmetric color scale per row and show RMSE
    on **observed label pixels only**.

    When ``show_completion`` is True, predictions are shown on the full
    footprint (so missing-label regions are visible) while Target keeps
    label holes; a Labels column marks observed vs missing spaxels.
    """
    n_maps = len(map_keys)
    n_samp = 0
    for row in rows:
        s = row.get("samples")
        if s is not None:
            n_samp = max(n_samp, min(show_n_individual, int(np.asarray(s).shape[0])))
    titles = ["Target"]
    if base is not None:
        titles.append("Base UNet")
    titles += ["Sample mean", "Sample std"]
    titles += [f"Sample {k}" for k in range(n_samp)]
    if show_completion:
        titles.append("Labels")
    titles += ["Target−UNet", "Target−mean"]
    ncols = len(titles)
    nrows = len(rows) * n_maps
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.25 * ncols, 2.4 * nrows), squeeze=False
    )

    def _rmse(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
        if m.sum() == 0:
            return float("nan")
        return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))

    def _resid_lim(*arrs: np.ndarray, m: np.ndarray) -> float:
        vals: list[float] = []
        for a in arrs:
            show = np.where(m, a, np.nan)
            finite = show[np.isfinite(show)]
            if finite.size:
                vals.append(float(np.nanpercentile(np.abs(finite), 98)))
        return max(max(vals) if vals else 0.15, 1e-4)

    for r_i, row in enumerate(rows):
        frac = row["frac"]  # float | None
        pred_mean = np.asarray(row["pred_mean"])
        pred_std = row.get("pred_std")
        samples = row.get("samples")
        for ch, key in enumerate(map_keys):
            ax_row = r_i * n_maps + ch
            m_lab = mask[ch] > 0
            if footprint is None:
                m_fp = m_lab
            else:
                fp_ch = footprint[ch] if footprint.ndim == 3 else footprint
                m_fp = fp_ch > 0
            # Predictions: footprint when showing completion; else labels.
            m_pred = m_fp if show_completion else m_lab
            tgt_ch = target[ch]
            mean_ch = pred_mean[ch]
            base_ch = None if base is None else base[ch]
            resid_unet = None if base_ch is None else (tgt_ch - base_ch)
            resid_mean = tgt_ch - mean_ch
            rmse_unet = (
                float("nan") if resid_unet is None else _rmse(tgt_ch, base_ch, m_lab)
            )
            rmse_mean = _rmse(tgt_ch, mean_ch, m_lab)
            lim = _resid_lim(
                *(a for a in (resid_unet, resid_mean) if a is not None),
                m=m_lab,
            )

            panels: list[tuple[str, np.ndarray | None, str, np.ndarray]] = [
                ("Target", tgt_ch, "map", m_lab)
            ]
            if base is not None:
                panels.append(("Base UNet", base_ch, "map", m_pred))
            panels.append(("Sample mean", mean_ch, "map", m_pred))
            panels.append(
                (
                    "Sample std",
                    None if pred_std is None else np.asarray(pred_std)[ch],
                    "std",
                    m_pred,
                )
            )
            for k in range(n_samp):
                img = None
                if samples is not None and k < np.asarray(samples).shape[0]:
                    img = np.asarray(samples)[k, ch]
                panels.append((f"Sample {k}", img, "map", m_pred))

            if show_completion:
                # 1 = observed label, 0 = footprint but missing label, nan = outside.
                lab_vis = np.full_like(tgt_ch, np.nan, dtype=np.float32)
                lab_vis[m_fp] = 0.0
                lab_vis[m_lab] = 1.0
                panels.append(("Labels\n(1=obs, 0=miss)", lab_vis, "labels", m_fp))

            # Residual pair — shared scale; RMSE on observed labels only.
            # ★ only meaningful for fair gen comparison at t=1 (from noise).
            fair = frac is None
            better_unet = fair and np.isfinite(rmse_unet) and (
                not np.isfinite(rmse_mean) or rmse_unet <= rmse_mean
            )
            better_mean = fair and np.isfinite(rmse_mean) and (
                not np.isfinite(rmse_unet) or rmse_mean < rmse_unet
            )
            unet_mark = " ★" if better_unet else ""
            mean_mark = " ★" if better_mean else ""
            panels.append(
                (
                    f"Target−UNet\nRMSE={rmse_unet:.4f}{unet_mark}",
                    resid_unet,
                    "resid",
                    m_lab,
                )
            )
            panels.append(
                (
                    f"Target−mean\nRMSE={rmse_mean:.4f}{mean_mark}",
                    resid_mean,
                    "resid",
                    m_lab,
                )
            )

            for j, (title, img, kind, m_show) in enumerate(panels):
                ax = axes[ax_row, j]
                ax.set_xticks([])
                ax.set_yticks([])
                if ax_row == 0 or kind == "resid":
                    ax.set_title(title, fontsize=8)
                if j == 0:
                    ax.set_ylabel(f"{_frac_label(frac)}\n{key}", fontsize=8)
                if img is None:
                    ax.set_facecolor("#ddd")
                    continue
                show = np.asarray(img, dtype=np.float32).copy()
                show = np.where(m_show, show, np.nan)
                if kind == "map":
                    im = ax.imshow(
                        show, origin="lower", cmap="viridis", vmin=MAP_VMIN, vmax=MAP_VMAX
                    )
                elif kind == "resid":
                    im = ax.imshow(
                        show,
                        origin="lower",
                        cmap="coolwarm",
                        vmin=-lim,
                        vmax=lim,
                    )
                elif kind == "labels":
                    im = ax.imshow(
                        show, origin="lower", cmap="gray", vmin=0.0, vmax=1.0
                    )
                else:
                    finite = show[np.isfinite(show)]
                    hi = float(np.nanpercentile(finite, 98)) if finite.size else 0.15
                    im = ax.imshow(
                        show, origin="lower", cmap="magma", vmin=0.0, vmax=max(hi, 1e-4)
                    )
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(
        f"{plateifu} — generator vs start noise\n"
        "t=1 from pure noise (fair vs UNet ★); t<1 = denoise GT+noise (near-identity at low t)"
    )
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
    t_label: str | None = None,
) -> None:
    """Target | Base? | Mean | Std | samples… with shared map scales."""
    del footprint  # reserved for future overlays
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
                hi = (
                    float(np.nanpercentile(show[np.isfinite(show)], 98))
                    if np.isfinite(show).any()
                    else 0.15
                )
                im = ax.imshow(show, origin="lower", cmap="magma", vmin=0.0, vmax=max(hi, 1e-4))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    title = plateifu if t_label is None else f"{plateifu} — {t_label}"
    fig.suptitle(title)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
