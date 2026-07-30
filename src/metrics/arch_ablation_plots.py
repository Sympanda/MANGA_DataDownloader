"""Aggregate architecture-ablation runs and write comparison plots/tables."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRIC_KEYS = ("rmse", "mae", "r2", "pearson_r", "bias", "median_abs_err")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def collect_cell_summaries(
    sweep_dir: Path,
    *,
    split: str = "test",
) -> list[dict[str, Any]]:
    """
    Load per-cell paper_eval summary rows + manifest metadata.

    Expects layout:
      <sweep>/runs/<cell>/paper_eval/<split>/summary_spaxel_stats.csv
      <sweep>/manifest.json
    """
    manifest_path = sweep_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    cells_meta = {c["name"]: c for c in manifest.get("cells", [])}

    rows_out: list[dict[str, Any]] = []
    runs_root = sweep_dir / "runs"
    if not runs_root.is_dir():
        return rows_out

    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        name = run_dir.name
        summary_path = run_dir / "paper_eval" / split / "summary_spaxel_stats.csv"
        if not summary_path.is_file():
            continue
        summary = _read_csv(summary_path)
        if not summary:
            continue

        meta = cells_meta.get(name, {})
        # Macro-average over channels (nanmean).
        agg = {k: float(np.nanmean([_f(r, k) for r in summary])) for k in METRIC_KEYS}
        # Also keep per-channel rmse/r2.
        for r in summary:
            ch = r.get("channel", r.get("map_key", "?"))
            for k in ("rmse", "mae", "r2"):
                agg[f"{k}__{ch}"] = _f(r, k)

        history_path = run_dir / "csv" / "train_val_history.csv"
        best_val = float("nan")
        best_epoch = float("nan")
        if history_path.is_file():
            hist = _read_csv(history_path)
            if hist and "val_loss" in hist[0]:
                losses = [_f(h, "val_loss") for h in hist]
                if any(np.isfinite(losses)):
                    i = int(np.nanargmin(losses))
                    best_val = losses[i]
                    best_epoch = _f(hist[i], "epoch")

        status = meta.get("status", "unknown")
        rows_out.append(
            {
                "cell": name,
                "split": split,
                "status": status,
                "architecture": meta.get("architecture", ""),
                "deep_supervision": meta.get("deep_supervision", ""),
                "spectrum": meta.get("spectrum", ""),
                "hr_cross_attn": meta.get("hr_cross_attn", ""),
                "film_injection": meta.get("film_injection", ""),
                "n_params": meta.get("n_params", ""),
                "note": meta.get("note", ""),
                "best_val_loss": best_val,
                "best_epoch": best_epoch,
                **agg,
            }
        )
    return rows_out


def write_summary_table(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return
    # Stable column order: identity first, then agg metrics, then per-channel.
    identity = [
        "cell",
        "split",
        "status",
        "architecture",
        "deep_supervision",
        "spectrum",
        "hr_cross_attn",
        "film_injection",
        "n_params",
        "best_val_loss",
        "best_epoch",
        "note",
    ]
    extras = [k for k in rows[0] if k not in identity]
    # Put macro metrics before per-channel.
    macros = [k for k in extras if "__" not in k]
    per_ch = sorted(k for k in extras if "__" in k)
    fieldnames = identity + macros + per_ch
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _cell_labels(rows: list[dict[str, Any]]) -> list[str]:
    return [str(r["cell"]) for r in rows]


def plot_metric_bars(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    out_path: Path,
    title: str | None = None,
    ylabel: str | None = None,
) -> None:
    if not rows:
        return
    labels = _cell_labels(rows)
    vals = [float(r.get(metric, float("nan"))) for r in rows]
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(labels)), 4.5))
    colors = []
    for r in rows:
        if r.get("hr_cross_attn") in (True, "True", "true", 1):
            colors.append("#c44e52")
        elif r.get("spectrum") == "on":
            colors.append("#4c72b0")
        else:
            colors.append("#55a868")
    ax.bar(labels, vals, color=colors, alpha=0.9)
    ax.set_ylabel(ylabel or metric.upper())
    ax.set_title(title or f"Architecture ablation — {metric}")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_channel_rmse(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not rows:
        return
    # Discover channels from first row keys.
    chans = sorted(
        {k.split("__", 1)[1] for k in rows[0] if k.startswith("rmse__")}
    )
    if not chans:
        return
    x = np.arange(len(chans))
    width = 0.8 / max(len(rows), 1)
    fig, ax = plt.subplots(figsize=(max(10, 1.4 * len(chans)), 5.0))
    for i, r in enumerate(rows):
        ys = [float(r.get(f"rmse__{c}", float("nan"))) for c in chans]
        ax.bar(x + i * width, ys, width=width, label=str(r["cell"]), alpha=0.9)
    ax.set_xticks(x + 0.4 * (len(rows) - 1) * width)
    ax.set_xticklabels(chans, rotation=25, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title("Per-channel RMSE by architecture")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_params_vs_rmse(rows: list[dict[str, Any]], out_path: Path) -> None:
    xs, ys, labels = [], [], []
    for r in rows:
        try:
            n = float(r.get("n_params") or float("nan"))
        except (TypeError, ValueError):
            n = float("nan")
        rmse = float(r.get("rmse", float("nan")))
        if np.isfinite(n) and np.isfinite(rmse):
            xs.append(n)
            ys.append(rmse)
            labels.append(str(r["cell"]))
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(xs, ys, s=60, c="#4c72b0")
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel("Trainable parameters")
    ax.set_ylabel("Macro RMSE")
    ax.set_title("Capacity vs error")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_delta_vs_baseline(
    rows: list[dict[str, Any]],
    *,
    baseline: str,
    metric: str,
    out_path: Path,
) -> None:
    base = next((r for r in rows if r["cell"] == baseline), None)
    if base is None or not np.isfinite(float(base.get(metric, float("nan")))):
        return
    bval = float(base[metric])
    labels, deltas = [], []
    for r in rows:
        if r["cell"] == baseline:
            continue
        v = float(r.get(metric, float("nan")))
        if not np.isfinite(v):
            continue
        # For RMSE/MAE lower is better → negative delta = improvement.
        # For R² higher is better → flip sign so negative still = worse.
        if metric in ("r2", "pearson_r"):
            d = bval - v
        else:
            d = v - bval
        labels.append(str(r["cell"]))
        deltas.append(d)
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(labels)), 4.5))
    colors = ["#55a868" if d < 0 else "#c44e52" for d in deltas]
    ax.bar(labels, deltas, color=colors, alpha=0.9)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_ylabel(f"Δ{metric} vs {baseline}\n(<0 = better)")
    ax.set_title(f"Change vs baseline ({baseline})")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_markdown_report(rows: list[dict[str, Any]], path: Path, *, baseline: str) -> None:
    lines = [
        "# Architecture ablation summary",
        "",
        f"Baseline for deltas: `{baseline}`",
        "",
        "| Cell | Arch | DS | Spec | HR | Params | RMSE | MAE | R² | best val_loss |",
        "|------|------|----|------|----|--------|------|-----|----|---------------|",
    ]
    for r in rows:
        lines.append(
            "| {cell} | {architecture} | {deep_supervision} | {spectrum} | {hr_cross_attn} | "
            "{n_params} | {rmse:.4f} | {mae:.4f} | {r2:.4f} | {best_val_loss:.4f} |".format(
                cell=r.get("cell", ""),
                architecture=r.get("architecture", ""),
                deep_supervision=r.get("deep_supervision", ""),
                spectrum=r.get("spectrum", ""),
                hr_cross_attn=r.get("hr_cross_attn", ""),
                n_params=r.get("n_params", ""),
                rmse=float(r.get("rmse", float("nan"))),
                mae=float(r.get("mae", float("nan"))),
                r2=float(r.get("r2", float("nan"))),
                best_val_loss=float(r.get("best_val_loss", float("nan"))),
            )
        )
    lines.extend(
        [
            "",
            "## How to read this",
            "",
            "- **A vs B**: plain UNet vs UNet++ (same spectrum/HR off).",
            "- **B vs C**: deep supervision on UNet++.",
            "- **C vs D**: spectrum package on top of UNet++ + DS.",
            "- **C vs E / D vs F**: HR cross-attention.",
            "- **D vs G / D vs H**: is UNet++ (+ DS) worth it once spectrum is on?",
            "",
            "Plots in this folder: macro RMSE/MAE/R² bars, per-channel RMSE, "
            "params-vs-RMSE, and deltas vs baseline.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_sweep(
    sweep_dir: Path,
    *,
    split: str = "test",
    baseline: str = "C_unetpp_ds",
) -> Path:
    """Build summary CSV + comparison plots under ``<sweep>/analysis/<split>/``."""
    rows = collect_cell_summaries(sweep_dir, split=split)
    out_dir = sweep_dir / "analysis" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    write_summary_table(rows, out_dir / "summary_metrics.csv")
    if not rows:
        (out_dir / "README.md").write_text(
            f"No completed paper_eval summaries found for split={split}.\n",
            encoding="utf-8",
        )
        return out_dir

    plot_metric_bars(rows, metric="rmse", out_path=out_dir / "macro_rmse.png", ylabel="Macro RMSE")
    plot_metric_bars(rows, metric="mae", out_path=out_dir / "macro_mae.png", ylabel="Macro MAE")
    plot_metric_bars(rows, metric="r2", out_path=out_dir / "macro_r2.png", ylabel="Macro R²")
    plot_per_channel_rmse(rows, out_dir / "per_channel_rmse.png")
    plot_params_vs_rmse(rows, out_dir / "params_vs_rmse.png")
    plot_delta_vs_baseline(rows, baseline=baseline, metric="rmse", out_path=out_dir / "delta_rmse_vs_baseline.png")
    plot_delta_vs_baseline(rows, baseline=baseline, metric="r2", out_path=out_dir / "delta_r2_vs_baseline.png")
    write_markdown_report(rows, out_dir / "README.md", baseline=baseline)
    return out_dir


__all__ = ["analyze_sweep", "collect_cell_summaries"]
