"""RePaint / synthetic-drop inpainting panels + neighbor-average baseline."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.metrics.plots import MAP_VMAX, MAP_VMIN

# 8-connected neighbours (include diagonals for smoother fills).
_NEIGH = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def neighbor_average_fill(
    target: np.ndarray,
    known_mask: np.ndarray,
    *,
    footprint: np.ndarray | None = None,
) -> np.ndarray:
    """
    Fill missing pixels by repeated neighbour averaging.

    Priority each round: pixels with the **most** already-filled 8-neighbours
    are filled first (frontier grows from known → inward).
    """
    tgt = np.asarray(target, dtype=np.float64)
    known = np.asarray(known_mask) > 0
    if footprint is None:
        domain = known | np.isfinite(tgt)
    else:
        domain = np.asarray(footprint) > 0

    h, w = tgt.shape
    filled = np.full((h, w), np.nan, dtype=np.float64)
    filled[known & domain] = tgt[known & domain]
    is_filled = known & domain
    todo = domain & ~is_filled

    # Safety cap: worst case one pixel per iteration.
    for _ in range(int(todo.sum()) + 5):
        if not todo.any():
            break
        # Count filled neighbours for every todo pixel.
        best_count = -1
        candidates: list[tuple[int, int]] = []
        ys, xs = np.where(todo)
        for y, x in zip(ys, xs):
            s = 0.0
            n = 0
            for dy, dx in _NEIGH:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and is_filled[yy, xx]:
                    s += filled[yy, xx]
                    n += 1
            if n > best_count:
                best_count = n
                candidates = [(y, x, s, n)]
            elif n == best_count and n > 0:
                candidates.append((y, x, s, n))

        if best_count <= 0:
            # Isolated holes: fall back to mean of currently filled domain.
            if is_filled.any():
                fallback = float(np.nanmean(filled[is_filled]))
            else:
                fallback = 0.0
            filled[todo] = fallback
            break

        # Fill all current best-count frontier pixels together (same “wave”).
        for y, x, s, n in candidates:
            filled[y, x] = s / n
            is_filled[y, x] = True
            todo[y, x] = False

    # Outside domain → nan for plotting; inside should be finite.
    out = filled.astype(np.float32)
    out[~domain] = np.nan
    return out


def heldout_rmse(pred: np.ndarray, target: np.ndarray, held_mask: np.ndarray) -> float:
    m = np.asarray(held_mask) > 0
    if m.sum() == 0:
        return 0.0
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    return float(np.sqrt(np.mean((p[m] - t[m]) ** 2)))


def plot_inpaint_drop_panel(
    *,
    plateifu: str,
    target: np.ndarray,
    base: np.ndarray | None,
    rows: list[dict[str, object]],
    out_path: Path,
) -> None:
    """
    One row per drop fraction (bottom row = 0% drop / 100% known).

    Columns: Target | Observed | Held-out | Base UNet | Neighbor avg | Inpaint mean
    RMSE_hold annotated on the last three (held-out pixels only). ★ = best.
    """
    titles = ["Target", "Observed", "Held-out", "Base UNet", "Neighbor avg", "Inpaint mean"]
    ncols = len(titles)
    nrows = len(rows)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.25 * ncols, 2.4 * nrows), squeeze=False
    )

    tgt = np.asarray(target[0] if target.ndim == 3 else target)

    for r_i, row in enumerate(rows):
        drop = float(row["drop_frac"])  # type: ignore[arg-type]
        known = np.asarray(row["known_mask"])
        if known.ndim == 3:
            known = known[0]
        held = np.asarray(row["heldout_mask"])
        if held.ndim == 3:
            held = held[0]
        mean = np.asarray(row["pred_mean"])
        if mean.ndim == 3:
            mean = mean[0]
        neigh = np.asarray(row["neighbor_fill"])
        if neigh.ndim == 3:
            neigh = neigh[0]
        rmse_inpaint = float(row.get("rmse_heldout_inpaint", float("nan")))  # type: ignore[arg-type]
        rmse_unet = float(row.get("rmse_heldout_unet", float("nan")))  # type: ignore[arg-type]
        rmse_neigh = float(row.get("rmse_heldout_neighbor", float("nan")))  # type: ignore[arg-type]
        keep_pct = 100.0 * (1.0 - drop)

        m_obs = known > 0
        observed = np.where(m_obs, tgt, np.nan)
        held_vis = np.where((known > 0) | (held > 0), held.astype(np.float32), np.nan)

        b = None
        if base is not None:
            b = np.asarray(base)
            if b.ndim == 3:
                b = b[0]

        scores = {
            "unet": rmse_unet,
            "neigh": rmse_neigh,
            "inpaint": rmse_inpaint,
        }
        finite = {k: v for k, v in scores.items() if np.isfinite(v)}
        best = min(finite, key=finite.get) if finite else None

        def _mark(key: str, val: float) -> str:
            if not np.isfinite(val):
                return "n/a"
            star = " ★" if best == key else ""
            return f"{val:.4f}{star}"

        panels: list[tuple[str, np.ndarray | None, str]] = [
            ("Target", tgt, "map"),
            ("Observed", observed, "map"),
            ("Held-out", held_vis, "mask"),
            (f"Base UNet\nRMSE_hold={_mark('unet', rmse_unet)}", b, "map"),
            (f"Neighbor avg\nRMSE_hold={_mark('neigh', rmse_neigh)}", neigh, "map"),
            (f"Inpaint mean\nRMSE_hold={_mark('inpaint', rmse_inpaint)}", mean, "map"),
        ]

        for j, (title, img, kind) in enumerate(panels):
            ax = axes[r_i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            if r_i == 0 or "RMSE" in title:
                ax.set_title(title, fontsize=8)
            if j == 0:
                if drop <= 1e-12:
                    ax.set_ylabel("drop 0%\n(100% known)", fontsize=8)
                else:
                    ax.set_ylabel(
                        f"drop {100 * drop:.0f}%\n({keep_pct:.0f}% known)", fontsize=8
                    )
            if img is None:
                ax.set_facecolor("#ddd")
                continue
            show = np.asarray(img, dtype=np.float32)
            if kind == "mask":
                im = ax.imshow(show, origin="lower", cmap="gray", vmin=0.0, vmax=1.0)
            else:
                im = ax.imshow(
                    show, origin="lower", cmap="viridis", vmin=MAP_VMIN, vmax=MAP_VMAX
                )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(
        f"{plateifu} — inpaint vs drop  (★ = lowest held-out RMSE among UNet / neighbor / inpaint)"
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def make_random_known_mask(
    label_mask: np.ndarray,
    *,
    drop_frac: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Keep a random subset of labelled pixels; drop_frac ∈ [0,1] are held out."""
    m = label_mask > 0
    known = m.copy()
    if drop_frac <= 0:
        return known.astype(np.float32)
    idx = np.argwhere(m)
    if idx.size == 0:
        return known.astype(np.float32)
    n_drop = int(round(float(drop_frac) * len(idx)))
    n_drop = min(n_drop, len(idx))
    if n_drop <= 0:
        return known.astype(np.float32)
    choice = rng.choice(len(idx), size=n_drop, replace=False)
    for i in choice:
        y, x = idx[i]
        known[y, x] = False
    return known.astype(np.float32)
