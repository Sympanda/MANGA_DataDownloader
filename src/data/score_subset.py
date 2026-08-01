"""Score-training galaxy subset selection and stratified sampling weights."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

import numpy as np

from src.data.splits import SplitName, read_split_csv

CoverageFeature = Literal["ha_flux", "hbeta_flux", "oiii_5007_flux", "nii_6584_flux", "ha_ew", "stellar_av"]


def load_coverage_meta(path: Path | str) -> list[dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Coverage meta CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def select_score_plateifus(
    *,
    coverage_csv: Path | str,
    split_csv: Path | str,
    split: SplitName = "train",
    feature: CoverageFeature = "ha_flux",
    min_coverage_pct: float = 99.0,
    max_coverage_pct: float | None = None,
) -> list[str]:
    """
    Select galaxies from ``split`` with feature coverage in
    ``[min_coverage_pct, max_coverage_pct]`` (max inclusive when set).

    Dense validation/test galaxies are never pulled into training.
    """
    allowed = read_split_csv(split_csv)[split]
    cov_col = f"{feature}_pct"
    max_cov = float("inf") if max_coverage_pct is None else float(max_coverage_pct)
    selected: list[str] = []
    for row in load_coverage_meta(coverage_csv):
        plateifu = row["plateifu"].strip()
        if plateifu not in allowed:
            continue
        try:
            cov = float(row[cov_col])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(cov) and float(min_coverage_pct) <= cov <= max_cov:
            selected.append(plateifu)
    return sorted(selected)


def stratified_sample_weights(
    plateifus: list[str],
    *,
    coverage_csv: Path | str,
    z_bins: int = 4,
    n_bins: int = 4,
    mass_bins: int = 4,
) -> np.ndarray:
    """
    Inverse-frequency weights over joint bins of redshift, Sersic n, and log mass.

    Galaxies missing metadata receive the median weight. Weights are not model inputs.
    """
    meta = {r["plateifu"].strip(): r for r in load_coverage_meta(coverage_csv)}
    z_vals: list[float] = []
    n_vals: list[float] = []
    m_vals: list[float] = []
    valid_idx: list[int] = []
    for i, p in enumerate(plateifus):
        row = meta.get(p)
        if row is None:
            continue
        try:
            z = float(row.get("z_clean") or row.get("z") or "nan")
            n = float(row.get("nsa_sersic_n_clean") or row.get("nsa_sersic_n") or "nan")
            m = float(row.get("log_mass") or "nan")
        except ValueError:
            continue
        if not (np.isfinite(z) and np.isfinite(n) and np.isfinite(m)):
            continue
        z_vals.append(z)
        n_vals.append(n)
        m_vals.append(m)
        valid_idx.append(i)

    weights = np.ones(len(plateifus), dtype=np.float64)
    if len(valid_idx) < 8:
        return weights

    z_arr = np.asarray(z_vals)
    n_arr = np.asarray(n_vals)
    m_arr = np.asarray(m_vals)
    z_edges = np.quantile(z_arr, np.linspace(0, 1, z_bins + 1))
    n_edges = np.quantile(n_arr, np.linspace(0, 1, n_bins + 1))
    m_edges = np.quantile(m_arr, np.linspace(0, 1, mass_bins + 1))
    # Ensure strictly increasing edges for digitize.
    for edges in (z_edges, n_edges, m_edges):
        for k in range(1, len(edges)):
            if edges[k] <= edges[k - 1]:
                edges[k] = edges[k - 1] + 1e-6

    zi = np.clip(np.digitize(z_arr, z_edges[1:-1], right=True), 0, z_bins - 1)
    ni = np.clip(np.digitize(n_arr, n_edges[1:-1], right=True), 0, n_bins - 1)
    mi = np.clip(np.digitize(m_arr, m_edges[1:-1], right=True), 0, mass_bins - 1)
    codes = zi * (n_bins * mass_bins) + ni * mass_bins + mi
    _, inverse, counts = np.unique(codes, return_inverse=True, return_counts=True)
    inv = 1.0 / counts.astype(np.float64)
    w_valid = inv[inverse]
    w_valid = w_valid / w_valid.mean()
    for i, w in zip(valid_idx, w_valid):
        weights[i] = float(w)
    return weights


def flag_ood_populations(
    plateifus: list[str],
    *,
    coverage_csv: Path | str,
    z_hi: float = 0.08,
    n_hi: float = 4.0,
    log_mass_hi: float = 10.8,
) -> dict[str, bool]:
    """Flag high-z / high-Sersic / high-mass systems as potentially OOD for dense-Ha training."""
    meta = {r["plateifu"].strip(): r for r in load_coverage_meta(coverage_csv)}
    out: dict[str, bool] = {}
    for p in plateifus:
        row = meta.get(p)
        if row is None:
            out[p] = True
            continue
        try:
            z = float(row.get("z_clean") or row.get("z") or "nan")
            n = float(row.get("nsa_sersic_n_clean") or row.get("nsa_sersic_n") or "nan")
            m = float(row.get("log_mass") or "nan")
        except ValueError:
            out[p] = True
            continue
        out[p] = bool(
            (np.isfinite(z) and z >= z_hi)
            or (np.isfinite(n) and n >= n_hi)
            or (np.isfinite(m) and m >= log_mass_hi)
        )
    return out
