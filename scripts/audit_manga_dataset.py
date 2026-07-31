"""Audit MaNGA Amara/Pipe3D label coverage joined to DRPall metadata."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

ROOT = Path(r"A:/MANGA")
DATA = ROOT / "manga_sdss_fits"
INDEX = DATA / "manga_dataset_index.csv"
DRPALL = ROOT / "drpall-v3_1_1.fits"
OUT = ROOT / "runs" / "dataset_audit"
OUT.mkdir(parents=True, exist_ok=True)

KEYS = [
    "ha_flux",
    "hbeta_flux",
    "oiii_5007_flux",
    "nii_6584_flux",
    "ha_ew",
    "stellar_av",
]
KEY_LABELS = {
    "ha_flux": "Ha flux",
    "hbeta_flux": "Hb flux",
    "oiii_5007_flux": "[OIII]5007",
    "nii_6584_flux": "[NII]6584",
    "ha_ew": "Ha EW",
    "stellar_av": "stellar Av",
}

MANGA_PRIMARY = 1 << 10
MANGA_SECONDARY = 1 << 11
MANGA_COLOR = 1 << 12


def clean(s: pd.Series, lo=None, hi=None) -> pd.Series:
    s = s.astype(float)
    s = s.where(s > -9000)
    if lo is not None:
        s = s.where(s >= lo)
    if hi is not None:
        s = s.where(s <= hi)
    return s


def hist_counts(arr, edges):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    counts, _ = np.histogram(arr, bins=edges)
    return counts.tolist(), int(arr.size)


def summarize(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return None
    return float(np.corrcoef(a[m], b[m])[0, 1])


def galaxy_dir(row, data: Path) -> Path:
    """Index paths are relative to manga_sdss_fits (data root)."""
    if "galaxy_dir" in row.index and pd.notna(row["galaxy_dir"]):
        p = Path(str(row["galaxy_dir"]))
        if not p.is_absolute():
            p = data / p
        return p
    pf = str(row["plateifu"]).replace("-", "_")
    return data / pf


def main() -> None:
    idx = pd.read_csv(INDEX)
    print("index rows", len(idx))
    print("columns", list(idx.columns))

    with fits.open(DRPALL) as hdul:
        t = hdul["MANGA"].data
        drp = pd.DataFrame(
            {
                "plateifu": np.array(t["plateifu"]).astype(str),
                "z": np.array(t["z"], dtype=float),
                "nsa_z": np.array(t["nsa_z"], dtype=float),
                "nsa_sersic_n": np.array(t["nsa_sersic_n"], dtype=float),
                "nsa_sersic_ba": np.array(t["nsa_sersic_ba"], dtype=float),
                "nsa_sersic_th50": np.array(t["nsa_sersic_th50"], dtype=float),
                "nsa_sersic_mass": np.array(t["nsa_sersic_mass"], dtype=float),
                "nsa_elpetro_mass": np.array(t["nsa_elpetro_mass"], dtype=float),
                "nsa_elpetro_th50_r": np.array(t["nsa_elpetro_th50_r"], dtype=float),
                "nsa_elpetro_ba": np.array(t["nsa_elpetro_ba"], dtype=float),
                "ifudesignsize": np.array(t["ifudesignsize"], dtype=float),
                "mngtarg1": np.array(t["mngtarg1"], dtype=np.int64),
                "mngtarg2": np.array(t["mngtarg2"], dtype=np.int64),
                "mngtarg3": np.array(t["mngtarg3"], dtype=np.int64),
                "ebvgal": np.array(t["ebvgal"], dtype=float),
                "exptime": np.array(t["exptime"], dtype=float),
                "bluesn2": np.array(t["bluesn2"], dtype=float),
                "redsn2": np.array(t["redsn2"], dtype=float),
            }
        )
    print("drpall", len(drp))

    if "has_amara_maps" in idx.columns:
        with_maps = idx[idx["has_amara_maps"].astype(bool)].copy()
    else:
        with_maps = idx.copy()

    rows = []
    missing = 0
    errors = 0
    n = len(with_maps)
    for i, (_, row) in enumerate(with_maps.iterrows()):
        gdir = galaxy_dir(row, DATA)
        if "amara_maps_npz" in row.index and pd.notna(row["amara_maps_npz"]):
            npz = Path(str(row["amara_maps_npz"]))
            if not npz.is_absolute():
                npz = DATA / npz
        else:
            npz = gdir / "amara_maps.npz"
        if not npz.exists():
            missing += 1
            continue
        try:
            a = np.load(npz)
            fp = a["native_footprint_mask"].astype(bool)
            fp_n = int(fp.sum())
            if fp_n == 0:
                continue
            pf = str(row.get("plateifu", gdir.name)).replace("_", "-")
            rec = {
                "plateifu": pf,
                "footprint_n": fp_n,
                "native_ny": int(a["native_ny"]) if "native_ny" in a.files else None,
                "native_nx": int(a["native_nx"]) if "native_nx" in a.files else None,
            }
            for k in KEYS:
                lm = a[f"{k}_loss_mask"].astype(bool)
                rec[f"{k}_pct"] = 100.0 * float(lm.sum()) / fp_n
                rec[f"{k}_n"] = int(lm.sum())
            rows.append(rec)
        except Exception as e:
            errors += 1
            if errors < 5:
                print("err", npz, e)
        if (i + 1) % 2000 == 0:
            print(f"processed {i+1}/{n}, kept {len(rows)}, missing {missing}")

    cov = pd.DataFrame(rows)
    print("coverage rows", len(cov), "missing", missing, "errors", errors)

    cov["plateifu"] = cov["plateifu"].astype(str).str.replace("_", "-", regex=False)
    drp["plateifu"] = drp["plateifu"].astype(str).str.strip()
    merged = cov.merge(drp, on="plateifu", how="left")
    print("merged", len(merged), "with z", merged["z"].notna().sum())

    merged["z_clean"] = clean(merged["z"], 0, 1)
    merged["nsa_sersic_n_clean"] = clean(merged["nsa_sersic_n"], 0.1, 10)
    merged["log_mass"] = np.log10(clean(merged["nsa_elpetro_mass"], 1e6, 1e13))
    merged["log_sersic_mass"] = np.log10(clean(merged["nsa_sersic_mass"], 1e6, 1e13))
    merged["th50_r"] = clean(merged["nsa_elpetro_th50_r"], 0.1, 100)
    merged["ba"] = clean(merged["nsa_elpetro_ba"], 0, 1)
    merged["sersic_ba"] = clean(merged["nsa_sersic_ba"], 0, 1)
    m1 = merged["mngtarg1"].fillna(0).astype(np.int64)
    merged["is_primary"] = (m1 & MANGA_PRIMARY) != 0
    merged["is_secondary"] = (m1 & MANGA_SECONDARY) != 0
    merged["is_color"] = (m1 & MANGA_COLOR) != 0

    merged.to_csv(OUT / "galaxy_coverage_meta.csv", index=False)
    print("saved csv", OUT / "galaxy_coverage_meta.csv")

    cov_edges = np.arange(0, 105, 5)
    feature_stats = {}
    for k in KEYS:
        col = f"{k}_pct"
        counts, nn = hist_counts(merged[col], cov_edges)
        feature_stats[k] = {
            "label": KEY_LABELS[k],
            "galaxy_pct_summary": summarize(merged[col]),
            "hist_5pct": {"edges": cov_edges.tolist(), "counts": counts, "n": nn},
            "pct_galaxies_ge": {
                str(t): float(100.0 * (merged[col] >= t).mean())
                for t in [50, 70, 80, 90, 95, 99]
            },
        }

    pixel_weighted = {
        k: float(100.0 * merged[f"{k}_n"].sum() / merged["footprint_n"].sum())
        for k in KEYS
    }

    z_edges = np.arange(0, 0.16, 0.01)
    z_counts, z_n = hist_counts(merged["z_clean"], z_edges)
    n_edges = np.arange(0, 6.5, 0.25)
    n_counts, n_n = hist_counts(merged["nsa_sersic_n_clean"], n_edges)
    m_edges = np.arange(8.0, 12.1, 0.2)
    m_counts, m_n = hist_counts(merged["log_mass"], m_edges)

    ifu_counts = merged["ifudesignsize"].value_counts(dropna=False).sort_index()
    ifu_size_dist = {
        str(int(k)) if pd.notna(k) else "nan": int(v) for k, v in ifu_counts.items()
    }

    modality = {}
    for c in idx.columns:
        if c.startswith("has_"):
            modality[c] = int(idx[c].astype(bool).sum())

    props = ["z_clean", "nsa_sersic_n_clean", "log_mass", "th50_r", "ba", "footprint_n"]

    def coverage_bin_table(feature_key):
        col = f"{feature_key}_pct"
        out_rows = []
        for lo in range(0, 100, 5):
            hi = lo + 5
            if hi < 100:
                mask = (merged[col] >= lo) & (merged[col] < hi)
            else:
                mask = (merged[col] >= lo) & (merged[col] <= hi)
            sub = merged.loc[mask]
            if len(sub) == 0:
                continue
            r = {
                "bin": f"{lo}-{hi}%",
                "n": int(len(sub)),
                "mean_cov": float(sub[col].mean()),
            }
            for p in props:
                s = summarize(sub[p])
                r[f"{p}_median"] = None if s is None else s["median"]
                r[f"{p}_mean"] = None if s is None else s["mean"]
                r[f"{p}_n"] = 0 if s is None else s["n"]
            r["frac_primary"] = float(sub["is_primary"].mean())
            r["frac_secondary"] = float(sub["is_secondary"].mean())
            r["frac_color"] = float(sub["is_color"].mean())
            if sub["ifudesignsize"].notna().any():
                r["ifu_mode"] = int(sub["ifudesignsize"].mode().iloc[0])
            else:
                r["ifu_mode"] = None
            out_rows.append(r)
        return out_rows

    bin_tables = {k: coverage_bin_table(k) for k in KEYS}

    def compare_high_low(feature_key, low_max=50, high_min=90):
        col = f"{feature_key}_pct"
        low = merged[merged[col] < low_max]
        high = merged[merged[col] >= high_min]
        mid = merged[(merged[col] >= low_max) & (merged[col] < high_min)]

        def block(df):
            return {
                "n": int(len(df)),
                "z": summarize(df["z_clean"]),
                "sersic_n": summarize(df["nsa_sersic_n_clean"]),
                "log_mass": summarize(df["log_mass"]),
                "th50_r": summarize(df["th50_r"]),
                "ba": summarize(df["ba"]),
                "footprint_n": summarize(df["footprint_n"]),
                "frac_primary": float(df["is_primary"].mean()) if len(df) else None,
                "frac_secondary": float(df["is_secondary"].mean()) if len(df) else None,
                "frac_color": float(df["is_color"].mean()) if len(df) else None,
            }

        return {
            "low_lt50": block(low),
            "mid_50_90": block(mid),
            "high_ge90": block(high),
        }

    comparisons = {k: compare_high_low(k) for k in KEYS}

    correlations = {}
    for k in KEYS:
        col = merged[f"{k}_pct"].values
        correlations[k] = {
            "z": corr(col, merged["z_clean"].values),
            "sersic_n": corr(col, merged["nsa_sersic_n_clean"].values),
            "log_mass": corr(col, merged["log_mass"].values),
            "th50_r": corr(col, merged["th50_r"].values),
            "ba": corr(col, merged["ba"].values),
            "footprint_n": corr(col, merged["footprint_n"].values.astype(float)),
            "redsn2": corr(col, clean(merged["redsn2"]).values),
        }

    # Coarse morphology proxy from Sersic n
    nser = merged["nsa_sersic_n_clean"]
    merged["morph_proxy"] = pd.cut(
        nser,
        bins=[0, 1.5, 2.5, 10],
        labels=["diskish_n<1.5", "intermediate", "bulgeish_n>2.5"],
    )
    morph_by_ha = (
        merged.assign(ha_bin=pd.cut(merged["ha_flux_pct"], bins=np.arange(0, 105, 5)))
        .groupby(["ha_bin", "morph_proxy"], observed=False)
        .size()
        .unstack(fill_value=0)
    )
    morph_table = []
    for bin_idx, row in morph_by_ha.iterrows():
        morph_table.append(
            {
                "bin": str(bin_idx),
                "diskish": int(row.get("diskish_n<1.5", 0)),
                "intermediate": int(row.get("intermediate", 0)),
                "bulgeish": int(row.get("bulgeish_n>2.5", 0)),
            }
        )

    summary = {
        "n_index": int(len(idx)),
        "n_with_amara_maps": int(len(merged)),
        "n_drpall": int(len(drp)),
        "modality_counts": modality,
        "pixel_weighted_coverage_pct": pixel_weighted,
        "feature_stats": feature_stats,
        "redshift_hist": {
            "edges": z_edges.tolist(),
            "counts": z_counts,
            "n": z_n,
            "summary": summarize(merged["z_clean"]),
        },
        "sersic_n_hist": {
            "edges": n_edges.tolist(),
            "counts": n_counts,
            "n": n_n,
            "summary": summarize(merged["nsa_sersic_n_clean"]),
        },
        "log_mass_hist": {
            "edges": m_edges.tolist(),
            "counts": m_counts,
            "n": m_n,
            "summary": summarize(merged["log_mass"]),
        },
        "ifu_size_dist": ifu_size_dist,
        "sample_flags": {
            "primary": int(merged["is_primary"].sum()),
            "secondary": int(merged["is_secondary"].sum()),
            "color_enhanced": int(merged["is_color"].sum()),
        },
        "bin_tables": bin_tables,
        "comparisons": comparisons,
        "correlations": correlations,
        "footprint_summary": summarize(merged["footprint_n"]),
        "morph_proxy_by_ha_bin": morph_table,
    }

    with open(OUT / "audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Wrote", OUT / "audit_summary.json")
    print("pixel weighted", pixel_weighted)
    print("z summary", summary["redshift_hist"]["summary"])
    print("ha coverage", feature_stats["ha_flux"]["galaxy_pct_summary"])
    print("correlations ha", correlations["ha_flux"])


if __name__ == "__main__":
    main()
