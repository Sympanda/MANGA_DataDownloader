"""Footprint fill / valid-pixel coverage for map target selection."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from manga_prep.targets.pipe3d_maps import load_amara_training_targets
from manga_prep.targets.pipe3d_phys_maps import load_amara_phys_training_targets


def map_coverage_stats(
    galaxy_dir: Path | str,
    *,
    target_source: str,
    target_keys: tuple[str, ...] | list[str],
    min_snr: float | None = None,
    require_sf_spaxel: bool = False,
) -> tuple[float, int, float]:
    """
    Return ``(n_valid_mean, footprint_n, fill_frac)``.

    ``n_valid_mean`` is the mean loss-mask count across ``target_keys``.
    ``fill_frac`` is the mean of (per-channel valid / footprint) — fair to small IFUs.
    """
    galaxy_dir = Path(galaxy_dir)
    keys = tuple(str(k) for k in target_keys)
    if not keys:
        return 0.0, 0, 0.0
    try:
        if target_source == "phys":
            bundle = load_amara_phys_training_targets(
                galaxy_dir,
                keys=keys,
                scaled=True,
                snr_min=min_snr,
                require_sf_spaxel=require_sf_spaxel,
            )
        else:
            bundle = load_amara_training_targets(galaxy_dir, scaled=True, keys=keys)
    except Exception:
        return 0.0, 0, 0.0

    footprint = np.asarray(bundle["footprint_mask"], dtype=np.uint8) > 0
    footprint_n = int(footprint.sum())
    if footprint_n <= 0:
        return 0.0, 0, 0.0

    fills: list[float] = []
    n_valid_sum = 0
    for key in keys:
        n = int(np.asarray(bundle["target_loss_masks"][key]).sum())
        n_valid_sum += n
        fills.append(n / float(footprint_n))
    n_valid_mean = float(n_valid_sum) / float(len(keys))
    fill_frac = float(np.mean(fills))
    return n_valid_mean, footprint_n, fill_frac


def _cache_path(
    data_root: Path,
    *,
    target_source: str,
    target_keys: tuple[str, ...],
    min_snr: float | None,
    require_sf_spaxel: bool,
) -> Path:
    payload = {
        "target_source": target_source,
        "target_keys": list(target_keys),
        "min_snr": min_snr,
        "require_sf_spaxel": bool(require_sf_spaxel),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return Path(data_root) / "stats" / f"map_coverage_{target_source}_{digest}.csv"


def load_or_build_coverage_table(
    data_root: Path | str,
    rows: list[dict],
    *,
    target_source: str,
    target_keys: tuple[str, ...],
    min_snr: float | None = None,
    require_sf_spaxel: bool = False,
    progress: bool = False,
) -> dict[str, tuple[float, int, float]]:
    """
    Map ``plateifu -> (n_valid_mean, footprint_n, fill_frac)``.

    Cached under ``data_root/stats/`` so full-dataset init stays cheap after the
    first scan.
    """
    data_root = Path(data_root)
    cache = _cache_path(
        data_root,
        target_source=target_source,
        target_keys=target_keys,
        min_snr=min_snr,
        require_sf_spaxel=require_sf_spaxel,
    )
    out: dict[str, tuple[float, int, float]] = {}
    if cache.is_file():
        with cache.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                out[row["plateifu"]] = (
                    float(row["n_valid_mean"]),
                    int(row["footprint_n"]),
                    float(row["fill_frac"]),
                )

    missing = [r for r in rows if r["plateifu"] not in out]
    if progress:
        if cache.is_file() and not missing:
            print(
                f"Map coverage cache hit: {cache.name} "
                f"({len(out):,} galaxies on disk)",
                flush=True,
            )
        elif missing:
            print(
                f"Map coverage: scoring {len(missing):,} galaxies "
                f"(cache={cache.name}"
                f"{', extending existing' if out else ', building new'})...",
                flush=True,
            )

    if missing:
        try:
            from tqdm import tqdm

            iterator = tqdm(missing, desc="Map coverage", unit="gal", disable=not progress)
        except Exception:
            iterator = missing
        for row in iterator:
            gdir = data_root / row["galaxy_dir"]
            stats = map_coverage_stats(
                gdir,
                target_source=target_source,
                target_keys=target_keys,
                min_snr=min_snr,
                require_sf_spaxel=require_sf_spaxel,
            )
            out[row["plateifu"]] = stats
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(
                fh,
                fieldnames=["plateifu", "n_valid_mean", "footprint_n", "fill_frac"],
            )
            w.writeheader()
            for plateifu in sorted(out):
                n_mean, fp_n, fill = out[plateifu]
                w.writerow(
                    {
                        "plateifu": plateifu,
                        "n_valid_mean": f"{n_mean:.6f}",
                        "footprint_n": fp_n,
                        "fill_frac": f"{fill:.6f}",
                    }
                )
        if progress:
            print(f"Map coverage cache written: {cache}", flush=True)
    return out
