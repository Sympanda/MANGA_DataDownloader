"""
Compute training-split asinh soft scales ``s_b`` for imaging bands and spectra.

For each channel, ``s_b`` is the p-th percentile of |flux| over finite samples
(footprint-masked for imaging). Saves 95 / 99 / 99.5 in one JSON under the data root.

Example:
  python -m manga_prep compute-input-scales --config config.jsonc
  python -m manga_prep compute-input-scales --data-root manga_sdss_fits --split-csv manga_sdss_fits/splits/default_split.csv
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from manga_prep.dataset.manga_dataset import (
    MangaGalaxyDataset,
    _LEGACY_BANDS,
    _SDSS_BANDS,
)
from manga_prep.io.input_scales import (
    DEFAULT_PERCENTILES,
    DEFAULT_SCALES_PATH,
    percentile_key,
    save_input_scales,
)
from src.config_loader import load_jsonc
from src.data.splits import filter_rows_by_split


def _sample_abs_finite(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    max_samples: int,
) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).ravel()
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return np.empty(0, dtype=np.float64)
    abs_vals = np.abs(finite)
    if abs_vals.size <= max_samples:
        return abs_vals
    idx = rng.choice(abs_vals.size, size=max_samples, replace=False)
    return abs_vals[idx]


def _percentiles_from_chunks(
    chunks: list[np.ndarray],
    percentiles: tuple[float, ...],
) -> dict[str, float]:
    if not chunks:
        raise RuntimeError("No samples collected for percentile estimate")
    stacked = np.concatenate(chunks, axis=0)
    if stacked.size == 0:
        raise RuntimeError("Empty sample buffer for percentile estimate")
    qs = np.percentile(stacked, list(percentiles))
    return {percentile_key(p): float(v) for p, v in zip(percentiles, qs)}


def _footprint_on_imaging(
    footprint: np.ndarray,
    imaging_hw: tuple[int, int],
) -> np.ndarray:
    """Nearest-resize Amara footprint onto the imaging canvas when sizes differ."""
    if footprint.shape == imaging_hw:
        return footprint
    # Integer tile when imaging is an exact Amara oversample.
    oy = imaging_hw[0] // footprint.shape[0]
    ox = imaging_hw[1] // footprint.shape[1]
    if oy * footprint.shape[0] == imaging_hw[0] and ox * footprint.shape[1] == imaging_hw[1]:
        return np.repeat(np.repeat(footprint, oy, axis=0), ox, axis=1)
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(footprint.astype(np.float32))[None, None]
    out = F.interpolate(t, size=imaging_hw, mode="nearest")
    return out[0, 0].numpy() > 0.5


def compute_input_scales(
    *,
    data_root: Path,
    split_csv: Path,
    split: str = "train",
    imaging_resolution: str = "aligned",
    aligned_oversample: int | None = None,
    percentiles: tuple[float, ...] = DEFAULT_PERCENTILES,
    max_samples_per_galaxy: int = 8192,
    seed: int = 0,
    limit: int | None = None,
    use_legacy: bool = False,
) -> dict[str, Any]:
    imaging_grid = "sdss_native" if imaging_resolution == "native" else "amara"
    oversample = 1 if imaging_grid == "sdss_native" else (
        int(aligned_oversample) if aligned_oversample is not None else 1
    )

    # Imaging + footprint for mask; spectra loaded in dedicated passes.
    ds_img = MangaGalaxyDataset(
        data_root,
        data_root / "manga_dataset_index.csv",
        include_sdss_imaging=True,
        include_legacy_imaging=use_legacy,
        include_targets=True,
        spectrum=None,
        require_all=False,
        align_imaging_to_amara_grid=True,
        prefer_aligned_cache=True,
        imaging_grid=imaging_grid,  # type: ignore[arg-type]
        aligned_oversample=oversample,
    )
    rows = filter_rows_by_split(ds_img.rows, split_csv, split)  # type: ignore[arg-type]
    if limit is not None:
        rows = rows[: int(limit)]
    if not rows:
        raise SystemExit(f"No galaxies for split={split!r} in {split_csv}")

    # Restrict dataset rows to the filtered set (preserve loading helpers).
    ds_img.rows = rows

    rng = np.random.default_rng(seed)
    sdss_chunks: list[list[np.ndarray]] = [[] for _ in _SDSS_BANDS]
    legacy_chunks: list[list[np.ndarray]] = [[] for _ in _LEGACY_BANDS]
    n_sdss = 0
    n_legacy = 0

    for i in tqdm(range(len(ds_img)), desc="imaging scales"):
        sample = ds_img[i]
        footprint = np.asarray(sample["footprint_mask"], dtype=np.float32) > 0.5
        if "sdss_imaging" in sample.get("inputs", {}):
            data = np.asarray(sample["inputs"]["sdss_imaging"]["data"], dtype=np.float32)
            footprint_img = _footprint_on_imaging(footprint, data.shape[-2:])
            for c in range(min(data.shape[0], len(_SDSS_BANDS))):
                vals = _sample_abs_finite(
                    data[c][footprint_img],
                    rng=rng,
                    max_samples=max_samples_per_galaxy,
                )
                if vals.size:
                    sdss_chunks[c].append(vals)
            n_sdss += 1
        if use_legacy and "legacy_imaging" in sample.get("inputs", {}):
            data = np.asarray(sample["inputs"]["legacy_imaging"]["data"], dtype=np.float32)
            bands = sample["inputs"]["legacy_imaging"]["bands"]
            footprint_img = _footprint_on_imaging(footprint, data.shape[-2:])
            for c, _band in enumerate(bands):
                if c >= len(legacy_chunks):
                    break
                vals = _sample_abs_finite(
                    data[c][footprint_img],
                    rng=rng,
                    max_samples=max_samples_per_galaxy,
                )
                if vals.size:
                    legacy_chunks[c].append(vals)
            n_legacy += 1

    sdss_block = {
        "bands": list(_SDSS_BANDS),
        "n_galaxies": n_sdss,
        "n_samples": [int(sum(ch.size for ch in chunks)) for chunks in sdss_chunks],
        "scales": {
            percentile_key(p): [
                float(np.percentile(np.concatenate(chunks), p)) if chunks else float("nan")
                for chunks in sdss_chunks
            ]
            for p in percentiles
        },
    }
    for key, vals in sdss_block["scales"].items():
        if any(not np.isfinite(v) or v <= 0 for v in vals):
            raise RuntimeError(f"Invalid SDSS scales at {key}: {vals}")

    legacy_block: dict[str, Any] | None = None
    if use_legacy:
        legacy_block = {
            "bands": list(_LEGACY_BANDS),
            "n_galaxies": n_legacy,
            "n_samples": [int(sum(ch.size for ch in chunks)) for chunks in legacy_chunks],
            "scales": {
                percentile_key(p): [
                    float(np.percentile(np.concatenate(chunks), p)) if chunks else float("nan")
                    for chunks in legacy_chunks
                ]
                for p in percentiles
            },
        }

    def _spectrum_block(mode: str) -> dict[str, Any]:
        ds = MangaGalaxyDataset(
            data_root,
            data_root / "manga_dataset_index.csv",
            include_sdss_imaging=False,
            include_legacy_imaging=False,
            include_targets=False,
            spectrum=mode,  # type: ignore[arg-type]
            spectrum_fallback=False,
            require_all=False,
            resample_spectrum=True,
        )
        spec_rows = filter_rows_by_split(ds.rows, split_csv, split)  # type: ignore[arg-type]
        # Keep only galaxies that actually have this spectrum type.
        flag = "has_fake_spectrum" if mode == "fake" else "has_real_spectrum"
        spec_rows = [r for r in spec_rows if r.get(flag)]
        if limit is not None:
            spec_rows = spec_rows[: int(limit)]
        ds.rows = spec_rows
        chunks: list[np.ndarray] = []
        for i in tqdm(range(len(ds)), desc=f"spectrum_{mode} scales"):
            sample = ds[i]
            flux = np.asarray(sample["inputs"]["spectrum"]["flux"], dtype=np.float32)
            vals = _sample_abs_finite(flux, rng=rng, max_samples=max_samples_per_galaxy)
            if vals.size:
                chunks.append(vals)
        scales = _percentiles_from_chunks(chunks, percentiles)
        return {
            "n_galaxies": len(ds),
            "n_samples": int(sum(ch.size for ch in chunks)),
            "scales": scales,
        }

    payload: dict[str, Any] = {
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root).replace("\\", "/"),
        "split_csv": str(split_csv).replace("\\", "/"),
        "split": split,
        "n_galaxies_imaging": len(rows),
        "imaging_resolution": imaging_resolution,
        "imaging_grid": imaging_grid,
        "aligned_oversample": oversample,
        "percentiles": [float(p) for p in percentiles],
        "method": "abs_finite",
        "footprint_masked": True,
        "max_samples_per_galaxy": int(max_samples_per_galaxy),
        "seed": int(seed),
        "sdss": sdss_block,
        "legacy": legacy_block,
        "spectrum_fake": _spectrum_block("fake"),
        "spectrum_real": _spectrum_block("real"),
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute train-split asinh soft scales for imaging + spectra."
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional config.jsonc for paths")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split-csv", type=Path, default=None)
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--imaging-resolution", choices=("aligned", "native"), default=None)
    parser.add_argument("--aligned-oversample", type=int, default=None)
    parser.add_argument("--use-legacy", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output JSON (default: {DEFAULT_SCALES_PATH})",
    )
    parser.add_argument("--max-samples-per-galaxy", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Debug: first N train galaxies")
    args = parser.parse_args(argv)

    data_root = args.data_root
    split_csv = args.split_csv
    imaging_resolution = args.imaging_resolution
    aligned_oversample = args.aligned_oversample
    use_legacy = bool(args.use_legacy)
    out_path = args.out

    if args.config is not None:
        cfg = load_jsonc(args.config)
        data_top = cfg.get("data", {})
        model_top = cfg.get("model", {})
        if data_root is None:
            data_root = Path(data_top.get("data_root", "manga_sdss_fits"))
        if split_csv is None:
            split_csv = Path(
                data_top.get("split", {}).get(
                    "split_csv_path", "manga_sdss_fits/splits/default_split.csv"
                )
            )
        if imaging_resolution is None:
            imaging_resolution = str(
                model_top.get("imaging_resolution", data_top.get("imaging_resolution", "aligned"))
            )
        if aligned_oversample is None and data_top.get("aligned_oversample") is not None:
            aligned_oversample = int(data_top["aligned_oversample"])
        use_legacy = use_legacy or bool(data_top.get("use_legacy", False))
        if out_path is None:
            norm = model_top.get("input_norm", {}) or {}
            if norm.get("scales_path"):
                out_path = Path(norm["scales_path"])

    if data_root is None:
        data_root = Path("manga_sdss_fits")
    if split_csv is None:
        split_csv = data_root / "splits" / "default_split.csv"
    if imaging_resolution is None:
        imaging_resolution = "aligned"
    if out_path is None:
        out_path = data_root / "stats" / "input_asinh_scales.json"

    payload = compute_input_scales(
        data_root=data_root,
        split_csv=split_csv,
        split=args.split,
        imaging_resolution=imaging_resolution,
        aligned_oversample=aligned_oversample,
        percentiles=DEFAULT_PERCENTILES,
        max_samples_per_galaxy=args.max_samples_per_galaxy,
        seed=args.seed,
        limit=args.limit,
        use_legacy=use_legacy,
    )
    save_input_scales(out_path, payload)
    print(f"Wrote {out_path}")
    print(f"  SDSS p99: {payload['sdss']['scales']['99']}")
    print(f"  spectrum_fake p99: {payload['spectrum_fake']['scales']['99']}")
    if payload.get("spectrum_real"):
        print(f"  spectrum_real p99: {payload['spectrum_real']['scales']['99']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
