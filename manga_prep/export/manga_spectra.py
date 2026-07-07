"""
Export MaNGA IFU spaxel spectra from local DRP LOGCUBE files into fast NPZ bundles.

Writes per galaxy:
    manga_sdss_fits/<plate>_<ifu>/amara_spectra.npz
    manga_sdss_fits/<plate>_<ifu>/amara_spectra_metadata.json

The NPZ stores wavelength plus FLUX/IVAR/MASK cubes in LOGCUBE axis order
(n_wave, ny, nx) as float32/uint8 for quick training-time loading.

Examples:
  python export_manga_spectra.py
  python export_manga_spectra.py 7495-3702
  python export_manga_spectra.py --workers 8 --skip-existing
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from astropy.io import fits

_PLATE_IFU_DIR = re.compile(r"^(\d+)_(\d+)$")


def parse_plateifu(s: str) -> tuple[str, str]:
    parts = s.replace(" ", "").split("-")
    if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
        raise ValueError(f"expected plate-ifu like 8485-1901, got {s!r}")
    return parts[0], parts[1]


def discover_plateifus_from_data_root(data_root: Path) -> list[str]:
    if not data_root.is_dir():
        return []
    out: list[str] = []
    for p in sorted(data_root.iterdir()):
        if not p.is_dir():
            continue
        m = _PLATE_IFU_DIR.match(p.name)
        if m:
            out.append(f"{m.group(1)}-{m.group(2)}")
    return out


def logcube_path(gal_dir: Path, plate: str, ifu: str) -> Path | None:
    for name in (
        f"manga-{plate}-{ifu}-LOGCUBE.fits.gz",
        f"manga-{plate}-{ifu}-LOGCUBE.fits",
    ):
        p = gal_dir / name
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def export_logcube_spectra(logcube: Path, *, include_ivar: bool = True) -> tuple[dict, dict]:
    with fits.open(logcube, memmap=True) as hdul:
        if "FLUX" not in hdul or hdul["FLUX"].data is None:
            raise ValueError(f"No FLUX extension in {logcube}")
        if "WAVE" not in hdul or hdul["WAVE"].data is None:
            raise ValueError(f"No WAVE extension in {logcube}")

        flux = np.asarray(hdul["FLUX"].data, dtype=np.float32)
        wave = np.asarray(hdul["WAVE"].data, dtype=np.float32)
        if flux.ndim != 3:
            raise ValueError(f"Expected FLUX to be 3D, got shape {flux.shape}")

        arrays: dict[str, np.ndarray] = {
            "wave": wave,
            "flux": flux,
            "native_shape": np.array(flux.shape[1:], dtype=np.int16),
            "n_wave": np.array(flux.shape[0], dtype=np.int32),
        }

        if include_ivar and "IVAR" in hdul and hdul["IVAR"].data is not None:
            arrays["ivar"] = np.asarray(hdul["IVAR"].data, dtype=np.float32)

        if "MASK" in hdul and hdul["MASK"].data is not None:
            arrays["mask"] = np.asarray(hdul["MASK"].data, dtype=np.uint8)

        header = hdul["FLUX"].header
        metadata = {
            "logcube_path": str(logcube),
            "plateifu": f"{header.get('PLATE', '')}-{header.get('IFUDSGN', header.get('IFU', ''))}".strip("-"),
            "wave_min": float(np.nanmin(wave)),
            "wave_max": float(np.nanmax(wave)),
            "n_wave": int(flux.shape[0]),
            "native_shape": [int(flux.shape[1]), int(flux.shape[2])],
            "flux_unit": str(header.get("BUNIT", "")),
            "axis_order": "flux[n_wave, y, x]; spectrum at spaxel (y,x) is flux[:, y, x]",
            "include_ivar": bool(include_ivar and "ivar" in arrays),
            "include_mask": "mask" in arrays,
        }

    # Spaxel has at least one finite flux value.
    arrays["spaxel_valid"] = np.isfinite(flux).any(axis=0).astype(np.uint8)
    metadata["valid_spaxel_count"] = int(arrays["spaxel_valid"].sum())
    return arrays, metadata


def write_amara_spectra(
    gal_dir: Path,
    *,
    include_ivar: bool = True,
    out_npz: str = "amara_spectra.npz",
    out_json: str = "amara_spectra_metadata.json",
) -> dict:
    plate, ifu = gal_dir.name.split("_", 1)
    logcube = logcube_path(gal_dir, plate, ifu)
    if logcube is None:
        raise FileNotFoundError(f"No LOGCUBE found in {gal_dir}")

    arrays, metadata = export_logcube_spectra(logcube, include_ivar=include_ivar)
    metadata["plateifu"] = f"{plate}-{ifu}"

    npz_path = gal_dir / out_npz
    json_path = gal_dir / out_json
    np.savez_compressed(npz_path, **arrays)
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "plateifu": metadata["plateifu"],
        "npz": npz_path,
        "metadata": json_path,
        **metadata,
    }


def load_amara_spectra(galaxy_dir: Path | str):
    """Load amara_spectra.npz from a manga_sdss_fits/<plate_ifu> folder."""
    npz_path = Path(galaxy_dir) / "amara_spectra.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}")
    return np.load(npz_path)


def export_one(job: tuple) -> dict:
    gal_dir, include_ivar, skip_existing = job
    npz_path = gal_dir / "amara_spectra.npz"
    if skip_existing and npz_path.exists() and npz_path.stat().st_size > 0:
        plate, ifu = gal_dir.name.split("_", 1)
        return {
            "plateifu": f"{plate}-{ifu}",
            "npz": npz_path,
            "skipped": True,
        }

    result = write_amara_spectra(gal_dir, include_ivar=include_ivar)
    result["skipped"] = False
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Export MaNGA LOGCUBE spaxel spectra to amara_spectra.npz in each "
            "manga_sdss_fits/<plate>_<ifu>/ folder."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "plateifu",
        nargs="*",
        help="Optional plate-ifu IDs. If omitted, all galaxy folders under --data-root are used.",
    )
    p.add_argument("--data-root", type=Path, default=Path("manga_sdss_fits"))
    p.add_argument("--workers", type=int, default=1, help="Parallel worker processes")
    p.add_argument("--skip-existing", action="store_true", help="Skip galaxies with amara_spectra.npz")
    p.add_argument("--no-ivar", action="store_true", help="Do not include IVAR cube in output")
    args = p.parse_args(argv)

    if args.plateifu:
        targets = list(args.plateifu)
    else:
        targets = discover_plateifus_from_data_root(args.data_root)
        if not targets:
            print(f"No galaxy folders found under {args.data_root}", file=sys.stderr)
            return 1

    jobs: list[tuple] = []
    for pi in targets:
        plate, ifu = parse_plateifu(pi)
        gal_dir = args.data_root / f"{plate}_{ifu}"
        if not gal_dir.is_dir():
            print(f"warning: missing folder {gal_dir}", file=sys.stderr)
            continue
        jobs.append((gal_dir, not args.no_ivar, args.skip_existing))

    if not jobs:
        print("No valid galaxy folders to process.", file=sys.stderr)
        return 1

    rows: list[dict] = []
    workers = max(1, int(args.workers))
    if workers == 1:
        for job in jobs:
            try:
                row = export_one(job)
                rows.append(row)
                action = "Skipped" if row.get("skipped") else "Wrote"
                print(f"{action} {row['plateifu']}: {row['npz']}")
            except Exception as e:
                plateifu = job[0].name.replace("_", "-")
                print(f"error: {plateifu}: {e}", file=sys.stderr)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(export_one, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                plateifu = job[0].name.replace("_", "-")
                try:
                    row = future.result()
                    rows.append(row)
                    action = "Skipped" if row.get("skipped") else "Wrote"
                    print(f"{action} {row['plateifu']}: {row['npz']}")
                except Exception as e:
                    print(f"error: {plateifu}: {e}", file=sys.stderr)

    if rows:
        manifest = args.data_root / "amara_spectra_manifest.csv"
        import csv

        fieldnames = sorted({k for row in rows for k in row.keys()})
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in sorted(rows, key=lambda r: r["plateifu"]):
                writer.writerow(row)
        print(f"Manifest: {manifest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
