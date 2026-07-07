"""
Export SDSS-fiber-like aperture spectra from MaNGA LOGCUBE spaxels.

For each galaxy with a local LOGCUBE, coadd spaxel spectra within a circular
aperture (default 3 arcsec diameter, matching legacy SDSS fibers).

Examples:
  # All galaxies under manga_sdss_fits/ that have a LOGCUBE
  python export_manga_aperture_spectra.py --workers 8

  python export_manga_aperture_spectra.py 7495-3702
  python export_manga_aperture_spectra.py --aperture-diameter 2 --workers 8
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from manga_prep.download.sdss_cutouts import discover_plateifus_from_data_root, parse_plateifu
from manga_prep.io.aperture_spectrum import (
    BOSS_FIBER_DIAMETER_ARCSEC,
    DEFAULT_APERTURE_DIAMETER_ARCSEC,
    DEFAULT_SUBPIXELS_PER_SPAXEL,
    SDSS_LEGACY_FIBER_DIAMETER_ARCSEC,
    write_fake_sdss_spectrum,
)


def export_one(job: tuple) -> dict:
    gal_dir, aperture_diameter, subpixels, skip_existing = job
    plate, ifu = gal_dir.name.split("_", 1)
    dia_tag = int(round(aperture_diameter * 10))
    npz_path = gal_dir / "fake_sdss_spectra" / f"manga-{plate}-{ifu}-fake-sdss-spectrum-{dia_tag}mas.npz"
    if skip_existing and npz_path.exists() and npz_path.stat().st_size > 0:
        return {"plateifu": f"{plate}-{ifu}", "npz": npz_path, "skipped": True}

    result = write_fake_sdss_spectrum(
        gal_dir,
        aperture_diameter_arcsec=aperture_diameter,
        subpixels=subpixels,
    )
    result["skipped"] = False
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build fake SDSS-fiber-like spectra from MaNGA LOGCUBE spaxels for every "
            "local galaxy folder (or selected plate-ifus)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("plateifu", nargs="*", help="Optional plate-ifu IDs.")
    parser.add_argument("--data-root", type=Path, default=Path("manga_sdss_fits"))
    parser.add_argument(
        "--aperture-diameter",
        type=float,
        default=DEFAULT_APERTURE_DIAMETER_ARCSEC,
        help=(
            "Circular aperture diameter in arcsec. "
            f"SDSS legacy={SDSS_LEGACY_FIBER_DIAMETER_ARCSEC}, BOSS={BOSS_FIBER_DIAMETER_ARCSEC}."
        ),
    )
    parser.add_argument(
        "--subpixels",
        type=int,
        default=DEFAULT_SUBPIXELS_PER_SPAXEL,
        help="Subpixel grid per spaxel edge for overlap integration.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)

    if args.plateifu:
        targets = list(args.plateifu)
    else:
        targets = discover_plateifus_from_data_root(args.data_root)
        if not targets:
            print(f"No galaxy folders found under {args.data_root}", file=sys.stderr)
            return 1

    jobs = []
    for pi in targets:
        plate, ifu = parse_plateifu(pi)
        gal_dir = args.data_root / f"{plate}_{ifu}"
        if not gal_dir.is_dir():
            print(f"warning: missing folder {gal_dir}", file=sys.stderr)
            continue
        jobs.append((gal_dir, float(args.aperture_diameter), int(args.subpixels), args.skip_existing))

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
            except Exception as exc:
                print(f"error: {job[0].name.replace('_', '-')}: {exc}", file=sys.stderr)
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
                except Exception as exc:
                    print(f"error: {plateifu}: {exc}", file=sys.stderr)

    if rows:
        manifest = args.data_root / "fake_sdss_spectra_manifest.csv"
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
