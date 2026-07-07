"""
Pre-export SDSS/Legacy cutouts aligned to the Amara grid (one-time, slow).

Training then loads small NPZ stacks instead of reprojecting FITS every sample.

Example:
  python -m manga_prep.export_aligned_imaging --survey sdss --workers 4
  python -m manga_prep.export_aligned_imaging --survey all --skip-existing --workers 8
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from manga_prep.io.aligned_cache import export_legacy_aligned, export_sdss_aligned
from manga_prep.dataset.index import _DIR_RE, read_manga_dataset_index


def _galaxy_dirs(data_root: Path) -> list[Path]:
    return sorted(p for p in data_root.iterdir() if p.is_dir() and _DIR_RE.match(p.name))


def _export_one(args: tuple[str, str, bool]) -> tuple[str, str | None]:
    gal_dir_str, survey, skip_existing = args
    gal_dir = Path(gal_dir_str)
    try:
        if survey in ("sdss", "all"):
            export_sdss_aligned(gal_dir, skip_existing=skip_existing)
        if survey in ("legacy", "all"):
            export_legacy_aligned(gal_dir, skip_existing=skip_existing)
        return gal_dir.name.replace("_", "-"), None
    except Exception as exc:
        return gal_dir.name.replace("_", "-"), str(exc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export aligned imaging NPZ caches for fast training.")
    parser.add_argument("--data-root", type=Path, default=Path("manga_sdss_fits"))
    parser.add_argument(
        "--survey",
        choices=("sdss", "legacy", "all"),
        default="sdss",
        help="Which imaging stacks to export.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--use-index",
        action="store_true",
        help="Only process galaxies flagged in manga_dataset_index.csv (recommended).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most N galaxies (debug).")
    args = parser.parse_args(argv)

    if not args.data_root.is_dir():
        raise SystemExit(f"Missing data root: {args.data_root}")

    galaxy_dirs = _galaxy_dirs(args.data_root)
    if args.use_index:
        index_path = args.data_root / "manga_dataset_index.csv"
        if not index_path.is_file():
            raise SystemExit(f"Missing index: {index_path} (run: python -m manga_prep build-index)")
        rows = read_manga_dataset_index(index_path)
        eligible: set[Path] = set()
        for row in rows:
            gal_dir = args.data_root / row["galaxy_dir"]
            if args.survey in ("sdss", "all") and row.get("has_sdss_imaging"):
                eligible.add(gal_dir)
            if args.survey in ("legacy", "all") and row.get("has_legacy_imaging"):
                eligible.add(gal_dir)
        galaxy_dirs = sorted(eligible)
    if args.limit is not None:
        galaxy_dirs = galaxy_dirs[: args.limit]

    tasks = [(str(gal_dir), args.survey, args.skip_existing) for gal_dir in galaxy_dirs]
    errors: list[tuple[str, str]] = []

    if args.workers <= 1:
        for task in tqdm(tasks, desc="Export aligned imaging", unit="galaxy"):
            plateifu, err = _export_one(task)
            if err:
                errors.append((plateifu, err))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_export_one, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Export aligned imaging", unit="galaxy"):
                plateifu, err = future.result()
                if err:
                    errors.append((plateifu, err))

    print(f"Processed {len(tasks)} galaxies under {args.data_root}")
    if errors:
        print(f"Errors: {len(errors)}")
        for plateifu, err in errors[:10]:
            print(f"  {plateifu}: {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    else:
        print("Done with no errors.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
