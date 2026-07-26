"""
Pre-export SDSS/Legacy cutouts aligned for fast training I/O.

Grids:
  amara        — Amara FoV / orientation (76×76, optional --oversample)
  sdss_native  — Amara-oriented, SDSS plate scale on a 196×196 canvas (HR)

Example:
  python -m manga_prep export-aligned-imaging --survey sdss --use-index --skip-existing --workers 8
  python -m manga_prep export-aligned-imaging --config config.jsonc --survey sdss --skip-existing --workers 8
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from manga_prep.dataset.index import _DIR_RE, read_manga_dataset_index
from manga_prep.io.aligned_cache import ImagingGrid, export_legacy_aligned, export_sdss_aligned
from manga_prep.io.imaging_alignment import SDSS_NATIVE_CANVAS
from src.config_loader import load_jsonc


def _galaxy_dirs(data_root: Path) -> list[Path]:
    return sorted(p for p in data_root.iterdir() if p.is_dir() and _DIR_RE.match(p.name))


def _export_one(args: tuple[str, str, bool, str, int, int]) -> tuple[str, str | None]:
    gal_dir_str, survey, skip_existing, grid, oversample, canvas = args
    gal_dir = Path(gal_dir_str)
    try:
        if survey in ("sdss", "all"):
            export_sdss_aligned(
                gal_dir,
                skip_existing=skip_existing,
                oversample=oversample,
                grid=grid,  # type: ignore[arg-type]
                canvas=canvas,
            )
        if survey in ("legacy", "all"):
            export_legacy_aligned(
                gal_dir,
                skip_existing=skip_existing,
                oversample=oversample,
                grid=grid,  # type: ignore[arg-type]
                canvas=canvas,
            )
        return gal_dir.name.replace("_", "-"), None
    except Exception as exc:
        return gal_dir.name.replace("_", "-"), str(exc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export aligned imaging NPZ caches for fast training.")
    parser.add_argument("--config", type=Path, default=None, help="Optional config.jsonc for paths / grid")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--survey",
        choices=("sdss", "legacy", "all"),
        default="sdss",
        help="Which imaging stacks to export.",
    )
    parser.add_argument(
        "--grid",
        choices=("amara", "sdss_native"),
        default=None,
        help="Output grid (default: from config imaging_resolution, else amara).",
    )
    parser.add_argument(
        "--oversample",
        type=int,
        default=None,
        help="Amara-grid oversample (ignored for sdss_native). Default 1.",
    )
    parser.add_argument("--canvas", type=int, default=SDSS_NATIVE_CANVAS, help="sdss_native canvas size")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--use-index",
        action="store_true",
        help="Only process galaxies flagged in manga_dataset_index.csv (recommended).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most N galaxies (debug).")
    args = parser.parse_args(argv)

    data_root = args.data_root
    grid: ImagingGrid | None = args.grid  # type: ignore[assignment]
    oversample = args.oversample

    if args.config is not None:
        cfg = load_jsonc(args.config)
        data_top = cfg.get("data", {})
        model_top = cfg.get("model", {})
        if data_root is None:
            data_root = Path(data_top.get("data_root", "manga_sdss_fits"))
        if grid is None:
            resolution = str(
                model_top.get(
                    "imaging_resolution",
                    data_top.get("imaging_resolution", "aligned"),
                )
            )
            grid = "sdss_native" if resolution == "native" else "amara"
        if oversample is None and data_top.get("aligned_oversample") is not None:
            oversample = int(data_top["aligned_oversample"])

    if data_root is None:
        data_root = Path("manga_sdss_fits")
    if grid is None:
        grid = "amara"
    if oversample is None:
        oversample = 1
    if grid == "sdss_native":
        oversample = 1
    if int(oversample) < 1:
        raise SystemExit(f"--oversample must be >= 1, got {oversample}")

    if not data_root.is_dir():
        raise SystemExit(f"Missing data root: {data_root}")

    galaxy_dirs = _galaxy_dirs(data_root)
    if args.use_index or args.config is not None:
        index_path = data_root / "manga_dataset_index.csv"
        if not index_path.is_file():
            raise SystemExit(f"Missing index: {index_path} (run: python -m manga_prep build-index)")
        rows = read_manga_dataset_index(index_path)
        eligible: set[Path] = set()
        for row in rows:
            gal_dir = data_root / row["galaxy_dir"]
            if args.survey in ("sdss", "all") and row.get("has_sdss_imaging"):
                eligible.add(gal_dir)
            if args.survey in ("legacy", "all") and row.get("has_legacy_imaging"):
                eligible.add(gal_dir)
        galaxy_dirs = sorted(eligible)
    if args.limit is not None:
        galaxy_dirs = galaxy_dirs[: args.limit]

    tasks = [
        (str(gal_dir), args.survey, args.skip_existing, grid, int(oversample), int(args.canvas))
        for gal_dir in galaxy_dirs
    ]
    errors: list[tuple[str, str]] = []

    print(
        f"Exporting aligned imaging grid={grid} oversample={oversample} under {data_root}",
        flush=True,
    )
    if args.workers <= 1:
        for task in tqdm(tasks, desc="Export aligned imaging", unit="galaxy"):
            plateifu, err = _export_one(task)
            if err:
                errors.append((plateifu, err))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_export_one, task) for task in tasks]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Export aligned imaging",
                unit="galaxy",
            ):
                plateifu, err = future.result()
                if err:
                    errors.append((plateifu, err))

    print(f"Processed {len(tasks)} galaxies under {data_root} (grid={grid})")
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
