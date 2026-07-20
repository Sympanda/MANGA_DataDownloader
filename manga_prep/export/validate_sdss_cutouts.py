"""
Scan SDSS ugriz cutouts for shape / metadata consistency.

Flags galaxies where band FITS differ in size, are missing, or are stale
(present on disk but not listed in metadata.json ugriz_files).

Usage:
  python -m manga_prep validate-sdss-cutouts
  python -m manga_prep validate-sdss-cutouts --data-root manga_sdss_fits --csv bad_cutouts.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from manga_prep.dataset.index import _SDSS_BANDS, _fits_shape

_ISSUE_FIELDS = (
    "plateifu",
    "issue",
    "expected_size_px",
    "band_shapes",
    "stale_bands",
    "missing_bands",
)


def _sdss_paths(gal_dir: Path, plate: str, ifu: str) -> dict[str, Path]:
    return {
        b: gal_dir / "sdss_cutouts" / f"sdss-{plate}-{ifu}-{b}.fits"
        for b in _SDSS_BANDS
    }


def inspect_galaxy(gal_dir: Path) -> list[dict[str, str]]:
    m = gal_dir.name.split("_", 1)
    if len(m) != 2:
        return []
    plate, ifu = m
    plateifu = f"{plate}-{ifu}"
    cutout_dir = gal_dir / "sdss_cutouts"
    if not cutout_dir.is_dir():
        return [{"plateifu": plateifu, "issue": "no_sdss_cutouts_dir"}]

    meta_path = cutout_dir / "metadata.json"
    meta: dict = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return [{"plateifu": plateifu, "issue": "bad_metadata_json"}]

    expected = meta.get("size_px")
    listed = set((meta.get("ugriz_files") or {}).keys())
    paths = _sdss_paths(gal_dir, plate, ifu)

    shapes: dict[str, tuple[int, int] | None] = {b: _fits_shape(paths[b]) for b in _SDSS_BANDS}
    missing = [b for b, sh in shapes.items() if sh is None]
    stale = [b for b in _SDSS_BANDS if b not in listed and shapes[b] is not None]
    present_shapes = {sh for sh in shapes.values() if sh is not None}

    issues: list[dict[str, str]] = []
    row_base = {
        "plateifu": plateifu,
        "expected_size_px": str(expected) if expected is not None else "",
        "band_shapes": str({b: shapes[b] for b in _SDSS_BANDS}),
        "stale_bands": ",".join(stale),
        "missing_bands": ",".join(missing),
    }

    if missing:
        issues.append({**row_base, "issue": "missing_bands"})
    if stale:
        issues.append({**row_base, "issue": "stale_bands"})
    if len(present_shapes) > 1:
        issues.append({**row_base, "issue": "mixed_band_shapes"})
    if expected is not None and len(present_shapes) == 1:
        sh = next(iter(present_shapes))
        if sh != (int(expected), int(expected)):
            issues.append({**row_base, "issue": "size_mismatch_metadata"})

    return issues


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate SDSS ugriz cutout consistency.")
    p.add_argument("--data-root", type=Path, default=Path("manga_sdss_fits"))
    p.add_argument("--csv", type=Path, default=None, help="Write issue rows to CSV")
    p.add_argument("--limit", type=int, default=20, help="Max example rows to print per issue type")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = args.data_root
    if not data_root.is_dir():
        print(f"Missing data root: {data_root}", file=sys.stderr)
        return 1

    all_issues: list[dict[str, str]] = []
    gal_dirs = sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.count("_") == 1)
    for gal_dir in gal_dirs:
        all_issues.extend(inspect_galaxy(gal_dir))

    by_issue: dict[str, list[dict[str, str]]] = {}
    for row in all_issues:
        by_issue.setdefault(row["issue"], []).append(row)

    n_gal = len(gal_dirs)
    n_bad = len({r["plateifu"] for r in all_issues})
    print(f"Scanned {n_gal} galaxies; {n_bad} with at least one issue.")
    for issue, rows in sorted(by_issue.items()):
        print(f"  {issue}: {len(rows)}")
        for row in rows[: args.limit]:
            extra = ""
            if row.get("stale_bands"):
                extra += f" stale=[{row['stale_bands']}]"
            if row.get("missing_bands"):
                extra += f" missing=[{row['missing_bands']}]"
            print(f"    {row['plateifu']}{extra}  shapes={row['band_shapes']}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_ISSUE_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_issues)
        print(f"Wrote {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
