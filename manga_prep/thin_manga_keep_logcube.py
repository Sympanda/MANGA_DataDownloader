"""
Remove extra MaNGA FITS under each galaxy folder, keeping only the DRP LOGCUBE.

Keeps (if present):
  manga-<plate>-<ifu>-LOGCUBE.fits.gz
  manga-<plate>-<ifu>-LOGCUBE.fits   (uncompressed variant)

Deletes other MaNGA SAS-style products in that folder, e.g. MAPS, DAP model
LOGCUBE-*, LINCUBE, RSS, Pipe3D, etc.

By default does NOT touch sdss_cutouts/ or legacy_cutouts/ (use --cutouts).

Default is dry-run; pass --apply to delete files.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

_DIR_RE = re.compile(r"^(\d+)_(\d+)$")


def discover_galaxy_dirs(data_root: Path) -> list[Path]:
    if not data_root.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(data_root.iterdir()):
        if p.is_dir() and _DIR_RE.match(p.name):
            out.append(p)
    return out


def logcube_keep_names(plate: str, ifu: str) -> set[str]:
    return {
        f"manga-{plate}-{ifu}-LOGCUBE.fits.gz",
        f"manga-{plate}-{ifu}-LOGCUBE.fits",
    }


def manga_fits_to_delete(gal_dir: Path, plate: str, ifu: str) -> list[Path]:
    """Paths to delete: MaNGA FITS in gal_dir root, excluding DRP LOGCUBE."""
    keep = logcube_keep_names(plate, ifu)
    out: list[Path] = []
    for p in gal_dir.iterdir():
        if not p.is_file():
            continue
        n = p.name
        if n in keep:
            continue
        if n.startswith(f"manga-{plate}-{ifu}-") or n.startswith(f"manga-{plate}-{ifu}."):
            out.append(p)
    return sorted(out)


def cutout_dirs(gal_dir: Path) -> list[Path]:
    return [gal_dir / "sdss_cutouts", gal_dir / "legacy_cutouts"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Delete MaNGA FITS except DRP LOGCUBE in each <plate>_<ifu> folder. "
            "Default: dry-run only; use --apply to delete."
        )
    )
    p.add_argument("--data-root", type=Path, default=Path("manga_sdss_fits"))
    p.add_argument(
        "--cutouts",
        action="store_true",
        help="Also remove sdss_cutouts/ and legacy_cutouts/ under each galaxy (entire dirs).",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files and dirs (default is list-only dry run).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N galaxy folders (0 = all).",
    )
    args = p.parse_args(argv)

    if not args.data_root.is_dir():
        print(f"Missing data root: {args.data_root}", file=sys.stderr)
        return 1

    dirs = discover_galaxy_dirs(args.data_root)
    if args.limit > 0:
        dirs = dirs[: args.limit]

    n_files = 0
    n_bytes = 0
    n_dirs = 0
    n_dir_bytes = 0

    for gal_dir in dirs:
        m = _DIR_RE.match(gal_dir.name)
        if not m:
            continue
        plate, ifu = m.group(1), m.group(2)
        for path in manga_fits_to_delete(gal_dir, plate, ifu):
            sz = path.stat().st_size if path.exists() else 0
            n_files += 1
            n_bytes += sz
            rel = path.relative_to(args.data_root)
            if args.apply:
                path.unlink(missing_ok=True)
                print(f"deleted file: {rel}")
            else:
                print(f"would delete: {rel}  ({sz / (1024**2):.1f} MiB)")

        if args.cutouts:
            for d in cutout_dirs(gal_dir):
                if not d.is_dir():
                    continue
                du = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                n_dirs += 1
                n_dir_bytes += du
                rel = d.relative_to(args.data_root)
                if args.apply:
                    shutil.rmtree(d, ignore_errors=False)
                    print(f"deleted dir:  {rel}/")
                else:
                    print(f"would delete dir: {rel}/  ({du / (1024**2):.1f} MiB)")

    mode = "APPLIED" if args.apply else "dry-run"
    print(
        f"\nSummary ({mode}): galaxies={len(dirs)}  "
        f"extra_manga_files={n_files}  (~{n_bytes / (1024**3):.2f} GiB)"
        + (f"  cutout_dirs={n_dirs}  (~{n_dir_bytes / (1024**3):.2f} GiB)" if args.cutouts else "")
    )
    if not args.apply and (n_files or (args.cutouts and n_dirs)):
        print("Re-run with --apply to delete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
