"""
Inventory completeness for local MaNGA + SDSS/Legacy cutout folders.

Scans folders like:
  manga_sdss_fits/<plate>_<ifu>/

Reports:
- complete_all_ifu_and_cutouts: DRP LOGCUBE + DAP MAPS + DAP model LOGCUBE + Pipe3D + full cutouts
- logcube_and_cutouts_only: DRP LOGCUBE + full cutouts, but missing all other IFU products
- missing_all_ifu: no IFU FITS files found at all
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


_DIR_RE = re.compile(r"^(\d+)_(\d+)$")
_UGRIZ = ("u", "g", "r", "i", "z")
_GRIZ = ("g", "r", "i", "z")
_GRZ = ("g", "r", "z")


@dataclass
class GalaxyFlags:
    plateifu: str
    has_drp_logcube: bool
    has_dap_maps: bool
    has_dap_model_logcube: bool
    has_pipe3d: bool
    has_any_ifu: bool
    has_cutout_jpeg: bool
    has_cutout_ugriz_all: bool
    has_cutout_metadata: bool
    has_cutouts_full: bool
    has_legacy_jpeg: bool
    has_legacy_grz_all: bool
    has_legacy_griz_all: bool
    has_legacy_i: bool
    has_legacy_metadata: bool


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _first_match_nonempty(gal_dir: Path, pattern: str) -> bool:
    return any(_nonempty(p) for p in gal_dir.glob(pattern))


def inspect_galaxy(gal_dir: Path) -> GalaxyFlags:
    m = _DIR_RE.match(gal_dir.name)
    if not m:
        raise ValueError(f"unexpected folder name: {gal_dir.name}")
    plate, ifu = m.group(1), m.group(2)
    plateifu = f"{plate}-{ifu}"

    has_drp_logcube = _nonempty(gal_dir / f"manga-{plate}-{ifu}-LOGCUBE.fits.gz")
    has_dap_maps = _first_match_nonempty(gal_dir, f"manga-{plate}-{ifu}-MAPS-*.fits.gz")
    has_dap_model_logcube = _first_match_nonempty(
        gal_dir, f"manga-{plate}-{ifu}-LOGCUBE-*.fits.gz"
    )
    has_pipe3d = _nonempty(gal_dir / f"manga-{plate}-{ifu}.Pipe3D.cube.fits.gz")

    has_any_ifu = any(
        (
            has_drp_logcube,
            has_dap_maps,
            has_dap_model_logcube,
            has_pipe3d,
            _first_match_nonempty(gal_dir, f"manga-{plate}-{ifu}-LINCUBE.fits.gz"),
            _first_match_nonempty(gal_dir, f"manga-{plate}-{ifu}-LOGRSS.fits.gz"),
            _first_match_nonempty(gal_dir, f"manga-{plate}-{ifu}-LINRSS.fits.gz"),
        )
    )

    cut = gal_dir / "sdss_cutouts"
    has_cutout_jpeg = _nonempty(cut / f"sdss-{plate}-{ifu}-color.jpg")
    has_cutout_ugriz_all = all(_nonempty(cut / f"sdss-{plate}-{ifu}-{b}.fits") for b in _UGRIZ)
    has_cutout_metadata = _nonempty(cut / "metadata.json")
    has_cutouts_full = has_cutout_jpeg and has_cutout_ugriz_all and has_cutout_metadata

    lcut = gal_dir / "legacy_cutouts"
    has_legacy_jpeg = _nonempty(lcut / f"legacy-{plate}-{ifu}-color.jpg")
    has_legacy_grz_all = all(_nonempty(lcut / f"legacy-{plate}-{ifu}-{b}.fits") for b in _GRZ)
    has_legacy_griz_all = all(_nonempty(lcut / f"legacy-{plate}-{ifu}-{b}.fits") for b in _GRIZ)
    has_legacy_i = _nonempty(lcut / f"legacy-{plate}-{ifu}-i.fits")
    has_legacy_metadata = _nonempty(lcut / "metadata.json")

    return GalaxyFlags(
        plateifu=plateifu,
        has_drp_logcube=has_drp_logcube,
        has_dap_maps=has_dap_maps,
        has_dap_model_logcube=has_dap_model_logcube,
        has_pipe3d=has_pipe3d,
        has_any_ifu=has_any_ifu,
        has_cutout_jpeg=has_cutout_jpeg,
        has_cutout_ugriz_all=has_cutout_ugriz_all,
        has_cutout_metadata=has_cutout_metadata,
        has_cutouts_full=has_cutouts_full,
        has_legacy_jpeg=has_legacy_jpeg,
        has_legacy_grz_all=has_legacy_grz_all,
        has_legacy_griz_all=has_legacy_griz_all,
        has_legacy_i=has_legacy_i,
        has_legacy_metadata=has_legacy_metadata,
    )


def summarize(flags: list[GalaxyFlags]) -> dict[str, object]:
    total = len(flags)

    complete_all_ifu_and_cutouts = [
        f
        for f in flags
        if f.has_drp_logcube
        and f.has_dap_maps
        and f.has_dap_model_logcube
        and f.has_pipe3d
        and f.has_cutouts_full
    ]
    logcube_and_cutouts_only = [
        f
        for f in flags
        if f.has_drp_logcube
        and f.has_cutouts_full
        and (not f.has_dap_maps)
        and (not f.has_dap_model_logcube)
        and (not f.has_pipe3d)
    ]
    missing_all_ifu = [f for f in flags if not f.has_any_ifu]
    logcube_sdss_ugriz_legacy_griz = [
        f
        for f in flags
        if f.has_drp_logcube and f.has_cutout_ugriz_all and f.has_legacy_griz_all
    ]
    logcube_sdss_ugriz_legacy_grz_missing_i = [
        f
        for f in flags
        if f.has_drp_logcube
        and f.has_cutout_ugriz_all
        and f.has_legacy_grz_all
        and (not f.has_legacy_i)
    ]

    return {
        "total_galaxy_folders": total,
        "complete_all_ifu_and_cutouts": len(complete_all_ifu_and_cutouts),
        "logcube_and_cutouts_only": len(logcube_and_cutouts_only),
        "missing_all_ifu": len(missing_all_ifu),
        "has_drp_logcube": sum(1 for f in flags if f.has_drp_logcube),
        "has_full_cutouts": sum(1 for f in flags if f.has_cutouts_full),
        "has_legacy_griz_all": sum(1 for f in flags if f.has_legacy_griz_all),
        "has_legacy_grz_all": sum(1 for f in flags if f.has_legacy_grz_all),
        "logcube_sdss_ugriz_legacy_griz": len(logcube_sdss_ugriz_legacy_griz),
        "logcube_sdss_ugriz_legacy_grz_missing_i": len(logcube_sdss_ugriz_legacy_grz_missing_i),
        "example_plateifu": {
            "complete_all_ifu_and_cutouts": [f.plateifu for f in complete_all_ifu_and_cutouts[:10]],
            "logcube_and_cutouts_only": [f.plateifu for f in logcube_and_cutouts_only[:10]],
            "missing_all_ifu": [f.plateifu for f in missing_all_ifu[:10]],
            "logcube_sdss_ugriz_legacy_griz": [f.plateifu for f in logcube_sdss_ugriz_legacy_griz[:10]],
            "logcube_sdss_ugriz_legacy_grz_missing_i": [
                f.plateifu for f in logcube_sdss_ugriz_legacy_grz_missing_i[:10]
            ],
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Inventory completeness of local MaNGA galaxy folders.")
    p.add_argument("--data-root", type=Path, default=Path("manga_sdss_fits"))
    p.add_argument("--json-out", type=Path, default=None, help="Optional path to write JSON report")
    p.add_argument(
        "--details-out",
        type=Path,
        default=None,
        help="Optional path to write per-galaxy flags JSONL",
    )
    args = p.parse_args(argv)

    if not args.data_root.is_dir():
        raise SystemExit(f"Missing data root: {args.data_root}")

    galaxy_dirs = sorted(p for p in args.data_root.iterdir() if p.is_dir() and _DIR_RE.match(p.name))
    flags = [inspect_galaxy(g) for g in galaxy_dirs]
    rep = summarize(flags)

    print(f"Data root: {args.data_root}")
    print(f"Total galaxy folders: {rep['total_galaxy_folders']}")
    print(f"Complete (all IFU + cutouts): {rep['complete_all_ifu_and_cutouts']}")
    print(f"LOGCUBE + cutouts only: {rep['logcube_and_cutouts_only']}")
    print(f"Missing all IFU: {rep['missing_all_ifu']}")
    print(f"Has DRP LOGCUBE: {rep['has_drp_logcube']}")
    print(f"Has full cutouts: {rep['has_full_cutouts']}")
    print(f"Has Legacy griz all bands: {rep['has_legacy_griz_all']}")
    print(f"Has Legacy grz bands: {rep['has_legacy_grz_all']}")
    print(
        "LOGCUBE + SDSS ugriz + Legacy griz: "
        f"{rep['logcube_sdss_ugriz_legacy_griz']}"
    )
    print(
        "LOGCUBE + SDSS ugriz + Legacy grz (missing i): "
        f"{rep['logcube_sdss_ugriz_legacy_grz_missing_i']}"
    )

    if args.json_out:
        args.json_out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"Saved report: {args.json_out}")

    if args.details_out:
        with args.details_out.open("w", encoding="utf-8") as f:
            for row in flags:
                f.write(json.dumps(asdict(row)) + "\n")
        print(f"Saved details: {args.details_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
