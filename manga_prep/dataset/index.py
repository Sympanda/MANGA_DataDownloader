"""
Build a unified index of per-galaxy training assets under manga_sdss_fits/.

The index records which modalities exist for each galaxy and paths to load them
without scanning the filesystem on every dataset init.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from astropy.io import fits

from manga_prep.io.aligned_cache import aligned_legacy_path_from_row, aligned_sdss_path_from_row

_DIR_RE = re.compile(r"^(\d+)_(\d+)$")
_SDSS_BANDS = ("u", "g", "r", "i", "z")
_LEGACY_BANDS = ("g", "r", "i", "z")

_INDEX_FIELDS = (
    "plateifu",
    "galaxy_dir",
    "ra_deg",
    "dec_deg",
    "has_sdss_imaging",
    "has_legacy_imaging",
    "sdss_imaging_valid",
    "legacy_imaging_valid",
    "has_amara_maps",
    "has_fake_spectrum",
    "has_real_spectrum",
    "amara_maps_npz",
    "fake_spectrum_npz",
    "real_spectrum_npz",
    "sdss_cutouts_dir",
    "legacy_cutouts_dir",
)


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _fits_shape(path: Path) -> tuple[int, int] | None:
    if not _nonempty(path):
        return None
    try:
        header = fits.getheader(path, 0)
    except OSError:
        return None
    if "NAXIS1" not in header or "NAXIS2" not in header:
        return None
    return int(header["NAXIS2"]), int(header["NAXIS1"])


def _consistent_band_shapes(paths: list[Path]) -> bool:
    shapes = [_fits_shape(path) for path in paths]
    if any(shape is None for shape in shapes):
        return False
    return len(set(shapes)) == 1


def sdss_imaging_ready(data_root: Path, row: dict) -> bool:
    cache_path = aligned_sdss_path_from_row(data_root, row)
    if _nonempty(cache_path):
        return True
    if "sdss_imaging_valid" in row and row["sdss_imaging_valid"] is not None:
        return bool(row["sdss_imaging_valid"])
    gal_dir = data_root / row["galaxy_dir"]
    plate, ifu = row["plateifu"].split("-", 1)
    paths = [gal_dir / "sdss_cutouts" / f"sdss-{plate}-{ifu}-{b}.fits" for b in _SDSS_BANDS]
    return _consistent_band_shapes(paths)


def legacy_imaging_ready(data_root: Path, row: dict) -> bool:
    cache_path = aligned_legacy_path_from_row(data_root, row)
    if _nonempty(cache_path):
        return True
    if "legacy_imaging_valid" in row and row["legacy_imaging_valid"] is not None:
        return bool(row["legacy_imaging_valid"])
    gal_dir = data_root / row["galaxy_dir"]
    plate, ifu = row["plateifu"].split("-", 1)
    grz = [gal_dir / "legacy_cutouts" / f"legacy-{plate}-{ifu}-{b}.fits" for b in ("g", "r", "z")]
    griz = [gal_dir / "legacy_cutouts" / f"legacy-{plate}-{ifu}-{b}.fits" for b in _LEGACY_BANDS]
    return _consistent_band_shapes(grz) or _consistent_band_shapes(griz)


def _read_json(path: Path) -> dict | None:
    if not _nonempty(path):
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _ra_dec_from_metadata(meta: dict | None) -> tuple[float | None, float | None]:
    if not meta:
        return None, None
    ra = meta.get("ra_deg", meta.get("obj_ra_deg", meta.get("ifu_ra_deg")))
    dec = meta.get("dec_deg", meta.get("obj_dec_deg", meta.get("ifu_dec_deg")))
    if ra is None or dec is None:
        return None, None
    return float(ra), float(dec)


def inspect_galaxy_index_row(gal_dir: Path, *, data_root: Path | None = None) -> dict:
    """Return one index row for a manga_sdss_fits/<plate>_<ifu>/ folder."""
    m = _DIR_RE.match(gal_dir.name)
    if not m:
        raise ValueError(f"unexpected galaxy folder name: {gal_dir.name}")

    plate, ifu = m.group(1), m.group(2)
    plateifu = f"{plate}-{ifu}"

    sdss_dir = gal_dir / "sdss_cutouts"
    legacy_dir = gal_dir / "legacy_cutouts"
    sdss_paths = [sdss_dir / f"sdss-{plate}-{ifu}-{b}.fits" for b in _SDSS_BANDS]
    has_sdss_imaging = all(_nonempty(path) for path in sdss_paths)
    sdss_imaging_valid = has_sdss_imaging and _consistent_band_shapes(sdss_paths)

    legacy_grz_paths = [legacy_dir / f"legacy-{plate}-{ifu}-{b}.fits" for b in ("g", "r", "z")]
    legacy_griz_paths = [legacy_dir / f"legacy-{plate}-{ifu}-{b}.fits" for b in _LEGACY_BANDS]
    has_legacy_imaging = all(_nonempty(path) for path in legacy_grz_paths) or all(
        _nonempty(path) for path in legacy_griz_paths
    )
    legacy_imaging_valid = _consistent_band_shapes(legacy_grz_paths) or _consistent_band_shapes(
        legacy_griz_paths
    )

    amara_maps_npz = gal_dir / "amara_maps.npz"
    has_amara_maps = _nonempty(amara_maps_npz)

    fake_matches = sorted((gal_dir / "fake_sdss_spectra").glob("manga-*-fake-sdss-spectrum-*.npz"))
    fake_spectrum_npz = fake_matches[0] if fake_matches and _nonempty(fake_matches[0]) else None
    has_fake_spectrum = fake_spectrum_npz is not None

    real_matches = sorted((gal_dir / "sdss_spectra").glob("sdss-*-spectrum.npz"))
    real_spectrum_npz = real_matches[0] if real_matches and _nonempty(real_matches[0]) else None
    has_real_spectrum = real_spectrum_npz is not None

    ra_deg = dec_deg = None
    for meta_path in (
        sdss_dir / "metadata.json",
        gal_dir / "fake_sdss_spectra" / "metadata.json",
        gal_dir / "amara_maps_metadata.json",
        legacy_dir / "metadata.json",
    ):
        ra_deg, dec_deg = _ra_dec_from_metadata(_read_json(meta_path))
        if ra_deg is not None and dec_deg is not None:
            break

    if data_root is not None:
        try:
            galaxy_dir_str = str(gal_dir.relative_to(data_root))
        except ValueError:
            galaxy_dir_str = str(gal_dir)
    else:
        galaxy_dir_str = str(gal_dir)

    def _rel(path: Path | None) -> str:
        if path is None:
            return ""
        if data_root is not None:
            try:
                return str(path.relative_to(data_root))
            except ValueError:
                pass
        return str(path)

    return {
        "plateifu": plateifu,
        "galaxy_dir": galaxy_dir_str.replace("\\", "/"),
        "ra_deg": "" if ra_deg is None else f"{ra_deg:.8f}",
        "dec_deg": "" if dec_deg is None else f"{dec_deg:.8f}",
        "has_sdss_imaging": has_sdss_imaging,
        "has_legacy_imaging": has_legacy_imaging,
        "sdss_imaging_valid": sdss_imaging_valid,
        "legacy_imaging_valid": legacy_imaging_valid,
        "has_amara_maps": has_amara_maps,
        "has_fake_spectrum": has_fake_spectrum,
        "has_real_spectrum": has_real_spectrum,
        "amara_maps_npz": _rel(amara_maps_npz if has_amara_maps else None),
        "fake_spectrum_npz": _rel(fake_spectrum_npz),
        "real_spectrum_npz": _rel(real_spectrum_npz),
        "sdss_cutouts_dir": _rel(sdss_dir if has_sdss_imaging else None),
        "legacy_cutouts_dir": _rel(legacy_dir if has_legacy_imaging else None),
    }


def build_manga_dataset_index(data_root: Path, *, galaxy_dirs: list[Path] | None = None) -> list[dict]:
    """Scan data_root and return index rows sorted by plateifu."""
    data_root = Path(data_root)
    if galaxy_dirs is None:
        galaxy_dirs = sorted(
            p for p in data_root.iterdir() if p.is_dir() and _DIR_RE.match(p.name)
        )
    rows = [inspect_galaxy_index_row(gal_dir, data_root=data_root) for gal_dir in galaxy_dirs]
    rows.sort(key=lambda row: row["plateifu"])
    return rows


def write_manga_dataset_index(rows: list[dict], out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_INDEX_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in _INDEX_FIELDS})
    return out_path


def _parse_bool_field(raw: str | None) -> bool:
    return str(raw or "").lower() in {"1", "true", "yes"}


def _parse_optional_bool_field(raw: str | None) -> bool | None:
    if raw is None or str(raw).strip() == "":
        return None
    return _parse_bool_field(raw)


def read_manga_dataset_index(index_path: Path) -> list[dict]:
    index_path = Path(index_path)
    with index_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or ()

    required_flags = (
        "has_sdss_imaging",
        "has_legacy_imaging",
        "has_amara_maps",
        "has_fake_spectrum",
        "has_real_spectrum",
    )
    optional_flags = ("sdss_imaging_valid", "legacy_imaging_valid")

    for row in rows:
        for flag in required_flags:
            row[flag] = _parse_bool_field(row.get(flag))
        for flag in optional_flags:
            if flag in fieldnames:
                row[flag] = _parse_optional_bool_field(row.get(flag))
    return rows


def summarize_index(rows: list[dict]) -> dict[str, int]:
    total = len(rows)
    return {
        "total": total,
        "has_sdss_imaging": sum(1 for r in rows if r["has_sdss_imaging"]),
        "has_legacy_imaging": sum(1 for r in rows if r["has_legacy_imaging"]),
        "has_amara_maps": sum(1 for r in rows if r["has_amara_maps"]),
        "has_fake_spectrum": sum(1 for r in rows if r["has_fake_spectrum"]),
        "has_real_spectrum": sum(1 for r in rows if r["has_real_spectrum"]),
        "has_maps_and_fake_spectrum": sum(
            1 for r in rows if r["has_amara_maps"] and r["has_fake_spectrum"]
        ),
        "has_all_training_modalities": sum(
            1
            for r in rows
            if r["has_sdss_imaging"]
            and r["has_legacy_imaging"]
            and r["has_amara_maps"]
            and r["has_fake_spectrum"]
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build unified manga_sdss_fits dataset index CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=Path("manga_sdss_fits"))
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: <data-root>/manga_dataset_index.csv)",
    )
    args = parser.parse_args(argv)

    if not args.data_root.is_dir():
        raise SystemExit(f"Missing data root: {args.data_root}")

    rows = build_manga_dataset_index(args.data_root)
    out_path = args.out or (args.data_root / "manga_dataset_index.csv")
    write_manga_dataset_index(rows, out_path)

    summary = summarize_index(rows)
    print(f"Wrote {out_path} ({summary['total']} galaxies)")
    for key, value in summary.items():
        if key == "total":
            continue
        print(f"  {key}: {value} ({100 * value / summary['total']:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
