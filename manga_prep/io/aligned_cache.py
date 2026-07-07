"""Pre-exported imaging aligned to the Amara / Pipe3D grid (fast training I/O)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from manga_prep.io.imaging_alignment import (
    _pipe3d_cube_path,
    native_shape_from_pipe3d,
    reproject_cutout_stack_to_amara_grid,
)

ALIGNED_SUBDIR = "aligned_imaging"
SDSS_ALIGNED_NAME = "sdss_aligned.npz"
LEGACY_ALIGNED_NAME = "legacy_aligned.npz"

_SDSS_BANDS = ("u", "g", "r", "i", "z")
_LEGACY_BANDS = ("g", "r", "i", "z")


def aligned_sdss_path(galaxy_dir: Path | str) -> Path:
    return Path(galaxy_dir) / ALIGNED_SUBDIR / SDSS_ALIGNED_NAME


def aligned_legacy_path(galaxy_dir: Path | str) -> Path:
    return Path(galaxy_dir) / ALIGNED_SUBDIR / LEGACY_ALIGNED_NAME


def aligned_sdss_path_from_row(data_root: Path, row: dict) -> Path:
    return aligned_sdss_path(data_root / row["galaxy_dir"])


def aligned_legacy_path_from_row(data_root: Path, row: dict) -> Path:
    return aligned_legacy_path(data_root / row["galaxy_dir"])


def _target_shape_from_amara(galaxy_dir: Path) -> tuple[int, int]:
    amara_path = Path(galaxy_dir) / "amara_maps.npz"
    if amara_path.is_file():
        with np.load(amara_path) as archive:
            return tuple(int(x) for x in archive["target_shape"])
    from manga_prep.targets.pipe3d_maps import DEFAULT_TARGET_SIZE

    return (DEFAULT_TARGET_SIZE, DEFAULT_TARGET_SIZE)


def _write_aligned_npz(
    out_path: Path,
    *,
    data: np.ndarray,
    bands: tuple[str, ...],
    target_shape: tuple[int, int],
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        data=np.asarray(data, dtype=np.float32),
        bands=np.asarray(bands, dtype="U1"),
        target_shape=np.asarray(target_shape, dtype=np.int32),
        aligned_to_amara_grid=np.array(True),
    )
    return out_path


def load_aligned_imaging(npz_path: Path | str) -> dict[str, object]:
    with np.load(npz_path) as archive:
        bands_raw = archive["bands"]
        if bands_raw.ndim == 0:
            bands = (str(bands_raw.item()),)
        else:
            bands = tuple(str(b) for b in bands_raw.tolist())
        return {
            "bands": bands,
            "data": np.asarray(archive["data"], dtype=np.float32),
            "aligned_to_amara_grid": True,
        }


def export_sdss_aligned(galaxy_dir: Path | str, *, skip_existing: bool = False) -> Path | None:
    galaxy_dir = Path(galaxy_dir)
    out_path = aligned_sdss_path(galaxy_dir)
    if skip_existing and out_path.is_file() and out_path.stat().st_size > 0:
        return out_path

    plateifu = galaxy_dir.name.replace("_", "-")
    plate, ifu = plateifu.split("-", 1)
    paths = [
        galaxy_dir / "sdss_cutouts" / f"sdss-{plate}-{ifu}-{band}.fits"
        for band in _SDSS_BANDS
    ]
    if not all(path.is_file() for path in paths):
        return None

    pipe3d_path = _pipe3d_cube_path(galaxy_dir)
    target_shape = _target_shape_from_amara(galaxy_dir)
    native_shape = native_shape_from_pipe3d(pipe3d_path)
    stack = reproject_cutout_stack_to_amara_grid(
        paths,
        pipe3d_path,
        target_shape=target_shape,
        native_shape=native_shape,
    )
    return _write_aligned_npz(
        out_path,
        data=stack,
        bands=_SDSS_BANDS,
        target_shape=target_shape,
    )


def count_aligned_caches(data_root: Path | str, rows: list[dict]) -> dict[str, int]:
    data_root = Path(data_root)
    sdss_cached = sum(1 for row in rows if aligned_sdss_path_from_row(data_root, row).is_file())
    legacy_cached = sum(
        1 for row in rows if aligned_legacy_path_from_row(data_root, row).is_file()
    )
    sdss_eligible = sum(1 for row in rows if row.get("has_sdss_imaging"))
    legacy_eligible = sum(1 for row in rows if row.get("has_legacy_imaging"))
    return {
        "sdss_cached": sdss_cached,
        "sdss_eligible": sdss_eligible,
        "legacy_cached": legacy_cached,
        "legacy_eligible": legacy_eligible,
    }


def export_legacy_aligned(galaxy_dir: Path | str, *, skip_existing: bool = False) -> Path | None:
    galaxy_dir = Path(galaxy_dir)
    out_path = aligned_legacy_path(galaxy_dir)
    if skip_existing and out_path.is_file() and out_path.stat().st_size > 0:
        return out_path

    plateifu = galaxy_dir.name.replace("_", "-")
    plate, ifu = plateifu.split("-", 1)
    for band_set in (_LEGACY_BANDS, ("g", "r", "z")):
        paths = [
            galaxy_dir / "legacy_cutouts" / f"legacy-{plate}-{ifu}-{band}.fits"
            for band in band_set
        ]
        if not all(path.is_file() for path in paths):
            continue

        pipe3d_path = _pipe3d_cube_path(galaxy_dir)
        target_shape = _target_shape_from_amara(galaxy_dir)
        native_shape = native_shape_from_pipe3d(pipe3d_path)
        stack = reproject_cutout_stack_to_amara_grid(
            paths,
            pipe3d_path,
            target_shape=target_shape,
            native_shape=native_shape,
        )
        return _write_aligned_npz(
            out_path,
            data=stack,
            bands=band_set,
            target_shape=target_shape,
        )
    return None
