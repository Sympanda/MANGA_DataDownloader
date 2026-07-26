"""Pre-exported imaging aligned to the Amara / Pipe3D grid (fast training I/O)."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from manga_prep.io.imaging_alignment import (
    SDSS_NATIVE_CANVAS,
    _pipe3d_cube_path,
    amara_aligned_pixel_shape,
    native_shape_from_pipe3d,
    reproject_cutout_stack_to_amara_grid,
    reproject_cutout_stack_to_sdss_native_grid,
)

ALIGNED_SUBDIR = "aligned_imaging"
SDSS_ALIGNED_STEM = "sdss_aligned"
LEGACY_ALIGNED_STEM = "legacy_aligned"
SDSS_NATIVE_NAME = "sdss_aligned_native.npz"
LEGACY_NATIVE_NAME = "legacy_aligned_native.npz"

ImagingGrid = Literal["amara", "sdss_native"]

_SDSS_BANDS = ("u", "g", "r", "i", "z")
_LEGACY_BANDS = ("g", "r", "i", "z")


def aligned_cache_name(stem: str, *, oversample: int = 1) -> str:
    os_factor = int(oversample)
    if os_factor < 1:
        raise ValueError(f"oversample must be >= 1, got {oversample}")
    if os_factor == 1:
        return f"{stem}.npz"
    return f"{stem}_os{os_factor}.npz"


def aligned_sdss_path(
    galaxy_dir: Path | str,
    *,
    grid: ImagingGrid = "amara",
    oversample: int = 1,
) -> Path:
    galaxy_dir = Path(galaxy_dir)
    if grid == "sdss_native":
        return galaxy_dir / ALIGNED_SUBDIR / SDSS_NATIVE_NAME
    return galaxy_dir / ALIGNED_SUBDIR / aligned_cache_name(SDSS_ALIGNED_STEM, oversample=oversample)


def aligned_legacy_path(
    galaxy_dir: Path | str,
    *,
    grid: ImagingGrid = "amara",
    oversample: int = 1,
) -> Path:
    galaxy_dir = Path(galaxy_dir)
    if grid == "sdss_native":
        return galaxy_dir / ALIGNED_SUBDIR / LEGACY_NATIVE_NAME
    return galaxy_dir / ALIGNED_SUBDIR / aligned_cache_name(LEGACY_ALIGNED_STEM, oversample=oversample)


def aligned_sdss_path_from_row(
    data_root: Path,
    row: dict,
    *,
    grid: ImagingGrid = "amara",
    oversample: int = 1,
) -> Path:
    return aligned_sdss_path(data_root / row["galaxy_dir"], grid=grid, oversample=oversample)


def aligned_legacy_path_from_row(
    data_root: Path,
    row: dict,
    *,
    grid: ImagingGrid = "amara",
    oversample: int = 1,
) -> Path:
    return aligned_legacy_path(data_root / row["galaxy_dir"], grid=grid, oversample=oversample)


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
    oversample: int = 1,
    grid: ImagingGrid = "amara",
    pixel_scale_arcsec: float | None = None,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # numpy savez appends .npz if missing — keep a real .npz suffix on the temp file.
    tmp_path = out_path.with_name(out_path.stem + ".tmp.npz")
    payload: dict[str, object] = {
        "data": np.asarray(data, dtype=np.float32),
        "bands": np.asarray(bands, dtype="U1"),
        "target_shape": np.asarray(target_shape, dtype=np.int32),
        "aligned_to_amara_grid": np.array(True),
        "aligned_oversample": np.array(int(oversample), dtype=np.int32),
        "grid": np.array(grid),
    }
    if pixel_scale_arcsec is not None:
        payload["pixel_scale_arcsec"] = np.array(float(pixel_scale_arcsec), dtype=np.float64)
    np.savez_compressed(tmp_path, **payload)
    tmp_path.replace(out_path)
    return out_path


def load_aligned_imaging(npz_path: Path | str) -> dict[str, object]:
    with np.load(npz_path) as archive:
        bands_raw = archive["bands"]
        if bands_raw.ndim == 0:
            bands = (str(bands_raw.item()),)
        else:
            bands = tuple(str(b) for b in bands_raw.tolist())
        oversample = 1
        if "aligned_oversample" in archive.files:
            oversample = int(np.asarray(archive["aligned_oversample"]).item())
        grid = "amara"
        if "grid" in archive.files:
            grid = str(np.asarray(archive["grid"]).item())
        out: dict[str, object] = {
            "bands": bands,
            "data": np.asarray(archive["data"], dtype=np.float32),
            "aligned_to_amara_grid": True,
            "aligned_oversample": oversample,
            "grid": grid,
        }
        if "pixel_scale_arcsec" in archive.files:
            out["pixel_scale_arcsec"] = float(np.asarray(archive["pixel_scale_arcsec"]).item())
        return out


def export_sdss_aligned(
    galaxy_dir: Path | str,
    *,
    skip_existing: bool = False,
    oversample: int = 1,
    grid: ImagingGrid = "amara",
    canvas: int = SDSS_NATIVE_CANVAS,
) -> Path | None:
    galaxy_dir = Path(galaxy_dir)
    oversample = int(oversample)
    out_path = aligned_sdss_path(galaxy_dir, grid=grid, oversample=oversample)
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

    if grid == "sdss_native":
        shape_out = (int(canvas), int(canvas))
        stack, scale = reproject_cutout_stack_to_sdss_native_grid(
            paths,
            pipe3d_path,
            shape_out=shape_out,
            target_shape=target_shape,
            native_shape=native_shape,
        )
        if stack.shape[-2:] != shape_out:
            raise RuntimeError(f"Native stack shape {stack.shape[-2:]} != expected {shape_out}")
        return _write_aligned_npz(
            out_path,
            data=stack,
            bands=_SDSS_BANDS,
            target_shape=target_shape,
            oversample=1,
            grid="sdss_native",
            pixel_scale_arcsec=scale,
        )

    stack = reproject_cutout_stack_to_amara_grid(
        paths,
        pipe3d_path,
        target_shape=target_shape,
        native_shape=native_shape,
        oversample=oversample,
    )
    expected = amara_aligned_pixel_shape(target_shape, oversample=oversample)
    if stack.shape[-2:] != expected:
        raise RuntimeError(f"Aligned stack shape {stack.shape[-2:]} != expected {expected}")
    return _write_aligned_npz(
        out_path,
        data=stack,
        bands=_SDSS_BANDS,
        target_shape=target_shape,
        oversample=oversample,
        grid="amara",
    )


def count_aligned_caches(
    data_root: Path | str,
    rows: list[dict],
    *,
    oversample: int = 1,
    grid: ImagingGrid = "amara",
) -> dict[str, int | str]:
    data_root = Path(data_root)
    os_factor = int(oversample)
    sdss_cached = sum(
        1
        for row in rows
        if aligned_sdss_path_from_row(data_root, row, grid=grid, oversample=os_factor).is_file()
    )
    legacy_cached = sum(
        1
        for row in rows
        if aligned_legacy_path_from_row(data_root, row, grid=grid, oversample=os_factor).is_file()
    )
    sdss_eligible = sum(1 for row in rows if row.get("has_sdss_imaging"))
    legacy_eligible = sum(1 for row in rows if row.get("has_legacy_imaging"))
    return {
        "sdss_cached": sdss_cached,
        "sdss_eligible": sdss_eligible,
        "legacy_cached": legacy_cached,
        "legacy_eligible": legacy_eligible,
        "oversample": os_factor,
        "grid": grid,
    }


def export_legacy_aligned(
    galaxy_dir: Path | str,
    *,
    skip_existing: bool = False,
    oversample: int = 1,
    grid: ImagingGrid = "amara",
    canvas: int = SDSS_NATIVE_CANVAS,
) -> Path | None:
    galaxy_dir = Path(galaxy_dir)
    oversample = int(oversample)
    out_path = aligned_legacy_path(galaxy_dir, grid=grid, oversample=oversample)
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
        if grid == "sdss_native":
            shape_out = (int(canvas), int(canvas))
            stack, scale = reproject_cutout_stack_to_sdss_native_grid(
                paths,
                pipe3d_path,
                shape_out=shape_out,
                target_shape=target_shape,
                native_shape=native_shape,
            )
            return _write_aligned_npz(
                out_path,
                data=stack,
                bands=band_set,
                target_shape=target_shape,
                oversample=1,
                grid="sdss_native",
                pixel_scale_arcsec=scale,
            )

        stack = reproject_cutout_stack_to_amara_grid(
            paths,
            pipe3d_path,
            target_shape=target_shape,
            native_shape=native_shape,
            oversample=oversample,
        )
        return _write_aligned_npz(
            out_path,
            data=stack,
            bands=band_set,
            target_shape=target_shape,
            oversample=oversample,
            grid="amara",
        )
    return None
