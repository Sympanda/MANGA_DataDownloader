"""
Align survey cutout FITS images to the Amara / Pipe3D spaxel grid via WCS.

SDSS frame cutouts are often in native camera orientation; Legacy cutouts use a
different pixel grid. Amara map targets live on the Pipe3D spaxel WCS (76×76
padded canvas). Reprojecting inputs onto that grid matches the comparison
workflow in sdss_legacy_fits_jpeg_comparison.ipynb.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.wcs import WCS
from reproject import reproject_interp

from manga_prep.targets.pipe3d_maps import DEFAULT_TARGET_SIZE
from manga_prep.io.fits_io import celestial_wcs_from_header, open_fits


def _pipe3d_cube_path(galaxy_dir: Path) -> Path:
    galaxy_dir = Path(galaxy_dir)
    matches = sorted(galaxy_dir.glob("manga-*.Pipe3D.cube.fits*"))
    if not matches:
        raise FileNotFoundError(f"No Pipe3D cube under {galaxy_dir}")
    return matches[0]


def native_shape_from_pipe3d(pipe3d_path: Path) -> tuple[int, int]:
    with open_fits(pipe3d_path, memmap=True) as hdul:
        data = hdul["SSP"].data
        return int(data.shape[1]), int(data.shape[2])


def amara_grid_wcs(
    pipe3d_path: Path,
    *,
    target_shape: tuple[int, int] = (DEFAULT_TARGET_SIZE, DEFAULT_TARGET_SIZE),
    native_shape: tuple[int, int] | None = None,
) -> WCS:
    """
    Celestial WCS for the padded Amara map canvas.

    CRPIX is shifted to account for center-padding from native Pipe3D shape to
    target_shape, so reprojected pixels align with amara_maps.npz arrays.
    """
    pipe3d_path = Path(pipe3d_path)
    if native_shape is None:
        native_shape = native_shape_from_pipe3d(pipe3d_path)

    native_y, native_x = map(int, native_shape)
    target_y, target_x = map(int, target_shape)

    with open_fits(pipe3d_path, memmap=True) as hdul:
        hdr = hdul[0].header.copy()

    y0 = (target_y - native_y) // 2
    x0 = (target_x - native_x) // 2
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = target_x
    hdr["NAXIS2"] = target_y
    hdr["CRPIX1"] = float(hdr["CRPIX1"]) + x0
    hdr["CRPIX2"] = float(hdr["CRPIX2"]) + y0

    return celestial_wcs_from_header(hdr)


def load_cutout_wcs(fits_path: Path) -> WCS:
    with open_fits(fits_path, memmap=True) as hdul:
        return celestial_wcs_from_header(hdul[0].header)


def reproject_cutout_to_amara_grid(
    fits_path: Path,
    pipe3d_path: Path,
    *,
    target_shape: tuple[int, int] = (DEFAULT_TARGET_SIZE, DEFAULT_TARGET_SIZE),
    native_shape: tuple[int, int] | None = None,
    order: str = "bilinear",
) -> np.ndarray:
    """Reproject a single-band cutout FITS onto the Amara/Pipe3D spaxel grid."""
    fits_path = Path(fits_path)
    pipe3d_path = Path(pipe3d_path)

    with open_fits(fits_path, memmap=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)

    if data.ndim != 2:
        raise ValueError(f"Expected 2D cutout in {fits_path}, got shape {data.shape}")

    input_wcs = load_cutout_wcs(fits_path)
    output_wcs = amara_grid_wcs(
        pipe3d_path,
        target_shape=target_shape,
        native_shape=native_shape,
    )
    reprojected, _ = reproject_interp(
        (data, input_wcs),
        output_wcs,
        shape_out=target_shape,
        order=order,
    )
    return np.asarray(reprojected, dtype=np.float32)


def reproject_cutout_stack_to_amara_grid(
    fits_paths: list[Path],
    pipe3d_path: Path,
    *,
    target_shape: tuple[int, int] = (DEFAULT_TARGET_SIZE, DEFAULT_TARGET_SIZE),
    native_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Reproject multiple band FITS cutouts; returns (n_band, H, W)."""
    planes = [
        reproject_cutout_to_amara_grid(
            path,
            pipe3d_path,
            target_shape=target_shape,
            native_shape=native_shape,
        )
        for path in fits_paths
    ]
    return np.stack(planes, axis=0)
