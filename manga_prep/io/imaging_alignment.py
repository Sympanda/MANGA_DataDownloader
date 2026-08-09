"""
Align survey cutout FITS images to the Amara / Pipe3D spaxel grid via WCS.

SDSS frame cutouts are often in native camera orientation; Legacy cutouts use a
different pixel grid. Amara map targets live on the Pipe3D spaxel WCS (76×76
padded canvas). Reprojecting inputs onto that grid matches the comparison
workflow in sdss_legacy_fits_jpeg_comparison.ipynb.

Two grids are supported for imaging:
- ``amara``: same FoV / orientation as Amara maps (optional integer oversample)
- ``sdss_native``: Amara center + orientation, but SDSS cutout plate scale on a
  fixed ~196×196 canvas (HR pipelines)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.wcs import WCS

from manga_prep.targets.pipe3d_maps import DEFAULT_TARGET_SIZE
from manga_prep.io.fits_io import celestial_wcs_from_header, open_fits

# Fixed canvas for SDSS-native aligned HR imaging (matches typical cutout size).
SDSS_NATIVE_CANVAS = 196


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


def pixel_scale_deg(wcs: WCS) -> float:
    """Return approximate |pixel scale| in degrees/pixel (axis 1)."""
    if wcs.wcs.has_cd():
        cd = np.asarray(wcs.wcs.cd, dtype=np.float64)
        return float(np.sqrt(cd[0, 0] ** 2 + cd[1, 0] ** 2))
    cdelt = np.asarray(wcs.wcs.cdelt, dtype=np.float64)
    return float(abs(cdelt[0]))


def pixel_scale_arcsec(wcs: WCS) -> float:
    return pixel_scale_deg(wcs) * 3600.0


def amara_grid_wcs(
    pipe3d_path: Path,
    *,
    target_shape: tuple[int, int] = (DEFAULT_TARGET_SIZE, DEFAULT_TARGET_SIZE),
    native_shape: tuple[int, int] | None = None,
    oversample: int = 1,
) -> WCS:
    """
    Celestial WCS for the padded Amara map canvas.

    CRPIX is shifted to account for center-padding from native Pipe3D shape to
    ``target_shape``, so reprojected pixels align with ``amara_maps.npz`` arrays.

    ``oversample`` > 1 keeps the same sky FoV as ``target_shape`` but uses a
    finer pixel grid (e.g. oversample=2 → 152×152 covering the same area as 76×76).
    """
    if int(oversample) < 1:
        raise ValueError(f"oversample must be >= 1, got {oversample}")

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

    wcs = celestial_wcs_from_header(hdr)
    if int(oversample) == 1:
        return wcs

    # Refine pixel scale: same FoV, oversample× more pixels along each axis.
    s = float(oversample)
    wcs = wcs.deepcopy()
    if wcs.wcs.has_cd():
        wcs.wcs.cd = np.asarray(wcs.wcs.cd, dtype=np.float64) / s
    else:
        wcs.wcs.cdelt = np.asarray(wcs.wcs.cdelt, dtype=np.float64) / s
    # FITS 1-indexed CRPIX: map pixel centres under integer grid refinement.
    crpix = np.asarray(wcs.wcs.crpix, dtype=np.float64)
    wcs.wcs.crpix = (crpix - 0.5) * s + 0.5
    return wcs


def amara_aligned_pixel_shape(
    target_shape: tuple[int, int] = (DEFAULT_TARGET_SIZE, DEFAULT_TARGET_SIZE),
    *,
    oversample: int = 1,
) -> tuple[int, int]:
    """Pixel (H, W) for an Amara-aligned canvas at the given oversample factor."""
    if int(oversample) < 1:
        raise ValueError(f"oversample must be >= 1, got {oversample}")
    return (int(target_shape[0]) * int(oversample), int(target_shape[1]) * int(oversample))


def sdss_native_aligned_wcs(
    pipe3d_path: Path,
    reference_cutout_wcs: WCS,
    *,
    shape_out: tuple[int, int] = (SDSS_NATIVE_CANVAS, SDSS_NATIVE_CANVAS),
    target_shape: tuple[int, int] = (DEFAULT_TARGET_SIZE, DEFAULT_TARGET_SIZE),
    native_shape: tuple[int, int] | None = None,
) -> WCS:
    """
    Amara-centered / Amara-oriented WCS at the SDSS cutout plate scale.

    Keeps survey pixel sampling for HR imaging while matching map sky axes.
    """
    shape_y, shape_x = map(int, shape_out)
    amara = amara_grid_wcs(
        pipe3d_path,
        target_shape=target_shape,
        native_shape=native_shape,
        oversample=1,
    )
    amara_scale = pixel_scale_deg(amara)
    cutout_scale = pixel_scale_deg(reference_cutout_wcs)
    if amara_scale <= 0 or cutout_scale <= 0:
        raise ValueError(
            f"Invalid pixel scales: amara={amara_scale}, cutout={cutout_scale}"
        )
    # Scale factor: Amara deg/pix / SDSS deg/pix → shrink CD so pixels are finer.
    s = amara_scale / cutout_scale

    wcs = amara.deepcopy()
    # Place CRPIX at the canvas centre (FITS 1-indexed).
    # Amara CRVAL (sky centre) maps to the centre of the native canvas.
    wcs.wcs.crpix = np.array([0.5 * (shape_x + 1), 0.5 * (shape_y + 1)], dtype=np.float64)
    if wcs.wcs.has_cd():
        wcs.wcs.cd = np.asarray(wcs.wcs.cd, dtype=np.float64) / s
    else:
        wcs.wcs.cdelt = np.asarray(wcs.wcs.cdelt, dtype=np.float64) / s
    return wcs


def load_cutout_wcs(fits_path: Path) -> WCS:
    with open_fits(fits_path, memmap=True) as hdul:
        return celestial_wcs_from_header(hdul[0].header)


def reproject_cutout_to_amara_grid(
    fits_path: Path,
    pipe3d_path: Path,
    *,
    target_shape: tuple[int, int] = (DEFAULT_TARGET_SIZE, DEFAULT_TARGET_SIZE),
    native_shape: tuple[int, int] | None = None,
    oversample: int = 1,
    order: str = "bilinear",
) -> np.ndarray:
    """
    Reproject a single-band cutout FITS onto the Amara/Pipe3D celestial grid.

    With ``oversample=1`` the output matches the Amara map canvas (typically 76×76).
    With ``oversample>1`` the output covers the same FoV on a finer pixel grid.
    """
    fits_path = Path(fits_path)
    pipe3d_path = Path(pipe3d_path)
    shape_out = amara_aligned_pixel_shape(target_shape, oversample=oversample)

    with open_fits(fits_path, memmap=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)

    if data.ndim != 2:
        raise ValueError(f"Expected 2D cutout in {fits_path}, got shape {data.shape}")

    input_wcs = load_cutout_wcs(fits_path)
    output_wcs = amara_grid_wcs(
        pipe3d_path,
        target_shape=target_shape,
        native_shape=native_shape,
        oversample=oversample,
    )
    from reproject import reproject_interp

    reprojected, _ = reproject_interp(
        (data, input_wcs),
        output_wcs,
        shape_out=shape_out,
        order=order,
    )
    return np.asarray(reprojected, dtype=np.float32)


def reproject_cutout_to_sdss_native_grid(
    fits_path: Path,
    pipe3d_path: Path,
    *,
    reference_cutout_wcs: WCS | None = None,
    shape_out: tuple[int, int] = (SDSS_NATIVE_CANVAS, SDSS_NATIVE_CANVAS),
    target_shape: tuple[int, int] = (DEFAULT_TARGET_SIZE, DEFAULT_TARGET_SIZE),
    native_shape: tuple[int, int] | None = None,
    order: str = "bilinear",
) -> np.ndarray:
    """
    Reproject a cutout onto an Amara-oriented canvas at SDSS plate scale.

    Output is typically 196×196 covering a larger FoV than the 76×76 Amara maps.
    """
    fits_path = Path(fits_path)
    pipe3d_path = Path(pipe3d_path)
    shape_out = (int(shape_out[0]), int(shape_out[1]))

    with open_fits(fits_path, memmap=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)

    if data.ndim != 2:
        raise ValueError(f"Expected 2D cutout in {fits_path}, got shape {data.shape}")

    input_wcs = load_cutout_wcs(fits_path)
    ref_wcs = reference_cutout_wcs if reference_cutout_wcs is not None else input_wcs
    output_wcs = sdss_native_aligned_wcs(
        pipe3d_path,
        ref_wcs,
        shape_out=shape_out,
        target_shape=target_shape,
        native_shape=native_shape,
    )
    from reproject import reproject_interp

    reprojected, _ = reproject_interp(
        (data, input_wcs),
        output_wcs,
        shape_out=shape_out,
        order=order,
    )
    return np.asarray(reprojected, dtype=np.float32)


def reproject_cutout_stack_to_amara_grid(
    fits_paths: list[Path],
    pipe3d_path: Path,
    *,
    target_shape: tuple[int, int] = (DEFAULT_TARGET_SIZE, DEFAULT_TARGET_SIZE),
    native_shape: tuple[int, int] | None = None,
    oversample: int = 1,
) -> np.ndarray:
    """Reproject multiple band FITS cutouts; returns (n_band, H, W)."""
    planes = [
        reproject_cutout_to_amara_grid(
            path,
            pipe3d_path,
            target_shape=target_shape,
            native_shape=native_shape,
            oversample=oversample,
        )
        for path in fits_paths
    ]
    return np.stack(planes, axis=0)


def reproject_cutout_stack_to_sdss_native_grid(
    fits_paths: list[Path],
    pipe3d_path: Path,
    *,
    shape_out: tuple[int, int] = (SDSS_NATIVE_CANVAS, SDSS_NATIVE_CANVAS),
    target_shape: tuple[int, int] = (DEFAULT_TARGET_SIZE, DEFAULT_TARGET_SIZE),
    native_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, float]:
    """
    Reproject band stack to SDSS-native aligned grid.

    Returns ``(n_band, H, W)`` and the reference plate scale in arcsec/pixel.
    Uses the first cutout's WCS as the shared plate-scale reference so all bands
    share one output grid.
    """
    if not fits_paths:
        raise ValueError("fits_paths must be non-empty")
    ref_wcs = load_cutout_wcs(fits_paths[0])
    scale = pixel_scale_arcsec(ref_wcs)
    planes = [
        reproject_cutout_to_sdss_native_grid(
            path,
            pipe3d_path,
            reference_cutout_wcs=ref_wcs,
            shape_out=shape_out,
            target_shape=target_shape,
            native_shape=native_shape,
        )
        for path in fits_paths
    ]
    return np.stack(planes, axis=0), float(scale)
