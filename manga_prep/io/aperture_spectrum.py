"""
Build SDSS-fiber-like aperture spectra from MaNGA LOGCUBE spaxels.

MaNGA DRP LOGCUBE spaxels are 0.5 arcsec on a WCS grid. SDSS legacy fibers
are 3 arcsec diameter (180 microns); BOSS/eBOSS fibers are 2 arcsec diameter.

This module coadds spaxel spectra within a circular aperture centered on the
IFU/galaxy pointing (LOGCUBE WCS reference pixel). Each spaxel contributes
in proportion to the area overlap between the spaxel square and the circular
aperture (partial-pixel weighting).

Units note:
- MaNGA LOGCUBE flux is 1e-17 erg/s/cm^2/Angstrom/spaxel.
- SDSS spec flux is 1e-17 erg/s/cm^2/Angstrom (integrated through the fiber).
- For spaxel flux integrated over the full spaxel area, a spaxel with overlap
  fraction f contributes f * flux_spaxel to the coadd.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

SDSS_LEGACY_FIBER_DIAMETER_ARCSEC = 3.0
BOSS_FIBER_DIAMETER_ARCSEC = 2.0
DEFAULT_APERTURE_DIAMETER_ARCSEC = SDSS_LEGACY_FIBER_DIAMETER_ARCSEC
DEFAULT_SUBPIXELS_PER_SPAXEL = 32


def logcube_path(gal_dir: Path, plate: str, ifu: str) -> Path | None:
    for name in (
        f"manga-{plate}-{ifu}-LOGCUBE.fits.gz",
        f"manga-{plate}-{ifu}-LOGCUBE.fits",
    ):
        path = gal_dir / name
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def spaxel_offsets_arcsec(ny: int, nx: int, header: fits.Header) -> tuple[np.ndarray, np.ndarray, float]:
    """Return spaxel-center offsets from the WCS reference pixel, in arcsec."""
    crpix1 = float(header["CRPIX1"])
    crpix2 = float(header["CRPIX2"])
    cd1_1 = float(header["CD1_1"]) * 3600.0
    cd2_2 = float(header["CD2_2"]) * 3600.0

    yy, xx = np.indices((ny, nx))
    offset_x = ((xx + 1) - crpix1) * cd1_1
    offset_y = ((yy + 1) - crpix2) * cd2_2
    spaxel_scale = 0.5 * (abs(cd1_1) + abs(cd2_2))
    return offset_x.astype(np.float32), offset_y.astype(np.float32), float(spaxel_scale)


def circular_aperture_overlap_weights(
    offset_x_arcsec: np.ndarray,
    offset_y_arcsec: np.ndarray,
    *,
    aperture_radius_arcsec: float,
    spaxel_scale_arcsec: float,
    subpixels: int = DEFAULT_SUBPIXELS_PER_SPAXEL,
) -> np.ndarray:
    """
    Fraction of each spaxel area that falls inside a circular aperture.

    Uses subpixel supersampling within each square spaxel. Returns weights in
    [0, 1], where 1 means the full spaxel is inside the aperture.
    """
    if subpixels < 4:
        raise ValueError("subpixels must be >= 4")

    half = float(spaxel_scale_arcsec) / 2.0
    step = float(spaxel_scale_arcsec) / subpixels
    local = (np.arange(subpixels, dtype=np.float64) + 0.5) * step - half

    dx = offset_x_arcsec[..., None, None] + local[None, None, :, None]
    dy = offset_y_arcsec[..., None, None] + local[None, None, None, :]
    inside = (dx * dx + dy * dy) <= float(aperture_radius_arcsec) ** 2
    return inside.mean(axis=(2, 3)).astype(np.float32)


def spaxel_radius_grid(ny: int, nx: int, header: fits.Header) -> tuple[np.ndarray, float]:
    """Return on-sky radius [arcsec] from WCS reference pixel for each spaxel."""
    wcs = WCS(header)
    if not wcs.has_celestial:
        raise ValueError("LOGCUBE FLUX header has no celestial WCS.")

    yy, xx = np.indices((ny, nx))
    pix = np.stack([xx + 1, yy + 1], axis=-1).reshape(-1, 2)
    world = wcs.celestial.all_pix2world(pix, 0)
    center_world = wcs.celestial.all_pix2world([[header["CRPIX1"], header["CRPIX2"]]], 0)[0]
    ra_c, dec_c = float(center_world[0]), float(center_world[1])

    ra = world[:, 0].reshape(ny, nx)
    dec = world[:, 1].reshape(ny, nx)
    ra_rad = np.radians(ra - ra_c) * np.cos(np.radians(dec_c))
    dec_rad = np.radians(dec - dec_c)
    radius_arcsec = np.degrees(np.hypot(ra_rad, dec_rad)) * 3600.0

    _, _, spaxel_scale = spaxel_offsets_arcsec(ny, nx, header)
    return radius_arcsec.astype(np.float32), float(spaxel_scale)


def spaxel_has_good_data(flux: np.ndarray, ivar: np.ndarray | None, mask: np.ndarray | None) -> np.ndarray:
    """Return (n_wave, ny, nx) bool cube of usable samples."""
    good = np.isfinite(flux)
    if ivar is not None:
        good &= np.isfinite(ivar) & (ivar > 0)
    if mask is not None:
        good &= (mask & 1) == 0
    return good


def coadd_aperture_spectrum(
    flux: np.ndarray,
    ivar: np.ndarray | None,
    overlap_weights: np.ndarray,
    *,
    good_cube: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Coadd spectra using partial-pixel overlap weights.

    overlap_weights has shape (ny, nx) with values in [0, 1].
    Returns flux_coadd, ivar_coadd, effective_weight_sum, n_spaxels_used.
    """
    if flux.ndim != 3:
        raise ValueError(f"Expected flux shape (n_wave, ny, nx), got {flux.shape}")

    if good_cube is None:
        good_cube = np.isfinite(flux)
        if ivar is not None:
            good_cube &= np.isfinite(ivar) & (ivar > 0)

    overlap = overlap_weights.astype(np.float64)[None, :, :]
    effective = overlap * good_cube
    if not effective.any():
        raise ValueError("Aperture contains no valid spaxel samples.")

    n_used = (effective > 0).sum(axis=(1, 2)).astype(np.int32)
    weight_sum = effective.sum(axis=(1, 2))
    flux_out = np.sum(flux.astype(np.float64) * effective, axis=(1, 2))

    ivar_out = np.full(flux.shape[0], np.nan, dtype=np.float64)
    if ivar is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            ivar_out = 1.0 / np.sum((effective * effective) / ivar.astype(np.float64), axis=(1, 2))
    else:
        ivar_out = 1.0 / np.maximum(weight_sum, 1.0)

    bad = weight_sum <= 0
    flux_out[bad] = np.nan
    ivar_out[bad] = np.nan

    return (
        flux_out.astype(np.float32),
        ivar_out.astype(np.float32),
        weight_sum.astype(np.float32),
        n_used,
    )


def extract_fiberlike_spectrum(
    logcube: Path,
    *,
    aperture_diameter_arcsec: float = DEFAULT_APERTURE_DIAMETER_ARCSEC,
    center_mode: str = "wcs_reference",
    subpixels: int = DEFAULT_SUBPIXELS_PER_SPAXEL,
) -> tuple[dict[str, np.ndarray], dict]:
    if center_mode != "wcs_reference":
        raise ValueError(f"Unsupported center_mode: {center_mode!r}")

    with fits.open(logcube, memmap=True) as hdul:
        flux = np.asarray(hdul["FLUX"].data, dtype=np.float32)
        wave = np.asarray(hdul["WAVE"].data, dtype=np.float32)
        ivar = None if "IVAR" not in hdul else np.asarray(hdul["IVAR"].data, dtype=np.float32)
        mask = None if "MASK" not in hdul else np.asarray(hdul["MASK"].data)
        header = hdul["FLUX"].header.copy()
        primary = hdul[0].header

    ny, nx = flux.shape[1], flux.shape[2]
    offset_x, offset_y, spaxel_scale = spaxel_offsets_arcsec(ny, nx, header)
    radius_arcsec, _ = spaxel_radius_grid(ny, nx, header)
    aperture_radius = float(aperture_diameter_arcsec) / 2.0

    overlap_weights = circular_aperture_overlap_weights(
        offset_x,
        offset_y,
        aperture_radius_arcsec=aperture_radius,
        spaxel_scale_arcsec=spaxel_scale,
        subpixels=subpixels,
    )
    good_cube = spaxel_has_good_data(flux, ivar, mask)

    flux_coadd, ivar_coadd, weight_sum, n_used = coadd_aperture_spectrum(
        flux,
        ivar,
        overlap_weights,
        good_cube=good_cube,
    )

    spaxel_area = float(spaxel_scale**2)
    effective_area = float(np.sum(overlap_weights) * spaxel_area)
    target_area = float(np.pi * aperture_radius**2)

    arrays = {
        "wave": wave,
        "flux": flux_coadd,
        "ivar": ivar_coadd,
        "overlap_weights": overlap_weights.astype(np.float32),
        "radius_arcsec": radius_arcsec,
        "weight_sum_per_wave": weight_sum,
        "n_spaxels_used": n_used,
        "aperture_diameter_arcsec": np.array(aperture_diameter_arcsec, dtype=np.float32),
        "spaxel_scale_arcsec": np.array(spaxel_scale, dtype=np.float32),
    }

    metadata = {
        "logcube_path": str(logcube),
        "plateifu": str(primary.get("PLATEIFU", "")),
        "obj_ra_deg": float(primary.get("OBJRA", np.nan)),
        "obj_dec_deg": float(primary.get("OBJDEC", np.nan)),
        "ifu_ra_deg": float(primary.get("IFURA", np.nan)),
        "ifu_dec_deg": float(primary.get("IFUDEC", np.nan)),
        "center_mode": center_mode,
        "coadd_method": "circular_aperture_partial_pixel_overlap",
        "subpixels_per_spaxel": int(subpixels),
        "aperture_diameter_arcsec": float(aperture_diameter_arcsec),
        "aperture_radius_arcsec": float(aperture_radius),
        "spaxel_scale_arcsec": float(spaxel_scale),
        "native_shape": [int(ny), int(nx)],
        "n_wave": int(wave.size),
        "n_spaxels_with_overlap": int(np.sum(overlap_weights > 0)),
        "n_spaxels_fully_inside": int(np.sum(overlap_weights >= 0.999)),
        "median_n_spaxels_used_per_wave": int(np.median(n_used)),
        "median_weight_sum_per_wave": float(np.median(weight_sum)),
        "effective_area_arcsec2": effective_area,
        "target_aperture_area_arcsec2": target_area,
        "wave_min": float(np.nanmin(wave)),
        "wave_max": float(np.nanmax(wave)),
        "flux_unit": "1E-17 erg/s/cm^2/Angstrom (partial-spaxel overlap sum)",
        "comparison_notes": {
            "sdss_legacy_fiber_diameter_arcsec": SDSS_LEGACY_FIBER_DIAMETER_ARCSEC,
            "boss_fiber_diameter_arcsec": BOSS_FIBER_DIAMETER_ARCSEC,
            "manga_spaxel_scale_arcsec": float(spaxel_scale),
            "spaxel_area_arcsec2": spaxel_area,
            "aperture_area_arcsec2": target_area,
            "overlap_weight_definition": "fraction of each 0.5 arcsec spaxel area inside the circular aperture",
            "flux_coadd_formula": "sum(spaxel_flux * overlap_fraction) over good spaxels",
            "ivar_coadd_formula": "1 / sum((overlap_fraction^2) / spaxel_ivar)",
        },
    }
    return arrays, metadata


def write_fake_sdss_spectrum(
    gal_dir: Path,
    *,
    aperture_diameter_arcsec: float = DEFAULT_APERTURE_DIAMETER_ARCSEC,
    subpixels: int = DEFAULT_SUBPIXELS_PER_SPAXEL,
    out_dir_name: str = "fake_sdss_spectra",
    out_npz: str | None = None,
    out_json: str = "metadata.json",
) -> dict:
    plate, ifu = gal_dir.name.split("_", 1)
    logcube = logcube_path(gal_dir, plate, ifu)
    if logcube is None:
        raise FileNotFoundError(f"No LOGCUBE found in {gal_dir}")

    arrays, metadata = extract_fiberlike_spectrum(
        logcube,
        aperture_diameter_arcsec=aperture_diameter_arcsec,
        subpixels=subpixels,
    )
    metadata["spectrum_type"] = "fake_sdss_fiber_aperture"
    metadata["is_real_sdss_fiber"] = False

    out_dir = gal_dir / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_npz is None:
        dia = int(round(aperture_diameter_arcsec * 10))
        out_npz = f"manga-{plate}-{ifu}-fake-sdss-spectrum-{dia}mas.npz"
    npz_path = out_dir / out_npz
    json_path = out_dir / out_json
    np.savez_compressed(npz_path, **arrays)
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "plateifu": f"{plate}-{ifu}",
        "npz": npz_path,
        "metadata": json_path,
        **metadata,
    }


# Backward-compatible alias.
write_fiberlike_spectrum = write_fake_sdss_spectrum


def load_fake_sdss_spectrum(galaxy_dir: Path | str):
    galaxy_dir = Path(galaxy_dir)
    search_dir = galaxy_dir / "fake_sdss_spectra"
    matches = sorted(search_dir.glob("manga-*-fake-sdss-spectrum-*.npz"))
    if not matches:
        raise FileNotFoundError(f"No fake SDSS spectrum NPZ found under {search_dir}")
    return np.load(matches[0])


load_fiberlike_spectrum = load_fake_sdss_spectrum
