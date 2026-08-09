"""
Physical-property Pipe3D maps (ages, metallicities, kinematics, SFR, …).

Writes a separate NPZ from the legacy emission-line targets in ``amara_maps.npz``:

  manga_sdss_fits/<plate>_<ifu>/amara_phys_maps.npz
  manga_sdss_fits/<plate>_<ifu>/amara_phys_maps_metadata.json

Each quantity with an error plane also stores ``*_snr`` so training can apply
spaxel-level S/N cuts without re-exporting.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
from astropy.io import fits
from astropy.table import Table

from manga_prep.targets.pipe3d_maps import (
    DEFAULT_TARGET_SIZE,
    center_pad,
    discover_pipe3d_cubes,
    extract_select_reg_footprint,
    infer_plateifu_from_path,
    max_native_shape,
    native_shape_from_pipe3d,
)

AMARA_PHYS_MAPS_NPZ = "amara_phys_maps.npz"
AMARA_PHYS_MAPS_META = "amara_phys_maps_metadata.json"

INTRINSIC_HA_HB = 2.86
K_HALPHA_CCM = 2.536
K_HBETA_CCM = 3.610
H_ALPHA_FLUX_UNIT = 1e-16
SFR_HA_COEFFICIENT = 5.5e-42
CM_PER_MPC = 3.0856775814913673e24
LN10 = np.log(10.0)

# Direct Pipe3D quantities written into amara_phys_maps.npz.
AMARA_PHYS_DIRECT_KEYS = (
    "lw_age",
    "mw_age",
    "lw_metallicity",
    "mw_metallicity",
    "ha_flux",
    "hbeta_flux",
    "oiii_5007_flux",
    "nii_6584_flux",
    "ha_ew",
    "stellar_av",
    "stellar_velocity",
    "stellar_sigma",
    "stellar_mass_density",
    "hb_abs_index",
    "d4000",
)

AMARA_PHYS_DERIVED_KEYS = (
    "balmer_decrement_ha_hb",
    "a_halpha_balmer",
    "log_sfr_halpha",
    "log_sigma_sfr_halpha",
    "gas_metallicity_o3n2_pp04",
)

PIPE3D_MAP_SPECS = [
    {
        "key": "lw_age",
        "label": "luminosity-weighted stellar age",
        "extension": "SSP",
        "plane": 5,
        "unit": "log10(yr)",
        "transform": "linear",
        "clip_min": 7.0,
        "clip_max": 10.2,
        "error_plane": 7,
    },
    {
        "key": "mw_age",
        "label": "mass-weighted stellar age",
        "extension": "SSP",
        "plane": 6,
        "unit": "log10(yr)",
        "transform": "linear",
        "clip_min": 7.0,
        "clip_max": 10.2,
        "error_plane": 7,
    },
    {
        "key": "lw_metallicity",
        "label": "luminosity-weighted stellar metallicity",
        "extension": "SSP",
        "plane": 8,
        "unit": "log10(Z/Zsun)",
        "transform": "linear",
        "clip_min": -2.5,
        "clip_max": 0.5,
        "error_plane": 10,
    },
    {
        "key": "mw_metallicity",
        "label": "mass-weighted stellar metallicity",
        "extension": "SSP",
        "plane": 9,
        "unit": "log10(Z/Zsun)",
        "transform": "linear",
        "clip_min": -2.5,
        "clip_max": 0.5,
        "error_plane": 10,
    },
    {
        "key": "ha_flux",
        "label": "Halpha flux",
        "extension": "FLUX_ELINES",
        "line_patterns": ["Halpha", "Ha"],
        "plane_offset": 0,
        "unit": "1e-16 erg/s/cm^2",
        "transform": "log10_positive",
        "clip_min": -5.0,
        "clip_max": 1.0,
        "error_plane_offset": 228,
    },
    {
        "key": "hbeta_flux",
        "label": "Hbeta flux",
        "extension": "FLUX_ELINES",
        "line_patterns": ["Hbeta", "Hb"],
        "plane_offset": 0,
        "unit": "1e-16 erg/s/cm^2",
        "transform": "log10_positive",
        "clip_min": -5.0,
        "clip_max": 1.0,
        "error_plane_offset": 228,
    },
    {
        "key": "oiii_5007_flux",
        "label": "[OIII]5007 flux",
        "extension": "FLUX_ELINES",
        "line_patterns": ["[OIII]5007", "[OIII] 5007", "OIII5007"],
        "plane_offset": 0,
        "unit": "1e-16 erg/s/cm^2",
        "transform": "log10_positive",
        "clip_min": -5.0,
        "clip_max": 1.0,
        "error_plane_offset": 228,
    },
    {
        "key": "nii_6584_flux",
        "label": "[NII]6584 flux",
        "extension": "FLUX_ELINES",
        "line_patterns": ["[NII]6584", "[NII] 6584", "NII6584"],
        "plane_offset": 0,
        "unit": "1e-16 erg/s/cm^2",
        "transform": "log10_positive",
        "clip_min": -5.0,
        "clip_max": 1.0,
        "error_plane_offset": 228,
    },
    {
        "key": "ha_ew",
        "label": "Halpha EW",
        "extension": "FLUX_ELINES",
        "line_patterns": ["Halpha", "Ha"],
        "plane_offset": 171,
        "unit": "Angstrom",
        "transform": "log10_negative_emission",
        "clip_min": 0.0,
        "clip_max": 3.0,
        "error_plane_offset": 399,
    },
    {
        "key": "stellar_av",
        "label": "stellar Av",
        "extension": "SSP",
        "plane": 11,
        "unit": "mag",
        "transform": "linear",
        "clip_min": 0.0,
        "clip_max": 3.0,
        "error_plane": 12,
    },
    {
        "key": "stellar_velocity",
        "label": "stellar velocity",
        "extension": "SSP",
        "plane": 13,
        "unit": "km/s",
        "transform": "linear",
        "clip_min": -300.0,
        "clip_max": 300.0,
        "error_plane": 14,
    },
    {
        "key": "stellar_sigma",
        "label": "stellar velocity dispersion",
        "extension": "SSP",
        "plane": 15,
        "unit": "km/s",
        "transform": "linear",
        "clip_min": 0.0,
        "clip_max": 300.0,
        "error_plane": 16,
    },
    {
        "key": "stellar_mass_density",
        "label": "stellar mass surface density",
        "extension": "SSP",
        "plane": 18,
        "unit": "log10(Msun/pc^2)",
        "transform": "linear",
        "clip_min": 0.0,
        "clip_max": 10.0,
        "error_plane": 20,
    },
    {
        "key": "hb_abs_index",
        "label": "Hbeta stellar absorption index",
        "extension": "INDICES",
        "plane": 1,
        "unit": "Angstrom",
        "transform": "linear",
        "clip_min": 0.0,
        "clip_max": 10.0,
        "error_plane": 10,
    },
    {
        "key": "d4000",
        "label": "D4000 spectral index",
        "extension": "INDICES",
        "plane": 5,
        "unit": "",
        "transform": "linear",
        "clip_min": 1.0,
        "clip_max": 2.5,
        "error_plane": 14,
    },
]

DERIVED_MAP_SPECS = [
    {
        "key": "balmer_decrement_ha_hb",
        "label": "Balmer decrement Halpha/Hbeta",
        "unit": "",
        "clip_min": 2.86,
        "clip_max": 8.0,
    },
    {
        "key": "a_halpha_balmer",
        "label": "Halpha attenuation from Balmer decrement",
        "unit": "mag",
        "clip_min": 0.0,
        "clip_max": 5.0,
    },
    {
        "key": "log_sfr_halpha",
        "label": "log10 Halpha SFR",
        "unit": "log10(Msun/yr)",
        "clip_min": -6.0,
        "clip_max": 0.0,
    },
    {
        "key": "log_sigma_sfr_halpha",
        "label": "log10 Halpha SFR surface density",
        "unit": "log10(Msun/yr/kpc^2)",
        "clip_min": -4.0,
        "clip_max": 0.0,
    },
    {
        "key": "gas_metallicity_o3n2_pp04",
        "label": "gas-phase metallicity PP04 O3N2",
        "unit": "12 + log(O/H)",
        "clip_min": 8.0,
        "clip_max": 9.0,
    },
]


def native_float_array(values):
    return np.array(values, dtype=np.float64, copy=True)


def _to_plain_value(value):
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    return value


def flux_elines_lookup(hdul):
    hdr = hdul["FLUX_ELINES"].header
    rows = []
    for i in range(57):
        raw_name = str(hdr.get(f"NAME{i}", ""))
        rows.append(
            {
                "line_index": i,
                "raw_name": raw_name,
                "clean_name": raw_name.replace("flux ", "").strip(),
                "wave": hdr.get(f"WAVE{i}", np.nan),
                "unit": hdr.get(f"UNIT{i}", ""),
            }
        )
    return rows


def find_line_index(line_lookup, patterns):
    for pattern in patterns:
        pattern_lower = pattern.lower()
        for row in line_lookup:
            if pattern_lower in str(row["clean_name"]).lower():
                return int(row["line_index"])
    return None


def _line_plane(spec, line_lookup):
    line_index = find_line_index(line_lookup, spec["line_patterns"])
    if line_index is None:
        raise KeyError(f"Could not find {spec['key']} in FLUX_ELINES.")
    return line_index + int(spec.get("plane_offset", 0))


def _line_error_plane(spec, line_lookup):
    line_index = find_line_index(line_lookup, spec["line_patterns"])
    if line_index is None:
        raise KeyError(f"Could not find {spec['key']} error in FLUX_ELINES.")
    return line_index + int(spec["error_plane_offset"])


def extract_direct_pipe3d_maps(path):
    maps = {}
    with fits.open(path, memmap=True) as hdul:
        line_lookup = flux_elines_lookup(hdul)
        extension_cache = {}
        for spec in PIPE3D_MAP_SPECS:
            extension = spec["extension"]
            if extension not in extension_cache:
                extension_cache[extension] = native_float_array(hdul[extension].data)
            cube = extension_cache[extension]
            if "plane" in spec:
                plane = int(spec["plane"])
            else:
                plane = _line_plane(spec, line_lookup)
            maps[spec["key"]] = np.asarray(cube[plane], dtype=np.float64)
            if "error_plane" in spec:
                maps[f"{spec['key']}_err"] = np.asarray(
                    cube[int(spec["error_plane"])],
                    dtype=np.float64,
                )
            elif "error_plane_offset" in spec:
                error_plane = _line_error_plane(spec, line_lookup)
                maps[f"{spec['key']}_err"] = np.asarray(cube[error_plane], dtype=np.float64)
    return maps


_DRPALL_ROW_CACHE: dict[str, dict[str, dict]] = {}


def _drpall_rows_by_plateifu(drpall_path) -> dict[str, dict]:
    """Load DRPall once per process and index by plateifu."""
    key = str(Path(drpall_path).resolve())
    cached = _DRPALL_ROW_CACHE.get(key)
    if cached is not None:
        return cached

    tab = Table.read(drpall_path, hdu=1)
    col_lookup = {name.lower(): name for name in tab.colnames}
    plateifu_col = col_lookup.get("plateifu")
    if plateifu_col is None:
        raise KeyError(f"Could not find plateifu column in {drpall_path}")

    rows: dict[str, dict] = {}
    colnames = list(tab.colnames)
    for i in range(len(tab)):
        plateifu = str(_to_plain_value(tab[plateifu_col][i])).strip()
        if not plateifu or plateifu in rows:
            continue
        row = tab[i]
        rows[plateifu] = {name: _to_plain_value(row[name]) for name in colnames}
    _DRPALL_ROW_CACHE[key] = rows
    return rows


def lookup_drpall_row(plateifu, drpall_path):
    rows = _drpall_rows_by_plateifu(drpall_path)
    plateifu = str(plateifu).strip()
    if plateifu not in rows:
        raise ValueError(f"Could not find plateifu={plateifu!r} in {drpall_path}")
    return rows[plateifu]


def positive_redshift_from_drpall_row(row):
    lower_lookup = {name.lower(): name for name in row}
    for key in ["nsa_z", "z", "zdist"]:
        actual = lower_lookup.get(key)
        if actual is None:
            continue
        try:
            value = float(row[actual])
        except Exception:
            continue
        if np.isfinite(value) and value > 0:
            return value, actual
    raise ValueError("Could not find a positive finite redshift in the DRPall row.")


def luminosity_distance_mpc_from_redshift(redshift, h0=70.0, om0=0.3):
    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u

    cosmo = FlatLambdaCDM(H0=h0 * u.km / u.s / u.Mpc, Om0=om0)
    return float(cosmo.luminosity_distance(float(redshift)).to(u.Mpc).value)


def spaxel_area_kpc2_from_redshift(redshift, h0=70.0, om0=0.3, arcsec_per_spaxel=0.5):
    luminosity_distance_mpc = luminosity_distance_mpc_from_redshift(redshift, h0=h0, om0=om0)
    angular_diameter_distance_mpc = luminosity_distance_mpc / (1.0 + float(redshift)) ** 2
    kpc_per_arcsec = angular_diameter_distance_mpc * 1000.0 / 206265.0
    spaxel_size_kpc = float(arcsec_per_spaxel) * kpc_per_arcsec
    return {
        "luminosity_distance_mpc": luminosity_distance_mpc,
        "angular_diameter_distance_mpc": angular_diameter_distance_mpc,
        "kpc_per_arcsec": kpc_per_arcsec,
        "spaxel_size_kpc": spaxel_size_kpc,
        "spaxel_area_kpc2": spaxel_size_kpc**2,
        "arcsec_per_spaxel": float(arcsec_per_spaxel),
        "cosmology_H0_km_s_Mpc": float(h0),
        "cosmology_Om0": float(om0),
    }


def _relative_error(value, error):
    value = np.asarray(value, dtype=np.float64)
    error = np.asarray(error, dtype=np.float64)
    out = np.full(value.shape, np.nan, dtype=np.float64)
    good = np.isfinite(value) & np.isfinite(error) & (value > 0) & (error >= 0)
    out[good] = error[good] / value[good]
    return out


def _log10_error(relative_error):
    return np.asarray(relative_error, dtype=np.float64) / LN10


def _line_snr(flux, error):
    snr = np.full(np.asarray(flux).shape, np.nan, dtype=np.float64)
    good = np.isfinite(flux) & np.isfinite(error) & (error > 0)
    snr[good] = flux[good] / error[good]
    return snr


def _kauffmann_2003(log_nii_ha):
    return 0.61 / (log_nii_ha - 0.05) + 1.30


def _kewley_2001(log_nii_ha):
    return 0.61 / (log_nii_ha - 0.47) + 1.19


def _science_template(shape):
    return np.full(shape, np.nan, dtype=np.float64)


def derive_science_maps(
    raw_maps,
    plateifu,
    drpall_path,
    snr_min=3.0,
    arcsec_per_spaxel=0.5,
    h0=70.0,
    om0=0.3,
):
    row = lookup_drpall_row(plateifu, drpall_path)
    redshift, redshift_source_column = positive_redshift_from_drpall_row(row)
    distance_metadata = spaxel_area_kpc2_from_redshift(
        redshift,
        h0=h0,
        om0=om0,
        arcsec_per_spaxel=arcsec_per_spaxel,
    )

    ha = raw_maps["ha_flux"]
    hb = raw_maps["hbeta_flux"]
    oiii = raw_maps["oiii_5007_flux"]
    nii = raw_maps["nii_6584_flux"]
    ha_err = raw_maps["ha_flux_err"]
    hb_err = raw_maps["hbeta_flux_err"]
    oiii_err = raw_maps["oiii_5007_flux_err"]
    nii_err = raw_maps["nii_6584_flux_err"]

    ha_snr = _line_snr(ha, ha_err)
    hb_snr = _line_snr(hb, hb_err)
    oiii_snr = _line_snr(oiii, oiii_err)
    nii_snr = _line_snr(nii, nii_err)

    line_valid = (
        (ha > 0)
        & (hb > 0)
        & (oiii > 0)
        & (nii > 0)
        & (ha_snr >= snr_min)
        & (hb_snr >= snr_min)
        & (oiii_snr >= snr_min)
        & (nii_snr >= snr_min)
    )

    log_nii_ha = _science_template(ha.shape)
    log_oiii_hbeta = _science_template(ha.shape)
    log_nii_ha[line_valid] = np.log10(nii[line_valid] / ha[line_valid])
    log_oiii_hbeta[line_valid] = np.log10(oiii[line_valid] / hb[line_valid])
    y_kauffmann = _kauffmann_2003(log_nii_ha)
    y_kewley = _kewley_2001(log_nii_ha)
    is_sf_bpt = line_valid & (log_oiii_hbeta < y_kauffmann)
    is_comp_bpt = line_valid & (log_oiii_hbeta >= y_kauffmann) & (log_oiii_hbeta < y_kewley)
    is_agn_bpt = line_valid & (log_oiii_hbeta >= y_kewley)
    bpt_class_code = np.zeros(ha.shape, dtype=np.uint8)
    bpt_class_code[is_sf_bpt] = 1
    bpt_class_code[is_comp_bpt] = 2
    bpt_class_code[is_agn_bpt] = 3

    o3n2 = _science_template(ha.shape)
    o3n2[line_valid] = np.log10((oiii[line_valid] / hb[line_valid]) / (nii[line_valid] / ha[line_valid]))
    rel_o3n2 = np.sqrt(
        _relative_error(oiii, oiii_err) ** 2
        + _relative_error(hb, hb_err) ** 2
        + _relative_error(nii, nii_err) ** 2
        + _relative_error(ha, ha_err) ** 2
    )
    o3n2_err = _log10_error(rel_o3n2)
    metallicity_valid = line_valid & is_sf_bpt & np.isfinite(o3n2) & (o3n2 >= -1.0) & (o3n2 <= 1.9)
    metallicity = _science_template(ha.shape)
    metallicity_err = _science_template(ha.shape)
    metallicity[metallicity_valid] = 8.73 - 0.32 * o3n2[metallicity_valid]
    metallicity_err[metallicity_valid] = 0.32 * o3n2_err[metallicity_valid]

    balmer_valid = (ha > 0) & (hb > 0) & (ha_snr >= snr_min) & (hb_snr >= snr_min)
    balmer_decrement = _science_template(ha.shape)
    balmer_decrement_err = _science_template(ha.shape)
    balmer_decrement[balmer_valid] = ha[balmer_valid] / hb[balmer_valid]
    rel_balmer = np.sqrt(_relative_error(ha, ha_err) ** 2 + _relative_error(hb, hb_err) ** 2)
    balmer_decrement_err[balmer_valid] = balmer_decrement[balmer_valid] * rel_balmer[balmer_valid]

    ratio = balmer_decrement / INTRINSIC_HA_HB
    ebv = _science_template(ha.shape)
    ebv_err = _science_template(ha.shape)
    good_ratio = balmer_valid & np.isfinite(ratio) & (ratio > 0)
    ebv[good_ratio] = (2.5 / (K_HBETA_CCM - K_HALPHA_CCM)) * np.log10(ratio[good_ratio])
    ebv_err[good_ratio] = (
        2.5
        / (K_HBETA_CCM - K_HALPHA_CCM)
        / LN10
        * (balmer_decrement_err[good_ratio] / balmer_decrement[good_ratio])
    )
    ebv = np.where(np.isfinite(ebv), np.maximum(ebv, 0.0), np.nan)
    a_halpha = K_HALPHA_CCM * ebv
    a_halpha_err = K_HALPHA_CCM * ebv_err

    distance_cm = distance_metadata["luminosity_distance_mpc"] * CM_PER_MPC
    lha_obs = 4.0 * np.pi * distance_cm**2 * ha * H_ALPHA_FLUX_UNIT
    lha_obs_err = 4.0 * np.pi * distance_cm**2 * ha_err * H_ALPHA_FLUX_UNIT
    lha_corr = lha_obs * 10.0 ** (0.4 * a_halpha)
    rel_lha_obs = _relative_error(lha_obs, lha_obs_err)
    rel_attenuation = 0.4 * LN10 * a_halpha_err
    lha_corr_err = lha_corr * np.sqrt(rel_lha_obs**2 + rel_attenuation**2)
    sfr = SFR_HA_COEFFICIENT * lha_corr
    sfr_err = SFR_HA_COEFFICIENT * lha_corr_err
    sfr_valid = (ha > 0) & balmer_valid & is_sf_bpt & np.isfinite(a_halpha)

    spaxel_area = distance_metadata["spaxel_area_kpc2"]
    sigma_sfr = sfr / spaxel_area
    sigma_sfr_err = sfr_err / spaxel_area

    log_sfr = _science_template(ha.shape)
    log_sfr_err = _science_template(ha.shape)
    positive_sfr = sfr_valid & (sfr > 0)
    log_sfr[positive_sfr] = np.log10(sfr[positive_sfr])
    log_sfr_err[positive_sfr] = _log10_error(_relative_error(sfr, sfr_err))[positive_sfr]

    log_sigma_sfr = _science_template(ha.shape)
    log_sigma_sfr_err = _science_template(ha.shape)
    positive_sigma = sfr_valid & (sigma_sfr > 0)
    log_sigma_sfr[positive_sigma] = np.log10(sigma_sfr[positive_sigma])
    log_sigma_sfr_err[positive_sigma] = _log10_error(
        _relative_error(sigma_sfr, sigma_sfr_err)
    )[positive_sigma]

    maps = {
        "ha_snr": ha_snr,
        "hbeta_snr": hb_snr,
        "oiii_5007_snr": oiii_snr,
        "nii_6584_snr": nii_snr,
        "bpt_valid": line_valid.astype(np.uint8),
        "is_sf_bpt": is_sf_bpt.astype(np.uint8),
        "is_comp_bpt": is_comp_bpt.astype(np.uint8),
        "is_agn_bpt": is_agn_bpt.astype(np.uint8),
        "bpt_class_code": bpt_class_code,
        "balmer_valid": balmer_valid.astype(np.uint8),
        "gas_metallicity_o3n2_valid": metallicity_valid.astype(np.uint8),
        "sfr_valid": sfr_valid.astype(np.uint8),
        "o3n2": np.where(line_valid, o3n2, np.nan),
        "o3n2_err": np.where(line_valid, o3n2_err, np.nan),
        "gas_metallicity_o3n2_pp04": metallicity,
        "gas_metallicity_o3n2_pp04_err": metallicity_err,
        "balmer_decrement_ha_hb": np.where(balmer_valid, balmer_decrement, np.nan),
        "balmer_decrement_ha_hb_err": np.where(balmer_valid, balmer_decrement_err, np.nan),
        "a_halpha_balmer": np.where(balmer_valid, a_halpha, np.nan),
        "a_halpha_balmer_err": np.where(balmer_valid, a_halpha_err, np.nan),
        "lha_obs": np.where(sfr_valid, lha_obs, np.nan),
        "lha_obs_err": np.where(sfr_valid, lha_obs_err, np.nan),
        "lha_corr": np.where(sfr_valid, lha_corr, np.nan),
        "lha_corr_err": np.where(sfr_valid, lha_corr_err, np.nan),
        "sfr_halpha": np.where(sfr_valid, sfr, np.nan),
        "sfr_halpha_err": np.where(sfr_valid, sfr_err, np.nan),
        "log_sfr_halpha": log_sfr,
        "log_sfr_halpha_err": log_sfr_err,
        "sigma_sfr_halpha": np.where(sfr_valid, sigma_sfr, np.nan),
        "sigma_sfr_halpha_err": np.where(sfr_valid, sigma_sfr_err, np.nan),
        "log_sigma_sfr_halpha": log_sigma_sfr,
        "log_sigma_sfr_halpha_err": log_sigma_sfr_err,
    }
    metadata = {
        "redshift": float(redshift),
        "redshift_source_column": redshift_source_column,
        **{key: float(value) for key, value in distance_metadata.items()},
        "intrinsic_ha_hb": INTRINSIC_HA_HB,
        "k_halpha_ccm": K_HALPHA_CCM,
        "k_hbeta_ccm": K_HBETA_CCM,
        "sfr_halpha_coefficient": SFR_HA_COEFFICIENT,
        "snr_min": float(snr_min),
        "bpt_class_code": {"unclassified": 0, "star_forming": 1, "composite": 2, "agn": 3},
        "n_bpt_valid": int(np.sum(line_valid)),
        "n_sf_bpt": int(np.sum(is_sf_bpt)),
        "n_comp_bpt": int(np.sum(is_comp_bpt)),
        "n_agn_bpt": int(np.sum(is_agn_bpt)),
        "n_sfr_valid": int(np.sum(sfr_valid)),
        "n_gas_metallicity_o3n2_valid": int(np.sum(metallicity_valid)),
    }
    return maps, metadata


def transform_map(values, transform):
    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if transform == "linear":
        good = np.isfinite(values)
        out[good] = values[good]
    elif transform == "log10_positive":
        good = np.isfinite(values) & (values > 0)
        out[good] = np.log10(values[good])
    elif transform == "log10_negative_emission":
        good = np.isfinite(values) & (values < 0)
        out[good] = np.log10(-values[good])
    else:
        raise ValueError(f"Unknown transform: {transform!r}")
    return out


def scale_map(values, spec, clip=True):
    transformed = transform_map(values, spec["transform"])
    clip_min = float(spec["clip_min"])
    clip_max = float(spec["clip_max"])
    if clip_max <= clip_min:
        raise ValueError(f"Invalid clipping range for {spec['key']}: {clip_min}, {clip_max}")
    clipped = np.clip(transformed, clip_min, clip_max) if clip else transformed
    return (clipped - clip_min) / (clip_max - clip_min)


def scale_linear_error_map(errors, spec):
    errors = np.asarray(errors, dtype=np.float64)
    scale_width = float(spec["clip_max"]) - float(spec["clip_min"])
    out = np.full(errors.shape, np.nan, dtype=np.float64)
    good = np.isfinite(errors) & (errors >= 0)
    out[good] = errors[good] / scale_width
    return out


def map_stats(raw_map, scaled_map):
    raw = np.asarray(raw_map, dtype=np.float64)
    scaled = np.asarray(scaled_map, dtype=np.float64)
    finite_raw = raw[np.isfinite(raw)]
    finite_scaled = scaled[np.isfinite(scaled)]
    return {
        "raw_finite_count": int(finite_raw.size),
        "raw_min": float(np.min(finite_raw)) if finite_raw.size else None,
        "raw_max": float(np.max(finite_raw)) if finite_raw.size else None,
        "scaled_finite_count": int(finite_scaled.size),
        "scaled_min": float(np.min(finite_scaled)) if finite_scaled.size else None,
        "scaled_max": float(np.max(finite_scaled)) if finite_scaled.size else None,
        "scaled_at_0_count": int(np.sum(finite_scaled == 0.0)) if finite_scaled.size else 0,
        "scaled_at_1_count": int(np.sum(finite_scaled == 1.0)) if finite_scaled.size else 0,
    }


def build_phys_arrays(
    pipe3d_path,
    target_shape=(DEFAULT_TARGET_SIZE, DEFAULT_TARGET_SIZE),
    clip=True,
    include_derived=False,
    drpall_path=None,
    snr_min=3.0,
    arcsec_per_spaxel=0.5,
):
    raw_maps = extract_direct_pipe3d_maps(pipe3d_path)
    first_map = next(iter(raw_maps.values()))
    native_y, native_x = first_map.shape
    plateifu = infer_plateifu_from_path(pipe3d_path)
    arrays = {
        "native_shape": np.array([native_y, native_x], dtype=np.int16),
        "native_ny": np.array(native_y, dtype=np.int16),
        "native_nx": np.array(native_x, dtype=np.int16),
        "native_spaxel_count": np.array(native_y * native_x, dtype=np.int32),
        "target_shape": np.array(target_shape, dtype=np.int16),
    }
    metadata = {
        "pipe3d_path": str(pipe3d_path),
        "plateifu": plateifu,
        "native_shape": [int(native_y), int(native_x)],
        "native_spaxel_count": int(native_y * native_x),
        "target_shape": [int(target_shape[0]), int(target_shape[1])],
        "clip_scaled_to_0_1": bool(clip),
        "maps": [],
        "derived_science": None,
        "product": "amara_phys_maps",
    }

    footprint = extract_select_reg_footprint(pipe3d_path, (native_y, native_x))
    padded_footprint = center_pad(footprint, target_shape, pad_value=0).astype(np.uint8)
    arrays["native_footprint_mask"] = padded_footprint

    for spec in PIPE3D_MAP_SPECS:
        key = spec["key"]
        raw_map = raw_maps[key]
        scaled_map = scale_map(raw_map, spec, clip=clip)
        arrays[f"{key}_raw"] = center_pad(raw_map, target_shape, pad_value=np.nan)
        arrays[f"{key}_scaled"] = center_pad(scaled_map, target_shape, pad_value=np.nan)
        feature_valid = np.isfinite(arrays[f"{key}_scaled"]).astype(np.uint8)
        arrays[f"{key}_valid_mask"] = feature_valid
        arrays[f"{key}_loss_mask"] = (
            padded_footprint.astype(bool) & feature_valid.astype(bool)
        ).astype(np.uint8)

        error_array_name = None
        snr_array_name = None
        error_key = f"{key}_err"
        if error_key in raw_maps:
            err_map = raw_maps[error_key]
            error_array_name = f"{key}_err_raw"
            snr_array_name = f"{key}_snr"
            arrays[error_array_name] = center_pad(err_map, target_shape, pad_value=np.nan)
            arrays[snr_array_name] = center_pad(_line_snr(raw_map, err_map), target_shape, pad_value=np.nan)
            # Convenience mask at the export-time snr_min (re-threshold with *_snr at train time).
            snr_ok = np.isfinite(arrays[snr_array_name]) & (arrays[snr_array_name] >= float(snr_min))
            arrays[f"{key}_snr_mask"] = (
                padded_footprint.astype(bool) & feature_valid.astype(bool) & snr_ok
            ).astype(np.uint8)

        metadata["maps"].append(
            {
                "key": key,
                "raw_array": f"{key}_raw",
                "error_array": error_array_name,
                "snr_array": snr_array_name,
                "scaled_array": f"{key}_scaled",
                "valid_mask": f"{key}_valid_mask",
                "loss_mask": f"{key}_loss_mask",
                "snr_mask": f"{key}_snr_mask" if snr_array_name else None,
                "label": spec["label"],
                "unit": spec["unit"],
                "transform": spec["transform"],
                "clip_min": float(spec["clip_min"]),
                "clip_max": float(spec["clip_max"]),
                "snr_min_for_snr_mask": float(snr_min),
                "scaled_formula": "(clip(transform(raw), clip_min, clip_max) - clip_min) / (clip_max - clip_min)",
                **map_stats(raw_map, scaled_map),
            }
        )

    if include_derived:
        if drpall_path is None:
            raise ValueError("Set drpall_path when include_derived=True so Halpha SFR can be calculated.")
        science_maps, science_metadata = derive_science_maps(
            raw_maps,
            plateifu=plateifu,
            drpall_path=drpall_path,
            snr_min=snr_min,
            arcsec_per_spaxel=arcsec_per_spaxel,
        )
        metadata["derived_science"] = science_metadata

        for key, value in science_maps.items():
            if key.endswith("_valid") or key.startswith("is_") or key in {"bpt_valid", "bpt_class_code"}:
                arrays[f"{key}_mask"] = center_pad(value.astype(np.uint8), target_shape, pad_value=0)
            elif key.endswith("_snr") and not key.endswith("_err"):
                # Line SNRs already stored above for direct maps; keep derived copies too.
                arrays[f"{key}_raw"] = center_pad(value, target_shape, pad_value=np.nan)
            else:
                arrays[f"{key}_raw"] = center_pad(value, target_shape, pad_value=np.nan)

        for spec in DERIVED_MAP_SPECS:
            key = spec["key"]
            raw_map = science_maps[key]
            err_map = science_maps.get(f"{key}_err")
            scaled_map = scale_map(
                raw_map,
                {
                    "key": key,
                    "transform": "linear",
                    "clip_min": spec["clip_min"],
                    "clip_max": spec["clip_max"],
                },
                clip=clip,
            )
            arrays[f"{key}_scaled"] = center_pad(scaled_map, target_shape, pad_value=np.nan)
            feature_valid = np.isfinite(arrays[f"{key}_scaled"]).astype(np.uint8)
            arrays[f"{key}_valid_mask"] = feature_valid
            arrays[f"{key}_loss_mask"] = (
                padded_footprint.astype(bool) & feature_valid.astype(bool)
            ).astype(np.uint8)
            scaled_error_array = None
            snr_array_name = None
            if err_map is not None:
                scaled_err = scale_linear_error_map(err_map, spec)
                scaled_error_array = f"{key}_scaled_err"
                arrays[scaled_error_array] = center_pad(scaled_err, target_shape, pad_value=np.nan)
                snr_array_name = f"{key}_snr"
                arrays[snr_array_name] = center_pad(_line_snr(raw_map, err_map), target_shape, pad_value=np.nan)
                snr_ok = np.isfinite(arrays[snr_array_name]) & (arrays[snr_array_name] >= float(snr_min))
                arrays[f"{key}_snr_mask"] = (
                    padded_footprint.astype(bool) & feature_valid.astype(bool) & snr_ok
                ).astype(np.uint8)
            metadata["maps"].append(
                {
                    "key": key,
                    "raw_array": f"{key}_raw",
                    "error_array": f"{key}_err_raw" if f"{key}_err_raw" in arrays else None,
                    "snr_array": snr_array_name,
                    "scaled_array": f"{key}_scaled",
                    "scaled_error_array": scaled_error_array,
                    "valid_mask": f"{key}_valid_mask",
                    "loss_mask": f"{key}_loss_mask",
                    "snr_mask": f"{key}_snr_mask" if snr_array_name else None,
                    "label": spec["label"],
                    "unit": spec["unit"],
                    "transform": "linear",
                    "clip_min": float(spec["clip_min"]),
                    "clip_max": float(spec["clip_max"]),
                    "snr_min_for_snr_mask": float(snr_min),
                    "scaled_formula": "(clip(raw, clip_min, clip_max) - clip_min) / (clip_max - clip_min)",
                    "scaled_error_formula": "raw_error / (clip_max - clip_min)",
                    **map_stats(raw_map, scaled_map),
                }
            )
    return arrays, metadata


def write_amara_phys_maps(
    pipe3d_path,
    galaxy_dir=None,
    target_shape=(DEFAULT_TARGET_SIZE, DEFAULT_TARGET_SIZE),
    clip=True,
    include_derived=False,
    drpall_path=None,
    snr_min=3.0,
    arcsec_per_spaxel=0.5,
):
    """Write physical-property maps into a galaxy folder as amara_phys_maps.npz."""
    pipe3d_path = Path(pipe3d_path)
    plateifu = infer_plateifu_from_path(pipe3d_path)
    if galaxy_dir is None:
        galaxy_dir = pipe3d_path.parent
    galaxy_dir = Path(galaxy_dir)
    galaxy_dir.mkdir(parents=True, exist_ok=True)

    arrays, metadata = build_phys_arrays(
        pipe3d_path,
        target_shape=target_shape,
        clip=clip,
        include_derived=include_derived,
        drpall_path=drpall_path,
        snr_min=snr_min,
        arcsec_per_spaxel=arcsec_per_spaxel,
    )
    npz_path = galaxy_dir / AMARA_PHYS_MAPS_NPZ
    json_path = galaxy_dir / AMARA_PHYS_MAPS_META
    np.savez_compressed(npz_path, **arrays)
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "plateifu": plateifu,
        "galaxy_dir": galaxy_dir,
        "npz": npz_path,
        "metadata": json_path,
        **metadata,
    }


def write_collaborator_phys_maps(
    pipe3d_path,
    out_dir,
    target_shape=(DEFAULT_TARGET_SIZE, DEFAULT_TARGET_SIZE),
    clip=True,
    include_derived=False,
    drpall_path=None,
    snr_min=3.0,
    arcsec_per_spaxel=0.5,
):
    """Write physical maps under a separate output tree (non in-place)."""
    pipe3d_path = Path(pipe3d_path)
    plateifu = infer_plateifu_from_path(pipe3d_path)
    galaxy_dir = Path(out_dir) / plateifu
    galaxy_dir.mkdir(parents=True, exist_ok=True)

    arrays, metadata = build_phys_arrays(
        pipe3d_path,
        target_shape=target_shape,
        clip=clip,
        include_derived=include_derived,
        drpall_path=drpall_path,
        snr_min=snr_min,
        arcsec_per_spaxel=arcsec_per_spaxel,
    )
    size_label = f"{int(target_shape[0])}x{int(target_shape[1])}"
    npz_path = galaxy_dir / f"{plateifu}_pipe3d_phys_maps_{size_label}.npz"
    json_path = galaxy_dir / f"{plateifu}_pipe3d_phys_maps_{size_label}_metadata.json"
    np.savez_compressed(npz_path, **arrays)
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"plateifu": plateifu, "npz": npz_path, "metadata": json_path, **metadata}


def load_amara_phys_maps(galaxy_dir):
    """Load amara_phys_maps.npz from a manga_sdss_fits/<plate_ifu> folder."""
    galaxy_dir = Path(galaxy_dir)
    npz_path = galaxy_dir / AMARA_PHYS_MAPS_NPZ
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}")
    return np.load(npz_path)


def _loss_mask_from_npz(arrays: dict[str, np.ndarray], key: str) -> np.ndarray:
    if f"{key}_loss_mask" in arrays:
        return arrays[f"{key}_loss_mask"].astype(np.uint8)
    footprint = arrays["native_footprint_mask"].astype(bool)
    valid = arrays[f"{key}_valid_mask"].astype(bool)
    return (footprint & valid).astype(np.uint8)


def snr_loss_mask(
    arrays: dict[str, np.ndarray],
    key: str,
    *,
    snr_min: float | None = None,
) -> np.ndarray:
    """
    Build a training mask for ``key``.

    Base mask is footprint ∩ valid (``*_loss_mask``).
    If ``snr_min`` is set, also require ``*_snr >= snr_min``.
    """
    base = _loss_mask_from_npz(arrays, key).astype(bool)
    if snr_min is None:
        return base.astype(np.uint8)
    if f"{key}_snr" not in arrays:
        raise KeyError(f"No SNR map for {key!r}; cannot apply snr_min={snr_min}")
    snr = arrays[f"{key}_snr"]
    return (base & np.isfinite(snr) & (snr >= float(snr_min))).astype(np.uint8)


def load_amara_phys_training_targets(
    galaxy_dir,
    *,
    keys: tuple[str, ...] | list[str] | None = None,
    scaled: bool = True,
    snr_min: float | None = None,
    require_sf_spaxel: bool = False,
) -> dict[str, object]:
    """
    Load physical-property targets and masks from amara_phys_maps.npz.

    ``snr_min`` re-thresholds spaxel masks from stored ``*_snr`` maps.
    ``require_sf_spaxel`` intersects loss masks with ``is_sf_bpt_mask`` when present.
    """
    if keys is None:
        keys = AMARA_PHYS_DIRECT_KEYS
    keys = tuple(keys)

    with load_amara_phys_maps(galaxy_dir) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}

    suffix = "_scaled" if scaled else "_raw"
    targets = {key: arrays[f"{key}{suffix}"].astype(np.float32) for key in keys}
    target_valid_masks = {key: arrays[f"{key}_valid_mask"].astype(np.uint8) for key in keys}
    target_loss_masks = {key: snr_loss_mask(arrays, key, snr_min=snr_min) for key in keys}
    if require_sf_spaxel:
        if "is_sf_bpt_mask" not in arrays:
            raise KeyError(
                "require_sf_spaxel=True but is_sf_bpt_mask missing; "
                "re-export amara_phys_maps.npz with --include-derived."
            )
        sf = arrays["is_sf_bpt_mask"].astype(bool)
        target_loss_masks = {
            key: (mask.astype(bool) & sf).astype(np.uint8)
            for key, mask in target_loss_masks.items()
        }
    target_snr = {
        key: arrays[f"{key}_snr"].astype(np.float32) if f"{key}_snr" in arrays else None
        for key in keys
    }

    out = {
        "targets": targets,
        "target_valid_masks": target_valid_masks,
        "target_loss_masks": target_loss_masks,
        "target_snr": target_snr,
        "footprint_mask": arrays["native_footprint_mask"].astype(np.uint8),
        "native_shape": tuple(int(x) for x in arrays["native_shape"]),
        "target_shape": tuple(int(x) for x in arrays["target_shape"]),
    }
    if "is_sf_bpt_mask" in arrays:
        out["is_sf_bpt_mask"] = arrays["is_sf_bpt_mask"].astype(np.uint8)
        out["bpt_class_code_mask"] = arrays["bpt_class_code_mask"].astype(np.uint8)
    return out


__all__ = [
    "AMARA_PHYS_DIRECT_KEYS",
    "AMARA_PHYS_DERIVED_KEYS",
    "AMARA_PHYS_MAPS_META",
    "AMARA_PHYS_MAPS_NPZ",
    "DEFAULT_TARGET_SIZE",
    "PIPE3D_MAP_SPECS",
    "build_phys_arrays",
    "discover_pipe3d_cubes",
    "load_amara_phys_maps",
    "load_amara_phys_training_targets",
    "max_native_shape",
    "native_shape_from_pipe3d",
    "snr_loss_mask",
    "write_amara_phys_maps",
    "write_collaborator_phys_maps",
]
