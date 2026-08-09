from pathlib import Path

import numpy as np
from astropy.table import Table


def kauffmann_2003(log_nii_ha):
    return 0.61 / (log_nii_ha - 0.05) + 1.30


def kewley_2001(log_nii_ha):
    return 0.61 / (log_nii_ha - 0.47) + 1.19


def _to_plain_value(value):
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    return value


def _column(table, name, default=np.nan):
    lower_lookup = {col.lower(): col for col in table.colnames}
    actual = lower_lookup.get(name.lower())
    if actual is None:
        return np.full(len(table), default, dtype=float)
    return np.asarray(table[actual], dtype=float)


def _text_column(table, name):
    lower_lookup = {col.lower(): col for col in table.colnames}
    actual = lower_lookup.get(name.lower())
    if actual is None:
        return np.array([""] * len(table), dtype=object)
    return np.array([str(_to_plain_value(value)).strip() for value in table[actual]], dtype=object)


def discover_local_plateifus(data_root):
    data_root = Path(data_root)
    return {
        path.name.replace("manga-", "").split(".Pipe3D")[0]
        for path in data_root.glob("**/manga-*.Pipe3D.cube.fits*")
    }


def make_global_flag_rows(
    pipe3d_catalog_path,
    plateifu_filter=None,
    max_ratio_err=0.3,
    min_ha_ew_emission=3.0,
    min_ha_ew_snr=3.0,
):
    table = Table.read(pipe3d_catalog_path, hdu=1)
    plateifu = _text_column(table, "plateifu")
    name = _text_column(table, "name")

    log_nii_ha = _column(table, "log_NII_Ha_ALL")
    log_oiii_hbeta = _column(table, "log_OIII_Hb_ALL")
    e_log_nii_ha = _column(table, "e_log_NII_Ha_ALL")
    e_log_oiii_hbeta = _column(table, "e_log_OIII_Hb_ALL")
    ew_ha = _column(table, "EW_Ha_ALL")
    e_ew_ha = _column(table, "e_EW_Ha_ALL")
    ha_hb = _column(table, "Ha_Hb_ALL")
    e_ha_hb = _column(table, "e_Ha_Hb_ALL")
    log_sfr_ha = _column(table, "log_SFR_Ha")
    e_log_sfr_ha = _column(table, "e_log_SFR_Ha")
    log_mass = _column(table, "log_Mass")
    e_log_mass = _column(table, "e_log_Mass")
    nsa_mstar = _column(table, "nsa_mstar")

    valid_bpt = np.isfinite(log_nii_ha) & np.isfinite(log_oiii_hbeta)
    y_kauffmann = kauffmann_2003(log_nii_ha)
    y_kewley = kewley_2001(log_nii_ha)
    is_sf = valid_bpt & (log_oiii_hbeta < y_kauffmann)
    is_comp = valid_bpt & (log_oiii_hbeta >= y_kauffmann) & (log_oiii_hbeta < y_kewley)
    is_agn = valid_bpt & (log_oiii_hbeta >= y_kewley)

    bpt_class_code = np.zeros(len(table), dtype=int)
    bpt_class_code[is_sf] = 1
    bpt_class_code[is_comp] = 2
    bpt_class_code[is_agn] = 3
    bpt_class = np.full(len(table), "UNCLASSIFIED", dtype=object)
    bpt_class[is_sf] = "SF"
    bpt_class[is_comp] = "COMP"
    bpt_class[is_agn] = "AGN"

    ha_ew_emission = -ew_ha
    ha_ew_emission_snr = np.full(len(table), np.nan, dtype=float)
    good_ew_err = np.isfinite(ha_ew_emission) & np.isfinite(e_ew_ha) & (e_ew_ha > 0)
    ha_ew_emission_snr[good_ew_err] = ha_ew_emission[good_ew_err] / e_ew_ha[good_ew_err]

    ratio_errors_good = (
        np.isfinite(e_log_nii_ha)
        & np.isfinite(e_log_oiii_hbeta)
        & (e_log_nii_ha <= max_ratio_err)
        & (e_log_oiii_hbeta <= max_ratio_err)
    )
    ha_ew_good = (
        np.isfinite(ha_ew_emission)
        & (ha_ew_emission >= min_ha_ew_emission)
        & np.isfinite(ha_ew_emission_snr)
        & (ha_ew_emission_snr >= min_ha_ew_snr)
    )
    global_bpt_sf_strict = is_sf & ratio_errors_good
    global_sf_ew_strict = global_bpt_sf_strict & ha_ew_good

    plateifu_filter = set(plateifu_filter or [])
    rows = []
    for i in range(len(table)):
        if plateifu_filter and plateifu[i] not in plateifu_filter:
            continue
        rows.append(
            {
                "plateifu": plateifu[i],
                "name": name[i],
                "global_bpt_valid": bool(valid_bpt[i]),
                "global_bpt_class": bpt_class[i],
                "global_bpt_class_code": int(bpt_class_code[i]),
                "global_bpt_sf": bool(is_sf[i]),
                "global_bpt_comp": bool(is_comp[i]),
                "global_bpt_agn": bool(is_agn[i]),
                "global_bpt_sf_strict": bool(global_bpt_sf_strict[i]),
                "global_sf_ew_strict": bool(global_sf_ew_strict[i]),
                "strict_max_ratio_err": float(max_ratio_err),
                "strict_min_ha_ew_emission": float(min_ha_ew_emission),
                "strict_min_ha_ew_snr": float(min_ha_ew_snr),
                "log_NII_Ha_ALL": log_nii_ha[i],
                "e_log_NII_Ha_ALL": e_log_nii_ha[i],
                "log_OIII_Hb_ALL": log_oiii_hbeta[i],
                "e_log_OIII_Hb_ALL": e_log_oiii_hbeta[i],
                "EW_Ha_ALL": ew_ha[i],
                "e_EW_Ha_ALL": e_ew_ha[i],
                "ha_ew_emission_ALL": ha_ew_emission[i],
                "ha_ew_emission_snr_ALL": ha_ew_emission_snr[i],
                "Ha_Hb_ALL": ha_hb[i],
                "e_Ha_Hb_ALL": e_ha_hb[i],
                "log_SFR_Ha": log_sfr_ha[i],
                "e_log_SFR_Ha": e_log_sfr_ha[i],
                "log_Mass": log_mass[i],
                "e_log_Mass": e_log_mass[i],
                "nsa_mstar": nsa_mstar[i],
            }
        )
    return rows
