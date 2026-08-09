"""Galaxy redshift helpers for FiLM conditioning."""
from __future__ import annotations

import json
from pathlib import Path

from manga_prep.targets.pipe3d_phys_maps import AMARA_PHYS_MAPS_META


def load_galaxy_redshift(galaxy_dir: Path | str) -> float | None:
    """
    Read redshift from ``amara_phys_maps_metadata.json`` (``derived_science.redshift``).

    Returns None when missing / non-positive / non-finite.
    """
    path = Path(galaxy_dir) / AMARA_PHYS_MAPS_META
    if not path.is_file():
        return None
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    ds = meta.get("derived_science") or {}
    z = ds.get("redshift")
    if z is None:
        return None
    try:
        z_f = float(z)
    except (TypeError, ValueError):
        return None
    if not (z_f > 0.0) or z_f != z_f:  # NaN check
        return None
    return z_f
