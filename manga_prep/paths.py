"""Shared path defaults for manga_prep scripts (relative to repo cwd)."""
from __future__ import annotations

from pathlib import Path

# Per-galaxy tree: manga_sdss_fits/<plate>_<ifu>/
DEFAULT_DATA_ROOT = Path("manga_sdss_fits")

# Shared caches written beside the repo (not inside per-galaxy folders)
DEFAULT_SDSS_SPPLATE_CACHE = Path("sdss_spplate_cache")
DEFAULT_LEGACY_BRICK_CACHE = Path("legacy_coadd_brick_cache")
DEFAULT_SPLITS_DIR = DEFAULT_DATA_ROOT / "splits"
