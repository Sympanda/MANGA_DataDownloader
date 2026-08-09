"""Pipe3D / Amara map target definitions and loaders."""

from manga_prep.targets.pipe3d_maps import (
    AMARA_TARGET_KEYS,
    DEFAULT_TARGET_SIZE,
    PIPE3D_MAP_SPECS,
    load_amara_maps,
    load_amara_training_targets,
)
from manga_prep.targets.pipe3d_phys_maps import (
    AMARA_PHYS_DERIVED_KEYS,
    AMARA_PHYS_DIRECT_KEYS,
    load_amara_phys_maps,
    load_amara_phys_training_targets,
)

__all__ = [
    "AMARA_TARGET_KEYS",
    "AMARA_PHYS_DIRECT_KEYS",
    "AMARA_PHYS_DERIVED_KEYS",
    "DEFAULT_TARGET_SIZE",
    "PIPE3D_MAP_SPECS",
    "load_amara_maps",
    "load_amara_training_targets",
    "load_amara_phys_maps",
    "load_amara_phys_training_targets",
]
