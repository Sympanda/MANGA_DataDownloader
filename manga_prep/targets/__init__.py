"""Pipe3D / Amara map target definitions and loaders."""

from manga_prep.targets.pipe3d_maps import (
    AMARA_TARGET_KEYS,
    DEFAULT_TARGET_SIZE,
    PIPE3D_MAP_SPECS,
    load_amara_maps,
    load_amara_training_targets,
)

__all__ = [
    "AMARA_TARGET_KEYS",
    "DEFAULT_TARGET_SIZE",
    "PIPE3D_MAP_SPECS",
    "load_amara_maps",
    "load_amara_training_targets",
]
