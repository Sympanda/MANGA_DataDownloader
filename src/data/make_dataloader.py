from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader

from manga_prep.dataset.manga_dataset import (
    GalaxySFFlag,
    MangaGalaxyDataset,
    TargetSource,
    collate_manga_batch,
)
from manga_prep.io.aligned_cache import ImagingGrid
from src.data.augmentation import AugmentConfig
from src.data.manga_split_dataset import MangaSplitDataset
from src.models.config import ImagingResolution


@dataclass
class DataConfig:
    data_root: Path = Path("manga_sdss_fits")
    index_path: Path | None = None
    split_csv_path: Path = Path("manga_sdss_fits/splits/default_split.csv")

    use_sdss: bool = True
    use_legacy: bool = False
    use_spectrum: bool = True
    spectrum_mode: str = "fake"  # "fake" | "real"
    spectrum_fallback: bool = True
    use_footprint_mask: bool = True

    # Target maps: legacy emission-line ("amara") or physical-property ("phys").
    target_source: TargetSource = "amara"
    target_keys: tuple[str, ...] | None = None
    min_snr: float | None = None
    galaxy_sf_flag: GalaxySFFlag | None = None
    require_sf_spaxel: bool = False
    # Coverage-aware selection (fill first so small IFUs are not punished).
    min_footprint_fill: float | None = None
    min_valid_pixels: float | None = None
    include_redshift: bool = False
    require_redshift: bool = False

    # "aligned" → Amara WCS 76×76. "native" → SDSS plate scale @ 196, Amara-oriented.
    imaging_resolution: ImagingResolution = "aligned"
    # Optional Amara-grid oversample (only when imaging_grid resolves to "amara").
    aligned_oversample: int | None = None
    # Explicit override; None → derived from imaging_resolution.
    imaging_grid: ImagingGrid | None = None

    # Side-stream high-res morphology for cross-attention (backbone stays Amara 76×76).
    include_hr_imaging: bool = False
    hr_survey: Literal["sdss", "legacy"] = "sdss"

    # Always True for training: survey cutouts are WCS-reprojected before the model.
    align_imaging_to_amara_grid: bool = True
    prefer_aligned_cache: bool = True
    require_all: bool = True

    augmentation: AugmentConfig = field(default_factory=AugmentConfig)

    def resolve_index_path(self) -> Path:
        if self.index_path is not None:
            return self.index_path
        return self.data_root / "manga_dataset_index.csv"

    def resolve_imaging_grid(self) -> ImagingGrid:
        if self.include_hr_imaging:
            # Backbone must stay on the Amara 76×76 grid when HR is a side stream.
            return "amara"
        if self.imaging_grid is not None:
            return self.imaging_grid
        if self.imaging_resolution == "native":
            return "sdss_native"
        return "amara"

    def resolve_aligned_oversample(self) -> int:
        if self.resolve_imaging_grid() == "sdss_native":
            return 1
        if self.aligned_oversample is not None:
            return int(self.aligned_oversample)
        return 1


def build_base_dataset(cfg: DataConfig) -> MangaGalaxyDataset:
    spectrum = cfg.spectrum_mode if cfg.use_spectrum else None
    return MangaGalaxyDataset(
        cfg.data_root,
        cfg.resolve_index_path(),
        include_sdss_imaging=cfg.use_sdss,
        include_legacy_imaging=cfg.use_legacy,
        include_targets=True,
        spectrum=spectrum,
        spectrum_fallback=cfg.spectrum_fallback,
        require_all=cfg.require_all,
        target_source=cfg.target_source,
        target_keys=cfg.target_keys,
        min_snr=cfg.min_snr,
        galaxy_sf_flag=cfg.galaxy_sf_flag,
        require_sf_spaxel=cfg.require_sf_spaxel,
        min_footprint_fill=cfg.min_footprint_fill,
        min_valid_pixels=cfg.min_valid_pixels,
        include_redshift=cfg.include_redshift,
        require_redshift=cfg.require_redshift,
        align_imaging_to_amara_grid=True,
        prefer_aligned_cache=bool(cfg.prefer_aligned_cache),
        imaging_grid=cfg.resolve_imaging_grid(),
        aligned_oversample=cfg.resolve_aligned_oversample(),
        write_aligned_cache=True,
        include_hr_imaging=cfg.include_hr_imaging,
        hr_survey=cfg.hr_survey,
    )


def make_manga_dataloaders(
    data_cfg: DataConfig,
    batching_cfg: dict[str, Any],
    *,
    base: MangaGalaxyDataset | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Returns (dl_train, dl_val, dl_test, dl_train_no_shuffle).

    Pass ``base`` to reuse an already-built (and coverage-filtered) dataset so
    the pre-check does not run twice.
    """
    if base is None:
        base = build_base_dataset(data_cfg)
    split_path = data_cfg.split_csv_path

    train_aug = AugmentConfig(
        enabled=data_cfg.augmentation.enabled,
        hflip=data_cfg.augmentation.hflip,
        vflip=data_cfg.augmentation.vflip,
        rot90=data_cfg.augmentation.rot90,
        p=data_cfg.augmentation.p,
    )
    no_aug = AugmentConfig(enabled=False)

    ds_train = MangaSplitDataset(base, split="train", split_csv_path=split_path, augment=train_aug)
    ds_val = MangaSplitDataset(base, split="val", split_csv_path=split_path, augment=no_aug)
    ds_test = MangaSplitDataset(base, split="test", split_csv_path=split_path, augment=no_aug)
    ds_train_eval = MangaSplitDataset(base, split="train", split_csv_path=split_path, augment=no_aug)

    train_bs = int(batching_cfg.get("train_batch_size", 8))
    eval_bs = int(batching_cfg.get("eval_batch_size", 16))
    num_workers = int(batching_cfg.get("num_workers", 0))
    pin_memory = bool(batching_cfg.get("pin_memory", torch.cuda.is_available()))

    loader_kwargs: dict[str, object] = {
        "num_workers": num_workers,
        "collate_fn": collate_manga_batch,
        "pin_memory": pin_memory,
    }
    # persistent_workers on Windows can silently kill the process between epochs
    if num_workers > 0 and sys.platform != "win32":
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    dl_train = DataLoader(ds_train, batch_size=train_bs, shuffle=True, **loader_kwargs)
    dl_val = DataLoader(ds_val, batch_size=eval_bs, shuffle=False, **loader_kwargs)
    dl_test = DataLoader(ds_test, batch_size=eval_bs, shuffle=False, **loader_kwargs)
    dl_train_no_shuffle = DataLoader(ds_train_eval, batch_size=eval_bs, shuffle=False, **loader_kwargs)
    return dl_train, dl_val, dl_test, dl_train_no_shuffle
