from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset

from manga_prep.dataset.manga_dataset import MangaGalaxyDataset, collate_manga_batch
from src.data.augmentation import AugmentConfig, augment_spatial_sample
from src.data.splits import filter_rows_by_split

SplitName = Literal["train", "val", "test"]


class MangaSplitDataset(Dataset):
    """MangaGalaxyDataset filtered by split CSV with optional spatial augmentation."""

    def __init__(
        self,
        base: MangaGalaxyDataset,
        *,
        split: SplitName,
        split_csv_path: Path | str,
        augment: AugmentConfig | None = None,
    ) -> None:
        self.base = base
        self.split = split
        self.augment = augment or AugmentConfig(enabled=False)
        self.rows = filter_rows_by_split(base.rows, split_csv_path, split)
        if not self.rows:
            raise ValueError(f"No galaxies for split={split!r} in {split_csv_path}")

        self._row_to_base_index = {row["plateifu"]: i for i, row in enumerate(base.rows)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        base_index = self._row_to_base_index[row["plateifu"]]
        sample = self.base[base_index]
        sample["split"] = self.split

        if not self.augment.enabled:
            return sample

        sdss = None
        legacy = None
        hr = None
        inputs = sample.get("inputs", {})
        if "sdss_imaging" in inputs:
            sdss = torch.from_numpy(inputs["sdss_imaging"]["data"].copy())
        if "legacy_imaging" in inputs:
            legacy = torch.from_numpy(inputs["legacy_imaging"]["data"].copy())
        if "hr_imaging" in inputs:
            hr = torch.from_numpy(inputs["hr_imaging"]["data"].copy())

        footprint = None
        targets_t = None
        masks_t = None
        if "footprint_mask" in sample:
            footprint = torch.from_numpy(sample["footprint_mask"].copy())
        if "targets" in sample:
            targets_t = {k: torch.from_numpy(v.copy()) for k, v in sample["targets"].items()}
        if "target_loss_masks" in sample:
            masks_t = {k: torch.from_numpy(v.copy()) for k, v in sample["target_loss_masks"].items()}

        sdss, legacy, hr, footprint, targets_t, masks_t = augment_spatial_sample(
            sdss=sdss,
            legacy=legacy,
            hr=hr,
            footprint=footprint,
            targets=targets_t,
            target_masks=masks_t,
            cfg=self.augment,
        )

        if sdss is not None and "inputs" in sample:
            sample["inputs"]["sdss_imaging"]["data"] = sdss.numpy()
        if legacy is not None and "inputs" in sample:
            sample["inputs"]["legacy_imaging"]["data"] = legacy.numpy()
        if hr is not None and "inputs" in sample:
            sample["inputs"]["hr_imaging"]["data"] = hr.numpy()
        if footprint is not None:
            sample["footprint_mask"] = footprint.numpy()
        if targets_t is not None:
            sample["targets"] = {k: v.numpy() for k, v in targets_t.items()}
        if masks_t is not None:
            sample["target_loss_masks"] = {k: v.numpy() for k, v in masks_t.items()}
        return sample


__all__ = ["MangaSplitDataset", "collate_manga_batch"]
