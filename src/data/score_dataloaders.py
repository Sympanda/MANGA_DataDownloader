"""Shared helpers for score-generator / score-corrector runners."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from manga_prep.dataset.manga_dataset import collate_manga_batch
from src.data.augmentation import AugmentConfig
from src.data.make_dataloader import DataConfig, build_base_dataset
from src.data.manga_split_dataset import MangaSplitDataset
from src.data.score_subset import select_score_plateifus, stratified_sample_weights
from src.models.input_prep import prepare_targets_and_masks
from src.models.map_score import ScoreNormStats


def filter_dataset_plateifus(ds: MangaSplitDataset, plateifus: set[str]) -> None:
    """In-place filter of MangaSplitDataset rows to an allowed plateifu set."""
    ds.rows = [r for r in ds.rows if r["plateifu"] in plateifus]
    if not ds.rows:
        raise ValueError("No galaxies left after score-subset filtering")


def make_score_dataloaders(
    data_cfg: DataConfig,
    batching_cfg: dict[str, Any],
    *,
    coverage_csv: Path | str,
    min_coverage_pct: float = 99.0,
    max_coverage_pct: float | None = None,
    feature: str = "ha_flux",
    use_stratified_weights: bool = True,
    plateifu_allowlist: list[str] | set[str] | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader, list[str]]:
    """
    Build loaders with:
    - train: train-split ∩ coverage band (+ optional stratified weights)
    - val/test: original val/test ∩ coverage band (no reweighting)

    ``plateifu_allowlist`` further restricts all splits (overfit diagnostics).
    """
    base = build_base_dataset(data_cfg)
    split_path = data_cfg.split_csv_path
    allow = None if plateifu_allowlist is None else set(plateifu_allowlist)

    train_ids = select_score_plateifus(
        coverage_csv=coverage_csv,
        split_csv=split_path,
        split="train",
        feature=feature,  # type: ignore[arg-type]
        min_coverage_pct=min_coverage_pct,
        max_coverage_pct=max_coverage_pct,
    )
    val_ids = select_score_plateifus(
        coverage_csv=coverage_csv,
        split_csv=split_path,
        split="val",
        feature=feature,  # type: ignore[arg-type]
        min_coverage_pct=min_coverage_pct,
        max_coverage_pct=max_coverage_pct,
    )
    test_ids = select_score_plateifus(
        coverage_csv=coverage_csv,
        split_csv=split_path,
        split="test",
        feature=feature,  # type: ignore[arg-type]
        min_coverage_pct=min_coverage_pct,
        max_coverage_pct=max_coverage_pct,
    )
    if allow is not None:
        train_ids = [p for p in train_ids if p in allow]
        val_ids = [p for p in val_ids if p in allow]
        test_ids = [p for p in test_ids if p in allow]

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

    filter_dataset_plateifus(ds_train, set(train_ids))
    filter_dataset_plateifus(ds_train_eval, set(train_ids))
    # Train-only allowlists (overfit) do not intersect the val/test splits; mirror
    # the train-eval rows so loaders stay non-empty.
    if val_ids:
        filter_dataset_plateifus(ds_val, set(val_ids))
    else:
        ds_val.rows = list(ds_train_eval.rows)
    if test_ids:
        filter_dataset_plateifus(ds_test, set(test_ids))
    else:
        ds_test.rows = list(ds_train_eval.rows)

    train_bs = int(batching_cfg.get("train_batch_size", 8))
    eval_bs = int(batching_cfg.get("eval_batch_size", 8))
    num_workers = int(batching_cfg.get("num_workers", 0))
    pin_memory = bool(batching_cfg.get("pin_memory", torch.cuda.is_available()))
    loader_kwargs: dict[str, object] = {
        "num_workers": num_workers,
        "collate_fn": collate_manga_batch,
        "pin_memory": pin_memory,
    }

    sampler = None
    shuffle = True
    if use_stratified_weights:
        plateifus = [r["plateifu"] for r in ds_train.rows]
        weights = stratified_sample_weights(plateifus, coverage_csv=coverage_csv)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )
        shuffle = False

    dl_train = DataLoader(
        ds_train, batch_size=train_bs, shuffle=shuffle, sampler=sampler, **loader_kwargs
    )
    dl_val = DataLoader(ds_val, batch_size=eval_bs, shuffle=False, **loader_kwargs)
    dl_test = DataLoader(ds_test, batch_size=eval_bs, shuffle=False, **loader_kwargs)
    dl_train_ns = DataLoader(ds_train_eval, batch_size=eval_bs, shuffle=False, **loader_kwargs)
    return dl_train, dl_val, dl_test, dl_train_ns, train_ids


@torch.no_grad()
def compute_score_norm_stats(
    dataloader: DataLoader,
    model_cfg,
    *,
    max_batches: int | None = None,
) -> ScoreNormStats:
    """Mean/std of scaled targets on **valid label pixels only** (not footprint fill)."""
    total = 0.0
    total_sq = 0.0
    n = 0
    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        targets, label_mask = prepare_targets_and_masks(batch, model_cfg)
        m = label_mask > 0
        vals = targets[m]
        if vals.numel() == 0:
            continue
        total += float(vals.sum().item())
        total_sq += float((vals**2).sum().item())
        n += int(vals.numel())
    if n == 0:
        raise RuntimeError("No labelled pixels found while computing score normalisation")
    mean = total / n
    var = max(total_sq / n - mean * mean, 1e-12)
    return ScoreNormStats(mean=mean, std=float(np.sqrt(var)))
