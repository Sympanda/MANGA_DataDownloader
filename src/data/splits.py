from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

import numpy as np

SplitName = Literal["train", "val", "test"]


def read_split_csv(path: Path | str) -> dict[SplitName, set[str]]:
    """Load plateifu -> split assignments from CSV (columns: plateifu, split)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Split CSV not found: {path}")

    splits: dict[SplitName, set[str]] = {"train": set(), "val": set(), "test": set()}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if "plateifu" not in (reader.fieldnames or []) or "split" not in (reader.fieldnames or []):
            raise ValueError(f"{path} must have columns: plateifu, split")
        for row in reader:
            plateifu = row["plateifu"].strip()
            split = row["split"].strip().lower()
            if split not in splits:
                raise ValueError(f"Unknown split {split!r} for {plateifu}")
            splits[split].add(plateifu)
    return splits


def write_split_csv(
    path: Path | str,
    assignments: dict[str, SplitName],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["plateifu", "galaxy_dir", "split"])
        writer.writeheader()
        for plateifu in sorted(assignments):
            galaxy_dir = plateifu.replace("-", "_")
            writer.writerow(
                {
                    "plateifu": plateifu,
                    "galaxy_dir": galaxy_dir,
                    "split": assignments[plateifu],
                }
            )


def make_random_splits(
    plateifus: list[str],
    *,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> dict[str, SplitName]:
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")

    rng = np.random.default_rng(seed)
    ids = list(plateifus)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_train = min(n_train, n - 2) if n >= 3 else max(1, n - 1)
    n_val = min(n_val, n - n_train - 1) if n - n_train > 1 else 0

    assignments: dict[str, SplitName] = {}
    for i, plateifu in enumerate(ids):
        if i < n_train:
            assignments[plateifu] = "train"
        elif i < n_train + n_val:
            assignments[plateifu] = "val"
        else:
            assignments[plateifu] = "test"
    return assignments


def filter_rows_by_split(
    rows: list[dict],
    split_csv_path: Path | str,
    split: SplitName,
) -> list[dict]:
    splits = read_split_csv(split_csv_path)
    allowed = splits[split]
    return [row for row in rows if row["plateifu"] in allowed]
