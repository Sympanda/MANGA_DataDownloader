from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data.splits import SplitName, read_split_csv, write_split_csv


def pool_train_val_fractions(base_split_csv: Path | str) -> tuple[float, float]:
    """Return train/val fractions relative to the non-test pool."""
    splits = read_split_csv(base_split_csv)
    n_train = len(splits["train"])
    n_val = len(splits["val"])
    denom = n_train + n_val
    if denom <= 0:
        return 8.0 / 9.0, 1.0 / 9.0
    return n_train / denom, n_val / denom


def make_member_split_assignments(
    base_split_csv: Path | str,
    *,
    member_seed: int,
    train_frac_pool: float | None = None,
    val_frac_pool: float | None = None,
) -> dict[str, SplitName]:
    """
    Keep test galaxies fixed from ``base_split_csv``; resplit train+val pool.

    Each ensemble member gets a different train/val partition of the non-test galaxies.
    """
    splits = read_split_csv(base_split_csv)
    test_ids = sorted(splits["test"])
    pool = sorted(splits["train"] | splits["val"])
    if train_frac_pool is None or val_frac_pool is None:
        train_frac_pool, val_frac_pool = pool_train_val_fractions(base_split_csv)

    rng = np.random.default_rng(int(member_seed))
    ids = list(pool)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(round(n * train_frac_pool))
    if n > 1:
        n_train = min(max(n_train, 1), n - 1)
    else:
        n_train = n

    assignments: dict[str, SplitName] = {}
    for plateifu in test_ids:
        assignments[plateifu] = "test"
    for i, plateifu in enumerate(ids):
        assignments[plateifu] = "train" if i < n_train else "val"
    return assignments


def write_member_split_csv(
    base_split_csv: Path | str,
    out_csv: Path | str,
    *,
    member_seed: int,
) -> dict[str, SplitName]:
    assignments = make_member_split_assignments(base_split_csv, member_seed=member_seed)
    write_split_csv(out_csv, assignments)
    return assignments


def write_ensemble_manifest(
    path: Path | str,
    *,
    ensemble_name: str,
    n_members: int,
    base_split_csv: str,
    member_seeds: list[int],
    config_path: str,
    user_snapshot: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ensemble_name": ensemble_name,
        "n_members": int(n_members),
        "base_split_csv": base_split_csv,
        "member_seeds": member_seeds,
        "config_path": config_path,
        "user_snapshot": user_snapshot or {},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_ensemble_manifest(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
