"""
Create train/val/test split CSV for MaNGA galaxies.

Usage:
  python -m src.data.make_splits --config config.jsonc --output manga_sdss_fits/splits/default_split.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

from manga_prep.manga_dataset import MangaGalaxyDataset
from src.config_loader import load_jsonc
from src.data.make_dataloader import DataConfig, build_base_dataset
from src.data.splits import make_random_splits, write_split_csv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create MaNGA train/val/test split CSV.")
    parser.add_argument("--config", type=Path, default=Path("config.jsonc"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_jsonc(args.config)
    data_raw = cfg.get("data", {})
    split_raw = data_raw.get("split", {})

    data_cfg = DataConfig(
        data_root=Path(data_raw.get("data_root", "manga_sdss_fits")),
        index_path=Path(data_raw["index_path"]) if data_raw.get("index_path") else None,
        use_sdss=bool(data_raw.get("use_sdss", True)),
        use_legacy=bool(data_raw.get("use_legacy", False)),
        use_spectrum=bool(data_raw.get("use_spectrum", True)),
        spectrum_mode=str(data_raw.get("spectrum_mode", "fake")),
        require_all=bool(data_raw.get("require_all", True)),
    )

    out_path = args.output or Path(
        split_raw.get("split_csv_path", "manga_sdss_fits/splits/default_split.csv")
    )
    seed = args.seed if args.seed is not None else int(split_raw.get("seed", 42))
    train_frac = float(split_raw.get("train_frac", 0.8))
    val_frac = float(split_raw.get("val_frac", 0.1))
    test_frac = float(split_raw.get("test_frac", 0.1))

    base: MangaGalaxyDataset = build_base_dataset(data_cfg)
    plateifus = [row["plateifu"] for row in base.rows]
    assignments = make_random_splits(
        plateifus,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )
    write_split_csv(out_path, assignments)

    n_train = sum(1 for s in assignments.values() if s == "train")
    n_val = sum(1 for s in assignments.values() if s == "val")
    n_test = sum(1 for s in assignments.values() if s == "test")
    print(f"Wrote {len(assignments):,} galaxies -> {out_path}")
    print(f"  train={n_train:,}  val={n_val:,}  test={n_test:,}  (seed={seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
