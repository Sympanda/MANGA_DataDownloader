"""
Train pixel-SED photometric baselines (Hα or configured target_keys).

Usage:
  python runner_pixel.py --config config_pixel.jsonc --variant linear --run-name pix_linear_ha --autoinc
  python runner_pixel.py --config config_pixel.jsonc --variant mlp --run-name pix_mlp_ha --autoinc
  python runner_pixel.py --config config_pixel.jsonc --eval-only --run-name pix_mlp_ha
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import signal
import sys
from pathlib import Path

import numpy as np
import torch

from runner import (
    _resolve_run_name,
    build_data_config,
    build_model_config,
    build_train_config,
    set_seed,
)
from src.config_loader import load_jsonc
from src.data.make_dataloader import build_base_dataset, make_manga_dataloaders
from src.metrics.residual_plots import evaluate_batch_forward_predictions
from src.metrics.plots import write_metrics_csv
from src.models.pixel_wrapper import PixelMapGenerator
from src.training.train import _load_checkpoint_state, run_training


def _run_eval_only(args: argparse.Namespace) -> int:
    training_top = load_jsonc(args.config).get("training", {})
    save_root = Path(training_top.get("logging", {}).get("root_dir", "runs/pixel_sed"))
    run_dir = save_root / args.run_name
    snap_path = run_dir / "config_used.json"
    if not snap_path.is_file():
        raise SystemExit(f"Run config not found: {snap_path}")

    snap = json.loads(snap_path.read_text(encoding="utf-8"))
    user_cfg = snap.get("user") or {}
    data_top = user_cfg.get("data", {})
    model_top = user_cfg.get("model", {})
    imaging_resolution = model_top.get("imaging_resolution", data_top.get("imaging_resolution", "aligned"))
    model_cfg = build_model_config(model_top, data_top, imaging_resolution=imaging_resolution)
    # Pixel models ignore footprint/spectrum in the network; keep loss masks available.
    model_cfg.use_spectrum = False
    data_cfg = build_data_config(data_top, imaging_resolution=imaging_resolution, model_top=model_top)
    train_cfg = build_train_config(user_cfg.get("training", training_top), run_name=args.run_name)

    ckpt_path = args.checkpoint or (run_dir / "ckpts" / "best.pt")
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    variant = str(model_top.get("variant", args.variant or "mlp"))
    hidden = int(model_top.get("hidden_channels", 32))
    set_seed(train_cfg.seed)
    _, dl_val, dl_test, dl_train_ns = make_manga_dataloaders(
        data_cfg,
        {
            "train_batch_size": train_cfg.train_batch_size,
            "eval_batch_size": train_cfg.eval_batch_size,
            "num_workers": user_cfg.get("training", training_top).get("batching", {}).get("num_workers", 0),
        },
    )
    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")
    model = PixelMapGenerator(model_cfg, variant=variant, hidden_channels=hidden).to(device)  # type: ignore[arg-type]
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(_load_checkpoint_state(ckpt))
    model.eval()

    split_loaders = {"train": dl_train_ns, "val": dl_val, "test": dl_test}
    for split in train_cfg.eval_splits:
        if split not in split_loaders:
            continue
        rows = evaluate_batch_forward_predictions(
            model,
            split_loaders[split],
            device=device,
            map_keys=tuple(model.config.target_keys),
            plots_dir=run_dir / "plots",
            split=split,
            max_plot=train_cfg.eval_max_plot,
        )
        csv_path = run_dir / "csv" / f"{split}_metrics.csv"
        write_metrics_csv(rows, csv_path)
        mse_vals = [float(r["mse_all"]) for r in rows if np.isfinite(float(r["mse_all"]))]
        print(f"  {split} mean mse_all={float(np.mean(mse_vals)):.6f} -> {csv_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    faulthandler.enable()
    if hasattr(faulthandler, "register"):
        try:
            faulthandler.register(signal.SIGTERM, file=sys.stderr, all_threads=True)
        except (AttributeError, ValueError, OSError):
            pass

    parser = argparse.ArgumentParser(description="Train pixel-SED photometric baselines.")
    parser.add_argument("--config", type=Path, default=Path("config_pixel.jsonc"))
    parser.add_argument("--run-name", type=str, default="pix_ha")
    parser.add_argument("--autoinc", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--variant", type=str, default=None, choices=["linear", "mlp"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    if args.eval_only:
        return _run_eval_only(args)

    user_cfg = load_jsonc(args.config)
    if args.epochs is not None:
        user_cfg.setdefault("training", {})["epochs"] = int(args.epochs)
    if args.device is not None:
        user_cfg.setdefault("training", {})["device"] = str(args.device)

    training_top = user_cfg.get("training", {})
    data_top = user_cfg.get("data", {})
    model_top = user_cfg.get("model", {})
    if args.variant is not None:
        model_top["variant"] = args.variant
        user_cfg["model"] = model_top

    imaging_resolution = model_top.get("imaging_resolution", data_top.get("imaging_resolution", "aligned"))
    if imaging_resolution != "aligned":
        raise SystemExit("Pixel SED baselines require imaging_resolution='aligned' (76×76).")

    model_cfg = build_model_config(model_top, data_top, imaging_resolution=imaging_resolution)
    model_cfg.use_spectrum = False
    model_cfg.footprint_mode = "loss_only"
    data_cfg = build_data_config(data_top, imaging_resolution=imaging_resolution, model_top=model_top)
    data_cfg.use_spectrum = False

    if not data_cfg.split_csv_path.is_file():
        raise SystemExit(f"Split CSV not found: {data_cfg.split_csv_path}")

    save_root = Path(training_top.get("logging", {}).get("root_dir", "runs/pixel_sed"))
    run_name = _resolve_run_name(save_root, args.run_name, args.autoinc)
    train_cfg = build_train_config(training_top, run_name=run_name)
    set_seed(train_cfg.seed)

    _ = build_base_dataset(data_cfg)
    dl_train, dl_val, dl_test, dl_train_ns = make_manga_dataloaders(
        data_cfg,
        {
            "train_batch_size": train_cfg.train_batch_size,
            "eval_batch_size": train_cfg.eval_batch_size,
            "num_workers": training_top.get("batching", {}).get("num_workers", 0),
        },
    )

    variant = str(model_top.get("variant", "mlp"))
    hidden = int(model_top.get("hidden_channels", 32))
    model = PixelMapGenerator(model_cfg, variant=variant, hidden_channels=hidden)  # type: ignore[arg-type]

    print("=" * 60)
    print("Pixel-SED photometric baseline")
    print("=" * 60)
    print(f"  run          : {run_name}")
    print(f"  variant      : {variant}")
    print(f"  targets      : {model_cfg.target_keys}")
    print(f"  imaging      : aligned ugriz → {model_cfg.target_spatial_size}²")
    print(f"  train/val/test batches: {len(dl_train)}/{len(dl_val)}/{len(dl_test)}")
    print("=" * 60)

    run_dirs = run_training(
        model,
        train_cfg,
        dl_train,
        dl_val,
        dl_test,
        dl_train_ns,
        user_snapshot=user_cfg,
    )
    print(f"Done. Artifacts in {run_dirs['root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
