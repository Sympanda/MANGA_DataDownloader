"""
Train residual models on top of a frozen base UNet/UNet++.

Usage:
  python runner_residual.py --config config_residual.jsonc --variant pixel --run-name res_pix_ha --autoinc
  python runner_residual.py --config config_residual.jsonc --variant local_cnn --run-name res_cnn_ha --autoinc
  python runner_residual.py --config config_residual.jsonc --variant gaussian --run-name res_gauss_ha --autoinc
  python runner_residual.py --config config_residual_all.jsonc --run-name res_pix_all --eval-only
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
from src.metrics.plots import write_metrics_csv
from src.metrics.residual_plots import evaluate_batch_forward_predictions
from src.models.base_loader import load_frozen_base_map_generator
from src.models.residual_wrapper import ResidualMapGenerator, assert_base_frozen
from src.training.train import _load_checkpoint_state, run_training


def _align_data_with_base(data_top: dict, base_cfg) -> dict:
    """Ensure the residual dataloader supplies modalities the frozen base needs."""
    out = dict(data_top)
    out["use_sdss"] = bool(base_cfg.use_sdss or data_top.get("use_sdss", True))
    out["use_legacy"] = bool(base_cfg.use_legacy or data_top.get("use_legacy", False))
    out["use_spectrum"] = bool(base_cfg.use_spectrum)
    out["use_footprint_mask"] = True
    out["imaging_resolution"] = "aligned"
    if base_cfg.use_hr_cross_attn:
        out.setdefault("include_hr_note", True)
    return out


def _resolve_channel_key(raw_channel) -> str | None:
    if raw_channel in (None, "", "all", "ALL"):
        return None
    return str(raw_channel)


def _build_residual_stack(user_cfg: dict):
    data_top = dict(user_cfg.get("data", {}))
    model_top = dict(user_cfg.get("model", {}))
    residual_top = dict(user_cfg.get("residual", {}))

    base_run_dir = residual_top.get("base_run_dir")
    if not base_run_dir:
        raise SystemExit("residual.base_run_dir is required (path to trained UNet run).")
    base_checkpoint = residual_top.get("base_checkpoint", "best.pt")
    channel_key = _resolve_channel_key(residual_top.get("channel_key", "ha_flux"))
    variant = str(residual_top.get("variant", "pixel"))
    hidden = int(residual_top.get("hidden_channels", 32))

    base_model, base_cfg, channel_index = load_frozen_base_map_generator(
        base_run_dir=base_run_dir,
        base_checkpoint=base_checkpoint,
        channel_key=channel_key,
    )

    data_top = _align_data_with_base(data_top, base_cfg)
    if "target_keys" not in model_top or not model_top.get("target_keys"):
        if channel_key is None:
            model_top["target_keys"] = list(base_cfg.target_keys)
        else:
            model_top["target_keys"] = [channel_key]
    imaging_resolution = "aligned"
    model_cfg = build_model_config(model_top, data_top, imaging_resolution=imaging_resolution)
    model_cfg.use_spectrum = False
    model_cfg.footprint_mode = "loss_only"
    model_cfg.use_hr_cross_attn = False
    if base_cfg.imaging_asinh_scales is not None:
        model_cfg.input_norm_mode = base_cfg.input_norm_mode
        model_cfg.imaging_asinh_scales = list(base_cfg.imaging_asinh_scales)
        model_cfg.imaging_clamp_min = base_cfg.imaging_clamp_min
        model_cfg.imaging_clamp_max = base_cfg.imaging_clamp_max
        model_cfg.input_norm_scales_path = base_cfg.input_norm_scales_path
        model_cfg.input_norm_imaging_percentile = base_cfg.input_norm_imaging_percentile

    model_top_for_data = dict(model_top)
    if base_cfg.use_hr_cross_attn:
        model_top_for_data["use_hr_cross_attention"] = True
        model_top_for_data["hr_survey"] = base_cfg.hr_survey
    data_cfg = build_data_config(
        data_top, imaging_resolution=imaging_resolution, model_top=model_top_for_data
    )
    data_cfg.use_spectrum = bool(base_cfg.use_spectrum)
    data_cfg.spectrum_mode = str(data_top.get("spectrum_mode", "fake"))

    model = ResidualMapGenerator(
        model_cfg,
        base_model=base_model,
        base_channel_index=channel_index,
        variant=variant,  # type: ignore[arg-type]
        hidden_channels=hidden,
        n_residual_samples=int(residual_top.get("n_residual_samples", 32)),
        gaussian_l1_weight=float(residual_top.get("gaussian_l1_weight", 0.1)),
    )
    assert_base_frozen(model)
    meta = {
        "base_run_dir": base_run_dir,
        "base_checkpoint": base_checkpoint,
        "channel_key": channel_key,
        "channel_index": channel_index,
        "variant": variant,
        "data_top": data_top,
        "model_top": model_top,
        "residual_top": residual_top,
    }
    return model, model_cfg, data_cfg, meta


def _run_eval_only(args: argparse.Namespace) -> int:
    training_top = load_jsonc(args.config).get("training", {})
    save_root = Path(training_top.get("logging", {}).get("root_dir", "runs/residual"))
    run_dir = save_root / args.run_name
    snap_path = run_dir / "config_used.json"
    if not snap_path.is_file():
        raise SystemExit(f"Run config not found: {snap_path}")

    snap = json.loads(snap_path.read_text(encoding="utf-8"))
    user_cfg = snap.get("user") or {}
    train_cfg = build_train_config(user_cfg.get("training", training_top), run_name=args.run_name)

    ckpt_path = args.checkpoint or (run_dir / "ckpts" / "best.pt")
    if not Path(ckpt_path).is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    set_seed(train_cfg.seed)
    model, model_cfg, data_cfg, meta = _build_residual_stack(user_cfg)
    _, dl_val, dl_test, dl_train_ns = make_manga_dataloaders(
        data_cfg,
        {
            "train_batch_size": train_cfg.train_batch_size,
            "eval_batch_size": train_cfg.eval_batch_size,
            "num_workers": user_cfg.get("training", training_top)
            .get("batching", {})
            .get("num_workers", 0),
        },
    )
    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(_load_checkpoint_state(ckpt))
    model.eval()

    print(f"Residual eval-only: {run_dir}")
    print(f"  checkpoint : {ckpt_path}")
    print(f"  variant    : {meta['variant']}")
    split_loaders = {"train": dl_train_ns, "val": dl_val, "test": dl_test}
    for split in train_cfg.eval_splits:
        if split not in split_loaders:
            continue
        rows = evaluate_batch_forward_predictions(
            model,
            split_loaders[split],
            device=device,
            map_keys=tuple(model_cfg.target_keys),
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

    parser = argparse.ArgumentParser(description="Train frozen-base residual models.")
    parser.add_argument("--config", type=Path, default=Path("config_residual.jsonc"))
    parser.add_argument("--run-name", type=str, default="res_ha")
    parser.add_argument("--autoinc", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        choices=["pixel", "local_cnn", "gaussian"],
    )
    parser.add_argument("--base-run-dir", type=Path, default=None)
    parser.add_argument("--base-checkpoint", type=str, default=None)
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
    residual_top = user_cfg.get("residual", {})

    if args.variant is not None:
        residual_top["variant"] = args.variant
    if args.base_run_dir is not None:
        residual_top["base_run_dir"] = str(args.base_run_dir)
    if args.base_checkpoint is not None:
        residual_top["base_checkpoint"] = str(args.base_checkpoint)
    user_cfg["residual"] = residual_top

    model, model_cfg, data_cfg, meta = _build_residual_stack(user_cfg)
    user_cfg["data"] = meta["data_top"]
    user_cfg["model"] = meta["model_top"]

    if not data_cfg.split_csv_path.is_file():
        raise SystemExit(f"Split CSV not found: {data_cfg.split_csv_path}")

    save_root = Path(training_top.get("logging", {}).get("root_dir", "runs/residual"))
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

    print("=" * 60)
    print("Frozen-base residual training")
    print("=" * 60)
    print(f"  run          : {run_name}")
    print(f"  variant      : {meta['variant']}")
    print(f"  base         : {meta['base_run_dir']} ({meta['base_checkpoint']})")
    print(f"  channel      : {meta['channel_key'] or 'all'} (idx={meta['channel_index']})")
    print(f"  targets      : {model_cfg.target_keys}")
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
