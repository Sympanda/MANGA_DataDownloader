"""
Train conditional residual diffusion on top of a frozen base UNet/UNet++.

Usage:
  python runner_residual_diffusion.py --config config_residual_diffusion.jsonc --run-name res_diff_ha --autoinc
"""
from __future__ import annotations

import argparse
import faulthandler
import signal
import sys
from pathlib import Path

from runner import (
    _resolve_run_name,
    build_data_config,
    build_model_config,
    build_train_config,
    set_seed,
)
from src.config_loader import load_jsonc
from src.data.make_dataloader import build_base_dataset, make_manga_dataloaders
from src.models.base_loader import load_frozen_base_map_generator
from src.models.residual_diffusion_wrapper import ResidualDiffusionMapGenerator
from src.training.train import run_training
from runner_residual import _align_data_with_base


def main(argv: list[str] | None = None) -> int:
    faulthandler.enable()
    if hasattr(faulthandler, "register"):
        try:
            faulthandler.register(signal.SIGTERM, file=sys.stderr, all_threads=True)
        except (AttributeError, ValueError, OSError):
            pass

    parser = argparse.ArgumentParser(description="Train residual diffusion models.")
    parser.add_argument("--config", type=Path, default=Path("config_residual_diffusion.jsonc"))
    parser.add_argument("--run-name", type=str, default="res_diff_ha")
    parser.add_argument("--autoinc", action="store_true")
    parser.add_argument("--base-run-dir", type=Path, default=None)
    parser.add_argument("--base-checkpoint", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    user_cfg = load_jsonc(args.config)
    if args.epochs is not None:
        user_cfg.setdefault("training", {})["epochs"] = int(args.epochs)
    if args.device is not None:
        user_cfg.setdefault("training", {})["device"] = str(args.device)

    training_top = user_cfg.get("training", {})
    data_top = user_cfg.get("data", {})
    model_top = user_cfg.get("model", {})
    residual_top = user_cfg.get("residual", {})
    diffusion_top = user_cfg.get("diffusion", {})

    if args.base_run_dir is not None:
        residual_top["base_run_dir"] = str(args.base_run_dir)
    if args.base_checkpoint is not None:
        residual_top["base_checkpoint"] = str(args.base_checkpoint)
    user_cfg["residual"] = residual_top

    base_run_dir = residual_top.get("base_run_dir")
    if not base_run_dir:
        raise SystemExit("residual.base_run_dir is required.")
    base_checkpoint = residual_top.get("base_checkpoint", "best.pt")
    raw_channel = residual_top.get("channel_key", "ha_flux")
    if raw_channel in (None, "", "all", "ALL"):
        channel_key = None
    else:
        channel_key = str(raw_channel)

    base_model, base_cfg, channel_index = load_frozen_base_map_generator(
        base_run_dir=base_run_dir,
        base_checkpoint=base_checkpoint,
        channel_key=channel_key,
    )

    data_top = _align_data_with_base(data_top, base_cfg)
    user_cfg["data"] = data_top
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

    if not data_cfg.split_csv_path.is_file():
        raise SystemExit(f"Split CSV not found: {data_cfg.split_csv_path}")

    save_root = Path(training_top.get("logging", {}).get("root_dir", "runs/residual_diffusion"))
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

    mults = tuple(int(x) for x in diffusion_top.get("channel_mults", [1, 2, 4]))
    model = ResidualDiffusionMapGenerator(
        model_cfg,
        base_model=base_model,
        base_channel_index=channel_index,
        diffusion_steps=int(diffusion_top.get("diffusion_steps", 1000)),
        ddim_steps=int(diffusion_top.get("ddim_steps", 50)),
        n_samples=int(diffusion_top.get("n_samples", 32)),
        base_channels=int(diffusion_top.get("base_channels", 32)),
        channel_mults=mults,
        schedule=str(diffusion_top.get("schedule", "linear")),
        use_footprint_cond=bool(diffusion_top.get("use_footprint_cond", True)),
    )

    print("=" * 60)
    print("Residual diffusion training")
    print("=" * 60)
    print(f"  run          : {run_name}")
    print(f"  base         : {base_run_dir} ({base_checkpoint})")
    print(f"  channel      : {channel_key}")
    print(f"  ddim_steps   : {diffusion_top.get('ddim_steps', 50)}")
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
