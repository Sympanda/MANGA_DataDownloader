"""
Single entry point for MaNGA map-model training + eval.

Usage:
  python runner.py --config config.jsonc --run-name exp_001 --autoinc
  python runner.py --config config.jsonc --run-name exp_001 --eval-only
  python -m src.data.make_splits --config config.jsonc   # create split CSV first
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

from src.config_loader import load_jsonc
from src.data.augmentation import AugmentConfig
from manga_prep.io.aligned_cache import count_aligned_caches
from src.data.make_dataloader import DataConfig, build_base_dataset, make_manga_dataloaders
from src.models.config import ModelConfig
from src.models.wrapper import MapGenerator
from src.metrics.plots import evaluate_map_predictions, write_metrics_csv
from src.training.train import TrainConfig, _load_checkpoint_state, run_training


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_run_name(root: Path, run_name: str, autoinc: bool) -> str:
    if not autoinc:
        return run_name
    candidate = run_name
    n = 2
    while (root / candidate).exists():
        candidate = f"{run_name}_{n}"
        n += 1
    return candidate


def build_data_config(data_top: dict, *, imaging_resolution: str = "aligned") -> DataConfig:
    aug = data_top.get("augmentation", {}) or {}
    split = data_top.get("split", {}) or {}
    resolution = str(data_top.get("imaging_resolution", imaging_resolution))
    oversample_raw = data_top.get("aligned_oversample", None)
    aligned_oversample = None if oversample_raw is None else int(oversample_raw)
    grid_raw = data_top.get("imaging_grid", None)
    imaging_grid = None if grid_raw is None else str(grid_raw)
    return DataConfig(
        data_root=Path(data_top.get("data_root", "manga_sdss_fits")),
        index_path=Path(data_top["index_path"]) if data_top.get("index_path") else None,
        split_csv_path=Path(split.get("split_csv_path", "manga_sdss_fits/splits/default_split.csv")),
        use_sdss=bool(data_top.get("use_sdss", True)),
        use_legacy=bool(data_top.get("use_legacy", False)),
        use_spectrum=bool(data_top.get("use_spectrum", True)),
        spectrum_mode=str(data_top.get("spectrum_mode", "fake")),
        spectrum_fallback=bool(data_top.get("spectrum_fallback", True)),
        use_footprint_mask=bool(data_top.get("use_footprint_mask", True)),
        imaging_resolution=resolution,  # type: ignore[arg-type]
        aligned_oversample=aligned_oversample,
        imaging_grid=imaging_grid,  # type: ignore[arg-type]
        align_imaging_to_amara_grid=True,
        prefer_aligned_cache=bool(data_top.get("prefer_aligned_cache", True)),
        require_all=bool(data_top.get("require_all", True)),
        augmentation=AugmentConfig(
            enabled=bool(aug.get("enabled", True)),
            hflip=bool(aug.get("hflip", True)),
            vflip=bool(aug.get("vflip", True)),
            rot90=bool(aug.get("rot90", True)),
            p=float(aug.get("p", 0.5)),
        ),
    )


def build_model_config(
    model_top: dict,
    data_top: dict,
    *,
    imaging_resolution: str = "aligned",
) -> ModelConfig:
    footprint_mode = model_top.get("footprint_mode", "spatial_channel")
    use_footprint = bool(data_top.get("use_footprint_mask", True))
    if footprint_mode == "loss_only":
        use_footprint_model = False
    else:
        use_footprint_model = use_footprint

    use_sdss = bool(data_top.get("use_sdss", True))
    use_legacy = bool(data_top.get("use_legacy", False))
    use_spectrum = bool(data_top.get("use_spectrum", True))

    norm_top = model_top.get("input_norm", {}) or {}
    input_norm_mode = str(norm_top.get("mode", "none"))
    imaging_asinh_scales = None
    spectrum_asinh_scale_fake = None
    spectrum_asinh_scale_real = None
    imaging_pct = float(norm_top.get("imaging_percentile", 99))
    spectrum_pct = float(norm_top.get("spectrum_percentile", 99))
    scales_path = norm_top.get("scales_path")
    if input_norm_mode == "asinh":
        from manga_prep.io.input_scales import (
            ensure_input_asinh_scales,
            normalize_percentile,
            resolve_runtime_asinh_scales,
        )

        imaging_pct = normalize_percentile(norm_top.get("imaging_percentile", 99))
        spectrum_pct = normalize_percentile(norm_top.get("spectrum_percentile", 99))
        scales_path = ensure_input_asinh_scales(
            data_top=data_top,
            model_top=model_top,
            imaging_resolution=imaging_resolution,
        )
        imaging_asinh_scales, spectrum_asinh_scale_fake, spectrum_asinh_scale_real = (
            resolve_runtime_asinh_scales(
                scales_path,
                imaging_percentile=imaging_pct,
                spectrum_percentile=spectrum_pct,
                use_sdss=use_sdss,
                use_legacy=use_legacy,
            )
        )

    # After asinh, raw flux clamps are usually wrong — default them off unless set.
    clamp_min = model_top.get("imaging_clamp_min", -5.0)
    clamp_max = model_top.get("imaging_clamp_max", 100.0)
    if input_norm_mode == "asinh" and "imaging_clamp_min" not in model_top:
        clamp_min = None
    if input_norm_mode == "asinh" and "imaging_clamp_max" not in model_top:
        clamp_max = None

    return ModelConfig(
        architecture=model_top.get("architecture", "unet"),
        output_head=model_top.get("output_head", "single"),
        use_sdss=use_sdss,
        use_legacy=use_legacy,
        use_spectrum=use_spectrum,
        use_footprint_mask=use_footprint_model,
        imaging_resolution=imaging_resolution,  # type: ignore[arg-type]
        spatial_pipeline=model_top.get("spatial_pipeline", "symmetric"),
        footprint_mode=footprint_mode,
        target_spatial_size=int(model_top.get("target_spatial_size", 76)),
        hr_project_mode=model_top.get("hr_project_mode", "bilinear"),
        imaging_clamp_min=clamp_min,
        imaging_clamp_max=clamp_max,
        input_norm_mode=input_norm_mode,  # type: ignore[arg-type]
        input_norm_scales_path=str(scales_path) if scales_path else None,
        input_norm_imaging_percentile=float(imaging_pct),
        input_norm_spectrum_percentile=float(spectrum_pct),
        imaging_asinh_scales=imaging_asinh_scales,
        spectrum_asinh_scale_fake=spectrum_asinh_scale_fake,
        spectrum_asinh_scale_real=spectrum_asinh_scale_real,
        base_channels=int(model_top.get("base_channels", 64)),
        bottleneck_multiplier=int(model_top.get("bottleneck_multiplier", 16)),
        n_down=int(model_top.get("n_down", 4)),
        dropout=float(model_top.get("dropout", 0.1)),
        upsample_mode=model_top.get("upsample_mode", "bilinear"),
        norm=model_top.get("norm", "gn"),
        residual_blocks=bool(model_top.get("residual_blocks", True)),
        cond_dim=int(model_top.get("cond_dim", 384)),
        film_injection=model_top.get("film_injection", "bottleneck"),
        spectrum_pooling=model_top.get("spectrum_pooling", "attention"),
        spectrum_use_wavelength=bool(model_top.get("spectrum_use_wavelength", True)),
        spectrum_use_ivar=bool(model_top.get("spectrum_use_ivar", True)),
        spectrum_wave_min=float(model_top.get("spectrum_wave_min", 3622.0)),
        spectrum_wave_max=float(model_top.get("spectrum_wave_max", 10354.0)),
        deep_supervision=bool(model_top.get("deep_supervision", False)),
        deep_supervision_weights=(
            [float(w) for w in model_top["deep_supervision_weights"]]
            if model_top.get("deep_supervision_weights") is not None
            else None
        ),
        deep_supervision_loss=model_top.get("deep_supervision_loss", "l1"),
        coarse_factor=int(model_top.get("coarse_factor", 2)),
        detail_scale_init=float(model_top.get("detail_scale_init", 0.1)),
        detail_scale_schedule=model_top.get("detail_scale_schedule"),
        losses=list(model_top.get("losses", ["charbonnier", "grad", "integration"])),
        loss_weights=[float(w) for w in model_top.get("loss_weights", [1.0, 0.1, 0.05])],
        loss_params=model_top.get("loss_params", {}),
    )


def _describe_inputs(model_cfg: ModelConfig, data_cfg: DataConfig) -> str:
    parts: list[str] = []
    if model_cfg.use_sdss:
        grid = data_cfg.resolve_imaging_grid()
        if grid == "sdss_native":
            res = "SDSS-native ~196×196 (Amara-oriented)"
        else:
            ov = data_cfg.resolve_aligned_oversample()
            res = f"Amara-aligned ×{ov}"
        parts.append(f"{model_cfg.n_sdss_bands} SDSS ({res})")
    if model_cfg.use_legacy:
        parts.append(f"{model_cfg.n_legacy_bands} Legacy")
    if model_cfg.uses_footprint_in_model():
        if model_cfg.footprint_mode == "spatial_channel":
            parts.append("footprint ch")
        elif model_cfg.footprint_mode == "fusion_concat":
            parts.append("footprint fusion")
    spec = " + spectrum (FiLM)" if model_cfg.use_spectrum and data_cfg.use_spectrum else ""
    pipeline = model_cfg.spatial_pipeline
    return (
        f"{model_cfg.backbone_input_channels()} ch backbone "
        f"({', '.join(parts)}){spec} | pipeline={pipeline} "
        f"-> {model_cfg.n_target_maps} maps @ {model_cfg.target_spatial_size}×{model_cfg.target_spatial_size}"
    )


def _run_eval_only(args: argparse.Namespace) -> int:
    training_top = load_jsonc(args.config).get("training", {})
    save_root = Path(training_top.get("logging", {}).get("root_dir", "runs/manga_maps"))
    run_dir = save_root / args.run_name
    snap_path = run_dir / "config_used.json"
    if not snap_path.is_file():
        raise SystemExit(f"Run config not found: {snap_path}\nUse --run-name for an existing run directory.")

    snap = json.loads(snap_path.read_text(encoding="utf-8"))
    user_cfg = snap.get("user") or {}
    data_top = user_cfg.get("data", {})
    model_top = user_cfg.get("model", {})
    imaging_resolution = model_top.get("imaging_resolution", data_top.get("imaging_resolution", "aligned"))
    model_cfg = build_model_config(model_top, data_top, imaging_resolution=imaging_resolution)
    data_cfg = build_data_config(data_top, imaging_resolution=imaging_resolution)
    train_cfg = build_train_config(user_cfg.get("training", training_top), run_name=args.run_name)

    ckpt_path = args.checkpoint or (run_dir / "ckpts" / "best.pt")
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    if not data_cfg.split_csv_path.is_file():
        raise SystemExit(f"Split CSV not found: {data_cfg.split_csv_path}")

    set_seed(train_cfg.seed)
    _, dl_val, dl_test, dl_train_ns = make_manga_dataloaders(
        data_cfg,
        {
            "train_batch_size": train_cfg.train_batch_size,
            "eval_batch_size": train_cfg.eval_batch_size,
            "num_workers": user_cfg.get("training", training_top).get("batching", {}).get("num_workers", 0),
        },
    )
    split_loaders = {"train": dl_train_ns, "val": dl_val, "test": dl_test}

    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")
    model = MapGenerator(model_cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(_load_checkpoint_state(ckpt))
    model.eval()

    plots_dir = run_dir / "plots"
    map_keys = tuple(model.config.target_keys)
    print(f"Eval-only: {run_dir}")
    print(f"  checkpoint : {ckpt_path}")
    print(f"  device     : {device}")
    print(f"  splits     : {train_cfg.eval_splits}")

    for split in train_cfg.eval_splits:
        if split not in split_loaders:
            print(f"  skip unknown split: {split!r}")
            continue
        print(f"Evaluating {split} ...")
        rows = evaluate_map_predictions(
            model,
            split_loaders[split],
            device=device,
            map_keys=map_keys,
            plots_dir=plots_dir,
            split=split,
            max_plot=train_cfg.eval_max_plot if split != "train" else min(4, train_cfg.eval_max_plot),
        )
        csv_path = run_dir / "csv" / f"{split}_metrics.csv"
        write_metrics_csv(rows, csv_path)
        mse_vals = [float(r["mse_all"]) for r in rows if np.isfinite(float(r["mse_all"]))]
        mean_mse = float(np.mean(mse_vals)) if mse_vals else float("nan")
        print(f"  {split} mean mse_all={mean_mse:.6f} -> {csv_path}")
        print(f"  plots -> {plots_dir}")

    return 0


def build_train_config(training_top: dict, *, run_name: str) -> TrainConfig:
    opt = training_top.get("optimizer", {}) or {}
    batch = training_top.get("batching", {}) or {}
    early = training_top.get("early_stopping", {}) or {}
    lr_sched = training_top.get("lr_schedule", {}) or {}
    log = training_top.get("logging", {}) or {}
    return TrainConfig(
        run_name=run_name,
        save_root=str(log.get("root_dir", "runs/manga_maps")),
        seed=int(training_top.get("seed", 42)),
        epochs=int(training_top.get("epochs", 100)),
        train_batch_size=int(batch.get("train_batch_size", 8)),
        eval_batch_size=int(batch.get("eval_batch_size", 16)),
        lr=float(opt.get("lr", 1e-3)),
        weight_decay=float(opt.get("weight_decay", 1e-4)),
        lr_schedule=str(lr_sched.get("name", "warmup_cosine")),
        lr_warmup_epochs=int(lr_sched.get("warmup_epochs", 5)),
        lr_min_ratio=float(lr_sched.get("min_lr_ratio", 0.01)),
        grad_clip=float(training_top.get("grad_clip", 1.0)),
        amp=bool(training_top.get("mixed_precision", True)),
        early_stop_patience=int(early.get("patience", 15)),
        early_stop_start_epoch=int(early.get("start_epoch", 1)),
        save_every=int(log.get("save_every_epochs", 5)),
        device=str(training_top.get("device", "cuda")),
        write_plots=bool(log.get("write_plots", True)),
        write_csv_history=bool(log.get("write_csv_history", True)),
        save_config_snapshot=bool(log.get("save_config_snapshot", True)),
        eval_max_plot=int(log.get("eval_max_plot", 8)),
        eval_splits=tuple(log.get("eval_splits", ["val", "test"])),
        run_post_train_eval=bool(log.get("run_post_train_eval", True)),
    )


def main(argv: list[str] | None = None) -> int:
    faulthandler.enable()
    if hasattr(faulthandler, "register"):
        try:
            import signal

            faulthandler.register(signal.SIGTERM, file=sys.stderr, all_threads=True)
        except (AttributeError, ValueError, OSError):
            pass

    parser = argparse.ArgumentParser(description="Train MaNGA conditional map model.")
    parser.add_argument("--config", type=Path, default=Path("config.jsonc"))
    parser.add_argument("--run-name", type=str, default="run_001")
    parser.add_argument("--autoinc", action="store_true", help="Auto-increment run name if exists")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; load checkpoint and run val/test plots + metrics",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint file (default: <run>/ckpts/best.pt)",
    )
    args = parser.parse_args(argv)

    if args.eval_only:
        return _run_eval_only(args)

    user_cfg = load_jsonc(args.config)
    training_top = user_cfg.get("training", {})
    data_top = user_cfg.get("data", {})
    model_top = user_cfg.get("model", {})
    imaging_resolution = model_top.get("imaging_resolution", data_top.get("imaging_resolution", "aligned"))
    model_cfg = build_model_config(model_top, data_top, imaging_resolution=imaging_resolution)
    data_cfg = build_data_config(data_top, imaging_resolution=imaging_resolution)

    save_root = Path(training_top.get("logging", {}).get("root_dir", "runs/manga_maps"))
    run_name = _resolve_run_name(save_root, args.run_name, args.autoinc)
    train_cfg = build_train_config(training_top, run_name=run_name)

    if not data_cfg.split_csv_path.is_file():
        raise SystemExit(
            f"Split CSV not found: {data_cfg.split_csv_path}\n"
            f"Create it with: python -m src.data.make_splits --config {args.config}"
        )

    set_seed(train_cfg.seed)
    base_dataset = build_base_dataset(data_cfg)
    dl_train, dl_val, dl_test, dl_train_ns = make_manga_dataloaders(
        data_cfg,
        {
            "train_batch_size": train_cfg.train_batch_size,
            "eval_batch_size": train_cfg.eval_batch_size,
            "num_workers": training_top.get("batching", {}).get("num_workers", 0),
        },
    )

    model = MapGenerator(model_cfg)
    print("=" * 60)
    print("MaNGA map training")
    print("=" * 60)
    print(f"  run          : {run_name}")
    print(f"  train/val/test batches: {len(dl_train)}/{len(dl_val)}/{len(dl_test)}")
    print(f"  architecture : {model_cfg.architecture}  head={model_cfg.output_head}")
    print(
        f"  conditioning : film={model_cfg.film_injection}  "
        f"deep_supervision={model_cfg.deep_supervision}"
    )
    print(f"  spatial pipe : {model_cfg.spatial_pipeline}  imaging={model_cfg.imaging_resolution} (grid={data_cfg.resolve_imaging_grid()})")
    print(f"  footprint    : {model_cfg.footprint_mode}")
    print(f"  inputs       : {_describe_inputs(model_cfg, data_cfg)}")
    print(f"  losses       : {list(zip(model_cfg.losses, model_cfg.loss_weights))}")
    print(f"  split csv    : {data_cfg.split_csv_path}")
    if data_cfg.use_sdss:
        grid = data_cfg.resolve_imaging_grid()
        oversample = data_cfg.resolve_aligned_oversample()
        counts = count_aligned_caches(
            base_dataset.data_root,
            base_dataset.rows,
            oversample=oversample,
            grid=grid,
        )
        cached, eligible = counts["sdss_cached"], counts["sdss_eligible"]
        print(f"  SDSS aligned cache ({grid}): {cached:,}/{eligible:,} galaxies")
        if cached < eligible:
            print(
                "  WARNING: missing aligned caches → each sample WCS-reprojects (very slow).\n"
                "  Pre-export once, then restart training:\n"
                f"       python -m manga_prep export-aligned-imaging --config {args.config} "
                f"--survey sdss --skip-existing --workers 8"
            )
    print("=" * 60)

    try:
        run_dirs = run_training(
            model,
            train_cfg,
            dl_train,
            dl_val,
            dl_test,
            dl_train_ns,
            user_snapshot=user_cfg,
        )
    except Exception as exc:
        print(f"\nTraining failed: {exc}", flush=True)
        raise

    print(f"Done. Artifacts in {run_dirs['root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
