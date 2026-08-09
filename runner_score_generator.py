"""
Train / eval the direct full-map score generator (Hα).

Usage:
  python runner_score_generator.py --config config_score_generator.jsonc --run-name score_gen_ha99 --autoinc
  python runner_score_generator.py --config config_score_generator.jsonc --run-name score_gen_ha99 --eval-only
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
from src.data.score_dataloaders import compute_score_norm_stats, make_score_dataloaders
from src.metrics.plots import write_metrics_csv
from src.metrics.score_plots import evaluate_score_samples
from src.models.map_score_wrapper import MapScoreModel
from src.training.train import _load_checkpoint_state, run_training


def _build_score_model(user_cfg: dict, *, score_norm, force_mode: str = "generator") -> MapScoreModel:
    data_top = user_cfg.get("data", {})
    model_top = user_cfg.get("model", {})
    score_top = user_cfg.get("score", {})
    imaging_resolution = "aligned"
    model_cfg = build_model_config(model_top, data_top, imaging_resolution=imaging_resolution)
    model_cfg.footprint_mode = "loss_only"
    model_cfg.use_hr_cross_attn = False
    model_cfg.target_keys = tuple(model_top.get("target_keys", ["ha_flux"]))
    model_cfg.n_target_maps = len(model_cfg.target_keys)

    mults = tuple(int(x) for x in score_top.get("channel_mults", [1, 2, 4, 4]))
    # Legacy checkpoints trained with label_mask in cond; default False for new runs.
    if "condition_on_label_mask" in score_top:
        cond_label = bool(score_top["condition_on_label_mask"])
    else:
        cond_label = True  # backward-compatible load of older runs
    model = MapScoreModel.build(
        model_cfg,
        mode=force_mode,  # type: ignore[arg-type]
        score_norm=score_norm,
        base_run_dir=score_top.get("base_run_dir"),
        base_checkpoint=score_top.get("base_checkpoint", "best.pt"),
        channel_key=str(score_top.get("channel_key", "ha_flux")),
        diffusion_steps=int(score_top.get("diffusion_steps", 1000)),
        ddim_steps=int(score_top.get("ddim_steps", 50)),
        n_samples=int(score_top.get("n_samples", 8)),
        base_channels=int(score_top.get("base_channels", 64)),
        channel_mults=mults,
        num_res_blocks=int(score_top.get("num_res_blocks", 2)),
        schedule=str(score_top.get("schedule", "linear")),
        ema_decay=float(score_top.get("ema_decay", 0.9999)),
        t_start_frac=float(score_top.get("t_start_frac", 0.25)),
        receive_base_as_cond=(force_mode == "corrector"),
        condition_on_label_mask=cond_label,
        unconditional=bool(score_top.get("unconditional", False)),
        use_min_snr=bool(score_top.get("use_min_snr", False)),
        min_snr_gamma=float(score_top.get("min_snr_gamma", 5.0)),
        use_bottleneck_attn=bool(score_top.get("use_bottleneck_attn", False)),
        attn_heads=int(score_top.get("attn_heads", 4)),
    )
    raw_fracs = score_top.get("eval_t_start_fracs")
    if raw_fracs is not None:
        model.eval_t_start_fracs = [float(x) for x in raw_fracs]
    elif force_mode == "generator":
        model.eval_t_start_fracs = [1.0, 0.5, 0.25, 0.1]
    else:
        model.eval_t_start_fracs = None
    return model


def _parse_t_fracs(raw: str | None) -> list[float] | None:
    if raw is None or str(raw).strip() == "":
        return None
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def _run_eval_only(args: argparse.Namespace) -> int:
    live_cfg = load_jsonc(args.config)
    training_top = live_cfg.get("training", {})
    save_root = Path(training_top.get("logging", {}).get("root_dir", "runs/score_generator"))
    run_dir = save_root / args.run_name
    snap = json.loads((run_dir / "config_used.json").read_text(encoding="utf-8"))
    user_cfg = snap.get("user") or {}
    score_top = dict(user_cfg.get("score", {}))
    train_cfg = build_train_config(user_cfg.get("training", training_top), run_name=args.run_name)

    data_top = user_cfg.get("data", {})
    model_top = user_cfg.get("model", {})
    data_cfg = build_data_config(data_top, imaging_resolution="aligned", model_top=model_top)
    data_cfg.use_spectrum = bool(data_top.get("use_spectrum", True))

    # Coverage band for this eval (defaults = training ≥99% dense set).
    cov_min = (
        float(args.coverage_min)
        if args.coverage_min is not None
        else float(score_top.get("min_coverage_pct", 99.0))
    )
    cov_max = float(args.coverage_max) if args.coverage_max is not None else None
    show_completion = bool(args.show_completion) or (cov_max is not None and cov_max < 99.0)

    from src.models.map_score import ScoreNormStats

    norm_snap = score_top.get("score_norm")
    batching = {
        "train_batch_size": train_cfg.train_batch_size,
        "eval_batch_size": train_cfg.eval_batch_size,
        "num_workers": user_cfg.get("training", {}).get("batching", {}).get("num_workers", 0),
    }
    coverage_csv = score_top.get("coverage_csv", "runs/dataset_audit/galaxy_coverage_meta.csv")
    feature = str(score_top.get("feature", "ha_flux"))

    # Norm stats always from the original training (≥99%) snapshot when available.
    if norm_snap:
        score_norm = ScoreNormStats.from_dict(norm_snap)
    else:
        _, _, _, dl_train_ns_dense, _ = make_score_dataloaders(
            data_cfg,
            batching,
            coverage_csv=coverage_csv,
            min_coverage_pct=float(score_top.get("min_coverage_pct", 99.0)),
            feature=feature,
            use_stratified_weights=False,
        )
        model_cfg = build_model_config(model_top, data_top, imaging_resolution="aligned")
        model_cfg.target_keys = tuple(model_top.get("target_keys", ["ha_flux"]))
        model_cfg.n_target_maps = len(model_cfg.target_keys)
        score_norm = compute_score_norm_stats(dl_train_ns_dense, model_cfg, max_batches=50)

    _, dl_val, dl_test, dl_train_ns, ids = make_score_dataloaders(
        data_cfg,
        batching,
        coverage_csv=coverage_csv,
        min_coverage_pct=cov_min,
        max_coverage_pct=cov_max,
        feature=feature,
        use_stratified_weights=False,
        plateifu_allowlist=score_top.get("overfit_plateifus"),
    )
    print(
        f"Eval coverage band: [{cov_min:g}, "
        f"{'inf' if cov_max is None else f'{cov_max:g}'}]  "
        f"n_train_band={len(ids)}  val_batches={len(dl_val)}  test_batches={len(dl_test)}"
    )

    model = _build_score_model(user_cfg, score_norm=score_norm, force_mode="generator")
    model.assert_generator_no_base_cond()
    model.eval_show_completion = show_completion
    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    ckpt_path = args.checkpoint or (run_dir / "ckpts" / "best.pt")
    model.load_state_dict(
        _load_checkpoint_state(torch.load(ckpt_path, map_location=device, weights_only=False))
    )
    model.eval()

    cli_fracs = _parse_t_fracs(getattr(args, "t_start_fracs", None))
    if cli_fracs is not None:
        t_fracs = cli_fracs
    elif args.t_start_frac is not None:
        t_fracs = [float(args.t_start_frac)]
    else:
        t_fracs = list(getattr(model, "eval_t_start_fracs", None) or [1.0, 0.5, 0.25, 0.1])

    max_plot = int(args.max_plot) if args.max_plot is not None else int(train_cfg.eval_max_plot)
    max_galaxies = int(args.max_galaxies) if args.max_galaxies is not None else max(max_plot, 16)

    if cov_max is not None:
        band_tag = f"cov_{cov_min:g}_{cov_max:g}".replace(".", "p")
    elif cov_min != float(score_top.get("min_coverage_pct", 99.0)):
        band_tag = f"cov_ge{cov_min:g}".replace(".", "p")
    else:
        band_tag = "dense99"
    plots_dir = run_dir / "plots" / band_tag

    split_loaders = {"train": dl_train_ns, "val": dl_val, "test": dl_test}
    eval_splits = list(args.splits.split(",")) if args.splits else list(train_cfg.eval_splits)
    for split in eval_splits:
        if split not in split_loaders:
            continue
        if len(split_loaders[split]) == 0:
            print(f"  {split}: empty loader for this coverage band; skip")
            continue
        use_ema = not bool(getattr(args, "no_ema", False))
        ema_tag = "ema" if use_ema else "noema"
        rows = evaluate_score_samples(
            model,
            split_loaders[split],
            device=device,
            map_keys=tuple(model.config.target_keys),
            plots_dir=plots_dir / ema_tag,
            split=split,
            max_plot=max_plot,
            max_galaxies=max_galaxies,
            n_samples=int(score_top.get("n_samples", 8)),
            ddim_steps=int(score_top.get("ddim_steps", 50)),
            t_start_fracs=t_fracs,
            seed=train_cfg.seed,
            use_ema=use_ema,
        )
        csv_path = run_dir / "csv" / f"{split}_metrics_{band_tag}_{ema_tag}.csv"
        write_metrics_csv(rows, csv_path)
        mse_vals = [float(r["mse_all"]) for r in rows if np.isfinite(float(r["mse_all"]))]
        print(
            f"  {split} n={len(rows)} use_ema={use_ema} mean mse_all="
            f"{float(np.mean(mse_vals)):.6f} -> {csv_path}"
        )
        print(f"  panels ({t_fracs}, max_plot={max_plot}) -> {plots_dir / ema_tag}")
    return 0


def main(argv: list[str] | None = None) -> int:
    faulthandler.enable()
    if hasattr(faulthandler, "register"):
        try:
            faulthandler.register(signal.SIGTERM, file=sys.stderr, all_threads=True)
        except (AttributeError, ValueError, OSError):
            pass

    parser = argparse.ArgumentParser(description="Train direct Hα score generator.")
    parser.add_argument("--config", type=Path, default=Path("config_score_generator.jsonc"))
    parser.add_argument("--run-name", type=str, default="score_gen_ha99")
    parser.add_argument("--autoinc", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--t-start-frac", type=float, default=None)
    parser.add_argument(
        "--t-start-fracs",
        type=str,
        default=None,
        help="Comma-separated start-noise fractions for multi-row panels, e.g. 1.0,0.5,0.25,0.1",
    )
    parser.add_argument("--max-plot", type=int, default=None, help="How many galaxies to plot")
    parser.add_argument(
        "--max-galaxies",
        type=int,
        default=None,
        help="How many galaxies to score for metrics CSV (default >= max-plot)",
    )
    parser.add_argument(
        "--coverage-min",
        type=float,
        default=None,
        help="Min Ha coverage %% for eval subset (default: training threshold)",
    )
    parser.add_argument(
        "--coverage-max",
        type=float,
        default=None,
        help="Max Ha coverage %% for eval subset (e.g. 80 for sparse galaxies)",
    )
    parser.add_argument(
        "--show-completion",
        action="store_true",
        help="Show predictions on full footprint + Labels column (auto for coverage-max < 99)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default=None,
        help="Comma-separated splits to eval, e.g. val,test",
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Sample with live denoiser weights (skip EMA). Diagnostic for EMA bugs.",
    )
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
    score_top = user_cfg.get("score", {})

    data_cfg = build_data_config(data_top, imaging_resolution="aligned", model_top=model_top)
    data_cfg.use_spectrum = bool(data_top.get("use_spectrum", True))

    save_root = Path(training_top.get("logging", {}).get("root_dir", "runs/score_generator"))
    run_name = _resolve_run_name(save_root, args.run_name, args.autoinc)
    train_cfg = build_train_config(training_top, run_name=run_name)
    set_seed(train_cfg.seed)

    dl_train, dl_val, dl_test, dl_train_ns, train_ids = make_score_dataloaders(
        data_cfg,
        {
            "train_batch_size": train_cfg.train_batch_size,
            "eval_batch_size": train_cfg.eval_batch_size,
            "num_workers": training_top.get("batching", {}).get("num_workers", 0),
        },
        coverage_csv=score_top.get("coverage_csv", "runs/dataset_audit/galaxy_coverage_meta.csv"),
        min_coverage_pct=float(score_top.get("min_coverage_pct", 99.0)),
        feature=str(score_top.get("feature", "ha_flux")),
        use_stratified_weights=bool(score_top.get("use_stratified_weights", True)),
    )

    model_cfg = build_model_config(model_top, data_top, imaging_resolution="aligned")
    model_cfg.target_keys = tuple(model_top.get("target_keys", ["ha_flux"]))
    model_cfg.n_target_maps = len(model_cfg.target_keys)
    score_norm = compute_score_norm_stats(dl_train_ns, model_cfg, max_batches=100)
    score_top["score_norm"] = score_norm.to_dict()
    score_top["n_train_galaxies"] = len(train_ids)
    user_cfg["score"] = score_top

    model = _build_score_model(user_cfg, score_norm=score_norm, force_mode="generator")
    model.assert_generator_no_base_cond()

    print("=" * 60)
    print("Direct score generator (full Hα map)")
    print("=" * 60)
    print(f"  run            : {run_name}")
    print(f"  train galaxies : {len(train_ids)} (Ha ≥ {score_top.get('min_coverage_pct')}%)")
    print(f"  score norm     : mean={score_norm.mean:.4f} std={score_norm.std:.4f}")
    print(f"  train/val/test : {len(dl_train)}/{len(dl_val)}/{len(dl_test)} batches")
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
