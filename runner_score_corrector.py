"""
Train / eval the full-map score corrector (SDEdit from frozen UNet Hα).

Usage:
  python runner_score_corrector.py --config config_score_corrector.jsonc --run-name score_corr_ha99 --autoinc
  python runner_score_corrector.py --config config_score_corrector.jsonc --run-name score_corr_ha99 --eval-only --t-start-frac 0.25
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import shutil
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
from runner_score_generator import _build_score_model
from src.config_loader import load_jsonc
from src.data.score_dataloaders import compute_score_norm_stats, make_score_dataloaders
from src.metrics.plots import write_metrics_csv
from src.metrics.score_plots import evaluate_score_samples
from src.training.train import _load_checkpoint_state, run_training


def _run_eval_only(args: argparse.Namespace) -> int:
    training_top = load_jsonc(args.config).get("training", {})
    save_root = Path(training_top.get("logging", {}).get("root_dir", "runs/score_corrector"))
    run_dir = save_root / args.run_name
    snap = json.loads((run_dir / "config_used.json").read_text(encoding="utf-8"))
    user_cfg = snap.get("user") or {}
    score_top = user_cfg.get("score", {})
    train_cfg = build_train_config(user_cfg.get("training", training_top), run_name=args.run_name)

    data_top = user_cfg.get("data", {})
    model_top = user_cfg.get("model", {})
    data_cfg = build_data_config(data_top, imaging_resolution="aligned", model_top=model_top)
    data_cfg.use_spectrum = bool(data_top.get("use_spectrum", True))

    from src.models.map_score import ScoreNormStats

    norm_snap = score_top.get("score_norm")
    _, dl_val, dl_test, dl_train_ns, _ = make_score_dataloaders(
        data_cfg,
        {
            "train_batch_size": train_cfg.train_batch_size,
            "eval_batch_size": train_cfg.eval_batch_size,
            "num_workers": user_cfg.get("training", {}).get("batching", {}).get("num_workers", 0),
        },
        coverage_csv=score_top.get("coverage_csv", "runs/dataset_audit/galaxy_coverage_meta.csv"),
        min_coverage_pct=float(score_top.get("min_coverage_pct", 99.0)),
        feature=str(score_top.get("feature", "ha_flux")),
        use_stratified_weights=False,
    )
    if norm_snap:
        score_norm = ScoreNormStats.from_dict(norm_snap)
    else:
        model_cfg = build_model_config(model_top, data_top, imaging_resolution="aligned")
        model_cfg.target_keys = tuple(model_top.get("target_keys", ["ha_flux"]))
        model_cfg.n_target_maps = len(model_cfg.target_keys)
        score_norm = compute_score_norm_stats(dl_train_ns, model_cfg, max_batches=50)

    model = _build_score_model(user_cfg, score_norm=score_norm, force_mode="corrector")
    # Corrector must receive base as conditioning.
    model.receive_base_as_cond = True
    model.assert_corrector_has_base_cond()

    # Rebuild denoiser cond width if build forced generator-style cond.
    # _build_score_model with force_mode=corrector sets receive_base_as_cond True via mode.
    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    ckpt_path = args.checkpoint or (run_dir / "ckpts" / "best.pt")
    model.load_state_dict(
        _load_checkpoint_state(torch.load(ckpt_path, map_location=device, weights_only=False))
    )
    model.eval()

    t_frac = (
        args.t_start_frac
        if args.t_start_frac is not None
        else float(score_top.get("t_start_frac", 0.25))
    )
    # Sample panels go in plots/ (next to loss curves). Also mirror into a
    # t-tagged subfolder so multi-t sweeps don't overwrite each other.
    t_tag = f"t{t_frac:.2f}".replace(".", "p")
    plots_main = run_dir / "plots"
    plots_t = plots_main / t_tag
    split_loaders = {"train": dl_train_ns, "val": dl_val, "test": dl_test}
    for split in train_cfg.eval_splits:
        if split not in split_loaders:
            continue
        rows = evaluate_score_samples(
            model,
            split_loaders[split],
            device=device,
            map_keys=tuple(model.config.target_keys),
            plots_dir=plots_main,
            split=split,
            max_plot=train_cfg.eval_max_plot,
            n_samples=int(score_top.get("n_samples", 4)),
            ddim_steps=int(score_top.get("ddim_steps", 50)),
            t_start_frac=t_frac,
            seed=train_cfg.seed,
        )
        plots_t.mkdir(parents=True, exist_ok=True)
        for row in rows:
            name = f"{split}_{str(row['plateifu']).replace('-', '_')}.png"
            src = plots_main / name
            if src.exists():
                shutil.copy2(src, plots_t / name)
        csv_path = run_dir / "csv" / f"{split}_metrics_{t_tag}.csv"
        write_metrics_csv(rows, csv_path)
        mse_vals = [float(r["mse_all"]) for r in rows if np.isfinite(float(r["mse_all"]))]
        print(f"  {split} t={t_frac:.2f} mean mse_all={float(np.mean(mse_vals)):.6f} -> {csv_path}")
        print(f"  sample panels -> {plots_main} (and {plots_t})")
    return 0


def main(argv: list[str] | None = None) -> int:
    faulthandler.enable()
    if hasattr(faulthandler, "register"):
        try:
            faulthandler.register(signal.SIGTERM, file=sys.stderr, all_threads=True)
        except (AttributeError, ValueError, OSError):
            pass

    parser = argparse.ArgumentParser(description="Train Hα score corrector (SDEdit).")
    parser.add_argument("--config", type=Path, default=Path("config_score_corrector.jsonc"))
    parser.add_argument("--run-name", type=str, default="score_corr_ha99")
    parser.add_argument("--autoinc", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--t-start-frac", type=float, default=None)
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
    if args.t_start_frac is not None:
        score_top["t_start_frac"] = float(args.t_start_frac)

    data_cfg = build_data_config(data_top, imaging_resolution="aligned", model_top=model_top)
    data_cfg.use_spectrum = bool(data_top.get("use_spectrum", True))

    save_root = Path(training_top.get("logging", {}).get("root_dir", "runs/score_corrector"))
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

    model = _build_score_model(user_cfg, score_norm=score_norm, force_mode="corrector")
    model.assert_corrector_has_base_cond()
    if model.base_model is not None:
        for p in model.base_model.parameters():
            if p.requires_grad:
                raise SystemExit("Frozen base model has trainable parameters")

    print("=" * 60)
    print("Score corrector (SDEdit from frozen UNet Hα)")
    print("=" * 60)
    print(f"  run            : {run_name}")
    print(f"  base           : {score_top.get('base_run_dir')}")
    print(f"  t_start_frac   : {score_top.get('t_start_frac', 0.25)}")
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
