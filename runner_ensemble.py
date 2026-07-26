"""
Train and evaluate an uncertainty-model ensemble.

Each member:
  - fixed test split (from base_split_csv)
  - resampled train/val from the combined train+val pool
  - unique init seed

Usage:
  python runner_ensemble.py --config config_uncertainty.jsonc --ensemble-name ens_v1 --n-models 5
  python runner_ensemble.py --config config_uncertainty.jsonc --ensemble-name ens_v1 --eval-only
  python runner_ensemble.py ... --start-member 2   # resume from member 2
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

from runner import build_data_config, build_model_config, build_train_config, set_seed
from src.config_loader import load_jsonc
from src.data.ensemble_splits import (
    load_ensemble_manifest,
    write_ensemble_manifest,
    write_member_split_csv,
)
from src.data.make_dataloader import make_manga_dataloaders
from src.metrics.uncertainty_plots import (
    evaluate_ensemble_predictions,
    write_calibration_csv,
    write_metrics_csv,
)
from src.models.uncertainty_wrapper import UncertaintyMapGenerator
from src.training.train import _load_checkpoint_state, run_training


def _member_name(index: int) -> str:
    return f"member_{index:02d}"


def _ensemble_root(save_root: Path, ensemble_name: str) -> Path:
    return save_root / ensemble_name


def _member_run_name(ensemble_name: str, index: int) -> str:
    return f"{ensemble_name}/members/{_member_name(index)}"


def _member_seed(base_seed: int, offset: int, index: int) -> int:
    return int(base_seed) + int(offset) + int(index) * 997


def _load_member_model(
    member_dir: Path,
    model_cfg,
    device: torch.device,
) -> UncertaintyMapGenerator:
    ckpt_path = member_dir / "ckpts" / "best.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing member checkpoint: {ckpt_path}")
    model = UncertaintyMapGenerator(model_cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(_load_checkpoint_state(ckpt))
    model.eval()
    return model


def _train_member(
    *,
    user_cfg: dict,
    ensemble_name: str,
    member_index: int,
    ensemble_dir: Path,
    base_split_csv: Path,
    base_seed: int,
    seed_offset: int,
    force: bool,
) -> bool:
    member_label = _member_name(member_index)
    member_run = _member_run_name(ensemble_name, member_index)
    member_dir = ensemble_dir / "members" / member_label
    best_pt = member_dir / "ckpts" / "best.pt"

    if best_pt.is_file() and not force:
        print(f"[SKIP] {member_label}: {best_pt} exists")
        return True

    member_seed = _member_seed(base_seed, seed_offset, member_index)
    split_dir = ensemble_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_csv = split_dir / f"{member_label}.csv"
    write_member_split_csv(base_split_csv, split_csv, member_seed=member_seed)

    cfg = json.loads(json.dumps(user_cfg))
    cfg.setdefault("data", {})["split"] = dict(cfg.get("data", {}).get("split", {}))
    cfg["data"]["split"]["split_csv_path"] = str(split_csv).replace("\\", "/")
    cfg.setdefault("training", {})["seed"] = member_seed

    training_top = cfg.get("training", {})
    data_top = cfg.get("data", {})
    model_top = cfg.get("model", {})
    imaging_resolution = model_top.get("imaging_resolution", data_top.get("imaging_resolution", "aligned"))
    model_cfg = build_model_config(model_top, data_top, imaging_resolution=imaging_resolution)
    data_cfg = build_data_config(data_top, imaging_resolution=imaging_resolution)
    train_cfg = build_train_config(training_top, run_name=member_run)

    set_seed(member_seed)
    dl_train, dl_val, dl_test, dl_train_ns = make_manga_dataloaders(
        data_cfg,
        {
            "train_batch_size": train_cfg.train_batch_size,
            "eval_batch_size": train_cfg.eval_batch_size,
            "num_workers": training_top.get("batching", {}).get("num_workers", 0),
        },
    )

    print("=" * 60)
    print(f"Ensemble member {member_index + 1}: {member_label}")
    print(f"  run       : {member_run}")
    print(f"  seed      : {member_seed}")
    print(f"  split csv : {split_csv}")
    print("=" * 60)

    model = UncertaintyMapGenerator(model_cfg)
    run_training(
        model,
        train_cfg,
        dl_train,
        dl_val,
        dl_test,
        dl_train_ns,
        user_snapshot=cfg,
    )
    return best_pt.is_file()


def _run_ensemble_eval(
    *,
    user_cfg: dict,
    ensemble_name: str,
    ensemble_dir: Path,
    n_members: int,
    device: torch.device,
    eval_max_plot: int | None = None,
) -> None:
    training_top = user_cfg.get("training", {})
    data_top = user_cfg.get("data", {})
    model_top = user_cfg.get("model", {})
    ensemble_top = user_cfg.get("ensemble", {})
    imaging_resolution = model_top.get("imaging_resolution", data_top.get("imaging_resolution", "aligned"))
    model_cfg = build_model_config(model_top, data_top, imaging_resolution=imaging_resolution)

    base_split_csv = Path(
        ensemble_top.get("base_split_csv", data_top.get("split", {}).get("split_csv_path"))
    )
    data_cfg = build_data_config(data_top, imaging_resolution=imaging_resolution)
    data_cfg.split_csv_path = base_split_csv
    train_cfg = build_train_config(training_top, run_name=ensemble_name)
    if eval_max_plot is not None:
        max_plot = int(eval_max_plot)
    else:
        max_plot = int(ensemble_top.get("eval_max_plot", train_cfg.eval_max_plot))

    _, _, dl_test, _ = make_manga_dataloaders(
        data_cfg,
        {
            "train_batch_size": train_cfg.train_batch_size,
            "eval_batch_size": train_cfg.eval_batch_size,
            "num_workers": training_top.get("batching", {}).get("num_workers", 0),
        },
    )

    models: list[UncertaintyMapGenerator] = []
    for i in range(n_members):
        member_dir = ensemble_dir / "members" / _member_name(i)
        models.append(_load_member_model(member_dir, model_cfg, device))

    eval_dir = ensemble_dir / "ensemble"
    plots_dir = eval_dir / "plots"
    csv_dir = eval_dir / "csv"
    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    secondary = str(ensemble_top.get("secondary_sigma", "epistemic"))
    print(f"Ensemble eval on test ({n_members} members) -> {eval_dir}")
    if max_plot <= 0:
        print("  per-galaxy plots: all test galaxies")
    else:
        print(f"  per-galaxy plots: first {max_plot} test galaxies")
    rows, calib_rows = evaluate_ensemble_predictions(
        models,
        dl_test,
        device=device,
        map_keys=tuple(model_cfg.target_keys),
        plots_dir=plots_dir,
        split="test",
        max_plot=max_plot,
        secondary_sigma=secondary,
    )
    write_metrics_csv(rows, csv_dir / "test_metrics.csv")
    write_calibration_csv(calib_rows, csv_dir / "test_calibration_bins.csv")

    mse_vals = [float(r["mse_all"]) for r in rows if np.isfinite(float(r["mse_all"]))]
    cov1 = [float(r["coverage_1sigma"]) for r in rows if np.isfinite(float(r["coverage_1sigma"]))]
    print(f"  test mean mse_all={float(np.mean(mse_vals)):.6f}")
    print(f"  test mean coverage@1σ={float(np.mean(cov1)):.3f}")
    print(f"  plots -> {plots_dir}")


def main(argv: list[str] | None = None) -> int:
    faulthandler.enable()
    if hasattr(faulthandler, "register"):
        try:
            faulthandler.register(signal.SIGTERM, file=sys.stderr, all_threads=True)
        except (AttributeError, ValueError, OSError):
            pass

    parser = argparse.ArgumentParser(description="Train/eval MaNGA uncertainty ensemble.")
    parser.add_argument("--config", type=Path, default=Path("config_uncertainty.jsonc"))
    parser.add_argument("--ensemble-name", type=str, required=True)
    parser.add_argument("--n-models", type=int, default=None, help="Number of ensemble members")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--start-member", type=int, default=0, help="Resume from this member index")
    parser.add_argument("--force", action="store_true", help="Retrain members even if best.pt exists")
    parser.add_argument("--device", type=str, default=None, help="Override config training.device")
    parser.add_argument(
        "--eval-max-plot",
        type=int,
        default=None,
        help="Max per-galaxy test PNGs (0 = all test galaxies; default: ensemble.eval_max_plot or logging.eval_max_plot)",
    )
    args = parser.parse_args(argv)

    user_cfg = load_jsonc(args.config)
    if args.device is not None:
        user_cfg.setdefault("training", {})["device"] = str(args.device)
    if user_cfg.get("model", {}).get("output_head") != "gaussian":
        raise SystemExit("config model.output_head must be 'gaussian'")

    training_top = user_cfg.get("training", {})
    data_top = user_cfg.get("data", {})
    ensemble_top = user_cfg.get("ensemble", {})
    save_root = Path(training_top.get("logging", {}).get("root_dir", "runs/manga_maps"))
    ensemble_dir = _ensemble_root(save_root, args.ensemble_name)
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = ensemble_dir / "manifest.json"
    base_split_csv = Path(
        ensemble_top.get("base_split_csv", data_top.get("split", {}).get("split_csv_path"))
    )
    if not base_split_csv.is_file():
        raise SystemExit(f"Base split CSV not found: {base_split_csv}")

    base_seed = int(training_top.get("seed", 42))
    seed_offset = int(ensemble_top.get("member_seed_offset", 1000))

    if manifest_path.is_file():
        manifest = load_ensemble_manifest(manifest_path)
        n_members = int(args.n_models or manifest.get("n_members", 0))
    else:
        n_members = int(args.n_models or 0)
        if n_members <= 0:
            raise SystemExit("--n-models is required for a new ensemble")
        member_seeds = [_member_seed(base_seed, seed_offset, i) for i in range(n_members)]
        write_ensemble_manifest(
            manifest_path,
            ensemble_name=args.ensemble_name,
            n_members=n_members,
            base_split_csv=str(base_split_csv).replace("\\", "/"),
            member_seeds=member_seeds,
            config_path=str(args.config).replace("\\", "/"),
            user_snapshot=user_cfg,
        )
        manifest = load_ensemble_manifest(manifest_path)

    n_members = int(args.n_models or manifest["n_members"])
    if args.n_models is not None and args.n_models != manifest.get("n_members"):
        manifest["n_members"] = n_members
        manifest["member_seeds"] = [_member_seed(base_seed, seed_offset, i) for i in range(n_members)]
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    device = torch.device(training_top.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    if not args.eval_only:
        for i in range(int(args.start_member), n_members):
            ok = _train_member(
                user_cfg=user_cfg,
                ensemble_name=args.ensemble_name,
                member_index=i,
                ensemble_dir=ensemble_dir,
                base_split_csv=base_split_csv,
                base_seed=base_seed,
                seed_offset=seed_offset,
                force=bool(args.force),
            )
            if not ok:
                print(f"[WARN] member {i} finished without best.pt — fix and resume with --start-member {i}")
                return 1

    _run_ensemble_eval(
        user_cfg=user_cfg,
        ensemble_name=args.ensemble_name,
        ensemble_dir=ensemble_dir,
        n_members=n_members,
        device=device,
        eval_max_plot=args.eval_max_plot,
    )
    print(f"Ensemble complete: {ensemble_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
