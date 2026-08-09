"""
Tiny-set diffusion overfit diagnostics (unconditional prior / conditional memorize).

Usage:
  python runner_score_overfit.py --config config_score_overfit_uncond.jsonc --run-name score_overfit_uncond --autoinc
  python runner_score_overfit.py --config config_score_overfit_cond.jsonc --run-name score_overfit_cond --autoinc
"""
from __future__ import annotations

import argparse
import faulthandler
import signal
import sys
from pathlib import Path

from runner import _resolve_run_name, build_data_config, build_model_config, build_train_config, set_seed
from runner_score_generator import _build_score_model
from src.config_loader import load_jsonc
from src.data.score_dataloaders import compute_score_norm_stats, make_score_dataloaders
from src.data.score_subset import select_score_plateifus
from src.training.train import run_training


def main(argv: list[str] | None = None) -> int:
    faulthandler.enable()
    if hasattr(faulthandler, "register"):
        try:
            faulthandler.register(signal.SIGTERM, file=sys.stderr, all_threads=True)
        except (AttributeError, ValueError, OSError):
            pass

    parser = argparse.ArgumentParser(description="Score-model overfit diagnostic.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name", type=str, default="score_overfit")
    parser.add_argument("--autoinc", action="store_true")
    parser.add_argument("--n-galaxies", type=int, default=None)
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
    score_top = user_cfg.get("score", {})
    if args.n_galaxies is not None:
        score_top["n_galaxies"] = int(args.n_galaxies)

    # Force overfit settings into score block for _build_score_model.
    score_top.setdefault("condition_on_label_mask", False)
    user_cfg["score"] = score_top

    data_cfg = build_data_config(data_top, imaging_resolution="aligned", model_top=model_top)
    data_cfg.use_spectrum = bool(data_top.get("use_spectrum", not score_top.get("unconditional", False)))

    save_root = Path(training_top.get("logging", {}).get("root_dir", "runs/score_overfit"))
    run_name = _resolve_run_name(save_root, args.run_name, args.autoinc)
    train_cfg = build_train_config(training_top, run_name=run_name)
    # Overfit: evaluate on the train galaxies themselves.
    train_cfg.eval_splits = ["train"]
    set_seed(train_cfg.seed)

    n_gal = int(score_top.get("n_galaxies", 16))
    coverage_csv = score_top.get("coverage_csv", "runs/dataset_audit/galaxy_coverage_meta.csv")
    train_ids_all = select_score_plateifus(
        coverage_csv=coverage_csv,
        split_csv=data_cfg.split_csv_path,
        split="train",
        feature=str(score_top.get("feature", "ha_flux")),  # type: ignore[arg-type]
        min_coverage_pct=float(score_top.get("min_coverage_pct", 99.0)),
    )
    train_ids = train_ids_all[:n_gal]
    if len(train_ids) < n_gal:
        raise SystemExit(f"Only found {len(train_ids)} dense galaxies; need {n_gal}")

    dl_train, dl_val, dl_test, dl_train_ns, _ = make_score_dataloaders(
        data_cfg,
        {
            "train_batch_size": train_cfg.train_batch_size,
            "eval_batch_size": train_cfg.eval_batch_size,
            "num_workers": training_top.get("batching", {}).get("num_workers", 0),
        },
        coverage_csv=coverage_csv,
        min_coverage_pct=float(score_top.get("min_coverage_pct", 99.0)),
        feature=str(score_top.get("feature", "ha_flux")),
        use_stratified_weights=False,
        plateifu_allowlist=train_ids,
    )
    del dl_val, dl_test  # overfit monitors train only

    model_cfg = build_model_config(model_top, data_top, imaging_resolution="aligned")
    model_cfg.target_keys = tuple(model_top.get("target_keys", ["ha_flux"]))
    model_cfg.n_target_maps = len(model_cfg.target_keys)
    if score_top.get("unconditional"):
        model_cfg.use_spectrum = False
    score_norm = compute_score_norm_stats(dl_train_ns, model_cfg, max_batches=None)
    score_top["score_norm"] = score_norm.to_dict()
    score_top["n_train_galaxies"] = len(train_ids)
    score_top["overfit_plateifus"] = train_ids
    user_cfg["score"] = score_top

    model = _build_score_model(user_cfg, score_norm=score_norm, force_mode="generator")
    model.eval_t_start_fracs = [1.0]

    kind = "UNCONDITIONAL prior" if score_top.get("unconditional") else "CONDITIONAL memorize"
    print("=" * 60)
    print(f"Score overfit — {kind}")
    print("=" * 60)
    print(f"  run         : {run_name}")
    print(f"  n_galaxies  : {len(train_ids)}")
    print(f"  plateifus   : {train_ids[:8]}{'...' if len(train_ids) > 8 else ''}")
    print(f"  uncond      : {bool(score_top.get('unconditional'))}")
    print(f"  label_mask  : {model.condition_on_label_mask}")
    print(f"  min_snr     : {model.use_min_snr} (gamma={model.min_snr_gamma})")
    print(f"  bottleneck  : {model.denoiser.use_bottleneck_attn}")
    print(f"  schedule    : {score_top.get('schedule')}")
    print(f"  score norm  : mean={score_norm.mean:.4f} std={score_norm.std:.4f}")
    print("=" * 60)

    run_dirs = run_training(
        model,
        train_cfg,
        dl_train,
        dl_train_ns,  # val = train for overfit monitoring
        dl_train_ns,
        dl_train_ns,
        user_snapshot=user_cfg,
    )
    print(f"Done. Artifacts in {run_dirs['root']}")
    print("Inspect train_*.png: samples should look galaxy-specific if the pipeline works.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
