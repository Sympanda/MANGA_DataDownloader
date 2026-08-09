"""
Capacity / overfit sense-check on a tiny train-only subset.

Picks ~N galaxies with the most valid supervised pixels, forces dropout=0,
disables augmentation + HR cross-attn (backbone + spectrum only), and trains
with no held-out val/test. Monitors train masked L1/MSE — if capacity is fine
we should drive train error near zero.

Usage:
  python scripts/overfit_tiny.py --config config.jsonc --n-galaxies 32 --run-name overfit_32 --autoinc
  python scripts/overfit_tiny.py --config config_phys_overfit.jsonc --n-galaxies 16 --run-name phys_overfit_16 --autoinc
  python scripts/overfit_tiny.py --config config.jsonc --n-galaxies 32 --epochs 500 --device cuda:0
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manga_prep.dataset.manga_dataset import collate_manga_batch  # noqa: E402
from manga_prep.dataset.map_coverage import map_coverage_stats  # noqa: E402
from runner import (  # noqa: E402
    _describe_inputs,
    _resolve_run_name,
    build_data_config,
    build_model_config,
    build_train_config,
    set_seed,
)
from src.config_loader import load_jsonc  # noqa: E402
from src.data.augmentation import AugmentConfig  # noqa: E402
from src.data.make_dataloader import build_base_dataset  # noqa: E402
from src.data.manga_split_dataset import MangaSplitDataset  # noqa: E402
from src.data.splits import filter_rows_by_split, write_split_csv  # noqa: E402
from src.metrics.plots import evaluate_map_predictions, write_metrics_csv  # noqa: E402
from src.models.wrapper import (  # noqa: E402
    MapGenerator,
    prepare_footprint_input,
    prepare_hr_imaging_input,
    prepare_imaging_input,
    prepare_redshift_input,
    prepare_spectrum_input,
    prepare_targets_and_masks,
)
from src.training.train import run_training  # noqa: E402


def _select_plateifus(
    base,
    *,
    source_split_csv: Path,
    n: int,
    seed: int,
) -> list[tuple[str, float, float]]:
    """
    Pick top-n train galaxies by footprint fill fraction, then mean valid pixels.

    Ranking by fill first avoids punishing small IFUs that are still well labelled.
    """
    train_rows = filter_rows_by_split(base.rows, source_split_csv, "train")
    print(
        f"Scoring {len(train_rows):,} train galaxies by fill fraction "
        f"+ mean valid pixels...",
        flush=True,
    )
    target_keys = tuple(base.target_keys)
    scored: list[tuple[str, float, float]] = []
    for row in tqdm(train_rows, desc="Select overfit galaxies", unit="gal", dynamic_ncols=True):
        gdir = base.data_root / row["galaxy_dir"]
        n_mean, _fp_n, fill = map_coverage_stats(
            gdir,
            target_source=base.target_source,
            target_keys=target_keys,
            min_snr=base.min_snr,
            require_sf_spaxel=base.require_sf_spaxel,
        )
        scored.append((row["plateifu"], fill, n_mean))

    rng = np.random.default_rng(seed)
    order = np.arange(len(scored))
    rng.shuffle(order)
    scored = [scored[i] for i in order]
    # Primary: fill_frac; secondary: mean valid pixels (both descending).
    scored.sort(key=lambda t: (t[1], t[2]), reverse=True)

    picked = [t for t in scored if t[1] > 0 and t[2] > 0][:n]
    if len(picked) < n:
        raise SystemExit(
            f"Only found {len(picked)} train galaxies with valid coverage "
            f"(requested {n}). Check data_root / target maps "
            f"(source={base.target_source!r})."
        )
    print(
        f"Selected {len(picked)} galaxies "
        f"(fill: {picked[-1][1]:.2%} .. {picked[0][1]:.2%}; "
        f"mean valid px: {picked[-1][2]:.0f} .. {picked[0][2]:.0f})",
        flush=True,
    )
    return picked


def _write_tiny_split(path: Path, plateifus: list[str]) -> None:
    # All rows are train; val/test intentionally empty — loaders built manually.
    write_split_csv(path, {p: "train" for p in plateifus})


@torch.no_grad()
def _train_fit_report(
    model: MapGenerator,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Masked L1 / MSE / 'near-perfect' fraction on the overfit set."""
    model.eval()
    map_keys = tuple(model.config.target_keys)
    per_map_l1 = {k: [] for k in map_keys}
    per_map_mse = {k: [] for k in map_keys}
    all_l1: list[float] = []
    all_mse: list[float] = []
    # Spaxel-level residuals for a rough "accuracy" sense-check.
    abs_err_parts: list[np.ndarray] = []
    abs_tgt_parts: list[np.ndarray] = []

    for batch in tqdm(loader, desc="Final train-set fit", unit="batch", dynamic_ncols=True):
        x = prepare_imaging_input(batch, model.config).to(device)
        x_hr = prepare_hr_imaging_input(batch, model.config)
        if x_hr is not None:
            x_hr = x_hr.to(device)
        footprint = prepare_footprint_input(batch, model.config)
        if footprint is not None:
            footprint = footprint.to(device)
        spec = prepare_spectrum_input(batch, model.config)
        if spec is not None:
            spec = spec.to(device)
        redshift = prepare_redshift_input(batch, model.config)
        if redshift is not None:
            redshift = redshift.to(device)
        targets, masks = prepare_targets_and_masks(batch, model.config)
        targets = targets.to(device)
        masks = masks.to(device)

        pred, _ = model.model(
            x,
            spectrum_flux=spec,
            footprint=footprint,
            x_hr=x_hr,
            redshift=redshift,
        )
        bsz = pred.shape[0]
        for i in range(bsz):
            sample_l1, sample_mse = [], []
            for ch, key in enumerate(map_keys):
                m = masks[i, ch] > 0
                if int(m.sum()) == 0:
                    sample_l1.append(float("nan"))
                    sample_mse.append(float("nan"))
                    continue
                err = pred[i, ch][m] - targets[i, ch][m]
                l1 = float(err.abs().mean().cpu())
                mse = float((err ** 2).mean().cpu())
                sample_l1.append(l1)
                sample_mse.append(mse)
                per_map_l1[key].append(l1)
                per_map_mse[key].append(mse)
                abs_err_parts.append(err.abs().detach().cpu().numpy())
                abs_tgt_parts.append(targets[i, ch][m].abs().detach().cpu().numpy())
            all_l1.append(float(np.nanmean(sample_l1)))
            all_mse.append(float(np.nanmean(sample_mse)))

    def _m(xs: list[float]) -> float:
        vals = [x for x in xs if np.isfinite(x)]
        return float(np.mean(vals)) if vals else float("nan")

    abs_err = np.concatenate(abs_err_parts) if abs_err_parts else np.array([])
    abs_tgt = np.concatenate(abs_tgt_parts) if abs_tgt_parts else np.array([])
    # Relative error threshold: |err| < 0.02 * (1 + |target|) counts as "near perfect".
    if abs_err.size:
        near = float(np.mean(abs_err < 0.02 * (1.0 + abs_tgt)))
        median_abs = float(np.median(abs_err))
    else:
        near = float("nan")
        median_abs = float("nan")

    out: dict[str, float] = {
        "masked_l1": _m(all_l1),
        "masked_mse": _m(all_mse),
        "median_abs_err": median_abs,
        "frac_near_perfect_2pct": near,
    }
    for k in map_keys:
        out[f"l1_{k}"] = _m(per_map_l1[k])
        out[f"mse_{k}"] = _m(per_map_mse[k])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Overfit a tiny MaNGA subset (train-only).")
    parser.add_argument("--config", type=Path, default=Path("config.jsonc"))
    parser.add_argument("--n-galaxies", type=int, default=32)
    parser.add_argument("--run-name", type=str, default="overfit_tiny")
    parser.add_argument("--autoinc", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--source-split",
        type=Path,
        default=None,
        help="Split CSV to sample train galaxies from (default: config data.split).",
    )
    args = parser.parse_args()

    user_cfg = load_jsonc(args.config)
    training_top = dict(user_cfg.get("training", {}) or {})
    data_top = dict(user_cfg.get("data", {}) or {})
    model_top = dict(user_cfg.get("model", {}) or {})

    # Force capacity-check settings.
    model_top["dropout"] = 0.0
    model_top["hr_attn_dropout"] = 0.0
    # HR is a separate ablation; overfit asks "can the backbone memorise?" only.
    model_top["use_hr_cross_attn"] = False
    model_top["use_hr_cross_attention"] = False
    data_top.setdefault("augmentation", {})
    data_top["augmentation"] = {
        **(data_top.get("augmentation") or {}),
        "enabled": False,
    }
    # No held-out eval during / after training.
    logging_top = dict(training_top.get("logging", {}) or {})
    logging_top["run_post_train_eval"] = False
    logging_top["eval_splits"] = ["train"]
    logging_top["eval_max_plot"] = min(args.n_galaxies, 16)
    training_top["logging"] = logging_top
    training_top["early_stopping"] = {
        "patience": 10**9,
        "start_epoch": 10**9,
    }
    opt = dict(training_top.get("optimizer", {}) or {})
    opt["weight_decay"] = 0.0
    opt["lr"] = float(args.lr)
    training_top["optimizer"] = opt
    # Constant LR — overfit doesn't need cosine decay.
    training_top["lr_schedule"] = {"name": "constant", "warmup_epochs": 0, "min_lr_ratio": 1.0}
    if args.epochs is not None:
        training_top["epochs"] = int(args.epochs)
    else:
        training_top.setdefault("epochs", 400)
    if args.device is not None:
        training_top["device"] = args.device
    if args.seed is not None:
        training_top["seed"] = int(args.seed)

    batching = dict(training_top.get("batching", {}) or {})
    if args.batch_size is not None:
        batching["train_batch_size"] = int(args.batch_size)
        batching["eval_batch_size"] = int(args.batch_size)
    else:
        # Fit the whole tiny set in one / few steps when possible.
        batching["train_batch_size"] = min(int(batching.get("train_batch_size", 16)), args.n_galaxies)
        # Keep eval == train: level-0 local HR xattn OOMs if eval packs the full tiny set.
        batching["eval_batch_size"] = batching["train_batch_size"]
    training_top["batching"] = batching

    imaging_resolution = model_top.get(
        "imaging_resolution", data_top.get("imaging_resolution", "aligned")
    )
    model_cfg = build_model_config(model_top, data_top, imaging_resolution=imaging_resolution)
    data_cfg = build_data_config(
        data_top, imaging_resolution=imaging_resolution, model_top=model_top
    )

    source_split = args.source_split or data_cfg.split_csv_path
    if not source_split.is_file():
        raise SystemExit(f"Source split CSV not found: {source_split}")

    save_root = Path(logging_top.get("root_dir", "runs/manga_maps"))
    run_name = _resolve_run_name(save_root, args.run_name, args.autoinc)
    train_cfg = build_train_config(training_top, run_name=run_name)
    # Trainer still needs a "val" loader; we reuse the train set (no held-out data).
    train_cfg.run_post_train_eval = False
    train_cfg.early_stop_patience = 10**9
    train_cfg.early_stop_start_epoch = 10**9

    set_seed(train_cfg.seed)
    base = build_base_dataset(data_cfg)
    picked = _select_plateifus(
        base,
        source_split_csv=source_split,
        n=args.n_galaxies,
        seed=train_cfg.seed,
    )
    plateifus = [p for p, _fill, _n in picked]

    run_dir = save_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tiny_split = run_dir / "tiny_split.csv"
    _write_tiny_split(tiny_split, plateifus)
    with (run_dir / "tiny_galaxies.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["plateifu", "fill_frac", "n_valid_mean"],
        )
        w.writeheader()
        for p, fill, n_mean in picked:
            w.writerow(
                {
                    "plateifu": p,
                    "fill_frac": f"{fill:.6f}",
                    "n_valid_mean": f"{n_mean:.3f}",
                }
            )

    data_cfg.split_csv_path = tiny_split
    data_cfg.augmentation = AugmentConfig(enabled=False)

    no_aug = AugmentConfig(enabled=False)
    ds_train = MangaSplitDataset(
        base, split="train", split_csv_path=tiny_split, augment=no_aug
    )
    # Same galaxies for the Trainer's "val" slot (monitor train loss only).
    ds_monitor = MangaSplitDataset(
        base, split="train", split_csv_path=tiny_split, augment=no_aug
    )

    num_workers = int(batching.get("num_workers", 0))
    pin_memory = bool(batching.get("pin_memory", torch.cuda.is_available()))
    loader_kwargs: dict[str, object] = {
        "num_workers": num_workers,
        "collate_fn": collate_manga_batch,
        "pin_memory": pin_memory,
    }
    if num_workers > 0 and sys.platform != "win32":
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    dl_train = DataLoader(
        ds_train,
        batch_size=train_cfg.train_batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    dl_monitor = DataLoader(
        ds_monitor,
        batch_size=train_cfg.eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    model = MapGenerator(model_cfg)

    # Snapshot includes the forced overfit overrides.
    snapshot = {
        "training": training_top,
        "data": {**data_top, "split": {**(data_top.get("split") or {}), "split_csv_path": str(tiny_split)}},
        "model": model_top,
        "overfit": {
            "n_galaxies": args.n_galaxies,
            "plateifus": plateifus,
            "source_split": str(source_split),
            "target_source": data_cfg.target_source,
            "target_keys": list(model_cfg.target_keys),
            "min_snr": data_cfg.min_snr,
            "galaxy_sf_flag": data_cfg.galaxy_sf_flag,
            "dropout": 0.0,
            "augmentation": False,
            "weight_decay": 0.0,
            "use_hr_cross_attn": False,
        },
    }

    print("=" * 60)
    print("MaNGA overfit-tiny (train-only capacity check)")
    print("=" * 60)
    print(f"  run          : {run_name}")
    print(f"  target_source: {data_cfg.target_source}")
    print(f"  target_keys  : {tuple(model_cfg.target_keys)}")
    if data_cfg.target_source == "phys":
        print(f"  min_snr      : {data_cfg.min_snr}  sf_flag={data_cfg.galaxy_sf_flag}")
    print(f"  n galaxies   : {len(plateifus)}  (top valid-pixel train subset)")
    print(f"  pixels/gal   : min={picked[-1][1]:,}  max={picked[0][1]:,}")
    print(f"  epochs       : {train_cfg.epochs}")
    print(f"  batch size   : {train_cfg.train_batch_size}")
    print(f"  dropout      : {model_cfg.dropout}  (HR cross-attn forced off)")
    print(f"  weight_decay : 0.0  aug=off  val/test=none")
    print(f"  architecture : {model_cfg.architecture}  head={model_cfg.output_head}")
    print(f"  inputs       : {_describe_inputs(model_cfg, data_cfg)}")
    print(f"  tiny split   : {tiny_split}")
    print("=" * 60)
    print(
        f"Starting training ({train_cfg.epochs} epochs, "
        f"{len(dl_train)} train batch(es)/epoch). "
        "Watch epoch loss lines below.",
        flush=True,
    )

    run_dirs = run_training(
        model,
        train_cfg,
        dl_train,
        dl_monitor,  # "val" = same train set
        dl_monitor,  # unused (post-eval off)
        dl_monitor,
        user_snapshot=snapshot,
    )

    print("Training finished. Computing final train-set fit report...", flush=True)
    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    report = _train_fit_report(model, dl_monitor, device)
    print("-" * 60)
    print("Final train-set fit (same N galaxies, no held-out data):")
    print(f"  masked L1              : {report['masked_l1']:.6e}")
    print(f"  masked MSE             : {report['masked_mse']:.6e}")
    print(f"  median |err|           : {report['median_abs_err']:.6e}")
    print(f"  frac |err|<2%(1+|y|)   : {report['frac_near_perfect_2pct']:.4f}")
    for key in model.config.target_keys:
        print(f"  L1[{key:16s}]: {report[f'l1_{key}']:.6e}")

    # Also dump standard per-galaxy MSE plots for visual inspection.
    plots_dir = Path(run_dirs["plots"])
    rows = evaluate_map_predictions(
        model,
        dl_monitor,
        device=device,
        map_keys=tuple(model.config.target_keys),
        plots_dir=plots_dir,
        split="train",
        max_plot=min(len(plateifus), train_cfg.eval_max_plot),
    )
    csv_path = Path(run_dirs["csv"]) / "train_overfit_metrics.csv"
    write_metrics_csv(rows, csv_path)
    report_path = Path(run_dirs["csv"]) / "overfit_summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  metrics csv  : {csv_path}")
    print(f"  summary json : {report_path}")
    print(f"  plots        : {plots_dir}")
    print("=" * 60)

    # Verdict vs typical full-data val L1 (~0.05). Overfit need not be
    # pixel-perfect (maps are soft / multi-modal); L1 << full-data baseline
    # already shows capacity is not the bottleneck.
    l1 = float(report["masked_l1"])
    near = float(report["frac_near_perfect_2pct"])
    if l1 < 1e-3 and near > 0.95:
        print("Verdict: near-perfect overfit — capacity looks fine.")
    elif l1 < 0.015:
        print("Verdict: strong train-set overfit (L1 well below ~0.05 full-data baseline).")
    elif l1 < 0.035:
        print(
            "Verdict: clear overfit / capacity OK — residuals remain "
            "(softer than truth is normal); not a capacity failure."
        )
    else:
        print(
            "Verdict: weak overfit — train L1 still near full-data levels; "
            "capacity / loss / labels may be limiting."
        )

    print(f"Done. Artifacts in {run_dirs['root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
