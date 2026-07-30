"""
HR-zero contribution diagnostic.

Forward a trained HR-cross-attn checkpoint twice on the same batches:
  1) normal hr_imaging
  2) hr_imaging replaced with zeros

Reports how much predictions / metrics change. Near-zero Δ means the model
is effectively ignoring the HR stream.

Usage:
  python scripts/hr_zero_contribution.py --run-name model_e_hr_xattn_5
  python scripts/hr_zero_contribution.py --run-name model_e_hr_xattn_5 --split val --max-batches 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runner import (  # noqa: E402
    build_data_config,
    build_model_config,
    build_train_config,
    make_manga_dataloaders,
    set_seed,
)
from src.models.wrapper import (  # noqa: E402
    MapGenerator,
    prepare_footprint_input,
    prepare_hr_imaging_input,
    prepare_imaging_input,
    prepare_spectrum_input,
    prepare_targets_and_masks,
)
from src.training.train import _load_checkpoint_state  # noqa: E402


def _masked_stats(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[float, float]:
    """Return (masked L1, masked MSE) averaged over valid spaxels."""
    m = mask > 0
    if int(m.sum()) == 0:
        return float("nan"), float("nan")
    err = pred[m] - target[m]
    return float(err.abs().mean().cpu()), float((err ** 2).mean().cpu())


@torch.no_grad()
def _forward_pair(model: MapGenerator, batch: dict, device: torch.device):
    cfg = model.config
    x = prepare_imaging_input(batch, cfg).to(device)
    footprint = prepare_footprint_input(batch, cfg)
    if footprint is not None:
        footprint = footprint.to(device)
    spec = prepare_spectrum_input(batch, cfg)
    if spec is not None:
        spec = spec.to(device)
    targets, masks = prepare_targets_and_masks(batch, cfg)
    targets = targets.to(device)
    masks = masks.to(device)

    x_hr = prepare_hr_imaging_input(batch, cfg)
    if x_hr is None:
        raise RuntimeError("Model does not use HR cross-attn; nothing to ablate.")
    x_hr = x_hr.to(device)
    x_hr_zero = torch.zeros_like(x_hr)

    pred_hr, _ = model.model(x, spectrum_flux=spec, footprint=footprint, x_hr=x_hr)
    pred_zero, _ = model.model(x, spectrum_flux=spec, footprint=footprint, x_hr=x_hr_zero)
    return pred_hr, pred_zero, targets, masks


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure HR contribution by zeroing hr_imaging.")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--root", type=Path, default=Path("runs/manga_maps"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--max-batches", type=int, default=0, help="0 = all batches")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    run_dir = args.root / args.run_name
    snap_path = run_dir / "config_used.json"
    if not snap_path.is_file():
        raise SystemExit(f"Run config not found: {snap_path}")

    snap = json.loads(snap_path.read_text(encoding="utf-8"))
    user_cfg = snap.get("user") or {}
    training_top = user_cfg.get("training", {})
    data_top = user_cfg.get("data", {})
    model_top = user_cfg.get("model", {})
    imaging_resolution = model_top.get(
        "imaging_resolution", data_top.get("imaging_resolution", "aligned")
    )
    model_cfg = build_model_config(model_top, data_top, imaging_resolution=imaging_resolution)
    data_cfg = build_data_config(
        data_top, imaging_resolution=imaging_resolution, model_top=model_top
    )
    train_cfg = build_train_config(training_top, run_name=args.run_name)

    if not model_cfg.use_hr_cross_attn:
        raise SystemExit(
            f"Run {args.run_name!r} has use_hr_cross_attn=false — HR-zero diag is N/A."
        )

    ckpt_path = args.checkpoint or (run_dir / "ckpts" / "best.pt")
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    device_str = args.device or train_cfg.device
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    set_seed(train_cfg.seed)

    dl_train, dl_val, dl_test, dl_train_ns = make_manga_dataloaders(
        data_cfg,
        {
            "train_batch_size": train_cfg.train_batch_size,
            "eval_batch_size": train_cfg.eval_batch_size,
            "num_workers": training_top.get("batching", {}).get("num_workers", 0),
        },
    )
    loaders = {"train": dl_train_ns, "val": dl_val, "test": dl_test}
    loader = loaders[args.split]

    model = MapGenerator(model_cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(_load_checkpoint_state(ckpt))
    model.eval()

    map_keys = tuple(model.config.target_keys)
    per_map_delta_l1: dict[str, list[float]] = {k: [] for k in map_keys}
    per_map_delta_mse: dict[str, list[float]] = {k: [] for k in map_keys}
    l1_hr_all: list[float] = []
    l1_zero_all: list[float] = []
    mse_hr_all: list[float] = []
    mse_zero_all: list[float] = []
    pred_delta_l1_all: list[float] = []
    n_galaxies = 0

    for bi, batch in enumerate(loader):
        if args.max_batches > 0 and bi >= args.max_batches:
            break
        pred_hr, pred_zero, targets, masks = _forward_pair(model, batch, device)
        bsz = pred_hr.shape[0]
        n_galaxies += bsz

        # |pred_hr - pred_zero| under the loss mask (does HR change the output?)
        for i in range(bsz):
            m_any = masks[i].sum(dim=0) > 0
            if int(m_any.sum()) == 0:
                continue
            d = (pred_hr[i] - pred_zero[i]).abs()
            # mean over maps at valid spaxels (union mask)
            pred_delta_l1_all.append(float(d[:, m_any].mean().cpu()))

            sample_l1_hr, sample_mse_hr = [], []
            sample_l1_z, sample_mse_z = [], []
            for ch, key in enumerate(map_keys):
                l1_h, mse_h = _masked_stats(pred_hr[i, ch], targets[i, ch], masks[i, ch])
                l1_z, mse_z = _masked_stats(pred_zero[i, ch], targets[i, ch], masks[i, ch])
                sample_l1_hr.append(l1_h)
                sample_mse_hr.append(mse_h)
                sample_l1_z.append(l1_z)
                sample_mse_z.append(mse_z)
                if np.isfinite(l1_h) and np.isfinite(l1_z):
                    per_map_delta_l1[key].append(l1_z - l1_h)
                if np.isfinite(mse_h) and np.isfinite(mse_z):
                    per_map_delta_mse[key].append(mse_z - mse_h)

            l1_hr_all.append(float(np.nanmean(sample_l1_hr)))
            l1_zero_all.append(float(np.nanmean(sample_l1_z)))
            mse_hr_all.append(float(np.nanmean(sample_mse_hr)))
            mse_zero_all.append(float(np.nanmean(sample_mse_z)))

    def _mean(xs: list[float]) -> float:
        vals = [x for x in xs if np.isfinite(x)]
        return float(np.mean(vals)) if vals else float("nan")

    mean_pred_delta = _mean(pred_delta_l1_all)
    mean_l1_hr = _mean(l1_hr_all)
    mean_l1_zero = _mean(l1_zero_all)
    mean_mse_hr = _mean(mse_hr_all)
    mean_mse_zero = _mean(mse_zero_all)

    print("=" * 60)
    print("HR-zero contribution diagnostic")
    print("=" * 60)
    print(f"  run          : {args.run_name}")
    print(f"  checkpoint   : {ckpt_path}")
    print(f"  split        : {args.split}  (n_galaxies={n_galaxies})")
    print(f"  HR levels    : {list(model_cfg.hr_cross_attn_levels)}")
    print(f"  HR n_down    : {model_cfg.hr_encoder_n_down}")
    print("-" * 60)
    print(f"  mean |pred_hr - pred_zero| (masked): {mean_pred_delta:.6e}")
    print(f"  masked L1   HR={mean_l1_hr:.6f}  zero={mean_l1_zero:.6f}  Δ(zero-HR)={mean_l1_zero - mean_l1_hr:+.6e}")
    print(f"  masked MSE  HR={mean_mse_hr:.6f}  zero={mean_mse_zero:.6f}  Δ(zero-HR)={mean_mse_zero - mean_mse_hr:+.6e}")
    print("-" * 60)
    print("  Per-map ΔL1 (zero - HR); positive ⇒ HR helped that map:")
    for key in map_keys:
        d = _mean(per_map_delta_l1[key])
        print(f"    {key:16s}  ΔL1={d:+.6e}")
    print("=" * 60)

    if mean_pred_delta < 1e-4:
        print("Verdict: HR stream is essentially ignored (|Δpred| ~ 0).")
    elif abs(mean_l1_zero - mean_l1_hr) < 1e-4:
        print("Verdict: HR changes predictions but does not improve L1.")
    else:
        print("Verdict: HR contributes a measurable metric change.")

    out = {
        "run_name": args.run_name,
        "split": args.split,
        "n_galaxies": n_galaxies,
        "mean_abs_pred_delta": mean_pred_delta,
        "masked_l1_hr": mean_l1_hr,
        "masked_l1_zero": mean_l1_zero,
        "masked_mse_hr": mean_mse_hr,
        "masked_mse_zero": mean_mse_zero,
        "per_map_delta_l1_zero_minus_hr": {k: _mean(v) for k, v in per_map_delta_l1.items()},
    }
    out_path = run_dir / "csv" / f"hr_zero_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
