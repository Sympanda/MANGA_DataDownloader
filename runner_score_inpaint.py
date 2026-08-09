"""
RePaint inpainting diagnostic on a trained score generator (no training).

Drops a fraction of labelled Ha pixels, keeps the rest as known truth, and
fills the holes with masked DDIM (RePaint). Multi-row panels: different drop
%%, bottom row = 0% drop (100% known).

Usage:
  python runner_score_inpaint.py --run-name score_gen_ha99_2 --max-plot 8
  python runner_score_inpaint.py --run-name score_gen_ha99_2 --drop-fracs 0.5,0.25,0.1,0.0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from runner import build_data_config, build_train_config
from runner_score_generator import _build_score_model
from src.config_loader import load_jsonc
from src.data.score_dataloaders import make_score_dataloaders
from src.metrics.inpaint_plots import (
    heldout_rmse,
    make_random_known_mask,
    neighbor_average_fill,
    plot_inpaint_drop_panel,
)
from src.metrics.plots import write_metrics_csv
from src.metrics.residual_plots import _move_batch_to_device
from src.models.map_score import ScoreNormStats
from src.training.train import _load_checkpoint_state


def _parse_fracs(raw: str) -> list[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("empty --drop-fracs")
    # Ensure 0% drop (100% known) is last.
    vals = [v for v in vals if v > 1e-12]
    vals = sorted(vals, reverse=True) + [0.0]
    return vals


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RePaint inpaint diagnostic (inference only).")
    parser.add_argument("--config", type=Path, default=Path("config_score_generator.jsonc"))
    parser.add_argument("--run-name", type=str, default="score_gen_ha99_2")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--max-plot", type=int, default=8)
    parser.add_argument("--max-galaxies", type=int, default=None)
    parser.add_argument(
        "--drop-fracs",
        type=str,
        default="0.5,0.25,0.1,0.0",
        help="Comma-separated fractions of labelled pixels to drop (0 = 100%% known).",
    )
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--ddim-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Use EMA weights (default: live denoiser; EMA has been unreliable).",
    )
    args = parser.parse_args(argv)

    live_cfg = load_jsonc(args.config)
    training_top = live_cfg.get("training", {})
    save_root = Path(training_top.get("logging", {}).get("root_dir", "runs/score_generator"))
    run_dir = save_root / args.run_name
    snap = json.loads((run_dir / "config_used.json").read_text(encoding="utf-8"))
    user_cfg = snap.get("user") or {}
    score_top = dict(user_cfg.get("score", {}))
    # Legacy generator ckpts were trained with label_mask conditioning.
    if "condition_on_label_mask" not in score_top:
        score_top["condition_on_label_mask"] = True
    user_cfg["score"] = score_top

    train_cfg = build_train_config(user_cfg.get("training", training_top), run_name=args.run_name)
    if args.device is not None:
        train_cfg.device = str(args.device)

    data_top = user_cfg.get("data", {})
    model_top = user_cfg.get("model", {})
    data_cfg = build_data_config(data_top, imaging_resolution="aligned", model_top=model_top)
    data_cfg.use_spectrum = bool(data_top.get("use_spectrum", True))

    norm_snap = score_top.get("score_norm")
    if not norm_snap:
        raise SystemExit("config_used.json missing score.score_norm")
    score_norm = ScoreNormStats.from_dict(norm_snap)

    _, dl_val, dl_test, dl_train_ns, _ = make_score_dataloaders(
        data_cfg,
        {
            "train_batch_size": train_cfg.train_batch_size,
            "eval_batch_size": 1,  # one galaxy at a time for clear drop masks
            "num_workers": user_cfg.get("training", {}).get("batching", {}).get("num_workers", 0),
        },
        coverage_csv=score_top.get("coverage_csv", "runs/dataset_audit/galaxy_coverage_meta.csv"),
        min_coverage_pct=float(score_top.get("min_coverage_pct", 99.0)),
        feature=str(score_top.get("feature", "ha_flux")),
        use_stratified_weights=False,
    )
    loaders = {"train": dl_train_ns, "val": dl_val, "test": dl_test}
    loader = loaders[args.split]

    model = _build_score_model(user_cfg, score_norm=score_norm, force_mode="generator")
    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    ckpt_path = args.checkpoint or (run_dir / "ckpts" / "best.pt")
    model.load_state_dict(
        _load_checkpoint_state(torch.load(ckpt_path, map_location=device, weights_only=False))
    )
    model.eval()

    drop_fracs = _parse_fracs(args.drop_fracs)
    n_samples = int(args.n_samples)
    ddim_steps = int(args.ddim_steps or score_top.get("ddim_steps", 50))
    max_plot = int(args.max_plot)
    max_galaxies = int(args.max_galaxies) if args.max_galaxies is not None else max_plot
    use_ema = bool(args.use_ema)

    out_dir = run_dir / "inpaint"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RePaint inpaint diagnostic (inference only)")
    print("=" * 60)
    print(f"  run        : {run_dir}")
    print(f"  out        : {out_dir}")
    print(f"  split      : {args.split}")
    print(f"  drop_fracs : {drop_fracs}  (last = 100% known)")
    print(f"  use_ema    : {use_ema}")
    print(f"  cond_label : {model.condition_on_label_mask}")
    print("=" * 60)

    rows_csv: list[dict[str, float | str]] = []
    n_done = 0
    for batch in loader:
        if n_done >= max_galaxies:
            break
        batch = _move_batch_to_device(batch, device)
        plateifu = str(batch["plateifu"][0])
        from src.models.input_prep import prepare_targets_and_masks

        _, full_label, footprint, base_ha = model._prepare_clean_map(batch)
        label_np = full_label[0, 0].detach().cpu().numpy()
        fp_np = footprint[0, 0].detach().cpu().numpy()
        targets_t, _ = prepare_targets_and_masks(batch, model.config)
        tgt_np = targets_t[0, 0].detach().cpu().numpy()
        base_np = None if base_ha is None else base_ha[0].detach().cpu().numpy()
        b0 = None if base_np is None else (base_np[0] if base_np.ndim == 3 else base_np)

        panel_rows: list[dict[str, object]] = []
        for drop in drop_fracs:
            rng = np.random.default_rng(args.seed + hash(plateifu) % 10_000 + int(1000 * drop))
            known_np = make_random_known_mask(label_np, drop_frac=drop, rng=rng)
            known = torch.from_numpy(known_np).to(device=device).view(1, 1, *known_np.shape)

            out = model.sample_inpaint(
                batch,
                known_mask=known,
                n_samples=n_samples,
                ddim_steps=ddim_steps,
                seed=args.seed,
                use_ema=use_ema,
            )
            held = out["heldout_mask"][0, 0].detach().cpu().numpy()
            pred = out["predictive_mean"][0, 0].detach().cpu().numpy()

            # Neighbor-average baseline: fill held-out from known, priority = most neighbours.
            neigh = neighbor_average_fill(tgt_np, known_np, footprint=fp_np)

            rmse_inpaint = heldout_rmse(pred, tgt_np, held)
            rmse_neigh = heldout_rmse(neigh, tgt_np, held)
            rmse_unet = (
                heldout_rmse(b0, tgt_np, held) if b0 is not None else float("nan")
            )

            panel_rows.append(
                {
                    "drop_frac": drop,
                    "known_mask": known_np,
                    "heldout_mask": held,
                    "pred_mean": pred,
                    "neighbor_fill": neigh,
                    "rmse_heldout_inpaint": rmse_inpaint,
                    "rmse_heldout_unet": rmse_unet,
                    "rmse_heldout_neighbor": rmse_neigh,
                }
            )
            rows_csv.append(
                {
                    "plateifu": plateifu,
                    "split": args.split,
                    "drop_frac": float(drop),
                    "keep_pct": float(100.0 * (1.0 - drop)),
                    "rmse_heldout_inpaint": rmse_inpaint,
                    "rmse_heldout_unet": rmse_unet,
                    "rmse_heldout_neighbor": rmse_neigh,
                    "n_heldout": int((held > 0).sum()),
                    "n_known": int((known_np > 0).sum()),
                }
            )

        if n_done < max_plot:
            plot_inpaint_drop_panel(
                plateifu=plateifu,
                target=tgt_np,
                base=base_np,
                rows=panel_rows,
                out_path=plots_dir / f"{args.split}_{plateifu.replace('-', '_')}.png",
            )
            print(f"  wrote {plots_dir / (args.split + '_' + plateifu.replace('-', '_') + '.png')}")

        n_done += 1

    csv_path = out_dir / f"{args.split}_inpaint_metrics.csv"
    write_metrics_csv(rows_csv, csv_path)
    print(f"Done. metrics -> {csv_path}")
    print(f"Plots -> {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
