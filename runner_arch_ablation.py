"""
Architecture ablation grid (no Optuna).

Trains a fixed set of big architecture variants (UNet vs UNet++, deep supervision,
spectrum on/off, HR cross-attn on/off), runs paper-style eval (RMSE/MAE/R²/…),
and writes comparison tables + plots.

Usage:
  # List cells
  python runner_arch_ablation.py --dry-run

  # Full core grid (default)
  python runner_arch_ablation.py --sweep-name arch_v1 --grid core

  # Subset / resume
  python runner_arch_ablation.py --sweep-name arch_v1 --only A_unet,C_unetpp_ds --skip-existing

  # Re-analyze completed runs only
  python runner_arch_ablation.py --sweep-name arch_v1 --analyze-only

Outputs under:
  runs/arch_ablation/<sweep-name>/
    configs/<cell>.json
    runs/<cell>/          # trainer artifacts + paper_eval/
    manifest.json
    analysis/{val,test}/  # summary CSV + comparison plots + README
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.ablation.arch_grid import ArchCell, GridName, apply_cell_to_config, get_grid
from src.config_loader import load_jsonc
from src.metrics.arch_ablation_plots import analyze_sweep


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _count_params(cfg_path: Path) -> int:
    """Build the model from a written config and count trainable params."""
    from runner import build_data_config, build_model_config
    from src.models.wrapper import MapGenerator

    user_cfg = load_jsonc(cfg_path)
    data_top = user_cfg.get("data", {})
    model_top = user_cfg.get("model", {})
    imaging_resolution = model_top.get(
        "imaging_resolution", data_top.get("imaging_resolution", "aligned")
    )
    model_cfg = build_model_config(model_top, data_top, imaging_resolution=imaging_resolution)
    model = MapGenerator(model_cfg)
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _run_cmd(cmd: list[str], *, cwd: Path) -> int:
    print(">>", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd))
    return int(proc.returncode)


def _cell_done(run_dir: Path) -> bool:
    return (run_dir / "ckpts" / "best.pt").is_file()


def _paper_eval_done(run_dir: Path, split: str) -> bool:
    return (run_dir / "paper_eval" / split / "summary_spaxel_stats.csv").is_file()


def _load_manifest(path: Path) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _upsert_cell_manifest(manifest: dict[str, Any], cell_row: dict[str, Any]) -> None:
    cells = manifest.setdefault("cells", [])
    for i, c in enumerate(cells):
        if c.get("name") == cell_row["name"]:
            cells[i] = {**c, **cell_row}
            return
    cells.append(cell_row)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train + evaluate a fixed architecture ablation grid (no Optuna)."
    )
    p.add_argument("--config", type=Path, default=Path("config.jsonc"), help="Base config")
    p.add_argument("--sweep-name", type=str, default="arch_v1")
    p.add_argument("--sweep-root", type=Path, default=Path("runs/arch_ablation"))
    p.add_argument("--grid", choices=("core", "extended"), default="core")
    p.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated cell names to run (default: all in --grid)",
    )
    p.add_argument("--skip-existing", action="store_true", help="Skip cells with best.pt")
    p.add_argument("--dry-run", action="store_true", help="Print cells and exit")
    p.add_argument("--analyze-only", action="store_true", help="Skip train/eval; only analyze")
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; run paper_eval + analysis on existing checkpoints",
    )
    p.add_argument("--splits", type=str, default="val,test", help="Eval splits for paper_eval")
    p.add_argument("--device", type=str, default=None, help="Override training.device")
    p.add_argument("--epochs", type=int, default=None, help="Override training.epochs")
    p.add_argument("--seed", type=int, default=None, help="Override training.seed")
    p.add_argument(
        "--baseline",
        type=str,
        default="C_unetpp_ds",
        help="Baseline cell name for delta plots",
    )
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument(
        "--paper-eval-batch-size",
        type=int,
        default=16,
        help="Batch size for paper_eval.py",
    )
    p.add_argument(
        "--max-spaxels",
        type=int,
        default=500_000,
        help="Max spaxels/channel kept in paper_eval plots",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = Path(__file__).resolve().parent
    base_cfg = load_jsonc(args.config)
    grid_name: GridName = args.grid  # type: ignore[assignment]
    cells = get_grid(grid_name)

    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        cells = [c for c in cells if c.name in wanted]
        missing = wanted - {c.name for c in cells}
        if missing:
            raise SystemExit(f"Unknown cell name(s): {sorted(missing)}")

    sweep_dir = args.sweep_root / args.sweep_name
    runs_root = sweep_dir / "runs"
    configs_dir = sweep_dir / "configs"
    manifest_path = sweep_dir / "manifest.json"

    print("=" * 64)
    print("Architecture ablation grid")
    print("=" * 64)
    print(f"  sweep     : {sweep_dir}")
    print(f"  grid      : {args.grid}  ({len(cells)} cells)")
    print(f"  base cfg  : {args.config}")
    for c in cells:
        print(
            f"    - {c.name:24s} arch={c.architecture:6s} ds={str(c.deep_supervision):5s} "
            f"spec={c.spectrum:3s} hr={str(c.hr_cross_attn):5s}  {c.note}"
        )
    print("=" * 64)

    if args.dry_run:
        return 0

    sweep_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(manifest_path)
    manifest.update(
        {
            "sweep_name": args.sweep_name,
            "grid": args.grid,
            "base_config": str(args.config),
            "updated_at": _utc_now(),
            "created_at": manifest.get("created_at", _utc_now()),
        }
    )
    # Ensure cell stubs exist for analysis even before training.
    for c in cells:
        _upsert_cell_manifest(manifest, c.to_dict() | {"status": "pending"})
    _write_json(manifest_path, manifest)

    if args.analyze_only:
        for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
            out = analyze_sweep(sweep_dir, split=split, baseline=args.baseline)
            print(f"Analysis -> {out}")
        return 0

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    failures: list[str] = []

    for cell in cells:
        cell_cfg_path = configs_dir / f"{cell.name}.json"
        run_dir = runs_root / cell.name

        cfg = apply_cell_to_config(base_cfg, cell)
        # Co-locate all runs under this sweep.
        cfg.setdefault("training", {}).setdefault("logging", {})
        cfg["training"]["logging"]["root_dir"] = str(runs_root).replace("\\", "/")
        # Keep post-train mse plots; paper_eval adds RMSE/R².
        cfg["training"]["logging"]["run_post_train_eval"] = True
        cfg["training"]["logging"]["eval_splits"] = ["val", "test"]
        if args.device is not None:
            cfg["training"]["device"] = args.device
        if args.epochs is not None:
            cfg["training"]["epochs"] = int(args.epochs)
        if args.seed is not None:
            cfg["training"]["seed"] = int(args.seed)

        _write_json(cell_cfg_path, cfg)

        n_params: int | None = None
        try:
            n_params = _count_params(cell_cfg_path)
        except Exception as exc:
            print(f"[{cell.name}] warning: could not count params: {exc}", flush=True)

        cell_row: dict[str, Any] = {
            **cell.to_dict(),
            "config_path": str(cell_cfg_path),
            "run_dir": str(run_dir),
            "n_params": n_params,
        }

        skip_train = args.eval_only or (args.skip_existing and _cell_done(run_dir))
        if skip_train and not _cell_done(run_dir):
            print(f"[{cell.name}] no checkpoint; skipping (eval-only / skip-existing)", flush=True)
            cell_row["status"] = "missing_ckpt"
            _upsert_cell_manifest(manifest, cell_row)
            _write_json(manifest_path, manifest)
            failures.append(cell.name)
            continue

        if not skip_train:
            print(f"\n[{cell.name}] training ...", flush=True)
            t0 = time.time()
            cell_row["status"] = "training"
            cell_row["started_at"] = _utc_now()
            _upsert_cell_manifest(manifest, cell_row)
            _write_json(manifest_path, manifest)

            rc = _run_cmd(
                [
                    args.python,
                    str(repo / "runner.py"),
                    "--config",
                    str(cell_cfg_path),
                    "--run-name",
                    cell.name,
                ],
                cwd=repo,
            )
            cell_row["train_seconds"] = round(time.time() - t0, 1)
            cell_row["train_returncode"] = rc
            if rc != 0 or not _cell_done(run_dir):
                cell_row["status"] = "train_failed"
                _upsert_cell_manifest(manifest, cell_row)
                _write_json(manifest_path, manifest)
                failures.append(cell.name)
                print(f"[{cell.name}] TRAIN FAILED (rc={rc})", flush=True)
                continue
            cell_row["status"] = "trained"
            print(f"[{cell.name}] train done in {cell_row['train_seconds']}s", flush=True)
        else:
            print(f"[{cell.name}] skip train (existing checkpoint)", flush=True)
            cell_row["status"] = "trained"

        # Paper eval (RMSE / MAE / R² / …)
        eval_ok = True
        for split in splits:
            if args.skip_existing and _paper_eval_done(run_dir, split):
                print(f"[{cell.name}] skip paper_eval split={split} (exists)", flush=True)
                continue
            print(f"[{cell.name}] paper_eval split={split} ...", flush=True)
            rc = _run_cmd(
                [
                    args.python,
                    str(repo / "paper_eval.py"),
                    "--root",
                    str(runs_root),
                    "--run-name",
                    cell.name,
                    "--split",
                    split,
                    "--device",
                    str(cfg["training"].get("device", "cuda")),
                    "--batch-size",
                    str(args.paper_eval_batch_size),
                    "--max-spaxels",
                    str(args.max_spaxels),
                ],
                cwd=repo,
            )
            if rc != 0 or not _paper_eval_done(run_dir, split):
                eval_ok = False
                print(f"[{cell.name}] paper_eval FAILED split={split} rc={rc}", flush=True)

        cell_row["status"] = "complete" if eval_ok else "eval_failed"
        cell_row["finished_at"] = _utc_now()
        if not eval_ok:
            failures.append(cell.name)
        _upsert_cell_manifest(manifest, cell_row)
        _write_json(manifest_path, manifest)

    # Comparison analysis across completed cells.
    for split in splits:
        out = analyze_sweep(sweep_dir, split=split, baseline=args.baseline)
        print(f"Analysis -> {out}")

    print("=" * 64)
    if failures:
        print(f"Finished with failures: {failures}")
        return 1
    print(f"Done. Sweep artifacts in {sweep_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
