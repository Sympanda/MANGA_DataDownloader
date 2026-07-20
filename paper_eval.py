"""
Generate paper-style evaluation plots for a trained run or ensemble.

Usage:
  python paper_eval.py --run-name manga_unetpp_v10
  python paper_eval.py --run-name ens_v1 --split test --device cuda:1
  python paper_eval.py --run-name ens_v1/members/member_00
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.metrics.paper_eval import run_paper_eval


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Paper eval: obs-vs-pred, summary stats, calibration (single or ensemble)."
    )
    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Run folder under --root (e.g. manga_unetpp_v10, ens_v1, ens_v1/members/member_00)",
    )
    parser.add_argument("--root", type=Path, default=Path("runs/manga_maps"))
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-spaxels",
        type=int,
        default=500_000,
        help="Max spaxels per channel kept for plots (subsampled if larger)",
    )
    parser.add_argument("--limit-batches", type=int, default=None, help="Debug: cap dataloader batches")
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=200,
        help="Bootstrap resamples for calibration curve error bands",
    )
    args = parser.parse_args(argv)

    out_dir = run_paper_eval(
        save_root=args.root,
        run_name=args.run_name,
        split=args.split,
        device=args.device,
        batch_size=args.batch_size,
        max_spaxels=args.max_spaxels,
        limit_batches=args.limit_batches,
        n_bootstrap=args.n_bootstrap,
    )
    print(f"Paper eval complete -> {out_dir}")
    for p in sorted(out_dir.glob("*")):
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
