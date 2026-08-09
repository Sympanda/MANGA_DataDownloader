from pathlib import Path
import argparse
import csv

from global_flags import discover_local_plateifus, make_global_flag_rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export galaxy-level Pipe3D global BPT/star-forming flags."
    )
    parser.add_argument(
        "--pipe3d-catalog",
        type=Path,
        required=True,
        help="Path to SDSS17Pipe3D_v3_1_1.fits.",
    )
    parser.add_argument("--out", type=Path, default=Path("pipe3d_global_flags.csv"))
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Optional local Pipe3D cube root. Used with --local-only.",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Only keep galaxies with local Pipe3D cubes under --data-root.",
    )
    parser.add_argument(
        "--max-ratio-err",
        type=float,
        default=0.3,
        help="Maximum global log-ratio error for global_sf_strict. Default: 0.3 dex.",
    )
    parser.add_argument(
        "--min-ha-ew-emission",
        type=float,
        default=3.0,
        help="Minimum positive-emission Halpha EW for global_sf_strict. Default: 3 Angstrom.",
    )
    parser.add_argument(
        "--min-ha-ew-snr",
        type=float,
        default=3.0,
        help="Minimum positive-emission Halpha EW S/N for global_sf_strict. Default: 3.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    plateifu_filter = None
    if args.local_only:
        if args.data_root is None:
            raise ValueError("--data-root is required with --local-only.")
        plateifu_filter = discover_local_plateifus(args.data_root)

    rows = make_global_flag_rows(
        args.pipe3d_catalog,
        plateifu_filter=plateifu_filter,
        max_ratio_err=args.max_ratio_err,
        min_ha_ew_emission=args.min_ha_ew_emission,
        min_ha_ew_snr=args.min_ha_ew_snr,
    )
    if not rows:
        raise ValueError("No rows matched the requested selection.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    n_sf = sum(row["global_bpt_sf"] for row in rows)
    n_strict = sum(row["global_bpt_sf_strict"] for row in rows)
    n_ew_strict = sum(row["global_sf_ew_strict"] for row in rows)
    n_agn = sum(row["global_bpt_agn"] for row in rows)
    n_comp = sum(row["global_bpt_comp"] for row in rows)
    print(f"Wrote {len(rows)} galaxies to {args.out}")
    print(
        f"Global BPT SF: {n_sf}; strict BPT SF: {n_strict}; "
        f"strict BPT+EW SF: {n_ew_strict}; composite: {n_comp}; AGN-like: {n_agn}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
