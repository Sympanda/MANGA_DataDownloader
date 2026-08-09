"""CLI: export physical-property Pipe3D maps to amara_phys_maps.npz."""
from __future__ import annotations

from pathlib import Path
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed

from manga_prep.targets.pipe3d_phys_maps import (
    AMARA_PHYS_MAPS_META,
    AMARA_PHYS_MAPS_NPZ,
    DEFAULT_TARGET_SIZE,
    discover_pipe3d_cubes,
    max_native_shape,
    write_amara_phys_maps,
    write_collaborator_phys_maps,
)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description=(
            "Export physical-property Pipe3D maps (ages, Z, kinematics, SFR, …) "
            "to amara_phys_maps.npz (does not overwrite legacy amara_maps.npz)."
        )
    )
    parser.add_argument(
        "plateifu",
        nargs="*",
        help="Optional plate-IFU IDs, e.g. 8485-1901. If omitted, all local Pipe3D cubes are used.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("manga_sdss_fits"))
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Write amara_phys_maps.npz into each manga_sdss_fits/<plate_ifu> folder.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("collaborator_pipe3d_phys_maps"),
        help="Output root when not using --in-place.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=DEFAULT_TARGET_SIZE,
        help=f"Padded square map size. Default: {DEFAULT_TARGET_SIZE}.",
    )
    parser.add_argument(
        "--auto-target-size",
        action="store_true",
        help="Pad to the largest native size in the selected local cubes.",
    )
    parser.add_argument(
        "--no-clip",
        action="store_true",
        help="Do not clip transformed values before scaling. Usually leave clipping on.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip galaxies that already have an exported .npz file.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes across galaxies. Default: 1.",
    )
    parser.add_argument(
        "--include-derived",
        action="store_true",
        help="Also export BPT-SF Halpha SFR and PP04 O3N2 gas metallicity maps with errors.",
    )
    parser.add_argument(
        "--drpall",
        type=Path,
        default=None,
        help="Path to drpall FITS. Required with --include-derived for redshift/distance.",
    )
    parser.add_argument(
        "--snr-min",
        type=float,
        default=3.0,
        help="S/N threshold for BPT/SFR/metallicity and per-map *_snr_mask. Default: 3.",
    )
    parser.add_argument(
        "--arcsec-per-spaxel",
        type=float,
        default=0.5,
        help="MaNGA spaxel size used for Sigma SFR area conversion. Default: 0.5.",
    )
    return parser.parse_args(argv)


def path_for_plateifu(data_root, plateifu):
    plateifu = str(plateifu).strip().replace("_", "-")
    plate, ifu = plateifu.split("-")
    return Path(data_root) / f"{plate}_{ifu}" / f"manga-{plateifu}.Pipe3D.cube.fits.gz"


def selected_cubes(data_root, plateifus):
    if plateifus:
        paths = [path_for_plateifu(data_root, plateifu) for plateifu in plateifus]
    else:
        paths = discover_pipe3d_cubes(data_root)
    missing = [path for path in paths if not path.exists()]
    if missing:
        formatted = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing Pipe3D cube(s):\n{formatted}")
    return paths


def export_one(job):
    (
        path,
        out_dir,
        target_shape,
        clip,
        in_place,
        skip_existing,
        include_derived,
        drpall_path,
        snr_min,
        arcsec_per_spaxel,
    ) = job

    if in_place:
        npz_path = path.parent / AMARA_PHYS_MAPS_NPZ
        metadata_path = path.parent / AMARA_PHYS_MAPS_META
    else:
        plateifu = path.name.replace("manga-", "").split(".Pipe3D")[0]
        size_label = f"{int(target_shape[0])}x{int(target_shape[1])}"
        npz_path = Path(out_dir) / plateifu / f"{plateifu}_pipe3d_phys_maps_{size_label}.npz"
        metadata_path = npz_path.with_name(npz_path.stem + "_metadata.json")

    if skip_existing and npz_path.exists():
        return {
            "plateifu": path.name.replace("manga-", "").split(".Pipe3D")[0],
            "native_ny": None,
            "native_nx": None,
            "native_spaxel_count": None,
            "target_ny": int(target_shape[0]),
            "target_nx": int(target_shape[1]),
            "npz": npz_path,
            "metadata": metadata_path,
            "n_sf_bpt": "",
            "n_sfr_valid": "",
            "skipped": True,
        }

    kwargs = dict(
        target_shape=target_shape,
        clip=clip,
        include_derived=include_derived,
        drpall_path=drpall_path,
        snr_min=snr_min,
        arcsec_per_spaxel=arcsec_per_spaxel,
    )
    if in_place:
        result = write_amara_phys_maps(path, galaxy_dir=path.parent, **kwargs)
    else:
        result = write_collaborator_phys_maps(path, out_dir=out_dir, **kwargs)

    derived = result.get("derived_science") or {}
    return {
        "plateifu": result["plateifu"],
        "native_ny": result["native_shape"][0],
        "native_nx": result["native_shape"][1],
        "native_spaxel_count": result["native_spaxel_count"],
        "target_ny": result["target_shape"][0],
        "target_nx": result["target_shape"][1],
        "npz": result["npz"],
        "metadata": result["metadata"],
        "n_sf_bpt": derived.get("n_sf_bpt", ""),
        "n_sfr_valid": derived.get("n_sfr_valid", ""),
        "skipped": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.include_derived and args.drpall is None:
        raise ValueError("--drpall is required with --include-derived.")
    paths = selected_cubes(args.data_root, args.plateifu)
    if not paths:
        raise FileNotFoundError(f"No Pipe3D cubes found under {args.data_root}")
    if args.auto_target_size:
        target_shape = max_native_shape(paths)
    else:
        target_shape = (args.target_size, args.target_size)

    if not args.in_place:
        args.out.mkdir(parents=True, exist_ok=True)
    jobs = [
        (
            path,
            args.out,
            target_shape,
            not args.no_clip,
            args.in_place,
            args.skip_existing,
            args.include_derived,
            args.drpall,
            args.snr_min,
            args.arcsec_per_spaxel,
        )
        for path in paths
    ]
    rows = []
    if int(args.workers) <= 1:
        for job in jobs:
            row = export_one(job)
            rows.append(row)
            action = "Skipped" if row.get("skipped") else "Wrote"
            print(f"{action} {row['plateifu']}: {row['npz']}")
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            futures = [executor.submit(export_one, job) for job in jobs]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                action = "Skipped" if row.get("skipped") else "Wrote"
                print(f"{action} {row['plateifu']}: {row['npz']}")

    rows = sorted(rows, key=lambda row: row["plateifu"])
    manifest_path = (args.data_root if args.in_place else args.out) / "amara_phys_maps_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
