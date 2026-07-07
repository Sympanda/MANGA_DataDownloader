from pathlib import Path
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed

from manga_prep.targets.pipe3d_maps import (
    DEFAULT_TARGET_SIZE,
    discover_pipe3d_cubes,
    max_native_shape,
    patch_amara_footprint,
    write_amara_maps,
    write_collaborator_maps,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export clipped 0-1 Pipe3D maps for collaborator map work."
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
        help="Write amara_maps.npz into each manga_sdss_fits/<plate_ifu> folder.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("collaborator_pipe3d_maps"),
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
        "--footprint-only",
        action="store_true",
        help="Only refresh native_footprint_mask and *_loss_mask in existing amara_maps.npz files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes across galaxies. Default: 1.",
    )
    return parser.parse_args()


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
    path, out_dir, target_shape, clip, in_place, skip_existing, footprint_only = job
    if in_place:
        npz_path = path.parent / "amara_maps.npz"
    else:
        plateifu = path.name.replace("manga-", "").split(".Pipe3D")[0]
        size_label = f"{int(target_shape[0])}x{int(target_shape[1])}"
        npz_path = Path(out_dir) / plateifu / f"{plateifu}_pipe3d_direct_maps_{size_label}.npz"

    if footprint_only:
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing existing NPZ for footprint refresh: {npz_path}")
        result = patch_amara_footprint(path, npz_path, target_shape=target_shape)
        return {
            "plateifu": path.name.replace("manga-", "").split(".Pipe3D")[0],
            "native_ny": result["native_shape"][0],
            "native_nx": result["native_shape"][1],
            "native_spaxel_count": None,
            "target_ny": int(target_shape[0]),
            "target_nx": int(target_shape[1]),
            "npz": result["npz"],
            "metadata": npz_path.with_name("amara_maps_metadata.json"),
            "skipped": False,
        }

    if skip_existing and npz_path.exists():
        metadata_path = npz_path.with_name(npz_path.stem + "_metadata.json")
        if in_place:
            metadata_path = npz_path.with_name("amara_maps_metadata.json")
        return {
            "plateifu": path.name.replace("manga-", "").split(".Pipe3D")[0],
            "native_ny": None,
            "native_nx": None,
            "native_spaxel_count": None,
            "target_ny": int(target_shape[0]),
            "target_nx": int(target_shape[1]),
            "npz": npz_path,
            "metadata": metadata_path,
            "skipped": True,
        }

    if in_place:
        result = write_amara_maps(
            path,
            galaxy_dir=path.parent,
            target_shape=target_shape,
            clip=clip,
        )
    else:
        result = write_collaborator_maps(
            path,
            out_dir=out_dir,
            target_shape=target_shape,
            clip=clip,
        )
    return {
        "plateifu": result["plateifu"],
        "native_ny": result["native_shape"][0],
        "native_nx": result["native_shape"][1],
        "native_spaxel_count": result["native_spaxel_count"],
        "target_ny": result["target_shape"][0],
        "target_nx": result["target_shape"][1],
        "npz": result["npz"],
        "metadata": result["metadata"],
        "skipped": False,
    }


def main():
    args = parse_args()
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
        (path, args.out, target_shape, not args.no_clip, args.in_place, args.skip_existing, args.footprint_only)
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
    manifest_path = (args.data_root if args.in_place else args.out) / "amara_maps_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
