"""
Bulk-download MaNGA DR17 data for many/all galaxies.

This orchestrates the per-target MaNGA pipeline (same products as
download_manga_sdss.py: DRP LOGCUBE, DAP MAPS, DAP model LOGCUBE, Pipe3D VAC).

SDSS and Legacy imaging cutouts are separate: run download_sdss_cutouts.py
and download_legacy_cutouts.py after MaNGA files exist under each galaxy folder.

It is resumable: existing files are skipped.

Examples:
  # Dry run first 5 targets
  python download_all_manga.py --limit 5 --dry-run

  # Full MaNGA DR17 for all targets in DRPall
  python download_all_manga.py
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from astropy.io import fits

from . import manga_sdss as dm


DRPALL_URL = f"{dm.BASE_DRP}/drpall-{dm.DRPVER}.fits"


def to_str(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore").strip()
    return str(x).strip()


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "manga-bulk-downloader/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    dest.write_bytes(data)


def ensure_drpall(path: Path) -> Path:
    if path.exists():
        return path
    print(f"Downloading DRPall catalog: {DRPALL_URL}")
    download_file(DRPALL_URL, path)
    print(f"saved: {path}")
    return path


def load_plateifus_from_drpall(drpall_path: Path) -> list[str]:
    with fits.open(drpall_path, memmap=True) as h:
        data = h[1].data
        names = [n.upper() for n in data.columns.names]
        if "PLATEIFU" not in names:
            raise RuntimeError(f"PLATEIFU column not found in {drpall_path}")
        col_name = data.columns.names[names.index("PLATEIFU")]
        vals = [to_str(v) for v in data[col_name]]
    # Deduplicate while preserving order
    seen = set()
    out = []
    for v in vals:
        if not v or "-" not in v:
            continue
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def planned_manga_files(plate: str, ifu: str, daptype: str) -> list[tuple[str, str]]:
    return dm.urls_for_plateifu(
        plate,
        ifu,
        daptype,
        want_maps=True,
        want_drp_logcube=True,
        want_drp_lincube=False,
        want_drp_logrss=False,
        want_drp_linrss=False,
        want_dap_logcube=True,
        want_pipe3d_vac=True,
    )


def maybe_download_with_retry(
    url: str, dest: Path, retries: int, *, verbose: bool = True
) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            dm.download_url(url, dest, verbose=verbose)
            return
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            last_err = e
            print(f"  attempt {attempt}/{retries} failed: {e}", file=sys.stderr)
            time.sleep(min(5 * attempt, 20))
    raise RuntimeError(f"failed after {retries} attempts: {url} ({last_err})")


def process_one(
    plateifu: str,
    *,
    out_root: Path,
    daptype: str,
    retries: int,
    file_workers: int,
    dry_run: bool,
) -> tuple[int, int]:
    plate, ifu = dm.parse_plateifu(plateifu)
    gal_dir = out_root / f"{plate}_{ifu}"
    pairs = planned_manga_files(plate, ifu, daptype)

    downloaded = 0
    skipped = 0
    print(f"\n[{plateifu}]")
    to_download: list[tuple[str, Path]] = []
    for url, fname in pairs:
        dest = gal_dir / fname
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  skip: {dest.name}")
            skipped += 1
            continue
        print(f"  get : {dest.name}")
        print(f"        {url}")
        if dry_run:
            continue
        to_download.append((url, dest))

    if (not dry_run) and to_download:
        workers = max(1, int(file_workers))
        if workers == 1 or len(to_download) == 1:
            for url, dest in to_download:
                maybe_download_with_retry(url, dest, retries, verbose=True)
                print(f"  done: {dest.name}", flush=True)
                downloaded += 1
        else:
            print(
                f"  parallel: {len(to_download)} files, workers={workers} (quiet per-file progress)",
                flush=True,
            )
            log_lock = threading.Lock()

            def _worker(url: str, dest: Path) -> None:
                maybe_download_with_retry(url, dest, retries, verbose=False)
                with log_lock:
                    mib = dest.stat().st_size / (1024**2)
                    print(f"  done: {dest.name} ({mib:.1f} MiB)", flush=True)

            with ThreadPoolExecutor(max_workers=workers) as ex:
                fut_map = {
                    ex.submit(_worker, url, dest): (url, dest) for url, dest in to_download
                }
                for fut in as_completed(fut_map):
                    url, dest = fut_map[fut]
                    try:
                        fut.result()
                        downloaded += 1
                    except Exception as e:
                        raise RuntimeError(f"failed: {dest.name} from {url}: {e}") from e

    return downloaded, skipped


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Bulk-download MaNGA DR17 galaxies from DRPall plateifu list. "
            "Default per target: DRP LOGCUBE + DAP MAPS + DAP model LOGCUBE + Pipe3D VAC. "
            "Use download_sdss_cutouts.py / download_legacy_cutouts.py for imaging."
        )
    )
    p.add_argument(
        "plateifu",
        nargs="*",
        help="Optional explicit plate-ifu list. If omitted, targets are read from DRPall.",
    )
    p.add_argument("--out-root", type=Path, default=Path("manga_sdss_fits"), help="Output root folder")
    p.add_argument(
        "--drpall",
        type=Path,
        default=Path("drpall-v3_1_1.fits"),
        help="Local DRPall FITS path (downloaded if missing)",
    )
    p.add_argument("--daptype", default=dm._DEFAULT_DAPTYPE, help="DAP analysis type")
    p.add_argument("--start", type=int, default=0, help="Start index in DRPall plateifu list")
    p.add_argument("--limit", type=int, default=0, help="Max number of galaxies (0 = all)")
    p.add_argument("--retries", type=int, default=3, help="Retries per file on network errors")
    p.add_argument(
        "--object-workers",
        type=int,
        default=1,
        help="Parallel workers across galaxies/plateifus (default 1).",
    )
    p.add_argument(
        "--file-workers",
        type=int,
        default=4,
        help="Parallel workers per galaxy for MaNGA file downloads (default 4).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print actions only; do not download")
    args = p.parse_args(argv)

    if args.plateifu:
        plateifus = args.plateifu
        print(f"Using explicit target list ({len(plateifus)} plate-ifu IDs).")
    else:
        drpall_path = ensure_drpall(args.drpall)
        plateifus = load_plateifus_from_drpall(drpall_path)
        if not plateifus:
            print("No plateifu targets found in DRPall.", file=sys.stderr)
            return 1

    start = max(0, args.start)
    end = len(plateifus) if args.limit <= 0 else min(len(plateifus), start + args.limit)
    todo = plateifus[start:end]
    print(f"Total candidate targets: {len(plateifus)}")
    print(f"Processing slice: [{start}:{end}] -> {len(todo)} targets")
    print(f"Output root: {args.out_root}")
    print(f"Galaxy workers: {max(1, args.object_workers)}")
    print(f"MaNGA file workers per galaxy: {max(1, args.file_workers)}")
    print(
        f"Max concurrent MaNGA file downloads: "
        f"{max(1, args.object_workers) * max(1, args.file_workers)}"
    )

    total_dl = 0
    total_skip = 0
    obj_workers = max(1, int(args.object_workers))
    file_workers = max(1, int(args.file_workers))

    if obj_workers == 1 or len(todo) <= 1:
        for i, pi in enumerate(todo, start=1):
            print(f"\n=== {i}/{len(todo)} ===")
            try:
                dl, sk = process_one(
                    pi,
                    out_root=args.out_root,
                    daptype=args.daptype,
                    retries=max(1, args.retries),
                    file_workers=file_workers,
                    dry_run=args.dry_run,
                )
                total_dl += dl
                total_skip += sk
            except Exception as e:
                print(f"ERROR [{pi}]: {e}", file=sys.stderr)
                continue
    else:
        print(f"\nParallel galaxy mode enabled: workers={obj_workers}")

        def _process_one_task(idx: int, plateifu: str) -> tuple[str, int, int]:
            print(f"\n=== {idx}/{len(todo)} :: {plateifu} ===", flush=True)
            dl, sk = process_one(
                plateifu,
                out_root=args.out_root,
                daptype=args.daptype,
                retries=max(1, args.retries),
                file_workers=file_workers,
                dry_run=args.dry_run,
            )
            return plateifu, dl, sk

        with ThreadPoolExecutor(max_workers=obj_workers) as ex:
            fut_map = {
                ex.submit(_process_one_task, i, pi): (i, pi)
                for i, pi in enumerate(todo, start=1)
            }
            for fut in as_completed(fut_map):
                i, pi = fut_map[fut]
                try:
                    done_pi, dl, sk = fut.result()
                    total_dl += dl
                    total_skip += sk
                    print(
                        f"[complete {i}/{len(todo)}] {done_pi}: downloaded={dl}, skipped={sk}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"ERROR [{pi}]: {e}", file=sys.stderr)
                    continue

    print("\nDone.")
    print(f"Downloaded files: {total_dl}")
    print(f"Skipped existing: {total_skip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


