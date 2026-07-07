"""
Download only Pipe3D VAC cubes for local MaNGA targets.

By default, targets are read from existing folders under --data-root
named <plate>_<ifu>. You can also pass explicit plate-ifu IDs.
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DR = "dr17"
DRPVER = "v3_1_1"
PIPE3DVER = "3.1.1"
BASE_PIPE3D = f"https://data.sdss.org/sas/{DR}/env/MANGA_PIPE3D/{DRPVER}/{PIPE3DVER}"

FOLDER_RE = re.compile(r"^(\d+)_(\d+)$")
PLATEIFU_RE = re.compile(r"^(\d+)-(\d+)$")


def parse_plateifu(s: str) -> tuple[str, str]:
    m = PLATEIFU_RE.match(s.strip())
    if not m:
        raise ValueError(f"Expected plate-ifu like 8485-1901, got {s!r}")
    return m.group(1), m.group(2)


def targets_from_data_root(data_root: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for p in sorted(data_root.iterdir()):
        if not p.is_dir():
            continue
        m = FOLDER_RE.match(p.name)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def download_url(url: str, dest: Path, chunk: int = 8 * 1024 * 1024) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "manga-pipe3d-only/1.0"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        total = resp.headers.get("Content-Length")
        n = int(total) if total else None
        got = 0
        with dest.open("wb") as f:
            while True:
                b = resp.read(chunk)
                if not b:
                    break
                f.write(b)
                got += len(b)
                if n:
                    pct = 100.0 * got / n
                    print(
                        f"\r  {dest.name}: {got / (1024**2):.1f}/{n / (1024**2):.1f} MiB ({pct:.1f}%)",
                        end="",
                        file=sys.stderr,
                    )
        if n:
            print(file=sys.stderr)


def build_job(plate: str, ifu: str, data_root: Path) -> tuple[str, Path, str]:
    fn = f"manga-{plate}-{ifu}.Pipe3D.cube.fits.gz"
    url = f"{BASE_PIPE3D}/{plate}/{fn}"
    dest = data_root / f"{plate}_{ifu}" / fn
    label = f"{plate}-{ifu}"
    return url, dest, label


def run_one(url: str, dest: Path, label: str, dry_run: bool, skip_existing: bool) -> tuple[str, str]:
    if skip_existing and dest.exists():
        return "skipped", f"{label} (exists)"

    if dry_run:
        return "dry-run", f"{label} -> {dest}"

    try:
        print(url)
        print(f" -> {dest}")
        download_url(url, dest)
        return "saved", f"{label} -> {dest}"
    except urllib.error.HTTPError as e:
        return "failed", f"{label}: HTTP {e.code} for {url}"
    except urllib.error.URLError as e:
        return "failed", f"{label}: URL error for {url}: {e}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Download only MaNGA Pipe3D VAC cubes. "
            "If no plateifu are passed, read targets from <data-root>/<plate>_<ifu>/."
        )
    )
    p.add_argument(
        "plateifu",
        nargs="*",
        help="Optional plate-ifu IDs, e.g. 8485-1901. If omitted, uses data-root folders.",
    )
    p.add_argument("--data-root", type=Path, default=Path("manga_sdss_fits"))
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel download workers across galaxies (default: 8).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print URLs/paths only")
    p.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-download even when local Pipe3D file already exists.",
    )
    args = p.parse_args(argv)

    if args.plateifu:
        targets = [parse_plateifu(x) for x in args.plateifu]
    else:
        if not args.data_root.exists():
            print(f"Data root not found: {args.data_root}", file=sys.stderr)
            return 2
        targets = targets_from_data_root(args.data_root)
        if not targets:
            print(f"No <plate>_<ifu> folders found in {args.data_root}", file=sys.stderr)
            return 2

    workers = max(1, int(args.workers))
    skip_existing = not args.no_skip_existing

    jobs = [build_job(plate, ifu, args.data_root) for plate, ifu in targets]
    saved = skipped = failed = dry_run = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(run_one, url, dest, label, args.dry_run, skip_existing)
            for url, dest, label in jobs
        ]
        for fut in as_completed(futs):
            status, msg = fut.result()
            if status == "saved":
                saved += 1
                print(f"saved: {msg}")
            elif status == "skipped":
                skipped += 1
                print(f"skip existing: {msg}")
            elif status == "dry-run":
                dry_run += 1
                print(f"dry-run: {msg}")
            else:
                failed += 1
                print(msg, file=sys.stderr)

    print(
        f"Summary: total={len(jobs)} saved={saved} skipped={skipped} dry_run={dry_run} failed={failed}"
    )
    if failed:
        print(f"Done with {failed} failures.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
