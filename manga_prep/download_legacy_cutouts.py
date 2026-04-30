"""
Download Legacy Surveys imaging cutouts for MaNGA targets.

This script is separate from SDSS cutouts. It reads RA/Dec from local MaNGA FITS
files under:
  manga_sdss_fits/<plate>_<ifu>/

and downloads Legacy Survey viewer cutouts:
  - JPEG: viewer/cutout.jpg
  - FITS: viewer/cutout.fits (per requested band, default grz)
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from astropy.io import fits

LEGACY_BASE = "https://www.legacysurvey.org/viewer"
_PLATE_IFU_DIR = re.compile(r"^(\d+)_(\d+)$")


def parse_plateifu(s: str) -> tuple[str, str]:
    parts = s.replace(" ", "").split("-")
    if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
        raise ValueError(f"expected plate-ifu like 8485-1901, got {s!r}")
    return parts[0], parts[1]


def discover_plateifus_from_data_root(data_root: Path) -> list[str]:
    if not data_root.is_dir():
        return []
    out: list[str] = []
    for p in sorted(data_root.iterdir()):
        if not p.is_dir():
            continue
        m = _PLATE_IFU_DIR.match(p.name)
        if m:
            out.append(f"{m.group(1)}-{m.group(2)}")
    return out


def candidate_fits_files(gal_dir: Path) -> list[Path]:
    names = sorted(gal_dir.glob("*.fits*"))

    def rank(p: Path) -> tuple[int, str]:
        n = p.name.lower()
        if "logcube.fits" in n and "logcube-" not in n:
            return (0, n)
        if "maps" in n:
            return (1, n)
        if ".pipe3d.cube" in n:
            return (2, n)
        return (3, n)

    return sorted(names, key=rank)


def _first_header_value(header: fits.Header, keys: list[str]) -> float | None:
    for k in keys:
        if k in header:
            try:
                return float(header[k])
            except Exception:
                continue
    return None


def resolve_radec_from_folder(gal_dir: Path) -> tuple[float, float, str]:
    tried: list[str] = []
    for path in candidate_fits_files(gal_dir):
        tried.append(path.name)
        try:
            hdr = fits.getheader(path, 0)
        except Exception:
            continue
        ra = _first_header_value(hdr, ["OBJRA", "IFURA", "RA", "CRVAL1"])
        dec = _first_header_value(hdr, ["OBJDEC", "IFUDEC", "DEC", "CRVAL2"])
        if ra is not None and dec is not None and -90.0 <= dec <= 90.0:
            return ra, dec, path.name
    raise RuntimeError(f"Could not resolve RA/Dec in {gal_dir}. Tried: {tried}")


def download_url(
    url: str,
    dest: Path,
    *,
    max_retries: int = 6,
    base_sleep_s: float = 1.5,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "manga-legacy-cutout/1.0"}
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = resp.read()
            dest.write_bytes(data)
            return
        except urllib.error.HTTPError as e:
            last_err = e
            # Retry on rate limit and transient server errors.
            if e.code not in (429, 500, 502, 503, 504) or attempt >= max_retries:
                raise
        except urllib.error.URLError as e:
            last_err = e
            if attempt >= max_retries:
                raise
        sleep_s = base_sleep_s * (2**attempt) + random.uniform(0.0, 0.6)
        time.sleep(min(sleep_s, 45.0))
    if last_err is not None:
        raise last_err
    raise RuntimeError("download failed without explicit exception")


def build_legacy_jpeg_url(
    ra: float,
    dec: float,
    *,
    layer: str,
    pixscale: float,
    size: int,
    bands: str,
) -> str:
    q = urllib.parse.urlencode(
        {
            "ra": f"{ra:.10f}",
            "dec": f"{dec:.10f}",
            "layer": layer,
            "pixscale": f"{pixscale:.6f}",
            "size": int(size),
            "bands": bands,
        }
    )
    return f"{LEGACY_BASE}/cutout.jpg?{q}"


def build_legacy_fits_url(
    ra: float,
    dec: float,
    *,
    layer: str,
    pixscale: float,
    size: int,
    bands: str,
) -> str:
    q = urllib.parse.urlencode(
        {
            "ra": f"{ra:.10f}",
            "dec": f"{dec:.10f}",
            "layer": layer,
            "pixscale": f"{pixscale:.6f}",
            "size": int(size),
            "bands": bands,
        }
    )
    return f"{LEGACY_BASE}/cutout.fits?{q}"


def legacy_complete(
    gal_dir: Path,
    plate: str,
    ifu: str,
    bands: str,
    with_jpeg: bool,
    *,
    fallback_grz: bool,
) -> bool:
    out = gal_dir / "legacy_cutouts"
    base: list[Path] = [out / "metadata.json"]
    if with_jpeg:
        base.append(out / f"legacy-{plate}-{ifu}-color.jpg")

    requested = base + [out / f"legacy-{plate}-{ifu}-{b}.fits" for b in bands]
    if all(p.is_file() and p.stat().st_size > 0 for p in requested):
        return True

    if fallback_grz and bands == "griz":
        grz = base + [out / f"legacy-{plate}-{ifu}-{b}.fits" for b in "grz"]
        return all(p.is_file() and p.stat().st_size > 0 for p in grz)
    return False


def run_one(
    plateifu: str,
    *,
    data_root: Path,
    layer: str,
    pixscale: float,
    size: int,
    bands: str,
    fallback_grz: bool,
    with_jpeg: bool,
    dry_run: bool,
    retries: int,
) -> tuple[str, str]:
    plate, ifu = parse_plateifu(plateifu)
    gal_dir = data_root / f"{plate}_{ifu}"
    if not gal_dir.is_dir():
        return "failed", f"missing folder {gal_dir}"

    if legacy_complete(
        gal_dir, plate, ifu, bands, with_jpeg, fallback_grz=fallback_grz
    ):
        return "skipped", "already complete"

    ra, dec, src = resolve_radec_from_folder(gal_dir)
    out = gal_dir / "legacy_cutouts"

    print(f"\n{plateifu}")
    print(f"  RA/Dec: {ra:.8f}, {dec:.8f}  (from {src})")
    if with_jpeg:
        jurl = build_legacy_jpeg_url(
            ra, dec, layer=layer, pixscale=pixscale, size=size, bands=bands
        )
        jpath = out / f"legacy-{plate}-{ifu}-color.jpg"
        print(f"  JPEG: {jurl}")
        print(f"  -> {jpath}")
    else:
        jurl, jpath = None, None

    if dry_run:
        return "done", ""

    try:
        if with_jpeg and jurl and jpath:
            download_url(jurl, jpath, max_retries=retries)
            print(f"saved: {jpath}")

        saved_bands: dict[str, str] = {}
        failed_bands: list[str] = []
        for b in bands:
            furl = build_legacy_fits_url(
                ra, dec, layer=layer, pixscale=pixscale, size=size, bands=b
            )
            fpath = out / f"legacy-{plate}-{ifu}-{b}.fits"
            try:
                download_url(furl, fpath, max_retries=retries)
                saved_bands[b] = fpath.name
                print(f"saved: {fpath}")
            except Exception as e:
                failed_bands.append(b)
                print(f"warning: band {b} failed for {plateifu}: {e}", file=sys.stderr)

        bands_used = bands
        if failed_bands:
            if fallback_grz and bands == "griz":
                print(f"  griz failed ({failed_bands}); retrying fallback bands=grz")
                saved_bands = {}
                failed_bands = []
                bands_used = "grz"
                for b in "grz":
                    furl = build_legacy_fits_url(
                        ra, dec, layer=layer, pixscale=pixscale, size=size, bands=b
                    )
                    fpath = out / f"legacy-{plate}-{ifu}-{b}.fits"
                    try:
                        download_url(furl, fpath, max_retries=retries)
                        saved_bands[b] = fpath.name
                        print(f"saved: {fpath}")
                    except Exception as e:
                        failed_bands.append(b)
                        print(
                            f"warning: fallback band {b} failed for {plateifu}: {e}",
                            file=sys.stderr,
                        )
            if failed_bands:
                return "failed", f"failed bands: {failed_bands}"

        meta = {
            "plateifu": plateifu,
            "ra_deg": ra,
            "dec_deg": dec,
            "radec_source_file": src,
            "source": "legacysurvey",
            "layer": layer,
            "pixscale_arcsec_per_pix": pixscale,
            "size_px": int(size),
            "bands_requested": bands,
            "bands_used": bands_used,
            "fallback_grz_enabled": bool(fallback_grz),
            "jpeg_file": jpath.name if jpath else None,
            "fits_files": saved_bands,
        }
        out.mkdir(parents=True, exist_ok=True)
        (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"saved: {out / 'metadata.json'}")
        return "done", ""
    except urllib.error.HTTPError as e:
        return "failed", f"HTTP {e.code}: {e}"
    except urllib.error.URLError as e:
        return "failed", f"URL error: {e}"
    except Exception as e:
        return "failed", str(e)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Download Legacy Survey cutouts for local MaNGA folders."
    )
    p.add_argument(
        "plateifu",
        nargs="*",
        help="Optional plate-ifu list. If omitted, process all <plate>_<ifu> folders.",
    )
    p.add_argument("--data-root", type=Path, default=Path("manga_sdss_fits"))
    p.add_argument("--workers", type=int, default=2, help="Parallel galaxies")
    p.add_argument("--layer", default="ls-dr10", help="Legacy viewer layer")
    p.add_argument("--pixscale", type=float, default=0.262, help="Arcsec per pixel")
    p.add_argument("--size", type=int, default=198, help="Cutout size in pixels")
    p.add_argument("--bands", default="griz", help="Bands string, e.g. griz")
    p.add_argument(
        "--no-fallback-grz",
        action="store_true",
        help="Disable automatic fallback from griz to grz if any requested band fails.",
    )
    p.add_argument("--no-jpeg", action="store_true", help="Skip JPEG")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--retries",
        type=int,
        default=6,
        help="Retries per HTTP request (429/5xx/URLError) with backoff.",
    )
    args = p.parse_args(argv)

    bands = "".join(ch for ch in args.bands.lower() if ch in "griz12")
    if not bands:
        raise SystemExit("No valid bands selected. Use a subset of griz12.")

    targets = args.plateifu or discover_plateifus_from_data_root(args.data_root)
    if not targets:
        raise SystemExit(f"No targets found under {args.data_root}")

    workers = max(1, int(args.workers))
    print(
        f"Targets: {len(targets)}  data-root: {args.data_root}  workers: {workers}  "
        f"layer: {args.layer}  bands: {bands}"
    )

    done = skipped = failed = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_map = {
            ex.submit(
                run_one,
                pi,
                data_root=args.data_root,
                layer=args.layer,
                pixscale=args.pixscale,
                size=args.size,
                bands=bands,
                fallback_grz=not args.no_fallback_grz,
                with_jpeg=not args.no_jpeg,
                dry_run=args.dry_run,
                retries=max(0, int(args.retries)),
            ): (i, pi)
            for i, pi in enumerate(targets, start=1)
        }
        for fut in as_completed(fut_map):
            i, pi = fut_map[fut]
            status, msg = fut.result()
            if status == "done":
                done += 1
                print(f"[{i}/{len(targets)}] DONE    {pi}")
            elif status == "skipped":
                skipped += 1
                print(f"[{i}/{len(targets)}] SKIP    {pi} ({msg})")
            else:
                failed += 1
                print(f"[{i}/{len(targets)}] FAILED  {pi}")
                if msg:
                    print(f"  {msg}")

    print("\nSummary:")
    print(f"  done   : {done}")
    print(f"  skipped: {skipped}")
    print(f"  failed : {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

