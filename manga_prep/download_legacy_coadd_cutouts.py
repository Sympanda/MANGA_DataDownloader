"""
Cut Legacy Survey imaging from public NERSC coadd bricks (bulk/science runs).

This avoids the Sky Viewer cutout endpoints (slow, frequent 5xx) by pulling
release brick FITS from:

  https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/

and slicing small WCS-aligned cutouts with astropy.

Brick geometry (where is the object?):
  - ``survey-bricks.fits`` gives one ``BRICKNAME`` per (RA, Dec) using columns
  ``RA1, RA2, DEC1, DEC2``. That is a single skypatch per source; the galaxy is
  not in two different geometric bricks at once.

NERSC ``.../north/...`` vs ``.../south/...`` paths (same brick name):
  - These are the Legacy Surveys file-tree split (MzLS/BASS "north" vs
  DECaLS-style "south" coadds), not two different brick maps. We try, in
  order, for each coadd: ``dr10/south``, ``dr10/north``, ``dr9/south``,
  ``dr9/north`` until a ``legacysurvey-...-image-*.fits.fz`` is found.
  - Public NERSC ``/cfs/cosmo/data/legacysurvey/dr11/`` is not available with the
  same layout at the time this was written; DR11 coadds are not searched here.

Outputs mirror ``download_legacy_cutouts.py`` under each galaxy folder:

  <data-root>/<plate>_<ifu>/legacy_cutouts/
    legacy-<plate>-<ifu>-<band>.fits
    metadata.json

Optional RGB JPEG (``--jpeg``) needs matplotlib. Default is FITS-only.

Requires: astropy, numpy. Uses the same RA/Dec resolution as the viewer script
(``resolve_radec_from_folder`` from local MaNGA FITS).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.utils.data import download_file
from astropy.wcs import WCS
import astropy.units as u

from .download_legacy_cutouts import (
    discover_plateifus_from_data_root,
    parse_plateifu,
    resolve_radec_from_folder,
)

NERSC_LEGACY = "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey"
SURVEY_BRICKS_URL = f"{NERSC_LEGACY}/dr10/survey-bricks.fits.gz"

# Same BRICKNAME; NERSC places coadds under one of these (try in order; 404s are
# expected where that release+hemi has no data for the brick or band).
NERSC_COADD_TRY_ORDER: tuple[tuple[str, str], ...] = (
    ("dr10", "south"),
    ("dr10", "north"),
    ("dr9", "south"),
    ("dr9", "north"),
)

_BRICKS_LOCK = threading.Lock()
_BRICK_DL_LOCKS: dict[str, threading.Lock] = {}
_BRICKS_TABLE: np.ndarray | None = None


def _brick_download_lock(brick: str) -> threading.Lock:
    with _BRICKS_LOCK:
        if brick not in _BRICK_DL_LOCKS:
            _BRICK_DL_LOCKS[brick] = threading.Lock()
        return _BRICK_DL_LOCKS[brick]


def load_survey_bricks_table() -> np.ndarray:
    global _BRICKS_TABLE
    with _BRICKS_LOCK:
        if _BRICKS_TABLE is None:
            path = download_file(SURVEY_BRICKS_URL, cache=True, show_progress=True)
            _BRICKS_TABLE = fits.getdata(path, 1)
        return _BRICKS_TABLE


def brick_ra_subdir(brickname: str) -> str:
    return str(int(brickname[:4]) // 10).zfill(3)


def find_brickname(bricks: np.ndarray, ra: float, dec: float) -> str | None:
    m = (
        (ra >= bricks["RA1"])
        & (ra <= bricks["RA2"])
        & (dec >= bricks["DEC1"])
        & (dec <= bricks["DEC2"])
    )
    idx = np.flatnonzero(m)
    if idx.size == 0:
        return None
    if idx.size > 1:
        r0 = np.asarray(bricks["RA"][idx], dtype=float)
        d0 = np.asarray(bricks["DEC"][idx], dtype=float)
        cosd = np.cos(np.deg2rad(dec))
        dist = (r0 - ra) ** 2 * cosd**2 + (d0 - dec) ** 2
        pick = int(idx[int(np.argmin(dist))])
    else:
        pick = int(idx[0])
    return str(bricks["BRICKNAME"][pick])


def brick_info_row(bricks: np.ndarray, brickname: str) -> dict | None:
    """
    Return RA/Dec bounds and center for the given brick, for JSON metadata.
    `brickname` must match ``survey-bricks`` BRICKNAME.
    """
    bcol = np.asarray(bricks["BRICKNAME"], dtype=object)
    s = np.vectorize(str)(bcol)
    m = s == str(brickname)
    if not np.any(m):
        return None
    r = bricks[m][0]
    return {
        "RA1": float(r["RA1"]),
        "RA2": float(r["RA2"]),
        "DEC1": float(r["DEC1"]),
        "DEC2": float(r["DEC2"]),
        "ra_center": float(r["RA"]),
        "dec_center": float(r["DEC"]),
    }


def ra_dec_in_brick_bounds(ra: float, dec: float, info: dict) -> bool:
    return (info["RA1"] <= ra <= info["RA2"]) and (info["DEC1"] <= dec <= info["DEC2"])


def coadd_image_url(release: str, hemi: str, brick: str, band: str) -> str:
    ra3 = brick_ra_subdir(brick)
    fn = f"legacysurvey-{brick}-image-{band}.fits.fz"
    return f"{NERSC_LEGACY}/{release}/{hemi}/coadd/{ra3}/{brick}/{fn}"


def _http_download(url: str, dest: Path, *, max_retries: int = 6) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "manga-legacy-coadd/1.0"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = resp.read()
            tmp.write_bytes(data)
            tmp.replace(dest)
            return
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 404:
                raise
            if e.code not in (429, 500, 502, 503, 504) or attempt >= max_retries:
                raise
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
            if attempt >= max_retries:
                raise
        sleep_s = 1.5 * (2**attempt) + random.uniform(0, 0.5)
        time.sleep(min(sleep_s, 45.0))
    if last:
        raise last
    raise RuntimeError("download failed")


def nersc_coadd_candidates() -> list[tuple[str, str]]:
    return list(NERSC_COADD_TRY_ORDER)


def ensure_brick_band_cached(
    brick: str,
    band: str,
    cache_dir: Path,
    max_retries: int,
) -> tuple[Path, str, str]:
    """Download coadd FITS if needed; return (local_path, release, hemi)."""
    dest_dir = cache_dir / brick
    dest = dest_dir / f"legacysurvey-{brick}-image-{band}.fits.fz"
    sidecar = dest.with_suffix(dest.suffix + ".source")

    with _brick_download_lock(brick):
        if dest.exists() and dest.stat().st_size > 0:
            if sidecar.exists():
                parts = sidecar.read_text(encoding="utf-8").split()
                if len(parts) >= 2:
                    return dest, parts[0], parts[1]
            return dest, "cached", "unknown"

        last_err: Exception | None = None
        for rel, hem in nersc_coadd_candidates():
            url = coadd_image_url(rel, hem, brick, band)
            try:
                _http_download(url, dest, max_retries=max_retries)
                sidecar.write_text(f"{rel} {hem}", encoding="utf-8")
                return dest, rel, hem
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code == 404:
                    if dest.exists():
                        dest.unlink(missing_ok=True)
                    sidecar.unlink(missing_ok=True)
                    continue
                raise
            except Exception as e:
                last_err = e
                if dest.exists():
                    dest.unlink(missing_ok=True)
                sidecar.unlink(missing_ok=True)
                continue
        raise FileNotFoundError(f"No coadd for brick={brick} band={band}: {last_err}")


def extract_nanomaggies_cutout(
    coadd_path: Path,
    ra: float,
    dec: float,
    size_px: int,
    out_path: Path,
) -> None:
    data = fits.getdata(str(coadd_path), ext=1, memmap=False)
    hdr = fits.getheader(str(coadd_path), ext=1)
    wcs = WCS(hdr)
    pos = SkyCoord(ra * u.deg, dec * u.deg)
    cut = Cutout2D(
        data,
        pos,
        (int(size_px), int(size_px)),
        wcs=wcs,
        mode="partial",
        fill_value=np.nan,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data=cut.data, header=cut.wcs.to_header()).writeto(
        out_path, overwrite=True
    )


def write_rgb_jpeg(
    paths_grz: dict[str, Path],
    out_jpeg: Path,
    *,
    percentiles: tuple[float, float] = (1.0, 99.0),
) -> None:
    import matplotlib

    matplotlib.use("agg")
    import matplotlib.pyplot as plt

    imgs = []
    for b in "grz":
        d = fits.getdata(str(paths_grz[b]), ext=0)
        imgs.append(np.asarray(d, dtype=float))
    rgb = []
    for im in imgs:
        lo, hi = np.percentile(np.nan_to_num(im, nan=0.0), percentiles)
        if hi <= lo:
            hi = lo + 1.0
        x = (im - lo) / (hi - lo)
        rgb.append(np.clip(x, 0.0, 1.0))
    stack = np.dstack(rgb)
    out_jpeg.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_jpeg, stack, origin="lower", pil_kwargs={"quality": 85})


def coadd_cutouts_complete(
    gal_dir: Path,
    plate: str,
    ifu: str,
    bands: str,
    with_jpeg: bool,
) -> bool:
    out = gal_dir / "legacy_cutouts"
    need = [out / f"legacy-{plate}-{ifu}-{b}.fits" for b in bands]
    if with_jpeg:
        need.append(out / f"legacy-{plate}-{ifu}-color.jpg")
    return all(p.is_file() and p.stat().st_size > 0 for p in need)


def run_one(
    plateifu: str,
    *,
    data_root: Path,
    brick_cache: Path,
    size_px: int,
    bands: str,
    fallback_grz: bool,
    with_jpeg: bool,
    force: bool,
    dry_run: bool,
    retries: int,
    bricks: np.ndarray,
) -> tuple[str, str]:
    plate, ifu = parse_plateifu(plateifu)
    gal_dir = data_root / f"{plate}_{ifu}"
    if not gal_dir.is_dir():
        return "failed", f"missing folder {gal_dir}"

    if (not force) and coadd_cutouts_complete(gal_dir, plate, ifu, bands, with_jpeg):
        return "skipped", "already complete"

    ra, dec, src = resolve_radec_from_folder(gal_dir)
    brick = find_brickname(bricks, ra, dec)
    if brick is None:
        return "failed", f"no survey brick for RA/Dec {ra},{dec}"
    binfo = brick_info_row(bricks, brick)
    in_bounds = (
        ra_dec_in_brick_bounds(ra, dec, binfo) if binfo is not None else None
    )

    out = gal_dir / "legacy_cutouts"
    print(f"\n{plateifu}  brick={brick}  RA/Dec={ra:.6f},{dec:.6f} ({src})")
    if binfo is not None:
        print(
            f"  survey-bricks bounds: RA [{binfo['RA1']:.4f},{binfo['RA2']:.4f}] "
            f"Dec [{binfo['DEC1']:.4f},{binfo['DEC2']:.4f}]  inside={in_bounds}"
        )

    if dry_run:
        for rel, hem in nersc_coadd_candidates():
            for b in bands:
                print(f"  would try: {coadd_image_url(rel, hem, brick, b)}")
        if with_jpeg:
            print(f"  would write JPEG {out / f'legacy-{plate}-{ifu}-color.jpg'}")
        return "done", ""

    meta_bands: dict[str, str] = {}
    release_used: str | None = None
    hemi_used: str | None = None
    saved_paths: dict[str, Path] = {}
    bands_used = bands

    def fetch_and_cut_band(b: str) -> None:
        nonlocal release_used, hemi_used
        coadd_fits, rel, hem = ensure_brick_band_cached(
            brick, b, brick_cache, max_retries=retries
        )
        release_used = rel
        hemi_used = hem
        dest = out / f"legacy-{plate}-{ifu}-{b}.fits"
        extract_nanomaggies_cutout(coadd_fits, ra, dec, size_px, dest)
        meta_bands[b] = dest.name
        saved_paths[b] = dest
        print(f"  saved {dest.name} (from {rel}/{hem} brick {brick})")

    try:
        failed_bands: list[str] = []
        for b in bands:
            try:
                fetch_and_cut_band(b)
            except FileNotFoundError:
                failed_bands.append(b)
                print(
                    f"  warning: no legacysurvey-*-image-{b}.fits.fz for brick {brick} on NERSC "
                    f"after trying dr10/s, dr10/n, dr9/s, dr9/n (object is in this brick by "
                    f"survey-bricks RA/Dec box; the band is often missing in a given coadd, "
                    f"e.g. i in dr9-north or z in shallow areas).",
                    file=sys.stderr,
                )

        if failed_bands and fallback_grz and bands == "griz":
            print(f"  griz missing ({failed_bands}); retrying grz", flush=True)
            bands_used = "grz"
            for b in "grz":
                if b in meta_bands:
                    continue
                try:
                    fetch_and_cut_band(b)
                except FileNotFoundError:
                    print(f"  warning: fallback band {b} missing", file=sys.stderr)

        if set(bands_used) - set(meta_bands.keys()):
            missing = sorted(set(bands_used) - set(meta_bands.keys()))
            return "failed", f"missing bands after fallback: {missing}"

        if with_jpeg:
            need = {k: saved_paths[k] for k in "grz" if k in saved_paths}
            if set("grz") <= set(need):
                jpath = out / f"legacy-{plate}-{ifu}-color.jpg"
                write_rgb_jpeg(need, jpath)
                print(f"  saved {jpath.name}")
            else:
                print("  skip JPEG: need g,r,z cutouts for RGB", file=sys.stderr)

        meta: dict = {
            "plateifu": plateifu,
            "ra_deg": ra,
            "dec_deg": dec,
            "radec_source_file": src,
            "source": "nersc_coadd",
            "brickname": brick,
            "object_inside_survey_bricks_box": in_bounds,
            "brick_info_from_survey_bricks": binfo,
            "nersc_coadd_search_order": [
                f"{rel}/{hem}" for rel, hem in nersc_coadd_candidates()
            ],
            "release_guess": release_used,
            "hemisphere_guess": hemi_used,
            "size_px": int(size_px),
            "bands_requested": bands,
            "bands_used": bands_used,
            "fallback_grz_enabled": bool(fallback_grz),
            "jpeg_file": (f"legacy-{plate}-{ifu}-color.jpg" if with_jpeg else None),
            "fits_files": meta_bands,
            "footprint_note": (
                "One BRICKNAME from survey-bricks; NERSC paths try dr10/s, dr10/n, dr9/s, dr9/n. "
                "DR11 is not on this CFS path."
            ),
        }
        out.mkdir(parents=True, exist_ok=True)
        (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return "done", ""
    except FileNotFoundError as e:
        return "failed", str(e)
    except Exception as e:
        return "failed", str(e)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Legacy Survey cutouts from NERSC release coadds (recommended for large batches). "
            "FITS are nanomaggy AB per pixel at native 0.262 arcsec/pixel."
        )
    )
    p.add_argument(
        "plateifu",
        nargs="*",
        help="Optional plate-ifu list. If omitted, process all <plate>_<ifu> folders.",
    )
    p.add_argument("--data-root", type=Path, default=Path("manga_sdss_fits"))
    p.add_argument(
        "--brick-cache",
        type=Path,
        default=Path("legacy_coadd_brick_cache"),
        help="Directory to cache full-brick legacysurvey-*-image-*.fits.fz files (reuse across galaxies).",
    )
    p.add_argument("--workers", type=int, default=2, help="Parallel galaxies (keep low; each brick is large).")
    p.add_argument("--size", type=int, default=198, help="Cutout size in pixels (native pixscale).")
    p.add_argument("--bands", default="grz", help="Bands to extract, e.g. grz or griz (i may 404 on older DR9 north).")
    p.add_argument(
        "--no-fallback-grz",
        action="store_true",
        help="When --bands griz, do not retry grz if any band is missing.",
    )
    p.add_argument("--jpeg", action="store_true", help="Write legacy-*-color.jpg from g,r,z (needs matplotlib).")
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-build cutouts even if legacy_cutouts files already exist (e.g. replacing viewer cutouts).",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--retries", type=int, default=6, help="Retries per brick FITS download.")
    args = p.parse_args(argv)

    bands = "".join(ch for ch in args.bands.lower() if ch in "griz")
    if not bands:
        raise SystemExit("No valid bands (use subset of griz).")

    targets = args.plateifu or discover_plateifus_from_data_root(args.data_root)
    if not targets:
        raise SystemExit(f"No targets under {args.data_root}")

    args.brick_cache.mkdir(parents=True, exist_ok=True)
    print("Loading survey-bricks (cached by astropy)...", flush=True)
    bricks = load_survey_bricks_table()

    workers = max(1, int(args.workers))
    print(
        f"Targets: {len(targets)}  workers: {workers}  size_px: {args.size}  bands: {bands}  "
        f"jpeg: {args.jpeg}  brick-cache: {args.brick_cache}"
    )

    done = skipped = failed = 0

    def task(pi: str) -> tuple[str, str, str]:
        st, msg = run_one(
            pi,
            data_root=args.data_root,
            brick_cache=args.brick_cache,
            size_px=args.size,
            bands=bands,
            fallback_grz=not args.no_fallback_grz,
            with_jpeg=args.jpeg,
            force=args.force,
            dry_run=args.dry_run,
            retries=max(0, int(args.retries)),
            bricks=bricks,
        )
        return pi, st, msg

    if workers == 1:
        for i, pi in enumerate(targets, start=1):
            _, st, msg = task(pi)
            if st == "done":
                done += 1
                print(f"[{i}/{len(targets)}] DONE    {pi}")
            elif st == "skipped":
                skipped += 1
                print(f"[{i}/{len(targets)}] SKIP    {pi} ({msg})")
            else:
                failed += 1
                print(f"[{i}/{len(targets)}] FAILED  {pi} {msg}")
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut_map = {ex.submit(task, pi): (i, pi) for i, pi in enumerate(targets, start=1)}
            for fut in as_completed(fut_map):
                i, pi = fut_map[fut]
                _, st, msg = fut.result()
                if st == "done":
                    done += 1
                    print(f"[{i}/{len(targets)}] DONE    {pi}")
                elif st == "skipped":
                    skipped += 1
                    print(f"[{i}/{len(targets)}] SKIP    {pi} ({msg})")
                else:
                    failed += 1
                    print(f"[{i}/{len(targets)}] FAILED  {pi} {msg}")

    print("\nSummary:")
    print(f"  done   : {done}")
    print(f"  skipped: {skipped}")
    print(f"  failed : {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
