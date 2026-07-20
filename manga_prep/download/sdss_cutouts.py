"""
Download SDSS SkyServer image cutouts for MaNGA plate-ifu targets.

This script is designed to work with folders created by download_manga_sdss.py:
    manga_sdss_fits/<plate>_<ifu>/

It tries to read RA/Dec from existing MaNGA FITS headers in that folder and then
downloads SkyServer ImgCutout images.

Notes:
- SkyServer getjpeg returns a color composite image (not separate u/g/r/i/z FITS frames).
- A getfits endpoint is included as an optional attempt; availability may vary.
- Ugriz uses SkyServer SQL + SAS `frame-*.fits.bz2` downloads and numpy cutouts (no astroquery).
  Optional `--ugriz-subprocess` runs that step in a child process if you still see native crashes.

Examples:
  # All galaxies under manga_sdss_fits/ (color JPEG + ugriz FITS)
  python download_sdss_cutouts.py
  python download_sdss_cutouts.py --no-ugriz --workers 4   # parallel JPEG only

  python download_sdss_cutouts.py 8485-1901
  python download_sdss_cutouts.py 8485-1901 --ugriz-only
  python download_sdss_cutouts.py 8485-1901 --size 1024 --scale 0.198
  python download_sdss_cutouts.py 8485-1901 --dry-run
"""
from __future__ import annotations

import argparse
import bz2
import csv
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from astropy.io import fits

SKYSERVER_BASE = "https://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout"
_UGRIZ = ("u", "g", "r", "i", "z")
_PLATE_IFU_DIR = re.compile(r"^(\d+)_(\d+)$")

def parse_plateifu(s: str) -> tuple[str, str]:
    parts = s.replace(" ", "").split("-")
    if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
        raise ValueError(f"expected plate-ifu like 8485-1901, got {s!r}")
    return parts[0], parts[1]


def discover_plateifus_from_data_root(data_root: Path) -> list[str]:
    """Subfolders named <plate>_<ifu> -> plate-ifu strings, sorted."""
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


def cutouts_fully_complete(
    gal_dir: Path,
    plate: str,
    ifu: str,
    *,
    require_jpeg: bool = True,
    require_ugriz: bool = True,
) -> bool:
    """Whether sdss_cutouts already has what we would fetch (for resume)."""
    cut = gal_dir / "sdss_cutouts"
    paths: list[Path] = []
    if require_jpeg:
        paths.append(cut / f"sdss-{plate}-{ifu}-color.jpg")
    if require_ugriz:
        for b in _UGRIZ:
            paths.append(cut / f"sdss-{plate}-{ifu}-{b}.fits")
    paths.append(cut / "metadata.json")
    if not all(p.exists() and p.stat().st_size > 0 for p in paths):
        return False
    if require_ugriz:
        meta = load_cutout_metadata(cut / "metadata.json")
        size_px = meta.get("size_px") if meta else None
        if size_px is not None:
            ok, _ = _verify_ugriz_shapes(cut, plate, ifu, int(size_px))
            if not ok:
                return False
        listed = set((meta or {}).get("ugriz_files") or {})
        for band in _UGRIZ:
            if band not in listed:
                return False
    return True


def load_cutout_metadata(meta_path: Path) -> dict | None:
    if not meta_path.is_file() or meta_path.stat().st_size <= 0:
        return None
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return raw if isinstance(raw, dict) else None


def metadata_matches_request(
    meta: dict | None,
    *,
    size: int,
    scale: float,
    ugriz_dr: int,
    no_ugriz: bool,
    skip_jpeg: bool,
) -> tuple[bool, str]:
    """Whether existing metadata matches the current request settings."""
    if not isinstance(meta, dict):
        return False, "missing/invalid metadata"

    got_size = meta.get("size_px")
    if got_size is None or int(got_size) != int(size):
        return False, f"size_px {got_size!r} != {int(size)}"

    got_scale = meta.get("scale_arcsec_per_px")
    if got_scale is None or abs(float(got_scale) - float(scale)) > 1e-9:
        return False, f"scale_arcsec_per_px {got_scale!r} != {float(scale)}"

    got_skip_jpeg = bool(meta.get("jpeg_skipped", False))
    if got_skip_jpeg != bool(skip_jpeg):
        return False, f"jpeg_skipped {got_skip_jpeg} != {bool(skip_jpeg)}"

    if not no_ugriz:
        got_dr = meta.get("ugriz_dr")
        if got_dr is None or int(got_dr) != int(ugriz_dr):
            return False, f"ugriz_dr {got_dr!r} != {int(ugriz_dr)}"

    return True, "metadata matches"


def _apply_blas_thread_env() -> None:
    for k, v in (
        ("OMP_NUM_THREADS", "1"),
        ("MKL_NUM_THREADS", "1"),
        ("OPENBLAS_NUM_THREADS", "1"),
        ("NUMEXPR_NUM_THREADS", "1"),
        ("VECLIB_MAXIMUM_THREADS", "1"),
    ):
        os.environ.setdefault(k, v)


def candidate_fits_files(gal_dir: Path) -> list[Path]:
    """Rank likely files that contain RA/Dec metadata."""
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
    """
    Resolve RA/Dec from local FITS headers.
    Returns: (ra_deg, dec_deg, filename_used)
    """
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

    raise RuntimeError(f"Could not resolve RA/Dec from FITS headers in {gal_dir}. Tried: {tried}")


def download_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "manga-sdss-cutout/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    dest.write_bytes(data)


def build_getjpeg_url(ra: float, dec: float, scale: float, size: int, opt: str) -> str:
    q = urllib.parse.urlencode(
        {
            "ra": f"{ra:.10f}",
            "dec": f"{dec:.10f}",
            "scale": f"{scale:.6f}",
            "width": int(size),
            "height": int(size),
            "opt": opt,
        }
    )
    return f"{SKYSERVER_BASE}/getjpeg?{q}"


def build_getfits_url(ra: float, dec: float, scale: float, size: int) -> str:
    q = urllib.parse.urlencode(
        {
            "ra": f"{ra:.10f}",
            "dec": f"{dec:.10f}",
            "scale": f"{scale:.6f}",
            "width": int(size),
            "height": int(size),
        }
    )
    return f"{SKYSERVER_BASE}/getfits?{q}"


def _skyserver_sql_rows(dr: int, sql: str) -> list[dict[str, str]]:
    """Run synchronous SQL on SkyServer; returns list of row dicts (string values)."""
    params = urllib.parse.urlencode({"cmd": sql, "format": "csv"})
    url = f"https://skyserver.sdss.org/dr{dr}/SkyServerWS/SearchTools/SqlSearch?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "manga-sdss-cutout/2.1"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    lines = [
        ln
        for ln in text.splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    if not lines:
        return []
    rdr = csv.DictReader(io.StringIO("\n".join(lines)))
    return [dict(row) for row in rdr]


def _nearest_field_params(ra: float, dec: float, dr: int) -> tuple[int, int, int, int] | None:
    """Return (run, rerun, camcol, field) for imaging covering (ra, dec)."""
    for arcmin in (0.5, 2.0, 10.0, 30.0):
        sql = (
            f"SELECT TOP 1 run, rerun, camcol, field "
            f"FROM PhotoObj WHERE objID = dbo.fGetNearestObjIdEq({ra:.10f},{dec:.10f},{arcmin})"
        )
        try:
            rows = _skyserver_sql_rows(dr, sql)
        except Exception as e:
            print(f"warning: SkyServer SQL failed ({arcmin}'): {e}", file=sys.stderr)
            continue
        if not rows:
            continue
        r0 = rows[0]
        try:
            run = int(float(r0.get("run", r0.get("Run", 0))))
            rerun = int(float(r0.get("rerun", r0.get("Rerun", 0))))
            camcol = int(float(r0.get("camcol", r0.get("Camcol", 0))))
            field = int(float(r0.get("field", r0.get("Field", 0))))
            if run > 0:
                return run, rerun, camcol, field
        except (TypeError, ValueError):
            continue
    # Slower fallback if fGetNearestObjIdEq is unavailable
    sql2 = (
        f"SELECT TOP 1 p.run, p.rerun, p.camcol, p.field FROM PhotoPrimary AS p "
        f"ORDER BY (power((p.ra-{ra:.10f})*cos(radians({dec:.10f})),2)+power(p.dec-{dec:.10f},2))"
    )
    try:
        rows = _skyserver_sql_rows(dr, sql2)
        if rows:
            r0 = rows[0]
            run = int(float(r0.get("run", r0.get("Run", 0))))
            rerun = int(float(r0.get("rerun", r0.get("Rerun", 0))))
            camcol = int(float(r0.get("camcol", r0.get("Camcol", 0))))
            field = int(float(r0.get("field", r0.get("Field", 0))))
            if run > 0:
                return run, rerun, camcol, field
    except Exception as e:
        print(f"warning: PhotoPrimary fallback SQL failed: {e}", file=sys.stderr)
    return None


def _download_first_ok(urls: list[str]) -> tuple[bytes, str] | None:
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "manga-sdss-cutout/2.1"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = resp.read()
            if data:
                return data, url
        except Exception:
            continue
    return None


def _sas_frame_urls(dr: int, rerun: int, run: int, camcol: int, field: int, band: str) -> list[str]:
    run6 = f"{run:06d}"
    field4 = f"{field:04d}"
    name = f"frame-{band}-{run6}-{camcol}-{field4}.fits.bz2"
    rel = f"photoObj/frames/{rerun}/{run}/{camcol}/{name}"
    # DR18 imaging often lives under prior-surveys/eboss; older layouts use sdss/ or boss/.
    bases = [
        f"https://dr{dr}.sdss.org/sas/dr{dr}/prior-surveys/sdss4-dr17-eboss/{rel}",
        f"https://dr{dr}.sdss.org/sas/dr{dr}/sdss/{rel}",
        f"https://dr{dr}.sdss.org/sas/dr{dr}/boss/{rel}",
        f"https://data.sdss.org/sas/dr{dr}/sdss/{rel}",
        f"https://data.sdss.org/sas/dr{dr}/boss/{rel}",
    ]
    return bases


def _world_to_pix_xy(ra: float, dec: float, hdr: fits.Header) -> tuple[float, float]:
    """
    RA/Dec (deg) -> pixel (x, y) using only FITS CD + small-angle tangent-plane offsets.

    Intentionally avoids astropy.wcs (wcslib/erfa), which can segfault on some Windows builds.
    Good enough for small SDSS cutouts around the reference pixel.
    """
    crval1 = float(hdr["CRVAL1"])
    crval2 = float(hdr["CRVAL2"])
    crpix1 = float(hdr["CRPIX1"])
    crpix2 = float(hdr["CRPIX2"])
    cd11 = float(hdr["CD1_1"]) if "CD1_1" in hdr else float(hdr.get("PC001001", 0.0))
    cd12 = float(hdr["CD1_2"]) if "CD1_2" in hdr else float(hdr.get("PC001002", 0.0))
    cd21 = float(hdr["CD2_1"]) if "CD2_1" in hdr else float(hdr.get("PC002001", 0.0))
    cd22 = float(hdr["CD2_2"]) if "CD2_2" in hdr else float(hdr.get("PC002002", 0.0))
    xi = (ra - crval1) * math.cos(math.radians(dec))
    eta = dec - crval2
    det = cd11 * cd22 - cd12 * cd21
    if abs(det) < 1e-30:
        raise RuntimeError("singular CD matrix in FITS header")
    inv11 = cd22 / det
    inv12 = -cd12 / det
    inv21 = -cd21 / det
    inv22 = cd11 / det
    dxi = inv11 * xi + inv12 * eta
    deta = inv21 * xi + inv22 * eta
    return float(crpix1 + dxi), float(crpix2 + deta)


def _numpy_cutout(
    data,
    hdr: fits.Header,
    ra: float,
    dec: float,
    size_px: int,
):
    import numpy as np

    data = np.asarray(data)
    x, y = _world_to_pix_xy(ra, dec, hdr)
    half = int(size_px) // 2
    ny, nx = data.shape[-2], data.shape[-1]
    x0 = int(round(x)) - half
    x1 = x0 + int(size_px)
    y0 = int(round(y)) - half
    y1 = y0 + int(size_px)
    pad_x0 = max(0, -x0)
    pad_y0 = max(0, -y0)
    x0c = max(0, x0)
    y0c = max(0, y0)
    x1c = min(nx, x1)
    y1c = min(ny, y1)
    sl = data[y0c:y1c, x0c:x1c]
    if sl.size == 0:
        raise RuntimeError("empty cutout slice")
    out = np.full((int(size_px), int(size_px)), np.nan, dtype=np.float32)
    sl = np.asarray(sl, dtype=np.float32)
    ox0 = pad_x0 + (x0c - x0)
    oy0 = pad_y0 + (y0c - y0)
    out[oy0 : oy0 + sl.shape[0], ox0 : ox0 + sl.shape[1]] = sl
    new_hdr = hdr.copy()
    new_hdr["CRPIX1"] = float(hdr["CRPIX1"]) - float(x0)
    new_hdr["CRPIX2"] = float(hdr["CRPIX2"]) - float(y0)
    new_hdr["NAXIS1"] = int(size_px)
    new_hdr["NAXIS2"] = int(size_px)
    if "NAXIS3" in new_hdr:
        del new_hdr["NAXIS3"]
    new_hdr["NAXIS"] = 2
    return out, new_hdr


def _ugriz_band_paths(out_dir: Path, plate: str, ifu: str) -> dict[str, Path]:
    return {b: out_dir / f"sdss-{plate}-{ifu}-{b}.fits" for b in _UGRIZ}


def _remove_ugriz_band_files(paths: dict[str, Path], bands: set[str]) -> None:
    for band in bands:
        path = paths[band]
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _verify_ugriz_shapes(
    out_dir: Path,
    plate: str,
    ifu: str,
    size_px: int,
    *,
    bands: tuple[str, ...] = _UGRIZ,
) -> tuple[bool, str]:
    """All listed bands must exist on disk with shape (size_px, size_px)."""
    paths = _ugriz_band_paths(out_dir, plate, ifu)
    shapes: dict[str, tuple[int, int] | None] = {}
    for band in bands:
        path = paths[band]
        if not path.is_file() or path.stat().st_size <= 0:
            return False, f"missing band {band}"
        try:
            header = fits.getheader(path, 0)
            h = int(header["NAXIS2"])
            w = int(header["NAXIS1"])
        except Exception as e:
            return False, f"bad header for band {band}: {e}"
        shapes[band] = (h, w)
        if (h, w) != (int(size_px), int(size_px)):
            return False, f"band {band} shape {(h, w)} != ({size_px}, {size_px})"
    if len(set(shapes.values())) != 1:
        return False, f"mixed band shapes: {shapes}"
    return True, "ok"


def _purge_unlisted_ugriz_bands(
    out_dir: Path,
    plate: str,
    ifu: str,
    listed_bands: set[str],
) -> list[str]:
    """Remove on-disk ugriz FITS not present in the latest download manifest."""
    removed: list[str] = []
    for band, path in _ugriz_band_paths(out_dir, plate, ifu).items():
        if band in listed_bands:
            continue
        if path.is_file():
            try:
                path.unlink()
                removed.append(band)
            except OSError:
                pass
    return removed


def download_ugriz_fits_cutouts(
    ra: float,
    dec: float,
    *,
    out_dir: Path,
    plate: str,
    ifu: str,
    size_px: int,
    scale_arcsec_per_px: float,
    data_release: int,
) -> dict[str, str]:
    """
    Download SDSS ugriz imaging cutouts via SkyServer SQL + SAS frame FITS (no astroquery).

    Avoids SkyCoord / Cutout2D / SDSS.get_images, which often crash native code on Windows.

    Returns a dict band -> saved filename.
    """
    dr = int(data_release)
    saved: dict[str, str] = {}
    out_dir.mkdir(parents=True, exist_ok=True)

    fp = _nearest_field_params(ra, dec, dr)
    if fp is None:
        print(f"warning: no PhotoObj / field near RA/Dec for DR{dr}", file=sys.stderr)
        return saved
    run, rerun, camcol, field = fp
    _ = scale_arcsec_per_px  # reserved for validation; pixel scale comes from frame WCS

    for band in _UGRIZ:
        urls = _sas_frame_urls(dr, rerun, run, camcol, field, band)
        got = _download_first_ok(urls)
        if got is None:
            print(f"warning: could not download frame for band {band}", file=sys.stderr)
            continue
        raw, src_url = got
        try:
            if raw.startswith(b"BZh") or (len(raw) >= 2 and raw[:2] == b"BZ"):
                raw = bz2.decompress(raw)
            with fits.open(io.BytesIO(raw), memmap=False) as hdul:
                hdu = hdul[0]
                data = hdu.data
                hdr = hdu.header
            if data is None:
                print(f"warning: empty frame data for band {band}", file=sys.stderr)
                continue
            cut, chdr = _numpy_cutout(data, hdr, ra, dec, int(size_px))
            out_hdu = fits.PrimaryHDU(data=cut, header=chdr)
        except Exception as e:
            print(f"warning: cutout failed for band {band}: {e}", file=sys.stderr)
            continue

        out_hdu.header["SRC_DR"] = (dr, "SDSS data release used")
        out_hdu.header["SRC_BAND"] = (band, "SDSS filter band")
        su = src_url.replace("https://", "")[:52]
        out_hdu.header["SASURL"] = (su, "SAS path")
        out_hdu.header["CUT_RA"] = (float(ra), "Cutout center RA [deg]")
        out_hdu.header["CUT_DEC"] = (float(dec), "Cutout center Dec [deg]")
        out_hdu.header["CUTSIZE"] = (int(size_px), "Cutout size [pixels]")
        out_hdu.header["CUTSCALE"] = (float(scale_arcsec_per_px), "Requested scale [arcsec/pix] (approx)")

        out_name = f"sdss-{plate}-{ifu}-{band}.fits"
        out_path = out_dir / out_name
        out_hdu.writeto(out_path, overwrite=True)
        saved[band] = out_name
        print(f"saved: {out_path}")

    return saved


def _subprocess_env() -> dict[str, str]:
    e = os.environ.copy()
    for k, v in (
        ("OMP_NUM_THREADS", "1"),
        ("MKL_NUM_THREADS", "1"),
        ("OPENBLAS_NUM_THREADS", "1"),
        ("NUMEXPR_NUM_THREADS", "1"),
        ("VECLIB_MAXIMUM_THREADS", "1"),
    ):
        e.setdefault(k, v)
    return e


def run_ugriz_fits_via_subprocess(
    ra: float,
    dec: float,
    *,
    out_dir: Path,
    plate: str,
    ifu: str,
    size_px: int,
    scale_arcsec_per_px: float,
    data_release: int,
) -> tuple[dict[str, str], str | None]:
    """
    Run download_ugriz_fits_cutouts in a child process. If astroquery/numpy segfaults,
    only the child dies; the parent gets a non-zero exit and stderr (when Python raises).

    Returns (saved dict, error_message). error_message is None on success.
    """
    worker = Path(__file__).resolve().parent.parent / "io" / "ugriz_worker.py"
    if not worker.is_file():
        return {}, f"missing {worker}"

    job = {
        "ra": ra,
        "dec": dec,
        "out_dir": str(out_dir.resolve()),
        "plate": plate,
        "ifu": ifu,
        "size_px": size_px,
        "scale_arcsec_per_px": scale_arcsec_per_px,
        "data_release": data_release,
    }

    fd, job_path = tempfile.mkstemp(suffix=".json", text=True)
    os.close(fd)
    jp = Path(job_path)
    try:
        jp.write_text(json.dumps(job), encoding="utf-8")
        cmd = [sys.executable, str(worker), str(jp)]
        kw = {
            "cwd": str(repo),
            "env": _subprocess_env(),
            "capture_output": True,
            "text": True,
            "timeout": 3600,
        }
        if sys.platform == "win32":
            kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        r = subprocess.run(cmd, **kw)
        if r.returncode != 0:
            err = (r.stderr or "").strip() or (r.stdout or "").strip()
            if not err:
                err = f"ugriz worker exit {r.returncode} (often a native crash in astroquery/numpy)"
            return {}, err
        raw = (r.stdout or "").strip()
        if not raw:
            return {}, "ugriz worker produced no output"
        last = raw.splitlines()[-1]
        return json.loads(last), None
    except subprocess.TimeoutExpired:
        return {}, "ugriz worker timed out (1h)"
    except json.JSONDecodeError as e:
        return {}, f"ugriz worker bad JSON: {e}"
    finally:
        try:
            jp.unlink(missing_ok=True)
        except OSError:
            pass


def run_cutouts_for_plateifu(
    plateifu: str,
    *,
    data_root: Path,
    size: int,
    scale: float,
    opt: str = "",
    with_fits: bool = False,
    no_ugriz: bool = False,
    skip_jpeg: bool = False,
    strict_ugriz: bool = False,
    ugriz_dr: int = 18,
    dry_run: bool = False,
    ugriz_subprocess: bool = False,
) -> int:
    """
    Download SkyServer JPEG (optional) and/or per-band ugriz FITS (SkyServer + SAS; no astroquery).

    - Default: JPEG + ugriz FITS (ugriz warnings only; use strict_ugriz to fail if incomplete).
    - skip_jpeg=True: ugriz FITS only (still needs local MaNGA FITS for RA/Dec).
    - ugriz_subprocess=True: run ugriz in a child process (optional safety if anything still crashes).

    Returns 0 on success, 1 on hard failure (missing data, JPEG error, or incomplete ugriz when strict).
    """
    plate, ifu = parse_plateifu(plateifu)
    gal_dir = data_root / f"{plate}_{ifu}"
    if not gal_dir.is_dir():
        print(f"Missing folder: {gal_dir}", file=sys.stderr)
        return 1

    if no_ugriz and skip_jpeg:
        print("Cannot combine no_ugriz with skip_jpeg (nothing to download).", file=sys.stderr)
        return 1

    try:
        ra, dec, source_name = resolve_radec_from_folder(gal_dir)
    except Exception as e:
        print(f"Could not resolve RA/Dec for {plateifu}: {e}", file=sys.stderr)
        return 1

    out_dir = gal_dir / "sdss_cutouts"
    jpeg_url = build_getjpeg_url(ra, dec, scale, size, opt)
    jpeg_path = out_dir / f"sdss-{plate}-{ifu}-color.jpg"
    meta_path = out_dir / "metadata.json"

    print(f"\n{plateifu}")
    print(f"  RA/Dec: {ra:.8f}, {dec:.8f}  (from {source_name})")
    if skip_jpeg:
        print("  mode: ugriz FITS only (SkyServer JPEG skipped)")
    else:
        print(f"  JPEG: {jpeg_url}")
        print(f"  -> {jpeg_path}")

    fits_url = None
    fits_path = None
    if not skip_jpeg and with_fits:
        fits_url = build_getfits_url(ra, dec, scale, size)
        fits_path = out_dir / f"sdss-{plate}-{ifu}-cutout.fits"
        print(f"  FITS: {fits_url}")
        print(f"  -> {fits_path}")

    if dry_run:
        if not no_ugriz:
            print(f"  ugriz: would query SDSS DR{ugriz_dr} for bands {', '.join(_UGRIZ)} -> {out_dir}")
        return 0

    if not skip_jpeg:
        try:
            download_url(jpeg_url, jpeg_path)
            print(f"saved: {jpeg_path}")
        except urllib.error.HTTPError as e:
            print(f"HTTP {e.code} downloading JPEG for {plateifu}: {e}", file=sys.stderr)
            return 1
        except urllib.error.URLError as e:
            print(f"URL error downloading JPEG for {plateifu}: {e}", file=sys.stderr)
            return 1

        if with_fits and fits_url and fits_path:
            try:
                download_url(fits_url, fits_path)
                print(f"saved: {fits_path}")
            except Exception as e:
                print(f"warning: FITS cutout download failed for {plateifu}: {e}", file=sys.stderr)

    ugriz_files: dict[str, str] = {}
    if not no_ugriz:
        # Drop leftover band files from partial / mixed-size downloads before fetching.
        _remove_ugriz_band_files(_ugriz_band_paths(out_dir, plate, ifu), set(_UGRIZ))
        print(
            f"  ugriz: querying SDSS (DR{ugriz_dr}) for {plateifu} — may take a minute…",
            flush=True,
        )
        if ugriz_subprocess:
            ugriz_files, sub_err = run_ugriz_fits_via_subprocess(
                ra,
                dec,
                out_dir=out_dir,
                plate=plate,
                ifu=ifu,
                size_px=size,
                scale_arcsec_per_px=scale,
                data_release=ugriz_dr,
            )
            if sub_err:
                print(f"error: ugriz failed for {plateifu}: {sub_err}", file=sys.stderr)
                return 1
        else:
            try:
                ugriz_files = download_ugriz_fits_cutouts(
                    ra=ra,
                    dec=dec,
                    out_dir=out_dir,
                    plate=plate,
                    ifu=ifu,
                    size_px=size,
                    scale_arcsec_per_px=scale,
                    data_release=ugriz_dr,
                )
            except Exception as e:
                print(f"error: ugriz cutout generation failed for {plateifu}: {e}", file=sys.stderr)
                return 1

    want_strict = (strict_ugriz or skip_jpeg) and not no_ugriz
    if want_strict and len(ugriz_files) < len(_UGRIZ):
        missing = [b for b in _UGRIZ if b not in ugriz_files]
        print(
            f"error: incomplete ugriz for {plateifu} (have {len(ugriz_files)}/{len(_UGRIZ)}; missing: {missing})",
            file=sys.stderr,
        )
        return 1

    if not no_ugriz and ugriz_files:
        removed = _purge_unlisted_ugriz_bands(out_dir, plate, ifu, set(ugriz_files.keys()))
        if removed:
            print(f"  removed stale bands not in download manifest: {removed}", flush=True)

        ok_shapes, shape_msg = _verify_ugriz_shapes(out_dir, plate, ifu, size)
        if not ok_shapes and len(ugriz_files) == len(_UGRIZ):
            print(f"warning: ugriz shape check failed for {plateifu}: {shape_msg}", file=sys.stderr)
        if want_strict and not ok_shapes:
            print(f"error: ugriz shape check failed for {plateifu}: {shape_msg}", file=sys.stderr)
            return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "plateifu": f"{plate}-{ifu}",
        "ra_deg": ra,
        "dec_deg": dec,
        "radec_source_file": source_name,
        "skyserver_dr": "dr17",
        "size_px": int(size),
        "scale_arcsec_per_px": float(scale),
        "jpeg_skipped": bool(skip_jpeg),
        "jpeg_url": None if skip_jpeg else jpeg_url,
        "jpeg_file": None if skip_jpeg else str(jpeg_path.name),
        "fits_url": fits_url,
        "fits_file": fits_path.name if fits_path else None,
        "ugriz_dr": int(ugriz_dr),
        "ugriz_files": ugriz_files,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved: {meta_path}")

    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "SDSS color JPEG (SkyServer) + per-band ugriz FITS (SAS frames + cutouts) for MaNGA targets. "
            "With no plate-ifu arguments, processes every galaxy folder under --data-root."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "plateifu",
        nargs="*",
        help="Plate-ifu IDs (e.g. 8485-1901). If omitted, all manga_sdss_fits/<plate>_<ifu>/ folders are used.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Parallelism only for --no-ugriz (JPEG-only). With ugriz downloads, galaxies "
            "always run one-by-one on the main thread (avoids native crashes from worker threads)."
        ),
    )
    p.add_argument("--data-root", type=Path, default=Path("manga_sdss_fits"), help="Root folder with <plate>_<ifu> subfolders")
    p.add_argument("--size", type=int, default=128, help="Cutout width/height in pixels")
    p.add_argument("--scale", type=float, default=0.198, help="Arcsec per pixel (default 0.198)")
    p.add_argument(
        "--opt",
        default="",
        help="SkyServer overlay options for JPEG (e.g. GLP). Default none.",
    )
    p.add_argument(
        "--with-fits",
        action="store_true",
        help="Also attempt SkyServer getfits cutout (endpoint availability varies).",
    )
    ugrp = p.add_mutually_exclusive_group()
    ugrp.add_argument(
        "--no-ugriz",
        action="store_true",
        help="JPEG (and optional SkyServer FITS) only; skip per-band ugriz FITS.",
    )
    ugrp.add_argument(
        "--ugriz-only",
        action="store_true",
        help="Per-band u/g/r/i/z FITS via astroquery only; skip SkyServer JPEG.",
    )
    p.add_argument(
        "--strict-ugriz",
        action="store_true",
        help="Fail if any ugriz band is missing (default when --ugriz-only).",
    )
    p.add_argument(
        "--ugriz-dr",
        type=int,
        default=18,
        help="SDSS data release for ugriz frame queries (default 18).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print resolved URLs and output paths only")
    p.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-check complete targets against metadata and re-run only if settings differ "
            "(size/scale/jpeg mode/ugriz DR)."
        ),
    )
    p.add_argument(
        "--ugriz-subprocess",
        action="store_true",
        help="Run ugriz in a child process (only if you still hit native crashes; default is in-process).",
    )
    args = p.parse_args(argv)

    _apply_blas_thread_env()

    if args.plateifu:
        targets = list(args.plateifu)
    else:
        targets = discover_plateifus_from_data_root(args.data_root)
        if not targets:
            print(
                f"No galaxy folders found under {args.data_root} (expected names like 8485_1901).",
                file=sys.stderr,
            )
            return 1

    require_jpeg = not args.ugriz_only
    require_ugriz = not args.no_ugriz
    workers = max(1, int(args.workers))

    # ThreadPoolExecutor always uses a worker thread (even with max_workers=1). On Windows,
    # astroquery + numpy/OpenBLAS inside worker threads often crash the process with no
    # traceback. Run the full JPEG+ugriz pipeline on the MAIN THREAD only.
    use_thread_pool = args.no_ugriz and workers > 1
    if not args.no_ugriz and workers > 1:
        print(
            "Note: --workers > 1 is ignored unless you pass --no-ugriz (JPEG-only). "
            "SDSS ugriz runs sequentially on the main thread to avoid native crashes.",
            file=sys.stderr,
        )

    print(
        f"Targets: {len(targets)}  data-root: {args.data_root}  "
        f"workers: {workers if use_thread_pool else 1} (main-thread ugriz)"
    )

    done = 0
    skipped = 0
    failed = 0

    def _run_one(idx: int, pi: str) -> tuple[str, str, str]:
        try:
            plate, ifu = parse_plateifu(pi)
        except ValueError as e:
            return "failed", pi, str(e)
        gal_dir = args.data_root / f"{plate}_{ifu}"
        if not gal_dir.is_dir():
            return "failed", pi, f"missing folder {gal_dir}"
        is_complete = cutouts_fully_complete(
            gal_dir,
            plate,
            ifu,
            require_jpeg=require_jpeg,
            require_ugriz=require_ugriz,
        )
        if is_complete:
            if not args.force:
                return "skipped", pi, "already complete"
            meta = load_cutout_metadata(gal_dir / "sdss_cutouts" / "metadata.json")
            same, reason = metadata_matches_request(
                meta,
                size=args.size,
                scale=args.scale,
                ugriz_dr=args.ugriz_dr,
                no_ugriz=args.no_ugriz,
                skip_jpeg=args.ugriz_only,
            )
            if same:
                return "skipped", pi, "already complete (matches requested settings)"
            print(f"  force-rerun: {pi} because {reason}", flush=True)
        rc = run_cutouts_for_plateifu(
            pi,
            data_root=args.data_root,
            size=args.size,
            scale=args.scale,
            opt=args.opt,
            with_fits=args.with_fits,
            no_ugriz=args.no_ugriz,
            skip_jpeg=args.ugriz_only,
            strict_ugriz=args.strict_ugriz,
            ugriz_dr=args.ugriz_dr,
            dry_run=args.dry_run,
            ugriz_subprocess=args.ugriz_subprocess,
        )
        return ("done", pi, "") if rc == 0 else ("failed", pi, f"exit {rc}")

    def _report(status: str, i: int, target: str, msg: str) -> None:
        nonlocal done, skipped, failed
        if status == "done":
            done += 1
            print(f"[{i}/{len(targets)}] DONE    {target}")
        elif status == "skipped":
            skipped += 1
            print(f"[{i}/{len(targets)}] SKIP    {target} ({msg})")
        else:
            failed += 1
            print(f"[{i}/{len(targets)}] FAILED  {target}")
            if msg:
                print(f"  {msg}", flush=True)

    if use_thread_pool:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut_map = {ex.submit(_run_one, i, pi): (i, pi) for i, pi in enumerate(targets, start=1)}
            for fut in as_completed(fut_map):
                i, pi = fut_map[fut]
                status, target, msg = fut.result()
                _report(status, i, target, msg)
    else:
        for i, pi in enumerate(targets, start=1):
            status, target, msg = _run_one(i, pi)
            _report(status, i, target, msg)

    print(f"\nSummary: done={done}  skipped={skipped}  failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

