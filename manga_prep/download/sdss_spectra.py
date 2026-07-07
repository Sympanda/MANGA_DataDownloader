"""
Download nearest SDSS fiber spectrum for each MaNGA target.

For every galaxy folder under manga_sdss_fits/<plate>_<ifu>/, this script:
1. Reads RA/Dec from local MaNGA FITS headers (same as download_sdss_cutouts.py)
2. Queries SkyServer SpecObj for the nearest SDSS fiber spectrum
3. Downloads the spectrum from SAS (BOSS spec file or legacy spPlate extraction)
4. Writes a fast NPZ bundle plus metadata.json under sdss_spectra/

Examples:
  python download_sdss_spectra.py
  python download_sdss_spectra.py 7495-3702
  python download_sdss_spectra.py --search-arcmin 2 --workers 4
  python download_sdss_spectra.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from astropy.io import fits

from manga_prep.download.sdss_cutouts import (
    discover_plateifus_from_data_root,
    parse_plateifu,
    resolve_radec_from_folder,
    _skyserver_sql_rows,
)

SAS_BASE = "https://data.sdss.org/sas/dr17"


def angular_sep_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    ra1r, dec1r = math.radians(ra1), math.radians(dec1)
    ra2r, dec2r = math.radians(ra2), math.radians(dec2)
    cos_d = (
        math.sin(dec1r) * math.sin(dec2r)
        + math.cos(dec1r) * math.cos(dec2r) * math.cos(ra1r - ra2r)
    )
    cos_d = min(1.0, max(-1.0, cos_d))
    return math.degrees(math.acos(cos_d)) * 3600.0


def query_nearest_specobj(ra: float, dec: float, *, dr: int, search_arcmin: float) -> dict | None:
    delta = float(search_arcmin) / 60.0
    sql = (
        "SELECT TOP 1 "
        "s.specObjID, s.plate, s.mjd, s.fiberID, s.run2d, s.survey, s.programname, "
        "s.ra, s.dec, s.z, s.zErr, s.class, s.subClass "
        "FROM SpecObj AS s "
        f"WHERE s.ra BETWEEN {ra - delta:.10f} AND {ra + delta:.10f} "
        f"AND s.dec BETWEEN {dec - delta:.10f} AND {dec + delta:.10f} "
        f"ORDER BY (POWER((s.ra-{ra:.10f})*COS(RADIANS({dec:.10f})),2)+POWER(s.dec-{dec:.10f},2))"
    )
    rows = _skyserver_sql_rows(dr, sql)
    return rows[0] if rows else None


def download_bytes(url: str, timeout: int = 300) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "manga-sdss-spectra/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def wave_from_loglam_header(header: fits.Header, n_pix: int) -> np.ndarray:
    coeff0 = float(header["COEFF0"])
    coeff1 = float(header["COEFF1"])
    pix = np.arange(n_pix, dtype=np.float64)
    return (10.0 ** (coeff0 + coeff1 * pix)).astype(np.float32)


def arrays_from_spec_fits(data: bytes) -> tuple[dict[str, np.ndarray], dict]:
    with fits.open(
        __import__("io").BytesIO(data),
        memmap=False,
        ignore_missing_end=True,
    ) as hdul:
        flux_hdu = hdul[0]
        flux = np.asarray(flux_hdu.data, dtype=np.float32).reshape(-1)
        wave = wave_from_loglam_header(flux_hdu.header, flux.size)
        arrays = {"wave": wave, "flux": flux}
        meta = {
            "source_format": "spec-fits",
            "n_pix": int(flux.size),
        }
        if len(hdul) > 1 and hdul[1].data is not None:
            ivar = np.asarray(hdul[1].data, dtype=np.float32).reshape(-1)
            if ivar.size == flux.size:
                arrays["ivar"] = ivar
                meta["include_ivar"] = True
        if len(hdul) > 2 and hdul[2].data is not None:
            mask = np.asarray(hdul[2].data).reshape(-1)
            if mask.size == flux.size:
                arrays["mask"] = mask.astype(np.uint8)
                meta["include_mask"] = True
    return arrays, meta


def arrays_from_spplate(data: bytes, fiber_id: int) -> tuple[dict[str, np.ndarray], dict]:
    fiber_idx = int(fiber_id) - 1
    with fits.open(
        __import__("io").BytesIO(data),
        memmap=False,
        ignore_missing_end=True,
    ) as hdul:
        flux_cube = np.asarray(hdul[0].data, dtype=np.float32)
        if flux_cube.ndim != 2:
            raise ValueError(f"Unexpected spPlate flux shape: {flux_cube.shape}")
        if not (0 <= fiber_idx < flux_cube.shape[0]):
            raise IndexError(f"fiberID {fiber_id} out of range for spPlate with {flux_cube.shape[0]} fibers")

        flux = flux_cube[fiber_idx]
        wave = wave_from_loglam_header(hdul[0].header, flux.size)
        arrays = {"wave": wave, "flux": flux}
        meta = {
            "source_format": "spPlate",
            "n_pix": int(flux.size),
            "fiber_index": int(fiber_idx),
        }

        if len(hdul) > 2 and hdul[2].data is not None:
            ivar_cube = np.asarray(hdul[2].data, dtype=np.float32)
            if ivar_cube.ndim == 2 and ivar_cube.shape == flux_cube.shape:
                arrays["ivar"] = ivar_cube[fiber_idx]
                meta["include_ivar"] = True

        if len(hdul) > 3 and hdul[3].data is not None:
            mask_cube = np.asarray(hdul[3].data)
            if mask_cube.ndim == 2 and mask_cube.shape == flux_cube.shape:
                arrays["mask"] = mask_cube[fiber_idx].astype(np.uint8)
                meta["include_mask"] = True

    return arrays, meta


def build_download_plan(spec: dict) -> list[tuple[str, str]]:
    plate = str(spec["plate"])
    mjd = str(spec["mjd"])
    fiber = str(spec["fiberID"])
    run2d = str(spec["run2d"])
    spec_name = f"spec-{plate}-{mjd}-{fiber}.fits"
    spplate_name = f"spPlate-{plate}-{mjd}.fits"

    plans: list[tuple[str, str]] = []
    for prefix in ("sdss", "eboss", "boss"):
        plans.append(
            (
                "spec-fits",
                f"{SAS_BASE}/{prefix}/spectro/BOSS_SPECTRO_REDUX/{run2d}/spectra/{plate}/{spec_name}",
            )
        )
        plans.append(
            (
                "spec-fits",
                f"{SAS_BASE}/{prefix}/spectro/redux/{run2d}/spectra/{plate}/{spec_name}",
            )
        )
    for prefix in ("sdss", "eboss", "boss"):
        plans.append(
            (
                "spPlate",
                f"{SAS_BASE}/{prefix}/spectro/redux/{run2d}/{plate}/{spplate_name}",
            )
        )
    return plans


def cached_spplate_path(cache_dir: Path, plate: str, mjd: str) -> Path:
    return cache_dir / f"spPlate-{plate}-{mjd}.fits"


def fetch_sdss_spectrum(spec: dict, *, cache_dir: Path) -> tuple[dict[str, np.ndarray], dict, str]:
    last_error = None
    for kind, url in build_download_plan(spec):
        try:
            if kind == "spPlate":
                cache_path = cached_spplate_path(cache_dir, spec["plate"], spec["mjd"])
                if cache_path.exists() and cache_path.stat().st_size > 0:
                    data = cache_path.read_bytes()
                    source_url = str(cache_path)
                else:
                    data = download_bytes(url)
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    cache_path.write_bytes(data)
                    source_url = url
                arrays, parse_meta = arrays_from_spplate(data, int(spec["fiberID"]))
                parse_meta["source_url"] = source_url
                return arrays, parse_meta, source_url

            data = download_bytes(url)
            arrays, parse_meta = arrays_from_spec_fits(data)
            parse_meta["source_url"] = url
            return arrays, parse_meta, url
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"Could not download SDSS spectrum for specObj {spec.get('specObjID')}: {last_error}")


def spectrum_complete(out_dir: Path, plate: str, ifu: str) -> bool:
    npz = out_dir / f"sdss-{plate}-{ifu}-spectrum.npz"
    meta = out_dir / "metadata.json"
    return npz.exists() and npz.stat().st_size > 0 and meta.exists() and meta.stat().st_size > 0


def run_spectrum_for_plateifu(
    plateifu: str,
    *,
    data_root: Path,
    cache_dir: Path,
    dr: int,
    search_arcmin: float,
    dry_run: bool = False,
) -> int:
    plate, ifu = parse_plateifu(plateifu)
    gal_dir = data_root / f"{plate}_{ifu}"
    if not gal_dir.is_dir():
        print(f"Missing folder: {gal_dir}", file=sys.stderr)
        return 1

    try:
        ra, dec, source_name = resolve_radec_from_folder(gal_dir)
    except Exception as exc:
        print(f"Could not resolve RA/Dec for {plateifu}: {exc}", file=sys.stderr)
        return 1

    spec = query_nearest_specobj(ra, dec, dr=dr, search_arcmin=search_arcmin)
    if spec is None:
        print(
            f"No SDSS spectrum found within {search_arcmin} arcmin for {plateifu} "
            f"at RA/Dec {ra:.6f}, {dec:.6f}",
            file=sys.stderr,
        )
        return 1

    spec_ra = float(spec["ra"])
    spec_dec = float(spec["dec"])
    sep_arcsec = angular_sep_arcsec(ra, dec, spec_ra, spec_dec)

    out_dir = gal_dir / "sdss_spectra"
    npz_path = out_dir / f"sdss-{plate}-{ifu}-spectrum.npz"
    meta_path = out_dir / "metadata.json"

    print(f"\n{plateifu}")
    print(f"  target RA/Dec: {ra:.8f}, {dec:.8f}  (from {source_name})")
    print(
        f"  nearest SDSS fiber: plate={spec['plate']} mjd={spec['mjd']} "
        f"fiber={spec['fiberID']} survey={spec.get('survey')} class={spec.get('class')}"
    )
    print(f"  fiber RA/Dec: {spec_ra:.8f}, {spec_dec:.8f}  separation={sep_arcsec:.2f} arcsec")
    print(f"  -> {npz_path}")

    if dry_run:
        for kind, url in build_download_plan(spec)[:4]:
            print(f"  would try [{kind}] {url}")
        return 0

    arrays, parse_meta, source_url = fetch_sdss_spectrum(spec, cache_dir=cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)

    metadata = {
        "plateifu": f"{plate}-{ifu}",
        "target_ra_deg": float(ra),
        "target_dec_deg": float(dec),
        "radec_source_file": source_name,
        "search_arcmin": float(search_arcmin),
        "separation_arcsec": float(sep_arcsec),
        "skyserver_dr": int(dr),
        "specObjID": spec.get("specObjID"),
        "sdss_plate": int(spec["plate"]),
        "sdss_mjd": int(spec["mjd"]),
        "sdss_fiberID": int(spec["fiberID"]),
        "sdss_run2d": str(spec["run2d"]),
        "sdss_survey": spec.get("survey"),
        "sdss_programname": spec.get("programname"),
        "sdss_ra_deg": spec_ra,
        "sdss_dec_deg": spec_dec,
        "sdss_z": float(spec["z"]) if spec.get("z") not in (None, "") else None,
        "sdss_z_err": float(spec["zErr"]) if spec.get("zErr") not in (None, "") else None,
        "sdss_class": spec.get("class"),
        "sdss_subClass": spec.get("subClass"),
        "source_url": source_url,
        "npz_file": npz_path.name,
        "wave_unit": "Angstrom",
        "flux_unit": "10^-17 erg/s/cm^2/Angstrom (SDSS spec convention)",
        **parse_meta,
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved: {npz_path}")
    print(f"saved: {meta_path}")
    return 0


def load_sdss_spectrum(galaxy_dir: Path | str):
    galaxy_dir = Path(galaxy_dir)
    matches = sorted((galaxy_dir / "sdss_spectra").glob("sdss-*-spectrum.npz"))
    if not matches:
        raise FileNotFoundError(f"No SDSS spectrum NPZ found under {galaxy_dir / 'sdss_spectra'}")
    return np.load(matches[0])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download nearest SDSS fiber spectrum for each local MaNGA target and "
            "write sdss_spectra/*.npz for fast loading."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "plateifu",
        nargs="*",
        help="Optional plate-ifu IDs. If omitted, all manga_sdss_fits/<plate>_<ifu>/ folders are used.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("manga_sdss_fits"))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("sdss_spplate_cache"),
        help="Cache directory for downloaded spPlate files (shared across galaxies).",
    )
    parser.add_argument("--dr", type=int, default=17, help="SkyServer / SAS data release")
    parser.add_argument(
        "--search-arcmin",
        type=float,
        default=2.0,
        help="Search radius for nearest SpecObj match.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers across galaxies")
    parser.add_argument("--dry-run", action="store_true", help="Resolve matches and print paths only")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if sdss_spectra output already exists.",
    )
    args = parser.parse_args(argv)

    if args.plateifu:
        targets = list(args.plateifu)
    else:
        targets = discover_plateifus_from_data_root(args.data_root)
        if not targets:
            print(f"No galaxy folders found under {args.data_root}", file=sys.stderr)
            return 1

    workers = max(1, int(args.workers))
    print(
        f"Targets: {len(targets)}  data-root: {args.data_root}  "
        f"search: {args.search_arcmin} arcmin  workers: {workers}"
    )

    todo: list[str] = []
    skipped = 0
    for pi in targets:
        plate, ifu = parse_plateifu(pi)
        out_dir = args.data_root / f"{plate}_{ifu}" / "sdss_spectra"
        if not args.force and spectrum_complete(out_dir, plate, ifu):
            skipped += 1
            print(f"SKIP    {pi} (already complete)")
            continue
        todo.append(pi)

    if not todo:
        print(f"Done. skipped={skipped}")
        return 0

    failed = 0
    done = 0

    def _run(pi: str) -> tuple[str, int]:
        rc = run_spectrum_for_plateifu(
            pi,
            data_root=args.data_root,
            cache_dir=args.cache_dir,
            dr=args.dr,
            search_arcmin=args.search_arcmin,
            dry_run=args.dry_run,
        )
        return pi, rc

    if workers == 1:
        for pi in todo:
            _, rc = _run(pi)
            if rc == 0:
                done += 1
            else:
                failed += 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run, pi): pi for pi in todo}
            for future in as_completed(futures):
                _, rc = future.result()
                if rc == 0:
                    done += 1
                else:
                    failed += 1

    print(f"Summary: done={done} skipped={skipped} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
