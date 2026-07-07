"""
Download MaNGA FITS from the SDSS SAS over HTTPS (no Marvin / Magrathea API).

Official layout (DR17): https://www.sdss4.org/dr17/manga/manga-data/data-access/

**â€œImagesâ€ (griz):** The DRP LOGCUBE file contains the reconstructed broadband images
(GIMG, RIMG, IIMG, ZIMG) and PSFs in FITS extensions â€” there is no separate
small â€œimage-onlyâ€ URL in the usual `stack/` tree. To get those, include `drp-cube`
(which is part of the default `all`).

Examples:
  python download_manga_sdss.py 8485-1901
  python download_manga_sdss.py 8485-1901 --what maps --dry-run
  python download_manga_sdss.py 8485-1901 --with-lin-cube --with-rss

For many galaxies, use rsync / wget -i as on the SDSS â€œData Accessâ€ page.
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

DR = "dr17"
DRPVER = "v3_1_1"
DAPVER = "3.1.0"
PIPE3DVER = "3.1.1"
_DEFAULT_DAPTYPE = "HYB10-MILESHC-MASTARHC2"

BASE_DRP = f"https://data.sdss.org/sas/{DR}/manga/spectro/redux/{DRPVER}"
BASE_DAP = f"https://data.sdss.org/sas/{DR}/manga/spectro/analysis/{DRPVER}/{DAPVER}"
BASE_PIPE3D = f"https://data.sdss.org/sas/{DR}/env/MANGA_PIPE3D/{DRPVER}/{PIPE3DVER}"


def urls_for_plateifu(
    plate: str,
    ifu: str,
    daptype: str,
    *,
    want_maps: bool,
    want_drp_logcube: bool,
    want_drp_lincube: bool,
    want_drp_logrss: bool,
    want_drp_linrss: bool,
    want_dap_logcube: bool,
    want_pipe3d_vac: bool,
) -> list[tuple[str, str]]:
    """Return (url, filename) pairs."""
    plate, ifu = str(plate), str(ifu)
    out: list[tuple[str, str]] = []

    stack = f"{BASE_DRP}/{plate}/stack"
    if want_drp_logcube:
        fn = f"manga-{plate}-{ifu}-LOGCUBE.fits.gz"
        out.append((f"{stack}/{fn}", fn))
    if want_drp_lincube:
        fn = f"manga-{plate}-{ifu}-LINCUBE.fits.gz"
        out.append((f"{stack}/{fn}", fn))
    if want_drp_logrss:
        fn = f"manga-{plate}-{ifu}-LOGRSS.fits.gz"
        out.append((f"{stack}/{fn}", fn))
    if want_drp_linrss:
        fn = f"manga-{plate}-{ifu}-LINRSS.fits.gz"
        out.append((f"{stack}/{fn}", fn))

    dapdir = f"{BASE_DAP}/{daptype}/{plate}/{ifu}"
    if want_maps:
        fn = f"manga-{plate}-{ifu}-MAPS-{daptype}.fits.gz"
        out.append((f"{dapdir}/{fn}", fn))
    if want_dap_logcube:
        fn = f"manga-{plate}-{ifu}-LOGCUBE-{daptype}.fits.gz"
        out.append((f"{dapdir}/{fn}", fn))

    if want_pipe3d_vac:
        fn = f"manga-{plate}-{ifu}.Pipe3D.cube.fits.gz"
        out.append((f"{BASE_PIPE3D}/{plate}/{fn}", fn))

    return out


def download_url(
    url: str, dest: Path, chunk: int = 8 * 1024 * 1024, *, verbose: bool = True
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "manga-downloader/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
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
                if verbose and n:
                    pct = 100.0 * got / n
                    print(
                        f"\r  {dest.name}  {got / (1024**2):.1f} / {n / (1024**2):.1f} MiB ({pct:.1f}%)",
                        end="",
                        file=sys.stderr,
                    )
        if verbose and n:
            print(file=sys.stderr)


def parse_plateifu(s: str) -> tuple[str, str]:
    parts = s.replace(" ", "").split("-")
    if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
        raise ValueError(f"expected plate-ifu like 8485-1901, got {s!r}")
    return parts[0], parts[1]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Download MaNGA FITS from SDSS SAS (HTTPS). "
            "Default: DRP LOGCUBE only (most complete across galaxies)."
        ),
    )
    p.add_argument("plateifu", nargs="+", help="One or more plate-ifu ids, e.g. 8485-1901")
    p.add_argument(
        "--what",
        choices=("all", "maps", "drp-cube", "dap-cube", "both"),
        default="drp-cube",
        help=(
            "drp-cube (default)= DRP LOGCUBE only; "
            "all = DAP MAPS + DRP LOGCUBE + DAP model LOGCUBE + Pipe3D VAC cube; "
            "maps / drp-cube / dap-cube = single product; "
            "both = MAPS + DRP LOGCUBE only (no DAP model cube)"
        ),
    )
    p.add_argument(
        "--with-lin-cube",
        action="store_true",
        help="Also download DRP LINCUBE (linear lambda cube). Large.",
    )
    p.add_argument(
        "--with-rss",
        action="store_true",
        help="Also download DRP LOGRSS and LINRSS. Very large.",
    )
    p.add_argument("--out", type=Path, default=Path("manga_sdss_fits"), help="Output directory")
    p.add_argument(
        "--daptype",
        default=_DEFAULT_DAPTYPE,
        help=f"DAP analysis folder name (default {_DEFAULT_DAPTYPE!r})",
    )
    p.add_argument(
        "--no-pipe3d-vac",
        action="store_true",
        help=(
            "Do not download per-galaxy Pipe3D VAC file "
            "(manga-PLATE-IFU.Pipe3D.cube.fits.gz)."
        ),
    )
    p.add_argument("--dry-run", action="store_true", help="Print URLs only")
    args = p.parse_args(argv)

    w = args.what
    if w == "all":
        want_maps = want_drp_log = want_dap_log = True
    elif w == "maps":
        want_maps, want_drp_log, want_dap_log = True, False, False
    elif w == "drp-cube":
        want_maps, want_drp_log, want_dap_log = False, True, False
    elif w == "dap-cube":
        want_maps, want_drp_log, want_dap_log = False, False, True
    else:  # both
        want_maps, want_drp_log, want_dap_log = True, True, False
    want_pipe3d_vac = (w == "all") and (not args.no_pipe3d_vac)

    tasks: list[tuple[str, Path]] = []
    for pi in args.plateifu:
        plate, ifu = parse_plateifu(pi)
        pairs = urls_for_plateifu(
            plate,
            ifu,
            args.daptype,
            want_maps=want_maps,
            want_drp_logcube=want_drp_log,
            want_drp_lincube=args.with_lin_cube,
            want_drp_logrss=args.with_rss,
            want_drp_linrss=args.with_rss,
            want_dap_logcube=want_dap_log,
            want_pipe3d_vac=want_pipe3d_vac,
        )
        for url, fname in pairs:
            tasks.append((url, args.out / f"{plate}_{ifu}" / fname))

    for url, dest in tasks:
        print(url)
        if args.dry_run:
            continue
        try:
            download_url(url, dest)
            print("saved:", dest)
        except urllib.error.HTTPError as e:
            print(f"HTTP {e.code} for {url}", file=sys.stderr)
            return 1
        except urllib.error.URLError as e:
            print(f"URL error for {url}: {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


