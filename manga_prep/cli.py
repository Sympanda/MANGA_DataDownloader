"""
Unified CLI for MaNGA data preparation.

Usage:
  python -m manga_prep <command> [args...]

Examples:
  python -m manga_prep download-manga-sdss 8485-1901
  python -m manga_prep download-sdss-cutouts --workers 4
  python -m manga_prep export-pipe3d-maps --in-place --workers 8
  python -m manga_prep build-index --data-root manga_sdss_fits
"""
from __future__ import annotations

import argparse
import sys
from typing import Callable

from manga_prep.download.all_manga import main as download_all_manga
from manga_prep.download.legacy_coadd_cutouts import main as download_legacy_coadd
from manga_prep.download.legacy_cutouts import main as download_legacy
from manga_prep.download.manga_sdss import main as download_manga_sdss
from manga_prep.download.pipe3d_only import main as download_pipe3d_only
from manga_prep.download.sdss_cutouts import main as download_sdss_cutouts
from manga_prep.download.sdss_spectra import main as download_sdss_spectra
from manga_prep.export.aligned_imaging import main as export_aligned_imaging
from manga_prep.export.aperture_spectra import main as export_aperture_spectra
from manga_prep.export.build_index import main as build_index
from manga_prep.export.inventory import main as inventory
from manga_prep.export.manga_spectra import main as export_manga_spectra
from manga_prep.export.pipe3d_maps import main as export_pipe3d_maps
from manga_prep.export.thin_logcube import main as thin_logcube
from manga_prep.io.ugriz_worker import main as ugriz_worker


COMMANDS: dict[str, tuple[Callable, str]] = {
    "download-manga-sdss": (download_manga_sdss, "Download MaNGA FITS from SDSS SAS"),
    "download-all-manga": (download_all_manga, "Bulk download from DRPall list"),
    "download-pipe3d": (download_pipe3d_only, "Download Pipe3D VAC cubes only"),
    "download-sdss-cutouts": (download_sdss_cutouts, "SDSS JPEG + ugriz FITS cutouts"),
    "download-legacy-cutouts": (download_legacy, "Legacy Sky Viewer cutouts"),
    "download-legacy-coadd": (download_legacy_coadd, "Legacy NERSC coadd cutouts (recommended)"),
    "download-sdss-spectra": (download_sdss_spectra, "Nearest SDSS fiber spectrum per galaxy"),
    "export-pipe3d-maps": (export_pipe3d_maps, "Export Amara map targets from Pipe3D"),
    "export-aperture-spectra": (export_aperture_spectra, "Fake SDSS-like aperture spectra from LOGCUBE"),
    "export-manga-spectra": (export_manga_spectra, "Full IFU spaxel cubes (not used by UNet)"),
    "export-aligned-imaging": (export_aligned_imaging, "Pre-align SDSS/Legacy to Amara grid"),
    "build-index": (build_index, "Build manga_dataset_index.csv"),
    "inventory": (inventory, "Completeness report for local galaxy folders"),
    "thin-logcube": (thin_logcube, "Strip large FITS, keep LOGCUBE only"),
    "ugriz-worker": (ugriz_worker, "Subprocess worker for ugriz downloads (internal)"),
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        print("Commands:")
        for name, (_, desc) in sorted(COMMANDS.items()):
            print(f"  {name:<28} {desc}")
        return 0

    cmd_name = argv[0]
    if cmd_name not in COMMANDS:
        print(f"Unknown command: {cmd_name!r}. Run: python -m manga_prep --help", file=sys.stderr)
        return 2

    fn, _ = COMMANDS[cmd_name]
    return int(fn(argv[1:]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
