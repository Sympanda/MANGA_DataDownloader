"""
Unified CLI for MaNGA data preparation.

Usage:
  python -m manga_prep <command> [args...]

Examples:
  python -m manga_prep download-manga-sdss 8485-1901
  python -m manga_prep download-sdss-cutouts --workers 4
  python -m manga_prep export-pipe3d-maps --in-place --workers 8
  python -m manga_prep export-pipe3d-phys-maps --in-place --include-derived --drpall drpall-v3_1_1.fits --workers 8
  python -m manga_prep build-index --data-root manga_sdss_fits
"""
from __future__ import annotations

import sys
from typing import Callable

# Lazy imports: some export commands pull heavy optional deps (e.g. reproject).
COMMAND_META: dict[str, tuple[str, str]] = {
    "download-manga-sdss": ("manga_prep.download.manga_sdss", "Download MaNGA FITS from SDSS SAS"),
    "download-all-manga": ("manga_prep.download.all_manga", "Bulk download from DRPall list"),
    "download-pipe3d": ("manga_prep.download.pipe3d_only", "Download Pipe3D VAC cubes only"),
    "download-sdss-cutouts": ("manga_prep.download.sdss_cutouts", "SDSS JPEG + ugriz FITS cutouts"),
    "download-legacy-cutouts": ("manga_prep.download.legacy_cutouts", "Legacy Sky Viewer cutouts"),
    "download-legacy-coadd": (
        "manga_prep.download.legacy_coadd_cutouts",
        "Legacy NERSC coadd cutouts (recommended)",
    ),
    "download-sdss-spectra": (
        "manga_prep.download.sdss_spectra",
        "Nearest SDSS fiber spectrum per galaxy",
    ),
    "export-pipe3d-maps": (
        "manga_prep.export.pipe3d_maps",
        "Export legacy emission-line Amara maps (amara_maps.npz)",
    ),
    "export-pipe3d-phys-maps": (
        "manga_prep.export.pipe3d_phys_maps",
        "Export physical-property Pipe3D maps (amara_phys_maps.npz)",
    ),
    "export-pipe3d-global-flags": (
        "manga_prep.export.pipe3d_global_flags",
        "Export galaxy-level BPT / star-forming flags CSV",
    ),
    "export-aperture-spectra": (
        "manga_prep.export.aperture_spectra",
        "Fake SDSS-like aperture spectra from LOGCUBE",
    ),
    "export-manga-spectra": (
        "manga_prep.export.manga_spectra",
        "Full IFU spaxel cubes (not used by UNet)",
    ),
    "export-aligned-imaging": (
        "manga_prep.export.aligned_imaging",
        "Pre-align SDSS/Legacy to Amara grid",
    ),
    "compute-input-scales": (
        "manga_prep.export.compute_input_scales",
        "Train-split asinh soft scales for imaging + spectra",
    ),
    "build-index": ("manga_prep.export.build_index", "Build manga_dataset_index.csv"),
    "inventory": ("manga_prep.export.inventory", "Completeness report for local galaxy folders"),
    "validate-sdss-cutouts": (
        "manga_prep.export.validate_sdss_cutouts",
        "Check ugriz FITS band shape / metadata consistency",
    ),
    "thin-logcube": ("manga_prep.export.thin_logcube", "Strip large FITS, keep LOGCUBE only"),
    "ugriz-worker": (
        "manga_prep.io.ugriz_worker",
        "Subprocess worker for ugriz downloads (internal)",
    ),
}


def _load_command(module_path: str) -> Callable:
    import importlib

    module = importlib.import_module(module_path)
    return module.main


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        print("Commands:")
        for name, (_, desc) in sorted(COMMAND_META.items()):
            print(f"  {name:<28} {desc}")
        return 0

    cmd_name = argv[0]
    if cmd_name not in COMMAND_META:
        print(f"Unknown command: {cmd_name!r}. Run: python -m manga_prep --help", file=sys.stderr)
        return 2

    module_path, _ = COMMAND_META[cmd_name]
    fn = _load_command(module_path)
    return int(fn(argv[1:]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
