"""
Run download_ugriz_fits_cutouts in a fresh Python process.

The parent (download_sdss_cutouts.py) invokes this so a segfault/stack overflow
inside astroquery/astropy/numpy on Windows kills only the child, not the bulk run.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        print("usage: ugriz_cutout_worker.py <job.json>", file=sys.stderr)
        return 2
    job_path = Path(argv[0])
    job = json.loads(job_path.read_text(encoding="utf-8"))

    try:
        from manga_prep.download.sdss_cutouts import download_ugriz_fits_cutouts

        out = download_ugriz_fits_cutouts(
            ra=float(job["ra"]),
            dec=float(job["dec"]),
            out_dir=Path(job["out_dir"]),
            plate=str(job["plate"]),
            ifu=str(job["ifu"]),
            size_px=int(job["size_px"]),
            scale_arcsec_per_px=float(job["scale_arcsec_per_px"]),
            data_release=int(job["data_release"]),
        )
        print(json.dumps(out))
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

