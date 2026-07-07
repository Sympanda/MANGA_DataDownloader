"""FITS I/O helpers with astropy WCS fix warnings suppressed."""
from __future__ import annotations

import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from astropy.io import fits
from astropy.wcs import FITSFixedWarning, WCS


@contextmanager
def suppress_fits_wcs_warnings() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FITSFixedWarning)
        yield


@contextmanager
def open_fits(path: Path | str, **kwargs: Any) -> Iterator[fits.HDUList]:
    with suppress_fits_wcs_warnings():
        with fits.open(path, **kwargs) as hdul:
            yield hdul


def celestial_wcs_from_header(hdr) -> WCS:
    with suppress_fits_wcs_warnings():
        wcs = WCS(hdr)
        if wcs.has_celestial:
            return wcs.celestial
        return WCS(hdr, naxis=2)
