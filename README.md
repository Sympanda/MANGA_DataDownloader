# MaNGA Data Prep (SDSS + Legacy)

Python launchers (in this folder) call `manga_prep.*` for downloads and cutouts. Every script supports **`--help`** for the authoritative option list; the sections below mirror that output.

**Typical order:** MaNGA FITS → SDSS cutouts → Legacy cutouts (Sky Viewer or NERSC coadd) → inventory.

---

## `download_manga_sdss.py`

Per-galaxy MaNGA products from SDSS SAS (HTTPS). Default is **DRP LOGCUBE** only; use `--what all` for DAP MAPS, DRP LOGCUBE, DAP model LOGCUBE, and Pipe3D VAC.

```bash
python download_manga_sdss.py 8485-1901
python download_manga_sdss.py --help
```

```
usage: download_manga_sdss.py [-h] [--what {all,maps,drp-cube,dap-cube,both}]
                              [--with-lin-cube] [--with-rss] [--out OUT]
                              [--daptype DAPTYPE] [--no-pipe3d-vac]
                              [--dry-run]
                              plateifu [plateifu ...]

Download MaNGA FITS from SDSS SAS (HTTPS). Default: DRP LOGCUBE only (most
complete across galaxies).

positional arguments:
  plateifu              One or more plate-ifu ids, e.g. 8485-1901

options:
  -h, --help            show this help message and exit
  --what {all,maps,drp-cube,dap-cube,both}
                        drp-cube (default)= DRP LOGCUBE only; all = DAP MAPS +
                        DRP LOGCUBE + DAP model LOGCUBE + Pipe3D VAC cube;
                        maps / drp-cube / dap-cube = single product; both =
                        MAPS + DRP LOGCUBE only (no DAP model cube)
  --with-lin-cube       Also download DRP LINCUBE (linear lambda cube). Large.
  --with-rss            Also download DRP LOGRSS and LINRSS. Very large.
  --out OUT             Output directory
  --daptype DAPTYPE     DAP analysis folder name (default 'HYB10-MILESHC-
                        MASTARHC2')
  --no-pipe3d-vac       Do not download per-galaxy Pipe3D VAC file (manga-
                        PLATE-IFU.Pipe3D.cube.fits.gz).
  --dry-run             Print URLs only
```

---

## `download_all_manga.py`

Bulk MaNGA from the DRPall plate–IFU list. **Imaging is not included**; use the SDSS and Legacy cutout scripts afterward.

```bash
python download_all_manga.py
python download_all_manga.py --limit 5 --dry-run
python download_all_manga.py --limit 50
python download_all_manga.py --help
```

```
usage: download_all_manga.py [-h] [--out-root OUT_ROOT] [--drpall DRPALL]
                             [--daptype DAPTYPE] [--start START]
                             [--limit LIMIT] [--retries RETRIES]
                             [--object-workers OBJECT_WORKERS]
                             [--file-workers FILE_WORKERS] [--dry-run]
                             [plateifu ...]

Bulk-download MaNGA DR17 galaxies from DRPall plateifu list. Default per
target: DRP LOGCUBE + DAP MAPS + DAP model LOGCUBE + Pipe3D VAC. Use
download_sdss_cutouts.py / download_legacy_cutouts.py for imaging.

positional arguments:
  plateifu              Optional explicit plate-ifu list. If omitted, targets
                        are read from DRPall.

options:
  -h, --help            show this help message and exit
  --out-root OUT_ROOT   Output root folder
  --drpall DRPALL       Local DRPall FITS path (downloaded if missing)
  --daptype DAPTYPE     DAP analysis type
  --start START         Start index in DRPall plateifu list
  --limit LIMIT         Max number of galaxies (0 = all)
  --retries RETRIES     Retries per file on network errors
  --object-workers OBJECT_WORKERS
                        Parallel workers across galaxies/plateifus (default
                        1).
  --file-workers FILE_WORKERS
                        Parallel workers per galaxy for MaNGA file downloads
                        (default 4).
  --dry-run             Print actions only; do not download
```

---

## `download_sdss_cutouts.py`

SDSS color JPEG (SkyServer) and per-band **ugriz** FITS (SAS). With no `plateifu` arguments, processes every `manga_sdss_fits/<plate>_<ifu>/` folder.

```bash
python download_sdss_cutouts.py
python download_sdss_cutouts.py --help
```

```
usage: download_sdss_cutouts.py [-h] [--workers WORKERS]
                                [--data-root DATA_ROOT] [--size SIZE]
                                [--scale SCALE] [--opt OPT] [--with-fits]
                                [--no-ugriz | --ugriz-only] [--strict-ugriz]
                                [--ugriz-dr UGRIZ_DR] [--dry-run]
                                [--ugriz-subprocess]
                                [plateifu ...]

SDSS color JPEG (SkyServer) + per-band ugriz FITS (SAS frames + cutouts) for
MaNGA targets. With no plate-ifu arguments, processes every galaxy folder
under --data-root.

positional arguments:
  plateifu              Plate-ifu IDs (e.g. 8485-1901). If omitted, all
                        manga_sdss_fits/<plate>_<ifu>/ folders are used.
                        (default: None)

options:
  -h, --help            show this help message and exit
  --workers WORKERS     Parallelism only for --no-ugriz (JPEG-only). With
                        ugriz downloads, galaxies always run one-by-one on the
                        main thread (avoids native crashes from worker
                        threads). (default: 1)
  --data-root DATA_ROOT
                        Root folder with <plate>_<ifu> subfolders (default:
                        manga_sdss_fits)
  --size SIZE           Cutout width/height in pixels (default: 128)
  --scale SCALE         Arcsec per pixel (default 0.198) (default: 0.198)
  --opt OPT             SkyServer overlay options for JPEG (e.g. GLP). Default
                        none. (default: )
  --with-fits           Also attempt SkyServer getfits cutout (endpoint
                        availability varies). (default: False)
  --no-ugriz            JPEG (and optional SkyServer FITS) only; skip per-band
                        ugriz FITS. (default: False)
  --ugriz-only          Per-band u/g/r/i/z FITS via astroquery only; skip
                        SkyServer JPEG. (default: False)
  --strict-ugriz        Fail if any ugriz band is missing (default when
                        --ugriz-only). (default: False)
  --ugriz-dr UGRIZ_DR   SDSS data release for ugriz frame queries (default
                        18). (default: 18)
  --dry-run             Print resolved URLs and output paths only (default:
                        False)
  --ugriz-subprocess    Run ugriz in a child process (only if you still hit
                        native crashes; default is in-process). (default:
                        False)
```

---

## `download_legacy_cutouts.py` (Sky Viewer)

Legacy Survey **cutout.jpg / cutout.fits** from `legacysurvey.org/viewer/`. Handy for small tests; for large batches prefer **`download_legacy_coadd_cutouts.py`**.

```bash
python download_legacy_cutouts.py
python download_legacy_cutouts.py --help
```

```
usage: download_legacy_cutouts.py [-h] [--data-root DATA_ROOT]
                                  [--workers WORKERS] [--layer LAYER]
                                  [--pixscale PIXSCALE] [--size SIZE]
                                  [--bands BANDS] [--no-fallback-grz]
                                  [--no-jpeg] [--dry-run] [--retries RETRIES]
                                  [plateifu ...]

Download Legacy Survey cutouts for local MaNGA folders.

positional arguments:
  plateifu              Optional plate-ifu list. If omitted, process all
                        <plate>_<ifu> folders.

options:
  -h, --help            show this help message and exit
  --data-root DATA_ROOT
  --workers WORKERS     Parallel galaxies
  --layer LAYER         Legacy viewer layer
  --pixscale PIXSCALE   Arcsec per pixel
  --size SIZE           Cutout size in pixels
  --bands BANDS         Bands string, e.g. griz
  --no-fallback-grz     Disable automatic fallback from griz to grz if any
                        requested band fails.
  --no-jpeg             Skip JPEG
  --dry-run
  --retries RETRIES     Retries per HTTP request (429/5xx/URLError) with
                        backoff.
```

---

## `download_legacy_coadd_cutouts.py` (NERSC bricks)

Public **release coadds** from `portal.nersc.gov`, then local WCS cutouts. Caches full bricks under **`legacy_coadd_brick_cache/`** (shared across galaxies in the same brick). Typical “production” run: 198×198, as many bands as the release allows, plus JPEG.

**Brick assignment:** a single (RA, Dec) maps to one `BRICKNAME` in `survey-bricks` using `RA1, RA2, DEC1, DEC2`. NERSC `.../north/...` vs `.../south/...` are the Legacy Surveys coadd *directory* split, not two different geometric bricks. For each coadd, the code tries, in order, **`dr10/south` → `dr10/north` → `dr9/south` → `dr9/north`** and uses the first URL that has `legacysurvey-<brick>-image-<band>.fits.fz` (stops as soon as one path works). A “band not found” warning after that order usually means the **filter was never released for that coadd** (e.g. no *i* in that DR/hemisphere), not a wrong brick. **DR11** is not on `https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr11/` with this layout, so it is not searched. Metadata per galaxy includes the brick’s RA/Dec box and a boolean that the object lies inside that box.

```bash
python download_legacy_coadd_cutouts.py --data-root manga_sdss_fits --size 198 --bands griz --jpeg
python download_legacy_coadd_cutouts.py --force --data-root manga_sdss_fits --size 198 --bands griz --jpeg
python download_legacy_coadd_cutouts.py --help
```

```
usage: download_legacy_coadd_cutouts.py [-h] [--data-root DATA_ROOT]
                                        [--brick-cache BRICK_CACHE]
                                        [--workers WORKERS] [--size SIZE]
                                        [--bands BANDS] [--no-fallback-grz]
                                        [--jpeg] [--force] [--dry-run]
                                        [--retries RETRIES]
                                        [plateifu ...]

Legacy Survey cutouts from NERSC release coadds (recommended for large
batches). FITS are nanomaggy AB per pixel at native 0.262 arcsec/pixel.

positional arguments:
  plateifu              Optional plate-ifu list. If omitted, process all
                        <plate>_<ifu> folders.

options:
  -h, --help            show this help message and exit
  --data-root DATA_ROOT
  --brick-cache BRICK_CACHE
                        Directory to cache full-brick
                        legacysurvey-*-image-*.fits.fz files (reuse across
                        galaxies).
  --workers WORKERS     Parallel galaxies (keep low; each brick is large).
  --size SIZE           Cutout size in pixels (native pixscale).
  --bands BANDS         Bands to extract, e.g. grz or griz (i may 404 on older
                        DR9 north).
  --no-fallback-grz     When --bands griz, do not retry grz if any band is
                        missing.
  --jpeg                Write legacy-*-color.jpg from g,r,z (needs
                        matplotlib).
  --force               Re-build cutouts even if legacy_cutouts files already
                        exist (e.g. replacing viewer cutouts).
  --dry-run
  --retries RETRIES     Retries per brick FITS download.
```

---

## `inventory_manga_completeness.py`

Scans `manga_sdss_fits/<plate>_<ifu>/` for DRP / DAP / Pipe3D / SDSS **sdss_cutouts** presence (Legacy is not in this report).

```bash
python inventory_manga_completeness.py
python inventory_manga_completeness.py --help
```

```
usage: inventory_manga_completeness.py [-h] [--data-root DATA_ROOT]
                                       [--json-out JSON_OUT]
                                       [--details-out DETAILS_OUT]

Inventory completeness of local MaNGA galaxy folders.

options:
  -h, --help            show this help message and exit
  --data-root DATA_ROOT
  --json-out JSON_OUT   Optional path to write JSON report
  --details-out DETAILS_OUT
                        Optional path to write per-galaxy flags JSONL
```

---

## Next step

After downloads, downstream scripts in this project can turn raw FITS into ML-ready tensors.
