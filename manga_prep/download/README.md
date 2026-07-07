# MaNGA Data Prep (Downloads & Exports)

Scripts live under `manga_prep/download/` and `manga_prep/export/`. Invoke via the unified CLI:

```bash
python -m manga_prep --help          # list all commands
python -m manga_prep <command> --help  # per-command options
```

**Typical order:** MaNGA FITS → SDSS cutouts → fake/real spectra → Pipe3D map export → aligned imaging cache → dataset index.

**Default data root:** `manga_sdss_fits/` (per-galaxy folders `manga_sdss_fits/<plate>_<ifu>/`).

All commands: `python -m manga_prep <command> --help`

---

## CLI command reference

| Command | Purpose |
|---------|---------|
| `download-manga-sdss` | Per-galaxy MaNGA FITS from SDSS SAS |
| `download-all-manga` | Bulk download from DRPall list |
| `download-pipe3d` | Pipe3D VAC cubes only |
| `download-sdss-cutouts` | SDSS JPEG + ugriz FITS cutouts |
| `download-legacy-cutouts` | Legacy Sky Viewer cutouts (small tests) |
| `download-legacy-coadd` | Legacy NERSC coadd cutouts (**recommended**) |
| `download-sdss-spectra` | Nearest SDSS fiber spectrum |
| `export-pipe3d-maps` | Amara 0–1 scaled map targets from Pipe3D |
| `export-aperture-spectra` | Fake SDSS-like aperture spectra from LOGCUBE |
| `export-manga-spectra` | Full IFU spaxel cubes (not used by UNet) |
| `export-aligned-imaging` | Pre-align SDSS/Legacy to Amara 76×76 grid |
| `build-index` | Build `manga_dataset_index.csv` |
| `inventory` | Completeness report for local folders |
| `thin-logcube` | Strip large FITS, keep LOGCUBE only |

---

## download-manga-sdss

Per-galaxy MaNGA products from SDSS SAS. Default: **DRP LOGCUBE** only; `--what all` adds DAP MAPS, DAP model LOGCUBE, and Pipe3D VAC.

```bash
python -m manga_prep download-manga-sdss 8485-1901
python -m manga_prep download-manga-sdss 8485-1901 --what all --dry-run
```

Key options: `--out` (default `manga_sdss_fits`), `--what {all,maps,drp-cube,dap-cube,both}`, `--no-pipe3d-vac`.

---

## download-pipe3d

Download **only** `manga-PLATE-IFU.Pipe3D.cube.fits.gz` for existing local folders. Skips existing files by default.

```bash
python -m manga_prep download-pipe3d --dry-run
python -m manga_prep download-pipe3d --workers 16
```

---

## download-all-manga

Bulk MaNGA from DRPall. **No imaging** — run cutout commands afterward.

```bash
python -m manga_prep download-all-manga --limit 50 --dry-run
```

---

## download-sdss-cutouts

SDSS color JPEG + per-band **ugriz** FITS. With no plate-ifu args, processes every folder under `--data-root`.

```bash
python -m manga_prep download-sdss-cutouts
python -m manga_prep download-sdss-cutouts --ugriz-only --workers 1
```

Output: `manga_sdss_fits/<plate>_<ifu>/sdss_cutouts/`

---

## download-legacy-coadd

Legacy Survey cutouts from **NERSC release coadds** (recommended for large batches). Caches bricks under `legacy_coadd_brick_cache/`.

```bash
python -m manga_prep download-legacy-coadd --size 198 --bands griz --jpeg
```

---

## download-sdss-spectra

Nearest SDSS fiber spectrum per MaNGA target.

```bash
python -m manga_prep download-sdss-spectra --workers 4
```

Output per galaxy: `sdss_spectra/sdss-<plate>-<ifu>-spectrum.npz`  
Shared cache: `sdss_spplate_cache/`

```python
from manga_prep.download.sdss_spectra import load_sdss_spectrum
spec = load_sdss_spectrum("manga_sdss_fits/7495_3702")
```

---

## export-pipe3d-maps

Export 6 Pipe3D science maps into `amara_maps.npz` (0–1 scaled, 76×76). See [`../targets/README.md`](../targets/README.md) for scaling rules.

```bash
python -m manga_prep export-pipe3d-maps --in-place --data-root manga_sdss_fits --workers 8 --skip-existing
```

```python
from manga_prep.targets.pipe3d_maps import load_amara_maps
maps = load_amara_maps("manga_sdss_fits/7495_3702")
```

---

## export-aperture-spectra

Fake SDSS-like spectra from MaNGA LOGCUBE spaxels (aperture coadd).

```bash
python -m manga_prep export-aperture-spectra --workers 8 --skip-existing
```

```python
from manga_prep.io.aperture_spectrum import load_fake_sdss_spectrum
spec = load_fake_sdss_spectrum("manga_sdss_fits/7495_3702")
```

---

## export-aligned-imaging

One-time WCS alignment of cutouts to the Amara grid (speeds up training).

```bash
python -m manga_prep export-aligned-imaging --survey sdss --use-index --skip-existing --workers 8
```

---

## build-index

Scan `manga_sdss_fits/` and write modality flags to `manga_dataset_index.csv`.

```bash
python -m manga_prep build-index --data-root manga_sdss_fits
```

---

## Per-galaxy folder layout

```text
manga_sdss_fits/<plate>_<ifu>/
  manga-<plate>-<ifu>-LOGCUBE.fits.gz
  manga-<plate>-<ifu>.Pipe3D.cube.fits.gz
  amara_maps.npz
  sdss_cutouts/sdss-<plate>-<ifu>-{u,g,r,i,z}.fits
  legacy_cutouts/legacy-<plate>-<ifu>-{g,r,i,z}.fits
  fake_sdss_spectra/manga-*-fake-sdss-spectrum-30mas.npz
  sdss_spectra/sdss-*-spectrum.npz          # optional
  aligned_imaging/sdss_aligned.npz          # optional cache
```

---

## Package layout

```
manga_prep/
  download/     # FITS & cutout downloads
  export/       # Maps, spectra, index, inventory
  targets/      # Pipe3D map definitions (library)
  dataset/      # PyTorch dataset
  io/           # FITS I/O, WCS alignment, caches
  cli.py        # Unified entry point
  paths.py      # DEFAULT_DATA_ROOT and cache paths
```
