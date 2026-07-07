# MaNGA UNet dataloader

Multimodal dataset for predicting Amara Pipe3D map targets from imaging and optional spectrum.

## Task

**Inputs**

- SDSS ugriz cutouts (`sdss_cutouts/*.fits`) — optional
- Legacy/DECaLS grz/griz cutouts (`legacy_cutouts/*.fits`) — optional
- 1D spectrum — optional (`spectrum=None`, `"real"`, or `"fake"`)

**Targets**

Six Amara map channels on a fixed **76×76** canvas (center-padded from native Pipe3D size):

| Key | Quantity |
|-----|----------|
| `ha_flux` | Hα flux |
| `hbeta_flux` | Hβ flux |
| `oiii_5007_flux` | [OIII]5007 flux |
| `nii_6584_flux` | [NII]6584 flux |
| `ha_ew` | Hα equivalent width |
| `stellar_av` | Stellar A_V |

Targets use **Amara fixed physical clipping** → 0–1 scaled values (reversible via `amara_maps_metadata.json`).

## Masks

Three mask types per galaxy (stored in `amara_maps.npz`):

| Mask | Source | Use |
|------|--------|-----|
| `native_footprint_mask` | Pipe3D `SELECT_REG` | IFU analysis region (hex footprint in rectangular grid) |
| `{feature}_valid_mask` | Amara transform (finite values) | Per-feature science validity |
| `{feature}_loss_mask` | `footprint & valid` | **Use this for training loss** |

Padding outside the native map is 0 in all masks. Do not compute loss on padded pixels.

## Spectrum modes

```python
spectrum=None    # imaging-only model
spectrum="fake"  # MaNGA 3″ aperture coadd (every galaxy)
spectrum="real"  # nearest SDSS fiber when available; falls back to fake by default
```

Spectra are resampled to a fixed wavelength grid (`3622–10354 Å`, 4563 bins) for batching.

## Input imaging orientation

Survey cutout FITS files are **reprojected onto the Pipe3D / Amara spaxel WCS**
(same 76×76 canvas as map targets). This matches the workflow in
`sdss_legacy_fits_jpeg_comparison.ipynb` — SDSS frame cutouts in particular are
not north-up in native orientation and must not be used raw.

Set `align_imaging_to_amara_grid=False` only for debugging raw cutouts.

Requires `reproject` (`pip install reproject`).

**Training uses raw aligned flux** — not the percentile scaling in the preview notebook
(that scaling is display-only).

## Conditional UNet

See `manga_models/` and `runner.py` (or `scripts/legacy/train_conditional_unet.py`).

v1 conditioning:
- **SDSS / Legacy**: channel-concat at UNet input (both aligned to Amara grid when `align_imaging_to_amara_grid=True`)
- **Footprint mask**: optional extra input channel
- **Spectrum**: 1D CNN → **FiLM at bottleneck only** (simple first pass)

```bash
python runner.py --config config.jsonc --run-name smoke --epochs 1
# or legacy: python scripts/legacy/train_conditional_unet.py --epochs 1 --batch-size 4 --max-batches 5 --device cpu
```

## Quick start

```python
from torch.utils.data import DataLoader
from manga_prep.manga_dataset import MangaGalaxyDataset, collate_manga_batch, masked_mse_loss

dataset = MangaGalaxyDataset(
    "manga_sdss_fits",
    include_sdss_imaging=True,
    include_targets=True,
    spectrum="fake",
    require_all=True,
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_manga_batch,
)

batch = next(iter(loader))
# batch["inputs"]["sdss_imaging"]  -> (B, 5, H, W)
# batch["inputs"]["spectrum"]["flux"] -> (B, 4563)
# batch["targets"]["ha_flux"]      -> (B, 76, 76)
# batch["target_loss_masks"]["ha_flux"] -> (B, 76, 76)
```

### Masked loss example

```python
import torch

pred = model(...)  # (B, 76, 76) for one target channel
loss = masked_mse_loss(
    pred,
    batch["targets"]["ha_flux"],
    batch["target_loss_masks"]["ha_flux"],
)
```

For multi-channel UNet output, apply `masked_mse_loss` per channel or stack channels and mask accordingly.

## Sample dict layout

```python
sample = dataset[i]
sample["plateifu"]
sample["inputs"]["sdss_imaging"]["data"]   # (5, H, W)
sample["inputs"]["spectrum"]["flux"]         # (4563,) if spectrum enabled
sample["targets"]["ha_flux"]                 # (76, 76)
sample["target_loss_masks"]["ha_flux"]       # (76, 76), uint8
sample["footprint_mask"]                     # (76, 76)
sample["native_shape"]                       # e.g. (44, 44)
```

## Index and exports

Build/update the unified index:

```bash
python -m manga_prep build-index --data-root manga_sdss_fits
```

Refresh footprint and loss masks in existing NPZ files (after `SELECT_REG` fix):

```bash
python -m manga_prep export-pipe3d-maps --in-place --data-root manga_sdss_fits --footprint-only --workers 8
```

Full re-export (new galaxies or changed scaling):

```bash
python -m manga_prep export-pipe3d-maps --in-place --data-root manga_sdss_fits --workers 8
```

## Target scaling (Amara defaults)

Documented in `manga_prep/targets/README.md`. Summary:

| Quantity | Transform | Clip range | Physical meaning |
|----------|-----------|------------|------------------|
| line fluxes | log10(flux) | [-5, 1] | 10⁻⁵ – 10 (Pipe3D flux units) |
| Hα EW | log10(-EW) | [0, 3] | 1 – 1000 Å emission EW |
| stellar Av | linear | [0, 3] | 0 – 3 mag |

Inverse transform:

```text
transformed = scaled * (clip_max - clip_min) + clip_min
physical = inverse_transform(transformed)
```

Raw values are always in `{feature}_raw` inside `amara_maps.npz`.

## UNet notes

- Fixed **76×76** output grid — no variable-size batches for maps.
- No explicit IFU size conditioning needed; `footprint_mask` and `target_loss_masks` encode geometry.
- Imaging normalization: raw cutouts are fine for a first pass; optional arcsinh per band can be added later.
- Preview samples: see `manga_dataloader_preview.ipynb`.

## Tests

```bash
python -m unittest tests.test_manga_dataset -v
```
