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

Survey cutout FITS files are **always WCS-reprojected onto the Pipe3D / Amara
spaxel grid in the dataloader** before the model sees them. This matches the
workflow in `sdss_legacy_fits_jpeg_comparison.ipynb` — SDSS frame cutouts in
particular are not north-up in native orientation and must not be used raw.

| Mode | Config | Imaging tensor |
|------|--------|----------------|
| `aligned` | `imaging_resolution: "aligned"` | Amara FoV / orientation, **76×76** |
| `native` | `imaging_resolution: "native"` | **SDSS plate scale**, fixed **196×196**, Amara-oriented (larger FoV than maps) |

Optional Amara-grid override: `data.aligned_oversample` (integer ≥ 1) only applies when
using the Amara grid (`aligned`). Pre-exported caches:
`aligned_imaging/sdss_aligned.npz` (76) or `sdss_aligned_native.npz` (196).

`align_imaging_to_amara_grid=False` raises in `MangaGalaxyDataset` (training path).
Raw cutout stacking (`_stack_native_imaging_bands`) remains only as a debug helper.

Requires `reproject` (`pip install reproject`).

**Training uses asinh-softened flux when `model.input_norm.mode="asinh"`** (see below).
Display notebooks may still use percentile stretch for visualization only.

## Input asinh scales

Per-band / per-spectrum soft scales ``s_b`` are estimated on the **train** split only:

```bash
python -m manga_prep compute-input-scales --config config.jsonc
# → manga_sdss_fits/stats/input_asinh_scales.json
```

The JSON stores percentiles **95 / 99 / 99.5** for:

- SDSS `u,g,r,i,z` (footprint-masked |flux|)
- optional Legacy bands
- fake aperture spectra
- real SDSS fiber spectra

Config picks which percentile to apply at train/eval time:

```jsonc
"input_norm": {
  "mode": "asinh",
  "scales_path": "manga_sdss_fits/stats/input_asinh_scales.json",
  "auto_compute": true,          // if file missing, runner builds it on train split
  "imaging_percentile": 99,      // 95 | 99 | 99.5 (alias 995)
  "spectrum_percentile": 99
}
```

If `scales_path` is absent when training starts and `auto_compute` is true (default),
`runner.py` / `build_model_config` calls the same computation as
`python -m manga_prep compute-input-scales`. Set `auto_compute: false` to fail fast
instead.

Runtime applies ``asinh(f / s_b)`` in `prepare_imaging_input` / `prepare_spectrum_input`
(before the model). Spectrum λ_norm and log1p(ivar) are unchanged.

## Conditional UNet

See `manga_models/` and `runner.py` (or `scripts/legacy/train_conditional_unet.py`).

v1 conditioning:
- **SDSS / Legacy**: channel-concat at UNet input (always Amara-WCS-aligned in the loader)
- **Footprint mask**: optional extra input channel
- **Spectrum**: 1D CNN → **FiLM** (`bottleneck` on deepest encoder, or multi-level `encoder`)
- **UNet++ deep supervision** (optional): 1×1 heads on nested full-res nodes; aux masked L1 + full loss on deepest


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

Physical-property maps (separate NPZ; does not overwrite `amara_maps.npz`):

```bash
python -m manga_prep export-pipe3d-phys-maps \
  --in-place --include-derived --drpall drpall-v3_1_1.fits \
  --data-root manga_sdss_fits --workers 8
```

See `manga_prep/targets/README.md` for keys, S/N masks, and global SF flags.

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
- Imaging / spectrum soft-norm: `asinh(f / s_b)` via `model.input_norm` (train-split
  percentiles in `manga_sdss_fits/stats/input_asinh_scales.json`). Compute with
  `python -m manga_prep compute-input-scales --config config.jsonc`, then pick
  `imaging_percentile` / `spectrum_percentile` ∈ {95, 99, 99.5}.
- Preview samples: see `manga_dataloader_preview.ipynb`.

## Tests

```bash
python -m unittest tests.test_manga_dataset -v
```
