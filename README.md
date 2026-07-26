# MaNGA Map Prediction

Predict **6 Amara Pipe3D map channels** (Hα flux, Hβ flux, [OIII], [NII], Hα EW, stellar Av) on a fixed **76×76** MaNGA grid from SDSS imaging (+ optional 1D spectrum via FiLM).

**Inputs:** 5-band SDSS `ugriz` cutouts (always WCS-aligned in the dataloader: 76×76 Amara grid or ~196 SDSS-native), optional IFU footprint, optional 1D spectrum.  
**Outputs:** 6 scaled map channels at **76×76** (supervision always on the Pipe3D / Amara grid).

## Quick start

```bash
conda activate manga
cd A:\MANGA

# 1. Data prep (see manga_prep/download/README.md for full pipeline)
python -m manga_prep download-manga-sdss 8485-1901
python -m manga_prep download-sdss-cutouts
python -m manga_prep export-pipe3d-maps --in-place --workers 8
python -m manga_prep export-aperture-spectra --workers 8
python -m manga_prep build-index
python -m manga_prep export-aligned-imaging --survey sdss --use-index --skip-existing --workers 8

# 2. Train/val/test split + training
python -m src.data.make_splits --config config.jsonc
python runner.py --config config.jsonc --run-name exp_001 --autoinc
```

All options are documented in **`config.jsonc`** (JSON with `//` comments).

## Spatial pipeline (config-swappable)

The model and dataloader support three spatial pipelines. Change only `config.jsonc` to swap — no code edits required. Checkpoints are **not** interchangeable across pipelines.

| `spatial_pipeline` | `imaging_resolution` | Description |
|--------------------|----------------------|-------------|
| `symmetric` | `aligned` | **Default.** 76×76 WCS-aligned SDSS + footprint channel → UNet/UNet++ → 76×76 maps |
| `symmetric` + `use_hr_cross_attn` | `aligned` | Same backbone; HR (SDSS-native / Legacy) conditions shallow levels via cross-attention |
| `hr_encoder` | `native` | SDSS-native ~196×196 (Amara-oriented) → HR encoder → project to 76×76 → footprint fusion → decoder |
| `hr_full` | `native` | Full UNet/UNet++ on SDSS-native imaging → resize predictions to 76×76 |
| `hr_multiscale` | `native` | HR pyramid fused into every UNet encoder level (legacy concat path) |

| `footprint_mode` | When to use |
|------------------|-------------|
| `spatial_channel` | With `symmetric` — footprint as a 6th input channel (current default) |
| `fusion_concat` | With `hr_encoder` / `hr_full` / `hr_multiscale` — footprint fused on the 76×76 grid |
| `loss_only` | Footprint not fed to the model; still used in masked losses |

**Config presets** (set in both `data` and `model`, or just `model` — `runner.py` merges them):

```jsonc
// Current default: 76×76 backbone + HR cross-attention
"imaging_resolution": "aligned",
"spatial_pipeline": "symmetric",
"footprint_mode": "spatial_channel",
"use_hr_cross_attn": true,
"hr_survey": "sdss",
"hr_cross_attn_levels": [0, 1]

// Ablation without HR
"use_hr_cross_attn": false

// Legacy HR multi-scale concat (SDSS-native as backbone)
"imaging_resolution": "native",
"spatial_pipeline": "hr_multiscale",
"footprint_mode": "fusion_concat",
"use_hr_cross_attn": false
```

**Notes:**
- Ground-truth maps are only defined at **76×76** (~0.5″/pix on the Pipe3D grid).
- With HR cross-attention, the UNet never sees resized HR pixels: HR is encoded to spatial tokens (K/V) and queried by shallow UNet features.
- **WCS alignment always happens in the dataloader** before the model. Pre-export both `sdss_aligned.npz` and `sdss_aligned_native.npz` when using HR cross-attn.
- Imaging/spectrum soft-norm: `asinh(f/s_b)` from train-split percentiles (auto-computed on first train if missing).
- Eval plots show the 76×76 backbone imaging; HR is an auxiliary stream.

## Project layout

```
MANGA/
├── config.jsonc              # Master training config (documented)
├── runner.py                 # Train + eval entry point
├── smoke_test.py             # Model forward-pass sanity check (all pipeline modes)
├── manga_prep/               # Data downloads, exports, dataset
│   ├── download/             # Download scripts + README
│   ├── export/               # Map/spectrum export, index, inventory
│   ├── targets/              # Pipe3D map specs & loaders (was amara_code)
│   ├── dataset/              # PyTorch dataset + index builder
│   └── io/                   # FITS I/O, WCS alignment, caches
├── src/                      # Config-driven training pipeline
│   ├── data/                 # Splits, augmentations, dataloaders
│   ├── models/               # UNet/UNet++, HR front-end, FiLM, losses
│   ├── training/             # Trainer, checkpoints, logging
│   └── metrics/              # Plots
├── scripts/                  # Utilities + legacy training scripts
│   └── legacy/               # train_conditional_unet.py, eval_conditional_unet.py
├── notebooks/                # Exploration notebooks
├── manga_models/             # Shared model definitions (used by runner + legacy scripts)
├── docs/                     # Dataloader & task documentation
├── tests/
└── manga_sdss_fits/          # Per-galaxy data (gitignored)
```

## Data locations

| Path | Purpose |
|------|---------|
| `manga_sdss_fits/<plate>_<ifu>/` | Per-galaxy FITS, cutouts, maps, spectra |
| `manga_sdss_fits/<plate>_<ifu>/sdss_cutouts/` | Native SDSS `ugriz` FITS (~196×196) |
| `manga_sdss_fits/<plate>_<ifu>/aligned_imaging/sdss_aligned.npz` | Cached 76×76 WCS-aligned SDSS stack |
| `manga_sdss_fits/<plate>_<ifu>/amara_maps.npz` | 76×76 targets, valid/loss masks, footprint |
| `manga_sdss_fits/manga_dataset_index.csv` | Modality flags for all galaxies |
| `manga_sdss_fits/splits/*.csv` | Train/val/test assignments (swap for ensembles) |
| `sdss_spplate_cache/` | Shared legacy SDSS spPlate FITS |
| `legacy_coadd_brick_cache/` | Shared Legacy Survey brick cache |
| `runs/manga_maps/<run_name>/` | Training checkpoints, plots, CSVs |

## Training pipeline

The pipeline (`runner.py`) mirrors [Galaxy_ILI](https://github.com/Sympanda/Galaxy_ILI):

- Single **JSONC config** drives everything; snapshot saved per run
- **CSV splits** for reproducible train/val/test (and ensemble variants)
- **Composable losses** with per-term weights (0 = off); mask-topology-aware — see [`docs/masked_losses.md`](docs/masked_losses.md)
- **UNet / UNet++**, optional **coarse/fine head**, **FiLM** spectrum conditioning (`bottleneck` or multi-level `encoder`), **UNet++ deep supervision**
- **Upsample modes:** `bilinear`, `transpose`, `pixel_shuffle`
- **Spatial pipelines:** symmetric (aligned), `hr_encoder`, `hr_full` — see table above
- Post-train **eval** on val/test with metrics CSV + prediction plots

Pre-v2 training scripts live in `scripts/legacy/` if you still need them.

## Documentation

- **Data downloads & exports:** [`manga_prep/download/README.md`](manga_prep/download/README.md)
- **Dataloader & ML task:** [`docs/manga_dataloader.md`](docs/manga_dataloader.md)
- **Map target scaling:** [`manga_prep/targets/README.md`](manga_prep/targets/README.md)
- **Config reference:** [`config.jsonc`](config.jsonc)
- **Mask-safe losses:** [`docs/masked_losses.md`](docs/masked_losses.md)
- **Ablations / config toggles (B–E):** [`docs/model_ablations.md`](docs/model_ablations.md)

## Tests

```bash
python smoke_test.py          # all arch × head × FiLM × spatial-pipeline combos
python -m unittest tests.test_losses tests.test_manga_dataset tests.test_conditional_unet -v
```

## Environment

```bash
conda env create -f environment.yml
conda activate manga
```
