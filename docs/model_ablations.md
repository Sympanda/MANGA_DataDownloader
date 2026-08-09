# Model ablations and config toggles

After the UNet++ / residual / balanced-loss fixes (**Model B**), spectral and
HR-imaging upgrades are controlled from `config.jsonc` (and `config_uncertainty.jsonc`).

See also: recommended experiment order in the architecture-fix notes (B → C → D → E).

## Quick reference

| Knob | Where | Values | Effect |
|------|--------|--------|--------|
| `model.architecture` | `config.jsonc` → `model` | `unetpp` / `unet` | Backbone |
| `model.deep_supervision` | same | `true` / `false` | UNet++ nested aux heads |
| `model.film_injection` | same | `encoder` / `bottleneck` / `none` | Spectrum FiLM sites |
| `model.spectrum_pooling` | same | `attention` / `avg` | Task 3 pooling |
| `model.spectrum_use_wavelength` | same | `true` / `false` | λ_norm channel |
| `model.spectrum_use_ivar` | same | `true` / `false` | log1p(ivar) channel |
| `model.spatial_pipeline` | same | see below | Imaging path |
| `model.use_hr_cross_attn` | same | `true` / `false` | HR morphology via cross-attention (side stream) |
| `model.hr_survey` | same | `sdss` / `legacy` | Which survey feeds the HR encoder |
| `model.hr_cross_attn_levels` | same | e.g. `[0,1]` | UNet spine levels receiving HR attention |
| `model.imaging_resolution` + `data.imaging_resolution` | both | `aligned` / `native` | Must match; both are **WCS-aligned in the dataloader** |
| `data.aligned_oversample` | `data` | int / omit | Pixel oversample on Amara FoV |

Keep **`data.imaging_resolution`** and **`model.imaging_resolution`** the same (forced to `aligned` when `use_hr_cross_attn`).

**Alignment is always done in the dataloader.** `aligned` = 1× Amara canvas (76×76).
`native` = SDSS plate scale on a ~196×196 Amara-oriented canvas (legacy HR-as-backbone paths).

---

## Model B — corrected deterministic baseline

Fixed dense UNet++, residual projection, per-(galaxy, map) losses.

```jsonc
"architecture": "unetpp",
"output_head": "single",
"deep_supervision": true,
"film_injection": "encoder",
"imaging_resolution": "aligned",
"spatial_pipeline": "symmetric",
"footprint_mode": "spatial_channel",
"use_hr_cross_attn": false,
"spectrum_pooling": "avg",
"spectrum_use_wavelength": false,
"spectrum_use_ivar": false
```

```powershell
python runner.py --config config.jsonc --run-name model_b --autoinc
```

---

## Model C — better spectrum (from B)

Attention pooling + wavelength (+ ivar if present). **Keep aligned / symmetric.**

```jsonc
"spectrum_pooling": "attention",
"spectrum_use_wavelength": true,
"spectrum_use_ivar": true,
"spectrum_wave_min": 3622.0,
"spectrum_wave_max": 10354.0
```

Defaults in `config.jsonc` already enable C on top of B.

---

## Model D — HR morphology via cross-attention (from B)

Keep the **76×76 aligned** backbone. Load SDSS-native (~196) as a side stream, encode to spatial tokens, and cross-attend into shallow UNet++ levels (queries = UNet features; keys/values = HR).

```jsonc
"imaging_resolution": "aligned",
"spatial_pipeline": "symmetric",
"footprint_mode": "spatial_channel",
"use_hr_cross_attn": true,
"hr_survey": "sdss",
"hr_cross_attn_levels": [0, 1],
"hr_encoder_n_down": 1,
"hr_attention_mode": "local",
"hr_attention_window": 7,
"spectrum_pooling": "avg",
"spectrum_use_wavelength": false,
"spectrum_use_ivar": false
```

Pre-export both Amara and SDSS-native caches. Ablate with `"use_hr_cross_attn": false`.

Sense-checks:
```powershell
# Does HR change predictions? (Δ ~ 0 ⇒ ignored)
python scripts/hr_zero_contribution.py --run-name model_d_hr_xattn --split val

# Can the model overfit ~32 galaxies? (capacity / label check; HR forced off)
python scripts/overfit_tiny.py --config config.jsonc --n-galaxies 32 --run-name overfit_32 --autoinc

# Same check on physical-property maps (detail / capacity)
python scripts/overfit_tiny.py --config config_phys_overfit.jsonc --n-galaxies 16 --run-name phys_overfit_16 --autoinc
```

```powershell
python runner.py --config config.jsonc --run-name model_d_hr_xattn --autoinc
```

---

## Model E — C + D combined

```jsonc
"imaging_resolution": "aligned",
"spatial_pipeline": "symmetric",
"footprint_mode": "spatial_channel",
"use_hr_cross_attn": true,
"hr_survey": "sdss",
"hr_cross_attn_levels": [0, 1],
"hr_encoder_n_down": 1,
"hr_attention_mode": "local",
"hr_attention_window": 7,
"spectrum_pooling": "attention",
"spectrum_use_wavelength": true,
"spectrum_use_ivar": true,
"deep_supervision": true,
"film_injection": "encoder"
```

Current `config.jsonc` default (shallow HR + local attn on levels 0 and 1).

---

## Spatial pipeline cheat-sheet

| `spatial_pipeline` | Imaging | Behaviour |
|--------------------|---------|-----------|
| `symmetric` | aligned 76×76 | Imaging (+ footprint channel) → UNet++ |
| `symmetric` + `use_hr_cross_attn` | aligned + HR side | Shallow levels get HR cross-attention |
| `hr_encoder` | native | HR encode → **single** deep map → resize 76×76 → UNet++ |
| `hr_multiscale` | native | HR pyramid → fuse into **every** UNet++ encoder level (legacy) |
| `hr_full` | native | Full-res UNet++ then resize maps to 76×76 |

Prefer **`use_hr_cross_attn`** over `hr_multiscale` for Task 4 (HR and MaNGA grids are not pixel-equivalent).

---

## Spectrum channel layout

When enabled, `SpectrumEncoder` input is `(B, C, n_wave)`:

1. flux (raw / nan-cleaned)
2. λ_norm ∈ [-1, 1] using `spectrum_wave_min` / `spectrum_wave_max`
3. `log1p(ivar)` (zeros → ones if ivar missing)

FiLM API is unchanged: encoder still emits `(B, cond_dim)`.

---

## Automated architecture grid (no Optuna)

Fixed factorial over the big knobs (UNet vs UNet++, deep supervision, spectrum,
HR cross-attn). Trains each cell, runs `paper_eval` (RMSE / MAE / R² / …), and
writes comparison plots under `runs/arch_ablation/<sweep>/analysis/`.

```powershell
# Preview cells
python runner_arch_ablation.py --dry-run --grid core

# Full core grid (~8 runs)
python runner_arch_ablation.py --sweep-name arch_v1 --grid core --device cuda:1

# Resume / subset
python runner_arch_ablation.py --sweep-name arch_v1 --only C_unetpp_ds,D_unetpp_ds_spec --skip-existing

# Re-plot only
python runner_arch_ablation.py --sweep-name arch_v1 --analyze-only
```

| Cell | Meaning |
|------|---------|
| `A_unet` | Plain UNet, imaging only |
| `B_unetpp` | UNet++ without DS |
| `C_unetpp_ds` | UNet++ + DS (baseline for deltas) |
| `D_unetpp_ds_spec` | + spectrum package (Model C) |
| `E_unetpp_ds_hr` | + HR cross-attn (Model D) |
| `F_unetpp_ds_spec_hr` | spectrum + HR (Model E) |
| `G_unet_spec` | UNet + spectrum |
| `H_unetpp_spec_nods` | UNet++ + spectrum, no DS |

`--grid extended` adds FiLM-off and UNet+spectrum+HR controls.

---

Pixel / grad / Laplacian losses average **per galaxy × map**, then over active maps.
No config flag — this is the corrected default.
