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
"hr_encoder_n_down": 3,
"spectrum_pooling": "avg",
"spectrum_use_wavelength": false,
"spectrum_use_ivar": false
```

Pre-export both Amara and SDSS-native caches. Ablate with `"use_hr_cross_attn": false`.

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
"spectrum_pooling": "attention",
"spectrum_use_wavelength": true,
"spectrum_use_ivar": true,
"deep_supervision": true,
"film_injection": "encoder"
```

Current `config.jsonc` default.

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

## Loss balancing (always on)

Pixel / grad / Laplacian losses average **per galaxy × map**, then over active maps.
No config flag — this is the corrected default.
