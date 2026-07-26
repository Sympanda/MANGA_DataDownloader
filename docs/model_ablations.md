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
| `model.imaging_resolution` + `data.imaging_resolution` | both | `aligned` / `native` | Must match |

Keep **`data.imaging_resolution`** and **`model.imaging_resolution`** the same.

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
"spectrum_pooling": "avg",              // optional: leave attention off for pure B
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

## Model D — multi-scale native SDSS (from B)

Encode **native** ugriz, build an HR feature pyramid, project each level onto the
76×76 UNet++ encoder spine (concat + conv fuse). Do **not** resample raw imaging
to 76×76 before the HR encoder.

```jsonc
"imaging_resolution": "native",   // also set data.imaging_resolution
"spatial_pipeline": "hr_multiscale",
"footprint_mode": "fusion_concat",  // or "loss_only"
"spectrum_pooling": "avg",
"spectrum_use_wavelength": false,
"spectrum_use_ivar": false
```

Geometric aug is disabled automatically for native imaging in `runner.py`.

```powershell
python runner.py --config config.jsonc --run-name model_d_hr_ms --autoinc
```

---

## Model E — C + D combined

```jsonc
"imaging_resolution": "native",
"spatial_pipeline": "hr_multiscale",
"footprint_mode": "fusion_concat",
"spectrum_pooling": "attention",
"spectrum_use_wavelength": true,
"spectrum_use_ivar": true,
"deep_supervision": true,
"film_injection": "encoder"
```

---

## Spatial pipeline cheat-sheet

| `spatial_pipeline` | Imaging | Behaviour |
|--------------------|---------|-----------|
| `symmetric` | aligned 76×76 | Imaging (+ footprint channel) → UNet++ |
| `hr_encoder` | native | HR encode → **single** deep map → resize 76×76 → UNet++ |
| `hr_multiscale` | native | HR pyramid → fuse into **every** UNet++ encoder level |
| `hr_full` | native | Full-res UNet++ then resize maps to 76×76 |

Prefer **`hr_multiscale`** over `hr_encoder` when testing Task 4.

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
