# Pipe3D Map Targets

Two map products live side-by-side under each galaxy folder:

| Product | File | Contents |
|---------|------|----------|
| Legacy emission-line Amara maps | `amara_maps.npz` | 6 line / Av channels |
| Physical-property maps | `amara_phys_maps.npz` | ages, Z, kinematics, mass density, indices, optional SFR / gas Z |

Legacy training still uses `amara_maps.npz`. New physical maps do **not** overwrite it.

## Legacy emission-line maps

**Code:** `manga_prep/targets/pipe3d_maps.py`  
**Export:** `python -m manga_prep export-pipe3d-maps`

| Key | Quantity |
|-----|----------|
| `ha_flux` | Hα flux |
| `hbeta_flux` | Hβ flux |
| `oiii_5007_flux` | [OIII]5007 flux |
| `nii_6584_flux` | [NII]6584 flux |
| `ha_ew` | Hα equivalent width |
| `stellar_av` | Stellar Av |

```bash
python -m manga_prep export-pipe3d-maps --in-place --data-root manga_sdss_fits --workers 8
```

```python
from manga_prep.targets.pipe3d_maps import load_amara_maps, load_amara_training_targets

maps = load_amara_maps("manga_sdss_fits/7495_3702")
ha = maps["ha_flux_scaled"]
bundle = load_amara_training_targets("manga_sdss_fits/7495_3702", scaled=True)
```

## Physical-property maps

**Code:** `manga_prep/targets/pipe3d_phys_maps.py`  
**Export:** `python -m manga_prep export-pipe3d-phys-maps`

Direct quantities include LW/MW stellar age & metallicity, line fluxes, Hα EW, stellar Av,
velocity / σ, stellar mass surface density, Hβ absorption, D4000.

With `--include-derived` (needs DRPall for redshift / distance): BPT masks, Hα SFR,
ΣSFR, Balmer decrement / A(Hα), PP04 O3N2 gas metallicity.

Each quantity with an error plane also stores:

- `{key}_err_raw` — raw uncertainty
- `{key}_snr` — value / error (spaxel S/N)
- `{key}_snr_mask` — footprint ∩ valid ∩ (S/N ≥ `--snr-min`, default 3)

Re-threshold at train time via `load_amara_phys_training_targets(..., snr_min=5.0)`.

```bash
# Direct physical maps only
python -m manga_prep export-pipe3d-phys-maps --in-place --data-root manga_sdss_fits --workers 8

# + derived SFR / metallicity / spaxel BPT (requires DRPall)
python -m manga_prep export-pipe3d-phys-maps \
  --in-place --include-derived --drpall drpall-v3_1_1.fits \
  --data-root manga_sdss_fits --workers 8
```

```python
from manga_prep.targets.pipe3d_phys_maps import (
    load_amara_phys_maps,
    load_amara_phys_training_targets,
)

maps = load_amara_phys_maps("manga_sdss_fits/7495_3702")
age = maps["mw_age_scaled"]
snr = maps["mw_age_snr"]

bundle = load_amara_phys_training_targets(
    "manga_sdss_fits/7495_3702",
    keys=["mw_age", "mw_metallicity", "stellar_mass_density", "d4000"],
    snr_min=3.0,
)
```

### Galaxy-level star-forming flags

Needs the integrated Pipe3D catalog (`SDSS17Pipe3D_v3_1_1.fits`):

```bash
python -m manga_prep export-pipe3d-global-flags \
  --pipe3d-catalog /path/to/SDSS17Pipe3D_v3_1_1.fits \
  --data-root manga_sdss_fits --local-only \
  --out manga_sdss_fits/pipe3d_global_flags.csv
```

`build-index` merges these into `manga_dataset_index.csv` when the CSV is present
(`global_bpt_sf`, `global_bpt_sf_strict`, `global_sf_ew_strict`, …).

## Scaling (shared conventions)

| Quantity | Transform | Clip range | Scaled meaning |
|----------|-----------|------------|----------------|
| line fluxes | log10(positive) | [-5, 1] | 1e-5 to 10 (Pipe3D flux units) |
| Hα EW | log10(-EW) for emission | [0, 3] | 1–1000 Å emission EW |
| stellar Av | linear | [0, 3] | 0–3 mag |
| LW/MW age | linear | [7, 10.2] | log10(yr) |
| LW/MW metallicity | linear | [-2.5, 0.5] | log10(Z/Z☉) |
| stellar velocity | linear | [-300, 300] | km/s |
| stellar σ | linear | [0, 300] | km/s |
| stellar mass density | linear | [0, 10] | log10(M☉/pc²) |
| Hβ abs / D4000 | linear | [0, 10] / [1, 2.5] | index units |
| log SFR / ΣSFR | linear | [-6, 0] / [-4, 0] | log10(M☉/yr[/kpc²]) |
| gas metallicity | linear | [8, 9] | 12 + log(O/H) |

## Physical-property training

Use ``config_phys.jsonc`` with ``runner.py``. Key ``data`` fields:

| Field | Meaning |
|-------|---------|
| ``target_source`` | ``"phys"`` → ``amara_phys_maps.npz`` |
| ``target_keys`` | Channel subset (must match ``model.target_keys``) |
| ``min_snr`` | Spaxel S/N cut on loss masks (``null`` = none) |
| ``galaxy_sf_flag`` | Keep SF galaxies only (``global_bpt_sf``, …) |
| ``require_sf_spaxel`` | Also mask to BPT-SF spaxels |

```bash
python runner.py --config config_phys.jsonc --run-name phys_v1 --autoinc
```
