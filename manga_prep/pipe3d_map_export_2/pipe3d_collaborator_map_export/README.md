# Pipe3D Map Export

Small standalone exporter for making padded, clipped, 0-1 scaled maps directly
from MaNGA Pipe3D VAC cubes.

It extracts these direct Pipe3D quantities:

- luminosity-weighted stellar age
- mass-weighted stellar age
- luminosity-weighted stellar metallicity
- mass-weighted stellar metallicity
- Halpha flux
- Hbeta flux
- [OIII]5007 flux
- [NII]6584 flux
- Halpha EW
- stellar Av
- stellar velocity
- stellar velocity dispersion
- stellar mass surface density
- Hbeta stellar absorption index
- D4000

No DRPall file is required for the direct Pipe3D maps. DRPall is required only
if you also ask for the derived Halpha SFR products, because SFR needs the
redshift-based luminosity distance and spaxel physical area.

## Install

```bash
pip install -r requirements.txt
```

## Expected Input

By default, the script expects Pipe3D cubes in folders like:

```text
manga_sdss_fits/
  7495_3702/
    manga-7495-3702.Pipe3D.cube.fits.gz
  8612_6103/
    manga-8612-6103.Pipe3D.cube.fits.gz
```

If the cubes are somewhere else, pass that folder with `--data-root`.

## Run

Export one galaxy:

```bash
python export_pipe3d_maps.py 7495-3702
```

Export selected galaxies:

```bash
python export_pipe3d_maps.py 7495-3702 8612-6103 --out collaborator_pipe3d_maps
```

Export every local Pipe3D cube:

```bash
python export_pipe3d_maps.py --data-root /path/to/pipe3d/cubes --out collaborator_pipe3d_maps
```

Run in parallel across galaxies:

```bash
python export_pipe3d_maps.py --data-root /path/to/pipe3d/cubes --out collaborator_pipe3d_maps --workers 8
```

Export direct maps plus derived star-forming Halpha SFR and gas metallicity maps:

```bash
python export_pipe3d_maps.py \
  --data-root /path/to/pipe3d/cubes \
  --drpall /path/to/drpall-v3_1_1.fits \
  --include-derived \
  --out collaborator_pipe3d_maps \
  --workers 8
```

Export a galaxy-level global BPT/star-forming flag table from the Pipe3D catalog:

```bash
python export_global_flags.py \
  --pipe3d-catalog /path/to/SDSS17Pipe3D_v3_1_1.fits \
  --out pipe3d_global_flags.csv
```

Export flags only for galaxies with local Pipe3D cubes:

```bash
python export_global_flags.py \
  --pipe3d-catalog /path/to/SDSS17Pipe3D_v3_1_1.fits \
  --data-root /path/to/pipe3d/cubes \
  --local-only \
  --out pipe3d_global_flags_local.csv
```

## Output

Each galaxy gets:

```text
collaborator_pipe3d_maps/
  PLATE-IFU/
    PLATE-IFU_pipe3d_direct_maps_76x76.npz
    PLATE-IFU_pipe3d_direct_maps_76x76_metadata.json
  manifest.csv
```

The `.npz` contains:

- `<quantity>_raw`: raw Pipe3D map, center-padded with `NaN`
- `<quantity>_err_raw`: raw error map, where available
- `<quantity>_scaled`: clipped 0-1 scaled map, center-padded with `NaN`
- `<quantity>_scaled_err`: scaled error map, for the derived scaled products
- `<quantity>_valid_mask`: `1` where the scaled map is finite, else `0`
- `native_footprint_mask`: `1` inside the original Pipe3D rectangular map
- `native_shape`, `native_ny`, `native_nx`, `native_spaxel_count`
- `target_shape`

Example loading:

```python
import numpy as np

maps = np.load("collaborator_pipe3d_maps/7495-3702/7495-3702_pipe3d_direct_maps_76x76.npz")

ha = maps["ha_flux_scaled"]
ha_mask = maps["ha_flux_valid_mask"]
native_shape = tuple(maps["native_shape"])
native_spaxel_count = int(maps["native_spaxel_count"])
```

Other direct Pipe3D arrays include:

```python
lw_age = maps["lw_age_scaled"]
mw_age = maps["mw_age_scaled"]
lw_metallicity = maps["lw_metallicity_scaled"]
mw_metallicity = maps["mw_metallicity_scaled"]

stellar_velocity = maps["stellar_velocity_scaled"]
stellar_sigma = maps["stellar_sigma_scaled"]
stellar_mass_density = maps["stellar_mass_density_raw"]
stellar_mass_density_err = maps["stellar_mass_density_err_raw"]
stellar_mass_density_scaled = maps["stellar_mass_density_scaled"]

hb_abs = maps["hb_abs_index_raw"]
hb_abs_err = maps["hb_abs_index_err_raw"]
hb_abs_scaled = maps["hb_abs_index_scaled"]

d4000 = maps["d4000_raw"]
d4000_err = maps["d4000_err_raw"]
d4000_scaled = maps["d4000_scaled"]
```

With `--include-derived`, useful additional arrays include:

```python
sfr = maps["sfr_halpha_raw"]
sfr_err = maps["sfr_halpha_err_raw"]
log_sfr_scaled = maps["log_sfr_halpha_scaled"]
log_sfr_scaled_err = maps["log_sfr_halpha_scaled_err"]
sfr_mask = maps["sfr_valid_mask"]

balmer = maps["balmer_decrement_ha_hb_scaled"]
a_halpha = maps["a_halpha_balmer_scaled"]
log_sigma_sfr = maps["log_sigma_sfr_halpha_scaled"]

metallicity = maps["gas_metallicity_o3n2_pp04_raw"]
metallicity_err = maps["gas_metallicity_o3n2_pp04_err_raw"]
metallicity_scaled = maps["gas_metallicity_o3n2_pp04_scaled"]
metallicity_scaled_err = maps["gas_metallicity_o3n2_pp04_scaled_err"]
metallicity_mask = maps["gas_metallicity_o3n2_valid_mask"]

sf_mask = maps["is_sf_bpt_mask"]
composite_mask = maps["is_comp_bpt_mask"]
agn_mask = maps["is_agn_bpt_mask"]
bpt_code = maps["bpt_class_code_mask"]
```

Choosing model inputs is just choosing array names. For example:

```python
feature_names = [
    "ha_flux_scaled",
    "hbeta_flux_scaled",
    "oiii_5007_flux_scaled",
    "nii_6584_flux_scaled",
    "stellar_av_scaled",
    "stellar_mass_density_scaled",
    "d4000_scaled",
    "mw_age_scaled",
    "mw_metallicity_scaled",
]

cube = np.stack([maps[name] for name in feature_names], axis=0)
```

## Scaling

The scaling is fixed so it can be applied without first downloading the full
dataset to estimate global ranges.

| Quantity | Transform before scaling | Clip range | Scaled meaning |
| --- | --- | --- | --- |
| LW stellar age | linear | `[7, 10.2]` | log stellar age in years |
| MW stellar age | linear | `[7, 10.2]` | log stellar age in years |
| LW stellar metallicity | linear | `[-2.5, 0.5]` | log stellar metallicity |
| MW stellar metallicity | linear | `[-2.5, 0.5]` | log stellar metallicity |
| Halpha flux | `log10(flux)` for positive flux | `[-5, 1]` | `1e-5` to `10` in Pipe3D flux units |
| Hbeta flux | `log10(flux)` for positive flux | `[-5, 1]` | `1e-5` to `10` in Pipe3D flux units |
| [OIII]5007 flux | `log10(flux)` for positive flux | `[-5, 1]` | `1e-5` to `10` in Pipe3D flux units |
| [NII]6584 flux | `log10(flux)` for positive flux | `[-5, 1]` | `1e-5` to `10` in Pipe3D flux units |
| Halpha EW | `log10(-EW)` for negative-emission Pipe3D EW | `[0, 3]` | `1` to `1000` Angstrom emission EW |
| stellar Av | linear | `[0, 3]` | `0` to `3` mag |
| stellar velocity | linear | `[-300, 300]` | km/s |
| stellar velocity dispersion | linear | `[0, 300]` | km/s |
| stellar mass surface density | linear | `[0, 10]` | log stellar mass surface density |
| Hbeta absorption index | linear | `[0, 10]` | Angstrom |
| D4000 | linear | `[1, 2.5]` | D4000 spectral break index |
| Balmer decrement | linear | `[2.86, 8]` | Halpha/Hbeta |
| Halpha attenuation | linear | `[0, 5]` | mag |
| log Halpha SFR | already `log10(SFR)` | `[-6, 0]` | `1e-6` to `1 Msun/yr` |
| log Halpha Sigma SFR | already `log10(Sigma SFR)` | `[-4, 0]` | `1e-4` to `1 Msun/yr/kpc^2` |
| gas metallicity | linear | `[8, 9]` | `12 + log(O/H)` from 8 to 9 |

Non-positive fluxes are invalid for log scaling. For Halpha EW, the current
Pipe3D files use negative values for emission EW, so non-negative EW values are
invalid for the scaled emission-EW map. Raw arrays are still included unchanged.

## Derived Products And Errors

The derived products follow the same workflow used in the notebooks:

- BPT classification uses Halpha, Hbeta, [OIII]5007, and [NII]6584 with
  `S/N >= 3` by default.
- Halpha SFR is calculated only for BPT star-forming spaxels with valid Balmer
  attenuation.
- Gas metallicity is the PP04 O3N2 calibration, also restricted to BPT
  star-forming spaxels and the PP04 O3N2 calibration range `[-1, 1.9]`.

Error propagation included:

- line-ratio log errors from the four Pipe3D line-flux errors
- Balmer decrement and Halpha attenuation errors
- observed and attenuation-corrected Halpha luminosity errors
- Halpha SFR and log Halpha SFR errors
- Sigma SFR and log Sigma SFR errors
- O3N2 and PP04 gas-metallicity errors

The calculation does not include uncertainty in DRPall redshift, luminosity
distance, cosmology, calibration constants, or the PP04 calibration scatter.

The BPT masks are the practical separation between Halpha dominated by star
formation and Halpha in non-star-forming regions:

```text
bpt_class_code_mask = 0  unclassified or low-S/N
bpt_class_code_mask = 1  star-forming
bpt_class_code_mask = 2  composite
bpt_class_code_mask = 3  AGN-like
```

This is a classification, not a fractional decomposition. A composite spaxel is
flagged as mixed/ambiguous; the script does not try to assign, for example, 60%
of its Halpha to star formation and 40% to AGN.

## Global Galaxy Flags

`export_global_flags.py` uses the integrated Pipe3D catalog columns:

- `log_NII_Ha_ALL`
- `log_OIII_Hb_ALL`
- `e_log_NII_Ha_ALL`
- `e_log_OIII_Hb_ALL`
- `EW_Ha_ALL`
- `e_EW_Ha_ALL`
- `Ha_Hb_ALL`
- `log_SFR_Ha`

It writes one row per galaxy with:

```text
global_bpt_class_code = 0  unclassified or missing global ratios
global_bpt_class_code = 1  globally star-forming
global_bpt_class_code = 2  globally composite
global_bpt_class_code = 3  globally AGN-like
```

It also writes `global_bpt_sf_strict`, which requires:

- global BPT class is star-forming
- both global log-ratio errors are below `--max-ratio-err` (`0.3` dex by default)

For a more conservative emission-strength flag, it writes `global_sf_ew_strict`,
which additionally requires:

- positive-emission Halpha EW is at least `--min-ha-ew-emission` (`3 Angstrom` by default)
- Halpha EW S/N is at least `--min-ha-ew-snr` (`3` by default)

This table is useful for testing whether a model performs better on galaxies
whose integrated spectrum is globally star-forming.

## Stellar Mass Density Versus Stellar Mass

Pipe3D provides stellar mass surface density, not a direct stellar mass per
spaxel map in this exporter. It is close in spirit to a stellar mass map because
each spaxel covers a small area, but it is not identical to stellar mass:

```text
stellar mass in a spaxel = stellar mass surface density x physical spaxel area
```

The physical spaxel area changes with galaxy distance. For ML map work, stellar
mass surface density is often the cleaner quantity because it describes the
local stellar population without multiplying by a distance-dependent area.

## Padding

The default target shape is `76 x 76`, which matches the largest native Pipe3D
map size found in the local sample used to build this exporter. If a future cube
is larger, the script raises an error rather than cropping.

To pad to the largest native shape in the selected local files:

```bash
python export_pipe3d_maps.py --auto-target-size --data-root /path/to/pipe3d/cubes
```
