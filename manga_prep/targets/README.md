# Pipe3D Map Targets

Library and export tooling for **6 Amara map channels** used as UNet training targets.

**Code:** `manga_prep/targets/pipe3d_maps.py`  
**Export CLI:** `python -m manga_prep export-pipe3d-maps`

## Target keys

| Key | Quantity |
|-----|----------|
| `ha_flux` | Hα flux |
| `hbeta_flux` | Hβ flux |
| `oiii_5007_flux` | [OIII]5007 flux |
| `nii_6584_flux` | [NII]6584 flux |
| `ha_ew` | Hα equivalent width |
| `stellar_av` | Stellar Av |

## Export

```bash
python -m manga_prep export-pipe3d-maps --in-place --data-root manga_sdss_fits --workers 8
```

Output per galaxy: `amara_maps.npz`, `amara_maps_metadata.json`

## Loading

```python
from manga_prep.targets.pipe3d_maps import load_amara_maps, load_amara_training_targets

maps = load_amara_maps("manga_sdss_fits/7495_3702")
ha = maps["ha_flux_scaled"]

bundle = load_amara_training_targets("manga_sdss_fits/7495_3702", scaled=True)
```

## Scaling

| Quantity | Transform | Clip range | Scaled meaning |
|----------|-----------|------------|----------------|
| Hα, Hβ, [OIII], [NII] flux | log10(positive) | [-5, 1] | 1e-5 to 10 (Pipe3D flux units) |
| Hα EW | log10(-EW) for emission | [0, 3] | 1–1000 Å emission EW |
| stellar Av | linear | [0, 3] | 0–3 mag |

Default padded canvas: **76×76** (`DEFAULT_TARGET_SIZE`).

## Note

Pipe3D map targets previously lived in `amara_code/`; they are now in `manga_prep/targets/`.
