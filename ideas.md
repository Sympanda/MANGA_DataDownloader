# Phys-map modelling ideas

Working notes for improving physical-property map prediction. Items marked
**done / in progress** are wired into `config_phys.jsonc` (or supporting code).
The rest are deferred.

## Done / in progress (phys config)

- **Keep SF-only filtering** (`galaxy_sf_flag: global_bpt_sf`) to make the first
  real runs easier; later prefer *conditioning* on SF class instead of dropping
  quiescent galaxies. *(wired in `config_phys.jsonc`)*
- **Coverage-aware galaxy selection** — `min_footprint_fill` + `min_valid_pixels`;
  overfit ranks by fill then mean valid pixels. *(code + phys configs)*
- **Redshift FiLM conditioning** — `use_redshift_cond` adds `log1p(z)` MLP into
  the FiLM vector (same path as spectrum). *(code + phys configs)*
- **2D FFT power-spectrum loss** — `fft_power` in `src/models/losses.py`.
  *(phys configs only)*
- **Loss retune for phys** — L1=1.0, grad=0.25, laplacian=0.08, fft_power=0.15.

## Deferred (come back later)

### Probabilistic / generative
- **cVAE** (or reuse residual/score diffusion) for multi-modal maps and sharper
  samples; addresses regression-to-the-mean blur after a solid deterministic
  baseline.
- Uncertainty head / ensemble calibration on phys channels.

### Conditioning
- FiLM on **galaxy SF class** (SF / composite / AGN), not just sample filtering.
- Spaxel-level `is_sf_bpt_mask` as a spatial channel or loss gate.
- IFU size / footprint area as a cheap scale cue alongside redshift.

### Losses & optimisation
- **Dynamic / uncertainty loss weighting** across channels (Kendall or GradNorm).
- **Log-Cosh** (likely redundant with Charbonnier).
- **SSIM** (low weight) for structural coherence.
- Per-channel loss weights or single-channel specialists (`ha_ew` vs mass density).
- Mild per-key SNR floors (line-like vs SSP products differ).

### Architecture / norms
- Already on **GroupNorm** (`norm: gn`) — good default; InstanceNorm / LayerNorm
  swaps are low priority versus data selection + conditioning.
- HR cross-attn ablations specifically for phys detail.

### Data curriculum
- Train on “richest coverage” subset as an ablation, then compare to full SF / all
  galaxies for selection-bias vs task-hardness.
- Judge success on coherent structure, not Pipe3D measurement pepper (overfit can
  memorise noise that should not generalise).

## Selection / conditioning rationale (short)

| Idea | Why |
|------|-----|
| Overfit works | Task is learnable; capacity OK; full-data blur ≠ impossible labels |
| SF filter | Homogenises emission-sensitive channels for early runs |
| Fill-fraction ranking | Fair to 19″ vs 127″ IFUs |
| Redshift cond | Distance / physical scale for surface-density-like maps |
| FFT power loss | Explicitly fights over-smooth predictions |
