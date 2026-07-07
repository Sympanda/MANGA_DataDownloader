# Mask-safe losses and regularisation

MaNGA map targets are only defined on **per-parameter valid masks** (`target_loss_masks` in the dataloader). Pixels outside a channel's mask have no ground truth and must not contribute to supervised losses.

All training losses in `src/models/losses.py` follow these rules.

## Tensor shapes

```text
pred, target, loss_mask : [B, C, H, W]
```

Each output channel `C` can have a **different** valid mask (sparse/low-SNR maps).

## Supervised losses (mask-only)

### Pixel losses (`charbonnier`, `l1`, `mse`)

Computed only where `loss_mask > 0`, normalised by the count of valid pixels:

```python
loss = (raw * mask).sum() / (mask.sum() + eps)
```

No target values are invented outside the mask.

### Gradient loss (`grad`)

Uses **horizontal and vertical finite differences**, not Sobel convolutions across mask holes.

A neighbour pair contributes only if **both** pixels are valid for that channel:

```python
valid_x = mask[:, :, :, 1:] & mask[:, :, :, :-1]
valid_y = mask[:, :, 1:, :] & mask[:, :, :-1, :]
```

This avoids learning artificial edges at valid/invalid boundaries.

### Laplacian loss (`laplacian`)

Optional. A 3×3 Laplacian stencil is evaluated only where **all nine** pixels in the stencil are valid. Use a **low weight** (e.g. `0.02`) or omit if unstable.

### Integration loss (`integration`)

For flux map channels (`ha_flux`, `hbeta_flux`, `oiii_5007_flux`, `nii_6584_flux`):

```python
pred_sum = (pred * mask).sum(dim=(-2, -1))
true_sum = (target * mask).sum(dim=(-2, -1))
```

Per-channel masks are used. **Normalise** the sum mismatch for training stability (default `mean`):

| `integration.normalize` | Formula | Scale |
|-------------------------|---------|-------|
| `mean` (default) | `\|pred_sum - true_sum\| / N_valid` | ~0.01–1, like pixel losses |
| `relative_sum` | `\|pred_sum - true_sum\| / max(\|true_sum\|, N_valid * eps)` | relative flux error |
| `raw` | `\|pred_sum - true_sum\|` | hundreds — can blow up AMP / grads |

```jsonc
"loss_params": { "integration": { "normalize": "mean" } }
```

## Regularisation (no targets)

These terms do **not** supervise pixels outside the mask. They only penalise pathological structure on valid regions.

### Prediction TV (`tv_pred`)

Weak total variation on `pred` across valid neighbour pairs (same `valid_x` / `valid_y` topology as gradient loss). Typical weight: `0.001`–`0.02` (default config: `0.005`).

### Residual penalties (`residual_amp`, `residual_tv`)

Only active when `output_head: "coarse_fine"`:

```text
pred = coarse_up + detail_scale * residual
```

- `residual_amp` — L1 on `|residual|` inside the valid mask
- `residual_tv` — pairwise TV on `residual` across valid neighbours

Typical weights: `0.001`–`0.01` each.

### Detail scale schedule

`detail_scale_init` (default `0.1`) is a learnable scale on the residual branch. Optional `detail_scale_schedule` in `config.jsonc` ramps the multiplier during early epochs so the coarse branch stabilises first:

```jsonc
"detail_scale_schedule": {
  "warmup_epochs": 20,
  "ramp_epochs": 60,
  "start": 0.0,
  "end": 1.0
}
```

Effective scale = `detail_scale_init * schedule_multiplier(epoch)`.

## Recommended configs

**Default (current `config.jsonc`):**

```jsonc
"losses": ["charbonnier", "grad", "integration", "tv_pred", "residual_tv"],
"loss_weights": [1.0, 0.05, 0.1, 0.005, 0.005]
```

**With mask-aware Laplacian:**

```jsonc
"losses": ["charbonnier", "grad", "laplacian", "integration", "tv_pred", "residual_tv"],
"loss_weights": [1.0, 0.05, 0.02, 0.1, 0.005, 0.005]
```

**Minimal:**

```jsonc
"losses": ["charbonnier", "grad", "integration"],
"loss_weights": [1.0, 0.05, 0.1]
```

Set any weight to `0` or remove the loss name to disable a term.

## Plots and metrics

Eval plots and per-galaxy MSE metrics use **masked** predictions only (`NaN` outside the per-channel loss mask). The rightmost map column is labelled `pred (masked)`.

## Implementation reference

| File | Role |
|------|------|
| `src/models/losses.py` | Loss functions and `compose_map_losses()` |
| `src/models/encoders.py` | `CoarseFineHead` with `coarse` / `residual` decomposition |
| `src/models/wrapper.py` | Passes `residual` and `epoch` into loss composition |
| `src/training/train.py` | Supplies `epoch` for detail-scale schedule |
