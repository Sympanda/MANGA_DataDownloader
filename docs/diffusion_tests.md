# Diffusion / score-model tests

Record of conditional diffusion experiments for MaNGA Hα maps (photometry → physical feature maps). Goal: recover fine structure / probabilistic maps beyond a frozen UNet mean. **Neither the score corrector nor the direct score generator beat the UNet as a point estimator on fair (`t=1`) sampling.**

Training used the Ha ≥99% coverage subset (train ∩ coverage only; val/test held out with zero plateifu overlap). Frozen base for fill/comparison: `runs/arch_ablation/arch_v1/runs/A_unet`.

---

## What we tried

### 1. Score corrector (SDEdit residual correction)

Train a conditional ε-prediction score model that starts from the **frozen UNet Ha** (plus noise at fraction `t_start`) and denoises toward the true map. Intended as a residual / detail corrector on top of the UNet.

**Outcome: did not work.** Sample means stayed near the base; true residuals (target − UNet) are structured, but predicted corrections were tiny / noisy. Residual Pearson on val (`t=0.10`) ≈ **0** (mean ≈ −0.008 over the plotted set). Varying `t_start` (0.1–0.5) did not recover structured residuals.

**Example run:** `runs/score_corrector/score_corr_ha99/`  
**Plots:** `.../plots/` (and `plots/t0p10/`, `t0p20/`, `t0p50/`) — Target | Base | Final | True residual | Pred residual + residual scatter.

### 2. Direct score generator (full-map from noise)

Same score UNet, but **no base as conditioning**. Inference at `t=1` starts from pure noise and generates Ha from ugriz (+ footprint / label mask / spectrum). Intermediate `t<1` rows in multi-t panels are **diagnostics only**: they noise the ground-truth map then denoise (near-identity at low `t`) — **not** a fair photometry→Ha test.

**Outcome: did not work as a point estimator.** On dense (≥99%) test galaxies, mean MSE of the sample mean at **`t=1` ≈ 0.076** (`test_metrics_dense99.csv`, n=48). Low-`t` MSE collapses (e.g. `t=0.10` ≈ 6×10⁻⁵) because of GT+noise reconstruction, not because the generator beats the UNet from photometry. Qualitatively, `t=1` samples are blotchy / over-bright vs the UNet; residual RMSE at `t=1` is typically ~**10× worse** than Target−UNet on the same panels.

Sparse (50–80% coverage) eval also run for completion visuals; metrics still only on observed labels — generator still not competitive with the UNet mean.

**Example run:** `runs/score_generator/score_gen_ha99_2/`  
**Plots:** `.../plots/dense99/`, `.../plots/cov_50_80/`

### 3. Related (earlier): residual-map DDPM

Separate experiment: diffuse the residual `R = Y − Ŷ_UNet` with a small conditional UNet (`runner_residual_diffusion.py`). Same broad conclusion as the score corrector — residual structure after a decent UNet is hard to recover from imaging alone. Not the main focus of the score runs above.

---

## Headline numbers (illustrative)

| Experiment | Metric | Value | Notes |
|------------|--------|------:|-------|
| Score corrector `score_corr_ha99` | val mean `resid_pearson` @ t=0.10 | ≈ −0.008 | No correlation with true residual |
| Score generator `score_gen_ha99_2` | test mean MSE @ **t=1** | ≈ 0.076 | Fair generation from noise |
| Score generator (same CSV) | test mean MSE @ t=0.10 | ≈ 6×10⁻⁵ | **Not fair** — GT+noise reconstr. |

CSVs:  
`runs/score_corrector/score_corr_ha99/csv/`  
`runs/score_generator/score_gen_ha99_2/csv/`

---

## Code / config map

| Role | Path |
|------|------|
| Score UNet + diffusion schedule / norm | `src/models/map_score.py` |
| Generator + corrector wrappers (`sample`, EMA) | `src/models/map_score_wrapper.py` |
| Ha≥99% subset + stratified weights | `src/data/score_subset.py` |
| Score dataloaders + score-space norm stats | `src/data/score_dataloaders.py` |
| Eval / multi-t panels / residual scatters | `src/metrics/score_plots.py` |
| Train/eval generator | `runner_score_generator.py` |
| Train/eval corrector | `runner_score_corrector.py` |
| Configs | `config_score_generator.jsonc`, `config_score_corrector.jsonc` |
| Acceptance tests | `tests/test_map_score.py` |
| Residual DDPM (related) | `src/models/residual_diffusion.py`, `src/models/residual_diffusion_wrapper.py`, `runner_residual_diffusion.py` |

**Commands (reference):**

```powershell
python runner_score_generator.py --config config_score_generator.jsonc --run-name score_gen_ha99 --autoinc
python runner_score_generator.py --config config_score_generator.jsonc --run-name score_gen_ha99_2 --eval-only --t-start-fracs 1.0

python runner_score_corrector.py --config config_score_corrector.jsonc --run-name score_corr_ha99 --autoinc
python runner_score_corrector.py --config config_score_corrector.jsonc --run-name score_corr_ha99 --eval-only --t-start-frac 0.25
```

---

## Takeaways (initial runs — provisional)

1. **Score corrector:** frozen-UNet SDEdit did not learn useful Ha residual structure on the first setup.
2. **Score generator:** fair (`t=1`) sample-mean MSE did not beat the UNet; low-`t` panels must not be read as photometry→map wins.
3. **Do not conclude diffusion cannot help** until the pipeline audit below is closed (EMA re-check, label-mask cond removed, tiny-set overfit tests).

---

## Pipeline audit (Aug 2026) — fixes landed

Following an audit of the score implementation, these corrections / diagnostics were added. **Prior negative results remain provisional until overfit tests pass.**

| Item | Status |
|------|--------|
| EMA `update_ema()` after optimizer step | Already present in `Trainer`; confirmed AMP + non-AMP. Checkpoint `best.pt` stores `_ema.*` keys. |
| Re-eval existing ckpt with `use_ema=False` | **Confirmed major issue.** `score_gen_ha99_2` test (n=8, t=1): `use_ema=False` mean MSE ≈ **0.005** vs `use_ema=True` ≈ **0.063** (~12×). Plots: `plots/dense99/noema/` vs `.../ema/`. EMA update hook exists, but the stored/applied EMA weights are much worse than the live denoiser — treat prior EMA-on results as invalid. |
| EMA integration test | `tests/test_map_score.py::EMAIntegrationTests` |
| Score-norm on **label** pixels only | `compute_score_norm_stats` fixed (was footprint) |
| Drop `label_mask` from generator/corrector cond | Default `condition_on_label_mask: false` (legacy loads omit key → keep old 7-ch cond) |
| Cosine schedule | Already supported (`"schedule": "cosine"`) |
| Min-SNR weighting | `use_min_snr` / `min_snr_gamma` on `MapScoreModel` |
| Bottleneck self-attention | `use_bottleneck_attn` / `attn_heads` on `CondMapScoreUNet` |
| Unconditional overfit | `config_score_overfit_uncond.jsonc` + `runner_score_overfit.py` |
| Conditional 16-gal overfit | `config_score_overfit_cond.jsonc` + same runner |

**Important evaluation note:** diffusion should not be judged only by sample-mean vs UNet MSE. The UNet approximates \(E[Y\mid X]\); diffusion models \(p(Y\mid X)\). Prefer coverage, power spectra, diversity, and individual samples.

### Diagnostic commands

```powershell
# Existing generator ckpt: live weights vs EMA
python runner_score_generator.py --config config_score_generator.jsonc --run-name score_gen_ha99_2 --eval-only --t-start-fracs 1.0 --max-plot 8 --no-ema
python runner_score_generator.py --config config_score_generator.jsonc --run-name score_gen_ha99_2 --eval-only --t-start-fracs 1.0 --max-plot 8

# Tiny-set overfits (must produce recognizable Ha before blaming data)
python runner_score_overfit.py --config config_score_overfit_uncond.jsonc --run-name score_overfit_uncond --autoinc
python runner_score_overfit.py --config config_score_overfit_cond.jsonc --run-name score_overfit_cond --autoinc
```

Extra code paths: `runner_score_overfit.py`, overfit configs above; model knobs in `src/models/map_score.py` / `map_score_wrapper.py`.
