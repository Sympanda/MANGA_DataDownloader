"""Quick diagnostic script for ens_v1 calibration."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

from src.metrics.paper_eval import _forward_member, _load_model, _move_batch, discover_run, _build_dataloader
from src.models.wrapper import prepare_targets_and_masks

ROOT = Path("runs/manga_maps")
RUN = "ens_v1"


def main() -> None:
    summary = pd.read_csv(ROOT / RUN / "paper_eval/test/summary_spaxel_stats.csv")
    summary["sigma_rmse_ratio"] = summary["rmse"] / summary["mean_sigma"]
    summary["sigma_mae_ratio"] = summary["mae"] / summary["mean_sigma"]
    print("=== Spaxel stats (full test) ===")
    print(
        summary[
            [
                "channel",
                "mean_sigma",
                "median_sigma",
                "rmse",
                "mae",
                "coverage_1sigma",
                "coverage_2sigma",
                "sigma_rmse_ratio",
            ]
        ].to_string(index=False)
    )

    cov = pd.read_csv(ROOT / RUN / "paper_eval/test/coverage_summary.csv")
    print("\n=== Coverage summary ===")
    print(cov.to_string(index=False))

    pg = pd.read_csv(ROOT / RUN / "paper_eval/test/per_galaxy_metrics.csv")
    for col in ["coverage_1sigma", "coverage_2sigma", "mse_all"]:
        if col in pg.columns:
            s = pg[col].dropna()
            print(
                f"\nPer-galaxy {col}: mean={s.mean():.3f} median={s.median():.3f} "
                f"p10={s.quantile(0.1):.3f} p90={s.quantile(0.9):.3f}"
            )

    # Sample a few batches to decompose sigma_epi vs sigma_ale
    ctx = discover_run(ROOT, RUN)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from src.metrics.paper_eval import _load_model as load_model

    models = [load_model(ctx, md, device) for md in ctx.member_dirs]
    loader = _build_dataloader(ctx, split="test", batch_size=16)

    epi_parts, ale_parts, tot_parts, err_parts = [], [], [], []
    n_batches = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        targets, masks = prepare_targets_and_masks(batch, ctx.model_cfg)
        member_preds, member_sigmas = [], []
        for model in models:
            p, s = _forward_member(model, batch, device)
            member_preds.append(p)
            member_sigmas.append(s)
        stacked = torch.stack(member_preds, dim=0)
        pred = stacked.mean(dim=0)
        sigma_epi = stacked.std(dim=0, unbiased=False)
        sigma_ale = torch.stack(member_sigmas, dim=0).mean(dim=0)
        sigma_tot = torch.sqrt(sigma_epi**2 + sigma_ale**2)
        err = torch.abs(pred - targets)

        m = masks > 0
        epi_parts.append(sigma_epi[m].cpu().numpy())
        ale_parts.append(sigma_ale[m].cpu().numpy())
        tot_parts.append(sigma_tot[m].cpu().numpy())
        err_parts.append(err[m].cpu().numpy())
        n_batches += 1
        if n_batches >= 20:
            break

    epi = np.concatenate(epi_parts)
    ale = np.concatenate(ale_parts)
    tot = np.concatenate(tot_parts)
    err = np.concatenate(err_parts)
    print("\n=== Sigma decomposition (20 test batches, all channels pooled) ===")
    for name, arr in [("sigma_epi", epi), ("sigma_ale", ale), ("sigma_tot", tot), ("|error|", err)]:
        print(f"  {name:12s} mean={arr.mean():.5f} median={np.median(arr):.5f} p90={np.quantile(arr,0.9):.5f}")
    frac_epi = np.mean(epi**2 / np.maximum(tot**2, 1e-12))
    frac_ale = np.mean(ale**2 / np.maximum(tot**2, 1e-12))
    print(f"  mean(epi²/tot²)={frac_epi:.3f}  mean(ale²/tot²)={frac_ale:.3f}")
    print(f"  coverage @1σ using tot: {np.mean(err <= tot):.3f}")
    print(f"  coverage @1σ using ale only: {np.mean(err <= ale):.3f}")
    print(f"  coverage @1σ using epi only: {np.mean(err <= epi):.3f}")
    # Scale factor needed for 68% coverage (Gaussian)
    scale = np.median(err / np.maximum(tot, 1e-8))
    k68 = stats.norm.ppf(0.84)
    print(f"  median |err|/σ_tot = {scale:.2f}  (need ~{k68:.2f} for 68% if Gaussian)")
    print(f"  implied σ inflation for 68%: {scale/k68:.2f}x")

    # Compare reported sigma vs NLL-consistent sigma = exp(0.5 * log_var)
    from src.models.wrapper import prepare_imaging_input, prepare_footprint_input, prepare_spectrum_input

    batch0 = _move_batch(next(iter(loader)), device)
    targets0, masks0 = prepare_targets_and_masks(batch0, ctx.model_cfg)
    x = prepare_imaging_input(batch0, ctx.model_cfg).to(device)
    footprint = prepare_footprint_input(batch0, ctx.model_cfg)
    if footprint is not None:
        footprint = footprint.to(device)
    spec = prepare_spectrum_input(batch0, ctx.model_cfg)
    if spec is not None:
        spec = spec.to(device)
    with torch.no_grad():
        pred0, aux0 = models[0].model(x, spectrum_flux=spec, footprint=footprint)
        log_var = aux0["log_var"]
        sig_reported = aux0["sigma"]
        sig_nll = torch.exp(0.5 * log_var.clamp(-6, 6))
        err0 = torch.abs(pred0 - targets0)
        m0 = masks0 > 0
        e = err0[m0].cpu().numpy()
        sr = sig_reported[m0].cpu().numpy()
        sn = sig_nll[m0].cpu().numpy()
        lv = log_var[m0].cpu().numpy()
    print("\n=== sigma parameterization check (member_00, 1 batch) ===")
    print(f"  log_var: mean={lv.mean():.3f} median={np.median(lv):.3f}")
    print(f"  sigma_reported (softplus): mean={sr.mean():.5f}")
    print(f"  sigma_nll (exp(0.5*log_var)): mean={sn.mean():.5f}  ratio={sn.mean()/max(sr.mean(),1e-8):.2f}x")
    print(f"  coverage@1σ reported={np.mean(e<=sr):.3f}  nll-consistent={np.mean(e<=sn):.3f}")

    # Ensemble on same batch with corrected aleatoric
    with torch.no_grad():
        member_preds, log_vars = [], []
        for model in models:
            p, aux = model.model(x, spectrum_flux=spec, footprint=footprint)
            member_preds.append(p)
            log_vars.append(aux["log_var"])
        stacked = torch.stack(member_preds, dim=0)
        pred_e = stacked.mean(dim=0)
        sigma_epi_b = stacked.std(dim=0, unbiased=False)
        sigma_ale_fix = torch.exp(0.5 * torch.stack(log_vars, dim=0).mean(dim=0).clamp(-6, 6))
        sigma_tot_fix = torch.sqrt(sigma_epi_b**2 + sigma_ale_fix**2)
        err_e = torch.abs(pred_e - targets0)[m0].cpu().numpy()
        st_fix = sigma_tot_fix[m0].cpu().numpy()
    print(f"  ensemble batch cov@1σ with fixed σ_tot={np.mean(err_e<=st_fix):.3f}")

    # Single member aleatoric only

    # Member val coverage from training if available
    mem0 = ROOT / RUN / "members/member_00"
    hist = mem0 / "history.csv"
    if hist.exists():
        h = pd.read_csv(hist)
        last = h.iloc[-1]
        cols = [c for c in h.columns if "coverage" in c.lower() or "nll" in c.lower()]
        print(f"\n=== member_00 last epoch metrics ===")
        for c in cols:
            if pd.notna(last.get(c)):
                print(f"  {c}: {last[c]}")


if __name__ == "__main__":
    main()
