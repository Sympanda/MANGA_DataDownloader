"""
Paper-style evaluation: pooled spaxel stats, obs-vs-pred, calibration, summary tables.

Supports single runs (point or gaussian head) and ensembles (manifest.json).
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader
from tqdm import tqdm

from manga_prep.targets.pipe3d_maps import AMARA_TARGET_KEYS
from runner import build_data_config, build_model_config
from src.data.ensemble_splits import load_ensemble_manifest
from src.data.make_dataloader import make_manga_dataloaders
from src.metrics.plots import plot_training_history
from src.metrics.plots import SIGMA_VMAX, SIGMA_VMIN
from src.metrics.uncertainty_plots import _coverage
from src.models.config import ModelConfig
from src.models.uncertainty_wrapper import UncertaintyMapGenerator
from src.models.wrapper import MapGenerator, prepare_footprint_input, prepare_imaging_input, prepare_spectrum_input, prepare_targets_and_masks
from src.training.train import _load_checkpoint_state

CHANNEL_LABELS: dict[str, str] = {
    "ha_flux": "Hα flux",
    "hbeta_flux": "Hβ flux",
    "oiii_5007_flux": "[O III]",
    "nii_6584_flux": "[N II]",
    "ha_ew": "Hα EW",
    "stellar_av": "Stellar Av",
}

FLUX_KEYS = frozenset({"ha_flux", "hbeta_flux", "oiii_5007_flux", "nii_6584_flux"})

# Nominal levels for reliability / TARP curves (central coverage & quantiles on [0, 1]).
NOMINAL_LEVELS = np.linspace(0.05, 0.95, 19)
DEFAULT_N_BOOTSTRAP = 200


@dataclass
class RunContext:
    run_name: str
    run_dir: Path
    kind: Literal["single", "ensemble", "member"]
    is_uncertainty: bool
    is_ensemble: bool
    user_cfg: dict
    model_cfg: ModelConfig
    member_dirs: list[Path] = field(default_factory=list)
    ensemble_name: str | None = None


@dataclass
class ChannelSpaxels:
    target: np.ndarray
    pred: np.ndarray
    residual: np.ndarray
    sigma: np.ndarray | None = None


@dataclass
class CoverageCounts:
    """Exact hit counts for coverage (no spaxel subsampling)."""

    n: int = 0
    within_1sigma: int = 0
    within_2sigma: int = 0

    def add(self, err: np.ndarray, sig: np.ndarray) -> None:
        valid = np.isfinite(err) & np.isfinite(sig) & (sig > 0)
        err = err[valid]
        sig = sig[valid]
        if err.size == 0:
            return
        self.n += int(err.size)
        self.within_1sigma += int(np.sum(err <= sig))
        self.within_2sigma += int(np.sum(err <= 2.0 * sig))

    def rate_1sigma(self) -> float:
        return float(self.within_1sigma / self.n) if self.n else float("nan")

    def rate_2sigma(self) -> float:
        return float(self.within_2sigma / self.n) if self.n else float("nan")


@dataclass
class EvalBundle:
    map_keys: tuple[str, ...]
    channels: dict[str, ChannelSpaxels]
    per_galaxy: list[dict[str, float | str]]
    has_sigma: bool
    is_ensemble: bool
    n_members: int
    coverage_spaxel: dict[str, CoverageCounts] = field(default_factory=dict)
    per_galaxy_cal: list[dict[str, np.ndarray | str]] = field(default_factory=list)


def _read_user_snapshot(path: Path) -> dict:
    snap = json.loads(path.read_text(encoding="utf-8"))
    return snap.get("user") or snap


def discover_run(save_root: Path, run_name: str) -> RunContext:
    run_dir = (save_root / run_name).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = load_ensemble_manifest(manifest_path)
        user_cfg = manifest.get("user_snapshot") or {}
        data_top = user_cfg.get("data", {})
        model_top = user_cfg.get("model", {})
        imaging_resolution = model_top.get("imaging_resolution", data_top.get("imaging_resolution", "aligned"))
        model_cfg = build_model_config(model_top, data_top, imaging_resolution=imaging_resolution)
        member_dirs = sorted(
            p for p in (run_dir / "members").iterdir() if p.is_dir() and (p / "ckpts" / "best.pt").is_file()
        )
        if not member_dirs:
            raise FileNotFoundError(f"No member checkpoints under {run_dir / 'members'}")
        return RunContext(
            run_name=run_name,
            run_dir=run_dir,
            kind="ensemble",
            is_uncertainty=model_cfg.output_head == "gaussian",
            is_ensemble=True,
            user_cfg=user_cfg,
            model_cfg=model_cfg,
            member_dirs=member_dirs,
            ensemble_name=run_name,
        )

    config_path = run_dir / "config_used.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"No config_used.json or manifest.json in {run_dir}")
    user_cfg = _read_user_snapshot(config_path)
    data_top = user_cfg.get("data", {})
    model_top = user_cfg.get("model", {})
    imaging_resolution = model_top.get("imaging_resolution", data_top.get("imaging_resolution", "aligned"))
    model_cfg = build_model_config(model_top, data_top, imaging_resolution=imaging_resolution)
    if not (run_dir / "ckpts" / "best.pt").is_file():
        raise FileNotFoundError(f"Missing checkpoint: {run_dir / 'ckpts' / 'best.pt'}")

    kind: Literal["single", "member"] = "single"
    ensemble_name = None
    if run_dir.parent.name == "members" and (run_dir.parent.parent / "manifest.json").is_file():
        kind = "member"
        ensemble_name = run_dir.parent.parent.name

    return RunContext(
        run_name=run_name,
        run_dir=run_dir,
        kind=kind,
        is_uncertainty=model_cfg.output_head == "gaussian",
        is_ensemble=False,
        user_cfg=user_cfg,
        model_cfg=model_cfg,
        member_dirs=[run_dir],
        ensemble_name=ensemble_name,
    )


def _build_dataloader(ctx: RunContext, *, split: str, batch_size: int) -> DataLoader:
    data_top = dict(ctx.user_cfg.get("data", {}))
    ensemble_top = ctx.user_cfg.get("ensemble", {})
    if ctx.is_ensemble:
        base_csv = ensemble_top.get("base_split_csv") or data_top.get("split", {}).get("split_csv_path")
        data_top.setdefault("split", {})["split_csv_path"] = base_csv
    data_cfg = build_data_config(data_top, imaging_resolution=ctx.model_cfg.imaging_resolution)
    data_cfg.augmentation.enabled = False
    _, dl_val, dl_test, _ = make_manga_dataloaders(
        data_cfg,
        {"train_batch_size": batch_size, "eval_batch_size": batch_size, "num_workers": 0},
    )
    loaders = {"val": dl_val, "test": dl_test}
    if split not in loaders:
        raise ValueError(f"Unknown split {split!r}; use val or test")
    return loaders[split]


def _load_model(ctx: RunContext, member_dir: Path, device: torch.device) -> torch.nn.Module:
    ckpt_path = member_dir / "ckpts" / "best.pt"
    snap_path = member_dir / "config_used.json"
    if snap_path.is_file():
        user_cfg = _read_user_snapshot(snap_path)
        data_top = user_cfg.get("data", {})
        model_top = user_cfg.get("model", {})
        imaging_resolution = model_top.get("imaging_resolution", data_top.get("imaging_resolution", "aligned"))
        model_cfg = build_model_config(model_top, data_top, imaging_resolution=imaging_resolution)
    else:
        model_cfg = ctx.model_cfg

    if model_cfg.output_head == "gaussian":
        model = UncertaintyMapGenerator(model_cfg)
    else:
        model = MapGenerator(model_cfg)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(_load_checkpoint_state(ckpt))
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def _forward_member(
    model: torch.nn.Module,
    batch: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    x = prepare_imaging_input(batch, model.config).to(device)
    footprint = prepare_footprint_input(batch, model.config)
    if footprint is not None:
        footprint = footprint.to(device)
    spec = prepare_spectrum_input(batch, model.config)
    if spec is not None:
        spec = spec.to(device)
    pred, aux = model.model(x, spectrum_flux=spec, footprint=footprint)
    sigma = aux.get("sigma") if isinstance(aux, dict) else None
    return pred, sigma


def _move_batch(batch: dict, device: torch.device) -> dict:
    out = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    if "inputs" in batch:
        inputs = dict(batch["inputs"])
        for key in ("sdss_imaging", "legacy_imaging"):
            if key in inputs:
                inputs[key] = inputs[key].to(device)
        if "spectrum" in inputs:
            inputs["spectrum"] = {sk: sv.to(device) for sk, sv in inputs["spectrum"].items()}
        out["inputs"] = inputs
    if "targets" in batch:
        out["targets"] = {k: v.to(device) for k, v in batch["targets"].items()}
    if "target_loss_masks" in batch:
        out["target_loss_masks"] = {k: v.to(device) for k, v in batch["target_loss_masks"].items()}
    if "footprint_mask" in batch:
        out["footprint_mask"] = batch["footprint_mask"].to(device)
    return out


class _SpaxelAccumulator:
    def __init__(self, map_keys: tuple[str, ...], *, has_sigma: bool, max_spaxels: int | None) -> None:
        self.map_keys = map_keys
        self._key_to_ch = {k: i for i, k in enumerate(map_keys)}
        self.has_sigma = has_sigma
        self.max_spaxels = max_spaxels
        self._chunks: dict[str, dict[str, list[np.ndarray]]] = {
            k: {"target": [], "pred": [], "residual": [], **({"sigma": []} if has_sigma else {})}
            for k in map_keys
        }
        self._cov_spaxel: dict[str, CoverageCounts] = {k: CoverageCounts() for k in map_keys}
        self._cov_spaxel_all = CoverageCounts()
        self.per_galaxy: list[dict[str, float | str]] = []
        self._per_galaxy_cal: list[dict[str, np.ndarray | str]] = []
        self._rng = np.random.default_rng(42)

    def _maybe_subsample(self, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        n = arrays["target"].size
        if self.max_spaxels is None or n <= self.max_spaxels:
            return arrays
        idx = self._rng.choice(n, size=self.max_spaxels, replace=False)
        return {k: v[idx] for k, v in arrays.items()}

    def add_batch(
        self,
        *,
        plateifus: list[str],
        targets: torch.Tensor,
        preds: torch.Tensor,
        masks: torch.Tensor,
        sigma: torch.Tensor | None,
    ) -> None:
        targets_np = targets.cpu().numpy()
        preds_np = preds.cpu().numpy()
        masks_np = masks.cpu().numpy()
        sigma_np = sigma.cpu().numpy() if sigma is not None else None

        for i, plateifu in enumerate(plateifus):
            row: dict[str, float | str] = {"plateifu": str(plateifu)}
            mse_vals = []
            for ch, key in enumerate(self.map_keys):
                m = masks_np[i, ch] > 0
                if not m.any():
                    row[f"mse_{key}"] = float("nan")
                    continue
                tgt = targets_np[i, ch][m]
                prd = preds_np[i, ch][m]
                res = prd - tgt
                chunk = {"target": tgt, "pred": prd, "residual": res}
                if sigma_np is not None:
                    chunk["sigma"] = sigma_np[i, ch][m]
                chunk = self._maybe_subsample(chunk)
                for name, arr in chunk.items():
                    self._chunks[key][name].append(arr)
                mse_vals.append(float(np.mean(res**2)))
                row[f"mse_{key}"] = mse_vals[-1]
            row["mse_all"] = float(np.nanmean(mse_vals)) if mse_vals else float("nan")
            for key in self.map_keys:
                if key not in FLUX_KEYS:
                    continue
                ch_idx = self._key_to_ch[key]
                m = masks_np[i, ch_idx] > 0
                if not m.any():
                    continue
                row[f"flux_target_{key}"] = float(np.sum(targets_np[i, ch_idx][m]))
                row[f"flux_pred_{key}"] = float(np.sum(preds_np[i, ch_idx][m]))
            if sigma_np is not None:
                gal_cov = CoverageCounts()
                gal_tgt_parts: list[np.ndarray] = []
                gal_prd_parts: list[np.ndarray] = []
                gal_sig_parts: list[np.ndarray] = []
                for ch, key in enumerate(self.map_keys):
                    m = masks_np[i, ch] > 0
                    if not m.any():
                        continue
                    tgt_ch = targets_np[i, ch][m]
                    prd_ch = preds_np[i, ch][m]
                    err = np.abs(prd_ch - tgt_ch)
                    sig = sigma_np[i, ch][m]
                    gal_tgt_parts.append(tgt_ch)
                    gal_prd_parts.append(prd_ch)
                    gal_sig_parts.append(sig)
                    self._cov_spaxel[key].add(err, sig)
                    self._cov_spaxel_all.add(err, sig)
                    gal_ch = CoverageCounts()
                    gal_ch.add(err, sig)
                    row[f"coverage_1sigma_{key}"] = gal_ch.rate_1sigma()
                    row[f"coverage_2sigma_{key}"] = gal_ch.rate_2sigma()
                    gal_cov.add(err, sig)
                row["coverage_1sigma"] = gal_cov.rate_1sigma()
                row["coverage_2sigma"] = gal_cov.rate_2sigma()
                if gal_tgt_parts:
                    self._per_galaxy_cal.append(
                        {
                            "plateifu": str(plateifu),
                            "target": np.concatenate(gal_tgt_parts),
                            "pred": np.concatenate(gal_prd_parts),
                            "sigma": np.concatenate(gal_sig_parts),
                        }
                    )
            self.per_galaxy.append(row)

    def finalize(self) -> EvalBundle:
        channels: dict[str, ChannelSpaxels] = {}
        for key in self.map_keys:
            chunks = self._chunks[key]
            channels[key] = ChannelSpaxels(
                target=np.concatenate(chunks["target"]) if chunks["target"] else np.array([]),
                pred=np.concatenate(chunks["pred"]) if chunks["pred"] else np.array([]),
                residual=np.concatenate(chunks["residual"]) if chunks["residual"] else np.array([]),
                sigma=np.concatenate(chunks["sigma"]) if chunks.get("sigma") else None,
            )
        cov = dict(self._cov_spaxel)
        cov["_all_channels"] = self._cov_spaxel_all
        return EvalBundle(
            map_keys=self.map_keys,
            channels=channels,
            per_galaxy=self.per_galaxy,
            has_sigma=self.has_sigma,
            is_ensemble=False,
            n_members=1,
            coverage_spaxel=cov,
            per_galaxy_cal=list(self._per_galaxy_cal),
        )


@torch.no_grad()
def collect_predictions(
    ctx: RunContext,
    dataloader: DataLoader,
    device: torch.device,
    *,
    max_spaxels: int | None = 500_000,
    limit_batches: int | None = None,
) -> EvalBundle:
    models = [_load_model(ctx, md, device) for md in ctx.member_dirs]
    has_sigma = ctx.is_uncertainty
    acc = _SpaxelAccumulator(ctx.model_cfg.target_keys, has_sigma=has_sigma, max_spaxels=max_spaxels)

    for bi, batch in enumerate(tqdm(dataloader, desc="paper eval", mininterval=1.0)):
        if limit_batches is not None and bi >= limit_batches:
            break
        batch = _move_batch(batch, device)
        targets, masks = prepare_targets_and_masks(batch, ctx.model_cfg)

        if len(models) == 1:
            pred, sigma = _forward_member(models[0], batch, device)
        else:
            member_preds = []
            member_sigmas = []
            for model in models:
                p, s = _forward_member(model, batch, device)
                member_preds.append(p)
                if s is not None:
                    member_sigmas.append(s)
            stacked = torch.stack(member_preds, dim=0)
            pred = stacked.mean(dim=0)
            if member_sigmas:
                sigma_epi = stacked.std(dim=0, unbiased=False)
                sigma_ale = torch.stack(member_sigmas, dim=0).mean(dim=0)
                sigma = torch.sqrt(sigma_epi**2 + sigma_ale**2)
            else:
                sigma = None

        acc.add_batch(
            plateifus=list(batch["plateifu"]),
            targets=targets,
            preds=pred,
            masks=masks,
            sigma=sigma,
        )

    bundle = acc.finalize()
    bundle.is_ensemble = ctx.is_ensemble and len(models) > 1
    bundle.n_members = len(models)
    bundle.has_sigma = has_sigma and any(
        bundle.channels[k].sigma is not None and bundle.channels[k].sigma.size > 0
        for k in bundle.map_keys
    )
    return bundle


def _channel_stats(ch: ChannelSpaxels) -> dict[str, float]:
    if ch.target.size == 0:
        return {k: float("nan") for k in ("n_spaxels", "rmse", "mae", "r2", "pearson_r", "bias", "median_abs_err")}
    err = ch.residual
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((ch.target - np.mean(ch.target)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    if ch.target.std() > 0 and ch.pred.std() > 0:
        pearson = float(np.corrcoef(ch.target, ch.pred)[0, 1])
    else:
        pearson = float("nan")
    return {
        "n_spaxels": float(ch.target.size),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": r2,
        "pearson_r": pearson,
        "bias": float(np.mean(err)),
        "median_abs_err": float(np.median(np.abs(err))),
    }


def build_summary_table(bundle: EvalBundle) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for key in bundle.map_keys:
        stats = _channel_stats(bundle.channels[key])
        ch = bundle.channels[key]
        row: dict[str, float | str] = {
            "channel": key,
            "label": CHANNEL_LABELS.get(key, key),
            **stats,
        }
        if bundle.has_sigma and ch.sigma is not None and ch.sigma.size:
            row["mean_sigma"] = float(np.mean(ch.sigma))
            row["median_sigma"] = float(np.median(ch.sigma))
            exact = bundle.coverage_spaxel.get(key)
            if exact is not None and exact.n > 0:
                row["coverage_1sigma"] = exact.rate_1sigma()
                row["coverage_2sigma"] = exact.rate_2sigma()
                row["n_spaxels_coverage"] = float(exact.n)
            else:
                row["coverage_1sigma"] = _coverage(np.abs(ch.residual), ch.sigma, k=1.0)
                row["coverage_2sigma"] = _coverage(np.abs(ch.residual), ch.sigma, k=2.0)
        rows.append(row)
    return rows


def build_coverage_summary(bundle: EvalBundle) -> list[dict[str, float | str]]:
    """Spaxel-level (exact, all test objects) and galaxy-level coverage tables."""
    if not bundle.has_sigma or not bundle.coverage_spaxel:
        return []

    rows: list[dict[str, float | str]] = []
    n_gal = len(bundle.per_galaxy)

    all_ch = bundle.coverage_spaxel.get("_all_channels")
    if all_ch is not None and all_ch.n > 0:
        rows.append(
            {
                "level": "spaxel",
                "scope": "all_channels",
                "channel": "_all",
                "label": "All channels (pooled spaxels)",
                "n_galaxies": float(n_gal),
                "n_units": float(all_ch.n),
                "coverage_1sigma": all_ch.rate_1sigma(),
                "coverage_2sigma": all_ch.rate_2sigma(),
                "target_1sigma": 0.68,
                "target_2sigma": 0.95,
            }
        )

    for key in bundle.map_keys:
        counts = bundle.coverage_spaxel.get(key)
        if counts is None or counts.n == 0:
            continue
        rows.append(
            {
                "level": "spaxel",
                "scope": "per_channel",
                "channel": key,
                "label": CHANNEL_LABELS.get(key, key),
                "n_galaxies": float(n_gal),
                "n_units": float(counts.n),
                "coverage_1sigma": counts.rate_1sigma(),
                "coverage_2sigma": counts.rate_2sigma(),
                "target_1sigma": 0.68,
                "target_2sigma": 0.95,
            }
        )

    if bundle.per_galaxy:
        gal_c1 = [float(r["coverage_1sigma"]) for r in bundle.per_galaxy if "coverage_1sigma" in r]
        gal_c2 = [float(r["coverage_2sigma"]) for r in bundle.per_galaxy if "coverage_2sigma" in r]
        gal_c1 = [v for v in gal_c1 if np.isfinite(v)]
        gal_c2 = [v for v in gal_c2 if np.isfinite(v)]
        if gal_c1:
            rows.append(
                {
                    "level": "galaxy",
                    "scope": "all_channels",
                    "channel": "_all",
                    "label": "All channels (per-galaxy mean)",
                    "n_galaxies": float(len(gal_c1)),
                    "n_units": float(len(gal_c1)),
                    "coverage_1sigma": float(np.mean(gal_c1)),
                    "coverage_2sigma": float(np.mean(gal_c2)),
                    "target_1sigma": 0.68,
                    "target_2sigma": 0.95,
                }
            )
            rows.append(
                {
                    "level": "galaxy",
                    "scope": "all_channels",
                    "channel": "_all",
                    "label": "All channels (per-galaxy median)",
                    "n_galaxies": float(len(gal_c1)),
                    "n_units": float(len(gal_c1)),
                    "coverage_1sigma": float(np.median(gal_c1)),
                    "coverage_2sigma": float(np.median(gal_c2)),
                    "target_1sigma": 0.68,
                    "target_2sigma": 0.95,
                }
            )

        for key in bundle.map_keys:
            c1 = [
                float(r[f"coverage_1sigma_{key}"])
                for r in bundle.per_galaxy
                if f"coverage_1sigma_{key}" in r and np.isfinite(float(r[f"coverage_1sigma_{key}"]))
            ]
            c2 = [
                float(r[f"coverage_2sigma_{key}"])
                for r in bundle.per_galaxy
                if f"coverage_2sigma_{key}" in r and np.isfinite(float(r[f"coverage_2sigma_{key}"]))
            ]
            if not c1:
                continue
            rows.append(
                {
                    "level": "galaxy",
                    "scope": "per_channel",
                    "channel": key,
                    "label": CHANNEL_LABELS.get(key, key),
                    "n_galaxies": float(len(c1)),
                    "n_units": float(len(c1)),
                    "coverage_1sigma": float(np.mean(c1)),
                    "coverage_2sigma": float(np.mean(c2)),
                    "target_1sigma": 0.68,
                    "target_2sigma": 0.95,
                }
            )

    return rows


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_obs_vs_pred_grid(bundle: EvalBundle, out_path: Path) -> None:
    keys = bundle.map_keys
    n = len(keys)
    ncols = 3
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows), dpi=150)
    axes = np.atleast_1d(axes).ravel()
    for ax, key in zip(axes, keys, strict=False):
        ch = bundle.channels[key]
        if ch.target.size == 0:
            ax.set_visible(False)
            continue
        hb = ax.hexbin(ch.target, ch.pred, gridsize=50, cmap="viridis", mincnt=1, linewidths=0.2)
        lo = float(min(ch.target.min(), ch.pred.min()))
        hi = float(max(ch.target.max(), ch.pred.max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="1:1")
        stats = _channel_stats(ch)
        ax.set_title(f"{CHANNEL_LABELS.get(key, key)}\nR²={stats['r2']:.3f}  RMSE={stats['rmse']:.4f}")
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
        fig.colorbar(hb, ax=ax, fraction=0.046)
    for ax in axes[len(keys) :]:
        ax.set_visible(False)
    title = "Observed vs predicted"
    if bundle.is_ensemble:
        title += f" (ensemble, n={bundle.n_members})"
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_residual_histogram(bundle: EvalBundle, out_path: Path) -> None:
    keys = bundle.map_keys
    ncols = 3
    nrows = int(math.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.5 * nrows), dpi=150)
    axes = np.atleast_1d(axes).ravel()
    for ax, key in zip(axes, keys, strict=False):
        ch = bundle.channels[key]
        if ch.residual.size == 0:
            ax.set_visible(False)
            continue
        ax.hist(ch.residual, bins=80, color="steelblue", alpha=0.85, density=True)
        ax.axvline(0.0, color="red", ls="--", lw=1)
        ax.set_title(CHANNEL_LABELS.get(key, key))
        ax.set_xlabel("Residual (pred − obs)")
    for ax in axes[len(keys) :]:
        ax.set_visible(False)
    fig.suptitle("Residual distributions")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_residual_vs_pred(bundle: EvalBundle, out_path: Path) -> None:
    keys = bundle.map_keys
    ncols = 3
    nrows = int(math.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), dpi=150)
    axes = np.atleast_1d(axes).ravel()
    for ax, key in zip(axes, keys, strict=False):
        ch = bundle.channels[key]
        if ch.pred.size == 0:
            ax.set_visible(False)
            continue
        ax.hexbin(ch.pred, ch.residual, gridsize=45, cmap="coolwarm", mincnt=1)
        ax.axhline(0.0, color="k", ls="--", lw=0.8)
        ax.set_title(CHANNEL_LABELS.get(key, key))
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual")
    for ax in axes[len(keys) :]:
        ax.set_visible(False)
    fig.suptitle("Residual vs predicted")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_per_galaxy_mse(bundle: EvalBundle, out_path: Path) -> None:
    if not bundle.per_galaxy:
        return
    keys = bundle.map_keys
    data = [[float(r.get(f"mse_{k}", float("nan"))) for r in bundle.per_galaxy] for k in keys]
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    parts = ax.violinplot(data, showmeans=True, showmedians=True)
    for body in parts["bodies"]:
        body.set_alpha(0.7)
    ax.set_xticks(range(1, len(keys) + 1))
    ax.set_xticklabels([CHANNEL_LABELS.get(k, k) for k in keys], rotation=25, ha="right")
    ax.set_ylabel("Per-galaxy MSE")
    ax.set_title("Per-galaxy error distribution")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_mse_cdf(bundle: EvalBundle, out_path: Path) -> None:
    if not bundle.per_galaxy:
        return
    mse = np.array([float(r["mse_all"]) for r in bundle.per_galaxy if np.isfinite(float(r["mse_all"]))])
    mse.sort()
    if mse.size == 0:
        return
    y = np.arange(1, mse.size + 1) / mse.size
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.plot(mse, y, lw=2)
    ax.set_xlabel("Per-galaxy MSE (mean over channels)")
    ax.set_ylabel("CDF")
    ax.set_title("Galaxy-level error CDF")
    ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_rmse_bars(summary: list[dict], out_path: Path) -> None:
    labels = [str(r["label"]) for r in summary]
    rmse = [float(r["rmse"]) for r in summary]
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.bar(labels, rmse, color="steelblue", alpha=0.85)
    ax.set_ylabel("RMSE")
    ax.set_title("Per-channel RMSE (pooled spaxels)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_coverage_summary(bundle: EvalBundle, out_dir: Path) -> None:
    if not bundle.has_sigma or not bundle.per_galaxy:
        return

    # Spaxel-level per channel (exact counts, all test objects)
    keys = bundle.map_keys
    spax_c1 = []
    spax_c2 = []
    for key in keys:
        counts = bundle.coverage_spaxel.get(key)
        if counts is None or counts.n == 0:
            spax_c1.append(float("nan"))
            spax_c2.append(float("nan"))
        else:
            spax_c1.append(counts.rate_1sigma())
            spax_c2.append(counts.rate_2sigma())

    gal_c1 = [float(r["coverage_1sigma"]) for r in bundle.per_galaxy if "coverage_1sigma" in r]
    gal_c2 = [float(r["coverage_2sigma"]) for r in bundle.per_galaxy if "coverage_2sigma" in r]
    gal_c1 = [v for v in gal_c1 if np.isfinite(v)]
    gal_c2 = [v for v in gal_c2 if np.isfinite(v)]

    labels = [CHANNEL_LABELS.get(k, k) for k in keys]
    x = np.arange(len(keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    ax.bar(x - width / 2, spax_c1, width, label="Spaxel 1σ", color="#4C72B0", alpha=0.85)
    ax.bar(x + width / 2, spax_c2, width, label="Spaxel 2σ", color="#55A868", alpha=0.85)
    ax.axhline(0.68, color="gray", ls=":", lw=1)
    ax.axhline(0.95, color="gray", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Observed coverage")
    ax.set_title(f"Spaxel-level coverage (all {len(bundle.per_galaxy)} test galaxies, exact counts)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "coverage_spaxel_by_channel.png", bbox_inches="tight")
    plt.close(fig)

    # Galaxy-level distribution (one value per galaxy, channels pooled)
    if gal_c1:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=150)
        axes[0].hist(gal_c1, bins=30, color="#4C72B0", alpha=0.85, edgecolor="white")
        axes[0].axvline(0.68, color="gray", ls=":", lw=1.5, label="target 68%")
        axes[0].axvline(float(np.mean(gal_c1)), color="red", ls="--", lw=1.2, label=f"mean={np.mean(gal_c1):.2f}")
        axes[0].set_xlabel("Per-galaxy coverage @ 1σ")
        axes[0].set_ylabel("Number of galaxies")
        axes[0].set_title("Galaxy-level 1σ coverage")
        axes[0].legend(fontsize=8)

        axes[1].hist(gal_c2, bins=30, color="#55A868", alpha=0.85, edgecolor="white")
        axes[1].axvline(0.95, color="gray", ls=":", lw=1.5, label="target 95%")
        axes[1].axvline(float(np.mean(gal_c2)), color="red", ls="--", lw=1.2, label=f"mean={np.mean(gal_c2):.2f}")
        axes[1].set_xlabel("Per-galaxy coverage @ 2σ")
        axes[1].set_title("Galaxy-level 2σ coverage")
        axes[1].legend(fontsize=8)
        fig.suptitle(f"Coverage per galaxy (channels pooled, n={len(gal_c1)} test galaxies)")
        fig.tight_layout()
        fig.savefig(out_dir / "coverage_galaxy_histogram.png", bbox_inches="tight")
        plt.close(fig)

    # Compare spaxel pooled vs galaxy mean (all channels)
    all_spax = bundle.coverage_spaxel.get("_all_channels")
    if all_spax is not None and all_spax.n > 0 and gal_c1:
        fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
        groups = ["Spaxel\n(all channels)", "Galaxy mean\n(all channels)"]
        c1 = [all_spax.rate_1sigma(), float(np.mean(gal_c1))]
        c2 = [all_spax.rate_2sigma(), float(np.mean(gal_c2))]
        xg = np.arange(2)
        ax.bar(xg - 0.2, c1, 0.35, label="1σ (68% target)", color="#4C72B0", alpha=0.85)
        ax.bar(xg + 0.2, c2, 0.35, label="2σ (95% target)", color="#55A868", alpha=0.85)
        ax.axhline(0.68, color="gray", ls=":", lw=1)
        ax.axhline(0.95, color="gray", ls=":", lw=1)
        ax.set_xticks(xg)
        ax.set_xticklabels(groups)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Observed coverage")
        ax.set_title("Spaxel vs galaxy-level coverage")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "coverage_spaxel_vs_galaxy.png", bbox_inches="tight")
        plt.close(fig)


@dataclass
class CalibrationCurves:
    nominals: np.ndarray
    coverage: np.ndarray
    coverage_lo: np.ndarray
    coverage_hi: np.ndarray
    tarp: np.ndarray
    tarp_lo: np.ndarray
    tarp_hi: np.ndarray
    pit: np.ndarray
    pit_ks_stat: float
    pit_ks_pvalue: float


def _valid_cal_mask(target: np.ndarray, pred: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return np.isfinite(target) & np.isfinite(pred) & np.isfinite(sigma) & (sigma > 0)


def _symmetric_coverage_observed(
    target: np.ndarray, pred: np.ndarray, sigma: np.ndarray, nominal_p: float
) -> float:
    m = _valid_cal_mask(target, pred, sigma)
    t, p, s = target[m], pred[m], sigma[m]
    if t.size == 0:
        return float("nan")
    k = stats.norm.ppf(0.5 + nominal_p / 2.0)
    return float(np.mean(np.abs(t - p) <= k * s))


def _tarp_observed(target: np.ndarray, pred: np.ndarray, sigma: np.ndarray, nominal_q: float) -> float:
    m = _valid_cal_mask(target, pred, sigma)
    t, p, s = target[m], pred[m], sigma[m]
    if t.size == 0:
        return float("nan")
    thresh = p + s * stats.norm.ppf(nominal_q)
    return float(np.mean(t <= thresh))


def _pit_values(target: np.ndarray, pred: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    m = _valid_cal_mask(target, pred, sigma)
    t, p, s = target[m], pred[m], sigma[m]
    if t.size == 0:
        return np.array([], dtype=np.float64)
    z = (t - p) / np.maximum(s, 1e-8)
    return stats.norm.cdf(z)


def _curve_symmetric_coverage(
    target: np.ndarray, pred: np.ndarray, sigma: np.ndarray, nominals: np.ndarray
) -> np.ndarray:
    return np.array([_symmetric_coverage_observed(target, pred, sigma, float(p)) for p in nominals])


def _curve_tarp(
    target: np.ndarray, pred: np.ndarray, sigma: np.ndarray, nominals: np.ndarray
) -> np.ndarray:
    return np.array([_tarp_observed(target, pred, sigma, float(q)) for q in nominals])


def _pit_ks_test(pit: np.ndarray) -> tuple[float, float]:
    if pit.size < 2:
        return float("nan"), float("nan")
    result = stats.kstest(pit, "uniform")
    return float(result.statistic), float(result.pvalue)


def _spaxel_calibration_with_bootstrap(
    per_galaxy_cal: list[dict[str, np.ndarray | str]],
    nominals: np.ndarray,
    n_boot: int,
    seed: int,
) -> CalibrationCurves | None:
    galaxies = [g for g in per_galaxy_cal if isinstance(g.get("target"), np.ndarray) and g["target"].size > 0]
    n_gal = len(galaxies)
    if n_gal == 0:
        return None

    tgt = np.concatenate([g["target"] for g in galaxies])  # type: ignore[arg-type]
    prd = np.concatenate([g["pred"] for g in galaxies])  # type: ignore[arg-type]
    sig = np.concatenate([g["sigma"] for g in galaxies])  # type: ignore[arg-type]
    pit = _pit_values(tgt, prd, sig)
    ks_stat, ks_p = _pit_ks_test(pit)

    cov = _curve_symmetric_coverage(tgt, prd, sig, nominals)
    tarp = _curve_tarp(tgt, prd, sig, nominals)

    rng = np.random.default_rng(seed)
    cov_boot = np.zeros((n_boot, len(nominals)))
    tarp_boot = np.zeros((n_boot, len(nominals)))
    for b in range(n_boot):
        idx = rng.integers(0, n_gal, size=n_gal)
        bt = np.concatenate([galaxies[i]["target"] for i in idx])  # type: ignore[arg-type]
        bp = np.concatenate([galaxies[i]["pred"] for i in idx])  # type: ignore[arg-type]
        bs = np.concatenate([galaxies[i]["sigma"] for i in idx])  # type: ignore[arg-type]
        cov_boot[b] = _curve_symmetric_coverage(bt, bp, bs, nominals)
        tarp_boot[b] = _curve_tarp(bt, bp, bs, nominals)

    return CalibrationCurves(
        nominals=nominals,
        coverage=cov,
        coverage_lo=np.percentile(cov_boot, 2.5, axis=0),
        coverage_hi=np.percentile(cov_boot, 97.5, axis=0),
        tarp=tarp,
        tarp_lo=np.percentile(tarp_boot, 2.5, axis=0),
        tarp_hi=np.percentile(tarp_boot, 97.5, axis=0),
        pit=pit,
        pit_ks_stat=ks_stat,
        pit_ks_pvalue=ks_p,
    )


def _galaxy_calibration_with_bootstrap(
    per_galaxy_cal: list[dict[str, np.ndarray | str]],
    nominals: np.ndarray,
    n_boot: int,
    seed: int,
) -> CalibrationCurves | None:
    galaxies = [g for g in per_galaxy_cal if isinstance(g.get("target"), np.ndarray) and g["target"].size > 0]
    n_gal = len(galaxies)
    if n_gal == 0:
        return None

    gal_cov = np.array(
        [_curve_symmetric_coverage(g["target"], g["pred"], g["sigma"], nominals) for g in galaxies]  # type: ignore[arg-type]
    )
    gal_tarp = np.array(
        [_curve_tarp(g["target"], g["pred"], g["sigma"], nominals) for g in galaxies]  # type: ignore[arg-type]
    )
    cov = gal_cov.mean(axis=0)
    tarp = gal_tarp.mean(axis=0)

    # One mean PIT per galaxy (galaxy-level PIT summary).
    pit = np.array(
        [float(np.mean(_pit_values(g["target"], g["pred"], g["sigma"]))) for g in galaxies]  # type: ignore[arg-type]
    )
    ks_stat, ks_p = _pit_ks_test(pit)

    rng = np.random.default_rng(seed)
    cov_boot = np.zeros((n_boot, len(nominals)))
    tarp_boot = np.zeros((n_boot, len(nominals)))
    for b in range(n_boot):
        idx = rng.integers(0, n_gal, size=n_gal)
        cov_boot[b] = gal_cov[idx].mean(axis=0)
        tarp_boot[b] = gal_tarp[idx].mean(axis=0)

    return CalibrationCurves(
        nominals=nominals,
        coverage=cov,
        coverage_lo=np.percentile(cov_boot, 2.5, axis=0),
        coverage_hi=np.percentile(cov_boot, 97.5, axis=0),
        tarp=tarp,
        tarp_lo=np.percentile(tarp_boot, 2.5, axis=0),
        tarp_hi=np.percentile(tarp_boot, 97.5, axis=0),
        pit=pit,
        pit_ks_stat=ks_stat,
        pit_ks_pvalue=ks_p,
    )


def _plot_pit_background(ax: plt.Axes, pit: np.ndarray, max_frac: float = 0.22) -> None:
    if pit.size == 0:
        return
    counts, edges = np.histogram(pit, bins=25, range=(0.0, 1.0), density=True)
    scale = max_frac / max(float(counts.max()), 1e-6)
    heights = counts * scale
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    ax.bar(
        centers,
        heights,
        width=widths,
        align="center",
        alpha=0.35,
        color="#9aa0a6",
        edgecolor="none",
        zorder=1,
        label="PIT density",
    )


def _plot_combined_calibration_panel(
    curves: CalibrationCurves,
    out_path: Path,
    *,
    title: str,
    n_galaxies: int,
    n_spaxels: int | None = None,
) -> None:
    x = curves.nominals
    fig, ax = plt.subplots(figsize=(6.2, 6.2), dpi=150)
    _plot_pit_background(ax, curves.pit)
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, zorder=2, label="Perfect calibration")
    ax.fill_between(
        x,
        curves.coverage_lo,
        curves.coverage_hi,
        color="#4C72B0",
        alpha=0.2,
        zorder=3,
    )
    ax.fill_between(
        x,
        curves.tarp_lo,
        curves.tarp_hi,
        color="#C44E52",
        alpha=0.2,
        zorder=3,
    )
    ax.plot(x, curves.coverage, color="#4C72B0", lw=2.0, zorder=4, label="Coverage (symmetric)")
    ax.plot(x, curves.tarp, color="#C44E52", lw=2.0, zorder=4, label="TARP (quantile)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Nominal level")
    ax.set_ylabel("Observed fraction")
    subtitle = f"n={n_galaxies} galaxies"
    if n_spaxels is not None:
        subtitle += f", {n_spaxels:,} spaxels"
    ax.set_title(f"{title}\n{subtitle}")
    ks_txt = (
        f"PIT KS: D={curves.pit_ks_stat:.3f}, p={curves.pit_ks_pvalue:.2e}"
        if np.isfinite(curves.pit_ks_stat)
        else "PIT KS: n/a"
    )
    ax.text(0.03, 0.97, ks_txt, transform=ax.transAxes, va="top", ha="left", fontsize=8)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25, linestyle="--")
    ax.set_aspect("equal")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_combined_calibration(
    bundle: EvalBundle,
    out_dir: Path,
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
) -> list[dict[str, float | str]]:
    if not bundle.has_sigma or not bundle.per_galaxy_cal:
        return []

    nominals = NOMINAL_LEVELS
    rows: list[dict[str, float | str]] = []

    spaxel = _spaxel_calibration_with_bootstrap(bundle.per_galaxy_cal, nominals, n_bootstrap, seed=42)
    if spaxel is not None:
        n_spax = int(spaxel.pit.size)
        _plot_combined_calibration_panel(
            spaxel,
            out_dir / "calibration_combined_spaxel.png",
            title="Spaxel-level calibration (coverage + TARP + PIT)",
            n_galaxies=len(bundle.per_galaxy_cal),
            n_spaxels=n_spax,
        )
        rows.append(
            {
                "level": "spaxel",
                "n_galaxies": float(len(bundle.per_galaxy_cal)),
                "n_units": float(n_spax),
                "pit_ks_stat": spaxel.pit_ks_stat,
                "pit_ks_pvalue": spaxel.pit_ks_pvalue,
                "coverage_at_68": float(spaxel.coverage[np.argmin(np.abs(nominals - 0.68))]),
                "n_bootstrap": float(n_bootstrap),
            }
        )

    galaxy = _galaxy_calibration_with_bootstrap(bundle.per_galaxy_cal, nominals, n_bootstrap, seed=43)
    if galaxy is not None:
        _plot_combined_calibration_panel(
            galaxy,
            out_dir / "calibration_combined_galaxy.png",
            title="Galaxy-level calibration (coverage + TARP + PIT)",
            n_galaxies=len(bundle.per_galaxy_cal),
        )
        rows.append(
            {
                "level": "galaxy",
                "n_galaxies": float(len(bundle.per_galaxy_cal)),
                "n_units": float(len(bundle.per_galaxy_cal)),
                "pit_ks_stat": galaxy.pit_ks_stat,
                "pit_ks_pvalue": galaxy.pit_ks_pvalue,
                "coverage_at_68": float(galaxy.coverage[np.argmin(np.abs(nominals - 0.68))]),
                "n_bootstrap": float(n_bootstrap),
            }
        )

    if rows:
        _write_csv(rows, out_dir / "calibration_diagnostics.csv")
    return rows


def _plot_calibration(bundle: EvalBundle, out_dir: Path, *, n_bootstrap: int = DEFAULT_N_BOOTSTRAP) -> None:
    if not bundle.has_sigma:
        return
    all_err = []
    all_sig = []
    for key in bundle.map_keys:
        ch = bundle.channels[key]
        if ch.sigma is None or ch.sigma.size == 0:
            continue
        all_err.append(np.abs(ch.residual))
        all_sig.append(ch.sigma)
    if not all_err:
        return
    err = np.concatenate(all_err)
    sig = np.concatenate(all_sig)
    valid = np.isfinite(err) & np.isfinite(sig) & (sig > 0)
    err, sig = err[valid], sig[valid]
    if err.size == 0:
        return

    # Reliability diagram
    nominal = np.array([0.5, 0.68, 0.9, 0.95])
    observed = []
    for q in nominal:
        k = {0.5: 0.674, 0.68: 1.0, 0.9: 1.645, 0.95: 1.96}.get(float(q), 1.0)
        observed.append(float(np.mean(err <= k * sig)))
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    ax.scatter(nominal, observed, s=80, zorder=3, label="Model")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Observed coverage")
    ax.set_title("Uncertainty calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_dir / "calibration_reliability.png", bbox_inches="tight")
    plt.close(fig)

    # Binned sigma vs RMSE
    edges = np.quantile(sig, np.linspace(0, 1, 11))
    edges = np.unique(edges)
    bin_centers = []
    bin_rmse = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        m = (sig >= lo) & (sig < hi) if hi < edges[-1] else (sig >= lo) & (sig <= hi)
        if not np.any(m):
            continue
        bin_centers.append(float(np.mean(sig[m])))
        bin_rmse.append(float(np.sqrt(np.mean(err[m] ** 2))))
    if bin_centers:
        fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
        ax.plot(bin_centers, bin_centers, "k--", label="Perfect")
        ax.scatter(bin_centers, bin_rmse, s=50, label="Binned RMSE")
        ax.set_xlabel("Mean predicted σ in bin")
        ax.set_ylabel("Observed RMSE")
        ax.set_title("Calibration curve")
        ax.legend()
        ax.grid(alpha=0.3, linestyle="--")
        fig.tight_layout()
        fig.savefig(out_dir / "calibration_curve.png", bbox_inches="tight")
        plt.close(fig)

    # Coverage bar chart (exact spaxel counts, all test objects)
    all_spax = bundle.coverage_spaxel.get("_all_channels")
    if all_spax is not None and all_spax.n > 0:
        cov1, cov2 = all_spax.rate_1sigma(), all_spax.rate_2sigma()
    else:
        cov1 = _coverage(err, sig, k=1.0)
        cov2 = _coverage(err, sig, k=2.0)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    ax.bar(["1σ (68%)", "2σ (95%)"], [cov1, cov2], color=["#4C72B0", "#55A868"], alpha=0.85)
    ax.axhline(0.68, color="gray", ls=":", lw=1)
    ax.axhline(0.95, color="gray", ls=":", lw=1)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Observed coverage")
    ax.set_title("Spaxel coverage (all channels, exact)")
    fig.tight_layout()
    fig.savefig(out_dir / "coverage_bars.png", bbox_inches="tight")
    plt.close(fig)

    # |residual| vs sigma
    idx = np.random.default_rng(0).choice(err.size, size=min(50_000, err.size), replace=False)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    ax.hexbin(sig[idx], err[idx], gridsize=45, cmap="magma", mincnt=1)
    sline = np.linspace(0, SIGMA_VMAX, 50)
    ax.plot(sline, sline, "c--", lw=1, label="|err|=σ")
    ax.plot(sline, 2 * sline, "y--", lw=1, label="|err|=2σ")
    ax.set_xlim(SIGMA_VMIN, SIGMA_VMAX)
    ax.set_xlabel("Predicted σ")
    ax.set_ylabel("|Residual|")
    ax.set_title("|Residual| vs σ")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "residual_vs_sigma.png", bbox_inches="tight")
    plt.close(fig)

    _plot_combined_calibration(bundle, out_dir, n_bootstrap=n_bootstrap)


def _plot_integrated_flux(bundle: EvalBundle, out_path: Path) -> None:
    flux_keys = [k for k in bundle.map_keys if k in FLUX_KEYS]
    if not flux_keys or not bundle.per_galaxy:
        return
    ncols = 2
    nrows = int(math.ceil(len(flux_keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.2 * nrows), dpi=150)
    axes = np.atleast_1d(axes).ravel()
    for ax, key in zip(axes, flux_keys, strict=False):
        obs = np.array([float(r[f"flux_target_{key}"]) for r in bundle.per_galaxy if f"flux_target_{key}" in r])
        prd = np.array([float(r[f"flux_pred_{key}"]) for r in bundle.per_galaxy if f"flux_pred_{key}" in r])
        if obs.size == 0:
            ax.set_visible(False)
            continue
        ax.scatter(obs, prd, s=12, alpha=0.5, edgecolors="none")
        lo, hi = float(min(obs.min(), prd.min())), float(max(obs.max(), prd.max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.2)
        if obs.std() > 0 and prd.std() > 0:
            r = float(np.corrcoef(obs, prd)[0, 1])
        else:
            r = float("nan")
        ax.set_title(f"{CHANNEL_LABELS.get(key, key)}  r={r:.3f}")
        ax.set_xlabel("Integrated observed")
        ax.set_ylabel("Integrated predicted")
    for ax in axes[len(flux_keys) :]:
        ax.set_visible(False)
    fig.suptitle("Integrated flux per galaxy")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _copy_learning_curves(ctx: RunContext, out_dir: Path) -> None:
    hist_paths: list[Path] = []
    if ctx.is_ensemble:
        for md in ctx.member_dirs:
            p = md / "csv" / "train_val_history.csv"
            if p.is_file():
                hist_paths.append(p)
    else:
        p = ctx.run_dir / "csv" / "train_val_history.csv"
        if p.is_file():
            hist_paths.append(p)
    if not hist_paths:
        return
    # Plot first member / single run
    import pandas as pd

    rows = pd.read_csv(hist_paths[0]).to_dict(orient="records")
    plot_training_history(rows, out_dir)


def generate_paper_plots(
    bundle: EvalBundle,
    ctx: RunContext,
    out_dir: Path,
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
) -> list[dict[str, float | str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary_table(bundle)
    coverage_summary = build_coverage_summary(bundle)

    _write_csv(summary, out_dir / "summary_spaxel_stats.csv")
    _write_csv(bundle.per_galaxy, out_dir / "per_galaxy_metrics.csv")
    if coverage_summary:
        _write_csv(coverage_summary, out_dir / "coverage_summary.csv")
    (out_dir / "summary_spaxel_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    meta = {
        "run_name": ctx.run_name,
        "kind": ctx.kind,
        "is_ensemble": bundle.is_ensemble,
        "is_uncertainty": bundle.has_sigma,
        "n_members": bundle.n_members,
        "map_keys": list(bundle.map_keys),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    _plot_obs_vs_pred_grid(bundle, out_dir / "obs_vs_pred_grid.png")
    _plot_residual_histogram(bundle, out_dir / "residual_histogram.png")
    _plot_residual_vs_pred(bundle, out_dir / "residual_vs_pred.png")
    _plot_per_galaxy_mse(bundle, out_dir / "per_galaxy_mse_violin.png")
    _plot_mse_cdf(bundle, out_dir / "mse_cdf.png")
    _plot_rmse_bars(summary, out_dir / "rmse_by_channel.png")
    _plot_integrated_flux(bundle, out_dir / "integrated_flux_scatter.png")
    _plot_calibration(bundle, out_dir, n_bootstrap=n_bootstrap)
    _plot_coverage_summary(bundle, out_dir)
    _copy_learning_curves(ctx, out_dir)

    return summary


def run_paper_eval(
    *,
    save_root: Path,
    run_name: str,
    split: str = "test",
    device: str = "cuda",
    batch_size: int = 32,
    max_spaxels: int | None = 500_000,
    limit_batches: int | None = None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
) -> Path:
    ctx = discover_run(save_root, run_name)
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    dataloader = _build_dataloader(ctx, split=split, batch_size=batch_size)
    bundle = collect_predictions(
        ctx,
        dataloader,
        torch_device,
        max_spaxels=max_spaxels,
        limit_batches=limit_batches,
    )
    out_dir = ctx.run_dir / "paper_eval" / split
    generate_paper_plots(bundle, ctx, out_dir, n_bootstrap=n_bootstrap)
    return out_dir
