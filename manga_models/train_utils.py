from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Iterable

import torch

from manga_models.config import ConditionalUNetConfig, MODEL_PRESETS
from manga_prep.io.aligned_cache import count_aligned_caches
from manga_prep.dataset.manga_dataset import MangaGalaxyDataset


TARGET_LABELS = {
    "ha_flux": "Hα flux",
    "hbeta_flux": "Hβ flux",
    "oiii_5007_flux": "[OIII]5007",
    "nii_6584_flux": "[NII]6584",
    "ha_ew": "Hα EW",
    "stellar_av": "Stellar Av",
}


def config_from_dict(raw: dict) -> ConditionalUNetConfig:
    valid = {f.name for f in fields(ConditionalUNetConfig)}
    kwargs = {k: v for k, v in raw.items() if k in valid}
    if "target_keys" in kwargs:
        kwargs["target_keys"] = tuple(kwargs["target_keys"])
    return ConditionalUNetConfig(**kwargs)


def load_config_from_run(run_dir: Path) -> ConditionalUNetConfig:
    path = run_dir / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    return config_from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_config_from_checkpoint(ckpt_path: Path) -> ConditionalUNetConfig:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "config" not in ckpt:
        raise KeyError(f"Checkpoint {ckpt_path} has no 'config' field; pass --run-dir with config.json")
    return config_from_dict(ckpt["config"])


def report_aligned_cache_status(dataset: MangaGalaxyDataset) -> None:
    counts = count_aligned_caches(dataset.data_root, dataset.rows)
    if dataset.include_sdss_imaging:
        cached, eligible = counts["sdss_cached"], counts["sdss_eligible"]
        print(f"  SDSS aligned cache: {cached:,}/{eligible:,} galaxies")
        if cached < eligible:
            print(
                "  tip: run  python -m manga_prep export-aligned-imaging "
                "--survey sdss --use-index --skip-existing --workers 8"
            )
    if dataset.include_legacy_imaging:
        cached, eligible = counts["legacy_cached"], counts["legacy_eligible"]
        print(f"  Legacy aligned cache: {cached:,}/{eligible:,} galaxies")
        if cached < eligible:
            print(
                "  tip: run  python -m manga_prep export-aligned-imaging "
                "--survey legacy --use-index --skip-existing --workers 8"
            )


def build_dataset_from_config(
    config: ConditionalUNetConfig,
    data_root: Path,
) -> MangaGalaxyDataset:
    return MangaGalaxyDataset(
        data_root,
        data_root / "manga_dataset_index.csv",
        include_sdss_imaging=config.use_sdss,
        include_legacy_imaging=config.use_legacy,
        include_targets=True,
        spectrum="fake" if config.use_spectrum else None,
        align_imaging_to_amara_grid=True,
        prefer_aligned_cache=True,
        require_all=True,
    )


def add_training_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=Path("manga_sdss_fits"))
    parser.add_argument("--use-sdss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-legacy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-spectrum", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-footprint-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--model-size",
        choices=tuple(MODEL_PRESETS.keys()),
        default="medium",
        help="Architecture + loss preset (small=v1, medium=recommended, large=heavier)",
    )
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--bottleneck-multiplier", type=int, choices=(8, 16), default=None)
    parser.add_argument("--cond-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument(
        "--upsample-mode",
        choices=("bilinear", "transpose"),
        default=None,
        help="Decoder upsampling: bilinear (default, no grid artifacts) or transpose (sharper but checkerboard risk)",
    )
    parser.add_argument("--loss-mse-weight", type=float, default=None)
    parser.add_argument("--loss-l1-weight", type=float, default=None)
    parser.add_argument("--loss-grad-weight", type=float, default=None)


def config_from_args(args: argparse.Namespace) -> ConditionalUNetConfig:
    preset = dict(MODEL_PRESETS[args.model_size])

    def _get(name: str):
        value = getattr(args, name, None)
        if value is not None:
            return value
        return preset.get(name.replace("_", "-") if False else name)  # noqa: silly

    # Map CLI dest names to preset/config field names
    overrides = {
        "base_channels": args.base_channels,
        "bottleneck_multiplier": args.bottleneck_multiplier,
        "cond_dim": args.cond_dim,
        "dropout": args.dropout,
        "upsample_mode": args.upsample_mode,
        "loss_mse_weight": args.loss_mse_weight,
        "loss_l1_weight": args.loss_l1_weight,
        "loss_grad_weight": args.loss_grad_weight,
    }
    for key, value in overrides.items():
        if value is not None:
            preset[key] = value

    return ConditionalUNetConfig(
        use_sdss=args.use_sdss,
        use_legacy=args.use_legacy,
        use_spectrum=args.use_spectrum,
        use_footprint_mask=args.use_footprint_mask,
        spectrum_injection="bottleneck" if args.use_spectrum else "none",
        **preset,
    )


def save_run_config(config: ConditionalUNetConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        **config.__dict__,
        "target_keys": list(config.target_keys),
        "input_channels": config.input_channels(),
    }
    (out_dir / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_training_device(device_str: str) -> torch.device:
    """Parse --device and fail fast with a clear message if CUDA was requested but unavailable."""
    requested = torch.device(device_str)
    if requested.type != "cuda":
        return requested

    if not torch.backends.cuda.is_built():
        raise SystemExit(
            "PyTorch in this environment is CPU-only (no CUDA build). "
            "Use --device cpu, or reinstall PyTorch with CUDA support, e.g.:\n"
            "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
        )

    if not torch.cuda.is_available():
        raise SystemExit(
            f"CUDA device {device_str!r} was requested but no GPU is available. "
            "Check nvidia-smi and drivers, or use --device cpu."
        )

    if requested.index is not None and requested.index >= torch.cuda.device_count():
        raise SystemExit(
            f"CUDA device {device_str!r} not found "
            f"({torch.cuda.device_count()} GPU(s) visible)."
        )

    return requested


def pick_eval_indices(n_total: int, n_samples: int, seed: int) -> list[int]:
    import numpy as np

    rng = np.random.default_rng(seed)
    n_samples = min(n_samples, n_total)
    return sorted(int(i) for i in rng.choice(n_total, size=n_samples, replace=False))
