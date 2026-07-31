"""Load and freeze a trained MapGenerator from a previous run directory."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.models.config import ModelConfig
from src.models.wrapper import MapGenerator
from src.training.train import _load_checkpoint_state


def _read_user_config(run_dir: Path) -> dict[str, Any]:
    snap_path = run_dir / "config_used.json"
    if not snap_path.is_file():
        raise FileNotFoundError(f"Base run config snapshot not found: {snap_path}")
    snap = json.loads(snap_path.read_text(encoding="utf-8"))
    user = snap.get("user") or {}
    if not user:
        raise ValueError(f"config_used.json in {run_dir} has no 'user' section")
    return user


def resolve_base_checkpoint(run_dir: Path, checkpoint: str | Path | None = None) -> Path:
    if checkpoint is not None:
        path = Path(checkpoint)
        if not path.is_file():
            # Allow checkpoint name relative to run_dir/ckpts
            alt = run_dir / "ckpts" / path.name
            if alt.is_file():
                return alt
            raise FileNotFoundError(f"Base checkpoint not found: {checkpoint}")
        return path
    best = run_dir / "ckpts" / "best.pt"
    if best.is_file():
        return best
    raise FileNotFoundError(f"No best.pt under {run_dir / 'ckpts'}")


def load_frozen_base_map_generator(
    *,
    base_run_dir: str | Path,
    base_checkpoint: str | Path | None = None,
    device: torch.device | str | None = None,
    channel_key: str | None = "ha_flux",
) -> tuple[MapGenerator, ModelConfig, int | None]:
    """
    Instantiate MapGenerator from a saved run, load weights with strict=True, freeze.

    Returns
    -------
    base_model, base_config, channel_index
        ``channel_index`` is the index of ``channel_key`` in the base target stack,
        or None if ``channel_key`` is None (use all channels).
    """
    from runner import build_model_config

    run_dir = Path(base_run_dir)
    user_cfg = _read_user_config(run_dir)
    data_top = user_cfg.get("data", {})
    model_top = user_cfg.get("model", {})
    imaging_resolution = model_top.get(
        "imaging_resolution", data_top.get("imaging_resolution", "aligned")
    )
    base_config = build_model_config(
        model_top, data_top, imaging_resolution=imaging_resolution
    )
    base_config.validate()

    ckpt_path = resolve_base_checkpoint(run_dir, base_checkpoint)
    model = MapGenerator(base_config)
    state = _load_checkpoint_state(
        torch.load(ckpt_path, map_location="cpu", weights_only=False)
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    model.requires_grad_(False)

    channel_index: int | None = None
    if channel_key is not None:
        if channel_key not in base_config.target_keys:
            raise ValueError(
                f"channel_key={channel_key!r} not in base target_keys={base_config.target_keys}"
            )
        channel_index = list(base_config.target_keys).index(channel_key)

    if device is not None:
        model.to(device)
    return model, base_config, channel_index


@torch.inference_mode()
def frozen_base_predict(
    base_model: MapGenerator,
    batch: dict[str, object],
    *,
    channel_index: int | None = None,
) -> torch.Tensor:
    """Run frozen base under inference_mode; optionally extract one channel."""
    was_training = base_model.training
    base_model.eval()
    pred_dict, _ = base_model(batch)
    maps = pred_dict["maps"]
    if channel_index is not None:
        maps = maps[:, channel_index : channel_index + 1]
    if was_training:
        base_model.train()
    return maps
