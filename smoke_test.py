"""Quick smoke test for the refactored pipeline (no real data)."""
from __future__ import annotations

import torch

from src.models.config import ModelConfig
from src.models.wrapper import MapGenerator


def _dummy_batch(cfg: ModelConfig, batch_size: int = 2) -> dict:
    target = cfg.target_spatial_size
    img_size = 76 if cfg.imaging_resolution == "aligned" else 196
    batch: dict = {"footprint_mask": torch.ones(batch_size, target, target)}
    inputs: dict = {}
    if cfg.use_sdss:
        inputs["sdss_imaging"] = torch.randn(batch_size, 5, img_size, img_size)
    if cfg.use_spectrum:
        inputs["spectrum"] = {
            "flux": torch.randn(batch_size, cfg.spectrum_n_wave),
            "wave": torch.randn(batch_size, cfg.spectrum_n_wave),
            "ivar": torch.randn(batch_size, cfg.spectrum_n_wave),
        }
    batch["inputs"] = inputs
    batch["targets"] = {k: torch.rand(batch_size, target, target) for k in cfg.target_keys}
    batch["target_loss_masks"] = {k: torch.ones(batch_size, target, target) for k in cfg.target_keys}
    return batch


def main() -> int:
    combos = [
        dict(spatial_pipeline="symmetric", imaging_resolution="aligned", footprint_mode="spatial_channel"),
        dict(spatial_pipeline="hr_encoder", imaging_resolution="native", footprint_mode="fusion_concat"),
        dict(spatial_pipeline="hr_full", imaging_resolution="native", footprint_mode="fusion_concat"),
        dict(spatial_pipeline="hr_encoder", imaging_resolution="native", footprint_mode="loss_only"),
    ]
    for arch in ("unet", "unetpp"):
        for head in ("single", "coarse_fine"):
            for film in ("bottleneck", "encoder"):
                for spatial in combos:
                    cfg = ModelConfig(
                        architecture=arch,
                        output_head=head,
                        film_injection=film,
                        cond_dim=64,
                        upsample_mode="pixel_shuffle",
                        **spatial,
                    )
                    model = MapGenerator(cfg)
                    pred, loss = model(_dummy_batch(cfg))
                    t = cfg.target_spatial_size
                    assert pred["maps"].shape == (2, cfg.n_target_maps, t, t)
                    assert "loss" in loss
                    print(
                        f"OK  arch={arch}  head={head}  film={film}  "
                        f"pipe={spatial['spatial_pipeline']}  res={spatial['imaging_resolution']}  "
                        f"fp={spatial['footprint_mode']}  loss={float(loss['loss']):.4f}"
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
