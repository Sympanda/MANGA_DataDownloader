from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from manga_prep.targets.pipe3d_maps import AMARA_TARGET_KEYS

UpsampleMode = Literal["bilinear", "transpose", "pixel_shuffle"]
SpectrumInjection = Literal["bottleneck", "none"]

MODEL_PRESETS: dict[str, dict[str, int | float | str]] = {
    # Original v1 capacity (~1.9M params with SDSS+spectrum+footprint).
    "small": {
        "base_channels": 32,
        "bottleneck_multiplier": 8,
        "cond_dim": 256,
        "dropout": 0.0,
        "upsample_mode": "bilinear",
        "loss_mse_weight": 1.0,
        "loss_l1_weight": 0.0,
        "loss_grad_weight": 0.0,
    },
    # Recommended: more capacity + regularisation + sharper loss.
    # transpose can cause checkerboard/grid artifacts in outputs; bilinear + conv is safer.
    "medium": {
        "base_channels": 64,
        "bottleneck_multiplier": 16,
        "cond_dim": 384,
        "dropout": 0.1,
        "upsample_mode": "bilinear",
        "loss_mse_weight": 0.5,
        "loss_l1_weight": 0.5,
        "loss_grad_weight": 0.1,
    },
    "large": {
        "base_channels": 96,
        "bottleneck_multiplier": 16,
        "cond_dim": 512,
        "dropout": 0.15,
        "upsample_mode": "bilinear",
        "loss_mse_weight": 0.4,
        "loss_l1_weight": 0.5,
        "loss_grad_weight": 0.15,
    },
}


@dataclass
class ConditionalUNetConfig:
    """
    Flexible conditioning switches for the map generator.

    Images (SDSS / Legacy) are concatenated at the UNet input — no separate
    image encoder in v1. Spectrum is encoded to a vector and injected at the
    bottleneck via FiLM (scale + shift). Footprint mask is an optional extra
    input channel (where the IFU lives on the 76×76 canvas).
    """

    use_sdss: bool = True
    use_legacy: bool = False
    use_spectrum: bool = True
    use_footprint_mask: bool = True

    # "bottleneck" = FiLM on deepest feature map (simple, good first pass).
    spectrum_injection: SpectrumInjection = "bottleneck"

    n_sdss_bands: int = 5
    n_legacy_bands: int = 4
    n_target_maps: int = len(AMARA_TARGET_KEYS)
    target_keys: tuple[str, ...] = field(default_factory=lambda: AMARA_TARGET_KEYS)

    base_channels: int = 64
    bottleneck_multiplier: int = 16
    dropout: float = 0.1
    upsample_mode: UpsampleMode = "bilinear"
    cond_dim: int = 384
    spectrum_n_wave: int = 4563

    # Training loss weights (masked; sum need not be 1).
    loss_mse_weight: float = 0.5
    loss_l1_weight: float = 0.5
    loss_grad_weight: float = 0.1

    def input_channels(self) -> int:
        channels = 0
        if self.use_sdss:
            channels += self.n_sdss_bands
        if self.use_legacy:
            channels += self.n_legacy_bands
        if self.use_footprint_mask:
            channels += 1
        if channels == 0:
            raise ValueError("Enable at least one image modality or footprint mask.")
        return channels

    def validate(self) -> None:
        if not self.use_sdss and not self.use_legacy and not self.use_footprint_mask:
            raise ValueError("Need SDSS, Legacy, or footprint mask as spatial input.")
        if self.use_spectrum and self.spectrum_injection == "none":
            raise ValueError("use_spectrum=True requires spectrum_injection='bottleneck'.")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.bottleneck_multiplier not in (8, 16):
            raise ValueError("bottleneck_multiplier must be 8 or 16")
        if self.upsample_mode not in ("bilinear", "transpose", "pixel_shuffle"):
            raise ValueError(f"Unknown upsample_mode: {self.upsample_mode!r}")
