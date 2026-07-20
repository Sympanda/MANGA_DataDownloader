from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from manga_prep.targets.pipe3d_maps import AMARA_TARGET_KEYS, DEFAULT_TARGET_SIZE

ArchitectureType = Literal["unet", "unetpp"]
OutputHeadType = Literal["single", "coarse_fine", "gaussian"]
FilmInjection = Literal["none", "bottleneck", "encoder"]
UpsampleMode = Literal["bilinear", "transpose", "pixel_shuffle"]
ImagingResolution = Literal["aligned", "native"]
SpatialPipeline = Literal["symmetric", "hr_encoder", "hr_full"]
FootprintMode = Literal["spatial_channel", "fusion_concat", "loss_only"]
HRProjectMode = Literal["bilinear", "learned"]


@dataclass
class ModelConfig:
    architecture: ArchitectureType = "unet"
    output_head: OutputHeadType = "single"

    use_sdss: bool = True
    use_legacy: bool = False
    use_spectrum: bool = True
    use_footprint_mask: bool = True

    n_sdss_bands: int = 5
    n_legacy_bands: int = 4
    n_target_maps: int = len(AMARA_TARGET_KEYS)
    target_keys: tuple[str, ...] = field(default_factory=lambda: AMARA_TARGET_KEYS)

    # Spatial input / output pipeline (swap via config without code changes).
    imaging_resolution: ImagingResolution = "aligned"
    spatial_pipeline: SpatialPipeline = "symmetric"
    footprint_mode: FootprintMode = "spatial_channel"
    target_spatial_size: int = DEFAULT_TARGET_SIZE
    hr_project_mode: HRProjectMode = "bilinear"

    imaging_clamp_min: float | None = -5.0
    imaging_clamp_max: float | None = 100.0

    base_channels: int = 64
    bottleneck_multiplier: int = 16
    n_down: int = 4
    dropout: float = 0.1
    upsample_mode: UpsampleMode = "bilinear"
    norm: str = "gn"
    residual_blocks: bool = True

    cond_dim: int = 384
    spectrum_n_wave: int = 4563
    film_injection: FilmInjection = "bottleneck"

    coarse_factor: int = 2
    detail_scale_init: float = 0.1
    detail_scale_schedule: dict[str, float | int] | None = None

    losses: list[str] = field(
        default_factory=lambda: ["charbonnier", "grad", "integration"]
    )
    loss_weights: list[float] = field(default_factory=lambda: [1.0, 0.1, 0.05])
    loss_params: dict[str, dict] = field(default_factory=lambda: {"charbonnier": {"eps": 1e-3}})

    def imaging_input_channels(self) -> int:
        channels = 0
        if self.use_sdss:
            channels += self.n_sdss_bands
        if self.use_legacy:
            channels += self.n_legacy_bands
        if channels == 0:
            raise ValueError("Enable at least one spatial imaging modality.")
        return channels

    def uses_footprint_in_model(self) -> bool:
        return self.use_footprint_mask and self.footprint_mode != "loss_only"

    def backbone_input_channels(self) -> int:
        if self.spatial_pipeline == "symmetric":
            channels = self.imaging_input_channels()
            if self.footprint_mode == "spatial_channel" and self.uses_footprint_in_model():
                channels += 1
            return channels
        if self.spatial_pipeline == "hr_encoder":
            return self.base_channels
        if self.spatial_pipeline == "hr_full":
            return self.imaging_input_channels()
        raise ValueError(f"Unknown spatial_pipeline: {self.spatial_pipeline!r}")

    def input_channels(self) -> int:
        """Backwards-compatible alias for backbone input width."""
        return self.backbone_input_channels()

    def validate(self) -> None:
        if self.bottleneck_multiplier not in (8, 16):
            raise ValueError("bottleneck_multiplier must be 8 or 16")
        active = [w for w in self.loss_weights if w > 0]
        if not active:
            raise ValueError("At least one loss weight must be > 0")
        if self.spatial_pipeline == "symmetric" and self.imaging_resolution == "native":
            raise ValueError(
                "spatial_pipeline='symmetric' requires imaging_resolution='aligned'. "
                "Use spatial_pipeline='hr_encoder' or 'hr_full' with native SDSS."
            )
        if self.footprint_mode == "spatial_channel" and self.spatial_pipeline != "symmetric":
            raise ValueError(
                "footprint_mode='spatial_channel' only applies to spatial_pipeline='symmetric'. "
                "Use footprint_mode='fusion_concat' or 'loss_only' for HR pipelines."
            )
        if self.footprint_mode == "fusion_concat" and not self.use_footprint_mask:
            raise ValueError("footprint_mode='fusion_concat' requires use_footprint_mask=true")


def effective_detail_scale_multiplier(
    config: ModelConfig,
    epoch: int | None,
) -> float:
    """
    Optional ramp for the coarse/fine residual branch during training.

    Returns a multiplier in [0, 1] applied to detail_scale_init. When epoch is
    None (eval) or no schedule is set, returns 1.0.
    """
    if epoch is None or config.output_head != "coarse_fine":
        return 1.0
    sched = config.detail_scale_schedule
    if not sched:
        return 1.0
    warmup = int(sched.get("warmup_epochs", 0))
    ramp = int(sched.get("ramp_epochs", 0))
    start = float(sched.get("start", 0.0))
    end = float(sched.get("end", 1.0))
    if epoch <= warmup:
        return start
    if ramp <= 0:
        return end
    t = min(1.0, (epoch - warmup) / ramp)
    return start + t * (end - start)
