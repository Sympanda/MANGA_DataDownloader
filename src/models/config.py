from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from manga_prep.targets.pipe3d_maps import AMARA_TARGET_KEYS, DEFAULT_TARGET_SIZE

ArchitectureType = Literal["unet", "unetpp"]
OutputHeadType = Literal["single", "coarse_fine", "gaussian"]
FilmInjection = Literal["none", "bottleneck", "encoder"]
UpsampleMode = Literal["bilinear", "transpose", "pixel_shuffle"]
ImagingResolution = Literal["aligned", "native"]
SpatialPipeline = Literal["symmetric", "hr_encoder", "hr_full", "hr_multiscale"]
FootprintMode = Literal["spatial_channel", "fusion_concat", "loss_only"]
HRProjectMode = Literal["bilinear", "learned"]
SpectrumPooling = Literal["avg", "attention"]
InputNormMode = Literal["none", "asinh"]
HRSurvey = Literal["sdss", "legacy"]
HRAttentionMode = Literal["local", "global"]


@dataclass
class ModelConfig:
    architecture: ArchitectureType = "unet"
    output_head: OutputHeadType = "single"

    use_sdss: bool = True
    use_legacy: bool = False
    use_spectrum: bool = True
    use_redshift_cond: bool = False
    use_footprint_mask: bool = True

    n_sdss_bands: int = 5
    n_legacy_bands: int = 4
    n_target_maps: int = len(AMARA_TARGET_KEYS)
    target_keys: tuple[str, ...] = field(default_factory=lambda: AMARA_TARGET_KEYS)

    def sync_target_maps(self) -> None:
        """Keep ``n_target_maps`` aligned with ``target_keys`` length."""
        self.n_target_maps = len(self.target_keys)

    # Spatial input / output pipeline (swap via config without code changes).
    imaging_resolution: ImagingResolution = "aligned"
    spatial_pipeline: SpatialPipeline = "symmetric"
    footprint_mode: FootprintMode = "spatial_channel"
    target_spatial_size: int = DEFAULT_TARGET_SIZE
    hr_project_mode: HRProjectMode = "bilinear"

    # High-res morphology via cross-attention (side stream; backbone stays 76×76 aligned).
    use_hr_cross_attn: bool = False
    hr_survey: HRSurvey = "sdss"  # "sdss" → native ~196; "legacy" → Legacy HR
    # Default: level 1 (38×38) only — dense attn at 76×76 OOMs; local window is cheap.
    hr_cross_attn_levels: tuple[int, ...] = (1,)
    hr_encoder_n_down: int = 3  # 196 → ~24 token grid at deepest level
    hr_attn_heads: int = 4
    hr_attn_dropout: float = 0.0
    hr_attention_mode: HRAttentionMode = "local"
    hr_attention_window: int = 7  # odd; K = window² local HR tokens per query

    imaging_clamp_min: float | None = -5.0
    imaging_clamp_max: float | None = 100.0

    # Input soft-normalization: asinh(f / s_b) with train-split percentile scales.
    # scales_path is resolved in runner → imaging_asinh_scales / spectrum_* filled.
    input_norm_mode: InputNormMode = "none"
    input_norm_scales_path: str | None = None
    input_norm_imaging_percentile: float = 99.0
    input_norm_spectrum_percentile: float = 99.0
    # Channel order matches imaging concat: SDSS ugriz then Legacy (if enabled).
    imaging_asinh_scales: list[float] | None = None
    # Separate HR stream scales (same physical units as the chosen hr_survey).
    hr_asinh_scales: list[float] | None = None
    spectrum_asinh_scale_fake: float | None = None
    spectrum_asinh_scale_real: float | None = None

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
    # Spectrum encoder (Task 3): pooling + optional λ / ivar channels.
    spectrum_pooling: SpectrumPooling = "attention"
    spectrum_use_wavelength: bool = True
    spectrum_use_ivar: bool = True
    # Fixed wavelength range for λ_norm (matches manga_prep default grid).
    spectrum_wave_min: float = 3622.0
    spectrum_wave_max: float = 10354.0

    # UNet++ deep supervision (Zhou et al.): 1×1 heads on full-res nested nodes.
    # Preferred over coarse_fine for multi-scale fidelity on this architecture.
    deep_supervision: bool = False
    # Per-level weights for auxiliary heads (deepest uses full loss).
    # None → linear ramp (1/L … (L-1)/L). Length must be n_down - 1 when set.
    deep_supervision_weights: list[float] | None = None
    # Aux-head loss (full loss stack stays on deepest).
    deep_supervision_loss: Literal["l1", "mse", "charbonnier", "grad", "laplacian"] = "l1"

    coarse_factor: int = 2
    detail_scale_init: float = 0.1
    detail_scale_schedule: dict[str, float | int] | None = None

    losses: list[str] = field(
        default_factory=lambda: ["charbonnier", "grad", "integration"]
    )
    loss_weights: list[float] = field(default_factory=lambda: [1.0, 0.1, 0.05])
    loss_params: dict[str, dict] = field(default_factory=lambda: {"charbonnier": {"eps": 1e-3}})

    def spectrum_input_channels(self) -> int:
        channels = 1  # flux
        if self.spectrum_use_wavelength:
            channels += 1
        if self.spectrum_use_ivar:
            channels += 1
        return channels

    def hr_imaging_channels(self) -> int:
        if self.hr_survey == "sdss":
            return self.n_sdss_bands
        if self.hr_survey == "legacy":
            return self.n_legacy_bands
        raise ValueError(f"Unknown hr_survey: {self.hr_survey!r}")

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
        if self.spatial_pipeline in ("hr_encoder", "hr_multiscale"):
            return self.base_channels
        if self.spatial_pipeline == "hr_full":
            return self.imaging_input_channels()
        raise ValueError(f"Unknown spatial_pipeline: {self.spatial_pipeline!r}")

    def input_channels(self) -> int:
        """Backwards-compatible alias for backbone input width."""
        return self.backbone_input_channels()

    def validate(self) -> None:
        if len(self.target_keys) == 0:
            raise ValueError("target_keys must be non-empty")
        if self.n_target_maps != len(self.target_keys):
            raise ValueError(
                f"n_target_maps={self.n_target_maps} != len(target_keys)={len(self.target_keys)}"
            )
        if self.bottleneck_multiplier not in (8, 16):
            raise ValueError("bottleneck_multiplier must be 8 or 16")
        active = [w for w in self.loss_weights if w > 0]
        if not active:
            raise ValueError("At least one loss weight must be > 0")
        if self.spatial_pipeline == "symmetric" and self.imaging_resolution == "native":
            raise ValueError(
                "spatial_pipeline='symmetric' requires imaging_resolution='aligned'. "
                "Use spatial_pipeline='hr_encoder', 'hr_multiscale', or 'hr_full' with native SDSS, "
                "or use_hr_cross_attn=true with aligned 76×76 backbone."
            )
        if self.use_hr_cross_attn:
            if self.spatial_pipeline != "symmetric":
                raise ValueError(
                    "use_hr_cross_attn requires spatial_pipeline='symmetric' "
                    "(76×76 aligned backbone + HR side-stream cross-attention)."
                )
            if self.imaging_resolution != "aligned":
                raise ValueError(
                    "use_hr_cross_attn requires imaging_resolution='aligned' "
                    "for the UNet++ backbone; HR is loaded separately."
                )
            levels = tuple(int(i) for i in self.hr_cross_attn_levels)
            if not levels:
                raise ValueError("hr_cross_attn_levels must be non-empty when use_hr_cross_attn=true")
            if any(i < 0 or i > self.n_down for i in levels):
                raise ValueError(
                    f"hr_cross_attn_levels must be in [0, n_down={self.n_down}], got {levels}"
                )
            if self.hr_encoder_n_down < 1:
                raise ValueError("hr_encoder_n_down must be >= 1")
            if self.hr_attention_mode not in ("local", "global"):
                raise ValueError(
                    f"hr_attention_mode must be 'local' or 'global', got {self.hr_attention_mode!r}"
                )
            if self.hr_attention_mode == "local":
                w = int(self.hr_attention_window)
                if w < 1 or w % 2 == 0:
                    raise ValueError(
                        f"hr_attention_window must be a positive odd integer, got {w}"
                    )
        if self.spatial_pipeline == "hr_multiscale" and self.imaging_resolution != "native":
            raise ValueError(
                "spatial_pipeline='hr_multiscale' requires imaging_resolution='native' "
                "so SDSS is encoded before the 76×76 target grid."
            )
        if self.footprint_mode == "spatial_channel" and self.spatial_pipeline != "symmetric":
            raise ValueError(
                "footprint_mode='spatial_channel' only applies to spatial_pipeline='symmetric'. "
                "Use footprint_mode='fusion_concat' or 'loss_only' for HR pipelines."
            )
        if self.footprint_mode == "fusion_concat" and not self.use_footprint_mask:
            raise ValueError("footprint_mode='fusion_concat' requires use_footprint_mask=true")
        if self.deep_supervision:
            if self.architecture != "unetpp":
                raise ValueError("deep_supervision requires architecture='unetpp'")
            if self.output_head != "single":
                raise ValueError(
                    "deep_supervision requires output_head='single' "
                    "(prefer UNet++ DS over coarse_fine / gaussian heads)."
                )
            if self.deep_supervision_weights is not None:
                expected = self.n_down - 1
                if len(self.deep_supervision_weights) != expected:
                    raise ValueError(
                        f"deep_supervision_weights must have length n_down-1={expected}, "
                        f"got {len(self.deep_supervision_weights)}"
                    )
        if self.film_injection != "none" and not (self.use_spectrum or self.use_redshift_cond):
            raise ValueError(
                "film_injection != 'none' requires use_spectrum and/or use_redshift_cond"
            )
        if self.input_norm_mode == "asinh":
            if self.imaging_asinh_scales is None:
                raise ValueError(
                    "input_norm_mode='asinh' requires imaging_asinh_scales "
                    "(load via input_norm.scales_path in config / runner)."
                )
            n_img = self.imaging_input_channels()
            if len(self.imaging_asinh_scales) != n_img:
                raise ValueError(
                    f"imaging_asinh_scales length {len(self.imaging_asinh_scales)} "
                    f"!= imaging channels {n_img}"
                )
            if any(float(s) <= 0 for s in self.imaging_asinh_scales):
                raise ValueError("imaging_asinh_scales must be > 0")
            if self.use_hr_cross_attn:
                if self.hr_asinh_scales is None:
                    raise ValueError(
                        "input_norm_mode='asinh' with use_hr_cross_attn requires hr_asinh_scales"
                    )
                if len(self.hr_asinh_scales) != self.hr_imaging_channels():
                    raise ValueError(
                        f"hr_asinh_scales length {len(self.hr_asinh_scales)} "
                        f"!= hr channels {self.hr_imaging_channels()}"
                    )
                if any(float(s) <= 0 for s in self.hr_asinh_scales):
                    raise ValueError("hr_asinh_scales must be > 0")
            if self.use_spectrum:
                if self.spectrum_asinh_scale_fake is None or self.spectrum_asinh_scale_real is None:
                    raise ValueError(
                        "input_norm_mode='asinh' with use_spectrum requires "
                        "spectrum_asinh_scale_fake and spectrum_asinh_scale_real"
                    )
                if float(self.spectrum_asinh_scale_fake) <= 0 or float(self.spectrum_asinh_scale_real) <= 0:
                    raise ValueError("spectrum asinh scales must be > 0")

    def resolved_deep_supervision_weights(self) -> list[float]:
        """Weights for auxiliary DS heads (excludes deepest, which uses the full loss)."""
        n_aux = self.n_down - 1
        if n_aux <= 0:
            return []
        if self.deep_supervision_weights is not None:
            return [float(w) for w in self.deep_supervision_weights]
        depth = self.n_down
        return [(i + 1) / depth for i in range(n_aux)]


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
