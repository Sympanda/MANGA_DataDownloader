from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.config import ModelConfig, effective_detail_scale_multiplier
from src.models.encoders import CoarseFineHead, FiLM2d, SpectrumEncoder
from src.models.hr_pipeline import FootprintFusion, GridProjector, HREncoder
from src.models.unet import UNetBackbone
from src.models.unetpp import UNetPPBackbone


def build_backbone(
    config: ModelConfig,
    *,
    in_channels: int | None = None,
    feature_channels: int | None = None,
) -> nn.Module:
    out_ch = feature_channels or config.n_target_maps
    common = dict(
        in_channels=in_channels if in_channels is not None else config.backbone_input_channels(),
        out_channels=out_ch,
        base_channels=config.base_channels,
        dropout=config.dropout,
        upsample_mode=config.upsample_mode,
        norm=config.norm,
    )
    if config.architecture == "unetpp":
        return UNetPPBackbone(**common, depth=config.n_down)
    return UNetBackbone(
        **common,
        bottleneck_multiplier=config.bottleneck_multiplier,
        n_down=config.n_down,
        residual=config.residual_blocks,
    )


class ConditionalMapModel(nn.Module):
    """Imaging (+ spectrum FiLM) -> Amara map prediction with optional HR front-ends."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        use_coarse_fine = config.output_head == "coarse_fine"
        backbone_out = config.base_channels if use_coarse_fine else config.n_target_maps

        self.hr_encoder: HREncoder | None = None
        self.grid_projector: GridProjector | None = None
        self.footprint_fusion: FootprintFusion | None = None

        if config.spatial_pipeline == "hr_encoder":
            self.hr_encoder = HREncoder(
                config.imaging_input_channels(),
                base_channels=config.base_channels,
                n_down=config.n_down,
                dropout=config.dropout,
                norm=config.norm,
            )
            self.grid_projector = GridProjector(
                self.hr_encoder.out_channels,
                config.base_channels,
                mode=config.hr_project_mode,
                dropout=config.dropout,
                norm=config.norm,
            )
            if config.footprint_mode == "fusion_concat":
                self.footprint_fusion = FootprintFusion(
                    config.base_channels,
                    dropout=config.dropout,
                    norm=config.norm,
                )
        elif config.spatial_pipeline == "hr_full" and config.footprint_mode == "fusion_concat":
            self.footprint_fusion = FootprintFusion(
                backbone_out,
                dropout=config.dropout,
                norm=config.norm,
            )

        self.backbone = build_backbone(config, feature_channels=backbone_out)
        self.output_head: CoarseFineHead | None = None
        if use_coarse_fine:
            self.output_head = CoarseFineHead(
                config.base_channels,
                config.n_target_maps,
                coarse_factor=config.coarse_factor,
                detail_scale_init=config.detail_scale_init,
            )

        self.spectrum_encoder: SpectrumEncoder | None = None
        self.bottleneck_film: FiLM2d | None = None
        self.encoder_film: nn.ModuleList | None = None

        if config.use_spectrum and config.film_injection != "none":
            self.spectrum_encoder = SpectrumEncoder(
                n_wave=config.spectrum_n_wave,
                out_dim=config.cond_dim,
            )
            if config.film_injection == "bottleneck":
                self.bottleneck_film = FiLM2d(
                    config.cond_dim,
                    self.backbone.bottleneck_channels,
                )
            elif config.film_injection == "encoder":
                level_channels = self.backbone.encoder_level_channels
                self.encoder_film = nn.ModuleList(
                    FiLM2d(config.cond_dim, ch) for ch in level_channels
                )

    def _target_size(self, footprint: torch.Tensor | None) -> tuple[int, int]:
        if footprint is not None:
            return int(footprint.shape[-2]), int(footprint.shape[-1])
        size = self.config.target_spatial_size
        return size, size

    def _prepare_backbone_input(
        self,
        x_imaging: torch.Tensor,
        footprint: torch.Tensor | None,
    ) -> torch.Tensor:
        cfg = self.config
        if cfg.spatial_pipeline == "symmetric":
            if cfg.footprint_mode == "spatial_channel" and footprint is not None:
                if footprint.ndim == x_imaging.ndim - 1:
                    footprint = footprint.unsqueeze(1)
                return torch.cat([x_imaging, footprint.float()], dim=1)
            return x_imaging

        if cfg.spatial_pipeline == "hr_encoder":
            assert self.hr_encoder is not None and self.grid_projector is not None
            target_size = self._target_size(footprint)
            features = self.hr_encoder(x_imaging)
            features = self.grid_projector(features, target_size)
            if self.footprint_fusion is not None and footprint is not None:
                features = self.footprint_fusion(features, footprint)
            return features

        if cfg.spatial_pipeline == "hr_full":
            return x_imaging

        raise ValueError(f"Unknown spatial_pipeline: {cfg.spatial_pipeline!r}")

    def _run_backbone(self, x_spatial: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        if cond is None:
            return self.backbone(x_spatial)

        if self.encoder_film is not None:
            hooks = [lambda h, film=film, c=cond: film(h, c) for film in self.encoder_film]
            if hasattr(self.backbone, "forward_with_encoder_hooks"):
                return self.backbone.forward_with_encoder_hooks(x_spatial, hooks)
            raise TypeError(f"{type(self.backbone).__name__} lacks forward_with_encoder_hooks")

        if self.bottleneck_film is not None:
            film = self.bottleneck_film

            def bottleneck_fn(b: torch.Tensor) -> torch.Tensor:
                return film(b, cond)

            return self.backbone.forward_with_bottleneck_hook(x_spatial, bottleneck_fn)

        return self.backbone(x_spatial)

    def _apply_output_head(
        self,
        features: torch.Tensor,
        *,
        detail_scale_multiplier: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if self.output_head is not None:
            maps, coarse, residual = self.output_head(
                features,
                detail_scale_multiplier=detail_scale_multiplier,
            )
            return maps, coarse, residual
        return features, None, None

    def _resize_to_target(
        self,
        maps: torch.Tensor,
        footprint: torch.Tensor | None,
    ) -> torch.Tensor:
        target_size = self._target_size(footprint)
        if maps.shape[-2:] == target_size:
            return maps
        return F.interpolate(maps, size=target_size, mode="bilinear", align_corners=False)

    def forward(
        self,
        x_imaging: torch.Tensor,
        *,
        spectrum_flux: torch.Tensor | None = None,
        footprint: torch.Tensor | None = None,
        detail_scale_multiplier: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cond = None
        if self.spectrum_encoder is not None and spectrum_flux is not None:
            cond = self.spectrum_encoder(spectrum_flux)

        x_backbone = self._prepare_backbone_input(x_imaging, footprint)
        features = self._run_backbone(x_backbone, cond)
        aux: dict[str, torch.Tensor] = {}

        if self.config.spatial_pipeline == "hr_full":
            target_size = self._target_size(footprint)
            features = F.interpolate(features, size=target_size, mode="bilinear", align_corners=False)
            if self.footprint_fusion is not None and footprint is not None:
                features = self.footprint_fusion(features, footprint)
            maps, coarse, residual = self._apply_output_head(
                features,
                detail_scale_multiplier=detail_scale_multiplier,
            )
            if coarse is not None:
                aux["coarse"] = coarse
            if residual is not None:
                aux["residual"] = residual
            return maps, aux

        maps, coarse, residual = self._apply_output_head(
            features,
            detail_scale_multiplier=detail_scale_multiplier,
        )
        if coarse is not None:
            aux["coarse"] = coarse
        if residual is not None:
            aux["residual"] = residual
        maps = self._resize_to_target(maps, footprint)
        return maps, aux
