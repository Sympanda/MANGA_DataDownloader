from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.config import ModelConfig
from src.models.encoders import CoarseFineHead, FiLM2d, SpectrumEncoder
from src.models.hr_pipeline import (
    CrossAttnHRBlock,
    FootprintFusion,
    GridProjector,
    HREncoder,
    HRLevelFusion,
)
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
        # Deep supervision uses external 1×1 heads on nested nodes; skip backbone outc.
        with_output_conv = not config.deep_supervision
        return UNetPPBackbone(**common, depth=config.n_down, with_output_conv=with_output_conv)
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
        use_gaussian = config.output_head == "gaussian"
        use_ds = config.deep_supervision
        if use_coarse_fine or use_ds:
            backbone_out = config.base_channels
        elif use_gaussian:
            backbone_out = 2 * config.n_target_maps
        else:
            backbone_out = config.n_target_maps

        self.hr_encoder: HREncoder | None = None
        self.hr_cross_encoder: HREncoder | None = None
        self.hr_cross_blocks: nn.ModuleDict | None = None
        self.grid_projector: GridProjector | None = None
        self.hr_fusions: nn.ModuleList | None = None
        self.footprint_fusion: FootprintFusion | None = None
        self._hr_pyramid: list[torch.Tensor] | None = None
        self._hr_cross_feat: torch.Tensor | None = None

        if config.use_hr_cross_attn:
            self.hr_cross_encoder = HREncoder(
                config.hr_imaging_channels(),
                base_channels=config.base_channels,
                n_down=config.hr_encoder_n_down,
                dropout=config.dropout,
                norm=config.norm,
            )
            unet_channels = self._encoder_level_channels_preview(config)
            blocks: dict[str, CrossAttnHRBlock] = {}
            for level in config.hr_cross_attn_levels:
                blocks[str(int(level))] = CrossAttnHRBlock(
                    unet_channels[int(level)],
                    self.hr_cross_encoder.out_channels,
                    num_heads=config.hr_attn_heads,
                    dropout=config.hr_attn_dropout,
                )
            self.hr_cross_blocks = nn.ModuleDict(blocks)

        if config.spatial_pipeline in ("hr_encoder", "hr_multiscale"):
            self.hr_encoder = HREncoder(
                config.imaging_input_channels(),
                base_channels=config.base_channels,
                n_down=config.n_down,
                dropout=config.dropout,
                norm=config.norm,
            )
            self.grid_projector = GridProjector(
                self.hr_encoder.level_channels[0]
                if config.spatial_pipeline == "hr_multiscale"
                else self.hr_encoder.out_channels,
                config.base_channels,
                mode=config.hr_project_mode,
                dropout=config.dropout,
                norm=config.norm,
            )
            if config.spatial_pipeline == "hr_multiscale":
                unet_channels = self._encoder_level_channels_preview(config)
                self.hr_fusions = nn.ModuleList(
                    [
                        HRLevelFusion(
                            hr_ch,
                            unet_ch,
                            dropout=config.dropout,
                            norm=config.norm,
                        )
                        for hr_ch, unet_ch in zip(self.hr_encoder.level_channels, unet_channels)
                    ]
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

        self.ds_heads: nn.ModuleList | None = None
        if use_ds:
            self.ds_heads = nn.ModuleList(
                [
                    nn.Conv2d(config.base_channels, config.n_target_maps, kernel_size=1)
                    for _ in range(config.n_down)
                ]
            )

        self.spectrum_encoder: SpectrumEncoder | None = None
        self.bottleneck_film: FiLM2d | None = None
        self.encoder_film: nn.ModuleList | None = None

        if config.use_spectrum and config.film_injection != "none":
            self.spectrum_encoder = SpectrumEncoder(
                n_wave=config.spectrum_n_wave,
                out_dim=config.cond_dim,
                in_channels=config.spectrum_input_channels(),
                pooling=config.spectrum_pooling,
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

    @staticmethod
    def _encoder_level_channels_preview(config: ModelConfig) -> list[int]:
        """Channel widths of backbone encoder spine (matches UNet / UNet++)."""
        c = config.base_channels
        if config.architecture == "unetpp":
            return [c * (2**i) for i in range(config.n_down + 1)]
        m = config.bottleneck_multiplier
        levels = [c]
        for i in range(config.n_down):
            levels.append(min(c * (2 ** (i + 1)), c * m))
        return levels

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
        self._hr_pyramid = None

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

        if cfg.spatial_pipeline == "hr_multiscale":
            assert (
                self.hr_encoder is not None
                and self.grid_projector is not None
                and self.hr_fusions is not None
            )
            target_size = self._target_size(footprint)
            pyramid = self.hr_encoder.forward_pyramid(x_imaging)
            self._hr_pyramid = pyramid
            # Stem HR features → target-grid UNet++ input (detail still encoded above).
            features = self.grid_projector(pyramid[0], target_size)
            if self.footprint_fusion is not None and footprint is not None:
                features = self.footprint_fusion(features, footprint)
            return features

        if cfg.spatial_pipeline == "hr_full":
            return x_imaging

        raise ValueError(f"Unknown spatial_pipeline: {cfg.spatial_pipeline!r}")

    def _hr_level_fusions(self) -> list | None:
        if self.hr_cross_blocks is not None and self._hr_cross_feat is not None:
            hr = self._hr_cross_feat
            n = self.config.n_down + 1
            fusions: list = [None] * n
            for key, block in self.hr_cross_blocks.items():
                level = int(key)

                def _fn(unet_h: torch.Tensor, block=block, hr=hr) -> torch.Tensor:
                    return block(unet_h, hr)

                fusions[level] = _fn
            return fusions

        if self.hr_fusions is None or self._hr_pyramid is None:
            return None
        pyramid = self._hr_pyramid

        def _make(fuse: HRLevelFusion, hr: torch.Tensor):
            def _fn(unet_h: torch.Tensor, fuse=fuse, hr=hr) -> torch.Tensor:
                return fuse(unet_h, hr)

            return _fn

        return [_make(fuse, hr) for fuse, hr in zip(self.hr_fusions, pyramid)]

    def _film_hooks(
        self, cond: torch.Tensor | None
    ) -> tuple[list | None, object | None]:
        """Return (encoder_hooks, bottleneck_fn) for the current FiLM mode."""
        if cond is None:
            return None, None
        if self.encoder_film is not None:
            hooks = [lambda h, film=film, c=cond: film(h, c) for film in self.encoder_film]
            return hooks, None
        if self.bottleneck_film is not None:
            film = self.bottleneck_film

            def bottleneck_fn(b: torch.Tensor) -> torch.Tensor:
                return film(b, cond)

            return None, bottleneck_fn
        return None, None

    def _run_backbone(self, x_spatial: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        encoder_hooks, bottleneck_fn = self._film_hooks(cond)
        level_fusions = self._hr_level_fusions()
        if encoder_hooks is not None or level_fusions is not None:
            hooks = encoder_hooks or [None] * (self.config.n_down + 1)
            if hasattr(self.backbone, "forward_with_encoder_hooks"):
                return self.backbone.forward_with_encoder_hooks(
                    x_spatial,
                    hooks,
                    bottleneck_fn=bottleneck_fn,
                    level_fusions=level_fusions,
                )
            raise TypeError(f"{type(self.backbone).__name__} lacks forward_with_encoder_hooks")
        if bottleneck_fn is not None:
            return self.backbone.forward_with_bottleneck_hook(x_spatial, bottleneck_fn)
        return self.backbone(x_spatial)

    def _run_backbone_nodes(
        self, x_spatial: torch.Tensor, cond: torch.Tensor | None
    ) -> dict[str, torch.Tensor]:
        assert isinstance(self.backbone, UNetPPBackbone)
        encoder_hooks, bottleneck_fn = self._film_hooks(cond)
        return self.backbone.forward_nodes(
            x_spatial,
            encoder_hooks=encoder_hooks,
            bottleneck_fn=bottleneck_fn,
            level_fusions=self._hr_level_fusions(),
        )

    def _split_gaussian_features(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        channels = self.config.n_target_maps
        mu = features[:, :channels]
        log_var = features[:, channels : 2 * channels]
        return mu, log_var

    def _apply_output_head(
        self,
        features: torch.Tensor,
        *,
        detail_scale_multiplier: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if self.config.output_head == "gaussian":
            mu, _log_var = self._split_gaussian_features(features)
            return mu, None, None
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

    def _finalize_gaussian_output(
        self,
        features: torch.Tensor,
        footprint: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mu, log_var = self._split_gaussian_features(features)
        target_size = self._target_size(footprint)
        maps = self._resize_to_target(mu, footprint)
        if log_var.shape[-2:] != target_size:
            log_var = F.interpolate(log_var, size=target_size, mode="bilinear", align_corners=False)
        log_var_clamped = log_var.clamp(
            min=self.config.loss_params.get("gaussian_nll", {}).get("min_log_var", -6.0),
            max=self.config.loss_params.get("gaussian_nll", {}).get("max_log_var", 6.0),
        )
        aux = {
            "log_var": log_var,
            "sigma": torch.exp(0.5 * log_var_clamped) + 1e-4,
        }
        return maps, aux

    def _forward_deep_supervision(
        self,
        x_backbone: torch.Tensor,
        cond: torch.Tensor | None,
        footprint: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        assert self.ds_heads is not None and isinstance(self.backbone, UNetPPBackbone)
        nodes = self._run_backbone_nodes(x_backbone, cond)
        keys = self.backbone.deep_supervision_keys()
        target_size = self._target_size(footprint)
        deep_maps: list[torch.Tensor] = []
        for head, key in zip(self.ds_heads, keys):
            feat = nodes[key]
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            if (
                self.config.spatial_pipeline == "hr_full"
                and self.footprint_fusion is not None
                and footprint is not None
            ):
                feat = self.footprint_fusion(feat, footprint)
            deep_maps.append(head(feat))

        maps = deep_maps[-1]
        aux: dict[str, torch.Tensor] = {
            "deep_maps": torch.stack(deep_maps, dim=0),
        }
        for i, pred in enumerate(deep_maps):
            aux[f"ds_{i}"] = pred
        return maps, aux

    def forward(
        self,
        x_imaging: torch.Tensor,
        *,
        spectrum: torch.Tensor | None = None,
        spectrum_flux: torch.Tensor | None = None,
        footprint: torch.Tensor | None = None,
        x_hr: torch.Tensor | None = None,
        detail_scale_multiplier: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        spec = spectrum if spectrum is not None else spectrum_flux
        cond = None
        if self.spectrum_encoder is not None and spec is not None:
            cond = self.spectrum_encoder(spec)

        self._hr_cross_feat = None
        if self.config.use_hr_cross_attn:
            if x_hr is None:
                raise ValueError("use_hr_cross_attn=true requires x_hr in forward()")
            assert self.hr_cross_encoder is not None
            self._hr_cross_feat = self.hr_cross_encoder(x_hr)

        x_backbone = self._prepare_backbone_input(x_imaging, footprint)

        if self.ds_heads is not None:
            return self._forward_deep_supervision(x_backbone, cond, footprint)

        features = self._run_backbone(x_backbone, cond)
        aux: dict[str, torch.Tensor] = {}

        if self.config.spatial_pipeline == "hr_full":
            target_size = self._target_size(footprint)
            features = F.interpolate(features, size=target_size, mode="bilinear", align_corners=False)
            if self.footprint_fusion is not None and footprint is not None:
                features = self.footprint_fusion(features, footprint)
            if self.config.output_head == "gaussian":
                return self._finalize_gaussian_output(features, footprint)
            maps, coarse, residual = self._apply_output_head(
                features,
                detail_scale_multiplier=detail_scale_multiplier,
            )
            if coarse is not None:
                aux["coarse"] = coarse
            if residual is not None:
                aux["residual"] = residual
            return maps, aux

        if self.config.output_head == "gaussian":
            return self._finalize_gaussian_output(features, footprint)

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
