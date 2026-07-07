from __future__ import annotations

import torch
import torch.nn as nn

from manga_models.config import ConditionalUNetConfig
from manga_models.encoders import FiLM2d, SpectrumEncoder
from manga_models.unet import UNetBackbone


class ConditionalMapUNet(nn.Module):
    """
    Conditional generator: imaging (+ optional footprint) + spectrum -> Amara maps.

    v1 design (simple):
    - SDSS / Legacy / footprint: channel-concat at UNet input (raw aligned flux).
    - Spectrum: 1D CNN -> FiLM at bottleneck only.
    """

    def __init__(self, config: ConditionalUNetConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        self.unet = UNetBackbone(
            in_channels=config.input_channels(),
            out_channels=config.n_target_maps,
            base_channels=config.base_channels,
            bottleneck_multiplier=config.bottleneck_multiplier,
            dropout=config.dropout,
            upsample_mode=config.upsample_mode,
        )

        self.spectrum_encoder: SpectrumEncoder | None = None
        self.bottleneck_film: FiLM2d | None = None
        if config.use_spectrum and config.spectrum_injection == "bottleneck":
            self.spectrum_encoder = SpectrumEncoder(
                n_wave=config.spectrum_n_wave,
                out_dim=config.cond_dim,
            )
            self.bottleneck_film = FiLM2d(
                config.cond_dim,
                self.unet.bottleneck_channels,
            )

    def forward(
        self,
        x_spatial: torch.Tensor,
        *,
        spectrum_flux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x_spatial : (B, C_in, H, W)
            Stacked SDSS/Legacy bands + optional footprint mask channel.
        spectrum_flux : (B, n_wave) or None
        """
        if self.spectrum_encoder is None or spectrum_flux is None:
            return self.unet(x_spatial)

        cond = self.spectrum_encoder(spectrum_flux)
        assert self.bottleneck_film is not None

        def apply_film(bottleneck: torch.Tensor) -> torch.Tensor:
            return self.bottleneck_film(bottleneck, cond)

        return self.unet.forward_with_bottleneck_hook(x_spatial, apply_film)
