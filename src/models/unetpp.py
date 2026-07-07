from __future__ import annotations

import torch
import torch.nn as nn

from src.models.unet import ConvBlock, SpatialUpsample2x, UpsampleMode, nested_upsample_channel_counts


class UNetPPBackbone(nn.Module):
    """
    UNet++ style nested skip connections for 76×76 inputs.
    Dense nodes X_{i,j}: i = depth level, j = nested index.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        base_channels: int = 64,
        depth: int = 4,
        dropout: float = 0.0,
        upsample_mode: UpsampleMode = "bilinear",
        norm: str = "bn",
    ) -> None:
        super().__init__()
        self.depth = depth
        c = base_channels
        block_kw = {"dropout": dropout, "norm": norm, "residual": True}

        self.x00 = ConvBlock(in_channels, c, **block_kw)
        self._base_channels = c
        self.downs = nn.ModuleList(
            [
                nn.Sequential(nn.MaxPool2d(2), ConvBlock(c * (2**i), c * (2 ** (i + 1)), **block_kw))
                for i in range(depth)
            ]
        )

        self.nested: nn.ModuleDict = nn.ModuleDict()
        for i in range(1, depth + 1):
            for j in range(1, i + 1):
                in_ch = c * (2 ** (i - j)) + c * (2 ** (i - j + 1))
                out_ch = c * (2 ** (i - j))
                self.nested[f"x{i}{j}"] = ConvBlock(in_ch, out_ch, **block_kw)

        self._bottleneck_channels = c  # final nested node X_{L,L} has base_channels
        self.outc = nn.Conv2d(c, out_channels, kernel_size=1)
        self.upsample_mode = upsample_mode
        self.upsamplers = nn.ModuleDict(
            {
                str(ch): SpatialUpsample2x(ch, upsample_mode)
                for ch in nested_upsample_channel_counts(c, depth)
            }
        )

    @property
    def bottleneck_channels(self) -> int:
        return self._bottleneck_channels

    def _upsample(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return self.upsamplers[str(x.shape[1])](x, ref=ref)

    @property
    def encoder_level_channels(self) -> list[int]:
        """Channel width at each encoder spine node x_i0 (for multi-scale FiLM)."""
        c = self._base_channels
        return [c * (2**i) for i in range(self.depth + 1)]

    def _forward_nodes(self, x: torch.Tensor, bottleneck_fn=None) -> dict[str, torch.Tensor]:
        nodes: dict[str, torch.Tensor] = {}
        nodes["x00"] = self.x00(x)
        for i, down in enumerate(self.downs):
            nodes[f"x{i + 1}0"] = down(nodes[f"x{i}0"])

        for i in range(1, self.depth + 1):
            for j in range(1, i + 1):
                up = self._upsample(nodes[f"x{i}{j - 1}"], nodes[f"x{i - j}0"])
                cat = torch.cat([nodes[f"x{i - j}0"], up], dim=1)
                nodes[f"x{i}{j}"] = self.nested[f"x{i}{j}"](cat)

        key = f"x{self.depth}{self.depth}"
        if bottleneck_fn is not None:
            nodes[key] = bottleneck_fn(nodes[key])
        return nodes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nodes = self._forward_nodes(x)
        return self.outc(nodes[f"x{self.depth}{self.depth}"])

    def forward_with_stem_hook(self, x: torch.Tensor, stem_fn) -> torch.Tensor:
        h = stem_fn(self.x00(x))
        nodes: dict[str, torch.Tensor] = {"x00": h}
        for i, down in enumerate(self.downs):
            nodes[f"x{i + 1}0"] = down(nodes[f"x{i}0"])
        for i in range(1, self.depth + 1):
            for j in range(1, i + 1):
                up = self._upsample(nodes[f"x{i}{j - 1}"], nodes[f"x{i - j}0"])
                cat = torch.cat([nodes[f"x{i - j}0"], up], dim=1)
                nodes[f"x{i}{j}"] = self.nested[f"x{i}{j}"](cat)
        return self.outc(nodes[f"x{self.depth}{self.depth}"])

    def forward_with_bottleneck_hook(self, x: torch.Tensor, bottleneck_fn) -> torch.Tensor:
        nodes = self._forward_nodes(x, bottleneck_fn=bottleneck_fn)
        return self.outc(nodes[f"x{self.depth}{self.depth}"])

    def forward_with_encoder_hooks(self, x: torch.Tensor, encoder_hooks: list) -> torch.Tensor:
        """Apply FiLM (or other hooks) after each encoder spine block x_i0."""
        h = self.x00(x)
        if len(encoder_hooks) > 0 and encoder_hooks[0] is not None:
            h = encoder_hooks[0](h)
        nodes: dict[str, torch.Tensor] = {"x00": h}
        for i, down in enumerate(self.downs):
            h = down(nodes[f"x{i}0"])
            hook_idx = i + 1
            if hook_idx < len(encoder_hooks) and encoder_hooks[hook_idx] is not None:
                h = encoder_hooks[hook_idx](h)
            nodes[f"x{i + 1}0"] = h
        for i in range(1, self.depth + 1):
            for j in range(1, i + 1):
                up = self._upsample(nodes[f"x{i}{j - 1}"], nodes[f"x{i - j}0"])
                cat = torch.cat([nodes[f"x{i - j}0"], up], dim=1)
                nodes[f"x{i}{j}"] = self.nested[f"x{i}{j}"](cat)
        return self.outc(nodes[f"x{self.depth}{self.depth}"])
