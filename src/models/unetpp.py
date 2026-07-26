from __future__ import annotations

import torch
import torch.nn as nn

from src.models.unet import ConvBlock, SpatialUpsample2x, UpsampleMode, nested_upsample_channel_counts


class UNetPPBackbone(nn.Module):
    """
    UNet++ with dense nested skip connections (Zhou et al.).

    Node X^{i,j} (key ``x{i}{j}``):
      - i = encoder level (0 = full resolution)
      - j = nested index along the skip pathway
      - X^{i,0} = encoder spine
      - X^{i,j} = H([X^{i,0}, …, X^{i,j-1}, U(X^{i+1,j-1})])

    Deep-supervision / final output sites are the full-resolution nodes
    X^{0,1} … X^{0,L} (keys x01…x0L).
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
        with_output_conv: bool = True,
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

        # Dense nested nodes X^{i,j} for j>=1.
        self.nested: nn.ModuleDict = nn.ModuleDict()
        for j in range(1, depth + 1):
            for i in range(depth - j + 1):
                # Concat: j prior nodes at level i (each c*2^i) + upsampled deeper node (c*2^{i+1})
                level_ch = c * (2**i)
                in_ch = j * level_ch + c * (2 ** (i + 1))
                self.nested[f"x{i}{j}"] = ConvBlock(in_ch, level_ch, **block_kw)

        self._bottleneck_channels = c * (2**depth)
        self.with_output_conv = with_output_conv
        self.outc: nn.Module | None
        if with_output_conv:
            self.outc = nn.Conv2d(c, out_channels, kernel_size=1)
        else:
            self.outc = None
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
        """Channel width at each encoder spine node X^{i,0} (for multi-scale FiLM)."""
        c = self._base_channels
        return [c * (2**i) for i in range(self.depth + 1)]

    def deep_supervision_keys(self) -> list[str]:
        """Full-resolution nested nodes X^{0,1} … X^{0,L} for deep supervision."""
        return [f"x0{j}" for j in range(1, self.depth + 1)]

    def nested_node_keys(self) -> list[str]:
        """All dense nested nodes X^{i,j} with j >= 1 (excludes encoder spine)."""
        keys: list[str] = []
        for j in range(1, self.depth + 1):
            for i in range(self.depth - j + 1):
                keys.append(f"x{i}{j}")
        return keys

    def build_nodes(
        self,
        x: torch.Tensor,
        *,
        encoder_hooks: list | None = None,
        bottleneck_fn=None,
        level_fusions: list | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Run encoder spine + dense nested decoder.

        FiLM hooks:
        - encoder_hooks[i] applied after spine node X^{i,0}
        - bottleneck_fn applied after deepest spine X^{L,0}
        - level_fusions[i] optional fuse (e.g. HR pyramid) after hooks on X^{i,0}
        """
        hooks = encoder_hooks or []
        fusions = level_fusions or []
        h = self.x00(x)
        if len(hooks) > 0 and hooks[0] is not None:
            h = hooks[0](h)
        if len(fusions) > 0 and fusions[0] is not None:
            h = fusions[0](h)
        nodes: dict[str, torch.Tensor] = {"x00": h}

        for i, down in enumerate(self.downs):
            h = down(nodes[f"x{i}0"])
            hook_idx = i + 1
            if hook_idx < len(hooks) and hooks[hook_idx] is not None:
                h = hooks[hook_idx](h)
            if hook_idx < len(fusions) and fusions[hook_idx] is not None:
                h = fusions[hook_idx](h)
            nodes[f"x{i + 1}0"] = h

        deep_key = f"x{self.depth}0"
        if bottleneck_fn is not None:
            if not (len(hooks) > self.depth and hooks[self.depth] is not None):
                nodes[deep_key] = bottleneck_fn(nodes[deep_key])

        # Dense UNet++: for each nest column j, build deep→shallow so U(X^{i+1,j-1}) exists.
        for j in range(1, self.depth + 1):
            for i in range(self.depth - j, -1, -1):
                up = self._upsample(nodes[f"x{i + 1}{j - 1}"], nodes[f"x{i}0"])
                priors = [nodes[f"x{i}{k}"] for k in range(j)]
                cat = torch.cat([*priors, up], dim=1)
                nodes[f"x{i}{j}"] = self.nested[f"x{i}{j}"](cat)
        return nodes

    def _project(self, nodes: dict[str, torch.Tensor]) -> torch.Tensor:
        feat = nodes[f"x0{self.depth}"]
        if self.outc is None:
            return feat
        return self.outc(feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._project(self.build_nodes(x))

    def forward_with_stem_hook(self, x: torch.Tensor, stem_fn) -> torch.Tensor:
        return self.forward_with_encoder_hooks(x, [stem_fn])

    def forward_with_bottleneck_hook(self, x: torch.Tensor, bottleneck_fn) -> torch.Tensor:
        return self._project(self.build_nodes(x, bottleneck_fn=bottleneck_fn))

    def forward_with_encoder_hooks(
        self,
        x: torch.Tensor,
        encoder_hooks: list,
        bottleneck_fn=None,
        *,
        level_fusions: list | None = None,
    ) -> torch.Tensor:
        """Apply FiLM (or other hooks) after each encoder spine block X^{i,0}."""
        return self._project(
            self.build_nodes(
                x,
                encoder_hooks=encoder_hooks,
                bottleneck_fn=bottleneck_fn,
                level_fusions=level_fusions,
            )
        )

    def forward_nodes(
        self,
        x: torch.Tensor,
        *,
        encoder_hooks: list | None = None,
        bottleneck_fn=None,
        level_fusions: list | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return all nested nodes (no output 1×1). Used for deep supervision."""
        return self.build_nodes(
            x,
            encoder_hooks=encoder_hooks,
            bottleneck_fn=bottleneck_fn,
            level_fusions=level_fusions,
        )
