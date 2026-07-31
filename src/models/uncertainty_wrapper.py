"""Heteroscedastic (μ+σ) map wrapper — thin specialisation of MapGenerator."""
from __future__ import annotations

from src.models.config import ModelConfig
from src.models.wrapper import MapGenerator


class UncertaintyMapGenerator(MapGenerator):
    """
    Heteroscedastic map model: μ + σ per target channel (gaussian output head).

    Public interface matches ``MapGenerator`` (``forward`` → ``pred_dict, loss_dict``).
    Requires ``config.output_head == "gaussian"``; NLL is composed via ``log_var`` in aux.
    """

    def __init__(self, config: ModelConfig) -> None:
        if config.output_head != "gaussian":
            raise ValueError(
                f"UncertaintyMapGenerator requires output_head='gaussian', got {config.output_head!r}"
            )
        super().__init__(config)
