from __future__ import annotations

import math


def warmup_cosine_lr(
    epoch: int,
    *,
    peak_lr: float,
    total_epochs: int,
    warmup_epochs: int = 0,
    min_lr_ratio: float = 0.01,
) -> float:
    """
    Per-epoch learning rate: linear warmup, then cosine decay to peak_lr * min_lr_ratio.

    ``epoch`` is 1-indexed (first training epoch = 1).
    """
    if total_epochs < 1:
        raise ValueError("total_epochs must be >= 1")
    peak_lr = float(peak_lr)
    min_lr = peak_lr * float(min_lr_ratio)
    warmup_epochs = max(0, int(warmup_epochs))

    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return peak_lr * epoch / warmup_epochs

    if total_epochs <= warmup_epochs:
        return min_lr

    # Cosine segment: epoch warmup+1 .. total_epochs maps to progress 0 .. 1.
    t = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    t = min(1.0, max(0.0, t))
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * t))


def resolve_lr(
    epoch: int,
    *,
    schedule: str,
    peak_lr: float,
    total_epochs: int,
    warmup_epochs: int = 0,
    min_lr_ratio: float = 0.01,
) -> float:
    """Return LR for ``epoch`` (1-indexed) under the named schedule."""
    if schedule == "constant":
        return float(peak_lr)
    if schedule == "warmup_cosine":
        return warmup_cosine_lr(
            epoch,
            peak_lr=peak_lr,
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
            min_lr_ratio=min_lr_ratio,
        )
    raise ValueError(f"Unknown lr schedule: {schedule!r}")
