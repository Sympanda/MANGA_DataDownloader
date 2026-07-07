"""Diagnose NaN training loss on real batches."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from manga_models.batch_utils import (
    masked_mse_loss_multichannel,
    prepare_spatial_input,
    prepare_spectrum_input,
    prepare_targets_and_masks,
)
from manga_models.conditional_unet import ConditionalMapUNet
from manga_models.config import ConditionalUNetConfig
from manga_models.train_utils import build_dataset_from_config
from manga_prep.dataset.manga_dataset import collate_manga_batch


def main() -> None:
    cfg = ConditionalUNetConfig(
        use_sdss=True,
        use_legacy=False,
        use_spectrum=True,
        use_footprint_mask=True,
    )
    ds = build_dataset_from_config(cfg, Path("manga_sdss_fits"))
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_manga_batch)
    batch = next(iter(loader))

    x = prepare_spatial_input(batch, cfg)
    spec = prepare_spectrum_input(batch, cfg)
    targets, masks = prepare_targets_and_masks(batch, cfg)

    print("spatial nan:", torch.isnan(x).sum().item(), "max:", float(x.max()))
    print("spec nan:", torch.isnan(spec).sum().item() if spec is not None else 0)
    print("targets nan:", torch.isnan(targets).sum().item())
    print("targets inf:", torch.isinf(targets).sum().item())
    print("mask sum per ch:", masks.sum(dim=(0, 2, 3)).tolist())

    pred = torch.randn_like(targets)
    loss = masked_mse_loss_multichannel(pred, targets, masks)
    print("loss:", float(loss.item()), "finite:", bool(torch.isfinite(loss).item()))

    # NaN * 0 demo
    t = torch.tensor([float("nan")])
    print("nan * 0 =", float((t * 0).item()))

    for i in range(len(batch["plateifu"])):
        t = targets[i : i + 1]
        m = masks[i : i + 1]
        p = pred[i : i + 1]
        l = masked_mse_loss_multichannel(p, t, m)
        print(
            f"  {batch['plateifu'][i]}: loss={float(l.item()):.6f} "
            f"nan_targets={int(torch.isnan(t).sum())} mask_sum={float(m.sum()):.0f}"
        )

    model = ConditionalMapUNet(cfg)
    with torch.no_grad():
        out = model(x, spectrum_flux=spec)
    print("model out nan:", torch.isnan(out).sum().item())
    loss2 = masked_mse_loss_multichannel(out, targets, masks)
    print("model loss:", float(loss2.item()), "finite:", bool(torch.isfinite(loss2).item()))


if __name__ == "__main__":
    main()
