"""
Train conditional UNet: imaging (+ spectrum) -> Amara map targets.

Example (simple first pass: SDSS + fake spectrum + footprint mask):
  python train_conditional_unet.py --epochs 1 --batch-size 4 --max-batches 5 --device cpu
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from manga_models.batch_utils import (
    compute_map_training_loss,
    prepare_spatial_input,
    prepare_spectrum_input,
    prepare_targets_and_masks,
)
from manga_models.conditional_unet import ConditionalMapUNet
from manga_models.train_utils import (
    add_training_data_args,
    build_dataset_from_config,
    config_from_args,
    report_aligned_cache_status,
    resolve_training_device,
    save_run_config,
)
from manga_prep.dataset.manga_dataset import collate_manga_batch


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config,
    *,
    epoch: int,
    max_batches: int | None = None,
    grad_clip: float | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    progress = tqdm(loader, desc=f"Epoch {epoch}", unit="batch", leave=True)
    for batch in progress:
        t0 = time.perf_counter()
        x = prepare_spatial_input(batch, config).to(device)
        spec = prepare_spectrum_input(batch, config)
        if spec is not None:
            spec = spec.to(device)
        targets, masks = prepare_targets_and_masks(batch, config)
        targets = targets.to(device)
        masks = masks.to(device)

        pred = model(x, spectrum_flux=spec)
        loss = compute_map_training_loss(pred, targets, masks, config)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_loss = float(loss.item())
        total_loss += batch_loss
        n_batches += 1
        elapsed = time.perf_counter() - t0
        progress.set_postfix(loss=f"{batch_loss:.5f}", sec=f"{elapsed:.1f}")

        if max_batches is not None and n_batches >= max_batches:
            break

    return total_loss / max(n_batches, 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train conditional Amara-map UNet.")
    add_training_data_args(parser)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/conditional_unet"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Max grad norm (0=disable)")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit batches per epoch (smoke test)")
    args = parser.parse_args(argv)

    config = config_from_args(args)
    config.validate()

    device = resolve_training_device(args.device)
    dataset = build_dataset_from_config(config, args.data_root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_manga_batch,
        pin_memory=device.type == "cuda",
    )

    model = ConditionalMapUNet(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    save_run_config(config, args.out_dir)

    print("=" * 60)
    print("Conditional UNet training")
    print("=" * 60)
    print(f"  galaxies     : {len(dataset):,}")
    print(f"  batches/epoch: {len(loader) if args.max_batches is None else min(len(loader), args.max_batches)}")
    print(f"  batch size   : {args.batch_size}")
    print(f"  input ch     : {config.input_channels()}  ->  {config.n_target_maps} target maps")
    print(f"  model size   : {args.model_size}")
    print(f"  base ch      : {config.base_channels}  bottleneck x{config.bottleneck_multiplier}")
    print(f"  dropout      : {config.dropout}")
    print(f"  upsample     : {config.upsample_mode}")
    print(f"  loss wts     : mse={config.loss_mse_weight} l1={config.loss_l1_weight} grad={config.loss_grad_weight}")
    print(f"  weight decay : {args.weight_decay}")
    print(f"  grad clip    : {args.grad_clip}")
    print(f"  SDSS         : {config.use_sdss}")
    print(f"  Legacy       : {config.use_legacy}")
    print(f"  spectrum     : {config.use_spectrum}")
    print(f"  footprint    : {config.use_footprint_mask}")
    print(f"  parameters   : {n_params:,}")
    print(f"  device       : {device}")
    print(f"  output dir   : {args.out_dir}")
    report_aligned_cache_status(dataset)
    print("=" * 60)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        t_epoch = time.perf_counter()
        avg_loss = train_one_epoch(
            model,
            loader,
            optimizer,
            device,
            config,
            epoch=epoch,
            max_batches=args.max_batches,
            grad_clip=args.grad_clip if args.grad_clip > 0 else None,
        )
        epoch_time = time.perf_counter() - t_epoch
        print(f"Epoch {epoch}/{args.epochs} complete  avg_masked_mse={avg_loss:.6f}  time={epoch_time:.1f}s")

        payload = {
            "epoch": epoch,
            "avg_masked_mse": avg_loss,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config.__dict__,
        }
        latest = args.out_dir / "latest.pt"
        torch.save(payload, latest)
        print(f"  saved {latest}")

        ckpt = args.out_dir / f"epoch_{epoch:03d}.pt"
        torch.save(payload, ckpt)
        print(f"  saved {ckpt}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best = args.out_dir / "best.pt"
            torch.save(payload, best)
            print(f"  new best -> {best}  (mse={best_loss:.6f})")

    print("Training finished.")
    print(f"  best masked MSE: {best_loss:.6f}")
    print(f"  evaluate with: python scripts/legacy/eval_conditional_unet.py --checkpoint {args.out_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
