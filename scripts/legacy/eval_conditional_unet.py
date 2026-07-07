"""
Evaluate a trained conditional UNet and save example prediction plots.

Uses the same training dataset by default (no held-out split while experimenting).

Example:
  python eval_conditional_unet.py --checkpoint runs/conditional_unet/epoch_010.pt
  python eval_conditional_unet.py --run-dir runs/conditional_unet --checkpoint epoch_010.pt --n-samples 4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from manga_models.batch_utils import (
    compute_map_training_loss,
    masked_mse_loss_multichannel,
    prepare_spatial_input,
    prepare_spectrum_input,
    prepare_targets_and_masks,
)
from manga_models.conditional_unet import ConditionalMapUNet
from manga_models.config import ConditionalUNetConfig
from manga_models.train_utils import (
    TARGET_LABELS,
    build_dataset_from_config,
    load_config_from_checkpoint,
    load_config_from_run,
    pick_eval_indices,
    resolve_training_device,
)
from manga_prep.dataset.manga_dataset import collate_manga_batch


def _percentile_norm(x: np.ndarray, lo: float = 5, hi: float = 99) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0)
    pos = x[x > 0]
    if pos.size == 0:
        return np.zeros_like(x)
    p_lo, p_hi = np.percentile(pos, [lo, hi])
    return np.clip((x - p_lo) / max(p_hi - p_lo, 1e-6), 0, 1)


def plot_galaxy_prediction(
    *,
    plateifu: str,
    sdss_r: np.ndarray | None,
    target: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    map_keys: tuple[str, ...],
    out_path: Path,
    epoch: int | None = None,
    show_full_prediction: bool = True,
) -> None:
    n_maps = len(map_keys)
    n_cols = 5 if show_full_prediction else 4
    fig, axes = plt.subplots(n_maps, n_cols, figsize=(3.2 * n_cols, 3.2 * n_maps), squeeze=False)

    if sdss_r is not None:
        axes[0, 0].imshow(_percentile_norm(sdss_r), origin="lower", cmap="gray", vmin=0, vmax=1)
        axes[0, 0].set_title("SDSS r (input)")
    else:
        axes[0, 0].axis("off")
        axes[0, 0].set_title("no SDSS")

    for row in range(n_maps):
        key = map_keys[row]
        tgt = target[row]
        prd = pred[row]
        m = mask[row].astype(bool)
        diff = np.where(m, prd - tgt, np.nan)

        if row > 0:
            axes[row, 0].axis("off")

        im1 = axes[row, 1].imshow(np.where(m, tgt, np.nan), origin="lower", cmap="viridis", vmin=0, vmax=1)
        axes[row, 1].set_title(f"{TARGET_LABELS.get(key, key)} target")
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046)

        im2 = axes[row, 2].imshow(np.where(m, prd, np.nan), origin="lower", cmap="viridis", vmin=0, vmax=1)
        axes[row, 2].set_title("predicted (masked)")
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046)

        vmax = np.nanpercentile(np.abs(diff), 99) if np.any(np.isfinite(diff)) else 1.0
        im3 = axes[row, 3].imshow(diff, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        axes[row, 3].set_title("pred - target")
        plt.colorbar(im3, ax=axes[row, 3], fraction=0.046)

        if show_full_prediction:
            im4 = axes[row, 4].imshow(prd, origin="lower", cmap="viridis", vmin=0, vmax=1)
            axes[row, 4].set_title("predicted (full)")
            plt.colorbar(im4, ax=axes[row, 4], fraction=0.046)

        for col in range(n_cols):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    title = f"{plateifu}"
    if epoch is not None:
        title += f"  (epoch {epoch})"
    fig.suptitle(title, y=1.01)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def evaluate_samples(
    model: ConditionalMapUNet,
    dataset,
    indices: list[int],
    device: torch.device,
    config: ConditionalUNetConfig,
    out_dir: Path,
    *,
    epoch: int | None = None,
    show_full_prediction: bool = True,
) -> dict[str, float]:
    model.eval()
    per_map_mse: dict[str, list[float]] = {k: [] for k in config.target_keys}
    training_losses: list[float] = []

    for idx in tqdm(indices, desc="Eval samples", unit="galaxy"):
        sample = dataset[idx]
        batch = collate_manga_batch([sample])

        x = prepare_spatial_input(batch, config).to(device)
        spec = prepare_spectrum_input(batch, config)
        if spec is not None:
            spec = spec.to(device)
        targets, masks = prepare_targets_and_masks(batch, config)
        targets = targets.to(device)
        masks = masks.to(device)

        pred = model(x, spectrum_flux=spec)
        loss = compute_map_training_loss(pred, targets, masks, config)
        training_losses.append(float(loss.item()))

        for ch, key in enumerate(config.target_keys):
            ch_loss = masked_mse_loss_multichannel(
                pred[:, ch : ch + 1],
                targets[:, ch : ch + 1],
                masks[:, ch : ch + 1],
            )
            per_map_mse[key].append(float(ch_loss.item()))

        plateifu = batch["plateifu"][0]
        sdss_r = None
        if config.use_sdss and "sdss_imaging" in batch.get("inputs", {}):
            sdss = batch["inputs"]["sdss_imaging"][0].cpu().numpy()
            bands = batch["inputs"].get("sdss_imaging_bands", ("u", "g", "r", "i", "z"))
            bidx = {b: i for i, b in enumerate(bands)}
            if "r" in bidx:
                sdss_r = sdss[bidx["r"]]

        plot_galaxy_prediction(
            plateifu=plateifu,
            sdss_r=sdss_r,
            target=targets[0].cpu().numpy(),
            pred=pred[0].cpu().numpy(),
            mask=masks[0].cpu().numpy(),
            map_keys=config.target_keys,
            out_path=out_dir / f"{plateifu.replace('-', '_')}.png",
            epoch=epoch,
            show_full_prediction=show_full_prediction,
        )

    summary = {
        "training_loss_mean": float(np.mean(training_losses)),
        "masked_mse_mean": float(np.mean([np.mean(v) for v in per_map_mse.values()])),
        **{f"mse_{k}": float(np.mean(v)) for k, v in per_map_mse.items()},
    }
    return summary


def resolve_checkpoint_path(args: argparse.Namespace) -> Path:
    ckpt = Path(args.checkpoint)
    if ckpt.is_file():
        return ckpt
    if args.run_dir is not None:
        candidate = Path(args.run_dir) / ckpt
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find checkpoint: {args.checkpoint}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate conditional UNet and plot predictions.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint or filename in --run-dir")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/conditional_unet"))
    parser.add_argument("--out-dir", type=Path, default=None, help="Default: <run-dir>/eval_<checkpoint_stem>")
    parser.add_argument("--data-root", type=Path, default=Path("manga_sdss_fits"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--indices", type=int, nargs="*", default=None, help="Optional explicit dataset indices")
    parser.add_argument(
        "--no-full-prediction",
        action="store_true",
        help="Omit the full-canvas predicted column (masked columns only).",
    )
    args = parser.parse_args(argv)

    ckpt_path = resolve_checkpoint_path(args)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    epoch = ckpt.get("epoch")

    try:
        config = load_config_from_checkpoint(ckpt_path)
    except KeyError:
        config = load_config_from_run(args.run_dir)

    out_dir = args.out_dir or (args.run_dir / f"eval_{ckpt_path.stem}")
    device = resolve_training_device(args.device)

    print(f"Checkpoint: {ckpt_path}")
    print(f"Epoch: {epoch}")
    print(f"Device: {device}")
    print(f"Output plots: {out_dir}")

    dataset = build_dataset_from_config(config, args.data_root)
    print(f"Dataset size: {len(dataset):,} galaxies")

    if args.indices:
        indices = list(args.indices)
    else:
        indices = pick_eval_indices(len(dataset), args.n_samples, args.seed)
    print(f"Evaluating indices: {indices}")

    model = ConditionalMapUNet(config).to(device)
    model.load_state_dict(ckpt["model_state"])

    summary = evaluate_samples(
        model,
        dataset,
        indices,
        device,
        config,
        out_dir,
        epoch=epoch,
        show_full_prediction=not args.no_full_prediction,
    )

    print("\nLoss summary:")
    print(f"  training loss: {summary['training_loss_mean']:.6f}")
    print(f"  masked MSE  : {summary['masked_mse_mean']:.6f}")
    for key in config.target_keys:
        print(f"  {key:16s}: {summary[f'mse_{key}']:.6f}")
    print(f"\nSaved {len(indices)} plot(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
