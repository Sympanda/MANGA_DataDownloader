from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics.plots import evaluate_map_predictions, write_metrics_csv
from src.training.lr_schedule import resolve_lr


@dataclass
class TrainConfig:
    run_name: str = "run_001"
    save_root: str = "runs/manga_maps"
    seed: int = 42
    epochs: int = 100
    train_batch_size: int = 8
    eval_batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lr_schedule: str = "warmup_cosine"  # constant | warmup_cosine
    lr_warmup_epochs: int = 5
    lr_min_ratio: float = 0.01
    grad_clip: float = 1.0
    amp: bool = True
    early_stop_patience: int = 15
    early_stop_start_epoch: int = 1
    save_every: int = 5
    device: str = "cuda"
    write_plots: bool = True
    write_csv_history: bool = True
    save_config_snapshot: bool = True
    eval_max_plot: int = 8
    # Post-train map eval: default skips full train split (8k+ galaxies).
    eval_splits: tuple[str, ...] = ("val", "test")
    run_post_train_eval: bool = True


def prepare_run_dirs(cfg: TrainConfig) -> dict[str, str]:
    run_dir = os.path.join(cfg.save_root, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    for sub in ("ckpts", "plots", "logs", "csv"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    return {
        "root": run_dir,
        "ckpts": os.path.join(run_dir, "ckpts"),
        "plots": os.path.join(run_dir, "plots"),
        "logs": os.path.join(run_dir, "logs"),
        "csv": os.path.join(run_dir, "csv"),
    }


def setup_logger(run_dirs: dict[str, str]) -> logging.Logger:
    logger = logging.getLogger(run_dirs["root"])
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(os.path.join(run_dirs["logs"], "run.log"))
    sh = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _to_float(x) -> float:
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def _flush_logger(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        handler.flush()


def _write_history_csv(history: list[dict[str, float]], csv_path: str | os.PathLike[str]) -> None:
    """Write loss history with stdlib csv (avoids pandas/pyarrow native crashes on Windows)."""
    if not history:
        return
    fieldnames = list(history[0].keys())
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def _persist_training_history(
    history: list[dict],
    run_dirs: dict[str, str],
    *,
    write_csv: bool,
    write_plots: bool,
    logger: logging.Logger,
) -> None:
    if not history:
        return
    rows: list[dict[str, float]] = [{k: float(v) for k, v in row.items()} for row in history]
    if write_csv:
        csv_path = os.path.join(run_dirs["csv"], "train_val_history.csv")
        _write_history_csv(rows, csv_path)
        logger.info(f"Wrote training history -> {csv_path}")
        _flush_logger(logger)
    if write_plots:
        try:
            from src.metrics.plots import plot_training_history

            plot_training_history(rows, run_dirs["plots"])
            logger.info(f"Wrote loss plots -> {run_dirs['plots']}")
        except Exception as exc:
            logger.exception(f"Loss plot failed (history CSV still saved): {exc}")
        _flush_logger(logger)


def _load_checkpoint_state(ckpt: object) -> dict[str, torch.Tensor]:
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt["model_state"]
    if isinstance(ckpt, dict):
        # Raw state_dict saved directly (Galaxy_ILI style).
        return ckpt  # type: ignore[return-value]
    raise TypeError(f"Unexpected checkpoint type: {type(ckpt)!r}")


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_cfg: TrainConfig,
        run_dirs: dict[str, str],
        logger: logging.Logger,
    ) -> None:
        self.model = model
        self.cfg = train_cfg
        self.dirs = run_dirs
        self.logger = logger
        self.device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=train_cfg.amp and self.device.type == "cuda")
        self.best_val = float("inf")
        self.bad_epochs = 0
        self.history: list[dict] = []

    def _set_lr(self, epoch: int) -> float:
        lr = resolve_lr(
            epoch,
            schedule=self.cfg.lr_schedule,
            peak_lr=self.cfg.lr,
            total_epochs=self.cfg.epochs,
            warmup_epochs=self.cfg.lr_warmup_epochs,
            min_lr_ratio=self.cfg.lr_min_ratio,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def _step(self, batch: dict, *, train: bool, epoch: int | None = None) -> dict[str, float]:
        batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        # nested tensors
        if "inputs" in batch:
            inputs = batch["inputs"]
            for key in ("sdss_imaging", "legacy_imaging", "hr_imaging"):
                if key in inputs:
                    inputs[key] = inputs[key].to(self.device)
            if "spectrum" in inputs:
                for sk in ("wave", "flux", "ivar"):
                    if sk in inputs["spectrum"]:
                        inputs["spectrum"][sk] = inputs["spectrum"][sk].to(self.device)
        if "targets" in batch:
            batch["targets"] = {k: v.to(self.device) for k, v in batch["targets"].items()}
        if "target_loss_masks" in batch:
            batch["target_loss_masks"] = {k: v.to(self.device) for k, v in batch["target_loss_masks"].items()}
        if "footprint_mask" in batch:
            batch["footprint_mask"] = batch["footprint_mask"].to(self.device)

        if train:
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            # Raw SDSS flux + UNet++/pixel_shuffle can overflow fp16; keep forward in fp32.
            with torch.amp.autocast("cuda", enabled=False):
                _, loss_dict = self.model(batch, epoch=epoch)
                loss = loss_dict["loss"]
            if not torch.isfinite(loss):
                plateifus = batch.get("plateifu", ["?"])
                self.logger.warning(
                    f"Skipping non-finite train loss (epoch={epoch}, "
                    f"plates={plateifus[:3]}{'...' if len(plateifus) > 3 else ''})"
                )
                return {k: float("nan") for k in loss_dict}
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                if self.cfg.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()
        else:
            self.model.eval()
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                _, loss_dict = self.model(batch, epoch=epoch)

        return {k: _to_float(v) for k, v in loss_dict.items()}

    def fit(self, dl_train: DataLoader, dl_val: DataLoader) -> list[dict]:
        history: list[dict] = []
        self.history = history
        for epoch in range(1, self.cfg.epochs + 1):
            cur_lr = self._set_lr(epoch)
            train_logs = []
            n_train = len(dl_train)
            pbar = tqdm(
                dl_train,
                desc=f"Epoch {epoch}/{self.cfg.epochs} [train]",
                leave=n_train <= 4,  # keep bar visible on tiny overfit loaders
                mininterval=0.1 if n_train <= 4 else 1.0,
                dynamic_ncols=True,
            )
            for batch in pbar:
                stats = self._step(batch, train=True, epoch=epoch)
                pbar.set_postfix({"loss": f"{stats.get('loss', 0):.4f}"})
                train_logs.append(stats)

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            val_logs = []
            self.model.eval()
            with torch.no_grad():
                n_val = len(dl_val)
                pbar = tqdm(
                    dl_val,
                    desc=f"Epoch {epoch}/{self.cfg.epochs} [val]",
                    leave=n_val <= 4,
                    mininterval=0.1 if n_val <= 4 else 1.0,
                    dynamic_ncols=True,
                )
                for batch in pbar:
                    stats = self._step(batch, train=False, epoch=epoch)
                    pbar.set_postfix({"loss": f"{stats.get('loss', 0):.4f}"})
                    val_logs.append(stats)
            pbar.close()

            def _mean(logs: list[dict]) -> dict[str, float]:
                if not logs:
                    return {}
                keys = logs[0].keys()
                return {
                    k: float(np.nanmean([d.get(k, float("nan")) for d in logs]))
                    for k in keys
                }

            train_mean = _mean(train_logs)
            val_mean = _mean(val_logs)
            row: dict[str, float] = {"epoch": float(epoch), "lr": float(cur_lr)}
            row.update({f"train_{k}": float(v) for k, v in train_mean.items()})
            row.update({f"val_{k}": float(v) for k, v in val_mean.items()})
            history.append(row)

            parts = [f"lr {cur_lr:.2e}", f"loss {train_mean.get('loss', 0):.4f}/{val_mean.get('loss', 0):.4f}"]
            for key in sorted(train_mean):
                if key == "loss":
                    continue
                parts.append(f"{key} {train_mean[key]:.4f}/{val_mean.get(key, 0):.4f}")
            self.logger.info(f"Epoch {epoch}: " + " | ".join(parts))
            _flush_logger(self.logger)

            cur = val_mean.get("loss", float("inf"))
            if cur < self.best_val:
                self.best_val = cur
                self.bad_epochs = 0
                # Galaxy_ILI-style: torch.save reads tensors in place; never .cpu() state_dict values.
                torch.save(self.model.state_dict(), os.path.join(self.dirs["ckpts"], "best.pt"))
                self.logger.info(f"  new best val_loss={cur:.6f}")
                _flush_logger(self.logger)
            elif epoch >= self.cfg.early_stop_start_epoch:
                self.bad_epochs += 1
                if self.bad_epochs >= self.cfg.early_stop_patience:
                    self.logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"(patience={self.cfg.early_stop_patience} from epoch "
                        f"{self.cfg.early_stop_start_epoch})"
                    )
                    _flush_logger(self.logger)
                    break

            if epoch % self.cfg.save_every == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "val_loss": cur,
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "config": self.model.config.__dict__,
                    },
                    os.path.join(self.dirs["ckpts"], f"epoch_{epoch}.pt"),
                )

        _persist_training_history(
            history,
            self.dirs,
            write_csv=self.cfg.write_csv_history,
            write_plots=self.cfg.write_plots,
            logger=self.logger,
        )

        best_path = os.path.join(self.dirs["ckpts"], "best.pt")
        if os.path.exists(best_path):
            self.model.load_state_dict(
                _load_checkpoint_state(torch.load(best_path, map_location=self.device, weights_only=False))
            )

        return history


def _resolve_eval_fn(model):
    cfg = getattr(model, "config", None)
    if cfg is not None and getattr(cfg, "output_head", None) == "gaussian":
        from src.metrics.uncertainty_plots import evaluate_uncertainty_predictions

        return evaluate_uncertainty_predictions
    return evaluate_map_predictions


def run_training(
    model: nn.Module,
    train_cfg: TrainConfig,
    dl_train: DataLoader,
    dl_val: DataLoader,
    dl_test: DataLoader,
    dl_train_no_shuffle: DataLoader,
    *,
    user_snapshot: dict[str, Any] | None = None,
) -> dict[str, str]:
    run_dirs = prepare_run_dirs(train_cfg)
    logger = setup_logger(run_dirs)

    if train_cfg.save_config_snapshot:
        snap = {"train": train_cfg.__dict__, "user": user_snapshot or {}}
        with open(os.path.join(run_dirs["root"], "config_used.json"), "w", encoding="utf-8") as fh:
            json.dump(snap, fh, indent=2)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")
    logger.info(f"Device: {train_cfg.device}")
    logger.info(f"Run directory: {run_dirs['root']}")
    logger.info(
        f"LR schedule: {train_cfg.lr_schedule} peak={train_cfg.lr:g} "
        f"warmup={train_cfg.lr_warmup_epochs} min_ratio={train_cfg.lr_min_ratio}"
    )
    logger.info(
        f"Early stop: patience={train_cfg.early_stop_patience} "
        f"from epoch {train_cfg.early_stop_start_epoch}"
    )

    trainer = Trainer(model, train_cfg, run_dirs, logger)
    history: list[dict] = []
    try:
        history = trainer.fit(dl_train, dl_val)
    except Exception:
        logger.exception("Training failed")
        _persist_training_history(
            trainer.history,
            run_dirs,
            write_csv=train_cfg.write_csv_history,
            write_plots=train_cfg.write_plots,
            logger=logger,
        )
        raise

    if not train_cfg.run_post_train_eval:
        logger.info("Post-train eval disabled (run_post_train_eval=false)")
        return run_dirs

    best_path = os.path.join(run_dirs["ckpts"], "best.pt")
    if not os.path.exists(best_path):
        logger.warning("No best.pt found; skipping post-train eval")
        return run_dirs

    split_loaders = {
        "train": dl_train_no_shuffle,
        "val": dl_val,
        "test": dl_test,
    }
    plots_dir = Path(run_dirs["plots"])
    map_keys = tuple(model.config.target_keys)
    eval_fn = _resolve_eval_fn(model)

    for split in train_cfg.eval_splits:
        if split not in split_loaders:
            logger.warning(f"Unknown eval split {split!r}; skipping")
            continue
        logger.info(f"Evaluating split={split} ...")
        try:
            rows = eval_fn(
                model,
                split_loaders[split],
                device=trainer.device,
                map_keys=map_keys,
                plots_dir=plots_dir,
                split=split,
                max_plot=train_cfg.eval_max_plot if split != "train" else min(4, train_cfg.eval_max_plot),
            )
            csv_path = os.path.join(run_dirs["csv"], f"{split}_metrics.csv")
            write_metrics_csv(rows, csv_path)
            mse_vals = [float(r["mse_all"]) for r in rows if np.isfinite(float(r["mse_all"]))]
            mean_mse = float(np.mean(mse_vals)) if mse_vals else float("nan")
            logger.info(f"  {split} mean mse_all={mean_mse:.6f} -> {csv_path}")
        except Exception:
            logger.exception(f"Eval failed for split={split}")
            raise

    return run_dirs
