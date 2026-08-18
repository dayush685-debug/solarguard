"""Orchestrates a full training run: epoch loop, LR scheduling, early stopping,
checkpointing on improvement. Architecture-agnostic — takes model/optimizer/loaders as
arguments rather than constructing BaselineCNN itself, so Phase 4/5 reuse this
unmodified by passing a different model in.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from solarguard.training.checkpoint import save_checkpoint
from solarguard.training.config import TrainingConfig
from solarguard.training.engine import evaluate, resolve_device, train_one_epoch


def _is_improvement(
    primary_value: float, best_value: float, val_loss: float, best_val_loss: float, min_delta: float
) -> bool:
    """Phase 3 §8 selection criterion, with a min_delta floor added after the file-4
    review: a strict `>` alone treats e.g. 0.5012000001 > 0.5012 as a real
    improvement, which is within GPU floating-point noise range for macro F1.

    Only the "genuine improvement" branch is gated by min_delta — the exact-tie
    tie-breaker (val_loss) is unchanged from the original approved logic."""
    genuine_improvement = primary_value > best_value + min_delta
    exact_tie_broken_by_loss = primary_value == best_value and val_loss < best_val_loss
    return genuine_improvement or exact_tie_broken_by_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    config: TrainingConfig,
    class_names: list[str],
) -> dict:
    """Test data never appears in this function's signature — there is no path for
    it to reach here even by mistake."""
    set_seed(config.seed)
    device = resolve_device(config.device)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience,
    )

    class_mapping = {name: i for i, name in enumerate(class_names)}
    checkpoint_path = config.experiment_dir / "checkpoint_best.pt"

    history: list[dict] = []
    best_value = -float("inf")
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    training_start = time.time()

    for epoch in range(1, config.max_epochs + 1):
        epoch_start = time.time()
        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device, class_names)
        epoch_elapsed = time.time() - epoch_start

        primary_value = val_metrics[config.primary_metric]
        scheduler.step(primary_value)

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["train_loss"],
            "val_loss": val_metrics["val_loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": round(epoch_elapsed, 3),
        }
        history.append(record)

        improved = _is_improvement(
            primary_value, best_value, val_metrics["val_loss"], best_val_loss, config.min_delta
        )
        if improved:
            best_value = primary_value
            best_val_loss = val_metrics["val_loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path, model, optimizer, epoch, config, best_value, class_mapping
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.early_stopping_patience:
            break

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_value": best_value,
        "primary_metric": config.primary_metric,
        "checkpoint_path": str(checkpoint_path),
        "stopped_early": epochs_without_improvement >= config.early_stopping_patience,
        "epochs_ran": epoch,
        "total_time_seconds": round(time.time() - training_start, 2),
    }
