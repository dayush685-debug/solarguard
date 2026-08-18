"""Checkpoint save/load. Saves everything needed to explain or resume a run without
looking anywhere else — the config, the epoch, the metric that earned this checkpoint
the "best" label, the class mapping, and the seed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from solarguard.training.config import TrainingConfig


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: TrainingConfig,
    best_metric_value: float,
    class_mapping: dict[str, int],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": config.to_dict(),
            "best_metric_name": config.primary_metric,
            "best_metric_value": best_metric_value,
            "class_mapping": class_mapping,
            "seed": config.seed,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint at {path}")

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
