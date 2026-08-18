"""One training epoch, one validation pass. Architecture-agnostic by construction —
takes `model` as a parameter and calls nothing specific to BaselineCNN, so Phase 4/5's
MobileNetV2 and EfficientNet reuse this file unmodified.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from solarguard.evaluation.metrics import compute_metrics


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print("WARNING: cuda requested but not available — falling back to cpu")
        return torch.device("cpu")
    return torch.device(requested)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    model.train()
    total_loss = 0.0
    n_samples = 0

    for images, labels in loader:
        if images.ndim != 4:
            raise ValueError(f"expected batched images (B,C,H,W), got shape {tuple(images.shape)}")

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size

    return {"train_loss": total_loss / n_samples}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    class_names: list[str],
    loss_prefix: str = "val",
) -> dict:
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = loss_fn(logits, labels)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    metrics = compute_metrics(y_true, y_pred, class_names)
    metrics[f"{loss_prefix}_loss"] = total_loss / n_samples
    return metrics
