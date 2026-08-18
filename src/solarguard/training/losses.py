"""Class-weighted loss for the baseline CNN. See PLANNING.md Phase 3 §3 and §6 for the
full rationale: unweighted cross-entropy would let the majority classes (Clean, Dusty)
dominate the training signal at a 2.73:1 imbalance ratio.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from torch import nn


def compute_class_weights(labels: pd.Series, class_to_idx: dict[str, int]) -> torch.Tensor:
    """Inverse-frequency weights: n_samples / (n_classes * count[class]).

    Pure function — has no knowledge of train/val/test splits, so it will silently
    compute the wrong thing if given the wrong labels. The one caller in this codebase
    is `class_weights_from_train_split`, which hardcodes train.csv specifically so that
    guarantee is visible at the call site rather than hidden behind a parameter.
    """
    counts = labels.value_counts()
    missing = set(class_to_idx) - set(counts.index)
    if missing:
        raise ValueError(
            f"class(es) {missing} have zero examples in the given labels — "
            "cannot compute an inverse-frequency weight for a class with no samples"
        )

    n_samples = len(labels)
    n_classes = len(class_to_idx)
    weights = torch.zeros(n_classes, dtype=torch.float32)
    for cls, idx in class_to_idx.items():
        weights[idx] = n_samples / (n_classes * counts[cls])
    return weights


def class_weights_from_train_split(splits_dir: Path) -> torch.Tensor:
    """The only sanctioned entry point for training code — reads train.csv, nothing else."""
    train_df = pd.read_csv(splits_dir / "train.csv")
    class_to_idx = json.loads((splits_dir / "class_mapping.json").read_text())
    return compute_class_weights(train_df["class"], class_to_idx)


def build_loss(class_weights: torch.Tensor) -> nn.CrossEntropyLoss:
    """CrossEntropyLoss expects raw logits (see models/baseline_cnn.py) — it applies
    log_softmax internally. Never pass it already-softmaxed output."""
    return nn.CrossEntropyLoss(weight=class_weights)
