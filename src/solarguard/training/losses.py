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
import torch.nn.functional as F


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


class FocalLoss(nn.Module):
    """Class-weighted focal loss (Lin et al., 2017):
    FL = -alpha_t * (1 - p_t)^gamma * log(p_t)

    At gamma=0 this is mathematically identical to nn.CrossEntropyLoss(weight=alpha),
    including its weighted-mean reduction sum(w_i * l_i) / sum(w_i) -- verified in
    tests/test_losses.py. That equivalence is what makes gamma a clean single-knob
    extension of the baseline rather than a different loss function.

    Motivation (Experiment 2): inverse-frequency class weighting corrects for RARITY.
    The baseline's worst class, Bird-drop (F1 0.545), is not rare -- 87 train / 19 val
    images and a near-neutral weight of 1.0345 -- it is CONFUSABLE. Focal modulation
    targets difficulty instead of frequency.

    Expects raw logits, same contract as build_loss (see models/baseline_cnn.py).
    """

    def __init__(self, weight: torch.Tensor, gamma: float = 2.0) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        self.register_buffer("weight", weight)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        loss = -((1.0 - pt) ** self.gamma) * log_pt

        alpha_t = self.weight[targets]
        # same reduction as CrossEntropyLoss(weight=...): weighted mean, not plain mean
        return (alpha_t * loss).sum() / alpha_t.sum()


def build_focal_loss(class_weights: torch.Tensor, gamma: float = 2.0) -> FocalLoss:
    """Experiment 2 loss. gamma=0 reduces exactly to build_loss()."""
    return FocalLoss(weight=class_weights, gamma=gamma)
