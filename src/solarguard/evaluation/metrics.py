"""Metric computation — separated from the training loop so the exact same function
computes validation metrics during Phase 3 and, unmodified, final test metrics in the
later evaluation phase. One implementation, never two competing versions.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    """y_true, y_pred: integer class indices (matching class_mapping.json order).
    Returns accuracy, macro F1, weighted F1, per-class precision/recall/F1/support,
    and the confusion matrix — the complete set Phase 3 §6 requires, in one place."""
    labels = list(range(len(class_names)))

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        class_names[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(class_names))
    }

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": class_names,
    }
