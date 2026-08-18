"""Reloads the best baseline checkpoint and regenerates the artifacts Phase 3 §7
actually specified but fit()'s history (aggregate numbers only) doesn't retain:
classification_report.json (full per-class precision/recall/F1) and the confusion
matrix (csv + png). Also cross-checks the reloaded metrics against the recorded
best-epoch history row as a consistency check.

Run: PYTHONPATH=src python scripts/generate_baseline_report.py <experiment_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from solarguard.data.datasets import build_train_val_dataloaders
from solarguard.models.baseline_cnn import BaselineCNN
from solarguard.training.checkpoint import load_checkpoint
from solarguard.training.engine import evaluate, resolve_device
from solarguard.training.losses import build_loss, class_weights_from_train_split

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = REPO_ROOT / "data" / "splits"
IMAGES_ROOT = REPO_ROOT / "data" / "candidates" / "PV_Panel_Defect_Dataset"


def main(experiment_dir: Path) -> None:
    class_to_idx = json.loads((SPLITS_DIR / "class_mapping.json").read_text())
    class_names = sorted(class_to_idx, key=class_to_idx.get)

    checkpoint_path = experiment_dir / "checkpoint_best.pt"
    device = resolve_device("cuda")
    model = BaselineCNN(num_classes=len(class_names))
    checkpoint = load_checkpoint(checkpoint_path, model, map_location=device)
    model.to(device)
    print(f"loaded checkpoint from epoch {checkpoint['epoch']}, "
          f"recorded best_metric_value={checkpoint['best_metric_value']:.4f}")

    _, val_loader = build_train_val_dataloaders(
        SPLITS_DIR, IMAGES_ROOT, REPO_ROOT / "configs" / "preprocessing.yaml",
        batch_size=32, num_workers=0, seed=42,
    )
    class_weights = class_weights_from_train_split(SPLITS_DIR).to(device)
    loss_fn = build_loss(class_weights)

    result = evaluate(model, val_loader, loss_fn, device, class_names)

    history = pd.read_csv(experiment_dir / "history.csv")
    recorded = history[history["epoch"] == checkpoint["epoch"]].iloc[0]
    print(f"\nconsistency check against recorded history row (epoch {checkpoint['epoch']}):")
    print(f"  recorded val_macro_f1={recorded['val_macro_f1']:.4f}  reloaded={result['macro_f1']:.4f}  "
          f"match={abs(recorded['val_macro_f1'] - result['macro_f1']) < 1e-6}")
    print(f"  recorded val_loss={recorded['val_loss']:.4f}  reloaded={result['val_loss']:.4f}  "
          f"match={abs(recorded['val_loss'] - result['val_loss']) < 1e-6}")

    classification_report = {
        "epoch": checkpoint["epoch"],
        "accuracy": result["accuracy"],
        "macro_f1": result["macro_f1"],
        "weighted_f1": result["weighted_f1"],
        "per_class": result["per_class"],
    }
    (experiment_dir / "classification_report.json").write_text(json.dumps(classification_report, indent=2))

    cm = np.array(result["confusion_matrix"])
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(experiment_dir / "confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — best checkpoint (epoch {checkpoint['epoch']})")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(experiment_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    print(f"\nper-class metrics (best checkpoint, epoch {checkpoint['epoch']}):")
    for cls, m in result["per_class"].items():
        print(f"  {cls:20s} precision={m['precision']:.3f}  recall={m['recall']:.3f}  "
              f"f1={m['f1']:.3f}  support={m['support']}")

    print(f"\nsaved: {experiment_dir / 'classification_report.json'}")
    print(f"saved: {experiment_dir / 'confusion_matrix.csv'}")
    print(f"saved: {experiment_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
