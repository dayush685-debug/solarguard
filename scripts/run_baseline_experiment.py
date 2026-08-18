"""Runs the approved SolarGuard baseline CNN experiment end-to-end: prints the
experiment manifest, trains with early stopping, saves all Phase 3 §7 artifacts,
generates loss/macro-F1 curves, and prints the final report. Test set is never
loaded or referenced anywhere in this script.

Run: PYTHONPATH=src python scripts/run_baseline_experiment.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — this runs from a script, not a notebook
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from solarguard.data.datasets import build_train_val_dataloaders
from solarguard.models.baseline_cnn import BaselineCNN, count_parameters
from solarguard.preprocessing.transforms import load_config
from solarguard.training.checkpoint import load_checkpoint
from solarguard.training.config import TrainingConfig
from solarguard.training.engine import evaluate, resolve_device
from solarguard.training.losses import build_loss, class_weights_from_train_split
from solarguard.training.train import fit, set_seed

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = REPO_ROOT / "data" / "splits"
IMAGES_ROOT = REPO_ROOT / "data" / "candidates" / "PV_Panel_Defect_Dataset"


def print_manifest(config, class_to_idx, class_names, split_counts, preprocessing_config, model, device, run_label=None):
    aug = preprocessing_config["train_augmentation"]
    print("=" * 72)
    print("SOLARGUARD BASELINE CNN - EXPERIMENT MANIFEST")
    if run_label:
        print(f"*** {run_label} ***")
    print("=" * 72)
    print(f"dataset manifest      : {SPLITS_DIR / 'manifest.csv'}")
    print(f"images root           : {IMAGES_ROOT}")
    print(f"class mapping         : {class_to_idx}")
    print(f"split counts          : train={split_counts['Train']}  val={split_counts['Valid']}  "
          f"test={split_counts['Test']} (test NOT loaded or used this run)")
    print(f"seed                  : {config.seed}")
    print(f"image size            : {preprocessing_config['image_size']}x{preprocessing_config['image_size']}, "
          f"interpolation={preprocessing_config['interpolation']}")
    print(f"augmentation          : h-flip p={aug['horizontal_flip']['probability']}, "
          f"rotation=+/-{aug['rotation']['max_degrees']}deg, "
          f"color_jitter(brightness={aug['color_jitter']['brightness']}, "
          f"contrast={aug['color_jitter']['contrast']}, saturation={aug['color_jitter']['saturation']}), "
          f"random_resized_crop scale=({aug['random_resized_crop']['scale_min']},"
          f"{aug['random_resized_crop']['scale_max']}), vertical_flip=OFF, random_erasing=OFF")
    print(f"model architecture    : BaselineCNN - 4 conv blocks (16/32/64/128 ch) + "
          f"GlobalAvgPool + Dropout(0.5) + Linear(128,{config.num_classes})")
    print(f"parameter count       : {count_parameters(model):,}")
    print(f"optimizer             : AdamW (lr={config.learning_rate}, weight_decay={config.weight_decay})")
    print(f"lr scheduler          : ReduceLROnPlateau (factor={config.lr_scheduler_factor}, "
          f"patience={config.lr_scheduler_patience}, mode=max)")
    print(f"batch size            : {config.batch_size}")
    print(f"max epochs            : {config.max_epochs}")
    print(f"early stopping        : patience={config.early_stopping_patience} epochs on {config.primary_metric}")
    print(f"min_delta             : {config.min_delta}")
    print(f"primary metric        : {config.primary_metric}")
    print(f"checkpoint selection  : highest {config.primary_metric} "
          f"(min improvement {config.min_delta}); ties broken by lowest val_loss")
    print(f"device                : {device}")
    print(f"experiment dir        : {config.experiment_dir}")
    print("=" * 72)


def main() -> None:
    run_label = sys.argv[1] if len(sys.argv) > 1 else None
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    if run_label:
        run_id = f"{run_id}_{run_label.lower().replace(' ', '_')}"
    experiment_dir = REPO_ROOT / "experiments" / "baseline_cnn" / run_id
    # num_workers=0 (not the TrainingConfig default of 4): a real DataLoader worker
    # crash was observed under num_workers=4 (OSError: broken data stream), but a full
    # sequential single-process decode of all 772 manifest images found zero failures —
    # this points to a transient OneDrive-sync/multiprocessing interaction on this
    # environment, not real file corruption. num_workers has no effect on what data is
    # trained on or in what order, so this is an infrastructure choice, not a change to
    # any experimental setting.
    config = TrainingConfig(experiment_dir=experiment_dir, num_workers=0)

    # Seed BEFORE constructing anything that draws from the global RNG. fit() also
    # calls set_seed() as its first line (so it stays correct when called directly,
    # e.g. in tests) — but BaselineCNN's default weight init happens here in main(),
    # before fit() is ever called, so without this line the model's initial weights
    # are never actually seeded. Confirmed by direct test: two BaselineCNN() calls in
    # the same unseeded process get different weights; two calls each preceded by
    # set_seed(42) get identical weights. This is why the first two full runs
    # (run_20260818_002504, run_20260818_005718) produced different results despite
    # both claiming seed=42 — not GPU non-determinism, a real ordering bug.
    set_seed(config.seed)

    class_to_idx = json.loads((SPLITS_DIR / "class_mapping.json").read_text())
    class_names = sorted(class_to_idx, key=class_to_idx.get)
    preprocessing_config = load_config(config.preprocessing_config_path)

    split_counts = {}
    for split_name, fname in [("Train", "train.csv"), ("Valid", "val.csv"), ("Test", "test.csv")]:
        split_counts[split_name] = len(pd.read_csv(SPLITS_DIR / fname))

    train_loader, val_loader = build_train_val_dataloaders(
        SPLITS_DIR, IMAGES_ROOT, config.preprocessing_config_path,
        batch_size=config.batch_size, num_workers=config.num_workers, seed=config.seed,
    )

    device = resolve_device(config.device)
    model = BaselineCNN(num_classes=config.num_classes, in_channels=config.in_channels)

    class_weights = class_weights_from_train_split(SPLITS_DIR).to(device)
    loss_fn = build_loss(class_weights)

    print_manifest(config, class_to_idx, class_names, split_counts, preprocessing_config, model, device, run_label)

    print("\nstarting training...\n")
    result = fit(model, train_loader, val_loader, loss_fn, config, class_names)

    # --- save artifacts (Phase 3 §7) ---
    experiment_dir.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(result["history"])
    history_df.to_csv(experiment_dir / "history.csv", index=False)

    config.save(experiment_dir / "config.yaml")
    (experiment_dir / "class_mapping.json").write_text(json.dumps(class_to_idx, indent=2))

    best_row = history_df[history_df["epoch"] == result["best_epoch"]].iloc[0].to_dict()
    final_row = history_df.iloc[-1].to_dict()
    metrics_summary = {
        "best_epoch": result["best_epoch"],
        "best_val_macro_f1": best_row["val_macro_f1"],
        "best_val_loss": best_row["val_loss"],
        "final_epoch": int(final_row["epoch"]),
        "final_train_loss": final_row["train_loss"],
        "final_val_loss": final_row["val_loss"],
        "final_val_accuracy": final_row["val_accuracy"],
        "final_val_macro_f1": final_row["val_macro_f1"],
        "final_val_weighted_f1": final_row["val_weighted_f1"],
        "stopped_early": result["stopped_early"],
        "epochs_ran": result["epochs_ran"],
        "total_time_seconds": result["total_time_seconds"],
        "checkpoint_path": result["checkpoint_path"],
    }
    (experiment_dir / "metrics.json").write_text(json.dumps(metrics_summary, indent=2))

    # --- curves ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    axes[0].axvline(result["best_epoch"], color="gray", linestyle="--", alpha=0.6, label="best epoch")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].set_title("Loss"); axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["val_macro_f1"], label="val_macro_f1")
    axes[1].plot(history_df["epoch"], history_df["val_accuracy"], label="val_accuracy", alpha=0.6)
    axes[1].axvline(result["best_epoch"], color="gray", linestyle="--", alpha=0.6, label="best epoch")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("score"); axes[1].set_title("Validation Macro-F1 vs Accuracy"); axes[1].legend()

    fig.tight_layout()
    curves_path = experiment_dir / "training_curves.png"
    fig.savefig(curves_path, dpi=120)
    plt.close(fig)

    # --- per-class metrics + confusion matrix from the SELECTED BEST checkpoint ---
    # fit()'s history only retains aggregate metrics; evaluate()'s full per-class/
    # confusion-matrix output is recomputed here from the reloaded best checkpoint,
    # not from training's last epoch — reloading also re-verifies the checkpoint
    # round-trips correctly rather than assuming it.
    best_model = BaselineCNN(num_classes=config.num_classes, in_channels=config.in_channels)
    checkpoint = load_checkpoint(result["checkpoint_path"], best_model, map_location=device)
    best_model.to(device)
    best_eval = evaluate(best_model, val_loader, loss_fn, device, class_names)

    assert checkpoint["epoch"] == result["best_epoch"]
    assert abs(best_eval["macro_f1"] - best_row["val_macro_f1"]) < 1e-6, (
        "reloaded checkpoint's macro_f1 does not match the recorded history row — "
        "checkpoint save/load is not faithful"
    )
    assert abs(best_eval["val_loss"] - best_row["val_loss"]) < 1e-6

    classification_report = {
        "epoch": checkpoint["epoch"],
        "accuracy": best_eval["accuracy"],
        "macro_f1": best_eval["macro_f1"],
        "weighted_f1": best_eval["weighted_f1"],
        "per_class": best_eval["per_class"],
    }
    (experiment_dir / "classification_report.json").write_text(json.dumps(classification_report, indent=2))

    cm = np.array(best_eval["confusion_matrix"])
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(experiment_dir / "confusion_matrix.csv")

    fig_cm, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — best checkpoint (epoch {checkpoint['epoch']})")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig_cm.colorbar(im)
    fig_cm.tight_layout()
    confusion_matrix_path = experiment_dir / "confusion_matrix.png"
    fig_cm.savefig(confusion_matrix_path, dpi=120)
    plt.close(fig_cm)

    # --- final report ---
    print("\n" + "=" * 72)
    print("BASELINE EXPERIMENT COMPLETE")
    print("=" * 72)
    print(f"best epoch                 : {result['best_epoch']}")
    print(f"best val macro_f1          : {best_row['val_macro_f1']:.4f}")
    print(f"best val loss (at best ep) : {best_row['val_loss']:.4f}")
    print(f"parameter count            : {count_parameters(model):,}")
    print(f"final epoch ran            : {final_row['epoch']:.0f}")
    print(f"final train_loss           : {final_row['train_loss']:.4f}")
    print(f"final val_loss             : {final_row['val_loss']:.4f}")
    print(f"final val_accuracy         : {final_row['val_accuracy']:.4f}")
    print(f"final val_macro_f1         : {final_row['val_macro_f1']:.4f}")
    print(f"final val_weighted_f1      : {final_row['val_weighted_f1']:.4f}")
    print(f"early stopping triggered   : {result['stopped_early']}")
    print(f"total training time        : {result['total_time_seconds']:.1f}s "
          f"({result['total_time_seconds']/60:.1f} min)")
    print(f"epochs run                 : {result['epochs_ran']} / {config.max_epochs} max")
    print(f"checkpoint (best)          : {result['checkpoint_path']}")
    print(f"history.csv                : {experiment_dir / 'history.csv'}")
    print(f"metrics.json               : {experiment_dir / 'metrics.json'}")
    print(f"config.yaml                : {experiment_dir / 'config.yaml'}")
    print(f"training_curves.png        : {curves_path}")
    print(f"classification_report.json : {experiment_dir / 'classification_report.json'}")
    print(f"confusion_matrix.csv/.png  : {confusion_matrix_path}")
    print("\nper-class metrics (best checkpoint):")
    for cls, m in best_eval["per_class"].items():
        print(f"  {cls:20s} precision={m['precision']:.3f}  recall={m['recall']:.3f}  "
              f"f1={m['f1']:.3f}  support={m['support']}")
    print(f"\ntest set NEVER loaded (no test DataLoader was constructed — build_train_val_dataloaders "
          f"only builds train/val), NEVER evaluated, and NOT used for model selection in this run.")
    print("=" * 72)


if __name__ == "__main__":
    main()
