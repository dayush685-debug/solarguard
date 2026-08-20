"""Multi-seed comparison runner over three arms: baseline (weighted CE), Experiment 2
(focal loss, gamma=2), and Experiment 3 (rotation augmentation disabled).

Seeding fix vs notebook cell 12: set_seed(seed) runs IMMEDIATELY BEFORE BaselineCNN(...)
is constructed, so weight initialization is actually governed by the requested seed.
Cell 12 built the model first and relied on fit() seeding afterwards, leaving init
dependent on whatever RNG earlier cells had consumed.

Nothing else differs between the two arms: same architecture, optimizer, LR, scheduler,
batch size, split, preprocessing, augmentation, and selection rule.

Test data is never loaded -- build_train_val_dataloaders has no test code path.

Run: PYTHONPATH=src python scripts/run_seed_sweep.py --arm baseline
     PYTHONPATH=src python scripts/run_seed_sweep.py --arm focal
     PYTHONPATH=src python scripts/run_seed_sweep.py --arm norot
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from solarguard.data.datasets import build_train_val_dataloaders
from solarguard.evaluation.metrics import compute_metrics
from solarguard.models.baseline_cnn import BaselineCNN, count_parameters
from solarguard.training.checkpoint import load_checkpoint
from solarguard.training.config import TrainingConfig
from solarguard.training.engine import evaluate, resolve_device
from solarguard.training.losses import build_focal_loss, build_loss, class_weights_from_train_split
from solarguard.training.train import fit, set_seed

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = REPO_ROOT / "data" / "splits"
IMAGES_ROOT = REPO_ROOT / "data" / "candidates" / "PV_Panel_Defect_Dataset"
SEEDS = [42, 123, 456]
FOCAL_GAMMA = 2.0

# arm -> (output dir, preprocessing config override or None for the default)
ARMS = {
    "baseline": ("baseline_3seed", None),                 # weighted CE (== focal gamma=0)
    "focal": ("exp2_focal_gamma2", None),                 # Exp 2: weighted focal, gamma=2
    "norot": ("exp3_no_rotation",                         # Exp 3: rotation disabled
              REPO_ROOT / "configs" / "preprocessing_norot.yaml"),
}


def run_one(arm: str, seed: int, device) -> dict:
    dir_name, config_override = ARMS[arm]
    experiment_dir = REPO_ROOT / "experiments" / dir_name / f"seed_{seed}"
    kwargs = dict(seed=seed, experiment_dir=experiment_dir, num_workers=0)
    if config_override is not None:
        kwargs["preprocessing_config_path"] = config_override
    config = TrainingConfig(**kwargs)

    class_to_idx = json.loads((SPLITS_DIR / "class_mapping.json").read_text())
    class_names = sorted(class_to_idx, key=class_to_idx.get)

    # ---------------- THE SEEDING FIX ----------------
    # set_seed BEFORE model construction, so init is governed by `seed`.
    set_seed(config.seed)
    model = BaselineCNN(num_classes=config.num_classes, in_channels=config.in_channels)
    # -------------------------------------------------

    train_loader, val_loader = build_train_val_dataloaders(
        SPLITS_DIR, IMAGES_ROOT, config.preprocessing_config_path,
        batch_size=config.batch_size, num_workers=config.num_workers, seed=config.seed,
    )

    class_weights = class_weights_from_train_split(SPLITS_DIR).to(device)
    # ONLY the focal arm changes the loss. baseline and norot both use weighted CE,
    # so Experiment 3's single changed variable really is rotation and nothing else.
    loss_fn = (
        build_focal_loss(class_weights, gamma=FOCAL_GAMMA) if arm == "focal"
        else build_loss(class_weights)
    )

    print(f"\n{'='*70}\narm={arm}  seed={seed}  loss={type(loss_fn).__name__}"
          f"{f' (gamma={FOCAL_GAMMA})' if arm == 'focal' else ''}"
          f"\npreprocessing={config.preprocessing_config_path.name}"
          f"\nparams={count_parameters(model):,}  dir={experiment_dir}\n{'='*70}")

    result = fit(model, train_loader, val_loader, loss_fn, config, class_names)

    # ---- artifacts (cell 12 saved only checkpoint_best.pt; these are required) ----
    experiment_dir.mkdir(parents=True, exist_ok=True)
    history = pd.DataFrame(result["history"])
    history.to_csv(experiment_dir / "history.csv", index=False)
    config.save(experiment_dir / "config.yaml")
    (experiment_dir / "class_mapping.json").write_text(json.dumps(class_to_idx, indent=2))

    # per-class metrics from the SELECTED best checkpoint, not the last epoch
    best_model = BaselineCNN(num_classes=config.num_classes, in_channels=config.in_channels)
    ckpt = load_checkpoint(result["checkpoint_path"], best_model, map_location=device)
    best_model.to(device)
    best = evaluate(best_model, val_loader, loss_fn, device, class_names)
    assert ckpt["epoch"] == result["best_epoch"]

    (experiment_dir / "classification_report.json").write_text(json.dumps({
        "epoch": ckpt["epoch"], "accuracy": best["accuracy"],
        "macro_f1": best["macro_f1"], "weighted_f1": best["weighted_f1"],
        "per_class": best["per_class"],
    }, indent=2))

    cm = np.array(best["confusion_matrix"])
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(experiment_dir / "confusion_matrix.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history["epoch"], history["train_loss"], label="train_loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="val_loss")
    axes[0].axvline(result["best_epoch"], color="gray", ls="--", alpha=.6, label="best")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend()
    axes[0].set_title(f"{arm} seed={seed} loss")
    axes[1].plot(history["epoch"], history["val_macro_f1"], label="val_macro_f1")
    axes[1].axvline(result["best_epoch"], color="gray", ls="--", alpha=.6)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("macro F1"); axes[1].legend()
    axes[1].set_title(f"{arm} seed={seed} macro-F1")
    fig.tight_layout(); fig.savefig(experiment_dir / "training_curves.png", dpi=120); plt.close(fig)

    row = {
        "arm": arm, "seed": seed, "best_epoch": result["best_epoch"],
        "val_macro_f1": best["macro_f1"], "val_accuracy": best["accuracy"],
        "val_weighted_f1": best["weighted_f1"], "val_loss": best["val_loss"],
        "epochs_ran": result["epochs_ran"], "stopped_early": result["stopped_early"],
        "total_time_seconds": result["total_time_seconds"],
    }
    for name in class_names:
        row[f"f1_{name}"] = best["per_class"][name]["f1"]
        row[f"precision_{name}"] = best["per_class"][name]["precision"]
        row[f"recall_{name}"] = best["per_class"][name]["recall"]
    (experiment_dir / "metrics.json").write_text(json.dumps(row, indent=2))
    print(f"  best_epoch={row['best_epoch']}  macro_f1={row['val_macro_f1']:.4f}  "
          f"acc={row['val_accuracy']:.4f}  wF1={row['val_weighted_f1']:.4f}")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    args = ap.parse_args()

    device = resolve_device("cuda")
    rows = [run_one(args.arm, s, device) for s in SEEDS]

    out = REPO_ROOT / "experiments" / ARMS[args.arm][0] / "summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    m = pd.DataFrame(rows)["val_macro_f1"]
    print(f"\n{'='*70}\narm={args.arm}  macro-F1 over {len(SEEDS)} seeds: "
          f"mean={m.mean():.4f}  std={m.std(ddof=1):.4f}  min={m.min():.4f}  max={m.max():.4f}")
    print(f"summary -> {out}\n{'='*70}")


if __name__ == "__main__":
    main()
