"""Phase 3 smoke test: proves the real pipeline (real images, real model, real
training/eval/checkpoint code) works end-to-end on a tiny slice of data, WITHOUT
committing to the full training run. Not a substitute for the real experiment —
just proof nothing is structurally broken before spending the time on one.

Run: PYTHONPATH=src python scripts/smoke_test_phase3.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from solarguard.data.datasets import SolarGuardDataset
from solarguard.evaluation.metrics import compute_metrics
from solarguard.models.baseline_cnn import BaselineCNN, count_parameters
from solarguard.preprocessing.transforms import build_eval_transform, build_train_transform, load_config
from solarguard.training.checkpoint import load_checkpoint, save_checkpoint
from solarguard.training.config import TrainingConfig
from solarguard.training.engine import evaluate, resolve_device, train_one_epoch
from solarguard.training.losses import build_loss, class_weights_from_train_split
from solarguard.training.train import fit

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = REPO_ROOT / "data" / "splits"
IMAGES_ROOT = REPO_ROOT / "data" / "candidates" / "PV_Panel_Defect_Dataset"


def main() -> None:
    config = TrainingConfig(batch_size=8)
    device = resolve_device(config.device)
    print(f"device: {device}")

    class_to_idx = json.loads((SPLITS_DIR / "class_mapping.json").read_text())
    class_names = sorted(class_to_idx, key=class_to_idx.get)
    preprocessing_config = load_config(config.preprocessing_config_path)

    # tiny slice — 16 real train images (2 batches of 8), 8 real val images (1 batch)
    train_df = pd.read_csv(SPLITS_DIR / "train.csv").head(16)
    val_df = pd.read_csv(SPLITS_DIR / "val.csv").head(8)
    print(f"smoke-test slice: {len(train_df)} train images, {len(val_df)} val images (real files, real labels)")

    train_dataset = SolarGuardDataset(train_df, IMAGES_ROOT, class_to_idx, build_train_transform(preprocessing_config))
    val_dataset = SolarGuardDataset(val_df, IMAGES_ROOT, class_to_idx, build_eval_transform(preprocessing_config))
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    print(f"train batches this epoch: {len(train_loader)}, val batches: {len(val_loader)}")

    model = BaselineCNN(num_classes=len(class_names)).to(device)
    print(f"model parameters: {count_parameters(model):,}")

    class_weights = class_weights_from_train_split(SPLITS_DIR).to(device)
    loss_fn = build_loss(class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    before = [p.clone() for p in model.parameters()]
    train_result = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
    after = list(model.parameters())
    changed = any(not torch.equal(b, a) for b, a in zip(before, after))
    print(f"\ntrain_one_epoch result: {train_result}")
    print(f"weights actually changed: {changed}")
    assert changed, "SMOKE TEST FAILED: no weights changed after a training step"

    val_result = evaluate(model, val_loader, loss_fn, device, class_names)
    print(f"\nevaluate() result: accuracy={val_result['accuracy']:.3f} "
          f"macro_f1={val_result['macro_f1']:.3f} val_loss={val_result['val_loss']:.4f}")
    assert set(val_result["per_class"].keys()) == set(class_names)
    assert len(val_result["confusion_matrix"]) == len(class_names)

    ckpt_path = REPO_ROOT / "experiments" / "baseline_cnn" / "_smoke_test_checkpoint.pt"
    save_checkpoint(ckpt_path, model, optimizer, epoch=1, config=config,
                     best_metric_value=val_result["macro_f1"], class_mapping=class_to_idx)
    fresh_model = BaselineCNN(num_classes=len(class_names)).to(device)
    checkpoint = load_checkpoint(ckpt_path, fresh_model, map_location=device)
    for p1, p2 in zip(model.parameters(), fresh_model.parameters()):
        assert torch.equal(p1, p2)
    print(f"\ncheckpoint save/load round-trip verified: {ckpt_path}")
    ckpt_path.unlink()  # this was only a smoke-test artifact, not a real experiment result

    # --- also exercise fit() itself (the actual orchestrator, incl. min_delta logic) ---
    # not covered by the checks above, which call train_one_epoch/evaluate directly
    fit_experiment_dir = REPO_ROOT / "experiments" / "baseline_cnn" / "_smoke_test_fit"
    fit_config = TrainingConfig(batch_size=8, max_epochs=2, experiment_dir=fit_experiment_dir)
    fit_model = BaselineCNN(num_classes=len(class_names))
    fit_result = fit(fit_model, train_loader, val_loader, loss_fn, fit_config, class_names)
    print(f"\nfit() smoke run (2 epochs, real orchestrator incl. min_delta logic): "
          f"best_epoch={fit_result['best_epoch']} best_value={fit_result['best_value']:.4f} "
          f"epochs_ran={fit_result['epochs_ran']}")
    assert len(fit_result["history"]) == fit_result["epochs_ran"]
    assert Path(fit_result["checkpoint_path"]).exists(), "fit() did not save a checkpoint"
    import shutil
    shutil.rmtree(fit_experiment_dir)  # smoke-test artifact only

    print("\nSMOKE TEST PASSED — real data, real model, real train/eval/checkpoint/fit code all work end-to-end.")
    print("This did NOT run the full baseline experiment. Awaiting approval before that.")


if __name__ == "__main__":
    main()
