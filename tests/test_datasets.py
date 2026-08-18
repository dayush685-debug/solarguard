"""Focused tests for src/solarguard/data/datasets.py — uses the real Phase 2 manifest
and real images, since this is the boundary where Phase 2's artifacts actually meet
PyTorch for the first time."""

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from solarguard.data.datasets import SolarGuardDataset, build_train_val_dataloaders
from solarguard.preprocessing.transforms import build_eval_transform, load_config

ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT / "data" / "splits"
IMAGES_ROOT = ROOT / "data" / "candidates" / "PV_Panel_Defect_Dataset"
PREPROCESSING_CONFIG = load_config(ROOT / "configs" / "preprocessing.yaml")


def test_dataset_length_matches_manifest():
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    class_to_idx = json.loads((SPLITS_DIR / "class_mapping.json").read_text())
    dataset = SolarGuardDataset(train_df, IMAGES_ROOT, class_to_idx, build_eval_transform(PREPROCESSING_CONFIG))
    assert len(dataset) == 540


def test_dataset_getitem_returns_correct_shape_and_valid_label():
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    class_to_idx = json.loads((SPLITS_DIR / "class_mapping.json").read_text())
    dataset = SolarGuardDataset(train_df, IMAGES_ROOT, class_to_idx, build_eval_transform(PREPROCESSING_CONFIG))

    image, label = dataset[0]
    assert image.shape == (3, 224, 224)
    assert isinstance(label, int)
    assert 0 <= label < len(class_to_idx)


def test_dataset_raises_clear_error_on_missing_file(tmp_path):
    fake_manifest = pd.DataFrame({"path": ["does/not/exist.jpg"], "class": ["Clean"]})
    dataset = SolarGuardDataset(fake_manifest, tmp_path, {"Clean": 0}, build_eval_transform(PREPROCESSING_CONFIG))
    with pytest.raises(RuntimeError, match="failed to load image"):
        dataset[0]


def test_build_train_val_dataloaders_produces_correct_batch_shapes():
    train_loader, val_loader = build_train_val_dataloaders(
        SPLITS_DIR, IMAGES_ROOT, ROOT / "configs" / "preprocessing.yaml",
        batch_size=8, num_workers=0, seed=42,
    )
    images, labels = next(iter(train_loader))
    assert images.shape == (8, 3, 224, 224)
    assert labels.shape == (8,)

    images, labels = next(iter(val_loader))
    assert images.shape[1:] == (3, 224, 224)


def test_train_loader_shuffles_val_loader_does_not():
    train_loader, val_loader = build_train_val_dataloaders(
        SPLITS_DIR, IMAGES_ROOT, ROOT / "configs" / "preprocessing.yaml",
        batch_size=8, num_workers=0, seed=42,
    )
    assert train_loader.sampler.__class__.__name__ != "SequentialSampler"
    from torch.utils.data import SequentialSampler
    assert isinstance(val_loader.sampler, SequentialSampler)


def test_no_test_dataloader_function_exists():
    """Structural enforcement of the test-set policy: there is no function in this
    module capable of building a test DataLoader during Phase 3."""
    import solarguard.data.datasets as datasets_module
    assert not hasattr(datasets_module, "build_test_dataloader")
    assert "test" not in build_train_val_dataloaders.__name__.lower()
