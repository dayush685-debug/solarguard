"""Focused tests for src/solarguard/training/checkpoint.py."""

from pathlib import Path

import pytest
import torch
from torch import nn

from solarguard.training.checkpoint import load_checkpoint, save_checkpoint
from solarguard.training.config import TrainingConfig


def _tiny_model() -> nn.Module:
    return nn.Linear(4, 3)


def test_save_and_load_round_trip_restores_weights(tmp_path: Path):
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters())
    config = TrainingConfig(experiment_dir=tmp_path)
    ckpt_path = tmp_path / "checkpoint_best.pt"

    save_checkpoint(ckpt_path, model, optimizer, epoch=5, config=config,
                     best_metric_value=0.87, class_mapping={"a": 0, "b": 1, "c": 2})

    fresh_model = _tiny_model()
    fresh_optimizer = torch.optim.AdamW(fresh_model.parameters())
    checkpoint = load_checkpoint(ckpt_path, fresh_model, fresh_optimizer)

    for original, restored in zip(model.parameters(), fresh_model.parameters()):
        assert torch.equal(original, restored)
    assert checkpoint["epoch"] == 5
    assert checkpoint["best_metric_value"] == 0.87


def test_checkpoint_contains_all_required_fields(tmp_path: Path):
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters())
    config = TrainingConfig(experiment_dir=tmp_path, seed=99)
    ckpt_path = tmp_path / "checkpoint_best.pt"

    save_checkpoint(ckpt_path, model, optimizer, epoch=1, config=config,
                     best_metric_value=0.5, class_mapping={"a": 0})

    checkpoint = load_checkpoint(ckpt_path, _tiny_model())
    for field in ["model_state_dict", "optimizer_state_dict", "epoch", "config",
                  "best_metric_name", "best_metric_value", "class_mapping", "seed"]:
        assert field in checkpoint, f"missing field: {field}"
    assert checkpoint["seed"] == 99
    assert checkpoint["best_metric_name"] == "macro_f1"


def test_load_missing_checkpoint_raises_clear_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        load_checkpoint(tmp_path / "does_not_exist.pt", _tiny_model())


def test_load_without_optimizer_still_restores_model(tmp_path: Path):
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters())
    config = TrainingConfig(experiment_dir=tmp_path)
    ckpt_path = tmp_path / "checkpoint_best.pt"
    save_checkpoint(ckpt_path, model, optimizer, epoch=1, config=config,
                     best_metric_value=0.5, class_mapping={"a": 0})

    fresh_model = _tiny_model()
    checkpoint = load_checkpoint(ckpt_path, fresh_model)  # no optimizer passed
    for original, restored in zip(model.parameters(), fresh_model.parameters()):
        assert torch.equal(original, restored)
