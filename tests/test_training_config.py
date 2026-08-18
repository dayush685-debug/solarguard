"""Focused tests for src/solarguard/training/config.py."""

import dataclasses
import json
from pathlib import Path

import pytest
import yaml

from solarguard.training.config import TrainingConfig

# ---- default values ----


def test_defaults_match_approved_phase3_design():
    config = TrainingConfig()
    assert config.learning_rate == 1e-3
    assert config.weight_decay == 1e-4
    assert config.batch_size == 32
    assert config.max_epochs == 100
    assert config.early_stopping_patience == 15
    assert config.primary_metric == "macro_f1"
    assert config.selection_tiebreaker == "val_loss"
    assert config.seed == 42
    assert config.num_classes == 6
    assert config.lr_scheduler_factor == 0.5
    assert config.lr_scheduler_patience == 7


def test_default_min_delta_is_documented_value():
    assert TrainingConfig().min_delta == 1e-4


def test_default_paths_point_inside_the_repo():
    config = TrainingConfig()
    assert config.splits_dir.name == "splits"
    assert config.splits_dir.exists(), "default splits_dir should resolve to the real data/splits"
    assert config.preprocessing_config_path.exists()


# ---- valid configuration ----


def test_valid_custom_configuration_accepted():
    config = TrainingConfig(learning_rate=5e-4, batch_size=16, max_epochs=50)
    assert config.learning_rate == 5e-4
    assert config.batch_size == 16
    assert config.max_epochs == 50


def test_weighted_f1_is_an_allowed_primary_metric():
    config = TrainingConfig(primary_metric="weighted_f1")
    assert config.primary_metric == "weighted_f1"


def test_config_is_immutable():
    config = TrainingConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.batch_size = 999  # type: ignore[misc]


# ---- invalid values ----


@pytest.mark.parametrize("bad_lr", [0, -1e-3])
def test_non_positive_learning_rate_rejected(bad_lr):
    with pytest.raises(ValueError, match="learning_rate"):
        TrainingConfig(learning_rate=bad_lr)


def test_negative_weight_decay_rejected():
    with pytest.raises(ValueError, match="weight_decay"):
        TrainingConfig(weight_decay=-0.01)


@pytest.mark.parametrize("bad_batch", [0, -8])
def test_non_positive_batch_size_rejected(bad_batch):
    with pytest.raises(ValueError, match="batch_size"):
        TrainingConfig(batch_size=bad_batch)


def test_zero_max_epochs_rejected():
    with pytest.raises(ValueError, match="max_epochs"):
        TrainingConfig(max_epochs=0)


def test_lr_scheduler_factor_out_of_range_rejected():
    with pytest.raises(ValueError, match="lr_scheduler_factor"):
        TrainingConfig(lr_scheduler_factor=1.5)


def test_single_class_rejected():
    with pytest.raises(ValueError, match="num_classes"):
        TrainingConfig(num_classes=1)


def test_negative_min_delta_rejected():
    with pytest.raises(ValueError, match="min_delta"):
        TrainingConfig(min_delta=-1e-4)


def test_zero_min_delta_is_allowed():
    """Zero is a legitimate choice (reverts to the original strict-> behavior) —
    only negative values are nonsensical."""
    config = TrainingConfig(min_delta=0.0)
    assert config.min_delta == 0.0


def test_min_delta_is_overridable():
    config = TrainingConfig(min_delta=1e-3)
    assert config.min_delta == 1e-3


# ---- metric / model-selection configuration ----


def test_accuracy_as_primary_metric_is_rejected_with_explanation():
    """Hard project rule: accuracy must never be the primary selection metric.
    Enforced structurally, not just documented."""
    with pytest.raises(ValueError, match="accuracy"):
        TrainingConfig(primary_metric="accuracy")


def test_unknown_primary_metric_rejected():
    with pytest.raises(ValueError, match="primary_metric"):
        TrainingConfig(primary_metric="f1_micro")


def test_unknown_tiebreaker_rejected():
    with pytest.raises(ValueError, match="selection_tiebreaker"):
        TrainingConfig(selection_tiebreaker="test_loss")


# ---- reproducibility-related settings ----


def test_seed_is_recorded_and_overridable():
    assert TrainingConfig().seed == 42
    assert TrainingConfig(seed=123).seed == 123


def test_to_dict_is_fully_serializable_json():
    config = TrainingConfig()
    d = config.to_dict()
    json.dumps(d)  # must not raise — proves every value is JSON-safe
    assert isinstance(d["splits_dir"], str)


def test_save_and_reload_yaml_round_trips(tmp_path: Path):
    config = TrainingConfig(learning_rate=2e-3, seed=7)
    out = tmp_path / "config.yaml"
    config.save(out)

    loaded = yaml.safe_load(out.read_text())
    assert loaded["learning_rate"] == 2e-3
    assert loaded["seed"] == 7
    assert loaded["primary_metric"] == "macro_f1"


def test_save_json_variant(tmp_path: Path):
    config = TrainingConfig()
    out = tmp_path / "config.json"
    config.save(out)
    loaded = json.loads(out.read_text())
    assert loaded["batch_size"] == 32
