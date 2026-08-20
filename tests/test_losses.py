"""Focused tests for src/solarguard/training/losses.py."""

import json
import math
from pathlib import Path

import pandas as pd
import pytest
import torch

from solarguard.training.losses import (build_focal_loss, build_loss,
                                        class_weights_from_train_split, compute_class_weights)

ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT / "data" / "splits"


def test_balanced_classes_all_get_weight_one():
    """If every class has equal representation, inverse-frequency weighting should
    reduce to a no-op — every weight exactly 1.0."""
    labels = pd.Series(["a", "b", "c"] * 10)  # 10 of each, 30 total
    weights = compute_class_weights(labels, {"a": 0, "b": 1, "c": 2})
    assert torch.allclose(weights, torch.ones(3))


def test_weights_match_hand_computed_formula():
    # 6 'a', 3 'b', 1 'c' -> n=10, n_classes=3
    labels = pd.Series(["a"] * 6 + ["b"] * 3 + ["c"] * 1)
    weights = compute_class_weights(labels, {"a": 0, "b": 1, "c": 2})
    assert weights[0].item() == pytest.approx(10 / (3 * 6))
    assert weights[1].item() == pytest.approx(10 / (3 * 3))
    assert weights[2].item() == pytest.approx(10 / (3 * 1))


def test_rarer_class_gets_larger_weight():
    labels = pd.Series(["common"] * 90 + ["rare"] * 10)
    weights = compute_class_weights(labels, {"common": 0, "rare": 1})
    assert weights[1] > weights[0]


def test_missing_class_raises_not_silently_produces_nan():
    labels = pd.Series(["a", "a", "b"])
    with pytest.raises(ValueError, match="zero examples"):
        compute_class_weights(labels, {"a": 0, "b": 1, "c": 2})


def test_class_weights_from_train_split_matches_phase2_cached_values():
    """This computation must agree with the values Phase 2's compute_statistics.py
    already saved to dataset_statistics.json — proving this is a faithful
    reimplementation, not a silently-diverged duplicate."""
    weights = class_weights_from_train_split(SPLITS_DIR)
    class_to_idx = json.loads((SPLITS_DIR / "class_mapping.json").read_text())

    stats = json.loads((ROOT / "data" / "final" / "dataset_statistics.json").read_text())
    expected = stats["class_weights_inverse_frequency"]["weights"]

    for cls, idx in class_to_idx.items():
        assert weights[idx].item() == pytest.approx(expected[cls], abs=1e-4), (
            f"weight for {cls} diverged from the Phase 2 cached value"
        )


def test_class_weights_from_train_split_never_touches_val_or_test():
    """Loading only train.csv is the whole leakage guard — assert the function's
    computed total matches train.csv's row count, not val/test's."""
    weights = class_weights_from_train_split(SPLITS_DIR)
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    class_to_idx = json.loads((SPLITS_DIR / "class_mapping.json").read_text())
    recomputed = compute_class_weights(train_df["class"], class_to_idx)
    assert torch.equal(weights, recomputed)


def test_build_loss_is_weighted_cross_entropy():
    weights = torch.tensor([1.0, 2.0, 0.5])
    loss_fn = build_loss(weights)
    assert isinstance(loss_fn, torch.nn.CrossEntropyLoss)
    assert torch.equal(loss_fn.weight, weights)


def test_weighted_loss_matches_hand_computed_value():
    """Two examples, 2 classes, weights [1.0, 3.0]. Verify PyTorch's weighted-mean
    reduction against a manual computation of the cross-entropy formula."""
    weights = torch.tensor([1.0, 3.0])
    loss_fn = build_loss(weights)

    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])  # example 0 predicts class 0 strongly, example 1 predicts class 1 strongly
    targets = torch.tensor([0, 1])  # both predictions are correct

    # hand-computed cross-entropy per example: -log(softmax(logits)[target])
    log_probs = torch.log_softmax(logits, dim=1)
    l0 = -log_probs[0, 0].item()
    l1 = -log_probs[1, 1].item()

    # PyTorch's weighted mean: sum(weight[y_i] * L_i) / sum(weight[y_i])
    expected = (weights[0].item() * l0 + weights[1].item() * l1) / (weights[0].item() + weights[1].item())

    actual = loss_fn(logits, targets).item()
    assert actual == pytest.approx(expected, abs=1e-5)


def test_focal_loss_gamma_zero_is_exactly_weighted_cross_entropy():
    """The scientific foundation of Experiment 2: gamma is a single knob on top of the
    baseline objective, not a different loss. At gamma=0 the focal loss must reproduce
    build_loss() in both value AND gradient -- gradient equality is the stronger claim,
    since that is what actually drives training."""
    weights = class_weights_from_train_split(SPLITS_DIR)
    ce = build_loss(weights)
    focal0 = build_focal_loss(weights, gamma=0.0)

    g = torch.Generator().manual_seed(0)
    for _ in range(10):
        logits = torch.randn(32, len(weights), generator=g)
        targets = torch.randint(0, len(weights), (32,), generator=g)
        assert ce(logits, targets).item() == pytest.approx(focal0(logits, targets).item(), abs=1e-5)

    logits = torch.randn(32, len(weights), generator=g, requires_grad=True)
    targets = torch.randint(0, len(weights), (32,), generator=g)
    ce(logits, targets).backward()
    grad_ce = logits.grad.clone()
    logits.grad = None
    focal0(logits, targets).backward()
    assert torch.allclose(grad_ce, logits.grad, atol=1e-6)


def test_focal_loss_gamma_two_actually_changes_the_objective():
    """Guards against a no-op experiment: gamma=2 must produce different gradients,
    and must down-weight easy examples far more than hard ones."""
    weights = class_weights_from_train_split(SPLITS_DIR)
    ce = build_loss(weights)
    focal2 = build_focal_loss(weights, gamma=2.0)

    g = torch.Generator().manual_seed(1)
    logits = torch.randn(32, len(weights), generator=g, requires_grad=True)
    targets = torch.randint(0, len(weights), (32,), generator=g)
    ce(logits, targets).backward()
    grad_ce = logits.grad.clone()
    logits.grad = None
    focal2(logits, targets).backward()
    assert not torch.allclose(grad_ce, logits.grad, atol=1e-6)

    easy = torch.tensor([[8.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    hard = torch.tensor([[1.0, 0.9, 0.0, 0.0, 0.0, 0.0]])
    t = torch.tensor([0])
    easy_ratio = focal2(easy, t).item() / ce(easy, t).item()
    hard_ratio = focal2(hard, t).item() / ce(hard, t).item()
    assert easy_ratio < hard_ratio
    assert easy_ratio < 0.01


def test_focal_loss_rejects_negative_gamma():
    weights = class_weights_from_train_split(SPLITS_DIR)
    with pytest.raises(ValueError, match="gamma"):
        build_focal_loss(weights, gamma=-1.0)
