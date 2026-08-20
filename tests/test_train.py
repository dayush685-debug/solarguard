"""Focused tests for src/solarguard/training/train.py's _is_improvement — the
min_delta threshold and the val_loss tie-breaker."""

import pytest
import torch

from solarguard.models.baseline_cnn import BaselineCNN
from solarguard.training.train import _is_improvement, set_seed


def test_genuine_improvement_above_min_delta_counts():
    assert _is_improvement(0.60, 0.50, val_loss=1.0, best_val_loss=1.0, min_delta=1e-4) is True


def test_marginal_change_within_min_delta_does_not_count():
    """The exact Q5 scenario: 0.5012000001 vs 0.5012 is not a tie mathematically,
    but it's far smaller than min_delta and must not trigger a save."""
    assert _is_improvement(
        0.5012000001, 0.5012, val_loss=0.9, best_val_loss=1.0, min_delta=1e-4
    ) is False


def test_worse_value_never_counts_regardless_of_val_loss():
    assert _is_improvement(0.40, 0.50, val_loss=0.1, best_val_loss=1.0, min_delta=1e-4) is False


def test_exact_tie_broken_by_lower_val_loss():
    assert _is_improvement(0.50, 0.50, val_loss=0.8, best_val_loss=1.0, min_delta=1e-4) is True


def test_exact_tie_not_improved_if_val_loss_is_not_better():
    assert _is_improvement(0.50, 0.50, val_loss=1.2, best_val_loss=1.0, min_delta=1e-4) is False


def test_value_exactly_at_min_delta_boundary_does_not_count():
    """Strict > : best_value + min_delta itself is not a genuine improvement."""
    assert _is_improvement(0.5001, 0.50, val_loss=1.0, best_val_loss=1.0, min_delta=1e-4) is False


def test_value_just_above_min_delta_boundary_counts():
    assert _is_improvement(0.50011, 0.50, val_loss=1.0, best_val_loss=1.0, min_delta=1e-4) is True


def test_min_delta_zero_reverts_to_original_strict_greater_than():
    assert _is_improvement(0.5000001, 0.50, val_loss=1.0, best_val_loss=1.0, min_delta=0.0) is True


@pytest.mark.parametrize("min_delta", [1e-4, 1e-3, 0.01])
def test_larger_min_delta_filters_more_aggressively(min_delta):
    # a fixed small improvement of 5e-4 should only pass for small-enough min_delta
    result = _is_improvement(0.5005, 0.50, val_loss=1.0, best_val_loss=1.0, min_delta=min_delta)
    assert result == (0.0005 > min_delta)


def test_model_weight_init_is_reproducible_when_seeded_before_construction():
    """Regression test for a real bug found running the baseline twice: seed=42 was
    set inside fit(), but BaselineCNN() was constructed in the caller BEFORE fit()
    ran — so weight init was never actually seeded, and the two runs produced
    different results despite both claiming seed=42. This is the fix's contract:
    set_seed() must run before model construction, not after."""
    set_seed(42)
    model_a = BaselineCNN(num_classes=6)

    set_seed(42)
    model_b = BaselineCNN(num_classes=6)

    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(p_a, p_b)


def test_model_weight_init_is_not_reproducible_without_seeding():
    """The negative case, proving the above test isn't trivially true — construction
    order relative to seeding genuinely matters."""
    model_a = BaselineCNN(num_classes=6)
    model_b = BaselineCNN(num_classes=6)

    differs = any(
        not torch.equal(p_a, p_b) for p_a, p_b in zip(model_a.parameters(), model_b.parameters())
    )
    assert differs, "unseeded model construction produced identical weights — unexpected"
