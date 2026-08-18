"""Focused tests for src/solarguard/training/engine.py — synthetic data, no real
images, so these stay fast (<1s). Real-data behavior is covered in test_datasets.py
and the end-to-end smoke test."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from solarguard.training.engine import evaluate, resolve_device, train_one_epoch

CLASS_NAMES = ["a", "b", "c"]


def _tiny_model() -> nn.Module:
    # small enough to train instantly, big enough to have real gradients
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, len(CLASS_NAMES)))


def _fake_loader(n: int = 20, batch_size: int = 4) -> DataLoader:
    images = torch.randn(n, 3, 8, 8)
    labels = torch.randint(0, len(CLASS_NAMES), (n,))
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


def test_one_training_step_changes_weights():
    model = _tiny_model()
    loader = _fake_loader()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    before = [p.clone() for p in model.parameters()]
    train_one_epoch(model, loader, loss_fn, optimizer, torch.device("cpu"))
    after = list(model.parameters())

    assert any(not torch.equal(b, a) for b, a in zip(before, after)), (
        "no parameter changed after a training epoch — backward/step did not run"
    )


def test_training_step_returns_finite_positive_loss():
    model = _tiny_model()
    loader = _fake_loader()
    result = train_one_epoch(
        model, loader, nn.CrossEntropyLoss(), torch.optim.AdamW(model.parameters()), torch.device("cpu")
    )
    assert "train_loss" in result
    assert result["train_loss"] > 0
    assert torch.isfinite(torch.tensor(result["train_loss"]))


def test_training_sets_train_mode():
    model = _tiny_model()
    train_one_epoch(
        model, _fake_loader(), nn.CrossEntropyLoss(),
        torch.optim.AdamW(model.parameters()), torch.device("cpu"),
    )
    assert model.training is True


def test_one_validation_step_returns_full_metric_suite():
    model = _tiny_model()
    result = evaluate(model, _fake_loader(), nn.CrossEntropyLoss(), torch.device("cpu"), CLASS_NAMES)
    for key in ["val_loss", "accuracy", "macro_f1", "weighted_f1", "per_class", "confusion_matrix"]:
        assert key in result, f"missing metric: {key}"
    assert set(result["per_class"].keys()) == set(CLASS_NAMES)


def test_evaluate_respects_custom_loss_prefix():
    """evaluate()'s default is "val" (what Phase 3 actually uses everywhere) — this
    confirms the loss_prefix parameter's behavior without changing that default or
    evaluate()'s other semantics, closing the gap identified in the file-4 review:
    previously nothing exercised any value other than the default."""
    model = _tiny_model()
    result = evaluate(
        model, _fake_loader(), nn.CrossEntropyLoss(), torch.device("cpu"), CLASS_NAMES,
        loss_prefix="test",
    )
    assert "test_loss" in result
    assert "val_loss" not in result

    default_result = evaluate(model, _fake_loader(), nn.CrossEntropyLoss(), torch.device("cpu"), CLASS_NAMES)
    assert "val_loss" in default_result
    assert "test_loss" not in default_result


def test_evaluation_sets_eval_mode():
    model = _tiny_model()
    evaluate(model, _fake_loader(), nn.CrossEntropyLoss(), torch.device("cpu"), CLASS_NAMES)
    assert model.training is False


def test_evaluation_does_not_change_weights():
    model = _tiny_model()
    before = [p.clone() for p in model.parameters()]
    evaluate(model, _fake_loader(), nn.CrossEntropyLoss(), torch.device("cpu"), CLASS_NAMES)
    after = list(model.parameters())
    assert all(torch.equal(b, a) for b, a in zip(before, after)), (
        "a parameter changed during evaluate() — validation must never backpropagate"
    )


def test_evaluation_leaves_no_gradients():
    """If evaluate() ever computed gradients, .grad would be populated. It never
    calls backward(), so every .grad must remain exactly as it started: None."""
    model = _tiny_model()
    assert all(p.grad is None for p in model.parameters())
    evaluate(model, _fake_loader(), nn.CrossEntropyLoss(), torch.device("cpu"), CLASS_NAMES)
    assert all(p.grad is None for p in model.parameters())


def test_eval_mode_is_deterministic_across_repeated_calls():
    model = _tiny_model()
    model.eval()
    x = torch.randn(4, 3, 8, 8)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.equal(out1, out2)


def test_train_one_epoch_rejects_malformed_input_shape():
    model = _tiny_model()
    bad_images = torch.randn(20, 3, 8)  # missing a dimension — not (B,C,H,W)
    bad_labels = torch.randint(0, 3, (20,))
    loader = DataLoader(TensorDataset(bad_images, bad_labels), batch_size=4)
    try:
        train_one_epoch(model, loader, nn.CrossEntropyLoss(), torch.optim.AdamW(model.parameters()), torch.device("cpu"))
        assert False, "expected a ValueError for malformed input shape"
    except ValueError as e:
        assert "shape" in str(e)


def test_resolve_device_respects_explicit_cpu():
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_falls_back_when_cuda_unavailable(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device = resolve_device("cuda")
    assert device == torch.device("cpu")
    assert "WARNING" in capsys.readouterr().out
