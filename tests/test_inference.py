"""Tests for the deployment inference path.

The most important test here is
`test_serving_preprocessing_is_identical_to_validation_preprocessing` — if serving and
validation preprocessing ever diverge, every reported metric becomes a claim about a
pipeline that is not the one actually running.
"""

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from solarguard.inference.predictor import SolarGuardPredictor, resolve_inference_device
from solarguard.models.baseline_cnn import BaselineCNN
from solarguard.preprocessing.transforms import build_eval_transform, load_config

ROOT = Path(__file__).resolve().parent.parent
DEPLOY_CKPT = ROOT / "models" / "solarguard_baseline_v1.pt"
PREPROC = ROOT / "configs" / "preprocessing.yaml"
CLASSES = ["Bird-drop", "Clean", "Dusty", "Electrical-damage", "Physical-Damage", "Snow-Covered"]


@pytest.fixture(scope="module")
def predictor() -> SolarGuardPredictor:
    return SolarGuardPredictor(DEPLOY_CKPT, PREPROC, device=torch.device("cpu"))


@pytest.fixture(scope="module")
def sample_image() -> Image.Image:
    return Image.new("RGB", (640, 480), (120, 130, 140))


# ---- deployment artifact ----

def test_deployment_artifact_exists():
    assert DEPLOY_CKPT.exists(), f"missing deployment checkpoint: {DEPLOY_CKPT}"


def test_deployment_artifact_has_no_optimizer_state():
    ck = torch.load(DEPLOY_CKPT, map_location="cpu", weights_only=False)
    assert "optimizer_state_dict" not in ck
    assert ck.get("inference_only") is True


def test_deployment_weights_match_locked_baseline():
    """The served weights must be the locked baseline's, bit for bit."""
    source = ROOT / "experiments" / "baseline_cnn" / "colab_run_20260818_170845" / "checkpoint_best.pt"
    if not source.exists():
        pytest.skip("source training checkpoint not present locally")
    src = torch.load(source, map_location="cpu", weights_only=False)["model_state_dict"]
    dst = torch.load(DEPLOY_CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
    assert set(src) == set(dst)
    for k in src:
        assert torch.equal(src[k], dst[k]), f"weight mismatch in {k}"


def test_deployment_artifact_records_provenance():
    ck = torch.load(DEPLOY_CKPT, map_location="cpu", weights_only=False)
    assert ck["trained_epoch"] == 23
    assert ck["val_metric_name"] == "macro_f1"
    assert round(ck["val_metric_value"], 4) == 0.7534
    assert ck["seed"] == 42


# ---- class mapping ----

def test_class_mapping_matches_repository_mapping(predictor):
    repo = json.loads((ROOT / "data" / "splits" / "class_mapping.json").read_text())
    assert predictor.class_mapping == repo


def test_class_names_are_index_ordered(predictor):
    assert predictor.class_names == CLASSES
    for i, name in enumerate(predictor.class_names):
        assert predictor.class_mapping[name] == i


# ---- preprocessing parity: the property that matters most ----

def test_serving_preprocessing_is_identical_to_validation_preprocessing(sample_image):
    """Serving must apply exactly the transform used to compute every validation metric."""
    p = SolarGuardPredictor(DEPLOY_CKPT, PREPROC, device=torch.device("cpu"))
    validation_transform = build_eval_transform(load_config(PREPROC))
    assert torch.equal(p.transform(sample_image), validation_transform(sample_image))


def test_serving_preprocessing_is_deterministic(predictor, sample_image):
    assert torch.equal(predictor.transform(sample_image), predictor.transform(sample_image))


def test_repeated_prediction_is_stable(predictor, sample_image):
    a = predictor.predict(sample_image)
    b = predictor.predict(sample_image)
    assert a["predicted_class"] == b["predicted_class"]
    assert a["confidence"] == pytest.approx(b["confidence"], abs=1e-9)


# ---- prediction output contract ----

def test_predict_returns_expected_structure(predictor, sample_image):
    r = predictor.predict(sample_image, top_k=3)
    assert set(r) == {"predicted_class", "confidence", "top_k", "all_probabilities", "logits"}
    assert r["predicted_class"] in CLASSES
    assert 0.0 <= r["confidence"] <= 1.0
    assert len(r["top_k"]) == 3
    assert len(r["logits"]) == 6


def test_probabilities_sum_to_one(predictor, sample_image):
    r = predictor.predict(sample_image)
    assert sum(r["all_probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
    assert set(r["all_probabilities"]) == set(CLASSES)


def test_top_k_is_descending_and_leads_with_prediction(predictor, sample_image):
    r = predictor.predict(sample_image, top_k=6)
    probs = [e["probability"] for e in r["top_k"]]
    assert probs == sorted(probs, reverse=True)
    assert r["top_k"][0]["class"] == r["predicted_class"]
    assert r["top_k"][0]["probability"] == pytest.approx(r["confidence"])


def test_model_stays_in_eval_mode_and_produces_no_gradients(predictor, sample_image):
    predictor.predict(sample_image)
    assert predictor.model.training is False
    assert all(p.grad is None for p in predictor.model.parameters())


# ---- input handling ----

@pytest.mark.parametrize("mode,size", [("RGB", (640, 480)), ("RGBA", (100, 100)),
                                        ("L", (300, 300)), ("RGB", (50, 900))])
def test_handles_varied_image_modes_and_sizes(predictor, mode, size):
    img = Image.new(mode, size, 128 if mode == "L" else (100, 110, 120, 255)[: len(mode)])
    r = predictor.predict(img)
    assert r["predicted_class"] in CLASSES


def test_rejects_non_image_input(predictor):
    with pytest.raises(TypeError, match="PIL.Image"):
        predictor.predict("not an image")


def test_rejects_invalid_top_k(predictor, sample_image):
    with pytest.raises(ValueError, match="top_k"):
        predictor.predict(sample_image, top_k=0)
    with pytest.raises(ValueError, match="top_k"):
        predictor.predict(sample_image, top_k=99)


def test_missing_checkpoint_raises_clearly():
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        SolarGuardPredictor(ROOT / "models" / "does_not_exist.pt", PREPROC)


def test_missing_preprocessing_config_raises_clearly():
    with pytest.raises(FileNotFoundError, match="preprocessing config not found"):
        SolarGuardPredictor(DEPLOY_CKPT, ROOT / "configs" / "nope.yaml")


# ---- metadata ----

def test_metadata_reports_served_model(predictor):
    m = predictor.metadata
    assert m["architecture"] == "BaselineCNN"
    assert m["parameters"] == 98_454
    assert m["num_classes"] == 6
    assert round(m["val_metric_value"], 4) == 0.7534


def test_resolve_inference_device_prefers_cpu_when_requested():
    assert resolve_inference_device(prefer_cuda=False) == torch.device("cpu")
