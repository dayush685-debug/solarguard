"""Phase 2 tests: preprocessing pipeline correctness and determinism."""

from pathlib import Path

import torch
from PIL import Image

from solarguard.preprocessing.transforms import build_eval_transform, build_train_transform, load_config

ROOT = Path(__file__).resolve().parent.parent
CONFIG = load_config(ROOT / "configs" / "preprocessing.yaml")

# A real RGBA sample confirmed present in dataset_statistics.json's mode_distribution
SAMPLE_IMAGE = ROOT / "data" / "candidates" / "PV_Panel_Defect_Dataset" / "train" / "Bird-drop" / "Bird (111).jpg"


def test_eval_transform_output_shape():
    tf = build_eval_transform(CONFIG)
    out = tf(Image.open(SAMPLE_IMAGE))
    assert out.shape == (3, CONFIG["image_size"], CONFIG["image_size"])


def test_train_transform_output_shape():
    tf = build_train_transform(CONFIG)
    out = tf(Image.open(SAMPLE_IMAGE))
    assert out.shape == (3, CONFIG["image_size"], CONFIG["image_size"])


def test_eval_transform_is_deterministic():
    """No augmentation in the eval path — the same image must always produce the
    identical tensor. This is what "no augmentation on val/test" means concretely."""
    tf = build_eval_transform(CONFIG)
    out1 = tf(Image.open(SAMPLE_IMAGE))
    out2 = tf(Image.open(SAMPLE_IMAGE))
    assert torch.equal(out1, out2)


def test_train_transform_is_stochastic():
    """Sanity check that augmentation is actually wired up, not silently a no-op."""
    tf = build_train_transform(CONFIG)
    outputs = [tf(Image.open(SAMPLE_IMAGE)) for _ in range(8)]
    assert not all(torch.equal(outputs[0], o) for o in outputs[1:])


def test_eval_transform_handles_rgba():
    rgba_img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
    tf = build_eval_transform(CONFIG)
    out = tf(rgba_img)
    assert out.shape == (3, CONFIG["image_size"], CONFIG["image_size"])


def test_normalization_matches_imagenet_stats():
    assert CONFIG["normalization"]["mean"] == [0.485, 0.456, 0.406]
    assert CONFIG["normalization"]["std"] == [0.229, 0.224, 0.225]
