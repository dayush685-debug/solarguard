"""Builds torchvision transform pipelines from configs/preprocessing.yaml.

Two entry points only: build_train_transform() and build_eval_transform(). The eval
transform is used for BOTH validation and test — there is deliberately no separate
"test transform," so it's structurally impossible for test preprocessing to drift from
validation preprocessing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

_INTERPOLATION = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}


def _to_rgb(img):
    """Module-level, not a lambda: DataLoader(num_workers>0) on Windows uses
    multiprocessing's 'spawn' start method, which pickles every transform to send it
    to worker processes. Local lambdas are not picklable; a plain module-level
    function is. (Linux's fork-based multiprocessing never hits this, which is why
    this only surfaced when num_workers was actually exercised, not in Phase 2's
    single-process tests.)"""
    return img.convert("RGB")


def load_config(path: str | Path = "configs/preprocessing.yaml") -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def build_train_transform(config: dict[str, Any]) -> T.Compose:
    size = config["image_size"]
    interp = _INTERPOLATION[config["interpolation"]]
    aug = config["train_augmentation"]
    norm = config["normalization"]

    steps: list = []

    if aug["random_resized_crop"]["enabled"]:
        crop = aug["random_resized_crop"]
        steps.append(T.RandomResizedCrop(
            size, scale=(crop["scale_min"], crop["scale_max"]), interpolation=interp,
        ))
    else:
        steps.append(T.Resize((size, size), interpolation=interp))

    steps.append(T.Lambda(_to_rgb))

    if aug["horizontal_flip"]["enabled"]:
        steps.append(T.RandomHorizontalFlip(p=aug["horizontal_flip"]["probability"]))

    if aug["rotation"]["enabled"]:
        steps.append(T.RandomRotation(degrees=aug["rotation"]["max_degrees"]))

    if aug["color_jitter"]["enabled"]:
        cj = aug["color_jitter"]
        steps.append(T.ColorJitter(
            brightness=cj["brightness"], contrast=cj["contrast"],
            saturation=cj["saturation"], hue=cj["hue"],
        ))

    steps.append(T.ToTensor())
    steps.append(T.Normalize(mean=norm["mean"], std=norm["std"]))

    # random_erasing operates on tensors, so it must come after ToTensor
    if aug["random_erasing"]["enabled"]:
        steps.append(T.RandomErasing())

    return T.Compose(steps)


def build_eval_transform(config: dict[str, Any]) -> T.Compose:
    """Used for BOTH validation and test — no augmentation, deterministic."""
    size = config["image_size"]
    interp = _INTERPOLATION[config["interpolation"]]
    norm = config["normalization"]

    return T.Compose([
        T.Resize((size, size), interpolation=interp),
        T.Lambda(_to_rgb),
        T.ToTensor(),
        T.Normalize(mean=norm["mean"], std=norm["std"]),
    ])
