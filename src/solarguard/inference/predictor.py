"""Single-image inference for a deployed SolarGuard checkpoint.

Two correctness properties this module exists to guarantee:

1. **Preprocessing cannot drift from evaluation.** The transform is built by
   `solarguard.preprocessing.transforms.build_eval_transform` — the same function that
   produced every validation metric in the repository. It is imported, never
   reimplemented. If it changes, serving changes with it.

2. **Class names come from the artifact, not from a separate file.** A deployment
   checkpoint is self-describing: it carries its own `class_mapping`. Reading labels
   from `data/splits/class_mapping.json` instead would allow a silent index/label
   mismatch if the two ever diverged, which would mislabel every prediction without
   raising anything.

Softmax is applied to logits for display purposes only. These values are NOT calibrated
probabilities — no calibration analysis has been performed on this model (see
docs/MODEL_CARD.md). Callers must not present them as reliability estimates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from solarguard.models.baseline_cnn import BaselineCNN, count_parameters
from solarguard.preprocessing.transforms import build_eval_transform, load_config


def resolve_inference_device(prefer_cuda: bool = True) -> torch.device:
    """CPU is the default target for serving. CUDA is used only if genuinely present."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SolarGuardPredictor:
    """Loads a deployment checkpoint and classifies single images.

    Accepts either an exported inference-only artifact (from
    scripts/export_deployment_checkpoint.py) or a raw training checkpoint — both carry
    `model_state_dict` and `class_mapping`.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        preprocessing_config_path: str | Path,
        device: torch.device | None = None,
    ) -> None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

        preprocessing_config_path = Path(preprocessing_config_path)
        if not preprocessing_config_path.exists():
            raise FileNotFoundError(
                f"preprocessing config not found: {preprocessing_config_path}"
            )

        self.device = device if device is not None else resolve_inference_device()
        self.checkpoint_path = checkpoint_path

        # NOTE: the checkpoint's embedded `config.preprocessing_config_path` is
        # deliberately ignored -- it is an absolute path from the training machine and
        # will not exist here. The config to use is passed in explicitly.
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if "model_state_dict" not in checkpoint:
            raise ValueError(f"{checkpoint_path} has no 'model_state_dict' -- not a SolarGuard checkpoint")
        if "class_mapping" not in checkpoint:
            raise ValueError(f"{checkpoint_path} has no 'class_mapping' -- cannot label predictions")

        self.class_mapping: dict[str, int] = checkpoint["class_mapping"]
        self.class_names: list[str] = sorted(self.class_mapping, key=self.class_mapping.get)

        indices = sorted(self.class_mapping.values())
        if indices != list(range(len(self.class_names))):
            raise ValueError(
                f"class_mapping indices are not contiguous from 0: {indices}. "
                "Predictions could not be reliably labelled."
            )

        num_classes = checkpoint.get("num_classes", len(self.class_names))
        in_channels = checkpoint.get("in_channels", 3)

        self.model = BaselineCNN(num_classes=num_classes, in_channels=in_channels)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = build_eval_transform(load_config(preprocessing_config_path))

        # provenance, surfaced so a UI can state exactly what it is serving
        self.metadata: dict[str, Any] = {
            "architecture": checkpoint.get("architecture", "BaselineCNN"),
            "parameters": count_parameters(self.model),
            "num_classes": num_classes,
            "trained_epoch": checkpoint.get("trained_epoch", checkpoint.get("epoch")),
            "seed": checkpoint.get("seed"),
            "val_metric_name": checkpoint.get("val_metric_name", checkpoint.get("best_metric_name")),
            "val_metric_value": checkpoint.get("val_metric_value", checkpoint.get("best_metric_value")),
            "source_checkpoint": checkpoint.get("source_checkpoint", str(checkpoint_path)),
            "device": str(self.device),
        }

    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = 3) -> dict[str, Any]:
        """Classify one PIL image.

        Returns predicted_class, confidence, top-k list, and the full probability
        distribution. `confidence` is a softmax output and is NOT calibrated.
        """
        if not isinstance(image, Image.Image):
            raise TypeError(f"expected a PIL.Image.Image, got {type(image).__name__}")
        if top_k < 1 or top_k > len(self.class_names):
            raise ValueError(f"top_k must be in 1..{len(self.class_names)}, got {top_k}")

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        if tensor.shape[1:] != (3, 224, 224):
            raise ValueError(f"preprocessing produced unexpected shape {tuple(tensor.shape)}")

        logits = self.model(tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0)

        ranked = torch.argsort(probabilities, descending=True)
        top = [
            {"class": self.class_names[i], "probability": float(probabilities[i])}
            for i in ranked[:top_k].tolist()
        ]

        return {
            "predicted_class": top[0]["class"],
            "confidence": top[0]["probability"],
            "top_k": top,
            "all_probabilities": {
                name: float(probabilities[idx]) for name, idx in self.class_mapping.items()
            },
            "logits": [float(v) for v in logits.squeeze(0)],
        }
