"""Immutable, validated training configuration for the baseline CNN. See PLANNING.md
Phase 3 §4-5-8 for the full rationale behind every default value.

Deliberately a plain stdlib dataclass, not a training-loop function's local variables —
this is the one object that gets serialized as an experiment's config.yaml (Phase 3 §7)
and the one thing later Phase 4/5 models will reuse the shape of.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Accuracy is intentionally excluded — see PLANNING.md Phase 3 §6: at 2.73:1 imbalance,
# accuracy can look fine while minority classes fail. This is a hard project rule,
# enforced here rather than left to documentation alone.
_ALLOWED_PRIMARY_METRICS = {"macro_f1", "weighted_f1"}
_ALLOWED_TIEBREAKERS = {"val_loss"}


@dataclass(frozen=True)
class TrainingConfig:
    # reproducibility
    seed: int = 42

    # data references (not copies — File 4 reads these paths, config doesn't own the data)
    splits_dir: Path = field(default_factory=lambda: _REPO_ROOT / "data" / "splits")
    preprocessing_config_path: Path = field(
        default_factory=lambda: _REPO_ROOT / "configs" / "preprocessing.yaml"
    )

    # architecture
    num_classes: int = 6
    in_channels: int = 3

    # optimizer (AdamW is a fixed choice for this baseline, not a config value —
    # see PLANNING.md Phase 3 §4)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # LR scheduler (ReduceLROnPlateau)
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 7

    # training loop
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 15
    num_workers: int = 4
    device: str = "cuda"

    # model selection (Phase 3 §8 — validation only, test never referenced here)
    primary_metric: str = "macro_f1"
    selection_tiebreaker: str = "val_loss"
    # Minimum genuine improvement in primary_metric required to count as "improved"
    # (added after the Phase 3 file-4 review: a strict `>` comparison alone treats
    # e.g. 0.5012000001 > 0.5012 as a real improvement, which is within GPU
    # floating-point noise range, not a meaningful result). Default 1e-4, chosen
    # relative to the actual validation set size (117 images): macro F1 only changes
    # when at least one prediction's argmax flips, and on 117 images the smallest
    # realistic such change is roughly 1/117 ~= 0.0085 — two orders of magnitude
    # above this threshold. 1e-4 is comfortably larger than pure floating-point
    # representation noise (~1e-7 to 1e-10 at this value's magnitude) while staying
    # comfortably below any real, discrete metric change this dataset can produce —
    # it filters noise without being able to filter out a genuine improvement.
    min_delta: float = 1e-4

    # artifacts
    experiment_dir: Path = field(
        default_factory=lambda: _REPO_ROOT / "experiments" / "baseline_cnn"
    )

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.max_epochs < 1:
            raise ValueError(f"max_epochs must be >= 1, got {self.max_epochs}")
        if self.early_stopping_patience < 1:
            raise ValueError(
                f"early_stopping_patience must be >= 1, got {self.early_stopping_patience}"
            )
        if not (0 < self.lr_scheduler_factor < 1):
            raise ValueError(
                f"lr_scheduler_factor must be in (0, 1), got {self.lr_scheduler_factor}"
            )
        if self.lr_scheduler_patience < 0:
            raise ValueError(
                f"lr_scheduler_patience must be >= 0, got {self.lr_scheduler_patience}"
            )
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {self.num_classes}")
        if self.in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {self.in_channels}")
        if self.primary_metric not in _ALLOWED_PRIMARY_METRICS:
            raise ValueError(
                f"primary_metric={self.primary_metric!r} is not allowed. "
                f"Must be one of {sorted(_ALLOWED_PRIMARY_METRICS)} — accuracy is "
                "deliberately excluded (see PLANNING.md Phase 3 §6: it can look fine "
                "while minority classes fail under this dataset's 2.73:1 imbalance)."
            )
        if self.selection_tiebreaker not in _ALLOWED_TIEBREAKERS:
            raise ValueError(
                f"selection_tiebreaker={self.selection_tiebreaker!r} is not allowed. "
                f"Must be one of {sorted(_ALLOWED_TIEBREAKERS)}."
            )
        if self.min_delta < 0:
            raise ValueError(
                f"min_delta must be non-negative, got {self.min_delta} — a negative "
                "value would let genuinely worse results count as improvements"
            )

    def to_dict(self) -> dict:
        d = asdict(self)
        # Path objects aren't YAML/JSON-safe by default — stringify for serialization
        for key in ("splits_dir", "preprocessing_config_path", "experiment_dir"):
            d[key] = str(d[key])
        return d

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            if path.suffix in (".yaml", ".yml"):
                yaml.safe_dump(self.to_dict(), f, sort_keys=False)
            else:
                json.dump(self.to_dict(), f, indent=2)
