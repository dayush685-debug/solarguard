"""PyTorch Dataset over the Phase 2 manifest. Deliberately thin — all preprocessing
logic lives in solarguard.preprocessing.transforms (Phase 2); this class only knows how
to turn a manifest row into (image_tensor, label_index).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from solarguard.preprocessing.transforms import build_eval_transform, build_train_transform, load_config


class SolarGuardDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, images_root: Path, class_to_idx: dict[str, int], transform):
        self.paths = manifest["path"].tolist()
        self.labels = manifest["class"].tolist()
        self.images_root = Path(images_root)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rel_path = self.paths[idx]
        full_path = self.images_root / rel_path
        try:
            image = Image.open(full_path)
            image.load()  # force decode now, not lazily — surfaces corrupt files here, not mid-batch
        except Exception as e:
            raise RuntimeError(f"failed to load image at {full_path} (manifest row {idx}): {e}") from e

        tensor = self.transform(image)
        label = self.class_to_idx[self.labels[idx]]
        return tensor, label


def _seed_worker(worker_id: int) -> None:
    """Each DataLoader worker is a separate process with its own RNG state — without
    this, augmentation randomness across workers would not be reproducibly seeded
    even though the main process's seed is set (Phase 3 design review §5)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_train_val_dataloaders(
    splits_dir: Path,
    images_root: Path,
    preprocessing_config_path: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    """Builds Train and Validation loaders only — deliberately no test loader exists
    here. Test data has no code path into Phase 3 training at all, not just a policy
    we remember to follow."""
    class_to_idx = json.loads((splits_dir / "class_mapping.json").read_text())
    preprocessing_config = load_config(preprocessing_config_path)

    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")

    train_dataset = SolarGuardDataset(
        train_df, images_root, class_to_idx, build_train_transform(preprocessing_config)
    )
    val_dataset = SolarGuardDataset(
        val_df, images_root, class_to_idx, build_eval_transform(preprocessing_config)
    )

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        worker_init_fn=_seed_worker, generator=generator, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        worker_init_fn=_seed_worker,
    )
    return train_loader, val_loader
