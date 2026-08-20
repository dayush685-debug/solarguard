"""Export an inference-only checkpoint from a verified training checkpoint.

Training checkpoints carry `optimizer_state_dict`, which is useless for inference and
roughly doubles the file size. This script strips it and writes a deployment artifact.

It is deliberately NON-DESTRUCTIVE: the source checkpoint is opened read-only and never
modified. The exported artifact is written to a separate path, and the script verifies
that every model weight in the export is bit-identical to the source before finishing.

The export drops `config`, whose embedded absolute paths (e.g.
`/content/Solar_Guard/configs/preprocessing.yaml`) refer to a machine that will not exist
at serving time. Preprocessing is instead pinned explicitly in the exported metadata.

Run: PYTHONPATH=src python scripts/export_deployment_checkpoint.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from solarguard.models.baseline_cnn import BaselineCNN, count_parameters

REPO_ROOT = Path(__file__).resolve().parent.parent

# The locked baseline: the frozen reference run, validated macro-F1 0.7534.
DEFAULT_SOURCE = (
    REPO_ROOT / "experiments" / "baseline_cnn" / "colab_run_20260818_170845" / "checkpoint_best.pt"
)
DEFAULT_TARGET = REPO_ROOT / "models" / "solarguard_baseline_v1.pt"

# Recorded so serving never has to guess, and never has to read the source checkpoint's
# absolute training-time paths.
PREPROCESSING = {
    "image_size": 224,
    "interpolation": "bilinear",
    "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "rgb_conversion": True,
    "augmentation": "none (evaluation transform only)",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    args = ap.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"source checkpoint not found: {args.source}")

    source = torch.load(args.source, map_location="cpu", weights_only=False)
    print(f"source : {args.source}")
    print(f"         {args.source.stat().st_size:,} bytes")
    print(f"         epoch={source['epoch']}  {source['best_metric_name']}="
          f"{source['best_metric_value']:.16f}  seed={source['seed']}")

    class_mapping = source["class_mapping"]
    num_classes = len(class_mapping)

    # Sanity-check the weights actually load into the architecture we will serve.
    probe = BaselineCNN(num_classes=num_classes, in_channels=3)
    probe.load_state_dict(source["model_state_dict"])
    print(f"         loads into BaselineCNN(num_classes={num_classes}): OK, "
          f"{count_parameters(probe):,} parameters")

    artifact = {
        "model_state_dict": source["model_state_dict"],
        "class_mapping": class_mapping,
        "architecture": "BaselineCNN",
        "num_classes": num_classes,
        "in_channels": 3,
        "preprocessing": PREPROCESSING,
        # provenance, so a served model can always be traced back to its experiment
        "source_checkpoint": str(args.source.relative_to(REPO_ROOT)).replace("\\", "/"),
        "trained_epoch": source["epoch"],
        "seed": source["seed"],
        "val_metric_name": source["best_metric_name"],
        "val_metric_value": source["best_metric_value"],
        "inference_only": True,
    }

    args.target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.target)

    # ---- verification: reload the export and compare weights tensor by tensor ----
    reloaded = torch.load(args.target, map_location="cpu", weights_only=False)
    src_sd, dst_sd = source["model_state_dict"], reloaded["model_state_dict"]
    assert set(src_sd) == set(dst_sd), "state_dict keys differ after export"
    identical = all(torch.equal(src_sd[k], dst_sd[k]) for k in src_sd)
    assert identical, "exported weights differ from source"
    assert "optimizer_state_dict" not in reloaded, "optimizer state was not stripped"

    check = BaselineCNN(num_classes=reloaded["num_classes"], in_channels=reloaded["in_channels"])
    check.load_state_dict(reloaded["model_state_dict"])

    src_size, dst_size = args.source.stat().st_size, args.target.stat().st_size
    print()
    print(f"target : {args.target}")
    print(f"         {dst_size:,} bytes  ({dst_size / src_size * 100:.1f}% of source, "
          f"{src_size - dst_size:,} bytes saved)")
    print(f"         optimizer_state_dict stripped : True")
    print(f"         weights bit-identical to source: {identical}")
    print(f"         reloads into BaselineCNN       : OK")
    print()
    print("source checkpoint was opened read-only and is unmodified.")


if __name__ == "__main__":
    main()
