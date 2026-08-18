"""Phase 2 Step A/B: inspection report on the 772 verified-unique images, plus Step D's
post-split leakage audit and Step G's class-weight computation. Writes
data/final/dataset_statistics.json.

Run: PYTHONPATH=src python scripts/compute_statistics.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_AUDIT = REPO_ROOT / "data" / "processed" / "pvpanel_audit_with_clusters.csv"
MANIFEST = REPO_ROOT / "data" / "splits" / "manifest.csv"
OUT_PATH = REPO_ROOT / "data" / "final" / "dataset_statistics.json"

VALID_CLASSES = {"Bird-drop", "Clean", "Dusty", "Electrical-damage", "Physical-Damage", "Snow-Covered"}


def main() -> None:
    audit = pd.read_csv(RAW_AUDIT)
    manifest = pd.read_csv(MANIFEST)
    reps = audit[audit["path"].isin(manifest["path"])].merge(
        manifest[["path", "split"]], on="path", suffixes=("_orig", "")
    )
    assert len(reps) == 772, f"expected 772, got {len(reps)}"

    stats: dict = {}

    # --- A: corruption / readability ---
    stats["corrupt_count"] = int(reps["corrupt"].sum())
    stats["suspicious_count"] = int(reps["suspicious"].sum())
    assert stats["corrupt_count"] == 0, "corrupt files found in the verified-unique population"

    # --- B: dimensions ---
    stats["width"] = {
        "min": int(reps.width.min()), "max": int(reps.width.max()),
        "mean": round(float(reps.width.mean()), 1), "median": int(reps.width.median()),
    }
    stats["height"] = {
        "min": int(reps.height.min()), "max": int(reps.height.max()),
        "mean": round(float(reps.height.mean()), 1), "median": int(reps.height.median()),
    }
    stats["distinct_resolutions"] = int(reps.groupby(["width", "height"]).ngroups)
    top_res = reps.groupby(["width", "height"]).size().sort_values(ascending=False).head(5)
    stats["most_common_resolutions"] = [
        {"width": int(w), "height": int(h), "count": int(c)} for (w, h), c in top_res.items()
    ]

    # --- aspect ratio ---
    ar = reps[["width", "height"]].max(axis=1) / reps[["width", "height"]].min(axis=1)
    stats["aspect_ratio"] = {
        "min": round(float(ar.min()), 3), "max": round(float(ar.max()), 3),
        "mean": round(float(ar.mean()), 3), "median": round(float(ar.median()), 3),
    }

    # --- channel / mode / format distribution ---
    stats["mode_distribution"] = reps["mode"].value_counts().to_dict()
    stats["format_distribution"] = reps["format"].value_counts().to_dict()
    stats["likely_grayscale_count"] = int(reps["likely_grayscale"].sum())

    # --- class distribution (raw representative counts, pre-split) ---
    class_counts = reps["label"].value_counts().to_dict()
    stats["class_distribution"] = class_counts
    stats["class_distribution_pct"] = {
        k: round(v / len(reps) * 100, 2) for k, v in class_counts.items()
    }
    stats["imbalance_ratio_majority_to_minority"] = round(
        max(class_counts.values()) / min(class_counts.values()), 2
    )

    # --- label verification ---
    invalid_labels = set(reps["label"].unique()) - VALID_CLASSES
    stats["label_verification"] = {
        "valid_classes": sorted(VALID_CLASSES),
        "invalid_labels_found": sorted(invalid_labels),
        "passed": len(invalid_labels) == 0,
    }
    assert not invalid_labels, f"unexpected labels: {invalid_labels}"

    # --- duplicate verification: confirm 0 exact and 0 near-dup pairs remain among the 772 ---
    n_unique_sha = reps["sha256"].nunique()
    stats["duplicate_verification"] = {
        "exact_duplicates_remaining": len(reps) - n_unique_sha,
        "near_dup_clusters_with_multiple_reps": int(
            (reps.groupby("near_dup_cluster").size() > 1).sum()
        ),
        "passed": (len(reps) - n_unique_sha == 0),
    }
    assert stats["duplicate_verification"]["passed"], "unexpected exact duplicate survived into the 772"

    # --- D: post-split leakage audit ---
    exact_leak = 0
    for _, grp in reps.groupby("sha256"):
        if grp["split"].nunique() > 1:
            exact_leak += 1
    near_leak = 0
    for _, grp in reps.groupby("near_dup_cluster"):
        if grp["split"].nunique() > 1:
            near_leak += 1
    stats["post_split_leakage_audit"] = {
        "exact_duplicate_groups_crossing_splits": exact_leak,
        "near_duplicate_clusters_crossing_splits": near_leak,
        "passed": (exact_leak == 0 and near_leak == 0),
    }
    assert stats["post_split_leakage_audit"]["passed"], "leakage detected across splits"

    # --- per-split class distribution ---
    split_table = reps.groupby(["split", "label"]).size().unstack(fill_value=0)
    stats["split_class_distribution"] = split_table.to_dict(orient="index")
    stats["split_totals"] = reps["split"].value_counts().to_dict()

    # --- G: class weights (inverse frequency, computed from TRAIN split only) ---
    train_counts = reps[reps["split"] == "Train"]["label"].value_counts()
    n_train = train_counts.sum()
    n_classes = len(train_counts)
    class_weights = {cls: round(n_train / (n_classes * cnt), 4) for cls, cnt in train_counts.items()}
    stats["class_weights_inverse_frequency"] = {
        "computed_from": "train split only",
        "formula": "n_train_samples / (n_classes * n_samples_in_class)",
        "weights": class_weights,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(stats, f, indent=2, default=str)

    print(json.dumps(stats, indent=2, default=str))
    print(f"\nsaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
