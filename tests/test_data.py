"""Phase 2 tests: verify the PV Panel Defect Dataset split manifest is internally
consistent, leakage-safe, and reproducible.

These tests check the *artifacts* produced by scripts/build_manifest.py and
scripts/compute_statistics.py rather than recomputing hashes from scratch — the full
1,574-file audit is already cached and versioned by the code that produced it
(src/solarguard/data/{hashing,audit,duplicates}.py).
"""

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "candidates" / "PV_Panel_Defect_Dataset"
MANIFEST_PATH = ROOT / "data" / "splits" / "manifest.csv"
AUDIT_PATH = ROOT / "data" / "processed" / "pvpanel_audit_with_clusters.csv"
STATS_PATH = ROOT / "data" / "final" / "dataset_statistics.json"

# The raw images are not committed (licence and size — see DATASET.md), so the tests that
# actually read them can only run once the dataset has been downloaded locally. They skip
# rather than fail when it is absent; every artifact-only test above still runs.
requires_dataset = pytest.mark.skipif(
    not DATA_RAW.exists(),
    reason="Optional PV Panel Defect Dataset not present; dataset-dependent test skipped.",
)

VALID_CLASSES = {"Bird-drop", "Clean", "Dusty", "Electrical-damage", "Physical-Damage", "Snow-Covered"}
VALID_SPLITS = {"Train", "Valid", "Test"}


@pytest.fixture(scope="module")
def manifest() -> pd.DataFrame:
    return pd.read_csv(MANIFEST_PATH)


@pytest.fixture(scope="module")
def audit() -> pd.DataFrame:
    return pd.read_csv(AUDIT_PATH)


def test_manifest_has_772_verified_unique_images(manifest):
    """Project rule: the 772 verified-unique images are the modeling population,
    never the raw 1,574 files."""
    assert len(manifest) == 772


def test_every_image_exactly_one_split(manifest):
    assert manifest["path"].is_unique


def test_class_labels_are_valid(manifest):
    invalid = set(manifest["class"].unique()) - VALID_CLASSES
    assert not invalid, f"unexpected class labels in manifest: {invalid}"


def test_split_labels_are_valid(manifest):
    invalid = set(manifest["split"].unique()) - VALID_SPLITS
    assert not invalid, f"unexpected split labels in manifest: {invalid}"


@requires_dataset
def test_files_exist(manifest):
    missing = [p for p in manifest["path"] if not (DATA_RAW / p).exists()]
    assert not missing, f"{len(missing)} manifest paths do not exist on disk, e.g. {missing[:5]}"


@requires_dataset
def test_images_are_readable(manifest):
    from PIL import Image
    unreadable = []
    for p in manifest["path"]:
        try:
            with Image.open(DATA_RAW / p) as img:
                img.verify()
        except Exception as e:
            unreadable.append((p, str(e)))
    assert not unreadable, f"{len(unreadable)} images failed to open, e.g. {unreadable[:3]}"


def test_split_counts_are_correct(manifest):
    counts = manifest["split"].value_counts().to_dict()
    assert sum(counts.values()) == 772
    assert counts["Train"] == 540
    assert counts["Valid"] == 117
    assert counts["Test"] == 115


def _exact_dup_shas_crossing(manifest, audit, split_a, split_b):
    merged = manifest.merge(audit[["path", "sha256"]], on="path", how="left")
    sub = merged[merged["split"].isin([split_a, split_b])]
    crossing = []
    for sha, group in sub.groupby("sha256"):
        if group["split"].nunique() > 1:
            crossing.append(sha)
    return crossing


def test_no_exact_duplicates_train_test(manifest, audit):
    assert _exact_dup_shas_crossing(manifest, audit, "Train", "Test") == []


def test_no_exact_duplicates_train_val(manifest, audit):
    assert _exact_dup_shas_crossing(manifest, audit, "Train", "Valid") == []


def test_no_exact_duplicates_val_test(manifest, audit):
    assert _exact_dup_shas_crossing(manifest, audit, "Valid", "Test") == []


def test_no_known_near_duplicate_crosses_splits(manifest, audit):
    merged = manifest.merge(audit[["path", "near_dup_cluster"]], on="path", how="left")
    assert merged["near_dup_cluster"].notna().all()
    for cluster_id, group in merged.groupby("near_dup_cluster"):
        assert group["split"].nunique() == 1, (
            f"near_dup_cluster {cluster_id} spans multiple splits: {group['path'].tolist()}"
        )
        # by construction each cluster contributes exactly one representative
        assert len(group) == 1


def test_no_label_conflict_clusters_in_manifest(manifest, audit):
    merged = manifest.merge(audit[["path", "near_dup_cluster"]], on="path", how="left")
    cluster_label_counts = audit.groupby("near_dup_cluster")["label"].nunique()
    for cluster_id in merged["near_dup_cluster"]:
        assert cluster_label_counts.loc[cluster_id] == 1, (
            f"manifest contains a representative from a label-conflict cluster ({cluster_id})"
        )


@requires_dataset
def test_split_is_reproducible_from_seed():
    """Re-running the split logic with the same seed must produce the identical
    test set — this is what makes the test set immune to accidental drift from
    any later preprocessing change."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from solarguard.data.duplicates import summarize_clusters, UnionFind
    from solarguard.data.provenance import extract_exif_hints, same_session_groups
    from solarguard.data.split import make_split

    audit = pd.read_csv(AUDIT_PATH)
    clusters = summarize_clusters(audit, "near_dup_cluster")
    conflict_ids = set(clusters[clusters.n_labels > 1]["cluster_id"])
    clean = audit[~audit["near_dup_cluster"].isin(conflict_ids)]
    reps = (clean.sort_values(["near_dup_cluster", "file_size_bytes", "path"],
                               ascending=[True, False, True])
                 .groupby("near_dup_cluster", as_index=False).first())

    exif_df = extract_exif_hints(DATA_RAW, reps["path"].tolist())
    sessions = same_session_groups(exif_df)
    uf = UnionFind(len(reps))
    path_to_idx = {p: i for i, p in enumerate(reps["path"])}
    for _, grp in sessions.groupby(["exif_make", "exif_model", "day"]):
        idxs = [path_to_idx[p] for p in grp["path"] if p in path_to_idx]
        for a, b in zip(idxs, idxs[1:]):
            uf.union(a, b)
    reps = reps.reset_index(drop=True)
    reps["split_group"] = [uf.find(i) for i in range(len(reps))]

    split_series, _ = make_split(reps, cluster_col="split_group", label_col="label",
                                  ratios=(0.70, 0.15, 0.15), seed=42)
    recomputed_test_paths = set(reps.loc[split_series.values == "Test", "path"])

    saved_test_paths = set(pd.read_csv(ROOT / "data" / "splits" / "test.csv")["path"])
    assert recomputed_test_paths == saved_test_paths, (
        "recomputing the split with the same seed produced a different test set — "
        "the split is not reproducible, or test.csv has drifted from the pipeline"
    )


def test_dataset_statistics_checks_all_passed():
    stats = json.loads(STATS_PATH.read_text())
    assert stats["duplicate_verification"]["passed"]
    assert stats["post_split_leakage_audit"]["passed"]
    assert stats["label_verification"]["passed"]
    assert stats["corrupt_count"] == 0
