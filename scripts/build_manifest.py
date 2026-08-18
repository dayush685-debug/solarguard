"""Phase 2: build the SolarGuard dataset manifest for PV Panel Defect Dataset.

Pipeline: raw audit (already computed, cached) -> exclude label-conflict clusters ->
one representative per near-duplicate cluster (772 verified-unique images) -> fold in
the tiny EXIF same-session signal as extra grouping constraints -> stratified group
split (70/15/15, seed=42) -> save manifest + reproducibility artifacts.

Run: PYTHONPATH=src python scripts/build_manifest.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from solarguard.data.duplicates import UnionFind, summarize_clusters
from solarguard.data.provenance import extract_exif_hints, same_session_groups
from solarguard.data.split import make_split, summarize_split

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = REPO_ROOT / "data" / "candidates" / "PV_Panel_Defect_Dataset"
RAW_AUDIT = REPO_ROOT / "data" / "processed" / "pvpanel_audit_with_clusters.csv"
FINAL_DIR = REPO_ROOT / "data" / "final"
SPLITS_DIR = REPO_ROOT / "data" / "splits"
SEED = 42
RATIOS = (0.70, 0.15, 0.15)


def main() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    audit = pd.read_csv(RAW_AUDIT)
    print(f"loaded raw audit: {len(audit)} files")
    assert audit["corrupt"].sum() == 0, "unexpected corrupt files in raw audit"

    # --- exclude label-conflict clusters (near-duplicate images with inconsistent labels) ---
    clusters = summarize_clusters(audit, "near_dup_cluster")
    conflict_ids = set(clusters[clusters.n_labels > 1]["cluster_id"])
    audit["is_label_conflict"] = audit["near_dup_cluster"].isin(conflict_ids)
    print(f"label-conflict clusters excluded: {len(conflict_ids)} clusters, "
          f"{audit['is_label_conflict'].sum()} images")

    clean = audit[~audit["is_label_conflict"]].copy()

    # --- one representative per near-duplicate cluster: largest file size, tie-break by path ---
    clean_sorted = clean.sort_values(
        ["near_dup_cluster", "file_size_bytes", "path"], ascending=[True, False, True]
    )
    reps = clean_sorted.groupby("near_dup_cluster", as_index=False).first()
    print(f"verified-unique representatives: {len(reps)}")
    assert len(reps) == 772, f"expected 772 verified-unique images, got {len(reps)}"

    # --- fold in the EXIF same-session signal as extra grouping constraints ---
    exif_df = extract_exif_hints(RAW_ROOT, reps["path"].tolist())
    sessions = same_session_groups(exif_df)
    print(f"EXIF coverage: {exif_df['exif_make'].notna().sum()}/{len(exif_df)} images; "
          f"same-session groups found: {sessions.groupby(['exif_make','exif_model','day']).ngroups}")

    uf = UnionFind(len(reps))
    path_to_idx = {p: i for i, p in enumerate(reps["path"])}
    for _, grp in sessions.groupby(["exif_make", "exif_model", "day"]):
        idxs = [path_to_idx[p] for p in grp["path"] if p in path_to_idx]
        for a, b in zip(idxs, idxs[1:]):
            uf.union(a, b)
    reps = reps.reset_index(drop=True)
    reps["split_group"] = [uf.find(i) for i in range(len(reps))]
    n_merged = reps["split_group"].nunique()
    print(f"split groups after EXIF-session merge: {n_merged} "
          f"(started from {reps['near_dup_cluster'].nunique()} singleton clusters)")

    # --- stratified group split, 70/15/15, seed=42 ---
    split_series, conflicts = make_split(
        reps, cluster_col="split_group", label_col="label", ratios=RATIOS, seed=SEED
    )
    reps["split"] = split_series.values
    assert len(conflicts) == 0, "unexpected label conflicts inside the clean representative set"

    print("\nfinal split distribution:")
    print(summarize_split(reps, "split", "label"))

    # --- save manifest + per-split CSVs ---
    manifest = reps[["path", "label", "split"]].rename(columns={"label": "class"})
    manifest = manifest.sort_values(["split", "class", "path"]).reset_index(drop=True)
    manifest.to_csv(SPLITS_DIR / "manifest.csv", index=False)
    # split values from make_split are "Train"/"Valid"/"Test"; save each as its own CSV too
    for split_value, fname in [("Train", "train.csv"), ("Valid", "val.csv"), ("Test", "test.csv")]:
        manifest[manifest["split"] == split_value].to_csv(SPLITS_DIR / fname, index=False)

    classes = sorted(manifest["class"].unique())
    class_mapping = {cls: i for i, cls in enumerate(classes)}
    with open(SPLITS_DIR / "class_mapping.json", "w") as f:
        json.dump(class_mapping, f, indent=2)

    split_config = {
        "seed": SEED,
        "ratios": {"train": RATIOS[0], "valid": RATIOS[1], "test": RATIOS[2]},
        "grouping_method": (
            "near-duplicate cluster (dHash<=5) collapsed to 1 representative per cluster, "
            "then merged with EXIF same-camera-same-day session groups where present "
            f"({n_merged} final groups from {len(reps)} images)"
        ),
        "stratification": "by class label",
        "source_population": "772 verified-unique images (see PLANNING.md for derivation from 1,574 raw files)",
    }
    with open(SPLITS_DIR / "split_config.json", "w") as f:
        json.dump(split_config, f, indent=2)

    # --- exclusions ledger ---
    kept_paths = set(reps["path"])
    excluded = audit[~audit["path"].isin(kept_paths)].copy()
    rep_by_cluster = reps.set_index("near_dup_cluster")["path"].to_dict()
    excluded["exclusion_reason"] = excluded["is_label_conflict"].map(
        {True: "label_conflict_cluster", False: "duplicate_of_representative"}
    )
    excluded["representative_kept"] = excluded.apply(
        lambda r: "" if r["is_label_conflict"] else rep_by_cluster.get(r["near_dup_cluster"], ""), axis=1
    )
    excl_out = excluded[["path", "split", "label", "near_dup_cluster", "exclusion_reason", "representative_kept"]]
    excl_out = excl_out.rename(columns={"split": "original_split", "label": "original_label"})
    excl_out.to_csv(FINAL_DIR / "exclusions.csv", index=False)

    print(f"\nsaved manifest ({len(manifest)} rows), train/val/test CSVs, class_mapping.json, "
          f"split_config.json, exclusions.csv ({len(excl_out)} rows)")
    print(f"sanity: {len(manifest)} kept + {len(excl_out)} excluded = {len(manifest) + len(excl_out)} "
          f"vs raw {len(audit)}")


if __name__ == "__main__":
    main()
