"""Leakage-safe train/val/test split.

No panel/session/source-ID metadata ships with SolarFCD (confirmed in Phase 0 —
the archive contains only images, no metadata CSV). In the absence of true
provenance grouping, the strongest defensible alternative is to group images by
*visual near-duplicate cluster* (see duplicates.py) so that near-identical copies
of the same underlying photo can never be split across train/val/test. This is
documented explicitly as a limitation, not presented as equivalent to true
source-level grouping: two genuinely different photos of the same physical panel
that don't happen to be near-duplicates will NOT be grouped by this method, and
that residual leakage risk cannot be ruled out with the data we have.

Clusters containing more than one class label are treated as label conflicts and
excluded from the automatic split entirely — they are reported separately for
manual review rather than silently assigned a label.
"""

from __future__ import annotations

import random

import pandas as pd


def make_split(
    df: pd.DataFrame,
    cluster_col: str,
    label_col: str = "label",
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> tuple[pd.Series, pd.DataFrame]:
    """Returns (split_series aligned to df.index, conflict_clusters dataframe).

    Rows belonging to a label-conflict cluster get split == "EXCLUDED_LABEL_CONFLICT"
    and are also returned in the conflicts dataframe for manual review.
    """
    assert abs(sum(ratios) - 1.0) < 1e-9, "ratios must sum to 1.0"
    train_r, val_r, test_r = ratios

    cluster_labels = df.groupby(cluster_col)[label_col].nunique()
    conflict_ids = set(cluster_labels[cluster_labels > 1].index)

    conflict_rows = df[df[cluster_col].isin(conflict_ids)].copy()

    clean = df[~df[cluster_col].isin(conflict_ids)]
    cluster_label = clean.groupby(cluster_col)[label_col].first()
    cluster_size = clean.groupby(cluster_col).size()

    rng = random.Random(seed)
    split_of_cluster: dict[int, str] = {}

    for label, cluster_ids in cluster_label.groupby(cluster_label).groups.items():
        cluster_ids = list(cluster_ids)
        rng.shuffle(cluster_ids)
        class_total = int(cluster_size.loc[cluster_ids].sum())
        target = {
            "Train": train_r * class_total,
            "Valid": val_r * class_total,
            "Test": test_r * class_total,
        }
        assigned = {"Train": 0, "Valid": 0, "Test": 0}
        for cid in cluster_ids:
            size = int(cluster_size.loc[cid])
            # assign to whichever split has the largest remaining deficit
            # (target - assigned), i.e. is furthest below its share so far
            deficits = {s: target[s] - assigned[s] for s in target}
            best = max(deficits, key=deficits.get)
            split_of_cluster[cid] = best
            assigned[best] += size

    result = pd.Series(index=df.index, dtype=object)
    result.loc[conflict_rows.index] = "EXCLUDED_LABEL_CONFLICT"
    clean_split = clean[cluster_col].map(split_of_cluster)
    result.loc[clean.index] = clean_split.values

    return result, conflict_rows


def summarize_split(df: pd.DataFrame, split_col: str, label_col: str = "label") -> pd.DataFrame:
    table = df.groupby([split_col, label_col]).size().unstack(fill_value=0)
    table["Total"] = table.sum(axis=1)
    table.loc["Total"] = table.sum(axis=0)
    return table
