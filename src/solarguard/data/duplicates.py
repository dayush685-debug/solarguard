"""Exact and near-duplicate detection over an audited image table (see audit.py).

Near-duplicate clustering is done with a full pairwise Hamming-distance matrix on
the dHash column, vectorized with numpy's bitwise_count. This is O(n^2) in memory
(~112 MB at n=3742) and is intentionally exact/brute-force rather than approximate
(e.g. LSH) — at this dataset size brute force is fast enough and easier to trust.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def exact_duplicate_groups(df: pd.DataFrame) -> dict[str, list[int]]:
    """sha256 -> list of row indices, only for hashes shared by 2+ files."""
    groups: dict[str, list[int]] = {}
    for sha, sub in df.dropna(subset=["sha256"]).groupby("sha256"):
        if len(sub) > 1:
            groups[sha] = sub.index.tolist()
    return groups


def near_duplicate_clusters(df: pd.DataFrame, threshold: int = 5, hash_col: str = "dhash") -> pd.Series:
    """Returns a cluster id (int) per row, aligned to df.index. Every row gets a
    cluster id — rows with no near-duplicate are singleton clusters of size 1.
    `threshold` is the max Hamming distance (out of 64 bits) to count as a match;
    5 is a commonly used dHash threshold for "same or near-identical image",
    documented here rather than treated as a universal constant."""
    valid = df.dropna(subset=[hash_col])
    idx = valid.index.to_numpy()
    hashes = valid[hash_col].astype(np.uint64).to_numpy()

    n = len(hashes)
    uf = UnionFind(n)

    # Vectorized pairwise Hamming distance via broadcasting; done in row blocks
    # to bound peak memory instead of building the full n x n matrix at once.
    block = 500
    for start in range(0, n, block):
        end = min(start + block, n)
        xor_block = np.bitwise_xor(hashes[start:end, None], hashes[None, :])
        dist_block = np.bitwise_count(xor_block)
        for local_i, global_i in enumerate(range(start, end)):
            close = np.where(dist_block[local_i] <= threshold)[0]
            for j in close:
                if j > global_i:
                    uf.union(global_i, j)

    # Map union-find roots to compact cluster ids, then align back to full df
    # (rows without a valid hash — e.g. corrupt files — get their own cluster).
    root_to_cluster: dict[int, int] = {}
    cluster_ids = np.empty(n, dtype=np.int64)
    next_id = 0
    for i in range(n):
        r = uf.find(i)
        if r not in root_to_cluster:
            root_to_cluster[r] = next_id
            next_id += 1
        cluster_ids[i] = root_to_cluster[r]

    result = pd.Series(index=df.index, dtype="Int64")
    result.loc[idx] = cluster_ids
    # Corrupt/unhashable rows: each gets its own singleton cluster.
    missing = df.index[~df.index.isin(idx)]
    for i, m in enumerate(missing):
        result.loc[m] = next_id + i
    return result


def summarize_clusters(df: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    """One row per cluster with size, classes involved, splits involved."""
    rows = []
    for cluster_id, sub in df.groupby(cluster_col):
        rows.append({
            "cluster_id": cluster_id,
            "size": len(sub),
            "labels": sorted(sub["label"].unique().tolist()),
            "n_labels": sub["label"].nunique(),
            "splits": sorted(sub["split"].unique().tolist()),
            "n_splits": sub["split"].nunique(),
            "paths": sub["path"].tolist(),
        })
    return pd.DataFrame(rows)
