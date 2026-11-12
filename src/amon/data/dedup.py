"""Cross-role near-duplicate removal. Train keeps. Held-out never keeps a cross-role copy."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from amon.data.textutil import user_text

# Lower rank keeps the row when a near-duplicate crosses roles.
ROLE_RANK = {
    "train": 0,
    "tune": 1,
    "pipeline": 2,
    "heldout_A1": 3,
    "heldout_A2": 4,
    "heldout_B": 5,
    "heldout_C": 6,
}


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def _shingles(text: str, width: int) -> list[str]:
    words = text.split()
    if len(words) < width:
        return [" ".join(words)] if words else [""]
    return [" ".join(words[i : i + width]) for i in range(len(words) - width + 1)]


def minhash_pairs(texts: list[str], num_perm: int, threshold: float, width: int, seed: int) -> list[tuple[int, int]]:
    """Pairs whose estimated Jaccard index is at least `threshold`."""
    from datasketch import MinHash, MinHashLSH

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    hashes = []
    for index, text in enumerate(texts):
        sketch = MinHash(num_perm=num_perm, seed=seed)
        for shingle in _shingles(text, width):
            sketch.update(shingle.encode())
        hashes.append(sketch)
        lsh.insert(str(index), sketch)
    pairs = []
    seen = set()
    for index, sketch in enumerate(hashes):
        for other in lsh.query(sketch):
            other_index = int(other)
            if other_index <= index:
                continue
            key = (index, other_index)
            if key in seen:
                continue
            seen.add(key)
            if sketch.jaccard(hashes[other_index]) >= threshold:
                pairs.append(key)
    return pairs


def embedding_pairs(vectors: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Pairs with cosine similarity at least `threshold`. Exact when n is small; LSH otherwise."""
    vectors = np.asarray(vectors, dtype=np.float64)
    if len(vectors) == 0:
        return []
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = vectors / norms
    if len(unit) <= 4000:
        sims = unit @ unit.T
        rows, cols = np.where(np.triu(sims, k=1) >= threshold)
        return list(zip(rows.tolist(), cols.tolist()))
    return _cosine_lsh_pairs(unit, threshold)


def _cosine_lsh_pairs(unit: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Random-projection buckets. High cosine pairs land in a shared bucket with high probability."""
    rng = np.random.default_rng(0)
    n_tables = 8
    n_bits = 8
    pairs: set[tuple[int, int]] = set()
    dim = unit.shape[1]
    for _ in range(n_tables):
        projection = rng.normal(size=(dim, n_bits))
        bits = unit @ projection > 0
        keys = bits.dot(1 << np.arange(n_bits))
        buckets: dict[int, list[int]] = defaultdict(list)
        for index, key in enumerate(keys.tolist()):
            buckets[int(key)].append(index)
        for members in buckets.values():
            if len(members) < 2 or len(members) > 5000:
                continue
            block = unit[members]
            sims = block @ block.T
            rows, cols = np.where(np.triu(sims, k=1) >= threshold)
            for row, col in zip(rows.tolist(), cols.tolist()):
                left, right = members[row], members[col]
                if left > right:
                    left, right = right, left
                pairs.add((left, right))
    return list(pairs)


def dedup_across_roles(items: pd.DataFrame, cfg) -> tuple[pd.DataFrame, dict]:
    """Drop cross-role near-duplicates and merge each cluster into one group_id.

    Same-role copies are kept. Their group ids are merged so grouped CV cannot split them.
    """
    out = items.reset_index(drop=True)
    if "_source_request" in out.columns:
        texts = out["_source_request"].fillna("").astype(str).tolist()
    else:
        texts = [user_text(value) for value in out["messages"].tolist()]
    width = int(getattr(cfg, "minhash_shingle_words", None) or cfg["minhash_shingle_words"])
    num_perm = int(getattr(cfg, "minhash_perm", None) or cfg["minhash_perm"])
    jaccard = float(getattr(cfg, "jaccard", None) or cfg["jaccard"])
    cosine = float(getattr(cfg, "embed_cosine", None) or cfg["embed_cosine"])
    seed = int(getattr(cfg, "seed", None) or cfg.get("seed", 1) if isinstance(cfg, dict) else getattr(cfg, "seed", 1))
    embed_fn = cfg["embed_fn"] if isinstance(cfg, dict) and "embed_fn" in cfg else getattr(cfg, "embed_fn", None)

    copy_pairs = minhash_pairs(texts, num_perm, jaccard, width, seed)
    if embed_fn is None:
        embed_pairs: list[tuple[int, int]] = []
    else:
        vectors = np.asarray(embed_fn(texts), dtype=np.float64)
        embed_pairs = embedding_pairs(vectors, cosine)

    union = _UnionFind(len(out))
    for left, right in copy_pairs + embed_pairs:
        union.union(left, right)

    clusters: dict[int, list[int]] = defaultdict(list)
    for index in range(len(out)):
        clusters[union.find(index)].append(index)

    roles = out["role"].astype(str).tolist()
    group_ids = out["group_id"].astype(str).tolist()
    drop: list[int] = []
    pair_counts: dict[str, dict[str, int]] = {"minhash": defaultdict(int), "embed": defaultdict(int)}
    for method, pairs in (("minhash", copy_pairs), ("embed", embed_pairs)):
        for left, right in pairs:
            key = "|".join(sorted((roles[left], roles[right])))
            pair_counts[method][key] += 1

    for members in clusters.values():
        if len(members) == 1:
            continue
        ranks = [(ROLE_RANK.get(roles[index], 99), index) for index in members]
        keeper_rank = min(rank for rank, _ in ranks)
        keepers = [index for rank, index in ranks if rank == keeper_rank]
        # Same-role cluster: keep every row. Cross-role: keep the best role only.
        distinct_roles = {roles[index] for index in members}
        if len(distinct_roles) > 1:
            keep = set(keepers)
            drop.extend(index for index in members if index not in keep)
        merged = min(group_ids[index] for index in members)
        for index in members:
            group_ids[index] = merged

    out = out.copy()
    out["group_id"] = group_ids
    kept = out.drop(index=drop).reset_index(drop=True)
    dropped_roles: dict[str, int] = defaultdict(int)
    for index in drop:
        dropped_roles[roles[index]] += 1
    report = {
        "pairs": {method: dict(counts) for method, counts in pair_counts.items()},
        "dropped_per_role": dict(dropped_roles),
        "n_input": int(len(items)),
        "n_kept": int(len(kept)),
        "n_dropped": int(len(drop)),
    }
    return kept, report


def cross_role_group_ids(items: pd.DataFrame) -> list[str]:
    """group_ids that still contain more than one role."""
    if items.empty:
        return []
    counts = items.groupby("group_id")["role"].nunique()
    return counts[counts > 1].index.astype(str).tolist()
