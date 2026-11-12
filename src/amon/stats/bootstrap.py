"""Paired bootstrap. Both arms are scored on the indices of one resample."""

from __future__ import annotations

import numpy as np


def _resample_indices(
    n: int,
    rng: np.random.Generator,
    clusters: np.ndarray | None,
    strata: np.ndarray | None,
) -> np.ndarray:
    idx = np.arange(n)
    if strata is None:
        strata = np.zeros(n, dtype=np.int64)
    else:
        strata = np.asarray(strata)
        if len(strata) != n:
            raise ValueError("strata length must equal n")
    if clusters is not None:
        clusters = np.asarray(clusters)
        if len(clusters) != n:
            raise ValueError("clusters length must equal n")
    parts: list[np.ndarray] = []
    for stratum in np.unique(strata):
        members = idx[strata == stratum]
        if clusters is None:
            parts.append(rng.choice(members, size=len(members), replace=True))
            continue
        labels = clusters[members]
        # Preserve first-seen order so the draw is a pure function of the seed.
        _, first = np.unique(labels, return_index=True)
        uniq = labels[np.sort(first)]
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        buckets: dict = {}
        for item, label in zip(members.tolist(), labels.tolist()):
            buckets.setdefault(label, []).append(item)
        chosen: list[int] = []
        for label in drawn.tolist():
            chosen.extend(buckets[label])
        parts.append(np.asarray(chosen, dtype=np.int64))
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def paired_bootstrap(
    stat_fn,
    n: int,
    B: int,
    seed: int,
    clusters: np.ndarray | None = None,
    strata: np.ndarray | None = None,
) -> np.ndarray:
    """B resamples. `stat_fn(indices)` is called once per resample, so paired arms share indices.

    Cluster resampling draws whole clusters with replacement. Strata are resampled
    separately, each back to its own size. The seed fixes the sequence.
    """
    if n < 0 or B < 0:
        raise ValueError("n and B must be non-negative")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(B):
        indices = _resample_indices(n, rng, clusters, strata)
        values.append(stat_fn(indices))
    return np.asarray(values)


def percentile_interval(samples: np.ndarray, ci: float = 0.95) -> tuple[float, float]:
    """Two-sided percentile interval. ci=0.95 uses the 2.5 and 97.5 percentiles."""
    samples = np.asarray(samples, dtype=np.float64)
    if samples.size == 0:
        raise ValueError("no bootstrap samples")
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(samples, [alpha, 1.0 - alpha])
    return float(lo), float(hi)
