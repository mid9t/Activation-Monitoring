"""Grouped bootstrap ensemble. Members share the primary probe's scaler."""

from __future__ import annotations

import numpy as np

from amon.probe.train import fit_probe


def group_bootstrap_indices(groups: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    _, first = np.unique(groups, return_index=True)
    uniq = groups[np.sort(first)]
    drawn = rng.choice(uniq, size=len(uniq), replace=True)
    buckets: dict = {}
    for index, label in enumerate(groups.tolist()):
        buckets.setdefault(label, []).append(index)
    chosen: list[int] = []
    for label in drawn.tolist():
        chosen.extend(buckets[label])
    return np.asarray(chosen, dtype=np.int64)


def fit_ensemble(X, y, groups, C: float, scaler, seeds: list[int], class_weight="balanced") -> list:
    members = []
    for seed in seeds:
        index = group_bootstrap_indices(groups, seed)
        members.append(fit_probe(X[index], np.asarray(y)[index], C, class_weight=class_weight, scaler=scaler))
    return members


def ensemble_std(members, X) -> np.ndarray:
    """Standard deviation of the member logits. Higher means more uncertain."""
    logits = np.stack([member.logit(X) for member in members], axis=0)
    return np.std(logits, axis=0, ddof=0)
