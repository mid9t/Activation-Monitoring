"""Item costs and the weighted cascade error.

E = sum_i c_i [yhat_i != y_i], with c_i = w_g(i) / N_g(i).
That equals 5 FNR + 2 FPR_hard + 50 FPR_ordinary when the weights are the Gate 1 weights.
"""

from __future__ import annotations

import numpy as np


def item_costs(groups: np.ndarray, weights: dict[str, float]) -> np.ndarray:
    """Per-item cost c_i = w_g / N_g. `groups` holds one error-group name per item."""
    groups = np.asarray(groups)
    if groups.ndim != 1:
        raise ValueError("groups must be a 1-d array of error-group names")
    missing = sorted(set(groups.tolist()) - set(weights))
    if missing:
        raise KeyError(f"no weight for error groups: {missing}")
    costs = np.empty(len(groups), dtype=np.float64)
    for name, weight in weights.items():
        mask = groups == name
        n = int(mask.sum())
        if n == 0:
            continue
        costs[mask] = float(weight) / n
    return costs


def weighted_error(pred: np.ndarray, y: np.ndarray, c: np.ndarray) -> float:
    """Sum of item costs on mistakes. Predictions and labels are 1 = harmful, 0 = benign."""
    pred = np.asarray(pred)
    y = np.asarray(y)
    c = np.asarray(c, dtype=np.float64)
    if pred.shape != y.shape or pred.shape != c.shape:
        raise ValueError("pred, y, and c must have the same shape")
    if pred.size == 0:
        return 0.0
    return float(np.sum(c * (pred != y)))
