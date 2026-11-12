"""L2 logistic regression on standardized last-token activations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


@dataclass
class Probe:
    """w and b are float64. logit() standardizes with the training mean and std and never clips."""

    w: np.ndarray
    b: float
    mean: np.ndarray
    std: np.ndarray

    def logit(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        std = np.where(self.std == 0, 1.0, self.std)
        standardized = (X - self.mean) / std
        return standardized @ self.w + float(self.b)


def _standardize(X: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None):
    X = np.asarray(X, dtype=np.float64)
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std, np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)


def fit_probe(X, y, C: float, class_weight="balanced", scaler=None) -> Probe:
    """Fit on standardized X. Pass `scaler=(mean, std)` to reuse the primary probe's scaler."""
    y = np.asarray(y)
    if scaler is None:
        standardized, mean, std = _standardize(X)
    else:
        mean, std = scaler
        standardized, mean, std = _standardize(X, mean, std)
    model = LogisticRegression(
        C=float(C),
        class_weight=class_weight,
        solver="lbfgs",
        tol=1e-6,
        max_iter=1000,
        random_state=0,
    )
    model.fit(standardized, y)
    return Probe(
        w=np.asarray(model.coef_, dtype=np.float64).reshape(-1),
        b=float(model.intercept_.reshape(-1)[0]),
        mean=mean,
        std=std,
    )


def _auroc(scores: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def _cv_auroc(X, y, groups, C: float, n_folds: int, class_weight: str) -> dict:
    groups = np.asarray(groups)
    y = np.asarray(y)
    n_groups = len(np.unique(groups))
    folds = min(int(n_folds), n_groups)
    if folds < 2:
        raise ValueError("need at least two groups for grouped cross-validation")
    splitter = GroupKFold(n_splits=folds)
    scores = []
    for train_index, val_index in splitter.split(np.zeros(len(y)), y, groups):
        probe = fit_probe(X[train_index], y[train_index], C, class_weight=class_weight)
        scores.append(_auroc(probe.logit(X[val_index]), y[val_index]))
    values = np.asarray(scores, dtype=np.float64)
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values, ddof=1)) if np.sum(~np.isnan(values)) > 1 else 0.0
    se = std / np.sqrt(np.sum(~np.isnan(values)))
    return {"mean": mean, "std": std, "se": float(se), "folds": [None if np.isnan(v) else float(v) for v in values]}


def _within_one_se(rows: list[dict]) -> list[dict]:
    best = max(rows, key=lambda row: row["mean"])
    floor = best["mean"] - best["se"]
    return [row for row in rows if row["mean"] >= floor]


def select_probe(X, y, groups, layers: list[int], Cs: list[float], cfg) -> dict:
    """Choose the layer at C = layer_search_C, then C at that layer.

    X has shape (n_layers, N, D) in the same order as `layers`.
    Within one standard error of the best: prefer layer 16, then the smallest C.
    """
    X = np.asarray(X)
    if X.ndim != 3 or X.shape[0] != len(layers):
        raise ValueError("X must have shape (n_layers, N, D) aligned with `layers`")
    y = np.asarray(y)
    groups = np.asarray(groups)
    search_C = float(cfg.layer_search_C if hasattr(cfg, "layer_search_C") else cfg["layer_search_C"])
    n_folds = int(cfg.cv_folds if hasattr(cfg, "cv_folds") else cfg["cv_folds"])
    class_weight = cfg.class_weight if hasattr(cfg, "class_weight") else cfg["class_weight"]
    layer_rows = []
    for index, layer in enumerate(layers):
        stats = _cv_auroc(X[index], y, groups, search_C, n_folds, class_weight)
        layer_rows.append({"layer": int(layer), "C": search_C, **stats})
    layer_candidates = _within_one_se(layer_rows)
    layer_candidates.sort(key=lambda row: (row["layer"] != 16, -row["mean"], row["layer"]))
    chosen_layer = int(layer_candidates[0]["layer"])
    layer_index = layers.index(chosen_layer)
    c_rows = []
    for C in Cs:
        stats = _cv_auroc(X[layer_index], y, groups, float(C), n_folds, class_weight)
        c_rows.append({"layer": chosen_layer, "C": float(C), **stats})
    c_candidates = _within_one_se(c_rows)
    c_candidates.sort(key=lambda row: (row["C"], -row["mean"]))
    chosen_C = float(c_candidates[0]["C"])
    return {
        "layer": chosen_layer,
        "C": chosen_C,
        "layer_scores": layer_rows,
        "C_scores": c_rows,
    }


def save_probe(probe: Probe, directory) -> None:
    from pathlib import Path

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "weights.npz", w=probe.w, b=np.asarray(probe.b))
    np.savez(directory / "scaler.npz", mean=probe.mean, std=probe.std)


def load_probe(directory) -> Probe:
    from pathlib import Path

    directory = Path(directory)
    weights = np.load(directory / "weights.npz")
    scaler = np.load(directory / "scaler.npz")
    return Probe(w=weights["w"], b=float(weights["b"]), mean=scaler["mean"], std=scaler["std"])
