"""Tied-covariance Gaussian over the three train cells. Score is min squared Mahalanobis."""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf


def _cholesky(cov: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        jitter = 1e-6 * np.eye(cov.shape[0])
        return np.linalg.cholesky(cov + jitter)


def mahalanobis_sq(X: np.ndarray, mean: np.ndarray, factor: np.ndarray) -> np.ndarray:
    """Squared Mahalanobis via a Cholesky factor and a triangular solve."""
    delta = np.asarray(X, dtype=np.float64) - mean
    solved = np.linalg.solve(factor, delta.T).T
    return np.sum(solved * solved, axis=1)


def fit_tied_density(X, cell: np.ndarray, classes: list[str]) -> dict:
    """One mean per cell and a Ledoit-Wolf tied covariance on the centered points."""
    X = np.asarray(X, dtype=np.float64)
    cell = np.asarray(cell)
    means = {}
    centered = []
    for name in classes:
        mask = cell == name
        if int(mask.sum()) < 2:
            raise ValueError(f"cell {name!r} needs at least two rows to estimate a mean")
        block = X[mask]
        mean = block.mean(axis=0)
        means[name] = mean
        centered.append(block - mean)
    pooled = np.concatenate(centered, axis=0)
    model = LedoitWolf().fit(pooled)
    factor = _cholesky(model.covariance_)
    return {
        "means": means,
        "factor": factor,
        "shrinkage": float(model.shrinkage_),
        "classes": list(classes),
        "global_mean": X.mean(axis=0),
    }


def min_mahalanobis_sq(X, density: dict) -> np.ndarray:
    distances = [mahalanobis_sq(X, density["means"][name], density["factor"]) for name in density["classes"]]
    return np.min(np.stack(distances, axis=0), axis=0)


def mahalanobis_plus_plus(X, cell: np.ndarray, classes: list[str]) -> np.ndarray:
    """Ablation: L2-normalize features before the tied Gaussian. Not a candidate score."""
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    density = fit_tied_density(X / norms, cell, classes)
    return min_mahalanobis_sq(X / norms, density)


def relative_mahalanobis(X, density: dict) -> np.ndarray:
    """Ablation: min class distance minus the distance to the overall mean."""
    class_d = min_mahalanobis_sq(X, density)
    background = mahalanobis_sq(X, density["global_mean"], density["factor"])
    return class_d - background


def distance_to_predicted_class(X, density: dict, probe_logit: np.ndarray) -> np.ndarray:
    """Ablation: harmful mean when the probe logit is positive, else the closer benign mean."""
    harmful = mahalanobis_sq(X, density["means"]["harmful"], density["factor"])
    benign_names = [name for name in density["classes"] if name != "harmful"]
    benign = np.min(
        np.stack([mahalanobis_sq(X, density["means"][name], density["factor"]) for name in benign_names], axis=0),
        axis=0,
    )
    positive = np.asarray(probe_logit) > 0
    return np.where(positive, harmful, benign)


def reference_quantiles(signal: np.ndarray) -> dict[str, float]:
    signal = np.asarray(signal, dtype=np.float64)
    return {f"q{int(q * 100)}": float(np.quantile(signal, q)) for q in (0.5, 0.9, 0.95, 0.99)}
