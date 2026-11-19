"""Harm-definition gap. Descriptive strata. Nothing here enters decide()."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from amon.cascade.headroom import headroom_counts
from amon.cascade.thresholds import llm_predict
from amon.io import write_json
from amon.stats.bootstrap import paired_bootstrap, percentile_interval
from amon.stats.wilson import wilson


def recall_by(
    strata: np.ndarray,
    log_odds: np.ndarray,
    y: np.ndarray,
    tau_llm: float,
    clusters: np.ndarray | None,
    B: int,
    seed: int,
    min_n: int = 30,
) -> pd.DataFrame:
    """Harmful rows only. Columns: stratum, n, recall, lo, hi, method ('wilson' or 'cluster_bootstrap')."""
    strata = np.asarray(strata).astype(str)
    log_odds = np.asarray(log_odds, dtype=np.float64)
    y = np.asarray(y)
    harmful = y == 1
    strata = strata[harmful]
    log_odds = log_odds[harmful]
    if clusters is not None:
        clusters = np.asarray(clusters)[harmful]
    pred = llm_predict(log_odds, tau_llm)
    labels = _pool(strata, min_n)
    rows = []
    for name in _order(labels):
        mask = labels == name
        n = int(mask.sum())
        k = int(pred[mask].sum())
        recall = (k / n) if n else 0.0
        if clusters is not None:
            lo, hi = _cluster_interval(pred[mask].astype(float), clusters[mask], B, seed)
            method = "cluster_bootstrap"
        else:
            lo, hi = wilson(k, n) if n else (float("nan"), float("nan"))
            method = "wilson"
        rows.append({"stratum": name, "n": n, "recall": recall, "lo": lo, "hi": hi, "method": method})
    return pd.DataFrame(rows)


def errors_by(
    strata: np.ndarray,
    z: np.ndarray,
    llm_pred: np.ndarray,
    y: np.ndarray,
    t0: float,
    t1: float,
    min_n: int = 30,
) -> pd.DataFrame:
    """Columns: stratum, n, F, U, I, U_share. Uses headroom_counts per stratum."""
    strata = np.asarray(strata).astype(str)
    z = np.asarray(z, dtype=np.float64)
    llm_pred = np.asarray(llm_pred)
    y = np.asarray(y)
    labels = _pool(strata, min_n)
    tokens = np.ones(len(y), dtype=np.float64)
    rows = []
    for name in _order(labels):
        mask = labels == name
        counts = headroom_counts(z[mask], llm_pred[mask], y[mask], t0, t1, tokens[mask])
        harmful = int(np.sum(y[mask] == 1))
        rows.append(
            {
                "stratum": name,
                "n": int(mask.sum()),
                "F": counts["F"],
                "U": counts["U"],
                "I": counts["I"],
                "U_share": (counts["U"] / harmful) if harmful else 0.0,
            }
        )
    return pd.DataFrame(rows)


def fpr_by(strata: np.ndarray, pred: np.ndarray, y: np.ndarray, min_n: int = 30) -> pd.DataFrame:
    """Benign rows only. Columns: stratum, n, fpr, lo, hi."""
    strata = np.asarray(strata).astype(str)
    pred = np.asarray(pred)
    y = np.asarray(y)
    benign = y == 0
    strata = strata[benign]
    pred = pred[benign]
    labels = _pool(strata, min_n)
    rows = []
    for name in _order(labels):
        mask = labels == name
        n = int(mask.sum())
        k = int(np.sum(pred[mask] == 1))
        lo, hi = wilson(k, n) if n else (float("nan"), float("nan"))
        rows.append({"stratum": name, "n": n, "fpr": (k / n) if n else 0.0, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def llm_miss_share(recall: pd.DataFrame, strata: np.ndarray, pred: np.ndarray, y: np.ndarray, min_n: int = 30) -> float:
    """Share of Llama Guard's harmful misses that sit in strata whose recall is below 0.5."""
    y = np.asarray(y)
    pred = np.asarray(pred)
    low = set(recall.loc[recall["recall"] < 0.5, "stratum"].astype(str))
    harmful = y == 1
    if not np.any(harmful & (pred == 0)):
        return 0.0
    harmful_labels = _pool(np.asarray(strata).astype(str)[harmful], min_n)
    miss_labels = harmful_labels[pred[harmful] == 0]
    if len(miss_labels) == 0:
        return 0.0
    return float(np.mean(np.isin(miss_labels, list(low))))


def write_category_bundle(directory: Path, tables: dict[str, pd.DataFrame], miss_share: float) -> None:
    """gate1/categories.json plus one CSV per table under gate1/categories/."""
    directory = Path(directory)
    folder = directory / "categories"
    folder.mkdir(parents=True, exist_ok=True)
    payload = {"llm_miss_share_low_recall_strata": float(miss_share), "tables": {}}
    for name, frame in tables.items():
        frame.to_csv(folder / f"{name}.csv", index=False)
        payload["tables"][name] = frame.to_dict(orient="records")
    write_json(directory / "categories.json", payload)


def _pool(strata: np.ndarray, min_n: int) -> np.ndarray:
    if len(strata) == 0:
        return strata
    counts = pd.Series(strata).value_counts()
    small = set(counts[counts < int(min_n)].index.astype(str))
    if not small:
        return strata
    return np.array(["other" if value in small else value for value in strata], dtype=object)


def _order(labels: np.ndarray) -> list[str]:
    seen = []
    for value in labels.tolist():
        if value not in seen:
            seen.append(value)
    return seen


def _cluster_interval(values: np.ndarray, clusters: np.ndarray, B: int, seed: int) -> tuple[float, float]:
    if len(values) == 0 or B <= 0:
        return float("nan"), float("nan")

    def stat(idx, sample=values):
        taken = sample[idx]
        return float(np.mean(taken)) if len(taken) else float("nan")

    samples = paired_bootstrap(stat, len(values), B, seed, clusters=clusters)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return float("nan"), float("nan")
    return percentile_interval(samples)
