"""Unit tests for the weighted error, the single threshold, and the band."""

import numpy as np
import pytest

from amon.cascade import (
    cascade_predict,
    fit_band,
    fit_single_threshold,
    headroom_counts,
    item_costs,
    weighted_error,
)


def test_weighted_error_matches_a_loop():
    rng = np.random.default_rng(0)
    n = 1000
    groups = rng.choice(["misses", "hard_negatives", "ordinary"], size=n)
    y = np.where(groups == "misses", 1, 0)
    pred = rng.integers(0, 2, size=n)
    weights = {"misses": 5.0, "hard_negatives": 2.0, "ordinary": 50.0}
    costs = item_costs(groups, weights)
    got = weighted_error(pred, y, costs)
    expected = 0.0
    for index in range(n):
        if pred[index] != y[index]:
            expected += costs[index]
    assert got == pytest.approx(expected)


def _exhaustive_threshold(scores, y, costs):
    order = np.argsort(scores, kind="mergesort")
    scores, y, costs = scores[order], y[order], costs[order]
    best_k = None
    best_error = None
    for k in range(len(scores) + 1):
        pred = np.zeros(len(scores), dtype=int)
        if k < len(scores):
            pred[k:] = 1
        # k == n predicts all benign; k == 0 predicts all harmful.
        # The value rule is "above the cut". Reconstruct the cut the same way the
        # exhaustive search over ranks does: items before k are not above the cut.
        error = float(np.sum(costs * (pred != y)))
        if best_error is None or error < best_error:
            best_error = error
            best_k = k
    return best_k, best_error


def test_fit_single_threshold_matches_exhaustive_search():
    rng = np.random.default_rng(1)
    n = 300
    scores = rng.normal(size=n)
    y = rng.integers(0, 2, size=n)
    costs = rng.random(n)
    threshold = fit_single_threshold(scores, y, costs)
    pred = (scores > threshold).astype(int)
    error = weighted_error(pred, y, costs)
    _, best = _exhaustive_threshold(scores, y, costs)
    assert error == pytest.approx(best)


def _exhaustive_band(z, llm, y, costs, k):
    order = np.argsort(z, kind="mergesort")
    z, llm, y, costs = z[order], llm[order], y[order], costs[order]
    best = None
    best_start = None
    for start in range(len(z) - k + 1):
        pred = np.empty(len(z), dtype=int)
        pred[:start] = 0
        pred[start + k :] = 1
        pred[start : start + k] = llm[start : start + k]
        error = float(np.sum(costs * (pred != y)))
        if best is None or error < best:
            best = error
            best_start = start
    return best, best_start


def test_fit_band_matches_all_pairs_and_realizes_the_budget():
    rng = np.random.default_rng(2)
    n = 2000
    z = rng.normal(size=n)
    y = rng.integers(0, 2, size=n)
    llm = rng.integers(0, 2, size=n)
    costs = rng.random(n)
    budget = 0.05
    t0, t1 = fit_band(z, llm, y, costs, budget)
    pred = cascade_predict(z, llm, t0, t1)
    got = weighted_error(pred, y, costs)
    k = int(round(budget * n))
    best, _ = _exhaustive_band(z, llm, y, costs, k)
    assert got == pytest.approx(best)
    inside = (z > t0) & (z < t1)
    assert int(inside.sum()) == k
    assert inside.mean() == pytest.approx(k / n)


def test_cascade_predict_sends_only_the_open_interval_to_the_llm():
    z = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    llm = np.array([1, 1, 0, 1, 0])
    pred = cascade_predict(z, llm, 0.5, 3.5)
    assert pred.tolist() == [0, 1, 0, 1, 1]
    sent = (z > 0.5) & (z < 3.5)
    assert np.array_equal(pred[sent], llm[sent])
    assert np.all(pred[z <= 0.5] == 0)
    assert np.all(pred[z >= 3.5] == 1)


def test_headroom_counts_match_a_12_item_example():
    z = np.arange(12, dtype=float)
    t0, t1 = 2.5, 8.5
    y = np.array([1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0])
    llm = np.array([1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1])
    tokens = np.ones(12)
    counts = headroom_counts(z, llm, y, t0, t1, tokens)
    assert counts["F"] == 2
    assert counts["U"] == 2
    assert counts["I"] == 4
    assert counts["F"] + counts["U"] + counts["I"] == counts["R"]
    assert counts["R"] == 8
    assert counts["realized_deferral_items"] == pytest.approx(0.5)
