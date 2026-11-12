"""Single thresholds and the two-threshold deferral band.

Predict 1 (harmful) when a score is strictly above a single threshold.
The band calls items at or below t0 benign, items at or above t1 harmful,
and sends the open interval (t0, t1) to the LLM.
"""

from __future__ import annotations

import numpy as np


def _cumsum0(values: np.ndarray) -> np.ndarray:
    """Prefix sums with a leading 0. out[i] = sum(values[:i])."""
    out = np.zeros(len(values) + 1, dtype=np.float64)
    if len(values):
        out[1:] = np.cumsum(values, dtype=np.float64)
    return out


def _suffix_sum0(values: np.ndarray) -> np.ndarray:
    """Suffix sums. out[i] = sum(values[i:]); out[len] = 0."""
    out = np.zeros(len(values) + 1, dtype=np.float64)
    if len(values):
        out[:-1] = np.cumsum(values[::-1], dtype=np.float64)[::-1]
    return out


def _midpoint_at(values: np.ndarray, left: int, right: int) -> float:
    """Halfway between neighbouring scores. ±inf when the cut is outside the range."""
    n = len(values)
    if left < 0:
        return float("-inf")
    if right >= n:
        return float("inf")
    return float(values[left] + values[right]) / 2.0


def fit_single_threshold(s, y, c) -> float:
    """Threshold that minimizes weighted error. Predict 1 for items strictly above the cut.

    Used for tau_llm (s = log-odds) and t_star (s = probe logit).
    """
    s = np.asarray(s, dtype=np.float64)
    y = np.asarray(y)
    c = np.asarray(c, dtype=np.float64)
    if s.shape != y.shape or s.shape != c.shape:
        raise ValueError("s, y, and c must have the same shape")
    if s.size == 0:
        raise ValueError("cannot fit a threshold on an empty set")
    order = np.argsort(s, kind="mergesort")
    s = s[order]
    y = y[order]
    c = c[order]
    harmful = c * (y == 1)
    benign = c * (y == 0)
    # fn[i] + fp[i] is the error when the cut sits after item i (items 0..i called benign).
    fn = np.cumsum(harmful)
    fp = np.sum(benign) - np.cumsum(benign)
    errors = np.concatenate([[np.sum(benign)], fn + fp])
    k = int(np.argmin(errors))
    return _midpoint_at(s, k - 1, k)


def fit_llm_threshold(log_odds, y, c) -> float:
    """tau_llm: the log-odds cut that minimizes LLM-only weighted error."""
    return fit_single_threshold(log_odds, y, c)


def fit_band(z, llm_pred, y, c, budget: float) -> tuple[float, float]:
    """Slide a window of k = round(budget * N) items. Return (t0, t1).

    Items below the window are called benign, items above it harmful,
    and items inside it take the LLM's decision. Ties take the lowest start.
    """
    z = np.asarray(z, dtype=np.float64)
    llm_pred = np.asarray(llm_pred)
    y = np.asarray(y)
    c = np.asarray(c, dtype=np.float64)
    if not (z.shape == llm_pred.shape == y.shape == c.shape):
        raise ValueError("z, llm_pred, y, and c must have the same shape")
    n = int(z.size)
    if n == 0:
        raise ValueError("cannot fit a band on an empty set")
    if not 0.0 <= float(budget) <= 1.0:
        raise ValueError(f"budget must be in [0, 1], got {budget}")
    order = np.argsort(z, kind="mergesort")
    z = z[order]
    llm_pred = llm_pred[order]
    y = y[order]
    c = c[order]
    k = int(round(float(budget) * n))
    k = min(max(k, 0), n)
    below = _cumsum0(c * (y == 1))
    above = _suffix_sum0(c * (y == 0))
    inside = _cumsum0(c * (llm_pred != y))
    starts = np.arange(n - k + 1)
    err = below[starts] + (inside[starts + k] - inside[starts]) + above[starts + k]
    start = int(np.argmin(err))
    t0 = _midpoint_at(z, start - 1, start)
    t1 = _midpoint_at(z, start + k - 1, start + k)
    return t0, t1


def cascade_predict(z, llm_pred, t0, t1) -> np.ndarray:
    """0 at or below t0, 1 at or above t1, otherwise the LLM's call."""
    z = np.asarray(z, dtype=np.float64)
    llm_pred = np.asarray(llm_pred)
    if z.shape != llm_pred.shape:
        raise ValueError("z and llm_pred must have the same shape")
    return np.where(z <= t0, 0, np.where(z >= t1, 1, llm_pred)).astype(np.int64, copy=False)


def llm_predict(log_odds, tau_llm) -> np.ndarray:
    """1 when log-odds are strictly above tau_llm."""
    return (np.asarray(log_odds, dtype=np.float64) > tau_llm).astype(np.int64)
