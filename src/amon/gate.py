"""Threshold fitting and Gate 1 counts. Held-out reads go through guard_read."""

from __future__ import annotations

import numpy as np

from amon.cascade.headroom import headroom_counts
from amon.cascade.metric import item_costs, weighted_error
from amon.cascade.thresholds import cascade_predict, fit_band, fit_llm_threshold, fit_single_threshold, llm_predict
from amon.guard import guard_read
from amon.stats.bootstrap import paired_bootstrap, percentile_interval


def fit_thresholds(z, log_odds, y, groups, weights: dict, budgets: list[float]) -> dict:
    """Fit tau_llm, the bands, then t_star. Order is fixed."""
    costs = item_costs(groups, weights)
    tau = fit_llm_threshold(log_odds, y, costs)
    llm = llm_predict(log_odds, tau)
    bands = {float(budget): fit_band(z, llm, y, costs, float(budget)) for budget in budgets}
    t_star = fit_single_threshold(z, y, costs)
    primary = bands[0.05]
    pred = cascade_predict(z, llm, primary[0], primary[1])
    n = len(z)
    realized = float(np.mean((z > primary[0]) & (z < primary[1]))) if n else 0.0
    return {
        "tau_llm": tau,
        "t0": primary[0],
        "t1": primary[1],
        "t_star": t_star,
        "t0_10": bands[0.10][0],
        "t1_10": bands[0.10][1],
        "realized_tune_deferral": realized,
        "llm_pred": llm,
        "costs": costs,
    }


def _paired_difference(pred_a, pred_b, y, groups, weights, clusters, B, seed):
    def statistic(idx):
        costs = item_costs(groups[idx], weights)
        return weighted_error(pred_a[idx], y[idx], costs) - weighted_error(pred_b[idx], y[idx], costs)

    samples = paired_bootstrap(statistic, len(y), B, seed, clusters=clusters, strata=groups)
    lo, hi = percentile_interval(samples, 0.95)
    return samples, (lo, hi)


def compare_probe_and_llm(z, log_odds, y, groups, weights, tau, t_star, clusters=None, B=2000, seed=1) -> dict:
    costs = item_costs(groups, weights)
    probe_pred = (np.asarray(z) > t_star).astype(int)
    llm_pred = (np.asarray(log_odds) > tau).astype(int)
    probe_error = weighted_error(probe_pred, y, costs)
    llm_error = weighted_error(llm_pred, y, costs)
    samples, interval = _paired_difference(probe_pred, llm_pred, y, groups, weights, clusters, B, seed)
    return {
        "probe_error": probe_error,
        "llm_error": llm_error,
        "difference_probe_minus_llm": probe_error - llm_error,
        "interval": interval,
        "n": int(len(y)),
        "llm_lower": bool(llm_error < probe_error),
    }


def variance_deff(z, log_odds, y, groups, weights, tau, t_star, clusters, B, seed) -> float:
    """Template-bootstrap variance divided by item-bootstrap variance. 1 when either is 0."""
    probe_pred = (np.asarray(z) > t_star).astype(int)
    llm_pred = (np.asarray(log_odds) > tau).astype(int)

    def statistic(idx):
        costs = item_costs(groups[idx], weights)
        return weighted_error(probe_pred[idx], y[idx], costs)

    clustered = paired_bootstrap(statistic, len(y), B, seed, clusters=clusters, strata=groups)
    items = paired_bootstrap(statistic, len(y), B, seed + 1, clusters=None, strata=groups)
    item_var = float(np.var(items))
    if item_var == 0:
        return 1.0
    return float(np.var(clustered) / item_var)


def condition_b(z, llm_pred, y, t0, t1, tokens, split: str) -> dict:
    guard_read(split, ["z", "log_odds", "label", "thresholds"])
    return headroom_counts(z, llm_pred, y, t0, t1, tokens)
