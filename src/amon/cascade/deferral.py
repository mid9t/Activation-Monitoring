"""Realized deferral, the budget-confound check, and the pipeline comparison.

At Gate 1 only the band exists. Frozen rules call rule_vs_band later.
The confound check uses the item rate. The token rate is reported and does not decide.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from amon.cascade.metric import item_costs, weighted_error
from amon.cascade.thresholds import cascade_predict, llm_predict
from amon.io import write_json
from amon.stats.bootstrap import paired_bootstrap, percentile_interval


def deferral_rates(defer: np.ndarray, tokens: np.ndarray) -> dict:
    """{'items': share of rows deferred, 'tokens': share of tokens deferred, 'n_deferred': int}."""
    defer = np.asarray(defer, dtype=bool)
    tokens = np.asarray(tokens, dtype=np.float64)
    if defer.shape != tokens.shape:
        raise ValueError("defer and tokens must have the same shape")
    n = int(defer.size)
    total = float(tokens.sum())
    return {
        "items": (float(defer.sum()) / n) if n else 0.0,
        "tokens": (float(tokens[defer].sum()) / total) if total else 0.0,
        "n_deferred": int(defer.sum()),
    }


def budget_check(rule_rate: float, band_rate: float, ratio: float, n: int) -> dict:
    """{'ratio': rule_rate / band_rate, 'confounded': bool}.

    confounded = rule_rate > ratio * band_rate. If band_rate is 0,
    confounded = rule_rate > 1 / n (the rule deferred anything at all).
    """
    rule_rate = float(rule_rate)
    band_rate = float(band_rate)
    if band_rate == 0.0:
        observed = float("inf") if rule_rate > 0.0 else 0.0
        floor = (1.0 / n) if n else 0.0
        confounded = rule_rate > floor
    else:
        observed = rule_rate / band_rate
        confounded = rule_rate > float(ratio) * band_rate
    return {"ratio": observed, "confounded": bool(confounded)}


def rule_vs_band(
    name: str,
    rule_defer: np.ndarray,
    band_defer: np.ndarray,
    tokens: np.ndarray,
    ratio: float,
) -> dict:
    """Both rates by items and by tokens, plus budget_check on the item rate."""
    rule = deferral_rates(rule_defer, tokens)
    band = deferral_rates(band_defer, tokens)
    return {
        "name": name,
        "rule": rule,
        "band": band,
        "budget": budget_check(rule["items"], band["items"], ratio, int(len(np.asarray(rule_defer)))),
    }


def mixed_set_errors(z, log_odds, y, groups, weights, thresholds, tokens, B, seed) -> dict:
    """Weighted error of probe-only, LLM-only and the 5% band on one mixed-class set.

    Each point estimate has a 95% paired-bootstrap interval stratified by group.
    The band's realized deferral is included.
    """
    z = np.asarray(z, dtype=np.float64)
    log_odds = np.asarray(log_odds, dtype=np.float64)
    y = np.asarray(y)
    groups = np.asarray(groups)
    tokens = np.asarray(tokens, dtype=np.float64)
    costs = item_costs(groups, weights)
    probe = llm_predict(z, thresholds["t_star"])
    llm = llm_predict(log_odds, thresholds["tau_llm"])
    band = cascade_predict(z, llm, thresholds["t0"], thresholds["t1"])
    inside = (z > thresholds["t0"]) & (z < thresholds["t1"])
    report = {
        "probe_only": _error_interval(probe, y, groups, weights, costs, B, seed),
        "llm_only": _error_interval(llm, y, groups, weights, costs, B, seed + 1),
        "band": _error_interval(band, y, groups, weights, costs, B, seed + 2),
        "realized_deferral": deferral_rates(inside, tokens),
    }
    return report


def share_of_f_recovered(
    rule_pred: np.ndarray, band_pred: np.ndarray, y: np.ndarray, f_mask: np.ndarray
) -> dict:
    """Directional-test scale on one shift.

    f_mask marks the band's fixable errors (F).
    net_share = (band errors - rule errors) / F, so a near-zero baseline error cannot decide it.
    Returns {'F', 'recovered', 'broken', 'net_share'} and net_share is None when F is 0.
    """
    rule_pred = np.asarray(rule_pred)
    band_pred = np.asarray(band_pred)
    y = np.asarray(y)
    f_mask = np.asarray(f_mask, dtype=bool)
    count = int(f_mask.sum())
    recovered = int(np.sum(f_mask & (rule_pred == y)))
    broken = int(np.sum((band_pred == y) & (rule_pred != y)))
    if count == 0:
        return {"F": 0, "recovered": recovered, "broken": broken, "net_share": None}
    return {
        "F": count,
        "recovered": recovered,
        "broken": broken,
        "net_share": (recovered - broken) / count,
    }


def generator_shift(trainval_z, trainval_y, trainval_groups, tune_z, tune_y, tune_groups, B, seed) -> dict:
    """Probe AUROC on train-val and on the tune pool, each with a group-stratified interval."""
    return {
        "trainval": _auroc_interval(trainval_z, trainval_y, trainval_groups, B, seed),
        "tune": _auroc_interval(tune_z, tune_y, tune_groups, B, seed + 1),
    }


def write_pipeline(directory: Path, payload: dict) -> None:
    write_json(Path(directory) / "pipeline.json", payload)


def write_generator_shift(directory: Path, payload: dict) -> None:
    write_json(Path(directory) / "generator_shift.json", payload)


def _error_interval(pred, y, groups, weights, costs, B, seed) -> dict:
    point = weighted_error(pred, y, costs)

    def stat(idx, prediction=pred, labels=y, grouped=groups):
        return weighted_error(prediction[idx], labels[idx], item_costs(grouped[idx], weights))

    samples = paired_bootstrap(stat, len(y), B, seed, strata=groups)
    lo, hi = percentile_interval(samples)
    return {"error": point, "lo": lo, "hi": hi}


def _auroc_interval(scores, y, groups, B, seed) -> dict:
    from sklearn.metrics import roc_auc_score

    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y)
    groups = np.asarray(groups)
    point = float(roc_auc_score(y, scores)) if len(np.unique(y)) > 1 else float("nan")

    def stat(idx, sample=scores, labels=y):
        taken_y = labels[idx]
        if len(np.unique(taken_y)) < 2:
            return point
        return float(roc_auc_score(taken_y, sample[idx]))

    samples = paired_bootstrap(stat, len(y), B, seed, strata=groups)
    lo, hi = percentile_interval(samples)
    return {"auroc": point, "lo": lo, "hi": hi}
