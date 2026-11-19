"""Deferral accounting, the pipeline check, and the generator-shift report."""

import numpy as np

from amon.cascade.deferral import (
    budget_check,
    generator_shift,
    mixed_set_errors,
    rule_vs_band,
    share_of_f_recovered,
    write_generator_shift,
    write_pipeline,
)
from amon.cascade.metric import item_costs, weighted_error
from amon.cascade.thresholds import cascade_predict, llm_predict
from amon.decide import label_outcome
from amon.io import read_json


def test_budget_check_ratio():
    assert budget_check(0.05, 0.03, 1.5, 100)["confounded"] is True
    assert budget_check(0.04, 0.03, 1.5, 100)["confounded"] is False


def test_budget_check_zero_band():
    assert budget_check(0.02, 0.0, 1.5, 100)["confounded"] is True
    assert budget_check(0.005, 0.0, 1.5, 100)["confounded"] is False


def test_rule_vs_band_reports_tokens():
    tokens = np.array([15.0, 15.0] + [70.0 / 98.0] * 98)
    rule = np.array([True, True] + [False] * 98)
    band = np.array([False, False, True, True] + [False] * 96)
    report = rule_vs_band("margin", rule, band, tokens, 1.5)
    assert report["rule"]["items"] == 0.02
    assert abs(report["rule"]["tokens"] - 0.30) < 1e-9
    assert report["budget"]["confounded"] is False


def test_mixed_set_errors_matches_weighted_error(tmp_path):
    rng = np.random.default_rng(0)
    n = 40
    groups = np.array(["misses"] * 20 + ["ordinary"] * 20)
    y = np.array([1] * 20 + [0] * 20)
    z = rng.normal(size=n)
    log_odds = rng.normal(size=n)
    weights = {"misses": 5, "ordinary": 50}
    thresholds = {"t_star": 0.1, "tau_llm": 0.0, "t0": -0.5, "t1": 0.5}
    tokens = np.ones(n)
    report = mixed_set_errors(z, log_odds, y, groups, weights, thresholds, tokens, B=30, seed=1)
    costs = item_costs(groups, weights)
    probe = llm_predict(z, thresholds["t_star"])
    llm = llm_predict(log_odds, thresholds["tau_llm"])
    band = cascade_predict(z, llm, thresholds["t0"], thresholds["t1"])
    assert report["probe_only"]["error"] == weighted_error(probe, y, costs)
    assert report["llm_only"]["error"] == weighted_error(llm, y, costs)
    assert report["band"]["error"] == weighted_error(band, y, costs)
    write_pipeline(tmp_path, report)
    payload = read_json(tmp_path / "pipeline.json")
    assert {"error", "lo", "hi"} <= set(payload["probe_only"])
    assert {"items", "tokens", "n_deferred"} <= set(payload["realized_deferral"])


def test_label_outcome():
    assert label_outcome("Positive", True) == "Budget-confounded positive"
    assert label_outcome("Null", True) == "Null"
    assert label_outcome("Inconclusive", False) == "Inconclusive"


def test_share_of_f_recovered():
    y = np.array([1] * 10 + [0])
    f_mask = np.array([True] * 10 + [False])
    band = np.array([0] * 10 + [0])
    rule = np.array([1] * 6 + [0] * 4 + [1])
    report = share_of_f_recovered(rule, band, y, f_mask)
    assert report["F"] == 10
    assert report["recovered"] == 6
    assert report["broken"] == 1
    assert report["net_share"] == 0.5
    empty = share_of_f_recovered(rule[:1], band[:1], y[:1], np.array([False]))
    assert empty["net_share"] is None


def test_generator_shift_report(tmp_path):
    rng = np.random.default_rng(2)
    def draw(n, shift):
        groups = np.array(["misses"] * n + ["ordinary"] * n)
        y = np.array([1] * n + [0] * n)
        scores = np.concatenate([rng.normal(loc=1.0 + shift, size=n), rng.normal(loc=-1.0, size=n)])
        return scores, y, groups

    left = draw(30, 0.0)
    right = draw(30, 0.4)
    report = generator_shift(*left, *right, B=40, seed=1)
    write_generator_shift(tmp_path, report)
    payload = read_json(tmp_path / "generator_shift.json")
    assert set(payload) == {"trainval", "tune"}
    for name in ("trainval", "tune"):
        assert {"auroc", "lo", "hi"} <= set(payload[name])
        assert payload[name]["lo"] <= payload[name]["auroc"] <= payload[name]["hi"]
    assert payload["trainval"]["auroc"] != payload["tune"]["auroc"]
