"""Probe selection prefers layer 16, then the smallest C, inside one standard error."""

import numpy as np

from amon.probe import fit_probe, select_probe


class Cfg:
    layer_search_C = 1e-3
    cv_folds = 2
    class_weight = "balanced"


def test_fit_is_deterministic_and_oriented():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    y = (X[:, 0] > 0).astype(int)
    first = fit_probe(X, y, C=1e-2)
    second = fit_probe(X, y, C=1e-2)
    assert np.array_equal(first.w, second.w)
    assert first.b == second.b
    logits = first.logit(X)
    assert logits[y == 1].mean() > logits[y == 0].mean()


def test_select_probe_can_return_layer_16():
    rng = np.random.default_rng(1)
    n = 60
    groups = np.repeat(np.arange(10), 6)
    signal = rng.normal(size=n)
    y = (signal > 0).astype(int)
    noise = rng.normal(size=(n, 4))
    informative = np.column_stack([signal, noise[:, 1:]])
    useless = rng.normal(size=(n, 4))
    X = np.stack([useless, informative], axis=0)
    chosen = select_probe(X, y, groups, layers=[8, 16], Cs=[1e-3, 1e-2], cfg=Cfg())
    assert chosen["layer"] == 16
    assert chosen["C"] in {1e-3, 1e-2}
