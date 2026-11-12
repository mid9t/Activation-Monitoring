"""Bootstrap, Wilson interval, and the minimum detectable gain."""

import numpy as np

from amon.stats import mdd_net_items, paired_bootstrap, wilson


def test_paired_bootstrap_repeats_and_shares_indices():
    seen = []

    def stat(idx):
        seen.append(np.asarray(idx).copy())
        arm_a = float(idx.sum())
        arm_b = float(np.square(idx).sum())
        return (arm_a, arm_b)

    first = paired_bootstrap(stat, n=20, B=10, seed=1)
    seen.clear()
    second = paired_bootstrap(stat, n=20, B=10, seed=1)
    assert np.array_equal(first, second)
    third = paired_bootstrap(lambda idx: float(idx.sum()), n=20, B=10, seed=2)
    assert not np.array_equal(first[:, 0], third)
    # One call per resample: both arms were computed from that same index vector.
    assert len(seen) == 10
    for idx, (arm_a, arm_b) in zip(seen, second):
        assert arm_a == float(idx.sum())
        assert arm_b == float(np.square(idx).sum())


def test_cluster_resampling_keeps_clusters_whole():
    clusters = np.array([0, 0, 1, 1, 2, 2])

    def stat(idx):
        return idx

    draws = paired_bootstrap(stat, n=6, B=25, seed=3, clusters=clusters)
    for row in draws:
        counts = {item: int(np.sum(row == item)) for item in range(6)}
        assert counts[0] == counts[1]
        assert counts[2] == counts[3]
        assert counts[4] == counts[5]


def test_strata_are_resampled_separately():
    strata = np.array([0] * 10 + [1] * 10)

    def stat(idx):
        labels = strata[idx]
        return ((labels == 0).sum(), (labels == 1).sum())

    draws = paired_bootstrap(stat, n=20, B=15, seed=4, strata=strata)
    assert np.all(draws[:, 0] == 10)
    assert np.all(draws[:, 1] == 10)


def test_wilson_3_of_30():
    lo, hi = wilson(3, 30)
    # Independent evaluation of the design's formula with z = 1.96.
    assert lo == pytest_approx(0.0346)
    assert hi == pytest_approx(0.2562)


def pytest_approx(value):
    import pytest

    return pytest.approx(value, abs=5e-5)


def test_mdd_for_300_discordant_items():
    gain = mdd_net_items(300, deff=1.0)
    assert gain == pytest_approx(2.8 * (300 ** 0.5))
    assert 45 < gain < 55
