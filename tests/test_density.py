"""The tied Gaussian is small at a class mean and larger far away."""

import numpy as np

from amon.density import fit_tied_density, min_mahalanobis_sq


def test_min_mahalanobis_is_smallest_at_a_class_mean():
    rng = np.random.default_rng(0)
    classes = ["harmful", "benign_wj", "ordinary"]
    centers = {"harmful": np.array([0.0, 0.0]), "benign_wj": np.array([8.0, 0.0]), "ordinary": np.array([0.0, 8.0])}
    blocks = []
    labels = []
    for name in classes:
        blocks.append(centers[name] + rng.normal(scale=0.1, size=(30, 2)))
        labels.extend([name] * 30)
    X = np.concatenate(blocks)
    density = fit_tied_density(X, np.array(labels), classes)
    at_mean = min_mahalanobis_sq(centers["harmful"].reshape(1, -1), density)[0]
    far = min_mahalanobis_sq(np.array([[20.0, 20.0]]), density)[0]
    assert at_mean < 1.0
    assert far > at_mean
    assert density["shrinkage"] >= 0.0
