"""Registered scores run the orientation test they were given."""

import numpy as np

from amon.registry import clear_registry, orientation_holds, register_score, run_orientation_tests


def test_registered_score_passes_its_orientation_test():
    clear_registry()
    signal = np.linspace(0, 1, 100)
    probe_error = (signal > 0.5).astype(float)
    assert orientation_holds(signal, probe_error)

    def test_fn():
        return orientation_holds(signal, probe_error)

    register_score("synthetic", "defer", test_fn)
    assert run_orientation_tests() == {"synthetic": True}
    clear_registry()
