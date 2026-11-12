"""Score registry. Each score declares an orientation and a unit test.

A signal that fails its orientation test is reported and not flipped.
"""

from __future__ import annotations

from collections.abc import Callable

_REGISTRY: dict[str, dict] = {}


def register_score(name: str, higher_means: str, test_fn: Callable[[], bool]) -> None:
    """Register `name`. `higher_means` is `defer` for Gate 2 signals."""
    if higher_means not in {"defer", "harmful"}:
        raise ValueError("higher_means must be 'defer' or 'harmful'")
    if not callable(test_fn):
        raise TypeError("test_fn must be callable")
    _REGISTRY[name] = {"higher_means": higher_means, "test_fn": test_fn}


def clear_registry() -> None:
    _REGISTRY.clear()


def registered_scores() -> dict[str, dict]:
    return dict(_REGISTRY)


def run_orientation_tests() -> dict[str, bool]:
    """Run every registered test. True means the declared orientation holds."""
    return {name: bool(entry["test_fn"]()) for name, entry in _REGISTRY.items()}


def orientation_holds(signal, probe_error) -> bool:
    """Top decile of `signal` has a higher probe-error rate than the bottom decile."""
    import numpy as np

    signal = np.asarray(signal, dtype=np.float64)
    probe_error = np.asarray(probe_error, dtype=np.float64)
    if signal.shape != probe_error.shape:
        raise ValueError("signal and probe_error must have the same shape")
    if signal.size < 10:
        return False
    lo = np.quantile(signal, 0.1)
    hi = np.quantile(signal, 0.9)
    bottom = probe_error[signal <= lo]
    top = probe_error[signal >= hi]
    if bottom.size == 0 or top.size == 0:
        return False
    return bool(top.mean() > bottom.mean())
