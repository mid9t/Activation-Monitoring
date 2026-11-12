"""Wilson score interval. z is 1.96, as fixed in the design document."""

from __future__ import annotations

import math

Z_95 = 1.96


def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided Wilson interval for k successes in n trials.

    alpha other than 0.05 is rejected: the design fixes z = 1.96.
    """
    if alpha != 0.05:
        raise ValueError("the design fixes the 95% Wilson interval (z = 1.96)")
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if n == 0:
        return (float("nan"), float("nan"))
    z = Z_95
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = p + z2 / (2.0 * n)
    margin = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return (center - margin) / denom, (center + margin) / denom
