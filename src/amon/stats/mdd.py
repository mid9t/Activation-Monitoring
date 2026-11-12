"""Minimum detectable net gain for a paired comparison on one shift."""

from __future__ import annotations

import math


def mdd_net_items(m_discordant: int, deff: float = 1.0) -> float:
    """2.8 * sqrt(m * DEFF). m is the number of items whose cascade outcome differs.

    DEFF is 1 for independent items. On A1 it is the template-bootstrap variance
    divided by the item-bootstrap variance.
    """
    if m_discordant < 0:
        raise ValueError("m_discordant must be non-negative")
    if deff < 0:
        raise ValueError("deff must be non-negative")
    return 2.8 * math.sqrt(m_discordant * deff)
