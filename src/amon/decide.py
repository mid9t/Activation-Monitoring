"""Gate 1 decision. The branch is mechanical; nothing is decided by eye."""

from __future__ import annotations

from amon.cascade.headroom import shift_passes
from amon.stats.mdd import mdd_net_items

# Ties in discounted F go to A, then C, then B.
SHIFT_TIE_RANK = {"A": 0, "C": 1, "B": 2}


def condition_a_pass(per_set: dict[str, dict]) -> bool:
    """Pass when the LLM's point estimate is lower on at least two of the four sets."""
    wins = 0
    for row in per_set.values():
        if row["llm_error"] < row["probe_error"]:
            wins += 1
    return wins >= 2


def kept_shift(discounts: dict[str, dict], min_f: float, min_frac: float) -> str | None:
    """Largest discounted F among shifts that pass (b). Ties: A, then C, then B."""
    passing = []
    for shift, row in discounts.items():
        if shift_passes(row["F_adj"], row["R_adj"], min_f, min_frac):
            passing.append(shift)
    if not passing:
        return None
    passing.sort(key=lambda shift: (-discounts[shift]["F_adj"], SHIFT_TIE_RANK.get(shift, 9)))
    return passing[0]


def detectable_gain(f_adj: float, deferred_n: int, deff: float) -> dict:
    """Smallest net gain the final test could detect if every deferred item can change.

    A shift cannot support a positive result when even fixing every adjusted F item
    stays under that gain.
    """
    gain = mdd_net_items(int(deferred_n), deff)
    return {
        "m_discordant_upper": int(deferred_n),
        "deff": float(deff),
        "min_detectable_net_gain": gain,
        "max_possible_net_gain": float(f_adj),
        "can_support_positive": float(f_adj) >= gain,
    }


def decide(
    condition_a: dict,
    discounts: dict[str, dict],
    handcheck: dict,
    *,
    min_f: float,
    min_frac: float,
    kill_mislabeled: int,
    materiality: float,
    deferred: dict[str, int],
    deff: dict[str, float],
) -> dict:
    """Apply the checks in order: (a), label noise, tie-break, then (b)."""
    a_pass = condition_a_pass(condition_a["sets"])
    errors = handcheck["r_F"]["k"] + handcheck["r_U"]["k"]
    label_kill = errors > kill_mislabeled
    tie = bool(handcheck.get("tie_break"))
    shift = None if (not a_pass or label_kill or tie) else kept_shift(discounts, min_f, min_frac)
    if not a_pass:
        branch = "fail_a"
    elif label_kill:
        branch = "kill_label_noise"
    elif tie:
        branch = "tie_break"
    elif shift is None:
        branch = "fail_b"
    else:
        branch = "pass"
    gains = {
        name: detectable_gain(discounts[name]["F_adj"], deferred.get(name, 0), deff.get(name, 1.0))
        for name in discounts
    }
    return {
        "branch": branch,
        "kept_shift": shift,
        "condition_a_pass": a_pass,
        "condition_b_pass": branch == "pass",
        "r_F": handcheck["r_F"],
        "r_U": handcheck["r_U"],
        "tie_break": tie,
        "mislabeled_errors": errors,
        "materiality_min_swap_frac": materiality,
        "detectable_gain": gains,
    }
