"""Condition (b) counts: fixable, unfixable, and in-band errors."""

from __future__ import annotations

import numpy as np

from amon.cascade.thresholds import cascade_predict


def headroom_counts(z, llm_pred, y, t0, t1, tokens) -> dict:
    """F, U, I, R, F/R, and realized deferral by items and by tokens.

    Outside the band the cascade uses the probe. F counts those probe errors
    that the LLM gets right; U counts those both get wrong. I counts LLM errors
    inside the band.
    """
    z = np.asarray(z, dtype=np.float64)
    llm_pred = np.asarray(llm_pred)
    y = np.asarray(y)
    tokens = np.asarray(tokens, dtype=np.float64)
    if not (z.shape == llm_pred.shape == y.shape == tokens.shape):
        raise ValueError("z, llm_pred, y, and tokens must have the same shape")
    outside = (z <= t0) | (z >= t1)
    cascade = cascade_predict(z, llm_pred, t0, t1)
    probe_wrong = cascade != y
    llm_right = llm_pred == y
    llm_wrong = ~llm_right
    f = int(np.sum(outside & probe_wrong & llm_right))
    u = int(np.sum(outside & probe_wrong & llm_wrong))
    i = int(np.sum(~outside & llm_wrong))
    r = f + u + i
    deferred = ~outside
    n = int(z.size)
    token_total = float(tokens.sum())
    return {
        "F": f,
        "U": u,
        "I": i,
        "R": r,
        "F_over_R": (f / r) if r else 0.0,
        "realized_deferral_items": (float(deferred.sum()) / n) if n else 0.0,
        "realized_deferral_tokens": (float(tokens[deferred].sum()) / token_total) if token_total else 0.0,
        "n": n,
        "n_deferred": int(deferred.sum()),
    }


def discount(f: float, u: float, r: float, r_f: float, r_u: float) -> dict:
    """Hand-check discount. Unclear labels are already inside r_F and r_U."""
    f_adj = f * (1.0 - r_f)
    r_adj = r - r_f * f - r_u * u
    frac = (f_adj / r_adj) if r_adj > 0 else 0.0
    return {"F_adj": f_adj, "R_adj": r_adj, "F_adj_over_R_adj": frac}


def shift_passes(f_adj: float, r_adj: float, min_f: float, min_frac: float) -> bool:
    if r_adj <= 0:
        return False
    return (f_adj >= min_f) and ((f_adj / r_adj) >= min_frac)
