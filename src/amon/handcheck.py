"""Blind hand-check sample and the mislabel discount."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from amon.cascade.headroom import discount, shift_passes
from amon.stats.wilson import wilson

SHIFTS = ("A", "B", "C")


def allocate(counts: dict[str, int], n: int, minimum: int) -> dict[str, int]:
    """Spread `n` draws across shifts in proportion to `counts`, with a floor per shift that has items."""
    eligible = {key: int(value) for key, value in counts.items() if int(value) > 0}
    alloc = {key: 0 for key in counts}
    if not eligible or n <= 0:
        return alloc
    n = min(n, sum(eligible.values()))
    shifts = sorted(eligible)
    floor = minimum
    while floor > 0 and floor * len(shifts) > n:
        floor -= 1
    for shift in shifts:
        alloc[shift] = min(floor, eligible[shift])
    leftover = n - sum(alloc.values())
    room = {shift: eligible[shift] - alloc[shift] for shift in shifts}
    weight_sum = sum(eligible[shift] for shift in shifts if room[shift] > 0)
    raw = {}
    for shift in shifts:
        raw[shift] = (leftover * eligible[shift] / weight_sum) if room[shift] > 0 and weight_sum else 0.0
    for shift in shifts:
        give = min(room[shift], int(np.floor(raw[shift])))
        alloc[shift] += give
    left = n - sum(alloc.values())
    order = sorted(shifts, key=lambda shift: (raw[shift] - np.floor(raw[shift]), eligible[shift]), reverse=True)
    for shift in order:
        if left <= 0:
            break
        if alloc[shift] < eligible[shift]:
            alloc[shift] += 1
            left -= 1
    return alloc


def _draw(frame: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if n <= 0 or frame.empty:
        return frame.iloc[0:0]
    index = rng.choice(len(frame), size=min(n, len(frame)), replace=False)
    return frame.iloc[np.sort(index)]


def sample_handcheck(items: pd.DataFrame, cfg: dict, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """30 F, 20 U, 15 decoys. Returns (sample.csv columns, key.csv columns) in shuffled order.

    `items` has item_id, text, shift, stratum in {F, U, decoy}, label, probe_call, llm_call.
    The caller marks stratum from the frozen thresholds before calling this.
    """
    hand = cfg["handcheck"] if isinstance(cfg, dict) else cfg
    rng = np.random.default_rng(seed)
    pieces = []
    for stratum, n, minimum in (
        ("F", int(hand["n_F"]), int(hand["min_per_shift"])),
        ("U", int(hand["n_U"]), int(hand["min_per_shift"])),
        ("decoy", int(hand["n_decoys"]), 1),
    ):
        pool = items.loc[items["stratum"] == stratum]
        counts = {shift: int((pool["shift"] == shift).sum()) for shift in SHIFTS}
        alloc = allocate(counts, n, minimum)
        for shift, take in alloc.items():
            pieces.append(_draw(pool.loc[pool["shift"] == shift], take, rng))
    chosen = pd.concat(pieces, ignore_index=True) if pieces else items.iloc[0:0]
    order = rng.permutation(len(chosen))
    chosen = chosen.iloc[order].reset_index(drop=True)
    chosen.insert(0, "hc_id", [f"hc_{index:04d}" for index in range(len(chosen))])
    sample = chosen[["hc_id", "text"]].copy()
    key = chosen[
        ["hc_id", "item_id", "stratum", "shift", "label", "probe_call", "llm_call"]
    ].copy()
    return sample, key


def key_sha256(path: Path) -> str:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return digest


def score_handcheck(key: pd.DataFrame, labels: pd.DataFrame, counts: dict[str, dict]) -> dict:
    """r_F and r_U, with unclear counted as mislabeled, plus the per-shift discount.

    `counts` maps a shift to F, U, R from stage 06. The tie-break recomputes pass (b)
    at the Wilson upper bound of r_F.
    """
    key = key.rename(columns={"label": "label_data"})
    labels = labels.rename(columns={"label": "label_human"})
    merged = key.merge(labels, on="hc_id", how="left")
    if merged["label_human"].isna().any():
        missing = int(merged["label_human"].isna().sum())
        raise ValueError(f"labels.csv is missing {missing} hc_id rows")

    def mislabeled(frame: pd.DataFrame) -> np.ndarray:
        human = frame["label_human"].astype(str).str.strip().str.lower()
        data = frame["label_data"].astype(str).str.strip().str.lower()
        return ((human == "unclear") | (human != data)).to_numpy()

    def rate(stratum: str) -> dict:
        frame = merged.loc[merged["stratum"] == stratum]
        k = int(mislabeled(frame).sum()) if len(frame) else 0
        n = int(len(frame))
        lo, hi = wilson(k, n) if n else (float("nan"), float("nan"))
        return {"k": k, "n": n, "rate": (k / n) if n else float("nan"), "wilson": [lo, hi]}

    r_f = rate("F")
    r_u = rate("U")
    decoys = merged.loc[merged["stratum"] == "decoy"]
    decoy_disagree = float(mislabeled(decoys).mean()) if len(decoys) else float("nan")
    discounted = {}
    point = {}
    upper = {}
    for shift, row in counts.items():
        point[shift] = discount(row["F"], row["U"], row["R"], r_f["rate"], r_u["rate"])
        upper[shift] = discount(row["F"], row["U"], row["R"], r_f["wilson"][1], r_u["rate"])
        discounted[shift] = point[shift]
    return {
        "r_F": r_f,
        "r_U": r_u,
        "decoy_disagreement": decoy_disagree,
        "discount": discounted,
        "discount_at_wilson_upper": upper,
        "tie_break": False,
    }


def tie_break_flag(result: dict, min_f: float, min_frac: float) -> bool:
    """True when any shift's pass/fail changes at the Wilson upper bound of r_F."""
    changed = False
    for shift, point in result["discount"].items():
        upper = result["discount_at_wilson_upper"][shift]
        point_pass = shift_passes(point["F_adj"], point["R_adj"], min_f, min_frac)
        upper_pass = shift_passes(upper["F_adj"], upper["R_adj"], min_f, min_frac)
        if point_pass != upper_pass:
            changed = True
    return changed
