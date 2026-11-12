"""WildJailbreak source-request groups, and the 50/50 split when those groups overlap."""

from __future__ import annotations

import pandas as pd

from amon.data.textutil import group_id_for, side_for_group, user_text


def assign_groups(items: pd.DataFrame) -> pd.DataFrame:
    """Fill group_id from the source request. Near-duplicate merges happen in dedup."""
    out = items.copy()
    if "_source_request" in out.columns:
        texts = out["_source_request"].fillna("").astype(str)
    else:
        texts = out["messages"].map(user_text)
    missing = out["group_id"].fillna("").astype(str) == ""
    out.loc[missing, "group_id"] = texts[missing].map(group_id_for)
    return out


def split_overlapping_wildjailbreak(items: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """If a source request is both vanilla and adversarial, assign the group by hash.

    Train keeps only the vanilla rows of groups assigned to train. Tune keeps only
    the adversarial rows of groups assigned to tune. Groups that do not overlap
    stay on their original side.
    """
    out = items.copy()
    wild = out["source"] == "wildjailbreak"
    if not wild.any():
        return out, {"overlapping_groups": 0, "dropped": 0}
    cells = out.loc[wild, "cell"].astype(str)
    vanilla = cells.str.startswith("vanilla")
    adversarial = cells.str.startswith("adversarial")
    vanilla_groups = set(out.loc[wild & vanilla, "group_id"])
    adversarial_groups = set(out.loc[wild & adversarial, "group_id"])
    overlap = vanilla_groups & adversarial_groups
    if not overlap:
        return out, {"overlapping_groups": 0, "dropped": 0}
    drop = []
    for group_id in overlap:
        side = side_for_group(str(group_id))
        member = wild & (out["group_id"] == group_id)
        if side == "train":
            drop.extend(out.index[member & adversarial].tolist())
            out.loc[member & vanilla, "role"] = "train"
        else:
            drop.extend(out.index[member & vanilla].tolist())
            out.loc[member & adversarial, "role"] = "tune"
    dropped = len(drop)
    out = out.drop(index=drop)
    return out, {"overlapping_groups": len(overlap), "dropped": dropped}
