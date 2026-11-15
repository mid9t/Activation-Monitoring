"""Role lists, smoke routing, and the seeded sample used by stages 02 and 03."""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROLES = [
    "train",
    "trainval",
    "tune",
    "heldout_A1",
    "heldout_A2",
    "heldout_B",
    "heldout_C",
    "pipeline",
]
GUARD_ROLES = [role for role in ROLES if role != "train"]


def parse_roles(text: str, *, allowed: list[str], default: list[str]) -> list[str]:
    """A comma list, or `default` when the argument is empty. Unknown names are rejected."""
    if not str(text).strip():
        return list(default)
    roles = [part.strip() for part in str(text).split(",") if part.strip()]
    unknown = [role for role in roles if role not in allowed]
    if unknown:
        known = ", ".join(allowed)
        raise SystemExit(f"unknown role(s): {', '.join(unknown)}. Known roles: {known}")
    return roles


def resolve_out_root(data_root: Path, out_root: Path, limit: int) -> Path:
    """Inputs stay on the data root. A --limit run writes under smoke/ unless $AMON_OUT is set."""
    data_root = Path(data_root)
    out_root = Path(out_root)
    if limit and out_root.resolve() == data_root.resolve():
        return data_root / "smoke"
    return out_root


def limit_items(items, limit: int, seed: int = 0):
    """Take `limit` rows, stratified by label, and keep the caller's row order.

    Both classes are included when `limit` is at least the number of classes, so the
    mass and generate checks see harmful and benign rows.
    """
    frame = items.reset_index(drop=True)
    if limit <= 0 or limit >= len(frame):
        return frame
    if "label" not in frame.columns:
        raise ValueError("--limit sampling needs a label column")
    labels = frame["label"].to_numpy()
    classes = [np.flatnonzero(labels == label) for label in dict.fromkeys(labels.tolist())]
    shares = np.array([len(idx) * limit / len(frame) for idx in classes], dtype=np.float64)
    counts = np.floor(shares).astype(int)
    leftover = int(limit - counts.sum())
    for index in np.argsort(-(shares - counts)):
        if leftover <= 0:
            break
        if counts[index] < len(classes[index]):
            counts[index] += 1
            leftover -= 1
    if limit >= len(classes):
        for index, pool in enumerate(classes):
            if counts[index] == 0 and len(pool):
                donor = int(np.argmax(counts))
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[index] += 1
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for pool, count in zip(classes, counts):
        count = int(count)
        if count <= 0:
            continue
        pick = rng.choice(pool, size=min(count, len(pool)), replace=False)
        chosen.extend(int(value) for value in np.atleast_1d(pick))
    chosen.sort()
    return frame.iloc[chosen].reset_index(drop=True)
