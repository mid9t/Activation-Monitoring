"""Refuse non-baseline reads of held-out splits before the Freeze."""

from __future__ import annotations

from pathlib import Path

# Columns a held-out read may request while prereg/freeze.json is absent.
ALLOWED_HELD_OUT = frozenset(
    {
        "item_id",
        "label",
        "group",
        "template_id",
        "request_id",
        "z",
        "log_odds",
        "thresholds",
    }
)

HELD_OUT_SPLITS = frozenset(
    {
        "heldout_A",
        "heldout_A1",
        "heldout_A2",
        "heldout_B",
        "heldout_C",
        "A",
        "A1",
        "A2",
        "B",
        "C",
    }
)


class GuardError(RuntimeError):
    """A held-out read asked for a column the Freeze has not unlocked."""


def repo_root() -> Path:
    """Directory that contains configs/ and prereg/."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs").is_dir() and (parent / "src").is_dir():
            return parent
    return Path.cwd()


def is_held_out(split: str) -> bool:
    return split in HELD_OUT_SPLITS or split.startswith("heldout_")


def guard_read(split: str, columns: list[str]) -> None:
    """Raise if `columns` goes beyond the baseline on a held-out split before the Freeze.

    Train, train-val, tune, and the pipeline split are not held out.
    Once prereg/freeze.json exists, held-out reads are allowed.
    """
    if not is_held_out(split):
        return
    freeze = repo_root() / "prereg" / "freeze.json"
    if freeze.is_file():
        return
    extra = [column for column in columns if column not in ALLOWED_HELD_OUT]
    if extra:
        raise GuardError(
            f"held-out split {split!r} cannot be read for {extra} "
            f"until {freeze} exists"
        )
