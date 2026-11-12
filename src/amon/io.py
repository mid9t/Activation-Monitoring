"""JSON that keeps infinities readable by Python's json loader."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def write_json(path: Path, payload) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=True, default=_default) + "\n")


def read_json(path: Path):
    return json.loads(Path(path).read_text())
