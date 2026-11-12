"""Acceptance checks for an activation cache."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from amon.cache.extract import forward_hidden, layer_name, left_pad
from amon.data.render import render_llama3, tokenize_rendered


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = left.astype(np.float64).ravel()
    right = right.astype(np.float64).ravel()
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    if denom == 0:
        return 1.0 if np.allclose(left, right) else 0.0
    return float(np.dot(left, right) / denom)


def verify_cache(split_dir: Path, ids: list[str], model, tok, n_check: int = 200) -> dict:
    """Row counts, finite non-zero rows, and agreement with unpadded single-item reruns."""
    split_dir = Path(split_dir)
    meta = pd.read_parquet(split_dir / "meta.parquet")
    problems = []
    if meta["item_id"].astype(str).tolist() != list(ids):
        problems.append("meta item_id order does not match the ids file")
    layers = []
    for path in sorted(split_dir.glob("last_L*.npy")):
        layers.append(int(path.stem.split("_L")[1]))
    arrays = {}
    for layer in layers:
        for position in ("last", "mean"):
            array = np.load(split_dir / f"{position}_{layer_name(layer)}.npy", mmap_mode="r")
            arrays[(position, layer)] = array
            if array.shape[0] != len(ids):
                problems.append(f"{position} L{layer} has {array.shape[0]} rows, expected {len(ids)}")
            if not np.isfinite(array).all():
                problems.append(f"{position} L{layer} contains a non-finite value")
            zero = np.all(array == 0, axis=1)
            if bool(np.any(zero)):
                problems.append(f"{position} L{layer} has an all-zero row")
    data_root = split_dir.parent.parent.parent
    items = pd.read_parquet(data_root / "processed" / "items.parquet").set_index("item_id")
    rng = np.random.default_rng(0)
    check_n = min(n_check, len(ids))
    chosen = rng.choice(len(ids), size=check_n, replace=False) if check_n else []
    pad_id = int(getattr(tok, "pad_token_id", None) or 0)
    min_cosine = 1.0
    import torch

    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        for index in chosen:
            item_id = ids[int(index)]
            messages = json.loads(items.loc[item_id, "messages"])
            sequence = tokenize_rendered(render_llama3(messages, tok), tok)
            ids_tensor, mask = left_pad([sequence], pad_id)
            hidden = forward_hidden(model, ids_tensor.to(device), mask.to(device), layers)
            for layer, (last, mean) in hidden.items():
                for position, vector in (("last", last), ("mean", mean)):
                    stored = np.asarray(arrays[(position, layer)][int(index)])
                    fresh = vector[0].detach().float().cpu().numpy()
                    score = _cosine(stored, fresh)
                    min_cosine = min(min_cosine, score)
    passed = not problems and (min_cosine >= 0.999 or check_n == 0)
    if min_cosine < 0.999 and check_n:
        problems.append(f"min cosine {min_cosine:.6f} is below 0.999")
    return {"pass": passed and not problems, "problems": problems, "min_cosine": min_cosine, "n_check": int(check_n)}
