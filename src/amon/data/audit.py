"""Ordinary-cell hygiene: template containment, owner exclusions, and the blinded audit."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from amon.data.catalog import ENGLISH
from amon.data.dedup import _shingles
from amon.provenance import sha256_file
from amon.stats.wilson import wilson

# A bracketed or braced span that is a template slot, not part of the request.
_PLACEHOLDER = re.compile(
    r"\[[^\[\]]*?(?:prompt|insert|question)[^\[\]]*\]"
    r"|\{[^{}]*?(?:prompt|insert|question)[^{}]*\}",
    re.IGNORECASE,
)

SELECTED_CODE = (
    "src/amon/data/build.py",
    "src/amon/data/dedup.py",
    "src/amon/data/groups.py",
    "src/amon/data/templates.py",
    "src/amon/data/embed.py",
    "src/amon/data/normalize.py",
    "src/amon/data/audit.py",
)


def normalize_text(text: str) -> str:
    """Lowercase, strip placeholder slots, and collapse whitespace."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        text = ""
    cleaned = _PLACEHOLDER.sub(" ", str(text).lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def shingle_hashes(text: str, width: int) -> set[int]:
    """blake2b-64 hashes of the width-word shingles of normalized text."""
    hashes: set[int] = set()
    for shingle in _shingles(normalize_text(text), width):
        if not shingle:
            continue
        digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest()
        hashes.add(int.from_bytes(digest, "little"))
    return hashes


def template_index(
    templates: pd.DataFrame, width: int, min_shingles: int
) -> tuple[dict[int, list[int]], np.ndarray]:
    """Shingle hash -> template row numbers, and each template's shingle count.

    Templates with fewer than `min_shingles` are left at size 0 and stay out of the index.
    """
    texts = _texts(templates)
    sizes = np.zeros(len(texts), dtype=np.int64)
    index: dict[int, list[int]] = {}
    for row, text in enumerate(texts):
        hashes = shingle_hashes(text, width)
        if len(hashes) < int(min_shingles):
            continue
        sizes[row] = len(hashes)
        for digest in hashes:
            index.setdefault(digest, []).append(row)
    return index, sizes


def template_containment(
    texts: list[str],
    index: dict[int, list[int]],
    sizes: np.ndarray,
    width: int,
) -> pd.DataFrame:
    """Columns: best_template_row (int, -1 if none), containment (float in [0, 1])."""
    n_templates = int(len(sizes))
    best_row = np.full(len(texts), -1, dtype=np.int64)
    best_score = np.zeros(len(texts), dtype=np.float64)
    if n_templates == 0 or not index:
        return pd.DataFrame({"best_template_row": best_row, "containment": best_score})
    counts = np.zeros(n_templates, dtype=np.int32)
    positive = sizes > 0
    for item, text in enumerate(texts):
        counts[:] = 0
        for digest in shingle_hashes(text, width):
            rows = index.get(digest)
            if rows:
                counts[rows] += 1
        if not counts.any():
            continue
        ratios = np.divide(counts, sizes, out=np.zeros(n_templates, dtype=np.float64), where=positive)
        row = int(np.argmax(ratios))
        score = float(ratios[row])
        if score > 0.0:
            best_row[item] = row
            best_score[item] = score
        if item and item % 50000 == 0:
            print(f"template containment {item}/{len(texts)}", flush=True)
    return pd.DataFrame({"best_template_row": best_row, "containment": best_score})


def flag_template_rows(pool: pd.DataFrame, templates: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Adds _template_containment and _template_id to pool; returns the report dict."""
    width = int(cfg.get("shingle_words", cfg.get("minhash_shingle_words", 5)))
    minimum = int(cfg["min_template_shingles"])
    threshold = float(cfg["template_containment"])
    index, sizes = template_index(templates, width, minimum)
    scored = template_containment(_texts(pool), index, sizes, width)
    out = pool.reset_index(drop=True).copy()
    out["_template_containment"] = scored["containment"].to_numpy()
    template_ids = _ids(templates)
    best = scored["best_template_row"].to_numpy()
    chosen = []
    for row in best.tolist():
        chosen.append("" if row < 0 or row >= len(template_ids) else template_ids[row])
    out["_template_id"] = chosen
    flagged = out["_template_containment"] >= threshold
    report = {
        "pool_size": int(len(out)),
        "flagged": int(flagged.sum()),
        "flagged_by_cell": _flagged_by_cell(out, flagged),
        "containment_histogram": _histogram(out["_template_containment"].to_numpy()),
        "top_templates": _top_templates(out.loc[flagged, "_template_id"], 20),
        "template_containment": threshold,
    }
    return out, report


def apply_exclusions(pool: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, int]:
    """Drops item_ids listed in the owner's exclusion file; a missing file means no exclusions."""
    path = Path(path)
    if not path.is_file():
        return pool, 0
    listed = pd.read_csv(path, dtype=str)
    if listed.empty or "item_id" not in listed.columns:
        return pool, 0
    banned = set(listed["item_id"].dropna().astype(str))
    if not banned:
        return pool, 0
    keep = ~pool["item_id"].astype(str).isin(banned)
    return pool.loc[keep].reset_index(drop=True), int((~keep).sum())


def exclusion_fingerprint(path: Path | str) -> str:
    """SHA-256 of the exclusion file. A missing file has a stable sentinel."""
    path = Path(path)
    if not path.is_file():
        return "missing"
    return sha256_file(path)


def audit_sample(
    pool: pd.DataFrame,
    flagged: pd.DataFrame,
    n_random: int,
    n_flagged: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (sheet, key). The sheet is shuffled and blinded; the key says which rows were flagged."""
    rng = np.random.default_rng(seed)
    random_rows = _draw(pool, n_random, rng)
    flagged_rows = _draw(flagged, n_flagged, rng)
    random_rows = random_rows.copy()
    flagged_rows = flagged_rows.copy()
    random_rows["kind"] = "random"
    flagged_rows["kind"] = "flagged"
    drawn = pd.concat([random_rows, flagged_rows], ignore_index=True)
    drawn["audit_id"] = [f"{index:04d}" for index in range(len(drawn))]
    key = pd.DataFrame(
        {
            "audit_id": drawn["audit_id"].astype(str),
            "item_id": drawn["item_id"].astype(str),
            "kind": drawn["kind"].astype(str),
            "containment": drawn["_template_containment"].astype(float)
            if "_template_containment" in drawn.columns
            else 0.0,
        }
    )
    order = rng.permutation(len(drawn))
    if len(drawn) > 1 and np.array_equal(order, np.arange(len(drawn))):
        order = order.copy()
        order[0], order[1] = order[1], order[0]
    sheet = pd.DataFrame(
        {
            "audit_id": drawn["audit_id"].to_numpy()[order],
            "text": _texts(drawn)[order] if len(drawn) else np.array([], dtype=object),
            "label": "",
        }
    )
    return sheet.reset_index(drop=True), key


def score_audit(sheet: pd.DataFrame, key: pd.DataFrame) -> dict:
    """Residual rate on the random rows and precision on the flagged rows."""
    joined = sheet.merge(key, on="audit_id", how="inner")
    labels = joined["label"].fillna("").astype(str).str.strip().str.lower()
    random = joined["kind"].eq("random")
    flagged = joined["kind"].eq("flagged")
    residual_hit = labels.isin(["harmful", "unclear"])
    precision_hit = labels.isin(["harmful", "jailbreak"])
    residual_k = int((random & residual_hit).sum())
    residual_n = int(random.sum())
    precision_k = int((flagged & precision_hit).sum())
    precision_n = int(flagged.sum())
    lo, hi = wilson(residual_k, residual_n) if residual_n else (float("nan"), float("nan"))
    rules = []
    if precision_n and precision_k < 24 and precision_n == 30:
        rules.append("raise_containment")
    elif precision_n and precision_k / precision_n < 0.8 and precision_n != 30:
        rules.append("raise_containment")
    if residual_k >= 3:
        rules.append("contingency_sweep")
    else:
        rules.append("exclude_and_rebuild")
    excluded = joined.loc[random & residual_hit, ["item_id"]].copy()
    excluded["reason"] = labels.loc[random & residual_hit].to_numpy()
    return {
        "residual_k": residual_k,
        "residual_n": residual_n,
        "residual_rate": (residual_k / residual_n) if residual_n else 0.0,
        "residual_wilson": [lo, hi],
        "precision_k": precision_k,
        "precision_n": precision_n,
        "precision": (precision_k / precision_n) if precision_n else 0.0,
        "rules": rules,
        "rule": rules[0],
        "exclusions": excluded.reset_index(drop=True),
    }


def _texts(frame: pd.DataFrame) -> np.ndarray:
    if "_source_request" in frame.columns:
        return frame["_source_request"].fillna("").astype(str).to_numpy()
    if "text" in frame.columns:
        return frame["text"].fillna("").astype(str).to_numpy()
    raise KeyError("pool rows need _source_request or text")


def _ids(frame: pd.DataFrame) -> list[str]:
    if "item_id" in frame.columns:
        return frame["item_id"].astype(str).tolist()
    if "template_id" in frame.columns:
        return frame["template_id"].astype(str).tolist()
    return [str(index) for index in range(len(frame))]


def _draw(frame: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if n <= 0 or frame.empty:
        return frame.iloc[0:0]
    if len(frame) <= n:
        return frame
    index = np.sort(rng.choice(len(frame), size=int(n), replace=False))
    return frame.iloc[index]


def _is_english(code: str) -> bool:
    return str(code or "").strip().lower().split("-")[0] in ENGLISH


def _flagged_by_cell(pool: pd.DataFrame, flagged: pd.Series) -> dict[str, int]:
    """Counts by the cell a flagged row would have been eligible for."""
    if not flagged.any():
        return {}
    rows = pool.loc[flagged]
    text = rows["_source_request"].fillna("").astype(str) if "_source_request" in rows.columns else pd.Series([""] * len(rows))
    code = rows["is_code"].astype(bool) if "is_code" in rows.columns else pd.Series(False, index=rows.index)
    from amon.data.textutil import is_python

    python = text.map(is_python) if len(rows) else pd.Series(dtype=bool)
    english = rows["lang"].map(_is_english) if "lang" in rows.columns else pd.Series(True, index=rows.index)
    cell = np.where(python, "python", np.where(code, "code", np.where(english, "ordinary", "non_english")))
    counts = pd.Series(cell).value_counts()
    return {str(name): int(count) for name, count in counts.items()}


def _histogram(scores: np.ndarray) -> dict[str, int]:
    counts, _edges = np.histogram(scores, bins=np.linspace(0.0, 1.0, 11))
    return {f"{index / 10:.1f}": int(count) for index, count in enumerate(counts.tolist())}


def _top_templates(template_ids: pd.Series, limit: int) -> list[dict]:
    if template_ids.empty:
        return []
    counts = template_ids.astype(str).value_counts()
    rows = []
    for template_id, count in counts.head(limit).items():
        if not template_id:
            continue
        rows.append({"template_id": str(template_id), "n": int(count)})
    return rows
