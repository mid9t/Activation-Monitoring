"""Blinded ordinary-cell audit. Draw the sheet, or score it after the owner labels it.

The key is written and its SHA-256 is printed. Do not open the key before the sheet is labeled.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

from amon.config import load_config, require_paths
from amon.data.audit import audit_sample, flag_template_rows, score_audit
from amon.guard import repo_root
from amon.io import read_json, write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ordinary-cell hand audit.")
    parser.add_argument("--score", action="store_true", help="Score a labeled sheet and write exclusions.")
    args = parser.parse_args(argv)
    if args.score:
        return _score()
    return _draw()


def _draw() -> int:
    cfg = load_config("data.yaml")
    audit_cfg = dict(cfg["ordinary_audit"])
    audit_cfg["shingle_words"] = int(cfg["dedup"]["minhash_shingle_words"])
    pool, templates = _load_pool()
    scored, report = flag_template_rows(pool, templates, audit_cfg)
    write_json(repo_root() / "prereg" / "audit" / "ordinary_audit_filter.json", report)
    threshold = float(audit_cfg["template_containment"])
    flagged = scored.loc[scored["_template_containment"] >= threshold]
    kept = scored.loc[scored["_template_containment"] < threshold]
    english = kept.loc[_english_non_code(kept)]
    sheet, key = audit_sample(
        english,
        flagged,
        int(audit_cfg["audit_n_random"]),
        int(audit_cfg["audit_n_flagged"]),
        int(cfg["seed"]),
    )
    sheet_path, key_path = _paths()
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(sheet_path, index=False)
    key.to_csv(key_path, index=False)
    digest = hashlib.sha256(key_path.read_bytes()).hexdigest()
    print(digest)
    print(f"sheet {sheet_path} rows {len(sheet)}", file=sys.stderr)
    return 0


def _score() -> int:
    sheet_path, key_path = _paths()
    sheet = pd.read_csv(sheet_path, dtype=str).fillna("")
    key = pd.read_csv(key_path, dtype=str)
    if (sheet["label"].astype(str).str.strip() == "").any():
        raise SystemExit("the sheet still has unlabeled rows")
    result = score_audit(sheet, key)
    exclusions = result.pop("exclusions")
    root = repo_root()
    out = root / "prereg" / "ordinary_exclusions.csv"
    exclusions.to_csv(out, index=False)
    payload = result
    previous = root / "prereg" / "audit" / "ordinary_audit_result.json"
    already = previous.is_file() and bool(read_json(previous).get("adjustment"))
    if "raise_containment" in result["rules"] and not already:
        payload["adjustment"] = _redraw_flagged(sheet, key, sheet_path, key_path)
        payload["rule"] = "raise_containment"
    else:
        payload["adjustment"] = None
    write_json(root / "prereg" / "audit" / "ordinary_audit_result.json", payload)
    print(payload["rule"])
    return 0


def _redraw_flagged(sheet, key, sheet_path, key_path) -> dict:
    """One allowed tightening, from 0.5 to 0.7, with 30 new flagged rows."""
    cfg = load_config("data.yaml")
    audit_cfg = dict(cfg["ordinary_audit"])
    audit_cfg["shingle_words"] = int(cfg["dedup"]["minhash_shingle_words"])
    audit_cfg["template_containment"] = 0.7
    pool, templates = _load_pool()
    scored, _report = flag_template_rows(pool, templates, audit_cfg)
    flagged = scored.loc[scored["_template_containment"] >= 0.7]
    flagged = flagged.loc[~flagged["item_id"].astype(str).isin(set(key["item_id"].astype(str)))]
    _extra_sheet, extra_key = audit_sample(flagged.iloc[0:0], flagged, 0, 30, int(cfg["seed"]) + 1)
    start = int(key["audit_id"].astype(int).max()) + 1 if len(key) else 0
    extra_key = extra_key.copy()
    extra_key["audit_id"] = [f"{start + index:04d}" for index in range(len(extra_key))]
    added = pd.DataFrame(
        {"audit_id": extra_key["audit_id"], "text": _texts(flagged, extra_key["item_id"]), "label": ""}
    )
    pd.concat([sheet, added], ignore_index=True).to_csv(sheet_path, index=False)
    pd.concat([key, extra_key], ignore_index=True).to_csv(key_path, index=False)
    data_path = repo_root() / "configs" / "data.yaml"
    text = data_path.read_text()
    data_path.write_text(text.replace("template_containment: 0.5", "template_containment: 0.7", 1))
    return {"template_containment": 0.7, "new_flagged": int(len(extra_key))}


def _texts(pool: pd.DataFrame, item_ids: pd.Series) -> list[str]:
    lookup = pool.set_index(pool["item_id"].astype(str))["_source_request"].astype(str)
    return [str(lookup.get(item, "")) for item in item_ids.astype(str)]


def _english_non_code(frame: pd.DataFrame) -> pd.Series:
    from amon.data.catalog import ENGLISH

    lang = frame["lang"].fillna("").astype(str).str.strip().str.lower().str.split("-").str[0]
    return lang.isin(ENGLISH) & ~frame["is_code"].astype(bool)


def _load_pool() -> tuple[pd.DataFrame, pd.DataFrame]:
    root = require_paths().data_root / "checkpoints" / "01"
    filtered = root / "wildchat_filtered.parquet"
    templates = root / "normalized_jailbreak_prompts.parquet"
    if not filtered.is_file() or not templates.is_file():
        raise SystemExit(
            "ordinary audit needs checkpoints/01/wildchat_filtered.parquet and "
            "normalized_jailbreak_prompts.parquet from stage 01"
        )
    return pd.read_parquet(filtered), pd.read_parquet(templates)


def _paths() -> tuple[Path, Path]:
    folder = repo_root() / "prereg" / "audit"
    return folder / "ordinary_audit_sheet.csv", folder / "ordinary_audit_key.csv"


if __name__ == "__main__":
    raise SystemExit(main())
