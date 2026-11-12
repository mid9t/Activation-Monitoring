"""Stage 03. Score Llama Guard. Train-val is scored before the tune pool."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from amon.config import load_config, require_paths
from amon.data.textutil import user_text
from amon.llm import Policy, answer_token_ids, score_guard
from amon.provenance import publish_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score Llama Guard 3.")
    parser.add_argument("--extra-category", default=None, help="Owner-supplied category line. Not used unless passed.")
    args = parser.parse_args(argv)
    paths = require_paths()
    models = load_config("models.yaml")
    llm_cfg = load_config("llm.yaml")
    sha = models["guard"]["revision"]
    policy = Policy(name=llm_cfg["policy"], extra_category=args.extra_category)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if paths.device == "cuda" else torch.float16
    tok = AutoTokenizer.from_pretrained(models["guard"]["repo"], revision=sha)
    tok.pad_token = tok.eos_token
    answer_token_ids(tok)
    model = AutoModelForCausalLM.from_pretrained(models["guard"]["repo"], revision=sha, torch_dtype=dtype)
    model.to(paths.device)
    cfg = SimpleNamespace(data_root=paths.data_root, token_budget_per_batch=32000, seed=1, model_sha=sha)
    out_dir = paths.data_root / "llm" / f"llama-guard-3-8b@{sha}"
    order = ["trainval", "train", "tune", "heldout_A1", "heldout_A2", "heldout_B", "heldout_C", "pipeline"]
    for role in order:
        path = score_guard(role, model, tok, policy, out_dir / f"{role}.parquet", cfg=cfg)
        frame = pd.read_parquet(path)
        share = float((frame["mass2"] >= float(llm_cfg["min_two_token_mass"])).mean())
        print(f"{role}: two-token mass >= {llm_cfg['min_two_token_mass']} on {share:.4f}")
        if share < 0.99:
            raise SystemExit(f"{role} failed the two-token mass check ({share:.4f})")
        publish_manifest(out_dir / "manifests" / role / "manifest.json", f"03_llm_{role}.json")
    _write_policy_review(paths.data_root, out_dir / "trainval.parquet")
    return 0


def _write_policy_review(root: Path, scores_path: Path) -> None:
    items = pd.read_parquet(root / "processed" / "items.parquet")
    scores = pd.read_parquet(scores_path)
    merged = items.merge(scores, on="item_id")
    harmful = merged.loc[(merged["role"] == "trainval") & (merged["label"] == 1) & (merged["log_odds"] <= 0)]
    missed = harmful.head(50)
    missed = missed.assign(text=missed["messages"].map(user_text))
    dest = root / "llm" / "policy_review.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    missed[["item_id", "text", "log_odds"]].to_csv(dest, index=False)
    print(f"wrote {len(missed)} missed harmful train-val items to {dest}")


if __name__ == "__main__":
    raise SystemExit(main())
