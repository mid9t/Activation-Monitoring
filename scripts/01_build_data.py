"""Stage 01. Build the item table, then stop for the owner to tag the pre-registration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from amon.config import load_config, require_paths
from amon.data import acceptance, normalize, render_and_tokenize, select_pools
from amon.data.embed import embed_texts
from amon.data.sources import download_raw, load_source
from amon.io import write_json
from amon.memo import write_draft
from amon.provenance import publish_manifest, sha256_file, write_manifest


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Build processed items and splits.").parse_args(argv)
    paths = require_paths()
    data_cfg = load_config("data.yaml")
    models = load_config("models.yaml")
    root = paths.data_root
    frames = {}
    for name in (
        "wildjailbreak",
        "wildchat",
        "jailbreak_prompts",
        "strongreject",
        "llama3_jailbreaks",
        "orbench",
        "xstest",
    ):
        spec = _spec(models, name)
        revision = spec["revision"]
        download_raw(name, spec["repo"], revision, root / "raw" / name / revision)
        raw = load_source(name, models)
        frames[name] = normalize(name, raw)
        print(f"{name}: {len(frames[name])} normalized rows")
    def embed_fn(texts):
        return embed_texts(texts, models["dedup"]["repo"], models["dedup"]["revision"])

    items, notes = select_pools(frames, data_cfg, embed_fn=embed_fn)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        models["monitored"]["repo"], revision=models["monitored"]["revision"]
    )
    items, tokens, token_report = render_and_tokenize(items, tok, int(data_cfg["max_tokens"]))
    notes["tokens"] = token_report
    verdict = acceptance(items, tokens, notes, data_cfg)
    processed = root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    public = items.drop(columns=[column for column in items.columns if column.startswith("_")])
    public.to_parquet(processed / "items.parquet", index=False)
    tokens.to_parquet(processed / "tokens.parquet", index=False)
    write_json(processed / "dedup_report.json", notes["dedup"])
    splits = processed / "splits"
    splits.mkdir(exist_ok=True)
    hashes = {}
    for role, frame in public.groupby("role"):
        path = splits / f"{role}.ids"
        ids = sorted(frame["item_id"].astype(str))
        path.write_text("".join(f"{item}\n" for item in ids))
        hashes[role] = sha256_file(path)
    if notes.get("a1_insertions") is not None:
        notes["a1_insertions"].to_parquet(processed / "a1_insertions.parquet", index=False)
    rubric = Path("prereg/rubric.md")
    rubric_sha = sha256_file(rubric) if rubric.is_file() else None
    write_draft(Path("prereg/gate1_memo.md"), hashes, int(data_cfg["seed"]), rubric_sha)
    manifest = write_manifest(
        processed,
        list((root / "raw").rglob("sha256.txt")),
        {"seed": data_cfg["seed"], "dtype": None, "device": paths.device, "model_sha": models["monitored"]["revision"]},
    )
    publish_manifest(manifest, "01_build_data.json")
    print(verdict)
    if not verdict["pass"]:
        print("stage 01 acceptance failed", file=sys.stderr)
        return 1
    print(
        "CHECKPOINT 1. Review prereg/gate1_memo.md, edit it, commit it, and tag gate1-prereg. "
        "The agent does not tag it.",
        file=sys.stderr,
    )
    return 0


def _spec(models: dict, name: str) -> dict:
    key = {
        "wildjailbreak": "wildjailbreak",
        "wildchat": "wildchat",
        "jailbreak_prompts": "jailbreak_prompts",
        "strongreject": "strongreject",
        "llama3_jailbreaks": "llama3_jailbreaks",
        "orbench": "orbench",
        "xstest": "xstest",
    }[name]
    spec = models["datasets"][key]
    if spec.get("revision") in (None, "", "TBD"):
        raise SystemExit(f"{key} has no pinned revision. Run scripts/00_check_access.py first.")
    return spec


if __name__ == "__main__":
    raise SystemExit(main())
