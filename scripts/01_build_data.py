"""Stage 01. Build the item table, then stop for the owner to tag the pre-registration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from amon.config import load_config, require_paths
from amon.data import acceptance, normalize, render_and_tokenize, select_pools
from amon.data.audit import SELECTED_CODE, exclusion_fingerprint
from amon.data.build import length_accounting
from amon.data.embed import embed_texts
from amon.data.resume import ResumeStore, source_hash
from amon.data.sources import download_raw, load_source
from amon.io import write_json
from amon.memo import write_draft
from amon.provenance import publish_manifest, sha256_file, write_manifest

SOURCES = (
    "wildjailbreak",
    "wildchat",
    "jailbreak_prompts",
    "strongreject",
    "llama3_jailbreaks",
    "orbench",
    "xstest",
)


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Build processed items and splits.").parse_args(argv)
    paths = require_paths()
    data_cfg = load_config("data.yaml")
    models = load_config("models.yaml")
    root = paths.data_root
    resume = ResumeStore(root)
    before, notes = _items(root, data_cfg, models, resume)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        models["monitored"]["repo"], revision=models["monitored"]["revision"]
    )
    items, tokens, token_report = render_and_tokenize(before, tok, int(data_cfg["max_tokens"]))
    notes["tokens"] = token_report
    verdict = acceptance(items, tokens, notes, data_cfg)
    verdict["role_cell"] = _with_targets(length_accounting(before, tokens), data_cfg)
    verdict["audit"] = notes.get("ordinary_audit", {})
    processed = root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    public = items.drop(columns=[column for column in items.columns if column.startswith("_")])
    public.to_parquet(processed / "items.parquet", index=False)
    tokens.to_parquet(processed / "tokens.parquet", index=False)
    write_json(processed / "dedup_report.json", notes["dedup"])
    write_json(processed / "acceptance.json", verdict)
    write_json(processed / "ordinary_audit.json", notes.get("ordinary_audit", {}))
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
    write_draft(Path("prereg/gate1_memo.draft.md"), hashes, int(data_cfg["seed"]), rubric_sha)
    outputs = [
        processed / "items.parquet",
        processed / "acceptance.json",
        processed / "ordinary_audit.json",
        *sorted(splits.glob("*.ids")),
    ]
    manifest = write_manifest(
        processed,
        list((root / "raw").rglob("sha256.txt")),
        {"seed": data_cfg["seed"], "dtype": None, "device": paths.device, "model_sha": models["monitored"]["revision"]},
        outputs=outputs,
    )
    publish_manifest(manifest, "01_build_data.json")
    print(verdict)
    if not verdict["pass"]:
        print("stage 01 acceptance failed", file=sys.stderr)
        return 1
    print(
        "CHECKPOINT 1. The draft is prereg/gate1_memo.draft.md. "
        "prereg/gate1_memo.md was not modified. The owner reviews, commits, and tags gate1-prereg. "
        "The agent does not tag it.",
        file=sys.stderr,
    )
    return 0


def _items(root, data_cfg: dict, models: dict, resume: ResumeStore):
    """Normalized sources and the selected item table, reloaded when a checkpoint matches."""
    code_norm = source_hash("src/amon/data/normalize.py", "src/amon/data/sources.py")
    frames = {}
    for name in SOURCES:
        spec = _spec(models, name)
        revision = spec["revision"]
        fingerprint = {"name": name, "revision": revision, "code": code_norm}
        cached = resume.load_frame(f"normalized_{name}", fingerprint)
        if cached is not None:
            frames[name] = cached
            print(f"resume {name}: {len(cached)} normalized rows", flush=True)
            continue
        download_raw(name, spec["repo"], revision, root / "raw" / name / revision)
        frames[name] = normalize(name, load_source(name, models))
        resume.save_frame(f"normalized_{name}", frames[name], fingerprint)
        print(f"{name}: {len(frames[name])} normalized rows", flush=True)

    embed_key = f"{models['dedup']['repo']}@{models['dedup']['revision']}"
    selected_fp = {
        "sources": {name: resume.frame_sha(f"normalized_{name}") for name in SOURCES},
        "data": source_hash("configs/data.yaml"),
        "code": source_hash(*SELECTED_CODE),
        "embed_key": embed_key,
        "exclusions": exclusion_fingerprint(data_cfg["ordinary_audit"]["exclusions"]),
    }
    selected = resume.load_selected(selected_fp)
    if selected is not None:
        items, notes = selected
        print(f"resume selected items: {len(items)}", flush=True)
        return items, notes

    def embed_fn(texts):
        return embed_texts(
            texts,
            models["dedup"]["repo"],
            models["dedup"]["revision"],
            resume=resume,
            resume_name="dedup_embed",
        )

    items, notes = select_pools(
        frames,
        data_cfg,
        embed_fn=embed_fn,
        resume=resume,
        wildchat_sha=resume.frame_sha("normalized_wildchat"),
    )
    resume.save_selected(items, notes, selected_fp)
    print(f"saved selected items: {len(items)}", flush=True)
    return items, notes


def _with_targets(rows: list[dict], cfg: dict) -> list[dict]:
    from amon.data.build import _target_map

    targets = _target_map(cfg)
    for row in rows:
        row["target"] = targets.get((row["role"], row["cell"]))
    return rows


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
