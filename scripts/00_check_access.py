"""Stage 00. Confirm every gated repo is readable and pin its commit SHA."""

from __future__ import annotations

import argparse
import sys

from amon.config import configs_dir, fill_tbd, load_config
from amon.data.sources import AccessError, check_readable


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Pin model and dataset revisions.").parse_args(argv)
    models = load_config("models.yaml")
    pinned = {}
    failures = []
    for key in ("monitored", "guard", "dedup"):
        repo = models[key]["repo"]
        try:
            pinned[f"{key}.revision"] = check_readable(repo, "model")
            print(f"readable model {repo} @ {pinned[f'{key}.revision']}")
        except AccessError as exc:
            failures.append(str(exc))
    for key, spec in models["datasets"].items():
        if key == "harmbench":
            print("harmbench is the official git repository and is fetched only as a fallback")
            continue
        try:
            pinned[key] = check_readable(spec["repo"], "dataset")
            print(f"readable dataset {spec['repo']} @ {pinned[key]}")
        except AccessError as exc:
            failures.append(str(exc))
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 2
    # fill_tbd matches the key text. Model revisions are nested; write them by unique key.
    path = configs_dir() / "models.yaml"
    text = path.read_text()
    for key in ("monitored", "guard", "dedup"):
        block_key = key
        sha = pinned[f"{key}.revision"]
        text = _fill_block(text, block_key, sha)
    for key, spec in models["datasets"].items():
        if key == "harmbench" or key not in pinned:
            continue
        text = _fill_block(text, key, pinned[key])
    path.write_text(text)
    print("pinned revisions written to configs/models.yaml")
    return 0


def _fill_block(text: str, key: str, sha: str) -> str:
    """Replace the first `revision: TBD` that follows `key:`."""
    import re

    pattern = rf"({re.escape(key)}:.*?revision:\s*)TBD"
    updated, count = re.subn(pattern, rf"\g<1>{sha}", text, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(f"could not pin {key}; revision is no longer TBD or the block moved")
    return updated


if __name__ == "__main__":
    raise SystemExit(main())
