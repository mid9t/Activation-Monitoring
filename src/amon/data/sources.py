"""Download the listed sources into raw/<source>/<revision>/ and load them untouched."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from amon.data.catalog import PIPELINE_SPLITS, a2_splits


class AccessError(RuntimeError):
    """A gated repo is not readable with the current token."""


def hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    return token or None


def dataset_revision(repo: str) -> str:
    from huggingface_hub import HfApi

    info = HfApi(token=hf_token()).dataset_info(repo)
    return info.sha


def model_revision(repo: str) -> str:
    from huggingface_hub import HfApi

    info = HfApi(token=hf_token()).model_info(repo)
    return info.sha


def check_readable(repo: str, kind: str) -> str:
    """Return the commit SHA, or raise AccessError naming the repo."""
    try:
        if kind == "model":
            return model_revision(repo)
        return dataset_revision(repo)
    except Exception as exc:  # Hugging Face wraps auth failures in several types.
        message = str(exc).lower()
        if any(word in message for word in ("401", "403", "gated", "unauthorized", "forbidden")):
            raise AccessError(f"not authorized to read {repo}. Accept its license and set HF_TOKEN.") from exc
        raise


def load_source(name: str, cfg) -> pd.DataFrame:
    """Raw rows for `name`, with the source's own columns.

    `cfg` is the models.yaml mapping (revisions must already be filled) plus an
    optional `root` pointing at the raw download. This function reads; stage 01
    is what writes raw/<source>/<sha>/.
    """
    token = hf_token()
    if name == "wildjailbreak":
        return _dataset_csv(cfg["datasets"]["wildjailbreak"], token, delimiter="\t")
    if name == "wildchat":
        return _load_hf(cfg["datasets"]["wildchat"]["repo"], cfg["datasets"]["wildchat"]["revision"], token)
    if name == "jailbreak_prompts":
        spec = cfg["datasets"]["jailbreak_prompts"]
        return _load_hf(spec["repo"], spec["revision"], token, config=spec["config"])
    if name == "strongreject":
        spec = cfg["datasets"]["strongreject"]
        return _load_hf(spec["repo"], spec["revision"], token, split=spec.get("split", "validation"))
    if name == "llama3_jailbreaks":
        return _bailey(cfg["datasets"]["llama3_jailbreaks"], cfg, token)
    if name == "orbench":
        return _orbench(cfg["datasets"]["orbench"], token)
    if name == "xstest":
        spec = cfg["datasets"]["xstest"]
        return _load_hf(spec["repo"], spec["revision"], token)
    if name == "harmbench":
        return _harmbench(cfg["datasets"]["harmbench"])
    raise KeyError(f"unknown source {name!r}")


def _load_hf(repo: str, revision: str, token: str | None, config: str | None = None, split: str | None = None) -> pd.DataFrame:
    from datasets import load_dataset

    if revision in (None, "", "TBD"):
        raise AccessError(f"{repo} has no pinned revision. Run stage 00 first.")
    kwargs = {"path": repo, "revision": revision, "token": token}
    if config:
        kwargs["name"] = config
    dataset = load_dataset(**kwargs)
    if split:
        if split not in dataset:
            raise AccessError(
                f"{repo} has no split {split!r}. Splits present: {list(dataset.keys())}. Stopping rather than substituting."
            )
        return dataset[split].to_pandas()
    if hasattr(dataset, "to_pandas"):
        return dataset.to_pandas()
    frames = [dataset[name].to_pandas() for name in dataset.keys()]
    return pd.concat(frames, ignore_index=True)


def _dataset_csv(spec: dict, token: str | None, delimiter: str = ",") -> pd.DataFrame:
    """WildJailbreak is published as TSV. Read the pinned file rather than a guessed schema."""
    from huggingface_hub import list_repo_files, hf_hub_download

    revision = spec["revision"]
    if revision in (None, "", "TBD"):
        raise AccessError(f"{spec['repo']} has no pinned revision. Run stage 00 first.")
    files = list_repo_files(spec["repo"], repo_type="dataset", revision=revision, token=token)
    tables = [name for name in files if name.endswith((".tsv", ".csv", ".parquet", ".json", ".jsonl"))]
    if not tables:
        raise AccessError(f"{spec['repo']} at {revision} has no data file. Files: {files[:20]}")
    preferred = [name for name in tables if name.endswith(".tsv") and "train" in name]
    chosen = preferred[0] if preferred else tables[0]
    path = hf_hub_download(spec["repo"], chosen, repo_type="dataset", revision=revision, token=token)
    if chosen.endswith(".parquet"):
        return pd.read_parquet(path)
    if chosen.endswith(".json") or chosen.endswith(".jsonl"):
        return pd.read_json(path, lines=chosen.endswith(".jsonl"))
    return pd.read_csv(path, sep="\t" if chosen.endswith(".tsv") else delimiter)


def _bailey(spec: dict, cfg: dict, token: str | None) -> pd.DataFrame:
    from datasets import load_dataset

    revision = spec["revision"]
    if revision in (None, "", "TBD"):
        raise AccessError(f"{spec['repo']} has no pinned revision. Run stage 00 first.")
    exclude = list(cfg.get("a2_exclude", ["harmful_prefill"]))
    wanted = set(a2_splits(exclude)) | set(PIPELINE_SPLITS)
    frames = []
    for split in sorted(wanted):
        dataset = load_dataset(spec["repo"], revision=revision, split=split, token=token)
        frame = dataset.to_pandas()
        frame["split"] = split
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _orbench(spec: dict, token: str | None) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    revision = spec["revision"]
    if revision in (None, "", "TBD"):
        raise AccessError(f"{spec['repo']} has no pinned revision. Run stage 00 first.")
    path = hf_hub_download(
        spec["repo"],
        spec["file"],
        repo_type="dataset",
        revision=revision,
        token=token,
    )
    return pd.read_csv(path)


def _harmbench(spec: dict) -> pd.DataFrame:
    """Official HarmBench repository, used only when fewer than 200 StrongREJECT prompts survive."""
    import io
    import urllib.request

    revision = spec.get("revision") or "main"
    if revision == "TBD":
        revision = "main"
    url = f"https://raw.githubusercontent.com/{spec['repo']}/{revision}/{spec['path']}"
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
    except Exception as exc:
        raise AccessError(f"could not read HarmBench at {url}") from exc
    return pd.read_csv(io.BytesIO(payload))


def download_raw(name: str, repo: str, revision: str, dest: Path) -> Path:
    """Save the pinned snapshot under dest and write sha256.txt. Does not log the token."""
    import hashlib

    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo,
        repo_type="dataset",
        revision=revision,
        local_dir=dest,
        token=hf_token(),
    )
    lines = []
    for path in sorted(item for item in dest.rglob("*") if item.is_file() and item.name != "sha256.txt"):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        lines.append(f"{digest.hexdigest()}  {path.relative_to(dest)}")
    (dest / "sha256.txt").write_text("\n".join(lines) + "\n")
    return dest
