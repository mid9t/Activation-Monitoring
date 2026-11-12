"""Stamps on every output directory, and the checks stage 09 uses to refuse a run."""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

TRACKED_LIBRARIES = (
    "torch",
    "transformers",
    "datasets",
    "scikit-learn",
    "numpy",
    "pandas",
    "pyarrow",
    "datasketch",
)


class DirtyTreeError(RuntimeError):
    """Stage 09 will not score inputs produced from a dirty tree."""


class PreregError(RuntimeError):
    """prereg/ does not match the gate1-prereg tag."""


def repo_root() -> Path:
    from amon.guard import repo_root as root

    return root()


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def git_state(cwd: Path | None = None) -> dict:
    """Commit and whether the working tree is dirty. Missing git is recorded, not invented."""
    root = cwd or repo_root()
    head = _git(["rev-parse", "HEAD"], root)
    if head.returncode != 0:
        return {"commit": None, "dirty": None, "error": head.stderr.strip()}
    status = _git(["status", "--porcelain"], root)
    dirty = bool(status.stdout.strip())
    return {"commit": head.stdout.strip(), "dirty": dirty, "error": None}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def canonical_config(cfg: dict) -> str:
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=_json_default)


def config_sha256(cfg: dict) -> str:
    return sha256_text(canonical_config(cfg))


def library_versions() -> dict[str, str | None]:
    import importlib.metadata as metadata

    versions: dict[str, str | None] = {}
    for name in TRACKED_LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def write_manifest(out_dir: Path, inputs: list[Path], cfg: dict) -> Path:
    """Write manifest.json. Records a dirty working tree; it does not clean it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    git = git_state()
    payload = {
        "git": git,
        "config_sha256": config_sha256(cfg),
        "inputs": {str(path): sha256_file(path) for path in inputs if Path(path).is_file()},
        "model_sha": cfg.get("model_sha"),
        "dtype": cfg.get("dtype"),
        "device": cfg.get("device"),
        "library_versions": library_versions(),
        "seed": cfg.get("seed"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")
    return path


def manifest_is_dirty(path: Path) -> bool:
    data = json.loads(Path(path).read_text())
    return bool(data.get("git", {}).get("dirty"))


def refuse_dirty_inputs(manifests: list[Path]) -> None:
    """09_gate1_decide refuses any input whose manifest came from a dirty tree."""
    dirty = [str(path) for path in manifests if manifest_is_dirty(path)]
    if dirty:
        raise DirtyTreeError("refusing dirty manifests: " + ", ".join(dirty))


def prereg_matches_tag(root: Path | None = None, tag: str = "gate1-prereg") -> None:
    """Raise unless prereg/ matches the tree of `tag`."""
    root = root or repo_root()
    tagged = _git(["rev-parse", "--verify", f"{tag}^{{commit}}"], root)
    if tagged.returncode != 0:
        raise PreregError(
            f"tag {tag} is missing. Checkpoint 1 is the owner committing and tagging prereg/."
        )
    diff = _git(["diff", "--exit-code", tag, "--", "prereg"], root)
    if diff.returncode != 0:
        detail = (diff.stdout or diff.stderr).strip()
        raise PreregError(f"prereg/ differs from {tag}. {detail}")


def publish_manifest(manifest: Path, name: str, manifests_dir: Path | None = None) -> Path:
    """Copy a manifest into the repo's manifests/ directory."""
    dest_dir = manifests_dir or (repo_root() / "manifests")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    dest.write_bytes(Path(manifest).read_bytes())
    return dest
