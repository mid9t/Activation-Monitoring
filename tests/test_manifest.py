"""Manifests record a dirty tree, and stage 09 refuses those inputs."""

import json
import subprocess
from pathlib import Path

import pytest

from amon.provenance import DirtyTreeError, git_state, publish_manifest, refuse_dirty_inputs, write_manifest


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def test_write_manifest_records_a_dirty_tree(tmp_path, monkeypatch):
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-m", "init")
    monkeypatch.setattr("amon.provenance.repo_root", lambda: tmp_path)
    monkeypatch.delenv("AMON_GIT_COMMIT", raising=False)
    monkeypatch.delenv("AMON_GIT_DIRTY", raising=False)
    monkeypatch.setenv("AMON_DATA", str(tmp_path))
    clean = write_manifest(tmp_path / "out-clean", [tracked], {"seed": 1})
    assert json.loads(clean.read_text())["git"]["dirty"] is False
    tracked.write_text("dirty\n")
    dirty = write_manifest(tmp_path / "out-dirty", [tracked], {"seed": 1})
    assert json.loads(dirty.read_text())["git"]["dirty"] is True
    with pytest.raises(DirtyTreeError):
        refuse_dirty_inputs([dirty])
    refuse_dirty_inputs([clean])
    recorded = json.loads(clean.read_text())
    assert recorded["inputs"] == {"tracked.txt": recorded["inputs"]["tracked.txt"]}
    assert recorded["outputs"] == {}
    assert recorded["smoke"] is False


def test_git_state_reads_the_env_when_git_is_missing(monkeypatch):
    monkeypatch.setenv("AMON_GIT_COMMIT", "abc123")
    monkeypatch.setenv("AMON_GIT_DIRTY", "0")

    def missing(*_args, **_kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr("amon.provenance._git", missing)
    state = git_state()
    assert state["commit"] == "abc123"
    assert state["dirty"] is False
    monkeypatch.delenv("AMON_GIT_COMMIT")
    missing_state = git_state()
    assert missing_state["commit"] is None
    assert missing_state["dirty"] is None
    assert missing_state["error"] == "git not installed"


def test_refuse_dirty_inputs_rejects_an_unknown_tree(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"git": {"commit": "abc", "dirty": None}}) + "\n")
    with pytest.raises(DirtyTreeError):
        refuse_dirty_inputs([path])


def test_publish_manifest_writes_to_amon_manifests_dir(tmp_path, monkeypatch):
    source = tmp_path / "manifest.json"
    source.write_text('{"smoke": false}\n')
    dest = tmp_path / "published"
    monkeypatch.setenv("AMON_MANIFESTS_DIR", str(dest))
    published = publish_manifest(source, "02_cache_train.json")
    assert published == dest / "02_cache_train.json"
    assert published.read_text() == source.read_text()
