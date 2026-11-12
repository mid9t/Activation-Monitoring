"""Manifests record a dirty tree, and stage 09 refuses those inputs."""

import json
import subprocess
from pathlib import Path

import pytest

from amon.provenance import DirtyTreeError, refuse_dirty_inputs, write_manifest


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
    clean = write_manifest(tmp_path / "out-clean", [tracked], {"seed": 1})
    assert json.loads(clean.read_text())["git"]["dirty"] is False
    tracked.write_text("dirty\n")
    dirty = write_manifest(tmp_path / "out-dirty", [tracked], {"seed": 1})
    assert json.loads(dirty.read_text())["git"]["dirty"] is True
    with pytest.raises(DirtyTreeError):
        refuse_dirty_inputs([dirty])
    refuse_dirty_inputs([clean])
