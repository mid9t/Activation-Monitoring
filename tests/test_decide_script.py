"""Stage 09's script refuses a dirty manifest and a prereg tree that is not the tag."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "09_gate1_decide.py"
_SPEC = importlib.util.spec_from_file_location("gate1_decide", _PATH)
gate1_decide = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gate1_decide)


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def test_decide_script_refuses_dirty_manifest(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"git": {"dirty": True, "commit": "abc"}}) + "\n")
    with pytest.raises(SystemExit):
        gate1_decide.main(["--manifest", str(manifest), "--check-only"])


def test_decide_script_refuses_prereg_that_differs_from_the_tag(tmp_path, monkeypatch):
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    prereg = tmp_path / "prereg"
    prereg.mkdir()
    (prereg / "gate1_memo.md").write_text("frozen\n")
    _git(tmp_path, "add", "prereg")
    _git(tmp_path, "commit", "-m", "prereg")
    _git(tmp_path, "tag", "gate1-prereg")
    (prereg / "gate1_memo.md").write_text("edited after the tag\n")
    monkeypatch.setattr(gate1_decide, "repo_root", lambda: tmp_path)
    with pytest.raises(SystemExit):
        gate1_decide.main(["--repo", str(tmp_path), "--check-prereg"])
