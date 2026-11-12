"""Held-out reads stay inside the baseline column set until the Freeze."""

import pytest

from amon.guard import GuardError, guard_read


def test_guard_read_blocks_non_baseline_heldout_columns(tmp_path, monkeypatch):
    monkeypatch.setattr("amon.guard.repo_root", lambda: tmp_path)
    (tmp_path / "prereg").mkdir()
    guard_read("train", ["density", "activation"])
    guard_read("heldout_A1", ["item_id", "label", "group", "template_id", "request_id", "z", "log_odds", "thresholds"])
    with pytest.raises(GuardError):
        guard_read("heldout_A1", ["z", "density"])
    with pytest.raises(GuardError):
        guard_read("heldout_B", ["activation_norm"])
    (tmp_path / "prereg" / "freeze.json").write_text("{}\n")
    guard_read("heldout_C", ["density"])
