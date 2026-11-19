"""Harm-category column and the descriptive stratum tables."""

import numpy as np
import pandas as pd
import pytest

from amon.cascade.headroom import headroom_counts
from amon.cascade.thresholds import llm_predict
from amon.categories import errors_by, recall_by, write_category_bundle
from amon.data.normalize import SchemaError, normalize
from amon.data.templates import build_templated_jailbreaks
from amon.guard import GuardError, guard_read
from amon.io import read_json


def test_a1_rows_inherit_request_category():
    templates = pd.DataFrame({"template_id": ["t1"], "text": ["Please do the following.\n\n"]})
    requests = pd.DataFrame(
        {"request_id": ["r1"], "text": ["the request"], "harm_category": ["Disinformation"]}
    )
    built = build_templated_jailbreaks(templates, requests, 1, 1, seed=1)
    assert built.iloc[0]["harm_category"] == "Disinformation"


def test_missing_category_column_stops():
    raw = pd.DataFrame({"content": [np.array(["write a harmful request"], dtype=object)]})
    with pytest.raises(SchemaError, match="category"):
        normalize("strongreject", raw)


def test_recall_by_pools_small_strata():
    strata = np.array(["a"] * 10 + ["b"] * 12)
    log_odds = np.ones(22)
    y = np.ones(22)
    table = recall_by(strata, log_odds, y, tau_llm=0.0, clusters=None, B=20, seed=1, min_n=30)
    assert table["stratum"].tolist() == ["other"]
    assert int(table.iloc[0]["n"]) == 22


def test_errors_by_sums_to_headroom(tmp_path):
    rng = np.random.default_rng(1)
    n = 80
    strata = np.array(["kept"] * 50 + ["tiny"] * 10 + ["small"] * 20)
    z = rng.normal(size=n)
    y = rng.integers(0, 2, size=n)
    llm = rng.integers(0, 2, size=n)
    table = errors_by(strata, z, llm, y, t0=-0.2, t1=0.2, min_n=30)
    assert int(table["F"].sum()) == headroom_counts(z, llm, y, -0.2, 0.2, np.ones(n))["F"]
    assert int(table["U"].sum()) == headroom_counts(z, llm, y, -0.2, 0.2, np.ones(n))["U"]
    assert int(table["I"].sum()) == headroom_counts(z, llm, y, -0.2, 0.2, np.ones(n))["I"]
    recall = recall_by(strata, z, y, 0.0, None, 20, 1, min_n=30)
    write_category_bundle(tmp_path, {"recall_A1": recall, "errors_A1": table}, 0.25)
    payload = read_json(tmp_path / "categories.json")
    assert payload["llm_miss_share_low_recall_strata"] == 0.25
    assert {"stratum", "n", "recall", "lo", "hi", "method"} <= set(payload["tables"]["recall_A1"][0])
    assert {"stratum", "n", "F", "U", "I", "U_share"} <= set(payload["tables"]["errors_A1"][0])
    assert (tmp_path / "categories" / "errors_A1.csv").is_file()


def test_guard_allows_metadata_only(tmp_path, monkeypatch):
    monkeypatch.setattr("amon.guard.repo_root", lambda: tmp_path)
    (tmp_path / "prereg").mkdir()
    guard_read("heldout_A1", ["cell", "harm_category"])
    with pytest.raises(GuardError):
        guard_read("heldout_A1", ["density"])
    assert llm_predict(np.array([0.2, -0.2]), 0.0).tolist() == [1, 0]
