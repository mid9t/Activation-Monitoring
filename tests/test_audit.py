"""Ordinary-cell template filter and the blinded audit sheet."""

import pandas as pd

from amon.data.audit import (
    apply_exclusions,
    audit_sample,
    exclusion_fingerprint,
    flag_template_rows,
    normalize_text,
    shingle_hashes,
)
from amon.data.build import select_pools
from amon.data.dedup import _shingles
from amon.data.resume import ResumeStore


def _words(prefix: str, count: int) -> str:
    return " ".join(f"{prefix}{index}" for index in range(count))


def _template() -> str:
    return _words("slot", 40)


def test_containment_finds_template_with_long_request():
    template = _template()
    request = _words("req", 300)
    item = f"{template} {request}"
    pool = pd.DataFrame({"item_id": ["x"], "_source_request": [item], "lang": ["en"], "is_code": [False]})
    templates = pd.DataFrame({"item_id": ["t"], "_source_request": [template]})
    scored, _report = flag_template_rows(
        pool, templates, {"template_containment": 0.5, "min_template_shingles": 20, "shingle_words": 5}
    )
    assert scored.iloc[0]["_template_containment"] >= 0.9
    left = set(_shingles(normalize_text(template), 5))
    right = set(_shingles(normalize_text(item), 5))
    jaccard = len(left & right) / len(left | right)
    assert jaccard < 0.5


def test_placeholder_is_stripped():
    cleaned = normalize_text("please [INSERT PROMPT HERE] now {prompt} answer")
    assert "insert" not in cleaned.split()
    assert "prompt" not in cleaned.split()
    bare = shingle_hashes("alpha beta gamma delta epsilon zeta", 5)
    wrapped = shingle_hashes("alpha beta [INSERT PROMPT HERE] gamma delta epsilon zeta", 5)
    assert bare == wrapped
    assert shingle_hashes("[INSERT PROMPT HERE] {prompt}", 5) == set()


def test_short_templates_are_skipped():
    short = _words("tiny", 14)  # 10 shingles of width 5
    pool = pd.DataFrame(
        {"item_id": ["x"], "_source_request": [short], "lang": ["en"], "is_code": [False]}
    )
    templates = pd.DataFrame({"item_id": ["t"], "_source_request": [short]})
    scored, report = flag_template_rows(
        pool, templates, {"template_containment": 0.5, "min_template_shingles": 20, "shingle_words": 5}
    )
    assert report["flagged"] == 0
    assert scored.iloc[0]["_template_containment"] == 0.0


def test_filter_runs_before_sampling():
    template = _template()
    clean = [
        _row(f"c{i}", _words(f"chat{i}", 12), "en", False)
        for i in range(36)
    ]
    flagged = [
        _row(f"f{i}", f"{template} {_words(f'req{i}', 80)}", "en", False)
        for i in range(4)
    ]
    other = [
        _row("fr1", _words("bonjour", 12), "fr", False),
        _row("fr2", _words("hola", 12), "es", False),
        _row("code1", "function f() {\n  return 1;\n}\nconst x = 1;\n", "en", True),
        _row("code2", "function g() {\n  return 2;\n}\nconst y = 2;\n", "en", True),
    ]
    wildchat = pd.DataFrame(clean + flagged + other)
    cfg = _cfg()
    items, notes = select_pools(_frames(wildchat, template), cfg, embed_fn=None)
    assert notes["ordinary_audit"]["flagged"] == 4
    assert not set(items["item_id"]).intersection({f"f{i}" for i in range(4)})
    train = items.loc[(items["role"].isin(["train", "trainval"])) & (items["cell"] == "ordinary")]
    tune = items.loc[(items["role"] == "tune") & (items["cell"] == "ordinary")]
    assert len(train) == 8
    assert len(tune) == 4
    assert len(items.loc[(items["role"] == "heldout_C") & (items["cell"] == "non_english")]) == 2
    assert len(items.loc[(items["role"] == "heldout_C") & (items["cell"] == "code")]) == 2


def test_exclusions_drop_ids_and_missing_file_is_noop(tmp_path):
    pool = pd.DataFrame({"item_id": ["a", "b"], "_source_request": ["one", "two"]})
    same, dropped = apply_exclusions(pool, tmp_path / "missing.csv")
    assert dropped == 0
    assert len(same) == 2
    path = tmp_path / "ordinary_exclusions.csv"
    path.write_text("item_id,reason\na,harmful\n")
    kept, dropped = apply_exclusions(pool, path)
    assert dropped == 1
    assert kept["item_id"].tolist() == ["b"]


def test_audit_sheet_is_blind():
    pool = pd.DataFrame(
        {
            "item_id": [f"r{i}" for i in range(8)],
            "_source_request": [f"random text {i} about weather" for i in range(8)],
            "_template_containment": [0.0] * 8,
        }
    )
    flagged = pd.DataFrame(
        {
            "item_id": [f"f{i}" for i in range(4)],
            "_source_request": [f"flagged text {i}" for i in range(4)],
            "_template_containment": [0.8] * 4,
        }
    )
    sheet, key = audit_sample(pool, flagged, 5, 3, seed=20261026)
    again, _key2 = audit_sample(pool, flagged, 5, 3, seed=20261026)
    assert list(sheet.columns) == ["audit_id", "text", "label"]
    assert sheet["label"].eq("").all()
    assert sheet["audit_id"].tolist() != key["audit_id"].tolist()
    assert sheet["audit_id"].tolist() == again["audit_id"].tolist()
    assert set(sheet["text"]) == set(again["text"])


def test_fingerprint_changes_with_exclusions(tmp_path):
    path = tmp_path / "ordinary_exclusions.csv"
    path.write_text("item_id,reason\na,harmful\n")
    store = ResumeStore(tmp_path)
    first = {"exclusions": exclusion_fingerprint(path), "code": "same"}
    store.save_selected(pd.DataFrame({"item_id": ["a"]}), {}, first)
    assert store.load_selected(first) is not None
    path.write_text("item_id,reason\na,harmful\nb,unclear\n")
    second = {"exclusions": exclusion_fingerprint(path), "code": "same"}
    assert first["exclusions"] != second["exclusions"]
    assert store.load_selected(second) is None


def _row(item_id, text, lang, code):
    return {
        "item_id": item_id,
        "source": "wildchat",
        "cell": "ordinary",
        "role": "",
        "label": 0,
        "group": "ordinary",
        "messages": "[]",
        "group_id": item_id,
        "template_id": "",
        "request_id": "",
        "lang": lang,
        "is_code": code,
        "harm_category": "",
        "_source_request": text,
        "_rendered": "",
        "_moderation_flagged": False,
    }


def _side(item_id, cell, role, group, label):
    return {
        "item_id": item_id,
        "source": "wildjailbreak",
        "cell": cell,
        "role": role,
        "label": label,
        "group": group,
        "messages": "[]",
        "group_id": item_id,
        "template_id": "",
        "request_id": "",
        "lang": "en",
        "is_code": False,
        "harm_category": "",
        "_source_request": f"unique {item_id} {_words(item_id, 8)}",
        "_rendered": "",
    }


def _frames(wildchat, template):
    wild = []
    for cell, role, group, label in (
        ("vanilla_harmful", "train", "misses", 1),
        ("vanilla_benign", "train", "hard_negatives", 0),
        ("adversarial_harmful", "tune", "misses", 1),
        ("adversarial_benign", "tune", "hard_negatives", 0),
    ):
        for index in range(4):
            wild.append(_side(f"{cell}-{index}", cell, role, group, label))
    return {
        "wildjailbreak": pd.DataFrame(wild),
        "wildchat": wildchat,
        "jailbreak_prompts": pd.DataFrame(
            {
                "item_id": ["template-1"],
                "source": "jailbreak_prompts",
                "_source_request": [template],
                "harm_category": [""],
            }
        ),
        "strongreject": pd.DataFrame(
            {
                "item_id": ["request-1"],
                "source": "strongreject",
                "_source_request": ["do the forbidden thing"],
                "harm_category": ["Fraud"],
            }
        ),
        "orbench": pd.DataFrame(
            [
                {
                    **_side("or-1", "orbench", "", "hard_negatives", 0),
                    "source": "orbench",
                }
            ]
        ),
        "xstest": pd.DataFrame(
            [
                {
                    **_side("xs-1", "xstest_safe", "heldout_B", "hard_negatives", 0),
                    "source": "xstest",
                }
            ]
        ),
    }


def _cfg():
    return {
        "seed": 1,
        "train": {"vanilla_harmful": 2, "vanilla_benign": 2, "ordinary": 8, "trainval_frac": 0.0},
        "tune": {"adversarial_harmful": 2, "adversarial_benign": 2, "ordinary": 4, "domain_slice": 0},
        "heldout": {
            "A1": {"n_templates": 1, "n_requests": 1},
            "A2": {"exclude_splits": []},
            "B": {"orbench_n": 1},
            "C": {"non_english": 2, "code": 2, "exclude_languages_in_tune_slice": True},
        },
        "wildchat": {"drop_moderation_flagged": True, "max_per_neardup_cluster": 2},
        "dedup": {"minhash_shingle_words": 5, "minhash_perm": 128, "jaccard": 0.8, "embed_cosine": 0.9},
        "ordinary_audit": {
            "template_containment": 0.5,
            "min_template_shingles": 20,
            "exclusions": "prereg/ordinary_exclusions.csv",
            "audit_n_random": 200,
            "audit_n_flagged": 30,
        },
    }

