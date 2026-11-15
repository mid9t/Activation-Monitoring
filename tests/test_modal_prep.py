"""Smoke routing, probe loading, and the Volume sync check."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from amon.provenance import sha256_file
from amon.splits import ROLES, limit_items, parse_roles, resolve_out_root


def _load(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", "_under_test"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_roles_and_limit_routing():
    root = Path("/data")
    assert resolve_out_root(root, root, limit=500) == root / "smoke"
    assert resolve_out_root(root, root / "smoke" / "abc", limit=500) == root / "smoke" / "abc"
    assert parse_roles("tune,heldout_B", allowed=ROLES, default=ROLES) == ["tune", "heldout_B"]
    with pytest.raises(SystemExit, match="unknown role"):
        parse_roles("nope", allowed=ROLES, default=ROLES)
    frame = pd.DataFrame({"item_id": list(range(10)), "label": [0, 1] * 5})
    chosen = limit_items(frame, 4, seed=0)
    assert len(chosen) == 4
    assert set(chosen["label"]) == {0, 1}
    assert list(chosen["item_id"]) == sorted(chosen["item_id"])


def test_unknown_role_is_rejected_before_the_model_loads():
    script = _load("02_cache_activations.py")
    with pytest.raises(SystemExit, match="unknown role"):
        script.main(["--roles", "nope"])


def test_limit_writes_only_the_named_role_under_smoke(tmp_path, monkeypatch):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    monkeypatch.setenv("AMON_DATA", str(tmp_path))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("AMON_DEVICE", "cpu")
    monkeypatch.setenv("AMON_CACHE_TMP", str(tmp_path / "scratch"))
    monkeypatch.delenv("AMON_OUT", raising=False)
    monkeypatch.delenv("AMON_GIT_COMMIT", raising=False)
    monkeypatch.delenv("AMON_MANIFESTS_DIR", raising=False)
    processed = tmp_path / "processed"
    splits = processed / "splits"
    splits.mkdir(parents=True)
    rows = []
    for role, n in (("train", 4), ("tune", 4)):
        ids = [f"{role}-{i}" for i in range(n)]
        (splits / f"{role}.ids").write_text("".join(f"{item}\n" for item in ids))
        for i, item in enumerate(ids):
            rows.append(
                {
                    "item_id": item,
                    "role": role,
                    "label": i % 2,
                    "messages": json.dumps([{"role": "user", "content": f"{role} hello {i}"}]),
                }
            )
    pd.DataFrame(rows).to_parquet(processed / "items.parquet", index=False)
    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        max_position_embeddings=64,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).eval()

    class Tok:
        pad_token_id = 0
        eos_token = "<eos>"

        def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
            return "<bos>" + messages[0]["content"] + "<asst>"

        def encode(self, text, add_special_tokens=False):
            return [1, *(3 + (ord(char) % 20) for char in text)]

    script = _load("02_cache_activations.py")
    monkeypatch.setattr(
        script,
        "load_config",
        lambda name: {
            "models.yaml": {"monitored": {"repo": "meta", "revision": "abc123"}},
            "cache.yaml": {
                "layers": [1, 2],
                "dtype_store": "float32",
                "token_budget_per_batch": 8,
                "verify_n": 2,
                "verify_min_cosine": 0.999,
            },
        }[name],
    )
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: Tok())
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", lambda *args, **kwargs: model)
    assert script.main(["--roles", "tune", "--limit", "2"]) == 0
    tune = tmp_path / "smoke" / "cache" / "llama3-8b-instruct@abc123" / "tune"
    assert (tune / "verify.json").is_file()
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "smoke" / "cache" / "llama3-8b-instruct@abc123" / "train").exists()
    manifest = json.loads((tmp_path / "smoke" / "manifests" / "02_cache_tune.json").read_text())
    assert manifest["smoke"] is True
    assert list(manifest["outputs"]) == sorted(manifest["outputs"])
    assert all(key.startswith("smoke/") for key in manifest["outputs"])
    assert len((tune / "split.ids").read_text().split()) == 2


def test_float32_stack_matches_float64_auroc(tmp_path):
    from amon.probe.train import _cv_auroc

    script = _load("04_train_probe.py")
    rng = np.random.default_rng(0)
    n, width = 40, 6
    groups = np.repeat(np.arange(8), 5)
    labels = np.tile([0, 1, 0, 1, 0], 8)
    matrix = rng.normal(scale=0.01, size=(n, width)).astype(np.float32)
    matrix[:, 0] = labels * 5.0
    split = tmp_path / "train"
    split.mkdir()
    ids = [f"i{i:02d}" for i in range(n)]
    (split / "split.ids").write_text("".join(f"{item}\n" for item in ids))
    for layer in (8, 12):
        np.save(split / f"last_L{layer:02d}.npy", matrix)
    items = pd.DataFrame({"item_id": ids, "label": labels, "group_id": groups.astype(str)})
    stacked, y, group_ids, _ids = script._stack(split, items, [8, 12])
    assert stacked.dtype == np.float32
    fresh = _cv_auroc(stacked[0], y, group_ids, 1e-3, 2, "balanced")["mean"]
    reference = _cv_auroc(np.asarray(stacked[0], dtype=np.float64), y, group_ids, 1e-3, 2, "balanced")["mean"]
    assert f"{fresh:.4f}" == f"{reference:.4f}"


def test_preflight_rejects_a_cache_that_does_not_match(tmp_path):
    script = _load("04_train_probe.py")
    items = pd.DataFrame(
        {"item_id": ["a", "b"], "role": ["train", "train"], "label": [0, 1], "group_id": ["g", "h"]}
    )
    items_path = tmp_path / "processed" / "items.parquet"
    items_path.parent.mkdir(parents=True)
    items.to_parquet(items_path, index=False)
    with pytest.raises(SystemExit, match="cache missing"):
        script._preflight(tmp_path, "abc", items)
    directory = tmp_path / "cache" / "llama3-8b-instruct@abc" / "train"
    directory.mkdir(parents=True)
    (directory / "split.ids").write_text("a\nb\n")
    (directory / "verify.json").write_text(json.dumps({"pass": True, "min_cosine": 1.0}))
    (directory / "manifest.json").write_text(
        json.dumps({"inputs": {"processed/items.parquet": sha256_file(items_path)}})
    )
    script._preflight(tmp_path, "abc", items)
    (directory / "verify.json").write_text(json.dumps({"pass": False, "min_cosine": 1.0}))
    with pytest.raises(SystemExit, match="cache missing"):
        script._preflight(tmp_path, "abc", items)


def test_verify_sync_fails_when_one_byte_changes(tmp_path):
    script = _load("verify_sync.py")
    root = tmp_path / "data"
    relative = "cache/role/last_L08.npy"
    path = root / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b"abc")
    manifest_dir = root / "manifests"
    manifest_dir.mkdir()
    payload = {
        "smoke": False,
        "outputs": {relative: {"sha256": sha256_file(path), "bytes": 3}},
    }
    (manifest_dir / "02_cache_train.json").write_text(json.dumps(payload))
    assert script.verify_outputs(manifest_dir, root) == []
    path.write_bytes(b"abd")
    problems = script.verify_outputs(manifest_dir, root)
    assert len(problems) == 1
    assert "hash mismatch" in problems[0]
    payload["smoke"] = True
    path.write_bytes(b"abc")
    payload["outputs"][relative]["sha256"] = sha256_file(path)
    (manifest_dir / "02_cache_train.json").write_text(json.dumps(payload))
    assert any("smoke" in problem for problem in script.verify_outputs(manifest_dir, root))


def test_pass2_requires_the_committed_policy(tmp_path, monkeypatch):
    from amon.llm import Policy, prompt_hash

    script = _load("03_score_llamaguard.py")
    monkeypatch.setenv("AMON_MANIFESTS_DIR", str(tmp_path))
    paths = type("Paths", (), {"data_root": tmp_path, "out_root": tmp_path})()
    with pytest.raises(SystemExit, match="policy_decided"):
        script._require_policy_decision(paths, "sha", {"policy_decided": False}, Policy())
    (tmp_path / "03_llm_trainval.json").write_text(json.dumps({"policy_sha256": "different"}))
    with pytest.raises(SystemExit, match="policy_sha256"):
        script._require_policy_decision(paths, "sha", {"policy_decided": True}, Policy())
    (tmp_path / "03_llm_trainval.json").write_text(
        json.dumps({"policy_sha256": prompt_hash(Policy().identity())})
    )
    script._require_policy_decision(paths, "sha", {"policy_decided": True}, Policy())
