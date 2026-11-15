"""Shard resume and float16 overflow for the activation cache."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from amon.cache.extract import _to_storage, cache_split


def _model():
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
    model = LlamaForCausalLM(config)
    model.eval()
    return model


class Tok:
    pad_token_id = 0
    eos_token = "<eos>"

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        return "<bos>" + messages[0]["content"] + "<asst>"

    def encode(self, text, add_special_tokens=False):
        return [1, *(3 + (ord(char) % 20) for char in text)]


def _items(n=6):
    return pd.DataFrame(
        {
            "item_id": [f"i{i}" for i in range(n)],
            "role": ["train"] * n,
            "label": [i % 2 for i in range(n)],
            "messages": [json.dumps([{"role": "user", "content": f"hello {i}"}]) for i in range(n)],
        }
    )


def _cfg(tmp, items, **extra):
    cfg = SimpleNamespace(
        layers=[1, 2],
        dtype_store="float16",
        token_budget_per_batch=1,
        data_root=tmp,
        items=items,
        seed=0,
        model_sha="abc",
        inputs=[],
        scratch_dir=None,
        shard_rows=2000,
        stop_after_batches=None,
    )
    for key, value in extra.items():
        setattr(cfg, key, value)
    return cfg


def test_resume_after_a_zeroed_row_matches_an_uninterrupted_run(tmp_path):
    model = _model()
    tok = Tok()
    items = _items()
    reference = tmp_path / "reference"
    cache_split("train", model, tok, _cfg(tmp_path, items, scratch_dir=tmp_path / "scratch-ref"), reference)
    partial_dir = tmp_path / "partial"
    partial = cache_split(
        "train",
        model,
        tok,
        _cfg(tmp_path, items, scratch_dir=tmp_path / "scratch-part", stop_after_batches=2),
        partial_dir,
    )
    assert partial["complete"] is False
    array = np.load(partial["scratch"] / "last_L01.npy", mmap_mode="r+")
    array[0] = 0
    array.flush()
    assert np.all(np.asarray(array[0]) == 0)
    del array
    (partial_dir / "shards" / "part-000099.npz.tmp").write_bytes(b"not a shard")
    finished = cache_split(
        "train",
        model,
        tok,
        _cfg(tmp_path, items, scratch_dir=tmp_path / "scratch-resume"),
        partial_dir,
    )
    assert finished["complete"] is True
    for name in ("last_L01.npy", "mean_L01.npy", "last_L02.npy", "mean_L02.npy"):
        assert (partial_dir / name).read_bytes() == (reference / name).read_bytes()
    resumed = np.load(partial_dir / "last_L01.npy")
    assert not np.any(np.all(resumed == 0, axis=1))


def test_float16_overflow_raises_and_leaves_written_rows(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="non-finite"):
        _to_storage(torch.tensor([float("nan")]), "float32")
    with pytest.raises(RuntimeError, match="float16"):
        _to_storage(torch.tensor([[60000.0]]), "float16")
    stored, peak = _to_storage(torch.tensor([[1.5, -2.0]]), "float16")
    assert stored.dtype == np.float16
    assert peak == pytest.approx(2.0)

    model = _model()
    tok = Tok()
    items = _items()
    out = tmp_path / "cache"
    cache_split(
        "train",
        model,
        tok,
        _cfg(tmp_path, items, scratch_dir=tmp_path / "scratch-a", stop_after_batches=1),
        out,
    )
    shard = next(path for path in (out / "shards").glob("part-*.npz") if ".tmp" not in path.name)
    before = shard.read_bytes()

    def huge(model, input_ids, attention_mask, layers):
        del attention_mask
        big = torch.full((input_ids.shape[0], model.config.hidden_size), 70000.0)
        return {layer: (big, big.clone()) for layer in layers}

    monkeypatch.setattr("amon.cache.extract.forward_hidden", huge)
    with pytest.raises(RuntimeError, match="float16"):
        cache_split(
            "train",
            model,
            tok,
            _cfg(tmp_path, items, scratch_dir=tmp_path / "scratch-b"),
            out,
        )
    assert shard.read_bytes() == before
