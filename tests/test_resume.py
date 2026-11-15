"""Stage 01 checkpoints: a second call continues from the last saved row."""

import json

import numpy as np
import pandas as pd

from amon.data.dedup import minhash_pairs
from amon.data.resume import ResumeStore


def test_frame_checkpoint_rejects_a_different_fingerprint(tmp_path):
    store = ResumeStore(tmp_path)
    frame = pd.DataFrame({"item_id": ["a"], "messages": ["[]"]})
    store.save_frame("normalized_demo", frame, {"revision": "abc"})
    assert store.load_frame("normalized_demo", {"revision": "abc"})["item_id"].tolist() == ["a"]
    assert store.load_frame("normalized_demo", {"revision": "other"}) is None


def test_selected_bundle_round_trips_a_note_frame(tmp_path):
    store = ResumeStore(tmp_path)
    items = pd.DataFrame({"item_id": ["a"], "role": ["train"], "_rendered": ["kept"]})
    notes = {
        "shortfalls": [],
        "counts": {("train", "ordinary"): 1},
        "a1_insertions": pd.DataFrame({"template_id": ["t"]}),
    }
    fingerprint = {"sources": {"wildchat": "sha"}, "code": "1"}
    store.save_selected(items, notes, fingerprint)
    loaded_items, loaded_notes = store.load_selected(fingerprint)
    assert loaded_items["_rendered"].tolist() == ["kept"]
    assert loaded_notes["a1_insertions"]["template_id"].tolist() == ["t"]
    assert loaded_notes["counts"]["train|ordinary"] == 1
    assert store.load_selected({"sources": {}, "code": "2"}) is None


def test_minhash_resumes_after_a_partial_signature_file(tmp_path):
    texts = [f"alpha beta gamma delta {index}" for index in range(6)]
    texts.append(texts[0])
    store = ResumeStore(tmp_path)
    full = minhash_pairs(texts, 32, 0.8, 2, 1, resume=store, resume_name="case")
    meta_path = store.root / "case.meta.json"
    meta = json.loads(meta_path.read_text())
    meta["done"] = 2
    meta_path.write_text(json.dumps(meta))
    array = np.load(store.root / "case.npy", mmap_mode="r+")
    array[2:] = 0
    array.flush()
    del array
    again = minhash_pairs(texts, 32, 0.8, 2, 1, resume=store, resume_name="case")
    repaired = np.load(store.root / "case.npy")
    assert repaired[5].any()
    assert set(map(tuple, again)) == set(map(tuple, full))
