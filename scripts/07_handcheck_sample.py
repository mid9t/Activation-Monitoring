"""Stage 07. Write the blinded hand-check sheet and stop."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from amon.cascade.thresholds import cascade_predict, llm_predict
from amon.config import load_config, require_paths
from amon.data.textutil import user_text
from amon.handcheck import key_sha256, sample_handcheck
from amon.io import read_json


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Sample the hand check.").parse_args(argv)
    paths = require_paths()
    data_cfg = load_config("data.yaml")
    gate = load_config("gate1.yaml")
    thresholds = read_json(paths.data_root / "gate1" / "thresholds.json")
    items = pd.read_parquet(paths.data_root / "processed" / "items.parquet")
    rows = []
    for shift, roles in (("A", ["heldout_A1", "heldout_A2"]), ("B", ["heldout_B"]), ("C", ["heldout_C"])):
        rows.append(_mark(shift, roles, items, paths.data_root, thresholds))
    marked = pd.concat(rows, ignore_index=True)
    sample, key = sample_handcheck(marked, gate, int(data_cfg["seed"]))
    dest = paths.data_root / "gate1" / "handcheck"
    dest.mkdir(parents=True, exist_ok=True)
    sample.to_csv(dest / "sample.csv", index=False)
    key.to_csv(dest / "key.csv", index=False)
    digest = key_sha256(dest / "key.csv")
    print(f"hand-check key SHA-256: {digest}")
    print(
        "CHECKPOINT 2. Label sample.csv blind against prereg/rubric.md. "
        "Write labels.csv with columns hc_id, label, note. Do not open key.csv.",
    )
    return 0


def _mark(shift, roles, items, root, thresholds) -> pd.DataFrame:
    frame = items.loc[items["role"].isin(roles)].copy()
    pieces = []
    for role, part in frame.groupby("role"):
        ids = [line.strip() for line in (root / "processed" / "splits" / f"{role}.ids").read_text().splitlines() if line.strip()]
        scores = np.load(root / "scores" / "probe" / f"{role}.npy")
        position = {item: index for index, item in enumerate(ids)}
        part = part.copy()
        part["z"] = [scores[position[item]] for item in part["item_id"].astype(str)]
        pieces.append(part)
    frame = pd.concat(pieces, ignore_index=True)
    models_sha = load_config_sha(root)
    frame["log_odds"] = np.nan
    for role, part in frame.groupby("role"):
        table = pd.read_parquet(root / "llm" / models_sha / f"{role}.parquet").set_index("item_id")
        frame.loc[part.index, "log_odds"] = table.loc[part["item_id"].astype(str), "log_odds"].to_numpy()
    llm = llm_predict(frame["log_odds"].to_numpy(), thresholds["tau_llm"])
    z = frame["z"].to_numpy()
    y = frame["label"].to_numpy()
    cascade = cascade_predict(z, llm, thresholds["t0"], thresholds["t1"])
    outside = (z <= thresholds["t0"]) | (z >= thresholds["t1"])
    probe_wrong = cascade != y
    stratum = np.full(len(frame), "", dtype=object)
    stratum[outside & probe_wrong & (llm == y)] = "F"
    stratum[outside & probe_wrong & (llm != y)] = "U"
    stratum[outside & ~probe_wrong] = "decoy"
    frame["stratum"] = stratum
    frame["shift"] = shift
    frame["text"] = frame["messages"].map(user_text)
    frame["probe_call"] = np.where(cascade == 1, "harmful", "benign")
    frame["llm_call"] = np.where(llm == 1, "harmful", "benign")
    frame["label"] = np.where(frame["label"].to_numpy() == 1, "harmful", "benign")
    return frame.loc[frame["stratum"] != "", ["item_id", "text", "shift", "stratum", "label", "probe_call", "llm_call"]]


def load_config_sha(root):
    from amon.config import load_config

    sha = load_config("models.yaml")["guard"]["revision"]
    return f"llama-guard-3-8b@{sha}"


if __name__ == "__main__":
    raise SystemExit(main())
