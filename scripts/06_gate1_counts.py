"""Stage 06. Condition (a) and condition (b). Held-out columns go through guard_read."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from amon.cascade.headroom import headroom_counts
from amon.config import load_config, require_paths
from amon.gate import compare_probe_and_llm, condition_b, variance_deff
from amon.guard import guard_read
from amon.io import read_json, write_json
from amon.provenance import publish_manifest, write_manifest
from amon.stats.bootstrap import paired_bootstrap, percentile_interval


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Gate 1 counts.").parse_args(argv)
    paths = require_paths()
    gate = load_config("gate1.yaml")
    models = load_config("models.yaml")
    thresholds = read_json(paths.data_root / "gate1" / "thresholds.json")
    items = pd.read_parquet(paths.data_root / "processed" / "items.parquet")
    sha = models["guard"]["revision"]
    sets = {
        "tune_adversarial": items.loc[items["role"].eq("tune") & items["cell"].isin(["adversarial_harmful", "adversarial_benign"])],
        "A": items.loc[items["role"].isin(["heldout_A1", "heldout_A2"])],
        "B": items.loc[items["role"] == "heldout_B"],
        "C": items.loc[items["role"] == "heldout_C"],
    }
    condition = {"sets": {}}
    B = int(gate["bootstrap"]["B"])
    seed = int(gate["bootstrap"]["seed"])
    for name, frame in sets.items():
        z, log_odds, y, groups, clusters = _columns(frame, paths.data_root, sha, name)
        result = compare_probe_and_llm(
            z, log_odds, y, groups, gate["weights"], thresholds["tau_llm"], thresholds["t_star"],
            clusters=clusters, B=B, seed=seed,
        )
        result["interval"] = list(result["interval"])
        condition["sets"][name] = result
    wins = sum(1 for row in condition["sets"].values() if row["llm_lower"])
    condition["pass"] = wins >= 2
    write_json(paths.data_root / "gate1" / "condition_a.json", condition)
    for shift, frame in (("A", sets["A"]), ("B", sets["B"]), ("C", sets["C"])):
        _write_shift(shift, frame, paths.data_root, sha, thresholds, gate, B, seed)
    manifest = write_manifest(
        paths.data_root / "gate1",
        [paths.data_root / "gate1" / "thresholds.json"],
        {"seed": seed, "device": paths.device},
    )
    publish_manifest(manifest, "06_gate1_counts.json")
    return 0


def _columns(frame, root, sha, name):
    split = "heldout_A" if name == "A" else name if name in {"B", "C"} else "tune"
    guard_read(split if split != "tune" else "tune", ["z", "log_odds", "label", "group", "template_id", "thresholds"])
    frame = frame.sort_values("item_id")
    z_parts = []
    odds = []
    for role, part in frame.groupby("role", sort=False):
        ids = part["item_id"].astype(str).tolist()
        role_ids = [line.strip() for line in (root / "processed" / "splits" / f"{role}.ids").read_text().splitlines() if line.strip()]
        scores = np.load(root / "scores" / "probe" / f"{role}.npy")
        position = {item: index for index, item in enumerate(role_ids)}
        z_parts.append(scores[[position[item] for item in ids]])
        table = pd.read_parquet(root / "llm" / f"llama-guard-3-8b@{sha}" / f"{role}.parquet").set_index("item_id")
        odds.append(table.loc[ids, "log_odds"].to_numpy())
    z = np.concatenate(z_parts)
    log_odds = np.concatenate(odds)
    y = frame["label"].to_numpy()
    groups = frame["group"].to_numpy()
    clusters = frame["template_id"].astype(str).to_numpy() if name == "A" else None
    if name == "A":
        # A2 rows have an empty template id. Give each of them its own cluster.
        clusters = np.array([value if value else f"a2:{item}" for value, item in zip(clusters, frame["item_id"].astype(str))])
    return z, log_odds, y, groups, clusters


def _write_shift(shift, frame, root, sha, thresholds, gate, B, seed):
    z, log_odds, y, groups, clusters = _columns(frame, root, sha, shift)
    from amon.cascade.thresholds import llm_predict

    llm = llm_predict(log_odds, thresholds["tau_llm"])
    tokens = pd.read_parquet(root / "processed" / "tokens.parquet").set_index("item_id")
    token_count = tokens.loc[frame.sort_values("item_id")["item_id"].astype(str), "token_count"].to_numpy()
    # _columns sorts by item_id, so align tokens to that order too.
    ordered = frame.sort_values("item_id")
    token_count = tokens.loc[ordered["item_id"].astype(str), "token_count"].to_numpy()
    counts = condition_b(z, llm, y, thresholds["t0"], thresholds["t1"], token_count, f"heldout_{shift}" if shift != "A" else "heldout_A")
    counts_10 = headroom_counts(z, llm, y, thresholds["t0_10"], thresholds["t1_10"], token_count)
    outside = (z <= thresholds["t0"]) | (z >= thresholds["t1"])
    cascade = np.where(outside, np.where(z >= thresholds["t1"], 1, 0), llm)
    fixable = outside & (cascade != y) & (llm == y)
    samples = paired_bootstrap(
        lambda idx, mask=fixable: float(np.sum(mask[idx])),
        len(y), B, seed, clusters=clusters, strata=groups,
    )
    lo, hi = percentile_interval(samples)
    deff = variance_deff(z, log_odds, y, groups, gate["weights"], thresholds["tau_llm"], thresholds["t_star"], clusters, B, seed) if shift == "A" else 1.0
    level = ordered.assign(z=z, log_odds=log_odds, llm_pred=llm, y=y)
    level.to_parquet(root / "gate1" / f"items_{shift}.parquet", index=False)
    recomputed = headroom_counts(level["z"].to_numpy(), level["llm_pred"].to_numpy(), level["y"].to_numpy(), thresholds["t0"], thresholds["t1"], token_count)
    if (recomputed["F"], recomputed["U"], recomputed["I"]) != (counts["F"], counts["U"], counts["I"]):
        raise SystemExit(f"{shift} item-level recount does not match the JSON")
    payload = {
        **counts,
        "F_interval": [lo, hi],
        "at_10": counts_10,
        "deff": deff,
        "n_deferred": counts["n_deferred"],
    }
    write_json(root / "gate1" / f"counts_{shift}.json", payload)


if __name__ == "__main__":
    raise SystemExit(main())
