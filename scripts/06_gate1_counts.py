"""Stage 06. Condition (a) and condition (b). Held-out columns go through guard_read."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from amon.cascade.deferral import generator_shift, mixed_set_errors, write_generator_shift, write_pipeline
from amon.cascade.headroom import headroom_counts
from amon.cascade.thresholds import llm_predict
from amon.categories import errors_by, fpr_by, llm_miss_share, recall_by, write_category_bundle
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
    _write_descriptions(items, paths.data_root, sha, thresholds, gate, B, seed)
    manifest = write_manifest(
        paths.data_root / "gate1",
        [paths.data_root / "gate1" / "thresholds.json"],
        {"seed": seed, "device": paths.device},
    )
    publish_manifest(manifest, "06_gate1_counts.json")
    return 0


def _columns(frame, root, sha, name):
    split = "heldout_A" if name == "A" else name if name in {"B", "C"} else "tune"
    guard_read(
        split if split != "tune" else "tune",
        ["z", "log_odds", "label", "group", "template_id", "cell", "harm_category", "thresholds"],
    )
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
    clusters = frame["template_id"].astype(str).to_numpy() if name in {"A", "A1"} else None
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
    for key in ("realized_deferral_items", "realized_deferral_tokens"):
        if key not in counts:
            raise SystemExit(f"{shift} is missing {key}")
    payload = {
        **counts,
        "F_interval": [lo, hi],
        "at_10": counts_10,
        "deff": deff,
        "n_deferred": counts["n_deferred"],
    }
    write_json(root / "gate1" / f"counts_{shift}.json", payload)


def _write_descriptions(items, root, sha, thresholds, gate, B, seed):
    """Harm-definition tables, the pipeline mixed-class check, and the generator-shift AUROCs."""
    if "harm_category" not in items.columns:
        items = items.copy()
        items["harm_category"] = ""
    tau = thresholds["tau_llm"]
    t_star = thresholds["t_star"]
    harmful = items.loc[
        items["role"].isin(["trainval", "tune"])
        & items["cell"].isin(["vanilla_harmful", "adversarial_harmful"])
    ]
    a1 = items.loc[items["role"] == "heldout_A1"]
    a2 = items.loc[items["role"] == "heldout_A2"]
    b = items.loc[items["role"] == "heldout_B"]
    c = items.loc[items["role"] == "heldout_C"]
    tables = {}
    recall_frames = []
    if len(harmful):
        arrays = _aligned(harmful, root, sha, "tune")
        table = recall_by(
            _stratum(harmful, "source", "cell"),
            arrays["log_odds"],
            arrays["y"],
            tau,
            None,
            B,
            seed,
        )
        tables["recall_trainval_tune"] = table
        recall_frames.append((harmful, arrays, table))
    if len(a1):
        arrays = _aligned(a1, root, sha, "A1")
        table = recall_by(
            a1.sort_values("item_id")["harm_category"].astype(str).to_numpy(),
            arrays["log_odds"],
            arrays["y"],
            tau,
            arrays["clusters"],
            B,
            seed,
        )
        tables["recall_A1"] = table
        tables["errors_A1"] = errors_by(
            a1.sort_values("item_id")["harm_category"].astype(str).to_numpy(),
            arrays["z"],
            llm_predict(arrays["log_odds"], tau),
            arrays["y"],
            thresholds["t0"],
            thresholds["t1"],
        )
        recall_frames.append((a1, arrays, table))
    if len(a2):
        ordered = a2.sort_values("item_id")
        arrays = _aligned(a2, root, sha, "A2")
        table = recall_by(ordered["cell"].astype(str).to_numpy(), arrays["log_odds"], arrays["y"], tau, None, B, seed)
        tables["recall_A2"] = table
        tables["errors_A2"] = errors_by(
            ordered["cell"].astype(str).to_numpy(),
            arrays["z"],
            llm_predict(arrays["log_odds"], tau),
            arrays["y"],
            thresholds["t0"],
            thresholds["t1"],
        )
        recall_frames.append((a2, arrays, table))
    for name, frame, split in (("fpr_B", b, "B"), ("fpr_C", c, "C")):
        if frame.empty:
            continue
        arrays = _aligned(frame, root, sha, split)
        ordered = frame.sort_values("item_id")
        strata = _stratum(ordered, "source", "cell")
        llm = fpr_by(strata, llm_predict(arrays["log_odds"], tau), arrays["y"])
        probe = fpr_by(strata, llm_predict(arrays["z"], t_star), arrays["y"])
        tables[name] = llm.merge(probe, on=["stratum", "n"], suffixes=("_llm", "_probe"))
    shares = []
    weights = []
    for frame, arrays, table in recall_frames:
        ordered = frame.sort_values("item_id")
        strata = (
            ordered["harm_category"].astype(str).to_numpy()
            if frame["role"].eq("heldout_A1").all()
            else ordered["cell"].astype(str).to_numpy()
            if frame["role"].eq("heldout_A2").all()
            else _stratum(ordered, "source", "cell")
        )
        pred = llm_predict(arrays["log_odds"], tau)
        misses = int(np.sum((arrays["y"] == 1) & (pred == 0)))
        if misses:
            shares.append(llm_miss_share(table, strata, pred, arrays["y"]) * misses)
            weights.append(misses)
    miss_share = (sum(shares) / sum(weights)) if weights else 0.0
    write_category_bundle(root / "gate1", tables, miss_share)
    pipeline = items.loc[items["role"] == "pipeline"]
    if len(pipeline):
        arrays = _aligned(pipeline, root, sha, "pipeline")
        tokens = _tokens(root, pipeline.sort_values("item_id"))
        write_pipeline(
            root / "gate1",
            mixed_set_errors(
                arrays["z"],
                arrays["log_odds"],
                arrays["y"],
                arrays["groups"],
                gate["weights"],
                thresholds,
                tokens,
                B,
                seed,
            ),
        )
    trainval = items.loc[items["role"] == "trainval"]
    tune = items.loc[items["role"] == "tune"]
    if len(trainval) and len(tune):
        left = _aligned(trainval, root, sha, "trainval")
        right = _aligned(tune, root, sha, "tune")
        write_generator_shift(
            root / "gate1",
            generator_shift(left["z"], left["y"], left["groups"], right["z"], right["y"], right["groups"], B, seed),
        )


def _aligned(frame, root, sha, name):
    z, log_odds, y, groups, clusters = _columns(frame, root, sha, name)
    return {"z": z, "log_odds": log_odds, "y": y, "groups": groups, "clusters": clusters}


def _stratum(frame, left: str, right: str):
    ordered = frame.sort_values("item_id")
    return (ordered[left].astype(str) + "|" + ordered[right].astype(str)).to_numpy()


def _tokens(root, ordered):
    tokens = pd.read_parquet(root / "processed" / "tokens.parquet").set_index("item_id")
    return tokens.loc[ordered["item_id"].astype(str), "token_count"].to_numpy()


if __name__ == "__main__":
    raise SystemExit(main())
