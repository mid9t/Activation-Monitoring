"""Stage 05. Fit tau_llm, the band, and t_star on the tune pool only."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from amon.cascade.metric import weighted_error
from amon.cascade.thresholds import cascade_predict
from amon.config import configs_dir, fill_tbd, load_config, require_paths
from amon.gate import fit_thresholds
from amon.io import write_json
from amon.provenance import publish_manifest, sha256_file, write_manifest


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Fit Gate 1 thresholds on the tune pool.").parse_args(argv)
    paths = require_paths()
    gate = load_config("gate1.yaml")
    if gate["thresholds"]["tau_llm"] != "TBD":
        raise SystemExit("thresholds are already written. Stage 05 does not rerun after they are frozen.")
    items = pd.read_parquet(paths.data_root / "processed" / "items.parquet")
    tune = items.loc[items["role"] == "tune"].sort_values("item_id")
    ids = [line.strip() for line in (paths.data_root / "processed" / "splits" / "tune.ids").read_text().splitlines() if line.strip()]
    tune = tune.set_index("item_id").loc[ids].reset_index()
    z = np.load(paths.data_root / "scores" / "probe" / "tune.npy")
    models = load_config("models.yaml")
    scores = pd.read_parquet(paths.data_root / "llm" / f"llama-guard-3-8b@{models['guard']['revision']}" / "tune.parquet")
    scores = scores.set_index("item_id").loc[ids]
    tokens = pd.read_parquet(paths.data_root / "processed" / "tokens.parquet").set_index("item_id").loc[ids]
    fitted = fit_thresholds(
        z,
        scores["log_odds"].to_numpy(),
        tune["label"].to_numpy(),
        tune["group"].to_numpy(),
        gate["weights"],
        gate["budgets"],
    )
    check = _subsample_check(z, fitted["llm_pred"], tune["label"].to_numpy(), fitted["costs"], 0.05)
    payload = {key: fitted[key] for key in ("tau_llm", "t0", "t1", "t_star", "t0_10", "t1_10", "realized_tune_deferral")}
    payload["subsample_check"] = check
    payload["input_hashes"] = {
        "tune_scores": sha256_file(paths.data_root / "scores" / "probe" / "tune.npy"),
    }
    out = paths.data_root / "gate1"
    write_json(out / "thresholds.json", payload)
    fill_tbd(
        configs_dir() / "gate1.yaml",
        {key: payload[key] for key in ("tau_llm", "t0", "t1", "t_star", "t0_10", "t1_10")},
    )
    manifest = write_manifest(out, [paths.data_root / "scores" / "probe" / "tune.npy"], {"seed": gate["bootstrap"]["seed"], "device": paths.device})
    publish_manifest(manifest, "05_thresholds.json")
    realized = payload["realized_tune_deferral"]
    if abs(realized - 0.05) > max(1.0 / len(z), 1e-6) or not check["match"]:
        raise SystemExit(f"threshold acceptance failed: deferral={realized}, check={check}")
    return 0


def _subsample_check(z, llm, y, costs, budget: float) -> dict:
    rng = np.random.default_rng(0)
    n = min(2000, len(z))
    take = rng.choice(len(z), size=n, replace=False)
    # The unit test is the all-pairs comparison. Here, recompute the window error
    # with an independent loop on the subsample and require the same minimum.
    sub_z, sub_llm, sub_y, sub_c = z[take], llm[take], y[take], costs[take]
    from amon.cascade.thresholds import fit_band

    t0, t1 = fit_band(sub_z, sub_llm, sub_y, sub_c, budget)
    got = weighted_error(cascade_predict(sub_z, sub_llm, t0, t1), sub_y, sub_c)
    order = np.argsort(sub_z, kind="mergesort")
    k = int(round(budget * n))
    best = None
    for start in range(n - k + 1):
        pred = np.empty(n, dtype=int)
        pred[order[:start]] = 0
        pred[order[start + k :]] = 1
        pred[order[start : start + k]] = sub_llm[order[start : start + k]]
        error = float(np.sum(sub_c * (pred != sub_y)))
        best = error if best is None else min(best, error)
    return {"match": abs(got - best) <= 1e-8, "window_error": got, "exhaustive_error": best}


if __name__ == "__main__":
    raise SystemExit(main())
