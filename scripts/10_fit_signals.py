"""Stage 10. Fit the two Gate 2 signals on train. Runs only after a pass."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from amon.cache.extract import load_rows
from amon.config import load_config, require_paths
from amon.density import fit_tied_density, min_mahalanobis_sq, reference_quantiles
from amon.guard import GuardError, guard_read
from amon.io import read_json, write_json
from amon.probe import ensemble_std, fit_ensemble, load_probe
from amon.provenance import publish_manifest, write_manifest
from amon.registry import clear_registry, orientation_holds, register_score, run_orientation_tests


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Fit Gate 2 signals.").parse_args(argv)
    paths = require_paths()
    decision = read_json(paths.data_root / "gate1" / "decision.json")
    if decision["branch"] != "pass":
        raise SystemExit(f"stage 10 runs only after a pass; branch is {decision['branch']}")
    probe_cfg = load_config("probe.yaml")
    signal_cfg = load_config("signals.yaml")
    models = load_config("models.yaml")
    layer = int(probe_cfg["selected"]["layer"])
    C = float(probe_cfg["selected"]["C"])
    sha = models["monitored"]["revision"]
    cache = paths.data_root / "cache" / f"llama3-8b-instruct@{sha}"
    items = pd.read_parquet(paths.data_root / "processed" / "items.parquet")
    train = _matrix(cache / "train", items.loc[items["role"] == "train"], layer)
    probe = load_probe(paths.data_root / "probes" / f"lr_L{layer}_C{C:.0e}")
    seeds = list(range(int(signal_cfg["ensemble"]["seeds"][0]), int(signal_cfg["ensemble"]["seeds"][1]) + 1))
    members = fit_ensemble(
        train["X"], train["y"], train["group_id"], C, (probe.mean, probe.std), seeds, probe_cfg["class_weight"]
    )
    density = fit_tied_density(train["X"], train["cell"], signal_cfg["density"]["classes"])
    clear_registry()
    for role in ("train", "trainval", "tune"):
        block = _matrix(cache / role, items.loc[items["role"] == role], layer)
        std = ensemble_std(members, block["X"])
        dist = min_mahalanobis_sq(block["X"], density)
        _save(paths.data_root / "signals" / "ensemble" / f"{role}.npy", std)
        _save(paths.data_root / "signals" / "density" / f"{role}.npy", dist)
        guard_read(role, ["ensemble", "density"])
    try:
        guard_read("heldout_A1", ["ensemble"])
    except GuardError:
        heldout_blocked = True
    else:
        heldout_blocked = False
    if not heldout_blocked:
        raise SystemExit("guard_read did not refuse a held-out signal")
    trainval = _matrix(cache / "trainval", items.loc[items["role"] == "trainval"], layer)
    z = np.load(paths.data_root / "scores" / "probe" / "trainval.npy")
    probe_error = ((z > 0) != trainval["y"]).astype(float)
    ensemble_signal = np.load(paths.data_root / "signals" / "ensemble" / "trainval.npy")
    density_signal = np.load(paths.data_root / "signals" / "density" / "trainval.npy")
    register_score("ensemble", "defer", lambda: orientation_holds(ensemble_signal, probe_error))
    register_score("density", "defer", lambda: orientation_holds(density_signal, probe_error))
    results = run_orientation_tests()
    quantiles = reference_quantiles(density_signal)
    write_json(
        paths.data_root / "signals" / "orientation.json",
        {"tests": results, "density_trainval_quantiles": quantiles, "shrinkage": density["shrinkage"]},
    )
    manifest = write_manifest(paths.data_root / "signals", [], {"seed": 0, "device": paths.device, "model_sha": sha})
    publish_manifest(manifest, "10_signals.json")
    return 0


def _matrix(split_dir, items, layer):
    ids = [line.strip() for line in (split_dir / "split.ids").read_text().splitlines() if line.strip()]
    ordered = items.set_index("item_id").loc[ids]
    cells = ordered["cell"].astype(str).to_numpy()
    mapped = np.array([
        "harmful" if cell == "vanilla_harmful" else "benign_wj" if cell == "vanilla_benign" else "ordinary"
        for cell in cells
    ])
    return {
        "X": np.asarray(load_rows(split_dir, "last", layer), dtype=np.float64),
        "y": ordered["label"].to_numpy(),
        "group_id": ordered["group_id"].to_numpy(),
        "cell": mapped,
    }


def _save(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(values, dtype=np.float64))


if __name__ == "__main__":
    raise SystemExit(main())
