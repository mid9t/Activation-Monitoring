"""Stage 08. Score the hand check. Stops if the tie-break fires."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from amon.config import load_config, require_paths
from amon.handcheck import score_handcheck, tie_break_flag
from amon.io import read_json, write_json


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Score hand-check labels.").parse_args(argv)
    paths = require_paths()
    gate = load_config("gate1.yaml")
    directory = paths.data_root / "gate1" / "handcheck"
    labels_path = directory / "labels.csv"
    if not labels_path.is_file():
        raise SystemExit("labels.csv is missing. Checkpoint 2 is still open.")
    labels = pd.read_csv(labels_path)
    if len(labels) < 65:
        raise SystemExit(f"labels.csv has {len(labels)} rows; checkpoint 2 needs 65.")
    key = pd.read_csv(directory / "key.csv")
    counts = {}
    for shift in ("A", "B", "C"):
        row = read_json(paths.data_root / "gate1" / f"counts_{shift}.json")
        counts[shift] = {"F": row["F"], "U": row["U"], "R": row["R"]}
    result = score_handcheck(key, labels, counts)
    flagged = tie_break_flag(result, float(gate["pass_b"]["min_F_adj"]), float(gate["pass_b"]["min_frac"]))
    result["tie_break"] = flagged
    write_json(directory / "results.json", result)
    errors = result["r_F"]["k"] + result["r_U"]["k"]
    if errors > int(gate["handcheck"]["kill_if_mislabeled_over"]):
        print(
            f"KILL label noise: {errors} of the confident errors are mislabeled. The owner decides the branch.",
            file=sys.stderr,
        )
    if flagged:
        raise SystemExit(
            "TIE-BREAK. A shift's verdict changes at the Wilson upper bound of r_F. "
            "Label 50 more confident errors from that shift."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
