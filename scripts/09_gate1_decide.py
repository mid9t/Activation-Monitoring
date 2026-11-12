"""Stage 09. Refuses a dirty manifest or a prereg/ tree that is not the tag."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from amon.decide import decide
from amon.io import read_json, write_json
from amon.llm import Policy, swap_classifier_messages
from amon.provenance import DirtyTreeError, PreregError, prereg_matches_tag, refuse_dirty_inputs


def repo_root() -> Path:
    from amon.guard import repo_root as root

    return root()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write gate1/decision.json or refuse to.")
    parser.add_argument("--manifest", action="append", default=[])
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--check-prereg", action="store_true")
    parser.add_argument("--repo", default=None)
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args(argv)
    root = Path(args.repo) if args.repo else repo_root()
    if args.check_prereg:
        try:
            prereg_matches_tag(root)
        except PreregError as exc:
            raise SystemExit(str(exc)) from exc
    if args.manifest:
        try:
            refuse_dirty_inputs([Path(path) for path in args.manifest])
        except DirtyTreeError as exc:
            raise SystemExit(str(exc)) from exc
    if args.check_only or args.check_prereg:
        return 0
    try:
        prereg_matches_tag(root)
    except PreregError as exc:
        raise SystemExit(str(exc)) from exc
    if not args.data_root:
        raise SystemExit("stage 09 needs --data-root once the checks pass")
    data = Path(args.data_root)
    condition_a = read_json(data / "gate1" / "condition_a.json")
    hand = read_json(data / "gate1" / "handcheck" / "results.json")
    discounts = hand["discount"]
    deferred = {}
    deff = {}
    for shift in ("A", "B", "C"):
        counts = read_json(data / "gate1" / f"counts_{shift}.json")
        deferred[shift] = int(counts["n_deferred"])
        deff[shift] = float(counts.get("deff", 1.0))
    from amon.config import load_config

    gate = load_config("gate1.yaml")
    decision = decide(
        condition_a,
        discounts,
        hand,
        min_f=float(gate["pass_b"]["min_F_adj"]),
        min_frac=float(gate["pass_b"]["min_frac"]),
        kill_mislabeled=int(gate["handcheck"]["kill_if_mislabeled_over"]),
        materiality=float(gate["materiality_min_swap_frac"]),
        deferred=deferred,
        deff=deff,
    )
    decision["inputs"] = {path: _sha(Path(path)) for path in args.manifest}
    write_json(data / "gate1" / "decision.json", decision)
    if decision["branch"] == "fail_a":
        prompt = swap_classifier_messages(
            [{"role": "user", "content": "REQUEST"}],
            Policy().identity(),
        )
        (data / "llm").mkdir(parents=True, exist_ok=True)
        (data / "llm" / "swap_prompt_template.json").write_text(
            __import__("json").dumps(prompt, indent=2) + "\n"
        )
        print("branch fail_a: swap prompt written. It is not run until the owner approves.", file=sys.stderr)
    return 0


def _sha(path: Path) -> str:
    from amon.provenance import sha256_file

    return sha256_file(path) if path.is_file() else ""


if __name__ == "__main__":
    raise SystemExit(main())
