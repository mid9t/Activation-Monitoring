"""Check a Volume download. Every stage 02–04 output must match its manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from amon.provenance import sha256_file


def verify_outputs(manifest_dir: Path, root: Path) -> list[str]:
    """Return a problem line for every smoke manifest, missing file, or hash mismatch."""
    problems = []
    for manifest in sorted(Path(manifest_dir).glob("0[2-4]_*.json")):
        data = json.loads(manifest.read_text())
        if data.get("smoke"):
            problems.append(f"{manifest.name}: smoke manifest in the real set")
        for rel, want in data.get("outputs", {}).items():
            path = root / rel
            if not path.is_file():
                problems.append(f"{manifest.name}: missing {rel}")
            elif path.stat().st_size != want["bytes"] or sha256_file(path) != want["sha256"]:
                problems.append(f"{manifest.name}: hash mismatch {rel}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check downloaded outputs against their manifests.")
    parser.add_argument("data_root", nargs="?", default=None, help="Defaults to $AMON_DATA.")
    parser.add_argument("--manifest-dir", default=None, help="Defaults to <data root>/manifests.")
    args = parser.parse_args(argv)
    root = Path(args.data_root) if args.data_root else Path(os.environ["AMON_DATA"])
    manifest_dir = Path(args.manifest_dir) if args.manifest_dir else root / "manifests"
    problems = verify_outputs(manifest_dir, root)
    print("\n".join(problems) or "all outputs match")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
