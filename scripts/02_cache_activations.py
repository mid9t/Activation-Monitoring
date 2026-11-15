"""Stage 02. Cache activations. One role per process; --limit writes under smoke/."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from amon.cache import cache_split, verify_cache
from amon.cache.extract import layer_name
from amon.config import load_config, require_paths
from amon.io import write_json
from amon.provenance import publish_manifest, write_manifest
from amon.splits import ROLES, parse_roles, resolve_out_root


def cache_dir(out_root: Path, sha: str, role: str) -> Path:
    return Path(out_root) / "cache" / f"llama3-8b-instruct@{sha}" / role


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cache residual-stream activations.")
    parser.add_argument("--roles", default="", help="Comma-separated roles. Default: every role.")
    parser.add_argument("--limit", type=int, default=0, help="Smoke run: N items per role, stratified by label.")
    args = parser.parse_args(argv)
    roles = parse_roles(args.roles, allowed=ROLES, default=ROLES)
    if args.limit < 0:
        raise SystemExit("--limit must be >= 0")
    paths = require_paths()
    cache_cfg = load_config("cache.yaml")
    models = load_config("models.yaml")
    sha = models["monitored"]["revision"]
    if sha in (None, "", "TBD"):
        raise SystemExit("monitored model revision is TBD. Run stage 00 first.")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if paths.device == "cuda" else torch.float16 if paths.device == "mps" else torch.float32
    tok = AutoTokenizer.from_pretrained(models["monitored"]["repo"], revision=sha)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(models["monitored"]["repo"], revision=sha, torch_dtype=dtype)
    model.to(paths.device)
    out_root = resolve_out_root(paths.data_root, paths.out_root, args.limit)
    smoke = args.limit > 0
    failed = False
    for role in roles:
        out = cache_dir(out_root, sha, role)
        items = _role_items(paths.data_root, role, args.limit) if args.limit else None
        cfg = SimpleNamespace(
            layers=cache_cfg["layers"],
            dtype_store=cache_cfg["dtype_store"],
            token_budget_per_batch=cache_cfg["token_budget_per_batch"],
            data_root=paths.data_root,
            seed=0 if args.limit else 1,
            model_sha=sha,
            inputs=[
                paths.data_root / "processed" / "items.parquet",
                paths.data_root / "processed" / "splits" / f"{role}.ids",
            ],
            items=items,
        )
        result = cache_split(role, model, tok, cfg, out)
        if not result["complete"]:
            raise SystemExit(f"{role}: cache stopped before every row was stored")
        ids = (out / "split.ids").read_text().split()
        n_check = int(cache_cfg["verify_n"])
        verdict = verify_cache(out, ids, model, tok, n_check=n_check, data_root=paths.data_root)
        verdict["peaks"] = result["peaks"]
        write_json(out / "verify.json", verdict)
        print(role, {key: verdict[key] for key in ("pass", "min_cosine", "n_check", "problems")})
        passed = verdict["pass"] and verdict["min_cosine"] >= float(cache_cfg["verify_min_cosine"])
        if not passed:
            failed = True
            continue
        outputs = [out / "split.ids", out / "meta.parquet", out / "verify.json"]
        for layer in cache_cfg["layers"]:
            for position in ("last", "mean"):
                outputs.append(out / f"{position}_{layer_name(int(layer))}.npy")
        manifest = write_manifest(
            out,
            cfg.inputs,
            {
                "seed": cfg.seed,
                "dtype": cache_cfg["dtype_store"],
                "device": paths.device,
                "model_sha": sha,
                "layers": [int(layer) for layer in cache_cfg["layers"]],
                "peaks": result["peaks"],
                "split": role,
                "smoke": smoke,
            },
            outputs=outputs,
        )
        publish_manifest(
            manifest,
            f"02_cache_{role}.json",
            manifests_dir=(out_root / "manifests") if smoke else None,
        )
        shutil.rmtree(out / "shards", ignore_errors=True)
    return 1 if failed else 0


def _role_items(data_root: Path, role: str, limit: int):
    from amon.splits import limit_items

    ids = [
        line.strip()
        for line in (data_root / "processed" / "splits" / f"{role}.ids").read_text().splitlines()
        if line.strip()
    ]
    items = pd.read_parquet(data_root / "processed" / "items.parquet").set_index("item_id").loc[ids].reset_index()
    return limit_items(items, limit, seed=0)


if __name__ == "__main__":
    raise SystemExit(main())
