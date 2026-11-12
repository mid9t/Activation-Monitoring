"""Stage 02. Cache activations. Runs on the GPU machine."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

from amon.cache import cache_split, verify_cache
from amon.config import load_config, require_paths
from amon.provenance import publish_manifest


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Cache residual-stream activations.").parse_args(argv)
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
    roles = ["train", "trainval", "tune", "heldout_A1", "heldout_A2", "heldout_B", "heldout_C", "pipeline"]
    failed = False
    for role in roles:
        out = paths.data_root / "cache" / f"llama3-8b-instruct@{sha}" / role
        cfg = SimpleNamespace(
            layers=cache_cfg["layers"],
            dtype_store=cache_cfg["dtype_store"],
            token_budget_per_batch=cache_cfg["token_budget_per_batch"],
            data_root=paths.data_root,
            seed=1,
            model_sha=sha,
            inputs=[paths.data_root / "processed" / "splits" / f"{role}.ids"],
            items=None,
        )
        cache_split(role, model, tok, cfg, out)
        ids = (out / "split.ids").read_text().split()
        verdict = verify_cache(out, ids, model, tok, n_check=int(cache_cfg["verify_n"]))
        print(role, verdict)
        publish_manifest(out / "manifest.json", f"02_cache_{role}.json")
        if not verdict["pass"] or verdict["min_cosine"] < float(cache_cfg["verify_min_cosine"]):
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
