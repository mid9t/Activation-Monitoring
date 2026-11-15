"""Stage 03. Score Llama Guard. Train-val is scored before the tune pool.

The policy comes only from configs/llm.yaml. Pass 1 is `--roles trainval`.
Pass 2 refuses to run until policy_decided is true and the train-val manifest
records the same policy hash.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from amon.config import load_config, require_paths
from amon.data.textutil import user_text
from amon.io import write_json
from amon.llm import Policy, answer_token_ids, generate_check, prompt_hash, render_guard_prompt, score_guard
from amon.provenance import publish_manifest, write_manifest
from amon.splits import GUARD_ROLES, ROLES, limit_items, parse_roles, resolve_out_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score Llama Guard 3.")
    parser.add_argument("--roles", default="", help="Comma-separated roles. Default: every role except train.")
    parser.add_argument("--limit", type=int, default=0, help="Smoke run: N items per role, stratified by label.")
    args = parser.parse_args(argv)
    roles = parse_roles(args.roles, allowed=ROLES, default=GUARD_ROLES)
    if args.limit < 0:
        raise SystemExit("--limit must be >= 0")
    paths = require_paths()
    models = load_config("models.yaml")
    llm_cfg = load_config("llm.yaml")
    sha = models["guard"]["revision"]
    if sha in (None, "", "TBD"):
        raise SystemExit("guard model revision is TBD. Run stage 00 first.")
    policy = Policy(name=str(llm_cfg["policy"]), extra_category=_extra(llm_cfg.get("extra_category")))
    if any(role != "trainval" for role in roles):
        _require_policy_decision(paths, sha, llm_cfg, policy)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if paths.device == "cuda" else torch.float16
    tok = AutoTokenizer.from_pretrained(models["guard"]["repo"], revision=sha)
    tok.pad_token = tok.eos_token
    answer_token_ids(tok)
    model = AutoModelForCausalLM.from_pretrained(models["guard"]["repo"], revision=sha, torch_dtype=dtype)
    model.to(paths.device)
    out_root = resolve_out_root(paths.data_root, paths.out_root, args.limit)
    smoke = args.limit > 0
    out_dir = out_root / "llm" / f"llama-guard-3-8b@{sha}"
    cfg = SimpleNamespace(
        data_root=paths.data_root,
        token_budget_per_batch=32000,
        seed=0 if args.limit else 1,
        model_sha=sha,
    )
    policy_sha = prompt_hash(policy.identity())
    for role in roles:
        items = _role_items(paths.data_root, role, args.limit)
        path = score_guard(role, model, tok, policy, out_dir / f"{role}.parquet", items=items, cfg=cfg)
        frame = pd.read_parquet(path)
        share = float((frame["mass2"] >= float(llm_cfg["min_two_token_mass"])).mean())
        print(f"{role}: two-token mass >= {llm_cfg['min_two_token_mass']} on {share:.4f}")
        if share < 0.99:
            raise SystemExit(f"{role} failed the two-token mass check ({share:.4f})")
        prompts = [_render(messages, tok, policy) for messages in items["messages"].tolist()]
        check = generate_check(
            model,
            tok,
            prompts,
            frame["item_id"].tolist(),
            frame["log_odds"].to_numpy(),
            n=int(llm_cfg["generate_check_n"]),
            seed=0,
        )
        check_path = out_dir / "checks" / f"generate_{role}.json"
        write_json(check_path, check)
        print(f"{role}: generate check agreement {check['agreement']:.4f} on {check['n']}")
        if not check["pass"]:
            raise SystemExit(f"{role} failed the generate check ({check['agreement']:.4f})")
        manifest = write_manifest(
            out_dir / "manifests" / role,
            [paths.data_root / "processed" / "items.parquet", paths.data_root / "processed" / "splits" / f"{role}.ids"],
            {
                "seed": cfg.seed,
                "dtype": str(getattr(model, "dtype", dtype)),
                "device": paths.device,
                "model_sha": sha,
                "policy": policy.identity(),
                "policy_sha256": policy_sha,
                "prompt_hash": policy_sha,
                "split": role,
                "smoke": smoke,
            },
            outputs=[path, check_path],
        )
        publish_manifest(
            manifest,
            f"03_llm_{role}.json",
            manifests_dir=(out_root / "manifests") if smoke else None,
        )
        if role == "trainval":
            _write_policy_review(paths.data_root, out_root, path)
    return 0


def _extra(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "null", "none"}:
        return None
    return text


def _require_policy_decision(paths, sha: str, llm_cfg: dict, policy: Policy) -> None:
    if not llm_cfg.get("policy_decided"):
        raise SystemExit(
            "policy_decided is false. Commit the policy decision before scoring roles other than trainval."
        )
    manifest_path = _trainval_manifest(paths, sha)
    if not manifest_path.is_file():
        raise SystemExit(f"missing train-val manifest at {manifest_path}")
    recorded = json.loads(manifest_path.read_text()).get("policy_sha256")
    if recorded != prompt_hash(policy.identity()):
        raise SystemExit("train-val policy_sha256 does not match configs/llm.yaml")


def _trainval_manifest(paths, sha: str) -> Path:
    name = "03_llm_trainval.json"
    candidates = []
    if os.environ.get("AMON_MANIFESTS_DIR"):
        candidates.append(Path(os.environ["AMON_MANIFESTS_DIR"]) / name)
    folder = f"llama-guard-3-8b@{sha}"
    candidates.append(paths.out_root / "llm" / folder / "manifests" / "trainval" / "manifest.json")
    candidates.append(paths.data_root / "llm" / folder / "manifests" / "trainval" / "manifest.json")
    from amon.guard import repo_root

    candidates.append(repo_root() / "manifests" / name)
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def _role_items(data_root: Path, role: str, limit: int):
    ids = [
        line.strip()
        for line in (data_root / "processed" / "splits" / f"{role}.ids").read_text().splitlines()
        if line.strip()
    ]
    items = pd.read_parquet(data_root / "processed" / "items.parquet").set_index("item_id").loc[ids].reset_index()
    if limit:
        return limit_items(items, limit, seed=0)
    return items


def _render(messages, tok, policy: Policy) -> str:
    parsed = json.loads(messages) if isinstance(messages, str) else messages
    return render_guard_prompt(parsed, tok, policy)


def _write_policy_review(data_root: Path, out_root: Path, scores_path: Path) -> None:
    items = pd.read_parquet(data_root / "processed" / "items.parquet")
    scores = pd.read_parquet(scores_path)
    merged = items.merge(scores, on="item_id")
    harmful = merged.loc[(merged["role"] == "trainval") & (merged["label"] == 1) & (merged["log_odds"] <= 0)]
    missed = harmful.head(50).assign(text=lambda frame: frame["messages"].map(user_text))
    dest = out_root / "llm" / "policy_review.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    missed[["item_id", "text", "log_odds"]].to_csv(dest, index=False)
    print(f"wrote {len(missed)} missed harmful train-val items to {dest}")


if __name__ == "__main__":
    raise SystemExit(main())
