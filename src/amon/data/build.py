"""Assemble the frozen item table from normalized frames. Downloads stay in sources.py."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from amon.data.catalog import ENGLISH, TUNE_SLICE_LANGUAGES
from amon.data.dedup import cross_role_group_ids, dedup_across_roles, minhash_pairs
from amon.data.groups import assign_groups, split_overlapping_wildjailbreak
from amon.data.render import check_token_batch, parse_rendered_llama3, render_llama3, tokenize_rendered
from amon.data.templates import build_templated_jailbreaks
from amon.data.textutil import BOS, is_code, is_python, messages_json, user_text

ITEM_COLUMNS = [
    "item_id",
    "source",
    "cell",
    "role",
    "label",
    "group",
    "messages",
    "group_id",
    "template_id",
    "request_id",
    "lang",
    "is_code",
]


def _cfg(cfg, key, default=None):
    if isinstance(cfg, dict):
        return cfg[key] if default is None and key in cfg else cfg.get(key, default)
    return getattr(cfg, key, default)


def _sample(frame: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if n <= 0 or frame.empty:
        return frame.iloc[0:0]
    if len(frame) <= n:
        return frame
    index = np.sort(rng.choice(len(frame), size=n, replace=False))
    return frame.iloc[index]


def _language(code: str) -> str:
    return str(code or "").strip().lower().split("-")[0]


def _is_english(code: str) -> bool:
    return _language(code) in ENGLISH


def filter_wildchat(frame: pd.DataFrame, cfg) -> pd.DataFrame:
    """First user turn, drop moderation flags, then cap near-duplicate clusters."""
    wildchat = _cfg(cfg, "wildchat")
    out = frame.copy()
    if "_moderation_flagged" in out.columns and wildchat.get("drop_moderation_flagged", True):
        out = out.loc[~out["_moderation_flagged"].astype(bool)]
    cap = int(wildchat.get("max_per_neardup_cluster", 2))
    if cap > 0 and len(out) > 1:
        texts = out["_source_request"].astype(str).tolist()
        dedup = _cfg(cfg, "dedup")
        pairs = minhash_pairs(
            texts,
            num_perm=int(dedup["minhash_perm"]),
            threshold=float(dedup["jaccard"]),
            width=int(dedup["minhash_shingle_words"]),
            seed=int(_cfg(cfg, "seed")),
        )
        parent = list(range(len(out)))

        def find(item: int) -> int:
            while parent[item] != item:
                parent[item] = parent[parent[item]]
                item = parent[item]
            return item

        for left, right in pairs:
            parent[find(right)] = find(left)
        keep = []
        buckets: dict[int, list[int]] = {}
        for index in range(len(out)):
            buckets.setdefault(find(index), []).append(index)
        positions = {item_id: index for index, item_id in enumerate(out["item_id"].tolist())}
        ordered = out.reset_index(drop=True)
        for members in buckets.values():
            chosen = sorted(members, key=lambda index: ordered.iloc[index]["item_id"])[:cap]
            keep.extend(chosen)
        out = ordered.iloc[sorted(keep)]
    return out.reset_index(drop=True)


def split_trainval(train: pd.DataFrame, frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out `frac` of train groups, stratified by cell. A group is not split."""
    if train.empty:
        return train.copy(), train.copy()
    rng = np.random.default_rng(seed)
    held = []
    for _, cell in train.groupby("cell", sort=True):
        groups = np.array(sorted(cell["group_id"].unique()))
        n = int(round(frac * len(groups)))
        if n <= 0 or n >= len(groups):
            continue
        chosen = set(rng.choice(groups, size=n, replace=False).tolist())
        held.extend(cell.index[cell["group_id"].isin(chosen)].tolist())
    trainval = train.loc[held].copy()
    trainval["role"] = "trainval"
    rest = train.drop(index=held).copy()
    rest["role"] = "train"
    return rest.reset_index(drop=True), trainval.reset_index(drop=True)


def select_pools(frames: dict[str, pd.DataFrame], cfg: dict, embed_fn=None) -> tuple[pd.DataFrame, dict]:
    """Sample, de-duplicate, and split. `frames` holds normalized tables keyed by source."""
    rng = np.random.default_rng(int(cfg["seed"]))
    notes: dict = {"shortfalls": []}
    wild = assign_groups(frames["wildjailbreak"])
    wild, overlap_report = split_overlapping_wildjailbreak(wild)
    notes["wildjailbreak_overlap"] = overlap_report

    def take(cell: str, role: str, n: int) -> pd.DataFrame:
        pool = wild.loc[(wild["cell"] == cell) & (wild["role"] == role)]
        chosen = _sample(pool, n, rng)
        if len(chosen) < n:
            notes["shortfalls"].append(
                {"cell": cell, "role": role, "requested": n, "got": int(len(chosen))}
            )
        return chosen

    train_h = take("vanilla_harmful", "train", int(cfg["train"]["vanilla_harmful"]))
    train_b = take("vanilla_benign", "train", int(cfg["train"]["vanilla_benign"]))
    tune_h = take("adversarial_harmful", "tune", int(cfg["tune"]["adversarial_harmful"]))
    tune_b = take("adversarial_benign", "tune", int(cfg["tune"]["adversarial_benign"]))

    chat = filter_wildchat(assign_groups(frames["wildchat"]), cfg)
    chat = chat.loc[~chat["is_code"].astype(bool)].copy()
    english = chat.loc[chat["lang"].map(_is_english)].copy()
    train_o = _sample(english, int(cfg["train"]["ordinary"]), rng)
    train_o["role"] = "train"
    train_o["cell"] = "ordinary"
    used = set(train_o["_source_request"].astype(str))
    tune_pool = english.loc[~english["_source_request"].astype(str).isin(used)]
    tune_o = _sample(tune_pool, int(cfg["tune"]["ordinary"]), rng)
    tune_o["role"] = "tune"
    tune_o["cell"] = "ordinary"
    if len(train_o) < int(cfg["train"]["ordinary"]):
        notes["shortfalls"].append(
            {"cell": "ordinary", "role": "train", "requested": int(cfg["train"]["ordinary"]), "got": int(len(train_o))}
        )
    if len(tune_o) < int(cfg["tune"]["ordinary"]):
        notes["shortfalls"].append(
            {"cell": "ordinary", "role": "tune", "requested": int(cfg["tune"]["ordinary"]), "got": int(len(tune_o))}
        )

    parts = [train_h, train_b, train_o, tune_h, tune_b, tune_o]
    domain_n = int(cfg["tune"].get("domain_slice", 0))
    slice_langs = TUNE_SLICE_LANGUAGES if domain_n > 0 else frozenset()
    if domain_n > 0:
        domain = chat.loc[chat["lang"].map(_language).isin(slice_langs) | chat["_source_request"].map(is_python)]
        domain = _sample(domain, domain_n, rng)
        domain["role"] = "tune"
        domain["cell"] = "domain_shift"
        domain["group"] = "ordinary"
        parts.append(domain)

    held_c = cfg["heldout"]["C"]
    non_english = chat.loc[~chat["lang"].map(_is_english) & ~chat["is_code"].astype(bool)]
    if held_c.get("exclude_languages_in_tune_slice", True) and slice_langs:
        non_english = non_english.loc[~non_english["lang"].map(_language).isin(slice_langs)]
    used_chat = used | set(tune_o["_source_request"].astype(str))
    non_english = non_english.loc[~non_english["_source_request"].astype(str).isin(used_chat)]
    code = chat.loc[chat["is_code"].astype(bool) & ~chat["_source_request"].map(is_python)]
    code = code.loc[~code["_source_request"].astype(str).isin(used_chat)]
    c_lang = _sample(non_english, int(held_c["non_english"]), rng)
    c_code = _sample(code, int(held_c["code"]), rng)
    for frame, cell in ((c_lang, "non_english"), (c_code, "code")):
        frame["role"] = "heldout_C"
        frame["cell"] = cell
        frame["group"] = "ordinary"
        frame["label"] = np.int8(0)
    parts.extend([c_lang, c_code])

    templates = frames["jailbreak_prompts"]
    requests = frames["strongreject"]
    tune_texts = set(pd.concat([tune_h, tune_b, tune_o], ignore_index=True)["_source_request"].map(str))
    templates = templates.loc[~templates["_source_request"].astype(str).isin(tune_texts)]
    train_harm_texts = set(train_h["_source_request"].map(lambda text: str(text).strip().lower()))
    requests = requests.loc[~requests["_source_request"].astype(str).str.strip().str.lower().isin(train_harm_texts)]
    template_frame = pd.DataFrame(
        {
            "template_id": templates["item_id"].astype(str),
            "text": templates["_source_request"].astype(str),
        }
    )
    request_frame = pd.DataFrame(
        {
            "request_id": requests["item_id"].astype(str),
            "text": requests["_source_request"].astype(str),
        }
    )
    a1 = build_templated_jailbreaks(
        template_frame,
        request_frame,
        n_templates=int(cfg["heldout"]["A1"]["n_templates"]),
        n_requests=int(cfg["heldout"]["A1"]["n_requests"]),
        seed=int(cfg["seed"]),
    )
    notes["a1_insertions"] = a1.attrs.get("insertions")
    if len(a1) < int(cfg["heldout"]["A1"]["n_templates"]) * int(cfg["heldout"]["A1"]["n_requests"]):
        notes["shortfalls"].append({"cell": "templated", "role": "heldout_A1", "got": int(len(a1))})
    if not a1.empty:
        parts.append(a1)

    if "llama3_jailbreaks" in frames:
        bailey = frames["llama3_jailbreaks"]
        exclude = set(cfg["heldout"]["A2"]["exclude_splits"])
        attacks = bailey.loc[
            bailey["cell"].astype(str).str.startswith("harmful_")
            & ~bailey["cell"].isin(exclude)
            & ~bailey["cell"].astype(str).str.contains("instructions")
            & (bailey["role"] == "heldout_A2")
        ]
        pipeline = bailey.loc[bailey["role"] == "pipeline"]
        parts.extend([attacks, pipeline])

    orbench = _sample(frames["orbench"], int(cfg["heldout"]["B"]["orbench_n"]), rng)
    orbench["role"] = "heldout_B"
    xstest = frames["xstest"]
    parts.extend([orbench, xstest])

    items = pd.concat(parts, ignore_index=True)
    items = assign_groups(items)
    dedup_cfg = dict(cfg["dedup"])
    dedup_cfg["seed"] = int(cfg["seed"])
    if embed_fn is not None:
        dedup_cfg["embed_fn"] = embed_fn
    items, dedup_report = dedup_across_roles(items, dedup_cfg)
    notes["dedup"] = dedup_report
    leftover = cross_role_group_ids(items)
    notes["cross_role_group_ids"] = leftover

    train_mask = items["role"] == "train"
    train, trainval = split_trainval(items.loc[train_mask], float(cfg["train"]["trainval_frac"]), int(cfg["seed"]))
    others = items.loc[~train_mask]
    items = pd.concat([train, trainval, others], ignore_index=True)
    notes["counts"] = items.groupby(["role", "cell"]).size().astype(int).to_dict()
    return items, notes


def render_and_tokenize(items: pd.DataFrame, tok, max_tokens: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Render with the Llama 3 template, drop items over `max_tokens`, and check the tail ids."""
    rendered = []
    round_trip_failures = []
    for record in items.itertuples(index=False):
        original = getattr(record, "_rendered", "") or ""
        if original:
            messages = parse_rendered_llama3(original)
            if not messages:
                messages = [{"role": "user", "content": user_text(original) or original}]
            text = render_llama3(messages, tok)
            if text != original:
                round_trip_failures.append(record.item_id)
        else:
            messages = json.loads(record.messages)
            text = render_llama3(messages, tok)
        rendered.append((messages, text))

    bos_id = _bos_id(tok)
    encoded = [tokenize_rendered(text, tok) for _, text in rendered]
    keep = []
    token_rows = []
    last5 = None
    for index, ids in enumerate(encoded):
        dropped = len(ids) > max_tokens or len(ids) < 3
        tail = ids[-5:] if len(ids) >= 5 else []
        token_rows.append(
            {
                "item_id": items.iloc[index]["item_id"],
                "token_count": int(len(ids)),
                "dropped": bool(dropped),
                "last5_ids": tail,
            }
        )
        if not dropped:
            keep.append(index)
            if last5 is None:
                last5 = tuple(tail)
    kept_ids = [encoded[index] for index in keep]
    check_token_batch(kept_ids, bos_id)
    kept = items.iloc[keep].copy()
    messages = [messages_json(rendered[index][0]) for index in keep]
    kept["messages"] = messages
    tokens = pd.DataFrame(token_rows)
    report = {
        "round_trip_failures": round_trip_failures,
        "dropped": int(tokens["dropped"].sum()),
        "drop_rate_by_source": _drop_rates(items, tokens),
    }
    return kept.reset_index(drop=True), tokens, report


def _bos_id(tok) -> int:
    if hasattr(tok, "bos_token_id") and tok.bos_token_id is not None:
        return int(tok.bos_token_id)
    ids = tokenize_rendered(BOS, tok)
    if len(ids) != 1:
        raise ValueError("<|begin_of_text|> must be a single token")
    return int(ids[0])


def _drop_rates(items: pd.DataFrame, tokens: pd.DataFrame) -> dict:
    merged = items[["item_id", "source"]].merge(tokens[["item_id", "dropped"]], on="item_id")
    rates = {}
    for source, frame in merged.groupby("source"):
        rates[str(source)] = float(frame["dropped"].mean()) if len(frame) else 0.0
    return rates


def acceptance(items: pd.DataFrame, tokens: pd.DataFrame, notes: dict, targets: dict) -> dict:
    """Stage 01 acceptance. A shortfall is a pass when it is listed with a reason."""
    problems = []
    if notes.get("cross_role_group_ids"):
        problems.append("cross-role near-duplicates remain")
    kept_ids = set(items["item_id"])
    kept_tokens = tokens.loc[tokens["item_id"].isin(kept_ids) & ~tokens["dropped"]]
    if kept_tokens.empty:
        problems.append("no items survived tokenization")
    else:
        tails = kept_tokens["last5_ids"].map(tuple)
        if tails.nunique() != 1:
            problems.append("last five token ids are not identical")
    within = []
    explained = []
    for (role, cell), target in _target_map(targets).items():
        if role == "train":
            got = int((items["role"].isin(["train", "trainval"]) & (items["cell"] == cell)).sum())
        else:
            got = int(((items["role"] == role) & (items["cell"] == cell)).sum())
        if target and abs(got - target) / target > 0.01:
            reason = next(
                (
                    row
                    for row in notes.get("shortfalls", [])
                    if row.get("role") == role and row.get("cell") == cell
                ),
                None,
            )
            token_drops = notes.get("tokens", {}).get("dropped", 0)
            gap = {"role": role, "cell": cell, "got": got, "target": target}
            if reason is None and not token_drops:
                within.append(gap)
            else:
                explained.append(gap)
    if within:
        problems.append(f"counts outside 1% without a logged shortfall: {within}")
    return {
        "pass": not problems,
        "problems": problems,
        "shortfalls": notes.get("shortfalls", []),
        "explained_gaps": explained,
    }


def _target_map(cfg: dict) -> dict[tuple[str, str], int]:
    return {
        ("train", "vanilla_harmful"): int(cfg["train"]["vanilla_harmful"]),
        ("train", "vanilla_benign"): int(cfg["train"]["vanilla_benign"]),
        ("train", "ordinary"): int(cfg["train"]["ordinary"]),
        ("tune", "adversarial_harmful"): int(cfg["tune"]["adversarial_harmful"]),
        ("tune", "adversarial_benign"): int(cfg["tune"]["adversarial_benign"]),
        ("tune", "ordinary"): int(cfg["tune"]["ordinary"]),
    }
