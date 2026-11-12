"""Map each source's raw columns onto the items schema. Unknown columns stop the run."""

from __future__ import annotations

import pandas as pd

from amon.data.textutil import group_id_for, is_code, is_python, item_id, messages_json

REQUIRED = {
    "wildjailbreak": {"vanilla", "adversarial", "data_type"},
    "wildchat": {"conversation"},
    "jailbreak_prompts": {"prompt"},
    "strongreject": set(),  # prompt or forbidden_prompt, checked below
    "llama3_jailbreaks": {"prompt"},
    "orbench": {"prompt"},
    "xstest": set(),
    "harmbench": set(),
}


class SchemaError(RuntimeError):
    """A dataset is missing a column this pipeline is specified to read."""


def _require(df: pd.DataFrame, columns: set[str], source: str) -> None:
    missing = columns - set(df.columns)
    if missing:
        raise SchemaError(
            f"{source} is missing {sorted(missing)}. Columns present: {list(df.columns)}"
        )


def _row_key(source: str, *parts: str) -> str:
    return item_id(source, "\n".join("" if part is None else str(part) for part in parts))


def _base_row(**kwargs) -> dict:
    row = {
        "item_id": kwargs["item_id"],
        "source": kwargs["source"],
        "cell": kwargs["cell"],
        "role": kwargs["role"],
        "label": kwargs["label"],
        "group": kwargs["group"],
        "messages": kwargs["messages"],
        "group_id": kwargs.get("group_id", ""),
        "template_id": kwargs.get("template_id", ""),
        "request_id": kwargs.get("request_id", ""),
        "lang": kwargs.get("lang", ""),
        "is_code": bool(kwargs.get("is_code", False)),
        "_source_request": kwargs.get("source_request", ""),
        "_rendered": kwargs.get("rendered", ""),
    }
    return row


def normalize(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Rows in the items schema, plus temporary `_source_request` and `_rendered` columns."""
    if name not in REQUIRED and name != "strongreject":
        raise SchemaError(f"no normalizer for {name!r}")
    if name == "wildjailbreak":
        return _wildjailbreak(df)
    if name == "wildchat":
        return _wildchat(df)
    if name == "jailbreak_prompts":
        return _prompts(df, "jailbreak_prompts", "template")
    if name == "strongreject":
        return _strongreject(df)
    if name == "llama3_jailbreaks":
        return _bailey(df)
    if name == "orbench":
        return _single_prompt(df, "orbench", "orbench", "heldout_B", 0, "hard_negatives")
    if name == "xstest":
        return _xstest(df)
    if name == "harmbench":
        return _harmbench(df)
    raise SchemaError(f"no normalizer for {name!r}")


def _wildjailbreak(df: pd.DataFrame) -> pd.DataFrame:
    _require(df, REQUIRED["wildjailbreak"], "wildjailbreak")
    rows = []
    for record in df.itertuples(index=False):
        data_type = str(getattr(record, "data_type"))
        vanilla = getattr(record, "vanilla")
        adversarial = getattr(record, "adversarial")
        vanilla = "" if vanilla is None or (isinstance(vanilla, float) and pd.isna(vanilla)) else str(vanilla)
        adversarial = "" if adversarial is None or (isinstance(adversarial, float) and pd.isna(adversarial)) else str(adversarial)
        harmful = "harmful" in data_type
        adversarial_row = data_type.startswith("adversarial")
        shown = adversarial if adversarial_row and adversarial else vanilla
        if not shown:
            continue
        source_request = vanilla or shown
        rows.append(
            _base_row(
                item_id=_row_key("wildjailbreak", data_type, vanilla, adversarial),
                source="wildjailbreak",
                cell=data_type,
                role="tune" if adversarial_row else "train",
                label=1 if harmful else 0,
                group="misses" if harmful else "hard_negatives",
                messages=messages_json([{"role": "user", "content": shown}]),
                group_id=group_id_for(source_request),
                lang="en",
                is_code=is_code(shown),
                source_request=source_request,
            )
        )
    return pd.DataFrame(rows)


def _first_user_turn(conversation) -> dict | None:
    if not isinstance(conversation, list) or not conversation:
        return None
    turn = conversation[0]
    if not isinstance(turn, dict) or turn.get("role") != "user":
        return None
    return turn


def _flagged(moderation, index: int = 0) -> bool:
    if moderation is None or (isinstance(moderation, float) and pd.isna(moderation)):
        return False
    entry = None
    if isinstance(moderation, list):
        if not moderation:
            return False
        entry = moderation[index] if index < len(moderation) else moderation[0]
    elif isinstance(moderation, dict):
        entry = moderation
    if not isinstance(entry, dict):
        return False
    return bool(entry.get("flagged", False))


def _wildchat(df: pd.DataFrame) -> pd.DataFrame:
    _require(df, REQUIRED["wildchat"], "wildchat")
    rows = []
    has_moderation = "openai_moderation" in df.columns
    has_hash = "conversation_hash" in df.columns
    for offset, record in enumerate(df.itertuples(index=False)):
        turn = _first_user_turn(getattr(record, "conversation"))
        if turn is None:
            continue
        content = str(turn.get("content") or "")
        if not content.strip():
            continue
        moderation = getattr(record, "openai_moderation", None) if has_moderation else None
        lang = str(turn.get("language") or getattr(record, "language", "") or "")
        identity = getattr(record, "conversation_hash", "") if has_hash else ""
        if not identity:
            identity = f"{offset}:{content}"
        rows.append(
            _base_row(
                item_id=_row_key("wildchat", str(identity)),
                source="wildchat",
                cell="ordinary",
                role="",
                label=0,
                group="ordinary",
                messages=messages_json([{"role": "user", "content": content}]),
                group_id=group_id_for(content),
                lang=lang,
                is_code=is_code(content) or is_python(content),
                source_request=content,
            )
        )
        rows[-1]["_moderation_flagged"] = _flagged(moderation)
        rows[-1]["_conversation_hash"] = str(identity)
    return pd.DataFrame(rows)


def _prompts(df: pd.DataFrame, source: str, cell: str) -> pd.DataFrame:
    _require(df, {"prompt"}, source)
    rows = []
    for offset, record in enumerate(df.itertuples(index=False)):
        text = str(getattr(record, "prompt") or "")
        if not text.strip():
            continue
        rows.append(
            _base_row(
                item_id=_row_key(source, str(offset), text),
                source=source,
                cell=cell,
                role="",
                label=1,
                group="misses",
                messages=messages_json([{"role": "user", "content": text}]),
                source_request=text,
                lang="en",
            )
        )
    return pd.DataFrame(rows)


def _strongreject(df: pd.DataFrame) -> pd.DataFrame:
    column = "forbidden_prompt" if "forbidden_prompt" in df.columns else "prompt" if "prompt" in df.columns else None
    if column is None:
        raise SchemaError(
            "StrongREJECT has neither 'forbidden_prompt' nor 'prompt'. "
            f"Columns present: {list(df.columns)}"
        )
    renamed = df.rename(columns={column: "prompt"})
    return _prompts(renamed, "strongreject", "forbidden")


def _bailey(df: pd.DataFrame) -> pd.DataFrame:
    _require(df, {"prompt"}, "llama3_jailbreaks")
    if "split" not in df.columns:
        raise SchemaError("llama3-jailbreaks rows need a 'split' column naming the source split")
    rows = []
    for record in df.itertuples(index=False):
        prompt = str(getattr(record, "prompt") or "")
        split = str(getattr(record, "split"))
        if not prompt.strip():
            continue
        harmful = split.startswith("harmful_")
        pipeline = "instructions" in split
        if pipeline:
            role = "pipeline"
            group = "misses" if harmful else "ordinary"
        elif split.startswith("harmful_"):
            role = "heldout_A2"
            group = "misses"
        else:
            continue
        rows.append(
            _base_row(
                item_id=_row_key("llama3_jailbreaks", split, prompt),
                source="llama3_jailbreaks",
                cell=split,
                role=role,
                label=1 if harmful else 0,
                group=group,
                messages="",
                rendered=prompt,
                lang="en",
                source_request=prompt,
            )
        )
    return pd.DataFrame(rows)


def _single_prompt(df, source, cell, role, label, group) -> pd.DataFrame:
    _require(df, {"prompt"}, source)
    rows = []
    for offset, record in enumerate(df.itertuples(index=False)):
        text = str(getattr(record, "prompt") or "")
        if not text.strip():
            continue
        rows.append(
            _base_row(
                item_id=_row_key(source, str(offset), text),
                source=source,
                cell=cell,
                role=role,
                label=label,
                group=group,
                messages=messages_json([{"role": "user", "content": text}]),
                lang="en",
                is_code=is_code(text),
                source_request=text,
            )
        )
    return pd.DataFrame(rows)


def _xstest(df: pd.DataFrame) -> pd.DataFrame:
    label_column = "label" if "label" in df.columns else "type" if "type" in df.columns else None
    text_column = "prompt" if "prompt" in df.columns else None
    if label_column is None or text_column is None:
        raise SchemaError(
            "XSTest needs a prompt column and a label or type column. "
            f"Columns present: {list(df.columns)}"
        )
    safe = df[df[label_column].astype(str).str.lower().str.startswith("safe")]
    renamed = safe.rename(columns={text_column: "prompt"})
    out = _single_prompt(renamed, "xstest", "xstest_safe", "heldout_B", 0, "hard_negatives")
    return out


def _harmbench(df: pd.DataFrame) -> pd.DataFrame:
    column = next((name for name in ("Behavior", "behavior", "prompt") if name in df.columns), None)
    if column is None:
        raise SchemaError(
            "HarmBench needs a Behavior column. "
            f"Columns present: {list(df.columns)}"
        )
    renamed = df.rename(columns={column: "prompt"})
    out = _single_prompt(renamed, "harmbench", "behavior", "heldout_A1", 1, "misses")
    return out
