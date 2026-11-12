"""Shared text, id, and code-detection helpers. No dataset I/O."""

from __future__ import annotations

import hashlib
import json
import re

BOS = "<|begin_of_text|>"
START = "<|start_header_id|>"
END = "<|end_header_id|>"
EOT = "<|eot_id|>"

# A line counts as code when it is an obvious statement, fence-adjacent marker, or tag.
_CODE_LINE = re.compile(
    r"^\s*("
    r"(def|class|import|from|return|elif|else|try|except|with|async|await|package|using)\b|"
    r"#include\b|"
    r"(function|const|let|var|public|private|protected)\b|"
    r"(if|for|while|switch)\s*\(|"
    r"</?[A-Za-z][\w:-]*|"
    r"[\{\}]\s*$|"
    r".+;\s*$"
    r")"
)
_PYTHON_HINT = re.compile(r"```(?:python|py)\b|(^|\n)\s*(def |class |import |from )", re.I)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def sha256_key(*parts: str) -> str:
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def item_id(source: str, row_key: str) -> str:
    """Hash of the source name and the row key."""
    return sha256_key(source, row_key)


def group_id_for(text: str) -> str:
    return sha256_key("group", normalize_text(text))


def is_code(text: str) -> bool:
    """A code fence, or at least three lines that match a code pattern."""
    if "```" in text:
        return True
    hits = sum(1 for line in text.splitlines() if _CODE_LINE.match(line))
    return hits >= 3


def is_python(text: str) -> bool:
    return bool(_PYTHON_HINT.search(text))


def messages_json(messages: list[dict]) -> str:
    return json.dumps(messages, ensure_ascii=False, separators=(",", ":"))


def user_text(messages: list[dict] | str) -> str:
    if isinstance(messages, str):
        messages = json.loads(messages)
    parts = [m["content"] for m in messages if m.get("role") == "user"]
    return "\n\n".join(parts)


def side_for_group(group_id: str) -> str:
    """Stable 50/50 assignment of an overlapping WildJailbreak group."""
    digest = hashlib.sha256(group_id.encode()).digest()
    return "train" if digest[0] % 2 == 0 else "tune"
