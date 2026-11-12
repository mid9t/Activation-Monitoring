"""Llama 3 chat rendering and its inverse, for Bailey's pre-rendered prompts."""

from __future__ import annotations

import re

from amon.data.textutil import BOS, END, EOT, START

_BLOCK = re.compile(
    r"<\|start_header_id\|>([A-Za-z0-9_]+)<\|end_header_id\|>\n\n(.*?)(?=<\|eot_id\|>|<\|start_header_id\|>|\Z)",
    re.DOTALL,
)
_GENERATION = f"{START}assistant{END}\n\n"


def llama3_prompt(messages: list[dict], add_generation_prompt: bool = True) -> str:
    """The Llama 3 Instruct layout: one BOS, trimmed turns, optional assistant header.

    No system turn is added. A system turn already in `messages` is kept, because
    dropping it would change the text the tokenizer was given.
    """
    parts = [BOS]
    for message in messages:
        content = str(message["content"]).strip()
        role = str(message["role"])
        parts.append(f"{START}{role}{END}\n\n{content}{EOT}")
    if add_generation_prompt:
        parts.append(_GENERATION)
    return "".join(parts)


def render_llama3(messages: list[dict], tok) -> str:
    """Chat template with add_generation_prompt=True and no added system prompt."""
    rendered = tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    if not isinstance(rendered, str):
        raise TypeError("apply_chat_template(..., tokenize=False) must return a string")
    return rendered


def parse_rendered_llama3(text: str) -> list[dict]:
    """Inverse of the Llama 3 layout. The trailing generation header is not a message."""
    body = text
    if body.startswith(BOS):
        body = body[len(BOS) :]
    if body.endswith(_GENERATION):
        body = body[: -len(_GENERATION)]
    messages = []
    for match in _BLOCK.finditer(body):
        role = match.group(1)
        content = match.group(2)
        # A block closed by eot is a real turn. A block that runs into the next
        # header, or into the end without eot, is an unclosed generation prompt.
        end = match.end()
        closed = body[end : end + len(EOT)] == EOT or content.endswith(EOT)
        if content.endswith(EOT):
            content = content[: -len(EOT)]
        if not closed and role == "assistant" and content == "":
            continue
        if not closed:
            continue
        messages.append({"role": role, "content": content})
    return messages


def tokenize_rendered(text: str, tok) -> list[int]:
    """Tokenize a rendered prompt. The string already contains BOS, so specials stay off."""
    ids = tok.encode(text, add_special_tokens=False)
    if hasattr(ids, "ids"):
        ids = ids.ids
    return list(ids)


def check_token_batch(encoded: list[list[int]], bos_id: int) -> None:
    """Every row has exactly one BOS, and every row ends with the same five ids."""
    if not encoded:
        raise ValueError("no tokenized prompts")
    last = None
    for ids in encoded:
        if ids.count(bos_id) != 1:
            raise ValueError("a rendered prompt does not contain exactly one <|begin_of_text|>")
        tail = tuple(ids[-5:])
        if len(tail) != 5:
            raise ValueError("a rendered prompt is shorter than five tokens")
        if last is None:
            last = tail
        elif tail != last:
            raise ValueError("rendered prompts do not share the same last five token ids")
