"""Llama 3 render and parse, including the shared assistant-header tail."""

import numpy as np

from amon.data.render import (
    llama3_prompt,
    parse_rendered_llama3,
    render_llama3,
    tokenize_rendered,
)
from amon.data.textutil import BOS, END, EOT, START

SPECIALS = {
    BOS: 128000,
    "<|end_of_text|>": 128001,
    START: 128006,
    END: 128007,
    EOT: 128009,
}


class TinyLlama3Tok:
    """Enough of a tokenizer to test the template contract without a gated download."""

    bos_token_id = SPECIALS[BOS]
    pad_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        text = llama3_prompt(messages, add_generation_prompt=add_generation_prompt)
        if tokenize:
            return self.encode(text, add_special_tokens=False)
        return text

    def encode(self, text, add_special_tokens=False):
        ids = []
        remaining = text
        while remaining:
            matched = False
            for token, token_id in SPECIALS.items():
                if remaining.startswith(token):
                    ids.append(token_id)
                    remaining = remaining[len(token) :]
                    matched = True
                    break
            if matched:
                continue
            if remaining.startswith("\n"):
                ids.append(10)
                remaining = remaining[1:]
                continue
            chunk = []
            while remaining and not any(remaining.startswith(token) for token in SPECIALS) and not remaining.startswith("\n"):
                chunk.append(remaining[0])
                remaining = remaining[1:]
            ids.append(1000 + (sum(map(ord, chunk)) % 500))
        return ids


def test_render_then_parse_is_the_identity_including_multiturn():
    tok = TinyLlama3Tok()
    rng = np.random.default_rng(5)
    for _ in range(50):
        turns = int(rng.integers(1, 4))
        messages = []
        for turn in range(turns):
            role = "user" if turn % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"turn {turn} body"})
        parsed = parse_rendered_llama3(render_llama3(messages, tok))
        assert parsed == messages


def test_every_prompt_has_one_bos_and_the_same_last_five_ids():
    tok = TinyLlama3Tok()
    conversations = [
        [{"role": "user", "content": "hello"}],
        [{"role": "user", "content": "a longer request about something else"}],
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"},
        ],
    ]
    encoded = [tokenize_rendered(render_llama3(messages, tok), tok) for messages in conversations]
    for ids in encoded:
        assert ids.count(SPECIALS[BOS]) == 1
    tails = [tuple(ids[-5:]) for ids in encoded]
    assert len(set(tails)) == 1
    assert len(tails[0]) == 5
