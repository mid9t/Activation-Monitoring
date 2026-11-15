"""Log-odds from a last-position logit row, and the safe/unsafe token check."""

import pytest
import torch

from amon.llm import Policy, answer_token_ids, render_guard_prompt, scores_from_logits, swap_classifier_messages


class Tok:
    def encode(self, text, add_special_tokens=False):
        table = {"safe": [3], "unsafe": [4]}
        if text not in table:
            raise KeyError(text)
        return table[text]

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        body = messages[0]["content"]
        header = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return f"<|begin_of_text|>user\n{body}\n{header}"


def test_scores_are_log_odds_not_probabilities():
    logits = torch.full((2, 1, 6), -10.0)
    logits[0, 0, 3] = 0.0
    logits[0, 0, 4] = 2.0
    logits[1, 0, 3] = 2.0
    logits[1, 0, 4] = 0.0
    scored = scores_from_logits(logits, safe_id=3, unsafe_id=4)
    assert scored["log_odds"][0] > 0
    assert scored["log_odds"][1] < 0
    assert scored["log_odds"][0] == scored["lp_unsafe"][0] - scored["lp_safe"][0]
    assert scored["mass2"][0] < 1.0


def test_answer_tokens_must_be_single_tokens():
    assert answer_token_ids(Tok()) == (3, 4)


def test_guard_prompt_ends_with_the_assistant_header_and_swap_uses_the_policy():
    prompt = render_guard_prompt([{"role": "user", "content": "hello"}], Tok(), Policy())
    assert prompt.rstrip().endswith("<|end_header_id|>") or "assistant<|end_header_id|>" in prompt[-80:]
    swapped = swap_classifier_messages([{"role": "user", "content": "hello"}], "policy text")
    assert "policy text" in swapped[0]["content"]
    assert "safe or unsafe" in swapped[0]["content"]


HEADER = "<|start_header_id|>assistant<|end_header_id|>"
CATS_END = "<END UNSAFE CONTENT CATEGORIES>"


class PieceTok:
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        body = messages[0]["content"]
        return (
            "<|begin_of_text|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{body}<|eot_id|>"
            "S1: Violent crimes.\n"
            f"{CATS_END}\n"
            f"{HEADER}"
        )

    def encode(self, text, add_special_tokens=False):
        if text == "\n\n":
            return [99]
        if text == "safe":
            return [3]
        if text == "unsafe":
            return [4]
        if text == HEADER:
            return [11, 12, 13]
        if text.endswith("\n\n") and text[:-2].endswith(HEADER):
            return [8, 11, 12, 13, 99]
        raise AssertionError(repr(text[-80:]))


def test_rendered_prompt_ends_with_the_header_and_one_newline_token():
    tok = PieceTok()
    prompt = render_guard_prompt([{"role": "user", "content": "hello"}], tok, Policy())
    assert prompt.endswith(HEADER + "\n\n")
    assert tok.encode(prompt, add_special_tokens=False)[-4:] == [11, 12, 13, 99]
    assert tok.encode("\n\n", add_special_tokens=False) == [99]
    assert answer_token_ids(tok) == (3, 4)


def test_custom_category_is_inserted_as_s15():
    tok = PieceTok()
    prompt = render_guard_prompt(
        [{"role": "user", "content": "hello"}],
        tok,
        Policy(extra_category="S15: Name.\nA description"),
    )
    assert prompt.index("S15: Name.") < prompt.index(CATS_END)
    assert prompt.endswith(HEADER + "\n\n")
    with pytest.raises(ValueError, match="S15"):
        render_guard_prompt([{"role": "user", "content": "hello"}], tok, Policy(extra_category="not a category"))


def test_last_position_logits_match_the_full_row_and_single_items():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from amon.cache.extract import left_pad
    from amon.llm.llamaguard import forward_guard_logits

    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        max_position_embeddings=64,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config)
    model.eval()
    sequences = [[1, 2, 3, 4, 5], [6, 7, 8], [9, 10, 11, 12]]
    ids, mask = left_pad(sequences, pad_id=0)
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    with torch.inference_mode():
        full = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False).logits[:, -1, :]
        kept = forward_guard_logits(model, ids, mask)
        got = kept[:, -1, :] if kept.ndim == 3 else kept
        assert torch.allclose(got.float(), full.float(), rtol=1e-4, atol=1e-4)
        for row, sequence in enumerate(sequences):
            one_ids, one_mask = left_pad([sequence], pad_id=0)
            single = forward_guard_logits(model, one_ids, one_mask)
            single = single[:, -1, :] if single.ndim == 3 else single
            cosine = torch.nn.functional.cosine_similarity(got[row].float(), single[0].float(), dim=0)
            assert float(cosine) >= 0.999


def test_generate_check_compares_the_first_line_with_the_scored_label():
    import torch

    from amon.llm import generate_check

    class Enc(dict):
        def to(self, _device):
            return self

    class Tok:
        def __call__(self, _text, return_tensors="pt", add_special_tokens=False):
            return Enc({"input_ids": torch.ones(1, 3, dtype=torch.long)})

        def decode(self, _ids, skip_special_tokens=True):
            return "unsafe\nS2"

    class Model:
        device = "cpu"

        def eval(self):
            return self

        def generate(self, **kwargs):
            ids = kwargs["input_ids"]
            return torch.cat([ids, torch.tensor([[4]])], dim=1)

    agreed = generate_check(Model(), Tok(), ["prompt"], ["item"], [1.5], n=1, seed=0)
    assert agreed["pass"] is True
    assert agreed["rows"][0]["generated"] == "unsafe"
    disagreed = generate_check(Model(), Tok(), ["prompt"], ["item"], [-1.5], n=1, seed=0)
    assert disagreed["pass"] is False
    assert disagreed["rows"][0]["scored"] == "safe"
