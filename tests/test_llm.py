"""Log-odds from a last-position logit row, and the safe/unsafe token check."""

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
