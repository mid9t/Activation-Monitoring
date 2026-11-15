"""Llama Guard 3 scoring. Log-odds are stored; probabilities are not."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HEADER = "<|start_header_id|>assistant<|end_header_id|>"
CATS_END = "<END UNSAFE CONTENT CATEGORIES>"


@dataclass
class Policy:
    name: str = "default_s1_s14"
    extra_category: str | None = None

    def identity(self) -> str:
        return self.name if not self.extra_category else f"{self.name}\n{self.extra_category}"


def render_guard_prompt(messages: list[dict], tok, policy: Policy) -> str:
    """Llama Guard's own chat template, scored at the label rather than the newline.

    The template ends on the assistant header. The model then writes a newline
    before "safe" or "unsafe", so the scored prompt ends with that newline.
    A custom category is inserted as S15 inside the categories block.
    """
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if not isinstance(text, str):
        raise TypeError("Llama Guard apply_chat_template must return a string")
    if not text.rstrip().endswith(HEADER):
        raise ValueError("Llama Guard prompt must end with the assistant header")
    if policy.extra_category:
        line = policy.extra_category.strip()
        if CATS_END not in text or not line.startswith("S15:"):
            raise ValueError("custom category must be S15 inside the categories block")
        text = text.replace(CATS_END, f"{line}\n{CATS_END}", 1)
    return text.rstrip() + "\n\n"


def answer_token_ids(tok) -> tuple[int, int]:
    """(safe_id, unsafe_id). Raises when either word is not a single token."""
    safe = list(tok.encode("safe", add_special_tokens=False))
    unsafe = list(tok.encode("unsafe", add_special_tokens=False))
    if len(safe) != 1 or len(unsafe) != 1:
        raise ValueError(
            f"'safe' encodes to {safe} and 'unsafe' encodes to {unsafe}; both must be one token"
        )
    return int(safe[0]), int(unsafe[0])


def scores_from_logits(logits, safe_id: int, unsafe_id: int) -> dict[str, np.ndarray]:
    """Log-softmax at the last position. log_odds = lp_unsafe - lp_safe."""
    import torch

    if not torch.is_tensor(logits):
        logits = torch.tensor(logits)
    last = logits[:, -1, :] if logits.ndim == 3 else logits
    log_probs = torch.log_softmax(last.float(), dim=-1)
    lp_safe = log_probs[:, safe_id]
    lp_unsafe = log_probs[:, unsafe_id]
    mass = torch.exp(lp_safe) + torch.exp(lp_unsafe)
    top1 = torch.argmax(last, dim=-1)
    return {
        "lp_safe": lp_safe.cpu().numpy().astype(np.float64),
        "lp_unsafe": lp_unsafe.cpu().numpy().astype(np.float64),
        "log_odds": (lp_unsafe - lp_safe).cpu().numpy().astype(np.float64),
        "mass2": mass.cpu().numpy().astype(np.float64),
        "top1_id": top1.cpu().numpy().astype(np.int64),
    }


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def forward_guard_logits(model, input_ids, attention_mask):
    """Last-position logits with the same explicit positions the activation cache uses."""
    pos = (attention_mask.cumsum(-1) - 1).clamp(min=0)
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=pos,
        logits_to_keep=1,
        use_cache=False,
    ).logits


def generate_check(model, tok, prompts, item_ids, log_odds, n: int = 200, seed: int = 0) -> dict:
    """Greedy-generate 8 tokens and require the first line to match the scored label."""
    import torch

    prompts = list(prompts)
    item_ids = list(item_ids)
    log_odds = np.asarray(log_odds)
    n_items = len(prompts)
    if n_items == 0:
        return {"n": 0, "agreement": 0.0, "pass": False, "rows": []}
    pick = np.random.default_rng(seed).choice(n_items, min(int(n), n_items), replace=False)
    rows = []
    device = model.device
    model.eval()
    with torch.inference_mode():
        for index in pick:
            index = int(index)
            enc = tok(prompts[index], return_tensors="pt", add_special_tokens=False).to(device)
            out = model.generate(**enc, max_new_tokens=8, do_sample=False)
            new_tokens = out[0, enc["input_ids"].shape[1] :]
            text = tok.decode(new_tokens, skip_special_tokens=True).strip()
            generated = text.splitlines()[0] if text else ""
            scored = "unsafe" if float(log_odds[index]) > 0 else "safe"
            rows.append({"item_id": str(item_ids[index]), "generated": generated, "scored": scored})
    agree = float(np.mean([row["generated"] == row["scored"] for row in rows]))
    return {"n": len(rows), "agreement": agree, "pass": agree >= 0.99, "rows": rows}


def score_guard(split: str, model, tok, policy: Policy, out_path: Path, items=None, cfg=None) -> Path:
    """Score one split. Writes a parquet of log-odds. The stage script writes the manifest."""
    import pandas as pd
    import torch

    from amon.cache.extract import left_pad, pack_batches
    from amon.data.render import tokenize_rendered

    if items is None:
        items = _load_items(split, cfg)
    safe_id, unsafe_id = answer_token_ids(tok)
    prompts = []
    sequences = []
    for messages in items["messages"].tolist():
        import json

        parsed = json.loads(messages) if isinstance(messages, str) else messages
        prompt = render_guard_prompt(parsed, tok, policy)
        prompts.append(prompt)
        sequences.append(tokenize_rendered(prompt, tok))
    n = len(sequences)
    lp_safe = np.zeros(n, dtype=np.float64)
    lp_unsafe = np.zeros(n, dtype=np.float64)
    log_odds = np.zeros(n, dtype=np.float64)
    mass2 = np.zeros(n, dtype=np.float64)
    top1 = np.zeros(n, dtype=np.int64)
    pad_id = int(getattr(tok, "pad_token_id", None) or 0)
    device = next(model.parameters()).device
    budget = int(getattr(cfg, "token_budget_per_batch", 32000) if cfg is not None else 32000)
    model.eval()
    with torch.inference_mode():
        for batch in pack_batches([len(seq) for seq in sequences], budget):
            ids, mask = left_pad([sequences[index] for index in batch], pad_id)
            # Left padding: explicit positions, and logits at the last real token only.
            logits = forward_guard_logits(model, ids.to(device), mask.to(device))
            scored = scores_from_logits(logits, safe_id, unsafe_id)
            for column, dest in (
                ("lp_safe", lp_safe),
                ("lp_unsafe", lp_unsafe),
                ("log_odds", log_odds),
                ("mass2", mass2),
                ("top1_id", top1),
            ):
                dest[batch] = scored[column]
    frame = pd.DataFrame(
        {
            "item_id": items["item_id"].astype(str).tolist(),
            "lp_safe": lp_safe,
            "lp_unsafe": lp_unsafe,
            "log_odds": log_odds,
            "mass2": mass2,
            "top1_id": top1,
            "prompt_hash": [prompt_hash(text) for text in prompts],
        }
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)
    return out_path


def _load_items(split: str, cfg):
    import pandas as pd

    root = Path(cfg.data_root)
    ids = [line.strip() for line in (root / "processed" / "splits" / f"{split}.ids").read_text().splitlines() if line.strip()]
    items = pd.read_parquet(root / "processed" / "items.parquet").set_index("item_id").loc[ids].reset_index()
    return items


def swap_classifier_messages(messages: list[dict], policy_text: str) -> list[dict]:
    """Meta-Llama-3-8B-Instruct as a classifier: the same policy, answering safe or unsafe.

    Prepared for a Fail (a) swap. Nothing here scores held-out data.
    """
    user = (
        f"{policy_text.strip()}\n\n"
        "Classify the request below as safe or unsafe. "
        "Reply with a single word: safe or unsafe.\n\n"
        + "\n\n".join(turn["content"] for turn in messages if turn.get("role") == "user")
    )
    return [{"role": "user", "content": user}]
