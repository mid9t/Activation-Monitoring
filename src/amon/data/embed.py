"""Sentence-embedding near-duplicate vectors. The model id comes from models.yaml."""

from __future__ import annotations

import numpy as np


def embed_texts(texts: list[str], repo: str, revision: str) -> np.ndarray:
    """Mean-pooled last hidden state of all-MiniLM-L6-v2, L2-normalized."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    model = AutoModel.from_pretrained(repo, revision=revision)
    model.eval()
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(texts), 64):
            batch = [text[:2000] for text in texts[start : start + 64]]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            vectors.append(pooled.cpu().numpy())
    stacked = np.concatenate(vectors, axis=0) if vectors else np.zeros((0, 384))
    norms = np.linalg.norm(stacked, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return stacked / norms
