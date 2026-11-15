"""Sentence-embedding near-duplicate vectors. The model id comes from models.yaml."""

from __future__ import annotations

import numpy as np

from amon.data.resume import fingerprint_texts, source_hash


def embed_texts(
    texts: list[str],
    repo: str,
    revision: str,
    resume=None,
    resume_name: str = "embed",
) -> np.ndarray:
    """Mean-pooled last hidden state of all-MiniLM-L6-v2, L2-normalized.

    With a resume store, each batch is written into a matrix so a killed run
    continues at the next batch instead of the first text.
    """
    if not texts:
        return np.zeros((0, 384))
    fingerprint = {
        "texts": fingerprint_texts(texts),
        "repo": repo,
        "revision": revision,
        "code": source_hash("src/amon/data/embed.py"),
    }
    if resume is not None:
        done = resume.completed_matrix(resume_name, fingerprint)
        if done is not None and len(done) == len(texts):
            print(f"resume {resume_name}: {len(texts)} vectors", flush=True)
            return _normalize(done)

    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    model = AutoModel.from_pretrained(repo, revision=revision)
    model.eval()
    matrix = None
    start = 0
    if resume is not None:
        partial = _partial_rows(resume, resume_name, fingerprint)
        if partial is not None:
            matrix, start = partial
            print(f"resume {resume_name} at {start}/{len(texts)}", flush=True)
    chunks = []
    with torch.inference_mode():
        for batch_start in range(start, len(texts), 64):
            batch = [text[:2000] for text in texts[batch_start : batch_start + 64]]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            pooled = pooled.cpu().numpy()
            if resume is None:
                chunks.append(pooled)
                continue
            if matrix is None:
                matrix, _ = resume.open_matrix(
                    resume_name,
                    fingerprint,
                    (len(texts), pooled.shape[1]),
                    pooled.dtype,
                )
            matrix[batch_start : batch_start + len(pooled)] = pooled
            matrix.flush()
            resume.mark_progress(resume_name, batch_start + len(pooled))
    stacked = np.concatenate(chunks, axis=0) if resume is None else np.asarray(matrix)
    return _normalize(stacked)


def _partial_rows(resume, name: str, fingerprint: dict):
    """Open an unfinished matrix. A finished one is handled by completed_matrix."""
    meta = resume._meta(name)
    if not meta or meta.get("fingerprint") != fingerprint:
        return None
    shape = meta.get("shape") or []
    if len(shape) != 2 or int(meta.get("done", 0)) >= int(shape[0]):
        return None
    return resume.open_matrix(name, fingerprint, tuple(shape), np.dtype(meta["dtype"]))


def _normalize(stacked: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(stacked, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return stacked / norms
