"""One forward pass per item. Left padding, explicit positions, resumable memmaps."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from amon.data.render import render_llama3, tokenize_rendered


def layer_name(layer: int) -> str:
    return f"L{layer:02d}"


def left_pad(sequences: list[list[int]], pad_id: int):
    """Left-pad to the longest sequence in the batch. Returns int64 tensors."""
    import torch

    width = max(len(seq) for seq in sequences)
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.long)
    for row, seq in enumerate(sequences):
        if not seq:
            continue
        ids[row, width - len(seq) :] = torch.tensor(seq, dtype=torch.long)
        mask[row, width - len(seq) :] = 1
    return ids, mask


def pack_batches(lengths: list[int], budget: int) -> list[list[int]]:
    """Greedy packs after the caller sorts by length. Cost is batch size times the padded width."""
    order = sorted(range(len(lengths)), key=lambda index: (lengths[index], index))
    batches: list[list[int]] = []
    current: list[int] = []
    current_max = 0
    for index in order:
        length = max(int(lengths[index]), 1)
        new_max = max(current_max, length)
        new_cost = (len(current) + 1) * new_max
        if current and new_cost > budget:
            batches.append(current)
            current = [index]
            current_max = length
        else:
            current.append(index)
            current_max = new_max
    if current:
        batches.append(current)
    return batches


def forward_hidden(model, input_ids, attention_mask, layers: list[int]) -> dict[int, tuple]:
    """Last-token and mean residual streams. Positions match an unpadded run.

    L means hidden_states[L], the output of block L. Index 0 is the embedding.
    The mean skips padding and the BOS token, whose position is 0.
    """
    import torch

    pos = (attention_mask.cumsum(-1) - 1).clamp(min=0)
    base = model.model if hasattr(model, "model") else model
    out = base(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=pos,
        output_hidden_states=True,
        use_cache=False,
    )
    keep = attention_mask * (pos > 0)
    keep_f = keep.to(dtype=out.hidden_states[0].dtype)
    denom = keep_f.sum(1, keepdim=True).clamp(min=1)
    result = {}
    for layer in layers:
        hidden = out.hidden_states[layer]
        last = hidden[:, -1]
        mean = (hidden * keep_f.unsqueeze(-1)).sum(1) / denom
        result[layer] = (last, mean)
    return result


def _to_storage(values, dtype_store: str):
    """Cast to the storage dtype. Non-finite values stop the run. Overflow stores float32."""
    import torch

    values = values.detach().float().cpu()
    if not torch.isfinite(values).all():
        raise RuntimeError("non-finite activation; refusing to store the cache")
    if dtype_store == "float16" and torch.max(values.abs()) >= 65504:
        return values.numpy().astype(np.float32), "float32"
    if dtype_store == "float16":
        return values.numpy().astype(np.float16), "float16"
    return values.numpy().astype(np.float32), "float32"


def cache_split(split: str, model, tok, cfg, out_dir: Path) -> Path:
    """Write last_L*.npy, mean_L*.npy, meta.parquet, and manifest.json. Resumable.

    Items and their id order come from cfg.items (a dataframe) or, if absent,
    from cfg.data_root / processed. Rows are written in .ids order.
    """
    import pandas as pd
    import torch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = _items_for_split(split, cfg)
    ids = items["item_id"].astype(str).tolist()
    (out_dir / "split.ids").write_text("".join(f"{item}\n" for item in ids))
    sequences = []
    for messages in items["messages"].tolist():
        parsed = json.loads(messages) if isinstance(messages, str) else messages
        sequences.append(tokenize_rendered(render_llama3(parsed, tok), tok))
    lengths = [len(seq) for seq in sequences]
    layers = [int(layer) for layer in cfg.layers]
    hidden = int(model.config.hidden_size)
    dtype_store = str(cfg.dtype_store)
    np_dtype = np.float16 if dtype_store == "float16" else np.float32
    stores = {}
    overflow = {}
    for layer in layers:
        for position in ("last", "mean"):
            path = out_dir / f"{position}_{layer_name(layer)}.npy"
            if path.exists():
                stores[(position, layer)] = np.load(path, mmap_mode="r+")
            else:
                stores[(position, layer)] = np.lib.format.open_memmap(
                    path, mode="w+", dtype=np_dtype, shape=(len(ids), hidden)
                )
    progress_path = out_dir / "rows_done.json"
    done = set(json.loads(progress_path.read_text())) if progress_path.exists() else set()
    pad_id = int(getattr(tok, "pad_token_id", None) or 0)
    device = next(model.parameters()).device
    model.eval()
    batches = pack_batches(lengths, int(cfg.token_budget_per_batch))
    norms = {layer: np.zeros(len(ids), dtype=np.float32) for layer in layers}
    if (out_dir / "meta.parquet").exists() and done:
        previous = pd.read_parquet(out_dir / "meta.parquet")
        for layer in layers:
            column = f"norm_{layer_name(layer)}"
            if column in previous.columns:
                norms[layer][:] = previous[column].to_numpy()
    with torch.inference_mode():
        for batch in batches:
            pending = [index for index in batch if index not in done]
            if not pending:
                continue
            ids_tensor, mask = left_pad([sequences[index] for index in pending], pad_id)
            ids_tensor = ids_tensor.to(device)
            mask = mask.to(device)
            hidden_states = forward_hidden(model, ids_tensor, mask, layers)
            for layer, (last, mean) in hidden_states.items():
                last_np, last_dtype = _to_storage(last, dtype_store)
                mean_np, mean_dtype = _to_storage(mean, dtype_store)
                overflow[layer_name(layer)] = last_dtype
                stores[("last", layer)] = _store_as(
                    stores[("last", layer)], last_dtype, out_dir, "last", layer, len(ids), hidden
                )
                stores[("mean", layer)] = _store_as(
                    stores[("mean", layer)], mean_dtype, out_dir, "mean", layer, len(ids), hidden
                )
                stores[("last", layer)][pending] = last_np
                stores[("mean", layer)][pending] = mean_np
                norms[layer][pending] = np.linalg.norm(last_np.astype(np.float32), axis=1)
            done.update(pending)
            progress_path.write_text(json.dumps(sorted(done)))
            for array in stores.values():
                array.flush()
    meta = pd.DataFrame({"item_id": ids, "token_count": lengths})
    for layer in layers:
        meta[f"norm_{layer_name(layer)}"] = norms[layer]
    meta.to_parquet(out_dir / "meta.parquet", index=False)
    from amon.provenance import write_manifest

    manifest_cfg = {
        "seed": getattr(cfg, "seed", None),
        "dtype": dtype_store,
        "device": str(device),
        "model_sha": getattr(cfg, "model_sha", None),
        "layers": layers,
        "overflow_dtype": overflow,
        "split": split,
    }
    inputs = [Path(path) for path in getattr(cfg, "inputs", [])]
    write_manifest(out_dir, inputs, manifest_cfg)
    return out_dir


def _store_as(array, dtype_name: str, out_dir: Path, position: str, layer: int, n: int, hidden: int):
    """Return a memmap of the requested dtype, rewriting the file if float16 overflowed."""
    wanted = np.float32 if dtype_name == "float32" else np.float16
    if array.dtype == wanted:
        return array
    path = out_dir / f"{position}_{layer_name(layer)}.npy"
    del array
    fresh = np.lib.format.open_memmap(path, mode="w+", dtype=wanted, shape=(n, hidden))
    return fresh


def _items_for_split(split: str, cfg):
    if getattr(cfg, "items", None) is not None:
        items = cfg.items
        if "role" in items.columns:
            items = items.loc[items["role"] == split]
        return items.reset_index(drop=True)
    import pandas as pd

    root = Path(cfg.data_root)
    ids_path = root / "processed" / "splits" / f"{split}.ids"
    ids = [line.strip() for line in ids_path.read_text().splitlines() if line.strip()]
    items = pd.read_parquet(root / "processed" / "items.parquet")
    items = items.set_index("item_id").loc[ids].reset_index()
    return items


def load_rows(split_dir: Path, position: str, layer: int) -> np.ndarray:
    return np.load(split_dir / f"{position}_{layer_name(layer)}.npy", mmap_mode="r")
