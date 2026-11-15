"""One forward pass per item. Left padding, explicit positions, shard checkpoints."""

from __future__ import annotations

import json
import os
import shutil
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


# Float16 saturates at 65,504. Massive activations sit at BOS, which is not stored.
# A peak this close to the limit is a bug, so the run stops instead of rewriting the file.
FLOAT16_PEAK = 60000
SHARD_ROWS = 2000


def _to_storage(values, dtype_store: str):
    """Cast to the storage dtype. Non-finite values and float16 overflow stop the run."""
    import torch

    values = values.detach().float().cpu()
    if not torch.isfinite(values).all():
        raise RuntimeError("non-finite activation; refusing to store the cache")
    peak = float(values.abs().max()) if values.numel() else 0.0
    if dtype_store == "float16" and peak >= FLOAT16_PEAK:
        raise RuntimeError(f"peak {peak:.0f} near the float16 limit; set dtype_store: float32")
    if dtype_store == "float16":
        return values.numpy().astype(np.float16), peak
    return values.numpy().astype(np.float32), peak


def _scratch_for(split: str, cfg) -> Path:
    custom = getattr(cfg, "scratch_dir", None)
    if custom:
        return Path(custom)
    return Path(os.environ.get("AMON_CACHE_TMP", "/tmp")) / "cache" / split


def _iter_shards(shard_dir: Path) -> list[Path]:
    if not shard_dir.is_dir():
        return []
    found = []
    for path in shard_dir.iterdir():
        if path.is_file() and path.name.startswith("part-") and path.suffix == ".npz" and ".tmp" not in path.name:
            found.append(path)
    return sorted(found)


def _next_serial(shards: list[Path]) -> int:
    serials = [int(path.stem.split("-")[1]) for path in shards]
    return max(serials) + 1 if serials else 0


def _read_ids(out_dir: Path) -> list[str]:
    return [line.strip() for line in (out_dir / "split.ids").read_text().splitlines() if line.strip()]


def _finals_complete(out_dir: Path, layers: list[int], n: int, hidden: int, np_dtype) -> bool:
    if not (out_dir / "meta.parquet").is_file() or not (out_dir / "split.ids").is_file():
        return False
    for layer in layers:
        for position in ("last", "mean"):
            path = out_dir / f"{position}_{layer_name(layer)}.npy"
            if not path.is_file():
                return False
            array = np.load(path, mmap_mode="r")
            if array.shape != (n, hidden) or array.dtype != np_dtype:
                return False
    return True


def _clear_partial(out_dir: Path) -> None:
    for path in out_dir.glob("*.npy"):
        path.unlink()
    meta = out_dir / "meta.parquet"
    if meta.exists():
        meta.unlink()
    peaks = out_dir / "peaks.json"
    if peaks.exists():
        peaks.unlink()
    shutil.rmtree(out_dir / "shards", ignore_errors=True)


def _open_memmaps(directory: Path, layers: list[int], n: int, hidden: int, np_dtype) -> dict:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    stores = {}
    for layer in layers:
        for position in ("last", "mean"):
            path = directory / f"{position}_{layer_name(layer)}.npy"
            stores[(position, layer)] = np.lib.format.open_memmap(
                path, mode="w+", dtype=np_dtype, shape=(n, hidden)
            )
    return stores


def _apply_shards(stores, shard_dir: Path, layers: list[int]) -> tuple[set[int], dict]:
    done: set[int] = set()
    peaks: dict[tuple[str, int], float] = {}
    for path in _iter_shards(shard_dir):
        with np.load(path) as data:
            indices = np.asarray(data["indices"], dtype=np.int64)
            for key in data.files:
                if key == "indices" or key.startswith("peak__"):
                    continue
                position, layer_text = key.split("_L")
                layer = int(layer_text)
                stores[(position, layer)][indices] = np.asarray(data[key])
            for layer in layers:
                for position in ("last", "mean"):
                    peak_key = f"peak__{position}_{layer_name(layer)}"
                    if peak_key in data.files:
                        peaks[(position, layer)] = max(peaks.get((position, layer), 0.0), float(data[peak_key]))
            done.update(int(index) for index in indices)
    return done, peaks


def _write_shard(shard_dir: Path, serial: int, indices: list[int], chunks: dict, peaks: dict) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    final = shard_dir / f"part-{serial:06d}.npz"
    tmp = shard_dir / f".part-{serial:06d}.tmp"
    payload = {"indices": np.asarray(indices, dtype=np.int64)}
    payload.update(chunks)
    for (position, layer), peak in peaks.items():
        payload[f"peak__{position}_{layer_name(layer)}"] = np.float64(peak)
    with tmp.open("wb") as handle:
        np.savez(handle, **payload)
    os.replace(tmp, final)


def _peaks_payload(peaks: dict) -> dict:
    payload: dict[str, dict[str, float]] = {}
    for (position, layer), peak in peaks.items():
        payload.setdefault(layer_name(layer), {})[position] = float(peak)
    return payload


def _load_peaks(out_dir: Path) -> dict:
    path = out_dir / "peaks.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def cache_split(split: str, model, tok, cfg, out_dir: Path) -> dict:
    """Write last_L*.npy and mean_L*.npy. Checkpoints are immutable shards.

    Local memmaps live on the container scratch disk. Every shard_rows rows, an
    npz of those row ids and arrays is renamed into place on `out_dir`. A
    preemption loses at most the open shard. The manifest is written by stage 02
    after verify_cache passes.

    Items and their id order come from cfg.items, or from cfg.data_root / processed.
    """
    import pandas as pd
    import torch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = _items_for_split(split, cfg)
    ids = items["item_id"].astype(str).tolist()
    layers = [int(layer) for layer in cfg.layers]
    hidden = int(model.config.hidden_size)
    dtype_store = str(cfg.dtype_store)
    np_dtype = np.float16 if dtype_store == "float16" else np.float32
    shard_dir = out_dir / "shards"
    if (out_dir / "split.ids").is_file() and _read_ids(out_dir) != ids:
        _clear_partial(out_dir)
    if (
        (out_dir / "split.ids").is_file()
        and _read_ids(out_dir) == ids
        and not _iter_shards(shard_dir)
        and _finals_complete(out_dir, layers, len(ids), hidden, np_dtype)
    ):
        return {"complete": True, "peaks": _load_peaks(out_dir), "out_dir": out_dir, "scratch": None}

    (out_dir / "split.ids").write_text("".join(f"{item}\n" for item in ids))
    sequences = []
    for messages in items["messages"].tolist():
        parsed = json.loads(messages) if isinstance(messages, str) else messages
        sequences.append(tokenize_rendered(render_llama3(parsed, tok), tok))
    lengths = [len(seq) for seq in sequences]
    scratch = _scratch_for(split, cfg)
    stores = _open_memmaps(scratch, layers, len(ids), hidden, np_dtype)
    done, peaks = _apply_shards(stores, shard_dir, layers)
    pad_id = int(getattr(tok, "pad_token_id", None) or 0)
    device = next(model.parameters()).device
    model.eval()
    batches = pack_batches(lengths, int(cfg.token_budget_per_batch))
    shard_rows = int(getattr(cfg, "shard_rows", SHARD_ROWS))
    stop_after = getattr(cfg, "stop_after_batches", None)
    serial = _next_serial(_iter_shards(shard_dir))
    buffer_idx: list[int] = []
    buffer = {(position, layer): [] for layer in layers for position in ("last", "mean")}

    def flush() -> None:
        nonlocal serial
        if not buffer_idx:
            return
        chunks = {}
        for layer in layers:
            for position in ("last", "mean"):
                rows = buffer[(position, layer)]
                chunks[f"{position}_{layer_name(layer)}"] = np.stack(rows, axis=0)
                buffer[(position, layer)] = []
        _write_shard(shard_dir, serial, list(buffer_idx), chunks, peaks)
        buffer_idx.clear()
        serial += 1

    computed = 0
    with torch.inference_mode():
        for batch in batches:
            pending = [index for index in batch if index not in done]
            if not pending:
                continue
            ids_tensor, mask = left_pad([sequences[index] for index in pending], pad_id)
            hidden_states = forward_hidden(model, ids_tensor.to(device), mask.to(device), layers)
            stored = {}
            batch_peaks = {}
            for layer, (last, mean) in hidden_states.items():
                last_np, last_peak = _to_storage(last, dtype_store)
                mean_np, mean_peak = _to_storage(mean, dtype_store)
                stored[("last", layer)] = last_np
                stored[("mean", layer)] = mean_np
                batch_peaks[("last", layer)] = last_peak
                batch_peaks[("mean", layer)] = mean_peak
            for key, array in stored.items():
                stores[key][list(pending)] = array
                for row, index in enumerate(pending):
                    buffer[key].append(np.asarray(array[row]))
            for key, peak in batch_peaks.items():
                peaks[key] = max(peaks.get(key, 0.0), peak)
            buffer_idx.extend(pending)
            done.update(pending)
            computed += 1
            if len(buffer_idx) >= shard_rows:
                flush()
            if stop_after is not None and computed >= int(stop_after):
                flush()
                for array in stores.values():
                    array.flush()
                return {
                    "complete": False,
                    "peaks": _peaks_payload(peaks),
                    "out_dir": out_dir,
                    "scratch": scratch,
                }
    flush()
    for key in list(stores):
        stores.pop(key).flush()
    for layer in layers:
        for position in ("last", "mean"):
            name = f"{position}_{layer_name(layer)}.npy"
            shutil.copyfile(scratch / name, out_dir / name)
    import gc

    gc.collect()
    norms = {}
    copied = {}
    for layer in layers:
        copied[layer] = np.load(out_dir / f"last_{layer_name(layer)}.npy", mmap_mode="r")
        norms[layer] = np.linalg.norm(np.asarray(copied[layer], dtype=np.float32), axis=1)
    meta = pd.DataFrame({"item_id": ids, "token_count": lengths})
    for layer in layers:
        meta[f"norm_{layer_name(layer)}"] = norms[layer]
    meta.to_parquet(out_dir / "meta.parquet", index=False)
    peak_payload = _peaks_payload(peaks)
    (out_dir / "peaks.json").write_text(json.dumps(peak_payload, indent=2, sort_keys=True) + "\n")
    copied.clear()
    gc.collect()
    shutil.rmtree(scratch, ignore_errors=True)
    return {"complete": True, "peaks": peak_payload, "out_dir": out_dir, "scratch": None}


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
