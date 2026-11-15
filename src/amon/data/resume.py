"""Checkpoints for stage 01. A later run continues from the last finished step."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from amon.guard import repo_root
from amon.provenance import sha256_file


def source_hash(*relative_paths: str) -> str:
    """SHA-256 of the named repo files, so a code change invalidates old checkpoints."""
    digest = hashlib.sha256()
    root = repo_root()
    for relative in relative_paths:
        digest.update(relative.encode())
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()


def fingerprint_texts(texts: list[str]) -> str:
    """Order-sensitive identity of a text list."""
    digest = hashlib.sha256()
    digest.update(str(len(texts)).encode())
    for text in texts:
        encoded = text.encode()
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        converted = {}
        for key, item in value.items():
            if isinstance(key, tuple):
                name = "|".join(str(part) for part in key)
            else:
                name = str(key)
            converted[name] = _jsonable(item)
        return converted
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class ResumeStore:
    """Files under `$AMON_DATA/checkpoints/01`. A checkpoint counts only when its fingerprint matches."""

    def __init__(self, data_root: Path):
        self.root = Path(data_root) / "checkpoints" / "01"
        self.root.mkdir(parents=True, exist_ok=True)

    def load_frame(self, name: str, fingerprint: dict) -> pd.DataFrame | None:
        meta = self._meta(name)
        path = self._file(name, ".parquet")
        if not self._fresh(meta, fingerprint, path):
            return None
        if sha256_file(path) != meta.get("parquet_sha256"):
            return None
        return pd.read_parquet(path)

    def save_frame(self, name: str, frame: pd.DataFrame, fingerprint: dict) -> str:
        path = self._file(name, ".parquet")
        temporary = path.with_name(path.name + ".tmp")
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
        digest = sha256_file(path)
        self._write_meta(
            name,
            {"fingerprint": fingerprint, "parquet_sha256": digest, "rows": int(len(frame))},
        )
        return digest

    def frame_sha(self, name: str) -> str | None:
        meta = self._meta(name)
        if meta is None:
            return None
        path = self._file(name, ".parquet")
        if not path.is_file() or sha256_file(path) != meta.get("parquet_sha256"):
            return None
        return meta["parquet_sha256"]

    def load_json(self, name: str, fingerprint: dict):
        meta = self._meta(name)
        path = self._file(name, ".json")
        if not self._fresh(meta, fingerprint, path):
            return None
        return json.loads(path.read_text())["payload"]

    def save_json(self, name: str, payload, fingerprint: dict) -> None:
        path = self._file(name, ".json")
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps({"payload": _jsonable(payload)}) + "\n")
        os.replace(temporary, path)
        self._write_meta(name, {"fingerprint": fingerprint})

    def save_selected(self, items: pd.DataFrame, notes: dict, fingerprint: dict) -> None:
        self.save_frame("selected_items", items, fingerprint)
        payload = {}
        frames = []
        for key, value in notes.items():
            if isinstance(value, pd.DataFrame):
                self.save_frame(f"selected_{key}", value, fingerprint)
                frames.append(key)
            else:
                payload[key] = value
        self.save_json("selected_notes", {"payload": payload, "frames": frames}, fingerprint)

    def load_selected(self, fingerprint: dict) -> tuple[pd.DataFrame, dict] | None:
        items = self.load_frame("selected_items", fingerprint)
        document = self.load_json("selected_notes", fingerprint)
        if items is None or document is None:
            return None
        notes = dict(document["payload"])
        for key in document["frames"]:
            frame = self.load_frame(f"selected_{key}", fingerprint)
            if frame is None:
                return None
            notes[key] = frame
        return items, notes

    def open_matrix(
        self,
        name: str,
        fingerprint: dict,
        shape: tuple[int, ...],
        dtype,
    ) -> tuple[np.ndarray, int]:
        """Return a memmap and how many rows are already filled. A fingerprint change starts over."""
        path = self._file(name, ".npy")
        meta = self._meta(name)
        if (
            meta
            and meta.get("fingerprint") == fingerprint
            and path.is_file()
            and tuple(meta.get("shape", [])) == tuple(shape)
        ):
            array = np.load(path, mmap_mode="r+")
            return array, min(int(meta.get("done", 0)), shape[0])
        if path.exists():
            path.unlink()
        array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
        self._write_meta(
            name,
            {
                "fingerprint": fingerprint,
                "shape": list(shape),
                "dtype": np.dtype(dtype).str,
                "done": 0,
            },
        )
        return array, 0

    def completed_matrix(self, name: str, fingerprint: dict) -> np.ndarray | None:
        meta = self._meta(name)
        path = self._file(name, ".npy")
        if not meta or meta.get("fingerprint") != fingerprint or not path.is_file():
            return None
        shape = meta.get("shape") or []
        if not shape or int(meta.get("done", 0)) != int(shape[0]):
            return None
        return np.load(path)

    def mark_progress(self, name: str, done: int) -> None:
        meta = self._meta(name)
        if meta is None:
            raise RuntimeError(f"no checkpoint metadata for {name}")
        meta["done"] = int(done)
        self._write_meta(name, meta)

    def _fresh(self, meta: dict | None, fingerprint: dict, path: Path) -> bool:
        return bool(meta and meta.get("fingerprint") == fingerprint and path.is_file())

    def _file(self, name: str, suffix: str) -> Path:
        return self.root / f"{name}{suffix}"

    def _meta_path(self, name: str) -> Path:
        return self.root / f"{name}.meta.json"

    def _meta(self, name: str) -> dict | None:
        path = self._meta_path(name)
        if not path.is_file():
            return None
        return json.loads(path.read_text())

    def _write_meta(self, name: str, meta: dict) -> None:
        path = self._meta_path(name)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(meta) + "\n")
        os.replace(temporary, path)
