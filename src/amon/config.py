"""Load the yaml configs. Environment variables are expanded from paths.yaml only."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

_ENV = re.compile(r"\$\{([A-Z0-9_]+)\}")


def repo_root() -> Path:
    from amon.guard import repo_root as root

    return root()


def configs_dir() -> Path:
    return repo_root() / "configs"


def _expand(value):
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in os.environ or os.environ[name] == "":
                raise EnvironmentError(
                    f"config references ${{{name}}} but that variable is unset"
                )
            return os.environ[name]

        return _ENV.sub(replace, value)
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return value


def load_yaml(path: Path) -> dict:
    data = yaml.safe_load(Path(path).read_text())
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"{path} must be a mapping")
    return data


def load_config(name: str) -> dict:
    path = configs_dir() / name
    data = load_yaml(path)
    if name == "paths.yaml":
        return _expand(data)
    return data


@dataclass(frozen=True)
class Paths:
    data_root: Path
    hf_home: Path
    device: str

    @classmethod
    def load(cls) -> "Paths":
        raw = load_config("paths.yaml")
        hf_home = raw.get("hf_home") or str(Path(raw["data_root"]) / "hf")
        device = raw.get("device") or os.environ.get("AMON_DEVICE", "cpu")
        return cls(data_root=Path(raw["data_root"]), hf_home=Path(hf_home), device=device)


def require_paths() -> Paths:
    """Fail with the variable name when the data root has not been configured."""
    return Paths.load()


def fill_tbd(path: Path, updates: dict[str, object]) -> None:
    """Replace `key: TBD` once. A value that is no longer TBD is left alone and reported."""
    import math
    import re

    text = Path(path).read_text()
    for key, value in updates.items():
        pattern = rf"({re.escape(str(key))}:\s*)TBD\b"
        if not re.search(pattern, text):
            raise RuntimeError(f"{key} in {path} is no longer TBD; refusing to change a frozen value")
        if isinstance(value, float) and math.isinf(value):
            scalar = ".inf" if value > 0 else "-.inf"
        elif isinstance(value, float):
            scalar = repr(value)
        else:
            scalar = str(value)
        text = re.sub(pattern, rf"\g<1>{scalar}", text, count=1)
    Path(path).write_text(text)
