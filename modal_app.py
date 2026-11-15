"""Run stages 02–04 on Modal. The Mac builds data, launches runs, and freezes choices."""

from __future__ import annotations

import os
import subprocess

import modal

app = modal.App("amon-gate1")
vol = modal.Volume.from_name("amon-data", create_if_missing=True, version=2)
hf = modal.Secret.from_name("huggingface")

# The Mac lock cannot pin Linux CUDA wheels. After the first image build, commit
# `pip freeze` from the image as requirements-linux.lock and this file will use it.
_LOCK = "requirements-linux.lock" if os.path.isfile("requirements-linux.lock") else "requirements.lock"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(_LOCK)
    .env(
        {
            "AMON_DATA": "/vol/amon",
            "HF_HOME": "/vol/hf",
            "AMON_DEVICE": "cuda",
            "AMON_MANIFESTS_DIR": "/vol/amon/manifests",
            "PYTHONPATH": "/root/repo/src",
        }
    )
    .add_local_dir("src", "/root/repo/src")
    .add_local_dir("configs", "/root/repo/configs")
    .add_local_dir("scripts", "/root/repo/scripts")
)
if os.path.isdir("manifests"):
    image = image.add_local_dir("manifests", "/root/repo/manifests")

GPU_KW = dict(
    image=image,
    volumes={"/vol": vol},
    secrets=[hf],
    gpu="A100-80GB",
    timeout=24 * 3600,
    max_containers=4,
    retries=modal.Retries(max_retries=10, initial_delay=0.0, backoff_coefficient=1.0),
)
CPU_KW = dict(image=image, volumes={"/vol": vol}, secrets=[hf], timeout=6 * 3600)

ROLES = ["train", "trainval", "tune", "heldout_A1", "heldout_A2", "heldout_B", "heldout_C", "pipeline"]


def _assert_cuda() -> None:
    import torch

    version = torch.version.cuda or ""
    assert torch.cuda.is_available() and version.startswith("13."), (
        f"expected CUDA 13.x, got available={torch.cuda.is_available()} version={version}"
    )


def _run(script, *args, commit, limit=0):
    env = {**os.environ, "AMON_GIT_COMMIT": commit, "AMON_GIT_DIRTY": "0"}
    if limit:
        env["AMON_OUT"] = f"/vol/amon/smoke/{commit[:7]}"
        env["AMON_MANIFESTS_DIR"] = f"{env['AMON_OUT']}/manifests"
    subprocess.run(
        ["python", f"/root/repo/scripts/{script}", *args],
        check=True,
        cwd="/root/repo",
        env=env,
    )
    vol.commit()


@app.function(**CPU_KW, cpu=4, memory=32768)
def prefetch(commit: str):
    """Download both models once, at the revisions stage 00 wrote into configs/models.yaml."""
    import yaml
    from huggingface_hub import snapshot_download

    models = yaml.safe_load(open("/root/repo/configs/models.yaml"))
    for key in ("monitored", "guard"):
        snapshot_download(
            models[key]["repo"],
            revision=models[key]["revision"],
            allow_patterns=["*.json", "*.safetensors", "tokenizer*"],
        )
    vol.commit()


@app.function(**CPU_KW, cpu=2, memory=8192)
def preflight(commit: str):
    """Hash /vol/amon/processed against the stage 01 manifest baked into the image."""
    import json
    from pathlib import Path

    from amon.provenance import sha256_file

    manifest_path = Path("/root/repo/manifests/01_build_data.json")
    if not manifest_path.is_file():
        raise SystemExit("manifests/01_build_data.json is not in the image. Commit stage 01 before preflight.")
    if not commit:
        raise SystemExit("preflight needs the git commit")
    data = json.loads(manifest_path.read_text())
    if data.get("git", {}).get("dirty") is True:
        raise SystemExit("stage 01 manifest comes from a dirty tree")
    if data.get("smoke"):
        raise SystemExit("stage 01 manifest is marked smoke")
    outputs = {rel: spec for rel, spec in (data.get("outputs") or {}).items() if rel.startswith("processed/")}
    if not outputs:
        raise SystemExit("stage 01 manifest has no processed/ outputs. Rebuild stage 01.")
    root = Path("/vol/amon")
    problems = []
    for rel, spec in outputs.items():
        path = root / rel
        if not path.is_file():
            problems.append(f"missing {rel}")
        elif path.stat().st_size != spec["bytes"] or sha256_file(path) != spec["sha256"]:
            problems.append(f"hash mismatch {rel}")
    if problems:
        raise SystemExit("preflight failed: " + "; ".join(problems))


@app.function(**GPU_KW)
def cache_role(role: str, commit: str, limit: int = 0):
    _assert_cuda()
    args = ["--roles", role]
    if limit:
        args += ["--limit", str(limit)]
    _run("02_cache_activations.py", *args, commit=commit, limit=limit)


@app.function(**GPU_KW)
def guard_roles(roles: str, commit: str, limit: int = 0):
    _assert_cuda()
    args = ["--roles", roles]
    if limit:
        args += ["--limit", str(limit)]
    _run("03_score_llamaguard.py", *args, commit=commit, limit=limit)


@app.function(**CPU_KW, cpu=16, memory=65536)
def train_probe(commit: str):
    _run("04_train_probe.py", "--no-fill", commit=commit)


def _clean_commit() -> str:
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    if status.stdout.strip():
        raise SystemExit("commit or stash changes before a Modal run; manifests must name a clean commit")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    return head.stdout.strip()


def _raise_failed(names: list[str], results: list) -> None:
    failed = [name for name, out in zip(names, results) if isinstance(out, Exception)]
    if failed:
        raise SystemExit(f"rerun with --roles {','.join(failed)}")


@app.local_entrypoint()
def main(step: str, limit: int = 0, roles: str = ""):
    commit = _clean_commit()
    if step == "prefetch":
        prefetch.remote(commit)
    elif step == "preflight":
        preflight.remote(commit)
    elif step == "cache":
        todo = roles.split(",") if roles else ROLES
        _raise_failed(
            todo,
            list(cache_role.map(todo, kwargs={"commit": commit, "limit": limit}, return_exceptions=True)),
        )
    elif step == "guard-trainval":
        guard_roles.remote("trainval", commit, limit)
    elif step == "guard-rest":
        rest = [role for role in ROLES if role not in ("train", "trainval")]
        _raise_failed(
            rest,
            list(
                guard_roles.map(
                    rest,
                    kwargs={"commit": commit, "limit": limit},
                    return_exceptions=True,
                )
            ),
        )
    elif step == "probe":
        train_probe.remote(commit)
    else:
        raise SystemExit(f"unknown step {step}")
