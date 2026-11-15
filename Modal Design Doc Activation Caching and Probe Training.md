# Modal Design Doc: Activation Caching and Probe Training

Nov 14, 2026 · @Ryu Kakuta

## Summary

The repo already implements stages 00–10 as local scripts; this design runs the three heavy stages on Modal without rewriting them:

- **Stage 02** caches Llama-3-8B-Instruct residual-stream activations on WildJailbreak and the held-out jailbreak sets.
- **Stage 03** scores Llama Guard 3.
- **Stage 04** trains the probe.

A new `modal_app.py` wraps the existing scripts and adds one shared Volume for data and model weights. Seven small code changes make the scripts safe on preemptible, git-less containers.

Estimated GPU cost is about $15–30 on an A100 80GB, well inside the $250 Modal credit; the price and runtime basis is in the cost section.

The repo was read at commit `7c347bf` (November 11, 2026). Everything else, including data building, threshold fitting, the hand check and the decision, stays on the Mac as the Gate 1 design document specifies.

## What the repo has today and what changes

The `amon` package (about 4,300 lines, with tests for every module) matches the Gate 1 design. The stage scripts assume one machine with a local GPU, git and a writable repo. Modal breaks those assumptions in three places:

- containers can be preempted and restarted;
- the copied repo has no `.git` directory;
- files written inside the container vanish unless they land on a Volume.

**What runs as-is.**

- `cache_split` already left-pads with explicit positions, writes memory-mapped `.npy` files in `.ids` order and resumes from `rows_done.json`.
- `score_guard` stores log-odds, not probabilities.
- `select_probe` and `fit_probe` implement the grouped-CV protocol.
- Config paths come from `AMON_DATA`, `HF_HOME` and `AMON_DEVICE`, so pointing them at a Volume needs no code change.

**Changes required before the first Modal run**

| # | File | Change | Why |
| --- | --- | --- | --- |
| 1 | `modal_app.py` (new) | Modal app, image, Volume, functions and local entrypoint (specified below) | Runs stages 02–04 remotely |
| 2 | `scripts/02_cache_activations.py`, `scripts/03_score_llamaguard.py` | Add `--roles a,b,c` and `--limit N` (writes to a separate `smoke/` tree) | One container per role, a pause between Llama Guard's train-val pass and the rest, and a cheap smoke run |
| 3 | `src/amon/provenance.py` | `git_state` falls back to `AMON_GIT_COMMIT` and `AMON_GIT_DIRTY` when git is absent. `publish_manifest` writes to `AMON_MANIFESTS_DIR` when set | Without git, manifests record `dirty: None`, which `refuse_dirty_inputs` silently accepts. Manifests written into the container's repo copy are lost |
| 4 | `src/amon/cache/extract.py` | On resume, drop from `rows_done` any row that is all zeros. On float16 overflow, raise instead of reallocating the file | A preemption can persist the progress file before the array data. `_store_as` currently reallocates the whole file, silently zeroing every row already written |
| 5 | `src/amon/llm/llamaguard.py`, `scripts/03_score_llamaguard.py` | Pass `position_ids` as caching does and request logits for the last position only (`logits_to_keep=1`; check the name in transformers 5.22.2). Add the 200-item greedy-generate check. Drop `train` from the default roles | Full-vocabulary logits for every position take about 8 GB per 32,000-token batch. The generate check is in the design but not the code. Train scores are not used at Gate 1 and add about 20% to the bill |
| 6 | `scripts/04_train_probe.py` | Load one layer at a time as float32. Add `--no-fill`, which writes the chosen layer and C to `probes/chosen.json` instead of editing `configs/probe.yaml` | Stacking 5 layers in float64 needs about 11 GB before CV copies. A config edit inside a container is lost, so freezing must happen in the local repo |
| 7 | `scripts/verify_sync.py` (new) | Recompute the SHA-256 of every downloaded file and compare it with its manifest | Acceptance check for the Volume-to-Mac copy |

## Architecture on Modal

The Mac builds data, launches runs and freezes choices in git. Modal containers only read inputs from one Volume and write outputs back to it.

&#91;embedded content: Modal architecture: Mac, one Volume, four functions\]

Nothing flows between containers directly. Every handoff passes through the Volume with a manifest, so any stage can be rerun alone and checked on the Mac.

## Modal setup

One app (`amon-gate1`), one image built from `requirements.lock`, one Volume (`amon-data`, version 2) and one secret (`huggingface`). Every function mounts the Volume at `/vol` and reads the repo from `/root/repo`.

**One-time setup on the Mac**

```bash
pip install modal
python3 -m modal setup                                   # browser login, stores the API token
modal secret create huggingface HF_TOKEN=hf_...          # the account that accepted the Meta and AI2 licenses
modal volume create amon-data --version=2
```

**Choices**

| Item | Choice | Reason |
| --- | --- | --- |
| Volume version | 2 | Version 2 writes only changed bytes inside a file. Version 1 can rewrite whole files, which is costly for incremental memory-map writes ([Modal Volumes](https://www.modal.com/docs/guide/volumes)) |
| Volume layout | `/vol/amon` = `AMON_DATA`; `/vol/hf` = `HF_HOME` | Paths stay config-driven; weights download once and are reused by every container |
| GPU | `A100-80GB` default; `H100` if cost per token is lower in the smoke run | The 32,000-token batch budget and full hidden-state output fit in 80 GB |
| Timeout | 24 hours per attempt (the maximum), `retries=modal.Retries(max_retries=3)` | Each retry gets a fresh timeout ([Modal timeouts](https://modal.com/docs/guide/timeouts.md)) |
| Preemption | Assume it happens | GPU functions can't opt out; Modal restarts the function on the same input, so every function must be resumable and idempotent ([Modal preemption](https://modal.com/docs/guide/preemption.md)) |
| Parallelism | One container per role via `.map` | Each role writes its own directory, so there are no concurrent writes to one file. Volumes are last-write-wins |
| Repo inside the image | `src/`, `configs/`, `scripts/`, `prereg/` copied to `/root/repo` | `amon.guard.repo_root()` finds the directory holding `configs/` and `src/` |

**`modal_app.py`** (new file at the repo root)

```python
import os, subprocess
import modal

app = modal.App("amon-gate1")
vol = modal.Volume.from_name("amon-data", create_if_missing=True, version=2)
hf = modal.Secret.from_name("huggingface")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements("requirements.lock")   # torch 2.15.0, transformers 5.22.2, ...
    .env({"AMON_DATA": "/vol/amon", "HF_HOME": "/vol/hf", "AMON_DEVICE": "cuda",
          "AMON_MANIFESTS_DIR": "/vol/amon/manifests", "PYTHONPATH": "/root/repo/src"})
    .add_local_dir("src", "/root/repo/src")             # local files go last in the image
    .add_local_dir("configs", "/root/repo/configs")
    .add_local_dir("scripts", "/root/repo/scripts")
    .add_local_dir("prereg", "/root/repo/prereg")
)
GPU_KW = dict(image=image, volumes={"/vol": vol}, secrets=[hf], gpu="A100-80GB",
              timeout=24 * 3600, retries=modal.Retries(max_retries=3))
CPU_KW = dict(image=image, volumes={"/vol": vol}, secrets=[hf], timeout=6 * 3600)

def _run(script, *args, commit):
    env = {**os.environ, "AMON_GIT_COMMIT": commit, "AMON_GIT_DIRTY": "0"}
    subprocess.run(["python", f"/root/repo/scripts/{script}", *args], check=True, cwd="/root/repo", env=env)
    vol.commit()

@app.function(**CPU_KW, cpu=4, memory=32768)
def prefetch(commit: str):            # download both models once, at the SHAs stage 00 wrote locally
    import yaml
    from huggingface_hub import snapshot_download
    models = yaml.safe_load(open("/root/repo/configs/models.yaml"))
    for key in ("monitored", "guard"):
        snapshot_download(models[key]["repo"], revision=models[key]["revision"],
                          allow_patterns=["*.json", "*.safetensors", "tokenizer*"])   # skips the original/ checkpoint copy
    vol.commit()

@app.function(**GPU_KW)
def cache_role(role: str, commit: str, limit: int = 0):
    _run("02_cache_activations.py", "--roles", role, *(["--limit", str(limit)] if limit else []), commit=commit)

@app.function(**GPU_KW)
def guard_roles(roles: str, commit: str, limit: int = 0, extra_category: str = ""):
    args = ["--roles", roles] + (["--limit", str(limit)] if limit else [])
    args += ["--extra-category", extra_category] if extra_category else []
    _run("03_score_llamaguard.py", *args, commit=commit)

@app.function(**CPU_KW, cpu=16, memory=65536)
def train_probe(commit: str):
    _run("04_train_probe.py", "--no-fill", commit=commit)

ROLES = ["train", "trainval", "tune", "heldout_A1", "heldout_A2", "heldout_B", "heldout_C", "pipeline"]

def _clean_commit():
    if subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip():
        raise SystemExit("commit or stash changes before a Modal run; manifests must name a clean commit")
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

@app.local_entrypoint()
def main(step: str, limit: int = 0, roles: str = "", extra_category: str = ""):
    commit = _clean_commit()
    if step == "prefetch":
        prefetch.remote(commit)
    elif step == "cache":
        todo = roles.split(",") if roles else ROLES
        list(cache_role.map(todo, kwargs={"commit": commit, "limit": limit}))
    elif step == "guard-trainval":
        guard_roles.remote("trainval", commit, limit)
    elif step == "guard-rest":
        rest = [r for r in ROLES if r not in ("train", "trainval")]
        list(guard_roles.map(rest, kwargs={"commit": commit, "limit": limit, "extra_category": extra_category}))
    elif step == "probe":
        train_probe.remote(commit)
    else:
        raise SystemExit(f"unknown step {step}")
```

Check two API details against the installed Modal client before the first run: the `kwargs=` argument of `.map`, and `Volume.from_name(..., version=2)`. If `.map` takes no keyword arguments in your version, use `.starmap` with tuples.

## Jailbreak data

Data is built on the Mac by the existing stage 01 and uploaded once; containers never download datasets. Stage 00 also runs locally, because it writes commit SHAs into `configs/models.yaml` and `configs/data.yaml`, and those edits must land in git before any Modal run.

| Role | Source (from `configs/data.yaml`) | Items | Label |
| --- | --- | --- | --- |
| train | WildJailbreak `vanilla_harmful`, `vanilla_benign`, WildChat English non-code | 67,500 | harmful / benign |
| trainval | 10% of train by `group_id` | 7,500 | as source |
| tune | WildJailbreak `adversarial_harmful`, `adversarial_benign`, WildChat ordinary | 100,000 | harmful / benign |
| heldout\_A1 | 100 in-the-wild jailbreak templates × 200 StrongREJECT requests | 20,000 | harmful |
| heldout\_A2 | Bailey et al.'s llama3-jailbreaks `harmful_*` except `harmful_prefill` | about 1,335 | harmful |
| heldout\_B | OR-Bench 80k sample + XSTest safe prompts | 20,250 | benign |
| heldout\_C | WildChat non-English and code first turns | 20,000 | benign |
| pipeline | Bailey's `*_instructions_*` splits, for the replication check only | about 19,900 | as split |

**Order of operations**

1. On the Mac: `make data`, which runs stage 01. Then run stage 00, commit `configs/`, and pass checkpoint 1 (tag `gate1-prereg`).
2. Upload the processed data:

   ```bash
   modal volume put amon-data "$AMON_DATA/processed" /amon/processed
   ```
3. Download the weights once: `modal run modal_app.py --step prefetch`, which writes both models into `/vol/hf` at the pinned SHAs.

The upload is about 1 GB. Raw downloads stay on the Mac; the Volume holds only `processed/`, so the split hashes in the memo cover exactly what the containers read.

## Caching job

`cache_role` runs the existing stage 02 for one role per container, eight containers in parallel, and writes ten float16 arrays per role straight onto the Volume. It reuses the repo's batching, explicit positions and verification.

**What one container does**

1. Load Meta-Llama-3-8B-Instruct in bf16 from `/vol/hf` at the pinned SHA, and set `tok.pad_token = tok.eos_token` as the script does.
2. Read `processed/items.parquet` and `processed/splits/<role>.ids`, render with the Llama 3 chat template and tokenize.
3. Pack length-sorted batches up to 32,000 padded tokens, left-pad, and pass `position_ids = (mask.cumsum(-1) - 1).clamp(min=0)`.
4. For layers 8, 12, 16, 20 and 24, store the last-token vector and the mean over real tokens after `<|begin_of_text|>`. Values must be finite; float16 overflow raises (change 4).
5. After each batch, write the rows into the memory-mapped files and update `rows_done.json`. Modal's background commits persist both every few seconds, and `_run` commits once more at the end.
6. Write `meta.parquet` (token counts and activation norms) and `manifest.json`, then run `verify_cache` on 200 random items rerun one at a time.

**Outputs per role**, under `/vol/amon/cache/llama3-8b-instruct@<sha>/<role>/`:

- `last_L08.npy` … `last_L24.npy` and `mean_L08.npy` … `mean_L24.npy`: float16, shape (N, 4096);
- `split.ids`, `rows_done.json`, `meta.parquet`, `manifest.json`.

**Resume after preemption or timeout.** Modal restarts the same input. `cache_split` reopens the existing arrays and skips rows listed in `rows_done.json`; change 4 first removes any listed row whose data is all zeros, which covers a progress file committed ahead of its data. Rerunning a finished role exits after verification.

**Commands**

```bash
modal run modal_app.py --step cache --limit 500      # smoke run: 500 items per role, into smoke/
modal run modal_app.py --step cache                  # full run, all eight roles in parallel
modal run modal_app.py --step cache --roles tune     # rerun one role
```

**Memory.** With `output_hidden_states=True`, one 32,000-token batch holds 33 hidden-state tensors of about 0.26 GB each in bf16, around 8.6 GB in total, beside the 16 GB of weights. That fits an 80 GB card. If the smoke run shows out-of-memory errors, halve `token_budget_per_batch` in `configs/cache.yaml` before the full run, and commit the change.

## Llama Guard scoring job

`guard_roles` runs stage 03 in two passes with an owner decision between them: train-val first, then the tune pool and the held-out roles under the frozen policy. It can run at the same time as caching, since the two jobs read the same inputs and write to different directories.

**Pass 1: train-val** (`modal run modal_app.py --step guard-trainval`)

1. Check that "safe" and "unsafe" are single tokens (already in `answer_token_ids`).
2. Run the new generate check (change 5) on 200 train-val items. The greedy first token must be one of the two answer tokens on at least 99% of items, and its sign must agree with the log-odds.
3. Score train-val and write `llm/policy_review.csv` with 50 missed harmful items.

**Owner decision.** Pull `policy_review.csv` (`modal volume get amon-data /amon/llm/policy_review.csv .`) and decide whether to add one category line. If one is added, rerun pass 1 with `--extra-category`, keep the version with the lower train-val weighted error, and commit the decision in `configs/llm.yaml`.

**Pass 2: the rest** (`modal run modal_app.py --step guard-rest [--extra-category "..."]`)

- This scores `tune`, `heldout_A1`, `heldout_A2`, `heldout_B`, `heldout_C` and `pipeline`, one container per role.
- Every role must reach a two-token mass of 0.9 or more on at least 99% of items, as the script already enforces.
- `prompt_hash` and `policy_sha256` must be identical across roles.

**Per batch.** Left-pad and pass explicit positions (change 5). Ask the model for logits at the last position only, then apply `scores_from_logits`, which stores `lp_safe`, `lp_unsafe`, `log_odds`, `mass2` and `top1_id` in float64.

**Resume.** `score_guard` writes one Parquet file per role at the end, so a preempted role restarts from zero. Each role takes well under an hour, so restarting is cheaper than adding checkpoints.

**Output.** `/vol/amon/llm/llama-guard-3-8b@<sha>/<role>.parquet`, plus `manifests/<role>/manifest.json`.

## Probe training job

`train_probe` runs stage 04 on a 16-core, 64 GB CPU container next to the cached arrays, so roughly 21 GB never has to leave the Volume before the probe is chosen. The chosen layer and C come back to the Mac, where they are frozen in git.

**Protocol** (unchanged from the repo and the Gate 1 design)

- **Data:** the `train` role's last-token vectors at layers 8, 12, 16, 20 and 24. Labels are 1 for harmful and 0 for benign; groups are `group_id`.
- **Model:** scikit-learn `LogisticRegression` with an L2 penalty, `class_weight="balanced"`, `lbfgs` and tolerance 1e-6, on features standardized with train statistics.
- **Search:** the layer at C = 1e-3 by 5-fold GroupKFold AUROC, then C over {1e-4 … 1e-1} at that layer. Within one standard error of the best, prefer layer 16, then the smallest C. That is 60 fits.
- **Sanity checks**, written to `sanity.json`:
  - Two fits give identical coefficients.
  - Mean z is higher on harmful train-val items than on benign ones.
  - With shuffled labels, train-val AUROC falls within 0.50 ± 0.02.
  - The pipeline replication on Bailey's instruction splits reaches recall at 1% FPR of at least 0.95.
  - The token-count AUROC is reported.
- **Scores:** float64 logit z for every role in `.ids` order, written to `scores/probe/<role>.npy` and never clipped.

**Change 6 in practice**

- `_stack` loads one layer at a time as float32: about 1.1 GB per layer for 67,500 × 4,096, instead of about 11 GB for all five layers in float64. Standardization still runs in float64, one fold at a time.
- `--no-fill` writes `probes/chosen.json` with the layer, C and every CV score, instead of editing `configs/probe.yaml`.

**Freezing the choice locally**

```bash
modal run modal_app.py --step probe
modal volume get amon-data /amon/probes "$AMON_DATA/"
python -c "import json; from amon.config import fill_tbd, configs_dir; c=json.load(open('$AMON_DATA/probes/chosen.json')); fill_tbd(configs_dir()/'probe.yaml', {'layer': c['layer'], 'C': c['C']})"
git commit -am "Freeze probe layer and C" && git push
```

**Where it could run instead.** The Mac works if it has 32 GB or more of memory and the cache has already been synced. The Modal container avoids both the wait for the copy and the memory pressure. CPU time is small next to the GPU jobs: 60 fits plus three extra fits for the sanity checks, on 54,000–67,500 rows each.

## Volume layout and sync to the Mac

The Volume mirrors `$AMON_DATA` under `/amon`, so every path in the repo means the same thing on both sides. The Mac stays the source of truth: each file comes back with its manifest and is accepted only when its hash matches.

```text
amon-data (Volume, version 2)
├── hf/                                         # HF_HOME: both models at pinned SHAs, about 32 GB
└── amon/                                       # AMON_DATA
    ├── processed/                              # uploaded from the Mac; read-only to containers
    ├── cache/llama3-8b-instruct@<sha>/<role>/  # stage 02, about 21 GB
    ├── llm/llama-guard-3-8b@<sha>/<role>.parquet
    ├── llm/policy_review.csv
    ├── probes/lr_L<layer>_C<C>/ and probes/chosen.json
    ├── scores/probe/<role>.npy
    ├── manifests/                              # AMON_MANIFESTS_DIR: copies for the repo's manifests/
    └── smoke/                                  # --limit runs; never mixed with real outputs
```

**Sync back**

```bash
for d in cache llm probes scores manifests; do
  modal volume get amon-data "/amon/$d" "$AMON_DATA/"
done
python scripts/verify_sync.py "$AMON_DATA"     # change 7: every file's SHA-256 against its manifest
cp "$AMON_DATA"/manifests/*.json manifests/ && git add manifests && git commit -m "Modal run manifests"
```

**Rules**

- Containers never write to `processed/`. If the data changes, rebuild locally and upload to a new path with new split hashes; never overwrite in place.
- Each role writes only inside its own directory, so parallel containers never touch the same file.
- Keep the Volume until Gate 1 is decided and the Mac copy is backed up, then delete `smoke/`. `hf/` can be deleted any time and re-fetched with `--step prefetch`.

## Cost and runtime

The full GPU work is about 6–7 A100-hours, or $15–18 at Modal's A100 80GB rate. Even budgeting double for retries, the smoke run and one rerun, it stays under $60 of the $250 credit. The throughput figure below is an assumption; the smoke run replaces it with a measured one.

| Job | Tokens through an 8B model | GPU-hours at an assumed 8,000 tokens/s | Cost at $2.50/h (A100 80GB) |
| --- | --- | --- | --- |
| Caching, 8 roles (about 257,000 items) | about 64 million | about 2.2 | about $6 |
| Llama Guard, 7 roles without train (about 189,000 items, about 200 policy tokens each) | about 90 million | about 3.1 | about $8 |
| Model loading (14 containers × about 3 min) and verification reruns | — | about 0.8 | about $2 |
| **Total** | about 154 million | **about 6–7** | **about $15–18** |

Modal's list rates are about $2.50/h for an A100 80GB and $3.95/h for an H100, billed per second; pinning a region multiplies the rate by 1.5–1.75 ([Spheron's summary of Modal pricing](https://www.spheron.network/blog/modal-gpu-pricing-2026-per-second-billing/)). An H100 pays off only if the smoke run shows at least 1.6 times the A100's tokens per second.

**Wall time.** All eight caching roles run in parallel, so wall time is set by the largest role: tune, at about 33 million tokens, takes about 70 minutes at the assumed rate. Llama Guard pass 2 runs alongside it. If your plan caps concurrent GPUs, `.map` queues the extra containers and wall time stretches; cost does not change.

**Smoke run first.** Run `--step cache --limit 500` and `--step guard-trainval --limit 500`. Read the tokens per second from the logs, recompute the table, and confirm that the full run fits the budget before starting it. The smoke run itself costs under $1.

## Tests and acceptance checks

New unit tests cover the seven changes and run locally with `make test`. The Modal runs are accepted only when the repo's existing stage checks pass on the Volume outputs and again after the copy to the Mac.

**New unit tests** (no GPU, no Modal)

- [ ] `git_state` returns the commit from `AMON_GIT_COMMIT` and `dirty: False` when git is missing, and `refuse_dirty_inputs` rejects a manifest whose `dirty` is `None`.
- [ ] `publish_manifest` writes to `AMON_MANIFESTS_DIR` when it is set.
- [ ] `cache_split` resume: interrupt after two batches, zero one finished row, rerun, and get arrays byte-identical to an uninterrupted run.
- [ ] Float16 overflow raises and leaves already-written rows untouched.
- [ ] Last-position-only Llama Guard logits equal the last row of full logits on a tiny random Llama config.
- [ ] Llama Guard with explicit positions matches single-item runs.
- [ ] `--limit` writes only under `smoke/`; `--roles` processes exactly the listed roles.
- [ ] Per-layer float32 `_stack` gives the same CV AUROC as the current code on a small synthetic cache, to four decimals.
- [ ] `verify_sync.py` fails when one byte of one file changes.

**Acceptance on Modal**

| Run | Passes when |
| --- | --- |
| Smoke | All jobs finish on 500 items per role; tokens per second recorded; the recomputed cost table fits the budget |
| Caching, per role | `verify_cache` passes: row counts match `.ids`, no NaN or all-zero rows, minimum cosine of 0.999 or more against single-item reruns. The manifest names a clean commit |
| Llama Guard pass 1 | Generate check of 99% or more; two-token mass of 0.9 or more on 99% of items; `policy_review.csv` written |
| Llama Guard pass 2 | The same mass check on every role; one policy hash across roles |
| Probe | All `sanity.json` checks pass; `chosen.json` written; scores exist for every role |
| Sync | `verify_sync.py` passes on the Mac and the manifests are committed |

**Preemption drill.** Once during the smoke run, stop a `cache_role` container in the Modal dashboard. Rerunning that role must finish without duplicate or zero rows and pass `verify_cache`.

## Open questions and resolutions

Research resolved all 26 open items: 5 checks passed, 6 new defects turned up, and 3 items need your decision. Item 5 is the most urgent: as written, Llama Guard is scored one token too early, so pass 1 would fail its own format check.

Devin can do every item marked Devin before the smoke run. Items 2, 22 and 26 are yours to decide. The sections above still show the original design; they change once you accept these resolutions.

| # | Item | What research found | Resolution | Owner | Status |
| --- | --- | --- | --- | --- | --- |
| 1 | Output hashes for verify\_sync | `write_manifest` hashes inputs only. Stage 04 passes no inputs; stage 03 lists its own parquet as an input. | Add `outputs` (SHA-256 and bytes per file, paths relative to the data root). `verify_sync.py` recomputes them on the Mac. | Devin | Proposed |
| 2 | Policy decision handoff | `--extra-category` is a free CLI string. Nothing ties pass 2 to the decision. | Record `extra_category` and `policy_decided` in `configs/llm.yaml` and commit before pass 2. Stage 03 reads only the config, and the CLI flag is removed. | Ryu decides; Devin enforces | Needs decision |
| 3 | `prereg/` folder missing | Confirmed absent at 7c347bf. Only stages 06, 09 and 10 read it, and all three run locally. | Drop `add_local_dir("prereg", …)`. No container can then see `freeze.json`. | Devin | Proposed |
| 4 | Stage 04 needs every role | It scores every role in `items.parquet` and reads train-val for its sanity checks. | Preflight: every role has a passing `verify.json`, a matching row count and the same `items.parquet` hash. | Devin | Proposed |
| 5 | Llama Guard verdict position (new) | Llama Guard 3 emits "\\n\\n" before "safe" or "unsafe". The repo scores the "\\n\\n" step. | Append "\\n\\n" to the rendered prompt before scoring. Keep the two-token mass check as the format gate. | Devin | Proposed |
| 6 | Extra category placement (new) | The repo inserts the line after the end-of-turn token, outside the user turn. | Insert it as `S15: …` inside the categories block, the way Meta's prompt utilities number custom categories. | Devin | Proposed |
| 7 | Progress file written before flush (new) | `rows_done.json` is written before the memmaps flush, so a snapshot can list rows that hold no data. | Replaced by the shards in item 10. | Devin | Proposed |
| 8 | `git_state` without git (new) | `subprocess.run(["git", …])` raises FileNotFoundError when git is missing; it does not return an error code. | Read `AMON_GIT_COMMIT` and `AMON_GIT_DIRTY` first, and catch FileNotFoundError. | Devin | Proposed |
| 9 | Dirty check passes unknown state (new) | `refuse_dirty_inputs` treats `dirty: None` as clean. | Refuse when `commit` or `dirty` is None. | Devin | Proposed |
| 10 | Cache resume and float16 overflow | Modal's docs do not cover memmaps under background commits. In Llama-3-8B, massive activations occur only at BOS, and neither stored position includes BOS. | Memmaps on local SSD, immutable shards on the Volume, and an error on overflow. | Devin | Proposed |
| 11 | `--roles`, `--limit` and smoke routing | `verify_cache` finds `processed/` by walking up from the cache path, so a cache under `smoke/` breaks it. | Add `AMON_OUT` for outputs, pass the data root into `verify_cache`, sample by seed and label, and flag smoke manifests. | Devin | Proposed |
| 12 | Verify verdict not saved (new) | Stage 02 prints the verdict and publishes the manifest before checking it. | Write `verify.json`, hash it as an output, and publish only on a pass. | Devin | Proposed |
| 13 | Generate check | `generate_check_n: 200` is configured, but no code uses it. | Greedy-generate 8 tokens on 200 seeded items per role and require at least 99% agreement with the scored label. | Devin | Proposed |
| 14 | `--no-fill` and `chosen.json` | Stage 04 calls `fill_tbd` inside the container, where the edit is lost. | Write `probes/chosen.json` (schema below); `fill_tbd` runs on the Mac. | Devin | Proposed |
| 15 | `.map(..., kwargs=)` | Verified: `map(*input_iterators, kwargs={}, order_outputs=True, return_exceptions=False)`. | Keep as written and add `return_exceptions=True`. | — | Verified |
| 16 | Volume v2 | Verified: pass `version=2` to `from_name`; v1 stays the default. v2 allows up to 262,144 files per directory. | Keep as written. | — | Verified |
| 17 | CUDA build of torch | The Linux torch 2.15.0 wheel pins cuda-toolkit 13.2.2. Modal's driver 610.57.04 provides CUDA 13.3. The Mac lock omits these Linux-only packages. | Compatible. Freeze a Linux lock inside the image once, and assert `torch.cuda.is_available()` at startup. | Devin | Verified |
| 18 | `logits_to_keep` | Verified in the 5.22.2 source: the argument exists, defaults to 0 (all positions), and an int N keeps the last N positions. | Use `logits_to_keep=1`. With left padding, the last position is the last real token. | — | Verified |
| 19 | `position_ids` in Llama Guard | Without them, every position shifts by the pad count. RoPE depends only on relative position, so the effect is numerical. | Pass them anyway, to match caching. | Devin | Proposed |
| 20 | Retries under preemption | Modal restarts a preempted input. The docs do not say whether that uses a retry. | Make each input idempotent and set `Retries(max_retries=10, initial_delay=0.0, backoff_coefficient=1.0)`. | Devin | Proposed |
| 21 | Throughput of 8,000 tokens/s | 8,000 tokens/s is about 41% of an A100's BF16 peak (16 GFLOP per token, 312 TFLOPS). | Measure it in the smoke run, then use the decision table below. | Smoke run | Proposed |
| 22 | Parallel GPUs | Starter allows 10 GPUs. Modal advises at most 5 concurrent commits. | Set `max_containers=4` on both GPU functions. Confirm your Modal plan. | Ryu | Needs decision |
| 23 | Storage and CPU cost | Volumes cost $0.09 per GiB-month with 1 TiB free, so the \~60 GB here costs nothing. CPU and memory add about $2. | Add both lines to the cost table. | — | Verified |
| 24 | Upload size and `processed/` | The 1 GB upload is an estimate, and nothing checks what arrived. | Run `du -sh` before `put`. A preflight compares the `items.parquet` SHA-256 with the stage 01 manifest. | Devin | Proposed |
| 25 | `apt_install("git")` | Needed today only because of item 8. | Remove it once item 8 lands. | Devin | Proposed |
| 26 | past.dev | Still unidentified; searches found nothing matching. | Not needed for Gate 1. Name it, or drop it from the resource map. | Ryu | Needs decision |

### Llama Guard prompt fix (items 5, 6, 13)

Llama Guard 3's template ends at the assistant header with no newline. The model then writes "\\n\\n" before the label, so the label is the second token, not the first. Meta's [customization notebook](https://github.com/meta-llama/llama-cookbook/blob/main/getting-started/responsible_ai/llama_guard/llama_guard_customization_via_prompting_and_fine_tuning.ipynb) prints outputs as "\\n\\nunsafe\\nS2". A 2026 audit of Llama Guard 3 got a block rate of zero when it scored the first token ([Zhang et al.](https://arxiv.org/pdf/2609.26178)). A [Hugging Face tutorial for Llama Guard 3](https://huggingface.co/meta-llama/Llama-Guard-3-8B/discussions/21) appends "\\n\\n" to the prompt for the same reason.

As written, stage 03 would fail its own two-token mass check on pass 1. That costs about an hour but corrupts nothing. The fix also moves the custom category into the categories block. That is where Meta's [prompt utilities](https://github.com/meta-llama/llama-cookbook/blob/main/src/llama_cookbook/inference/prompt_format_utils.py) number custom categories (S1 to Sn):

```python
# src/amon/llm/llamaguard.py
HEADER = "<|start_header_id|>assistant<|end_header_id|>"
CATS_END = "<END UNSAFE CONTENT CATEGORIES>"

def render_guard_prompt(messages, tok, policy):
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if not text.rstrip().endswith(HEADER):
        raise ValueError("Llama Guard prompt must end with the assistant header")
    if policy.extra_category:
        line = policy.extra_category.strip()      # "S15: <name>.\n<description>"
        if CATS_END not in text or not line.startswith("S15:"):
            raise ValueError("custom category must be S15 inside the categories block")
        text = text.replace(CATS_END, f"{line}\n{CATS_END}", 1)
    return text.rstrip() + "\n\n"                  # score the label, not the newline
```

Add a unit test that the tokenized prompt ends with the assistant header ids followed by the single "\\n\\n" token. The label ids stay `safe` and `unsafe` with no leading space, which matches the trained output.

**Generate check (item 13).** For each role, greedy-generate on 200 seeded items from the same prompts and compare with the scored label:

```python
def generate_check(model, tok, prompts, item_ids, log_odds, n=200, seed=0):
    pick = np.random.default_rng(seed).choice(len(prompts), min(n, len(prompts)), replace=False)
    rows = []
    for i in pick:
        enc = tok(prompts[i], return_tensors="pt", add_special_tokens=False).to(model.device)
        out = model.generate(**enc, max_new_tokens=8, do_sample=False)
        text = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
        rows.append({"item_id": item_ids[i], "generated": text.splitlines()[0] if text else "",
                     "scored": "unsafe" if log_odds[i] > 0 else "safe"})
    agree = float(np.mean([r["generated"] == r["scored"] for r in rows]))
    return {"n": len(rows), "agreement": agree, "pass": agree >= 0.99, "rows": rows}
```

Write the result to `llm/llama-guard-3-8b@<sha>/checks/generate_<role>.json` and hash it as an output. Stage 03 stops if `pass` is false. The check takes about 2 minutes per role.

**Whether S15 is worth adding.** Llama Guard 3 was trained on S1 to S14 only. In the original [Llama Guard paper](https://arxiv.org/abs/2312.06674), zero-shot category prompts raised AUPRC on OpenAI Moderation from 0.837 to 0.847, and few-shot prompts to 0.872. So adopt S15 only if it measurably lowers train-val misses (item 2).

### Manifests and sync (items 1, 8, 9, 12)

Each manifest gains an `outputs` map with a SHA-256 and a byte count per file, keyed by path relative to `$AMON_DATA`. Relative keys let the same manifest match on the Volume (`/vol/amon/…`) and on the Mac. Input keys become relative for the same reason. DVC records a hash and a size per output in the same way ([DVC docs](https://doc.dvc.org/user-guide/project-structure/dvc-files)).

```python
# src/amon/provenance.py
def git_state(cwd=None):
    if os.environ.get("AMON_GIT_COMMIT"):                 # set by modal_app._run
        return {"commit": os.environ["AMON_GIT_COMMIT"],
                "dirty": os.environ.get("AMON_GIT_DIRTY") == "1", "error": None, "source": "env"}
    try:
        head = _git(["rev-parse", "HEAD"], cwd or repo_root())
    except FileNotFoundError:                             # git absent: record it, never crash
        return {"commit": None, "dirty": None, "error": "git not installed"}
    ...                                                   # unchanged below

def data_rel(path):
    return str(Path(path).resolve().relative_to(Path(os.environ["AMON_DATA"]).resolve()))

def write_manifest(out_dir, inputs, cfg, outputs=()):
    ...
    payload["inputs"] = {data_rel(p): sha256_file(p) for p in inputs if Path(p).is_file()}
    payload["outputs"] = {data_rel(p): {"sha256": sha256_file(p), "bytes": Path(p).stat().st_size}
                          for p in outputs}
    payload["smoke"] = bool(cfg.get("smoke", False))

def refuse_dirty_inputs(manifests):
    bad = [str(m) for m in manifests
           if (g := json.loads(Path(m).read_text()).get("git", {})).get("commit") is None
           or g.get("dirty") is not False]               # None now counts as dirty
    if bad:
        raise DirtyTreeError("refusing manifests without a clean commit: " + ", ".join(bad))
```

| Stage | Files listed in `outputs` |
| --- | --- |
| 02, per role | `split.ids`, `meta.parquet`, ten `last_L*.npy` and `mean_L*.npy` files, `verify.json` |
| 03, per role | `<role>.parquet`, `checks/generate_<role>.json` |
| 04 | the probe directory's files, `probes/chosen.json`, `scores/probe/<role>.npy` for every role |

**Stage 02 order (item 12).** Run `cache_split`, then `verify_cache`, then write `verify.json`, then write the manifest with `verify.json` among its outputs. Publish the manifest only if the verdict passes. Otherwise exit 1 and leave it unpublished.

**`scripts/verify_sync.py`.** Run it on the Mac after `modal volume get`:

```python
def main(manifest_dir="manifests") -> int:
    root, problems = Path(os.environ["AMON_DATA"]), []
    for m in sorted(Path(manifest_dir).glob("0[2-4]_*.json")):
        data = json.loads(m.read_text())
        if data.get("smoke"):
            problems.append(f"{m.name}: smoke manifest in the real set")
        for rel, want in data.get("outputs", {}).items():
            p = root / rel
            if not p.is_file():
                problems.append(f"{m.name}: missing {rel}")
            elif p.stat().st_size != want["bytes"] or sha256_file(p) != want["sha256"]:
                problems.append(f"{m.name}: hash mismatch {rel}")
    print("\n".join(problems) or "all outputs match")
    return 1 if problems else 0
```

These hashes prove the download is intact. They do not prove the run is reproducible. A rerun or a resume can give slightly different bits, because GPU kernel results depend on batch composition ([Thinking Machines, summarized by Simon Willison](https://simonwillison.net/2025/Sep/11/defeating-nondeterminism)). Reproducibility stays with the cosine ≥ 0.999 check in `verify_cache`.

### Cache shards (items 7, 10)

Write the memmaps on the container's local SSD and checkpoint immutable shards to the Volume. A shard is either whole or absent, so resuming needs no progress file and no guessing from all-zero rows.

Modal's [Volumes guide](https://modal.com/docs/guide/volumes.md) says nothing about memmaps or open files during background commits, and says the last write wins. Its [dataset ingestion guide](https://modal.com/docs/guide/dataset-ingestion.md) recommends doing heavy IO in `/tmp` on local SSD. The container disk quota defaults to 512 GiB ([Resources](https://modal.com/docs/guide/resources)), and one role needs a few GB.

```text
/tmp/cache/<role>/{last,mean}_L*.npy                          # local memmaps, rebuilt after a restart
/vol/amon/cache/llama3-8b-instruct@<sha>/<role>/shards/part-000042.npz   # row ids + 10 arrays
/vol/amon/cache/llama3-8b-instruct@<sha>/<role>/{last,mean}_L*.npy       # final copy, written once
```

1. On start, list `shards/part-*.npz` on the Volume and ignore any `*.tmp`. The rows they hold are done.
2. Copy those rows into fresh local memmaps under `/tmp`.
3. Run the remaining batches. Every 2,000 rows (about 160 MB), write `part-NNNNNN.npz.tmp` with the row indices and all ten arrays, then `os.replace` it to `.npz`.
4. After the last batch, flush, copy each `.npy` to the Volume, and run `verify_cache`. Then write `verify.json` and the manifest, and delete `shards/`.
5. Call `vol.commit()`.

A preemption loses at most one shard of work. In the preemption drill, check that the killed container left only whole `.npz` files and that the rerun skipped them.

**Float16 overflow.** The float16 maximum is 65,504. Massive activations, the only values near that range, sit at the starting token and the first delimiter ([Sun et al.](https://eric-mingjie.github.io/massive-activations/index.html)). In Llama-3-8B they appear at BOS only, not at delimiters ([House of Cards](https://arxiv.org/html/2410.01866v2)). The last position is the generation header's final token, and the mean already skips BOS. So overflow is not expected, and if it happens it is a bug. Delete `_store_as`, raise instead, and record the peak value:

```python
def _to_storage(values, dtype_store):
    values = values.detach().float().cpu()
    if not torch.isfinite(values).all():
        raise RuntimeError("non-finite activation; refusing to store the cache")
    peak = float(values.abs().max())
    if dtype_store == "float16" and peak >= 60000:          # 8% headroom under 65,504
        raise RuntimeError(f"peak {peak:.0f} near the float16 limit; set dtype_store: float32")
    return values.numpy().astype(np.float16 if dtype_store == "float16" else np.float32), peak
```

The manifest stores the peak per layer and position, so the smoke run shows the real headroom. If float32 is ever needed, the cache grows from about 21 GB to 42 GB, which still fits.

### Policy decision (item 2)

The decision lives in `configs/llm.yaml` and is committed before pass 2. Stage 03 reads it only from there, so drop `--extra-category` from stage 03 and from `modal_app.py`.

```yaml
# configs/llm.yaml
policy: default_s1_s14
extra_category: null        # or "S15: <name>.\n<one-paragraph description>"
policy_decided: false       # set true in the same commit as extra_category
min_two_token_mass: 0.9
generate_check_n: 200
threshold_rule: fit_on_tune
```

1. Pass 1 scores train-val under the default policy and writes `llm/policy_review.csv` (50 missed harmful items).
2. You read the misses. If a pattern falls outside S1 to S14, draft one S15 line; otherwise leave `extra_category: null`.
3. If you drafted S15, rescore train-val with it and compare misses and false positives. Decide using train-val alone, before any tune or held-out score exists.
4. Set `policy_decided: true`, commit, and tag `policy-decided`.
5. Run pass 2. Stage 03 refuses if `policy_decided` is false, or if the train-val manifest's `policy_sha256` differs from the config's.

Every role ends with the same `policy_sha256`. Stage 05 checks that all eight `03_llm_*.json` manifests share one hash, so no analysis mixes policies.

### Smoke routing, preflights and chosen.json (items 4, 11, 14, 24)

**Smoke routing (item 11).** Inputs (`processed/`, `hf/`) are always read from `AMON_DATA`. Outputs go to a new `out_root: ${AMON_OUT}` in `configs/paths.yaml`, which defaults to the data root. A smoke run points it at its own folder:

```python
# modal_app.py
def _run(script, *args, commit, limit=0):
    env = {**os.environ, "AMON_GIT_COMMIT": commit, "AMON_GIT_DIRTY": "0"}
    if limit:
        env["AMON_OUT"] = f"/vol/amon/smoke/{commit[:7]}"
        env["AMON_MANIFESTS_DIR"] = f"{env['AMON_OUT']}/manifests"
    subprocess.run(["python", f"/root/repo/scripts/{script}", *args], check=True, cwd="/root/repo", env=env)
    vol.commit()
```

- `--limit N` takes N items per role, sampled with seed 0 and stratified by label, so the mass and generate checks see both classes. Chosen rows keep the role's `.ids` order.
- `--roles a,b` takes a comma list and rejects unknown role names.
- `verify_cache` gains a `data_root` argument in place of `split_dir.parent.parent.parent`.
- Smoke manifests carry `"smoke": true`. Stage 09 and `verify_sync.py` refuse them.

**Upload check (item 24).** Stage 01's manifest lists `items.parquet` and the split `.ids` files under `outputs` (item 1). A small CPU step compares them after the upload:

```bash
du -sh "$AMON_DATA/processed"                                  # replaces the 1 GB estimate
modal volume put amon-data "$AMON_DATA/processed" /amon/processed
modal run modal_app.py --step preflight                       # hashes /vol/amon/processed against 01_build_data.json
```

**Stage 04 preflight (item 4).** `train_probe` checks every role before it loads anything:

```python
items = pd.read_parquet(root / "processed/items.parquet")
for role, n in items["role"].value_counts().items():
    d = root / f"cache/llama3-8b-instruct@{sha}/{role}"
    verdict = json.loads((d / "verify.json").read_text())
    if not verdict["pass"] or verdict["min_cosine"] < 0.999 or len((d / "split.ids").read_text().split()) != n:
        raise SystemExit(f"{role}: cache missing, unverified or the wrong size")
```

**`chosen.json` (item 14).** With `--no-fill`, stage 04 skips `fill_tbd` and writes `probes/chosen.json` before the sanity gate:

```text
{
  "layer": int, "C": float, "position": "last",
  "tie_rule": "one_se_prefer_layer16_then_smallest_C",
  "cv": <contents of cv.json>, "probe_dir": "probes/lr_L<layer>_C<C>",
  "sanity_pass": bool, "git_commit": str
}
```

On the Mac, after `verify_sync.py` passes, `fill_tbd` copies `layer` and `C` into `configs/probe.yaml`, and you commit. Stage 04 still exits 1 when `sanity_pass` is false.

### Runtime, retries, environment and cost (items 17, 20 to 23)

**Throughput (item 21).** A forward pass costs about 2 FLOPs per parameter per token ([FLOP guide](https://leo1oel.github.io/blog/2026-05/flop)), so about 16 GFLOP per token for an 8B model. The A100 80GB peaks at 312 BF16 TFLOPS ([NVIDIA datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/a100-80gb-datasheet-update-a4-nvidia-1485612-r12-web.pdf)). So 8,000 tokens/s means 128 TFLOPS, or 41% model FLOPs utilization. For scale, PaLM 540B trained at 46.2% and GPT-3 at 21.3% ([PaLM paper](https://arxiv.org/pdf/2204.02311)). 41% is reachable for large batched forward passes but not guaranteed. The smoke run prints tokens per second, excluding model load, and the table below sets the response:

| Smoke-run throughput (tokens/s) | Full run (A100 hours) | GPU cost | Action |
| --- | --- | --- | --- |
| 8,000 or more | 6.1 or less | $15 or less | Proceed |
| 6,000 | 7.9 | $20 | Proceed |
| 4,000 | 11.5 | $29 | Proceed after checking `attn_implementation="sdpa"` and the padded-to-real token ratio |
| Under 3,000 | over 15 | over $38 | Stop and profile before the full run |

The table assumes 154M tokens (64M cached, 90M through Llama Guard) plus 0.8 hours of loading and checks, at $2.50 per hour. An H100 costs 1.58 times as much per hour on [Modal's price list](https://modal.com/pricing), so switch only if it runs more than 1.58 times faster.

**Retries and concurrency (items 20, 22).** Modal restarts a preempted input on the same input but does not say whether that uses a retry ([Preemption](https://modal.com/docs/guide/preemption.md)). Its [long-training example](https://modal.com/docs/examples/long-training.md) uses `max_retries=10`, which it calls the current maximum, with no delay. With item 10's shards every restart is cheap, so allow the maximum. Starter allows 10 GPUs at once ([Pricing](https://modal.com/pricing)), but the [Volumes guide](https://modal.com/docs/guide/volumes.md) advises at most 5 concurrent commits, so cap at 4 containers:

```python
GPU_KW = dict(image=image, volumes={"/vol": vol}, secrets=[hf], gpu="A100-80GB",
              timeout=24 * 3600, max_containers=4,
              retries=modal.Retries(max_retries=10, initial_delay=0.0, backoff_coefficient=1.0))

results = list(cache_role.map(todo, kwargs={"commit": commit, "limit": limit}, return_exceptions=True))
failed = [role for role, out in zip(todo, results) if isinstance(out, Exception)]
if failed:
    raise SystemExit(f"rerun with --roles {','.join(failed)}")
```

With `return_exceptions=True`, one failed role no longer cancels the others ([Failures and retries](https://modal.com/docs/guide/retries.md)). Stage 03 still reruns a whole role after a restart, which costs at most about an hour. Add shards there too only if the preemption drill shows frequent restarts.

**CUDA build (item 17).** On Linux, torch 2.15.0 pins `cuda-toolkit==13.2.2` and `nvidia-cudnn-cu13` ([PyPI](https://pypi.org/project/torch/2.15.0/)). Modal's driver 610.57.04 provides CUDA 13.3, and the toolkit must not exceed the host's version ([CUDA on Modal](https://modal.com/docs/guide/cuda.md)), so the pair works. The Mac lock cannot pin these Linux-only packages. Build the image once from `requirements.lock`, run `pip freeze` inside it, and commit the result as `requirements-linux.lock` for the image to use. Every GPU function starts with `assert torch.cuda.is_available() and torch.version.cuda.startswith("13.")`.

**Full cost (item 23).**

| Line | Basis | Cost |
| --- | --- | --- |
| A100 80GB | 6.1 hours at $2.50 per hour | about $15 |
| Probe training CPU | 16 cores and 64 GiB at $1.27 per hour, about 1 hour assumed | about $1.30 |
| Prefetch CPU | 4 cores and 32 GiB at $0.44 per hour, under 1 hour | under $0.50 |
| Volume | about 60 GB, inside the free 1 TiB per month | $0 |
| Total | at 8,000 tokens/s | about $17, against a $60 cap |

## Sources

- [mid9t/Activation-Monitoring](https://github.com/mid9t/Activation-Monitoring), read at commit `7c347bf` (November 11, 2026)
- [Modal Volumes](https://www.modal.com/docs/guide/volumes): commits, version 2 partial writes, CLI transfer, limits
- [Modal timeouts](https://modal.com/docs/guide/timeouts.md): 24-hour maximum, retries
- [Modal preemption](https://modal.com/docs/guide/preemption.md): GPU functions are preemptible and restart on the same input
- [Spheron: Modal GPU pricing 2026](https://www.spheron.network/blog/modal-gpu-pricing-2026-per-second-billing/)
- [Gate 1 Design Document (Agent Handoff)](https://claude.ai/artifact/PW42b2Fx4BheWf144oeQk1) and [Gate 1 Implementation Plan](https://claude.ai/artifact/6nJptfFNLXtLZKHpVjLtqo)

**Added for the open questions** (pages opened November 15, 2026)

- Modal: [Function.map and starmap](https://modal.com/docs/sdk/py/latest/Function), [App.function parameters](https://modal.com/docs/sdk/py/latest/App), [Volumes](https://modal.com/docs/guide/volumes.md), [Preemption](https://modal.com/docs/guide/preemption.md), [Failures and retries](https://modal.com/docs/guide/retries.md), [Retries](https://modal.com/docs/sdk/py/latest/Retries), [Long-training example](https://modal.com/docs/examples/long-training.md), [Resources](https://modal.com/docs/guide/resources), [Dataset ingestion](https://modal.com/docs/guide/dataset-ingestion.md), [CUDA on Modal](https://modal.com/docs/guide/cuda.md), [Pricing](https://modal.com/pricing)
- Packages: [transformers 5.22.2](https://pypi.org/project/transformers/5.22.2/) (`modeling_llama.py` and `slice_logits_to_keep` read from the installed source), [torch 2.15.0 metadata](https://pypi.org/project/torch/2.15.0/)
- Llama Guard: [Llama Guard 3 8B model card](https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard3/8B/MODEL_CARD.md), [Meta prompt utilities](https://github.com/meta-llama/llama-cookbook/blob/main/src/llama_cookbook/inference/prompt_format_utils.py), [Meta customization notebook](https://github.com/meta-llama/llama-cookbook/blob/main/getting-started/responsible_ai/llama_guard/llama_guard_customization_via_prompting_and_fine_tuning.ipynb), [Hugging Face classifier tutorial](https://huggingface.co/meta-llama/Llama-Guard-3-8B/discussions/21), [Inan et al., Llama Guard (2023)](https://arxiv.org/abs/2312.06674), [Zhang et al., Unread or Unenforced? (2026)](https://arxiv.org/pdf/2609.26178)
- Activations and numerics: [Sun et al., Massive Activations in Large Language Models (2024)](https://eric-mingjie.github.io/massive-activations/index.html), [House of Cards: Massive Weights in LLMs](https://arxiv.org/html/2410.01866v2), [Thinking Machines on batch invariance, summarized by Simon Willison](https://simonwillison.net/2025/Sep/11/defeating-nondeterminism)
- Compute: [NVIDIA A100 80GB datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/a100-80gb-datasheet-update-a4-nvidia-1485612-r12-web.pdf), [Chowdhery et al., PaLM (2022)](https://arxiv.org/pdf/2204.02311), [FLOP accounting guide](https://leo1oel.github.io/blog/2026-05/flop)
- Provenance: [DVC output entries](https://doc.dvc.org/user-guide/project-structure/dvc-files)
- Repo: [mid9t/Activation-Monitoring at 7c347bf](https://github.com/mid9t/Activation-Monitoring/tree/7c347bf)
