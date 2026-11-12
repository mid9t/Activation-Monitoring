"""Held-out A1: sampled templates crossed with sampled requests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from amon.data.catalog import PLACEHOLDERS
from amon.data.textutil import group_id_for, is_code, item_id, messages_json, normalize_text


def _insert(template: str, request: str) -> tuple[str, str]:
    folded = template.lower()
    for placeholder in PLACEHOLDERS:
        start = folded.find(placeholder.lower())
        if start == -1:
            continue
        composed = template[:start] + request + template[start + len(placeholder) :]
        return composed, "placeholder"
    return template.rstrip() + "\n\n" + request, "append"


def build_templated_jailbreaks(
    templates: pd.DataFrame,
    requests: pd.DataFrame,
    n_templates: int,
    n_requests: int,
    seed: int,
) -> pd.DataFrame:
    """Cross `n_templates` templates with `n_requests` requests.

    Callers filter token length and near-duplicates of the tune pool first.
    The sample uses `seed` and takes every surviving row when fewer than asked remain.
    """
    templates = templates.drop_duplicates(subset=["text"]).reset_index(drop=True)
    requests = requests.drop_duplicates(subset=["text"]).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    n_templates = min(int(n_templates), len(templates))
    n_requests = min(int(n_requests), len(requests))
    if n_templates == 0 or n_requests == 0:
        return pd.DataFrame()
    template_index = np.sort(rng.choice(len(templates), size=n_templates, replace=False))
    request_index = np.sort(rng.choice(len(requests), size=n_requests, replace=False))
    rows = []
    insertions = []
    for t_i in template_index.tolist():
        template = templates.iloc[int(t_i)]
        for r_i in request_index.tolist():
            request = requests.iloc[int(r_i)]
            text, mode = _insert(str(template["text"]), str(request["text"]))
            template_id = str(template["template_id"])
            request_id = str(request["request_id"])
            identity = item_id("a1", f"{template_id}\n{request_id}")
            rows.append(
                {
                    "item_id": identity,
                    "source": "a1",
                    "cell": "templated",
                    "role": "heldout_A1",
                    "label": np.int8(1),
                    "group": "misses",
                    "messages": messages_json([{"role": "user", "content": text}]),
                    "group_id": group_id_for(text),
                    "template_id": template_id,
                    "request_id": request_id,
                    "lang": "en",
                    "is_code": bool(is_code(text)),
                    "_source_request": normalize_text(str(request["text"])),
                    "_rendered": "",
                }
            )
            insertions.append(
                {
                    "item_id": identity,
                    "template_id": template_id,
                    "request_id": request_id,
                    "insertion_mode": mode,
                }
            )
    frame = pd.DataFrame(rows)
    frame.attrs["insertions"] = pd.DataFrame(insertions)
    return frame
