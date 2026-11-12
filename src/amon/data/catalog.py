"""Dataset locations named by the design document. Revisions come from models.yaml."""

from __future__ import annotations

# Instruction splits are the pipeline check, not held-out A2.
# The implementation plan's A2 count (about 1,335) is the harmful_* attack
# splits with harmful_prefill and the instruction splits removed.
PIPELINE_SPLITS = (
    "harmful_instructions_train",
    "harmful_instructions_test",
    "benign_instructions_train",
    "benign_instructions_test",
)

# Languages the optional tune domain slice would use. Held-out C excludes them
# only when that slice is actually built (domain_slice > 0).
TUNE_SLICE_LANGUAGES = frozenset({"fr", "de", "es", "pt", "it"})

ENGLISH = frozenset({"en", "eng", "english"})

PLACEHOLDERS = (
    "[INSERT PROMPT HERE]",
    "[INSERT PROMPT]",
    "[PROMPT]",
    "[QUERY]",
    "{prompt}",
    "{{prompt}}",
)


def a2_splits(exclude: list[str]) -> list[str]:
    """harmful_* attack splits, minus the configured exclusions and the pipeline splits."""
    attacks = [
        "harmful_autodan",
        "harmful_best_of_n",
        "harmful_gcg",
        "harmful_human_mt",
        "harmful_misc",
        "harmful_msj",
        "harmful_pair",
        "harmful_simple_adaptive",
    ]
    banned = set(exclude) | set(PIPELINE_SPLITS)
    return [name for name in attacks if name not in banned]
