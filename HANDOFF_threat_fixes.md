# Threat-fix handoff

## 1. Status

Blocked at owner checkpoint 1. Items 1–3 are implemented and tested on `gate1-threat-fixes`. The two clean rebuilds have not been run. Stage 01 will also stop until StrongREJECT has a `category` column: the pinned validation files do not.

## 2. Commits

- `b109cde` Add the ordinary-cell filter and the Gate 1 threat reports.
- `efaa648` memo: pre-register threat fixes 1-3.

## 3. Acceptance

| # | Criterion | Result | Evidence |
| --- | --- | --- | --- |
| 1 | `make test` passes, including the 19 new tests | Pass | 76 passed in 8.05s. The 19 tests are the 7 in `tests/test_audit.py`, the 5 in `tests/test_categories.py`, and the 7 in `tests/test_deferral.py`. |
| 2 | No flagged template row is in any ordinary cell | Blocked | Not checked on a rebuilt `items.parquet`. `test_filter_runs_before_sampling` checks it on a synthetic pool. |
| 3 | Every owner exclusion is absent from `items.parquet` | Blocked | `prereg/ordinary_exclusions.csv` does not exist yet. A missing file drops nothing. |
| 4 | Filter precision at least 80%, or the one recorded adjustment | Blocked | The sheet is unlabeled. |
| 5 | Every A1 row has a non-empty `harm_category` | Blocked | The copy is implemented. The pinned StrongREJECT frame has no `category` column, so normalize raises `SchemaError` before A1 is built. |
| 6 | Stage 06 on synthetic fixtures writes `categories.json` and `pipeline.json` | Pass | `test_errors_by_sums_to_headroom` writes `categories.json` with `stratum`, `n`, `recall`, `lo`, `hi`, `method` and `stratum`, `n`, `F`, `U`, `I`, `U_share`. `test_mixed_set_errors_matches_weighted_error` writes `pipeline.json` with `error`, `lo`, `hi` and `items`, `tokens`, `n_deferred`. |
| 7 | Two fresh stage 01 runs give identical hashes | Blocked | Rebuilds wait on the labeled exclusion list and on the StrongREJECT category column. |
| 8 | `acceptance.json` passes and the stage 01 manifest shows `dirty: false` | Blocked | Stage 01 was not rerun. |
| 9 | `prereg/gate1_memo.md` is unchanged by the build; memo edits are their own commit | Pass | Memo edits are `efaa648`. Stage 01 writes `prereg/gate1_memo.draft.md` and does not write `prereg/gate1_memo.md`. The build has not been run. |

## 4. Audit numbers

Drawn from the existing `wildchat_filtered` checkpoint (628,786 rows) with containment 0.5. The key SHA-256 is `0212e5849ab45be924956817ee8625ec942cb8d1fc10789e1c43805a5268817b`. The sheet has 230 rows and was not opened for labeling.

- Pool size: 628,786.
- Flagged: 1,021. By destination cell: ordinary 645, non_english 303, python 66, code 7.
- Filter precision: not available. The 30 flagged rows are unlabeled.
- Residual rate: not available. The 200 random rows are unlabeled.
- Decision rule: none. `--score` has not been run.

Counts are in `prereg/audit/ordinary_audit_filter.json`.

## 5. Rebuild

Not run. There are no run-1 or run-2 hashes and no realized counts from this branch.

## 6. Owner actions left

1. Commit `prereg/rubric.md`. It is not in the repo.
2. Label `prereg/audit/ordinary_audit_sheet.csv`. Each `label` is harmful, benign, or unclear. Leave the key closed until that is done.
3. Run `scripts/01b_ordinary_audit.py --score`. That writes `prereg/ordinary_exclusions.csv` and `prereg/audit/ordinary_audit_result.json`.
4. Give StrongREJECT a `category` column, or amend the rule. The pinned validation files have `clf_label`, `proxy_clf_label`, `instructions`, `content`, `answer_prompt`, `gen_target`, `proxy_gen_target`. The normalizer stops rather than inferring a category from the text. The same columns are on the `keywords` and `violence` configs.
5. After the exclusion list exists, run the two clean rebuilds in the handoff. Move `checkpoints/01` aside first. Confirm `diff` of the two hash files is empty, `acceptance.json` passes, and the manifest has `dirty: false`.
6. Fill section 17 and the realized-count table from that rebuild, then commit and tag `gate1-prereg`. The agent does not tag.

## 7. Deviations from this doc

- Held-out C's code cell is sampled from the cleaned WildChat pool before the non-code cut. The previous code dropped every `is_code` row before that sample, so the code cell was empty. The handoff describes held-out C as non-English and non-Python code, and the filter test requires that cell to reach its target.
- Flagged Python rows are counted as `python`. With `domain_slice: 0` they are not eligible for ordinary, non-English, or the non-Python code cell.
- The audit draw uses the saved `wildchat_filtered` parquet. That is the pool after `_filtered_wildchat`. Rebuilding it would rerun minhash because `build.py` changed.
- Stage 01 writes its draft to `prereg/gate1_memo.draft.md`. Per-role length caps, dirty-tree refusal, and HarmBench removal are named as rebuild-commit work specified elsewhere. They are not in this handoff, so they are not implemented.
- The contingency sweep stays off. No dataset text was sent to a judge.
