# Activation Monitoring

An open-weights, public-data follow-up to Kramár et al., [Building Production-Ready Probes for Gemini](https://arxiv.org/abs/2601.11516) (DeepMind, 2026).

The harness is the `amon` package. It trains a linear activation probe on Llama-3-8B-Instruct, scores every prompt with Llama Guard 3, and tests whether any signal other than the probe score can choose a better set of prompts to send to the guard. The baseline it has to beat is the two-threshold cascade from that paper, fit at a fixed 5% deferral budget.

The decision rules are pre-registered in [`prereg/gate1_memo.md`](prereg/gate1_memo.md). The October schedule and the compute plan that produced this harness are kept in [`docs/gate1_implementation_plan.md`](docs/gate1_implementation_plan.md).

## What DeepMind showed

Kramár et al. monitor Gemini 2.5 Flash for cyber-offensive prompts. A small probe reads the model’s residual stream and classifies most requests on its own. Requests whose probe score falls between two thresholds go to a prompted Gemini classifier. Their cascade is

```text
score ≤ t0          → benign
t0 < score < t1     → asked of the language-model classifier
score ≥ t1          → harmful
```

which is their Equation 11. A symmetric band around the probe’s decision threshold is a simpler heuristic; the two free thresholds are the optimal rule of this form (Appendix E). On their cyber evaluation, deferring to Gemini 2.5 Flash about 8% of the time (under 10% in the text) beats both the probe alone and the prompted model alone, at roughly a fiftieth of the prompted model’s cost. The same work put probes of this kind into user-facing Gemini.

The error they minimize (Equation 12) weights a miss by 5, a false alarm on a hard negative by 2, and a false alarm on ordinary traffic by 50. Ordinary-traffic false alarms dominate because benign traffic is common and a false alarm there is expensive. Their cascade plots have wide intervals. The paper closes by inviting further work on the open-weights experiments in Appendix A, which re-run the probe architectures on Gemma-2 9B and public classification tasks. This repository takes the cascade itself onto a public harmful-request benchmark with an open model.

Their main engineering result is the probe. Mean-pooled linear probes lose accuracy when the context gets long, and they introduce MultiMax and max-of-rolling-means attention probes, plus an architecture search with AlphaEvolve, to close that gap. Long context, multi-turn conversations, and adaptive attacks are the shifts they treat as production-relevant. Jailbreaks stay hard for every method they try.

## What this repository tests

This study keeps their cascade and their weights, and asks a question they leave open. Once `t0` and `t1` are chosen to minimize weighted error at a fixed deferral budget, does any other quantity computed from the same activations pick a better 5% of prompts to send to the second stage?

The baseline rule is their score band: defer the prompts whose probe logit sits between `t0` and `t1`. The candidate rules replace that interval with a set of the same size chosen by another signal:

- **Ensemble disagreement.** Standard deviation of the logit across 20 logistic probes fit on grouped bootstrap resamples of the training set. High disagreement means defer.
- **Class-conditional density.** Minimum squared Mahalanobis distance to three Gaussians fit on the training activations (harmful, WildJailbreak-benign, ordinary chat), with one Ledoit–Wolf covariance shared across classes. Far from every training class means defer.

The probe logit itself is the margin, and it is the band. Token count and activation norm are controls a signal has to beat before it is kept. A kept rule is then frozen and scored once on held-out shifts. A win is a lower weighted cascade error than the band, with a paired-bootstrap interval that excludes zero. A null is an open replication of the cascade plus a count of how many confident probe errors the guard could still fix (the headroom). That count is the ceiling on what any better deferral rule could gain.

Held-out prompts are locked until that freeze. `amon.guard` refuses any read of a held-out split that asks for a non-baseline signal before `prereg/freeze.json` exists.

## Where the two studies differ

| | Kramár et al. | This study |
| --- | --- | --- |
| Model | Gemini 2.5 Flash | Llama-3-8B-Instruct, revision pinned in `configs/models.yaml` |
| Task | Cyber-offensive input monitoring | Harmful-request input monitoring, across hazard types |
| Probe | Several architectures. The linear baseline mean-pools token activations at a middle layer. The probes they select are MultiMax and max-of-rolling-means attention. | L2 logistic regression on the last prompt token, at one layer chosen from {8, 12, 16, 20, 24}, with layer 16 preferred on a tie. The cache also stores the mean over prompt tokens. |
| Second stage | Prompted Gemini 2.5 Flash, Pro, or Flash-Lite, parsed as a hard label | Llama Guard 3 8B. The score is the log-odds of unsafe versus safe, then a threshold. |
| Data | Internal Gemini traffic and expert-labeled cyber sets | Public sets only. See the table below. |
| Shifts the result is about | Long context, multi-turn, existing jailbreaks, adaptive red teaming | (A) jailbreak templates and released jailbreaks, (B) benign prompts that sound harmful, (C) code and non-English chat |
| Cascade | Equation 11. They sweep `(t0, t1)` and report operating points that call the language model on under 10% of queries. | The same rule. The decision uses a budget of 5% of tune-pool items. A 10% budget is reported beside it. |
| Weights | Equation 12: 5 / 2 / 50, as a weighted average of per-dataset rates | The same 5 / 2 / 50. Each item’s cost is its group’s weight divided by the group’s size, so a smaller group still contributes its full weight. |
| Where thresholds are fit | A validation mix. One frontier plot sweeps thresholds and reads the test set. | The tune pool only. Held-out splits are scored with thresholds that are already frozen. |
| Past the band | The band is the method. | Whether ensemble disagreement or density beats the band at the same budget. |

Two consequences of those choices are worth stating on their own.

The probe here is the linear probe from their Section 3.1.1, read at the last prompt token. Their MultiMax and attention probes answer a long-context question. Items longer than 2,048 tokens are dropped, so every prompt in this study stays short. Long context, multi-turn inputs, adaptive attacks, and output monitoring are the shifts their paper treats as production-relevant; the shifts here are jailbreaks, harmful-sounding benign prompts, and new ordinary traffic.

The second stage returns a probability, so it receives the same kind of weighted-error threshold as the probe. Their prompted classifiers return a parsed label, and the cascade uses that label directly.

The metric uses their weights and pools them differently. Equation 12 averages each dataset’s error rate and then weights those rates. Here a miss, a hard-negative false alarm, and an ordinary false alarm are three groups, and an item in a group of size `N` costs `w / N`. A group that came out smaller than planned still contributes its full 5, 2, or 50. On a held-out shift that contains only one class, the comparison is that class’s error rate.

## Data

Every label comes from the source dataset. A blinded hand check of confident errors estimates how often that label is wrong; those labels are used only to discount the headroom count.

| Role | What it is | Used for |
| --- | --- | --- |
| Train | WildJailbreak vanilla harmful and vanilla benign, plus ordinary English WildChat turns | Probe, ensemble, and density |
| Tune | WildJailbreak adversarial harmful and adversarial benign, plus a disjoint ordinary WildChat slice | Llama Guard threshold, the score band, and every deferral rule |
| Held-out A | 100 in-the-wild jailbreak templates applied to StrongREJECT requests, plus Bailey et al.’s released Llama-3 jailbreaks | Jailbreak shift. Harmful. |
| Held-out B | OR-Bench and the safe prompts from XSTest | Benign prompts that sound harmful |
| Held-out C | WildChat in languages and code styles held out of the tune slice | New ordinary traffic |
| Pipeline | Bailey et al.’s harmful and benign instruction splits | A mixed-class check that no decision uses |

Train and tune never share a source request. Near-duplicates across roles are removed with MinHash and with MiniLM embeddings. Ordinary WildChat rows that contain a jailbreak template are dropped before sampling, and a blinded audit of the remainder can exclude more. The build seed is `20261026`. About 228,000 items remain after the length drop.

## How a run proceeds

Data lives under `$AMON_DATA`, outside the repository. Model and dataset revisions are pinned in `configs/models.yaml` and loaded at that revision.

```bash
export AMON_DATA=~/amon-data
export HF_HOME=$AMON_DATA/hf
export AMON_DEVICE=cuda          # or mps
```

`make test` runs the unit tests. The numbered scripts in `scripts/` are the stages, and each one reads only the outputs of the stages before it.

| Stage | What it writes |
| --- | --- |
| `00_check_access` | Confirms the gated model and dataset licenses |
| `01_build_data` | One item table and the frozen split files |
| `01b_ordinary_audit` | The ordinary-traffic exclusion audit |
| `02_cache_activations` | Last-token and mean-pooled residual stream at layers 8, 12, 16, 20, and 24 |
| `03_score_llamaguard` | Log-odds for every item |
| `04_train_probe` | The logistic probe. Layer and `C` are chosen by grouped cross-validation on train. |
| `05_fit_thresholds` | Llama Guard’s threshold, the 5% and 10% bands, and the probe-only threshold, all on the tune pool |
| `06`–`09` | Headroom, the hand-check sample, and the Gate 1 decision |
| `10_fit_signals` | Ensemble and density on train, train-val, and tune. Runs only after Gate 1 passes, and the guard still blocks held-out splits. |

Gate 1 asks whether the cascade has anything left to gain: the guard has to beat the probe on shifted data, and one held-out shift has to contain enough confident probe errors that the guard would correct. Those two conditions, the discount for mislabeled hand-check items, and the rule for which shift the final comparison uses are specified in the [Gate 1 memo](prereg/gate1_memo.md). The memo also records the departures from each equation in Kramár et al.

## References

- Kramár et al., [Building Production-Ready Probes for Gemini](https://arxiv.org/abs/2601.11516). The cascade (Equation 11) and the weights (Equation 12) that this study reimplements on public data.
- Bailey et al., [Obfuscated Activations Bypass LLM Latent-Space Defenses](https://arxiv.org/abs/2412.09565). Source of the released Llama-3 jailbreaks, and the reason a probe is treated as a cost-saving stage rather than a security boundary.
- Jiang et al., [WildTeaming at Scale](https://arxiv.org/abs/2406.18510). WildJailbreak, the train and tune harmful and hard-negative cells.
- Cunningham et al., [Cost-Effective Constitutional Classifiers via Representation Re-use](https://alignment.anthropic.com/2025/cheap-monitors/) (2025). The earlier probe-to-language-model cascade that Kramár et al. cite.
