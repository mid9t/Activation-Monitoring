# Activation Monitoring: Five-Week Project Plan

Do epistemic deferral signals beat the optimal score band in a probe-to-LLM cascade?

October 12 – November 13, 2026 · Prepared October 10, 2026

## Goal

By Friday, November 13, complete and write up a pre-specified test, on Llama-3-8B-Instruct with public data, of whether any epistemic deferral signal lowers the error of a probe-to-LLM cascade below DeepMind’s optimal two-threshold score band on held-out distribution shifts. Release the harness with it.

Either outcome is a contribution. A win is new evidence for a better deferral rule. A null is an independent open-weights replication of the cascade with larger error sets, plus a measured ceiling on what any better rule could gain. DeepMind asks for follow-up on open weights and open data, and its own cascade comparisons had mostly overlapping confidence intervals.

## Setup, fixed before week 1

- **Model and task:** Llama-3-8B-Instruct, the model Bailey et al. monitored and released jailbreak sets for. The task is input monitoring for harmful requests.

- **First stage:** logistic regression on last-prompt-token activations at one middle layer. Hold back a stronger probe (a small MLP over several layers) for the week-4 robustness check.

- **Second stage:** an open guard model, or the 8B model prompted as a classifier, run on every item.

- **Data roles:**

  - **Train:** WildJailbreak’s direct harmful and direct benign requests, plus ordinary chat.

  - **Tune shift:** WildJailbreak’s jailbreak-style harmful and benign cells. All fitting and Gate 2 happen here.

  - **Held-out shifts**, 20,000 items or more each where the data allows:

    - (A) public jailbreak templates applied to held-out harmful requests, the recipe DeepMind used for its jailbreak set, plus Bailey’s released jailbreaks;

    - (B) benign-but-harmful-sounding prompts from a separate source;

    - (C) new benign traffic such as code and non-English prompts.

- **Deferral signals:** margin (the baseline), bootstrap-ensemble disagreement, class-conditional density, and evidential vacuity. Token count and activation norm are controls.

- **Metric:** weighted cascade error using DeepMind’s weights (50 for false alarms on ordinary traffic, 5 for misses, 2 for false alarms on hard negatives), at 5% deferral with 10% as a secondary point. Fix thresholds on the tune shift and report the realized deferral rate on each held-out shift.

- **Out of scope:** adaptive attacks, long context, output monitoring, a second model. These were cut from the earlier 8–10 week scope.

Compute is small: an estimated one to two GPU-days on a single card for caching and LLM labels.

## Schedule

| Week      | Work                                                                                                                                | Ends with  |
|---------------|-----------------------------------------------------------------------------------------------------------------------------------------|----------------|
| 1 (Oct 12–16) | Cache activations, train the probe, get LLM labels for every item, fit the optimal band, count headroom, hand-check 50 confident errors | Gate 1         |
| 2 (Oct 19–23) | Compute all signals on the tune shift; run the exchange test                                                                            | Gate 2         |
| 3 (Oct 26–30) | Fit at most two deferral rules on the tune shift; confound and stability checks                                                         | Freeze         |
| 4 (Nov 2–6)   | Open the held-out shifts once; paired bootstrap; repeat with the stronger probe                                                         | Final decision |
| 5 (Nov 9–13)  | Write up; release code, cached features and harness                                                                                     | Ship           |

## Gates and decision rules

### Gate 1 (Oct 16): is there anything to gain?

This uses margin only. On held-out shifts, only baseline counts are allowed at this point.

- **Pass** needs both:

  - (a) the LLM classifier has lower weighted error than the probe on at least two shifted sets;

  - (b) at 5% deferral, at least 200 confident probe errors that the LLM would fix remain on one held-out shift, and they make up a quarter of the cascade’s remaining errors after discounting the label-error rate from the hand check.

- **Fail (a):** swap the second stage once, allowing two days. If it still fails, apply kill criterion 1.

- **Fail (b):** skip weeks 2–3 and take the null branch. Report the headroom and the open replication, and spend the spare time on the stronger probe and a second LLM classifier.

### Gate 2 (Oct 23): does any signal find those errors?

Tune shift only.

- **Pass:** for at least one non-control signal, both of these hold:

  - the confident items in its top 1% of traffic have a higher net fix rate (LLM fixes minus LLM breaks) than the least useful 1% inside the band, with the bootstrap interval for the difference above zero;

  - it beats token count on the same test.

- **Fail:** take the null branch, with the per-signal exchange rates as part of the result.

### Freeze (Oct 30)

Before opening held-out data, write down the rules, budget, weights, shifts and success test. Carry forward at most two rules: a swap rule on the best signal, and a margin-plus-signal rule.

### Final decision (Nov 6)

- **Positive:** the frozen rule lowers weighted cascade error against the optimal band at 5% deferral, with a 95% paired-bootstrap interval excluding zero, on at least two held-out shifts, and this holds with the stronger probe.

- **Partial:** it wins on one shift only, or only with the weak probe. Report it as shift-specific with no general claim.

- **Null:** anything else. Report the intervals and the headroom.

## Kill criteria

1.  **No usable second stage** after one swap. Stop the cascade study and switch to the density audit (the four WildJailbreak cells plus benign shift) on the same activations.

2.  **Label noise.** More than 15 of the 50 hand-checked confident errors are mislabeled, and relabeling isn’t feasible in three days.

3.  **Schedule.** No optimal band and headroom count by Wednesday, October 21. Cut to one held-out shift, drop the evidential arm, and skip the stronger probe.

4.  **Per signal.** Drop any signal that fails to beat token count, or whose rank correlation with a kept signal exceeds 0.95.

5.  **Scope.** Add nothing from the out-of-scope list, whatever the results suggest.

## In production

- **What it becomes:** a deferral policy, a small deterministic function of probe-side features, for a probe-first cascade. DeepMind reports deploying misuse-mitigation probes in user-facing Gemini.

- **Who would use it:** lab safety teams, and anyone self-hosting an open-weights model with a guard model. In DeepMind’s results the cascade called the LLM less than 10% of the time.

- **Cost:** it reuses activations the model already computes and adds a few small heads or one distance computation per request.

- **Rollout path:**

  1. Shadow mode: log what each rule would defer and act on nothing.

  2. Set thresholds to a deferral budget on recent traffic and track the realized deferral rate; a jump signals drift.

  3. Send a sample of high-uncertainty items to human review and feed the labels back into probe training, the loop Anthropic describes for anomaly detectors.

  4. Only then let the rule route requests.

- **What a deployer must redo:** validation on their own probes and traffic, including long-context and multi-turn inputs.

- **Design constraints:**

  - Keep the rule deterministic, since DeepMind notes that randomized policies invite repeated-query attacks.

  - Treat it as a cost and coverage component, not a security boundary: Bailey et al. showed that attacks trained against a Gaussian detector also fool input-based probes.

- **If the result is null:** deployers can keep the two-threshold band with evidence behind it, and the headroom figure tells them how much a better rule could ever buy.

## Sources

- Kramár et al., [Building Production-Ready Probes For Gemini](https://arxiv.org/abs/2601.11516)

- Bailey et al., [Obfuscated Activations Bypass LLM Latent-Space Defenses](https://arxiv.org/html/2412.09565)

- Jiang et al., [WildTeaming at Scale (WildJailbreak)](https://arxiv.org/abs/2406.18510)

- Anthropic, [Recommendations for Technical AI Safety Research Directions](https://alignment.anthropic.com/2025/recommended-directions/)
