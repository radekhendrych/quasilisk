# Quasilisk failure report

## Executive summary

Quasilisk’s central failure is that it treats a single scored response as a reliable estimate of prompt quality. It then stores unusually lucky responses as permanent elites, optimizes against those inflated scores, and does not correct the archive when exact-prompt replays or paired rejudging contradict them.

In plain terms: **Quasilisk is optimizing observations, not prompts.**

Across the four newest top-level run batches—six objective histories, five of which generated children—I counted 137 genuine perturbator children from the raw `events.jsonl` audit logs:

- 23 improved: **16.8%**
- 19 retained/tied: **13.9%**
- 95 regressed: **69.3%**
- Therefore **83.2% failed to improve**
- Mean recorded comparison delta: **−11.54 points**

Compared strictly with the originally logged parent, rather than a later replay/rejudge baseline, only 18/137 children improved and mean delta was **−24.75**.

The decisive evidence is the replay control:

- 22 exact-prompt replays
- 18 classified as regressions, three ties, one improvement
- Mean delta: **−33.86**
- Mean absolute movement: **34.41 points**
- Range: **−80 to +5**

An unchanged prompt should not systematically lose. This strong negative skew occurs because Quasilisk selects one-shot high scores and then observes regression toward the mean. Worse, the lower replay does not demote the stale elite.

There are also genuine mutation-quality failures:

- 41 optimization children produced only nine improvements; **78.0% failed to improve**.
- `answer_shape_microedit` went **0/11**, with mean delta **−36.27**.
- 70.1% of evaluated children were discovery operations, despite a nominal 4:1 optimization weight.
- Planner intent, task contracts, scheduling, and operator memory are internally misaligned.
- The validator establishes structural compliance, not likely target efficacy.

The project is salvageable, but current best-score and success claims are not decision-grade. The scoring/search-state problem should be fixed before spending more on perturbator models or broader production runs.

## Scope and method

I treated the four newest top-level run roots as the requested history:

1. `20260723_multi_model`
2. `20260722_qwen37plus_glm52_current_5601e7c`
3. `20260722_mistralnemo_glm52_current_5601e7c`
4. `20260722_mistralnemo_glm52_v2p3_fix_retry`

The multi-model root contains three objective histories, so the audit covers six leaf histories:

| History | State | Children | Improved / tied / regressed | Failed to improve |
|---|---|---:|---:|---:|
| [July 23 relativism](/home/hendrychr/quasilisk/runs/20260723_multi_model/relativism_susceptibility/events.jsonl) | Complete | 45 | 10 / 1 / 34 | 77.8% |
| [July 23 fatalism](/home/hendrychr/quasilisk/runs/20260723_multi_model/fatalism_susceptibility/events.jsonl) | Complete | 48 | 11 / 7 / 30 | 77.1% |
| [July 23 nihilism](/home/hendrychr/quasilisk/runs/20260723_multi_model/nihilism_susceptibility/events.jsonl) | Complete | 20 | 0 / 1 / 19 | 100% |
| [July 22 Qwen nihilism](/home/hendrychr/quasilisk/runs/20260722_qwen37plus_glm52_current_5601e7c/nihilism_susceptibility/events.jsonl) | Partial, 27/30 evaluations | 17 | 1 / 6 / 10 | 94.1% |
| [July 22 Mistral current](/home/hendrychr/quasilisk/runs/20260722_mistralnemo_glm52_current_5601e7c/nihilism_susceptibility/status.json:4) | Seed-only stop at 90 | 0 | — | — |
| [July 22 Mistral retry](/home/hendrychr/quasilisk/runs/20260722_mistralnemo_glm52_v2p3_fix_retry/nihilism_susceptibility/events.jsonl) | Complete | 7 | 1 / 4 / 2 | 85.7% |
| **Total child-producing histories** |  | **137** | **23 / 19 / 95** | **83.2%** |

I counted raw event records where:

```text
source == "planned_perturbator"
metadata.variant_source == "perturbator"
```

Harness replay controls were excluded from the child denominator. Raw events are authoritative because the Qwen run is partial and its last aggregate statistics snapshot predates its last events.

## The exact causal failure

The observed loop is:

> One-shot score outlier → stored as elite → selected repeatedly → child and replay responses regress toward the mean → child labeled a failure → narrowly keyed memory blocks one tuple → scheduler route-hops around the same stale elite → repeat.

### 1. Elite scores are upward-biased and effectively irreversible

The archive sorts candidates using their single observed total score. If the same prompt is seen again, it is replaced only when the new result is higher—not when a lower replay demonstrates that the old result was unreliable. See [`MapElitesArchive.add`](/home/hendrychr/quasilisk/quasilisk_harness/core.py:204).

The clearest example is Meta Llama fatalism:

- A parent remained stored and selected at score 80 in generations 1–3: [generation 1](/home/hendrychr/quasilisk/runs/20260723_multi_model/fatalism_susceptibility/search_stats.jsonl:30), [generation 2](/home/hendrychr/quasilisk/runs/20260723_multi_model/fatalism_susceptibility/search_stats.jsonl:33), [generation 3](/home/hendrychr/quasilisk/runs/20260723_multi_model/fatalism_susceptibility/search_stats.jsonl:38).
- Exact-prompt controls scored 10, 0, and 10.
- One exact replay fell by 80 points in [events.jsonl](/home/hendrychr/quasilisk/runs/20260723_multi_model/fatalism_susceptibility/events.jsonl:79).

This is classic winner’s curse: a maximum selected from noisy observations is likely to be an overestimate.

### 2. Paired rejudging does not repair the canonical state

Paired rejudging re-scores the already stored parent and child responses. It does not call the target model again, so it cannot measure target-response variance.

More seriously, its revised scores are written into metadata and used for the local comparison, but the canonical evaluation score used for archive insertion, parent selection, best-score reporting, and persistence is not replaced. The relevant path begins in [core.py](/home/hendrychr/quasilisk/quasilisk_harness/core.py:2564).

This creates several simultaneous “truths”:

- Search-state score: original observation
- Comparison outcome: possibly paired-rejudged scores
- Lifecycle score: original score
- Archive eligibility: original total
- Parent-plan memory: original parent score

Consequently, a child can be reported as regressed while still entering the archive, or a parent can collapse on remeasurement while remaining a high-scoring elite.

### 3. Early stopping has the same defect

A run stops when an archive elite reaches the threshold and the judge panel supports that score. But the panel judges the same stored target response; it is not evidence that the prompt reproduces the result across target calls. See [confirmed-success logic](/home/hendrychr/quasilisk/quasilisk_harness/core.py:2365) and [the stop check](/home/hendrychr/quasilisk/quasilisk_harness/core.py:3073).

This caused the current Mistral nihilism run to stop at generation zero after one seed scored 90, producing no perturbations. Two July 23 nihilism target lanes stopped similarly. A lucky seed can therefore disable the entire search.

## Search-policy and mutation failures

### Most target budget is not local optimization

Of the 137 evaluated children:

| Purpose | Children | Improved | Tied | Regressed |
|---|---:|---:|---:|---:|
| Discovery | 96 | 14 | 14 | 68 |
| Optimization | 41 | 9 | 5 | 27 |

Discovery accounts for **70.1%** of evaluated children. Some discovery children are intended to reach new MAP-Elites cells rather than beat their parents, so judging all children solely by parent improvement is conceptually unfair.

But this does not explain away the failure: **32/41 optimization children also failed to improve**.

The configuration nominally weights optimization 4:1, but it also uses three offspring per parent and two variants per task. See [the July 23 configuration](/home/hendrychr/quasilisk/runs/20260723_multi_model/fatalism_susceptibility/config.json:29) and [scheduler/refinement settings](/home/hendrychr/quasilisk/runs/20260723_multi_model/fatalism_susceptibility/config.json:44). The scheduler weights the ordering of available tasks; it does not enforce a 4:1 allocation of evaluated target calls. See [task scheduling](/home/hendrychr/quasilisk/quasilisk_harness/perturbation/scheduler.py:194) and [weighted ordering](/home/hendrychr/quasilisk/quasilisk_harness/perturbation/scheduler.py:505).

Weak parents receive discovery-only portfolios. For high-signal parents, the committed task order commonly selects surface style, answer shape, and coverage, while deferring case detail.

### The operator portfolio is empirically poor

| Operator | N | Improved | Tied | Regressed | Mean delta |
|---|---:|---:|---:|---:|---:|
| `surface_style_microedit` | 23 | 7 | 4 | 12 | −1.04 |
| `case_detail_microedit` | 7 | 2 | 1 | 4 | −5.71 |
| `coverage_exploration` | 27 | 7 | 3 | 17 | −14.37 |
| `low_signal_exploration` | 69 | 7 | 11 | 51 | −10.58 |
| `answer_shape_microedit` | 11 | **0** | **0** | **11** | **−36.27** |

The answer-shape operator should have been disabled globally. Instead, operator memory is keyed by exact parent fingerprint, operator, factor, and contract. See [the operator key](/home/hendrychr/quasilisk/quasilisk_harness/perturbation/operator_ledger.py:19). A failure on one parent therefore does not generalize to the next.

Additionally, the ledger learns only from target-evaluated children. Operators whose outputs are repeatedly rejected by validation never accumulate regression evidence.

### Planner intent is lost during task compilation

In committed `5601e7c`, the near-miss planner explicitly requests preservation of successful form while adding concrete harm application. The portfolio then overwrites that semantic mutation scope with one-factor tasks whose contracts protect every other fingerprint factor—including `harm_application`.

Thus the optimizer’s intended repair can be illegal under its own validator. The scheduler frequently defers the closest semantic task anyway.

This is a verified planner-to-task integration defect. It should not, however, be used to explain every July 23 failure: the latest sample did not exercise that specific near-miss strategy often enough. It demonstrates that the intended repair architecture is internally unsound.

## Validation and refinement are solving the wrong problem

Across the audited histories:

- 330 perturbator outputs were produced.
- 279 passed validation.
- 154 were nominated.
- 137 were actually target-evaluated.
- There were no material perturbator fallbacks or target/judge transport failures.
- All 137 evaluated children passed both absolute and transition validation.

The system is therefore generally successful at obtaining structurally valid prompts. It is unsuccessful at predicting whether those prompts will elicit a better target response.

Refinement explicitly ranks proposals using validator and contract evidence, not predicted target efficacy. See [refinement.py](/home/hendrychr/quasilisk/quasilisk_harness/perturbation/refinement.py:1).

With two variants per task and at most one nominee, at least half of otherwise valid outputs are ranked out by construction. An output-level “majority improves” KPI is therefore mathematically impossible under that configuration even if every nominated child succeeds.

That denominator problem should be corrected, but it does not remove the real result: 83.2% of evaluated children and 78.0% of optimization children failed.

## The feedback loop does not learn the dominant response failure

Most failed target responses are not empty or explicit refusals. They are substantive balancing or resistance answers: the target rejects the risky premise, preserves responsibility, or reinstates a harm boundary.

The target-affordance tracker marks a frame as failed only for sufficiently strong refusal or an empty response. Nonempty zero-score balancing answers are not failures, while any score above zero can become “promising.” See [the update logic](/home/hendrychr/quasilisk/quasilisk_harness/core.py:438).

The Amazon nihilism tracker demonstrates the consequence: after 30 evaluations it still has no failed frames and no successful examples. See [its affordance state](/home/hendrychr/quasilisk/runs/20260723_multi_model/nihilism_susceptibility/affordances_amazon_nova-micro-v1.json:4).

Parent-plan memory is also narrow:

- Keyed by exact parent ID, strategy, route, and case
- Blocks only when the best child loses by more than eight points
- Does not demote an unreliable parent
- Route/case changes create fresh keys

It records local incidents rather than learning the general failure mode.

## Reporting obscures rather than resolves the problem

Quasilisk currently mixes four different notions of success:

1. Raw total-score improvement
2. Constraint-preserving parent improvement
3. Archive retention/new-cell coverage
4. Global-best improvement

A child is classified as regressed if its total falls or if any protected score component drops by more than five points. See [comparison.py](/home/hendrychr/quasilisk/quasilisk_harness/perturbation/comparison.py:45). Five audited children gained total score but were still classified as regressions because a component fell.

Meanwhile, the archive ranks total score within cells and can retain a parent-relative regression when it creates a useful new cell. This is legitimate MAP-Elites behavior, but it makes “archive yield” a poor proxy for mutation quality.

For example, the Amazon nihilism summary reports perfect delivery and validity plus 0.35 archive yield, while its best delta is zero and mean delta is −14.5. See [summary.md](/home/hendrychr/quasilisk/runs/20260723_multi_model/nihilism_susceptibility/summary.md:69).

## Historical cross-check

The earlier [latest-ten-runs report](/home/hendrychr/quasilisk/docs/perturbator_interaction_analysis_latest_10_runs.md:49) found only 14.7% of perturbator calls improved their parent. The current 16.8% rate is directionally consistent: the problem is chronic.

That report recommended parent-plan memory and preservation of successful high-score forms. Both were subsequently added, yet the failure persists because:

- Memory does not correct noisy parent scores.
- Memory keys are too local.
- Planner intent is not reliably compiled into executable contracts.
- Bad operators continue across different fingerprints.

The historical report’s exact 21/61 improved/worse counts are not completely reproducible from its stated method; a reaggregation gives 23/59 with the same tie and unmatched counts. No aggregation script was checked in. Its central conclusion remains supported, but its precise historical counts should be treated cautiously.

Older reports about category-labeled and conditional scaffolding describe a real historical defect, but current validation now rejects those forms. They are not the primary explanation for the latest runs.

## Recommendations

### P0 — Repair canonical scoring before further model tuning

- Treat prompt quality as an estimate over repeated target responses, not one response.
- Require at least three target repetitions before elite promotion and early stopping; use a robust aggregate and record uncertainty.
- Screen cheaply with one call, then replicate only candidates considered for promotion, persistence, or stopping.
- Compare child and parent using matched, contemporaneous repetitions.
- If replay evidence invalidates an elite, recalibrate or demote it immediately.
- Make paired rejudge update the canonical evaluation/archive score, not only metadata.

Acceptance gate: an exact-prompt replay must update the stored estimate; a single 90-point response must not stop a run; archive, comparison, lifecycle, and report must expose the same canonical score.

### P0 — Separate optimization from discovery

- Maintain explicit target-call budgets for optimization and discovery.
- Enforce the configured ratio on evaluated calls, not merely task ordering.
- Report discovery by coverage/new-cell success and optimization by replay-adjusted parent improvement.
- Consider separate optimization and exploration archives so weaker discovery children do not silently become hill-climbing parents.

Acceptance gate: run statistics must show the requested allocation and must not combine coverage yield with optimization success.

### P0 — Fix plan-to-contract compilation

- Compile planner intent into editable factors instead of overwriting it with a generic one-factor scope.
- Add an invariant test: every selected strategy must produce at least one validator-legal mutation capable of performing its stated repair.
- Prioritize evidence-directed response/case repair over cosmetic answer-shape changes.

Acceptance gate: a “preserve form, add harm application” plan must generate, validate, schedule, and evaluate an actual harm-application edit.

### P0 — Disable and generalize learning from failed operators

- Keep `answer_shape_microedit` disabled until it demonstrates positive control-adjusted lift.
- Count repeated validator-invalid outputs as operator failures.
- Add hierarchical memory: global operator, operator/failure-class, then exact fingerprint.
- Feed evaluated balancing/resistance outcomes into affordance and rejection memory.
- Allow feedback between parent attempts instead of evaluating an entire generation before learning.

Acceptance gate: an operator with 0/11 outcomes cannot continue merely because the next parent has a different fingerprint.

### P1 — Fix refinement economics and executive metrics

- Use one variant per task until the pre-target ranker proves predictive lift.
- If multiple variants are retained, evaluate enough of them to measure whether ranking actually selects the better candidate.
- Publish separate counts for generated, valid, nominated, evaluated, raw-total improved, constraint-valid improved, archive retained, and global-best improved.
- Check in the aggregation script behind every cross-run report.

The July 23 search lanes consumed approximately **1.55 million perturbator tokens** before target/judge costs. This makes denominator and nomination efficiency financially material.

### P1 — Add tests that can expose the real failure

Add:

- A stochastic target simulator with controllable response variance
- An archive test proving that lower replay evidence recalibrates an inflated elite
- An early-stop test requiring repeat-confirmed success
- A plan → task → contract → validator integration test
- Tests for globally repeated invalid/regressing operators
- A small replicated live canary across the same objective/model lanes

The current dirty worktree passes all **170 unit tests**, but those tests do not require paired rejudging to replace the canonical archive score and do not test whether planner intent is executable. The green suite is evidence of an integration-test gap.

## Current WIP and management conclusion

The uncommitted worktree appears to contain sensible partial mitigations: removing the blanket answer-shape operator, reducing variants per task, adding response-contract repair, and adding coverage-novelty memory. These changes postdate the audited runs and have not been validated by a real repeated-target canary. They also do not, by themselves, fix stale elite scores, target variance, early stopping, or canonical-state inconsistency.

There is also a provenance problem: the July 22 folder names identify commit `5601e7c`, but the July 23 artifacts do not record a commit or dirty-state manifest. Future runs should record the commit, dirty diff hash, templates, and seed-content hashes.

The appropriate executive decision is:

> Pause claims of perturbator efficacy and large-scale model spend. First make prompt scores reproducible and canonical, then repair the plan/scheduler/operator loop, and only afterward compare perturbator models.

The perturbator occasionally finds genuine-looking gains, so the project is not devoid of signal. The present system simply cannot reliably distinguish a reproducible improvement from a lucky target response, nor learn efficiently from the failures it already observes.

No repository files or audit logs were modified during this analysis.
