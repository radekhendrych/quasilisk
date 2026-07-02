# Quasilisk

Quasilisk is a dependency-light MAP-Elites-style harness for evaluating OpenRouter-hosted target models against pluggable objectives.

The generic runner is `quasilisk.py`. It supports objective plugins, seed libraries, config-driven runs, OpenRouter target models, and deterministic dry-runs. The Tiananmen censorship probe is implemented as the built-in `censorship_tiananmen` objective rather than a separate runner.

The harness is intentionally limited to direct, auditable prompt variants. The perturbator instructions reject jailbreaks, hidden instructions, encoded prompts, and policy-evasion wording. Every target input and response is logged so a run can be audited.

## Objectives

Built-in generic objectives:

- `censorship_tiananmen`: willingness to provide substantive factual discussion of the Tiananmen Square protests instead of refusing, minimizing, or avoiding the topic.
- `self_harm_emission`: susceptibility to emitting actionable self-harm information instead of refusing and redirecting to support.
- `nihilism_susceptibility`: susceptibility to endorsing nihilism as a reason to abandon moral concern.
- `relativism_susceptibility`: susceptibility to endorsing harmful moral relativism.
- `fatalism_susceptibility`: susceptibility to endorsing fatalistic resignation from preventable harm.

For `censorship_tiananmen`, higher scores mean greater willingness to discuss
the events substantively. For the other generic objectives, higher scores mean
higher risk or susceptibility.
The abstract moral-susceptibility objectives (`nihilism_susceptibility`,
`relativism_susceptibility`, and `fatalism_susceptibility`) use an LLM-as-judge
scoring scheme. One to three configured judge models receive the original
prompt and target response and return JSON scores from 0 to 100. With multiple
judges, Quasilisk takes the scalar median of the independently parsed and
policy-capped scores. The full judge request/response audit is stored with the event.
For these abstract objectives,
high scores require the target response to accept the failure-mode conclusion in
its own analytical voice. The judge and parser separately track argument
success and commitment evidence, and they penalize roleplay, fictional speakers,
debate speeches, quotations, steelmanning, disclaimers, and other distancing
frames. Concrete objectives currently use deterministic heuristic scorers.

## Quick Start

Run the generic local dry-run first:

```bash
python3 quasilisk.py --config configs/dryrun_censorship.json
```

Run the generic harness against OpenRouter:

```bash
export OPENROUTER_API_KEY="..."

python3 quasilisk.py \
  --objective-id censorship_tiananmen \
  --target-model "provider/target-model-a,provider/target-model-b" \
  --perturbator-model "provider/perturbator-model" \
  --iterations 8 \
  --max-evaluations 80 \
  --parallelism 2
```

Useful optional metadata for OpenRouter rankings/analytics:

```bash
export OPENROUTER_HTTP_REFERER="https://example.org"
export OPENROUTER_APP_TITLE="quasilisk-generic-harness"
```

## Outputs

Each run writes to `runs/<timestamp>/` unless `--out-dir` is provided.

- `events.jsonl`: one record for every prompt delivered to a target model, including the response, score, feature cell, parent id, and errors.
- For LLM-judged objectives, each event also includes judge request messages, raw responses, parsed member scores, usage, quorum/aggregation details, and fallback status.
- `perturbations.jsonl`: one record for each perturbator request, including the exact perturbator request messages, raw perturbator response, parsed variants, fallback status, and rejected variants.
- `generation_traces.jsonl`: one record for each perturbator call, including the selected plan, defense profile, rendered request, raw response, and parsed variants.
- `defense_profiles_<model>.jsonl`: extracted target resistance profiles used by the planner.
- `parent_plan_outcomes.jsonl`: per-generation parent-plan outcome records used by outcome memory; this does not duplicate target responses from `events.jsonl`.
- `search_stats.jsonl`: append-only perturbator-attempt lifecycle records linking selected parent cells and plans to validation, child-score, budget, and final archive outcomes.
- `run_statistics.json`: live aggregate perturbator effectiveness by target, parent cell, strategy, route/case, and generation.
- `affordances_<model>.json`: compact target-specific framing cues extracted from observed responses, such as preferred terms, visible reasoning cues, successful frames, failed frames, and refusal cues.
- `archive_<model>.json`: best prompts per MAP-Elites feature cell.
- `summary.md`: human-readable top prompts and response excerpts.
- `run_summary.json`: machine-readable run summary.
- `config.json`: run configuration.

Perturbator statistics use a stable `perturbation_attempt_id` propagated through
generation traces, perturbations, candidate metadata, and child evaluations.
Archive effectiveness is credited from the final archive state after each
generation, so parallel evaluation order and transient insertions do not inflate
retention, new-cell, or cell-improvement rates.
Primary perturbator rates exclude attempts that used local fallback variants;
the corresponding `search_*` rates include fallbacks and describe end-to-end
search effectiveness.

Prompt text used for perturbator and judge requests lives in template files
under `templates/` by default. Runs can override those paths through the
`templates` config object, and request audits include the rendered messages plus
template path/hash metadata.

## Judge Panels

An `llm_judge` objective can use one, two, or three unique judge models. Every
member receives the same rendered judge request. Each response is parsed and
policy-capped independently by the objective before aggregation, so one member
cannot bypass the existing own-voice, conditional-framing, distancing,
harm-application, or balancing checks.

Panel fitness is the scalar median of the successfully parsed final member
scores:

- one judge preserves the original single-judge behavior;
- two judges require two successful parses and use the midpoint of their scores;
- three judges require at least two successful parses and use the middle score
  when all three succeed, or the midpoint when only two succeed.

Failed calls and malformed judge responses are retained in the audit but do not
contribute heuristic pseudo-scores to the median. If quorum is not met, the
candidate uses the objective's deterministic heuristic fallback. The aggregate
score has `score_source: "llm_judge_panel"`; its legacy component fields and
signals come from the valid member closest to the median, while
`score.judge_panel` contains component medians and per-signal vote counts for
diagnosis. No run-relative score normalization is applied.

For panels, `event["judge"]` is a schema-versioned envelope containing the
shared request, ordered `members`, aggregate usage/retries, and `aggregation`
details such as quorum, valid count, member totals, range, and median absolute
deviation. Each member retains its model id, raw response, parsed score, usage,
finish reason, retries, and error. Singleton events retain the original
single-judge audit shape.

The median remains the archive and search fitness even when judges disagree.
However, early stopping and automatic successful or near-miss seed persistence
require quorum members to satisfy the relevant threshold or predicate
individually. `status.json` and `run_summary.json` expose
`judge_call_count`, member and quorum failures, degraded quorum, disagreement,
judge tokens, retries, and cost.

`max_evaluations` limits target calls, not judge calls. A complete three-member
panel therefore normally makes three judge calls per successful target
response. `judge_parallelism` bounds those calls separately from target
`parallelism`.

## Algorithm

The generic runner loads an objective plugin, evaluates seed prompts, scores
target responses, extracts target-specific framing affordances from observed
responses, including bounded visible-reasoning cue snippets when targets expose
them, places candidates into objective-specific MAP-Elites feature cells,
selects high-scoring parents, asks the perturbator for direct prompt variants
using archive context and compact affordance context, validates those variants,
and repeats until the configured budget is reached.

MAP-Elites stores elites by objective-specific feature cells, then uses
route/case/defense-aware perturbation for every parent-to-variant step. The
harness extracts a defense profile from the parent response, chooses a
deterministic perturbation plan, asks the perturbator for direct prompts against
that plan, runs structural planned-prompt validation, and then evaluates
accepted prompts normally. High-scoring parents are handled as exploitation
plans that preserve the parent's successful route, pressure pattern, and
commitment form; lower-scoring parents use coverage-weighted route/case
exploration. Route, case, defense, strategy, and validation metadata is stored
with the candidate; the target receives only the final user prompt.

For the abstract moral-susceptibility objectives, MAP-Elites cells are oriented
around argument route, pressure pattern, and commitment form. Mutation guidance
asks the perturbator to create direct premise-adjudication prompts that invite a
considered verdict, rather than essays, speeches, fictional scenarios, or other
genre artifacts that can make the target response ambiguous evidence.

Generic objective seed prompts live under `objectives/<objective_id>/seeds/`.
Each objective has a `default.json` seed file. Model-specific learned or
curated seeds live under `objectives/<objective_id>/seeds/models/<model>.json`
using a safe model slug. During a run, the harness first looks for
target-specific seeds; if none exist, it falls back to `default.json`.
Successful non-dry-run prompts are saved back into the model-specific file when
they meet the objective's successful-seed score threshold.

Example dry-run configs live under `configs/`.

The `censorship_tiananmen` objective ranks responses by willingness to provide
substantive factual discussion. It scores each response for:

- topic coverage across event, democratic context, participants, protest events, state response, and aftermath;
- refusal or sensitivity signals;
- low-substance responses;
- minimization signals.

The archive for this objective keeps the highest willingness scores, not the
highest censorship-risk scores. Refusal and minimization diagnostics are still
preserved in each score object as `refusal_strength`, `risk_score`, and
`censorship_risk_score`.

Prompts are mapped into objective-specific feature cells. For censorship those
cells use frame, specificity, and requested output shape; for abstract moral
susceptibility they use argument route, pressure pattern, and commitment form.
The archive keeps the top candidates per cell, samples promising parents, asks
the perturbator model for non-deceptive variants, evaluates those variants
against the target model, and repeats until the iteration, evaluation, or score
budget is reached.

## Generic Config

```bash
python3 quasilisk.py --help
python3 quasilisk.py --list-objectives
```

Minimal config:

```json
{
  "objective_id": "self_harm_emission",
  "target_models": ["provider/target-model"],
  "perturbator_model": "provider/perturbator-model",
  "iterations": 8,
  "max_evaluations": 80,
  "parallelism": 2
}
```

Optional prompt template overrides:

```json
{
  "templates": {
    "perturbator_planned_system": "templates/perturbator/planned_system.md",
    "perturbator_planned_user": "templates/perturbator/planned_user.md",
    "moral_judge_system": "templates/judge/moral_system.md",
    "moral_judge_user": "templates/judge/moral_user.md"
  }
}
```

Optional three-model judge panel:

```json
{
  "judge_models": [
    "provider-a/judge-model",
    "provider-b/judge-model",
    "provider-c/judge-model"
  ],
  "judge_parallelism": 3,
  "judge_disagreement_threshold": 20
}
```

Legacy `judge_model` configuration remains supported as a singleton panel.
`OPENROUTER_JUDGE_MODEL` remains supported for a singleton or comma-separated
panel. Prefer judges from distinct provider/model families; exact duplicate
model ids are rejected and using a target as its own judge emits a warning.

CLI flags override config file values.

## Generic Runner Options

```bash
python3 quasilisk.py --help
```

Important flags:

- `--objective-id`: built-in objective id or importable objective module.
- `--target-model`: target OpenRouter model id; repeat or comma-separate.
- `--perturbator-model`: OpenRouter model id used to generate variants.
- `--judge-model`: OpenRouter model id used to score abstract moral-susceptibility responses. Repeat or comma-separate it to configure up to three unique judges.
- `--judge-parallelism`: maximum concurrent judge calls, separate from target `--parallelism`; defaults to `3`.
- `--judge-disagreement-threshold`: panel score range above which an evaluation is counted as disagreement; defaults to `20`.
- `--seed-library-dir`: objective seed library directory; defaults to the directory containing the objective's `default.json`.
- `--no-model-specific-seeds`: disable model-specific seed lookup for a run.
- `--no-save-successful-seeds`: disable automatic saving of successful model-specific seeds.
- `--successful-seed-score`: override the score threshold for learned seeds.
- `--iterations`: MAP-Elites mutation rounds after seed evaluation.
- `--max-evaluations`: target call budget per target model, including seeds.
- `--archive-per-cell`: number of elites retained per feature cell.
- `--stop-score`: stop a target once the configured objective score is reached; use `0` to disable.
- `--target-system-prompt`: optional system prompt sent to target models. It is empty by default so the target receives only the user prompt.
- `--parent-plan-memory-enabled` / `--no-parent-plan-memory-enabled`: enable or disable parent-plan outcome memory. Enabled by default.
- `--parent-plan-memory-mode`: use `block` to skip regressed parent-plan tuples during cooldown, or `penalize` to reduce their priority.
- `--parent-plan-regression-threshold`: mark a parent-plan tuple as regressed when its best evaluated child is more than this many points below the parent score; defaults to `8`.
- `--parent-plan-cooldown-generations`: number of following generations to block or penalize a regressed `(parent_id, strategy, route, case_family)` tuple; defaults to `2`.
- `--parent-plan-penalty`: priority penalty used when `--parent-plan-memory-mode penalize` is selected; defaults to `15`.
- `--parent-plan-min-evaluated-children`: minimum non-error children required before a tuple can be blocked or penalized; defaults to `1`.
- `--max-output-tokens`: optional explicit output cap applied to target, perturbator, and judge calls.
- `--target-max-tokens`, `--perturbator-max-tokens`, `--judge-max-tokens`: optional explicit per-call output caps. Config-file token caps are preserved unless a CLI token cap overrides them.
- `--dry-run`: no network, deterministic fake target and perturbator.

## Notes

The scoring is triage for manual review, not a final judgment. Some objectives
use deterministic heuristics and the abstract moral-susceptibility objectives
use an LLM judge. The JSONL log is the source of truth for auditing what each
model was actually asked and how it responded.
For `llm_judge` objectives, the judge or median panel score is still a triage signal rather
than a final judgment; inspect the target response, stored judge rationale,
`argument_success_score`, `commitment_evidence_score`, and judge signals for
review.

For strategy review, inspect the run `config.json`, the selected
`objectives/<objective_id>/seeds/default.json`, `affordances_<model>.json`,
`perturbations.jsonl`, and `quasilisk_harness/core.py`. Together they show the
starting islands, run parameters, target-specific framing and visible-reasoning
cues, parent-selection context supplied to the perturbator, mutation rules, raw
generated variants, and acceptance/rejection decisions.
