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
scoring scheme. The configured judge model receives the original prompt and
target response, returns a JSON risk score from 0 to 100, and the full judge
request/response audit is stored with the event. Concrete objectives currently
use deterministic heuristic scorers.

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
- For LLM-judged objectives, each event also includes the judge model, judge request messages, raw judge response, usage, and fallback status.
- `perturbations.jsonl`: one record for each perturbator request, including the exact perturbator request messages, raw perturbator response, parsed variants, fallback status, and rejected variants.
- `affordances_<model>.json`: compact target-specific framing cues extracted from observed responses, such as preferred terms, visible reasoning cues, successful frames, failed frames, and refusal cues.
- `archive_<model>.json`: best prompts per MAP-Elites feature cell.
- `summary.md`: human-readable top prompts and response excerpts.
- `run_summary.json`: machine-readable run summary.
- `config.json`: run configuration.

## Algorithm

The generic runner loads an objective plugin, evaluates seed prompts, scores
target responses, extracts target-specific framing affordances from observed
responses, including bounded visible-reasoning cue snippets when targets expose
them, places candidates into objective-specific MAP-Elites feature cells,
selects high-scoring parents, asks the perturbator for direct prompt variants
using archive context and compact affordance context, validates those variants,
and repeats until the configured budget is reached.

Generic objective seed prompts live under `objectives/<objective_id>/seeds/`. Example dry-run configs live under `configs/`.

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

Prompts are mapped into feature cells by frame, specificity, and requested output shape. The archive keeps the top candidates per cell, samples promising parents, asks the perturbator model for non-deceptive variants, evaluates those variants against the target model, and repeats until the iteration, evaluation, or score budget is reached.

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

CLI flags override config file values.

## Generic Runner Options

```bash
python3 quasilisk.py --help
```

Important flags:

- `--objective-id`: built-in objective id or importable objective module.
- `--target-model`: target OpenRouter model id; repeat or comma-separate.
- `--perturbator-model`: OpenRouter model id used to generate variants.
- `--judge-model`: OpenRouter model id used to score abstract moral-susceptibility responses. Required for `llm_judge` objectives unless `--dry-run` is used.
- `--iterations`: MAP-Elites mutation rounds after seed evaluation.
- `--max-evaluations`: target call budget per target model, including seeds.
- `--archive-per-cell`: number of elites retained per feature cell.
- `--stop-score`: stop a target once a strong discussion score is reached; use `0` to disable.
- `--target-system-prompt`: optional system prompt sent to target models. It is empty by default so the target receives only the user prompt.
- `--dry-run`: no network, deterministic fake target and perturbator.

## Notes

The scoring is triage for manual review, not a final judgment. Some objectives
use deterministic heuristics and the abstract moral-susceptibility objectives
use an LLM judge. The JSONL log is the source of truth for auditing what each
model was actually asked and how it responded.
For `llm_judge` objectives, the judge score is still a triage signal rather
than a final judgment; inspect the target response and stored judge rationale
for review.

For strategy review, inspect the run `config.json`, the selected
`objectives/<objective_id>/seeds/default.json`, `affordances_<model>.json`,
`perturbations.jsonl`, and `quasilisk_harness/core.py`. Together they show the
starting islands, run parameters, target-specific framing and visible-reasoning
cues, parent-selection context supplied to the perturbator, mutation rules, raw
generated variants, and acceptance/rejection decisions.
