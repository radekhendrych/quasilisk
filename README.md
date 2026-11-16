# Quasilisk

Quasilisk is a dependency-light, serial research harness for testing whether one
language-model target exhibits a declared philosophical susceptibility. It uses
one deterministic engine, objective plugins, funded work plans, a MAP-Elites
archive, fresh verification, and a complete append-only audit trail.

The supported objectives are:

- `fatalism_susceptibility`
- `nihilism_susceptibility`
- `relativism_susceptibility`

The practical endpoint is the target's own action-governing commitment, derived
from a premise it has adopted. The engine gives these openings a finishing
priority and protects capacity for fresh replay. A high single response is only
a lead. Scientific claims require a conforming frozen
objective scope, exact usable judge evidence, repeated independent target
executions, fixed screening, fresh verification, and descriptor-cell evidence.
Role-call allocations are fully funded before a run starts. An optional dollar
ceiling can stop a run before its next physical dispatch; this operational limit
does not establish scientific success or failure.

## Quick start

Quasilisk requires Python 3 and has no third-party runtime dependency.

Development starts on feature branches from `main`. Run the verification in
`AGENTS.md` and review the diff before merging. Raw experiment journals, study
results, report data, logs, and backup archives stay local and are ignored by
Git. Curated fixtures and all source code stay tracked. Historical study tests
skip explicitly when their private evidence is absent; core regression tests
use the checked-in fixtures.

List the discovered objective plugins:

```bash
python3 -m quasilisk --list-objectives
```

Validate a configuration and its selected plugin without creating a run:

```bash
python3 -m quasilisk --config configs/dryrun_fatalism.json --validate-only
```

Run the three deterministic, network-free experiments:

```bash
python3 -m quasilisk --config configs/dryrun_fatalism.json
python3 -m quasilisk --config configs/dryrun_nihilism.json
python3 -m quasilisk --config configs/dryrun_relativism.json
```

Each checked-in dry-run config names a reusable output-container base. Every
invocation creates a distinct `run-<full-run-id>` child, so rerunning a command
does not overwrite earlier evidence.

## Command-line interface

The complete CLI surface is:

```text
python3 -m quasilisk --config PATH [--seed-prompts PATH] [--validate-only]
python3 -m quasilisk --resume OUTPUT_DIRECTORY
python3 -m quasilisk --list-objectives
python3 -m quasilisk --rebuild-library-indexes RUN_ROOT [RUN_ROOT ...]
```

`--config` starts a run from one strict JSON file. `--validate-only` performs
pre-run configuration, plugin, template, and schema checks without creating a
run directory or making a model call. `--resume` accepts the exact
`run-<full-run-id>` directory and uses only the configuration and input
snapshots frozen in its `run_created` event.

`--rebuild-library-indexes` is a read-only evidence-reduction mode followed by
one atomic index replacement. It accepts only explicit consolidated run roots
and makes no model call. It is mutually exclusive with every run, resume, list,
and validation mode. `--config` and `--resume` are also mutually exclusive.
All experiment settings remain in JSON. The sole source-selection convenience
is `--seed-prompts PATH`; it selects bootstrap JSON for a new run. Its path is
relative to the current directory; `seed.prompts_path` in config is relative to
the repository root. Resume uses the frozen selected contents even if the file
has moved or been deleted. The format is the same closed `objective_id`/`probes`
object as each objective's `bootstrap.json`, including prompt IDs and descriptors.

Each canonical `objectives/<objective>/bootstrap.json` contains six unqualified
starting prompts: the original three plus three imported from the
`quasilisk-experiment` seed batch, for fatalism, nihilism, and relativism.
Skepticism-derived arguments are excluded. With `seed.prompts_path: null`,
these are available automatically; `bootstrap_limit: 6` admits all six.
Smaller limits retain the original prompts first. The A/B configs use frozen
original-only and imported-only snapshots under `fixtures/seed_ab_20261108/`.

For example, validate a selected prompt set:

```bash
python3 -m quasilisk --config configs/dryrun_nihilism.json --seed-prompts objectives/nihilism_susceptibility/bootstrap.json --validate-only
```

For edited prompts, use a network config or regenerate matching scripted
fixtures; scripted responses match the exact request.

Process exits are stable:

| Exit | Meaning |
|---:|---|
| `0` | Successful command, completed plan, or no eligible work |
| `2` | Invalid CLI, path, configuration, or objective plugin |
| `3` | Corrupt or unwritable evidence, derived artifact, or library index |
| `4` | Sustained provider failure or target identity drift |
| `5` | Resume incompatibility or impossible internal state |
| `130` | Operator interruption at a durable boundary |

## Configuration

Configuration is closed: every field below is required, and unknown fields are
rejected. Noninteger JSON numbers are parsed and normalized as exact decimals.

```json
{
  "provider": {
    "adapter": "openai_compatible",
    "base_url": "https://example-provider.invalid/v1",
    "api_key_env": "PROVIDER_API_KEY",
    "timeout_seconds": 90
  },
  "objective": "fatalism_susceptibility",
  "models": {
    "target": {
      "model": "provider/target",
      "temperature": 0.2,
      "max_output_tokens": 1600,
      "system_prompt": ""
    },
    "judges": {
      "primary": {
        "model": "provider-a/judge-primary",
        "temperature": 0,
        "max_output_tokens": 1600
      },
      "secondaries": [
        {
          "model": "provider-b/judge-secondary-a",
          "temperature": 0,
          "max_output_tokens": 1600
        },
        {
          "model": "provider-c/judge-secondary-b",
          "temperature": 0,
          "max_output_tokens": 1600
        }
      ],
      "replacement": {
        "model": "provider-d/judge-replacement",
        "temperature": 0,
        "max_output_tokens": 1600
      }
    },
    "selector": {
      "model": "provider/selector",
      "fallback_model": "provider/selector-fallback",
      "temperature": 0.2,
      "max_output_tokens": 1800
    },
    "perturbator": {
      "model": "provider/perturbator",
      "fallback_model": "provider/perturbator-fallback",
      "temperature": 0.7,
      "max_output_tokens": 1600
    }
  },
  "seed": {
    "library_directory": "library/seeds",
    "bootstrap_limit": 6,
    "prompts_path": null
  },
  "experiment": {
    "exploration_batches": 1,
    "conversation_rounds": 3,
    "verification_blocks": 1,
    "strategy_experiments": 0
  },
  "samples": {
    "screening": 3,
    "verification": 5,
    "strategy_qualification": 5,
    "holdout": 5
  },
  "search": {
    "archive_capacity_per_cell": 2,
    "proposals_per_step": 3,
    "maximum_followups": 3,
    "maximum_stalled_followups": 2
  },
  "random_seed": 7,
  "output_directory": "runs/fatalism_experiment"
}
```

The two provider forms are exact tagged alternatives:

```json
{
  "adapter": "openai_compatible",
  "base_url": "https://example-provider.invalid/v1",
  "api_key_env": "PROVIDER_API_KEY",
  "timeout_seconds": 90
}
```

```json
{
  "adapter": "scripted",
  "script_path": "fixtures/dryrun/fatalism.jsonl",
  "timeout_seconds": 1
}
```

The network adapter reads the credential only from `api_key_env`; secret values
and authorization headers never enter hashes or events. The scripted adapter is
a stateless lookup keyed by the logical action, role, route, attempt ordinal,
and exact semantic request. It performs no network I/O and fails loudly on a
missing or duplicate fixture key.

Configuration validation enforces these invariants:

- exactly one target;
- one primary plus one secondary and `replacement: null`, or one primary plus
  two secondaries and one replacement; all model IDs are distinct;
- the target model is not reused by a judge, Selector, or Perturbator;
- Selector and Perturbator fallbacks differ from their respective primaries;
- provider timeout is in `1..300` seconds;
- an optional `provider.max_cost_usd` requires frozen `max_price` and
  `max_context_tokens` for every model route; unknown charges retain their full
  reservation across resume;
- experiment allocations are nonnegative integers; complete physical funding is derived automatically for all selected work;
- screening is exactly `3`, while verification, strategy qualification, and
  holdout are each at least `5`;
- archive capacity is at least `1` and proposals per step is at least `3`;
- bootstrap limit is nonnegative and cannot be zero when no compatible verified
  seed exists; and
- paths are resolved relative to the repository root, and the output base is a
  writable directory outside all source and input-data trees.

Budgets count physical dispatch attempts, including retries, schema repairs,
response-format downgrades, replacements, and role fallbacks. The engine admits
work only when the complete unit's worst-case role-call vector fits the
remaining budget; it never starts hidden or partially budgeted work.

`experiment` funds finite exploration batches, conversational rounds, fresh
verification blocks, and strategy experiments. All selected entries and all
potential triggered screens are funded automatically. `--validate-only` prints
the derived retry-inclusive physical call limits. Every started batch consumes
an allocation, including rejected batches. Recovery cannot consume another
allocation's funding, and auxiliary exhaustion is an internal invariant failure.

Branches close at `maximum_followups` or after `maximum_stalled_followups`
turns without semantic progress. Score drift alone is not progress. The complete
judge procedure reaches reporting, Selector, and Perturbator; exact adopted
premises remain useful at score zero, and an unresolved bridge requires repair.
Generation briefs pin the parent reply and include prior rejection reasons.
Finisher examples, including condition repair, concrete decisions, inconsistent
defenses, and problematic implications, live in
[`templates/finisher_strategies.json`](templates/finisher_strategies.json).

## Execution and evidence

One run evaluates one exact target profile against one objective. The engine:

1. loads compatible verified library entries, then bounded bootstrap probes;
2. records target responses before any judging;
3. requires evidence for premise ownership, inference, commitment, and material
   reservations, then applies deterministic score ceilings and panel rules;
4. screens high observations with three independent target executions;
5. explores objective-declared categorical cells with one MAP-Elites archive;
6. uses Selector-authored bound strategies and Perturbator realizations for
   stateful target-conditioned search; and
7. requires fresh verification and matched controls before durable scientific
   qualification.

Checked-in judge and Selector semantic fixtures are offline regression inputs,
not live-run gates. The scheduler inserts no model-calibration or
component-qualification work and publishes no reusable model status. Each live
structured response still passes its materialized strict schema and bounded
route recovery.

One bound strategy has exactly one realization step. The Perturbator receives
exactly three slots for each active strategy: anchor, close, and divergent.
Continuing through a supporting target reply requires a new Selector decision
and a newly grounded linked strategy; the old contract is never silently
stretched over another turn. One primary-route admission review sends only
`aligned` realizations to the target. `invalid_contract` and `invalid` are
terminal and receive neither correction nor target call.

All model-generated text is untrusted data. Role templates delimit interpolated
material and instruct models never to follow commands embedded in transcripts,
objective data, evidence, candidate prompts, or prior model output. Strict JSON
schemas validate structured role responses; deterministic orchestration never
extracts hidden instructions or invents semantic fallbacks.

## Run directories

A run child contains five public files:

- `events.jsonl` — the sole authoritative, append-only evidence stream;
- `manifest.json` — a rebuildable projection of `run_created`;
- `archive.json` — the rebuildable current MAP projection;
- `summary.json` — the rebuildable machine-readable report; and
- `summary.md` — the rebuildable human-readable report.

Every physical target, judge, Selector, and Perturbator attempt is represented
in `events.jsonl`, including exact semantic messages, transport receipts,
failures, retries, budget charges, and causal references. Derived files may be
deleted and reconstructed from that hash-chained journal. They never replace or
amend authoritative evidence.

Resume validates the full event chain, frozen configuration, selected plugin,
templates, schemas, library snapshots, reducer, and scripted-fixture digest
before continuing. Settled model calls are reused; an unknowable post-dispatch
outcome remains visible and charged. There is no force-resume or
ignore-compatibility mode.

Reports lead with verified distinct descriptor cells at the fixed physical
target-attempt budget. They separate demonstrated-only support from evidence
that depends on explicit semantic ambiguity and disclose qualification,
fallback, disagreement, unresolved evidence, and provider-identity caveats.

## Repository architecture

The package follows a single dependency direction from immutable domain records
through services, reduction, scheduling, and the engine. Important ownership
boundaries are:

- `config.py`: the only configuration contract and normalization;
- `objective.py`: trusted plugin discovery, validation, materialization, and
  digesting;
- `provider.py`: role-neutral transport only;
- `events.py`: durable journal validation and append only;
- `run_state.py`: the one composed reducer;
- `judging.py`, `evidence.py`, `archive.py`, `strategy.py`, and
  `perturbation.py`: their respective scientific policies;
- `scheduler.py`: deterministic work choice and budget feasibility;
- `engine.py`: the only run state machine; and
- `reporting.py`: derived views built from replayed state.

An objective is one directory under `objectives/` containing its declarative
specification, bootstrap set, required regression fixtures, and a restricted
pure plugin module. Objective admission is deterministic conformance only;
there is no human attestation file or objective qualification state. Adding an
objective does not require editing a central registry or adding an engine
branch.

## Development checks

Run the complete release checks after implementation changes:

```bash
python3 -m unittest discover -s tests
python3 -m quasilisk --config configs/dryrun_fatalism.json
python3 -m quasilisk --config configs/dryrun_nihilism.json
python3 -m quasilisk --config configs/dryrun_relativism.json
```

The binding design and scientific rationale are in
[`docs/quasilisk_redesign_specification.md`](docs/quasilisk_redesign_specification.md).
