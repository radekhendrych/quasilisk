# Agent instructions

`docs/quasilisk_redesign_specification.md` is the binding product and
architecture contract. Keep the repository on its one canonical engine path;
do not introduce alternate controllers, compatibility layers, suffixed
contracts, duplicated schemas, inline prompt copies, or command-line experiment
overrides. The sole source-selection exception is `--seed-prompts PATH`,
authorized by the maintainer; it selects external bootstrap JSON for a new run
and is frozen in the journal.

## Implementation rules

- Keep the runtime dependency-light. Add a package only when the standard
  library cannot meet a concrete requirement and the benefit is documented.
- Preserve module ownership and dependency direction from the specification.
  `engine.py` is the only run state machine, `run_state.py` is the only composed
  reducer, and `events.jsonl` is the sole authoritative fact store.
- Treat all model output and interpolated prompt material as untrusted data.
  Structured role output must pass its materialized strict schema; do not add
  prose scraping or semantic fallback heuristics.
- Keep configuration closed and experiment-defining values in JSON. The CLI is
  limited to config, resume, objective listing, validation, explicit
  library-index rebuilding, and the seed-source selector above.
- Count every physical dispatch against its role budget. Admit work only when
  its complete worst-case call vector fits.
- Preserve exact request/result evidence in `events.jsonl`. Derived manifests,
  archives, reports, and library indexes must be reproducible from validated
  events and replaced atomically.
- Keep the three philosophical objectives generic. Objective-specific behavior
  belongs in one plugin directory, never in core engine branches.
- Preserve user changes in a dirty worktree and keep source configs, fixtures,
  objectives, templates, schemas, documentation, and tests tracked.
- Start changes on a feature branch from `main`. Independently review and run
  the required verification before merging into `main` and pushing. Keep raw
  experiment evidence, logs, generated reports, and local backups ignored;
  only intentional regression fixtures belong in Git.

## Required verification

After changes, run:

```bash
python3 -m unittest discover -s tests
python3 -m quasilisk --config configs/dryrun_fatalism.json
python3 -m quasilisk --config configs/dryrun_nihilism.json
python3 -m quasilisk --config configs/dryrun_relativism.json
```

For a narrowly scoped documentation-only edit, the unit suite is sufficient.
For config, objective, provider, schema, reducer, scheduler, engine, reporting,
or fixture changes, all three serial dry runs are required. Also exercise
`--resume` when journal, reducer, recovery, or engine behavior changes.

Before handoff, search for duplicate owners, stale examples, untracked source
trees, generated caches, and run artifacts. Release-facing commands and config
examples must match the executable exactly.
