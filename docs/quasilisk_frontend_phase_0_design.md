# Quasilisk Observatory — Phase 0 design draft

## Status

This document is a review draft, not yet a binding amendment to
`quasilisk_redesign_specification.md`. It fixes the intended product shape,
information architecture, interaction model, visual language, and technical
boundaries early enough to avoid a later frontend redesign.

The accompanying static prototype is in `docs/frontend_prototype/`. It uses
representative data, writes nothing, makes no model calls, and contains no
scientific decision logic.

Representative captures:

![Run overview](frontend_prototype/preview-overview.png)

![Model-role configuration](frontend_prototype/preview-configure.png)

## 1. Product decision

The frontend is provisionally named **Quasilisk Observatory**. It is a
local-first browser application for configuring one canonical Quasilisk run,
observing its durable progress, and interpreting its evidence without
misrepresenting exploratory observations as verified findings.

The primary audience is a technically literate researcher who may understand
model evaluation without knowing Quasilisk's internal entity vocabulary. The
interface must remain useful to an expert without requiring a novice to read
raw JSONL before understanding a result.

The visual concept is an **epistemic instrument**: a restrained dark research
console built around evidence lattices, nonmerging argument traces, exact
checkpoints, and explicit uncertainty. It should feel investigative and
hacking-adjacent, but not theatrical, militarized, or like generic “Matrix”
decoration.

## 2. Goals and non-goals

### Goals

- Launch and resume runs through the one existing CLI and engine path.
- Make every experiment-defining value visible before launch.
- Lead with the verified-cell endpoint and fixed target-call denominator.
- Visually separate observations, high observations, screens, and verification.
- Make every reported claim navigable to exact event references.
- Expose complete call, retry, fallback, identity, token, and cost accounting.
- Organize hundreds of runs without replaying every journal at startup.
- Render all model-controlled text as inert, untrusted text.
- Work without third-party runtime packages or externally hosted assets.

### Non-goals for the first release

- Remote or multi-user hosting.
- A second controller, scheduler, reducer, or scientific schema.
- Editing objectives, templates, fixtures, or response schemas.
- An ad hoc prompt playground or isolated “retry this prompt” action.
- Concurrent runs launched by the same UI process.
- Automatic aggregation across targets or incompatible scientific scopes.
- Run deletion or destructive evidence management.
- A historical compatibility layer for superseded runtime contracts.
- Frontend prose parsing, score inference, or semantic fallback logic.

## 3. Binding architectural boundary

```text
browser assets
      |
      | same-origin loopback HTTP
      v
quasilisk.ui server
      |-- canonical config validation
      |-- exact CLI subprocess launch/resume/interrupt
      |-- read-only projection serving
      |
      +------------------------------+
                                     v
CLI -> engine.py -> events.jsonl -> run_state.py -> reporting.py
       only state     only facts       only reducer    all derived views
```

The UI is a leaf dependency. Core modules never import it. The UI may consume
public configuration, objective, event-integrity, replay, and reporting APIs,
but it owns no scientific policy.

The existing `python3 -m quasilisk` CLI remains unchanged. The UI receives a
separate operational entry point:

```text
python3 -m quasilisk.ui
```

No `--ui` mode and no experiment-setting command-line flags are added.

The proposed source shape is:

```text
quasilisk/ui/
  __main__.py       local startup and browser opening only
  server.py         loopback HTTP, closed routes, process lifecycle
  catalog.py        run discovery and projection-freshness checks
  static/
    index.html
    app.css
    app.js
    components/     buildless ES modules when the source grows
```

The browser implementation uses buildless HTML, CSS, ES modules, Web
Components where useful, and inline SVG. The Python server uses standard
library HTTP and subprocess facilities. A dependency may be reconsidered only
if a concrete accessibility or browser-compatibility requirement cannot be met
reliably with the standard platform.

## 4. Canonical launch lifecycle

1. The user selects a checked-in configuration, imports JSON, or completes the
   guided form.
2. The browser sends one complete config object. There is no separate frontend
   config contract.
3. The server writes canonical JSON atomically beneath the ignored runtime
   area `runs/.ui/configs/`, with no secret value in it.
4. The server invokes the existing `--validate-only` path and returns its exact
   config digest, scientific-scope digest, and validation errors.
5. Launch requires a second explicit action from the review screen.
6. The server invokes the existing `--config` mode as a subprocess.
7. The unique run child is discovered through its promptly published manifest.
8. A safe stop sends `SIGINT`; no hard-kill action is presented in the normal UI.
9. Resume invokes the existing `--resume` mode after compatibility validation.

Only a process started by the current UI session exposes the stop control.
Externally launched runs are visible but are never signaled through guessed
PIDs. One UI launch may be active at a time to prevent accidental duplicated
spend and ambiguous run association.

## 5. Durable live projections

The UI must not repeatedly parse a growing multi-megabyte journal in the
browser. Instead, the canonical engine publishes derived checkpoints:

- `manifest.json` immediately after the durable `run_created` event and before
  the first provider dispatch;
- `archive.json` and `summary.json` after each durably completed work-unit
  boundary; and
- the normal final bundle after `run_terminated`.

Each projection contains `run_id`, `schema_revision`, and an exact
`source_checkpoint` with sequence, event ID, and event hash. Replacement stays
atomic. These files remain disposable projections; `events.jsonl` remains the
only authority.

An intermediate projection failure must not influence scheduling or consume a
model budget. The UI marks the last projection as stale and the engine retains
the diagnostic on stderr. A final derived-artifact failure keeps the existing
terminal semantics and exit status.

Detailed evidence pages replay only the selected checkpoint through the
canonical reducer. An in-memory cache is keyed by `(run_id, event_hash)` and is
discardable. No SQLite database, browser-side evidence database, or persistent
UI index is introduced.

## 6. Information architecture

### Global destinations

1. **Run library** — organization and discovery.
2. **New run** — guided configuration and exact JSON review.
3. **Selected run** — one persistent run context with six sections:
   - Overview
   - Evidence atlas
   - Conversation trace
   - Strategies
   - Operations
   - Audit trail

The desktop shell uses a persistent left rail and compact checkpoint bar. On a
narrow screen, the rail keeps only Run library and New run; selected-run
sections remain in the horizontal tab strip, avoiding duplicate navigation.

### Result hierarchy

Every run is organized in this order:

1. verified distinct cells and target-call denominator;
2. scope and caveats;
3. evidence ladder and complete score distributions;
4. exploratory leads, explicitly labeled as unverified;
5. strategy, realization, treatment, control, and holdout evidence;
6. operational accounting and model identity;
7. exact event audit.

The product must never make the maximum score the hero metric.

## 7. Screen contracts

### 7.1 Run library

The library shows summary cards for total runs, current-schema reports, runs
with verified cells, and indexed storage, followed by a dense sortable table.

Default columns are run/time, objective, target, evidence state, target budget,
terminal state, and open action. Filters cover objective, target, terminal
reason, time, evidence state, and projection health.

The catalog reads small manifest and summary projections. It does not replay
all journals. Missing, stale, incompatible, and corrupt projections remain
visible as explicit attention states rather than disappearing.

Cross-run comparison is deferred. A later comparison view may align compatible
scientific scopes side by side, but it must not create a synthetic aggregate.

### 7.2 New-run wizard

The four fixed steps are:

1. **Objective** — title, scientific question, failure definition, boundaries,
   score anchors, and descriptor summary.
2. **Models** — provider, credential environment name, target, four distinct
   judge identities, Selector primary/fallback, and Perturbator
   primary/fallback.
3. **Experiment** — all four physical budgets, screening, verification,
   strategy qualification, holdout, proposals per step, cell capacity, seed
   policy, random seed, and output base.
4. **Review** — canonical validation result, exact normalized JSON, config
   digest, scientific-scope digest, credential presence, call envelope, and
   launch confirmation.

Presets are form-fill operations only. They never hide values or become a
second experiment contract. Imported JSON passes through the same canonical
validator and then populates the same four steps.

Credential values are never accepted. The UI displays only whether the named
environment variable exists in the server process.

### 7.3 Live run

The live state presents the last durable work-unit checkpoint, role-budget
consumption, active work-unit label, process ownership, current checkpoint, and
projection age. Scientific counts update only at reducer-backed checkpoints.

The UI distinguishes:

- **Starting** — process launched, run child not yet discovered;
- **Running** — owned subprocess alive;
- **Interrupt requested** — SIGINT sent, awaiting a durable boundary;
- **Open/incomplete** — no terminal event and no owned process;
- **Terminated** — exact terminal reason and resumability;
- **Attention required** — stale, missing, corrupt, or incompatible evidence.

Mere `.events.lock` file existence never establishes that a run is active.

### 7.4 Overview

The headline sentence describes evidence state, not an inferred property of the
target. For example:

> No descriptor cell reached repeat-supported verification.

It is followed immediately by the bounded interpretation:

> Five observations were confirmed high, but none completed the fixed evidence
> ladder within this run's budget. This is not evidence that the target is
> unsusceptible.

The four headline metrics are verified cells, target attempts, completed target
responses, and occupied cells. The evidence ladder then shows usable responses,
panel-confirmed highs, promising screens, and verified cells. Judges are
explicitly described as assessments of one response, not independent target
replications.

### 7.5 Evidence atlas

The atlas is the primary distinctive visualization. It renders the complete
declared descriptor grid, including empty cells. For a three-descriptor
objective:

- argument move is the vertical axis;
- application scope is the horizontal axis; and
- objective route is a tab/facet.

Cell state is encoded by color, shape, border, and text:

- empty — neutral outline;
- observed — desaturated filled square;
- exploratory high/lead — amber diamond or inset border;
- promising — stronger amber plus an explicit label;
- verified — cyan ring and “verified” text;
- failed — rose slash;
- inconclusive — hatched neutral state.

Selecting a cell opens retained entries, full score distributions, origin,
strategy reference, screen/verification state, evidence kind, and event links.
The UI does not compute those states from scores.

### 7.6 Conversation trace

The trace uses a deterministic layered tree, not a force-directed graph.
Nonmerging siblings stay visibly separate. Each node exposes:

- exact prompt and target response;
- conversation and scientific keys;
- role and turn;
- panel score and status;
- bound-strategy purpose, diagnosis, move, and predicted transition;
- realization lane (anchor, close, or divergent);
- objective-support and prompt-antecedent guardrails; and
- exact event references.

Long text may initially show an excerpt, but “Open exact response” must expose
the verbatim event-backed text. It is inserted with `textContent`, never parsed
as HTML or trusted Markdown.

### 7.7 Strategies

Strategy cards keep five concepts visibly separate: template, bound strategy,
realization, episode, and scenario. Recovery and advance intent receive labels,
not only colors.

Qualification views show treatment, parent baseline, strategy-agnostic control,
sealed holdout, complete distributions, and the exact qualification decision.
When qualification never began, the interface states that plainly rather than
showing empty charts.

### 7.8 Operations and audit

Operations presents role budgets, physical attempts, retry/repair/fallback
routes, failures, unknown outcomes, latency, tokens, cost-receipt coverage, and
effective identities.

Audit is cursor-paginated. It lists event sequence, timestamp, type, role and
purpose, event ID, causes, and exact payload access. Large raw bodies are loaded
only when requested. Known artifact downloads are limited to the five public
run files.

## 8. Visual system

### Design language

- Graphite surfaces and fine coordinate grids evoke an instrument, not a game.
- Cropped panel corners and coordinate labels (`E / 01`, `M / 02`) create a
  recognizable Quasilisk visual grammar.
- The Evidence Atlas and nonmerging Trace are the central identity; decoration
  does not substitute for them.
- Human-readable prose uses a system sans stack. IDs, checkpoints, and exact
  values use a system monospace stack. No external font request is made.
- Animation is limited to live-state transitions and is removed when
  `prefers-reduced-motion` is set.

### Semantic palette

| Color | Meaning |
|---|---|
| Acid lime | primary action, active navigation, integrity/validation state |
| Cyan | scientifically verified evidence only |
| Amber | high but unverified evidence, budget terminal state, caution |
| Role blue | target/model text and transport success |
| Violet | strategy, Selector, and branch structure |
| Rose | failure, corruption, and destructive-risk messaging |
| Gray | neutral, empty, unavailable, or unresolved |

Color never acts alone. Text, symbols, borders, and patterns duplicate every
meaning.

## 9. Accessibility and responsive requirements

- WCAG 2.2 AA contrast for all required text and controls.
- Complete keyboard operation and visible focus.
- Minimum 44-pixel touch targets at narrow breakpoints.
- Skip link and semantic landmarks.
- Accessible tables alongside every chart or diagram.
- `aria-selected`, `aria-current`, and live-region announcements for stateful
  navigation and wizard progress.
- No hover-only evidence.
- No auto-advancing content.
- Reduced-motion support.
- The result hierarchy remains intact at 390 CSS pixels; dense tables may use a
  labeled horizontal scroller.

## 10. Local security contract

- Bind only to `127.0.0.1`; remote binding is outside the first release.
- Reject unrecognized Host and Origin values.
- Use a random per-process CSRF token and SameSite session cookie.
- Apply a restrictive CSP with no external scripts, fonts, frames, or network
  origins.
- Reject unknown request fields and oversized bodies.
- Resolve every run/config path beneath explicit allowed roots before access.
- Expose only allowlisted artifact filenames.
- Never accept an API-key value through HTTP.
- Render model output, prompts, provider bodies, errors, and metadata as inert
  text.
- Do not render arbitrary Markdown to HTML.
- Do not persist stdout/stderr as a second scientific journal.

## 11. Draft HTTP surface

```text
GET  /api/session
GET  /api/objectives
POST /api/config/validate
GET  /api/runs
POST /api/runs
GET  /api/runs/{run_id}/overview
GET  /api/runs/{run_id}/atlas
GET  /api/runs/{run_id}/trace?root=...&cursor=...
GET  /api/runs/{run_id}/strategies
GET  /api/runs/{run_id}/operations
GET  /api/runs/{run_id}/events?after=...&limit=...
GET  /api/runs/{run_id}/artifacts/{known_filename}
POST /api/runs/{run_id}/resume
POST /api/runs/{run_id}/interrupt
```

All run-specific projection responses carry the same source-checkpoint shape.
The browser and server ship together, so there is one current API contract and
no version-suffixed compatibility routes.

## 12. Performance budgets

- Initial static assets: target below 250 KB uncompressed, excluding optional
  screenshots.
- Catalog startup reads manifest/summary projections only.
- Catalog responses are paginated; client rows are virtualized after 100.
- Run sections load lazily.
- Journals and provider bodies never enter the initial payload.
- Replayed detail projections are cached only in memory by event hash.
- A 400-run catalog and an 18 MB journal are mandatory test fixtures.

## 13. Critique and resolution record

Six passes were completed against browser-rendered prototypes before this draft
was submitted.

| Pass | Focus | Issues found | Resolution |
|---:|---|---|---|
| 1 | Scientific integrity | Headline could read as evidence of target unsusceptibility; retry annotation was not exact; ladder numerals were rotated. | Rephrased the headline as a verification-state claim, used only report-supported counts, and replaced rotated diamonds with legible clipped nodes. |
| 2 | Navigation and onboarding | Library/new-run pages retained a run checkpoint and breadcrumb; New run did not stay selected; catalog lacked organization above its table. | Made global chrome context-aware, added library summary measures, and added exact JSON import beside the wizard. |
| 3 | Subject fit and originality | Cyan conflated target text with verified evidence; “highest scoring” privileged a maximum; “verified” ambiguously described journal integrity; branch siblings were cryptic. | Reserved evidence semantics, renamed the checkpoint to “integrity OK,” selected a high-observation lineage without ranking it as truth, and introduced an explicit anchor/close/divergent branch navigator. |
| 4 | Accessibility and responsive use | Narrow screens duplicated navigation; the prototype label obscured content; touch targets and secondary contrast were weak; tab state was not fully announced. | Removed duplicate mobile destinations, raised contrast, added 44-pixel targets, objective-card focus, reduced-motion behavior, and explicit tab selection state. |
| 5 | Architecture and config completeness | Model setup hid secondary/replacement/fallback identities; sampling/search fields were absent; JSON review was inert. | Expanded the wizard to the complete closed configuration structure, exposed independence validation and all sampling/search settings, and implemented an exact JSON disclosure. |
| 6 | Visual semantics and evidence fidelity | Operational success still borrowed verification cyan; transcript snippets could be mistaken for full evidence; the aesthetic risked becoming a generic neon dashboard. | Locked colors to meanings, moved transport/model content to role blue, labeled excerpts with exact-response actions, and retained originality through the atlas, coordinate grammar, and nonmerging trace rather than ornamental effects. |

## 14. Decisions proposed for approval

Approval of this draft freezes these high-cost design choices before
implementation:

1. Product concept and provisional name: Quasilisk Observatory.
2. Local-first browser application with no remote mode in release one.
3. Separate `python3 -m quasilisk.ui` entry point; canonical CLI remains exact.
4. Four-step new-run wizard ending in complete JSON review.
5. Six-section run information architecture.
6. Evidence Atlas as the primary results visualization.
7. Deterministic nonmerging Conversation Trace.
8. Verified-first information hierarchy and fixed semantic palette.
9. Work-unit-boundary report checkpoints rather than browser journal replay.
10. Buildless, dependency-light frontend implementation.

## 15. Items intentionally left adjustable

These can change during implementation without redesigning the product:

- exact spacing and type scale;
- minor icon shapes and panel ornament;
- catalog column widths and default sort;
- pagination size;
- wording of nonbinding helper text;
- whether the app opens the browser automatically;
- port selection and other non-scientific UI-server settings.

## 16. Review checklist

The prototype reviewer should answer:

- Does “Observatory” fit the product?
- Is the restrained dark instrument aesthetic original enough without becoming
  distracting?
- Is the difference between a high observation and a verified cell immediate?
- Is the objective-selection language appropriately sober for philosophical
  susceptibility research?
- Does the four-step configuration flow expose enough detail without feeling
  like raw JSON entry?
- Are Atlas, Trace, Strategies, Operations, and Audit the right permanent
  divisions?
- Is desktop information density comfortable?
- Should narrow screens support launching runs, or become read-only below a
  chosen width?

After approval, the accepted decisions should be incorporated into the binding
redesign specification before production UI code is added.
