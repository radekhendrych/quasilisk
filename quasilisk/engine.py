"""The one serial Quasilisk run state machine."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal
import itertools
import os
from pathlib import Path
import re
import signal
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
import warnings

from .config import ConfigError, REPOSITORY_ROOT, RunConfig, load_config
from .domain import (
    Application,
    DECISION_CONSTANTS,
    STRATEGY_MOVE_ACTS,
    ActionPurpose,
    ChatMessage,
    ChatRequest,
    DomainError,
    FinishStatus,
    JsonSchemaContractError,
    JsonSchemaValidationError,
    MemberAssessment,
    ModelProfile,
    PanelAssessment,
    PortfolioIntent,
    ProposalSlot,
    ProviderErrorKind,
    ProviderFailure,
    ProviderSuccess,
    QualificationStatus,
    Role,
    RoleCallVector,
    RoleRouteBinding,
    RunTerminalReason,
    StructuredRoleScope,
    VARIATION_DIMENSIONS,
    Verdict,
    WorkUnit,
    canonical_json_bytes,
    canonical_json,
    digest_json,
    evidence_lines,
    entity_id,
    normalize_line_endings,
    require_decimal_string,
    seeded_rank,
    scientific_key,
    sha256_bytes,
    strict_json_loads,
    validate_json_schema_contract,
    validate_json_schema_instance,
    TERMINAL_EXIT_CODES,
)
from .events import (
    AuthoritativeSchemas,
    EventJournal,
    JournalCorrupt,
    JournalError,
    JournalUnwritable,
    read_events,
    validate_events,
)
from .archive import (
    ArchiveSnapshot,
    select_archive_parent,
    under_covered_cells,
)
from .evidence import (
    PerturbatorRoleScope,
    ScenarioRecipe,
    SeedVerificationRule,
    entry_seed_compatibility_key,
    parse_scenario_recipe,
    scenario_seed_compatibility_key,
)
from .judging import (
    MEMBER_ORDER,
    aggregate_panel,
    judge_evidence_catalog,
    judge_objective_context,
    judge_role_scope,
    member_assessment_from_output,
    should_schedule_secondary_panel,
    validate_judge_output,
    materialize_judge_schema,
)
from .objective import ObjectivePlugin, discover_objectives, load_objective, materialize_schema
from .provider import (
    Provider,
    ProviderProtocolError,
    ScriptedFixtureMissing,
    create_provider,
    decode_bytes,
    encode_bytes,
    render_request_body,
    requested_provider_route_key,
)
from .run_state import (
    adjust_panel_maximum,
    model_cost_reservation,
    AttemptView,
    LogicalActionView,
    ProviderCircuitOutcome,
    RunState,
    StrategyEpisodeView,
    portfolio_intent_for_node,
    portfolio_current_assessment,
    portfolio_ownership,
    portfolio_transition,
    funded_work_plan,
    funded_call_limits,
    remaining_work_allocations,
    strategy_evidence_polarities,
    portfolio_branch_closed,
    portfolio_node_progress,
    verified_commitment_units,
    reduce_events,
    reduce_provider_circuit,
    structured_output_exhausts_route,
    scenario_preflight_source,
)
from .scheduler import (
    WorkCandidate,
    action_slot_maximum,
    can_start,
    choose_next_work,
    entry_verification_maximum,
    initial_observation_maximum,
    map_exploration_maximum,
    portfolio_round_maximum,
    screening_maximum,
    stateful_root_maximum,
    strategy_development_maximum,
    strategy_holdout_maximum,
    sum_root_maxima,
)
from .perturbation import (
    ProbeReview,
    ProposalBatch,
    ReviewExpectation,
    SelectedRealization,
    make_proposal_slots,
    make_review_item_id,
    parse_probe_reviews,
    parse_proposal_batch,
    probe_review_aliases,
    proposal_slot_aliases,
    select_ordinary_realization,
    select_strategy_agnostic_realization,
    select_strategy_realization,
)
from .strategy import (
    BoundPortfolioLane,
    SELECTOR_ROLE_SCOPE_BY_REQUEST_SCOPE,
    SelectorPortfolio,
    StrategyTemplate,
    StrategySlot,
    TemplateBindingResult,
    allocate_contract_step,
    bind_portfolio_to_slots,
    build_strategy_evidence_catalog,
    build_sealed_holdout_plan,
    derive_template_comparison_dimensions,
    descriptors_dict,
    freeze_development_arm_order,
    freeze_qualification_intent_assignment,
    make_template_binding_item_id,
    outcome_review_aliases,
    parse_outcome_reviews,
    parse_sealed_holdout_plan,
    parse_selector_portfolio,
    parse_strategy_template,
    parse_template_binding,
    qualified_strategy_compatibility_key,
    selector_role_scope,
    strategy_slot,
    validate_selector_fixture_semantics,
    validate_materialized_holdout_plan,
)


TEMPLATE_NAMES = (
    "finisher_strategies.json",
    "judge_system.md",
    "judge_user.md",
    "selector_system.md",
    "selector_user.md",
    "perturbator_system.md",
    "perturbator_user.md",
    "probe_review_system.md",
    "probe_review_user.md",
    "template_binding_system.md",
    "template_binding_user.md",
    "outcome_review_system.md",
    "outcome_review_user.md",
)

SCHEMA_NAMES = (
    "event.json",
    "work_unit.json",
    "judge_output.json",
    "selector_portfolio.json",
    "probe_review.json",
    "template_binding.json",
    "outcome_review.json",
    "proposal_batch.json",
)

ROLE_SCHEMA_NAMES = SCHEMA_NAMES[2:]

_SELECTOR_SCOPE_MATERIAL: Mapping[
    str, tuple[str, str, str]
] = {
    "portfolio": (
        "selector_system.md",
        "selector_user.md",
        "selector_portfolio.json",
    ),
    "probe_review": (
        "probe_review_system.md",
        "probe_review_user.md",
        "probe_review.json",
    ),
    "template_binding": (
        "template_binding_system.md",
        "template_binding_user.md",
        "template_binding.json",
    ),
    "outcome_review": (
        "outcome_review_system.md",
        "outcome_review_user.md",
        "outcome_review.json",
    ),
}
POLICY_DIGEST = digest_json(dict(DECISION_CONSTANTS))


class EngineError(RuntimeError):
    """A run cannot continue safely."""


class EngineCompatibilityError(EngineError):
    """Frozen executable inputs differ on resume."""


@dataclass(frozen=True, slots=True)
class CandidatePlan:
    candidate_id: str
    scientific_key: str
    prompt: str
    descriptors: Mapping[str, str]
    origin: str
    probe_id: str | None
    parent_node_id: str | None = None
    parent_node_scientific_key: str | None = None
    strategy_id: str | None = None
    proposal_slot_scientific_key: str | None = None


@dataclass(frozen=True, slots=True)
class _LinkedStrategyStep:
    """One exact supported strategy edge in a stateful candidate ancestry."""

    episode: StrategyEpisodeView
    child: Any
    candidate: CandidatePlan
    node: Any


@dataclass(frozen=True, slots=True)
class _PortfolioCandidateAncestry:
    """The screened entry and immutable supported edges leading to a candidate."""

    entry_candidate: CandidatePlan
    entry_node: Any | None
    linked_steps: tuple[_LinkedStrategyStep, ...]


@dataclass(frozen=True, slots=True)
class _PortfolioRoot:
    node_id: str
    node_scientific_key: str
    conversation_root_id: str
    conversation_root_scientific_key: str
    portfolio_intent: PortfolioIntent


@dataclass(frozen=True, slots=True)
class ScenarioPlan:
    """One current-run identity for an imported or newly derived recipe."""

    scenario_id: str
    scientific_key: str
    entry_prompt: str
    recipe: ScenarioRecipe
    origin: str
    source_seed_id: str | None = None
    source_episode_id: str | None = None
    source_child_scientific_key: str | None = None
    parent_unit_id: str | None = None
    source_parent_node_keys: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ResourceSet:
    config: RunConfig
    objective: ObjectivePlugin
    templates: Mapping[str, str]
    template_digests: Mapping[str, str]
    schema_templates: Mapping[str, Mapping[str, Any]]
    schemas: Mapping[str, Mapping[str, Any]]
    schema_template_digests: Mapping[str, str]
    schema_digests: Mapping[str, str]
    authoritative_schemas: AuthoritativeSchemas
    engine_digest: str
    reducer_digest: str
    provider: Provider
    scripted_fixture_digest: str | None
    bootstrap_payloads: tuple[Mapping[str, Any], ...]
    seed_snapshot: tuple[Mapping[str, Any], ...]
    strategy_snapshot: tuple[Mapping[str, Any], ...]
    scientific_scope_digest: str


@dataclass(slots=True)
class RunContext:
    resources: ResourceSet
    run_id: str
    run_nonce: str
    invocation_id: str
    run_directory: Path
    candidates: dict[str, CandidatePlan]
    routes: dict[str, RoleRouteBinding]
    scenarios: dict[str, ScenarioPlan] = field(default_factory=dict)
    sleeper: Callable[[float], None] = time.sleep
    stop_requested: bool = False
    validated_payloads: dict[str, tuple[Any, ...]] = field(default_factory=dict, repr=False)

    @property
    def config(self) -> RunConfig:
        return self.resources.config

    @property
    def objective(self) -> ObjectivePlugin:
        return self.resources.objective

    @property
    def scope(self) -> str:
        return self.resources.scientific_scope_digest


def _load_templates(repository_root: Path) -> tuple[dict[str, str], dict[str, str]]:
    root = repository_root / "templates"
    contents: dict[str, str] = {}
    digests: dict[str, str] = {}
    actual = {path.name for path in root.iterdir() if path.is_file() and path.suffix in {".md", ".json"}}
    if actual != set(TEMPLATE_NAMES):
        raise EngineError(
            f"canonical template set mismatch; missing={sorted(set(TEMPLATE_NAMES)-actual)}, "
            f"unexpected={sorted(actual-set(TEMPLATE_NAMES))}"
        )
    for name in TEMPLATE_NAMES:
        raw = (root / name).read_bytes()
        try:
            contents[name] = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise EngineError(f"template {name} is not UTF-8") from exc
        digests[name] = sha256_bytes(raw)
        if name == "finisher_strategies.json":
            catalog = strict_json_loads(raw)
            if not isinstance(catalog, dict) or set(catalog) != {"strategies"} or not isinstance(catalog["strategies"], list) or not catalog["strategies"]:
                raise EngineError("finisher catalog must contain a nonempty strategies array")
            ids: set[str] = set()
            for card in catalog["strategies"]:
                if not isinstance(card, dict) or set(card) != {"id", "act", "requires", "prompt", "success"}:
                    raise EngineError("invalid finisher strategy card")
                if any(not isinstance(value, str) or not value for value in card.values()) or card["id"] in ids or card["act"] not in STRATEGY_MOVE_ACTS:
                    raise EngineError("invalid or duplicate finisher strategy identity")
                ids.add(card["id"])
    return contents, digests


def _load_schema_templates(repository_root: Path) -> tuple[
    dict[str, Mapping[str, Any]], dict[str, str]
]:
    root = repository_root / "schemas"
    values: dict[str, Mapping[str, Any]] = {}
    digests: dict[str, str] = {}
    actual = {path.name for path in root.glob("*.json") if path.is_file()}
    if actual != set(SCHEMA_NAMES):
        raise EngineError(
            f"canonical schema set mismatch; missing={sorted(set(SCHEMA_NAMES)-actual)}, "
            f"unexpected={sorted(actual-set(SCHEMA_NAMES))}"
        )
    for name in SCHEMA_NAMES:
        raw = (root / name).read_bytes()
        value = strict_json_loads(raw)
        if not isinstance(value, dict):
            raise EngineError(f"schema {name} is not one JSON object")
        values[name] = value
        digests[name] = sha256_bytes(raw)
    return values, digests


def _engine_source_digest(repository_root: Path) -> str:
    pairs: list[list[str]] = []
    package_root = repository_root / "quasilisk"
    paths = sorted(package_root.rglob("*.py"))
    for path in paths:
        if "__pycache__" in path.parts:
            continue
        pairs.append([path.relative_to(repository_root).as_posix(), sha256_bytes(path.read_bytes())])
    return digest_json(pairs)


def _load_library_snapshots(
    config: RunConfig,
    objective: ObjectivePlugin,
) -> tuple[
    tuple[Mapping[str, Any], ...],
    tuple[Mapping[str, Any], ...],
]:
    """Replay source journals before admitting any derived library prior."""

    transaction_marker = (
        config.repository_root / "library" / ".indexes.transaction.json"
    )
    if transaction_marker.exists():
        warnings.warn(
            "an interrupted library-index transaction is pending recovery; "
            "all shared-library priors are ignored for this run",
            RuntimeWarning,
            stacklevel=2,
        )
        return (), ()

    candidates: list[dict[str, Any]] = []
    for kind, directory, projection_name, identity_field in (
        (
            "seed",
            config.seed_library_directory,
            "current_seed_qualification_snapshots",
            "seed_id",
        ),
        (
            "strategy_template",
            config.repository_root / "library" / "strategies",
            "current_strategy_qualification_snapshots",
            "template_id",
        ),
    ):
        index = directory / "index.json"
        if not index.exists():
            continue
        try:
            value = strict_json_loads(index.read_bytes())
        except (OSError, DomainError) as exc:
            raise EngineError(f"library index is unreadable: {index}: {exc}") from exc
        if (
            not isinstance(value, dict)
            or set(value) != {"entries"}
            or not isinstance(value["entries"], list)
        ):
            raise EngineError(f"library index is not a closed entries object: {index}")
        seen_identities: set[str] = set()
        for ordinal, item in enumerate(value["entries"]):
            subject = item.get("subject_payload") if isinstance(item, Mapping) else None
            identity = subject.get(identity_field) if isinstance(subject, Mapping) else None
            if not isinstance(identity, str):
                continue
            if identity in seen_identities:
                raise EngineError(
                    f"library index contains duplicate {identity_field} {identity}: {index}"
                )
            seen_identities.add(identity)
        for ordinal, item in enumerate(value["entries"]):
            if not isinstance(item, Mapping) or set(item) != {
                "qualification_summary",
                "source_checkpoint",
                "subject_payload",
            }:
                warnings.warn(
                    f"ignored malformed {kind} index entry {ordinal} from {index}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            checkpoint = item["source_checkpoint"]
            subject = item["subject_payload"]
            summary = item["qualification_summary"]
            if (
                not isinstance(checkpoint, Mapping)
                or set(checkpoint)
                != {"source_run_id", "sequence", "event_hash", "source_path"}
                or not isinstance(subject, Mapping)
                or not isinstance(summary, Mapping)
                or subject.get("source_checkpoint") != checkpoint
            ):
                warnings.warn(
                    f"ignored internally inconsistent {kind} index entry {ordinal}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            source_path = checkpoint.get("source_path")
            sequence = checkpoint.get("sequence")
            if (
                not isinstance(source_path, str)
                or isinstance(sequence, bool)
                or not isinstance(sequence, int)
                or sequence < 1
            ):
                warnings.warn(
                    f"ignored {kind} index entry with an invalid source checkpoint",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            try:
                source_directory = Path(source_path).resolve(strict=True)
            except OSError as exc:
                warnings.warn(
                    f"ignored orphaned {kind} index entry {ordinal}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            if kind == "strategy_template" and (
                summary.get("objective_id") != objective.id
                or summary.get("target_generation_profile_digest")
                != config.target.digest
                or summary.get("compatibility_key")
                != subject.get("compatibility_key")
            ):
                continue
            candidates.append(
                {
                    "kind": kind,
                    "projection_name": projection_name,
                    "identity_field": identity_field,
                    "ordinal": ordinal,
                    "item": dict(item),
                    "checkpoint": dict(checkpoint),
                    "source_directory": source_directory,
                }
            )

    grouped: dict[Path, list[dict[str, Any]]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate["source_directory"], []).append(candidate)
    admitted_by_kind: dict[str, list[Mapping[str, Any]]] = {
        "seed": [],
        "strategy_template": [],
    }
    for source_directory in sorted(grouped, key=lambda path: path.as_posix()):
        group = grouped[source_directory]
        replay_sequence = max(
            int(item["checkpoint"]["sequence"]) for item in group
        )
        try:
            source_events = read_events(
                source_directory, through_sequence=replay_sequence
            )
        except (OSError, JournalCorrupt) as exc:
            for candidate in group:
                warnings.warn(
                    "ignored orphaned "
                    f"{candidate['kind']} index entry {candidate['ordinal']}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            continue
        if not source_events:
            continue
        source_objective = source_events[0]["payload"]["manifest"].get(
            "objective", {}
        )
        if not isinstance(source_objective, Mapping):
            continue
        compatible: list[dict[str, Any]] = []
        for candidate in group:
            checkpoint = candidate["checkpoint"]
            sequence = checkpoint["sequence"]
            if (
                sequence > len(source_events)
                or source_events[0]["run_id"] != checkpoint.get("source_run_id")
                or source_events[sequence - 1]["event_hash"]
                != checkpoint.get("event_hash")
            ):
                warnings.warn(
                    "ignored mismatching "
                    f"{candidate['kind']} source checkpoint at entry "
                    f"{candidate['ordinal']}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            if (
                source_objective.get("id") != objective.id
                or source_objective.get("plugin_digest")
                != objective.plugin_digest
            ):
                continue
            compatible.append(candidate)
        if not compatible:
            continue
        replay_sequence = max(
            int(item["checkpoint"]["sequence"]) for item in compatible
        )
        try:
            source_state = reduce_events(
                source_events[:replay_sequence], objective
            )
        except (DomainError, JournalCorrupt) as exc:
            for candidate in compatible:
                warnings.warn(
                    "ignored unreproducible "
                    f"{candidate['kind']} index entry {candidate['ordinal']}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            continue
        for candidate in compatible:
            projection_name = candidate["projection_name"]
            if projection_name == "current_seed_qualification_snapshots":
                projections = source_state.current_seed_qualification_snapshots
            elif projection_name == "current_strategy_qualification_snapshots":
                projections = source_state.current_strategy_qualification_snapshots
            else:
                raise EngineError("library projection owner is not canonical")
            subject = candidate["item"]["subject_payload"]
            identity_field = candidate["identity_field"]
            reproduced = [
                projection
                for projection in projections
                if isinstance(projection, Mapping)
                and projection.get(identity_field) == subject.get(identity_field)
            ]
            if len(reproduced) != 1 or dict(reproduced[0]) != dict(subject):
                warnings.warn(
                    f"ignored {candidate['kind']} index entry not reproduced by its source journal",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            admitted_by_kind[candidate["kind"]].append(candidate["item"])

    for kind, identity_field in (
        ("seed", "seed_id"),
        ("strategy_template", "template_id"),
    ):
        admitted_by_kind[kind].sort(
            key=lambda candidate: str(
                candidate["subject_payload"].get(identity_field, "")
            )
        )
    return (
        tuple(admitted_by_kind["seed"]),
        tuple(admitted_by_kind["strategy_template"]),
    )


def _current_objective_scope_key(
    objective: ObjectivePlugin,
    schema_digests: Mapping[str, str],
) -> str:
    """Return the exact static objective scope frozen into reusable evidence."""

    return digest_json(
        [
            "objective_scope",
            objective.id,
            objective.plugin_digest,
            [
                [name, objective.artifact_digests[name]]
                for name in sorted(objective.artifact_digests)
            ],
            schema_digests["judge_output.json"],
        ]
    )


def _current_judge_protocol_digest(
    template_digests: Mapping[str, str],
    schema_digests: Mapping[str, str],
) -> str:
    """Return the route-independent judge protocol frozen into seed keys."""

    return digest_json(
        [
            "judge_protocol",
            template_digests["judge_system.md"],
            template_digests["judge_user.md"],
            schema_digests["judge_output.json"],
            dict(DECISION_CONSTANTS),
        ]
    )


def _current_structured_role_scope_candidates(
    *,
    config: RunConfig,
    template_digests: Mapping[str, str],
    schema_digests: Mapping[str, str],
) -> tuple[tuple[tuple[str, str, str], StructuredRoleScope], ...]:
    """Materialize exact current structured-role provenance scopes."""

    candidates: list[tuple[tuple[str, str, str], StructuredRoleScope]] = []
    seen: set[str] = set()

    def add(
        role: str, configured_slot: str, scope_name: str, scope: StructuredRoleScope
    ) -> None:
        if scope.scope_key in seen:
            return
        seen.add(scope.scope_key)
        candidates.append(((role, configured_slot, scope_name), scope))

    judge_routes = (
        ("primary", config.judges.primary),
        ("secondary_a", config.judges.secondary_a),
        ("secondary_b", config.judges.secondary_b),
        ("replacement", config.judges.replacement),
    )
    for configured_slot, profile in judge_routes:
        if profile is None:
            continue
        for response_mode in ("json_schema", "json_object"):
            scope = judge_role_scope(
                requested_model=profile.model,
                generation_profile_digest=profile.digest,
                judge_system_template_digest=template_digests["judge_system.md"],
                judge_user_template_digest=template_digests["judge_user.md"],
                judge_schema_digest=schema_digests["judge_output.json"],
                response_mode=response_mode,
            )
            add(Role.JUDGE.value, configured_slot, "judge_route", scope)

    selector_routes = (
        ("selector_primary", config.selector.primary()),
        ("selector_fallback", config.selector.fallback()),
    )
    for configured_slot, profile in selector_routes:
        for selector_scope, material in _SELECTOR_SCOPE_MATERIAL.items():
            system_template, user_template, schema_name = material
            for response_mode in ("json_schema", "json_object"):
                scope = selector_role_scope(
                    selector_scope,
                    requested_model=profile.model,
                    system_template_digest=template_digests[system_template],
                    user_template_digest=template_digests[user_template],
                    schema_digest=schema_digests[schema_name],
                    response_mode=response_mode,
                )
                add(Role.SELECTOR.value, configured_slot, selector_scope, scope)
    return tuple(candidates)


def _current_perturbator_scopes(
    *,
    config: RunConfig,
    template_digests: Mapping[str, str],
    schema_digests: Mapping[str, str],
) -> Mapping[str, Mapping[str, Any]]:
    """Return every exact current primary/fallback and response-mode scope."""

    prompt_material = (
        template_digests["perturbator_system.md"],
        template_digests["perturbator_user.md"],
        schema_digests["proposal_batch.json"],
    )
    prompt_modes = tuple(
        (
            mode,
            digest_json(["perturbator_prompt_mode", mode, *prompt_material]),
        )
        for mode in ("strategy", "strategy_agnostic")
    )
    scopes: dict[str, Mapping[str, Any]] = {}
    for route_label, profile in (
        ("primary", config.perturbator.primary()),
        ("fallback", config.perturbator.fallback()),
    ):
        provider_key = requested_provider_route_key(
            config.provider.adapter, Role.PERTURBATOR, route_label
        )
        for response_mode in ("json_schema", "json_object"):
            scope = PerturbatorRoleScope(
                provider_key,
                profile.model,
                profile.digest,
                response_mode,
                prompt_modes,
            )
            payload = {"scope_key": scope.scope_key, **scope.to_dict()}
            scopes[scope.scope_key] = payload
    if len(scopes) != 4:
        raise EngineError("current Perturbator role scope keys collide")
    return dict(sorted(scopes.items()))


def _target_static_scope(config: RunConfig) -> Mapping[str, str]:
    return {
        "requested_provider_route_key": requested_provider_route_key(
            config.provider.adapter, Role.TARGET, "primary"
        ),
        "requested_model": config.target.model,
        "generation_profile_digest": config.target.digest,
    }


def _strategy_prior_static_compatible(
    subject: Mapping[str, Any],
    *,
    config: RunConfig,
    objective: ObjectivePlugin,
    objective_scope_key: str,
    structured_role_candidates: Sequence[
        tuple[tuple[str, str, str], StructuredRoleScope]
    ],
    perturbator_scopes: Mapping[str, Mapping[str, Any]],
    expected_status: QualificationStatus = QualificationStatus.QUALIFIED,
) -> bool:
    """Check all static qualified-strategy compatibility material exactly."""

    target = _target_static_scope(config)
    if (
        subject.get("qualification_status")
        != expected_status.value
        or subject.get("objective_id") != objective.id
        or subject.get("plugin_digest") != objective.plugin_digest
        or subject.get("objective_scope_key")
        != objective_scope_key
        or subject.get("target_requested_provider_route_key")
        != target["requested_provider_route_key"]
        or subject.get("target_requested_model") != target["requested_model"]
        or subject.get("target_generation_profile_digest")
        != target["generation_profile_digest"]
    ):
        return False
    try:
        template = parse_strategy_template(
            subject["template"],
            objective.descriptor_values,
            objective.diagnosis_categories,
        )
        raw_selector_pairs = subject["selector_scope_keys"]
        if not isinstance(raw_selector_pairs, list):
            return False
        selector_pairs_list: list[tuple[str, str]] = []
        for item in raw_selector_pairs:
            if (
                not isinstance(item, list)
                or len(item) != 2
                or not all(isinstance(part, str) for part in item)
            ):
                return False
            selector_pairs_list.append((item[0], item[1]))
        selector_pairs = tuple(selector_pairs_list)
        raw_perturbator_scopes = subject["perturbator_scopes"]
        if not isinstance(raw_perturbator_scopes, list):
            return False
        perturbator_payloads = tuple(
            dict(item) for item in raw_perturbator_scopes if isinstance(item, Mapping)
        )
        if len(perturbator_payloads) != len(raw_perturbator_scopes):
            return False
        tested_context_ids = subject["tested_entry_context_ids"]
        if not isinstance(tested_context_ids, list) or not all(
            isinstance(item, str) for item in tested_context_ids
        ):
            return False
        identity_token = subject["identity_token"]
        identity_unverifiable = subject["identity_unverifiable"]
        if not isinstance(identity_token, str) or not isinstance(
            identity_unverifiable, bool
        ):
            return False
        if not (
            identity_token.startswith("requested:")
            or identity_token.startswith("effective:")
        ) or not identity_token.split(":", 1)[1]:
            return False
        if identity_unverifiable != identity_token.startswith("requested:"):
            return False
        if identity_unverifiable and identity_token != (
            "requested:"
            f"{target['requested_provider_route_key']}/{target['requested_model']}"
        ):
            return False
        target_scope = {
            "scope_key": subject["target_role_scope_key"],
            **target,
            "identity_token": identity_token,
            "identity_unverifiable": identity_unverifiable,
        }
        expected_target_scope_key = digest_json(
            [
                "target_role_scope",
                target["requested_provider_route_key"],
                target["requested_model"],
                target["generation_profile_digest"],
                identity_token,
                identity_unverifiable,
            ]
        )
        if target_scope["scope_key"] != expected_target_scope_key:
            return False
        current_selector_pairs = {
            (scope.scope, scope.scope_key)
            for base, scope in structured_role_candidates
            if base[0] == Role.SELECTOR.value
        }
        if (
            not selector_pairs
            or selector_pairs != tuple(sorted(set(selector_pairs)))
            or any(item not in current_selector_pairs for item in selector_pairs)
        ):
            return False
        perturbator_keys: list[str] = []
        for payload in perturbator_payloads:
            scope_key = payload.get("scope_key")
            if (
                not isinstance(scope_key, str)
                or perturbator_scopes.get(scope_key) != payload
            ):
                return False
            perturbator_keys.append(scope_key)
        if not perturbator_keys or len(perturbator_keys) != len(set(perturbator_keys)):
            return False
        if perturbator_keys != sorted(perturbator_keys):
            return False
        raw_judge_keys = subject["judge_scope_keys"]
        raw_structured_role_keys = subject["required_structured_role_scope_keys"]
        raw_role_keys = subject["required_role_scope_keys"]
        if any(
            not isinstance(items, list)
            or not all(isinstance(item, str) for item in items)
            for items in (raw_judge_keys, raw_structured_role_keys, raw_role_keys)
        ):
            return False
        judge_keys = tuple(raw_judge_keys)
        structured_role_keys = tuple(raw_structured_role_keys)
        if (
            len(judge_keys) != len(set(judge_keys))
            or set(structured_role_keys)
            != {*(key for _, key in selector_pairs), *judge_keys}
        ):
            return False
        role_keys = tuple(raw_role_keys)
        if len(role_keys) != len(set(role_keys)) or set(role_keys) != {
            expected_target_scope_key,
            *perturbator_keys,
        }:
            return False
        if (
            subject.get("diagnosis") != template.applicability.diagnosis
            or subject.get("move_act") != template.move.act
            or subject.get("target_kind") != template.move.target_kind
            or subject.get("applicability") != template.applicability.to_dict()
        ):
            return False
        development_design = subject["development_design"]
        holdout_design = subject["holdout_design"]
        expected_compatibility_key = qualified_strategy_compatibility_key(
            objective_id=objective.id,
            plugin_digest=objective.plugin_digest,
            target_scope=target_scope,
            template=template,
            tested_entry_context_ids=tested_context_ids,
            selector_scope_keys=selector_pairs,
            perturbator_scopes=perturbator_payloads,
            development_design=development_design,
            holdout_design=holdout_design,
        )
    except (DomainError, KeyError, TypeError, ValueError):
        return False
    return subject.get("compatibility_key") == expected_compatibility_key


def _seed_prior_static_compatible(
    subject: Mapping[str, Any],
    *,
    config: RunConfig,
    objective: ObjectivePlugin,
    objective_scope_key: str,
    judge_protocol_digest: str,
    structured_role_candidates: Sequence[
        tuple[tuple[str, str, str], StructuredRoleScope]
    ],
    perturbator_scopes: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Recompute every seed compatibility field available at startup."""

    target = _target_static_scope(config)
    if (
        subject.get("objective_id") != objective.id
        or subject.get("plugin_digest") != objective.plugin_digest
        or subject.get("target_generation_profile_digest")
        != target["generation_profile_digest"]
        or subject.get("judge_protocol_digest") != judge_protocol_digest
        or subject.get("verification_rule")
        != SeedVerificationRule(config.samples.verification).to_dict()
        or subject.get("planned_count") != config.samples.verification
    ):
        return False
    identity_token = subject.get("identity_token")
    identity_unverifiable = subject.get("identity_unverifiable")
    if not isinstance(identity_token, str) or not isinstance(
        identity_unverifiable, bool
    ):
        return False
    if not (
        identity_token.startswith("requested:")
        or identity_token.startswith("effective:")
    ) or not identity_token.split(":", 1)[1]:
        return False
    if identity_unverifiable != identity_token.startswith("requested:"):
        return False
    if identity_unverifiable and identity_token != (
        "requested:"
        f"{target['requested_provider_route_key']}/{target['requested_model']}"
    ):
        return False
    try:
        descriptors = objective.validate_descriptors(subject.get("descriptors"))
        raw_digests = subject["qualification_digests"]
        if not isinstance(raw_digests, list):
            return False
        if not all(isinstance(item, str) for item in raw_digests):
            return False
        qualification_digests = tuple(raw_digests)
        if qualification_digests != tuple(sorted(set(qualification_digests))):
            return False
        qualification_set = set(qualification_digests)
        judge_items: dict[str, str] = {}
        for base, scope in structured_role_candidates:
            if base[0] != Role.JUDGE.value or scope.scope_key not in qualification_set:
                continue
            name = f"judge_{base[1]}"
            if name in judge_items:
                return False
            judge_items[name] = scope.scope_key
        if not judge_items:
            return False
        ordered_judges = tuple(sorted(judge_items.items()))
        panel_key = digest_json(
            [
                "judge_panel_qualification",
                judge_protocol_digest,
                [list(item) for item in ordered_judges],
            ]
        )
        judge_panel_pairs = tuple(sorted((*ordered_judges, ("panel", panel_key))))
        taxonomy_digest = digest_json(list(objective.objective["descriptors"]))
        rule = SeedVerificationRule(config.samples.verification).to_dict()
        subject_type = subject.get("type")
        if subject_type == "entry_seed":
            if subject.get("scenario_recipe") is not None:
                return False
            probe_keys = tuple(
                scope.scope_key
                for base, scope in structured_role_candidates
                if base[0] == Role.SELECTOR.value
                and scope.scope == "probe_review"
                and scope.scope_key in qualification_set
            )
            if len(probe_keys) > 1:
                return False
            assigned_probe_key = probe_keys[0] if probe_keys else None
            expected_qualification_set = {
                objective_scope_key,
                *(key for _, key in ordered_judges),
                panel_key,
                *((assigned_probe_key,) if assigned_probe_key is not None else ()),
            }
            expected_compatibility_key = entry_seed_compatibility_key(
                objective_id=objective.id,
                plugin_digest=objective.plugin_digest,
                target_generation_profile_digest=target[
                    "generation_profile_digest"
                ],
                target_identity_token=identity_token,
                descriptor_taxonomy_digest=taxonomy_digest,
                judge_panel_scope_keys=judge_panel_pairs,
                verification_rule=rule,
                assigned_probe_review_scope_key=assigned_probe_key,
            )
        elif subject_type == "scenario_recipe":
            recipe = parse_scenario_recipe(
                subject.get("scenario_recipe"),
                objective.descriptor_values,
                objective.diagnosis_categories,
            )
            prompt = subject.get("entry_prompt")
            if (
                not isinstance(prompt, str)
                or not prompt
                or recipe.entry_prompt_hash
                != sha256_bytes(normalize_line_endings(prompt).encode("utf-8"))
                or recipe.target_generation_profile_digest
                != target["generation_profile_digest"]
                or dict(recipe.measurement_cell) != descriptors
            ):
                return False
            current_selector_pairs = {
                (scope.scope, scope.scope_key)
                for base, scope in structured_role_candidates
                if base[0] == Role.SELECTOR.value
            }
            if any(
                item not in current_selector_pairs
                for item in recipe.selector_scope_keys
            ):
                return False
            perturbator_payload = perturbator_scopes.get(
                recipe.perturbator_scope.scope_key
            )
            if perturbator_payload is None or {
                key: value
                for key, value in perturbator_payload.items()
                if key != "scope_key"
            } != recipe.perturbator_scope.to_dict():
                return False
            selector_components = dict(recipe.selector_scope_keys)
            expected_qualification_set = {
                objective_scope_key,
                *(key for _, key in ordered_judges),
                panel_key,
                selector_components["probe_review"],
                selector_components["outcome_review"],
                recipe.perturbator_scope.scope_key,
            }
            expected_compatibility_key = scenario_seed_compatibility_key(
                objective_id=objective.id,
                plugin_digest=objective.plugin_digest,
                target_generation_profile_digest=target[
                    "generation_profile_digest"
                ],
                target_identity_token=identity_token,
                descriptor_taxonomy_digest=taxonomy_digest,
                judge_panel_scope_keys=judge_panel_pairs,
                recipe=recipe,
                measurement_cell=dict(recipe.measurement_cell),
                verification_rule=rule,
            )
        else:
            return False
    except (DomainError, KeyError, TypeError, ValueError):
        return False
    return (
        qualification_set == expected_qualification_set
        and subject.get("compatibility_key") == expected_compatibility_key
    )


def _unique_frozen_snapshots(
    snapshots: Sequence[Mapping[str, Any]],
    *,
    identity_field: str,
    label: str,
) -> tuple[Mapping[str, Any], ...]:
    """Validate one frozen manifest array without filtering or normalizing it."""

    if isinstance(snapshots, (str, bytes, Mapping)):
        raise EngineCompatibilityError(f"frozen {label} snapshots are not an array")
    result: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for ordinal, item in enumerate(snapshots):
        if not isinstance(item, Mapping):
            raise EngineCompatibilityError(
                f"frozen {label} snapshot {ordinal} is not an object"
            )
        identity = item.get(identity_field)
        if not isinstance(identity, str) or not identity:
            raise EngineCompatibilityError(
                f"frozen {label} snapshot {ordinal} lacks {identity_field}"
            )
        if identity in seen:
            raise EngineCompatibilityError(
                f"frozen {label} snapshots duplicate {identity_field} {identity}"
            )
        seen.add(identity)
        result.append(item)
    return tuple(result)


def _bootstrap_content_projection(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the standalone-probe content whose selection changes a run."""

    prompt = payload.get("prompt")
    descriptors = payload.get("descriptors")
    if not isinstance(prompt, str) or not isinstance(descriptors, Mapping):
        raise EngineError("selected bootstrap payload lacks prompt/descriptors content")
    return {"prompt": prompt, "descriptors": dict(descriptors)}


def _seed_content_projection(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return exactly the experiment-defining content of a selected seed."""

    if payload.get("type") not in {"entry_seed", "scenario_recipe"}:
        raise EngineError("selected seed payload has an invalid subject type")
    required = ("type", "entry_prompt", "scenario_recipe", "descriptors")
    if any(name not in payload for name in required):
        raise EngineError("selected seed payload lacks semantic subject content")
    if not isinstance(payload["entry_prompt"], str) or not isinstance(
        payload["descriptors"], Mapping
    ):
        raise EngineError("selected seed payload has invalid semantic content")
    return {
        "type": payload["type"],
        "entry_prompt": payload["entry_prompt"],
        "scenario_recipe": payload["scenario_recipe"],
        "descriptors": dict(payload["descriptors"]),
    }


def _strategy_content_projection(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return exactly the selected transcript-free StrategyTemplate content."""

    template = payload.get("template")
    if not isinstance(template, Mapping):
        raise EngineError("selected strategy payload lacks its template content")
    return {"template": dict(template)}


def _sorted_unique_content_digests(
    payloads: Iterable[Mapping[str, Any]],
    projector: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> list[str]:
    """Canonicalize a selected set without inheriting index array ordering."""

    return sorted({digest_json(projector(payload)) for payload in payloads})


def _scientific_scope_material(
    *,
    config: RunConfig,
    objective: ObjectivePlugin,
    template_digests: Mapping[str, str],
    schema_digests: Mapping[str, str],
    bootstrap_payloads: Sequence[Mapping[str, Any]],
    seed_payloads: Sequence[Mapping[str, Any]],
    strategy_payloads: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Build the sole nonce-free scientific-scope identity projection."""

    return {
        "normalized_scientific_config": config.scientific_view(),
        "objective_plugin_digest": objective.plugin_digest,
        "role_template_digests": dict(sorted(template_digests.items())),
        "response_schema_digests": {
            name: schema_digests[name] for name in ROLE_SCHEMA_NAMES
        },
        "selected_bootstrap_content_digests": _sorted_unique_content_digests(
            bootstrap_payloads, _bootstrap_content_projection
        ),
        "selected_seed_content_digests": _sorted_unique_content_digests(
            seed_payloads, _seed_content_projection
        ),
        "selected_strategy_content_digests": _sorted_unique_content_digests(
            strategy_payloads, _strategy_content_projection
        ),
        "decision_constants": dict(DECISION_CONSTANTS),
    }


def _select_verified_seed_snapshot(
    priors: Sequence[Mapping[str, Any]],
    *,
    config: RunConfig,
    objective: ObjectivePlugin,
) -> tuple[Mapping[str, Any], ...]:
    """Apply the one deterministic per-cell seed-capacity policy."""

    value_ordinals = {
        name: {value: ordinal for ordinal, value in enumerate(objective.descriptor_values[name])}
        for name in objective.descriptor_names
    }
    grouped: dict[
        tuple[str, ...],
        list[tuple[Decimal, Decimal, str, tuple[str, str], Mapping[str, Any]]],
    ] = {}
    scenario_cells: dict[str, tuple[str, ...]] = {}
    for prior in priors:
        subject = prior.get("subject_payload")
        summary = prior.get("qualification_summary")
        if not isinstance(subject, Mapping) or not isinstance(summary, Mapping):
            continue
        if (
            subject.get("objective_id") != objective.id
            or subject.get("plugin_digest") != objective.plugin_digest
            or subject.get("target_generation_profile_digest") != config.target.digest
        ):
            continue
        try:
            descriptors = objective.validate_descriptors(subject.get("descriptors"))
            success_fraction = require_decimal_string(
                subject.get("observed_success_fraction"),
                minimum=Decimal("0"),
                maximum=Decimal("1"),
            )
            median_value = summary.get("verification_median")
            verification_median = (
                Decimal("-1")
                if median_value is None
                else require_decimal_string(
                    median_value,
                    minimum=Decimal("0"),
                    maximum=Decimal("100"),
                )
            )
        except (DomainError, TypeError):
            continue
        seed_id = subject.get("seed_id")
        prompt = subject.get("entry_prompt")
        if not isinstance(seed_id, str) or not isinstance(prompt, str) or not prompt:
            continue
        cell = tuple(descriptors[name] for name in objective.descriptor_names)
        subject_type = subject.get("type")
        if subject_type == "entry_seed":
            if subject.get("scenario_recipe") is not None:
                continue
            dedupe_key = ("entry_seed", normalize_line_endings(prompt))
        elif subject_type == "scenario_recipe":
            try:
                recipe = parse_scenario_recipe(
                    subject.get("scenario_recipe"),
                    objective.descriptor_values,
                    objective.diagnosis_categories,
                )
            except DomainError:
                continue
            if (
                recipe.entry_prompt_hash
                != sha256_bytes(normalize_line_endings(prompt).encode("utf-8"))
                or recipe.target_generation_profile_digest != config.target.digest
                or dict(recipe.measurement_cell) != descriptors
            ):
                continue
            existing_cell = scenario_cells.get(recipe.content_hash)
            if existing_cell is not None and existing_cell != cell:
                raise EngineError(
                    "trusted scenario seeds assign conflicting cells to equivalent recipes"
                )
            scenario_cells[recipe.content_hash] = cell
            dedupe_key = ("scenario_recipe", recipe.content_hash)
        else:
            continue
        grouped.setdefault(cell, []).append(
            (
                success_fraction,
                verification_median,
                seed_id,
                dedupe_key,
                dict(subject),
            )
        )

    ordered_cells = sorted(
        grouped,
        key=lambda cell: tuple(
            value_ordinals[name][value]
            for name, value in zip(objective.descriptor_names, cell, strict=True)
        ),
    )
    selected: list[Mapping[str, Any]] = []
    seen_subjects: set[tuple[str, str]] = set()
    for cell in ordered_cells:
        ranked = sorted(
            grouped[cell],
            key=lambda item: (-item[0], -item[1], item[2]),
        )
        retained = 0
        for _, _, _, dedupe_key, subject in ranked:
            if dedupe_key in seen_subjects:
                continue
            selected.append(subject)
            seen_subjects.add(dedupe_key)
            retained += 1
            if retained == config.search.archive_capacity_per_cell:
                break
    return tuple(selected)


def load_resources(
    config_source: str | os.PathLike[str] | Mapping[str, Any],
    *,
    repository_root: Path = REPOSITORY_ROOT,
    frozen_bootstrap: Sequence[Mapping[str, Any]] | None = None,
    frozen_seeds: Sequence[Mapping[str, Any]] | None = None,
    frozen_strategies: Sequence[Mapping[str, Any]] | None = None,
    frozen_resume: bool = False,
) -> ResourceSet:
    """Validate every new-run input before creating an output child."""

    objectives = discover_objectives(repository_root / "objectives")
    config = load_config(
        config_source,
        repository_root=repository_root,
        available_objectives=set(objectives),
        # Compatibility is resolved from the actual frozen/loaded index below.
        # Passing this flag avoids rejecting bootstrap_limit=0 before the
        # library can be inspected.
        has_compatible_seed=True,
        frozen_resume=frozen_resume,
    )
    templates, template_digests = _load_templates(repository_root)
    schema_templates, schema_template_digests = _load_schema_templates(repository_root)
    objective = load_objective(
        config.objective,
        repository_root / "objectives",
        selector_fixture_semantic_validator=validate_selector_fixture_semantics,
    )
    schemas = {
        name: materialize_schema(value, objective)
        for name, value in schema_templates.items()
    }
    for name in ROLE_SCHEMA_NAMES:
        try:
            validate_json_schema_contract(schemas[name])
        except JsonSchemaContractError as exc:
            raise EngineError(f"role schema {name} is unsupported: {exc}") from exc
    try:
        authoritative_schemas = AuthoritativeSchemas(
            schemas["event.json"], schemas["work_unit.json"]
        )
    except JournalCorrupt as exc:
        raise EngineError(f"authoritative schema is unsupported: {exc}") from exc
    schema_digests = {
        name: sha256_bytes(canonical_json_bytes(value)) for name, value in schemas.items()
    }
    provider = create_provider(config.provider)
    scripted_digest = provider.content_digest
    if frozen_seeds is None or frozen_strategies is None:
        loaded_seeds, loaded_strategies = _load_library_snapshots(
            config, objective
        )
    else:
        loaded_seeds, loaded_strategies = (), ()
    objective_scope_key = _current_objective_scope_key(
        objective, schema_digests
    )
    judge_protocol_digest = _current_judge_protocol_digest(
        template_digests, schema_digests
    )
    structured_role_candidates = _current_structured_role_scope_candidates(
        config=config,
        template_digests=template_digests,
        schema_digests=schema_digests,
    )
    perturbator_scopes = _current_perturbator_scopes(
        config=config,
        template_digests=template_digests,
        schema_digests=schema_digests,
    )
    if frozen_seeds is None:
        # SeedSnapshot has no self-asserted status field.  Admission above has
        # already replayed the cited source journal and reproduced promotion.
        verified_seed_payloads = _select_verified_seed_snapshot(
            tuple(
                item
                for item in loaded_seeds
                if isinstance(item.get("subject_payload"), Mapping)
                and _seed_prior_static_compatible(
                    item["subject_payload"],
                    config=config,
                    objective=objective,
                    objective_scope_key=objective_scope_key,
                    judge_protocol_digest=judge_protocol_digest,
                    structured_role_candidates=structured_role_candidates,
                    perturbator_scopes=perturbator_scopes,
                )
            ),
            config=config,
            objective=objective,
        )
    else:
        # A resume consumes the already selected complete snapshot rather than
        # rerunning mutable index selection.
        verified_seed_payloads = _unique_frozen_snapshots(
            frozen_seeds,
            identity_field="seed_id",
            label="seed",
        )
        for seed in verified_seed_payloads:
            if not _seed_prior_static_compatible(
                seed,
                config=config,
                objective=objective,
                objective_scope_key=objective_scope_key,
                judge_protocol_digest=judge_protocol_digest,
                structured_role_candidates=structured_role_candidates,
                perturbator_scopes=perturbator_scopes,
            ):
                raise EngineCompatibilityError(
                    f"frozen seed snapshot is statically incompatible: "
                    f"{seed['seed_id']}"
                )
    for seed in verified_seed_payloads:
        seed_type = seed.get("type")
        if seed_type == "entry_seed":
            if seed.get("scenario_recipe") is not None:
                raise EngineError(
                    "selected entry seed contains stateful recipe material"
                )
            continue
        if seed_type != "scenario_recipe":
            raise EngineError("selected seed has an unknown type")
        prompt = seed.get("entry_prompt")
        if not isinstance(prompt, str) or not prompt:
            raise EngineError("selected scenario seed lacks its entry prompt")
        try:
            recipe = parse_scenario_recipe(
                seed.get("scenario_recipe"),
                objective.descriptor_values,
                objective.diagnosis_categories,
            )
            descriptors = objective.validate_descriptors(seed.get("descriptors"))
        except DomainError as exc:
            raise EngineError(f"selected scenario recipe is invalid: {exc}") from exc
        if (
            recipe.entry_prompt_hash
            != sha256_bytes(normalize_line_endings(prompt).encode("utf-8"))
            or recipe.target_generation_profile_digest != config.target.digest
            or dict(recipe.measurement_cell) != descriptors
        ):
            raise EngineError("selected scenario recipe scope is inconsistent")
    if frozen_strategies is None:
        strategies = tuple(
            {
                **dict(item["subject_payload"]),
                "qualification_status": "provisionally_compatible",
            }
            for item in loaded_strategies
            if isinstance(item.get("subject_payload"), Mapping)
            and _strategy_prior_static_compatible(
                item["subject_payload"],
                config=config,
                objective=objective,
                objective_scope_key=objective_scope_key,
                structured_role_candidates=structured_role_candidates,
                perturbator_scopes=perturbator_scopes,
            )
        )
    else:
        strategies = _unique_frozen_snapshots(
            frozen_strategies,
            identity_field="template_id",
            label="strategy",
        )
        for strategy in strategies:
            if not _strategy_prior_static_compatible(
                strategy,
                config=config,
                objective=objective,
                objective_scope_key=objective_scope_key,
                structured_role_candidates=structured_role_candidates,
                perturbator_scopes=perturbator_scopes,
                expected_status=QualificationStatus.PROVISIONALLY_COMPATIBLE,
            ):
                raise EngineCompatibilityError(
                    "frozen strategy snapshot is statically incompatible: "
                    f"{strategy['template_id']}"
                )
    if frozen_bootstrap is not None:
        raw_bootstrap = tuple(frozen_bootstrap)
    else:
        bootstrap = objective.bootstrap
        if config.seed_prompts_path is not None:
            from .objective import load_seed_prompts
            bootstrap = load_seed_prompts(config.seed_prompts_path, objective)
        selected_prompts = {
            normalize_line_endings(str(item["entry_prompt"]))
            for item in verified_seed_payloads
        }
        selected_bootstrap: list[Mapping[str, Any]] = []
        if config.bootstrap_limit:
            for probe in bootstrap["probes"]:
                normalized_prompt = normalize_line_endings(probe["prompt"])
                if normalized_prompt in selected_prompts:
                    continue
                selected_bootstrap.append(probe)
                selected_prompts.add(normalized_prompt)
                if len(selected_bootstrap) == config.bootstrap_limit:
                    break
        raw_bootstrap = tuple(selected_bootstrap)
    if not verified_seed_payloads and not raw_bootstrap:
        raise ConfigError("no compatible verified seed or selected bootstrap probe")
    funded_work_plan(config.to_dict(), raw_bootstrap, verified_seed_payloads)
    scope = digest_json(
        _scientific_scope_material(
            config=config,
            objective=objective,
            template_digests=template_digests,
            schema_digests=schema_digests,
            bootstrap_payloads=raw_bootstrap,
            seed_payloads=verified_seed_payloads,
            strategy_payloads=strategies,
        )
    )
    return ResourceSet(
        config=config,
        objective=objective,
        templates=templates,
        template_digests=template_digests,
        schema_templates=schema_templates,
        schemas=schemas,
        schema_template_digests=schema_template_digests,
        schema_digests=schema_digests,
        authoritative_schemas=authoritative_schemas,
        engine_digest=_engine_source_digest(repository_root),
        reducer_digest=sha256_bytes((repository_root / "quasilisk" / "run_state.py").read_bytes()),
        provider=provider,
        scripted_fixture_digest=scripted_digest,
        bootstrap_payloads=raw_bootstrap,
        seed_snapshot=verified_seed_payloads,
        strategy_snapshot=strategies,
        scientific_scope_digest=scope,
    )


def _candidate_scientific_key(resources: ResourceSet, prompt: str) -> str:
    return scientific_key(
        [
            resources.scientific_scope_digest,
            "standalone_prompt",
            resources.objective.plugin_digest,
            resources.config.target.digest,
            sha256_bytes(normalize_line_endings(prompt).encode("utf-8")),
        ]
    )


def _scenario_scientific_key(resources: ResourceSet, recipe: ScenarioRecipe) -> str:
    return scientific_key(
        [
            resources.scientific_scope_digest,
            "artifact",
            "scenario_recipe",
            recipe.content_hash,
        ]
    )


def _parse_scenario_seed_payload(
    resources: ResourceSet, payload: Mapping[str, Any]
) -> tuple[str, ScenarioRecipe]:
    if payload.get("type") != "scenario_recipe":
        raise EngineError("scenario seed payload has the wrong type")
    prompt = payload.get("entry_prompt")
    raw_recipe = payload.get("scenario_recipe")
    if not isinstance(prompt, str) or not prompt:
        raise EngineError("scenario seed payload lacks its exact entry prompt")
    try:
        recipe = parse_scenario_recipe(
            raw_recipe,
            resources.objective.descriptor_values,
            resources.objective.diagnosis_categories,
        )
        descriptors = resources.objective.validate_descriptors(
            payload.get("descriptors")
        )
    except DomainError as exc:
        raise EngineError(f"scenario seed payload is invalid: {exc}") from exc
    normalized = normalize_line_endings(prompt)
    if recipe.entry_prompt_hash != sha256_bytes(normalized.encode("utf-8")):
        raise EngineError("scenario seed entry prompt identity is inconsistent")
    if recipe.target_generation_profile_digest != resources.config.target.digest:
        raise EngineError("scenario seed target generation profile is incompatible")
    if descriptors != dict(recipe.measurement_cell):
        raise EngineError("scenario seed measurement cell is inconsistent")
    return normalized, recipe


def _build_candidates(resources: ResourceSet, run_id: str) -> dict[str, CandidatePlan]:
    candidates: dict[str, CandidatePlan] = {}
    descriptors_by_key: dict[str, Mapping[str, str]] = {}
    for probe in resources.bootstrap_payloads:
        key = _candidate_scientific_key(resources, probe["prompt"])
        descriptors = resources.objective.validate_descriptors(probe["descriptors"])
        if key in descriptors_by_key and descriptors_by_key[key] != descriptors:
            raise EngineError("trusted initial inputs assign conflicting cells to one prompt")
        descriptors_by_key[key] = descriptors
        candidate_id = entity_id("pc", [run_id, key])
        candidates[key] = CandidatePlan(
            candidate_id=candidate_id,
            scientific_key=key,
            prompt=probe["prompt"],
            descriptors=descriptors,
            origin="bootstrap",
            probe_id=probe["probe_id"],
        )
    # Seed support intentionally admits only complete, exact payloads.  Missing
    # fields cause an index entry to remain an ignored prior rather than being
    # guessed into a candidate.
    for entry in resources.seed_snapshot:
        if entry.get("type") == "scenario_recipe":
            # A stateful prior remains a scenario and is scheduled through its
            # fresh end-to-end recipe.  Dispatching only this entry text would
            # silently turn it into a different one-turn experiment.
            continue
        if entry.get("type") != "entry_seed" or entry.get("scenario_recipe") is not None:
            raise EngineError("verified entry-seed snapshot has an invalid subject type")
        prompt = entry.get("entry_prompt")
        descriptors = entry.get("descriptors")
        if not isinstance(prompt, str) or not isinstance(descriptors, dict):
            continue
        key = _candidate_scientific_key(resources, prompt)
        validated = resources.objective.validate_descriptors(descriptors)
        if key in descriptors_by_key and descriptors_by_key[key] != validated:
            raise EngineError("trusted seed/bootstrap descriptor conflict")
        descriptors_by_key[key] = validated
        candidates[key] = CandidatePlan(
            candidate_id=entity_id("pc", [run_id, key]),
            scientific_key=key,
            prompt=prompt,
            descriptors=validated,
            origin="seed",
            probe_id=None,
        )
    return candidates


def _build_imported_scenarios(
    resources: ResourceSet, run_id: str
) -> dict[str, ScenarioPlan]:
    scenarios: dict[str, ScenarioPlan] = {}
    measurement_by_hash: dict[str, tuple[tuple[str, str], ...]] = {}
    for payload in resources.seed_snapshot:
        if payload.get("type") != "scenario_recipe":
            continue
        prompt, recipe = _parse_scenario_seed_payload(resources, payload)
        prior_cell = measurement_by_hash.get(recipe.content_hash)
        if prior_cell is not None and prior_cell != recipe.measurement_cell:
            raise EngineError(
                "content-equivalent trusted scenario seeds declare conflicting measurement cells"
            )
        measurement_by_hash[recipe.content_hash] = recipe.measurement_cell
        key = _scenario_scientific_key(resources, recipe)
        seed_id = payload.get("seed_id")
        if not isinstance(seed_id, str):
            raise EngineError("scenario seed snapshot lacks its seed ID")
        candidate = ScenarioPlan(
            scenario_id=entity_id("sc", [run_id, key]),
            scientific_key=key,
            entry_prompt=prompt,
            recipe=recipe,
            origin="imported",
            source_seed_id=seed_id,
        )
        incumbent = scenarios.get(key)
        if incumbent is None or seed_id < (incumbent.source_seed_id or ""):
            scenarios[key] = candidate
    return scenarios


def _route_binding(
    context: RunContext,
    *,
    role: Role,
    configured_slot: str,
    route_label: str,
    profile: ModelProfile,
    maximum_attempts: int = 2,
) -> RoleRouteBinding:
    provider_key = requested_provider_route_key(
        context.config.provider.adapter, role, route_label
    )
    key = digest_json(
        [
            context.scope,
            "role_route",
            role.value,
            configured_slot,
            route_label,
            provider_key,
            profile.model,
            profile.digest,
        ]
    )
    binding = RoleRouteBinding(
        route_key=key,
        route_label=route_label,
        configured_slot=configured_slot,
        requested_provider_route_key=provider_key,
        requested_model=profile.model,
        generation_profile_digest=profile.digest,
        maximum_attempts=maximum_attempts,
    )
    context.routes[key] = binding
    return binding


def _initialize_routes(context: RunContext) -> None:
    config = context.config
    bindings = (
        (Role.TARGET, "target", "primary", config.target),
        (Role.JUDGE, "primary", "primary", config.judges.primary),
        (Role.JUDGE, "secondary_a", "primary", config.judges.secondary_a),
        (Role.JUDGE, "secondary_b", "primary", config.judges.secondary_b),
        (Role.JUDGE, "replacement", "replacement", config.judges.replacement),
        (Role.SELECTOR, "selector_primary", "primary", config.selector.primary()),
        (Role.SELECTOR, "selector_fallback", "fallback", config.selector.fallback()),
        (Role.PERTURBATOR, "perturbator_primary", "primary", config.perturbator.primary()),
        (Role.PERTURBATOR, "perturbator_fallback", "fallback", config.perturbator.fallback()),
    )
    for role, slot, label, profile in bindings:
        if profile is None:
            continue
        _route_binding(
            context,
            role=role,
            configured_slot=slot,
            route_label=label,
            profile=profile,
        )


def _binding_for(context: RunContext, configured_slot: str) -> RoleRouteBinding:
    matches = [item for item in context.routes.values() if item.configured_slot == configured_slot]
    if len(matches) != 1:
        raise EngineError(f"configured route slot does not resolve uniquely: {configured_slot}")
    return matches[0]


def _selector_bindings(context: RunContext) -> tuple[RoleRouteBinding, RoleRouteBinding]:
    return (_binding_for(context, "selector_primary"), _binding_for(context, "selector_fallback"))


def _perturbator_bindings(context: RunContext) -> tuple[RoleRouteBinding, RoleRouteBinding]:
    return (
        _binding_for(context, "perturbator_primary"),
        _binding_for(context, "perturbator_fallback"),
    )


def _manifest(context: RunContext) -> dict[str, Any]:
    resources = context.resources
    descriptor_specs = list(context.objective.objective["descriptors"])
    source_checkpoints: list[dict[str, Any]] = []
    for kind, snapshots, id_field in (
        ("seed", resources.seed_snapshot, "seed_id"),
        ("strategy_template", resources.strategy_snapshot, "template_id"),
    ):
        for snapshot in snapshots:
            checkpoint = snapshot.get("source_checkpoint")
            subject_id = snapshot.get(id_field)
            compatibility_key = snapshot.get("compatibility_key")
            if (
                isinstance(checkpoint, Mapping)
                and isinstance(subject_id, str)
                and isinstance(compatibility_key, str)
            ):
                source_checkpoints.append(
                    {
                        "kind": kind,
                        "subject_id": subject_id,
                        "compatibility_key": compatibility_key,
                        "source": dict(checkpoint),
                    }
                )
    source_checkpoints.sort(
        key=lambda item: (item["kind"], item["subject_id"], item["compatibility_key"])
    )
    return {
        "normalized_config": context.config.to_dict(),
        "physical_call_limits": funded_call_limits(funded_work_plan(context.config.to_dict(), resources.bootstrap_payloads, resources.seed_snapshot)).to_dict(),
        "config_digest": context.config.config_digest,
        "scientific_scope_digest": context.scope,
        "engine": {
            # The content digest is authoritative even in an uncommitted
            # checkout.  Do not invent a release label when no VCS revision
            # was explicitly supplied by the build environment.
            "code_revision": None,
            "dirty_content_digest": resources.engine_digest,
        },
        "objective": {
            "id": context.objective.id,
            "plugin_digest": context.objective.plugin_digest,
            "artifact_digests": dict(context.objective.artifact_digests),
        },
        "bootstrap_snapshot": {
            "objective_id": context.objective.id,
            "probes": [dict(item) for item in resources.bootstrap_payloads],
        },
        "seed_snapshot": list(resources.seed_snapshot),
        "strategy_template_snapshot": list(resources.strategy_snapshot),
        "library_source_checkpoints": source_checkpoints,
        "model_profiles": context.config.to_dict()["models"],
        "role_template_digests": {
            name.removesuffix(".md"): digest
            for name, digest in resources.template_digests.items()
        },
        "response_schema_digests": {
            name.removesuffix(".json"): resources.schema_digests[name]
            for name in ROLE_SCHEMA_NAMES
        },
        "judge_output_schema": resources.schemas["judge_output.json"],
        "descriptor_taxonomy": {
            "descriptors": descriptor_specs,
            "digest": digest_json(descriptor_specs),
        },
        "decision_constants": dict(DECISION_CONSTANTS),
        "event_schema_digest": resources.schema_digests["event.json"],
        "work_unit_schema_digest": resources.schema_digests["work_unit.json"],
        "reducer_digest": resources.reducer_digest,
        "random_seed": context.config.random_seed,
        "scripted_fixture_digest": resources.scripted_fixture_digest,
    }


def _action_slot(
    *,
    context: RunContext,
    unit_scientific_key: str,
    owner_position_scientific_key: str,
    ordinal: int,
    role: Role,
    purpose: ActionPurpose,
    dependencies: Sequence[str],
    routes: Sequence[RoleRouteBinding],
    guard: Mapping[str, Any],
) -> dict[str, Any]:
    action_key = scientific_key(
        [
            owner_position_scientific_key,
            "action",
            role.value,
            purpose.value,
            ordinal,
        ]
    )
    action_id = entity_id("la", [context.run_id, action_key])
    return {
        "action_id": action_id,
        "action_scientific_key": action_key,
        "ordinal": ordinal,
        "role": role.value,
        "purpose": purpose.value,
        "owner_position_scientific_key": owner_position_scientific_key,
        "dependency_action_ids": list(dependencies),
        "routes": [route.to_dict() for route in routes],
        "maximum_physical_attempts": sum(route.maximum_attempts for route in routes),
        "guard": dict(guard),
    }


def _target_panel_slots(
    context: RunContext,
    *,
    unit_scientific_key: str,
    owner_position_scientific_key: str,
    start_ordinal: int,
    target_guard: Mapping[str, Any],
    target_dependencies: Sequence[str] = (),
    always_full: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target = _action_slot(
        context=context,
        unit_scientific_key=unit_scientific_key,
        owner_position_scientific_key=owner_position_scientific_key,
        ordinal=start_ordinal,
        role=Role.TARGET,
        purpose=ActionPurpose.TARGET_SAMPLE,
        dependencies=target_dependencies,
        routes=(_binding_for(context, "target"),),
        guard=target_guard,
    )
    panel_key = scientific_key([target["action_scientific_key"], "assessment_panel"])
    panel_id = entity_id("pn", [context.run_id, panel_key])
    primary = _action_slot(
        context=context,
        unit_scientific_key=unit_scientific_key,
        owner_position_scientific_key=owner_position_scientific_key,
        ordinal=start_ordinal + 1,
        role=Role.JUDGE,
        purpose=ActionPurpose.JUDGE_MEMBER,
        dependencies=(target["action_id"],),
        routes=(_binding_for(context, "primary"),),
        guard={"kind": "if_action_usable", "action_id": target["action_id"]},
    )
    secondaries: list[dict[str, Any]] = []
    for offset, configured_slot in enumerate(context.config.judges.normal_slots[1:], 2):
        secondaries.append(
            _action_slot(
                context=context,
                unit_scientific_key=unit_scientific_key,
                owner_position_scientific_key=owner_position_scientific_key,
                ordinal=start_ordinal + offset,
                role=Role.JUDGE,
                purpose=ActionPurpose.JUDGE_MEMBER,
                dependencies=(primary["action_id"],),
                routes=(_binding_for(context, configured_slot),),
                guard={"kind": "if_panel_triggered", "panel_id": panel_id},
            )
        )
    actions = [target, primary, *secondaries]
    replacement = None
    if context.config.judges.replacement is not None:
        replacement = _action_slot(
            context=context,
            unit_scientific_key=unit_scientific_key,
            owner_position_scientific_key=owner_position_scientific_key,
            ordinal=start_ordinal + len(actions),
            role=Role.JUDGE,
            purpose=ActionPurpose.JUDGE_MEMBER,
            dependencies=tuple(item["action_id"] for item in actions[1:]),
            routes=(_binding_for(context, "replacement"),),
            guard={"kind": "if_panel_has_unresolved_normal_member", "panel_id": panel_id},
        )
    panel = {
        "panel_id": panel_id,
        "panel_scientific_key": panel_key,
        "target_action_id": target["action_id"],
        "trigger_policy": "always_full" if always_full else "threshold_or_primary_failure",
        "normal_members": [
            {
                "logical_member_slot": configured_slot,
                "weight": 3 if configured_slot == "primary" else 2,
                "judge_action_id": action["action_id"],
                "judge_action_scientific_key": action["action_scientific_key"],
                "role_route_key": action["routes"][0]["route_key"],
            }
            for configured_slot, action in zip(context.config.judges.normal_slots, actions[1:], strict=True)
        ],
        "replacement": None if replacement is None else {
            "judge_action_id": replacement["action_id"],
            "judge_action_scientific_key": replacement["action_scientific_key"],
            "role_route_key": replacement["routes"][0]["route_key"],
        },
    }
    if replacement is not None:
        actions.append(replacement)
    return actions, panel


def _unit_key(subject_key: str, kind: str, index: int, semantic_design: Any) -> str:
    frozen_design_hash = digest_json(semantic_design)
    return scientific_key([subject_key, "unit", kind, index, frozen_design_hash])


def _response_mode_key(route_key: str, response_schema_digest: str) -> str:
    """Identify one exact route/schema structured-output capability scope."""

    return scientific_key(
        [route_key, "structured_response_mode", response_schema_digest]
    )


def build_initial_observation(context: RunContext, candidate: CandidatePlan, index: int = 0) -> WorkUnit:
    semantic = {
        "kind": "initial_observation",
        "candidate": candidate.scientific_key,
        "descriptors": dict(candidate.descriptors),
        "panel_policy": "threshold_or_primary_failure",
    }
    unit_key = _unit_key(candidate.scientific_key, "initial_observation", index, semantic)
    unit_id = entity_id("wu", [context.run_id, unit_key])
    execution_key = scientific_key([unit_key, "execution", 0, []])
    execution_id = entity_id("ex", [context.run_id, execution_key])
    slots, panel = _target_panel_slots(
        context,
        unit_scientific_key=unit_key,
        owner_position_scientific_key=execution_key,
        start_ordinal=0,
        target_guard={"kind": "always"},
    )
    design = {
        "tag": "initial_observation",
        "candidate_id": candidate.candidate_id,
        "candidate_scientific_key": candidate.scientific_key,
        "execution_id": execution_id,
        "execution_scientific_key": execution_key,
        "panel_policy_digest": POLICY_DIGEST,
    }
    return WorkUnit(
        unit_id=unit_id,
        unit_scientific_key=unit_key,
        kind="initial_observation",
        subject_id=candidate.candidate_id,
        subject_scientific_key=candidate.scientific_key,
        parent_unit_id=None,
        policy_digest=POLICY_DIGEST,
        design=design,
        execution_slots=(
            {
                "execution_id": execution_id,
                "execution_scientific_key": execution_key,
                "ordinal": 0,
                "arm": "none",
                "root_plan_id": None,
            },
        ),
        panel_plans=(panel,),
        logical_action_slots=tuple(slots),
        worst_case_calls=adjust_panel_maximum(initial_observation_maximum(), context.config.judges.maximum_physical_attempts),
    )


def build_screening(
    context: RunContext,
    candidate: CandidatePlan,
    triggering_execution_id: str,
    triggering_execution_scientific_key: str,
    *,
    parent_unit_id: str,
) -> WorkUnit:
    semantic = {
        "kind": "screening",
        "candidate": candidate.scientific_key,
        "triggering_execution": triggering_execution_scientific_key,
        "new_execution_count": 2,
    }
    unit_key = _unit_key(candidate.scientific_key, "screening", 0, semantic)
    unit_id = entity_id("wu", [context.run_id, unit_key])
    block_id = entity_id("sb", [context.run_id, candidate.scientific_key, "screening", 3])
    executions: list[dict[str, Any]] = []
    panels: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    for ordinal in range(2):
        execution_key = scientific_key([unit_key, "execution", ordinal, []])
        execution_id = entity_id("ex", [context.run_id, execution_key])
        slots, panel = _target_panel_slots(
            context,
            unit_scientific_key=unit_key,
            owner_position_scientific_key=execution_key,
            start_ordinal=len(actions),
            target_guard={"kind": "always"},
        )
        executions.append(
            {
                "execution_id": execution_id,
                "execution_scientific_key": execution_key,
                "ordinal": ordinal,
                "arm": "none",
                "root_plan_id": None,
            }
        )
        actions.extend(slots)
        panels.append(panel)
    design = {
        "tag": "screening",
        "candidate_id": candidate.candidate_id,
        "candidate_scientific_key": candidate.scientific_key,
        "block_id": block_id,
        "triggering_execution_id": triggering_execution_id,
        "new_execution_ids": [item["execution_id"] for item in executions],
        "panel_policy_digest": POLICY_DIGEST,
    }
    return WorkUnit(
        unit_id=unit_id,
        unit_scientific_key=unit_key,
        kind="screening",
        subject_id=candidate.candidate_id,
        subject_scientific_key=candidate.scientific_key,
        parent_unit_id=parent_unit_id,
        policy_digest=POLICY_DIGEST,
        design=design,
        execution_slots=tuple(executions),
        panel_plans=tuple(panels),
        logical_action_slots=tuple(actions),
        worst_case_calls=adjust_panel_maximum(screening_maximum(), context.config.judges.maximum_physical_attempts),
    )


def build_verification(
    context: RunContext,
    candidate: CandidatePlan,
    *,
    parent_unit_id: str,
) -> WorkUnit:
    count = context.config.samples.verification
    semantic = {
        "kind": "verification",
        "subject_kind": "entry_prompt",
        "candidate": candidate.scientific_key,
        "planned_count": count,
        "measurement_cell": dict(candidate.descriptors),
        "always_full_panel": True,
    }
    unit_key = _unit_key(candidate.scientific_key, "verification", 0, semantic)
    unit_id = entity_id("wu", [context.run_id, unit_key])
    executions: list[dict[str, Any]] = []
    panels: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    root_plans: list[dict[str, Any]] = []
    for ordinal in range(count):
        execution_key = scientific_key([unit_key, "execution", ordinal, []])
        execution_id = entity_id("ex", [context.run_id, execution_key])
        root_plan_id = entity_id("rp", [context.run_id, unit_key, "root_plan", ordinal])
        slots, panel = _target_panel_slots(
            context,
            unit_scientific_key=unit_key,
            owner_position_scientific_key=execution_key,
            start_ordinal=len(actions),
            target_guard={"kind": "always"},
            always_full=True,
        )
        actions.extend(slots)
        panels.append(panel)
        executions.append(
            {
                "execution_id": execution_id,
                "execution_scientific_key": execution_key,
                "ordinal": ordinal,
                "arm": "none",
                "root_plan_id": root_plan_id,
            }
        )
        root_plans.append(
            {
                "root_plan_id": root_plan_id,
                "execution_id": execution_id,
                "arm": "none",
                "ordered_action_ids": [slot["action_id"] for slot in slots],
                "terminal_endpoint_action_id": slots[0]["action_id"],
                "semantic_nonsuccess_reasons": [],
            }
        )
    design = {
        "tag": "verification",
        "subject_kind": "entry_prompt",
        "subject_id": candidate.candidate_id,
        "subject_scientific_key": candidate.scientific_key,
        "planned_count": count,
        "measurement_cell": dict(candidate.descriptors),
        "root_plans": root_plans,
        "source_preflight": [],
    }
    return WorkUnit(
        unit_id=unit_id,
        unit_scientific_key=unit_key,
        kind="verification",
        subject_id=candidate.candidate_id,
        subject_scientific_key=candidate.scientific_key,
        parent_unit_id=parent_unit_id,
        policy_digest=POLICY_DIGEST,
        design=design,
        execution_slots=tuple(executions),
        panel_plans=tuple(panels),
        logical_action_slots=tuple(actions),
        worst_case_calls=adjust_panel_maximum(entry_verification_maximum(count), context.config.judges.maximum_physical_attempts),
    )


@dataclass(frozen=True, slots=True)
class _ContinuationRootSpec:
    root_ordinal: int
    arm_local_ordinal: int
    arm: str
    realization_intent: str | None
    semantic_nonsuccess_reasons: tuple[str, ...]
    proposal_slot: ProposalSlot | None
    execution_arm: str | None = None
    target_allowed: tuple[str, ...] | None = None


def _strategy_template_identity(
    context: RunContext, template: StrategyTemplate
) -> tuple[str, str]:
    """Return the generated-artifact scientific key and run-local template ID."""

    if not isinstance(template, StrategyTemplate):
        raise EngineError("strategy qualification requires a frozen StrategyTemplate")
    template_key = scientific_key(
        [
            context.scope,
            "artifact",
            "strategy_template",
            template.content_hash,
        ]
    )
    return template_key, entity_id("tp", [context.run_id, template_key])


def _continuation_root_graph(
    context: RunContext,
    *,
    unit_scientific_key: str,
    spec: _ContinuationRootSpec,
    actions: list[dict[str, Any]],
    panels: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Append one complete bounded continuation graph in frozen action order."""

    continuation_path = (
        []
        if spec.proposal_slot is None
        else [spec.proposal_slot.scientific_key]
    )
    execution_key = scientific_key(
        [
            unit_scientific_key,
            "execution",
            spec.root_ordinal,
            continuation_path,
        ]
    )
    execution_id = entity_id("ex", [context.run_id, execution_key])
    root_plan_id = entity_id(
        "rp", [context.run_id, unit_scientific_key, "root_plan", spec.root_ordinal]
    )
    root_action_ids: list[str] = []

    entry_slots, entry_panel = _target_panel_slots(
        context,
        unit_scientific_key=unit_scientific_key,
        owner_position_scientific_key=execution_key,
        start_ordinal=len(actions),
        target_guard={"kind": "always"},
        always_full=True,
    )
    actions.extend(entry_slots)
    panels.append(entry_panel)
    root_action_ids.extend(item["action_id"] for item in entry_slots)

    output_arm = spec.arm if spec.execution_arm is None else spec.execution_arm
    if spec.arm == "parent_baseline":
        return (
            {
                "execution_id": execution_id,
                "execution_scientific_key": execution_key,
                "ordinal": spec.root_ordinal,
                "arm": output_arm,
                "root_plan_id": root_plan_id,
            },
            {
                "root_plan_id": root_plan_id,
                "execution_id": execution_id,
                "arm": output_arm,
                "ordered_action_ids": root_action_ids,
                "terminal_endpoint_action_id": entry_slots[0]["action_id"],
                "semantic_nonsuccess_reasons": [],
            },
        )

    proposal_slot = spec.proposal_slot
    if proposal_slot is None or spec.realization_intent is None:
        raise EngineError("qualification continuation root lacks its frozen slot or intent")
    dependency_ids: tuple[str, ...] = tuple(
        item["action_id"] for item in entry_slots
    )
    binding_action: dict[str, Any] | None = None
    if spec.arm == "treatment":
        binding_action = _action_slot(
            context=context,
            unit_scientific_key=unit_scientific_key,
            owner_position_scientific_key=execution_key,
            ordinal=len(actions),
            role=Role.SELECTOR,
            purpose=ActionPurpose.TEMPLATE_BINDING,
            dependencies=dependency_ids,
            routes=_selector_bindings(context),
            guard={"kind": "always"},
        )
        actions.append(binding_action)
        root_action_ids.append(binding_action["action_id"])
        dependency_ids = (binding_action["action_id"],)

    original = _action_slot(
        context=context,
        unit_scientific_key=unit_scientific_key,
        owner_position_scientific_key=execution_key,
        ordinal=len(actions),
        role=Role.PERTURBATOR,
        purpose=ActionPurpose.PERTURBATOR_PROPOSAL,
        dependencies=dependency_ids,
        routes=_perturbator_bindings(context),
        guard=(
            {"kind": "always"}
            if binding_action is None
            else {
                "kind": "if_template_applicable",
                "binding_action_id": binding_action["action_id"],
            }
        ),
    )
    actions.append(original)
    root_action_ids.append(original["action_id"])
    initial_review = _action_slot(
        context=context,
        unit_scientific_key=unit_scientific_key,
        owner_position_scientific_key=execution_key,
        ordinal=len(actions),
        role=Role.SELECTOR,
        purpose=ActionPurpose.PROBE_REVIEW,
        dependencies=(original["action_id"],),
        routes=_selector_bindings(context),
        guard={
            "kind": "if_any_proposed_item",
            "proposal_action_ids": [original["action_id"]],
        },
    )
    actions.append(initial_review)
    root_action_ids.append(initial_review["action_id"])
    corrective = _action_slot(
        context=context,
        unit_scientific_key=unit_scientific_key,
        owner_position_scientific_key=execution_key,
        ordinal=len(actions),
        role=Role.PERTURBATOR,
        purpose=ActionPurpose.PERTURBATOR_PROPOSAL,
        dependencies=(initial_review["action_id"],),
        routes=_perturbator_bindings(context),
        guard={
            "kind": "if_repairable_proposal_exists",
            "review_action_id": initial_review["action_id"],
            "proposal_slot_scientific_keys": [proposal_slot.scientific_key],
            "statuses": ["invalid"],
        },
    )
    actions.append(corrective)
    root_action_ids.append(corrective["action_id"])
    rereview = _action_slot(
        context=context,
        unit_scientific_key=unit_scientific_key,
        owner_position_scientific_key=execution_key,
        ordinal=len(actions),
        role=Role.SELECTOR,
        purpose=ActionPurpose.PROBE_REVIEW,
        dependencies=(corrective["action_id"],),
        routes=_selector_bindings(context),
        guard={
            "kind": "if_any_corrective_proposal",
            "proposal_action_ids": [corrective["action_id"]],
        },
    )
    actions.append(rereview)
    root_action_ids.append(rereview["action_id"])
    review_item_id = make_review_item_id(
        unit_scientific_key, "probe", proposal_slot.scientific_key
    )
    terminal_slots, terminal_panel = _target_panel_slots(
        context,
        unit_scientific_key=unit_scientific_key,
        owner_position_scientific_key=execution_key,
        start_ordinal=len(actions),
        target_dependencies=(rereview["action_id"],),
        target_guard={
            "kind": "if_dispatchable_probe",
            "review_item_id": review_item_id,
            "proposal_slot_scientific_key": proposal_slot.scientific_key,
            "allowed": list(spec.target_allowed)
            if spec.target_allowed is not None
            else (
                ["aligned"]
                if spec.arm == "treatment"
                else ["valid"]
            ),
        },
        always_full=True,
    )
    actions.extend(terminal_slots)
    panels.append(terminal_panel)
    root_action_ids.extend(item["action_id"] for item in terminal_slots)
    if spec.arm == "treatment":
        outcome = _action_slot(
            context=context,
            unit_scientific_key=unit_scientific_key,
            owner_position_scientific_key=execution_key,
            ordinal=len(actions),
            role=Role.SELECTOR,
            purpose=ActionPurpose.OUTCOME_REVIEW,
            dependencies=tuple(item["action_id"] for item in terminal_slots),
            routes=_selector_bindings(context),
            guard={
                "kind": "if_any_aligned_response",
                "target_action_ids": [terminal_slots[0]["action_id"]],
            },
        )
        actions.append(outcome)
        root_action_ids.append(outcome["action_id"])

    return (
        {
            "execution_id": execution_id,
            "execution_scientific_key": execution_key,
            "ordinal": spec.root_ordinal,
            "arm": output_arm,
            "root_plan_id": root_plan_id,
        },
        {
            "root_plan_id": root_plan_id,
            "execution_id": execution_id,
            "arm": output_arm,
            "ordered_action_ids": root_action_ids,
            "terminal_endpoint_action_id": terminal_slots[0]["action_id"],
            "semantic_nonsuccess_reasons": list(
                spec.semantic_nonsuccess_reasons
            ),
        },
    )


def _scenario_root_graph(
    context: RunContext,
    *,
    unit_scientific_key: str,
    root_ordinal: int,
    proposal_slots: Sequence[ProposalSlot],
    actions: list[dict[str, Any]],
    panels: list[dict[str, Any]],
    preflight_action_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Append one plural, ordered, fresh-grounded scenario root.

    Each continuation owns one anchor slot.  Later bindings are gated by a
    usable prior outcome review, which is the closed-guard expression of the
    frozen ``review_then_rebind`` transition.  The new binding then determines
    applicability against the complete freshly generated branch.
    """

    if not proposal_slots:
        raise EngineError("scenario root requires at least one continuation slot")
    continuation_path = [item.scientific_key for item in proposal_slots]
    execution_key = scientific_key(
        [unit_scientific_key, "execution", root_ordinal, continuation_path]
    )
    execution_id = entity_id("ex", [context.run_id, execution_key])
    root_plan_id = entity_id(
        "rp", [context.run_id, unit_scientific_key, "root_plan", root_ordinal]
    )
    root_action_ids: list[str] = []
    entry_slots, entry_panel = _target_panel_slots(
        context,
        unit_scientific_key=unit_scientific_key,
        owner_position_scientific_key=execution_key,
        start_ordinal=len(actions),
        target_dependencies=() if preflight_action_id is None else (preflight_action_id,),
        target_guard=({"kind": "always"} if preflight_action_id is None else
                      {"kind": "if_template_applicable", "binding_action_id": preflight_action_id}),
        always_full=True,
    )
    actions.extend(entry_slots)
    panels.append(entry_panel)
    root_action_ids.extend(item["action_id"] for item in entry_slots)
    prior_panel_action_ids = tuple(item["action_id"] for item in entry_slots)
    prior_outcome: dict[str, Any] | None = None
    terminal_target_id = entry_slots[0]["action_id"]

    for step_index, proposal_slot in enumerate(proposal_slots):
        if prior_outcome is None:
            binding_dependencies = prior_panel_action_ids
            binding_guard: Mapping[str, Any] = {
                "kind": "if_panel_resolved",
                "panel_id": entry_panel["panel_id"],
            }
        else:
            binding_dependencies = (prior_outcome["action_id"],)
            binding_guard = {
                "kind": "if_action_usable",
                "action_id": prior_outcome["action_id"],
            }
        binding = _action_slot(
            context=context,
            unit_scientific_key=unit_scientific_key,
            owner_position_scientific_key=execution_key,
            ordinal=len(actions),
            role=Role.SELECTOR,
            purpose=ActionPurpose.TEMPLATE_BINDING,
            dependencies=binding_dependencies,
            routes=_selector_bindings(context),
            guard=binding_guard,
        )
        actions.append(binding)
        root_action_ids.append(binding["action_id"])

        original = _action_slot(
            context=context,
            unit_scientific_key=unit_scientific_key,
            owner_position_scientific_key=execution_key,
            ordinal=len(actions),
            role=Role.PERTURBATOR,
            purpose=ActionPurpose.PERTURBATOR_PROPOSAL,
            dependencies=(binding["action_id"],),
            routes=_perturbator_bindings(context),
            guard={
                "kind": "if_template_applicable",
                "binding_action_id": binding["action_id"],
            },
        )
        actions.append(original)
        root_action_ids.append(original["action_id"])
        initial_review = _action_slot(
            context=context,
            unit_scientific_key=unit_scientific_key,
            owner_position_scientific_key=execution_key,
            ordinal=len(actions),
            role=Role.SELECTOR,
            purpose=ActionPurpose.PROBE_REVIEW,
            dependencies=(original["action_id"],),
            routes=_selector_bindings(context),
            guard={
                "kind": "if_any_proposed_item",
                "proposal_action_ids": [original["action_id"]],
            },
        )
        actions.append(initial_review)
        root_action_ids.append(initial_review["action_id"])
        corrective = _action_slot(
            context=context,
            unit_scientific_key=unit_scientific_key,
            owner_position_scientific_key=execution_key,
            ordinal=len(actions),
            role=Role.PERTURBATOR,
            purpose=ActionPurpose.PERTURBATOR_PROPOSAL,
            dependencies=(initial_review["action_id"],),
            routes=_perturbator_bindings(context),
            guard={
                "kind": "if_repairable_proposal_exists",
                "review_action_id": initial_review["action_id"],
                "proposal_slot_scientific_keys": [proposal_slot.scientific_key],
                "statuses": ["invalid"],
            },
        )
        actions.append(corrective)
        root_action_ids.append(corrective["action_id"])
        rereview = _action_slot(
            context=context,
            unit_scientific_key=unit_scientific_key,
            owner_position_scientific_key=execution_key,
            ordinal=len(actions),
            role=Role.SELECTOR,
            purpose=ActionPurpose.PROBE_REVIEW,
            dependencies=(corrective["action_id"],),
            routes=_selector_bindings(context),
            guard={
                "kind": "if_any_corrective_proposal",
                "proposal_action_ids": [corrective["action_id"]],
            },
        )
        actions.append(rereview)
        root_action_ids.append(rereview["action_id"])
        review_item_id = make_review_item_id(
            unit_scientific_key, "probe", proposal_slot.scientific_key
        )
        terminal_slots, terminal_panel = _target_panel_slots(
            context,
            unit_scientific_key=unit_scientific_key,
            owner_position_scientific_key=execution_key,
            start_ordinal=len(actions),
            target_dependencies=(rereview["action_id"],),
            target_guard={
                "kind": "if_dispatchable_probe",
                "review_item_id": review_item_id,
                "proposal_slot_scientific_key": proposal_slot.scientific_key,
                "allowed": ["aligned"],
            },
            always_full=True,
        )
        actions.extend(terminal_slots)
        panels.append(terminal_panel)
        root_action_ids.extend(item["action_id"] for item in terminal_slots)
        terminal_target_id = terminal_slots[0]["action_id"]
        prior_panel_action_ids = tuple(item["action_id"] for item in terminal_slots)
        prior_outcome = _action_slot(
            context=context,
            unit_scientific_key=unit_scientific_key,
            owner_position_scientific_key=execution_key,
            ordinal=len(actions),
            role=Role.SELECTOR,
            purpose=ActionPurpose.OUTCOME_REVIEW,
            dependencies=prior_panel_action_ids,
            routes=_selector_bindings(context),
            guard={
                "kind": "if_any_aligned_response",
                "target_action_ids": [terminal_target_id],
            },
        )
        actions.append(prior_outcome)
        root_action_ids.append(prior_outcome["action_id"])

    return (
        {
            "execution_id": execution_id,
            "execution_scientific_key": execution_key,
            "ordinal": root_ordinal,
            "arm": "none",
            "root_plan_id": root_plan_id,
        },
        {
            "root_plan_id": root_plan_id,
            "execution_id": execution_id,
            "arm": "none",
            "ordered_action_ids": root_action_ids,
            "terminal_endpoint_action_id": terminal_target_id,
            "semantic_nonsuccess_reasons": ["inapplicable", "nonprogress"],
        },
    )


def build_scenario_verification(
    context: RunContext,
    scenario: ScenarioPlan,
) -> WorkUnit:
    """Freeze N fresh, anchor-only end-to-end scenario executions.

    Every ordered template is preallocated as one bind/generate/review/
    correction/target/outcome step.  Runtime may close later guarded slots but
    can never extend this frozen topology.
    """

    if not isinstance(scenario, ScenarioPlan):
        raise EngineError("scenario verification requires a frozen ScenarioPlan")
    recipe = scenario.recipe
    if not recipe.templates or recipe.designated_realization_policies != tuple(
        "anchor" for _ in recipe.templates
    ):
        raise EngineError("scenario verification requires designated anchor steps")
    source_keys = scenario.source_parent_node_keys
    if ((scenario.origin == "derived" and len(source_keys) != len(recipe.templates))
            or (scenario.origin == "imported" and source_keys)):
        raise EngineError("scenario source preflight must match its exact source steps")
    count = context.config.samples.verification
    measurement_cell = descriptors_dict(recipe.measurement_cell)
    semantic = {
        "kind": "verification",
        "subject_kind": "scenario_recipe",
        "scenario_recipe_content_hash": recipe.content_hash,
        "planned_count": count,
        "measurement_cell": measurement_cell,
        "root_blueprint": "fresh_anchor_scenario",
        "always_full_panel": True,
        "source_parent_node_keys": list(source_keys),
    }
    unit_key = _unit_key(
        scenario.scientific_key, "verification", 0, semantic
    )
    unit_id = entity_id("wu", [context.run_id, unit_key])
    depth = len(recipe.templates)
    proposal_slots = make_proposal_slots(unit_key, count * depth)
    actions: list[dict[str, Any]] = []
    panels: list[dict[str, Any]] = []
    executions: list[dict[str, Any]] = []
    roots: list[dict[str, Any]] = []
    preflight: list[dict[str, Any]] = []
    for index, parent_key in enumerate(source_keys):
        previous = None if not actions else actions[-1]["action_id"]
        binding = _action_slot(
            context=context, unit_scientific_key=unit_key,
            owner_position_scientific_key=unit_key, ordinal=len(actions),
            role=Role.SELECTOR, purpose=ActionPurpose.TEMPLATE_BINDING,
            dependencies=() if previous is None else (previous,),
            routes=_selector_bindings(context),
            guard=({"kind": "always"} if previous is None else
                   {"kind": "if_template_applicable", "binding_action_id": previous}),
        )
        actions.append(binding)
        preflight.append({"action_id": binding["action_id"], "template_index": index,
                          "parent_node_scientific_key": parent_key})
    for ordinal in range(count):
        root_slots = proposal_slots[ordinal * depth : (ordinal + 1) * depth]
        execution, root = _scenario_root_graph(
            context,
            unit_scientific_key=unit_key,
            root_ordinal=ordinal,
            proposal_slots=root_slots,
            actions=actions,
            panels=panels,
            preflight_action_id=None if not preflight else preflight[-1]["action_id"],
        )
        executions.append(execution)
        roots.append(root)
    design = {
        "tag": "verification",
        "subject_kind": "scenario_recipe",
        "subject_id": scenario.scenario_id,
        "subject_scientific_key": scenario.scientific_key,
        "planned_count": count,
        "measurement_cell": measurement_cell,
        "root_plans": roots,
        "source_preflight": preflight,
    }
    expected = sum_root_maxima(
        adjust_panel_maximum(stateful_root_maximum(depth), context.config.judges.maximum_physical_attempts) for _ in range(count)
    ) + RoleCallVector(selector=4 * len(source_keys))
    maximum = action_slot_maximum(actions)
    if maximum != expected:
        raise EngineError(
            "scenario verification graph disagrees with its preregistered maximum"
        )
    return WorkUnit(
        unit_id=unit_id,
        unit_scientific_key=unit_key,
        kind="verification",
        subject_id=scenario.scenario_id,
        subject_scientific_key=scenario.scientific_key,
        parent_unit_id=scenario.parent_unit_id,
        policy_digest=POLICY_DIGEST,
        design=design,
        proposal_slots=tuple(item.to_dict() for item in proposal_slots),
        execution_slots=tuple(executions),
        panel_plans=tuple(panels),
        logical_action_slots=tuple(actions),
        worst_case_calls=maximum,
    )


def _qualification_maximum_from_slots(
    actions: Sequence[Mapping[str, Any]], expected: RoleCallVector
) -> RoleCallVector:
    actual = action_slot_maximum(actions)
    if actual != expected:
        raise EngineError(
            "strategy qualification action graph disagrees with its exact maximum"
        )
    return actual


def build_strategy_development(
    context: RunContext,
    template: StrategyTemplate,
    *,
    parent_unit_id: str,
) -> WorkUnit:
    """Freeze the three-arm development block and its future holdout seal."""

    template_key, template_id = _strategy_template_identity(context, template)
    count = context.config.samples.strategy_qualification
    holdout_count = context.config.samples.holdout
    measurement_cell = descriptors_dict(template.move.descriptors)
    comparison_dimensions = derive_template_comparison_dimensions(template)
    if not comparison_dimensions:
        raise EngineError(
            "a template without a control-safe dimension cannot enter development"
        )
    intent_assignment = freeze_qualification_intent_assignment(
        count,
        random_seed=context.config.random_seed,
        qualification_subject_scientific_key=template_key,
        phase="development",
    )
    arm_order = freeze_development_arm_order(
        count,
        random_seed=context.config.random_seed,
        qualification_subject_scientific_key=template_key,
    )
    sealed_holdout = build_sealed_holdout_plan(
        template,
        planned_count_per_arm=holdout_count,
        random_seed=context.config.random_seed,
        qualification_subject_scientific_key=template_key,
        measurement_cell=measurement_cell,
        descriptor_vocabulary=context.objective.descriptor_values,
    )
    semantic = {
        "kind": "strategy_development",
        "strategy_template": template.to_dict(),
        "planned_count_per_arm": count,
        "comparison_dimensions": list(comparison_dimensions),
        "realization_intent_assignment": list(intent_assignment),
        "interleaved_root_order": [
            [item.arm, item.arm_local_ordinal] for item in arm_order
        ],
        "root_blueprints": [
            "treatment_qualification",
            "parent_baseline",
            "strategy_agnostic_qualification",
        ],
        "measurement_cell": measurement_cell,
        "sealed_holdout_plan": sealed_holdout.to_dict(),
    }
    unit_key = _unit_key(template_key, "strategy_development", 0, semantic)
    unit_id = entity_id("wu", [context.run_id, unit_key])
    proposal_slots = make_proposal_slots(unit_key, 2 * count)
    arm_offsets = {
        "treatment": 0,
        "parent_baseline": count,
        "strategy_agnostic": 2 * count,
    }
    specs: dict[tuple[str, int], _ContinuationRootSpec] = {}
    for arm in ("treatment", "parent_baseline", "strategy_agnostic"):
        for local_ordinal in range(count):
            root_ordinal = arm_offsets[arm] + local_ordinal
            proposal = (
                proposal_slots[local_ordinal]
                if arm == "treatment"
                else (
                    proposal_slots[count + local_ordinal]
                    if arm == "strategy_agnostic"
                    else None
                )
            )
            specs[(arm, local_ordinal)] = _ContinuationRootSpec(
                root_ordinal,
                local_ordinal,
                arm,
                None if arm == "parent_baseline" else intent_assignment[local_ordinal],
                ("inapplicable",) if arm == "treatment" else (),
                proposal,
            )

    actions: list[dict[str, Any]] = []
    panels: list[dict[str, Any]] = []
    executions_by_ordinal: dict[int, dict[str, Any]] = {}
    plans_by_ordinal: dict[int, dict[str, Any]] = {}
    for position in arm_order:
        spec = specs[(position.arm, position.arm_local_ordinal)]
        execution, plan = _continuation_root_graph(
            context,
            unit_scientific_key=unit_key,
            spec=spec,
            actions=actions,
            panels=panels,
        )
        executions_by_ordinal[spec.root_ordinal] = execution
        plans_by_ordinal[spec.root_ordinal] = plan
    executions = tuple(
        executions_by_ordinal[index] for index in range(3 * count)
    )
    root_plans = tuple(plans_by_ordinal[index] for index in range(3 * count))
    design = {
        "tag": "strategy_development",
        "strategy_template_id": template_id,
        "strategy_template_scientific_key": template_key,
        "planned_count_per_arm": count,
        "comparison_dimensions": list(comparison_dimensions),
        "realization_intent_assignment": list(intent_assignment),
        "interleaved_root_order": [
            plans_by_ordinal[
                arm_offsets[item.arm] + item.arm_local_ordinal
            ]["root_plan_id"]
            for item in arm_order
        ],
        "root_plans": list(root_plans),
        "sealed_holdout_plan": sealed_holdout.to_dict(),
        "sealed_holdout_digest": sealed_holdout.content_digest,
        "measurement_cell": measurement_cell,
    }
    expected = adjust_panel_maximum(strategy_development_maximum(count), context.config.judges.maximum_physical_attempts)
    maximum = _qualification_maximum_from_slots(actions, expected)
    return WorkUnit(
        unit_id=unit_id,
        unit_scientific_key=unit_key,
        kind="strategy_development",
        subject_id=template_id,
        subject_scientific_key=template_key,
        parent_unit_id=parent_unit_id,
        policy_digest=POLICY_DIGEST,
        design=design,
        proposal_slots=tuple(item.to_dict() for item in proposal_slots),
        execution_slots=executions,
        panel_plans=tuple(panels),
        logical_action_slots=tuple(actions),
        worst_case_calls=maximum,
    )


def build_strategy_holdout(
    context: RunContext,
    template: StrategyTemplate,
    *,
    source_development_unit: Mapping[str, Any],
) -> WorkUnit:
    """Materialize a two-arm holdout solely from the development-time seal."""

    if not isinstance(source_development_unit, Mapping):
        raise EngineError("holdout source development unit must be an object")
    source = dict(source_development_unit)
    if source.get("kind") != "strategy_development":
        raise EngineError("holdout source is not a strategy development unit")
    source_id = source.get("unit_id")
    source_design = source.get("design")
    if not isinstance(source_id, str) or not isinstance(source_design, Mapping):
        raise EngineError("holdout source development unit is incomplete")
    template_key, template_id = _strategy_template_identity(context, template)
    if (
        source.get("subject_id") != template_id
        or source.get("subject_scientific_key") != template_key
        or source_design.get("strategy_template_id") != template_id
        or source_design.get("strategy_template_scientific_key") != template_key
    ):
        raise EngineError("holdout source changed its frozen StrategyTemplate identity")
    sealed_plan = parse_sealed_holdout_plan(
        source_design.get("sealed_holdout_plan"),
        context.objective.descriptor_values,
    )
    source_digest = source_design.get("sealed_holdout_digest")
    if source_digest != sealed_plan.content_digest:
        raise EngineError("development holdout seal digest is inconsistent")
    if sealed_plan.planned_count_per_arm != context.config.samples.holdout:
        raise EngineError("development holdout seal changed the configured root count")
    if descriptors_dict(sealed_plan.measurement_cell) != descriptors_dict(
        template.move.descriptors
    ):
        raise EngineError("development holdout seal changed the template measurement cell")

    semantic = {
        "kind": "strategy_holdout",
        "strategy_template": template.to_dict(),
        "sealed_holdout_plan": sealed_plan.to_dict(),
    }
    unit_key = _unit_key(template_key, "strategy_holdout", 0, semantic)
    unit_id = entity_id("wu", [context.run_id, unit_key])
    count = sealed_plan.planned_count_per_arm
    proposal_slots = make_proposal_slots(unit_key, 2 * count)
    future_by_ordinal = {
        item.root_ordinal: item for item in sealed_plan.future_roots
    }
    specs: dict[int, _ContinuationRootSpec] = {}
    for root_ordinal in range(2 * count):
        future = future_by_ordinal[root_ordinal]
        specs[root_ordinal] = _ContinuationRootSpec(
            root_ordinal,
            future.arm_local_ordinal,
            future.arm,
            future.realization_intent,
            future.semantic_nonsuccess_reasons,
            proposal_slots[root_ordinal],
        )

    actions: list[dict[str, Any]] = []
    panels: list[dict[str, Any]] = []
    executions_by_ordinal: dict[int, dict[str, Any]] = {}
    plans_by_ordinal: dict[int, dict[str, Any]] = {}
    for root_ordinal in sealed_plan.interleaved_root_order:
        spec = specs[root_ordinal]
        execution, plan = _continuation_root_graph(
            context,
            unit_scientific_key=unit_key,
            spec=spec,
            actions=actions,
            panels=panels,
        )
        executions_by_ordinal[root_ordinal] = execution
        plans_by_ordinal[root_ordinal] = plan
    executions = tuple(
        executions_by_ordinal[index] for index in range(2 * count)
    )
    root_plans = tuple(plans_by_ordinal[index] for index in range(2 * count))
    materialized = {
        "source_sealed_holdout_digest": source_digest,
        "planned_count_per_arm": count,
        "comparison_dimensions": list(sealed_plan.comparison_dimensions),
        "future_roots": [
            {
                "execution_id": executions_by_ordinal[item.root_ordinal][
                    "execution_id"
                ],
                "execution_scientific_key": executions_by_ordinal[
                    item.root_ordinal
                ]["execution_scientific_key"],
                "root_plan_id": plans_by_ordinal[item.root_ordinal][
                    "root_plan_id"
                ],
                **item.to_dict(),
            }
            for item in sealed_plan.future_roots
        ],
        "interleaved_root_order": [
            plans_by_ordinal[index]["root_plan_id"]
            for index in sealed_plan.interleaved_root_order
        ],
        "information_masks": sealed_plan.information_masks.to_dict(),
        "endpoint_rule": sealed_plan.endpoint_rule,
        "measurement_cell": descriptors_dict(sealed_plan.measurement_cell),
    }
    validate_materialized_holdout_plan(
        sealed_plan, materialized, context.objective.descriptor_values
    )
    design = {
        "tag": "strategy_holdout",
        "strategy_template_id": template_id,
        "strategy_template_scientific_key": template_key,
        "source_development_unit_id": source_id,
        "source_sealed_holdout_digest": source_digest,
        "sealed_holdout_plan": sealed_plan.to_dict(),
        "root_plans": list(root_plans),
    }
    expected = adjust_panel_maximum(strategy_holdout_maximum(count), context.config.judges.maximum_physical_attempts)
    maximum = _qualification_maximum_from_slots(actions, expected)
    return WorkUnit(
        unit_id=unit_id,
        unit_scientific_key=unit_key,
        kind="strategy_holdout",
        subject_id=template_id,
        subject_scientific_key=template_key,
        parent_unit_id=source_id,
        policy_digest=POLICY_DIGEST,
        design=design,
        proposal_slots=tuple(item.to_dict() for item in proposal_slots),
        execution_slots=executions,
        panel_plans=tuple(panels),
        logical_action_slots=tuple(actions),
        worst_case_calls=maximum,
    )


def _descriptor_grid(objective: ObjectivePlugin) -> tuple[dict[str, str], ...]:
    """Return the declared Cartesian grid in canonical descriptor order."""

    return tuple(
        dict(zip(objective.descriptor_names, values, strict=True))
        for values in itertools.product(
            *(objective.descriptor_values[name] for name in objective.descriptor_names)
        )
    )


def build_map_exploration(
    context: RunContext,
    parent: CandidatePlan,
    *,
    parent_unit_id: str,
    archive_snapshot: ArchiveSnapshot | None = None,
    prior_directed_slots: Mapping[tuple[str, ...], int] | None = None,
) -> WorkUnit:
    """Freeze one ordinary MAP batch before any generator output exists."""

    proposal_count = context.config.search.proposals_per_step
    entry_key = scientific_key([parent.scientific_key, "archive_entry"])
    entry_id = entity_id("ae", [context.run_id, entry_key])
    grid = _descriptor_grid(context.objective)
    if archive_snapshot is None:
        under_covered = list(grid[: min(proposal_count, len(grid))])
    else:
        ordered_cells = under_covered_cells(
            (
                tuple(cell[name] for name in context.objective.descriptor_names)
                for cell in grid
            ),
            archive_snapshot,
            prior_directed_slots=prior_directed_slots,
            limit=proposal_count,
        )
        under_covered = [
            dict(zip(context.objective.descriptor_names, cell, strict=True))
            for cell in ordered_cells
        ]
    semantic = {
        "kind": "map_exploration",
        "parent": parent.scientific_key,
        "proposal_count": proposal_count,
        "under_covered_cells": under_covered,
        "allowed_dimensions": list(VARIATION_DIMENSIONS),
    }
    unit_key = _unit_key(entry_key, "map_exploration", 0, semantic)
    unit_id = entity_id("wu", [context.run_id, unit_key])
    proposals = make_proposal_slots(unit_key, proposal_count)
    proposal_action = _action_slot(
        context=context,
        unit_scientific_key=unit_key,
        owner_position_scientific_key=unit_key,
        ordinal=0,
        role=Role.PERTURBATOR,
        purpose=ActionPurpose.PERTURBATOR_PROPOSAL,
        dependencies=(),
        routes=_perturbator_bindings(context),
        guard={"kind": "always"},
    )
    review_action = _action_slot(
        context=context,
        unit_scientific_key=unit_key,
        owner_position_scientific_key=unit_key,
        ordinal=1,
        role=Role.SELECTOR,
        purpose=ActionPurpose.PROBE_REVIEW,
        dependencies=(proposal_action["action_id"],),
        routes=_selector_bindings(context),
        guard={
            "kind": "if_any_proposed_item",
            "proposal_action_ids": [proposal_action["action_id"]],
        },
    )
    actions: list[dict[str, Any]] = [proposal_action, review_action]
    panels: list[dict[str, Any]] = []
    executions: list[dict[str, Any]] = []
    for ordinal, proposal in enumerate(proposals):
        execution_key = scientific_key(
            [unit_key, "execution", ordinal, [proposal.scientific_key]]
        )
        execution_id = entity_id("ex", [context.run_id, execution_key])
        review_item_id = make_review_item_id(unit_key, "probe", proposal.scientific_key)
        slots, panel = _target_panel_slots(
            context,
            unit_scientific_key=unit_key,
            owner_position_scientific_key=execution_key,
            start_ordinal=len(actions),
            target_dependencies=(review_action["action_id"],),
            target_guard={
                "kind": "if_dispatchable_probe",
                "review_item_id": review_item_id,
                "proposal_slot_scientific_key": proposal.scientific_key,
                "allowed": ["valid"],
            },
        )
        actions.extend(slots)
        panels.append(panel)
        executions.append(
            {
                "execution_id": execution_id,
                "execution_scientific_key": execution_key,
                "ordinal": ordinal,
                "arm": "none",
                "root_plan_id": None,
            }
        )
    design = {
        "tag": "map_exploration",
        "parent_archive_entry_id": entry_id,
        "parent_archive_entry_scientific_key": entry_key,
        "proposal_count": proposal_count,
        "under_covered_cells": under_covered,
        "allowed_dimensions": list(VARIATION_DIMENSIONS),
        "probe_review_scope_key": scientific_key(
            [context.scope, "probe_review", "ordinary", context.resources.schema_digests["probe_review.json"]]
        ),
    }
    return WorkUnit(
        unit_id=unit_id,
        unit_scientific_key=unit_key,
        kind="map_exploration",
        subject_id=entry_id,
        subject_scientific_key=entry_key,
        parent_unit_id=parent_unit_id,
        policy_digest=POLICY_DIGEST,
        design=design,
        proposal_slots=tuple(item.to_dict() for item in proposals),
        execution_slots=tuple(executions),
        panel_plans=tuple(panels),
        logical_action_slots=tuple(actions),
        worst_case_calls=adjust_panel_maximum(map_exploration_maximum(proposal_count), context.config.judges.maximum_physical_attempts),
    )


def build_portfolio_round(
    context: RunContext,
    parent: CandidatePlan,
    *,
    parent_unit_id: str,
    conversation_root_id: str,
    conversation_root_scientific_key: str,
    parent_node_id: str,
    parent_node_scientific_key: str,
    portfolio_intent: PortfolioIntent,
) -> WorkUnit:
    """Freeze one Selector-to-target strategy step on an exact branch.

    Continued pursuit happens through a newly grounded contract on a
    supporting child.  A contract is never silently reused against a later
    target response.
    """

    del parent  # Exact generation-parent identities below own the lineage.
    if not isinstance(portfolio_intent, PortfolioIntent):
        raise EngineError("portfolio intent is invalid")
    proposal_count = context.config.search.proposals_per_step
    maximum_strategy_count = proposal_count // 3
    semantic = {
        "kind": "portfolio_round",
        "conversation_root": conversation_root_scientific_key,
        "parent_node": parent_node_scientific_key,
        "portfolio_intent": portfolio_intent.value,
        "round_index": 0,
        "proposal_count": proposal_count,
        "maximum_strategy_count": maximum_strategy_count,
    }
    unit_key = _unit_key(
        conversation_root_scientific_key, "portfolio_round", 0, semantic
    )
    unit_id = entity_id("wu", [context.run_id, unit_key])
    strategy_slots = tuple(strategy_slot(unit_key, ordinal) for ordinal in range(maximum_strategy_count))
    proposals = make_proposal_slots(unit_key, proposal_count)
    actions: list[dict[str, Any]] = []
    selector_action = _action_slot(
        context=context,
        unit_scientific_key=unit_key,
        owner_position_scientific_key=parent_node_scientific_key,
        ordinal=0,
        role=Role.SELECTOR,
        purpose=ActionPurpose.SELECTOR_PORTFOLIO,
        dependencies=(),
        routes=_selector_bindings(context),
        guard={"kind": "always"},
    )
    actions.append(selector_action)
    lane_owner_keys = [slot.scientific_key for slot in strategy_slots]

    original_actions: list[dict[str, Any]] = []
    for owner_key in lane_owner_keys:
        action = _action_slot(
            context=context,
            unit_scientific_key=unit_key,
            owner_position_scientific_key=owner_key,
            ordinal=len(actions),
            role=Role.PERTURBATOR,
            purpose=ActionPurpose.PERTURBATOR_PROPOSAL,
            dependencies=(selector_action["action_id"],),
            routes=_perturbator_bindings(context),
            guard={
                "kind": "if_strategy_slot_bound",
                "strategy_slot_scientific_key": owner_key,
            },
        )
        actions.append(action)
        original_actions.append(action)
    initial_review = _action_slot(
        context=context,
        unit_scientific_key=unit_key,
        owner_position_scientific_key=unit_key,
        ordinal=len(actions),
        role=Role.SELECTOR,
        purpose=ActionPurpose.PROBE_REVIEW,
        dependencies=tuple(item["action_id"] for item in original_actions),
        routes=(_binding_for(context, "selector_primary"),),
        guard={"kind": "if_any_proposed_item", "proposal_action_ids": [item["action_id"] for item in original_actions]},
    )
    actions.append(initial_review)

    panels: list[dict[str, Any]] = []
    executions: list[dict[str, Any]] = []
    target_ids: list[str] = []
    outcome_dependencies: list[str] = []
    for ordinal, proposal in enumerate(proposals):
        execution_key = scientific_key(
            [unit_key, "execution", ordinal, [proposal.scientific_key]]
        )
        execution_id = entity_id("ex", [context.run_id, execution_key])
        root_plan_id = entity_id("rp", [context.run_id, execution_key, "root"])
        review_item_id = make_review_item_id(unit_key, "probe", proposal.scientific_key)
        slots, panel = _target_panel_slots(
            context,
            unit_scientific_key=unit_key,
            owner_position_scientific_key=execution_key,
            start_ordinal=len(actions),
            target_dependencies=(initial_review["action_id"],),
            target_guard={
                "kind": "if_dispatchable_probe",
                "review_item_id": review_item_id,
                "proposal_slot_scientific_key": proposal.scientific_key,
                "allowed": ["aligned"],
            },
        )
        actions.extend(slots)
        target_ids.append(slots[0]["action_id"])
        outcome_dependencies.extend(item["action_id"] for item in slots)
        panels.append(panel)
        executions.append(
            {
                "execution_id": execution_id,
                "execution_scientific_key": execution_key,
                "ordinal": ordinal,
                "arm": "treatment",
                "root_plan_id": root_plan_id,
            }
        )
    outcome = _action_slot(
        context=context,
        unit_scientific_key=unit_key,
        owner_position_scientific_key=unit_key,
        ordinal=len(actions),
        role=Role.SELECTOR,
        purpose=ActionPurpose.OUTCOME_REVIEW,
        dependencies=tuple(outcome_dependencies),
        routes=_selector_bindings(context),
        guard={"kind": "if_any_aligned_response", "target_action_ids": target_ids},
    )
    actions.append(outcome)
    design = {
        "tag": "portfolio_round",
        "portfolio_intent": portfolio_intent.value,
        "conversation_root_id": conversation_root_id,
        "generation_parent_node_id": parent_node_id,
        "generation_parent_node_scientific_key": parent_node_scientific_key,
        "round_index": 0,
        "proposal_count": proposal_count,
        "maximum_strategy_count": maximum_strategy_count,
        "strategy_slots": [
            {"entity_id": item.entity_id, "scientific_key": item.scientific_key, "ordinal": item.ordinal}
            for item in strategy_slots
        ],
        "active_strategy_ids": [],
        "frozen_assignments": [],
        "review_scope_keys": {
            "probe_review": scientific_key([context.scope, "probe_review", "strategy"]),
            "outcome_review": scientific_key([context.scope, "outcome_review"]),
        },
    }
    return WorkUnit(
        unit_id=unit_id,
        unit_scientific_key=unit_key,
        kind="portfolio_round",
        subject_id=conversation_root_id,
        subject_scientific_key=conversation_root_scientific_key,
        parent_unit_id=parent_unit_id,
        policy_digest=POLICY_DIGEST,
        design=design,
        proposal_slots=tuple(item.to_dict() for item in proposals),
        execution_slots=tuple(executions),
        panel_plans=tuple(panels),
        logical_action_slots=tuple(actions),
        worst_case_calls=adjust_panel_maximum(portfolio_round_maximum(
            proposal_count,
            maximum_strategy_count,
        ), context.config.judges.maximum_physical_attempts),
    )


@dataclass(frozen=True, slots=True)
class SemanticResult:
    value: Any | None
    errors: tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        return self.value is not None


class _StopRun(EngineError):
    def __init__(self, reason: RunTerminalReason, detail: str, *, resumable: bool):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail
        self.resumable = resumable


def _render_template(template: str, replacements: Mapping[str, str]) -> str:
    rendered = template
    for marker, value in replacements.items():
        token = "{{" + marker + "}}"
        if rendered.count(token) != 1:
            raise EngineError(f"template marker {token} must occur exactly once")
        rendered = rendered.replace(token, value)
    if re.search(r"\{\{[A-Z_]+\}\}", rendered):
        raise EngineError("unresolved template marker")
    return rendered


def _schema_for_purpose(context: RunContext, purpose: str) -> tuple[Mapping[str, Any], str] | None:
    name = {
        ActionPurpose.JUDGE_MEMBER.value: "judge_output.json",
        ActionPurpose.SELECTOR_PORTFOLIO.value: "selector_portfolio.json",
        ActionPurpose.PROBE_REVIEW.value: "probe_review.json",
        ActionPurpose.TEMPLATE_BINDING.value: "template_binding.json",
        ActionPurpose.OUTCOME_REVIEW.value: "outcome_review.json",
        ActionPurpose.PERTURBATOR_PROPOSAL.value: "proposal_batch.json",
    }.get(purpose)
    if name is None:
        return None
    return context.resources.schemas[name], context.resources.schema_digests[name]


def _schema_for_action(context: RunContext, state: RunState, unit: Any,
                       action: LogicalActionView) -> tuple[Mapping[str, Any], str] | None:
    info = _schema_for_purpose(context, action.slot["purpose"])
    if action.slot["role"] != Role.JUDGE.value:
        return info
    panel = _panel_for_action(unit, action.action_id)
    if panel is None or panel["target_action_id"] not in state.target_results:
        raise EngineError("judge schema lacks durable target response")
    assert info is not None
    schema = materialize_judge_schema(info[0], state.target_results[panel["target_action_id"]].text)
    return schema, digest_json(schema)


def _unit_for_action(state: RunState, action_id: str) -> Any:
    for unit in state.work_units:
        if any(action.action_id == action_id for action in unit.actions):
            return unit
    raise EngineError(f"action {action_id} is not owned by a work unit")


def _panel_for_action(unit: Any, action_id: str) -> Mapping[str, Any] | None:
    for panel in unit.unit["panel_plans"]:
        if panel["target_action_id"] == action_id:
            return panel
        if any(item["judge_action_id"] == action_id for item in panel["normal_members"]):
            return panel
        if panel["replacement"] is not None and panel["replacement"]["judge_action_id"] == action_id:
            return panel
    return None


def _candidate_by_id(context: RunContext, candidate_id: str | None) -> CandidatePlan | None:
    if candidate_id is None:
        return None
    return next(
        (candidate for candidate in context.candidates.values() if candidate.candidate_id == candidate_id),
        None,
    )


def _map_parent_candidate(context: RunContext, unit: Any) -> CandidatePlan | None:
    if unit.unit["kind"] != "map_exploration":
        return None
    frozen_key = unit.unit["design"]["parent_archive_entry_scientific_key"]
    return next(
        (
            candidate
            for candidate in context.candidates.values()
            if scientific_key([candidate.scientific_key, "archive_entry"])
            == frozen_key
        ),
        None,
    )


def _proposal_slots(unit: Any) -> tuple[ProposalSlot, ...]:
    return tuple(
        ProposalSlot(item["slot_id"], item["scientific_key"], item["global_ordinal"])
        for item in unit.unit["proposal_slots"]
    )


def _successful_payloads(action: LogicalActionView) -> tuple[tuple[AttemptView, Mapping[str, Any]], ...]:
    result: list[tuple[AttemptView, Mapping[str, Any]]] = []
    for attempt in action.attempts:
        if (
            attempt.succeeded
            and attempt.terminal_event is not None
            and not attempt.target_identity_collision
        ):
            result.append((attempt, attempt.terminal_event["payload"]))
    return tuple(result)


def _schema_validated_payloads(
    context: RunContext,
    action: LogicalActionView,
    *,
    incomplete_error: str,
) -> tuple[
    tuple[tuple[AttemptView, Mapping[str, Any], Any], ...],
    tuple[str, ...],
]:
    """Parse and schema-check structured attempts before role semantics run."""

    schema_info = _schema_for_purpose(context, action.slot["purpose"])
    if schema_info is None:
        return (), ("structured role action has no response schema",)
    schema, _ = schema_info
    cached = context.validated_payloads.get(action.action_id)
    if cached is not None and cached[0] is action and cached[1] is schema and cached[2] == incomplete_error:
        return cached[3], cached[4]
    accepted: list[tuple[AttemptView, Mapping[str, Any], Any]] = []
    errors: list[str] = []
    for attempt, payload in _successful_payloads(action):
        if payload["finish_status"] != FinishStatus.COMPLETE.value:
            errors.append(incomplete_error)
            continue
        try:
            value = strict_json_loads(payload["text"])
            validate_json_schema_instance(value, schema)
        except JsonSchemaValidationError as exc:
            errors.extend(exc.errors)
            continue
        except DomainError as exc:
            errors.append(str(exc))
            continue
        accepted.append((attempt, payload, value))
    result = tuple(accepted), tuple(errors)
    # LogicalActionView is immutable and replaced by each reducer checkpoint.
    # Cache only that exact action/schema pair, never across new evidence.
    context.validated_payloads[action.action_id] = (action, schema, incomplete_error, *result)
    return result


def _target_result(action: LogicalActionView, state: RunState) -> SemanticResult:
    result = state.target_results.get(action.action_id)
    if result is not None:
        return SemanticResult(result)
    errors: list[str] = []
    for _, payload in _successful_payloads(action):
        if payload["finish_status"] != FinishStatus.COMPLETE.value:
            errors.append("target response is truncated or incomplete")
        elif not payload["text"]:
            errors.append("target response is empty")
    return SemanticResult(None, tuple(errors) or ("no authoritative usable target response",))


def _normal_slot_for_action(panel: Mapping[str, Any], action_id: str) -> str | None:
    for member in panel["normal_members"]:
        if member["judge_action_id"] == action_id:
            return str(member["logical_member_slot"])
    return None


def _primary_member(
    context: RunContext, state: RunState, unit: Any, panel: Mapping[str, Any]
) -> MemberAssessment | None:
    primary_id = panel["normal_members"][0]["judge_action_id"]
    action = state.logical_actions[primary_id]
    result = _judge_result(context, state, unit, action, forced_slot="primary")
    return result.value if isinstance(result.value, MemberAssessment) else None


def _panel_is_required(
    context: RunContext, state: RunState, unit: Any, panel: Mapping[str, Any]
) -> bool:
    # A skipped or exhausted target has no observation to judge. "Always full"
    # controls panel membership only after a usable target response exists.
    if panel["target_action_id"] not in state.target_results:
        return False
    if panel["trigger_policy"] == "always_full":
        return True
    primary = _primary_member(context, state, unit, panel)
    if primary is not None:
        return should_schedule_secondary_panel(primary)
    primary_action = state.logical_actions[panel["normal_members"][0]["judge_action_id"]]
    return _action_closed(context, state, unit, primary_action)


def _replacement_slot(
    context: RunContext, state: RunState, unit: Any, panel: Mapping[str, Any]
) -> str | None:
    if not _panel_is_required(context, state, unit, panel):
        return None
    for member in panel["normal_members"]:
        action = state.logical_actions[member["judge_action_id"]]
        parsed = _judge_result(
            context,
            state,
            unit,
            action,
            forced_slot=str(member["logical_member_slot"]),
        )
        if not parsed.usable and _action_closed(context, state, unit, action):
            return str(member["logical_member_slot"])
    return None


def _judge_result(
    context: RunContext,
    state: RunState,
    unit: Any,
    action: LogicalActionView,
    *,
    forced_slot: str | None = None,
) -> SemanticResult:
    panel = _panel_for_action(unit, action.action_id)
    if panel is None:
        return SemanticResult(None, ("judge result is not an observation member",))
    target = state.target_results.get(panel["target_action_id"])
    if target is None:
        return SemanticResult(None, ("target observation is unavailable",))
    slot = forced_slot or _normal_slot_for_action(panel, action.action_id)
    if slot is None and panel["replacement"] is not None and panel["replacement"]["judge_action_id"] == action.action_id:
        slot = _replacement_slot(context, state, unit, panel)
    if slot is None:
        return SemanticResult(None, ("replacement has no unresolved scheduled member",))
    validated, structural_errors = _schema_validated_payloads(
        context,
        action,
        incomplete_error="judge response is truncated or incomplete",
    )
    errors = list(structural_errors)
    for attempt, payload, _ in validated:
        parsed = validate_judge_output(
            payload["text"],
            target.text,
            schema=context.resources.schemas["judge_output.json"],
        )
        if parsed.output is None:
            errors.extend(parsed.errors)
            continue
        observation_id = _target_observation_id(
            context, state, panel["target_action_id"]
        )
        return SemanticResult(
            member_assessment_from_output(
                parsed.output,
                assessment_id=entity_id("ma", [context.run_id, attempt.physical_attempt_id]),
                observation_id=observation_id,
                logical_member_slot=slot,
                effective_model_id=payload["effective_model_id"],
                role_route_key=attempt.intended["role_route_key"],
                source_attempt_id=attempt.physical_attempt_id,
            )
        )
    return SemanticResult(None, tuple(errors) or ("no authoritative judge response",))


def _panel_assessment(
    context: RunContext,
    state: RunState,
    unit: Any,
    panel: Mapping[str, Any],
) -> tuple[MemberAssessment | None, PanelAssessment | None, bool]:
    primary = _primary_member(context, state, unit, panel)
    required = _panel_is_required(context, state, unit, panel)
    if not required:
        return primary, None, False
    members: list[MemberAssessment] = []
    for member_plan in panel["normal_members"]:
        result = _judge_result(
            context,
            state,
            unit,
            state.logical_actions[member_plan["judge_action_id"]],
            forced_slot=str(member_plan["logical_member_slot"]),
        )
        if isinstance(result.value, MemberAssessment):
            members.append(result.value)
    if panel["replacement"] is not None:
        replacement_action = state.logical_actions[panel["replacement"]["judge_action_id"]]
        replacement = _judge_result(context, state, unit, replacement_action)
        if isinstance(replacement.value, MemberAssessment):
            # Do not retain both a failed slot and its replacement.  A normal
            # structurally usable member always wins its own logical slot.
            members = [
                member
                for member in members
                if member.logical_member_slot != replacement.value.logical_member_slot
            ]
            members.append(replacement.value)
    observation_id = _target_observation_id(
        context, state, panel["target_action_id"]
    )
    target = state.target_results.get(panel["target_action_id"])
    assessment = aggregate_panel(
        panel["panel_id"],
        observation_id,
        members,
        target_effective_model_id=None if target is None else target.effective_model_id,
    )
    return primary, assessment, True


def _target_observation_key(state: RunState, logical_action_id: str) -> str:
    """Return the nonce-free observation key for one durable target action."""

    action = state.logical_actions.get(logical_action_id)
    if action is None or action.slot["role"] != Role.TARGET.value:
        raise EngineError("target result does not resolve to its logical action")
    return scientific_key([action.slot["action_scientific_key"], "observation"])


def _target_observation_id(
    context: RunContext, state: RunState, logical_action_id: str
) -> str:
    """Return the one run-local ID for the logical target observation."""

    del context
    if logical_action_id not in state.logical_actions:
        raise EngineError("target observation names an unknown logical action")
    return entity_id("to", [logical_action_id])


def _conversation_lineage_identity(
    context: RunContext,
    state: RunState,
    unit: Any,
    logical_action_id: str,
    candidate: CandidatePlan,
) -> Mapping[str, str]:
    """Derive one target turn's exact root/node/observation identity tree."""

    action = state.logical_actions.get(logical_action_id)
    if action is None or action.slot["role"] != Role.TARGET.value:
        raise EngineError("conversation node names a non-target logical action")
    execution = next(
        (
            item
            for item in unit.unit["execution_slots"]
            if item["execution_scientific_key"]
            == action.slot["owner_position_scientific_key"]
        ),
        None,
    )
    if execution is None:
        raise EngineError("target logical action has no frozen sample execution")

    if candidate.parent_node_id is None:
        root_key = scientific_key(
            [execution["execution_scientific_key"], "conversation_root"]
        )
        root_id = entity_id(
            "cr", [execution["execution_id"], "conversation_root"]
        )
        node_parent_id = root_id
        node_parent_key = root_key
    else:
        root_id = unit.unit["design"].get("conversation_root_id")
        if root_id is None:
            # Stateful verification/qualification remains inside the same
            # preallocated fresh execution rather than borrowing a recorded
            # search root.
            root_id = entity_id(
                "cr", [execution["execution_id"], "conversation_root"]
            )
            root_key = scientific_key(
                [execution["execution_scientific_key"], "conversation_root"]
            )
        else:
            root_key = unit.unit.get("subject_scientific_key")
        if not isinstance(root_id, str) or not isinstance(root_key, str):
            raise EngineError("stateful target lacks its frozen conversation root")
        if candidate.parent_node_scientific_key is None:
            raise EngineError("stateful candidate lacks its parent scientific key")
        node_parent_id = candidate.parent_node_id
        node_parent_key = candidate.parent_node_scientific_key

    observation_key = _target_observation_key(state, logical_action_id)
    observation_id = _target_observation_id(
        context, state, logical_action_id
    )
    exact_turn = normalize_line_endings(candidate.prompt)
    node_key = scientific_key(
        [
            node_parent_key,
            "conversation_node",
            sha256_bytes(exact_turn.encode("utf-8")),
            observation_key,
        ]
    )
    node_id = entity_id(
        "cn", [node_parent_id, exact_turn, observation_id]
    )
    return {
        "conversation_root_id": root_id,
        "conversation_root_scientific_key": root_key,
        "node_id": node_id,
        "node_scientific_key": node_key,
        "observation_id": observation_id,
        "observation_scientific_key": observation_key,
    }


def _map_proposal_result(
    context: RunContext, state: RunState, unit: Any, action: LogicalActionView
) -> SemanticResult:
    validated, structural_errors = _schema_validated_payloads(
        context,
        action,
        incomplete_error="proposal response is truncated or incomplete",
    )
    errors = list(structural_errors)
    for _, payload, value in validated:
        try:
            parsed = parse_proposal_batch(
                value,
                _proposal_slots(unit),
            )
            return SemanticResult(parsed)
        except DomainError as exc:
            errors.append(str(exc))
    return SemanticResult(None, tuple(errors) or ("no authoritative proposal batch",))


def _map_review_result(
    context: RunContext, state: RunState, unit: Any, action: LogicalActionView
) -> SemanticResult:
    proposal_action = next(
        item
        for item in unit.actions
        if item.slot["purpose"] == ActionPurpose.PERTURBATOR_PROPOSAL.value
    )
    batch_result = _map_proposal_result(context, state, unit, proposal_action)
    if not isinstance(batch_result.value, ProposalBatch):
        return SemanticResult(None, ("no usable proposal batch to review",))
    expectations = tuple(
        ReviewExpectation(
            make_review_item_id(unit.unit["unit_scientific_key"], "probe", slot.scientific_key),
            proposal,
        )
        for slot, proposal in zip(
            _proposal_slots(unit), batch_result.value.items, strict=True
        )
        if proposal.proposed
    )
    if not expectations:
        return SemanticResult(())
    validated, structural_errors = _schema_validated_payloads(
        context,
        action,
        incomplete_error="probe review is truncated or incomplete",
    )
    errors = list(structural_errors)
    for _, _, value in validated:
        try:
            return SemanticResult(
                parse_probe_reviews(
                    value,
                    expectations,
                    context.objective.descriptor_values,
                    mode="ordinary",
                )
            )
        except DomainError as exc:
            errors.append(str(exc))
    return SemanticResult(None, tuple(errors) or ("no authoritative probe review",))


_STRATEGY_QUALIFICATION_KINDS = frozenset(
    {"strategy_development", "strategy_holdout"}
)


def _qualification_source_episode(state: RunState, unit: Any) -> StrategyEpisodeView:
    """Resolve the one frozen in-run template provenance without output parsing."""

    source_unit_id = unit.unit.get("parent_unit_id")
    if unit.unit["kind"] == "strategy_holdout":
        source_development_id = unit.unit["design"]["source_development_unit_id"]
        if source_unit_id != source_development_id:
            raise EngineError("holdout parent differs from its source development unit")
        source_development = next(
            (
                item
                for item in state.work_units
                if item.unit_id == source_development_id
            ),
            None,
        )
        if source_development is None:
            raise EngineError("holdout source development unit is missing")
        source_unit_id = source_development.unit.get("parent_unit_id")
    if not isinstance(source_unit_id, str):
        raise EngineError("strategy qualification lacks its source portfolio unit")

    matches: list[StrategyEpisodeView] = []
    for episode in state.strategy_episodes.values():
        template = episode.rounds[0].lane.item.template_candidate
        if template is None:
            continue
        template_key = scientific_key(
            [
                state.scientific_scope_digest,
                "artifact",
                "strategy_template",
                template.content_hash,
            ]
        )
        if (
            template_key == unit.unit["subject_scientific_key"]
            and any(item.work_unit_id == source_unit_id for item in episode.rounds)
        ):
            matches.append(episode)
    if not matches:
        raise EngineError("strategy qualification cannot recover its frozen source episode")
    return min(matches, key=lambda item: item.strategy_scientific_key)


def _qualification_template(
    context: RunContext, state: RunState, unit: Any
) -> StrategyTemplate:
    episode = _qualification_source_episode(state, unit)
    template = episode.rounds[0].lane.item.template_candidate
    if not isinstance(template, StrategyTemplate):
        raise EngineError("strategy qualification source has no valid frozen template")
    template_key, template_id = _strategy_template_identity(context, template)
    if (
        unit.unit["subject_id"] != template_id
        or unit.unit["subject_scientific_key"] != template_key
        or unit.unit["design"]["strategy_template_id"] != template_id
        or unit.unit["design"]["strategy_template_scientific_key"] != template_key
    ):
        raise EngineError("strategy qualification template identity changed")
    return template


def _qualification_entry_candidate(
    context: RunContext, state: RunState, unit: Any
) -> CandidatePlan:
    episode = _qualification_source_episode(state, unit)
    return _entry_candidate_for_episode(context, state, episode)


def _qualification_root_context(
    unit: Any, action: LogicalActionView
) -> tuple[Mapping[str, Any], Mapping[str, Any], int, int, ProposalSlot | None, str | None]:
    """Return root plan, execution, ordinals, optional slot, and frozen intent."""

    root_plan = next(
        (
            item
            for item in unit.unit["design"]["root_plans"]
            if action.action_id in item["ordered_action_ids"]
        ),
        None,
    )
    if root_plan is None:
        raise EngineError("qualification action is absent from every frozen root plan")
    execution = next(
        (
            item
            for item in unit.unit["execution_slots"]
            if item["execution_id"] == root_plan["execution_id"]
        ),
        None,
    )
    if execution is None or execution["root_plan_id"] != root_plan["root_plan_id"]:
        raise EngineError("qualification root plan lacks its execution slot")
    root_ordinal = execution["ordinal"]
    arm = root_plan["arm"]
    if unit.unit["kind"] == "strategy_development":
        count = unit.unit["design"]["planned_count_per_arm"]
        offsets = {
            "treatment": 0,
            "parent_baseline": count,
            "strategy_agnostic": 2 * count,
        }
        if arm not in offsets:
            raise EngineError("development root has an unknown arm")
        arm_local_ordinal = root_ordinal - offsets[arm]
        if not 0 <= arm_local_ordinal < count:
            raise EngineError("development root ordinal disagrees with its arm")
        intent = (
            None
            if arm == "parent_baseline"
            else unit.unit["design"]["realization_intent_assignment"][
                arm_local_ordinal
            ]
        )
        proposal_ordinal = (
            None
            if arm == "parent_baseline"
            else (
                arm_local_ordinal
                if arm == "treatment"
                else count + arm_local_ordinal
            )
        )
    else:
        sealed = unit.unit["design"]["sealed_holdout_plan"]
        future = next(
            (
                item
                for item in sealed["future_roots"]
                if item["root_ordinal"] == root_ordinal
            ),
            None,
        )
        if future is None or future["arm"] != arm:
            raise EngineError("holdout execution differs from its sealed future root")
        arm_local_ordinal = future["arm_local_ordinal"]
        intent = future["realization_intent"]
        proposal_ordinal = root_ordinal
    proposal_slot = (
        None
        if proposal_ordinal is None
        else next(
            (
                item
                for item in _proposal_slots(unit)
                if item.global_ordinal == proposal_ordinal
            ),
            None,
        )
    )
    if proposal_ordinal is not None and proposal_slot is None:
        raise EngineError("qualification root lacks its preallocated proposal slot")
    return (
        root_plan,
        execution,
        root_ordinal,
        arm_local_ordinal,
        proposal_slot,
        intent,
    )


def _qualification_root_actions(
    state: RunState, root_plan: Mapping[str, Any]
) -> tuple[LogicalActionView, ...]:
    try:
        return tuple(
            state.logical_actions[action_id]
            for action_id in root_plan["ordered_action_ids"]
        )
    except KeyError as exc:
        raise EngineError("qualification root cites an unknown logical action") from exc


def _qualification_target_actions(
    state: RunState, root_plan: Mapping[str, Any]
) -> tuple[LogicalActionView, ...]:
    return tuple(
        item
        for item in _qualification_root_actions(state, root_plan)
        if item.slot["role"] == Role.TARGET.value
    )


def _qualification_fresh_branch(
    context: RunContext,
    state: RunState,
    unit: Any,
    action: LogicalActionView,
) -> tuple[list[dict[str, str]], str, str, LogicalActionView]:
    root_plan, _, _, _, _, _ = _qualification_root_context(unit, action)
    targets = _qualification_target_actions(state, root_plan)
    if not targets:
        raise EngineError("qualification root has no entry target")
    entry = targets[0]
    target = state.target_results.get(entry.action_id)
    if target is None or not target.usable:
        raise EngineError("qualification fresh branch lacks its entry target response")
    if not entry.attempts:
        raise EngineError("qualification entry target lacks exact request evidence")
    messages = [dict(item) for item in entry.attempts[0].intended["messages"]]
    messages.append({"role": "assistant", "content": target.text})
    return messages, _target_observation_key(state, entry.action_id), target.text, entry


def _qualification_binding_action(
    state: RunState, root_plan: Mapping[str, Any]
) -> LogicalActionView:
    bindings = tuple(
        item
        for item in _qualification_root_actions(state, root_plan)
        if item.slot["purpose"] == ActionPurpose.TEMPLATE_BINDING.value
    )
    if len(bindings) != 1:
        raise EngineError("treatment qualification root lacks one template binding")
    return bindings[0]


def _qualification_binding_result(
    context: RunContext, state: RunState, unit: Any, action: LogicalActionView
) -> SemanticResult:
    root_plan, execution, _, _, _, _ = _qualification_root_context(unit, action)
    if root_plan["arm"] != "treatment":
        return SemanticResult(None, ("control root cannot bind a StrategyTemplate",))
    template = _qualification_template(context, state, unit)
    _, turn_ref, turn_text, _ = _qualification_fresh_branch(
        context, state, unit, action
    )
    binding_id = make_template_binding_item_id(
        unit.unit["unit_scientific_key"],
        execution["execution_scientific_key"],
        unit.unit["design"]["strategy_template_scientific_key"],
        0,
    )
    validated, structural_errors = _schema_validated_payloads(
        context,
        action,
        incomplete_error="template binding response is truncated or incomplete",
    )
    errors = list(structural_errors)
    for _, _, value in validated:
        try:
            return SemanticResult(
                parse_template_binding(
                    value,
                    expected_binding_item_id=binding_id,
                    template=template,
                    descriptor_vocabulary=context.objective.descriptor_values,
                    fresh_turn_text_by_ref={turn_ref: turn_text},
                )
            )
        except DomainError as exc:
            errors.append(str(exc))
    return SemanticResult(
        None, tuple(errors) or ("no authoritative template binding",)
    )


def _qualification_proposal_result(
    context: RunContext, state: RunState, unit: Any, action: LogicalActionView
) -> SemanticResult:
    root_plan, _, _, _, proposal_slot, intent = _qualification_root_context(
        unit, action
    )
    if proposal_slot is None or intent is None:
        return SemanticResult(None, ("qualification control has no proposal slot",))
    root_actions = _qualification_root_actions(state, root_plan)
    proposal_actions = tuple(
        item
        for item in root_actions
        if item.slot["purpose"] == ActionPurpose.PERTURBATOR_PROPOSAL.value
    )
    if action not in proposal_actions or len(proposal_actions) != 2:
        raise EngineError("qualification continuation lacks its bounded generation pair")
    is_corrective = action.action_id == proposal_actions[1].action_id
    if is_corrective:
        review_action = next(
            item
            for item in root_actions
            if item.slot["purpose"] == ActionPurpose.PROBE_REVIEW.value
        )
        reviews = _qualification_review_result(
            context, state, unit, review_action
        ).value
        if not isinstance(reviews, tuple):
            return SemanticResult(
                None, ("qualification correction lacks its initial review",)
            )
    validated, structural_errors = _schema_validated_payloads(
        context,
        action,
        incomplete_error="qualification proposal response is truncated or incomplete",
    )
    errors = list(structural_errors)
    for _, _, value in validated:
        try:
            parsed = parse_proposal_batch(
                value,
                (proposal_slot,),
            )
            return SemanticResult(parsed)
        except DomainError as exc:
            errors.append(str(exc))
        except EngineError as exc:
            errors.append(str(exc))
    return SemanticResult(
        None, tuple(errors) or ("no authoritative qualification proposal batch",)
    )


def _qualification_review_result(
    context: RunContext, state: RunState, unit: Any, action: LogicalActionView
) -> SemanticResult:
    root_plan, _, _, _, proposal_slot, _ = _qualification_root_context(unit, action)
    if proposal_slot is None:
        return SemanticResult(None, ("qualification review has no proposal slot",))
    root_actions = _qualification_root_actions(state, root_plan)
    review_actions = tuple(
        item
        for item in root_actions
        if item.slot["purpose"] == ActionPurpose.PROBE_REVIEW.value
    )
    proposal_actions = tuple(
        item
        for item in root_actions
        if item.slot["purpose"] == ActionPurpose.PERTURBATOR_PROPOSAL.value
    )
    if action not in review_actions or len(review_actions) != 2 or len(proposal_actions) != 2:
        raise EngineError("qualification continuation review graph is incomplete")
    index = 0 if action.action_id == review_actions[0].action_id else 1
    batch = _qualification_proposal_result(
        context, state, unit, proposal_actions[index]
    ).value
    if not isinstance(batch, ProposalBatch):
        return SemanticResult(None, ("qualification review lacks a usable proposal batch",))
    proposal = next((item for item in batch.items if item.proposed), None)
    if proposal is None:
        return SemanticResult(())
    expectation = ReviewExpectation(
        make_review_item_id(
            unit.unit["unit_scientific_key"],
            "probe",
            proposal_slot.scientific_key,
        ),
        proposal,
    )
    arm = root_plan["arm"]
    mode = "strategy" if arm == "treatment" else "ordinary"
    contracts: Mapping[str, Any] | None = None
    if arm == "treatment":
        binding = _qualification_binding_result(
            context,
            state,
            unit,
            _qualification_binding_action(state, root_plan),
        ).value
        if not isinstance(binding, TemplateBindingResult) or binding.contract is None:
            return SemanticResult(None, ("strategy review lacks a fresh bound contract",))
        contract = binding.contract
        contracts = {proposal_slot.slot_id: contract.move.descriptors}
    validated, structural_errors = _schema_validated_payloads(
        context,
        action,
        incomplete_error="qualification probe review is truncated or incomplete",
    )
    errors = list(structural_errors)
    for _, _, value in validated:
        try:
            reviews = parse_probe_reviews(
                value,
                (expectation,),
                context.objective.descriptor_values,
                mode=mode,
                contract_descriptors_by_slot=contracts,
            )
            return SemanticResult(reviews)
        except DomainError as exc:
            errors.append(str(exc))
        except EngineError as exc:
            errors.append(str(exc))
    return SemanticResult(
        None, tuple(errors) or ("no authoritative qualification probe review",)
    )


def _qualification_selected_realization(
    context: RunContext,
    state: RunState,
    unit: Any,
    target_action: LogicalActionView,
) -> SelectedRealization | None:
    root_plan, _, _, _, proposal_slot, _ = _qualification_root_context(
        unit, target_action
    )
    if proposal_slot is None:
        return None
    root_actions = _qualification_root_actions(state, root_plan)
    proposals = tuple(
        item
        for item in root_actions
        if item.slot["purpose"] == ActionPurpose.PERTURBATOR_PROPOSAL.value
    )
    reviews = tuple(
        item
        for item in root_actions
        if item.slot["purpose"] == ActionPurpose.PROBE_REVIEW.value
    )
    if len(proposals) != 2 or len(reviews) != 2:
        raise EngineError("qualification realization graph is incomplete")
    original_batch = _qualification_proposal_result(
        context, state, unit, proposals[0]
    ).value
    initial_reviews = _qualification_review_result(
        context, state, unit, reviews[0]
    ).value
    if not isinstance(original_batch, ProposalBatch) or not isinstance(
        initial_reviews, tuple
    ):
        return None
    original = next(
        (
            item
            for item in original_batch.proposed_items
            if item.slot_id == proposal_slot.slot_id
        ),
        None,
    )
    original_review = next(
        (
            item
            for item in initial_reviews
            if item.review_item_id
            == make_review_item_id(
                unit.unit["unit_scientific_key"],
                "probe",
                proposal_slot.scientific_key,
            )
        ),
        None,
    )
    if original is None or original_review is None:
        return None
    corrective = None
    corrective_review = None
    corrective_batch = _qualification_proposal_result(
        context, state, unit, proposals[1]
    ).value
    if isinstance(corrective_batch, ProposalBatch):
        corrective = next(
            (
                item
                for item in corrective_batch.proposed_items
                if item.slot_id == proposal_slot.slot_id
            ),
            None,
        )
    if corrective is not None:
        rereviews = _qualification_review_result(
            context, state, unit, reviews[1]
        ).value
        if isinstance(rereviews, tuple):
            corrective_review = next(
                (
                    item
                    for item in rereviews
                    if item.review_item_id == original_review.review_item_id
                ),
                None,
            )
    if (corrective is None) != (corrective_review is None):
        return None
    if root_plan["arm"] == "treatment":
        return select_strategy_realization(
            original, original_review, corrective, corrective_review
        )
    return select_strategy_agnostic_realization(
        original, original_review, corrective, corrective_review
    )


def _qualification_outcome_request(
    context: RunContext, state: RunState, unit: Any, action: LogicalActionView
) -> tuple[dict[str, Any], dict[str, tuple[str, str]]]:
    root_plan, _, _, _, proposal_slot, _ = _qualification_root_context(unit, action)
    if root_plan["arm"] != "treatment" or proposal_slot is None:
        raise EngineError("only a treatment root may request an outcome review")
    targets = _qualification_target_actions(state, root_plan)
    if len(targets) != 2:
        raise EngineError("treatment root lacks its entry and terminal targets")
    terminal = targets[1]
    guard = action.slot["guard"]
    if (
        guard.get("kind") != "if_any_aligned_response"
        or guard.get("target_action_ids") != [terminal.action_id]
    ):
        raise EngineError("qualification outcome guard changed its terminal target")
    selected = _qualification_selected_realization(
        context, state, unit, terminal
    )
    target = state.target_results.get(terminal.action_id)
    observation = state.target_observations.get(terminal.action_id)
    panel_plan = _panel_for_action(unit, terminal.action_id)
    panel = None if panel_plan is None else state.panel_views.get(panel_plan["panel_id"])
    if (
        selected is None
        or selected.review.status != "aligned"
        or target is None
        or not target.usable
        or observation is None
        or panel is None
        or not panel.resolved
    ):
        return {"items": []}, {}
    binding = _qualification_binding_result(
        context,
        state,
        unit,
        _qualification_binding_action(state, root_plan),
    ).value
    if not isinstance(binding, TemplateBindingResult) or binding.contract is None:
        raise EngineError("treatment outcome review lacks its bound contract")
    contract = binding.contract
    _, _, before_text, _ = _qualification_fresh_branch(
        context, state, unit, action
    )
    review_id = make_review_item_id(
        unit.unit["unit_scientific_key"], "outcome", proposal_slot.scientific_key
    )
    expected = {
        review_id: (observation.scientific_key, observation.response)
    }
    item = {
        "review_alias": outcome_review_aliases(expected)[review_id],
        "transition_criterion": contract.success.to_dict(),
        "before_target": {
            "role": "assistant",
            "content": before_text,
        },
        "intervening_user": {
            "role": "user",
            "content": selected.proposal.user_turn,
        },
        "after_target": {
            "role": "assistant",
            "content": observation.response,
        },
    }
    return {"items": [item]}, expected


def _qualification_outcome_result(
    context: RunContext, state: RunState, unit: Any, action: LogicalActionView
) -> SemanticResult:
    _, expected = _qualification_outcome_request(context, state, unit, action)
    if not expected:
        return SemanticResult(())
    validated, structural_errors = _schema_validated_payloads(
        context,
        action,
        incomplete_error="qualification outcome review is truncated or incomplete",
    )
    errors = list(structural_errors)
    for _, _, value in validated:
        try:
            return SemanticResult(parse_outcome_reviews(value, expected))
        except DomainError as exc:
            errors.append(str(exc))
    return SemanticResult(
        None, tuple(errors) or ("no authoritative qualification outcome review",)
    )


def _is_scenario_verification(unit: Any) -> bool:
    return bool(
        unit.unit.get("kind") == "verification"
        and unit.unit.get("design", {}).get("subject_kind") == "scenario_recipe"
    )


def _scenario_for_unit(context: RunContext, unit: Any) -> ScenarioPlan:
    scenario = context.scenarios.get(unit.unit["subject_scientific_key"])
    if scenario is None:
        raise EngineError("scenario verification subject is not hydrated")
    if (
        scenario.scenario_id != unit.unit["subject_id"]
        or descriptors_dict(scenario.recipe.measurement_cell)
        != unit.unit["design"]["measurement_cell"]
    ):
        raise EngineError("scenario verification subject changed after freeze")
    return scenario


def _scenario_root_context(
    unit: Any, action: LogicalActionView
) -> tuple[Mapping[str, Any], Mapping[str, Any], ProposalSlot]:
    roots = [
        item
        for item in unit.unit["design"]["root_plans"]
        if action.action_id in item["ordered_action_ids"]
    ]
    if len(roots) != 1:
        raise EngineError("scenario action does not belong to one frozen root")
    root = roots[0]
    execution = next(
        (
            item
            for item in unit.unit["execution_slots"]
            if item["execution_id"] == root["execution_id"]
        ),
        None,
    )
    if execution is None or execution["root_plan_id"] != root["root_plan_id"]:
        raise EngineError("scenario root lacks its frozen execution")
    root_actions = tuple(
        item for item in unit.actions if item.action_id in root["ordered_action_ids"]
    )
    bindings = tuple(
        item
        for item in root_actions
        if item.slot["purpose"] == ActionPurpose.TEMPLATE_BINDING.value
    )
    if not bindings:
        raise EngineError("scenario root lacks continuation bindings")
    step_index = max(
        (
            index
            for index, binding in enumerate(bindings)
            if binding.slot["ordinal"] <= action.slot["ordinal"]
        ),
        default=0,
    )
    proposal_ordinal = execution["ordinal"] * len(bindings) + step_index
    proposal = next(
        (
            item
            for item in _proposal_slots(unit)
            if item.global_ordinal == proposal_ordinal
        ),
        None,
    )
    if proposal is None:
        raise EngineError("scenario root lacks its designated proposal slot")
    return root, execution, proposal


def _scenario_step_index(
    unit: Any, root: Mapping[str, Any], action: LogicalActionView
) -> int:
    bindings = tuple(
        item
        for item in unit.actions
        if item.action_id in root["ordered_action_ids"]
        and item.slot["purpose"] == ActionPurpose.TEMPLATE_BINDING.value
    )
    if not bindings:
        raise EngineError("scenario root lacks continuation bindings")
    return max(
        (
            index
            for index, binding in enumerate(bindings)
            if binding.slot["ordinal"] <= action.slot["ordinal"]
        ),
        default=0,
    )


def _scenario_root_actions(
    state: RunState, root: Mapping[str, Any]
) -> tuple[LogicalActionView, ...]:
    try:
        return tuple(
            state.logical_actions[action_id]
            for action_id in root["ordered_action_ids"]
        )
    except KeyError as exc:
        raise EngineError("scenario root names an unknown action") from exc


def _scenario_target_actions(
    state: RunState, root: Mapping[str, Any]
) -> tuple[LogicalActionView, ...]:
    return tuple(
        item
        for item in _scenario_root_actions(state, root)
        if item.slot["role"] == Role.TARGET.value
    )


def _scenario_fresh_branch(
    state: RunState, unit: Any, action: LogicalActionView
) -> tuple[list[dict[str, str]], str, str, LogicalActionView]:
    root, _, _ = _scenario_root_context(unit, action)
    targets = _scenario_target_actions(state, root)
    step_index = _scenario_step_index(unit, root, action)
    if len(targets) <= step_index:
        raise EngineError("scenario continuation has no freshly generated parent target")
    parent = targets[step_index]
    result = state.target_results.get(parent.action_id)
    if result is None or not result.usable or not parent.attempts:
        raise EngineError("scenario continuation lacks its fresh parent response")
    messages = [dict(item) for item in parent.attempts[0].intended["messages"]]
    messages.append({"role": "assistant", "content": result.text})
    return messages, _target_observation_key(state, parent.action_id), result.text, parent


def _scenario_binding_action(
    state: RunState,
    unit: Any,
    root: Mapping[str, Any],
    action: LogicalActionView,
) -> LogicalActionView:
    matches = tuple(
        item
        for item in _scenario_root_actions(state, root)
        if item.slot["purpose"] == ActionPurpose.TEMPLATE_BINDING.value
    )
    step_index = _scenario_step_index(unit, root, action)
    if step_index >= len(matches):
        raise EngineError("scenario step lacks its template binding")
    return matches[step_index]


def _scenario_binding_result(
    context: RunContext, state: RunState, unit: Any, action: LogicalActionView
) -> SemanticResult:
    scenario = _scenario_for_unit(context, unit)
    source = scenario_preflight_source(unit, action.action_id, state.conversation_nodes, state.target_observations)
    if source is None:
        root, execution, _ = _scenario_root_context(unit, action)
        step_index = _scenario_step_index(unit, root, action)
        owner_key = execution["execution_scientific_key"]
        _, turn_ref, turn_text, _ = _scenario_fresh_branch(state, unit, action)
    else:
        step_index, observation = source
        owner_key = unit.unit["unit_scientific_key"]
        turn_ref, turn_text = observation.scientific_key, observation.response
    template = scenario.recipe.templates[step_index]
    template_key, _ = _strategy_template_identity(context, template)
    binding_id = make_template_binding_item_id(
        unit.unit["unit_scientific_key"],
        owner_key,
        template_key,
        step_index,
    )
    validated, structural_errors = _schema_validated_payloads(
        context,
        action,
        incomplete_error="scenario template binding is truncated or incomplete",
    )
    errors = list(structural_errors)
    for _, _, value in validated:
        try:
            return SemanticResult(
                parse_template_binding(
                    value,
                    expected_binding_item_id=binding_id,
                    template=template,
                    descriptor_vocabulary=context.objective.descriptor_values,
                    fresh_turn_text_by_ref={turn_ref: turn_text},
                )
            )
        except DomainError as exc:
            errors.append(str(exc))
    return SemanticResult(
        None, tuple(errors) or ("no authoritative scenario template binding",)
    )


def _scenario_proposal_result(
    context: RunContext, state: RunState, unit: Any, action: LogicalActionView
) -> SemanticResult:
    scenario = _scenario_for_unit(context, unit)
    root, _, proposal_slot = _scenario_root_context(unit, action)
    step_index = _scenario_step_index(unit, root, action)
    actions = _scenario_root_actions(state, root)
    all_proposals = tuple(
        item
        for item in actions
        if item.slot["purpose"] == ActionPurpose.PERTURBATOR_PROPOSAL.value
    )
    proposals = all_proposals[2 * step_index : 2 * step_index + 2]
    if len(proposals) != 2 or action not in proposals:
        raise EngineError("scenario root lacks its bounded proposal pair")
    if action.action_id == proposals[1].action_id:
        all_reviews = tuple(
            item
            for item in actions
            if item.slot["purpose"] == ActionPurpose.PROBE_REVIEW.value
        )
        review = all_reviews[2 * step_index]
        if not isinstance(
            _scenario_review_result(context, state, unit, review).value, tuple
        ):
            return SemanticResult(None, ("scenario correction lacks initial review",))
    validated, structural_errors = _schema_validated_payloads(
        context,
        action,
        incomplete_error="scenario proposal is truncated or incomplete",
    )
    errors = list(structural_errors)
    for _, _, value in validated:
        try:
            return SemanticResult(
                parse_proposal_batch(
                    value,
                    (proposal_slot,),
                )
            )
        except DomainError as exc:
            errors.append(str(exc))
    return SemanticResult(None, tuple(errors) or ("no authoritative scenario proposal",))


def _scenario_review_result(
    context: RunContext, state: RunState, unit: Any, action: LogicalActionView
) -> SemanticResult:
    root, _, proposal_slot = _scenario_root_context(unit, action)
    step_index = _scenario_step_index(unit, root, action)
    actions = _scenario_root_actions(state, root)
    all_reviews = tuple(
        item
        for item in actions
        if item.slot["purpose"] == ActionPurpose.PROBE_REVIEW.value
    )
    all_proposals = tuple(
        item
        for item in actions
        if item.slot["purpose"] == ActionPurpose.PERTURBATOR_PROPOSAL.value
    )
    reviews = all_reviews[2 * step_index : 2 * step_index + 2]
    proposals = all_proposals[2 * step_index : 2 * step_index + 2]
    if len(reviews) != 2 or len(proposals) != 2 or action not in reviews:
        raise EngineError("scenario review graph is incomplete")
    index = 0 if action.action_id == reviews[0].action_id else 1
    batch = _scenario_proposal_result(
        context, state, unit, proposals[index]
    ).value
    if not isinstance(batch, ProposalBatch):
        return SemanticResult(None, ("scenario review lacks a usable proposal",))
    proposal = next(iter(batch.proposed_items), None)
    if proposal is None:
        return SemanticResult(())
    binding = _scenario_binding_result(
        context, state, unit, _scenario_binding_action(state, unit, root, action)
    ).value
    if not isinstance(binding, TemplateBindingResult) or binding.contract is None:
        return SemanticResult(None, ("scenario review lacks an applicable binding",))
    contract = binding.contract
    expectation = ReviewExpectation(
        make_review_item_id(
            unit.unit["unit_scientific_key"], "probe", proposal_slot.scientific_key
        ),
        proposal,
    )
    validated, structural_errors = _schema_validated_payloads(
        context,
        action,
        incomplete_error="scenario probe review is truncated or incomplete",
    )
    errors = list(structural_errors)
    for _, _, value in validated:
        try:
            return SemanticResult(
                parse_probe_reviews(
                    value,
                    (expectation,),
                    context.objective.descriptor_values,
                    mode="strategy",
                    contract_descriptors_by_slot={
                        proposal_slot.slot_id: contract.move.descriptors
                    },
                )
            )
        except DomainError as exc:
            errors.append(str(exc))
    return SemanticResult(
        None, tuple(errors) or ("no authoritative scenario probe review",)
    )


def _scenario_selected_realization(
    context: RunContext,
    state: RunState,
    unit: Any,
    target_action: LogicalActionView,
) -> SelectedRealization | None:
    root, _, proposal_slot = _scenario_root_context(unit, target_action)
    step_index = _scenario_step_index(unit, root, target_action)
    actions = _scenario_root_actions(state, root)
    all_proposals = tuple(
        item
        for item in actions
        if item.slot["purpose"] == ActionPurpose.PERTURBATOR_PROPOSAL.value
    )
    all_reviews = tuple(
        item
        for item in actions
        if item.slot["purpose"] == ActionPurpose.PROBE_REVIEW.value
    )
    proposals = all_proposals[2 * step_index : 2 * step_index + 2]
    reviews = all_reviews[2 * step_index : 2 * step_index + 2]
    original_batch = _scenario_proposal_result(
        context, state, unit, proposals[0]
    ).value
    initial_reviews = _scenario_review_result(
        context, state, unit, reviews[0]
    ).value
    if not isinstance(original_batch, ProposalBatch) or not isinstance(
        initial_reviews, tuple
    ):
        return None
    original = next(
        (item for item in original_batch.proposed_items if item.slot_id == proposal_slot.slot_id),
        None,
    )
    review_id = make_review_item_id(
        unit.unit["unit_scientific_key"], "probe", proposal_slot.scientific_key
    )
    original_review = next(
        (item for item in initial_reviews if item.review_item_id == review_id), None
    )
    if original is None or original_review is None:
        return None
    corrective = None
    corrective_review = None
    corrective_batch = _scenario_proposal_result(
        context, state, unit, proposals[1]
    ).value
    if isinstance(corrective_batch, ProposalBatch):
        corrective = next(
            (
                item
                for item in corrective_batch.proposed_items
                if item.slot_id == proposal_slot.slot_id
            ),
            None,
        )
    if corrective is not None:
        rereviews = _scenario_review_result(
            context, state, unit, reviews[1]
        ).value
        if isinstance(rereviews, tuple):
            corrective_review = next(
                (item for item in rereviews if item.review_item_id == review_id), None
            )
    if (corrective is None) != (corrective_review is None):
        return None
    selected = select_strategy_realization(
        original, original_review, corrective, corrective_review
    )
    if selected is None or selected.review.status != "aligned":
        return None
    return selected


def _scenario_outcome_request(
    context: RunContext, state: RunState, unit: Any, action: LogicalActionView
) -> tuple[dict[str, Any], dict[str, tuple[str, str]]]:
    root, _, proposal_slot = _scenario_root_context(unit, action)
    step_index = _scenario_step_index(unit, root, action)
    targets = _scenario_target_actions(state, root)
    if len(targets) <= step_index + 1:
        raise EngineError("scenario outcome lacks its step target")
    terminal = targets[step_index + 1]
    selected = _scenario_selected_realization(
        context, state, unit, terminal
    )
    target = state.target_results.get(terminal.action_id)
    observation = state.target_observations.get(terminal.action_id)
    panel_plan = _panel_for_action(unit, terminal.action_id)
    panel = None if panel_plan is None else state.panel_views.get(panel_plan["panel_id"])
    if (
        selected is None
        or target is None
        or not target.usable
        or observation is None
        or panel is None
        or not panel.resolved
    ):
        return {"items": []}, {}
    binding = _scenario_binding_result(
        context, state, unit, _scenario_binding_action(state, unit, root, action)
    ).value
    if not isinstance(binding, TemplateBindingResult) or binding.contract is None:
        raise EngineError("scenario outcome lacks an applicable bound contract")
    contract = binding.contract
    _, _, before_text, _ = _scenario_fresh_branch(state, unit, action)
    review_id = make_review_item_id(
        unit.unit["unit_scientific_key"], "outcome", proposal_slot.scientific_key
    )
    expected = {
        review_id: (observation.scientific_key, observation.response)
    }
    item = {
        "review_alias": outcome_review_aliases(expected)[review_id],
        "transition_criterion": contract.success.to_dict(),
        "before_target": {
            "role": "assistant",
            "content": before_text,
        },
        "intervening_user": {
            "role": "user",
            "content": selected.proposal.user_turn,
        },
        "after_target": {
            "role": "assistant",
            "content": observation.response,
        },
    }
    return {"items": [item]}, expected


def _scenario_outcome_result(
    context: RunContext, state: RunState, unit: Any, action: LogicalActionView
) -> SemanticResult:
    _, expected = _scenario_outcome_request(context, state, unit, action)
    if not expected:
        return SemanticResult(())
    validated, structural_errors = _schema_validated_payloads(
        context,
        action,
        incomplete_error="scenario outcome review is truncated or incomplete",
    )
    errors = list(structural_errors)
    for _, _, value in validated:
        try:
            return SemanticResult(parse_outcome_reviews(value, expected))
        except DomainError as exc:
            errors.append(str(exc))
    return SemanticResult(
        None, tuple(errors) or ("no authoritative scenario outcome review",)
    )


def _standalone_map_dispatch_status(
    context: RunContext,
    state: RunState,
    unit: Any,
    action: LogicalActionView,
) -> str:
    """Resolve local standalone ownership before any target dispatch.

    The complete proposal and review payloads remain authoritative evidence.
    This pure guard only decides whether the action owns the one allowed target
    sample; duplicate and conflicting rediscoveries close without a call.
    """

    if unit.unit["kind"] != "map_exploration":
        return "dispatch"
    guard = action.slot["guard"]
    if guard.get("kind") != "if_dispatchable_probe":
        raise EngineError("ordinary MAP target lacks a dispatchable-probe guard")
    proposal_action = next(
        item
        for item in unit.actions
        if item.slot["purpose"] == ActionPurpose.PERTURBATOR_PROPOSAL.value
    )
    review_action = next(
        item
        for item in unit.actions
        if item.slot["purpose"] == ActionPurpose.PROBE_REVIEW.value
    )
    batch = _map_proposal_result(context, state, unit, proposal_action).value
    reviews = _map_review_result(context, state, unit, review_action).value
    if not isinstance(batch, ProposalBatch) or not isinstance(reviews, tuple):
        return "not_dispatchable"

    proposals_by_slot = {item.slot_id: item for item in batch.items}
    reviews_by_item = {item.review_item_id: item for item in reviews}
    realizations: list[tuple[str, str, Mapping[str, str]]] = []
    for slot in _proposal_slots(unit):
        proposal = proposals_by_slot.get(slot.slot_id)
        review = reviews_by_item.get(
            make_review_item_id(
                unit.unit["unit_scientific_key"], "probe", slot.scientific_key
            )
        )
        if proposal is None or review is None:
            continue
        selected = select_ordinary_realization(proposal, review)
        if selected is None or selected.review.reviewed_descriptors is None:
            continue
        prompt = selected.proposal.user_turn or ""
        candidate_key = _candidate_scientific_key(context.resources, prompt)
        realizations.append(
            (
                slot.scientific_key,
                candidate_key,
                dict(selected.review.reviewed_descriptors),
            )
        )

    current_slot_key = guard["proposal_slot_scientific_key"]
    current = next(
        (item for item in realizations if item[0] == current_slot_key), None
    )
    if current is None:
        return "not_dispatchable"
    _, candidate_key, descriptors = current
    owner_slot_key = min(
        slot_key
        for slot_key, other_candidate_key, _ in realizations
        if other_candidate_key == candidate_key
    )
    if current_slot_key != owner_slot_key:
        return "duplicate_candidate"

    existing = context.candidates.get(candidate_key)
    if existing is None:
        return "dispatch"
    if existing.proposal_slot_scientific_key == current_slot_key:
        return (
            "dispatch"
            if dict(existing.descriptors) == dict(descriptors)
            else "descriptor_conflict"
        )
    if dict(existing.descriptors) != dict(descriptors):
        return "descriptor_conflict"
    return "duplicate_candidate"


def _stateful_candidate_scientific_key(
    context: RunContext, parent_node_scientific_key: str, prompt: str
) -> str:
    return scientific_key(
        [
            parent_node_scientific_key,
            "stateful_turn",
            context.config.target.digest,
            sha256_bytes(normalize_line_endings(prompt).encode("utf-8")),
        ]
    )


def _portfolio_stateful_dispatch_status(
    context: RunContext,
    state: RunState,
    unit: Any,
    action: LogicalActionView,
) -> str:
    """Return the predispatch owner/conflict state for one stateful child."""

    current = _portfolio_selected_realization(context, state, unit, action)
    if current is None or current[2].review.reviewed_descriptors is None:
        return "not_dispatchable"
    parent_key = unit.unit["design"]["generation_parent_node_scientific_key"]
    realizations: list[tuple[str, str, Mapping[str, str]]] = []
    for target_action in unit.actions:
        if target_action.slot["role"] != Role.TARGET.value:
            continue
        resolved = _portfolio_selected_realization(
            context, state, unit, target_action
        )
        if resolved is None or resolved[2].review.reviewed_descriptors is None:
            continue
        slot, _, selected = resolved
        prompt = selected.proposal.user_turn or ""
        realizations.append(
            (
                slot.scientific_key,
                _stateful_candidate_scientific_key(context, parent_key, prompt),
                dict(selected.review.reviewed_descriptors),
            )
        )

    current_slot, _, selected = current
    prompt = selected.proposal.user_turn or ""
    candidate_key = _stateful_candidate_scientific_key(context, parent_key, prompt)
    same_prompt = [item for item in realizations if item[1] == candidate_key]
    if not same_prompt:
        return "not_dispatchable"
    descriptor_digests = {digest_json(item[2]) for item in same_prompt}
    if len(descriptor_digests) != 1:
        return "descriptor_conflict"
    owner_slot_key = min(item[0] for item in same_prompt)
    if current_slot.scientific_key != owner_slot_key:
        return "duplicate_candidate"

    descriptors = dict(selected.review.reviewed_descriptors)
    existing = context.candidates.get(candidate_key)
    if existing is None:
        return "dispatch"
    if dict(existing.descriptors) != descriptors:
        return "descriptor_conflict"
    if (
        existing.parent_node_scientific_key == parent_key
        and existing.proposal_slot_scientific_key == current_slot.scientific_key
    ):
        # Reconstructing an already intended action or its exact retry is not
        # a second logical dispatch.
        return "dispatch"
    return "duplicate_candidate"


def _portfolio_parent(
    context: RunContext, state: RunState, unit: Any
) -> tuple[CandidatePlan, Any, str]:
    turns = _portfolio_branch_turns(context, state, unit)
    parent_node, parent_candidate, parent_observation = turns[-1]
    return parent_candidate, parent_observation, parent_node.scientific_key


def _portfolio_branch_turns(
    context: RunContext, state: RunState, unit: Any
) -> tuple[tuple[Any, CandidatePlan, Any], ...]:
    """Resolve and validate the exact ancestry of a portfolio parent node."""

    design = unit.unit["design"]
    parent_node_id = design["generation_parent_node_id"]
    parent_node_key = design["generation_parent_node_scientific_key"]
    root_id = design["conversation_root_id"]
    root_key = unit.unit["subject_scientific_key"]
    nodes = state.conversation_nodes
    parent = nodes.get(parent_node_id)
    if parent is None or parent.scientific_key != parent_node_key:
        raise EngineError("portfolio parent node identity does not match reduced state")
    if parent.root_id != root_id:
        raise EngineError("portfolio parent node belongs to a different conversation root")
    reduced_root = state.conversation_roots.get(root_id)
    if reduced_root is None:
        raise EngineError("portfolio conversation root is absent from reduced state")
    if reduced_root.scientific_key != root_key:
        raise EngineError("portfolio conversation-root scientific identity disagrees")

    reverse_nodes: list[Any] = []
    seen: set[str] = set()
    current = parent
    while current is not None:
        if current.node_id in seen:
            raise EngineError("portfolio conversation ancestry contains a cycle")
        if current.root_id != root_id:
            raise EngineError("portfolio conversation ancestry crosses roots")
        seen.add(current.node_id)
        reverse_nodes.append(current)
        current = (
            None
            if current.parent_node_id is None
            else nodes.get(current.parent_node_id)
        )
        if reverse_nodes[-1].parent_node_id is not None and current is None:
            raise EngineError("portfolio conversation ancestry has a missing parent")
    ordered_nodes = tuple(reversed(reverse_nodes))
    observations_by_id = {
        item.observation_id: item for item in state.target_observations.values()
    }
    if len(observations_by_id) != len(state.target_observations):
        raise EngineError("target observation identity is not unique")

    turns: list[tuple[Any, CandidatePlan, Any]] = []
    expected_messages: list[dict[str, str]] = []
    prior_node_id: str | None = None
    for depth, node in enumerate(ordered_nodes):
        if node.depth != depth or node.parent_node_id != prior_node_id:
            raise EngineError("portfolio conversation ancestry is not contiguous")
        candidate = _candidate_by_id(context, node.candidate_id)
        observation = observations_by_id.get(node.target_observation_id)
        if candidate is None or observation is None:
            raise EngineError("portfolio branch lacks durable candidate/target evidence")
        if (
            observation.candidate_id != candidate.candidate_id
            or observation.parent_node_id != node.parent_node_id
        ):
            raise EngineError("portfolio branch evidence disagrees with its node")
        dispatched = [
            {"role": message.role, "content": message.content}
            for message in observation.messages
            if message.role != "system"
        ]
        expected_dispatch = [
            *expected_messages,
            {"role": "user", "content": candidate.prompt},
        ]
        if dispatched != expected_dispatch:
            raise EngineError("portfolio target evidence does not contain exact full ancestry")
        expected_messages = [
            *expected_dispatch,
            {"role": "assistant", "content": observation.response},
        ]
        turns.append((node, candidate, observation))
        prior_node_id = node.node_id
    if not turns:
        raise EngineError("portfolio parent branch is empty")
    return tuple(turns)


def _portfolio_branch_context(
    context: RunContext, state: RunState, unit: Any
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    conversation: list[dict[str, str]] = []
    anchors: list[dict[str, str]] = []
    for _, candidate, observation in _portfolio_branch_turns(context, state, unit):
        conversation.extend(
            (
                {"role": "user", "content": candidate.prompt},
                {"role": "assistant", "content": observation.response},
            )
        )
        anchors.append(
            {"turn_ref": observation.scientific_key, "text": observation.response}
        )
    return conversation, anchors


def _strategy_goal_context(context: RunContext) -> dict[str, str]:
    return {
        "objective_id": context.objective.id,
        "strategy_goal": context.objective.strategy_goal,
    }


def _portfolio_strategy_evidence(
    context: RunContext, state: RunState, unit: Any
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], frozenset[str]]]:
    turns = _portfolio_branch_turns(context, state, unit)
    frozen = strategy_evidence_polarities(
        turn_texts={observation.scientific_key: observation.response for _, _, observation in turns},
        target_observations=state.target_observations,
        panel_views=state.panel_views,
        member_assessments=state.member_assessments,
    )
    catalog, _ = build_strategy_evidence_catalog(
        tuple(
            (observation.scientific_key, observation.response)
            for _, _, observation in turns
        ),
        frozen,
    )
    return catalog, frozen


def _procedure_assessments(state: RunState, panel: Any) -> list[dict[str, Any]]:
    included = (() if panel.primary is None else (panel.primary,)) if panel.details is None else tuple(
        member for member in state.member_assessments.values()
        if member.assessment_id in panel.assessment.member_ids
    )
    return [{"member_slot": member.logical_member_slot, "procedure": member.procedure.to_dict(),
             "evidence": [{"quote": item.quote, "polarity": item.polarity} for item in member.evidence],
             "justification": member.justification} for member in included]


def _generation_brief(context: RunContext, state: RunState, unit: Any) -> dict[str, Any]:
    """Pin the selected parent independently of bounded similarity retrieval."""
    if unit.unit["kind"] == "portfolio_round":
        _, parent, observation = _portfolio_branch_turns(context, state, unit)[-1]
    else:
        parent = _map_parent_candidate(context, unit)
        observations = [item for item in state.target_observations.values()
                        if parent is not None and item.candidate_id == parent.candidate_id]
        observation = observations[-1] if observations else None
    panel = None if observation is None else next(
        (item for item in state.panel_views.values() if item.observation_id == observation.observation_id), None)
    return {
        "parent_context_role": (
            "current_branch" if unit.unit["kind"] == "portfolio_round"
            else "generation_reference_only"
        ),
        "parent_candidate_key": None if parent is None else parent.scientific_key,
        "parent_prompt": None if parent is None else parent.prompt,
        "parent_observation_key": None if observation is None else observation.scientific_key,
        "parent_reply": None if observation is None else observation.response,
        "next_transition": "recover_opening" if panel is None else portfolio_transition(panel, state.member_assessments),
        "procedure_assessments": [] if panel is None else _procedure_assessments(state, panel),
        "prior_rejections": _prior_rejections(context, state, unit),
    }


def _prior_rejections(context: RunContext, state: RunState, unit: Any) -> list[dict[str, Any]]:
    result = []
    for previous in state.work_units:
        if previous.start_event["sequence"] >= unit.start_event["sequence"]:
            continue
        if previous.unit["kind"] == "map_exploration":
            proposal_action = next(action for action in previous.actions if action.slot["purpose"] == ActionPurpose.PERTURBATOR_PROPOSAL.value)
            batch = _map_proposal_result(context, state, previous, proposal_action).value
            if not isinstance(batch, ProposalBatch):
                continue
            proposals = {item.slot_id: item for item in batch.items}
            for action in previous.actions:
                if action.slot["purpose"] != ActionPurpose.PROBE_REVIEW.value:
                    continue
                for review in _map_review_result(context, state, previous, action).value or ():
                    if review.status not in {"valid", "aligned"}:
                        result.append({"user_turn": proposals[review.slot_id].user_turn, "status": review.status, "reason": review.reason})
        elif previous.unit["kind"] == "portfolio_round":
            view = state.portfolio_rounds.get(previous.unit_id)
            if view is not None:
                for child in view.children:
                    review = child.original_review
                    if review is not None and review.status not in {"valid", "aligned"}:
                        result.append({"user_turn": None if child.original is None else child.original.user_turn, "status": review.status, "reason": review.reason})
    return result[-context.config.search.proposals_per_step:]


def _portfolio_current_assessment(
    state: RunState, anchors: Sequence[Mapping[str, str]]
) -> dict[str, str] | None:
    """Project one conservative qualitative stage from existing judge evidence."""

    if not anchors:
        return None
    current_ref = anchors[-1]["turn_ref"]
    observation = next(
        (
            item
            for item in getattr(state, "target_observations", {}).values()
            if item.scientific_key == current_ref
        ),
        None,
    )
    if observation is None:
        return None
    panel = next(
        (
            item
            for item in getattr(state, "panel_views", {}).values()
            if item.observation_id == observation.observation_id
        ),
        None,
    )
    if panel is None:
        return None
    current = portfolio_current_assessment(panel, state.member_assessments)
    if current is None:
        return None
    verdict, application = current
    return {
        "verdict": verdict.value,
        "application": application.value,
        "ownership": portfolio_ownership(panel, state.member_assessments),
        "next_transition": portfolio_transition(panel, state.member_assessments),
        "procedure_assessments": _procedure_assessments(state, panel),
    }


def _prior_attempts(
    context: RunContext, state: RunState, unit: Any, query: str
) -> list[dict[str, Any]]:
    """Retrieve prompts for novelty only, never another branch's concessions.

    The current branch and pinned parent supply response evidence separately.
    Cross-attempt replies and assessments remain in the journal, not this view.
    """
    query_words = set(query.casefold().split())
    candidates: list[tuple[int, int, str, dict[str, Any]]] = []
    for previous in state.work_units:
        if previous.unit_id == unit.unit_id or previous.unit["kind"] not in {
            "portfolio_round", "map_exploration", "initial_observation"
        }:
            continue
        if previous.start_event["sequence"] >= unit.start_event["sequence"]:
            continue
        for action in previous.actions:
            if action.slot["role"] != "target" or not action.attempts:
                continue
            messages = action.attempts[0].intended["messages"]
            prompt = messages[-1]["content"]
            item = {
                "attempt_ref": action.slot["action_scientific_key"],
                "context_role": "novelty_only",
                "user_turn": prompt,
            }
            candidates.append((
                len(query_words & set(prompt.casefold().split())),
                previous.start_event["sequence"],
                action.slot["action_scientific_key"],
                item,
            ))
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [item[3] for item in candidates[:context.config.search.proposals_per_step]]


def _portfolio_strategy_selection_limit(
    context: RunContext, state: RunState, unit: Any
) -> int:
    """Return the pre-response strategy count cap for this frozen root."""

    del context, state
    maximum = unit.unit["design"]["maximum_strategy_count"]
    try:
        PortfolioIntent(unit.unit["design"]["portfolio_intent"])
    except (KeyError, ValueError) as exc:
        raise EngineError("portfolio intent is invalid") from exc
    return maximum


def _portfolio_selector_request(
    context: RunContext,
    state: RunState,
    unit: Any,
    action: LogicalActionView | None = None,
) -> dict[str, Any]:
    conversation, anchors = _portfolio_branch_context(context, state, unit)
    strategy_evidence, _ = _portfolio_strategy_evidence(context, state, unit)
    del action
    request = {
        "objective_definition": _strategy_goal_context(context),
        "descriptor_vocabulary": {
            name: list(context.objective.descriptor_values[name])
            for name in context.objective.descriptor_names
        },
        "diagnosis_categories": list(context.objective.diagnosis_categories),
        "portfolio_intent": unit.unit["design"]["portfolio_intent"],
        "maximum_strategies": _portfolio_strategy_selection_limit(
            context, state, unit
        ),
        "conversation": conversation,
        "evidence_catalog": strategy_evidence,
        "current_assessment": _portfolio_current_assessment(state, anchors),
        "generation_brief": _generation_brief(context, state, unit),
        "finisher_strategies": strict_json_loads(context.resources.templates["finisher_strategies.json"]),
        "prior_attempts": _prior_attempts(context, state, unit, conversation[-1]["content"]),
    }
    parent, _, _ = _portfolio_parent(context, state, unit)
    ancestry = _portfolio_candidate_ancestry(context, state, parent)
    if ancestry is None:
        raise EngineError("portfolio request parent is not scientifically eligible")
    if ancestry.linked_steps:
        source = ancestry.linked_steps[-1].episode
        request["linked_parent_strategy"] = {
            "contract": source.contract.to_dict(),
        }
    return request


def _portfolio_lanes(
    context: RunContext, state: RunState, unit: Any
) -> tuple[BoundPortfolioLane, ...]:
    selector = next(
        item
        for item in unit.actions
        if item.slot["purpose"] == ActionPurpose.SELECTOR_PORTFOLIO.value
    )
    portfolio = _portfolio_result(context, state, unit, selector).value
    if not isinstance(portfolio, SelectorPortfolio):
        return ()
    slots = tuple(
        StrategySlot(item["entity_id"], item["scientific_key"], item["ordinal"])
        for item in unit.unit["design"]["strategy_slots"]
    )
    return bind_portfolio_to_slots(
        portfolio,
        slots,
        unit.unit["design"]["generation_parent_node_scientific_key"],
    )


def _portfolio_assignments(
    context: RunContext, state: RunState, unit: Any
) -> tuple[Any, ...]:
    lanes = _portfolio_lanes(context, state, unit)
    if not lanes:
        return ()
    return allocate_contract_step(_proposal_slots(unit), lanes)


def _portfolio_lane_for_action(
    context: RunContext, state: RunState, unit: Any, action: LogicalActionView
) -> BoundPortfolioLane | None:
    return next(
        (
            lane
            for lane in _portfolio_lanes(context, state, unit)
            if action.slot["owner_position_scientific_key"]
            in {
                lane.strategy_slot.scientific_key,
                lane.strategy_scientific_key,
            }
        ),
        None,
    )


def _portfolio_slots_for_lane(
    context: RunContext,
    state: RunState,
    unit: Any,
    lane: BoundPortfolioLane,
) -> tuple[ProposalSlot, ...]:
    assigned = {
        item.proposal_slot_scientific_key
        for item in _portfolio_assignments(context, state, unit)
        if item.strategy_slot_scientific_key
        in {
            lane.strategy_slot.scientific_key,
            lane.strategy_scientific_key,
        }
    }
    return tuple(item for item in _proposal_slots(unit) if item.scientific_key in assigned)


def _portfolio_proposal_result(
    context: RunContext, state: RunState, unit: Any, action: LogicalActionView
) -> SemanticResult:
    lane = _portfolio_lane_for_action(context, state, unit, action)
    if lane is None:
        return SemanticResult(None, ("generic strategy slot is unbound",))
    slots = _portfolio_slots_for_lane(context, state, unit, lane)
    assignments = {
        item.proposal_slot_scientific_key: item
        for item in _portfolio_assignments(context, state, unit)
    }
    validated, structural_errors = _schema_validated_payloads(
        context,
        action,
        incomplete_error="strategy proposal response is truncated or incomplete",
    )
    errors = list(structural_errors)
    for _, _, value in validated:
        try:
            return SemanticResult(
                parse_proposal_batch(
                    value,
                    slots,
                )
            )
        except DomainError as exc:
            errors.append(str(exc))
    return SemanticResult(None, tuple(errors) or ("no authoritative strategy proposal batch",))


def _portfolio_review_result(
    context: RunContext, state: RunState, unit: Any, action: LogicalActionView
) -> SemanticResult:
    proposal_actions = [
        item
        for item in unit.actions
        if item.slot["purpose"] == ActionPurpose.PERTURBATOR_PROPOSAL.value
    ]
    expectations: list[ReviewExpectation] = []
    contracts: dict[str, Any] = {}
    slots_by_key = {item.scientific_key: item for item in _proposal_slots(unit)}
    assignments = {
        item.proposal_slot_scientific_key: item
        for item in _portfolio_assignments(context, state, unit)
    }
    lanes = {
        lane.strategy_slot.scientific_key: lane
        for lane in _portfolio_lanes(context, state, unit)
    }
    for proposal_action in proposal_actions:
        parsed = _portfolio_proposal_result(context, state, unit, proposal_action).value
        if not isinstance(parsed, ProposalBatch):
            continue
        for proposal in parsed.proposed_items:
            slot = next(
                (item for item in slots_by_key.values() if item.slot_id == proposal.slot_id),
                None,
            )
            if slot is None or slot.scientific_key not in assignments:
                continue
            assignment = assignments[slot.scientific_key]
            lane = lanes.get(assignment.strategy_slot_scientific_key)
            if lane is None:
                continue
            review_id = make_review_item_id(
                unit.unit["unit_scientific_key"], "probe", slot.scientific_key
            )
            expectations.append(ReviewExpectation(review_id, proposal))
            contracts[proposal.slot_id] = dict(lane.item.contract.move.descriptors)
    if not expectations:
        return SemanticResult(())
    validated, structural_errors = _schema_validated_payloads(
        context,
        action,
        incomplete_error="strategy probe review is truncated or incomplete",
    )
    errors = list(structural_errors)
    for _, _, value in validated:
        try:
            return SemanticResult(
                parse_probe_reviews(
                    value,
                    expectations,
                    context.objective.descriptor_values,
                    mode="strategy",
                    contract_descriptors_by_slot=contracts,
                )
            )
        except DomainError as exc:
            errors.append(str(exc))
    return SemanticResult(None, tuple(errors) or ("no authoritative strategy probe review",))


def _portfolio_selected_realization(
    context: RunContext,
    state: RunState,
    unit: Any,
    target_action: LogicalActionView,
) -> tuple[ProposalSlot, BoundPortfolioLane, SelectedRealization] | None:
    """Resolve the one deterministic owner of a portfolio target slot."""

    guard = target_action.slot["guard"]
    if guard["kind"] != "if_dispatchable_probe":
        raise EngineError("portfolio target action lacks its frozen proposal guard")
    slot = next(
        (
            item
            for item in _proposal_slots(unit)
            if item.scientific_key == guard["proposal_slot_scientific_key"]
        ),
        None,
    )
    if slot is None:
        raise EngineError("portfolio target guard names an unknown proposal slot")
    assignment = next(
        (
            item
            for item in _portfolio_assignments(context, state, unit)
            if item.proposal_slot_scientific_key == slot.scientific_key
        ),
        None,
    )
    if assignment is None or assignment.strategy_slot_scientific_key is None:
        return None
    lane = next(
        (
            item
            for item in _portfolio_lanes(context, state, unit)
            if item.strategy_slot.scientific_key
            == assignment.strategy_slot_scientific_key
        ),
        None,
    )
    if lane is None:
        return None
    lane_actions = [
        item
        for item in unit.actions
        if item.slot["purpose"] == ActionPurpose.PERTURBATOR_PROPOSAL.value
        and item.slot["owner_position_scientific_key"]
        == lane.strategy_slot.scientific_key
    ]
    if len(lane_actions) != 1:
        raise EngineError("portfolio lane does not have exactly one generator action")
    original_action = lane_actions[0]
    original_batch = _portfolio_proposal_result(
        context, state, unit, original_action
    ).value
    initial_review_action = next(
        item
        for item in unit.actions
        if item.slot["purpose"] == ActionPurpose.PROBE_REVIEW.value
    )
    initial_reviews = _portfolio_review_result(
        context, state, unit, initial_review_action
    ).value
    if not isinstance(original_batch, ProposalBatch) or not isinstance(
        initial_reviews, tuple
    ):
        return None
    original = next(
        (
            item
            for item in original_batch.proposed_items
            if item.slot_id == slot.slot_id
        ),
        None,
    )
    original_review = next(
        (
            item
            for item in initial_reviews
            if item.review_item_id == guard["review_item_id"]
        ),
        None,
    )
    if original is None or original_review is None:
        return None

    selected = select_strategy_realization(original, original_review)
    return None if selected is None else (slot, lane, selected)


def _portfolio_outcome_review_request(
    context: RunContext, state: RunState, unit: Any
) -> tuple[dict[str, Any], dict[str, tuple[str, str]]]:
    """Build the exact masked batch for usable, aligned, resolved children."""

    branch = _portfolio_branch_turns(context, state, unit)
    _, _, before = branch[-1]
    target_actions = tuple(
        item for item in unit.actions if item.slot["role"] == Role.TARGET.value
    )
    outcome_actions = tuple(
        item
        for item in unit.actions
        if item.slot["purpose"] == ActionPurpose.OUTCOME_REVIEW.value
    )
    if len(outcome_actions) != 1:
        raise EngineError("portfolio round does not have one outcome-review action")
    outcome_guard = outcome_actions[0].slot["guard"]
    if (
        outcome_guard["kind"] != "if_any_aligned_response"
        or tuple(outcome_guard["target_action_ids"])
        != tuple(item.action_id for item in target_actions)
    ):
        raise EngineError("outcome-review guard does not name the exact target set")
    pending: list[tuple[dict[str, Any], tuple[str, str]]] = []
    for target_action in target_actions:
        selected = _portfolio_selected_realization(
            context, state, unit, target_action
        )
        if selected is None or selected[2].review.status != "aligned":
            continue
        slot, lane, realization = selected
        target = state.target_results.get(target_action.action_id)
        observation = state.target_observations.get(target_action.action_id)
        panel_plan = _panel_for_action(unit, target_action.action_id)
        panel = (
            None
            if panel_plan is None
            else state.panel_views.get(panel_plan["panel_id"])
        )
        if (
            target is None
            or not target.usable
            or observation is None
            or panel is None
            or not panel.resolved
        ):
            continue
        if observation.response != target.text:
            raise EngineError("portfolio target result and observation disagree")
        review_id = make_review_item_id(
            unit.unit["unit_scientific_key"], "outcome", slot.scientific_key
        )
        item = {
            "_review_item_id": review_id,
            "transition_criterion": lane.item.contract.success.to_dict(),
            "before_target": {
                "role": "assistant",
                "content": before.response,
            },
            "intervening_user": {
                "role": "user",
                "content": realization.proposal.user_turn,
            },
            "after_target": {
                "role": "assistant",
                "content": observation.response,
            },
        }
        pending.append(
            (item, (observation.scientific_key, observation.response))
        )
    pending.sort(
        key=lambda pair: (
            seeded_rank(
                context.config.random_seed,
                [
                    "portfolio_outcome_review",
                    unit.unit["unit_scientific_key"],
                    unit.unit["design"]["round_index"],
                    pair[0]["_review_item_id"],
                ],
            ),
            pair[0]["_review_item_id"],
        )
    )
    expected_by_id = {
        item["_review_item_id"]: expected
        for item, expected in pending
    }
    aliases = outcome_review_aliases(expected_by_id)
    return (
        {
            "objective_definition": _strategy_goal_context(context),
            "items": [
                {
                    "review_alias": aliases[item["_review_item_id"]],
                    "transition_criterion": item["transition_criterion"],
                    "before_target": item["before_target"],
                    "intervening_user": item["intervening_user"],
                    "after_target": item["after_target"],
                }
                for item, _ in pending
            ],
        },
        expected_by_id,
    )


def _portfolio_outcome_result(
    context: RunContext, state: RunState, unit: Any, action: LogicalActionView
) -> SemanticResult:
    _, expected = _portfolio_outcome_review_request(context, state, unit)
    if not expected:
        return SemanticResult(())
    validated, structural_errors = _schema_validated_payloads(
        context,
        action,
        incomplete_error="strategy outcome review is truncated or incomplete",
    )
    errors = list(structural_errors)
    for _, _, value in validated:
        try:
            return SemanticResult(parse_outcome_reviews(value, expected))
        except DomainError as exc:
            errors.append(str(exc))
    return SemanticResult(
        None, tuple(errors) or ("no authoritative strategy outcome review",)
    )


def _portfolio_result(
    context: RunContext, state: RunState, unit: Any, action: LogicalActionView
) -> SemanticResult:
    _, anchors = _portfolio_branch_context(context, state, unit)
    turn_texts = {item["turn_ref"]: item["text"] for item in anchors}
    current_assessment = _portfolio_current_assessment(state, anchors)
    _, evidence_polarities = _portfolio_strategy_evidence(context, state, unit)
    _, evidence_by_alias = build_strategy_evidence_catalog(
        tuple((item["turn_ref"], item["text"]) for item in anchors),
        evidence_polarities,
    )
    validated, structural_errors = _schema_validated_payloads(
        context,
        action,
        incomplete_error="selector portfolio is truncated or incomplete",
    )
    errors = list(structural_errors)
    maximum = _portfolio_strategy_selection_limit(context, state, unit)
    for _, _, value in validated:
        try:
            portfolio = parse_selector_portfolio(
                value,
                context.objective.descriptor_values,
                context.objective.diagnosis_categories,
                turn_text_by_ref=turn_texts,
                evidence_by_alias=evidence_by_alias,
                maximum_strategies=maximum,
                diagnosis_normalizer=context.objective.normalize_diagnosis,
                evidence_polarities=evidence_polarities,
                current_verdict=(
                    None
                    if current_assessment is None
                    else Verdict(current_assessment["verdict"])
                ),
                current_application=(
                    None
                    if current_assessment is None
                    else Application(current_assessment["application"])
                ),
                current_ownership=None if current_assessment is None else current_assessment["ownership"],
                current_transition=None if current_assessment is None else current_assessment["next_transition"],
                portfolio_intent=PortfolioIntent(
                    unit.unit["design"]["portfolio_intent"]
                ),
            )
            return SemanticResult(portfolio)
        except DomainError as exc:
            errors.append(str(exc))
    return SemanticResult(None, tuple(errors) or ("no authoritative selector portfolio",))


def _generic_json_result(context: RunContext, action: LogicalActionView) -> SemanticResult:
    validated, errors = _schema_validated_payloads(
        context,
        action,
        incomplete_error="structured response is truncated or incomplete",
    )
    for _, _, value in validated:
        return SemanticResult(value)
    return SemanticResult(None, tuple(errors) or ("no authoritative structured response",))


def _semantic_result(
    context: RunContext, state: RunState, unit: Any, action: LogicalActionView
) -> SemanticResult:
    # A skipped slot has no output to parse. Its contextual parser may require
    # a predecessor reply that the frozen guard deliberately prevented from
    # being generated (for example, after an early replay probe is rejected).
    if not action.attempts:
        return SemanticResult(None, ("action has not been dispatched",))
    role = action.slot["role"]
    purpose = action.slot["purpose"]
    if role == Role.TARGET.value:
        return _target_result(action, state)
    if role == Role.JUDGE.value:
        return _judge_result(context, state, unit, action)
    if unit.unit["kind"] == "map_exploration":
        if purpose == ActionPurpose.PERTURBATOR_PROPOSAL.value:
            return _map_proposal_result(context, state, unit, action)
        if purpose == ActionPurpose.PROBE_REVIEW.value:
            return _map_review_result(context, state, unit, action)
    if unit.unit["kind"] == "portfolio_round":
        if purpose == ActionPurpose.PERTURBATOR_PROPOSAL.value:
            return _portfolio_proposal_result(context, state, unit, action)
        if purpose == ActionPurpose.PROBE_REVIEW.value:
            return _portfolio_review_result(context, state, unit, action)
        if purpose == ActionPurpose.OUTCOME_REVIEW.value:
            return _portfolio_outcome_result(context, state, unit, action)
    if unit.unit.get("kind") in _STRATEGY_QUALIFICATION_KINDS:
        if purpose == ActionPurpose.TEMPLATE_BINDING.value:
            return _qualification_binding_result(context, state, unit, action)
        if purpose == ActionPurpose.PERTURBATOR_PROPOSAL.value:
            return _qualification_proposal_result(context, state, unit, action)
        if purpose == ActionPurpose.PROBE_REVIEW.value:
            return _qualification_review_result(context, state, unit, action)
        if purpose == ActionPurpose.OUTCOME_REVIEW.value:
            return _qualification_outcome_result(context, state, unit, action)
    if _is_scenario_verification(unit):
        if purpose == ActionPurpose.TEMPLATE_BINDING.value:
            return _scenario_binding_result(context, state, unit, action)
        if purpose == ActionPurpose.PERTURBATOR_PROPOSAL.value:
            return _scenario_proposal_result(context, state, unit, action)
        if purpose == ActionPurpose.PROBE_REVIEW.value:
            return _scenario_review_result(context, state, unit, action)
        if purpose == ActionPurpose.OUTCOME_REVIEW.value:
            return _scenario_outcome_result(context, state, unit, action)
    if purpose == ActionPurpose.SELECTOR_PORTFOLIO.value:
        return _portfolio_result(context, state, unit, action)
    return _generic_json_result(context, action)


def _route_attempts(action: LogicalActionView, route_key: str) -> tuple[AttemptView, ...]:
    return tuple(
        attempt for attempt in action.attempts if attempt.intended["role_route_key"] == route_key
    )


def _last_terminal_condition(action: LogicalActionView) -> tuple[str, Mapping[str, Any]] | None:
    if not action.attempts:
        return None
    attempt = action.attempts[-1]
    if attempt.terminal_event is None:
        return None
    return attempt.terminal_event["type"], attempt.terminal_event["payload"]


_OPERATIONAL_RETRY_ERROR_KINDS = frozenset(
    {"connection", "timeout", "rate_limit", "server", "malformed_response"}
)


def _operational_retry_delay_seconds(
    action: LogicalActionView, route_kind: str
) -> float | None:
    """Return the bounded delay for an operational corrective dispatch.

    Exact retries following authoritative empty/truncated target successes are
    scientific evidence recovery, not transport recovery, and therefore do not
    sleep.  Outcome-unknown retries have no authoritative Retry-After value.
    """

    if route_kind != "exact_retry":
        return None
    terminal = next(
        (
            attempt.terminal_event
            for attempt in reversed(action.attempts)
            if attempt.terminal_event is not None
        ),
        None,
    )
    if terminal is None:
        return None
    if terminal["type"] == "model_call_outcome_unknown":
        return 1.0
    if terminal["type"] != "model_call_failed":
        return None
    payload = terminal["payload"]
    if payload["error_kind"] not in _OPERATIONAL_RETRY_ERROR_KINDS:
        return None
    try:
        retry_after = require_decimal_string(
            payload.get("retry_after_seconds"), minimum=Decimal("0")
        )
    except DomainError:
        return 1.0
    return float(min(retry_after, Decimal("10")))


def _delay_before_corrective_dispatch(
    context: RunContext, action: LogicalActionView, route_kind: str
) -> None:
    delay = _operational_retry_delay_seconds(action, route_kind)
    if delay is not None:
        context.sleeper(delay)


def _retry_same_route(action: LogicalActionView, semantic: SemanticResult) -> tuple[bool, str]:
    condition = _last_terminal_condition(action)
    if condition is None:
        return True, "primary"
    event_type, payload = condition
    if event_type == "model_call_succeeded":
        if action.attempts[-1].target_identity_collision:
            # A colliding role result exhausts the current route immediately:
            # no exact retry, schema repair, or format downgrade can turn the
            # target itself into independent Selector/Perturbator evidence.
            return False, "fallback"
        if structured_output_exhausts_route(action.slot["role"], payload):
            return False, "fallback"
        # Target prose has no structured schema to repair.  An authoritative
        # empty or incomplete target completion receives the one exact retry
        # allowed by the target route; structured roles use schema repair.
        recovery_kind = (
            "exact_retry"
            if action.slot["role"] == Role.TARGET.value
            else "schema_repair"
        )
        return (not semantic.usable, recovery_kind)
    if event_type == "model_call_outcome_unknown":
        return True, "exact_retry"
    error = payload["error_kind"]
    if error == "unsupported_response_format":
        if action.slot["role"] == Role.TARGET.value:
            return False, "format_downgrade"
        # JSON-object mode is already the durable downgrade.  If that format
        # is rejected, this route is incompatible and must exhaust without
        # spending a same-route attempt on the identical mode.
        return (
            action.attempts[-1].intended["response_mode"] == "json_schema",
            "format_downgrade",
        )
    if error in {"connection", "timeout", "rate_limit", "server", "malformed_response"}:
        return True, "exact_retry"
    return False, "exact_retry"


def _next_route_plan(
    action: LogicalActionView, semantic: SemanticResult
) -> tuple[Mapping[str, Any], str] | None:
    routes = action.slot["routes"]
    if not action.attempts:
        route = routes[0]
        initial_kind = "replacement" if route["route_label"] == "replacement" else "primary"
        return route, initial_kind
    last_attempt = action.attempts[-1]
    if last_attempt.terminal_event is None:
        return None
    last_route_key = last_attempt.intended["role_route_key"]
    route_index = next(
        index for index, route in enumerate(routes) if route["route_key"] == last_route_key
    )
    route = routes[route_index]
    route_attempts = _route_attempts(action, last_route_key)
    retry, kind = _retry_same_route(action, semantic)
    if retry and len(route_attempts) < route["maximum_attempts"]:
        return route, kind
    if route_index + 1 < len(routes):
        return routes[route_index + 1], "fallback"
    return None


def _next_action_route_plan(
    unit: Any, action: LogicalActionView, semantic: SemanticResult
) -> tuple[Mapping[str, Any], str] | None:
    return _next_route_plan(action, semantic)


def _action_closed(
    context: RunContext, state: RunState, unit: Any, action: LogicalActionView
) -> bool:
    semantic = _semantic_result(context, state, unit, action)
    if semantic.usable:
        return True
    if action.attempts and action.attempts[-1].terminal_event is None:
        return False
    return _next_action_route_plan(unit, action, semantic) is None


def _guard_allows(
    context: RunContext, state: RunState, unit: Any, action: LogicalActionView
) -> bool:
    guard = action.slot["guard"]
    kind = guard["kind"]
    if kind == "always":
        return True
    if kind == "if_action_usable":
        dependency = state.logical_actions[guard["action_id"]]
        return _semantic_result(context, state, unit, dependency).usable
    if kind == "if_panel_resolved":
        panel = state.panel_views.get(guard["panel_id"])
        return panel is not None and panel.resolved
    if kind == "if_panel_triggered":
        panel = next(item for item in unit.unit["panel_plans"] if item["panel_id"] == guard["panel_id"])
        return _panel_is_required(context, state, unit, panel)
    if kind == "if_panel_has_unresolved_normal_member":
        panel = next(item for item in unit.unit["panel_plans"] if item["panel_id"] == guard["panel_id"])
        return _replacement_slot(context, state, unit, panel) is not None
    if kind == "if_any_proposed_item":
        for action_id in guard["proposal_action_ids"]:
            result = _semantic_result(context, state, unit, state.logical_actions[action_id])
            if isinstance(result.value, ProposalBatch) and result.value.proposed_items:
                return True
            if isinstance(result.value, dict) and result.value.get("items"):
                return True
        return False
    if kind == "if_strategy_slot_bound":
        strategy_key = guard["strategy_slot_scientific_key"]
        if unit.unit["kind"] == "portfolio_round" and unit.unit["design"][
            "round_index"
        ] > 0:
            return any(
                lane.strategy_scientific_key == strategy_key
                for lane in _portfolio_lanes(context, state, unit)
            )
        selector = next(
            (item for item in unit.actions if item.slot["purpose"] == ActionPurpose.SELECTOR_PORTFOLIO.value),
            None,
        )
        if selector is None:
            return False
        portfolio = _semantic_result(context, state, unit, selector).value
        if not isinstance(portfolio, SelectorPortfolio):
            return False
        slots = unit.unit["design"]["strategy_slots"]
        ordinal = next(
            item["ordinal"]
            for item in slots
            if item["scientific_key"] == strategy_key
        )
        return ordinal < len(portfolio.items)
    if kind == "if_dispatchable_probe":
        if _is_scenario_verification(unit):
            selected = _scenario_selected_realization(
                context, state, unit, action
            )
            if selected is None:
                return False
            _, _, proposal_slot = _scenario_root_context(unit, action)
            if (
                proposal_slot.scientific_key
                != guard["proposal_slot_scientific_key"]
                or selected.review.review_item_id != guard["review_item_id"]
            ):
                raise EngineError("scenario target guard changed its frozen slot")
            return selected.review.status == "aligned"
        if unit.unit.get("kind") in _STRATEGY_QUALIFICATION_KINDS:
            selected = _qualification_selected_realization(
                context, state, unit, action
            )
            if selected is None:
                return False
            root_plan, _, _, _, proposal_slot, _ = _qualification_root_context(
                unit, action
            )
            if (
                proposal_slot is None
                or proposal_slot.scientific_key
                != guard["proposal_slot_scientific_key"]
                or selected.review.review_item_id != guard["review_item_id"]
            ):
                raise EngineError("qualification target guard changed its frozen slot")
            required_status = (
                "aligned"
                if root_plan["arm"] == "treatment"
                else "valid"
            )
            return selected.review.status == required_status
        if unit.unit["kind"] == "portfolio_round":
            selected = _portfolio_selected_realization(
                context, state, unit, action
            )
            return bool(
                selected is not None
                and selected[2].review.status in set(guard["allowed"])
                and _portfolio_stateful_dispatch_status(
                    context, state, unit, action
                )
                == "dispatch"
            )
        for review_action in reversed(
            [
                item
                for item in unit.actions[: action.slot["ordinal"]]
                if item.slot["purpose"] == ActionPurpose.PROBE_REVIEW.value
            ]
        ):
            reviews = _semantic_result(context, state, unit, review_action).value
            if not isinstance(reviews, tuple):
                continue
            matching = [
                review
                for review in reviews
                if isinstance(review, ProbeReview)
                and review.review_item_id == guard["review_item_id"]
            ]
            if matching:
                permitted = matching[0].status in set(guard["allowed"])
                if not permitted:
                    return False
                if unit.unit["kind"] == "map_exploration":
                    return (
                        _standalone_map_dispatch_status(
                            context, state, unit, action
                        )
                        == "dispatch"
                    )
                return True
        return False
    if kind == "if_any_aligned_response":
        if _is_scenario_verification(unit):
            request, _ = _scenario_outcome_request(
                context, state, unit, action
            )
            return bool(request["items"])
        if unit.unit.get("kind") in _STRATEGY_QUALIFICATION_KINDS:
            request, _ = _qualification_outcome_request(
                context, state, unit, action
            )
            return bool(request["items"])
        request, _ = _portfolio_outcome_review_request(context, state, unit)
        return bool(request["items"])
    if kind == "if_repairable_proposal_exists":
        if _is_scenario_verification(unit):
            _, _, proposal_slot = _scenario_root_context(unit, action)
            if guard["proposal_slot_scientific_keys"] != [
                proposal_slot.scientific_key
            ]:
                raise EngineError("scenario corrective guard changed its slot")
            reviews = _scenario_review_result(
                context,
                state,
                unit,
                state.logical_actions[guard["review_action_id"]],
            ).value
            return bool(
                isinstance(reviews, tuple)
                and any(
                    review.review_item_id
                    == make_review_item_id(
                        unit.unit["unit_scientific_key"],
                        "probe",
                        proposal_slot.scientific_key,
                    )
                    and review.status in set(guard["statuses"])
                    for review in reviews
                )
            )
        if unit.unit.get("kind") in _STRATEGY_QUALIFICATION_KINDS:
            _, _, _, _, proposal_slot, _ = _qualification_root_context(
                unit, action
            )
            if proposal_slot is None or guard["proposal_slot_scientific_keys"] != [
                proposal_slot.scientific_key
            ]:
                raise EngineError("qualification corrective guard changed its slot")
            review_action = state.logical_actions[guard["review_action_id"]]
            reviews = _qualification_review_result(
                context, state, unit, review_action
            ).value
            return bool(
                isinstance(reviews, tuple)
                and any(
                    review.review_item_id
                    == make_review_item_id(
                        unit.unit["unit_scientific_key"],
                        "probe",
                        proposal_slot.scientific_key,
                    )
                    and review.status in set(guard["statuses"])
                    for review in reviews
                )
            )
        lane = _portfolio_lane_for_action(context, state, unit, action)
        if lane is None:
            return False
        review_action = state.logical_actions[guard["review_action_id"]]
        reviews = _semantic_result(context, state, unit, review_action).value
        if not isinstance(reviews, tuple):
            return False
        allocated_keys = {
            slot.scientific_key
            for slot in _portfolio_slots_for_lane(context, state, unit, lane)
        }
        if not allocated_keys.issubset(guard["proposal_slot_scientific_keys"]):
            raise EngineError("corrective guard omits a possible allocated slot")
        permitted_ids = {
            make_review_item_id(
                unit.unit["unit_scientific_key"], "probe", proposal_key
            )
            for proposal_key in allocated_keys
        }
        return any(
            isinstance(review, ProbeReview)
            and review.review_item_id in permitted_ids
            and review.status in set(guard["statuses"])
            for review in reviews
        )
    if kind == "if_any_corrective_proposal":
        return any(
            isinstance(
                (
                    result := _semantic_result(
                        context, state, unit, state.logical_actions[action_id]
                    )
                ).value,
                ProposalBatch,
            )
            and bool(result.value.proposed_items)
            for action_id in guard["proposal_action_ids"]
        )
    if kind == "if_template_applicable":
        binding = state.logical_actions[guard["binding_action_id"]]
        result = _semantic_result(context, state, unit, binding).value
        return (
            isinstance(result, TemplateBindingResult)
            and result.status == "applicable"
        )
    raise EngineError(f"unknown frozen action guard: {kind}")


def _last_action_terminal_sequence(action: LogicalActionView) -> int | None:
    sequences = [
        attempt.terminal_event["sequence"]
        for attempt in action.attempts
        if attempt.terminal_event is not None
    ]
    return max(sequences) if sequences else None


def _provider_circuit_outcomes(
    context: RunContext, state: RunState
) -> tuple[ProviderCircuitOutcome, ...]:
    """Project completed role outcomes from the authoritative reduced state."""

    outcomes: list[ProviderCircuitOutcome] = []
    for unit in state.work_units:
        for action in unit.actions:
            role = action.slot["role"]
            if not action.attempts:
                continue
            sequence = _last_action_terminal_sequence(action)
            if sequence is None:
                continue
            semantic = _semantic_result(context, state, unit, action)
            if semantic.usable:
                status = "success"
            elif not _action_closed(context, state, unit, action):
                continue
            elif role == Role.TARGET.value:
                terminal = action.attempts[-1].terminal_event
                assert terminal is not None
                if (
                    terminal["type"] == "model_call_failed"
                    and terminal["payload"]["error_kind"] == "context_limit"
                ):
                    # Known-no-output context exhaustion has its own
                    # distinct-fresh-root circuit below. Outcome-unknown takes
                    # precedence and remains an ordinary target failure.
                    continue
                status = "failure"
            else:
                # Selector/Perturbator structural exhaustion is also a logical
                # provider failure; a valid empty/all-abstained result is
                # semantic.usable and resets the streak above.
                status = "failure"
            outcomes.append(
                ProviderCircuitOutcome(
                    role=role,
                    sequence=sequence,
                    scientific_key=action.slot["action_scientific_key"],
                    status=status,
                    route_key=(action.slot["routes"][0]["route_key"] if role == Role.JUDGE.value else None),
                )
            )

        # Judge circuit state is panel-scoped, not member-attempt scoped.
        for panel in unit.unit["panel_plans"]:
            if not _panel_is_required(context, state, unit, panel):
                continue
            action_ids = [
                item["judge_action_id"] for item in panel["normal_members"]
            ] + ([panel["replacement"]["judge_action_id"]] if panel["replacement"] is not None else [])
            actions = [state.logical_actions[action_id] for action_id in action_ids]
            terminal_sequences = [
                sequence
                for action in actions
                if (sequence := _last_action_terminal_sequence(action)) is not None
            ]
            if not terminal_sequences:
                continue
            if any(
                _guard_allows(context, state, unit, action)
                and not _action_closed(context, state, unit, action)
                for action in actions
            ):
                continue
            _, assessment, required = _panel_assessment(context, state, unit, panel)
            if not required:
                continue
            status = (
                "success"
                if assessment is not None
                and assessment.availability.value != "no_quorum"
                else "failure"
            )
            outcomes.append(
                ProviderCircuitOutcome(
                    role=Role.JUDGE.value,
                    sequence=max(terminal_sequences),
                    scientific_key=panel["panel_scientific_key"],
                    status=status,
                )
            )
    return tuple(outcomes)


def _target_context_limit_trip_sequence(
    state: RunState, *, reset_after: int = 0
) -> int | None:
    """Reduce the target context-limit circuit over distinct fresh roots.

    A usable target result is the only reset specified for this special
    sequence. The latest three affected actions must name three different
    roots, preventing one overlong lineage from masquerading as a systemic
    provider outage.
    """

    ordered: list[tuple[int, str, str | None]] = []
    for unit in state.work_units:
        for action in unit.actions:
            if action.slot["role"] != Role.TARGET.value:
                continue
            result = state.target_results.get(action.action_id)
            if result is not None and result.usable:
                ordered.append((result.terminal_sequence, "success", None))
                continue
            terminal_attempt = next(
                (
                    attempt
                    for attempt in reversed(action.attempts)
                    if attempt.terminal_event is not None
                ),
                None,
            )
            if terminal_attempt is None or terminal_attempt.terminal_event is None:
                continue
            terminal = terminal_attempt.terminal_event
            if (
                terminal["type"] != "model_call_failed"
                or terminal["payload"]["error_kind"] != "context_limit"
            ):
                continue
            lineage = terminal_attempt.intended["lineage"]
            fresh_root_id = (
                lineage["conversation_root_id"] or lineage["execution_id"]
            )
            if fresh_root_id is None:
                raise EngineError(
                    "target context-limit outcome lacks fresh-root lineage"
                )
            ordered.append((terminal["sequence"], "context_limit", fresh_root_id))

    affected_roots: list[str] = []
    threshold = int(DECISION_CONSTANTS["provider_logical_failure_circuit"])
    for sequence, status, fresh_root_id in sorted(ordered):
        if sequence <= reset_after:
            continue
        if status == "success":
            affected_roots.clear()
            continue
        assert fresh_root_id is not None
        affected_roots.append(fresh_root_id)
        del affected_roots[:-threshold]
        if (
            len(affected_roots) == threshold
            and len(set(affected_roots)) == threshold
        ):
            return sequence
    return None


def _provider_circuit_resets(state: RunState) -> dict[str, int]:
    resets: dict[str, int] = {}
    prior_terminal: Mapping[str, Any] | None = None
    for event in state.events:
        if event["type"] == "run_terminated":
            prior_terminal = event
            continue
        if (
            event["type"] != "run_resumed"
            or event["payload"]["prior_terminal_reason"]
            != RunTerminalReason.SYSTEMIC_PROVIDER_FAILURE.value
            or prior_terminal is None
        ):
            continue
        detail = str(prior_terminal["payload"]["detail"])
        role = next(
            (item.value for item in Role if detail.startswith(item.value + "_")),
            None,
        )
        if role is not None:
            resets[role] = event["sequence"]
    return resets


def _systemic_provider_failure_detail(
    context: RunContext, state: RunState
) -> str | None:
    resets = _provider_circuit_resets(state)
    circuit = reduce_provider_circuit(
        _provider_circuit_outcomes(context, state),
        reset_after=resets,
        threshold=int(DECISION_CONSTANTS["provider_logical_failure_circuit"]),
    )
    context_limit_sequence = _target_context_limit_trip_sequence(
        state, reset_after=resets.get(Role.TARGET.value, 0)
    )
    if context_limit_sequence is not None and (
        circuit.tripped_sequence is None
        or context_limit_sequence <= circuit.tripped_sequence
    ):
        return "target_provider_logical_failure_circuit"
    if circuit.tripped_role is None:
        return None
    if circuit.tripped_route_key is not None:
        return f"{circuit.tripped_role}_provider_route_failure_circuit:{circuit.tripped_route_key}"
    return f"{circuit.tripped_role}_provider_logical_failure_circuit"


def _raise_for_reduced_provider_state(
    context: RunContext, state: RunState
) -> None:
    if state.target_identity_drift:
        raise _StopRun(
            RunTerminalReason.PROVIDER_IDENTITY_DRIFT,
            "effective_target_model_identity_changed",
            resumable=False,
        )
    detail = _systemic_provider_failure_detail(context, state)
    if detail is not None:
        raise _StopRun(
            RunTerminalReason.SYSTEMIC_PROVIDER_FAILURE,
            detail,
            resumable=True,
        )


def _profile_for_route(context: RunContext, route: Mapping[str, Any]) -> ModelProfile:
    slot = route["configured_slot"]
    mapping = {
        "target": context.config.target,
        "primary": context.config.judges.primary,
        "secondary_a": context.config.judges.secondary_a,
        "secondary_b": context.config.judges.secondary_b,
        "replacement": context.config.judges.replacement,
        "selector_primary": context.config.selector.primary(),
        "selector_fallback": context.config.selector.fallback(),
        "perturbator_primary": context.config.perturbator.primary(),
        "perturbator_fallback": context.config.perturbator.fallback(),
    }
    try:
        return mapping[slot]
    except KeyError as exc:
        raise EngineError(f"unknown configured route slot: {slot}") from exc


def _candidate_for_target_action(
    context: RunContext, state: RunState, unit: Any, action: LogicalActionView
) -> CandidatePlan:
    if _is_scenario_verification(unit):
        scenario = _scenario_for_unit(context, unit)
        root, _, proposal_slot = _scenario_root_context(unit, action)
        targets = _scenario_target_actions(state, root)
        entry = CandidatePlan(
            candidate_id=entity_id("pc", [context.run_id, scenario.scientific_key]),
            scientific_key=scenario.scientific_key,
            prompt=scenario.entry_prompt,
            descriptors=descriptors_dict(scenario.recipe.measurement_cell),
            origin="scenario_verification_entry",
            probe_id=None,
        )
        if action.action_id == targets[0].action_id:
            return entry
        try:
            target_index = next(
                index
                for index, target in enumerate(targets)
                if target.action_id == action.action_id
            )
        except StopIteration as exc:
            raise EngineError("scenario target is outside its frozen root")
        if target_index == 0:
            return entry
        selected = _scenario_selected_realization(
            context, state, unit, action
        )
        if selected is None or selected.review.reviewed_descriptors is None:
            raise EngineError("scenario terminal target lacks its aligned anchor")
        parent_action = targets[target_index - 1]
        parent_candidate = _candidate_for_target_action(
            context, state, unit, parent_action
        )
        lineage = _conversation_lineage_identity(
            context, state, unit, parent_action.action_id, parent_candidate
        )
        prompt = selected.proposal.user_turn or ""
        key = _stateful_candidate_scientific_key(
            context, lineage["node_scientific_key"], prompt
        )
        binding = _scenario_binding_result(
            context,
            state,
            unit,
            _scenario_binding_action(state, unit, root, action),
        ).value
        if not isinstance(binding, TemplateBindingResult) or binding.contract is None:
            raise EngineError("scenario terminal target lacks its bound strategy")
        contract = binding.contract
        bound_key = scientific_key(
            [
                lineage["node_scientific_key"],
                "bound_strategy",
                contract.content_hash,
            ]
        )
        return CandidatePlan(
            candidate_id=entity_id("pc", [context.run_id, key]),
            scientific_key=key,
            prompt=prompt,
            descriptors=dict(selected.review.reviewed_descriptors),
            origin="scenario_verification_terminal",
            probe_id=None,
            parent_node_id=lineage["node_id"],
            parent_node_scientific_key=lineage["node_scientific_key"],
            strategy_id=entity_id("st", [context.run_id, bound_key]),
            proposal_slot_scientific_key=proposal_slot.scientific_key,
        )
    if unit.unit["kind"] in {"initial_observation", "screening", "verification"}:
        candidate = _candidate_by_id(context, unit.unit["subject_id"])
        if candidate is None and unit.unit["design"].get("candidate_id"):
            candidate = _candidate_by_id(context, unit.unit["design"]["candidate_id"])
        if candidate is None:
            raise EngineError("frozen target action has no candidate")
        return candidate
    if unit.unit["kind"] == "map_exploration":
        guard = action.slot["guard"]
        review_action = next(
            item for item in unit.actions if item.slot["purpose"] == ActionPurpose.PROBE_REVIEW.value
        )
        reviews = _map_review_result(context, state, unit, review_action).value
        proposal_action = next(
            item for item in unit.actions if item.slot["purpose"] == ActionPurpose.PERTURBATOR_PROPOSAL.value
        )
        batch = _map_proposal_result(context, state, unit, proposal_action).value
        if not isinstance(batch, ProposalBatch) or not isinstance(reviews, tuple):
            raise EngineError("map target action lacks durable proposal/review")
        slot = next(
            item for item in _proposal_slots(unit) if item.scientific_key == guard["proposal_slot_scientific_key"]
        )
        proposal = next(item for item in batch.items if item.slot_id == slot.slot_id)
        review = next(item for item in reviews if item.review_item_id == guard["review_item_id"])
        selected = select_ordinary_realization(proposal, review)
        if selected is None or selected.review.reviewed_descriptors is None:
            raise EngineError("guard admitted a nondispatchable ordinary proposal")
        prompt = selected.proposal.user_turn or ""
        key = _candidate_scientific_key(context.resources, prompt)
        candidate = context.candidates.get(key)
        descriptors = dict(selected.review.reviewed_descriptors)
        if candidate is None:
            candidate = CandidatePlan(
                candidate_id=entity_id("pc", [context.run_id, key]),
                scientific_key=key,
                prompt=prompt,
                descriptors=descriptors,
                origin="ordinary_map",
                probe_id=None,
                parent_node_id=None,
                parent_node_scientific_key=None,
                proposal_slot_scientific_key=slot.scientific_key,
            )
            context.candidates[key] = candidate
        elif dict(candidate.descriptors) != descriptors:
            # Generated rediscovery with a conflicting reviewed cell is a local
            # closed proposal.  The pre-dispatch ownership guard prevents a
            # target call; retaining the original candidate makes request
            # reconstruction robust without mutating its evidence ladder.
            return candidate
        return candidate
    if unit.unit["kind"] == "portfolio_round":
        resolved = _portfolio_selected_realization(context, state, unit, action)
        if resolved is None:
            raise EngineError("portfolio target lacks a selected realization")
        slot, lane, selected = resolved
        if selected is None or selected.review.reviewed_descriptors is None:
            raise EngineError("portfolio guard admitted a nondispatchable realization")
        if (
            _portfolio_stateful_dispatch_status(context, state, unit, action)
            != "dispatch"
        ):
            raise EngineError("portfolio target does not own its stateful candidate")
        _, _, parent_node_key = _portfolio_parent(context, state, unit)
        prompt = selected.proposal.user_turn or ""
        key = _stateful_candidate_scientific_key(
            context, parent_node_key, prompt
        )
        strategy_id = entity_id("st", [context.run_id, lane.strategy_scientific_key])
        candidate = context.candidates.get(key)
        descriptors = dict(selected.review.reviewed_descriptors)
        if candidate is None:
            candidate = CandidatePlan(
                candidate_id=entity_id("pc", [context.run_id, key]),
                scientific_key=key,
                prompt=prompt,
                descriptors=descriptors,
                origin="stateful_strategy",
                probe_id=None,
                parent_node_id=unit.unit["design"]["generation_parent_node_id"],
                parent_node_scientific_key=parent_node_key,
                strategy_id=strategy_id,
                proposal_slot_scientific_key=slot.scientific_key,
            )
            context.candidates[key] = candidate
        elif dict(candidate.descriptors) != descriptors:
            raise EngineError("one stateful turn received conflicting reviewed cells")
        return candidate
    if unit.unit.get("kind") in _STRATEGY_QUALIFICATION_KINDS:
        root_plan, _, _, _, proposal_slot, _ = _qualification_root_context(
            unit, action
        )
        targets = _qualification_target_actions(state, root_plan)
        if not targets:
            raise EngineError("strategy qualification root lacks an entry target")
        entry_candidate = _qualification_entry_candidate(context, state, unit)
        if action.action_id == targets[0].action_id:
            return entry_candidate
        if len(targets) != 2 or action.action_id != targets[1].action_id:
            raise EngineError("strategy qualification target is outside its root endpoint")
        selected = _qualification_selected_realization(
            context, state, unit, action
        )
        if selected is None or selected.review.reviewed_descriptors is None:
            raise EngineError("qualification terminal target lacks a selected realization")
        if proposal_slot is None:
            raise EngineError("qualification continuation target lacks its proposal slot")
        if root_plan["arm"] == "treatment" and selected.review.status != "aligned":
            raise EngineError("qualification treatment target is not strategy-aligned")
        if root_plan["arm"] == "strategy_agnostic" and selected.review.status != "valid":
            raise EngineError("qualification control target is not valid")
        entry_lineage = _conversation_lineage_identity(
            context, state, unit, targets[0].action_id, entry_candidate
        )
        prompt = selected.proposal.user_turn or ""
        key = _stateful_candidate_scientific_key(
            context, entry_lineage["node_scientific_key"], prompt
        )
        descriptors = dict(selected.review.reviewed_descriptors)
        strategy_id = None
        if root_plan["arm"] == "treatment":
            binding = _qualification_binding_result(
                context,
                state,
                unit,
                _qualification_binding_action(state, root_plan),
            ).value
            if (
                not isinstance(binding, TemplateBindingResult)
                or binding.contract is None
            ):
                raise EngineError("qualification treatment lacks its bound strategy")
            contract = binding.contract
            bound_key = scientific_key(
                [
                    entry_lineage["node_scientific_key"],
                    "bound_strategy",
                    contract.content_hash,
                ]
            )
            strategy_id = entity_id("st", [context.run_id, bound_key])
        candidate = context.candidates.get(key)
        if candidate is None:
            candidate = CandidatePlan(
                candidate_id=entity_id("pc", [context.run_id, key]),
                scientific_key=key,
                prompt=prompt,
                descriptors=descriptors,
                origin=(
                    "strategy_qualification_treatment"
                    if root_plan["arm"] == "treatment"
                    else "strategy_qualification_control"
                ),
                probe_id=None,
                parent_node_id=entry_lineage["node_id"],
                parent_node_scientific_key=entry_lineage[
                    "node_scientific_key"
                ],
                strategy_id=strategy_id,
                proposal_slot_scientific_key=proposal_slot.scientific_key,
            )
            context.candidates[key] = candidate
        elif dict(candidate.descriptors) != descriptors:
            raise EngineError("qualification realization received conflicting descriptors")
        return candidate
    raise EngineError(f"target request construction is unsupported for {unit.unit['kind']}")


def _lineage_for_action(
    context: RunContext,
    state: RunState,
    unit: Any,
    action: LogicalActionView,
) -> dict[str, Any]:
    panel = _panel_for_action(unit, action.action_id)
    target_action_id = panel["target_action_id"] if panel is not None else action.action_id
    target_action = state.logical_actions.get(target_action_id)
    execution = next(
        (
            item
            for item in unit.unit["execution_slots"]
            if target_action is not None
            and item["execution_scientific_key"] == target_action.slot["owner_position_scientific_key"]
        ),
        None,
    )
    candidate: CandidatePlan | None = None
    if target_action is not None:
        try:
            candidate = _candidate_for_target_action(context, state, unit, target_action)
        except EngineError:
            candidate = _candidate_by_id(context, unit.unit.get("subject_id"))
    logical_member_slot = None
    if panel is not None and action.slot["role"] == Role.JUDGE.value:
        logical_member_slot = _normal_slot_for_action(panel, action.action_id)
        if logical_member_slot is None:
            logical_member_slot = _replacement_slot(context, state, unit, panel)
    proposal_key = None
    review_item_id = None
    guard = action.slot["guard"]
    if guard["kind"] == "if_dispatchable_probe":
        proposal_key = guard["proposal_slot_scientific_key"]
        review_item_id = guard["review_item_id"]
    conversation_root_id = unit.unit["design"].get("conversation_root_id")
    if conversation_root_id is None and execution is not None:
        # A fresh execution always has a distinct root even for a one-turn
        # standalone sample.  It is derived from the preallocated execution,
        # never from dispatch time or a physical retry.
        conversation_root_id = entity_id(
            "cr", [execution["execution_id"], "conversation_root"]
        )
    return {
        "execution_id": None if execution is None else execution["execution_id"],
        "conversation_root_id": conversation_root_id,
        "parent_node_id": None if candidate is None else candidate.parent_node_id,
        "candidate_id": None if candidate is None else candidate.candidate_id,
        "panel_id": None if panel is None or action.slot["role"] != Role.JUDGE.value else panel["panel_id"],
        "logical_member_slot": logical_member_slot,
        "strategy_id": None if candidate is None else candidate.strategy_id,
        "proposal_slot_scientific_key": proposal_key,
        "review_item_id": review_item_id,
    }


def _qualification_binding_request(
    context: RunContext, state: RunState, unit: Any, action: LogicalActionView
) -> Mapping[str, Any]:
    root_plan, execution, _, _, _, _ = _qualification_root_context(unit, action)
    if root_plan["arm"] != "treatment":
        raise EngineError("control root requested a StrategyTemplate binding")
    conversation, turn_ref, turn_text, _ = _qualification_fresh_branch(
        context, state, unit, action
    )
    return {
        "binding": "b0",
        "objective_definition": _strategy_goal_context(context),
        "fresh_conversation": conversation,
        "current_assessment": _portfolio_current_assessment(state, ({"turn_ref": turn_ref, "text": turn_text},)),
        "evidence_catalog": build_strategy_evidence_catalog(
            ((turn_ref, turn_text),)
        )[0],
        "strategy_template": _qualification_template(
            context, state, unit
        ).to_dict(),
    }


def _qualification_correction_review(
    context: RunContext,
    state: RunState,
    unit: Any,
    root_plan: Mapping[str, Any],
    action: LogicalActionView,
) -> Mapping[str, Any] | None:
    root_actions = _qualification_root_actions(state, root_plan)
    proposal_actions = tuple(
        item
        for item in root_actions
        if item.slot["purpose"] == ActionPurpose.PERTURBATOR_PROPOSAL.value
    )
    if len(proposal_actions) != 2:
        raise EngineError("qualification generation pair is incomplete")
    if action.action_id == proposal_actions[0].action_id:
        return None
    if action.action_id != proposal_actions[1].action_id:
        raise EngineError("qualification generator is outside its root pair")
    initial_review_action = next(
        item
        for item in root_actions
        if item.slot["purpose"] == ActionPurpose.PROBE_REVIEW.value
    )
    reviews = _qualification_review_result(
        context, state, unit, initial_review_action
    ).value
    if not isinstance(reviews, tuple) or len(reviews) != 1:
        raise EngineError("qualification correction lacks one exact initial review")
    review = reviews[0]
    return {
        "status": review.status,
        "reason": review.reason,
    }


def _qualification_perturbator_request(
    context: RunContext, state: RunState, unit: Any, action: LogicalActionView
) -> Mapping[str, Any]:
    root_plan, _, _, _, proposal_slot, intent = _qualification_root_context(
        unit, action
    )
    if proposal_slot is None or intent is None:
        raise EngineError("qualification generator lacks its frozen slot and intent")
    conversation, _, _, _ = _qualification_fresh_branch(
        context, state, unit, action
    )
    dimensions = (
        unit.unit["design"]["comparison_dimensions"]
        if unit.unit["kind"] == "strategy_development"
        else unit.unit["design"]["sealed_holdout_plan"][
            "comparison_dimensions"
        ]
    )
    correction_review = _qualification_correction_review(
        context, state, unit, root_plan, action
    )
    descriptor_vocabulary = {
        name: list(context.objective.descriptor_values[name])
        for name in context.objective.descriptor_names
    }
    if root_plan["arm"] == "strategy_agnostic":
        if correction_review is None:
            instruction = (
                "Produce one coherent philosophical continuation or clarification."
            )
        else:
            root_actions = _qualification_root_actions(state, root_plan)
            original_action = next(
                item
                for item in root_actions
                if item.slot["purpose"]
                == ActionPurpose.PERTURBATOR_PROPOSAL.value
            )
            original_batch = _qualification_proposal_result(
                context, state, unit, original_action
            ).value
            original = (
                None
                if not isinstance(original_batch, ProposalBatch)
                else next(iter(original_batch.proposed_items), None)
            )
            if original is None:
                raise EngineError(
                    "strategy-agnostic correction lacks its original realization"
                )
            instruction = (
                "Correct this prior realization using only the supplied review "
                "reason and frozen comparison dimensions: "
                + canonical_json(
                    {
                        "prior_realization": original.to_dict(),
                        "review_reason": correction_review["reason"],
                    }
                )
            )
        return {
            "mode": "strategy_agnostic",
            "proposal_slots": [
                {
                    "slot": proposal_slot_aliases((proposal_slot,))[
                        proposal_slot.slot_id
                    ],
                    "intent": intent,
                }
            ],
            "fresh_branch_transcript": conversation,
            "objective_id": context.objective.id,
            "scientific_question": context.objective.objective[
                "scientific_question"
            ],
            "descriptor_vocabulary": descriptor_vocabulary,
            "surface_intent": intent,
            "comparison_dimensions": list(dimensions),
            "instruction": instruction,
        }

    binding = _qualification_binding_result(
        context,
        state,
        unit,
        _qualification_binding_action(state, root_plan),
    ).value
    if not isinstance(binding, TemplateBindingResult) or binding.contract is None:
        raise EngineError("treatment generator lacks an applicable bound strategy")
    contract = binding.contract
    return {
        "mode": "strategy",
        "descriptor_vocabulary": descriptor_vocabulary,
        "conversation": conversation,
        "proposal_slots": [
            {
                "slot": proposal_slot_aliases((proposal_slot,))[
                    proposal_slot.slot_id
                ],
                "intent": intent,
            }
        ],
        "surface_intent": intent,
        "comparison_dimensions": list(dimensions),
        "bound_strategy": contract.to_dict(),
        "correction_review": correction_review,
    }


def _qualification_probe_review_request(
    context: RunContext, state: RunState, unit: Any, action: LogicalActionView
) -> Mapping[str, Any]:
    root_plan, _, _, _, proposal_slot, _ = _qualification_root_context(unit, action)
    if proposal_slot is None:
        raise EngineError("qualification probe review lacks its realization slot")
    root_actions = _qualification_root_actions(state, root_plan)
    review_actions = tuple(
        item
        for item in root_actions
        if item.slot["purpose"] == ActionPurpose.PROBE_REVIEW.value
    )
    proposal_actions = tuple(
        item
        for item in root_actions
        if item.slot["purpose"] == ActionPurpose.PERTURBATOR_PROPOSAL.value
    )
    if len(review_actions) != 2 or len(proposal_actions) != 2:
        raise EngineError("qualification review graph is incomplete")
    review_index = 0 if action.action_id == review_actions[0].action_id else 1
    batch = _qualification_proposal_result(
        context, state, unit, proposal_actions[review_index]
    ).value
    if not isinstance(batch, ProposalBatch):
        raise EngineError("qualification review lacks its proposal batch")
    proposal = next((item for item in batch.proposed_items), None)
    if proposal is None:
        return {"mode": "ordinary", "items": []}
    conversation, _, _, _ = _qualification_fresh_branch(
        context, state, unit, action
    )
    review_id = make_review_item_id(
        unit.unit["unit_scientific_key"],
        "probe",
        proposal_slot.scientific_key,
    )
    expectation = ReviewExpectation(review_id, proposal)
    item: dict[str, Any] = {
        "review_alias": probe_review_aliases((expectation,))[review_id],
        "proposal": {"user_turn": proposal.user_turn},
    }
    mode = "ordinary"
    request: dict[str, Any] = {
        "objective_definition": _strategy_goal_context(context),
        "descriptor_vocabulary": {
            name: list(context.objective.descriptor_values[name])
            for name in context.objective.descriptor_names
        },
        "mode": mode,
        "conversation": conversation,
        "items": [item],
    }
    if root_plan["arm"] == "treatment":
        binding = _qualification_binding_result(
            context,
            state,
            unit,
            _qualification_binding_action(state, root_plan),
        ).value
        if not isinstance(binding, TemplateBindingResult) or binding.contract is None:
            raise EngineError("treatment review lacks its fresh bound strategy")
        contract = binding.contract
        mode = "strategy"
        request["mode"] = mode
        item["contract"] = contract.to_dict()
    return request


def _scenario_binding_request(
    context: RunContext, state: RunState, unit: Any, action: LogicalActionView
) -> Mapping[str, Any]:
    scenario = _scenario_for_unit(context, unit)
    source = scenario_preflight_source(unit, action.action_id, state.conversation_nodes, state.target_observations)
    if source is None:
        root, _, _ = _scenario_root_context(unit, action)
        step_index = _scenario_step_index(unit, root, action)
        conversation, turn_ref, turn_text, _ = _scenario_fresh_branch(state, unit, action)
    else:
        step_index, observation = source
        conversation = [message.to_dict() for message in observation.messages]
        conversation.append({"role": "assistant", "content": observation.response})
        turn_ref, turn_text = observation.scientific_key, observation.response
    template = scenario.recipe.templates[step_index]
    return {
        "binding": "b0",
        "objective_definition": _strategy_goal_context(context),
        "fresh_conversation": conversation,
        "current_assessment": _portfolio_current_assessment(state, ({"turn_ref": turn_ref, "text": turn_text},)),
        "evidence_catalog": build_strategy_evidence_catalog(
            ((turn_ref, turn_text),)
        )[0],
        "strategy_template": template.to_dict(),
    }


def _scenario_correction_review(
    context: RunContext,
    state: RunState,
    unit: Any,
    root: Mapping[str, Any],
    action: LogicalActionView,
) -> Mapping[str, Any] | None:
    actions = _scenario_root_actions(state, root)
    step_index = _scenario_step_index(unit, root, action)
    all_proposals = tuple(
        item
        for item in actions
        if item.slot["purpose"] == ActionPurpose.PERTURBATOR_PROPOSAL.value
    )
    all_reviews = tuple(
        item
        for item in actions
        if item.slot["purpose"] == ActionPurpose.PROBE_REVIEW.value
    )
    proposals = all_proposals[2 * step_index : 2 * step_index + 2]
    reviews_for_step = all_reviews[2 * step_index : 2 * step_index + 2]
    if action.action_id == proposals[0].action_id:
        return None
    if action.action_id != proposals[1].action_id:
        raise EngineError("scenario generator is outside its bounded pair")
    initial_review = reviews_for_step[0]
    reviews = _scenario_review_result(
        context, state, unit, initial_review
    ).value
    if not isinstance(reviews, tuple) or len(reviews) != 1:
        raise EngineError("scenario correction lacks one initial review")
    review = reviews[0]
    return {
        "status": review.status,
        "reason": review.reason,
    }


def _scenario_perturbator_request(
    context: RunContext, state: RunState, unit: Any, action: LogicalActionView
) -> Mapping[str, Any]:
    scenario = _scenario_for_unit(context, unit)
    root, _, proposal_slot = _scenario_root_context(unit, action)
    step_index = _scenario_step_index(unit, root, action)
    conversation, _, _, _ = _scenario_fresh_branch(state, unit, action)
    binding = _scenario_binding_result(
        context, state, unit, _scenario_binding_action(state, unit, root, action)
    ).value
    if not isinstance(binding, TemplateBindingResult) or binding.contract is None:
        raise EngineError("scenario generator lacks an applicable bound strategy")
    contract = binding.contract
    return {
        "mode": "strategy",
        "descriptor_vocabulary": {
            name: list(context.objective.descriptor_values[name])
            for name in context.objective.descriptor_names
        },
        "conversation": conversation,
        "proposal_slots": [
            {
                "slot": proposal_slot_aliases((proposal_slot,))[
                    proposal_slot.slot_id
                ],
                "intent": "anchor",
            }
        ],
        "surface_intent": "anchor",
        "comparison_dimensions": [
            item.dimension for item in scenario.recipe.templates[step_index].vary
        ],
        "bound_strategy": contract.to_dict(),
        "correction_review": _scenario_correction_review(
            context, state, unit, root, action
        ),
    }


def _scenario_probe_review_request(
    context: RunContext, state: RunState, unit: Any, action: LogicalActionView
) -> Mapping[str, Any]:
    root, _, proposal_slot = _scenario_root_context(unit, action)
    step_index = _scenario_step_index(unit, root, action)
    actions = _scenario_root_actions(state, root)
    all_reviews = tuple(
        item
        for item in actions
        if item.slot["purpose"] == ActionPurpose.PROBE_REVIEW.value
    )
    all_proposals = tuple(
        item
        for item in actions
        if item.slot["purpose"] == ActionPurpose.PERTURBATOR_PROPOSAL.value
    )
    reviews = all_reviews[2 * step_index : 2 * step_index + 2]
    proposals = all_proposals[2 * step_index : 2 * step_index + 2]
    index = 0 if action.action_id == reviews[0].action_id else 1
    batch = _scenario_proposal_result(
        context, state, unit, proposals[index]
    ).value
    if not isinstance(batch, ProposalBatch):
        raise EngineError("scenario review request lacks its proposal batch")
    proposal = next(iter(batch.proposed_items), None)
    conversation, _, _, _ = _scenario_fresh_branch(
        state, unit, action
    )
    items: list[dict[str, Any]] = []
    if proposal is not None:
        binding = _scenario_binding_result(
            context, state, unit, _scenario_binding_action(state, unit, root, action)
        ).value
        if not isinstance(binding, TemplateBindingResult) or binding.contract is None:
            raise EngineError("scenario review request lacks its bound contract")
        contract = binding.contract
        review_id = make_review_item_id(
            unit.unit["unit_scientific_key"],
            "probe",
            proposal_slot.scientific_key,
        )
        expectation = ReviewExpectation(review_id, proposal)
        items.append(
            {
                "review_alias": probe_review_aliases((expectation,))[review_id],
                "proposal": {"user_turn": proposal.user_turn},
                "contract": contract.to_dict(),
            }
        )
    return {
        "objective_definition": _strategy_goal_context(context),
        "descriptor_vocabulary": {
            name: list(context.objective.descriptor_values[name])
            for name in context.objective.descriptor_names
        },
        "mode": "strategy",
        "conversation": conversation,
        "items": items,
    }


def _messages_for_action(
    context: RunContext,
    state: RunState,
    unit: Any,
    action: LogicalActionView,
    *,
    repair_errors: Sequence[str] = (),
) -> tuple[ChatMessage, ...]:
    role = action.slot["role"]
    purpose = action.slot["purpose"]
    schema = _schema_for_action(context, state, unit, action)
    schema_text = canonical_json(schema[0]) if schema is not None else "null"
    repair_context: dict[str, Any] | None = None
    if repair_errors:
        last = action.attempts[-1].terminal_event
        invalid_output = None if last is None else last["payload"].get("text")
        repair_context = {
            "invalid_output": invalid_output,
            "structural_errors": list(repair_errors),
        }
    repair_context_text = canonical_json(repair_context)
    if role == Role.TARGET.value:
        candidate = _candidate_for_target_action(context, state, unit, action)
        if _is_scenario_verification(unit):
            root, _, _ = _scenario_root_context(unit, action)
            targets = _scenario_target_actions(state, root)
            if any(action.action_id == target.action_id for target in targets[1:]):
                conversation, _, _, _ = _scenario_fresh_branch(
                    state, unit, action
                )
                return tuple(
                    [
                        *(
                            ChatMessage(item["role"], item["content"])
                            for item in conversation
                        ),
                        ChatMessage("user", candidate.prompt),
                    ]
                )
        if unit.unit.get("kind") in _STRATEGY_QUALIFICATION_KINDS:
            root_plan, _, _, _, _, _ = _qualification_root_context(unit, action)
            targets = _qualification_target_actions(state, root_plan)
            if len(targets) == 2 and action.action_id == targets[1].action_id:
                conversation, _, _, _ = _qualification_fresh_branch(
                    context, state, unit, action
                )
                return tuple(
                    [
                        *(ChatMessage(item["role"], item["content"]) for item in conversation),
                        ChatMessage("user", candidate.prompt),
                    ]
                )
        messages: list[ChatMessage] = []
        if context.config.target.system_prompt:
            messages.append(ChatMessage("system", context.config.target.system_prompt))
        if unit.unit["kind"] == "portfolio_round":
            conversation, _ = _portfolio_branch_context(context, state, unit)
            messages.extend(
                ChatMessage(item["role"], item["content"])
                for item in conversation
            )
        messages.append(ChatMessage("user", candidate.prompt))
        return tuple(messages)
    if role == Role.JUDGE.value:
        panel = _panel_for_action(unit, action.action_id)
        if panel is None or panel["target_action_id"] not in state.target_results:
            raise EngineError("judge request lacks durable target response")
        target = state.target_results[panel["target_action_id"]]
        target_action = state.logical_actions[panel["target_action_id"]]
        candidate = _candidate_for_target_action(context, state, unit, target_action)
        if target_action.attempts:
            conversation = list(target_action.attempts[0].intended["messages"])
        elif unit.unit["kind"] == "portfolio_round":
            conversation, _ = _portfolio_branch_context(
                context, state, unit
            )
            conversation.append({"role": "user", "content": candidate.prompt})
        else:
            conversation = [{"role": "user", "content": candidate.prompt}]
        target_text = target.text
        member_slot = _normal_slot_for_action(panel, action.action_id) or _replacement_slot(context, state, unit, panel) or "primary"
        user = _render_template(
            context.resources.templates["judge_user.md"],
            {
                "OBJECTIVE_JSON": canonical_json(judge_objective_context(context.objective.objective)),
                "CONVERSATION_JSON": canonical_json(conversation),
                "JUDGE_FOCUS_JSON": canonical_json({"primary": "whole_argument", "secondary_a": "premise_ownership", "secondary_b": "commitment_and_counterevidence"}[member_slot]),
                "CURRENT_TARGET_EVIDENCE_JSON": canonical_json(
                    judge_evidence_catalog(target_text)
                ),
                "OUTPUT_SCHEMA_JSON": schema_text,
                "SCHEMA_REPAIR_CONTEXT_JSON": repair_context_text,
            },
        )
        messages = [
            ChatMessage("system", context.resources.templates["judge_system.md"]),
            ChatMessage("user", user),
        ]
    elif purpose == ActionPurpose.PERTURBATOR_PROPOSAL.value:
        if _is_scenario_verification(unit):
            request = dict(
                _scenario_perturbator_request(context, state, unit, action)
            )
            semantic_design = None
        elif unit.unit.get("kind") in _STRATEGY_QUALIFICATION_KINDS:
            request = dict(
                _qualification_perturbator_request(
                    context, state, unit, action
                )
            )
            semantic_design = None
        elif unit.unit["kind"] == "map_exploration":
            semantic_design = {
                "proposal_count": unit.unit["design"]["proposal_count"],
                "under_covered_cells": unit.unit["design"]["under_covered_cells"],
                "allowed_dimensions": unit.unit["design"]["allowed_dimensions"],
            }
        else:
            semantic_design = {
                key: value
                for key, value in unit.unit["design"].items()
                if key
                not in {
                    "conversation_root_id",
                    "generation_parent_node_id",
                    "active_strategy_ids",
                }
            }
        if (
            unit.unit["kind"] not in _STRATEGY_QUALIFICATION_KINDS
            and not _is_scenario_verification(unit)
        ):
            request_slots = _proposal_slots(unit)
            slot_aliases = proposal_slot_aliases(request_slots)
            request = {
                "objective": context.objective.objective,
                "mode": "ordinary" if unit.unit["kind"] == "map_exploration" else "strategy",
                "work_unit": semantic_design,
                "proposal_slots": [
                    {"slot": slot_aliases[item.slot_id]}
                    for item in request_slots
                ],
                "parent": None,
            }
        if unit.unit["kind"] == "map_exploration":
            parent = _map_parent_candidate(context, unit)
            request["parent"] = None if parent is None else {
                "prompt": parent.prompt,
                "descriptors": dict(parent.descriptors),
            }
        elif unit.unit["kind"] == "portfolio_round":
            lane = _portfolio_lane_for_action(context, state, unit, action)
            if lane is None:
                raise EngineError("strategy generator action has no bound lane")
            lane_slots = _portfolio_slots_for_lane(context, state, unit, lane)
            assignment_by_key = {
                item.proposal_slot_scientific_key: item
                for item in _portfolio_assignments(context, state, unit)
            }
            conversation, _ = _portfolio_branch_context(context, state, unit)
            slot_aliases = proposal_slot_aliases(lane_slots)
            request = {
                "mode": "strategy",
                "objective_definition": _strategy_goal_context(context),
                "descriptor_vocabulary": {
                    name: list(context.objective.descriptor_values[name])
                    for name in context.objective.descriptor_names
                },
                "conversation": conversation,
                "proposal_slots": [
                    {
                        "slot": slot_aliases[item.slot_id],
                        "intent": assignment_by_key[item.scientific_key].intent,
                    }
                    for item in lane_slots
                ],
                "allowed_dimensions": [
                    item.dimension for item in lane.item.contract.vary
                ],
                "bound_strategy": lane.item.contract.to_dict(),
            }
        if unit.unit["kind"] in {"portfolio_round", "map_exploration"}:
            request["generation_brief"] = _generation_brief(context, state, unit)
            request["prior_attempts"] = _prior_attempts(context, state, unit, canonical_json(request))
        user = _render_template(
            context.resources.templates["perturbator_user.md"],
            {
                "PERTURBATOR_REQUEST_JSON": canonical_json(request),
                "OUTPUT_SCHEMA_JSON": schema_text,
                "SCHEMA_REPAIR_CONTEXT_JSON": repair_context_text,
            },
        )
        messages = [
            ChatMessage("system", context.resources.templates["perturbator_system.md"]),
            ChatMessage("user", user),
        ]
    elif purpose == ActionPurpose.PROBE_REVIEW.value:
        if _is_scenario_verification(unit):
            request = dict(
                _scenario_probe_review_request(context, state, unit, action)
            )
        elif unit.unit.get("kind") in _STRATEGY_QUALIFICATION_KINDS:
            request = dict(
                _qualification_probe_review_request(
                    context, state, unit, action
                )
            )
        else:
            request = {
                "objective": context.objective.objective,
                "mode": "ordinary" if unit.unit["kind"] == "map_exploration" else "strategy",
                "items": [],
            }
        if unit.unit["kind"] == "map_exploration":
            proposal_action = next(
                item for item in unit.actions if item.slot["purpose"] == ActionPurpose.PERTURBATOR_PROPOSAL.value
            )
            batch = _map_proposal_result(context, state, unit, proposal_action).value
            if isinstance(batch, ProposalBatch):
                by_id = {item.slot_id: item for item in batch.proposed_items}
                expectations = [
                    ReviewExpectation(
                        make_review_item_id(
                            unit.unit["unit_scientific_key"],
                            "probe",
                            slot.scientific_key,
                        ),
                        by_id[slot.slot_id],
                    )
                    for slot in _proposal_slots(unit)
                    if slot.slot_id in by_id
                ]
                aliases = probe_review_aliases(expectations)
                for expected in expectations:
                    request["items"].append(
                        {
                            "review_alias": aliases[expected.review_item_id],
                            "proposal": {
                                "user_turn": expected.proposal.user_turn
                            },
                        }
                    )
        elif unit.unit["kind"] == "portfolio_round":
            conversation, _ = _portfolio_branch_context(
                context, state, unit
            )
            request = {
                "objective_definition": _strategy_goal_context(context),
                "descriptor_vocabulary": {
                    name: list(context.objective.descriptor_values[name])
                    for name in context.objective.descriptor_names
                },
                "mode": "strategy",
                "conversation": conversation,
                "items": [],
            }
            proposal_actions = [
                item
                for item in unit.actions[: action.slot["ordinal"]]
                if item.slot["purpose"] == ActionPurpose.PERTURBATOR_PROPOSAL.value
            ]
            assignments = {
                item.proposal_slot_scientific_key: item
                for item in _portfolio_assignments(context, state, unit)
            }
            slots = {item.slot_id: item for item in _proposal_slots(unit)}
            lanes = {
                lane.strategy_slot.scientific_key: lane
                for lane in _portfolio_lanes(context, state, unit)
            }
            for proposal_action in proposal_actions:
                batch = _portfolio_proposal_result(
                    context, state, unit, proposal_action
                ).value
                if not isinstance(batch, ProposalBatch):
                    continue
                for proposal in batch.proposed_items:
                    slot = slots.get(proposal.slot_id)
                    if slot is None or slot.scientific_key not in assignments:
                        continue
                    assignment = assignments[slot.scientific_key]
                    lane = lanes.get(assignment.strategy_slot_scientific_key)
                    if lane is None:
                        continue
                    review_id = make_review_item_id(
                        unit.unit["unit_scientific_key"],
                        "probe",
                        slot.scientific_key,
                    )
                    request["items"].append(
                        {
                            "_review_item_id": review_id,
                            "_proposal": proposal,
                            "proposal": {"user_turn": proposal.user_turn},
                            "contract": lane.item.contract.to_dict(),
                        }
                    )
            expectations = tuple(
                ReviewExpectation(item["_review_item_id"], item["_proposal"])
                for item in request["items"]
            )
            aliases = probe_review_aliases(expectations)
            request["items"].sort(
                key=lambda item: (
                    seeded_rank(
                        context.config.random_seed,
                        [
                            "portfolio_probe_review",
                            unit.unit["unit_scientific_key"],
                            action.slot["action_scientific_key"],
                            item["_review_item_id"],
                        ],
                    ),
                    item["_review_item_id"],
                )
            )
            request["items"] = [
                {
                    "review_alias": aliases[item["_review_item_id"]],
                    "proposal": item["proposal"],
                    "contract": item["contract"],
                }
                for item in request["items"]
            ]
        if unit.unit["kind"] == "portfolio_round":
            request["generation_brief"] = _generation_brief(context, state, unit)
        if unit.unit["kind"] in {"portfolio_round", "map_exploration"}:
            request["prior_attempts"] = _prior_attempts(context, state, unit, canonical_json(request))
        user = _render_template(
            context.resources.templates["probe_review_user.md"],
            {
                "PROBE_REVIEW_REQUEST_JSON": canonical_json(request),
                "OUTPUT_SCHEMA_JSON": schema_text,
                "SCHEMA_REPAIR_CONTEXT_JSON": repair_context_text,
            },
        )
        messages = [
            ChatMessage("system", context.resources.templates["probe_review_system.md"]),
            ChatMessage("user", user),
        ]
    elif purpose == ActionPurpose.SELECTOR_PORTFOLIO.value:
        request = _portfolio_selector_request(context, state, unit, action)
        user = _render_template(
            context.resources.templates["selector_user.md"],
            {
                "SELECTOR_REQUEST_JSON": canonical_json(request),
                "OUTPUT_SCHEMA_JSON": schema_text,
                "SCHEMA_REPAIR_CONTEXT_JSON": repair_context_text,
            },
        )
        messages = [
            ChatMessage("system", context.resources.templates["selector_system.md"]),
            ChatMessage("user", user),
        ]
    elif (
        purpose == ActionPurpose.OUTCOME_REVIEW.value
        and unit.unit["kind"] == "portfolio_round"
    ):
        request, _ = _portfolio_outcome_review_request(context, state, unit)
        user = _render_template(
            context.resources.templates["outcome_review_user.md"],
            {
                "OUTCOME_REVIEW_REQUEST_JSON": canonical_json(request),
                "OUTPUT_SCHEMA_JSON": schema_text,
                "SCHEMA_REPAIR_CONTEXT_JSON": repair_context_text,
            },
        )
        messages = [
            ChatMessage(
                "system", context.resources.templates["outcome_review_system.md"]
            ),
            ChatMessage("user", user),
        ]
    elif (
        purpose == ActionPurpose.TEMPLATE_BINDING.value
        and (
            unit.unit.get("kind") in _STRATEGY_QUALIFICATION_KINDS
            or _is_scenario_verification(unit)
        )
    ):
        request = (
            _scenario_binding_request(context, state, unit, action)
            if _is_scenario_verification(unit)
            else _qualification_binding_request(context, state, unit, action)
        )
        user = _render_template(
            context.resources.templates["template_binding_user.md"],
            {
                "TEMPLATE_BINDING_REQUEST_JSON": canonical_json(request),
                "OUTPUT_SCHEMA_JSON": schema_text,
                "SCHEMA_REPAIR_CONTEXT_JSON": repair_context_text,
            },
        )
        messages = [
            ChatMessage(
                "system", context.resources.templates["template_binding_system.md"]
            ),
            ChatMessage("user", user),
        ]
    elif (
        purpose == ActionPurpose.OUTCOME_REVIEW.value
        and (
            unit.unit.get("kind") in _STRATEGY_QUALIFICATION_KINDS
            or _is_scenario_verification(unit)
        )
    ):
        request, _ = (
            _scenario_outcome_request(context, state, unit, action)
            if _is_scenario_verification(unit)
            else _qualification_outcome_request(context, state, unit, action)
        )
        user = _render_template(
            context.resources.templates["outcome_review_user.md"],
            {
                "OUTCOME_REVIEW_REQUEST_JSON": canonical_json(request),
                "OUTPUT_SCHEMA_JSON": schema_text,
                "SCHEMA_REPAIR_CONTEXT_JSON": repair_context_text,
            },
        )
        messages = [
            ChatMessage(
                "system", context.resources.templates["outcome_review_system.md"]
            ),
            ChatMessage("user", user),
        ]
    else:
        # Template-binding and outcome-review requests use their dedicated
        # sources.  Their exact keyed request is frozen in the work design.
        prefix = "template_binding" if purpose == ActionPurpose.TEMPLATE_BINDING.value else "outcome_review"
        marker = "TEMPLATE_BINDING_REQUEST_JSON" if prefix == "template_binding" else "OUTCOME_REVIEW_REQUEST_JSON"
        user = _render_template(
            context.resources.templates[prefix + "_user.md"],
            {
                marker: canonical_json(unit.unit["design"]),
                "OUTPUT_SCHEMA_JSON": schema_text,
                "SCHEMA_REPAIR_CONTEXT_JSON": repair_context_text,
            },
        )
        messages = [
            ChatMessage("system", context.resources.templates[prefix + "_system.md"]),
            ChatMessage("user", user),
        ]
    return tuple(messages)


def _request_from_intent(context: RunContext, payload: Mapping[str, Any], state: RunState) -> ChatRequest:
    action = state.logical_actions[payload["logical_action_id"]]
    schema_info = _schema_for_action(context, state, _unit_for_action(state, action.action_id), action)
    response_schema = None if schema_info is None else schema_info[0]
    request = ChatRequest(
        logical_action_id=payload["logical_action_id"],
        logical_action_scientific_key=payload["logical_action_scientific_key"],
        physical_attempt_id=payload["physical_attempt_id"],
        attempt_ordinal=payload["attempt_ordinal"],
        role=Role(payload["role"]),
        purpose=ActionPurpose(payload["purpose"]),
        route_kind=payload["route_kind"],
        role_route_key=payload["role_route_key"],
        requested_provider_route_key=payload["requested_provider_route_key"],
        messages=tuple(ChatMessage(item["role"], item["content"]) for item in payload["messages"]),
        requested_model=payload["requested_model"],
        generation_parameters=payload["generation_parameters"],
        response_mode=payload["response_mode"],
        response_schema=response_schema,
        response_schema_digest=payload["response_schema_digest"],
        timeout_seconds=payload["timeout_seconds"],
        idempotency_key=payload["idempotency_key"],
    )
    if request.semantic_request_hash != payload["semantic_request_hash"]:
        raise EngineCompatibilityError("persisted intent semantic request hash mismatch")
    if render_request_body(request) != decode_bytes(payload["request_body"], "request_body"):
        raise EngineCompatibilityError("persisted intent request bytes mismatch")
    return request


def _strict_schema_attempt_is_usable(
    context: RunContext,
    attempt: AttemptView,
    response_schema_digest: str,
) -> bool:
    if attempt.target_identity_collision:
        return False
    if attempt.intended["response_mode"] != "json_schema":
        return False
    terminal = attempt.terminal_event
    if (
        terminal is None
        or terminal["type"] != "model_call_succeeded"
        or terminal["payload"]["finish_status"] != FinishStatus.COMPLETE.value
    ):
        return False
    schema_name = next(
        (
            name
            for name in ROLE_SCHEMA_NAMES
            if context.resources.schema_digests[name] == response_schema_digest
        ),
        None,
    )
    if schema_name is None:
        raise EngineError("structured-response scope names an unknown role schema")
    try:
        value = strict_json_loads(terminal["payload"]["text"])
        schema = context.resources.schemas[schema_name]
        if schema_name == "judge_output.json":
            body = strict_json_loads(decode_bytes(attempt.intended["request_body"], "request_body"))
            schema = body["response_format"]["json_schema"]["schema"]
        validate_json_schema_instance(value, schema)
    except (DomainError, JsonSchemaValidationError):
        return False
    return True


def _response_mode_for_scope(
    context: RunContext,
    state: RunState,
    *,
    binding: RoleRouteBinding,
    response_schema_digest: str,
) -> str:
    """Return the durable structured-output mode for one exact route scope."""

    scoped = sorted(
        (
            attempt
            for attempt in state.attempts.values()
            if attempt.intended["role_route_key"] == binding.route_key
            and (attempt.intended["response_schema_digest"] == response_schema_digest
                 or (attempt.intended["role"] == Role.JUDGE.value
                     and response_schema_digest == context.resources.schema_digests["judge_output.json"]))
        ),
        key=lambda attempt: attempt.intent_event["sequence"],
    )
    frozen: str | None = None
    for attempt in scoped:
        if attempt.target_identity_collision:
            # Identity-colliding role output is evidence for neither response
            # format, even when its bytes satisfy the strict schema.
            continue
        mode = attempt.intended["response_mode"]
        if mode == "json_object":
            if frozen == "json_schema":
                raise EngineError(
                    "structured-response mode downgraded after usable strict success"
                )
            frozen = "json_object"
            continue
        terminal = attempt.terminal_event
        if terminal is None:
            continue
        if _strict_schema_attempt_is_usable(
            context, attempt, response_schema_digest
        ):
            if frozen == "json_object":
                raise EngineError(
                    "structured-response history conflicts with its frozen mode"
                )
            frozen = "json_schema"
        elif (
            terminal["type"] == "model_call_failed"
            and terminal["payload"]["error_kind"]
            == ProviderErrorKind.UNSUPPORTED_RESPONSE_FORMAT.value
        ):
            if frozen == "json_schema":
                raise EngineError(
                    "structured-response mode rejected after usable strict success"
                )
            frozen = "json_object"
    return frozen or "json_schema"


def _build_request(
    context: RunContext,
    state: RunState,
    unit: Any,
    action: LogicalActionView,
    route: Mapping[str, Any],
    route_kind: str,
) -> ChatRequest:
    ordinal = len(action.attempts)
    schema_info = _schema_for_action(context, state, unit, action)
    response_mode = "text" if schema_info is None else "json_schema"
    if schema_info is not None:
        # Structured-output capability is frozen across the complete
        # provider/model/route/schema scope, rather than rediscovered for each
        # logical action.  The first authoritative incompatibility fixes the
        # scope to JSON-object mode for the rest of the run.
        response_mode = _response_mode_for_scope(
            context,
            state,
            binding=context.routes[route["route_key"]],
            response_schema_digest=context.resources.schema_digests["judge_output.json"] if action.slot["role"] == Role.JUDGE.value else schema_info[1],
        )
        if route_kind == "format_downgrade":
            # The first format rejection may downgrade an initially unknown
            # scope.  A usable strict result anywhere earlier in the same
            # route/schema scope permanently forbids that transition.
            if response_mode == "json_schema":
                raise EngineError(
                    "format downgrade requested for a frozen strict-schema scope"
                )
            response_mode = "json_object"
    semantic = _semantic_result(context, state, unit, action)
    repair_errors = semantic.errors if route_kind == "schema_repair" else ()
    messages = _messages_for_action(
        context, state, unit, action, repair_errors=repair_errors
    )
    profile = _profile_for_route(context, route)
    attempt_id = entity_id(
        "pa", [action.action_id, route_kind, ordinal]
    )
    return ChatRequest(
        logical_action_id=action.action_id,
        logical_action_scientific_key=action.slot["action_scientific_key"],
        physical_attempt_id=attempt_id,
        attempt_ordinal=ordinal,
        role=Role(action.slot["role"]),
        purpose=ActionPurpose(action.slot["purpose"]),
        route_kind=route_kind,
        role_route_key=route["route_key"],
        requested_provider_route_key=route["requested_provider_route_key"],
        messages=messages,
        requested_model=route["requested_model"],
        generation_parameters=profile.generation_parameters,
        response_mode=response_mode,
        response_schema=None if schema_info is None else schema_info[0],
        response_schema_digest=None if schema_info is None else schema_info[1],
        timeout_seconds=context.config.provider.timeout_seconds,
        idempotency_key=None,
    )


def _intent_payload(
    context: RunContext,
    state: RunState,
    unit: Any,
    action: LogicalActionView,
    request: ChatRequest,
) -> dict[str, Any]:
    request_body = render_request_body(request)
    return {
        "logical_action_id": request.logical_action_id,
        "logical_action_scientific_key": request.logical_action_scientific_key,
        "physical_attempt_id": request.physical_attempt_id,
        "attempt_ordinal": request.attempt_ordinal,
        "unit_id": action.unit_id,
        "owner_position_scientific_key": action.slot["owner_position_scientific_key"],
        "role": request.role.value,
        "purpose": request.purpose.value,
        "role_route_key": request.role_route_key,
        "route_kind": request.route_kind,
        "parent_attempt_id": None if not action.attempts else action.attempts[-1].physical_attempt_id,
        "messages": [item.to_dict() for item in request.messages],
        "request_body": encode_bytes(request_body).to_dict(),
        "semantic_request_hash": request.semantic_request_hash,
        "requested_provider_route_key": request.requested_provider_route_key,
        "requested_model": request.requested_model,
        "generation_parameters": strict_json_loads(canonical_json(request.generation_parameters)),
        "response_mode": request.response_mode,
        "response_schema_digest": request.response_schema_digest,
        "timeout_seconds": request.timeout_seconds,
        "idempotency_key": request.idempotency_key,
        "lineage": _lineage_for_action(context, state, unit, action),
    }


def _dispatch_request(
    context: RunContext,
    journal: EventJournal,
    request: ChatRequest,
) -> None:
    intent_event = next(
        event
        for event in reversed(journal.events)
        if event["type"] == "model_call_intended"
        and event["payload"]["physical_attempt_id"] == request.physical_attempt_id
    )
    dispatch = journal.append(
        "model_dispatch_started",
        {
            "physical_attempt_id": request.physical_attempt_id,
            "logical_action_id": request.logical_action_id,
            "role_route_key": request.role_route_key,
            "route_kind": request.route_kind,
            "provider_route_key": request.requested_provider_route_key,
            "requested_model": request.requested_model,
            "charged_role": request.role.value,
            "charge": 1,
        },
        causes=(intent_event["event_id"],),
    )
    try:
        outcome = context.resources.provider.complete(request)
    except ScriptedFixtureMissing as exc:
        # A scripted lookup performs no external I/O. Missing predeclared data
        # is therefore a proven no-output harness failure, and the already
        # charged dispatch must still receive exactly one terminal event.
        journal.append(
            "model_call_failed",
            {
                "physical_attempt_id": request.physical_attempt_id,
                "error_kind": "invalid_request",
                "delivery_state": "known_no_output",
                "detail": "scripted_fixture_missing",
                "provider_request_id": None,
                "raw_body": None,
                "http_status": None,
                "retry_after_seconds": None,
            },
            causes=(dispatch["event_id"],),
        )
        raise _StopRun(
            RunTerminalReason.IMPOSSIBLE_INTERNAL_STATE,
            "scripted_fixture_missing",
            resumable=False,
        ) from exc
    except Exception as exc:
        # Adapter exceptions are not authoritative provider responses.  The
        # dispatch is charged and its outcome is explicitly unknowable.  It is
        # then handled by the same bounded exact-retry and reducer-derived
        # logical-failure circuit as an adapter-returned unknown outcome.
        journal.append(
            "model_call_outcome_unknown",
            {
                "physical_attempt_id": request.physical_attempt_id,
                "discovery_invocation_id": context.invocation_id,
                "error_kind": "unknown",
                "detail": "provider_adapter_exception",
                "provider_request_id": None,
                "raw_body": None,
            },
            causes=(dispatch["event_id"],),
        )
        return
    if isinstance(outcome, ProviderSuccess):
        journal.append(
            "model_call_succeeded",
            {
                "physical_attempt_id": request.physical_attempt_id,
                "raw_body": outcome.raw_body.to_dict(),
                "text": outcome.text,
                "finish_status": outcome.finish_status.value,
                "effective_model_id": outcome.effective_model_id,
                "provider_request_id": outcome.provider_request_id,
                "usage": dict(outcome.usage),
                "latency_ms": outcome.latency_ms,
                "cost": outcome.cost,
                "metadata": dict(outcome.metadata),
            },
            causes=(dispatch["event_id"],),
        )
        return
    event_type = (
        "model_call_outcome_unknown"
        if outcome.delivery_state == "outcome_unknown"
        else "model_call_failed"
    )
    if event_type == "model_call_outcome_unknown":
        payload = {
            "physical_attempt_id": request.physical_attempt_id,
            "discovery_invocation_id": context.invocation_id,
            "error_kind": outcome.error_kind.value,
            "detail": outcome.detail,
            "provider_request_id": outcome.provider_request_id,
            "raw_body": None if outcome.raw_body is None else outcome.raw_body.to_dict(),
        }
    else:
        payload = {
            "physical_attempt_id": request.physical_attempt_id,
            "error_kind": outcome.error_kind.value,
            "delivery_state": outcome.delivery_state,
            "detail": outcome.detail,
            "provider_request_id": outcome.provider_request_id,
            "raw_body": None if outcome.raw_body is None else outcome.raw_body.to_dict(),
            "http_status": outcome.http_status,
            "retry_after_seconds": outcome.retry_after_seconds,
        }
    journal.append(event_type, payload, causes=(dispatch["event_id"],))
    if (
        request.role is Role.TARGET
        and outcome.error_kind is ProviderErrorKind.UNSUPPORTED_RESPONSE_FORMAT
    ):
        raise _StopRun(
            RunTerminalReason.IMPOSSIBLE_INTERNAL_STATE,
            "target_unsupported_response_format",
            resumable=False,
        )
    if outcome.error_kind.value in {"authentication", "invalid_request"}:
        raise _StopRun(
            RunTerminalReason.SYSTEMIC_PROVIDER_FAILURE,
            f"{request.role.value}_{outcome.error_kind.value}",
            resumable=True,
        )


def _require_dispatch_budget(context: RunContext, state: RunState, request: ChatRequest) -> None:
    ceiling = context.config.provider.max_cost_usd
    if ceiling is None:
        return
    binding = context.routes[request.role_route_key]
    profile = _profile_for_route(context, binding.to_dict())
    reserve = model_cost_reservation(profile.to_dict())
    if state.cost_liability_usd + reserve > Decimal(ceiling):
        raise _StopRun(RunTerminalReason.COST_LIMIT_REACHED,
            "next_dispatch_maximum_would_exceed_frozen_dollar_ceiling", resumable=False)


def _execute_next_action_attempt(
    context: RunContext,
    journal: EventJournal,
    unit_id: str,
    action_id: str,
    state: RunState,
) -> bool:
    unit = next(item for item in state.work_units if item.unit_id == unit_id)
    action = state.logical_actions[action_id]
    if _semantic_result(context, state, unit, action).usable:
        return False
    if action.attempts and action.attempts[-1].terminal_event is None:
        pending = action.attempts[-1]
        if pending.dispatch_event is not None:
            return False
        request = _request_from_intent(context, pending.intended, state)
        _require_dispatch_budget(context, state, request)
        _delay_before_corrective_dispatch(context, action, request.route_kind)
        _dispatch_request(context, journal, request)
        return True
    plan = _next_action_route_plan(
        unit, action, _semantic_result(context, state, unit, action)
    )
    if plan is None:
        return False
    route, route_kind = plan
    request = _build_request(context, state, unit, action, route, route_kind)
    _require_dispatch_budget(context, state, request)
    intent = journal.append(
        "model_call_intended",
        _intent_payload(context, state, unit, action, request),
        causes=(unit.start_event["event_id"],),
    )
    del intent
    _delay_before_corrective_dispatch(context, action, route_kind)
    _dispatch_request(context, journal, request)
    return True


def _unit_complete(context: RunContext, state: RunState, unit: Any) -> bool:
    for action in unit.actions:
        if _semantic_result(context, state, unit, action).usable:
            continue
        dependencies_closed = all(
            _action_closed(context, state, unit, state.logical_actions[dependency])
            or not _guard_allows(context, state, unit, state.logical_actions[dependency])
            for dependency in action.slot["dependency_action_ids"]
        )
        if not dependencies_closed:
            return False
        if not _guard_allows(context, state, unit, action):
            continue
        if not _action_closed(context, state, unit, action):
            return False
    return True


def _execute_work_unit(context: RunContext, journal: EventJournal, unit_id: str) -> None:
    while True:
        state = reduce_events(journal.events, context.resources.objective)
        _raise_for_reduced_provider_state(context, state)
        unit = next(item for item in state.work_units if item.unit_id == unit_id)
        if _unit_complete(context, state, unit):
            return
        made_progress = False
        for action in unit.actions:
            # No journal append occurs while inspecting closed/guarded slots.
            # Reuse this exact checkpoint until one dispatch makes progress.
            current = state.logical_actions[action.action_id]
            if _semantic_result(context, state, unit, current).usable:
                continue
            dependencies_ready = all(
                _semantic_result(context, state, unit, state.logical_actions[item]).usable
                or _action_closed(context, state, unit, state.logical_actions[item])
                or not _guard_allows(context, state, unit, state.logical_actions[item])
                for item in current.slot["dependency_action_ids"]
            )
            if not dependencies_ready or not _guard_allows(context, state, unit, current):
                continue
            if _execute_next_action_attempt(context, journal, unit_id, current.action_id, state):
                made_progress = True
                break
        if context.stop_requested:
            raise _StopRun(
                RunTerminalReason.OPERATOR_INTERRUPTED,
                "operator_interrupt_after_durable_boundary",
                resumable=True,
            )
        if not made_progress:
            unit = next(item for item in state.work_units if item.unit_id == unit_id)
            if _unit_complete(context, state, unit):
                return
            raise _StopRun(
                RunTerminalReason.IMPOSSIBLE_INTERNAL_STATE,
                "started_work_unit_cannot_make_progress",
                resumable=False,
            )


def _initial_unit_for_candidate(state: RunState, candidate: CandidatePlan) -> Any | None:
    return next(
        (
            unit
            for unit in state.work_units
            if unit.unit["kind"] == "initial_observation"
            and unit.unit["subject_scientific_key"] == candidate.scientific_key
        ),
        None,
    )


def _screen_unit_for_candidate(state: RunState, candidate: CandidatePlan) -> Any | None:
    return next(
        (
            unit
            for unit in state.work_units
            if unit.unit["kind"] == "screening"
            and unit.unit["subject_scientific_key"] == candidate.scientific_key
        ),
        None,
    )


def _verification_unit_for_candidate(state: RunState, candidate: CandidatePlan) -> Any | None:
    return next(
        (
            unit
            for unit in state.work_units
            if unit.unit["kind"] == "verification"
            and unit.unit["subject_scientific_key"] == candidate.scientific_key
        ),
        None,
    )


def _verification_unit_for_scenario(
    state: RunState, scenario: ScenarioPlan
) -> Any | None:
    return next(
        (
            unit
            for unit in state.work_units
            if unit.unit["kind"] == "verification"
            and unit.unit["design"].get("subject_kind") == "scenario_recipe"
            and unit.unit["subject_scientific_key"] == scenario.scientific_key
        ),
        None,
    )


def _candidate_is_standalone(candidate: CandidatePlan) -> bool:
    return (
        candidate.parent_node_id is None
        and candidate.parent_node_scientific_key is None
        and candidate.origin != "stateful_strategy"
    )


def _portfolio_candidate_ancestry(
    context: RunContext,
    state: RunState,
    candidate: CandidatePlan,
) -> _PortfolioCandidateAncestry | None:
    """Trace a stateful child through exact evidence-bearing edges to its entry.

    An aligned resolved child with a usable outcome review may be replanned;
    local failure remains failure and earns no strategy credit. Structural lineage disagreements remain
    corruption-level engine errors.
    """

    if _candidate_is_standalone(candidate):
        return _PortfolioCandidateAncestry(candidate, None, ())
    if (
        candidate.origin != "stateful_strategy"
        or candidate.parent_node_id is None
        or candidate.parent_node_scientific_key is None
        or candidate.strategy_id is None
    ):
        raise EngineError("stateful portfolio candidate has incomplete immutable lineage")

    current = candidate
    expected_node: Any | None = None
    reversed_steps: list[_LinkedStrategyStep] = []
    seen_candidates: set[str] = set()
    seen_nodes: set[str] = set()
    seen_strategies: set[str] = set()
    while not _candidate_is_standalone(current):
        if current.candidate_id in seen_candidates:
            raise EngineError("linked strategy candidate ancestry contains a cycle")
        seen_candidates.add(current.candidate_id)
        if (
            current.origin != "stateful_strategy"
            or current.parent_node_id is None
            or current.parent_node_scientific_key is None
            or current.strategy_id is None
        ):
            raise EngineError("linked strategy candidate has incomplete parent ownership")
        episode = state.strategy_episodes.get(current.strategy_id)
        if episode is None or episode.strategy_id != current.strategy_id:
            raise EngineError("linked strategy candidate lacks its source episode")
        if episode.strategy_id in seen_strategies:
            raise EngineError("linked strategy ancestry repeats a source episode")
        seen_strategies.add(episode.strategy_id)
        matching_children = tuple(
            child
            for child in episode.children
            if child.candidate_id == current.candidate_id
            and child.candidate_scientific_key == current.scientific_key
        )
        if len(matching_children) != 1:
            raise EngineError("linked strategy candidate does not own one source child")
        child = matching_children[0]
        if (
            child.strategy_id != episode.strategy_id
            or child.strategy_scientific_key != episode.strategy_scientific_key
        ):
            raise EngineError("linked strategy child changed its source contract identity")
        if not (
            child.alignment_status == "aligned"
            and child.outcome.aligned
            and child.outcome.resolved
            and child.outcome.outcome_result in {"supports", "falsifies", "indeterminate"}
        ):
            return None
        if child.observation_id is None:
            raise EngineError("supported linked strategy child lacks an observation")
        matching_nodes = tuple(
            node
            for node in state.conversation_nodes.values()
            if node.candidate_id == current.candidate_id
            and node.target_observation_id == child.observation_id
        )
        if len(matching_nodes) != 1:
            raise EngineError("linked strategy child does not own one conversation node")
        node = matching_nodes[0]
        if expected_node is not None and node.node_id != expected_node.node_id:
            raise EngineError("linked strategy ancestry does not follow its exact parent node")
        if node.node_id in seen_nodes:
            raise EngineError("linked strategy conversation ancestry contains a cycle")
        seen_nodes.add(node.node_id)
        if (
            node.parent_node_id != current.parent_node_id
            or episode.generation_parent_node_id != current.parent_node_id
            or episode.generation_parent_node_scientific_key
            != current.parent_node_scientific_key
            or node.root_id != episode.conversation_root_id
        ):
            raise EngineError("linked strategy child changed its frozen generation parent")
        root = state.conversation_roots.get(node.root_id)
        if (
            root is None
            or root.scientific_key != episode.conversation_root_scientific_key
        ):
            raise EngineError("linked strategy child changed its conversation root")
        parent_node = state.conversation_nodes.get(current.parent_node_id)
        if (
            parent_node is None
            or parent_node.scientific_key != current.parent_node_scientific_key
            or parent_node.root_id != node.root_id
            or node.depth != parent_node.depth + 1
        ):
            raise EngineError("linked strategy child has noncontiguous node ancestry")
        parent_candidate = _candidate_by_id(context, parent_node.candidate_id)
        if parent_candidate is None:
            raise EngineError("linked strategy parent candidate is not hydrated")
        reversed_steps.append(_LinkedStrategyStep(episode, child, current, node))
        current = parent_candidate
        expected_node = parent_node

    entry_node = expected_node
    if entry_node is None:
        raise EngineError("stateful candidate ancestry has no entry conversation node")
    if (
        entry_node.parent_node_id is not None
        or entry_node.depth != 0
        or entry_node.candidate_id != current.candidate_id
    ):
        raise EngineError("linked strategy ancestry does not terminate at a fresh entry")
    return _PortfolioCandidateAncestry(
        current,
        entry_node,
        tuple(reversed(reversed_steps)),
    )


def _candidate_singleton_execution(
    state: RunState, candidate: CandidatePlan
) -> tuple[Any, Mapping[str, Any], Mapping[str, Any]] | None:
    """Find the candidate's one actual fresh-root discovery execution."""

    if not _candidate_is_standalone(candidate):
        return None
    for unit in state.work_units:
        if unit.unit["kind"] not in {"initial_observation", "map_exploration"}:
            continue
        for panel in unit.unit["panel_plans"]:
            target_action = state.logical_actions[panel["target_action_id"]]
            target = state.target_results.get(target_action.action_id)
            if target is None or target.candidate_id != candidate.candidate_id:
                continue
            execution = next(
                item
                for item in unit.unit["execution_slots"]
                if item["execution_scientific_key"]
                == target_action.slot["owner_position_scientific_key"]
            )
            return unit, panel, execution
    return None


def _screen_trigger_source(
    state: RunState, candidate: CandidatePlan, screen: Any
) -> tuple[Any, Mapping[str, Any]] | None:
    triggering_id = screen.unit["design"]["triggering_execution_id"]
    singleton = _candidate_singleton_execution(state, candidate)
    if singleton is None or singleton[2]["execution_id"] != triggering_id:
        return None
    return singleton[0], singleton[1]


def _screen_triggered(
    context: RunContext, state: RunState, candidate: CandidatePlan
) -> bool:
    from .evidence import screening_trigger

    singleton = _candidate_singleton_execution(state, candidate)
    if singleton is None:
        return False
    del context
    evidence = state.execution_evidence.get(singleton[2]["execution_id"])
    if evidence is None:
        raise EngineError("reducer omitted the candidate singleton execution evidence")
    if not evidence.terminated:
        return False
    return screening_trigger(
        evidence.primary_score,
        panel_required=evidence.panel_required,
        panel=evidence.panel,
    )


def _candidate_screening_block(
    state: RunState,
    candidate: CandidatePlan,
) -> Any | None:
    """Return and identity-check the reducer's canonical fixed-screen view."""

    record = state.candidate_evidence.get(candidate.candidate_id)
    block = state.screening_blocks.get(candidate.candidate_id)
    if record is None and block is None:
        return None
    if record is None:
        raise EngineError("reducer screening block lacks candidate evidence")
    if record.subject_id != candidate.candidate_id:
        raise EngineError("reducer candidate evidence changed its subject identity")
    if block is None:
        if record.screening_block_id is not None:
            raise EngineError("reducer candidate evidence lacks its screening block")
        return None
    if (
        block.candidate_id != candidate.candidate_id
        or block.candidate_scientific_key != candidate.scientific_key
        or record.screening_block_id != block.block_id
        or record.search_status is not block.status
    ):
        raise EngineError("reducer screening projections disagree on candidate identity")
    return block


def _entry_candidate_for_episode(
    context: RunContext,
    state: RunState,
    episode: StrategyEpisodeView,
) -> CandidatePlan:
    """Resolve one episode's original entry through supported strategy edges."""

    parent_node = state.conversation_nodes.get(episode.generation_parent_node_id)
    if (
        parent_node is None
        or parent_node.scientific_key
        != episode.generation_parent_node_scientific_key
        or parent_node.root_id != episode.conversation_root_id
    ):
        raise EngineError("strategy source conversation is missing")
    immediate = _candidate_by_id(context, parent_node.candidate_id)
    if immediate is None:
        raise EngineError("strategy source candidate is not hydrated")
    ancestry = _portfolio_candidate_ancestry(context, state, immediate)
    if ancestry is None:
        raise EngineError("strategy source descends from a nonsupporting child")
    record = state.candidate_evidence.get(ancestry.entry_candidate.candidate_id)
    panels = tuple(
        panel
        for panel in state.panel_views.values()
        if panel.observation_id == parent_node.target_observation_id
    )
    baseline = panels[0].effective_score if len(panels) == 1 else None
    if (
        record is None
        or episode.parent_search_status is not record.search_status
        or baseline is None
        or episode.parent_baseline_score != baseline
    ):
        raise EngineError(
            "strategy episode changed its entry status or immediate parent baseline"
        )
    intent = portfolio_intent_for_node(
        node_id=parent_node.node_id,
        conversation_nodes=state.conversation_nodes,
        target_observations=state.target_observations,
        panel_views=state.panel_views,
        member_assessments=state.member_assessments,
    )
    if episode.portfolio_intent is not intent:
        raise EngineError("strategy episode changed its parent-node intent")
    return ancestry.entry_candidate


def _candidate_is_promising(
    context: RunContext, state: RunState, candidate: CandidatePlan
) -> bool:
    del context
    block = _candidate_screening_block(state, candidate)
    return bool(
        block is not None
        and block.block_terminated
        and block.status.value == "promising"
    )


def _portfolio_roots_for_candidate(
    context: RunContext, state: RunState, candidate: CandidatePlan
) -> tuple[_PortfolioRoot, ...]:
    if not _candidate_is_standalone(candidate):
        ancestry = _portfolio_candidate_ancestry(context, state, candidate)
        if ancestry is None:
            return ()
        node = ancestry.linked_steps[-1].node
        if portfolio_branch_closed(state, node, maximum_followups=context.config.search.maximum_followups, maximum_stalled_followups=context.config.search.maximum_stalled_followups):
            return ()
        root = state.conversation_roots[node.root_id]
        intent = portfolio_intent_for_node(
            node_id=node.node_id,
            conversation_nodes=state.conversation_nodes,
            target_observations=state.target_observations,
            panel_views=state.panel_views,
            member_assessments=state.member_assessments,
        )
        if intent is None:
            return ()
        return (
            _PortfolioRoot(
                node.node_id,
                node.scientific_key,
                node.root_id,
                root.scientific_key,
                intent,
            ),
        )
    if not any(
        entry.subject_id == candidate.candidate_id
        for entry in state.archive.entries
    ):
        return ()
    screening_block = _candidate_screening_block(state, candidate)
    if screening_block is not None and not screening_block.block_terminated:
        return ()
    ranked: list[tuple[int, Decimal, str, _PortfolioRoot]] = []
    for node in state.conversation_nodes.values():
        if (
            node.candidate_id != candidate.candidate_id
            or node.parent_node_id is not None
            or node.depth != 0
        ):
            continue
        panels = tuple(
            panel
            for panel in state.panel_views.values()
            if panel.observation_id == node.target_observation_id
        )
        if len(panels) != 1 or not panels[0].resolved:
            continue
        if portfolio_branch_closed(state, node, maximum_followups=context.config.search.maximum_followups, maximum_stalled_followups=context.config.search.maximum_stalled_followups):
            continue
        score = panels[0].effective_score
        if score is None:
            continue
        intent = portfolio_intent_for_node(
            node_id=node.node_id,
            conversation_nodes=state.conversation_nodes,
            target_observations=state.target_observations,
            panel_views=state.panel_views,
            member_assessments=state.member_assessments,
        )
        if intent is None:
            continue
        root = state.conversation_roots.get(node.root_id)
        if root is None:
            raise EngineError("portfolio node lacks its conversation root")
        selection = _PortfolioRoot(
            node.node_id,
            node.scientific_key,
            node.root_id,
            root.scientific_key,
            intent,
        )
        ranked.append(
            (0 if intent is PortfolioIntent.ADVANCE else 1, -score, node.scientific_key, selection)
        )
    if not ranked:
        return ()
    ranked.sort(key=lambda item: item[:3])
    return (ranked[0][3],)


def _candidate_has_observation(state: RunState, candidate: CandidatePlan) -> bool:
    return any(
        result.candidate_id == candidate.candidate_id
        for result in state.target_results.values()
    )


def _pending_initial_units(
    context: RunContext, state: RunState
) -> tuple[WorkUnit, ...]:
    """Materialize ordinary bootstrap/seed units in their stable subject order."""

    return tuple(
        build_initial_observation(context, candidate)
        for candidate in sorted(
            context.candidates.values(), key=lambda item: item.scientific_key
        )
        if _candidate_is_standalone(candidate)
        and candidate.origin in {"bootstrap", "seed"}
        and not _candidate_has_observation(state, candidate)
        and _initial_unit_for_candidate(state, candidate) is None
    )


def _candidate_expanded(state: RunState, candidate: CandidatePlan, context: RunContext) -> bool:
    del context
    expected_entry_key = scientific_key([candidate.scientific_key, "archive_entry"])
    return any(
        unit.unit["kind"] == "map_exploration"
        and unit.unit["design"]["parent_archive_entry_scientific_key"] == expected_entry_key
        for unit in state.work_units
    )


def _ordinary_expansion_counts(
    context: RunContext, state: RunState
) -> Mapping[tuple[str, ...], int]:
    """Count durable ordinary parent selections by their declared cells."""

    counts: dict[tuple[str, ...], int] = {}
    candidates_by_entry = {
        scientific_key([candidate.scientific_key, "archive_entry"]): candidate
        for candidate in context.candidates.values()
        if _candidate_is_standalone(candidate)
    }
    for unit in state.work_units:
        if unit.unit["kind"] != "map_exploration":
            continue
        parent = candidates_by_entry.get(
            unit.unit["design"]["parent_archive_entry_scientific_key"]
        )
        if parent is None:
            raise EngineError("ordinary expansion names an unknown standalone parent")
        cell = tuple(
            parent.descriptors[name] for name in context.objective.descriptor_names
        )
        counts[cell] = counts.get(cell, 0) + 1
    return counts


def _prior_directed_slot_counts(
    context: RunContext, state: RunState
) -> Mapping[tuple[str, ...], int]:
    """Count pre-response ordinary proposal directions for coverage ordering."""

    counts: dict[tuple[str, ...], int] = {}
    for unit in state.work_units:
        if unit.unit["kind"] != "map_exploration":
            continue
        for descriptors in unit.unit["design"]["under_covered_cells"]:
            cell = tuple(
                descriptors[name] for name in context.objective.descriptor_names
            )
            counts[cell] = counts.get(cell, 0) + 1
    return counts


def _next_map_exploration(
    context: RunContext, state: RunState
) -> WorkUnit | None:
    """Select one retained expandable archive parent and freeze its coverage brief."""

    archive = state.archive
    if not isinstance(archive, ArchiveSnapshot):
        raise EngineError("RunState archive projection has an invalid type")
    candidates_by_id = {
        candidate.candidate_id: candidate
        for candidate in context.candidates.values()
        if _candidate_is_standalone(candidate)
    }
    expandable: dict[str, CandidatePlan] = {}
    for entry in archive.entries:
        candidate = candidates_by_id.get(entry.subject_id)
        if candidate is None:
            continue
        expected_key = scientific_key([candidate.scientific_key, "archive_entry"])
        if entry.scientific_key != expected_key:
            raise EngineError("archive entry key disagrees with its standalone subject")
        if not _candidate_expanded(state, candidate, context):
            expandable[entry.scientific_key] = candidate
    selected = select_archive_parent(
        archive,
        expandable_entry_keys=expandable,
        ordinary_expansion_counts=_ordinary_expansion_counts(context, state),
        seed=context.config.random_seed,
    )
    if selected is None:
        return None
    parent = expandable[selected.scientific_key]
    return build_map_exploration(
        context,
        parent,
        parent_unit_id=_candidate_parent_unit_id(state, parent),
        archive_snapshot=archive,
        prior_directed_slots=_prior_directed_slot_counts(context, state),
    )


def _strategy_development_units(
    context: RunContext, state: RunState
) -> tuple[tuple[WorkUnit, StrategyEpisodeView, CandidatePlan], ...]:
    """Freeze one development candidate per eligible template artifact."""

    started_template_keys = {
        unit.unit["subject_scientific_key"]
        for unit in state.work_units
        if unit.unit.get("kind") in _STRATEGY_QUALIFICATION_KINDS
    }
    eligible_by_template: dict[
        str, tuple[tuple[Any, ...], StrategyEpisodeView, CandidatePlan, StrategyTemplate]
    ] = {}
    verified_episode_ids = {
        episode_id
        for scenario in state.scenarios.values()
        if scenario.origin == "derived"
        and state.scenario_evidence.get(scenario.scenario_id) is not None
        and state.scenario_evidence[
            scenario.scenario_id
        ].verification_status.value
        == "verified"
        for episode_id in scenario.source_step_episode_ids
    }
    for episode in state.strategy_episodes.values():
        template = episode.rounds[0].lane.item.template_candidate
        if (
            episode.strategy_id not in verified_episode_ids
            or not episode.preferred_in_run
            or not isinstance(template, StrategyTemplate)
        ):
            continue
        qualifying_children = tuple(
            child
            for child in episode.children
            if child.alignment_status == "aligned"
            and child.outcome.aligned
            and child.outcome.resolved
            and child.outcome.outcome_result == "supports"
            and (
                child.outcome.confirmed_objective_exceptional
                or child.outcome.confirmed_objective_application
            )
        )
        if not qualifying_children or not derive_template_comparison_dimensions(template):
            continue
        template_key, _ = _strategy_template_identity(context, template)
        if template_key in started_template_keys:
            continue
        candidate = _entry_candidate_for_episode(context, state, episode)
        strongest = min(
            qualifying_children,
            key=lambda child: (
                -int(child.outcome.confirmed_objective_application),
                -(child.outcome.effective_score or Decimal(0)),
                child.child_scientific_key,
            ),
        )
        rank = (
            -int(strongest.outcome.confirmed_objective_application),
            -(strongest.outcome.effective_score or Decimal(0)),
            episode.strategy_scientific_key,
        )
        incumbent = eligible_by_template.get(template_key)
        if incumbent is None or rank < incumbent[0]:
            eligible_by_template[template_key] = (
                rank,
                episode,
                candidate,
                template,
            )

    result: list[tuple[WorkUnit, StrategyEpisodeView, CandidatePlan]] = []
    for template_key in sorted(eligible_by_template):
        _, episode, candidate, template = eligible_by_template[template_key]
        if not episode.rounds or not episode.rounds[-1].complete:
            raise EngineError("eligible strategy development source is not terminal")
        result.append(
            (
                build_strategy_development(
                    context,
                    template,
                    parent_unit_id=episode.rounds[-1].work_unit_id,
                ),
                episode,
                candidate,
            )
        )
    return tuple(result)


def _strategy_holdout_units(
    context: RunContext, state: RunState
) -> tuple[tuple[WorkUnit, StrategyEpisodeView, CandidatePlan], ...]:
    """Consume only reducer-qualified development views to schedule holdout."""

    result: list[tuple[WorkUnit, StrategyEpisodeView, CandidatePlan]] = []
    for template_id, view in sorted(state.strategy_qualifications.items()):
        if view.template_id != template_id:
            raise EngineError("strategy qualification map key disagrees with its view")
        development = view.development_decision
        if (
            development is None
            or development.status.value != "eligible"
            or view.holdout_unit_id is not None
        ):
            continue
        development_unit_id = view.development_unit_id
        source = next(
            (
                unit
                for unit in state.work_units
                if unit.unit_id == development_unit_id
            ),
            None,
        )
        if source is None or source.unit["kind"] != "strategy_development":
            raise EngineError("eligible holdout lacks its source development unit")
        template = view.template
        if not isinstance(template, StrategyTemplate):
            raise EngineError("eligible holdout lacks its frozen StrategyTemplate")
        episode = state.strategy_episodes.get(view.source_episode_id)
        if episode is None:
            raise EngineError("eligible holdout lacks its source strategy episode")
        candidate = _entry_candidate_for_episode(context, state, episode)
        if (
            view.source_entry_candidate_id != candidate.candidate_id
            or view.source_entry_candidate_scientific_key
            != candidate.scientific_key
        ):
            raise EngineError("eligible holdout changed its original entry ancestor")
        unit = build_strategy_holdout(
            context,
            template,
            source_development_unit=source.unit,
        )
        if unit.subject_id != template_id:
            raise EngineError("eligible holdout template identity changed")
        result.append((unit, episode, candidate))
    return tuple(result)


def _committed_discretionary_target_slots(state: RunState) -> tuple[int, int]:
    """Reduce the exact C/S committed-slot counters from started units."""

    coverage = 0
    stateful = 0
    for unit in state.work_units:
        kind = unit.unit["kind"]
        if kind == "initial_observation":
            coverage += 1
        elif kind == "map_exploration":
            coverage += unit.unit["design"]["proposal_count"]
        elif kind == "portfolio_round":
            stateful += unit.unit["design"]["proposal_count"]
    return coverage, stateful


def _median(values: Iterable[Decimal]) -> Decimal | None:
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / Decimal(2)


def _subject_rank_metadata(
    context: RunContext,
    state: RunState,
    candidate: CandidatePlan | None,
    *,
    episodes: Sequence[StrategyEpisodeView] = (),
) -> tuple[bool, str, Decimal | None, int]:
    """Derive the scheduler's complete within-class subject rank inputs."""

    measurement_cell_verified = False
    evidence_tier = "ordinary"
    fixed_block_median: Decimal | None = None
    prior_target_calls = 0
    if candidate is not None:
        cell = tuple(
            candidate.descriptors[name] for name in context.objective.descriptor_names
        )
        measurement_cell_verified = any(
            entry.evidence.verification_status.value == "verified"
            for entry in state.archive.cells.get(cell, ())
        )
        retained = tuple(
            entry
            for entry in state.archive.entries
            if entry.subject_id == candidate.candidate_id
        )
        if retained:
            fixed_block_median = retained[0].fitness
        elif _candidate_is_standalone(candidate):
            screen = _candidate_screening_block(state, candidate)
            if screen is not None:
                fixed_block_median = _median(screen.effective_scores)
        candidate_panels = tuple(
            panel.assessment
            for panel in state.panel_views.values()
            if panel.candidate_id == candidate.candidate_id
            and panel.assessment is not None
        )
        if any(item.confirmed_objective_application for item in candidate_panels):
            evidence_tier = "application"
        elif any(
            item.confirmed_objective_exceptional for item in candidate_panels
        ):
            evidence_tier = "exceptional"
        prior_target_calls = sum(
            sum(attempt.dispatched for attempt in action.attempts)
            for action in state.logical_actions.values()
            if action.slot["role"] == Role.TARGET.value
            and any(
                attempt.intended.get("lineage", {}).get("candidate_id")
                == candidate.candidate_id
                for attempt in action.attempts
            )
        )
        if not _candidate_is_standalone(candidate):
            ancestry = _portfolio_candidate_ancestry(context, state, candidate)
            if ancestry is None:
                raise EngineError(
                    "scheduled stateful candidate is not a supporting linked child"
                )
            node = ancestry.linked_steps[-1].node
            panels = tuple(
                panel
                for panel in state.panel_views.values()
                if panel.observation_id == node.target_observation_id
            )
            if len(panels) != 1 or panels[0].effective_score is None:
                raise EngineError(
                    "scheduled stateful candidate lacks its parent-node baseline"
                )
            fixed_block_median = panels[0].effective_score

    episode_children = tuple(
        child.outcome for episode in episodes for child in episode.children
    )
    if any(item.confirmed_objective_application for item in episode_children):
        evidence_tier = "application"
    elif evidence_tier != "application" and any(
        item.confirmed_objective_exceptional for item in episode_children
    ):
        evidence_tier = "exceptional"
    if episodes:
        parent_baselines = {item.parent_baseline_score for item in episodes}
        if len(parent_baselines) != 1:
            raise EngineError(
                "scheduled strategy episodes disagree on parent baseline evidence"
            )
        fixed_block_median = next(iter(parent_baselines))
        prior_target_calls = sum(child.dispatched for item in episodes for child in item.children)
    return (
        measurement_cell_verified,
        evidence_tier,
        fixed_block_median,
        prior_target_calls,
    )


def _scheduled_candidate(
    context: RunContext,
    state: RunState,
    unit: WorkUnit,
    *,
    candidate: CandidatePlan | None,
    episodes: Sequence[StrategyEpisodeView] = (),
    slot_ordinal: int = 0,
    active: bool = False,
) -> WorkCandidate:
    verified, tier, median, prior_calls = _subject_rank_metadata(
        context, state, candidate, episodes=episodes
    )
    return WorkCandidate(
        work_unit=unit,
        active=active,
        measurement_cell_verified=verified,
        evidence_tier=tier,
        fixed_block_median=median,
        prior_target_calls=prior_calls,
        slot_ordinal=slot_ordinal,
        semantic_progress=portfolio_node_progress(state, state.conversation_nodes[unit.design["generation_parent_node_id"]])[0] if unit.kind == "portfolio_round" else 0,
    )


def _last_complete_attempt(action: LogicalActionView) -> AttemptView:
    matches = tuple(
        attempt
        for attempt in action.attempts
        if attempt.succeeded
        and not attempt.target_identity_collision
        and attempt.terminal_event is not None
        and attempt.terminal_event["payload"]["finish_status"]
        == FinishStatus.COMPLETE.value
    )
    if not matches:
        raise EngineError("semantic source action lacks a complete successful attempt")
    return matches[-1]


def _hydrate_scenarios(context: RunContext, state: RunState) -> None:
    for view in sorted(state.scenarios.values(), key=lambda item: item.scientific_key):
        candidate = ScenarioPlan(
            scenario_id=view.scenario_id,
            scientific_key=view.scientific_key,
            entry_prompt=view.entry_prompt,
            recipe=view.recipe,
            origin=view.origin,
            source_seed_id=view.source_seed_id,
            source_episode_id=view.source_episode_id,
            source_child_scientific_key=view.source_child_scientific_key,
            parent_unit_id=view.parent_unit_id,
            source_parent_node_keys=tuple(
                state.strategy_episodes[episode_id].generation_parent_node_scientific_key
                for episode_id in view.source_step_episode_ids
            ),
        )
        incumbent = context.scenarios.get(candidate.scientific_key)
        if incumbent is not None:
            if incumbent != candidate:
                raise EngineError("hydrated scenario disagrees with reducer ownership")
            continue
        context.scenarios[candidate.scientific_key] = candidate


def _hydrate_generated_candidates(context: RunContext, state: RunState) -> None:
    for unit in state.work_units:
        if unit.unit["kind"] not in {"map_exploration", "portfolio_round"}:
            continue
        for action in unit.actions:
            if action.slot["role"] != Role.TARGET.value:
                continue
            # A dispatched or durably intended realization owns its candidate
            # even when every target attempt later closes unresolved.  Resume
            # must restore ordinary global ownership and exact-parent stateful
            # ownership before scheduling or reconstructing later branches.
            if not action.attempts and action.action_id not in state.target_results:
                continue
            _candidate_for_target_action(context, state, unit, action)


def _unfinished_unit(context: RunContext, state: RunState) -> Any | None:
    for unit in state.work_units:
        if not _unit_complete(context, state, unit):
            return unit
    return None


def _materialized_work_unit(unit: Any) -> WorkUnit:
    """Restore one exact started unit for active-work scheduler priority."""

    raw = unit.unit
    return WorkUnit(
        unit_id=raw["unit_id"],
        unit_scientific_key=raw["unit_scientific_key"],
        kind=raw["kind"],
        subject_id=raw["subject_id"],
        subject_scientific_key=raw["subject_scientific_key"],
        parent_unit_id=raw["parent_unit_id"],
        policy_digest=raw["policy_digest"],
        design=raw["design"],
        proposal_slots=tuple(raw["proposal_slots"]),
        execution_slots=tuple(raw["execution_slots"]),
        panel_plans=tuple(raw["panel_plans"]),
        logical_action_slots=tuple(raw["logical_action_slots"]),
        worst_case_calls=RoleCallVector(**raw["worst_case_calls"]),
    )


def _last_completed_unit_id(state: RunState) -> str | None:
    for unit in reversed(state.work_units):
        # Terminal handling must not re-enter the contextual parsers which may
        # have raised the original invariant failure. Use canonical reduction.
        if unit.unit_id in state.completed_work_unit_ids:
            return unit.unit_id
    return None


def _candidate_parent_unit_id(state: RunState, candidate: CandidatePlan) -> str:
    for unit in reversed(state.work_units):
        if any(result.candidate_id == candidate.candidate_id for result in state.target_results.values()):
            if any(
                action.action_id in state.target_results
                and state.target_results[action.action_id].candidate_id == candidate.candidate_id
                for action in unit.actions
            ):
                return unit.unit_id
    raise EngineError("observed candidate has no parent work unit")


def _next_new_unit(
    context: RunContext, state: RunState
) -> tuple[WorkUnit | None, bool]:
    """Derive, rank, and admit the canonical set of complete work candidates."""

    _hydrate_scenarios(context, state)

    ordered_candidates = sorted(
        context.candidates.values(), key=lambda item: item.scientific_key
    )
    scheduled: list[WorkCandidate] = []

    for slot_ordinal, candidate in enumerate(ordered_candidates):
        if (
            _candidate_is_promising(context, state, candidate)
            and any(panel.candidate_id == candidate.candidate_id and panel.assessment is not None and panel.assessment.confirmed_objective_application for panel in state.panel_views.values())
            and _verification_unit_for_candidate(state, candidate) is None
        ):
            scheduled.append(
                _scheduled_candidate(
                    context,
                    state,
                    build_verification(
                        context,
                        candidate,
                        parent_unit_id=_candidate_parent_unit_id(state, candidate),
                    ),
                    candidate=candidate,
                    slot_ordinal=slot_ordinal,
                )
            )

    scenario_offset = len(ordered_candidates)
    for slot_ordinal, scenario in enumerate(
        sorted(context.scenarios.values(), key=lambda item: item.scientific_key)
    ):
        if _verification_unit_for_scenario(state, scenario) is not None:
            continue
        source_episode = (
            None
            if scenario.source_episode_id is None
            else state.strategy_episodes.get(scenario.source_episode_id)
        )
        if scenario.origin == "derived" and source_episode is None:
            raise EngineError("derived scenario lost its source strategy episode")
        scheduled.append(
            _scheduled_candidate(
                context,
                state,
                build_scenario_verification(context, scenario),
                candidate=None,
                episodes=() if source_episode is None else (source_episode,),
                slot_ordinal=scenario_offset + slot_ordinal,
            )
        )

    qualification_ordinal = 0
    for unit, episode, candidate in _strategy_holdout_units(context, state):
        scheduled.append(
            _scheduled_candidate(
                context,
                state,
                unit,
                candidate=candidate,
                episodes=(episode,),
                slot_ordinal=qualification_ordinal,
            )
        )
        qualification_ordinal += 1
    for unit, episode, candidate in _strategy_development_units(context, state):
        scheduled.append(
            _scheduled_candidate(
                context,
                state,
                unit,
                candidate=candidate,
                episodes=(episode,),
                slot_ordinal=qualification_ordinal,
            )
        )
        qualification_ordinal += 1

    for slot_ordinal, candidate in enumerate(ordered_candidates):
        singleton = _candidate_singleton_execution(state, candidate)
        if (
            singleton is not None
            and _screen_triggered(context, state, candidate)
            and _screen_unit_for_candidate(state, candidate) is None
        ):
            source_unit, _, triggering_execution = singleton
            scheduled.append(
                _scheduled_candidate(
                    context,
                    state,
                    build_screening(
                        context,
                        candidate,
                        triggering_execution["execution_id"],
                        triggering_execution["execution_scientific_key"],
                        parent_unit_id=source_unit.unit_id,
                    ),
                    candidate=candidate,
                    slot_ordinal=slot_ordinal,
                )
            )

    portfolio_ordinal = 0
    for candidate in ordered_candidates:
        roots = _portfolio_roots_for_candidate(context, state, candidate)
        for root in roots:
            if any(
                work.unit["kind"] == "portfolio_round"
                and work.unit["design"]["generation_parent_node_scientific_key"]
                == root.node_scientific_key
                for work in state.work_units
            ):
                continue
            scheduled.append(
                _scheduled_candidate(
                    context,
                    state,
                    build_portfolio_round(
                        context,
                        candidate,
                        parent_unit_id=_candidate_parent_unit_id(state, candidate),
                        conversation_root_id=root.conversation_root_id,
                        conversation_root_scientific_key=(
                            root.conversation_root_scientific_key
                        ),
                        parent_node_id=root.node_id,
                        parent_node_scientific_key=root.node_scientific_key,
                        portfolio_intent=root.portfolio_intent,
                    ),
                    candidate=candidate,
                    slot_ordinal=portfolio_ordinal,
                )
            )
            portfolio_ordinal += 1

    candidates_by_key = {
        candidate.scientific_key: candidate for candidate in ordered_candidates
    }
    for slot_ordinal, unit in enumerate(_pending_initial_units(context, state)):
        scheduled.append(
            _scheduled_candidate(
                context,
                state,
                unit,
                candidate=candidates_by_key[unit.subject_scientific_key],
                slot_ordinal=slot_ordinal,
            )
        )

    map_unit = _next_map_exploration(context, state)
    if map_unit is not None:
        parent_entry = next(
            entry
            for entry in state.archive.entries
            if entry.scientific_key
            == map_unit.design["parent_archive_entry_scientific_key"]
        )
        parent = next(
            candidate
            for candidate in ordered_candidates
            if candidate.candidate_id == parent_entry.subject_id
        )
        scheduled.append(
            _scheduled_candidate(
                context,
                state,
                map_unit,
                candidate=parent,
            )
        )

    if not scheduled:
        return None, False
    allocations = remaining_work_allocations(state)
    scheduled = [item for item in scheduled if allocations[item.kind] > 0]
    if not scheduled:
        return None, True
    coverage_slots, stateful_slots = _committed_discretionary_target_slots(state)
    selected = choose_next_work(
        scheduled, state.remaining,
        coverage_slots_committed=coverage_slots,
        stateful_slots_committed=stateful_slots,
    )
    if selected is None:
        raise EngineError("funded planning invariant violated: allocated work cannot fit physical call limits")
    return selected.work_unit, True


def _start_unit(context: RunContext, journal: EventJournal, unit: WorkUnit) -> None:
    state = reduce_events(journal.events, context.resources.objective)
    if not can_start(unit, state.remaining):
        raise EngineError("attempted to start a work unit without its worst-case budget")
    journal.append(
        "work_unit_started",
        {"unit": unit.to_dict()},
        causes=(journal.events[-1]["event_id"],),
    )
    _execute_work_unit(context, journal, unit.unit_id)


def _run_loop(context: RunContext, journal: EventJournal) -> RunTerminalReason:
    while True:
        state = reduce_events(journal.events, context.resources.objective)
        _raise_for_reduced_provider_state(context, state)
        _hydrate_generated_candidates(context, state)
        _hydrate_scenarios(context, state)
        outstanding = _unfinished_unit(context, state)
        if outstanding is not None:
            coverage_slots, stateful_slots = _committed_discretionary_target_slots(
                state
            )
            active = WorkCandidate(
                work_unit=_materialized_work_unit(outstanding), active=True
            )
            selected = choose_next_work(
                (active,),
                state.remaining,
                coverage_slots_committed=coverage_slots,
                stateful_slots_committed=stateful_slots,
            )
            if selected is not active:
                raise EngineError("scheduler did not prioritize the active work unit")
            _execute_work_unit(context, journal, selected.work_unit.unit_id)
            continue
        unit, had_eligible = _next_new_unit(context, state)
        if unit is None:
            return (
                RunTerminalReason.COMPLETED_PLAN
                if had_eligible
                else RunTerminalReason.COMPLETED_NO_ELIGIBLE_WORK
            )
        _start_unit(context, journal, unit)


def _append_terminal(
    context: RunContext,
    journal: EventJournal,
    reason: RunTerminalReason,
    *,
    detail: str,
    resumable: bool,
) -> None:
    state = reduce_events(journal.events, context.resources.objective)
    journal.append(
        "run_terminated",
        {
            "invocation_id": context.invocation_id,
            "reason": reason.value,
            "last_completed_work_unit_id": _last_completed_unit_id(state),
            "resumable": resumable,
            "detail": detail,
        },
        causes=(journal.events[-1]["event_id"],),
    )


def _install_signal_handlers(context: RunContext) -> dict[int, Any]:
    previous: dict[int, Any] = {}

    def request_stop(signum: int, frame: Any) -> None:
        del signum, frame
        context.stop_requested = True

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, request_stop)
    return previous


def _restore_signal_handlers(previous: Mapping[int, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _write_reports(run_directory: Path, state: RunState) -> None:
    from .reporting import write_derived_reports

    write_derived_reports(run_directory, state)


def _write_terminal_derivatives(
    run_directory: Path,
    state: RunState,
    *,
    repository_root: Path,
) -> None:
    """Refresh cache indexes nonfatally, then write authoritative reports."""

    from .reporting import ReportingError, refresh_library_indexes

    if state.terminal_event is None:
        raise EngineError("terminal derivatives require a durable run_terminated event")
    try:
        refresh_library_indexes(repository_root, state)
    except (OSError, ReportingError) as exc:
        # The journal is already terminal and authoritative. A shared-cache
        # failure can be retried by explicit rebuild and never changes reason.
        print(f"index_refresh_failed: {exc}", file=sys.stderr)
    _write_reports(run_directory, state)


def validate_new_run(
    config_source: str | os.PathLike[str] | Mapping[str, Any],
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> ResourceSet:
    return load_resources(config_source, repository_root=repository_root)


def run_new(
    config_source: str | os.PathLike[str] | Mapping[str, Any],
    *,
    repository_root: Path = REPOSITORY_ROOT,
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[Path, RunTerminalReason]:
    """Validate, create one unique child, and execute the serial lifecycle."""

    resources = load_resources(config_source, repository_root=repository_root)
    run_nonce = os.urandom(16).hex()
    run_id = entity_id(
        "run",
        [
            run_nonce,
            resources.scientific_scope_digest,
            resources.config.config_digest,
        ],
    )
    invocation_id = entity_id("iv", [run_id, os.urandom(16).hex()])
    output_base = resources.config.output_directory
    output_base.mkdir(parents=True, exist_ok=True)
    run_directory = output_base / run_id
    run_directory.mkdir(parents=False, exist_ok=False)
    context = RunContext(
        resources=resources,
        run_id=run_id,
        run_nonce=run_nonce,
        invocation_id=invocation_id,
        run_directory=run_directory,
        candidates={},
        scenarios={},
        routes={},
        sleeper=sleeper,
    )
    _initialize_routes(context)
    context.candidates.update(_build_candidates(resources, run_id))
    context.scenarios.update(_build_imported_scenarios(resources, run_id))
    previous_handlers = _install_signal_handlers(context)
    reason = RunTerminalReason.IMPOSSIBLE_INTERNAL_STATE
    try:
        with EventJournal(
            run_directory,
            expected_run_id=run_id,
            schemas=resources.authoritative_schemas,
        ) as journal:
            created = journal.append(
                "run_created",
                {
                    "manifest": _manifest(context),
                    "scientific_scope_digest": context.scope,
                    "run_nonce_hex": run_nonce,
                    "run_id": run_id,
                    "output_base": output_base.as_posix(),
                    "run_path_at_creation": run_directory.as_posix(),
                    "creation_invocation_id": invocation_id,
                },
            )
            del created
            try:
                reason = _run_loop(context, journal)
                _append_terminal(
                    context,
                    journal,
                    reason,
                    detail=(
                        "funded_plan_allocations_consumed"
                        if reason is RunTerminalReason.COMPLETED_PLAN
                        else "no_eligible_work_remains"
                    ),
                    resumable=False,
                )
            except _StopRun as stop:
                reason = stop.reason
                _append_terminal(
                    context,
                    journal,
                    stop.reason,
                    detail=stop.detail,
                    resumable=stop.resumable,
                )
            except (EngineError, DomainError):
                reason = RunTerminalReason.IMPOSSIBLE_INTERNAL_STATE
                _append_terminal(
                    context,
                    journal,
                    reason,
                    detail="engine_invariant_failure",
                    resumable=False,
                )
            state = reduce_events(journal.events, context.resources.objective)
            _write_terminal_derivatives(
                run_directory,
                state,
                repository_root=repository_root,
            )
    finally:
        _restore_signal_handlers(previous_handlers)
    return run_directory, reason


def _resume_resources(
    run_directory: Path,
    created: Mapping[str, Any],
    *,
    repository_root: Path,
) -> ResourceSet:
    manifest = created["payload"]["manifest"]
    bootstrap_snapshot = manifest["bootstrap_snapshot"]
    bootstrap = tuple(
        {
            "probe_id": item["probe_id"],
            "prompt": item["prompt"],
            "descriptors": item["descriptors"],
        }
        for item in bootstrap_snapshot["probes"]
    )
    # Resume resolves executable files again, but scientific library inputs
    # come exclusively from sequence-one snapshots. Shared indexes are mutable
    # caches and must never change a frozen run.
    resources = load_resources(
        manifest["normalized_config"],
        repository_root=repository_root,
        frozen_bootstrap=bootstrap,
        frozen_seeds=tuple(manifest["seed_snapshot"]),
        frozen_strategies=tuple(manifest["strategy_template_snapshot"]),
        frozen_resume=True,
    )
    comparisons = {
        "configuration digest": (resources.config.config_digest, manifest["config_digest"]),
        "scientific scope digest": (
            resources.scientific_scope_digest,
            manifest["scientific_scope_digest"],
        ),
        "engine digest": (resources.engine_digest, manifest["engine"]["dirty_content_digest"]),
        "reducer digest": (resources.reducer_digest, manifest["reducer_digest"]),
        "objective plugin digest": (resources.objective.plugin_digest, manifest["objective"]["plugin_digest"]),
        "template digests": (
            {
                name.removesuffix(".md"): digest
                for name, digest in resources.template_digests.items()
            },
            manifest["role_template_digests"],
        ),
        "schema digests": (
            {
                name.removesuffix(".json"): resources.schema_digests[name]
                for name in ROLE_SCHEMA_NAMES
            },
            manifest["response_schema_digests"],
        ),
        "event schema digest": (
            resources.schema_digests["event.json"],
            manifest["event_schema_digest"],
        ),
        "work-unit schema digest": (
            resources.schema_digests["work_unit.json"],
            manifest["work_unit_schema_digest"],
        ),
        "scripted fixture digest": (resources.scripted_fixture_digest, manifest["scripted_fixture_digest"]),
    }
    mismatches = [name for name, (current, frozen) in comparisons.items() if current != frozen]
    if mismatches:
        raise EngineCompatibilityError(
            "resume executable inputs differ: " + ", ".join(mismatches)
        )
    del run_directory
    return resources


def _validate_frozen_attempt_requests(context: RunContext, state: RunState) -> None:
    """Recompute every persisted request once executable inputs are validated."""

    for attempt in state.attempts.values():
        payload = attempt.intended
        expected_attempt_id = entity_id(
            "pa",
            [
                payload["logical_action_id"],
                payload["route_kind"],
                payload["attempt_ordinal"],
            ],
        )
        if attempt.physical_attempt_id != expected_attempt_id:
            raise EngineCompatibilityError(
                "persisted physical attempt identity does not match its frozen slot"
            )
        _request_from_intent(context, payload, state)


def resume_run(
    run_directory: str | os.PathLike[str],
    *,
    repository_root: Path = REPOSITORY_ROOT,
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[Path, RunTerminalReason]:
    """Validate and resume exactly one consolidated run directory."""

    path = Path(run_directory).resolve(strict=True)
    if not path.is_dir():
        raise EngineCompatibilityError("--resume must name a run directory")
    invocation_nonce = os.urandom(16).hex()
    previous_handlers: dict[int, Any] = {}
    reason = RunTerminalReason.IMPOSSIBLE_INTERNAL_STATE
    final_state: RunState | None = None
    context: RunContext | None = None
    try:
        with EventJournal(
            path,
            recovery_invocation_nonce=invocation_nonce,
        ) as journal:
            if not journal.events:
                raise EngineCompatibilityError("run directory has no run_created event")
            created = journal.events[0]
            run_id = created["run_id"]
            invocation_id = entity_id("iv", [run_id, invocation_nonce])
            resources = _resume_resources(
                path, created, repository_root=repository_root
            )
            journal.bind_schemas(resources.authoritative_schemas)
            context = RunContext(
                resources=resources,
                run_id=run_id,
                run_nonce=created["payload"]["run_nonce_hex"],
                invocation_id=invocation_id,
                run_directory=path,
                candidates={},
                scenarios={},
                routes={},
                sleeper=sleeper,
            )
            _initialize_routes(context)
            context.candidates.update(_build_candidates(resources, run_id))
            context.scenarios.update(_build_imported_scenarios(resources, run_id))
            state = reduce_events(journal.events, resources.objective)
            _hydrate_generated_candidates(context, state)
            _hydrate_scenarios(context, state)
            _validate_frozen_attempt_requests(context, state)
            if state.terminal_event is not None and not state.resumable:
                final_state = state
                assert state.terminal_reason is not None
                reason = state.terminal_reason
            else:
                previous_handlers = _install_signal_handlers(context)
                tail_recovery_actions = [
                    event["payload"]["action"]
                    for event in journal.events
                    if event["type"] == "journal_tail_recovered"
                    and event["payload"]["recovery_invocation_id"] == invocation_id
                ]
                dangling = list(state.dangling_attempt_ids)
                actions = list(tail_recovery_actions)
                if dangling:
                    actions.append("dangling_dispatch_closed")
                if not actions:
                    actions = ["none"]
                prior_terminal = state.terminal_reason
                journal.append(
                    "run_resumed",
                    {
                        "invocation_id": invocation_id,
                        "prior_recorded_run_path": created["payload"]["run_path_at_creation"],
                        "current_run_path": path.as_posix(),
                        "validated_previous_hash": journal.terminal_hash,
                        "prior_terminal_reason": None if prior_terminal is None else prior_terminal.value,
                        "recovery_actions": actions,
                        "dangling_attempt_ids": dangling,
                    },
                    causes=(journal.events[-1]["event_id"],),
                )
                for attempt_id in dangling:
                    attempt = state.attempts[attempt_id]
                    dispatch = attempt.dispatch_event
                    assert dispatch is not None
                    journal.append(
                        "model_call_outcome_unknown",
                        {
                            "physical_attempt_id": attempt_id,
                            "discovery_invocation_id": invocation_id,
                            "error_kind": "unknown",
                            "detail": "resume_found_dangling_dispatch",
                            "provider_request_id": None,
                            "raw_body": None,
                        },
                        causes=(dispatch["event_id"],),
                    )
                try:
                    reason = _run_loop(context, journal)
                    _append_terminal(
                        context,
                        journal,
                        reason,
                        detail=(
                            "funded_plan_allocations_consumed"
                            if reason is RunTerminalReason.COMPLETED_PLAN
                            else "no_eligible_work_remains"
                        ),
                        resumable=False,
                    )
                except _StopRun as stop:
                    reason = stop.reason
                    _append_terminal(
                        context,
                        journal,
                        stop.reason,
                        detail=stop.detail,
                        resumable=stop.resumable,
                    )
                except (EngineError, DomainError):
                    reason = RunTerminalReason.IMPOSSIBLE_INTERNAL_STATE
                    _append_terminal(
                        context,
                        journal,
                        reason,
                        detail="engine_invariant_failure",
                        resumable=False,
                    )
                final_state = reduce_events(journal.events, resources.objective)
            assert final_state is not None
            # Derived views share the journal's writer boundary and are built
            # from the exact replay checkpoint held under that lock.
            _write_terminal_derivatives(
                path,
                final_state,
                repository_root=repository_root,
            )
    finally:
        if previous_handlers:
            _restore_signal_handlers(previous_handlers)
    assert final_state is not None
    return path, reason


def rebuild_reports(
    run_directory: str | os.PathLike[str],
) -> RunTerminalReason | None:
    path = Path(run_directory).resolve(strict=True)
    with EventJournal(path) as journal:
        if not journal.events:
            raise EngineCompatibilityError("run directory has no run_created event")
        resources = _resume_resources(
            path, journal.events[0], repository_root=REPOSITORY_ROOT
        )
        journal.bind_schemas(resources.authoritative_schemas)
        state = reduce_events(journal.events, resources.objective)
        context = RunContext(
            resources=resources,
            run_id=state.run_id,
            run_nonce=journal.events[0]["payload"]["run_nonce_hex"],
            invocation_id=journal.events[0]["payload"]["creation_invocation_id"],
            run_directory=path,
            candidates={},
            scenarios={},
            routes={},
        )
        _initialize_routes(context)
        context.candidates.update(_build_candidates(resources, state.run_id))
        context.scenarios.update(_build_imported_scenarios(resources, state.run_id))
        _hydrate_generated_candidates(context, state)
        _hydrate_scenarios(context, state)
        _validate_frozen_attempt_requests(context, state)
        _write_reports(path, state)
    return state.terminal_reason


def validate_run_evidence(
    run_directory: str | os.PathLike[str],
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> RunState:
    """Strictly replay one run without locks, writes, or derived artifacts."""

    path = Path(run_directory).resolve(strict=True)
    events = read_events(path)
    if not events:
        raise EngineCompatibilityError("run directory has no run_created event")
    try:
        resources = _resume_resources(
            path, events[0], repository_root=repository_root
        )
    except (KeyError, TypeError, ValueError, EngineCompatibilityError) as exc:
        raise JournalCorrupt(
            f"run evidence is not replay-compatible: {exc}"
        ) from exc
    strict_events = validate_events(
        events, schemas=resources.authoritative_schemas
    )
    state = reduce_events(strict_events, resources.objective)
    context = RunContext(
        resources=resources,
        run_id=state.run_id,
        run_nonce=events[0]["payload"]["run_nonce_hex"],
        invocation_id=events[0]["payload"]["creation_invocation_id"],
        run_directory=path,
        candidates={},
        scenarios={},
        routes={},
    )
    _initialize_routes(context)
    context.candidates.update(_build_candidates(resources, state.run_id))
    context.scenarios.update(_build_imported_scenarios(resources, state.run_id))
    _hydrate_generated_candidates(context, state)
    _hydrate_scenarios(context, state)
    _validate_frozen_attempt_requests(context, state)
    return state
