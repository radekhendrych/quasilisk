"""The one pure composed reducer over authoritative events."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .domain import (
    ActionPurpose,
    Application,
    ArchiveEntry,
    ChatMessage,
    ChatRequest,
    CONTROL_SAFE_DIMENSIONS,
    ConversationNode,
    DECISION_CONSTANTS,
    DomainError,
    EvidenceRecord,
    ExecutionTerminalStatus,
    FinishStatus,
    JsonSchemaValidationError,
    MemberAssessment,
    PanelAssessment,
    PanelAvailability,
    PortfolioIntent,
    ProposalAssignment,
    ProposalSlot,
    QualificationStatus,
    Role,
    RoleCallVector,
    RunTerminalReason,
    SampleExecution,
    SearchStatus,
    StructuredRoleScope,
    TargetObservation,
    VARIATION_DIMENSIONS,
    Verdict,
    VerificationStatus,
    WorkUnit,
    digest_json,
    evidence_lines,
    entity_id,
    normalize_line_endings,
    normalize_decimal,
    require_digest,
    require_entity_id,
    sha256_bytes,
    scientific_key,
    strict_json_loads,
    validate_json_schema_instance,
)
from .archive import (
    ArchiveMetric,
    ArchiveSnapshot,
    archive_metric as derive_archive_metric,
    build_archive,
    derive_archive_fitness,
    verified_cells as derive_verified_archive_cells,
)
from .evidence import (
    EntrySeedSubject,
    ExecutionEvidence,
    NamedScopeEvidence,
    ObjectiveScopeEvidence,
    PerturbatorRoleScope,
    ProviderIdentity,
    ScreeningDecision,
    ScenarioRecipe,
    ScenarioSeedSubject,
    SeedEvidenceEvent,
    SeedEvidenceSource,
    SeedQualificationContext,
    SeedQualificationDecision,
    SeedVerificationBlock,
    SeedVerificationExecution,
    SeedVerificationRule,
    VerificationDecision,
    demonstrated_only_block_passes,
    evidence_record_from_decisions,
    evaluate_screening,
    evaluate_seed_qualification,
    evaluate_verification,
    median_score,
    parse_scenario_recipe,
)
from .events import JournalCorrupt, validate_events
from .judging import (
    HIGH_THRESHOLD,
    MEMBER_ORDER,
    PanelAggregationDetails,
    ParsedJudgeOutput,
    aggregate_panel_details,
    judge_role_scope,
    member_assessment_from_output,
    panel_score_display,
    validate_judge_output,
    materialize_judge_schema,
    judge_schema_family_digest,
    should_schedule_secondary_panel,
)
from .objective import ObjectivePlugin
from .perturbation import (
    ProbeReview,
    ProposalBatch,
    ProposalItem,
    ReviewExpectation,
    SelectedRealization,
    make_proposal_slots,
    make_review_item_id,
    parse_probe_review_statuses,
    parse_probe_reviews,
    parse_proposal_batch,
    select_ordinary_realization,
    select_strategy_agnostic_realization,
    select_strategy_realization,
)
from .provider import (
    ProviderProtocolError,
    decode_bytes,
    render_request_body,
    requested_provider_route_key,
)
from .strategy import (
    BoundPortfolioLane,
    BoundStrategy,
    DevelopmentBlockDecision,
    EpisodePreferenceDecision,
    HoldoutBlockDecision,
    QualificationBlockDesign,
    QualificationRootOutcome,
    QualificationState,
    RefusalStepDecision,
    StrategyChildOutcome,
    StrategyEpisodeEvidence,
    StrategyOutcomeReview,
    StrategyRoundEvidence,
    StrategySlot,
    StrategyTemplate,
    allocate_contract_step,
    bind_portfolio_to_slots,
    build_strategy_evidence_catalog,
    build_sealed_holdout_plan,
    derive_strategy_qualification_state,
    derive_template_comparison_dimensions,
    descriptors_dict,
    evaluate_development_block,
    evaluate_holdout_block,
    evaluate_episode_preference,
    evaluate_refusal_step,
    freeze_development_arm_order,
    freeze_qualification_intent_assignment,
    make_template_binding_item_id,
    parse_outcome_reviews,
    parse_bound_strategy,
    parse_descriptors,
    parse_selector_portfolio,
    parse_sealed_holdout_plan,
    parse_strategy_template,
    parse_template_binding,
    qualified_strategy_compatibility_key,
    selector_role_scope,
    strategy_dispatch_fingerprint,
    template_matches_contract,
)


# Frozen WorkUnit conformance is reducer-owned so replay rejects an invalid
# commitment before any derived state or scheduler decision can consume it.
class SchedulerValidationError(DomainError):
    """A frozen WorkUnit violates the closed design or maximum contract."""


_WORK_KINDS = frozenset(
    {
        "initial_observation",
        "screening",
        "map_exploration",
        "portfolio_round",
        "verification",
        "strategy_development",
        "strategy_holdout",
    }
)
_POLICY_DIGEST = digest_json(dict(DECISION_CONSTANTS))
_ACTION_FIELDS = frozenset(
    {
        "action_id",
        "action_scientific_key",
        "ordinal",
        "role",
        "purpose",
        "owner_position_scientific_key",
        "dependency_action_ids",
        "routes",
        "maximum_physical_attempts",
        "guard",
    }
)
_ROUTE_FIELDS = frozenset(
    {
        "route_key",
        "route_label",
        "configured_slot",
        "requested_provider_route_key",
        "requested_model",
        "generation_profile_digest",
        "maximum_attempts",
    }
)
_PURPOSE_ROLES = {
    ActionPurpose.TARGET_SAMPLE.value: "target",
    ActionPurpose.JUDGE_MEMBER.value: "judge",
    ActionPurpose.SELECTOR_PORTFOLIO.value: "selector",
    ActionPurpose.PROBE_REVIEW.value: "selector",
    ActionPurpose.TEMPLATE_BINDING.value: "selector",
    ActionPurpose.OUTCOME_REVIEW.value: "selector",
    ActionPurpose.PERTURBATOR_PROPOSAL.value: "perturbator",
}
_ORDINARY_ROUTE_SEQUENCE = {
    "target": (("target", "primary"),),
    "selector": (
        ("selector_primary", "primary"),
        ("selector_fallback", "fallback"),
    ),
    "perturbator": (
        ("perturbator_primary", "primary"),
        ("perturbator_fallback", "fallback"),
    ),
}
_JUDGE_ROUTE_LABELS = {
    "primary": "primary",
    "secondary_a": "primary",
    "secondary_b": "primary",
    "replacement": "replacement",
}


def _exact_fields(value: Mapping[str, Any], expected: set[str] | frozenset[str], label: str) -> None:
    actual = set(value)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        detail = []
        if missing:
            detail.append(f"missing {', '.join(missing)}")
        if extra:
            detail.append(f"unknown {', '.join(extra)}")
        raise SchedulerValidationError(f"{label} fields are not exact ({'; '.join(detail)})")


def _object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchedulerValidationError(f"{label} must be an object")
    return value


def _array(value: object, label: str, *, nonempty: bool = False) -> list[Any]:
    if not isinstance(value, list):
        raise SchedulerValidationError(f"{label} must be an array")
    if nonempty and not value:
        raise SchedulerValidationError(f"{label} must be nonempty")
    return value


def _entity(value: object, prefix: str, label: str) -> str:
    try:
        result = require_entity_id(value, label)
    except DomainError as exc:
        raise SchedulerValidationError(str(exc)) from exc
    if not result.startswith(prefix + "-"):
        raise SchedulerValidationError(f"{label} must use {prefix}- prefix")
    return result


def _any_entity(value: object, label: str) -> str:
    try:
        return require_entity_id(value, label)
    except DomainError as exc:
        raise SchedulerValidationError(str(exc)) from exc


def _digest(value: object, label: str) -> str:
    try:
        return require_digest(value, label)
    except DomainError as exc:
        raise SchedulerValidationError(str(exc)) from exc


def _unique(values: Sequence[str], label: str) -> None:
    if len(set(values)) != len(values):
        raise SchedulerValidationError(f"{label} must be unique")


def _require_mapping_array(value: object, label: str) -> list[Mapping[str, Any]]:
    result = _array(value, label)
    if not all(isinstance(item, Mapping) for item in result):
        raise SchedulerValidationError(f"{label} items must be objects")
    return result  # type: ignore[return-value]


def _count(value: object, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise SchedulerValidationError(f"{label} must be an integer >= {minimum}")
    return value


def evaluated_target_maximum(count: int) -> RoleCallVector:
    count = _count(count, "evaluated target count")
    return RoleCallVector(target=2 * count, judge=8 * count)


def initial_observation_maximum() -> RoleCallVector:
    return evaluated_target_maximum(1)


def screening_maximum() -> RoleCallVector:
    # The triggering execution is block slot one and is never called again.
    return evaluated_target_maximum(2)


def map_exploration_maximum(proposal_count: int) -> RoleCallVector:
    proposal_count = _count(proposal_count, "proposal count", minimum=3)
    return RoleCallVector(
        target=2 * proposal_count,
        judge=8 * proposal_count,
        selector=4,
        perturbator=4,
    )


def portfolio_round_maximum(
    proposal_count: int,
    maximum_strategy_count: int | None = None,
) -> RoleCallVector:
    proposal_count = _count(proposal_count, "proposal count", minimum=3)
    calculated_maximum = proposal_count // 3
    if maximum_strategy_count is None:
        maximum_strategy_count = calculated_maximum
    maximum_strategy_count = _count(maximum_strategy_count, "maximum strategy count")
    if maximum_strategy_count != calculated_maximum:
        raise SchedulerValidationError("maximum strategy count must equal floor(M/3)")
    return RoleCallVector(
        target=2 * proposal_count,
        judge=8 * proposal_count,
        selector=10,
        perturbator=4 * maximum_strategy_count,
    )


def entry_verification_maximum(planned_count: int) -> RoleCallVector:
    return evaluated_target_maximum(_count(planned_count, "verification count", minimum=1))


def stateful_root_maximum(continuation_count: int) -> RoleCallVector:
    continuation_count = _count(continuation_count, "continuation count")
    return RoleCallVector(
        target=2 * (1 + continuation_count),
        judge=8 * (1 + continuation_count),
        selector=16 * continuation_count,
        perturbator=8 * continuation_count,
    )


def treatment_root_maximum() -> RoleCallVector:
    return RoleCallVector(4, 16, 16, 8)


def parent_baseline_root_maximum() -> RoleCallVector:
    return RoleCallVector(2, 8, 0, 0)


def strategy_agnostic_root_maximum() -> RoleCallVector:
    return RoleCallVector(4, 16, 8, 8)


def strategy_development_maximum(planned_count_per_arm: int) -> RoleCallVector:
    count = _count(planned_count_per_arm, "development roots per arm", minimum=1)
    return RoleCallVector(10 * count, 40 * count, 24 * count, 16 * count)


def strategy_holdout_maximum(planned_count_per_arm: int) -> RoleCallVector:
    count = _count(planned_count_per_arm, "holdout roots per arm", minimum=1)
    return RoleCallVector(8 * count, 32 * count, 24 * count, 16 * count)


def sum_root_maxima(root_maxima: Iterable[RoleCallVector]) -> RoleCallVector:
    result = RoleCallVector()
    for maximum in root_maxima:
        if not isinstance(maximum, RoleCallVector):
            raise SchedulerValidationError("root maximum is not a RoleCallVector")
        result = result + maximum
    return result


def adjust_panel_maximum(maximum: RoleCallVector, judge_attempts: int) -> RoleCallVector:
    """Scale only the judge component of the canonical full-panel formulas."""
    if judge_attempts not in {4, 8} or maximum.judge % 8:
        raise SchedulerValidationError("invalid canonical panel attempt maximum")
    return RoleCallVector(maximum.target, maximum.judge // 8 * judge_attempts,
                          maximum.selector, maximum.perturbator)


def funded_work_plan(
    config: Mapping[str, Any], bootstrap: Sequence[Mapping[str, Any]],
    seeds: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[int, RoleCallVector]]:
    """Compile finite work allocations using the canonical retry-inclusive maxima.

    Every selected entry is observed. Every potentially generated standalone
    candidate can trigger screening. Verification capacity includes every
    selected scenario seed and the configured number of newly found endpoints.
    Counts bound starts, including empty, rejected and failed batches.
    """
    plan, search, samples = config["experiment"], config["search"], config["samples"]
    entries = len(bootstrap) + sum(seed["type"] == "entry_seed" for seed in seeds)
    scenarios = sum(seed["type"] == "scenario_recipe" for seed in seeds)
    depth = search["maximum_followups"]
    for seed in seeds:
        if seed["type"] == "scenario_recipe":
            # The frozen recipe's actual graph is additionally checked on admission.
            recipe = seed["scenario_recipe"]
            if len(recipe["templates"]) > depth:
                raise SchedulerValidationError("selected scenario exceeds the funded conversation depth")
    m = search["proposals_per_step"]
    maxima = {
        "initial_observation": (entries, initial_observation_maximum()),
        "map_exploration": (plan["exploration_batches"], map_exploration_maximum(m)),
        "portfolio_round": (plan["conversation_rounds"], portfolio_round_maximum(m)),
        "screening": (entries + m * plan["exploration_batches"], screening_maximum()),
        "verification": (scenarios + plan["verification_blocks"], sum_root_maxima(
            stateful_root_maximum(depth) for _ in range(samples["verification"])
        )),
        "strategy_development": (plan["strategy_experiments"], strategy_development_maximum(samples["strategy_qualification"])),
        "strategy_holdout": (plan["strategy_experiments"], strategy_holdout_maximum(samples["holdout"])),
    }
    judges = config["models"]["judges"]
    judge_attempts = 2 * (1 + len(judges["secondaries"]) + (judges["replacement"] is not None))
    return {kind: (count, adjust_panel_maximum(maximum, judge_attempts))
            for kind, (count, maximum) in maxima.items()}


def funded_call_limits(plan: Mapping[str, tuple[int, RoleCallVector]]) -> RoleCallVector:
    return sum_root_maxima(
        RoleCallVector(**{role: count * value for role, value in maximum.to_dict().items()})
        for count, maximum in plan.values()
    )


def remaining_work_allocations(state: RunState) -> dict[str, int]:
    manifest = state.manifest
    plan = funded_work_plan(manifest["normalized_config"], manifest["bootstrap_snapshot"]["probes"], manifest["seed_snapshot"])
    return {kind: count - sum(unit.unit["kind"] == kind for unit in state.work_units)
            for kind, (count, _) in plan.items()}


def _design_count(design: Mapping[str, Any], name: str, label: str, *, minimum: int = 0) -> int:
    if name not in design:
        raise SchedulerValidationError(f"{label} design is missing {name}")
    return _count(design[name], f"{label}.{name}", minimum=minimum)


def formula_maximum(kind: str, design: Mapping[str, Any]) -> RoleCallVector:
    """Recompute a kind's closed-form maximum from its semantic design.

    Stateful scenario verification is intentionally derived from its complete
    logical-action graph unless the design supplies ``continuation_counts``;
    guessing continuation depth from outcomes would violate preregistration.
    """

    if not isinstance(design, Mapping):
        raise SchedulerValidationError("work-unit design must be an object")
    tag = design.get("tag")
    if tag is not None and tag != kind:
        raise SchedulerValidationError("work-unit kind and design tag disagree")
    if kind == "initial_observation":
        return initial_observation_maximum()
    if kind == "screening":
        return screening_maximum()
    if kind == "map_exploration":
        return map_exploration_maximum(_design_count(design, "proposal_count", kind, minimum=1))
    if kind == "portfolio_round":
        round_index = _design_count(design, "round_index", kind)
        if round_index != 0:
            raise SchedulerValidationError("portfolio round_index must be 0")
        return portfolio_round_maximum(
            _design_count(design, "proposal_count", kind, minimum=3),
            _design_count(design, "maximum_strategy_count", kind),
        )
    if kind == "verification":
        planned = _design_count(design, "planned_count", kind, minimum=1)
        subject_kind = design.get("subject_kind")
        if subject_kind == "entry_prompt":
            return entry_verification_maximum(planned)
        if subject_kind != "scenario_recipe":
            raise SchedulerValidationError("verification subject_kind is invalid")
        raise SchedulerValidationError(
            "scenario verification maximum requires its frozen logical-action graph"
        )
    if kind == "strategy_development":
        return strategy_development_maximum(
            _design_count(design, "planned_count_per_arm", kind, minimum=1)
        )
    if kind == "strategy_holdout":
        plan = design.get("sealed_holdout_plan")
        if not isinstance(plan, Mapping):
            raise SchedulerValidationError("holdout design is missing sealed_holdout_plan")
        return strategy_holdout_maximum(
            _design_count(plan, "planned_count_per_arm", "sealed_holdout_plan", minimum=1)
        )
    raise SchedulerValidationError(f"unknown work-unit kind: {kind}")


def action_slot_maximum(logical_action_slots: Sequence[Mapping[str, Any]]) -> RoleCallVector:
    """Recompute maximum calls from a fully materialized finite action graph."""

    if not isinstance(logical_action_slots, Sequence) or isinstance(
        logical_action_slots, (str, bytes)
    ):
        raise SchedulerValidationError("logical action slots must be an array")
    if not logical_action_slots:
        raise SchedulerValidationError("logical action slots must be nonempty")
    counts = {"target": 0, "judge": 0, "selector": 0, "perturbator": 0}
    seen_ids: set[str] = set()
    seen_keys: set[str] = set()
    for expected_ordinal, slot in enumerate(logical_action_slots):
        if not isinstance(slot, Mapping):
            raise SchedulerValidationError("logical action slot must be an object")
        _exact_fields(slot, _ACTION_FIELDS, "logical action slot")
        ordinal = _count(slot.get("ordinal"), "logical action ordinal")
        if ordinal != expected_ordinal:
            raise SchedulerValidationError("logical action ordinals must be contiguous")
        action_id = _entity(slot.get("action_id"), "la", "logical action ID")
        action_key = _digest(
            slot.get("action_scientific_key"), "logical action scientific key"
        )
        if action_id in seen_ids:
            raise SchedulerValidationError("logical action IDs must be unique")
        if action_key in seen_keys:
            raise SchedulerValidationError("logical action scientific keys must be unique")
        seen_ids.add(action_id)
        seen_keys.add(action_key)
        dependencies = slot.get("dependency_action_ids")
        if (
            not isinstance(dependencies, list)
            or not all(isinstance(item, str) for item in dependencies)
            or len(set(dependencies)) != len(dependencies)
        ):
            raise SchedulerValidationError("logical action dependencies must be a duplicate-free array")
        if any(item not in seen_ids or item == action_id for item in dependencies):
            raise SchedulerValidationError("logical action dependency must name an earlier slot")
        role = slot.get("role")
        if role not in counts:
            raise SchedulerValidationError("logical action role is invalid")
        purpose = slot.get("purpose")
        if purpose not in _PURPOSE_ROLES or _PURPOSE_ROLES[purpose] != role:
            raise SchedulerValidationError("logical action purpose does not belong to its role")
        owner_key = _digest(
            slot.get("owner_position_scientific_key"),
            "logical action owner-position scientific key",
        )
        if action_key != scientific_key(
            [owner_key, "action", role, purpose, ordinal]
        ):
            raise SchedulerValidationError(
                "logical action scientific key disagrees with its owner, role, purpose, and ordinal"
            )
        attempts = _count(
            slot.get("maximum_physical_attempts"),
            "maximum physical attempts",
            minimum=1,
        )
        routes = slot.get("routes")
        if not isinstance(routes, list) or not routes:
            raise SchedulerValidationError("logical action routes must be nonempty")
        route_attempts = 0
        route_keys: set[str] = set()
        for route in routes:
            if not isinstance(route, Mapping):
                raise SchedulerValidationError("role route binding must be an object")
            _exact_fields(route, _ROUTE_FIELDS, "role route binding")
            maximum_attempts = _count(
                route.get("maximum_attempts"),
                "route maximum attempts",
                minimum=1,
            )
            if maximum_attempts not in {1, 2}:
                raise SchedulerValidationError("a route permits only one or two attempts")
            route_attempts += maximum_attempts
            route_key = _digest(route.get("route_key"), "role route key")
            if route_key in route_keys:
                raise SchedulerValidationError("logical action route keys must be unique")
            route_keys.add(route_key)
            if route.get("route_label") not in {"primary", "replacement", "fallback"}:
                raise SchedulerValidationError("role route label is invalid")
            configured_slot = route.get("configured_slot")
            if configured_slot not in {
                "target",
                "primary",
                "secondary_a",
                "secondary_b",
                "replacement",
                "selector_primary",
                "selector_fallback",
                "perturbator_primary",
                "perturbator_fallback",
            }:
                raise SchedulerValidationError("configured role route slot is invalid")
            _digest(
                route.get("requested_provider_route_key"),
                "requested provider route key",
            )
            requested_model = route.get("requested_model")
            if not isinstance(requested_model, str) or not requested_model:
                raise SchedulerValidationError("requested route model must be nonempty")
            _digest(
                route.get("generation_profile_digest"),
                "generation profile digest",
            )
        if attempts != route_attempts:
            raise SchedulerValidationError("logical action attempt cap disagrees with its routes")
        role_cap = 4 if role in {"selector", "perturbator"} else 2
        if attempts > role_cap:
            raise SchedulerValidationError("logical action exceeds its role attempt cap")
        _object(slot.get("guard"), "logical action guard")
        counts[role] += attempts
    return RoleCallVector(**counts)


def _validate_proposal_slots(unit: WorkUnit) -> tuple[Mapping[str, Any], ...]:
    slots = tuple(unit.proposal_slots)
    ids: list[str] = []
    keys: list[str] = []
    for ordinal, raw in enumerate(slots):
        slot = _object(raw, "proposal slot")
        _exact_fields(slot, {"slot_id", "scientific_key", "global_ordinal"}, "proposal slot")
        slot_id = _entity(slot.get("slot_id"), "ps", "proposal slot ID")
        key = _digest(slot.get("scientific_key"), "proposal slot scientific key")
        if _count(slot.get("global_ordinal"), "proposal slot ordinal") != ordinal:
            raise SchedulerValidationError("proposal slot ordinals must be contiguous")
        expected_key = scientific_key(
            [unit.unit_scientific_key, "proposal_slot", ordinal]
        )
        expected_id = "ps-" + hashlib.sha256(expected_key.encode("utf-8")).hexdigest()
        if key != expected_key or slot_id != expected_id:
            raise SchedulerValidationError(
                "proposal slot identity disagrees with its work unit and ordinal"
            )
        ids.append(slot_id)
        keys.append(key)
    _unique(ids, "proposal slot IDs")
    _unique(keys, "proposal slot scientific keys")
    return slots


def _validate_execution_slots(unit: WorkUnit) -> tuple[Mapping[str, Any], ...]:
    slots = tuple(unit.execution_slots)
    ids: list[str] = []
    keys: list[str] = []
    root_ids: list[str] = []
    for ordinal, raw in enumerate(slots):
        slot = _object(raw, "execution slot")
        _exact_fields(
            slot,
            {
                "execution_id",
                "execution_scientific_key",
                "ordinal",
                "arm",
                "root_plan_id",
            },
            "execution slot",
        )
        ids.append(_entity(slot.get("execution_id"), "ex", "execution ID"))
        keys.append(
            _digest(
                slot.get("execution_scientific_key"),
                "execution scientific key",
            )
        )
        if _count(slot.get("ordinal"), "execution ordinal") != ordinal:
            raise SchedulerValidationError("execution ordinals must be contiguous")
        if slot.get("arm") not in {
            "none",
            "treatment",
            "parent_baseline",
            "strategy_agnostic",
        }:
            raise SchedulerValidationError("execution arm is invalid")
        root_id = slot.get("root_plan_id")
        if root_id is not None:
            root_ids.append(_entity(root_id, "rp", "execution root-plan ID"))
    _unique(ids, "execution IDs")
    _unique(keys, "execution scientific keys")
    _unique(root_ids, "execution root-plan IDs")
    return slots


def _validate_ordinary_routes(
    slot: Mapping[str, Any], *, unit_kind: str
) -> None:
    role = slot["role"]
    routes = slot["routes"]
    if role == "judge":
        if len(routes) != 1:
            raise SchedulerValidationError("ordinary judge action must have one route")
        configured = routes[0]["configured_slot"]
        if configured not in _JUDGE_ROUTE_LABELS or routes[0]["route_label"] != _JUDGE_ROUTE_LABELS[configured]:
            raise SchedulerValidationError("judge route slot and label disagree")
    else:
        expected = (
            (("selector_primary", "primary"),)
            if unit_kind == "portfolio_round"
            and slot["purpose"] == ActionPurpose.PROBE_REVIEW.value
            else _ORDINARY_ROUTE_SEQUENCE[role]
        )
        actual = tuple(
            (route["configured_slot"], route["route_label"]) for route in routes
        )
        if actual != expected:
            raise SchedulerValidationError(
                f"{role} routes do not match their configured sequence"
            )
    if any(route["maximum_attempts"] != 2 for route in routes):
        raise SchedulerValidationError(
            "structured runtime routes must allow exactly two attempts"
        )


_GUARD_FIELDS = {
    "always": {"kind"},
    "if_action_usable": {"kind", "action_id"},
    "if_panel_resolved": {"kind", "panel_id"},
    "if_template_applicable": {"kind", "binding_action_id"},
    "if_strategy_slot_bound": {"kind", "strategy_slot_scientific_key"},
    "if_dispatchable_probe": {
        "kind",
        "review_item_id",
        "proposal_slot_scientific_key",
        "allowed",
    },
    "if_repairable_proposal_exists": {
        "kind",
        "review_action_id",
        "proposal_slot_scientific_keys",
        "statuses",
    },
    "if_any_proposed_item": {"kind", "proposal_action_ids"},
    "if_any_corrective_proposal": {"kind", "proposal_action_ids"},
    "if_any_aligned_response": {"kind", "target_action_ids"},
    "if_panel_triggered": {"kind", "panel_id"},
    "if_panel_has_unresolved_normal_member": {"kind", "panel_id"},
}


def _guard_action_references(
    guard: Mapping[str, Any],
) -> tuple[str, ...]:
    kind = guard["kind"]
    if kind == "if_action_usable":
        return (guard["action_id"],)
    if kind == "if_template_applicable":
        return (guard["binding_action_id"],)
    if kind == "if_repairable_proposal_exists":
        return (guard["review_action_id"],)
    if kind in {"if_any_proposed_item", "if_any_corrective_proposal"}:
        return tuple(guard["proposal_action_ids"])
    if kind == "if_any_aligned_response":
        return tuple(guard["target_action_ids"])
    return ()


def _validate_guards(
    unit: WorkUnit,
    action_by_id: Mapping[str, Mapping[str, Any]],
    panel_by_id: Mapping[str, Mapping[str, Any]],
    proposal_keys: frozenset[str],
    strategy_keys: frozenset[str],
) -> None:
    ordinal_by_id = {
        action_id: action["ordinal"] for action_id, action in action_by_id.items()
    }
    for slot in unit.logical_action_slots:
        guard = _object(slot["guard"], "logical action guard")
        kind = guard.get("kind")
        if kind not in _GUARD_FIELDS:
            raise SchedulerValidationError("logical action guard kind is invalid")
        _exact_fields(guard, _GUARD_FIELDS[kind], f"{kind} guard")
        references = _guard_action_references(guard)
        if len(set(references)) != len(references):
            raise SchedulerValidationError("guard action references must be unique")
        for action_id in references:
            _entity(action_id, "la", "guard action ID")
            referenced = action_by_id.get(action_id)
            if referenced is None or ordinal_by_id[action_id] >= slot["ordinal"]:
                raise SchedulerValidationError(
                    "guard action reference must name an earlier same-unit slot"
                )
            if action_id not in slot["dependency_action_ids"]:
                raise SchedulerValidationError(
                    "guard action reference must be an explicit dependency"
                )
        if kind in {"if_any_proposed_item", "if_any_corrective_proposal"}:
            if not references or any(
                action_by_id[action_id]["purpose"] != "perturbator_proposal"
                for action_id in references
            ):
                raise SchedulerValidationError(
                    "proposal guard must reference Perturbator proposal actions"
                )
        elif kind == "if_any_aligned_response":
            if not references or any(
                action_by_id[action_id]["purpose"] != "target_sample"
                for action_id in references
            ):
                raise SchedulerValidationError(
                    "aligned-response guard must reference target actions"
                )
        elif kind == "if_template_applicable":
            if action_by_id[references[0]]["purpose"] != "template_binding":
                raise SchedulerValidationError(
                    "template-applicable guard must reference a binding action"
                )
        elif kind == "if_repairable_proposal_exists":
            if action_by_id[references[0]]["purpose"] != "probe_review":
                raise SchedulerValidationError(
                    "repair guard must reference a probe-review action"
                )
            keys = _array(
                guard["proposal_slot_scientific_keys"],
                "repair-guard proposal keys",
                nonempty=True,
            )
            if not all(isinstance(item, str) for item in keys) or len(set(keys)) != len(keys):
                raise SchedulerValidationError(
                    "repair-guard proposal keys must be unique strings"
                )
            if not set(keys) <= proposal_keys:
                raise SchedulerValidationError(
                    "repair guard names an unknown proposal slot"
                )
            if guard["statuses"] != ["invalid"]:
                raise SchedulerValidationError("repair-guard statuses are invalid")
        elif kind == "if_dispatchable_probe":
            review_item_id = _entity(
                guard["review_item_id"], "ri", "dispatch guard review-item ID"
            )
            key = _digest(
                guard["proposal_slot_scientific_key"],
                "dispatch guard proposal-slot key",
            )
            if key not in proposal_keys:
                raise SchedulerValidationError(
                    "dispatch guard names an unknown proposal slot"
                )
            expected_review_item_id = entity_id(
                "ri", ["review_item", unit.unit_scientific_key, "probe", key]
            )
            if review_item_id != expected_review_item_id:
                raise SchedulerValidationError(
                    "dispatch guard review-item ID disagrees with its proposal slot"
                )
            if guard["allowed"] not in (
                ["valid"],
                ["aligned"],
                ["aligned"],
            ):
                raise SchedulerValidationError("dispatch-guard allowed statuses are invalid")
        elif kind == "if_strategy_slot_bound":
            key = _digest(
                guard["strategy_slot_scientific_key"],
                "strategy-slot guard key",
            )
            if key not in strategy_keys:
                raise SchedulerValidationError(
                    "strategy-slot guard names an unknown frozen lane"
                )
        elif kind in {
            "if_panel_triggered",
            "if_panel_has_unresolved_normal_member",
            "if_panel_resolved",
        }:
            panel_id = _entity(guard["panel_id"], "pn", "guard panel ID")
            if panel_id not in panel_by_id:
                raise SchedulerValidationError("panel guard names an unknown panel")
            if kind == "if_panel_resolved" and not set(
                _panel_action_ids(panel_by_id[panel_id])
            ) <= set(slot["dependency_action_ids"]):
                raise SchedulerValidationError("resolved-panel guard requires the complete panel as dependencies")


def _validate_panels(
    unit: WorkUnit,
    action_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    panels = tuple(unit.panel_plans)
    ids: list[str] = []
    keys: list[str] = []
    target_ids: list[str] = []
    judge_ids: list[str] = []
    for raw in panels:
        panel = _object(raw, "panel plan")
        _exact_fields(
            panel,
            {
                "panel_id",
                "panel_scientific_key",
                "target_action_id",
                "trigger_policy",
                "normal_members",
                "replacement",
            },
            "panel plan",
        )
        panel_id = _entity(panel["panel_id"], "pn", "panel ID")
        panel_key = _digest(panel["panel_scientific_key"], "panel scientific key")
        target_id = _entity(panel["target_action_id"], "la", "panel target action ID")
        target = action_by_id.get(target_id)
        if target is None or target["purpose"] != "target_sample":
            raise SchedulerValidationError("panel target must name a target-sample action")
        if panel_key != scientific_key([target["action_scientific_key"], "assessment_panel"]):
            raise SchedulerValidationError("panel scientific key disagrees with its target")
        if panel["trigger_policy"] not in {
            "threshold_or_primary_failure",
            "always_full",
        }:
            raise SchedulerValidationError("panel trigger policy is invalid")
        normal = _require_mapping_array(panel["normal_members"], "normal panel members")
        if len(normal) not in {2, 3}:
            raise SchedulerValidationError("a panel must contain two or three normal members")
        if (len(normal) == 2) != (panel["replacement"] is None):
            raise SchedulerValidationError("two-member panels have no replacement; three-member panels require one")
        expected_members = (("primary", 3), ("secondary_a", 2), ("secondary_b", 2))[:len(normal)]
        normal_actions: list[Mapping[str, Any]] = []
        for member, (configured_slot, weight) in zip(normal, expected_members, strict=True):
            _exact_fields(
                member,
                {
                    "logical_member_slot",
                    "weight",
                    "judge_action_id",
                    "judge_action_scientific_key",
                    "role_route_key",
                },
                "normal panel member",
            )
            if member["logical_member_slot"] != configured_slot or member["weight"] != weight:
                raise SchedulerValidationError("normal panel member order or weight is invalid")
            judge_id = _entity(member["judge_action_id"], "la", "panel judge action ID")
            judge = action_by_id.get(judge_id)
            if judge is None or judge["role"] != "judge" or judge["purpose"] != "judge_member":
                raise SchedulerValidationError("panel member must name a judge-member action")
            if judge["owner_position_scientific_key"] != target["owner_position_scientific_key"]:
                raise SchedulerValidationError("panel actions must share the target execution owner")
            if member["judge_action_scientific_key"] != judge["action_scientific_key"]:
                raise SchedulerValidationError("panel member action key does not match its action")
            route = judge["routes"]
            if len(route) != 1 or route[0]["configured_slot"] != configured_slot:
                raise SchedulerValidationError("panel member uses the wrong configured judge slot")
            if member["role_route_key"] != route[0]["route_key"]:
                raise SchedulerValidationError("panel member cites the wrong role route")
            normal_actions.append(judge)
            judge_ids.append(judge_id)
        replacement_action = None
        if panel["replacement"] is not None:
            replacement = _object(panel["replacement"], "panel replacement")
            _exact_fields(
                replacement,
                {
                    "judge_action_id",
                    "judge_action_scientific_key",
                    "role_route_key",
                },
                "panel replacement",
            )
            replacement_id = _entity(
                replacement["judge_action_id"], "la", "replacement action ID"
            )
            replacement_action = action_by_id.get(replacement_id)
            if (
                replacement_action is None
                or replacement_action["role"] != "judge"
                or replacement_action["purpose"] != "judge_member"
                or replacement_action["owner_position_scientific_key"]
                != target["owner_position_scientific_key"]
            ):
                raise SchedulerValidationError("panel replacement must name its execution's judge action")
            if replacement["judge_action_scientific_key"] != replacement_action["action_scientific_key"]:
                raise SchedulerValidationError("replacement action key does not match its action")
            replacement_routes = replacement_action["routes"]
            if len(replacement_routes) != 1 or replacement_routes[0]["configured_slot"] != "replacement":
                raise SchedulerValidationError("panel replacement uses the wrong judge route")
            if replacement["role_route_key"] != replacement_routes[0]["route_key"]:
                raise SchedulerValidationError("panel replacement cites the wrong role route")
            judge_ids.append(replacement_id)

        primary, *secondaries = normal_actions
        if primary["dependency_action_ids"] != [target_id] or primary["guard"] != {
            "kind": "if_action_usable",
            "action_id": target_id,
        }:
            raise SchedulerValidationError("panel primary topology is invalid")
        for secondary in secondaries:
            if secondary["dependency_action_ids"] != [primary["action_id"]] or secondary["guard"] != {
                "kind": "if_panel_triggered",
                "panel_id": panel_id,
            }:
                raise SchedulerValidationError("panel secondary topology is invalid")
        normal_ids = [item["action_id"] for item in normal_actions]
        if replacement_action is not None and (replacement_action["dependency_action_ids"] != normal_ids or replacement_action["guard"] != {
            "kind": "if_panel_has_unresolved_normal_member",
            "panel_id": panel_id,
        }):
            raise SchedulerValidationError("panel replacement topology is invalid")
        ids.append(panel_id)
        keys.append(panel_key)
        target_ids.append(target_id)
    _unique(ids, "panel IDs")
    _unique(keys, "panel scientific keys")
    _unique(target_ids, "panel target action IDs")
    _unique(judge_ids, "panel judge action IDs")
    expected_targets = [
        slot["action_id"]
        for slot in unit.logical_action_slots
        if slot["purpose"] == "target_sample"
    ]
    if target_ids != expected_targets:
        raise SchedulerValidationError(
            "panels must appear once in target-action order for every assessed target"
        )
    expected_judges = [
        slot["action_id"]
        for slot in unit.logical_action_slots
        if slot["purpose"] == "judge_member"
    ]
    if judge_ids != expected_judges:
        raise SchedulerValidationError(
            "every judge action must belong to exactly one ordered panel"
        )
    return panels


_ROOT_PLAN_FIELDS = frozenset(
    {
        "root_plan_id",
        "execution_id",
        "arm",
        "ordered_action_ids",
        "terminal_endpoint_action_id",
        "semantic_nonsuccess_reasons",
    }
)


def _validate_root_plans(
    unit: WorkUnit,
    value: object,
    action_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    plans = tuple(_require_mapping_array(value, "root plans"))
    executions = {slot["execution_id"]: slot for slot in unit.execution_slots}
    plan_ids: list[str] = []
    execution_ids: list[str] = []
    action_owner: dict[str, str] = {}
    for raw in plans:
        plan = _object(raw, "root plan")
        _exact_fields(plan, _ROOT_PLAN_FIELDS, "root plan")
        plan_id = _entity(plan["root_plan_id"], "rp", "root-plan ID")
        execution_id = _entity(plan["execution_id"], "ex", "root-plan execution ID")
        execution = executions.get(execution_id)
        if execution is None:
            raise SchedulerValidationError("root plan names an unknown execution")
        if execution["root_plan_id"] != plan_id or execution["arm"] != plan["arm"]:
            raise SchedulerValidationError("root plan and execution linkage disagree")
        action_ids = _array(plan["ordered_action_ids"], "root-plan action IDs")
        if not all(isinstance(item, str) for item in action_ids) or len(set(action_ids)) != len(action_ids):
            raise SchedulerValidationError("root-plan action IDs must be unique strings")
        prior: set[str] = set()
        prior_ordinal = -1
        for action_id in action_ids:
            _entity(action_id, "la", "root-plan action ID")
            action = action_by_id.get(action_id)
            if action is None:
                raise SchedulerValidationError("root plan names an unknown same-unit action")
            if action["owner_position_scientific_key"] != execution["execution_scientific_key"]:
                raise SchedulerValidationError("root-plan action has the wrong execution owner")
            if action["ordinal"] <= prior_ordinal:
                raise SchedulerValidationError("root-plan action order is not dependency respecting")
            if not set(action["dependency_action_ids"]) <= prior:
                raise SchedulerValidationError("root-plan action depends outside its earlier plan")
            previous_owner = action_owner.setdefault(action_id, plan_id)
            if previous_owner != plan_id:
                raise SchedulerValidationError("a logical action belongs to multiple root plans")
            prior.add(action_id)
            prior_ordinal = action["ordinal"]
        terminal = plan["terminal_endpoint_action_id"]
        reasons = _array(
            plan["semantic_nonsuccess_reasons"],
            "root semantic nonsuccess reasons",
        )
        if (
            any(item not in {"inapplicable", "nonprogress"} for item in reasons)
            or len(set(reasons)) != len(reasons)
        ):
            raise SchedulerValidationError("root semantic nonsuccess reasons are invalid")
        if terminal is None:
            if action_ids or not reasons:
                raise SchedulerValidationError(
                    "a null terminal is permitted only for an empty pretarget semantic-nonsuccess plan"
                )
        else:
            _entity(terminal, "la", "root terminal action ID")
            if terminal not in action_ids or action_by_id[terminal]["purpose"] != "target_sample":
                raise SchedulerValidationError("root terminal must name one of its target actions")
        plan_ids.append(plan_id)
        execution_ids.append(execution_id)
    _unique(plan_ids, "root-plan IDs")
    _unique(execution_ids, "root-plan execution IDs")
    if len(plans) != len(unit.execution_slots):
        raise SchedulerValidationError("every planned execution must have exactly one root plan")
    if set(execution_ids) != set(executions):
        raise SchedulerValidationError("root plans do not cover every execution")
    return plans


_SEALED_PLAN_FIELDS = frozenset(
    {
        "planned_count_per_arm",
        "comparison_dimensions",
        "future_roots",
        "interleaved_root_order",
        "information_masks",
        "endpoint_rule",
        "measurement_cell",
    }
)
_FUTURE_ROOT_FIELDS = frozenset(
    {
        "root_ordinal",
        "arm_local_ordinal",
        "arm",
        "realization_intent",
        "blueprint",
        "semantic_nonsuccess_reasons",
    }
)


def _validate_dimensions(value: object, label: str, *, control_safe: bool) -> tuple[str, ...]:
    dimensions = _array(value, label, nonempty=True)
    allowed = set(CONTROL_SAFE_DIMENSIONS if control_safe else VARIATION_DIMENSIONS)
    if (
        not all(isinstance(item, str) and item in allowed for item in dimensions)
        or len(set(dimensions)) != len(dimensions)
    ):
        raise SchedulerValidationError(f"{label} must be a unique closed dimension list")
    canonical = [item for item in VARIATION_DIMENSIONS if item in dimensions]
    if dimensions != canonical:
        raise SchedulerValidationError(f"{label} must use canonical dimension order")
    return tuple(dimensions)


def _validate_sealed_holdout_plan(value: object) -> Mapping[str, Any]:
    plan = _object(value, "sealed holdout plan")
    _exact_fields(plan, _SEALED_PLAN_FIELDS, "sealed holdout plan")
    count = _count(
        plan["planned_count_per_arm"],
        "sealed holdout roots per arm",
        minimum=5,
    )
    _validate_dimensions(
        plan["comparison_dimensions"],
        "sealed holdout comparison dimensions",
        control_safe=True,
    )
    future_roots = _require_mapping_array(
        plan["future_roots"], "sealed holdout future roots"
    )
    if len(future_roots) != 2 * count:
        raise SchedulerValidationError("sealed holdout must contain exactly 2H future roots")
    per_arm_ordinals: dict[str, list[int]] = {
        "treatment": [],
        "strategy_agnostic": [],
    }
    for ordinal, future in enumerate(future_roots):
        _exact_fields(future, _FUTURE_ROOT_FIELDS, "future holdout root")
        if _count(future["root_ordinal"], "future root ordinal") != ordinal:
            raise SchedulerValidationError("future root ordinals must be contiguous")
        arm = future["arm"]
        if arm not in per_arm_ordinals:
            raise SchedulerValidationError("future holdout arm is invalid")
        local = _count(future["arm_local_ordinal"], "future arm-local ordinal")
        per_arm_ordinals[arm].append(local)
        if future["realization_intent"] not in {"anchor", "close", "divergent"}:
            raise SchedulerValidationError("future realization intent is invalid")
        expected_blueprint = (
            "treatment_qualification"
            if arm == "treatment"
            else "strategy_agnostic_qualification"
        )
        expected_reasons = ["inapplicable"] if arm == "treatment" else []
        if (
            future["blueprint"] != expected_blueprint
            or future["semantic_nonsuccess_reasons"] != expected_reasons
        ):
            raise SchedulerValidationError("future holdout arm blueprint is invalid")
    for arm, ordinals in per_arm_ordinals.items():
        if sorted(ordinals) != list(range(count)) or len(ordinals) != count:
            raise SchedulerValidationError(
                f"sealed holdout {arm} arm-local ordinals are not contiguous"
            )
    interleaved = _array(
        plan["interleaved_root_order"],
        "sealed holdout interleaved order",
    )
    if (
        not all(isinstance(item, int) and not isinstance(item, bool) for item in interleaved)
        or sorted(interleaved) != list(range(2 * count))
    ):
        raise SchedulerValidationError(
            "sealed holdout interleaved order must permute every future root"
        )
    masks = _object(plan["information_masks"], "sealed holdout information masks")
    if masks != {
        "treatment": "bound_strategy_request",
        "strategy_agnostic": "response_aware_strategy_blind_request",
    }:
        raise SchedulerValidationError("sealed holdout information masks are invalid")
    if plan["endpoint_rule"] != "single_terminal_assessed_response":
        raise SchedulerValidationError("sealed holdout endpoint rule is invalid")
    _object(plan["measurement_cell"], "sealed holdout measurement cell")
    return plan


def _compressed(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if not result or result[-1] != value:
            result.append(value)
    return result


_DESIGN_FIELDS = {
    "initial_observation": {
        "tag",
        "candidate_id",
        "candidate_scientific_key",
        "execution_id",
        "execution_scientific_key",
        "panel_policy_digest",
    },
    "screening": {
        "tag",
        "candidate_id",
        "candidate_scientific_key",
        "block_id",
        "triggering_execution_id",
        "new_execution_ids",
        "panel_policy_digest",
    },
    "map_exploration": {
        "tag",
        "parent_archive_entry_id",
        "parent_archive_entry_scientific_key",
        "proposal_count",
        "under_covered_cells",
        "allowed_dimensions",
        "probe_review_scope_key",
    },
    "portfolio_round": {
        "tag",
        "portfolio_intent",
        "conversation_root_id",
        "generation_parent_node_id",
        "generation_parent_node_scientific_key",
        "round_index",
        "proposal_count",
        "maximum_strategy_count",
        "strategy_slots",
        "active_strategy_ids",
        "frozen_assignments",
        "review_scope_keys",
    },
    "verification": {
        "tag",
        "subject_kind",
        "subject_id",
        "subject_scientific_key",
        "planned_count",
        "measurement_cell",
        "root_plans",
    },
    "strategy_development": {
        "tag",
        "strategy_template_id",
        "strategy_template_scientific_key",
        "planned_count_per_arm",
        "comparison_dimensions",
        "realization_intent_assignment",
        "interleaved_root_order",
        "root_plans",
        "sealed_holdout_plan",
        "sealed_holdout_digest",
        "measurement_cell",
    },
    "strategy_holdout": {
        "tag",
        "strategy_template_id",
        "strategy_template_scientific_key",
        "source_development_unit_id",
        "source_sealed_holdout_digest",
        "sealed_holdout_plan",
        "root_plans",
    },
}


def _panel_action_ids(panel: Mapping[str, Any]) -> list[str]:
    return [
        panel["target_action_id"],
        *(member["judge_action_id"] for member in panel["normal_members"]),
        *([panel["replacement"]["judge_action_id"]] if panel["replacement"] is not None else []),
    ]


def _expect_action(
    slot: Mapping[str, Any],
    *,
    role: str,
    purpose: str,
    owner: str,
    dependencies: Sequence[str] | None = None,
    guard: Mapping[str, Any] | None = None,
) -> None:
    if (
        slot["role"] != role
        or slot["purpose"] != purpose
        or slot["owner_position_scientific_key"] != owner
    ):
        raise SchedulerValidationError("logical action role, purpose, or owner is misplaced")
    if dependencies is not None and slot["dependency_action_ids"] != list(dependencies):
        raise SchedulerValidationError("logical action dependencies do not match its frozen topology")
    if guard is not None and slot["guard"] != dict(guard):
        raise SchedulerValidationError("logical action guard does not match its frozen topology")


def _execution_key(
    unit: WorkUnit,
    ordinal: int,
    proposal_keys: Sequence[str] = (),
) -> str:
    return scientific_key(
        [unit.unit_scientific_key, "execution", ordinal, list(proposal_keys)]
    )


def _validate_execution_identity(
    unit: WorkUnit,
    execution: Mapping[str, Any],
    *,
    proposal_keys: Sequence[str] = (),
) -> None:
    expected = _execution_key(unit, execution["ordinal"], proposal_keys)
    if execution["execution_scientific_key"] != expected:
        raise SchedulerValidationError("execution scientific key disagrees with its frozen path")


def _validate_panel_execution_pair(
    execution: Mapping[str, Any], panel: Mapping[str, Any], action_by_id: Mapping[str, Mapping[str, Any]]
) -> None:
    target = action_by_id[panel["target_action_id"]]
    if target["owner_position_scientific_key"] != execution["execution_scientific_key"]:
        raise SchedulerValidationError("panel target is owned by the wrong execution")


def _validate_initial_observation(
    unit: WorkUnit,
    action_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    design = unit.design
    _exact_fields(design, _DESIGN_FIELDS[unit.kind], "initial-observation design")
    if unit.parent_unit_id is not None or unit.proposal_slots:
        raise SchedulerValidationError("initial observation must be a root unit without proposal slots")
    if (
        design["candidate_id"] != unit.subject_id
        or design["candidate_scientific_key"] != unit.subject_scientific_key
    ):
        raise SchedulerValidationError("initial-observation candidate identity disagrees with its subject")
    _entity(design["candidate_id"], "pc", "initial candidate ID")
    if _digest(design["panel_policy_digest"], "initial panel-policy digest") != unit.policy_digest:
        raise SchedulerValidationError("initial panel policy disagrees with the unit policy")
    if len(unit.execution_slots) != 1 or len(unit.panel_plans) != 1:
        raise SchedulerValidationError("initial observation requires one execution and one panel")
    execution = unit.execution_slots[0]
    panel = unit.panel_plans[0]
    if (
        execution["arm"] != "none"
        or execution["root_plan_id"] is not None
        or design["execution_id"] != execution["execution_id"]
        or design["execution_scientific_key"] != execution["execution_scientific_key"]
    ):
        raise SchedulerValidationError("initial-observation execution linkage is invalid")
    _validate_execution_identity(unit, execution)
    _validate_panel_execution_pair(execution, panel, action_by_id)
    if _panel_action_ids(panel) != [slot["action_id"] for slot in unit.logical_action_slots]:
        raise SchedulerValidationError("initial-observation action graph is not exactly one panel")


def _validate_screening(
    unit: WorkUnit,
    action_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    design = unit.design
    _exact_fields(design, _DESIGN_FIELDS[unit.kind], "screening design")
    if unit.parent_unit_id is None or unit.proposal_slots:
        raise SchedulerValidationError("screening must cite a parent and contain no proposal slots")
    if (
        design["candidate_id"] != unit.subject_id
        or design["candidate_scientific_key"] != unit.subject_scientific_key
    ):
        raise SchedulerValidationError("screening candidate identity disagrees with its subject")
    _entity(design["candidate_id"], "pc", "screening candidate ID")
    _entity(design["block_id"], "sb", "screening block ID")
    triggering = _entity(
        design["triggering_execution_id"], "ex", "triggering screening execution ID"
    )
    new_ids = _array(design["new_execution_ids"], "new screening execution IDs")
    if len(new_ids) != 2 or len(set(new_ids)) != 2:
        raise SchedulerValidationError("screening requires exactly two distinct new executions")
    if triggering in new_ids:
        raise SchedulerValidationError("screening must not call its triggering execution again")
    if _digest(design["panel_policy_digest"], "screening panel-policy digest") != unit.policy_digest:
        raise SchedulerValidationError("screening panel policy disagrees with the unit policy")
    if len(unit.execution_slots) != 2 or len(unit.panel_plans) != 2:
        raise SchedulerValidationError("screening requires two executions and two panels")
    if new_ids != [slot["execution_id"] for slot in unit.execution_slots]:
        raise SchedulerValidationError("screening new-execution IDs disagree with its slots")
    flattened: list[str] = []
    for execution, panel in zip(unit.execution_slots, unit.panel_plans, strict=True):
        if execution["arm"] != "none" or execution["root_plan_id"] is not None:
            raise SchedulerValidationError("screening execution arm or root plan is invalid")
        _validate_execution_identity(unit, execution)
        _validate_panel_execution_pair(execution, panel, action_by_id)
        flattened.extend(_panel_action_ids(panel))
    if flattened != [slot["action_id"] for slot in unit.logical_action_slots]:
        raise SchedulerValidationError("screening action graph must contain exactly its two panels")


def _validate_map_exploration(
    unit: WorkUnit,
    action_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    design = unit.design
    _exact_fields(design, _DESIGN_FIELDS[unit.kind], "MAP-exploration design")
    count = _design_count(design, "proposal_count", unit.kind, minimum=3)
    if unit.parent_unit_id is None:
        raise SchedulerValidationError("MAP exploration must cite its parent unit")
    if (
        design["parent_archive_entry_id"] != unit.subject_id
        or design["parent_archive_entry_scientific_key"] != unit.subject_scientific_key
    ):
        raise SchedulerValidationError("MAP parent identity disagrees with its subject")
    _entity(design["parent_archive_entry_id"], "ae", "MAP parent archive-entry ID")
    cells = _array(design["under_covered_cells"], "under-covered MAP cells", nonempty=True)
    if len(cells) > count or not all(isinstance(item, Mapping) for item in cells):
        raise SchedulerValidationError("MAP under-covered cells exceed the frozen proposal count")
    if len({digest_json(item) for item in cells}) != len(cells):
        raise SchedulerValidationError("MAP under-covered cells must be unique")
    if design["allowed_dimensions"] != list(VARIATION_DIMENSIONS):
        raise SchedulerValidationError("MAP variation dimensions are not the closed vocabulary")
    _digest(design["probe_review_scope_key"], "MAP probe-review scope key")
    if (
        len(unit.proposal_slots) != count
        or len(unit.execution_slots) != count
        or len(unit.panel_plans) != count
    ):
        raise SchedulerValidationError("MAP slot and panel cardinalities must equal M")
    actions = tuple(unit.logical_action_slots)
    if len(actions) != 2 + sum(len(_panel_action_ids(panel)) for panel in unit.panel_plans):
        raise SchedulerValidationError("MAP action graph cardinality is invalid")
    proposal_action, review_action = actions[:2]
    _expect_action(
        proposal_action,
        role="perturbator",
        purpose="perturbator_proposal",
        owner=unit.unit_scientific_key,
        dependencies=(),
        guard={"kind": "always"},
    )
    _expect_action(
        review_action,
        role="selector",
        purpose="probe_review",
        owner=unit.unit_scientific_key,
        dependencies=(proposal_action["action_id"],),
        guard={
            "kind": "if_any_proposed_item",
            "proposal_action_ids": [proposal_action["action_id"]],
        },
    )
    flattened = [proposal_action["action_id"], review_action["action_id"]]
    for ordinal, (proposal, execution, panel) in enumerate(
        zip(unit.proposal_slots, unit.execution_slots, unit.panel_plans, strict=True)
    ):
        if execution["arm"] != "none" or execution["root_plan_id"] is not None:
            raise SchedulerValidationError("MAP execution arm or root plan is invalid")
        _validate_execution_identity(unit, execution, proposal_keys=(proposal["scientific_key"],))
        _validate_panel_execution_pair(execution, panel, action_by_id)
        target = action_by_id[panel["target_action_id"]]
        if target["dependency_action_ids"] != [review_action["action_id"]] or target["guard"].get(
            "proposal_slot_scientific_key"
        ) != proposal["scientific_key"] or target["guard"].get("allowed") != ["valid"]:
            raise SchedulerValidationError("MAP target does not own its reviewed proposal slot")
        flattened.extend(_panel_action_ids(panel))
    if flattened != [slot["action_id"] for slot in actions]:
        raise SchedulerValidationError("MAP action graph order is invalid")


def _validate_strategy_slots(
    unit: WorkUnit, value: object, expected_count: int
) -> tuple[Mapping[str, Any], ...]:
    slots = tuple(_require_mapping_array(value, "portfolio strategy slots"))
    if len(slots) != expected_count:
        raise SchedulerValidationError("portfolio must freeze exactly Kmax strategy slots")
    ids: list[str] = []
    keys: list[str] = []
    for ordinal, slot in enumerate(slots):
        _exact_fields(slot, {"entity_id", "scientific_key", "ordinal"}, "strategy slot")
        if _count(slot["ordinal"], "strategy slot ordinal") != ordinal:
            raise SchedulerValidationError("strategy slot ordinals must be contiguous")
        key = _digest(slot["scientific_key"], "strategy slot scientific key")
        expected_key = scientific_key([unit.unit_scientific_key, "strategy_slot", ordinal])
        if key != expected_key or slot["entity_id"] != entity_id("ss", [key]):
            raise SchedulerValidationError("strategy slot identity is invalid")
        ids.append(_entity(slot["entity_id"], "ss", "strategy slot ID"))
        keys.append(key)
    _unique(ids, "strategy slot IDs")
    _unique(keys, "strategy slot scientific keys")
    return slots


def _validate_portfolio_assignments(
    design: Mapping[str, Any],
    proposal_keys: Sequence[str],
    maximum_strategy_count: int,
) -> tuple[str, ...]:
    round_index = design["round_index"]
    active_ids = _array(design["active_strategy_ids"], "active strategy IDs")
    for active_id in active_ids:
        _entity(active_id, "st", "active strategy ID")
    _unique(active_ids, "active strategy IDs")
    assignments = _require_mapping_array(
        design["frozen_assignments"], "frozen portfolio assignments"
    )
    if round_index != 0 or active_ids or assignments:
        raise SchedulerValidationError(
            "a portfolio contract step must have round_index 0 and no prebound assignments"
        )
    return ()


def _validate_portfolio_round(
    unit: WorkUnit,
    action_by_id: Mapping[str, Mapping[str, Any]],
) -> frozenset[str]:
    design = unit.design
    _exact_fields(design, _DESIGN_FIELDS[unit.kind], "portfolio-round design")
    if unit.parent_unit_id is None:
        raise SchedulerValidationError("portfolio round must cite its parent unit")
    try:
        PortfolioIntent(design["portfolio_intent"])
    except (KeyError, ValueError) as exc:
        raise SchedulerValidationError("portfolio intent is invalid") from exc
    if design["conversation_root_id"] != unit.subject_id:
        raise SchedulerValidationError("portfolio conversation root disagrees with its subject")
    _entity(design["conversation_root_id"], "cr", "portfolio conversation-root ID")
    _entity(design["generation_parent_node_id"], "cn", "generation parent-node ID")
    parent_key = _digest(
        design["generation_parent_node_scientific_key"],
        "generation parent-node scientific key",
    )
    round_index = _design_count(design, "round_index", unit.kind)
    if round_index != 0:
        raise SchedulerValidationError("portfolio round index must be 0")
    count = _design_count(design, "proposal_count", unit.kind, minimum=3)
    maximum_strategy_count = _design_count(
        design, "maximum_strategy_count", unit.kind, minimum=1
    )
    if maximum_strategy_count != count // 3:
        raise SchedulerValidationError("portfolio Kmax must equal floor(M/3)")
    strategy_slots = _validate_strategy_slots(unit, design["strategy_slots"], maximum_strategy_count)
    proposal_keys = [slot["scientific_key"] for slot in unit.proposal_slots]
    active_strategy_keys = _validate_portfolio_assignments(
        design, proposal_keys, maximum_strategy_count
    )
    generic_strategy_keys = {
        slot["scientific_key"] for slot in strategy_slots
    }
    if generic_strategy_keys & set(active_strategy_keys):
        raise SchedulerValidationError(
            "later active strategies cannot reuse generic strategy-slot identities"
        )
    review_scopes = _object(design["review_scope_keys"], "portfolio review-scope keys")
    _exact_fields(review_scopes, {"probe_review", "outcome_review"}, "portfolio review-scope keys")
    _digest(review_scopes["probe_review"], "portfolio probe-review scope key")
    _digest(review_scopes["outcome_review"], "portfolio outcome-review scope key")
    if (
        len(unit.proposal_slots) != count
        or len(unit.execution_slots) != count
        or len(unit.panel_plans) != count
    ):
        raise SchedulerValidationError("portfolio slot and panel cardinalities must equal M")
    actions = tuple(unit.logical_action_slots)
    expected_count = maximum_strategy_count + 3 + sum(len(_panel_action_ids(panel)) for panel in unit.panel_plans)
    if len(actions) != expected_count:
        raise SchedulerValidationError("portfolio action graph cardinality is invalid")
    cursor = 0
    flattened: list[str] = []
    selector = actions[cursor]
    _expect_action(
        selector,
        role="selector",
        purpose="selector_portfolio",
        owner=parent_key,
        dependencies=(),
        guard={"kind": "always"},
    )
    flattened.append(selector["action_id"])
    cursor += 1
    generic_keys = [slot["scientific_key"] for slot in strategy_slots]
    lane_owner_keys = [
        *active_strategy_keys,
        *generic_keys[len(active_strategy_keys) :],
    ]
    originals = actions[cursor : cursor + maximum_strategy_count]
    cursor += maximum_strategy_count
    for action, owner in zip(originals, lane_owner_keys, strict=True):
        dependencies = () if selector is None else (selector["action_id"],)
        _expect_action(
            action,
            role="perturbator",
            purpose="perturbator_proposal",
            owner=owner,
            dependencies=dependencies,
            guard={
                "kind": "if_strategy_slot_bound",
                "strategy_slot_scientific_key": owner,
            },
        )
        flattened.append(action["action_id"])
    initial_review = actions[cursor]
    cursor += 1
    original_ids = [action["action_id"] for action in originals]
    _expect_action(
        initial_review,
        role="selector",
        purpose="probe_review",
        owner=unit.unit_scientific_key,
        dependencies=original_ids,
        guard={"kind": "if_any_proposed_item", "proposal_action_ids": original_ids},
    )
    flattened.append(initial_review["action_id"])
    all_panel_action_ids: list[str] = []
    target_ids: list[str] = []
    for proposal, execution, panel in zip(
        unit.proposal_slots, unit.execution_slots, unit.panel_plans, strict=True
    ):
        if execution["arm"] != "treatment" or execution["root_plan_id"] is None:
            raise SchedulerValidationError("portfolio execution arm or root identity is invalid")
        _validate_execution_identity(unit, execution, proposal_keys=(proposal["scientific_key"],))
        _validate_panel_execution_pair(execution, panel, action_by_id)
        target = action_by_id[panel["target_action_id"]]
        if target["dependency_action_ids"] != [initial_review["action_id"]] or target["guard"] != {
            "kind": "if_dispatchable_probe",
            "review_item_id": target["guard"].get("review_item_id"),
            "proposal_slot_scientific_key": proposal["scientific_key"],
            "allowed": ["aligned"],
        }:
            raise SchedulerValidationError("portfolio target does not own its reviewed proposal slot")
        panel_ids = _panel_action_ids(panel)
        flattened.extend(panel_ids)
        all_panel_action_ids.extend(panel_ids)
        target_ids.append(target["action_id"])
    outcome = actions[cursor + len(all_panel_action_ids)]
    _expect_action(
        outcome,
        role="selector",
        purpose="outcome_review",
        owner=unit.unit_scientific_key,
        dependencies=all_panel_action_ids,
        guard={"kind": "if_any_aligned_response", "target_action_ids": target_ids},
    )
    flattened.append(outcome["action_id"])
    if flattened != [slot["action_id"] for slot in actions]:
        raise SchedulerValidationError("portfolio action graph order is invalid")
    return frozenset((*generic_keys, *active_strategy_keys))


def _validate_verification(
    unit: WorkUnit,
    action_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    design = unit.design
    _exact_fields(design, _DESIGN_FIELDS[unit.kind], "verification design")
    if design["subject_id"] != unit.subject_id or design["subject_scientific_key"] != unit.subject_scientific_key:
        raise SchedulerValidationError("verification design subject disagrees with its unit")
    subject_kind = design["subject_kind"]
    if subject_kind not in {"entry_prompt", "scenario_recipe"}:
        raise SchedulerValidationError("verification subject kind is invalid")
    _entity(
        design["subject_id"],
        "pc" if subject_kind == "entry_prompt" else "sc",
        "verification subject ID",
    )
    if subject_kind == "entry_prompt":
        if unit.parent_unit_id is None:
            raise SchedulerValidationError("entry verification must cite its parent unit")
    elif unit.parent_unit_id is not None:
        _entity(unit.parent_unit_id, "wu", "derived scenario parent unit ID")
    count = _design_count(design, "planned_count", unit.kind, minimum=5)
    _object(design["measurement_cell"], "verification measurement cell")
    if len(unit.execution_slots) != count:
        raise SchedulerValidationError("verification must contain exactly N executions")
    roots = _validate_root_plans(unit, design["root_plans"], action_by_id)
    if len(roots) != count:
        raise SchedulerValidationError("verification must contain exactly N root plans")
    if [root["execution_id"] for root in roots] != [
        execution["execution_id"] for execution in unit.execution_slots
    ]:
        raise SchedulerValidationError("verification root plans must follow execution order")
    flattened: list[str] = []
    for execution, root in zip(unit.execution_slots, roots, strict=True):
        if execution["arm"] != "none":
            raise SchedulerValidationError("verification execution arm must be none")
        if subject_kind == "entry_prompt":
            _validate_execution_identity(unit, execution)
            if root["semantic_nonsuccess_reasons"]:
                raise SchedulerValidationError("entry verification cannot predeclare semantic nonsuccess")
            owned_panels = [
                panel
                for panel in unit.panel_plans
                if action_by_id[panel["target_action_id"]]["owner_position_scientific_key"]
                == execution["execution_scientific_key"]
            ]
            if len(owned_panels) != 1 or root["ordered_action_ids"] != _panel_action_ids(owned_panels[0]):
                raise SchedulerValidationError("entry-verification root must be exactly one assessed target")
            if root["terminal_endpoint_action_id"] != owned_panels[0]["target_action_id"]:
                raise SchedulerValidationError("entry-verification terminal endpoint is invalid")
            if owned_panels[0]["trigger_policy"] != "always_full":
                raise SchedulerValidationError("entry verification requires a full panel")
        else:
            if len(unit.proposal_slots) % count:
                raise SchedulerValidationError(
                    "scenario proposal slots must divide evenly across roots"
                )
            depth = len(unit.proposal_slots) // count
            if depth < 1:
                raise SchedulerValidationError(
                    "scenario verification requires at least one continuation"
                )
            root_index = execution["ordinal"]
            if root_index < 0 or root_index >= count:
                raise SchedulerValidationError(
                    "scenario execution ordinal is outside its frozen roots"
                )
            root_proposals = unit.proposal_slots[
                root_index * depth : (root_index + 1) * depth
            ]
            proposal_keys = tuple(
                proposal["scientific_key"] for proposal in root_proposals
            )
            _validate_execution_identity(
                unit, execution, proposal_keys=proposal_keys
            )
            if root["semantic_nonsuccess_reasons"] != [
                "inapplicable",
                "nonprogress",
            ]:
                raise SchedulerValidationError(
                    "scenario semantic nonsuccess reasons are not canonical"
                )
            panel_width = len(_panel_action_ids(unit.panel_plans[0]))
            expected_length = panel_width + (6 + panel_width) * depth
            root_actions = [
                action_by_id[action_id] for action_id in root["ordered_action_ids"]
            ]
            if len(root_actions) != expected_length:
                raise SchedulerValidationError(
                    "scenario root action cardinality is invalid"
                )
            owner = execution["execution_scientific_key"]
            entry_panel = _panel_for_target(unit, root_actions[0]["action_id"])
            entry_ids = _panel_action_ids(entry_panel)
            if root["ordered_action_ids"][:len(entry_ids)] != entry_ids:
                raise SchedulerValidationError(
                    "scenario entry target panel is misplaced"
                )
            if entry_panel["trigger_policy"] != "always_full":
                raise SchedulerValidationError(
                    "scenario entry target requires a full panel"
                )
            _validate_panel_execution_pair(execution, entry_panel, action_by_id)
            prior_outcome: Mapping[str, Any] | None = None
            prior_target_id = entry_panel["target_action_id"]
            cursor = len(entry_ids)
            final_target_id = prior_target_id
            for step_index, proposal_key in enumerate(proposal_keys):
                binding = root_actions[cursor]
                cursor += 1
                if prior_outcome is None:
                    binding_dependencies = tuple(entry_ids)
                    binding_guard = {
                        "kind": "if_panel_resolved",
                        "panel_id": entry_panel["panel_id"],
                    }
                else:
                    binding_dependencies = (prior_outcome["action_id"],)
                    binding_guard = {
                        "kind": "if_action_usable",
                        "action_id": prior_outcome["action_id"],
                    }
                _expect_action(
                    binding,
                    role="selector",
                    purpose="template_binding",
                    owner=owner,
                    dependencies=binding_dependencies,
                    guard=binding_guard,
                )
                original = root_actions[cursor]
                cursor += 1
                _expect_action(
                    original,
                    role="perturbator",
                    purpose="perturbator_proposal",
                    owner=owner,
                    dependencies=(binding["action_id"],),
                    guard={
                        "kind": "if_template_applicable",
                        "binding_action_id": binding["action_id"],
                    },
                )
                review = root_actions[cursor]
                cursor += 1
                _expect_action(
                    review,
                    role="selector",
                    purpose="probe_review",
                    owner=owner,
                    dependencies=(original["action_id"],),
                    guard={
                        "kind": "if_any_proposed_item",
                        "proposal_action_ids": [original["action_id"]],
                    },
                )
                corrective = root_actions[cursor]
                cursor += 1
                _expect_action(
                    corrective,
                    role="perturbator",
                    purpose="perturbator_proposal",
                    owner=owner,
                    dependencies=(review["action_id"],),
                    guard={
                        "kind": "if_repairable_proposal_exists",
                        "review_action_id": review["action_id"],
                        "proposal_slot_scientific_keys": [proposal_key],
                        "statuses": ["invalid"],
                    },
                )
                rereview = root_actions[cursor]
                cursor += 1
                _expect_action(
                    rereview,
                    role="selector",
                    purpose="probe_review",
                    owner=owner,
                    dependencies=(corrective["action_id"],),
                    guard={
                        "kind": "if_any_corrective_proposal",
                        "proposal_action_ids": [corrective["action_id"]],
                    },
                )
                target = root_actions[cursor]
                panel = _panel_for_target(unit, target["action_id"])
                panel_ids = _panel_action_ids(panel)
                if root["ordered_action_ids"][cursor : cursor + len(panel_ids)] != panel_ids:
                    raise SchedulerValidationError(
                        "scenario continuation target panel is misplaced"
                    )
                if panel["trigger_policy"] != "always_full":
                    raise SchedulerValidationError(
                        "scenario continuation requires a full panel"
                    )
                _validate_panel_execution_pair(execution, panel, action_by_id)
                _expect_action(
                    target,
                    role="target",
                    purpose="target_sample",
                    owner=owner,
                    dependencies=(rereview["action_id"],),
                    guard={
                        "kind": "if_dispatchable_probe",
                        "review_item_id": make_review_item_id(
                            unit.unit_scientific_key, "probe", proposal_key
                        ),
                        "proposal_slot_scientific_key": proposal_key,
                        "allowed": ["aligned"],
                    },
                )
                cursor += len(panel_ids)
                outcome = root_actions[cursor]
                cursor += 1
                _expect_action(
                    outcome,
                    role="selector",
                    purpose="outcome_review",
                    owner=owner,
                    dependencies=tuple(panel_ids),
                    guard={
                        "kind": "if_any_aligned_response",
                        "target_action_ids": [target["action_id"]],
                    },
                )
                prior_outcome = outcome
                prior_target_id = target["action_id"]
                final_target_id = target["action_id"]
            if cursor != len(root_actions):
                raise SchedulerValidationError(
                    "scenario root contains actions outside its frozen steps"
                )
            if root["terminal_endpoint_action_id"] != final_target_id:
                raise SchedulerValidationError(
                    "scenario terminal endpoint is not its final target"
                )
        flattened.extend(root["ordered_action_ids"])
    if flattened != [slot["action_id"] for slot in unit.logical_action_slots]:
        raise SchedulerValidationError("verification root plans must partition the action graph")
    if subject_kind == "entry_prompt":
        if unit.proposal_slots or len(unit.panel_plans) != count:
            raise SchedulerValidationError("entry verification cannot contain proposal slots and needs N panels")
    else:
        depth = len(unit.proposal_slots) // count
        if len(unit.panel_plans) != count * (1 + depth):
            raise SchedulerValidationError(
                "scenario verification panel count is invalid"
            )
    return roots


def _panel_for_target(
    unit: WorkUnit, target_id: str
) -> Mapping[str, Any]:
    matches = [panel for panel in unit.panel_plans if panel["target_action_id"] == target_id]
    if len(matches) != 1:
        raise SchedulerValidationError("qualification target does not resolve to one panel")
    return matches[0]


def _validate_qualification_root(
    unit: WorkUnit,
    execution: Mapping[str, Any],
    root: Mapping[str, Any],
    proposal_key: str | None,
) -> None:
    arm = execution["arm"]
    action_by_id = {slot["action_id"]: slot for slot in unit.logical_action_slots}
    actions = [action_by_id[action_id] for action_id in root["ordered_action_ids"]]
    owner = execution["execution_scientific_key"]
    panel_width = len(_panel_action_ids(_panel_for_target(unit, actions[0]["action_id"])))
    if arm == "parent_baseline":
        if proposal_key is not None or len(actions) != panel_width:
            raise SchedulerValidationError("parent-baseline root topology is invalid")
        panel = _panel_for_target(unit, actions[0]["action_id"])
        if root["ordered_action_ids"] != _panel_action_ids(panel):
            raise SchedulerValidationError("parent-baseline root must be one panel")
        if root["terminal_endpoint_action_id"] != actions[0]["action_id"] or root["semantic_nonsuccess_reasons"]:
            raise SchedulerValidationError("parent-baseline endpoint or nonsuccess reasons are invalid")
        return
    if proposal_key is None:
        raise SchedulerValidationError("continuation qualification root lacks its proposal slot")
    if arm == "treatment":
        expected_length = 2 * panel_width + 6
        terminal_index = panel_width + 5
    elif arm == "strategy_agnostic":
        expected_length = 2 * panel_width + 4
        terminal_index = panel_width + 4
    else:
        raise SchedulerValidationError("qualification root arm is invalid")
    if len(actions) != expected_length:
        raise SchedulerValidationError("qualification root action cardinality is invalid")
    entry_panel = _panel_for_target(unit, actions[0]["action_id"])
    if root["ordered_action_ids"][:panel_width] != _panel_action_ids(entry_panel):
        raise SchedulerValidationError("qualification entry target panel is misplaced")
    if entry_panel["trigger_policy"] != "always_full":
        raise SchedulerValidationError("qualification entry target requires a full panel")
    cursor = panel_width
    entry_ids = root["ordered_action_ids"][:panel_width]
    if arm == "treatment":
        binding = actions[cursor]
        _expect_action(
            binding,
            role="selector",
            purpose="template_binding",
            owner=owner,
            dependencies=entry_ids,
            guard={"kind": "always"},
        )
        cursor += 1
        original_guard = {
            "kind": "if_template_applicable",
            "binding_action_id": binding["action_id"],
        }
        original_dependencies = (binding["action_id"],)
    else:
        original_guard = {"kind": "always"}
        original_dependencies = tuple(entry_ids)
    original = actions[cursor]
    _expect_action(
        original,
        role="perturbator",
        purpose="perturbator_proposal",
        owner=owner,
        dependencies=original_dependencies,
        guard=original_guard,
    )
    cursor += 1
    review = actions[cursor]
    _expect_action(
        review,
        role="selector",
        purpose="probe_review",
        owner=owner,
        dependencies=(original["action_id"],),
        guard={
            "kind": "if_any_proposed_item",
            "proposal_action_ids": [original["action_id"]],
        },
    )
    cursor += 1
    correction = actions[cursor]
    statuses = ["invalid"]
    _expect_action(
        correction,
        role="perturbator",
        purpose="perturbator_proposal",
        owner=owner,
        dependencies=(review["action_id"],),
        guard={
            "kind": "if_repairable_proposal_exists",
            "review_action_id": review["action_id"],
            "proposal_slot_scientific_keys": [proposal_key],
            "statuses": statuses,
        },
    )
    cursor += 1
    rereview = actions[cursor]
    _expect_action(
        rereview,
        role="selector",
        purpose="probe_review",
        owner=owner,
        dependencies=(correction["action_id"],),
        guard={
            "kind": "if_any_corrective_proposal",
            "proposal_action_ids": [correction["action_id"]],
        },
    )
    cursor += 1
    terminal = actions[cursor]
    terminal_panel = _panel_for_target(unit, terminal["action_id"])
    if root["ordered_action_ids"][cursor : cursor + len(_panel_action_ids(terminal_panel))] != _panel_action_ids(terminal_panel):
        raise SchedulerValidationError("qualification terminal target panel is misplaced")
    if terminal_panel["trigger_policy"] != "always_full":
        raise SchedulerValidationError("qualification terminal target requires a full panel")
    allowed = ["aligned"] if arm == "treatment" else ["valid"]
    if terminal["dependency_action_ids"] != [rereview["action_id"]] or terminal["guard"] != {
        "kind": "if_dispatchable_probe",
        "review_item_id": terminal["guard"].get("review_item_id"),
        "proposal_slot_scientific_key": proposal_key,
        "allowed": allowed,
    }:
        raise SchedulerValidationError("qualification terminal dispatch guard is invalid")
    cursor += len(_panel_action_ids(terminal_panel))
    if arm == "treatment":
        outcome = actions[cursor]
        terminal_panel_ids = _panel_action_ids(terminal_panel)
        _expect_action(
            outcome,
            role="selector",
            purpose="outcome_review",
            owner=owner,
            dependencies=terminal_panel_ids,
            guard={
                "kind": "if_any_aligned_response",
                "target_action_ids": [terminal["action_id"]],
            },
        )
    if root["terminal_endpoint_action_id"] != actions[terminal_index]["action_id"]:
        raise SchedulerValidationError("qualification root terminal endpoint is invalid")
    expected_reasons = ["inapplicable"] if arm == "treatment" else []
    if root["semantic_nonsuccess_reasons"] != expected_reasons:
        raise SchedulerValidationError("qualification root semantic nonsuccess reasons are invalid")


def _validate_qualification_partition(
    unit: WorkUnit,
    roots: Sequence[Mapping[str, Any]],
    root_order: Sequence[str],
    proposal_key_by_execution: Mapping[str, str | None],
) -> None:
    root_by_id = {root["root_plan_id"]: root for root in roots}
    if len(root_by_id) != len(roots) or set(root_order) != set(root_by_id) or len(root_order) != len(roots):
        raise SchedulerValidationError("qualification interleaved root order is not a permutation")
    execution_by_id = {slot["execution_id"]: slot for slot in unit.execution_slots}
    flattened: list[str] = []
    for root_id in root_order:
        root = root_by_id[root_id]
        execution = execution_by_id[root["execution_id"]]
        proposal_key = proposal_key_by_execution[execution["execution_id"]]
        _validate_qualification_root(unit, execution, root, proposal_key)
        flattened.extend(root["ordered_action_ids"])
    if flattened != [slot["action_id"] for slot in unit.logical_action_slots]:
        raise SchedulerValidationError("qualification root order does not partition the action graph")


def _validate_strategy_development(
    unit: WorkUnit,
    action_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    design = unit.design
    _exact_fields(design, _DESIGN_FIELDS[unit.kind], "strategy-development design")
    if unit.parent_unit_id is None:
        raise SchedulerValidationError("strategy development must cite its parent unit")
    if design["strategy_template_id"] != unit.subject_id or design["strategy_template_scientific_key"] != unit.subject_scientific_key:
        raise SchedulerValidationError("development template identity disagrees with its subject")
    _entity(design["strategy_template_id"], "tp", "development strategy-template ID")
    count = _design_count(design, "planned_count_per_arm", unit.kind, minimum=5)
    dimensions = _validate_dimensions(
        design["comparison_dimensions"],
        "development comparison dimensions",
        control_safe=True,
    )
    intents = _array(
        design["realization_intent_assignment"],
        "development realization-intent assignment",
    )
    if len(intents) != count or any(item not in {"anchor", "close", "divergent"} for item in intents):
        raise SchedulerValidationError("development must freeze one valid intent per arm-local root")
    sealed = _validate_sealed_holdout_plan(design["sealed_holdout_plan"])
    if design["sealed_holdout_digest"] != digest_json(sealed):
        raise SchedulerValidationError("development sealed-holdout digest is incorrect")
    if tuple(sealed["comparison_dimensions"]) != dimensions or sealed["measurement_cell"] != design["measurement_cell"]:
        raise SchedulerValidationError("development and future holdout design changed comparison scope")
    _object(design["measurement_cell"], "development measurement cell")
    if len(unit.proposal_slots) != 2 * count or len(unit.execution_slots) != 3 * count:
        raise SchedulerValidationError("development slot cardinalities must be 2N proposals and 3N executions")
    roots = _validate_root_plans(unit, design["root_plans"], action_by_id)
    if len(roots) != 3 * count:
        raise SchedulerValidationError("development must contain exactly 3N root plans")
    expected_arms = ["treatment"] * count + ["parent_baseline"] * count + ["strategy_agnostic"] * count
    if [slot["arm"] for slot in unit.execution_slots] != expected_arms:
        raise SchedulerValidationError("development executions are not in canonical arm-major order")
    if [root["execution_id"] for root in roots] != [slot["execution_id"] for slot in unit.execution_slots]:
        raise SchedulerValidationError("development root plans must follow execution order")
    proposal_by_execution: dict[str, str | None] = {}
    for ordinal, execution in enumerate(unit.execution_slots):
        if ordinal < count:
            proposal = unit.proposal_slots[ordinal]["scientific_key"]
        elif ordinal < 2 * count:
            proposal = None
        else:
            proposal = unit.proposal_slots[count + ordinal - 2 * count]["scientific_key"]
        _validate_execution_identity(
            unit,
            execution,
            proposal_keys=() if proposal is None else (proposal,),
        )
        proposal_by_execution[execution["execution_id"]] = proposal
    order = _array(design["interleaved_root_order"], "development interleaved root order")
    for root_id in order:
        _entity(root_id, "rp", "development interleaved root-plan ID")
    _validate_qualification_partition(unit, roots, order, proposal_by_execution)
    return roots


def _validate_strategy_holdout(
    unit: WorkUnit,
    action_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    design = unit.design
    _exact_fields(design, _DESIGN_FIELDS[unit.kind], "strategy-holdout design")
    source_id = _entity(
        design["source_development_unit_id"],
        "wu",
        "holdout source-development unit ID",
    )
    if unit.parent_unit_id != source_id:
        raise SchedulerValidationError("holdout parent must be its cited development unit")
    if design["strategy_template_id"] != unit.subject_id or design["strategy_template_scientific_key"] != unit.subject_scientific_key:
        raise SchedulerValidationError("holdout template identity disagrees with its subject")
    _entity(design["strategy_template_id"], "tp", "holdout strategy-template ID")
    sealed = _validate_sealed_holdout_plan(design["sealed_holdout_plan"])
    if design["source_sealed_holdout_digest"] != digest_json(sealed):
        raise SchedulerValidationError("holdout source seal does not digest its reproduced plan")
    count = sealed["planned_count_per_arm"]
    if len(unit.proposal_slots) != 2 * count or len(unit.execution_slots) != 2 * count:
        raise SchedulerValidationError("holdout must contain exactly 2H proposals and executions")
    roots = _validate_root_plans(unit, design["root_plans"], action_by_id)
    if len(roots) != 2 * count:
        raise SchedulerValidationError("holdout must contain exactly 2H root plans")
    future_roots = sealed["future_roots"]
    expected_arms = [future["arm"] for future in future_roots]
    if [slot["arm"] for slot in unit.execution_slots] != expected_arms:
        raise SchedulerValidationError("holdout materialized arms disagree with the sealed plan")
    if [root["execution_id"] for root in roots] != [slot["execution_id"] for slot in unit.execution_slots]:
        raise SchedulerValidationError("holdout root plans must follow execution order")
    proposal_by_execution: dict[str, str] = {}
    for ordinal, execution in enumerate(unit.execution_slots):
        proposal = unit.proposal_slots[ordinal]["scientific_key"]
        _validate_execution_identity(unit, execution, proposal_keys=(proposal,))
        proposal_by_execution[execution["execution_id"]] = proposal
        if roots[ordinal]["semantic_nonsuccess_reasons"] != future_roots[ordinal]["semantic_nonsuccess_reasons"]:
            raise SchedulerValidationError("holdout root nonsuccess rules disagree with the seal")
    root_order = [roots[ordinal]["root_plan_id"] for ordinal in sealed["interleaved_root_order"]]
    _validate_qualification_partition(unit, roots, root_order, proposal_by_execution)
    return roots


def _validate_route_consistency(unit: WorkUnit) -> None:
    by_key: dict[str, Mapping[str, Any]] = {}
    by_slot: dict[str, str] = {}
    for action in unit.logical_action_slots:
        for route in action["routes"]:
            route_key = route["route_key"]
            previous = by_key.setdefault(route_key, route)
            fields = set(_ROUTE_FIELDS) - {"maximum_attempts"}
            if any(previous[field] != route[field] for field in fields):
                raise SchedulerValidationError("one role route key has inconsistent frozen metadata")
            configured_slot = route["configured_slot"]
            previous_key = by_slot.setdefault(configured_slot, route_key)
            if previous_key != route_key:
                raise SchedulerValidationError("one configured role slot has multiple route keys")


def _validate_proposal_usage(unit: WorkUnit) -> None:
    proposal_keys = [slot["scientific_key"] for slot in unit.proposal_slots]
    dispatched = [
        slot["guard"]["proposal_slot_scientific_key"]
        for slot in unit.logical_action_slots
        if slot["guard"]["kind"] == "if_dispatchable_probe"
    ]
    if proposal_keys and (
        len(dispatched) != len(proposal_keys) or set(dispatched) != set(proposal_keys)
    ):
        raise SchedulerValidationError(
            "every proposal slot must own exactly one dispatch guard"
        )
    if not proposal_keys and dispatched:
        raise SchedulerValidationError("a dispatch guard requires a preallocated proposal slot")


def _validate_work_unit_semantics(
    unit: WorkUnit, *, expected_policy_digest: str = _POLICY_DIGEST
) -> None:
    if not isinstance(unit, WorkUnit):
        raise SchedulerValidationError("expected a WorkUnit")
    _entity(unit.unit_id, "wu", "work-unit ID")
    _digest(unit.unit_scientific_key, "work-unit scientific key")
    if unit.kind not in _WORK_KINDS:
        raise SchedulerValidationError("work-unit kind is invalid")
    _any_entity(unit.subject_id, "work-unit subject ID")
    _digest(unit.subject_scientific_key, "work-unit subject scientific key")
    if unit.parent_unit_id is not None:
        _entity(unit.parent_unit_id, "wu", "parent work-unit ID")
    if (
        _digest(unit.policy_digest, "work-unit policy digest")
        != expected_policy_digest
    ):
        raise SchedulerValidationError("work-unit policy digest is not the canonical policy")
    design = _object(unit.design, "work-unit design")
    if design.get("tag") != unit.kind:
        raise SchedulerValidationError("work-unit kind and design tag disagree")
    if not unit.logical_action_slots:
        raise SchedulerValidationError("work unit must freeze a nonempty logical-action graph")
    action_slot_maximum(unit.logical_action_slots)
    action_by_id = {
        slot["action_id"]: slot for slot in unit.logical_action_slots
    }
    proposals = _validate_proposal_slots(unit)
    _validate_execution_slots(unit)
    panels = _validate_panels(unit, action_by_id)
    panel_by_id = {panel["panel_id"]: panel for panel in panels}
    _validate_route_consistency(unit)
    strategy_keys: frozenset[str] = frozenset()
    for slot in unit.logical_action_slots:
        _validate_ordinary_routes(slot, unit_kind=unit.kind)
    if unit.kind == "initial_observation":
        _validate_initial_observation(unit, action_by_id)
    elif unit.kind == "screening":
        _validate_screening(unit, action_by_id)
    elif unit.kind == "map_exploration":
        _validate_map_exploration(unit, action_by_id)
    elif unit.kind == "portfolio_round":
        strategy_keys = _validate_portfolio_round(unit, action_by_id)
    elif unit.kind == "verification":
        _validate_verification(unit, action_by_id)
    elif unit.kind == "strategy_development":
        _validate_strategy_development(unit, action_by_id)
    elif unit.kind == "strategy_holdout":
        _validate_strategy_holdout(unit, action_by_id)
    _validate_guards(
        unit,
        action_by_id,
        panel_by_id,
        frozenset(slot["scientific_key"] for slot in proposals),
        strategy_keys,
    )
    _validate_proposal_usage(unit)


def recompute_work_unit_maximum(unit: WorkUnit) -> RoleCallVector:
    if not isinstance(unit, WorkUnit):
        raise SchedulerValidationError("expected a WorkUnit")
    requires_action_graph = (
        unit.kind == "verification"
        and unit.design.get("subject_kind") == "scenario_recipe"
    )
    by_formula: RoleCallVector | None = None
    try:
        by_formula = formula_maximum(unit.kind, unit.design)
        if unit.panel_plans:
            panel = unit.panel_plans[0]
            judge_attempts = 2 * (len(panel["normal_members"]) + (panel["replacement"] is not None))
            by_formula = adjust_panel_maximum(by_formula, judge_attempts)
    except SchedulerValidationError:
        if not requires_action_graph or not unit.logical_action_slots:
            raise
    if unit.logical_action_slots:
        by_actions = action_slot_maximum(unit.logical_action_slots)
        if by_formula is not None and by_actions != by_formula:
            raise SchedulerValidationError("action graph and semantic maximum disagree")
        return by_actions
    assert by_formula is not None
    return by_formula


def validate_work_unit_maximum(
    unit: WorkUnit, *, expected_policy_digest: str = _POLICY_DIGEST
) -> RoleCallVector:
    _validate_work_unit_semantics(
        unit, expected_policy_digest=expected_policy_digest
    )
    calculated = recompute_work_unit_maximum(unit)
    if calculated != unit.worst_case_calls:
        raise SchedulerValidationError("recorded work-unit maximum is incorrect")
    return calculated



TERMINAL_CALL_TYPES = frozenset(
    {"model_call_succeeded", "model_call_failed", "model_call_outcome_unknown"}
)
PROVIDER_CIRCUIT_ROLES = ("target", "judge", "selector", "perturbator")
_TERMINAL_RESUMABILITY = {
    RunTerminalReason.COMPLETED_NO_ELIGIBLE_WORK.value: False,
    RunTerminalReason.COMPLETED_PLAN.value: False,
    RunTerminalReason.COST_LIMIT_REACHED.value: False,
    RunTerminalReason.OPERATOR_INTERRUPTED.value: True,
    RunTerminalReason.SYSTEMIC_PROVIDER_FAILURE.value: True,
    RunTerminalReason.PROVIDER_IDENTITY_DRIFT.value: False,
    RunTerminalReason.EVIDENCE_UNWRITABLE.value: True,
    RunTerminalReason.EVIDENCE_CORRUPT.value: False,
    RunTerminalReason.IMPOSSIBLE_INTERNAL_STATE.value: False,
}


@dataclass(frozen=True, slots=True)
class ProviderCircuitOutcome:
    """One completed logical role outcome in deterministic journal order."""

    role: str
    sequence: int
    scientific_key: str
    status: str
    route_key: str | None = None

    def __post_init__(self) -> None:
        if self.role not in PROVIDER_CIRCUIT_ROLES:
            raise DomainError("provider circuit outcome has an invalid role")
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence <= 0
        ):
            raise DomainError("provider circuit outcome sequence must be positive")
        require_digest(self.scientific_key, "provider circuit outcome scientific key")
        if self.status not in {"success", "failure"}:
            raise DomainError("provider circuit outcome status is invalid")
        if self.route_key is not None:
            require_digest(self.route_key, "provider circuit route key")


@dataclass(frozen=True, slots=True)
class ProviderCircuitState:
    """Pure reduction of consecutive exhausted logical-role outcomes."""

    target: int = 0
    judge: int = 0
    selector: int = 0
    perturbator: int = 0
    tripped_role: str | None = None
    tripped_sequence: int | None = None
    tripped_route_key: str | None = None

    def streak(self, role: str) -> int:
        if role not in PROVIDER_CIRCUIT_ROLES:
            raise DomainError("unknown provider circuit role")
        return int(getattr(self, role))


def reduce_provider_circuit(
    outcomes: Iterable[ProviderCircuitOutcome],
    *,
    reset_after: Mapping[str, int] | None = None,
    threshold: int = 3,
) -> ProviderCircuitState:
    """Reduce role outcomes without a mutable counter or side-channel event.

    ``reset_after`` implements the explicit-resume circuit window: outcomes at
    or before a role's reset sequence remain audited but do not participate in
    the new consecutive-failure streak.
    """

    if isinstance(threshold, bool) or threshold <= 0:
        raise DomainError("provider circuit threshold must be positive")
    resets = dict(reset_after or {})
    if any(
        role not in PROVIDER_CIRCUIT_ROLES
        or isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or sequence < 0
        for role, sequence in resets.items()
    ):
        raise DomainError("provider circuit reset map is invalid")
    counts = {role: 0 for role in PROVIDER_CIRCUIT_ROLES}
    tripped_role: str | None = None
    tripped_sequence: int | None = None
    tripped_route_key: str | None = None
    route_counts: dict[tuple[str, str], int] = {}
    ordered = sorted(
        tuple(outcomes),
        key=lambda item: (item.sequence, item.scientific_key, item.role),
    )
    for outcome in ordered:
        if outcome.sequence <= resets.get(outcome.role, 0):
            continue
        if outcome.route_key is not None:
            key = (outcome.role, outcome.route_key)
            route_counts[key] = 0 if outcome.status == "success" else route_counts.get(key, 0) + 1
            if route_counts[key] >= threshold and tripped_role is None:
                tripped_role, tripped_sequence = outcome.role, outcome.sequence
                tripped_route_key = outcome.route_key
            continue
        if outcome.status == "success":
            counts[outcome.role] = 0
            continue
        counts[outcome.role] += 1
        if counts[outcome.role] >= threshold and tripped_role is None:
            tripped_role = outcome.role
            tripped_sequence = outcome.sequence
    return ProviderCircuitState(
        target=counts["target"],
        judge=counts["judge"],
        selector=counts["selector"],
        perturbator=counts["perturbator"],
        tripped_role=tripped_role,
        tripped_sequence=tripped_sequence,
        tripped_route_key=tripped_route_key,
    )


@dataclass(frozen=True, slots=True)
class AttemptView:
    physical_attempt_id: str
    intent_event: Mapping[str, Any]
    dispatch_event: Mapping[str, Any] | None
    terminal_event: Mapping[str, Any] | None
    # Derived from authoritative receipts, never persisted as a second fact.
    # It marks Selector/Perturbator successes whose non-null effective identity
    # equals the target identity pinned for this replay projection.
    target_identity_collision: bool = False

    @property
    def intended(self) -> Mapping[str, Any]:
        return self.intent_event["payload"]

    @property
    def dispatched(self) -> bool:
        return self.dispatch_event is not None

    @property
    def terminal_type(self) -> str | None:
        return None if self.terminal_event is None else str(self.terminal_event["type"])

    @property
    def succeeded(self) -> bool:
        return self.terminal_type == "model_call_succeeded"

    @property
    def output_text(self) -> str | None:
        if not self.succeeded or self.terminal_event is None:
            return None
        return self.terminal_event["payload"]["text"]


@dataclass(frozen=True, slots=True)
class LogicalActionView:
    action_id: str
    unit_id: str
    slot: Mapping[str, Any]
    attempts: tuple[AttemptView, ...]

    @property
    def successful_attempt(self) -> AttemptView | None:
        # A provider success is an authoritative transport fact, not proof that
        # the role-level result is usable.  The bounded recovery policy may
        # therefore record an invalid/truncated successful response followed by
        # a schema repair or exact retry that also succeeds.  Role projections
        # choose the first structurally usable attempt; this transport-level
        # convenience returns the latest authoritative success only.
        return next(
            (attempt for attempt in reversed(self.attempts) if attempt.succeeded),
            None,
        )

    @property
    def successful_attempts(self) -> tuple[AttemptView, ...]:
        return tuple(attempt for attempt in self.attempts if attempt.succeeded)

    @property
    def dispatched_attempt_count(self) -> int:
        return sum(attempt.dispatched for attempt in self.attempts)

    @property
    def maximum_attempts(self) -> int:
        value = self.slot.get("maximum_physical_attempts")
        return value if isinstance(value, int) else 0


@dataclass(frozen=True, slots=True)
class WorkUnitView:
    start_event: Mapping[str, Any]
    unit: Mapping[str, Any]
    actions: tuple[LogicalActionView, ...]

    @property
    def unit_id(self) -> str:
        return str(self.unit["unit_id"])


@dataclass(frozen=True, slots=True)
class TargetResultView:
    logical_action_id: str
    candidate_id: str | None
    execution_id: str | None
    conversation_root_id: str | None
    parent_node_id: str | None
    text: str
    finish_status: str
    effective_model_id: str | None
    physical_attempt_id: str
    terminal_event_id: str
    terminal_sequence: int

    @property
    def usable(self) -> bool:
        return bool(self.text) and self.finish_status == "complete"


@dataclass(frozen=True, slots=True)
class ConversationRootView:
    """One immutable conversation root derived from a frozen execution."""

    root_id: str
    scientific_key: str
    execution_ids: tuple[str, ...]
    node_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PanelView:
    """Complete reducer view of one frozen assessment panel."""

    panel_id: str
    target_action_id: str
    observation_id: str | None
    candidate_id: str | None
    execution_id: str | None
    target_event_id: str | None
    triggered: bool
    primary: MemberAssessment | None
    details: PanelAggregationDetails | None
    invalid_member_actions: tuple[str, ...]
    member_validation_errors: Mapping[str, tuple[str, ...]]
    terminated: bool

    @property
    def assessment(self) -> PanelAssessment | None:
        return None if self.details is None else self.details.assessment

    @property
    def effective_score(self) -> Decimal | None:
        if self.triggered:
            panel = self.assessment
            if panel is None or panel.availability is PanelAvailability.NO_QUORUM:
                return None
            return panel.weighted_score
        return None if self.primary is None else self.primary.score

    @property
    def resolved(self) -> bool:
        if not self.terminated:
            return False
        if self.triggered:
            panel = self.assessment
            return bool(
                panel is not None
                and panel.availability is not PanelAvailability.NO_QUORUM
            )
        return self.primary is not None

    @property
    def panel_required(self) -> bool:
        return self.triggered

    @property
    def half_even_score_display(self) -> str | None:
        """Return the canonical six-place display of the exact panel rational."""

        assessment = self.assessment
        return None if assessment is None else panel_score_display(assessment)

    @property
    def disputed_high(self) -> bool:
        """Return whether a quorum-level high lacks objective confirmation."""

        assessment = self.assessment
        return bool(
            assessment is not None
            and assessment.availability is not PanelAvailability.NO_QUORUM
            and assessment.weighted_score is not None
            and assessment.weighted_score >= HIGH_THRESHOLD
            and not assessment.confirmed_objective_high
        )


def portfolio_ownership(panel: PanelView, member_assessments: Mapping[str, MemberAssessment]) -> str:
    """Require the same ownership from a quorum, or the sole primary in search."""
    if panel.details is None:
        return "unclear" if panel.primary is None else panel.primary.ownership
    if panel.assessment is None or panel.assessment.availability is PanelAvailability.NO_QUORUM:
        return "unclear"
    members = [member for member in member_assessments.values() if member.assessment_id in panel.assessment.member_ids]
    if any(member.ownership == "conditional" for member in members):
        return "conditional"
    if any(member.ownership == "unclear" for member in members):
        return "unclear"
    if any(member.ownership == "absent" for member in members) and any(member.ownership == "adopted" for member in members):
        return "unclear"
    for status in ("adopted", "conditional", "absent"):
        if sum(member.ownership == status for member in members) >= 2:
            return status
    return "unclear"


def portfolio_current_assessment(
    panel: PanelView,
    member_assessments: Mapping[str, MemberAssessment],
) -> tuple[Verdict, Application] | None:
    """Project the one commitment stage shared by request building and replay."""

    if panel.details is None:
        if panel.primary is None:
            return None
        return panel.primary.verdict, panel.primary.application

    assessment = panel.details.assessment
    if assessment.availability is PanelAvailability.NO_QUORUM:
        return None
    try:
        verdict = Verdict(assessment.objective_failure)
    except ValueError as exc:
        raise JournalCorrupt(
            "current panel has no objective verdict classification"
        ) from exc
    included = frozenset(assessment.member_ids)
    members = tuple(
        member
        for member in member_assessments.values()
        if member.observation_id == panel.observation_id
        and member.assessment_id in included
    )
    if not members:
        raise JournalCorrupt("current panel has no included member evidence")
    if assessment.confirmed_objective_application:
        application = Application.ACTION_GOVERNING
    elif any(
        member.application in {Application.ABSTRACT, Application.ACTION_GOVERNING}
        for member in members
    ):
        application = Application.ABSTRACT
    else:
        application = Application.NONE
    return verdict, application


@dataclass(frozen=True, slots=True)
class PortfolioChildView:
    """One preallocated strategy slot from generation through outcome review."""

    work_unit_id: str
    proposal_slot: ProposalSlot
    assignment: ProposalAssignment
    strategy_id: str
    strategy_scientific_key: str
    original: ProposalItem | None
    original_review: ProbeReview | None
    selected: SelectedRealization | None
    alignment_status: str | None
    target_action_id: str
    execution_id: str
    target_attempt_ids: tuple[str, ...]
    dispatched_attempt_ids: tuple[str, ...]
    candidate_id: str | None
    candidate_scientific_key: str | None
    child_scientific_key: str
    observation_id: str | None
    observation_scientific_key: str | None
    panel_id: str
    terminal: bool
    outcome_review: StrategyOutcomeReview | None
    dispatch_fingerprint: str | None
    outcome: StrategyChildOutcome

    @property
    def allocated(self) -> bool:
        return True

    @property
    def dispatched(self) -> bool:
        return self.outcome.dispatched

    @property
    def resolved(self) -> bool:
        return self.outcome.resolved

    @property
    def credited(self) -> bool:
        return bool(
            self.outcome.aligned
            and self.outcome.resolved
            and self.outcome.outcome_result == "supports"
        )


@dataclass(frozen=True, slots=True)
class StrategyRoundView:
    """The reducer-owned evidence for one strategy lane in one frozen round."""

    work_unit_id: str
    work_unit_scientific_key: str
    portfolio_intent: PortfolioIntent
    round_index: int
    strategy_id: str
    strategy_scientific_key: str
    lane: BoundPortfolioLane
    assignments: tuple[ProposalAssignment, ...]
    children: tuple[PortfolioChildView, ...]
    evidence: StrategyRoundEvidence
    refusal: RefusalStepDecision
    complete: bool


@dataclass(frozen=True, slots=True)
class PortfolioRoundView:
    """A content-bound portfolio and every immutable lane/slot consequence."""

    work_unit_id: str
    work_unit_scientific_key: str
    portfolio_intent: PortfolioIntent
    conversation_root_id: str
    conversation_root_scientific_key: str
    generation_parent_node_id: str
    generation_parent_node_scientific_key: str
    round_index: int
    lanes: tuple[BoundPortfolioLane, ...]
    assignments: tuple[ProposalAssignment, ...]
    strategy_rounds: tuple[StrategyRoundView, ...]
    children: tuple[PortfolioChildView, ...]
    outcome_reviews: tuple[StrategyOutcomeReview, ...]
    complete: bool


@dataclass(frozen=True, slots=True)
class StrategyEpisodeView:
    """The sole composed projection used for strategy scheduling and credit."""

    strategy_id: str
    strategy_scientific_key: str
    portfolio_intent: PortfolioIntent
    conversation_root_id: str
    conversation_root_scientific_key: str
    generation_parent_node_id: str
    generation_parent_node_scientific_key: str
    diagnosis: str
    contract: BoundStrategy
    rounds: tuple[StrategyRoundView, ...]
    children: tuple[PortfolioChildView, ...]
    parent_baseline_score: Decimal
    parent_search_status: SearchStatus
    evidence: StrategyEpisodeEvidence
    preference: EpisodePreferenceDecision
    supporting_children: tuple[str, ...]
    dispatch_fingerprints: tuple[str, ...]
    refusal: RefusalStepDecision

    @property
    def preferred_in_run(self) -> bool:
        return self.preference.preferred_in_run


@dataclass(frozen=True, slots=True)
class ScenarioRecipeView:
    """One reducer-reconstructed stateful recipe and its exact source lineage."""

    scenario_id: str
    scientific_key: str
    entry_prompt: str
    recipe: ScenarioRecipe
    origin: str
    source_seed_id: str | None
    source_episode_id: str | None
    source_child_scientific_key: str | None
    parent_unit_id: str | None
    source_step_episode_ids: tuple[str, ...]
    source_step_child_scientific_keys: tuple[str, ...]
    source_unit_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.scenario_id.startswith("sc-"):
            raise DomainError("scenario recipe view requires an sc- entity ID")
        require_entity_id(self.scenario_id, "scenario recipe view ID")
        require_digest(self.scientific_key, "scenario recipe scientific key")
        if not isinstance(self.entry_prompt, str) or not self.entry_prompt:
            raise DomainError("scenario recipe view requires its entry prompt")
        if not isinstance(self.recipe, ScenarioRecipe):
            raise DomainError("scenario recipe view requires one frozen recipe")
        if self.origin not in {"imported", "derived"}:
            raise DomainError("scenario recipe origin is invalid")
        if self.origin == "imported":
            if (
                self.source_seed_id is None
                or self.source_episode_id is not None
                or self.source_child_scientific_key is not None
                or self.parent_unit_id is not None
                or self.source_step_episode_ids
                or self.source_step_child_scientific_keys
                or self.source_unit_ids
            ):
                raise DomainError("imported scenario recipe lineage is invalid")
            require_entity_id(self.source_seed_id, "imported scenario source seed ID")
        else:
            if (
                self.source_seed_id is not None
                or self.source_episode_id is None
                or self.source_child_scientific_key is None
                or self.parent_unit_id is None
                or not self.source_step_episode_ids
                or len(self.source_step_episode_ids)
                != len(self.source_step_child_scientific_keys)
                or len(self.source_step_episode_ids) != len(self.recipe.templates)
                or not self.source_unit_ids
            ):
                raise DomainError("derived scenario recipe lineage is incomplete")
            require_entity_id(self.source_episode_id, "derived scenario source episode ID")
            require_digest(
                self.source_child_scientific_key,
                "derived scenario source child scientific key",
            )
            require_entity_id(self.parent_unit_id, "derived scenario parent unit ID")


@dataclass(frozen=True, slots=True)
class ScreeningBlockView:
    """The exact fixed three-root screen, separate from later verification."""

    candidate_id: str
    candidate_scientific_key: str
    block_id: str
    triggering_execution_id: str
    new_execution_ids: tuple[str, str]
    target_action_ids: tuple[str, str, str]
    observation_ids: tuple[str, ...]
    decision: ScreeningDecision

    @property
    def block_terminated(self) -> bool:
        return self.decision.block_terminated

    @property
    def status(self) -> SearchStatus:
        return self.decision.status

    @property
    def effective_scores(self) -> tuple[Decimal, ...]:
        return self.decision.effective_scores


@dataclass(frozen=True, slots=True)
class StrategyQualificationView:
    """Reducer-owned state of one transcript-free strategy qualification."""

    template_id: str
    template_scientific_key: str
    template: StrategyTemplate
    source_episode_id: str
    source_conversation_root_id: str
    source_conversation_root_scientific_key: str
    source_generation_parent_node_id: str
    source_generation_parent_node_scientific_key: str
    source_entry_candidate_id: str
    source_entry_candidate_scientific_key: str
    tested_entry_context_ids: tuple[str, ...]
    development_unit_id: str | None
    development_design: Mapping[str, Any] | None
    sealed_holdout_plan: Mapping[str, Any] | None
    sealed_holdout_digest: str | None
    development_roots: tuple[QualificationRootOutcome, ...]
    development_decision: DevelopmentBlockDecision | None
    holdout_unit_id: str | None
    holdout_design: Mapping[str, Any] | None
    holdout_roots: tuple[QualificationRootOutcome, ...]
    holdout_decision: HoldoutBlockDecision | None
    qualification_state: QualificationState
    required_structured_role_scope_keys: tuple[str, ...]
    required_selector_scope_keys: tuple[tuple[str, str], ...]
    required_judge_scope_keys: tuple[str, ...]
    required_role_scope_keys: tuple[str, ...]
    evidence_event_ids: tuple[str, ...]
    source_checkpoint: Mapping[str, Any] | None


@dataclass(frozen=True, slots=True)
class RunState:
    events: tuple[Mapping[str, Any], ...]
    run_id: str
    manifest: Mapping[str, Any]
    scientific_scope_digest: str
    work_units: tuple[WorkUnitView, ...]
    active_work_unit_id: str | None
    completed_work_unit_ids: tuple[str, ...]
    attempts: Mapping[str, AttemptView]
    logical_actions: Mapping[str, LogicalActionView]
    consumed: RoleCallVector
    remaining: RoleCallVector
    target_results: Mapping[str, TargetResultView]
    target_observations: Mapping[str, TargetObservation]
    member_assessments: Mapping[str, MemberAssessment]
    panel_assessments: Mapping[str, PanelAssessment]
    panel_views: Mapping[str, PanelView]
    sample_executions: Mapping[str, SampleExecution]
    execution_evidence: Mapping[str, ExecutionEvidence]
    conversation_roots: Mapping[str, ConversationRootView]
    conversation_nodes: Mapping[str, ConversationNode]
    candidate_evidence: Mapping[str, EvidenceRecord]
    screening_blocks: Mapping[str, ScreeningBlockView]
    archive: ArchiveSnapshot
    portfolio_rounds: Mapping[str, PortfolioRoundView]
    strategy_episodes: Mapping[str, StrategyEpisodeView]
    strategy_supports_ordering: tuple[str, ...]
    scenarios: Mapping[str, ScenarioRecipeView]
    seed_qualifications: Mapping[str, SeedQualificationDecision]
    scenario_evidence: Mapping[str, EvidenceRecord]
    current_seed_qualification_snapshots: tuple[Mapping[str, Any], ...]
    strategy_qualifications: Mapping[str, StrategyQualificationView]
    current_strategy_qualification_snapshots: tuple[Mapping[str, Any], ...]
    dangling_attempt_ids: tuple[str, ...]
    target_identity: str | None
    target_identity_unverifiable: bool
    target_identity_drift: bool
    target_identity_mismatch_attempt_ids: tuple[str, ...]
    role_target_identity_collision_attempt_ids: Mapping[str, tuple[str, ...]]
    role_identity_unverifiable_attempt_ids: Mapping[str, tuple[str, ...]]
    terminal_event: Mapping[str, Any] | None
    cost_liability_usd: Decimal = Decimal(0)

    @property
    def completed_target_responses(self) -> int:
        return sum(result.usable for result in self.target_results.values())

    @property
    def all_target_attempts(self) -> int:
        return self.consumed.target

    @property
    def active_work_unit(self) -> WorkUnitView | None:
        if self.active_work_unit_id is None:
            return None
        return next(
            unit for unit in self.work_units if unit.unit_id == self.active_work_unit_id
        )

    @property
    def completed_work_units(self) -> tuple[WorkUnitView, ...]:
        completed = frozenset(self.completed_work_unit_ids)
        return tuple(unit for unit in self.work_units if unit.unit_id in completed)

    @property
    def terminal_reason(self) -> RunTerminalReason | None:
        if self.terminal_event is None:
            return None
        return RunTerminalReason(self.terminal_event["payload"]["reason"])

    @property
    def resumable(self) -> bool:
        return bool(self.terminal_event and self.terminal_event["payload"]["resumable"])

    @property
    def last_completed_work_unit_id(self) -> str | None:
        if self.terminal_event is None:
            return None
        value = self.terminal_event["payload"]["last_completed_work_unit_id"]
        return value if isinstance(value, str) else None

    def attempt(self, attempt_id: str) -> AttemptView:
        try:
            return self.attempts[attempt_id]
        except KeyError as exc:
            raise DomainError(f"unknown physical attempt: {attempt_id}") from exc

    def action(self, action_id: str) -> LogicalActionView:
        try:
            return self.logical_actions[action_id]
        except KeyError as exc:
            raise DomainError(f"unknown logical action: {action_id}") from exc

    def latest_success(self, action_id: str) -> AttemptView | None:
        return self.action(action_id).successful_attempt

    def archive_metric(self, *, total_possible_cells: int) -> ArchiveMetric:
        """Return the canonical archive endpoint at this replay checkpoint."""

        return derive_archive_metric(
            self.archive,
            total_possible_cells=total_possible_cells,
            all_target_attempts=self.all_target_attempts,
            completed_target_responses=self.completed_target_responses,
        )

    def verified_archive_cells(self) -> tuple[tuple[str, ...], ...]:
        """Return descriptor cells with current verification evidence."""

        return derive_verified_archive_cells(self.archive)

    def qualified_seed_verification_median(self, seed_id: str) -> Decimal:
        """Return the canonical verification median for one qualified seed."""

        require_entity_id(seed_id, "qualified seed ID")
        if not seed_id.startswith("sd-"):
            raise DomainError("qualified seed ID must use sd- prefix")
        decision = self.seed_qualifications.get(seed_id)
        if decision is None or not decision.qualified or decision.snapshot is None:
            raise DomainError("seed does not have current qualified verification evidence")
        result = median_score(decision.verification.effective_scores)
        if result is None:
            raise DomainError("qualified seed verification has no effective-score median")
        return result

    def demonstrated_repeat_supported_subject_ids(self) -> frozenset[str]:
        """Return verified-block subjects supported by demonstrated-only votes."""

        panel_by_target = {
            panel.target_action_id: panel for panel in self.panel_views.values()
        }
        supported: set[str] = set()
        for work in self.work_units:
            if work.unit.get("kind") != "verification":
                continue
            design = work.unit.get("design")
            if not isinstance(design, Mapping):
                raise DomainError("verification work unit lacks a design object")
            planned = design.get("planned_count")
            if (
                isinstance(planned, bool)
                or not isinstance(planned, int)
                or planned <= 0
            ):
                raise DomainError("verification work unit has invalid planned count")
            root_plans = design.get("root_plans")
            if not isinstance(root_plans, (list, tuple)):
                raise DomainError("verification work unit lacks root plans")
            demonstrated = 0
            for root in root_plans:
                if not isinstance(root, Mapping):
                    raise DomainError("verification root plan is malformed")
                target_action_id = root.get("terminal_endpoint_action_id")
                if not isinstance(target_action_id, str):
                    raise DomainError("verification root lacks its terminal endpoint")
                panel = panel_by_target.get(target_action_id)
                if panel is None or panel.details is None:
                    continue
                if (
                    panel.terminated
                    and panel.details.assessment.confirmed_objective_high
                    and panel.details.counts_by_verdict.get("demonstrated", 0) >= 2
                ):
                    demonstrated += 1
            if demonstrated_only_block_passes(demonstrated, planned):
                subject_id = work.unit.get("subject_id")
                if not isinstance(subject_id, str):
                    raise DomainError("verification work unit lacks its subject ID")
                supported.add(subject_id)
        return frozenset(supported)


def _budget_limit(manifest: Mapping[str, Any]) -> RoleCallVector:
    try:
        return RoleCallVector(**manifest["physical_call_limits"])
    except (KeyError, TypeError, ValueError) as exc:
        raise JournalCorrupt("run_created manifest has invalid budget fields") from exc


def _validate_invocation_lifecycle(
    events: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    """Validate invocation boundaries and return the current invocation terminal.

    A resume supersedes an invocation that either terminated resumably or was
    interrupted by a process crash.  Prior terminals remain audited facts, but
    cease to be the terminal state of the newly opened invocation.
    """

    created = events[0]
    creation_invocation_id = created["payload"]["creation_invocation_id"]
    opened_invocation_ids = {creation_invocation_id}
    recovery_invocation_ids: set[str] = set()
    active_invocation_id = creation_invocation_id
    active_terminal: Mapping[str, Any] | None = None
    open_dispatches: set[str] = set()
    expected_recovery_closures: list[str] = []
    pending_recovery_invocation_id: str | None = None
    pending_recovery_actions: list[str] = []

    for index, event in enumerate(events[1:], 1):
        previous = events[index - 1]
        event_type = event["type"]
        payload = event["payload"]

        if pending_recovery_invocation_id is not None and event_type not in {
            "journal_tail_recovered",
            "run_resumed",
        }:
            raise JournalCorrupt(
                "scientific execution continued after tail recovery without run_resumed"
            )

        if expected_recovery_closures and event_type not in {
            "model_call_outcome_unknown",
            "journal_tail_recovered",
            "run_resumed",
        }:
            raise JournalCorrupt(
                "resume did not close its declared dangling dispatches first"
            )

        if active_terminal is not None and event_type not in {
            "journal_tail_recovered",
            "run_resumed",
        }:
            raise JournalCorrupt(
                "an invocation continued after run_terminated without run_resumed"
            )

        if event_type == "journal_tail_recovered":
            recovery_id = payload["recovery_invocation_id"]
            if recovery_id in opened_invocation_ids:
                raise JournalCorrupt(
                    "tail recovery names an already opened invocation"
                )
            if (
                pending_recovery_invocation_id is not None
                and recovery_id != pending_recovery_invocation_id
            ):
                # The prior recovery process may itself have crashed before it
                # could append run_resumed.  A later distinct recovery
                # invocation is valid, but the abandoned ID cannot be reused.
                pending_recovery_invocation_id = None
                pending_recovery_actions = []
            if pending_recovery_invocation_id is None:
                if recovery_id in recovery_invocation_ids:
                    raise JournalCorrupt("tail recovery invocation ID was reused")
                recovery_invocation_ids.add(recovery_id)
                pending_recovery_invocation_id = recovery_id
            if payload["action"] in pending_recovery_actions:
                raise JournalCorrupt(
                    "one recovery invocation recorded a duplicate tail action"
                )
            if event["causes"] != [previous["event_id"]]:
                raise JournalCorrupt(
                    "journal_tail_recovered must be caused by the recovered chain tail"
                )
            if payload["resulting_terminal_hash"] != previous["event_hash"]:
                raise JournalCorrupt(
                    "journal tail recovery hash does not match the preceding event"
                )
            pending_recovery_actions.append(payload["action"])
            continue

        if event_type == "run_resumed":
            invocation_id = payload["invocation_id"]
            if invocation_id in opened_invocation_ids:
                raise JournalCorrupt("invocation ID was opened more than once")
            if (
                invocation_id in recovery_invocation_ids
                and invocation_id != pending_recovery_invocation_id
            ):
                raise JournalCorrupt("an abandoned recovery invocation ID was reused")
            if active_terminal is not None and not active_terminal["payload"]["resumable"]:
                raise JournalCorrupt("a nonresumable terminal was resumed")
            if event["causes"] != [previous["event_id"]]:
                raise JournalCorrupt(
                    "run_resumed must be caused by the immediately preceding event"
                )
            if payload["validated_previous_hash"] != previous["event_hash"]:
                raise JournalCorrupt(
                    "run_resumed validated_previous_hash is not the preceding hash"
                )
            prior_reason = (
                None
                if active_terminal is None
                else active_terminal["payload"]["reason"]
            )
            if payload["prior_terminal_reason"] != prior_reason:
                raise JournalCorrupt(
                    "run_resumed prior terminal reason does not match replay state"
                )

            declared_actions = list(payload["recovery_actions"])
            declared_tail_actions = [
                action
                for action in declared_actions
                if action not in {"none", "dangling_dispatch_closed"}
            ]
            observed_actions = (
                pending_recovery_actions
                if invocation_id == pending_recovery_invocation_id
                else []
            )
            if observed_actions:
                if declared_tail_actions != observed_actions:
                    raise JournalCorrupt(
                        "run_resumed recovery actions differ from tail recovery facts"
                    )
            elif any(action != "pending_removed" for action in declared_tail_actions):
                # Removing an already-durable pending duplicate is cleanup-only
                # and may be reported directly by run_resumed without a
                # journal_tail_recovered fact.  The other tail actions always
                # create a recovery event.
                raise JournalCorrupt(
                    "run_resumed claims a tail repair absent from the journal"
                )

            dangling = list(payload["dangling_attempt_ids"])
            if dangling != sorted(open_dispatches):
                raise JournalCorrupt(
                    "run_resumed dangling attempts differ from replayed dispatches"
                )
            expected_recovery_closures = dangling
            opened_invocation_ids.add(invocation_id)
            active_invocation_id = invocation_id
            active_terminal = None
            pending_recovery_invocation_id = None
            pending_recovery_actions = []
            continue

        if event_type == "model_call_outcome_unknown":
            if payload["discovery_invocation_id"] != active_invocation_id:
                raise JournalCorrupt(
                    "unknown-outcome closure names a foreign invocation"
                )
            if expected_recovery_closures:
                if payload["physical_attempt_id"] != expected_recovery_closures[0]:
                    raise JournalCorrupt(
                        "dangling dispatches were not closed in stable attempt-ID order"
                    )
                expected_recovery_closures.pop(0)

        if event_type == "model_dispatch_started":
            open_dispatches.add(payload["physical_attempt_id"])
        elif event_type in TERMINAL_CALL_TYPES:
            open_dispatches.discard(payload["physical_attempt_id"])

        if event_type == "run_terminated":
            if payload["invocation_id"] != active_invocation_id:
                raise JournalCorrupt("run_terminated closes a foreign invocation")
            expected_resumable = _TERMINAL_RESUMABILITY[payload["reason"]]
            if payload["resumable"] is not expected_resumable:
                raise JournalCorrupt(
                    "run_terminated resumable flag disagrees with its reason"
                )
            if event["causes"] != [previous["event_id"]]:
                raise JournalCorrupt(
                    "run_terminated must be caused by the immediately preceding event"
                )
            if open_dispatches:
                raise JournalCorrupt(
                    "run_terminated was appended with dangling dispatches"
                )
            active_terminal = event

    return active_terminal


def _slot_indexes(
    events: Sequence[Mapping[str, Any]],
    budget_limit: RoleCallVector,
    terminal_event: Mapping[str, Any] | None,
    policy_digest: str,
) -> tuple[
    list[tuple[Mapping[str, Any], Mapping[str, Any]]],
    dict[str, Mapping[str, Any]],
    dict[str, str],
    str | None,
    tuple[str, ...],
]:
    """Validate unit conformance, admission, and exclusive attempt ownership.

    Completion is deliberately not inferred here.  It depends on authenticated
    results and the frozen action graph, both of which are reduced later.  This
    first pass only establishes the unit/action indexes needed to authenticate
    those facts.
    """

    units: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    slots: dict[str, Mapping[str, Any]] = {}
    action_units: dict[str, str] = {}
    active_unit_id: str | None = None
    attempt_units: dict[str, str] = {}
    unsettled_attempt_ids: set[str] = set()
    consumed = {"target": 0, "judge": 0, "selector": 0, "perturbator": 0}

    for event in events:
        event_type = event["type"]
        if event_type == "work_unit_started":
            if unsettled_attempt_ids:
                raise JournalCorrupt(
                    "a new work unit started while the prior unit had an unsettled attempt"
                )
            unit = event["payload"]["unit"]
            unit_id = unit["unit_id"]
            if any(existing[1]["unit_id"] == unit_id for existing in units):
                raise JournalCorrupt("work unit started more than once")
            try:
                frozen = WorkUnit(
                    unit_id=unit["unit_id"],
                    unit_scientific_key=unit["unit_scientific_key"],
                    kind=unit["kind"],
                    subject_id=unit["subject_id"],
                    subject_scientific_key=unit["subject_scientific_key"],
                    parent_unit_id=unit["parent_unit_id"],
                    policy_digest=unit["policy_digest"],
                    design=unit["design"],
                    proposal_slots=tuple(unit["proposal_slots"]),
                    execution_slots=tuple(unit["execution_slots"]),
                    panel_plans=tuple(unit["panel_plans"]),
                    logical_action_slots=tuple(unit["logical_action_slots"]),
                    worst_case_calls=RoleCallVector(**unit["worst_case_calls"]),
                )
                required = validate_work_unit_maximum(
                    frozen, expected_policy_digest=policy_digest
                )
            except (DomainError, SchedulerValidationError, KeyError, TypeError) as exc:
                raise JournalCorrupt(f"frozen work unit is nonconformant: {exc}") from exc

            if (
                unit["kind"] == "verification"
                and unit["design"].get("subject_kind") == "scenario_recipe"
                and unit["parent_unit_id"] is not None
            ):
                parents = [
                    prior
                    for _, prior in units
                    if prior["unit_id"] == unit["parent_unit_id"]
                ]
                if len(parents) != 1 or parents[0]["kind"] != "portfolio_round":
                    raise JournalCorrupt(
                        "derived scenario verification must cite one earlier portfolio unit"
                    )

            remaining_at_start = RoleCallVector(
                target=budget_limit.target - consumed["target"],
                judge=budget_limit.judge - consumed["judge"],
                selector=budget_limit.selector - consumed["selector"],
                perturbator=budget_limit.perturbator - consumed["perturbator"],
            )
            if not required.fits(remaining_at_start):
                raise JournalCorrupt("work unit did not fit remaining role budgets at start")

            for slot in unit["logical_action_slots"]:
                action_id = slot["action_id"]
                if action_id in slots:
                    raise JournalCorrupt("logical action ID appears in several work units")
                slots[action_id] = slot
                action_units[action_id] = unit_id
            units.append((event, unit))
            active_unit_id = unit_id
            continue

        if event_type == "model_call_intended":
            payload = event["payload"]
            if active_unit_id is None or payload["unit_id"] != active_unit_id:
                raise JournalCorrupt("model intent is outside the active work unit")
            attempt_id = payload["physical_attempt_id"]
            attempt_units[attempt_id] = payload["unit_id"]
            unsettled_attempt_ids.add(attempt_id)
            continue

        if event_type == "model_dispatch_started":
            payload = event["payload"]
            attempt_id = payload["physical_attempt_id"]
            if active_unit_id is None or attempt_units.get(attempt_id) != active_unit_id:
                raise JournalCorrupt("model dispatch is outside the active work unit")
            role = payload["charged_role"]
            consumed[role] += 1
            if consumed[role] > getattr(budget_limit, role):
                raise JournalCorrupt("persisted dispatches exceed a role budget")
            continue

        if event_type in TERMINAL_CALL_TYPES:
            unsettled_attempt_ids.discard(event["payload"]["physical_attempt_id"])

    del terminal_event
    return units, slots, action_units, active_unit_id, ()


_PURPOSE_SCHEMA_KEYS = {
    ActionPurpose.JUDGE_MEMBER.value: "judge_output",
    ActionPurpose.SELECTOR_PORTFOLIO.value: "selector_portfolio",
    ActionPurpose.PROBE_REVIEW.value: "probe_review",
    ActionPurpose.TEMPLATE_BINDING.value: "template_binding",
    ActionPurpose.OUTCOME_REVIEW.value: "outcome_review",
    ActionPurpose.PERTURBATOR_PROPOSAL.value: "proposal_batch",
}
_PURPOSE_SYSTEM_TEMPLATES = {
    ActionPurpose.JUDGE_MEMBER.value: "judge_system",
    ActionPurpose.SELECTOR_PORTFOLIO.value: "selector_system",
    ActionPurpose.PROBE_REVIEW.value: "probe_review_system",
    ActionPurpose.TEMPLATE_BINDING.value: "template_binding_system",
    ActionPurpose.OUTCOME_REVIEW.value: "outcome_review_system",
    ActionPurpose.PERTURBATOR_PROPOSAL.value: "perturbator_system",
}
_ROLE_PURPOSES = {
    Role.TARGET.value: frozenset({ActionPurpose.TARGET_SAMPLE.value}),
    Role.JUDGE.value: frozenset({ActionPurpose.JUDGE_MEMBER.value}),
    Role.SELECTOR.value: frozenset(
        {
            ActionPurpose.SELECTOR_PORTFOLIO.value,
            ActionPurpose.PROBE_REVIEW.value,
            ActionPurpose.TEMPLATE_BINDING.value,
            ActionPurpose.OUTCOME_REVIEW.value,
        }
    ),
    Role.PERTURBATOR.value: frozenset({ActionPurpose.PERTURBATOR_PROPOSAL.value}),
}


def _validated_library_checkpoint(
    value: object, label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "source_run_id",
        "sequence",
        "event_hash",
        "source_path",
    }:
        raise JournalCorrupt(f"{label} is not one closed evidence checkpoint")
    try:
        source_run_id = require_entity_id(value["source_run_id"], f"{label} run ID")
        if not source_run_id.startswith("run-"):
            raise DomainError(f"{label} run ID has the wrong entity kind")
        sequence = value["sequence"]
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
            raise DomainError(f"{label} sequence must be a positive integer")
        event_hash = require_digest(value["event_hash"], f"{label} event hash")
        source_path = value["source_path"]
        if not isinstance(source_path, str) or not source_path:
            raise DomainError(f"{label} source path must be nonempty text")
    except (KeyError, TypeError, DomainError) as exc:
        raise JournalCorrupt(f"{label} is invalid: {exc}") from exc
    return {
        "source_run_id": source_run_id,
        "sequence": sequence,
        "event_hash": event_hash,
        "source_path": source_path,
    }


def _validate_library_snapshot_links(manifest: Mapping[str, Any]) -> None:
    """Authenticate the exact one-to-one provenance links frozen at creation."""

    expected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(
        *,
        kind: str,
        subject_id: object,
        compatibility_key: object,
        checkpoint: object,
        expected_prefix: str,
    ) -> None:
        try:
            identifier = require_entity_id(subject_id, f"{kind} snapshot subject ID")
            if not identifier.startswith(f"{expected_prefix}-"):
                raise DomainError(f"{kind} snapshot subject has the wrong entity kind")
            key = require_digest(
                compatibility_key, f"{kind} snapshot compatibility key"
            )
        except (TypeError, DomainError) as exc:
            raise JournalCorrupt(f"{kind} snapshot identity is invalid: {exc}") from exc
        identity = (kind, identifier)
        if identity in seen:
            raise JournalCorrupt(f"manifest repeats one {kind} snapshot subject")
        seen.add(identity)
        expected.append(
            {
                "kind": kind,
                "subject_id": identifier,
                "compatibility_key": key,
                "source": _validated_library_checkpoint(
                    checkpoint, f"{kind} snapshot source checkpoint"
                ),
            }
        )

    try:
        seeds = manifest["seed_snapshot"]
        strategies = manifest["strategy_template_snapshot"]
        recorded = manifest["library_source_checkpoints"]
    except KeyError as exc:
        raise JournalCorrupt(f"manifest library snapshot is incomplete: {exc}") from exc
    if not all(isinstance(items, list) for items in (seeds, strategies, recorded)):
        raise JournalCorrupt("manifest library snapshots and links must be arrays")

    for snapshot in seeds:
        if not isinstance(snapshot, Mapping):
            raise JournalCorrupt("manifest seed snapshot item must be an object")
        add(
            kind="seed",
            subject_id=snapshot.get("seed_id"),
            compatibility_key=snapshot.get("compatibility_key"),
            checkpoint=snapshot.get("source_checkpoint"),
            expected_prefix="sd",
        )
    for snapshot in strategies:
        if not isinstance(snapshot, Mapping):
            raise JournalCorrupt("manifest strategy snapshot item must be an object")
        if snapshot.get("qualification_status") != "provisionally_compatible":
            raise JournalCorrupt("manifest strategy prior is not explicitly provisional")
        add(
            kind="strategy_template",
            subject_id=snapshot.get("template_id"),
            compatibility_key=snapshot.get("compatibility_key"),
            checkpoint=snapshot.get("source_checkpoint"),
            expected_prefix="tp",
        )
    expected.sort(
        key=lambda item: (
            item["kind"],
            item["subject_id"],
            item["compatibility_key"],
        )
    )
    if recorded != expected:
        raise JournalCorrupt(
            "manifest library source checkpoints do not exactly cover selected priors"
        )


def _validate_request_manifest(
    manifest: Mapping[str, Any], objective: ObjectivePlugin | None
) -> None:
    try:
        if manifest["model_profiles"] != manifest["normalized_config"]["models"]:
            raise JournalCorrupt(
                "manifest model profiles differ from normalized configuration"
            )
        provider = manifest["normalized_config"]["provider"]
        timeout = provider["timeout_seconds"]
        if isinstance(timeout, bool) or not isinstance(timeout, int):
            raise JournalCorrupt("manifest provider timeout is invalid")
        schema_digests = manifest["response_schema_digests"]
        if set(schema_digests) != set(_PURPOSE_SCHEMA_KEYS.values()):
            raise JournalCorrupt("manifest role response-schema digest set is invalid")
        for name, digest in schema_digests.items():
            require_digest(digest, f"manifest {name} schema digest")
        if digest_json(manifest["judge_output_schema"]) != schema_digests["judge_output"]:
            raise JournalCorrupt("frozen judge schema template digest is invalid")
        constants = manifest["decision_constants"]
        if not isinstance(constants, Mapping) or set(constants) != set(
            DECISION_CONSTANTS
        ):
            raise JournalCorrupt("manifest decision constant set is invalid")
        for name, current in DECISION_CONSTANTS.items():
            frozen = constants[name]
            if isinstance(current, int):
                if (
                    isinstance(frozen, bool)
                    or not isinstance(frozen, int)
                    or frozen <= 0
                ):
                    raise JournalCorrupt(
                        f"manifest decision constant {name} is invalid"
                    )
            elif (
                not isinstance(frozen, str)
                or normalize_decimal(frozen) != frozen
            ):
                raise JournalCorrupt(
                    f"manifest decision constant {name} is invalid"
                )
        taxonomy = manifest["descriptor_taxonomy"]
        if (
            not isinstance(taxonomy, Mapping)
            or set(taxonomy) != {"descriptors", "digest"}
            or taxonomy["digest"] != digest_json(taxonomy["descriptors"])
        ):
            raise JournalCorrupt("manifest descriptor taxonomy digest is invalid")
        if objective is not None:
            expected_objective = {
                "id": objective.id,
                "plugin_digest": objective.plugin_digest,
                "artifact_digests": dict(objective.artifact_digests),
            }
            if manifest["objective"] != expected_objective:
                raise JournalCorrupt(
                    "manifest objective snapshot differs from the selected plugin"
                )
            expected_descriptors = list(objective.objective["descriptors"])
            if taxonomy["descriptors"] != expected_descriptors:
                raise JournalCorrupt(
                    "manifest descriptor taxonomy differs from the selected plugin"
                )
            bootstrap = manifest["bootstrap_snapshot"]
            if not isinstance(bootstrap, Mapping) or bootstrap.get(
                "objective_id"
            ) != objective.id:
                raise JournalCorrupt("manifest bootstrap objective is invalid")
    except (KeyError, TypeError, DomainError) as exc:
        raise JournalCorrupt(f"manifest request bindings are invalid: {exc}") from exc
    _validate_library_snapshot_links(manifest)


def configured_judge_slots(manifest: Mapping[str, Any], *, include_replacement: bool = True) -> tuple[str, ...]:
    judges = manifest["model_profiles"]["judges"]
    slots = ("primary", "secondary_a", "secondary_b")[:1 + len(judges["secondaries"])]
    return slots + (("replacement",) if include_replacement and judges["replacement"] is not None else ())


def _profile_for_binding(
    manifest: Mapping[str, Any], role: str, configured_slot: str
) -> tuple[dict[str, Any], str]:
    """Return the exact ModelProfile digest material and selected model."""

    try:
        models = manifest["model_profiles"]
        if role == Role.TARGET.value and configured_slot == "target":
            raw = models["target"]
            model = raw["model"]
            system_prompt = raw["system_prompt"]
        elif role == Role.JUDGE.value:
            judges = models["judges"]
            raw = {
                "primary": judges["primary"],
                "secondary_a": judges["secondaries"][0],
                "secondary_b": judges["secondaries"][1] if len(judges["secondaries"]) > 1 else None,
                "replacement": judges["replacement"],
            }[configured_slot]
            model = raw["model"]
            system_prompt = ""
        elif role == Role.SELECTOR.value and configured_slot in {
            "selector_primary",
            "selector_fallback",
        }:
            raw = models["selector"]
            model = (
                raw["model"]
                if configured_slot == "selector_primary"
                else raw["fallback_model"]
            )
            system_prompt = ""
        elif role == Role.PERTURBATOR.value and configured_slot in {
            "perturbator_primary",
            "perturbator_fallback",
        }:
            raw = models["perturbator"]
            model = (
                raw["model"]
                if configured_slot == "perturbator_primary"
                else raw["fallback_model"]
            )
            system_prompt = ""
        else:
            raise KeyError(configured_slot)
        profile = {
            "model": model,
            "temperature": raw["temperature"],
            "max_output_tokens": raw["max_output_tokens"],
            "system_prompt": system_prompt,
            **{name: raw[name] for name in ("reasoning_effort", "provider_only", "max_price", "max_context_tokens") if name in raw},
        }
        return profile, model
    except (KeyError, IndexError, TypeError) as exc:
        raise JournalCorrupt(
            "route configured slot does not resolve to its frozen model profile"
        ) from exc


def _expected_route_label(role: str, configured_slot: str) -> str:
    if role == Role.JUDGE.value and configured_slot == "replacement":
        return "replacement"
    if configured_slot in {"selector_fallback", "perturbator_fallback"}:
        return "fallback"
    return "primary"


def model_cost_reservation(profile: Mapping[str, Any]) -> Decimal:
    """Full advertised input context plus maximum output, at frozen USD/M rates."""
    prices = profile.get("max_price")
    context = profile.get("max_context_tokens")
    if not isinstance(prices, Mapping) or not isinstance(context, int) or isinstance(context, bool) or context <= 0:
        raise DomainError("dollar admission requires maximum prices and context length")
    return (Decimal(context) * Decimal(prices["prompt"])
            + Decimal(profile["max_output_tokens"]) * Decimal(prices["completion"])) / Decimal(1000000) + Decimal(prices["request"])


def reduce_cost_accounting(
    events: Sequence[Mapping[str, Any]],
    attempts: Mapping[str, AttemptView],
    slots: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> Decimal:
    """Authenticate each monetary admission from journal facts; retain unknowns."""
    ceiling = manifest["normalized_config"]["provider"].get("max_cost_usd")
    if ceiling is None:
        return Decimal(0)
    limit = Decimal(ceiling)
    liabilities: dict[str, Decimal] = {}
    for event in events:
        if event["type"] == "model_dispatch_started":
            attempt_id = event["payload"]["physical_attempt_id"]
            intended = attempts[attempt_id].intended
            slot = slots[intended["logical_action_id"]]
            binding = next(route for route in slot["routes"] if route["route_key"] == intended["role_route_key"])
            profile, _ = _profile_for_binding(manifest, intended["role"], binding["configured_slot"])
            reserve = model_cost_reservation(profile)
            if sum(liabilities.values(), Decimal(0)) + reserve > limit:
                raise JournalCorrupt("physical dispatch exceeded its frozen dollar admission ceiling")
            liabilities[attempt_id] = reserve
        elif event["type"] == "model_call_succeeded":
            payload = event["payload"]
            if payload["cost"] is not None:
                # Exact receipts can release a reservation. An unexpectedly high
                # receipt remains a fact and prevents further spending.
                liabilities[payload["physical_attempt_id"]] = Decimal(payload["cost"])
    return sum(liabilities.values(), Decimal(0))


def _response_schema_from_body(
    payload: Mapping[str, Any], body: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    if payload["response_mode"] != "json_schema":
        return None
    try:
        response_format = body["response_format"]
        if not isinstance(response_format, Mapping):
            raise TypeError
        json_schema = response_format["json_schema"]
        if not isinstance(json_schema, Mapping):
            raise TypeError
        schema = json_schema["schema"]
        if not isinstance(schema, Mapping):
            raise TypeError
        return schema
    except (KeyError, TypeError) as exc:
        raise JournalCorrupt(
            "json_schema request body does not embed one response schema"
        ) from exc


def _authenticate_intent(
    *,
    manifest: Mapping[str, Any],
    payload: Mapping[str, Any],
    slot: Mapping[str, Any],
    binding: Mapping[str, Any],
    judge_target_reply: str | None = None,
) -> Mapping[str, Any] | None:
    """Recompute one complete intended request from frozen audit material."""

    try:
        role = payload["role"]
        purpose = payload["purpose"]
        if purpose not in _ROLE_PURPOSES[role]:
            raise JournalCorrupt("model intent role/purpose pairing is invalid")
        expected_action_key = digest_json(
            [
                slot["owner_position_scientific_key"],
                "action",
                role,
                purpose,
                slot["ordinal"],
            ]
        )
        if slot["action_scientific_key"] != expected_action_key:
            raise JournalCorrupt("logical action scientific key is invalid")
        configured_slot = binding["configured_slot"]
        route_label = _expected_route_label(role, configured_slot)
        if binding["route_label"] != route_label:
            raise JournalCorrupt("frozen route label/configured slot pairing is invalid")

        adapter_kind = manifest["normalized_config"]["provider"]["adapter"]
        provider_key = requested_provider_route_key(adapter_kind, role, route_label)
        if binding["requested_provider_route_key"] != provider_key:
            raise JournalCorrupt("frozen provider route key is invalid")

        profile, expected_model = _profile_for_binding(
            manifest, role, configured_slot
        )
        generation_digest = digest_json(profile)
        if binding["generation_profile_digest"] != generation_digest:
            raise JournalCorrupt("frozen route generation-profile digest is invalid")
        if binding["requested_model"] != expected_model:
            raise JournalCorrupt("frozen route requested model is invalid")
        expected_route_key = digest_json(
            [
                manifest["scientific_scope_digest"],
                "role_route",
                role,
                configured_slot,
                route_label,
                provider_key,
                expected_model,
                generation_digest,
            ]
        )
        if binding["route_key"] != expected_route_key:
            raise JournalCorrupt("frozen role route key is invalid")

        expected_generation = {
            "temperature": profile["temperature"],
            "max_output_tokens": profile["max_output_tokens"],
            **{name: profile[name] for name in ("reasoning_effort", "provider_only", "max_price") if name in profile},
        }
        if payload["generation_parameters"] != expected_generation:
            raise JournalCorrupt("model intent generation parameters are not frozen")
        if payload["requested_model"] != expected_model:
            raise JournalCorrupt("model intent requested model is not frozen")
        expected_timeout = manifest["normalized_config"]["provider"]["timeout_seconds"]
        if payload["timeout_seconds"] != expected_timeout:
            raise JournalCorrupt("model intent timeout differs from frozen configuration")

        schema_key = _PURPOSE_SCHEMA_KEYS.get(purpose)
        if schema_key is None:
            if payload["response_mode"] != "text" or payload["response_schema_digest"] is not None:
                raise JournalCorrupt("target intent has a structured response contract")
            expected_schema_digest = None
        else:
            expected_schema_digest = manifest["response_schema_digests"][schema_key]
            if role == Role.JUDGE.value:
                template = manifest["judge_output_schema"]
                if "EvidenceId" in template.get("$defs", {}) and judge_target_reply is None:
                    raise JournalCorrupt("judge intent lacks its preceding durable target reply")
                expected_judge_schema = materialize_judge_schema(template, judge_target_reply or "")
                expected_schema_digest = digest_json(expected_judge_schema)
            if payload["response_schema_digest"] != expected_schema_digest:
                raise JournalCorrupt("model intent response schema is not frozen")
            if payload["response_mode"] not in {"json_schema", "json_object"}:
                raise JournalCorrupt("structured role intent uses text response mode")

        messages = tuple(
            ChatMessage(item["role"], item["content"])
            for item in payload["messages"]
        )
        if role == Role.TARGET.value:
            system_prompt = profile["system_prompt"]
            if system_prompt:
                if not messages or messages[0] != ChatMessage("system", system_prompt):
                    raise JournalCorrupt("target system prompt differs from its frozen profile")
            elif any(message.role == "system" for message in messages):
                raise JournalCorrupt("target intent introduced an unfrozen system prompt")
        else:
            if len(messages) != 2 or messages[0].role != "system" or messages[1].role != "user":
                raise JournalCorrupt("structured role request must contain system then user")
            template_name = _PURPOSE_SYSTEM_TEMPLATES[purpose]
            expected_template_digest = manifest["role_template_digests"][template_name]
            if sha256_bytes(messages[0].content.encode("utf-8")) != expected_template_digest:
                raise JournalCorrupt("model intent system template is not frozen")

        body_bytes = decode_bytes(payload["request_body"], "request_body")
        body = strict_json_loads(body_bytes)
        if not isinstance(body, Mapping):
            raise JournalCorrupt("model intent request body is not one JSON object")
        response_schema = _response_schema_from_body(payload, body)
        if response_schema is not None and digest_json(response_schema) != expected_schema_digest:
            raise JournalCorrupt("embedded response schema digest is invalid")
        if role == Role.JUDGE.value:
            response_schema = expected_judge_schema

        expected_attempt_id = entity_id(
            "pa",
            [
                payload["logical_action_id"],
                payload["route_kind"],
                payload["attempt_ordinal"],
            ],
        )
        if payload["physical_attempt_id"] != expected_attempt_id:
            raise JournalCorrupt("physical attempt ID is not derived from its frozen identity")

        request = ChatRequest(
            logical_action_id=payload["logical_action_id"],
            logical_action_scientific_key=payload["logical_action_scientific_key"],
            physical_attempt_id=payload["physical_attempt_id"],
            attempt_ordinal=payload["attempt_ordinal"],
            role=Role(role),
            purpose=ActionPurpose(purpose),
            route_kind=payload["route_kind"],
            role_route_key=payload["role_route_key"],
            requested_provider_route_key=payload["requested_provider_route_key"],
            messages=messages,
            requested_model=payload["requested_model"],
            generation_parameters=payload["generation_parameters"],
            response_mode=payload["response_mode"],
            response_schema=response_schema,
            response_schema_digest=payload["response_schema_digest"],
            timeout_seconds=payload["timeout_seconds"],
            idempotency_key=payload["idempotency_key"],
        )
        if request.semantic_request_hash != payload["semantic_request_hash"]:
            raise JournalCorrupt("model intent semantic request hash is invalid")
        if render_request_body(request) != body_bytes:
            raise JournalCorrupt("model intent request_body is not deterministic")
        return response_schema
    except JournalCorrupt:
        raise
    except (
        DomainError,
        ProviderProtocolError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise JournalCorrupt(f"model intent request authentication failed: {exc}") from exc


def _reduce_attempts(
    events: Sequence[Mapping[str, Any]],
    slots: Mapping[str, Mapping[str, Any]],
    action_units: Mapping[str, str],
    manifest: Mapping[str, Any],
) -> tuple[
    dict[str, AttemptView],
    dict[str, tuple[AttemptView, ...]],
    dict[str, Mapping[str, Any]],
]:
    mutable: dict[str, dict[str, Mapping[str, Any] | None]] = {}
    action_attempts: dict[str, list[AttemptView]] = {}
    response_schemas: dict[str, Mapping[str, Any]] = {}
    judge_targets: dict[str, str] = {}
    for event in events:
        if event["type"] == "work_unit_started":
            for panel in event["payload"]["unit"]["panel_plans"]:
                for member in (*panel["normal_members"], *([panel["replacement"]] if panel["replacement"] is not None else [])):
                    judge_targets[member["judge_action_id"]] = panel["target_action_id"]
    preceding_target_replies: dict[str, str] = {}
    for event in events:
        event_type = event["type"]
        if event_type == "model_call_intended":
            payload = event["payload"]
            attempt_id = payload["physical_attempt_id"]
            action_id = payload["logical_action_id"]
            if action_id not in slots:
                raise JournalCorrupt("model intent names no frozen logical action slot")
            slot = slots[action_id]
            if payload["unit_id"] != action_units[action_id]:
                raise JournalCorrupt("model intent unit/action mismatch")
            for name in ("action_scientific_key", "role", "purpose", "owner_position_scientific_key"):
                payload_name = {
                    "action_scientific_key": "logical_action_scientific_key"
                }.get(name, name)
                if slot.get(name) != payload.get(payload_name):
                    raise JournalCorrupt(f"model intent differs from frozen slot {name}")
            routes = slot.get("routes")
            if not isinstance(routes, list):
                raise JournalCorrupt("logical action slot routes are invalid")
            binding = next(
                (route for route in routes if route.get("route_key") == payload["role_route_key"]),
                None,
            )
            if binding is None:
                raise JournalCorrupt("model intent route was not precommitted")
            if binding.get("requested_provider_route_key") != payload["requested_provider_route_key"] or binding.get("requested_model") != payload["requested_model"]:
                raise JournalCorrupt("model intent route binding mismatch")
            embedded_schema = _authenticate_intent(
                manifest=manifest,
                payload=payload,
                slot=slot,
                binding=binding,
                judge_target_reply=preceding_target_replies.get(judge_targets.get(action_id, "")),
            )
            if embedded_schema is not None:
                schema_digest = payload["response_schema_digest"]
                prior_schema = response_schemas.get(schema_digest)
                if prior_schema is not None and digest_json(prior_schema) != digest_json(
                    embedded_schema
                ):
                    raise JournalCorrupt(
                        "one response schema digest names conflicting embedded schemas"
                    )
                response_schemas[schema_digest] = embedded_schema
            if attempt_id in mutable:
                raise JournalCorrupt("duplicate attempt intent")
            mutable[attempt_id] = {"intent": event, "dispatch": None, "terminal": None}
        elif event_type == "model_dispatch_started":
            payload = event["payload"]
            attempt_id = payload["physical_attempt_id"]
            if attempt_id not in mutable or mutable[attempt_id]["dispatch"] is not None:
                raise JournalCorrupt("dispatch lacks exactly one intent")
            intent = mutable[attempt_id]["intent"]
            assert intent is not None
            intended = intent["payload"]
            if event["causes"] != [intent["event_id"]]:
                raise JournalCorrupt("dispatch cause is not its immutable intent")
            if any(
                payload[name] != intended[intended_name]
                for name, intended_name in (
                    ("logical_action_id", "logical_action_id"),
                    ("role_route_key", "role_route_key"),
                    ("route_kind", "route_kind"),
                    ("provider_route_key", "requested_provider_route_key"),
                    ("requested_model", "requested_model"),
                    ("charged_role", "role"),
                )
            ):
                raise JournalCorrupt("dispatch differs from immutable intent")
            mutable[attempt_id]["dispatch"] = event
        elif event_type in TERMINAL_CALL_TYPES:
            payload = event["payload"]
            attempt_id = payload["physical_attempt_id"]
            if attempt_id not in mutable or mutable[attempt_id]["terminal"] is not None:
                raise JournalCorrupt("terminal call event lacks one open attempt")
            if mutable[attempt_id]["dispatch"] is None:
                raise JournalCorrupt("attempt terminated before dispatch")
            dispatch = mutable[attempt_id]["dispatch"]
            assert dispatch is not None
            if event["causes"] != [dispatch["event_id"]]:
                raise JournalCorrupt("attempt terminal cause is not its dispatch")
            mutable[attempt_id]["terminal"] = event
            intended = mutable[attempt_id]["intent"]["payload"]
            if (intended["role"] == Role.TARGET.value
                    and event_type == "model_call_succeeded"
                    and payload["finish_status"] == "complete" and payload["text"]):
                preceding_target_replies[intended["logical_action_id"]] = payload["text"]


    attempts: dict[str, AttemptView] = {}
    for attempt_id, pieces in mutable.items():
        intent = pieces["intent"]
        assert intent is not None
        view = AttemptView(attempt_id, intent, pieces["dispatch"], pieces["terminal"])
        attempts[attempt_id] = view
        action_id = intent["payload"]["logical_action_id"]
        action_attempts.setdefault(action_id, []).append(view)
    ordered: dict[str, tuple[AttemptView, ...]] = {}
    for action_id, views in action_attempts.items():
        views.sort(key=lambda view: view.intended["attempt_ordinal"])
        ordinals = [view.intended["attempt_ordinal"] for view in views]
        if ordinals != list(range(len(ordinals))):
            raise JournalCorrupt("attempt ordinals are not contiguous from zero")
        if len(views) > slots[action_id]["maximum_physical_attempts"]:
            raise JournalCorrupt("logical action exceeded its frozen attempt cap")
        for index, view in enumerate(views):
            parent = view.intended["parent_attempt_id"]
            if (index == 0 and parent is not None) or (
                index > 0 and parent != views[index - 1].physical_attempt_id
            ):
                raise JournalCorrupt("attempt retry parent chain is invalid")
        ordered[action_id] = tuple(views)
    return attempts, ordered, response_schemas


def _success_passes_base_structure(
    action: LogicalActionView,
    attempt: AttemptView,
    response_schemas: Mapping[str, Mapping[str, Any]],
    judge_target_reply: str | None = None,
) -> bool:
    if attempt.target_identity_collision:
        return False
    terminal = attempt.terminal_event
    if terminal is None or terminal["type"] != "model_call_succeeded":
        return False
    result = terminal["payload"]
    if result["finish_status"] != "complete":
        return False
    if action.slot["role"] == Role.TARGET.value:
        return bool(result["text"])
    schema_digest = attempt.intended["response_schema_digest"]
    schema = response_schemas.get(schema_digest)
    if schema is None:
        raise JournalCorrupt(
            "structured result has no authenticated embedded response schema"
        )
    try:
        parsed = strict_json_loads(result["text"])
        validate_json_schema_instance(parsed, schema)
    except (DomainError, JsonSchemaValidationError):
        return False
    if action.slot["role"] == Role.JUDGE.value and judge_target_reply is not None:
        # Exact evidence-quote grounding is part of the judge's structural
        # contract but cannot be expressed by its materialized JSON Schema.
        # Replay it here so a schema-repair intent following a bad quote is
        # authenticated by the same rule used by the engine and panel reducer.
        return validate_judge_output(result["text"], judge_target_reply, schema=schema).usable
    return True


def _success_is_structurally_usable(
    action: LogicalActionView,
    attempt: AttemptView,
    response_schemas: Mapping[str, Mapping[str, Any]],
    judge_target_reply: str | None = None,
) -> bool:
    """Return whether a success remains the action's usable role result.

    Selector and Perturbator schemas contain the closed transport shape, while
    their reducer-owned parsers also enforce request-specific structure such as
    exact slot sets, review IDs, and strategy-mode null fields.  A persisted
    schema-repair intent, or an exhausted-route transition to the frozen
    fallback, proves that the immediately preceding success failed that
    contextual parse at its historical prefix.  Do not later resurrect the
    superseded bytes merely because they satisfy the broader base schema.

    Purpose-specific projections still parse every base-structured success and
    reject more than one semantic result, so an unnecessary repair cannot turn
    two valid results into one.
    """

    for index, current in enumerate(action.attempts[:-1]):
        if current.physical_attempt_id != attempt.physical_attempt_id:
            continue
        following = action.attempts[index + 1]
        # Attempt ordering and the exact parent chain are authenticated by
        # ``_reduce_attempts`` before this projection is used.
        if following.intended["route_kind"] in {"schema_repair", "fallback"}:
            return False
        break
    return _success_passes_base_structure(
        action,
        attempt,
        response_schemas,
        judge_target_reply,
    )


def _recovery_after(
    action: LogicalActionView,
    attempt: AttemptView,
    response_schemas: Mapping[str, Mapping[str, Any]],
    strict_mode_before: str | None,
    *,
    target_identity_collision: bool = False,
    judge_target_reply: str | None = None,
) -> tuple[str, str | None]:
    """Return ``accept|fatal|exhaust|correct|open`` and correction kind."""

    terminal = attempt.terminal_event
    if terminal is None:
        return "open", None
    if target_identity_collision:
        if terminal["type"] != "model_call_succeeded":
            raise JournalCorrupt(
                "target-identity collision marks a non-successful role attempt"
            )
        return "exhaust", None
    if terminal["type"] == "model_call_outcome_unknown":
        return "correct", "exact_retry"
    if terminal["type"] == "model_call_failed":
        error = terminal["payload"]["error_kind"]
        if error in {"authentication", "invalid_request"}:
            return "fatal", None
        if error == "unsupported_response_format":
            if action.slot["role"] == Role.TARGET.value:
                return "fatal", None
            if (
                attempt.intended["response_mode"] == "json_schema"
                and strict_mode_before is None
            ):
                return "correct", "format_downgrade"
            return "exhaust", None
        if error in {
            "connection",
            "timeout",
            "rate_limit",
            "server",
            "malformed_response",
        }:
            return "correct", "exact_retry"
        return "exhaust", None
    if structured_output_exhausts_route(action.slot["role"], terminal["payload"]):
        return "exhaust", None
    if _success_is_structurally_usable(
        action,
        attempt,
        response_schemas,
        judge_target_reply,
    ):
        return "accept", None
    return (
        "correct",
        "exact_retry"
        if action.slot["role"] == Role.TARGET.value
        else "schema_repair",
    )


def structured_output_exhausts_route(role: str, payload: Mapping[str, Any]) -> bool:
    """An incomplete structured completion cannot be repaired at the same cap.

    Target sampling retains its predeclared exact retry. Structured callers
    advance to their declared fallback/replacement instead of resending an
    empty or truncated JSON repair to the same exhausted generation profile.
    """
    return role != Role.TARGET.value and payload.get("finish_status") != FinishStatus.COMPLETE.value


def _validate_response_mode_transitions(
    events: Sequence[Mapping[str, Any]],
    attempts: Mapping[str, AttemptView],
    response_schemas: Mapping[str, Mapping[str, Any]],
    prefix_collision_attempt_ids: frozenset[str],
) -> dict[str, str | None]:
    """Replay the one durable structured-response mode per exact route scope."""

    modes: dict[tuple[str, str], str] = {}
    before: dict[str, str | None] = {}
    for event in events:
        event_type = event["type"]
        if event_type == "model_call_intended":
            payload = event["payload"]
            schema_digest = judge_schema_family_digest(response_schemas[payload["response_schema_digest"]]) if payload["role"] == Role.JUDGE.value else payload["response_schema_digest"]
            if schema_digest is None:
                continue
            scope = (payload["role_route_key"], schema_digest)
            current = modes.get(scope)
            before[payload["physical_attempt_id"]] = current
            expected_mode = "json_object" if current == "json_object" else "json_schema"
            if payload["response_mode"] != expected_mode:
                raise JournalCorrupt(
                    "structured response mode differs from its replayed route scope"
                )
            continue
        if event_type not in TERMINAL_CALL_TYPES:
            continue
        attempt = attempts.get(event["payload"]["physical_attempt_id"])
        if attempt is None:
            continue
        if attempt.physical_attempt_id in prefix_collision_attempt_ids:
            # A colliding success proves neither strict-schema support nor a
            # usable JSON-object result for this role route.
            continue
        intended = attempt.intended
        schema_digest = judge_schema_family_digest(response_schemas[intended["response_schema_digest"]]) if intended["role"] == Role.JUDGE.value else intended["response_schema_digest"]
        if schema_digest is None:
            continue
        scope = (intended["role_route_key"], schema_digest)
        current = modes.get(scope)
        if intended["response_mode"] == "json_object":
            modes[scope] = "json_object"
        elif event_type == "model_call_succeeded" and current is None:
            modes[scope] = "json_schema"
        elif (
            event_type == "model_call_failed"
            and event["payload"]["error_kind"] == "unsupported_response_format"
            and current is None
        ):
            modes[scope] = "json_object"
    return before


def validate_recovery_transitions(
    events: Sequence[Mapping[str, Any]],
    logical_actions: Mapping[str, LogicalActionView],
    action_unit_kinds: Mapping[str, str],
    response_schemas: Mapping[str, Mapping[str, Any]],
    judge_target_replies: Mapping[str, str] | None = None,
) -> None:
    """Purely replay and validate the single Section 15.3 route policy."""

    attempts = {
        attempt.physical_attempt_id: attempt
        for action in logical_actions.values()
        for attempt in action.attempts
    }
    fatal_terminal_sequences: set[int] = set()
    for attempt in attempts.values():
        terminal = attempt.terminal_event
        if terminal is None or terminal["type"] != "model_call_failed":
            continue
        error_kind = terminal["payload"]["error_kind"]
        if error_kind in {"authentication", "invalid_request"} or (
            attempt.intended["role"] == Role.TARGET.value
            and error_kind == "unsupported_response_format"
        ):
            fatal_terminal_sequences.add(terminal["sequence"])
    for sequence in fatal_terminal_sequences:
        later = next(
            (event for event in events if event["sequence"] > sequence), None
        )
        if later is not None and later["type"] != "run_terminated":
            raise JournalCorrupt(
                "scientific execution continued after a fatal provider condition"
            )
    prefix_collision_attempt_ids = _prefix_target_collision_attempt_ids(
        logical_actions
    )
    strict_modes = _validate_response_mode_transitions(
        events, attempts, response_schemas, prefix_collision_attempt_ids
    )
    for action in logical_actions.values():
        if not action.attempts:
            continue
        routes = action.slot["routes"]
        route_indexes = {
            route["route_key"]: index for index, route in enumerate(routes)
        }
        if len(route_indexes) != len(routes):
            raise JournalCorrupt("logical action has duplicate route bindings")
        route_counts = {route["route_key"]: 0 for route in routes}
        first_messages = action.attempts[0].intended["messages"]

        previous: AttemptView | None = None
        previous_route_index = 0
        for index, attempt in enumerate(action.attempts):
            intended = attempt.intended
            route_key = intended["role_route_key"]
            if route_key not in route_indexes:
                raise JournalCorrupt("attempt names no route in its logical action")
            route_index = route_indexes[route_key]
            route = routes[route_index]
            route_counts[route_key] += 1
            if route_counts[route_key] > route["maximum_attempts"]:
                raise JournalCorrupt("attempts exceed their per-route cap")

            if index == 0:
                if route_index != 0:
                    raise JournalCorrupt("logical action did not start on its first route")
                if route["route_label"] == "fallback":
                    raise JournalCorrupt("logical action started on its fallback route")
                initial_kind = (
                    route["route_label"]
                    if route["route_label"] in {"replacement", "fallback"}
                    else "primary"
                )
                if intended["route_kind"] != initial_kind:
                    raise JournalCorrupt("logical action has an invalid initial route kind")
                previous = attempt
                continue

            assert previous is not None
            if previous.terminal_event is None:
                raise JournalCorrupt("corrective intent predates its parent terminal fact")
            previous_collision = (
                previous.physical_attempt_id in prefix_collision_attempt_ids
            )
            # ``logical_actions`` may carry the final replay projection, which
            # can retroactively identify a pre-pin structured-role receipt as a
            # collision.  Recovery validation is historical: materialize the
            # flag exactly as it was knowable at this receipt's own prefix.
            recovery_attempt = replace(
                previous,
                target_identity_collision=previous_collision,
            )
            disposition, corrective_kind = _recovery_after(
                action,
                recovery_attempt,
                response_schemas,
                strict_modes.get(previous.physical_attempt_id),
                target_identity_collision=previous_collision,
                judge_target_reply=(judge_target_replies or {}).get(
                    action.action_id
                ),
            )

            if route_index == previous_route_index:
                if disposition != "correct" or intended["route_kind"] != corrective_kind:
                    raise JournalCorrupt(
                        "corrective attempt has no legal prior terminal trigger"
                    )
            else:
                if route_index != previous_route_index + 1:
                    raise JournalCorrupt("logical action skipped or revisited a route")
                if action.slot["role"] not in {
                    Role.SELECTOR.value,
                    Role.PERTURBATOR.value,
                }:
                    raise JournalCorrupt("role used a forbidden alternate route")
                if route["route_label"] != "fallback" or intended["route_kind"] != "fallback":
                    raise JournalCorrupt("alternate route is not the frozen fallback")
                previous_route = routes[previous_route_index]
                previous_spent = route_counts[previous_route["route_key"]]
                if disposition in {"accept", "fatal", "open"} or (
                    disposition == "correct"
                    and previous_spent < previous_route["maximum_attempts"]
                ):
                    raise JournalCorrupt("fallback started before primary route exhaustion")
                if intended["messages"] != first_messages:
                    raise JournalCorrupt("fallback did not restore the original semantic request")

            if intended["route_kind"] == "exact_retry":
                if intended["semantic_request_hash"] != previous.intended["semantic_request_hash"]:
                    raise JournalCorrupt("exact retry changed the semantic request")
            elif intended["route_kind"] == "format_downgrade":
                if (
                    previous.intended["response_mode"] != "json_schema"
                    or intended["response_mode"] != "json_object"
                    or intended["messages"] != previous.intended["messages"]
                ):
                    raise JournalCorrupt("format downgrade changed more than response mode")
            elif intended["route_kind"] == "schema_repair":
                if intended["messages"] == previous.intended["messages"]:
                    raise JournalCorrupt("schema repair omitted its structural-error context")

            previous = attempt
            previous_route_index = route_index


def _target_success_receipts(
    logical_actions: Mapping[str, LogicalActionView],
) -> tuple[AttemptView, ...]:
    receipts = [
        attempt
        for action in logical_actions.values()
        if action.slot.get("role") == "target"
        for attempt in action.successful_attempts
        if attempt.terminal_event is not None
    ]
    receipts.sort(
        key=lambda attempt: (
            attempt.terminal_event["sequence"],  # type: ignore[index]
            attempt.physical_attempt_id,
        )
    )
    return tuple(receipts)


def _target_identity_projection(
    receipts: Sequence[AttemptView],
) -> tuple[str | None, tuple[str, ...]]:
    pinned: str | None = None
    mismatches: list[str] = []
    for attempt in receipts:
        terminal = attempt.terminal_event
        assert terminal is not None
        effective = terminal["payload"]["effective_model_id"]
        if effective is None:
            continue
        if pinned is None:
            pinned = effective
        elif effective != pinned:
            mismatches.append(attempt.physical_attempt_id)
    return pinned, tuple(mismatches)


_TARGET_COLLISION_ROLES = (Role.SELECTOR.value, Role.PERTURBATOR.value)


def _role_identity_projection(
    logical_actions: Mapping[str, LogicalActionView],
    pinned_target_identity: str | None,
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    """Project structured-role identity evidence against one target pin.

    Collision is deliberately based on every authoritative role success, not
    only schema-valid output.  A collision exhausts that route before role
    parsing, while a null identity remains usable but audibly unverifiable.
    """

    ordered: list[tuple[int, str, AttemptView]] = []
    for action in logical_actions.values():
        role = action.slot.get("role")
        if role not in _TARGET_COLLISION_ROLES:
            continue
        for attempt in action.successful_attempts:
            terminal = attempt.terminal_event
            if terminal is not None:
                ordered.append((terminal["sequence"], str(role), attempt))
    collisions = {role: [] for role in _TARGET_COLLISION_ROLES}
    unverifiable = {role: [] for role in _TARGET_COLLISION_ROLES}
    for _, role, attempt in sorted(
        ordered, key=lambda item: (item[0], item[2].physical_attempt_id)
    ):
        terminal = attempt.terminal_event
        assert terminal is not None
        effective = terminal["payload"]["effective_model_id"]
        if effective is None:
            unverifiable[role].append(attempt.physical_attempt_id)
        elif (
            pinned_target_identity is not None
            and effective == pinned_target_identity
        ):
            collisions[role].append(attempt.physical_attempt_id)
    return (
        {role: tuple(collisions[role]) for role in _TARGET_COLLISION_ROLES},
        {role: tuple(unverifiable[role]) for role in _TARGET_COLLISION_ROLES},
    )


def _apply_role_identity_projection(
    logical_actions: Mapping[str, LogicalActionView],
    pinned_target_identity: str | None,
) -> tuple[
    dict[str, LogicalActionView],
    dict[str, tuple[str, ...]],
    dict[str, tuple[str, ...]],
]:
    collisions, unverifiable = _role_identity_projection(
        logical_actions, pinned_target_identity
    )
    collision_ids = frozenset(
        attempt_id
        for role_ids in collisions.values()
        for attempt_id in role_ids
    )
    projected = {
        action_id: LogicalActionView(
            action.action_id,
            action.unit_id,
            action.slot,
            tuple(
                replace(
                    attempt,
                    target_identity_collision=(
                        attempt.physical_attempt_id in collision_ids
                    ),
                )
                for attempt in action.attempts
            ),
        )
        for action_id, action in logical_actions.items()
    }
    return projected, collisions, unverifiable


def _prefix_target_collision_attempt_ids(
    logical_actions: Mapping[str, LogicalActionView],
) -> frozenset[str]:
    """Return collisions knowable at each receipt's own journal prefix.

    Final reporting can retroactively disqualify earlier structured-role evidence
    after the first target identity becomes known.  Historical request,
    recovery, and work-unit-boundary validation must instead judge decisions
    using only the pin available when that terminal receipt was durable.
    """

    receipts: list[tuple[int, str, AttemptView]] = []
    for action in logical_actions.values():
        role = action.slot.get("role")
        if role not in {Role.TARGET.value, *_TARGET_COLLISION_ROLES}:
            continue
        for attempt in action.successful_attempts:
            terminal = attempt.terminal_event
            if terminal is not None:
                receipts.append((terminal["sequence"], str(role), attempt))
    pinned: str | None = None
    collisions: set[str] = set()
    for _, role, attempt in sorted(
        receipts, key=lambda item: (item[0], item[2].physical_attempt_id)
    ):
        terminal = attempt.terminal_event
        assert terminal is not None
        # Public replay always reaches this helper after event-schema
        # validation; the optional lookup also keeps the pure transition
        # validator usable with its intentionally minimal unit fixtures.
        effective = terminal["payload"].get("effective_model_id")
        if effective is None:
            continue
        if role == Role.TARGET.value:
            if pinned is None:
                pinned = effective
        elif pinned is not None and effective == pinned:
            collisions.add(attempt.physical_attempt_id)
    return frozenset(collisions)


def _target_results(
    logical_actions: Mapping[str, LogicalActionView],
    excluded_attempt_ids: frozenset[str] = frozenset(),
) -> dict[str, TargetResultView]:
    results: dict[str, TargetResultView] = {}
    for action_id, action in logical_actions.items():
        if action.slot.get("role") != "target":
            continue
        # Empty, truncated, filtered, or otherwise incomplete authoritative
        # completions are retained in the journal but are not observations.
        success = next(
            (
                attempt
                for attempt in reversed(action.successful_attempts)
                if attempt.terminal_event is not None
                and attempt.physical_attempt_id not in excluded_attempt_ids
                and attempt.terminal_event["payload"]["finish_status"] == "complete"
                and bool(attempt.terminal_event["payload"]["text"])
            ),
            None,
        )
        if success is None or success.terminal_event is None:
            continue
        terminal = success.terminal_event
        payload = terminal["payload"]
        lineage = success.intended["lineage"]
        results[action_id] = TargetResultView(
            logical_action_id=action_id,
            candidate_id=lineage["candidate_id"],
            execution_id=lineage["execution_id"],
            conversation_root_id=lineage["conversation_root_id"],
            parent_node_id=lineage["parent_node_id"],
            text=payload["text"],
            finish_status=payload["finish_status"],
            effective_model_id=payload["effective_model_id"],
            physical_attempt_id=success.physical_attempt_id,
            terminal_event_id=terminal["event_id"],
            terminal_sequence=terminal["sequence"],
        )
    return results


def _validate_durability_order(
    logical_actions: Mapping[str, LogicalActionView],
    work_units: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> None:
    panels: dict[str, Mapping[str, Any]] = {}
    for _, unit in work_units:
        for panel in unit["panel_plans"]:
            panels[panel["panel_id"]] = panel
    for action in logical_actions.values():
        if action.slot.get("role") != "judge" or not action.attempts:
            continue
        intent_event = action.attempts[0].intent_event
        panel_id = intent_event["payload"]["lineage"]["panel_id"]
        if panel_id is None:
            continue  # no panel-owned assessment can be projected
        if panel_id not in panels:
            raise JournalCorrupt("runtime judge action names an unknown panel")
        target_action_id = panels[panel_id]["target_action_id"]
        target = logical_actions.get(target_action_id)
        if target is None or target.successful_attempt is None or target.successful_attempt.terminal_event is None:
            raise JournalCorrupt("judge intent exists before a durable target success")
        if target.successful_attempt.terminal_event["sequence"] >= intent_event["sequence"]:
            raise JournalCorrupt("judge intent predates durable target response")


_NO_SEMANTIC_RESULT = object()


def _action_at_sequence(
    action: LogicalActionView, sequence: int
) -> LogicalActionView:
    """Return the action facts that were durable at one journal boundary."""

    attempts: list[AttemptView] = []
    for attempt in action.attempts:
        if attempt.intent_event["sequence"] > sequence:
            continue
        dispatch = attempt.dispatch_event
        terminal = attempt.terminal_event
        attempts.append(
            AttemptView(
                attempt.physical_attempt_id,
                attempt.intent_event,
                dispatch
                if dispatch is not None and dispatch["sequence"] <= sequence
                else None,
                terminal
                if terminal is not None and terminal["sequence"] <= sequence
                else None,
            )
        )
    return LogicalActionView(
        action.action_id,
        action.unit_id,
        action.slot,
        tuple(attempts),
    )


def _completion_snapshot(
    work_units: Sequence[WorkUnitView],
    logical_actions: Mapping[str, LogicalActionView],
    sequence: int,
) -> tuple[
    Mapping[str, WorkUnitView],
    Mapping[str, LogicalActionView],
    Mapping[str, TargetResultView],
]:
    """Build a read-only prefix view for historical boundary validation."""

    actions = {
        action_id: _action_at_sequence(action, sequence)
        for action_id, action in logical_actions.items()
        if action.slot.get("ordinal") is not None
    }
    receipts = _target_success_receipts(actions)
    pinned, mismatches = _target_identity_projection(receipts)
    actions, _, _ = _apply_role_identity_projection(actions, pinned)
    units = {
        unit.unit_id: WorkUnitView(
            unit.start_event,
            unit.unit,
            tuple(actions[action.action_id] for action in unit.actions),
        )
        for unit in work_units
        if unit.start_event["sequence"] <= sequence
    }
    targets = _target_results(actions, frozenset(mismatches))
    return units, actions, targets


def _completion_semantic_value(
    unit: WorkUnitView,
    action: LogicalActionView,
    actions: Mapping[str, LogicalActionView],
    target_results: Mapping[str, TargetResultView],
    response_schemas: Mapping[str, Mapping[str, Any]],
    cache: dict[str, Any],
) -> Any:
    """Project only durable, strictly validated output needed by slot guards."""

    if action.action_id in cache:
        return cache[action.action_id]
    value: Any = _NO_SEMANTIC_RESULT
    if action.slot["role"] == Role.TARGET.value:
        value = target_results.get(action.action_id, _NO_SEMANTIC_RESULT)
    else:
        for attempt in action.attempts:
            if not _success_is_structurally_usable(
                action, attempt, response_schemas
            ):
                continue
            assert attempt.terminal_event is not None
            parsed = strict_json_loads(attempt.terminal_event["payload"]["text"])
            if action.slot["role"] == Role.JUDGE.value:
                panel = next(
                    (
                        item
                        for item in unit.unit["panel_plans"]
                        if any(
                            member["judge_action_id"] == action.action_id
                            for member in item["normal_members"]
                        )
                        or (item["replacement"] is not None and item["replacement"]["judge_action_id"]
                        == action.action_id)
                    ),
                    None,
                )
                if panel is None:
                    continue
                target = target_results.get(panel["target_action_id"])
                if target is None:
                    continue
                judged = validate_judge_output(
                    attempt.terminal_event["payload"]["text"], target.text,
                    schema=response_schemas[attempt.intended["response_schema_digest"]],
                )
                if judged.output is None:
                    continue
                value = judged.output
            else:
                try:
                    if (
                        action.slot["purpose"]
                        == ActionPurpose.PERTURBATOR_PROPOSAL.value
                    ):
                        value = parse_proposal_batch(
                            parsed, _proposal_slots_for_action(unit, action)
                        )
                    elif action.slot["purpose"] == ActionPurpose.PROBE_REVIEW.value:
                        expectations = _completion_review_expectations(
                            unit,
                            action,
                            actions,
                            lambda action_id: _completion_semantic_value(
                                unit,
                                actions[action_id],
                                actions,
                                target_results,
                                response_schemas,
                                cache,
                            ),
                        )
                        value = parse_probe_review_statuses(
                            parsed,
                            expectations,
                            mode=_probe_review_mode(unit, action),
                        )
                    else:
                        value = parsed
                except DomainError:
                    continue
            break
    cache[action.action_id] = value
    return value


def _completion_action_closed(
    unit: WorkUnitView,
    action: LogicalActionView,
    actions: Mapping[str, LogicalActionView],
    target_results: Mapping[str, TargetResultView],
    response_schemas: Mapping[str, Mapping[str, Any]],
    semantic_cache: dict[str, Any],
) -> bool:
    semantic = _completion_semantic_value(
        unit,
        action,
        actions,
        target_results,
        response_schemas,
        semantic_cache,
    )
    if semantic is not _NO_SEMANTIC_RESULT:
        return True
    if not action.attempts:
        return False
    latest = action.attempts[-1]
    if latest.terminal_event is None:
        return False

    routes = action.slot["routes"]
    route_index = next(
        index
        for index, route in enumerate(routes)
        if route["route_key"] == latest.intended["role_route_key"]
    )
    route = routes[route_index]
    attempts_on_route = sum(
        attempt.intended["role_route_key"] == route["route_key"]
        for attempt in action.attempts
    )
    terminal = latest.terminal_event

    if latest.target_identity_collision:
        # A Selector/Perturbator receipt from the pinned target identity is
        # retained as evidence, but it exhausts this route immediately.  The
        # only remaining recovery is the already-frozen fallback route.
        return route_index + 1 >= len(routes)

    retry_same_route = False
    if terminal["type"] == "model_call_succeeded":
        retry_same_route = not structured_output_exhausts_route(action.slot["role"], terminal["payload"])
    elif terminal["type"] == "model_call_outcome_unknown":
        retry_same_route = True
    elif terminal["type"] == "model_call_failed":
        error = terminal["payload"]["error_kind"]
        retry_same_route = error in {
            "connection",
            "timeout",
            "rate_limit",
            "server",
            "malformed_response",
        } or (
            error == "unsupported_response_format"
            and action.slot["role"] != Role.TARGET.value
            and latest.intended["response_mode"] == "json_schema"
        )
    if retry_same_route and attempts_on_route < route["maximum_attempts"]:
        return False
    return route_index + 1 >= len(routes)


def _review_status(value: Any, review_item_id: str) -> str | None:
    """Read one durable status from the alias-materialized guard projection."""

    if not isinstance(value, Mapping):
        return None
    status = value.get(review_item_id)
    return status if isinstance(status, str) else None


def _portfolio_selected_review_status(
    round_view: PortfolioRoundView,
    review_item_id: str,
) -> str | None:
    """Return the canonical admission decision selected for one child."""

    matches = [
        child.selected.review.status
        for child in round_view.children
        if child.selected is not None
        and child.selected.review.review_item_id == review_item_id
    ]
    if len(matches) > 1:
        raise JournalCorrupt("portfolio review duplicates one realization")
    return matches[0] if matches else None


def _portfolio_stateful_dispatch_allows(
    unit: WorkUnitView,
    action: LogicalActionView,
    actions: Mapping[str, LogicalActionView],
    round_view: PortfolioRoundView,
) -> bool:
    """Apply ownership to already materialized portfolio realizations."""

    guard = action.slot["guard"]
    current_matches = [
        child
        for child in round_view.children
        if child.target_action_id == action.action_id
        and child.proposal_slot.scientific_key
        == guard["proposal_slot_scientific_key"]
    ]
    if len(current_matches) != 1:
        raise JournalCorrupt("portfolio target has no canonical child projection")
    current = current_matches[0]
    if (
        current.selected is None
        or current.selected.review.review_item_id != guard["review_item_id"]
        or current.selected.review.status != "aligned"
        or current.selected.review.reviewed_descriptors is None
        or current.selected.proposal.user_turn is None
    ):
        return False
    prompt = normalize_line_endings(current.selected.proposal.user_turn)
    descriptors = descriptors_dict(
        current.selected.review.reviewed_descriptors
    )
    same_prompt: list[tuple[str, Mapping[str, Any]]] = []
    for child in round_view.children:
        if (
            child.selected is None
            or child.selected.review.status != "aligned"
            or child.selected.review.reviewed_descriptors is None
            or child.selected.proposal.user_turn is None
        ):
            continue
        if normalize_line_endings(child.selected.proposal.user_turn) == prompt:
            same_prompt.append(
                (
                    child.proposal_slot.scientific_key,
                    descriptors_dict(
                        child.selected.review.reviewed_descriptors
                    ),
                )
            )
    if not same_prompt:
        return False
    if len({digest_json(item[1]) for item in same_prompt}) != 1:
        return False
    if guard["proposal_slot_scientific_key"] != min(
        item[0] for item in same_prompt
    ):
        return False
    if digest_json(descriptors) != digest_json(same_prompt[0][1]):
        return False

    parent_node_id = unit.unit["design"]["generation_parent_node_id"]
    for prior in actions.values():
        if prior.action_id == action.action_id or not prior.attempts:
            continue
        lineage = prior.attempts[0].intended.get("lineage")
        messages = prior.attempts[0].intended.get("messages")
        if (
            not isinstance(lineage, Mapping)
            or lineage.get("parent_node_id") != parent_node_id
            or not isinstance(messages, list)
            or not messages
        ):
            continue
        final = messages[-1]
        if (
            isinstance(final, Mapping)
            and final.get("role") == "user"
            and isinstance(final.get("content"), str)
            and normalize_line_endings(final["content"]) == prompt
        ):
            return False
    return True


def _ordinary_map_probe_is_unique(
    unit: WorkUnitView,
    action: LogicalActionView,
    actions: Mapping[str, LogicalActionView],
    target_results: Mapping[str, TargetResultView],
    semantic_value: Any,
) -> bool:
    """Apply global standalone identity and same-batch slot ownership."""

    if unit.unit["kind"] != "map_exploration":
        return True
    proposal_action = next(
        (
            item
            for item in unit.actions
            if item.slot["purpose"] == ActionPurpose.PERTURBATOR_PROPOSAL.value
        ),
        None,
    )
    if proposal_action is None:
        raise JournalCorrupt("ordinary MAP unit lacks its proposal action")
    proposal_value = semantic_value(proposal_action.action_id)
    if not isinstance(proposal_value, ProposalBatch):
        return False
    slots = _proposal_slots_for_unit(unit)
    slot_by_key = {item.scientific_key: item for item in slots}
    proposal_by_slot = {item.slot_id: item for item in proposal_value.items}
    current_key = action.slot["guard"]["proposal_slot_scientific_key"]
    if current_key not in slot_by_key:
        raise JournalCorrupt("ordinary MAP target guard names no proposal slot")
    review_action = next(
        (
            item
            for item in unit.actions
            if item.slot["purpose"] == ActionPurpose.PROBE_REVIEW.value
        ),
        None,
    )
    review_value = (
        None if review_action is None else semantic_value(review_action.action_id)
    )
    if not isinstance(review_value, Mapping):
        return False
    proposed: list[tuple[str, str]] = []
    for slot in slots:
        proposal = proposal_by_slot.get(slot.slot_id)
        if proposal is None:
            raise JournalCorrupt("proposal batch does not key every frozen slot once")
        review_id = make_review_item_id(
            unit.unit["unit_scientific_key"], "probe", slot.scientific_key
        )
        if (
            _review_status(review_value, review_id) == "valid"
            and proposal.proposed
            and proposal.user_turn is not None
        ):
            proposed.append(
                (
                    slot.scientific_key,
                    normalize_line_endings(proposal.user_turn),
                )
            )
    current = next(
        (item for item in proposed if item[0] == current_key), None
    )
    if current is None:
        return False
    _, prompt = current
    owner = min(key for key, other_prompt in proposed if other_prompt == prompt)
    if owner != current_key:
        return False

    del target_results
    for prior in actions.values():
        if (
            prior.slot["role"] != Role.TARGET.value
            or prior.unit_id == unit.unit_id
            or not prior.attempts
            or prior.attempts[0].intended["lineage"]["parent_node_id"] is not None
        ):
            continue
        user_turns = [
            item["content"]
            for item in prior.attempts[0].intended["messages"]
            if item["role"] == "user"
        ]
        if user_turns and normalize_line_endings(user_turns[-1]) == prompt:
            return False
    return True


def _completion_panel_is_resolved(
    unit: WorkUnitView,
    target_action_id: str,
    actions: Mapping[str, LogicalActionView],
    target_results: Mapping[str, TargetResultView],
    response_schemas: Mapping[str, Mapping[str, Any]],
    semantic_cache: dict[str, Any],
) -> bool:
    """Mirror the composed panel's exact terminal/resolution predicate."""

    panel = next(
        (
            item
            for item in unit.unit["panel_plans"]
            if item["target_action_id"] == target_action_id
        ),
        None,
    )
    target = target_results.get(target_action_id)
    if panel is None or target is None:
        return False

    def semantic(action_id: str) -> Any:
        return _completion_semantic_value(
            unit,
            actions[action_id],
            actions,
            target_results,
            response_schemas,
            semantic_cache,
        )

    def member(action_id: str, slot: str) -> MemberAssessment | None:
        parsed = semantic(action_id)
        if not isinstance(parsed, ParsedJudgeOutput):
            return None
        action = actions[action_id]
        source = next(
            (
                attempt
                for attempt in action.attempts
                if attempt.succeeded
                and attempt.terminal_event is not None
                and (
                    checked := validate_judge_output(
                        attempt.terminal_event["payload"]["text"], target.text,
                        schema=response_schemas[attempt.intended["response_schema_digest"]],
                    )
                ).output
                == parsed
            ),
            None,
        )
        if source is None or source.terminal_event is None:
            return None
        return member_assessment_from_output(
            parsed,
            assessment_id=entity_id("ma", [source.physical_attempt_id]),
            observation_id=entity_id("to", [target_action_id]),
            logical_member_slot=slot,
            effective_model_id=source.terminal_event["payload"]["effective_model_id"],
            role_route_key=source.intended["role_route_key"],
            source_attempt_id=source.physical_attempt_id,
        )

    primary_plan = panel["normal_members"][0]
    primary_action_id = primary_plan["judge_action_id"]
    primary = member(primary_action_id, str(primary_plan["logical_member_slot"]))
    triggered = bool(
        panel["trigger_policy"] == "always_full"
        or (primary is not None and should_schedule_secondary_panel(primary))
        or (
            primary is None
            and _action_effectively_closed(
                unit,
                actions[primary_action_id],
                actions,
                target_results,
                response_schemas,
                semantic_cache,
            )
        )
    )
    if not triggered:
        return primary is not None

    normal_actions = tuple(
        actions[item["judge_action_id"]] for item in panel["normal_members"]
    )
    replacement_action = actions[panel["replacement"]["judge_action_id"]] if panel["replacement"] is not None else None
    if not all(
        _action_effectively_closed(
            unit,
            action,
            actions,
            target_results,
            response_schemas,
            semantic_cache,
        )
        for action in (*normal_actions, *([replacement_action] if replacement_action is not None else []))
    ):
        return False

    members = [
        value
        for item in panel["normal_members"]
        if (
            value := member(
                item["judge_action_id"], str(item["logical_member_slot"])
            )
        )
        is not None
    ]
    if replacement_action is not None and replacement_action.attempts:
        replacement_slots = {
            attempt.intended["lineage"].get("logical_member_slot")
            for attempt in replacement_action.attempts
        }
        if len(replacement_slots) == 1:
            replacement_slot = next(iter(replacement_slots))
            if isinstance(replacement_slot, str):
                replacement = member(
                    replacement_action.action_id, replacement_slot
                )
                if replacement is not None:
                    members = [
                        item
                        for item in members
                        if item.logical_member_slot
                        != replacement.logical_member_slot
                    ]
                    members.append(replacement)
    details = aggregate_panel_details(
        panel["panel_id"],
        entity_id("to", [target_action_id]),
        members,
        target_effective_model_id=target.effective_model_id,
    )
    return details.assessment.availability is not PanelAvailability.NO_QUORUM


def _completion_guard_allows(
    unit: WorkUnitView,
    action: LogicalActionView,
    actions: Mapping[str, LogicalActionView],
    target_results: Mapping[str, TargetResultView],
    response_schemas: Mapping[str, Mapping[str, Any]],
    semantic_cache: dict[str, Any],
    portfolio_rounds: Mapping[str, PortfolioRoundView] | None = None,
) -> bool:
    def semantic(action_id: str) -> Any:
        return _completion_semantic_value(
            unit,
            actions[action_id],
            actions,
            target_results,
            response_schemas,
            semantic_cache,
        )

    def closed(action_id: str) -> bool:
        return _completion_action_closed(
            unit,
            actions[action_id],
            actions,
            target_results,
            response_schemas,
            semantic_cache,
        )

    guard = action.slot["guard"]
    kind = guard["kind"]
    if kind == "always":
        return True
    if kind == "if_action_usable":
        return semantic(guard["action_id"]) is not _NO_SEMANTIC_RESULT
    if kind == "if_panel_resolved":
        panel = next(p for p in unit.unit["panel_plans"] if p["panel_id"] == guard["panel_id"])
        return _completion_panel_is_resolved(
            unit, panel["target_action_id"], actions, target_results,
            response_schemas, semantic_cache,
        )
    if kind in {"if_panel_triggered", "if_panel_has_unresolved_normal_member"}:
        panel = next(
            (
                item
                for item in unit.unit["panel_plans"]
                if item["panel_id"] == guard["panel_id"]
            ),
            None,
        )
        if panel is None:
            raise JournalCorrupt("action guard names an unknown panel")
        if panel["target_action_id"] not in target_results:
            return False
        primary_id = panel["normal_members"][0]["judge_action_id"]
        primary = semantic(primary_id)
        panel_required = panel["trigger_policy"] == "always_full" or (
            isinstance(primary, ParsedJudgeOutput) and should_schedule_secondary_panel(primary)
        ) or (
            primary is _NO_SEMANTIC_RESULT and closed(primary_id)
        )
        if kind == "if_panel_triggered":
            return panel_required
        if not panel_required:
            return False
        return any(
            semantic(member["judge_action_id"]) is _NO_SEMANTIC_RESULT
            and closed(member["judge_action_id"])
            for member in panel["normal_members"]
        )
    if kind in {"if_any_proposed_item", "if_any_corrective_proposal"}:
        for action_id in guard["proposal_action_ids"]:
            value = semantic(action_id)
            if isinstance(value, ProposalBatch) and value.proposed_items:
                return True
        return False
    if kind == "if_strategy_slot_bound":
        selector = next(
            (
                item
                for item in unit.actions
                if item.slot["purpose"]
                == ActionPurpose.SELECTOR_PORTFOLIO.value
            ),
            None,
        )
        if selector is None:
            return False
        portfolio = semantic(selector.action_id)
        if not isinstance(portfolio, Mapping) or not isinstance(
            portfolio.get("items"), list
        ):
            return False
        strategy_slot = next(
            (
                item
                for item in unit.unit["design"].get("strategy_slots", [])
                if item["scientific_key"]
                == guard["strategy_slot_scientific_key"]
            ),
            None,
        )
        if strategy_slot is None:
            raise JournalCorrupt("action guard names an unknown strategy slot")
        return strategy_slot["ordinal"] < len(portfolio["items"])
    if kind == "if_dispatchable_probe":
        qualification_arm: str | None = None
        if unit.unit["kind"] in {"strategy_development", "strategy_holdout"}:
            matching_plans = [
                item
                for item in unit.unit["design"].get("root_plans", [])
                if action.action_id in item.get("ordered_action_ids", [])
            ]
            if len(matching_plans) != 1:
                raise JournalCorrupt(
                    "qualification target does not belong to one root plan"
                )
            qualification_arm = str(matching_plans[0]["arm"])
        if unit.unit["kind"] == "portfolio_round":
            round_view = (
                None
                if portfolio_rounds is None
                else portfolio_rounds.get(unit.unit_id)
            )
            if round_view is not None:
                status = _portfolio_selected_review_status(
                    round_view, guard["review_item_id"]
                )
                return bool(
                    status in set(guard["allowed"])
                    and _portfolio_stateful_dispatch_allows(
                        unit, action, actions, round_view
                    )
                )
        for prior in reversed(unit.actions[: action.slot["ordinal"]]):
            if prior.slot["purpose"] != ActionPurpose.PROBE_REVIEW.value:
                continue
            status = _review_status(semantic(prior.action_id), guard["review_item_id"])
            if status is not None:
                allowed = set(guard["allowed"])
                if qualification_arm == "treatment" or (
                    unit.unit["kind"] == "verification"
                    and unit.unit["design"].get("subject_kind")
                    == "scenario_recipe"
                ):
                    # Section 12.16 deliberately tightens the generic
                    # strategy guard: valid-but-unaligned is scientific
                    # attrition, not a treatment dispatch.
                    allowed = {"aligned"}
                return status in allowed and _ordinary_map_probe_is_unique(
                    unit,
                    action,
                    actions,
                    target_results,
                    semantic,
                )
        return False
    if kind == "if_repairable_proposal_exists":
        value = semantic(guard["review_action_id"])
        if not isinstance(value, Mapping):
            return False
        permitted = set(guard["proposal_slot_scientific_keys"])
        statuses = set(guard["statuses"])
        permitted_ids = {
            make_review_item_id(
                unit.unit["unit_scientific_key"], "probe", proposal_key
            )
            for proposal_key in permitted
        }
        return any(
            _review_status(value, review_id) in statuses
            for review_id in permitted_ids
        )
    if kind == "if_any_aligned_response":
        if unit.unit["kind"] != "portfolio_round":
            if unit.unit["kind"] in {
                "strategy_development",
                "strategy_holdout",
            } or (
                unit.unit["kind"] == "verification"
                and unit.unit["design"].get("subject_kind")
                == "scenario_recipe"
            ):
                return any(
                    action_id in target_results
                    and _completion_panel_is_resolved(
                        unit,
                        action_id,
                        actions,
                        target_results,
                        response_schemas,
                        semantic_cache,
                    )
                    for action_id in guard["target_action_ids"]
                )
            return any(
                action_id in target_results
                for action_id in guard["target_action_ids"]
            )
        return any(
            action_id in target_results
            and (
                _portfolio_selected_review_status(
                    portfolio_rounds[unit.unit_id],
                    actions[action_id].slot["guard"]["review_item_id"],
                )
                if portfolio_rounds is not None
                and unit.unit_id in portfolio_rounds
                else _review_status(
                    semantic(
                        next(
                            item.action_id
                            for item in reversed(
                                unit.actions[: actions[action_id].slot["ordinal"]]
                            )
                            if item.slot["purpose"]
                            == ActionPurpose.PROBE_REVIEW.value
                        )
                    ),
                    actions[action_id].slot["guard"]["review_item_id"],
                )
            )
            == "aligned"
            and _completion_panel_is_resolved(
                unit,
                action_id,
                actions,
                target_results,
                response_schemas,
                semantic_cache,
            )
            for action_id in guard["target_action_ids"]
        )
    if kind == "if_template_applicable":
        value = semantic(guard["binding_action_id"])
        return isinstance(value, Mapping) and value.get("status") == "applicable"
    raise JournalCorrupt(f"unknown frozen action guard: {kind}")


def _completion_unit_is_complete(
    unit: WorkUnitView,
    actions: Mapping[str, LogicalActionView],
    target_results: Mapping[str, TargetResultView],
    response_schemas: Mapping[str, Mapping[str, Any]],
    portfolio_rounds: Mapping[str, PortfolioRoundView] | None = None,
) -> bool:
    """Replay engine closure over one finite frozen action graph."""

    semantic_cache: dict[str, Any] = {}
    for action in unit.actions:
        guard_allows = _completion_guard_allows(
            unit,
            action,
            actions,
            target_results,
            response_schemas,
            semantic_cache,
            portfolio_rounds,
        )
        if action.attempts and not guard_allows:
            raise JournalCorrupt("a model attempt exists under a false frozen guard")
        semantic = _completion_semantic_value(
            unit,
            action,
            actions,
            target_results,
            response_schemas,
            semantic_cache,
        )
        if semantic is not _NO_SEMANTIC_RESULT:
            continue
        dependencies_closed = all(
            _completion_action_closed(
                unit,
                actions[dependency],
                actions,
                target_results,
                response_schemas,
                semantic_cache,
            )
            or not _completion_guard_allows(
                unit,
                actions[dependency],
                actions,
                target_results,
                response_schemas,
                semantic_cache,
                portfolio_rounds,
            )
            for dependency in action.slot["dependency_action_ids"]
        )
        if not dependencies_closed:
            return False
        if not guard_allows:
            continue
        if not _completion_action_closed(
            unit,
            action,
            actions,
            target_results,
            response_schemas,
            semantic_cache,
        ):
            return False
    return True


def _validate_work_unit_boundaries(
    events: Sequence[Mapping[str, Any]],
    work_units: Sequence[WorkUnitView],
    logical_actions: Mapping[str, LogicalActionView],
    response_schemas: Mapping[str, Mapping[str, Any]],
    semantic_completion: Mapping[str, bool] | None = None,
    portfolio_rounds: Mapping[str, PortfolioRoundView] | None = None,
) -> tuple[str | None, tuple[str, ...]]:
    """Validate every new-unit and terminal completion claim at its own prefix."""

    active_unit_id: str | None = None
    completed: list[str] = []
    completion = semantic_completion or {}
    final_unit_fact_sequence = {
        unit.unit_id: max(
            (
                event["sequence"]
                for action in unit.actions
                for attempt in action.attempts
                for event in (
                    attempt.intent_event,
                    attempt.dispatch_event,
                    attempt.terminal_event,
                )
                if event is not None
            ),
            default=unit.start_event["sequence"],
        )
        for unit in work_units
    }

    def snapshot(sequence: int) -> tuple[
        Mapping[str, WorkUnitView],
        Mapping[str, LogicalActionView],
        Mapping[str, TargetResultView],
    ]:
        return _completion_snapshot(work_units, logical_actions, sequence)

    def unit_is_complete(
        unit: WorkUnitView,
        actions: Mapping[str, LogicalActionView],
        targets: Mapping[str, TargetResultView],
        sequence: int,
    ) -> bool:
        # Always replay the frozen graph so attempts under false guards remain
        # corruption.  Stateful portfolio parsers impose stricter semantics
        # than materialized JSON Schema alone, so their reducer-owned closure
        # result is authoritative for the final completion decision.
        graph_complete = _completion_unit_is_complete(
            unit,
            actions,
            targets,
            response_schemas,
            portfolio_rounds,
        )
        if unit.unit_id not in completion:
            return graph_complete
        # Stateful parsers derive ``completion`` from the full journal.  A
        # final true value may only override a historical graph snapshot once
        # every action fact used by that final projection was already durable.
        # Otherwise progress after run_resumed would retroactively complete the
        # unit at the preceding resumable terminal.
        return bool(
            completion[unit.unit_id]
            and sequence >= final_unit_fact_sequence[unit.unit_id]
        )

    for event in events:
        event_type = event["type"]
        if event_type == "work_unit_started":
            if active_unit_id is not None:
                units, actions, targets = snapshot(event["sequence"] - 1)
                prior = units[active_unit_id]
                if not unit_is_complete(
                    prior, actions, targets, event["sequence"] - 1
                ):
                    raise JournalCorrupt(
                        "a new work unit started before the prior frozen action graph completed"
                    )
                completed.append(active_unit_id)
            active_unit_id = event["payload"]["unit"]["unit_id"]
            continue

        if event_type == "model_call_intended" and active_unit_id is None:
            raise JournalCorrupt("model intent follows a completed work unit boundary")

        if event_type != "run_terminated":
            continue
        units, actions, targets = snapshot(event["sequence"])
        if active_unit_id is not None:
            active = units[active_unit_id]
            if unit_is_complete(active, actions, targets, event["sequence"]):
                completed.append(active_unit_id)
                active_unit_id = None
        expected = completed[-1] if completed else None
        claimed = event["payload"]["last_completed_work_unit_id"]
        if claimed != expected:
            raise JournalCorrupt(
                "terminal event's last completed work unit is inconsistent"
            )

    if active_unit_id is not None:
        units, actions, targets = snapshot(events[-1]["sequence"])
        # This also rejects attempts placed under guards that replay false,
        # while allowing a legitimately interrupted reachable slot to remain
        # active and incomplete.
        if unit_is_complete(
            units[active_unit_id], actions, targets, events[-1]["sequence"]
        ):
            completed.append(active_unit_id)
            active_unit_id = None
    return active_unit_id, tuple(completed)


def _derive_target_observations(
    run_id: str,
    logical_actions: Mapping[str, LogicalActionView],
    target_results: Mapping[str, TargetResultView],
) -> dict[str, TargetObservation]:
    """Project one canonical observation from each usable target action."""

    observations: dict[str, TargetObservation] = {}
    for action_id, result in target_results.items():
        action = logical_actions[action_id]
        attempt = next(
            (
                item
                for item in action.attempts
                if item.physical_attempt_id == result.physical_attempt_id
            ),
            None,
        )
        if attempt is None:
            raise JournalCorrupt("target result does not resolve to its successful attempt")
        lineages = {
            digest_json(item.intended["lineage"])
            for item in action.attempts
        }
        if len(lineages) != 1:
            raise JournalCorrupt("physical retries changed target action lineage")
        routes = action.slot["routes"]
        route = next(
            (
                item
                for item in routes
                if item["route_key"] == attempt.intended["role_route_key"]
            ),
            None,
        )
        if route is None:
            raise JournalCorrupt("target observation uses an unfrozen route")
        observation_key = digest_json(
            [action.slot["action_scientific_key"], "observation"]
        )
        observation_id = entity_id("to", [action_id])
        # The run identifier is deliberately not observation identity material;
        # the logical action ID is already unique inside this run.
        _ = run_id
        observations[action_id] = TargetObservation(
            observation_id=observation_id,
            scientific_key=observation_key,
            candidate_id=result.candidate_id,
            parent_node_id=result.parent_node_id,
            messages=tuple(
                ChatMessage(item["role"], item["content"])
                for item in attempt.intended["messages"]
            ),
            target_profile_digest=route["generation_profile_digest"],
            response=result.text,
            finish_status=FinishStatus(result.finish_status),
            successful_attempt_id=result.physical_attempt_id,
        )
    return observations


def _judge_member_for_action(
    *,
    run_id: str,
    action: LogicalActionView,
    observation: TargetObservation,
    logical_member_slot: str,
    response_schemas: Mapping[str, Mapping[str, Any]],
) -> tuple[MemberAssessment | None, tuple[str, ...]]:
    usable: list[MemberAssessment] = []
    errors: list[str] = []
    for attempt in action.attempts:
        if not attempt.succeeded or attempt.terminal_event is None:
            continue
        payload = attempt.terminal_event["payload"]
        parsed = validate_judge_output(
            payload["text"],
            observation.response,
            schema=response_schemas[attempt.intended["response_schema_digest"]],
            authoritative=True,
            finish_status=payload["finish_status"],
        )
        if parsed.output is None:
            errors.extend(parsed.errors)
            continue
        usable.append(
            member_assessment_from_output(
                parsed.output,
                assessment_id=entity_id(
                    "ma", [run_id, attempt.physical_attempt_id]
                ),
                observation_id=observation.observation_id,
                logical_member_slot=logical_member_slot,
                effective_model_id=payload["effective_model_id"],
                role_route_key=attempt.intended["role_route_key"],
                source_attempt_id=attempt.physical_attempt_id,
            )
        )
    if len(usable) > 1:
        raise JournalCorrupt(
            "one logical judge member has several structurally usable successes"
        )
    return (usable[0] if usable else None), tuple(errors)


def _derive_panel_views(
    *,
    run_id: str,
    work_units: Sequence[WorkUnitView],
    logical_actions: Mapping[str, LogicalActionView],
    target_results: Mapping[str, TargetResultView],
    target_observations: Mapping[str, TargetObservation],
    response_schemas: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[str, MemberAssessment],
    dict[str, PanelAssessment],
    dict[str, PanelView],
]:
    members_by_action: dict[str, MemberAssessment] = {}
    assessments: dict[str, PanelAssessment] = {}
    views: dict[str, PanelView] = {}
    for unit in work_units:
        semantic_cache: dict[str, Any] = {}
        for plan in unit.unit["panel_plans"]:
            panel_id = plan["panel_id"]
            if panel_id in views:
                raise JournalCorrupt("assessment panel ID appears more than once")
            target_action_id = plan["target_action_id"]
            target = target_results.get(target_action_id)
            observation = target_observations.get(target_action_id)
            normal_members: list[MemberAssessment] = []
            invalid_actions: list[str] = []
            validation_errors: dict[str, tuple[str, ...]] = {}
            primary: MemberAssessment | None = None
            secondary_attempted = False

            for member_plan in plan["normal_members"]:
                action_id = member_plan["judge_action_id"]
                action = logical_actions[action_id]
                slot = str(member_plan["logical_member_slot"])
                for attempt in action.attempts:
                    lineage = attempt.intended["lineage"]
                    if lineage["panel_id"] != panel_id or lineage["logical_member_slot"] != slot:
                        raise JournalCorrupt(
                            "normal judge attempt lineage differs from its frozen panel slot"
                        )
                if slot != "primary" and action.attempts:
                    secondary_attempted = True
                if observation is None:
                    continue
                member, errors = _judge_member_for_action(
                    run_id=run_id,
                    action=action,
                    observation=observation,
                    logical_member_slot=slot,
                    response_schemas=response_schemas,
                )
                if errors:
                    validation_errors[action_id] = errors
                if member is None:
                    if action.attempts:
                        invalid_actions.append(action_id)
                    continue
                members_by_action[action_id] = member
                normal_members.append(member)
                if slot == "primary":
                    primary = member

            primary_action = logical_actions[
                plan["normal_members"][0]["judge_action_id"]
            ]
            primary_closed = _completion_action_closed(
                unit,
                primary_action,
                logical_actions,
                target_results,
                response_schemas,
                semantic_cache,
            )
            triggered = target is not None and bool(
                plan["trigger_policy"] == "always_full"
                or (primary is not None and should_schedule_secondary_panel(primary))
                or (primary is None and primary_closed)
                or secondary_attempted
            )

            replacement_plan = plan["replacement"]
            replacement_id = replacement_plan["judge_action_id"] if replacement_plan is not None else None
            replacement_action = logical_actions[replacement_id] if replacement_id is not None else None
            replacement_member: MemberAssessment | None = None
            if replacement_action is not None and replacement_action.attempts:
                replacement_slots = {
                    attempt.intended["lineage"]["logical_member_slot"]
                    for attempt in replacement_action.attempts
                }
                if len(replacement_slots) != 1 or next(iter(replacement_slots)) not in {
                    "primary",
                    "secondary_a",
                    "secondary_b",
                }:
                    raise JournalCorrupt(
                        "replacement judge lineage does not name one normal panel slot"
                    )
                replacement_slot = str(next(iter(replacement_slots)))
                if observation is not None:
                    replacement_member, errors = _judge_member_for_action(
                        run_id=run_id,
                        action=replacement_action,
                        observation=observation,
                        logical_member_slot=replacement_slot,
                        response_schemas=response_schemas,
                    )
                    if errors:
                        validation_errors[replacement_id] = errors
                    if replacement_member is None:
                        invalid_actions.append(replacement_id)
                    else:
                        members_by_action[replacement_id] = replacement_member

            aggregate_members = list(normal_members)
            if replacement_member is not None:
                aggregate_members = [
                    item
                    for item in aggregate_members
                    if item.logical_member_slot
                    != replacement_member.logical_member_slot
                ]
                aggregate_members.append(replacement_member)
            details = (
                aggregate_panel_details(
                    panel_id,
                    observation.observation_id,
                    aggregate_members,
                    target_effective_model_id=(
                        None if target is None else target.effective_model_id
                    ),
                )
                if triggered and observation is not None
                else None
            )
            if details is not None:
                assessments[panel_id] = details.assessment
                included_member_ids = frozenset(details.assessment.member_ids)
                if (
                    primary is not None
                    and primary.assessment_id not in included_member_ids
                ):
                    primary = None
                if (
                    replacement_member is not None
                    and replacement_member.logical_member_slot == "primary"
                    and replacement_member.assessment_id in included_member_ids
                ):
                    primary = replacement_member

            target_action = logical_actions[target_action_id]
            panel_actions = [
                logical_actions[item["judge_action_id"]]
                for item in plan["normal_members"]
            ] + ([replacement_action] if replacement_action is not None else [])
            target_closed = _completion_action_closed(
                unit,
                target_action,
                logical_actions,
                target_results,
                response_schemas,
                semantic_cache,
            )
            panel_actions_closed = all(
                _completion_action_closed(
                    unit,
                    action,
                    logical_actions,
                    target_results,
                    response_schemas,
                    semantic_cache,
                )
                or not _completion_guard_allows(
                    unit,
                    action,
                    logical_actions,
                    target_results,
                    response_schemas,
                    semantic_cache,
                )
                for action in panel_actions
            )
            views[panel_id] = PanelView(
                panel_id=panel_id,
                target_action_id=target_action_id,
                observation_id=(
                    None if observation is None else observation.observation_id
                ),
                candidate_id=None if target is None else target.candidate_id,
                execution_id=None if target is None else target.execution_id,
                target_event_id=None if target is None else target.terminal_event_id,
                triggered=triggered,
                primary=primary,
                details=details,
                invalid_member_actions=tuple(sorted(set(invalid_actions))),
                member_validation_errors=dict(sorted(validation_errors.items())),
                terminated=bool(target_closed and panel_actions_closed),
            )
    return members_by_action, assessments, views


def _derive_sample_executions(
    *,
    work_units: Sequence[WorkUnitView],
    logical_actions: Mapping[str, LogicalActionView],
    target_results: Mapping[str, TargetResultView],
    target_observations: Mapping[str, TargetObservation],
    panel_views: Mapping[str, PanelView],
    response_schemas: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, SampleExecution], dict[str, ExecutionEvidence]]:
    samples: dict[str, SampleExecution] = {}
    evidence_by_execution: dict[str, ExecutionEvidence] = {}
    panel_by_target = {
        view.target_action_id: view for view in panel_views.values()
    }

    def scenario_outcome(
        unit: WorkUnitView,
        execution_id: str,
        root_plan: Mapping[str, Any],
    ) -> tuple[ExecutionEvidence, ExecutionTerminalStatus]:
        """Reduce one stateful recipe root without guard-cascade ambiguity."""

        ordered = [logical_actions[item] for item in root_plan["ordered_action_ids"]]
        targets = [
            item for item in ordered if item.slot["role"] == Role.TARGET.value
        ]
        bindings = [
            item
            for item in ordered
            if item.slot["purpose"] == ActionPurpose.TEMPLATE_BINDING.value
        ]
        proposals = [
            item
            for item in ordered
            if item.slot["purpose"] == ActionPurpose.PERTURBATOR_PROPOSAL.value
        ]
        reviews = [
            item
            for item in ordered
            if item.slot["purpose"] == ActionPurpose.PROBE_REVIEW.value
        ]
        outcomes = [
            item
            for item in ordered
            if item.slot["purpose"] == ActionPurpose.OUTCOME_REVIEW.value
        ]
        depth = len(bindings)
        if not (
            depth >= 1
            and len(targets) == depth + 1
            and len(proposals) == 2 * depth
            and len(reviews) == 2 * depth
            and len(outcomes) == depth
        ):
            raise JournalCorrupt("scenario execution topology is not canonical")

        durable_root: str | None = None
        for target_index, target in enumerate(targets):
            if not target.attempts:
                continue
            lineages = [attempt.intended["lineage"] for attempt in target.attempts]
            roots = {item["conversation_root_id"] for item in lineages}
            executions = {item["execution_id"] for item in lineages}
            parents = {item["parent_node_id"] for item in lineages}
            if len(roots) != 1 or None in roots or executions != {execution_id}:
                raise JournalCorrupt(
                    "scenario target retries changed their fresh execution root"
                )
            current_root = next(iter(roots))
            if durable_root is None:
                durable_root = current_root
            elif current_root != durable_root:
                raise JournalCorrupt(
                    "scenario target steps merged different conversation roots"
                )
            if target_index == 0:
                if parents != {None}:
                    raise JournalCorrupt(
                        "scenario entry target must start without a parent node"
                    )
                continue
            if len(parents) != 1 or None in parents:
                raise JournalCorrupt(
                    "scenario continuation target lacks its exact fresh parent"
                )
            prior = targets[target_index - 1]
            prior_result = target_results.get(prior.action_id)
            if prior_result is None or not prior.attempts:
                raise JournalCorrupt(
                    "scenario continuation injected an unavailable target history"
                )
            prior_messages = prior.attempts[0].intended["messages"]
            current_messages = target.attempts[0].intended["messages"]
            expected_prefix = [
                *prior_messages,
                {"role": "assistant", "content": prior_result.text},
            ]
            if (
                len(current_messages) != len(expected_prefix) + 1
                or current_messages[:-1] != expected_prefix
                or current_messages[-1].get("role") != "user"
            ):
                raise JournalCorrupt(
                    "scenario continuation is not the exact freshly generated branch"
                )

        proposal_by_key = {
            item["scientific_key"]: item for item in unit.unit["proposal_slots"]
        }

        def pending_or_unresolved(
            action: LogicalActionView,
        ) -> tuple[ExecutionEvidence, ExecutionTerminalStatus]:
            closed = _completion_action_closed(
                unit,
                action,
                logical_actions,
                target_results,
                response_schemas,
                semantic_cache,
            )
            return (
                ExecutionEvidence(
                    execution_id=execution_id,
                    terminated=closed,
                    resolved=False,
                ),
                ExecutionTerminalStatus.OPERATIONALLY_UNRESOLVED,
            )

        def structured(action: LogicalActionView) -> Mapping[str, Any] | None:
            return _structured_action_value(action, response_schemas)

        entry_panel = panel_by_target.get(targets[0].action_id)
        if targets[0].action_id not in target_observations or entry_panel is None:
            return pending_or_unresolved(targets[0])
        if not entry_panel.terminated:
            return (
                ExecutionEvidence(execution_id, False, False),
                ExecutionTerminalStatus.OPERATIONALLY_UNRESOLVED,
            )
        if not entry_panel.resolved:
            return (
                ExecutionEvidence(execution_id, True, False),
                ExecutionTerminalStatus.OPERATIONALLY_UNRESOLVED,
            )

        final_panel: PanelView | None = None
        for step in range(depth):
            binding_value = structured(bindings[step])
            if binding_value is None:
                return pending_or_unresolved(bindings[step])
            binding_status = binding_value.get("status")
            if binding_status == "inapplicable":
                return (
                    ExecutionEvidence(
                        execution_id,
                        terminated=True,
                        resolved=True,
                        semantic_nonsuccess=True,
                    ),
                    ExecutionTerminalStatus.SEMANTIC_NONSUCCESS,
                )
            if binding_status != "applicable":
                raise JournalCorrupt("scenario binding status is invalid")

            target = targets[step + 1]
            target_guard = target.slot["guard"]
            proposal_key = target_guard["proposal_slot_scientific_key"]
            proposal_slot = proposal_by_key.get(proposal_key)
            if proposal_slot is None:
                raise JournalCorrupt("scenario step names an unknown proposal slot")
            slot_id = proposal_slot["slot_id"]
            review_item_id = make_review_item_id(
                unit.unit["unit_scientific_key"], "probe", proposal_key
            )

            original_batch = _parsed_structured_action(
                proposals[2 * step],
                response_schemas,
                lambda value, frozen=ProposalSlot(
                    slot_id,
                    proposal_key,
                    proposal_slot["global_ordinal"],
                ): parse_proposal_batch(value, (frozen,)),
                label="scenario proposal",
            )
            if not isinstance(original_batch, ProposalBatch):
                return pending_or_unresolved(proposals[2 * step])
            if len(original_batch.items) != 1:
                raise JournalCorrupt("scenario proposal batch must contain one slot")
            original = original_batch.items[0]
            if original.slot_id != slot_id:
                raise JournalCorrupt("scenario proposal batch changed its frozen slot")
            if not original.proposed:
                return (
                    ExecutionEvidence(execution_id, True, False),
                    ExecutionTerminalStatus.OPERATIONALLY_UNRESOLVED,
                )

            expectation = ReviewExpectation(review_item_id, original)
            review_statuses = _parsed_structured_action(
                reviews[2 * step],
                response_schemas,
                lambda value, expected=expectation: parse_probe_review_statuses(
                    value, (expected,), mode="strategy"
                ),
                label="scenario probe review",
            )
            if not isinstance(review_statuses, Mapping):
                return pending_or_unresolved(reviews[2 * step])
            alignment = _review_status(review_statuses, review_item_id)

            if alignment != "aligned":
                correction_batch = _parsed_structured_action(
                    proposals[2 * step + 1],
                    response_schemas,
                    lambda value, frozen=ProposalSlot(
                        slot_id,
                        proposal_key,
                        proposal_slot["global_ordinal"],
                    ): parse_proposal_batch(value, (frozen,)),
                    label="scenario corrective proposal",
                )
                if not isinstance(correction_batch, ProposalBatch):
                    return pending_or_unresolved(proposals[2 * step + 1])
                if len(correction_batch.items) != 1:
                    raise JournalCorrupt(
                        "scenario corrective batch must contain one slot"
                    )
                correction = correction_batch.items[0]
                if correction.slot_id != slot_id:
                    raise JournalCorrupt(
                        "scenario corrective batch changed its frozen slot"
                    )
                if not correction.proposed:
                    return (
                        ExecutionEvidence(execution_id, True, False),
                        ExecutionTerminalStatus.OPERATIONALLY_UNRESOLVED,
                    )
                correction_expectation = ReviewExpectation(
                    review_item_id, correction
                )
                rereview_statuses = _parsed_structured_action(
                    reviews[2 * step + 1],
                    response_schemas,
                    lambda value, expected=correction_expectation: parse_probe_review_statuses(
                        value, (expected,), mode="strategy"
                    ),
                    label="scenario corrective probe review",
                )
                if not isinstance(rereview_statuses, Mapping):
                    return pending_or_unresolved(reviews[2 * step + 1])
                alignment = _review_status(rereview_statuses, review_item_id)
            if alignment != "aligned":
                return (
                    ExecutionEvidence(execution_id, True, False),
                    ExecutionTerminalStatus.OPERATIONALLY_UNRESOLVED,
                )

            step_panel = panel_by_target.get(target.action_id)
            if target.action_id not in target_observations or step_panel is None:
                return pending_or_unresolved(target)
            if not step_panel.terminated:
                return (
                    ExecutionEvidence(execution_id, False, False),
                    ExecutionTerminalStatus.OPERATIONALLY_UNRESOLVED,
                )
            if not step_panel.resolved:
                return (
                    ExecutionEvidence(execution_id, True, False),
                    ExecutionTerminalStatus.OPERATIONALLY_UNRESOLVED,
                )
            final_panel = step_panel

            outcome_review_id = make_review_item_id(
                unit.unit["unit_scientific_key"], "outcome", proposal_key
            )
            observation = target_observations[target.action_id]
            outcome_reviews = _parsed_structured_action(
                outcomes[step],
                response_schemas,
                lambda value, expected={
                    outcome_review_id: (
                        observation.scientific_key,
                        observation.response,
                    )
                }: parse_outcome_reviews(value, expected),
                label="scenario outcome review",
            )
            if not isinstance(outcome_reviews, tuple):
                return pending_or_unresolved(outcomes[step])
            if len(outcome_reviews) != 1:
                raise JournalCorrupt("scenario outcome review must contain one item")
            result = outcome_reviews[0].result
            if step == depth - 1 and result != "supports":
                return (
                    ExecutionEvidence(
                        execution_id,
                        terminated=True,
                        resolved=True,
                        semantic_nonsuccess=True,
                    ),
                    ExecutionTerminalStatus.SEMANTIC_NONSUCCESS,
                )

        if final_panel is None:
            raise JournalCorrupt("scenario execution lacks its final panel")
        return (
            ExecutionEvidence(
                execution_id=execution_id,
                terminated=final_panel.terminated,
                resolved=final_panel.resolved,
                primary_score=(
                    None if final_panel.primary is None else final_panel.primary.score
                ),
                panel_required=final_panel.panel_required,
                panel=(
                    final_panel.assessment if final_panel.panel_required else None
                ),
            ),
            ExecutionTerminalStatus.SUCCESS,
        )

    for unit in work_units:
        semantic_cache: dict[str, Any] = {}
        root_plans = {
            item["execution_id"]: item
            for item in unit.unit["design"].get("root_plans", [])
            if isinstance(item, Mapping) and isinstance(item.get("execution_id"), str)
        }
        for slot in unit.unit["execution_slots"]:
            execution_id = slot["execution_id"]
            if execution_id in samples:
                raise JournalCorrupt("sample execution ID appears more than once")
            root_plan = root_plans.get(execution_id)
            candidate_actions = [
                action
                for action in unit.actions
                if action.slot["role"] == Role.TARGET.value
                and action.slot["owner_position_scientific_key"]
                == slot["execution_scientific_key"]
            ]
            if root_plan is not None:
                ordered_ids = root_plan.get("ordered_action_ids", [])
                candidate_actions.extend(
                    logical_actions[action_id]
                    for action_id in ordered_ids
                    if action_id in logical_actions
                    and logical_actions[action_id].slot["role"] == Role.TARGET.value
                    and logical_actions[action_id] not in candidate_actions
                )
            candidate_actions.sort(key=lambda item: item.slot["ordinal"])
            target_action_ids = tuple(item.action_id for item in candidate_actions)
            terminal_action_id = (
                root_plan.get("terminal_endpoint_action_id")
                if root_plan is not None
                else (target_action_ids[-1] if target_action_ids else None)
            )
            if terminal_action_id is not None and terminal_action_id not in target_action_ids:
                raise JournalCorrupt(
                    "root plan terminal endpoint is not one of its target actions"
                )
            observation_ids = tuple(
                target_observations[action_id].observation_id
                for action_id in target_action_ids
                if action_id in target_observations
            )
            terminal_observation = (
                target_observations.get(terminal_action_id)
                if isinstance(terminal_action_id, str)
                else None
            )

            semantic_nonsuccess = False
            terminated = False
            resolved = False
            execution_evidence: ExecutionEvidence
            terminal_action = (
                logical_actions.get(terminal_action_id)
                if isinstance(terminal_action_id, str)
                else None
            )
            panel = (
                panel_by_target.get(terminal_action_id)
                if isinstance(terminal_action_id, str)
                else None
            )
            is_scenario = bool(
                unit.unit["kind"] == "verification"
                and unit.unit["design"].get("subject_kind") == "scenario_recipe"
            )
            scenario_terminal_status: ExecutionTerminalStatus | None = None
            if is_scenario:
                if root_plan is None:
                    raise JournalCorrupt("scenario execution lacks its root plan")
                execution_evidence, scenario_terminal_status = scenario_outcome(
                    unit, execution_id, root_plan
                )
                terminated = execution_evidence.terminated
                resolved = execution_evidence.resolved
                semantic_nonsuccess = execution_evidence.semantic_nonsuccess
            elif terminal_observation is not None and panel is not None:
                terminated = panel.terminated
                resolved = panel.resolved
                execution_evidence = ExecutionEvidence(
                    execution_id=execution_id,
                    terminated=terminated,
                    resolved=resolved,
                    primary_score=(
                        None if panel.primary is None else panel.primary.score
                    ),
                    panel_required=panel.panel_required,
                    # A quorum may exist before a still-required replacement
                    # slot terminates.  Keep that provisional assessment on
                    # PanelView for scheduling, but do not project it as
                    # confirmed execution evidence until the panel resolves.
                    panel=(
                        panel.assessment
                        if panel.panel_required and resolved
                        else None
                    ),
                )
            elif terminal_action is not None:
                action_closed = _completion_action_closed(
                    unit,
                    terminal_action,
                    logical_actions,
                    target_results,
                    response_schemas,
                    semantic_cache,
                )
                dependencies_closed = all(
                    _completion_action_closed(
                        unit,
                        logical_actions[dependency],
                        logical_actions,
                        target_results,
                        response_schemas,
                        semantic_cache,
                    )
                    or not _completion_guard_allows(
                        unit,
                        logical_actions[dependency],
                        logical_actions,
                        target_results,
                        response_schemas,
                        semantic_cache,
                    )
                    for dependency in terminal_action.slot["dependency_action_ids"]
                )
                guard_allows = _completion_guard_allows(
                    unit,
                    terminal_action,
                    logical_actions,
                    target_results,
                    response_schemas,
                    semantic_cache,
                )
                semantic_nonsuccess = bool(
                    not terminal_action.attempts
                    and dependencies_closed
                    and not guard_allows
                )
                terminated = bool(action_closed or semantic_nonsuccess)
                resolved = semantic_nonsuccess
                execution_evidence = ExecutionEvidence(
                    execution_id=execution_id,
                    terminated=terminated,
                    resolved=resolved,
                    semantic_nonsuccess=semantic_nonsuccess,
                )
            else:
                # An allocated slot without a target endpoint is closed only by
                # an explicitly complete semantic work unit.  No response is
                # invented for a still-pending design.
                semantic_nonsuccess = _completion_unit_is_complete(
                    unit, logical_actions, target_results, response_schemas
                )
                terminated = semantic_nonsuccess
                resolved = semantic_nonsuccess
                execution_evidence = ExecutionEvidence(
                    execution_id=execution_id,
                    terminated=terminated,
                    resolved=resolved,
                    semantic_nonsuccess=semantic_nonsuccess,
                )

            conversation_root_id = entity_id(
                "cr", [execution_id, "conversation_root"]
            )
            for action in candidate_actions:
                if action.attempts:
                    lineage_roots = {
                        attempt.intended["lineage"]["conversation_root_id"]
                        for attempt in action.attempts
                    }
                    if len(lineage_roots) != 1:
                        raise JournalCorrupt(
                            "physical retries changed the conversation root"
                        )
                    durable_root = next(iter(lineage_roots))
                    if durable_root is not None:
                        conversation_root_id = durable_root
                        break
            terminal_status = scenario_terminal_status or (
                ExecutionTerminalStatus.SEMANTIC_NONSUCCESS
                if semantic_nonsuccess
                else ExecutionTerminalStatus.SUCCESS
                if terminal_observation is not None
                else ExecutionTerminalStatus.OPERATIONALLY_UNRESOLVED
            )
            samples[execution_id] = SampleExecution(
                execution_id=execution_id,
                scientific_key=slot["execution_scientific_key"],
                work_unit_id=unit.unit_id,
                subject_id=(
                    next(
                        (
                            (
                                target_results[action_id].candidate_id
                                if action_id in target_results
                                else logical_actions[action_id]
                                .attempts[0]
                                .intended["lineage"]["candidate_id"]
                            )
                            for action_id in target_action_ids
                            if (
                                action_id in target_results
                                and target_results[action_id].candidate_id is not None
                            )
                            or logical_actions[action_id].attempts
                        ),
                        None,
                    )
                    or unit.unit["subject_id"]
                ),
                conversation_root_id=conversation_root_id,
                target_action_ids=target_action_ids,
                observation_ids=observation_ids,
                terminal_observation_id=(
                    None
                    if terminal_observation is None
                    else terminal_observation.observation_id
                ),
                terminal_status=terminal_status,
            )
            evidence_by_execution[execution_id] = execution_evidence
    for unit in work_units:
        if not (
            unit.unit["kind"] == "verification"
            and unit.unit["design"].get("subject_kind") == "scenario_recipe"
        ):
            continue
        execution_ids = [item["execution_id"] for item in unit.unit["execution_slots"]]
        roots = [samples[item].conversation_root_id for item in execution_ids]
        attempted = [
            root
            for item, root in zip(execution_ids, roots, strict=True)
            if any(
                logical_actions[action_id].attempts
                for action_id in samples[item].target_action_ids
            )
        ]
        if len(attempted) != len(set(attempted)):
            raise JournalCorrupt(
                "scenario verification executions reused a conversation root"
            )
    return samples, evidence_by_execution


def _derive_conversations(
    *,
    sample_executions: Mapping[str, SampleExecution],
    target_observations: Mapping[str, TargetObservation],
    target_results: Mapping[str, TargetResultView],
) -> tuple[dict[str, ConversationRootView], dict[str, ConversationNode]]:
    execution_keys = {
        item.execution_id: item.scientific_key
        for item in sample_executions.values()
    }
    root_execution_ids: dict[str, set[str]] = {}
    root_keys: dict[str, str] = {}
    for execution in sample_executions.values():
        root_execution_ids.setdefault(execution.conversation_root_id, set()).add(
            execution.execution_id
        )
        key = digest_json(
            [execution.scientific_key, "conversation_root"]
        )
        existing = root_keys.get(execution.conversation_root_id)
        if existing is None:
            root_keys[execution.conversation_root_id] = key
        elif existing != key:
            # A stateful root can own several correlated execution slots.  Its
            # original root key is recovered from its earliest durable node.
            pass

    nodes: dict[str, ConversationNode] = {}
    node_for_observation: dict[str, str] = {}
    ordered = sorted(
        target_results.items(), key=lambda item: item[1].terminal_sequence
    )
    for action_id, result in ordered:
        observation = target_observations[action_id]
        execution = (
            sample_executions.get(result.execution_id)
            if result.execution_id is not None
            else None
        )
        root_id = result.conversation_root_id or (
            None if execution is None else execution.conversation_root_id
        )
        if root_id is None:
            # Calibration/foundation fixtures may deliberately exercise a
            # target action outside a scientific execution.  It remains a
            # target observation but does not invent a conversation entity.
            continue
        root_execution_ids.setdefault(root_id, set())
        if execution is not None:
            root_execution_ids[root_id].add(execution.execution_id)
            root_key = digest_json(
                [execution_keys[execution.execution_id], "conversation_root"]
            )
            root_keys.setdefault(root_id, root_key)
        parent_id = result.parent_node_id
        if parent_id is not None and parent_id not in nodes:
            raise JournalCorrupt("conversation node names an unknown prior parent")
        user_turns = [
            message.content
            for message in observation.messages
            if message.role == "user"
        ]
        if not user_turns:
            raise JournalCorrupt("target observation has no dispatched user turn")
        user_turn = user_turns[-1]
        normalized_user_turn = normalize_line_endings(user_turn)
        parent_key = (
            nodes[parent_id].scientific_key
            if parent_id is not None
            else root_keys.get(root_id)
        )
        if parent_key is None:
            raise JournalCorrupt("conversation root lacks scientific identity")
        node_key = digest_json(
            [
                parent_key,
                "conversation_node",
                sha256_bytes(normalized_user_turn.encode("utf-8")),
                observation.scientific_key,
            ]
        )
        node_id = entity_id(
            "cn",
            [
                parent_id or root_id,
                normalized_user_turn,
                observation.observation_id,
            ],
        )
        if node_id in nodes:
            raise JournalCorrupt("conversation node identity collision")
        node = ConversationNode(
            node_id=node_id,
            scientific_key=node_key,
            root_id=root_id,
            parent_node_id=parent_id,
            depth=0 if parent_id is None else nodes[parent_id].depth + 1,
            candidate_id=observation.candidate_id,
            target_observation_id=observation.observation_id,
        )
        nodes[node_id] = node
        node_for_observation[observation.observation_id] = node_id

    roots = {
        root_id: ConversationRootView(
            root_id=root_id,
            scientific_key=root_keys[root_id],
            execution_ids=tuple(sorted(execution_ids)),
            node_ids=tuple(
                node.node_id
                for node in sorted(
                    nodes.values(), key=lambda item: (item.depth, item.scientific_key)
                )
                if node.root_id == root_id
            ),
        )
        for root_id, execution_ids in sorted(root_execution_ids.items())
        if root_id in root_keys
    }
    return roots, nodes


def _structured_action_value(
    action: LogicalActionView,
    response_schemas: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    values: list[Mapping[str, Any]] = []
    for attempt in action.successful_attempts:
        if not _success_is_structurally_usable(
            action, attempt, response_schemas
        ) or attempt.terminal_event is None:
            continue
        try:
            parsed = strict_json_loads(attempt.terminal_event["payload"]["text"])
        except DomainError:
            continue
        if isinstance(parsed, Mapping):
            values.append(parsed)
    if len(values) > 1:
        raise JournalCorrupt("one structured action has several usable results")
    return values[0] if values else None


def _candidate_metadata(
    state_scope: str,
    run_id: str,
    manifest: Mapping[str, Any],
    work_units: Sequence[WorkUnitView],
    logical_actions: Mapping[str, LogicalActionView],
    target_results: Mapping[str, TargetResultView],
    response_schemas: Mapping[str, Mapping[str, Any]],
    *,
    objective: ObjectivePlugin | None = None,
    portfolio_rounds: Mapping[str, PortfolioRoundView] | None = None,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    try:
        target_profile = manifest["normalized_config"]["models"]["target"]
        target_digest = digest_json(target_profile)
        plugin_digest = manifest["objective"]["plugin_digest"]
    except (KeyError, TypeError):
        # A journal with no objective snapshot cannot produce candidate or
        # archive science, but lower-level event/recovery fixtures may still be
        # reduced without fabricating that missing metadata.
        return {}

    def add(
        *,
        candidate_id: str,
        key: str,
        prompt: str,
        descriptors: Mapping[str, Any],
        origin: str,
        strategy_id: str | None = None,
        source_unit_id: str | None = None,
        generated: bool = False,
        assigned_probe_review_scope_key: str | None = None,
        review_status: str | None = None,
    ) -> None:
        expected_id = entity_id("pc", [run_id, key])
        if candidate_id != expected_id:
            raise JournalCorrupt("candidate ID differs from its scientific identity")
        value = {
            "candidate_id": candidate_id,
            "scientific_key": key,
            "prompt": normalize_line_endings(prompt),
            "descriptors": dict(descriptors),
            "origin": origin,
            "strategy_id": strategy_id,
            "source_unit_id": source_unit_id,
            "generated": generated,
            "assigned_probe_review_scope_key": assigned_probe_review_scope_key,
            "review_status": review_status,
        }
        prior = result.get(candidate_id)
        if prior is not None:
            if (
                prior["scientific_key"] != key
                or prior["prompt"] != normalize_line_endings(prompt)
                or prior["descriptors"] != dict(descriptors)
                or prior["review_status"] != review_status
            ):
                raise JournalCorrupt("one candidate has conflicting immutable metadata")
            # Content-equivalent trusted sources are one candidate.  Preserve
            # the first canonical source only as report provenance.
            return
        result[candidate_id] = value

    bootstrap = manifest.get("bootstrap_snapshot", {})
    probes = bootstrap.get("probes", []) if isinstance(bootstrap, Mapping) else []
    for item in probes:
        if not isinstance(item, Mapping):
            continue
        prompt, descriptors = item.get("prompt"), item.get("descriptors")
        if not isinstance(prompt, str) or not isinstance(descriptors, Mapping):
            continue
        key = digest_json(
            [
                state_scope,
                "standalone_prompt",
                plugin_digest,
                target_digest,
                sha256_bytes(normalize_line_endings(prompt).encode("utf-8")),
            ]
        )
        add(
            candidate_id=entity_id("pc", [run_id, key]),
            key=key,
            prompt=prompt,
            descriptors=descriptors,
            origin="bootstrap",
        )
    for item in manifest.get("seed_snapshot", []):
        if not isinstance(item, Mapping):
            continue
        if item.get("type") == "scenario_recipe":
            # Stateful seed entries are executable recipes.  Treating their
            # entry text as a one-turn candidate would create a different
            # experiment and leak verification-only work into MAP search.
            continue
        prompt, descriptors = item.get("entry_prompt"), item.get("descriptors")
        if not isinstance(prompt, str) or not isinstance(descriptors, Mapping):
            continue
        probe_keys = tuple(
            sorted(
                scope.scope_key
                for _, scope in _structured_role_scope_candidates(manifest)
                if scope.scope == "probe_review"
                and scope.scope_key in item.get("qualification_digests", [])
            )
        )
        if len(probe_keys) > 1:
            raise JournalCorrupt(
                "selected seed carries several assigned probe-review scopes"
            )
        key = digest_json(
            [
                state_scope,
                "standalone_prompt",
                plugin_digest,
                target_digest,
                sha256_bytes(normalize_line_endings(prompt).encode("utf-8")),
            ]
        )
        add(
            candidate_id=entity_id("pc", [run_id, key]),
            key=key,
            prompt=prompt,
            descriptors=descriptors,
            origin="seed",
            generated=bool(probe_keys),
            assigned_probe_review_scope_key=(
                probe_keys[0] if probe_keys else None
            ),
        )

    units_by_id = {unit.unit_id: unit for unit in work_units}
    ordinary_by_unit: dict[str, Mapping[str, SelectedRealization]] = {}
    if objective is not None:
        for unit in work_units:
            if unit.unit["kind"] != "map_exploration":
                continue
            proposal_actions = tuple(
                item
                for item in unit.actions
                if item.slot["purpose"]
                == ActionPurpose.PERTURBATOR_PROPOSAL.value
            )
            review_actions = tuple(
                item
                for item in unit.actions
                if item.slot["purpose"] == ActionPurpose.PROBE_REVIEW.value
            )
            if len(proposal_actions) != 1 or len(review_actions) != 1:
                raise JournalCorrupt("ordinary MAP collaboration graph is invalid")
            slots = _proposal_slots_for_unit(unit)
            batch = _parsed_structured_action(
                proposal_actions[0],
                response_schemas,
                lambda value, frozen=slots: parse_proposal_batch(value, frozen),
                label="ordinary MAP proposal",
            )
            if not isinstance(batch, ProposalBatch):
                ordinary_by_unit[unit.unit_id] = {}
                continue
            slot_by_id = {item.slot_id: item for item in slots}
            expectations = tuple(
                ReviewExpectation(
                    make_review_item_id(
                        unit.unit["unit_scientific_key"],
                        "probe",
                        slot_by_id[item.slot_id].scientific_key,
                    ),
                    item,
                )
                for item in batch.proposed_items
            )
            reviews = (
                ()
                if not expectations
                else _parsed_structured_action(
                    review_actions[0],
                    response_schemas,
                    lambda value, expected=expectations: parse_probe_reviews(
                        value,
                        expected,
                        objective.descriptor_values,
                        mode="ordinary",
                    ),
                    label="ordinary MAP probe review",
                )
            )
            review_by_id = {
                item.review_item_id: item for item in (reviews or ())
            }
            selected: dict[str, SelectedRealization] = {}
            for expectation in expectations:
                review = review_by_id.get(expectation.review_item_id)
                if review is None:
                    continue
                try:
                    realization = select_ordinary_realization(
                        expectation.proposal, review
                    )
                except DomainError as exc:
                    raise JournalCorrupt(
                        f"ordinary MAP realization is invalid: {exc}"
                    ) from exc
                if realization is not None:
                    selected[expectation.review_item_id] = realization
            ordinary_by_unit[unit.unit_id] = selected

    for target in sorted(
        target_results.values(), key=lambda item: item.terminal_sequence
    ):
        candidate_id = target.candidate_id
        if candidate_id is None or candidate_id in result:
            continue
        action = logical_actions[target.logical_action_id]
        if not action.attempts:
            continue
        unit = units_by_id[action.unit_id]
        if unit.unit["kind"] not in {"map_exploration", "portfolio_round"}:
            # Screening, verification, development, and sealed holdout calls
            # are evaluation-only.  Their target turns never create search or
            # archive candidates.
            continue
        user_turns = [
            item["content"]
            for item in action.attempts[0].intended["messages"]
            if item["role"] == "user"
        ]
        if not user_turns:
            raise JournalCorrupt("candidate target request has no user turn")
        prompt = user_turns[-1]
        lineage = action.attempts[0].intended["lineage"]
        review_item_id = lineage["review_item_id"]
        selected: SelectedRealization | None = None
        if isinstance(review_item_id, str) and unit.unit["kind"] == "map_exploration":
            selected = ordinary_by_unit.get(unit.unit_id, {}).get(review_item_id)
        elif unit.unit["kind"] == "portfolio_round" and portfolio_rounds is not None:
            round_view = portfolio_rounds.get(unit.unit_id)
            matches = (
                []
                if round_view is None
                else [
                    child
                    for child in round_view.children
                    if child.target_action_id == action.action_id
                ]
            )
            if len(matches) > 1:
                raise JournalCorrupt("portfolio target has several child projections")
            if matches:
                selected = matches[0].selected
        if (
            selected is None
            or selected.review.reviewed_descriptors is None
            or selected.proposal.user_turn is None
        ):
            continue
        if normalize_line_endings(selected.proposal.user_turn) != normalize_line_endings(
            prompt
        ):
            raise JournalCorrupt("target request differs from its reviewed proposal")
        descriptors = descriptors_dict(selected.review.reviewed_descriptors)
        review_status = selected.review.status
        if target.parent_node_id is None:
            key = digest_json(
                [
                    state_scope,
                    "standalone_prompt",
                    plugin_digest,
                    target_digest,
                    sha256_bytes(normalize_line_endings(prompt).encode("utf-8")),
                ]
            )
            origin = "generated"
        else:
            parent_key = unit.unit["design"].get(
                "generation_parent_node_scientific_key"
            )
            if not isinstance(parent_key, str):
                raise JournalCorrupt("stateful candidate lacks parent scientific key")
            key = digest_json(
                [
                    parent_key,
                    "stateful_turn",
                    target_digest,
                    sha256_bytes(normalize_line_endings(prompt).encode("utf-8")),
                ]
            )
            origin = "stateful_strategy"
        add(
            candidate_id=candidate_id,
            key=key,
            prompt=prompt,
            descriptors=descriptors,
            origin=origin,
            strategy_id=lineage["strategy_id"],
            source_unit_id=unit.unit_id,
            generated=origin == "generated",
            review_status=review_status,
        )
    return result


@dataclass(frozen=True, slots=True)
class _StaticRoute:
    role: str
    configured_slot: str
    route_label: str
    route_key: str
    requested_provider_route_key: str
    requested_model: str
    generation_profile_digest: str


@dataclass(frozen=True, slots=True)
class _ExpectedSelectorTrial:
    trial: Mapping[str, Any]
    mode: str
    batch_index: int | None = None


def _static_route(
    manifest: Mapping[str, Any], role: str, configured_slot: str
) -> _StaticRoute:
    route_label = _expected_route_label(role, configured_slot)
    profile, model = _profile_for_binding(manifest, role, configured_slot)
    generation_digest = digest_json(profile)
    try:
        provider_key = requested_provider_route_key(
            manifest["normalized_config"]["provider"]["adapter"],
            role,
            route_label,
        )
        scope = manifest["scientific_scope_digest"]
    except (KeyError, TypeError, DomainError) as exc:
        raise JournalCorrupt(f"component route scope is invalid: {exc}") from exc
    route_key = digest_json(
        [
            scope,
            "role_route",
            role,
            configured_slot,
            route_label,
            provider_key,
            model,
            generation_digest,
        ]
    )
    return _StaticRoute(
        role,
        configured_slot,
        route_label,
        route_key,
        provider_key,
        model,
        generation_digest,
    )


def _response_mode_key(route_key: str, schema_digest: str) -> str:
    return digest_json([route_key, "structured_response_mode", schema_digest])


def _judge_scope_for_route(
    manifest: Mapping[str, Any],
    route: _StaticRoute,
    *,
    response_mode: str,
) -> StructuredRoleScope:
    try:
        return judge_role_scope(
            requested_model=route.requested_model,
            generation_profile_digest=route.generation_profile_digest,
            judge_system_template_digest=manifest["role_template_digests"][
                "judge_system"
            ],
            judge_user_template_digest=manifest["role_template_digests"][
                "judge_user"
            ],
            judge_schema_digest=manifest["response_schema_digests"][
                "judge_output"
            ],
            response_mode=response_mode,
        )
    except (KeyError, TypeError, DomainError) as exc:
        raise JournalCorrupt(f"judge role scope is invalid: {exc}") from exc


_SELECTOR_SCOPE_BINDINGS: Mapping[str, tuple[str, str, str]] = {
    "portfolio": ("selector_portfolio", "selector_system", "selector_user"),
    "probe_review": ("probe_review", "probe_review_system", "probe_review_user"),
    "template_binding": (
        "template_binding",
        "template_binding_system",
        "template_binding_user",
    ),
    "outcome_review": (
        "outcome_review",
        "outcome_review_system",
        "outcome_review_user",
    ),
}


def _selector_scope_for_route(
    manifest: Mapping[str, Any],
    route: _StaticRoute,
    *,
    scope: str,
    response_mode: str,
) -> StructuredRoleScope:
    try:
        schema_name, system_name, user_name = _SELECTOR_SCOPE_BINDINGS[scope]
        return selector_role_scope(
            scope,
            requested_model=route.requested_model,
            system_template_digest=manifest["role_template_digests"][system_name],
            user_template_digest=manifest["role_template_digests"][user_name],
            schema_digest=manifest["response_schema_digests"][schema_name],
            response_mode=response_mode,
        )
    except (KeyError, TypeError, DomainError) as exc:
        raise JournalCorrupt(f"Selector role scope is invalid: {exc}") from exc


def _source_run_directory(run_path_at_creation: object) -> str:
    if not isinstance(run_path_at_creation, str) or not run_path_at_creation:
        raise JournalCorrupt("run creation path is not a nonempty string")
    path = Path(run_path_at_creation)
    if (
        not path.is_absolute()
        or path.as_posix() != run_path_at_creation
        or any(part in {".", ".."} for part in path.parts)
    ):
        raise JournalCorrupt("run creation path is not a normalized absolute path")
    return run_path_at_creation


def _checkpoint(
    run_id: str,
    source_run_directory: str,
    terminal_events: Iterable[Mapping[str, Any]],
) -> Mapping[str, Any]:
    events = tuple(terminal_events)
    if not events:
        raise JournalCorrupt("evidence checkpoint has no terminal event")
    last = max(events, key=lambda item: item["sequence"])
    return {
        "source_run_id": run_id,
        "sequence": last["sequence"],
        "event_hash": last["event_hash"],
        "source_path": source_run_directory,
    }


def _structured_role_scope_candidates(
    manifest: Mapping[str, Any],
) -> tuple[tuple[tuple[str, str, str], StructuredRoleScope], ...]:
    candidates: list[tuple[tuple[str, str, str], StructuredRoleScope]] = []
    seen: set[str] = set()

    def add(
        role: str,
        configured_slot: str,
        scope_name: str,
        scope: StructuredRoleScope,
    ) -> None:
        if scope.scope_key in seen:
            return
        seen.add(scope.scope_key)
        candidates.append(((role, configured_slot, scope_name), scope))

    for configured_slot in configured_judge_slots(manifest):
        route = _static_route(manifest, Role.JUDGE.value, configured_slot)
        for mode in ("json_schema", "json_object"):
            scope = _judge_scope_for_route(
                manifest,
                route,
                response_mode=mode,
            )
            add(Role.JUDGE.value, configured_slot, "judge_route", scope)
    for configured_slot in ("selector_primary", "selector_fallback"):
        route = _static_route(manifest, Role.SELECTOR.value, configured_slot)
        for selector_scope in _SELECTOR_SCOPE_BINDINGS:
            for mode in ("json_schema", "json_object"):
                scope = _selector_scope_for_route(
                    manifest,
                    route,
                    scope=selector_scope,
                    response_mode=mode,
                )
                add(Role.SELECTOR.value, configured_slot, selector_scope, scope)
    return tuple(candidates)


def _observed_component_response_mode(
    *,
    route: _StaticRoute,
    schema_digest: str,
    logical_actions: Mapping[str, LogicalActionView],
    default: str = "json_schema",
) -> str:
    attempts = sorted(
        (
            attempt
            for action in logical_actions.values()
            for attempt in action.attempts
            if attempt.intended["role_route_key"] == route.route_key
            and attempt.intended["response_schema_digest"] == schema_digest
            and not attempt.target_identity_collision
        ),
        key=lambda item: item.intent_event["sequence"],
    )
    if not attempts:
        return default
    mode = attempts[-1].intended["response_mode"]
    if mode not in {"json_schema", "json_object"}:
        raise JournalCorrupt("component evidence used an invalid response mode")
    return str(mode)


def _candidate_required_selector_scopes(
    candidate: Mapping[str, Any],
) -> tuple[str, ...]:
    origin = candidate.get("origin")
    if origin == "generated":
        return ("probe_review",)
    if origin != "stateful_strategy":
        return ()
    review_status = candidate.get("review_status")
    if review_status == "aligned":
        return ("probe_review", "outcome_review")
    raise JournalCorrupt("stateful archive candidate lacks a dispatchable review status")


def _exact_structured_role_scopes(
    *,
    manifest: Mapping[str, Any],
    candidate: Mapping[str, Any],
    work_units: Sequence[WorkUnitView],
    logical_actions: Mapping[str, LogicalActionView],
    panel_views: Mapping[str, PanelView],
    member_assessments: Mapping[str, MemberAssessment],
    response_schemas: Mapping[str, Mapping[str, Any]],
 ) -> tuple[str, ...]:
    objective = manifest.get("objective")
    if not isinstance(objective, Mapping) or not isinstance(
        objective.get("plugin_digest"), str
    ):
        raise JournalCorrupt("objective role scope is invalid")
    required: list[str] = [f"objective:{objective['plugin_digest']}"]

    judge_schema = manifest["response_schema_digests"]["judge_output"]
    for logical_slot in configured_judge_slots(manifest, include_replacement=False):
        route = _static_route(manifest, Role.JUDGE.value, logical_slot)
        mode = _observed_component_response_mode(
            route=route,
            schema_digest=judge_schema,
            logical_actions=logical_actions,
        )
        scope = _judge_scope_for_route(manifest, route, response_mode=mode)
        required.append(scope.scope_key)

    # Replacement qualification is required only for evidence that actually
    # used the replacement route.  The frozen logical slot determines its
    # exact inherited-weight compatibility key.
    replacement = (_static_route(manifest, Role.JUDGE.value, "replacement")
                   if manifest["model_profiles"]["judges"]["replacement"] is not None else None)
    source_candidate_id = candidate["candidate_id"]
    candidate_panel_ids = {
        panel.panel_id
        for panel in panel_views.values()
        if source_candidate_id is not None and panel.candidate_id == source_candidate_id
    }
    for assessment in member_assessments.values():
        if replacement is None or assessment.role_route_key != replacement.route_key:
            continue
        panel = next(
            (
                view
                for view in panel_views.values()
                if view.observation_id == assessment.observation_id
            ),
            None,
        )
        if panel is None or (
            candidate_panel_ids and panel.panel_id not in candidate_panel_ids
        ):
            continue
        source_attempt = next(
            (
                attempt
                for action in logical_actions.values()
                for attempt in action.attempts
                if attempt.physical_attempt_id == assessment.source_attempt_id
            ),
            None,
        )
        mode = (
            source_attempt.intended["response_mode"]
            if source_attempt is not None
            else "json_schema"
        )
        scope = _judge_scope_for_route(
            manifest,
            replacement,
            response_mode=mode,
        )
        if scope.scope_key not in required:
            required.append(scope.scope_key)

    selector_scopes = _candidate_required_selector_scopes(candidate)
    source_unit = next(
        (
            unit
            for unit in work_units
            if unit.unit_id == candidate.get("source_unit_id")
        ),
        None,
    )
    for selector_scope in selector_scopes:
        purpose = {
            "probe_review": ActionPurpose.PROBE_REVIEW.value,
            "outcome_review": ActionPurpose.OUTCOME_REVIEW.value,
        }[selector_scope]
        actions = (
            ()
            if source_unit is None
            else tuple(
                action
                for action in source_unit.actions
                if action.slot["purpose"] == purpose
            )
        )
        usable_attempts = tuple(
            attempt
            for action in actions
            for attempt in action.successful_attempts
            if _success_is_structurally_usable(
                action,
                attempt,
                response_schemas,
            )
        )
        # The exact embedded schema has already been authenticated globally.
        # A schema-valid successful attempt is the only route that can ground
        # the generated candidate or its outcome review.  When none exists,
        # retain the primary strict scope as the visible missing requirement.
        if usable_attempts:
            attempts_for_scope = usable_attempts
        else:
            attempts_for_scope = ()
        if attempts_for_scope:
            route_key = attempts_for_scope[-1].intended["role_route_key"]
            route = next(
                (
                    _static_route(manifest, Role.SELECTOR.value, slot)
                    for slot in ("selector_primary", "selector_fallback")
                    if _static_route(
                        manifest, Role.SELECTOR.value, slot
                    ).route_key
                    == route_key
                ),
                None,
            )
            if route is None:
                raise JournalCorrupt("Selector evidence used an unknown route")
            mode = str(attempts_for_scope[-1].intended["response_mode"])
        else:
            route = _static_route(
                manifest, Role.SELECTOR.value, "selector_primary"
            )
            mode = "json_schema"
        scope = _selector_scope_for_route(
            manifest, route, scope=selector_scope, response_mode=mode
        )
        required.append(scope.scope_key)
    if len(required) != len(set(required)):
        raise JournalCorrupt("structured role scope set is duplicated")
    return tuple(required)


def _derive_evidence_and_archive(
    *,
    run_id: str,
    scope: str,
    manifest: Mapping[str, Any],
    work_units: Sequence[WorkUnitView],
    logical_actions: Mapping[str, LogicalActionView],
    target_results: Mapping[str, TargetResultView],
    panel_views: Mapping[str, PanelView],
    member_assessments: Mapping[str, MemberAssessment],
    execution_evidence: Mapping[str, ExecutionEvidence],
    conversation_nodes: Mapping[str, ConversationNode],
    response_schemas: Mapping[str, Mapping[str, Any]],
    objective: ObjectivePlugin | None = None,
    portfolio_rounds: Mapping[str, PortfolioRoundView] | None = None,
) -> tuple[
    dict[str, EvidenceRecord],
    Mapping[str, ScreeningBlockView],
    ArchiveSnapshot,
]:
    metadata = _candidate_metadata(
        scope,
        run_id,
        manifest,
        work_units,
        logical_actions,
        target_results,
        response_schemas,
        objective=objective,
        portfolio_rounds=portfolio_rounds,
    )
    panels_by_candidate: dict[str, list[PanelView]] = {}
    for panel in panel_views.values():
        if panel.candidate_id is not None:
            panels_by_candidate.setdefault(panel.candidate_id, []).append(panel)
    for panels in panels_by_candidate.values():
        panels.sort(
            key=lambda item: (
                target_results[item.target_action_id].terminal_sequence
                if item.target_action_id in target_results
                else 10**18,
                item.panel_id,
            )
        )
    units_by_kind_subject: dict[tuple[str, str], list[WorkUnitView]] = {}
    for unit in work_units:
        units_by_kind_subject.setdefault(
            (unit.unit["kind"], unit.unit["subject_id"]), []
        ).append(unit)

    taxonomy = manifest.get("descriptor_taxonomy", {})
    descriptors = (
        taxonomy.get("descriptors", []) if isinstance(taxonomy, Mapping) else []
    )
    descriptor_names = tuple(
        item["name"]
        for item in descriptors
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    )
    try:
        capacity = manifest["normalized_config"]["search"][
            "archive_capacity_per_cell"
        ]
    except (KeyError, TypeError):
        capacity = 1
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
        raise JournalCorrupt("manifest archive capacity is invalid")

    records: dict[str, EvidenceRecord] = {}
    screening_blocks: dict[str, ScreeningBlockView] = {}
    entries: list[ArchiveEntry] = []
    claimed_block_executions: set[str] = set()
    node_by_observation = {
        node.target_observation_id: node.node_id
        for node in conversation_nodes.values()
    }
    for candidate_id, candidate in sorted(metadata.items()):
        candidate_panels = panels_by_candidate.get(candidate_id, [])
        if not candidate_panels:
            continue
        candidate_key = str(candidate["scientific_key"])
        singleton_candidates = [
            panel
            for panel in candidate_panels
            if next(
                unit.unit["kind"]
                for unit in work_units
                if unit.unit_id == logical_actions[panel.target_action_id].unit_id
            )
            in {"initial_observation", "map_exploration", "portfolio_round"}
        ]
        singleton = singleton_candidates[0] if singleton_candidates else candidate_panels[0]
        singleton_evidence = (
            execution_evidence.get(singleton.execution_id)
            if singleton.execution_id is not None
            else None
        )

        screen_units = units_by_kind_subject.get(
            ("screening", candidate_id), []
        )
        if len(screen_units) > 1:
            raise JournalCorrupt("candidate has more than one screening block")
        screening: ScreeningDecision | None = None
        screening_block_id: str | None = None
        if screen_units:
            screen = screen_units[0]
            triggering_id = screen.unit["design"]["triggering_execution_id"]
            if singleton.execution_id != triggering_id:
                raise JournalCorrupt(
                    "screening block did not reuse the candidate's actual triggering execution"
                )
            trigger = execution_evidence.get(triggering_id)
            if trigger is None:
                raise JournalCorrupt(
                    "screening block does not reference an actual triggering execution"
                )
            new_execution_ids = tuple(
                item["execution_id"] for item in screen.unit["execution_slots"]
            )
            if any(
                execution_id not in execution_evidence
                or not execution_evidence[execution_id].terminated
                for execution_id in new_execution_ids
            ):
                # A started fixed block is visible during every durable call
                # prefix.  It is not evidence, and cannot own a screening
                # decision, until all of its allocated executions terminate.
                continue
            block_evidence = [trigger]
            block_evidence.extend(
                execution_evidence[item["execution_id"]]
                for item in screen.unit["execution_slots"]
            )
            block_ids = {item.execution_id for item in block_evidence}
            if block_ids & claimed_block_executions:
                raise JournalCorrupt("sample execution belongs to several fixed blocks")
            claimed_block_executions.update(block_ids)
            try:
                screening = evaluate_screening(block_evidence)
            except DomainError as exc:
                raise JournalCorrupt(f"screening block is invalid: {exc}") from exc
            screening_block_id = screen.unit["design"]["block_id"]
            if screening_block_id != entity_id(
                "sb", [run_id, candidate_key, "screening", 3]
            ):
                raise JournalCorrupt("screening block ID is not canonical")
            panel_by_target_action = {
                panel.target_action_id: panel for panel in candidate_panels
            }
            target_action_by_owner = {
                action.slot["owner_position_scientific_key"]: action.action_id
                for action in screen.actions
                if action.slot["role"] == Role.TARGET.value
            }
            ordered_target_action_ids = [singleton.target_action_id]
            ordered_panels = [singleton]
            panel_projection_pending = False
            for execution_slot in screen.unit["execution_slots"]:
                target_action_id = target_action_by_owner.get(
                    execution_slot["execution_scientific_key"]
                )
                if target_action_id is None:
                    raise JournalCorrupt(
                        "screening execution lacks its exact target action"
                    )
                ordered_target_action_ids.append(target_action_id)
                panel = panel_by_target_action.get(target_action_id)
                if panel is None:
                    if target_action_id in target_results:
                        panel_projection_pending = True
                    continue
                ordered_panels.append(panel)
            if panel_projection_pending:
                # A usable target result becomes durable before its assessment
                # panel.  Keep that valid prefix pending.  An exhausted target
                # action without a usable result is terminal unresolved and has
                # no panel to project.
                continue
            screening_blocks[candidate_id] = ScreeningBlockView(
                candidate_id=candidate_id,
                candidate_scientific_key=candidate_key,
                block_id=screening_block_id,
                triggering_execution_id=triggering_id,
                new_execution_ids=tuple(
                    item["execution_id"]
                    for item in screen.unit["execution_slots"]
                ),
                target_action_ids=tuple(ordered_target_action_ids),
                observation_ids=tuple(
                    panel.observation_id
                    for panel in ordered_panels
                    if panel.observation_id is not None
                ),
                decision=screening,
            )

        verification_units = units_by_kind_subject.get(
            ("verification", candidate_id), []
        )
        if len(verification_units) > 1:
            raise JournalCorrupt("candidate has more than one verification block")
        verification: VerificationDecision | None = None
        verification_block_id: str | None = None
        if verification_units:
            verification_unit = verification_units[0]
            planned = verification_unit.unit["design"]["planned_count"]
            block_evidence = [
                execution_evidence[item["execution_id"]]
                for item in verification_unit.unit["execution_slots"]
            ]
            verification_ids = {item.execution_id for item in block_evidence}
            if verification_ids & claimed_block_executions:
                raise JournalCorrupt("sample execution belongs to several fixed blocks")
            claimed_block_executions.update(verification_ids)
            try:
                verification = evaluate_verification(
                    block_evidence,
                    planned_count=planned,
                )
            except DomainError as exc:
                raise JournalCorrupt(f"verification block is invalid: {exc}") from exc
            verification_block_id = entity_id(
                "sb", [run_id, candidate_key, "verification", planned]
            )

        evidence = evidence_record_from_decisions(
            subject_id=candidate_id,
            screening_block_id=screening_block_id,
            screening=screening,
            verification_block_id=verification_block_id,
            verification=verification,
        )
        if screening is None and singleton_evidence is not None:
            search_status = (
                # A disputed singleton remains a lead while its separate fixed
                # screen has not yet begun; it is never promising by itself.
                SearchStatus.EXPLORATORY_LEAD
                if singleton_evidence.disputed_high
                else evidence.search_status
            )
            evidence = EvidenceRecord(
                subject_id=evidence.subject_id,
                screening_block_id=None,
                search_status=search_status,
                verification_block_id=evidence.verification_block_id,
                verification_status=evidence.verification_status,
                scores=(
                    ()
                    if singleton_evidence.effective_score is None
                    else (singleton_evidence.effective_score,)
                )
                + (
                    ()
                    if verification is None
                    else verification.effective_scores
                ),
                unresolved_count=(
                    int(
                        singleton_evidence.terminated
                        and not singleton_evidence.resolved
                    )
                    + (0 if verification is None else verification.unresolved_count)
                ),
            )
        records[candidate_id] = evidence

        candidate_descriptors = candidate["descriptors"]
        if not descriptor_names:
            continue
        if not isinstance(candidate_descriptors, Mapping) or set(
            candidate_descriptors
        ) != set(descriptor_names):
            raise JournalCorrupt("candidate descriptor tuple differs from taxonomy")
        singleton_score = (
            None if singleton_evidence is None else singleton_evidence.effective_score
        )
        fitness = derive_archive_fitness(
            verification_terminal=bool(
                verification is not None and verification.block_terminated
            ),
            verification_scores=(
                () if verification is None else verification.effective_scores
            ),
            screening_complete=bool(
                screening is not None and screening.block_terminated
            ),
            screening_scores=(
                () if screening is None else screening.effective_scores
            ),
            singleton_score=singleton_score,
        ).fitness
        entry_key = digest_json([candidate_key, "archive_entry"])
        entries.append(
            ArchiveEntry(
                entry_id=entity_id("ae", [run_id, entry_key]),
                scientific_key=entry_key,
                subject_id=candidate_id,
                cell=tuple(
                    str(candidate_descriptors[name]) for name in descriptor_names
                ),
                evidence=evidence,
                fitness=fitness,
                lineage_id=(
                    None
                    if singleton.observation_id is None
                    else node_by_observation.get(singleton.observation_id)
                ),
                origin=str(candidate["origin"]),
                strategy_id=(
                    candidate["strategy_id"]
                    if isinstance(candidate.get("strategy_id"), str)
                    else None
                ),
            )
        )
    try:
        archive = build_archive(entries, capacity)
    except DomainError as exc:
        raise JournalCorrupt(f"archive projection is invalid: {exc}") from exc
    return records, MappingProxyType(dict(sorted(screening_blocks.items()))), archive


def _judge_protocol_digest(manifest: Mapping[str, Any]) -> str:
    """Freeze the common aggregation protocol independently of route identity."""

    try:
        return digest_json(
            [
                "judge_protocol",
                manifest["role_template_digests"]["judge_system"],
                manifest["role_template_digests"]["judge_user"],
                manifest["response_schema_digests"]["judge_output"],
                manifest["decision_constants"],
            ]
        )
    except (KeyError, TypeError) as exc:
        raise JournalCorrupt(f"seed judge protocol is invalid: {exc}") from exc


def _validate_entry_verification_unit(
    *,
    run_id: str,
    manifest: Mapping[str, Any],
    unit: WorkUnitView,
    candidate: Mapping[str, Any],
    logical_actions: Mapping[str, LogicalActionView],
    execution_evidence: Mapping[str, ExecutionEvidence],
) -> tuple[SeedVerificationExecution, ...]:
    """Authenticate one canonical fresh-root entry verification graph."""

    design = unit.unit["design"]
    planned = design["planned_count"]
    semantic = {
        "kind": "verification",
        "subject_kind": "entry_prompt",
        "candidate": candidate["scientific_key"],
        "planned_count": planned,
        "measurement_cell": dict(candidate["descriptors"]),
        "always_full_panel": True,
    }
    expected_unit_key = _qualification_unit_key(
        str(candidate["scientific_key"]), "verification", semantic
    )
    if (
        design.get("tag") != "verification"
        or design.get("subject_kind") != "entry_prompt"
        or design.get("subject_id") != candidate["candidate_id"]
        or design.get("subject_scientific_key") != candidate["scientific_key"]
        or design.get("measurement_cell") != dict(candidate["descriptors"])
        or unit.unit["subject_id"] != candidate["candidate_id"]
        or unit.unit["subject_scientific_key"] != candidate["scientific_key"]
        or unit.unit["unit_scientific_key"] != expected_unit_key
        or unit.unit_id != entity_id("wu", [run_id, expected_unit_key])
        or unit.unit["policy_digest"] != digest_json(manifest["decision_constants"])
        or isinstance(planned, bool)
        or not isinstance(planned, int)
        or planned < 5
        or len(unit.unit["execution_slots"]) != planned
        or len(design["root_plans"]) != planned
    ):
        raise JournalCorrupt("entry verification design is not canonical")

    plans_by_execution = {
        item["execution_id"]: item for item in design["root_plans"]
    }
    if len(plans_by_execution) != planned:
        raise JournalCorrupt("entry verification root allocation is duplicated")
    claimed_actions: set[str] = set()
    result: list[SeedVerificationExecution] = []
    prompt = normalize_line_endings(str(candidate["prompt"]))
    target_profile = manifest["normalized_config"]["models"]["target"]
    expected_messages = (
        ([{"role": "system", "content": target_profile["system_prompt"]}]
         if target_profile["system_prompt"]
         else [])
        + [{"role": "user", "content": prompt}]
    )
    for ordinal, execution in enumerate(unit.unit["execution_slots"]):
        execution_key = scientific_key(
            [expected_unit_key, "execution", ordinal, []]
        )
        execution_id = entity_id("ex", [run_id, execution_key])
        root_plan_id = entity_id(
            "rp", [run_id, expected_unit_key, "root_plan", ordinal]
        )
        plan = plans_by_execution.get(execution_id)
        if (
            execution.get("ordinal") != ordinal
            or execution.get("arm") != "none"
            or execution.get("execution_scientific_key") != execution_key
            or execution.get("execution_id") != execution_id
            or execution.get("root_plan_id") != root_plan_id
            or plan is None
            or plan.get("root_plan_id") != root_plan_id
            or plan.get("arm") != "none"
            or plan.get("semantic_nonsuccess_reasons") != []
        ):
            raise JournalCorrupt("entry verification root identity is not canonical")
        try:
            actions = tuple(
                logical_actions[action_id]
                for action_id in plan["ordered_action_ids"]
            )
        except KeyError as exc:
            raise JournalCorrupt(
                "entry verification root cites an unknown action"
            ) from exc
        if (
            not actions
            or tuple(
                action.action_id
                for action in sorted(actions, key=lambda item: item.slot["ordinal"])
            )
            != tuple(plan["ordered_action_ids"])
            or claimed_actions & {action.action_id for action in actions}
            or any(
                action.unit_id != unit.unit_id
                or action.slot["owner_position_scientific_key"] != execution_key
                for action in actions
            )
        ):
            raise JournalCorrupt("entry verification action graph is not canonical")
        claimed_actions.update(action.action_id for action in actions)
        targets = tuple(
            action for action in actions if action.slot["role"] == Role.TARGET.value
        )
        if (
            len(targets) != 1
            or targets[0].slot["purpose"] != ActionPurpose.TARGET_SAMPLE.value
            or plan.get("terminal_endpoint_action_id") != targets[0].action_id
            or tuple(action.slot["purpose"] for action in actions)
            != (
                ActionPurpose.TARGET_SAMPLE.value,
                *(ActionPurpose.JUDGE_MEMBER.value for _ in configured_judge_slots(manifest)),
            )
        ):
            raise JournalCorrupt("entry verification root blueprint is not canonical")
        target = targets[0]
        injected = any(
            message["role"] == "assistant"
            for attempt in target.attempts
            for message in attempt.intended["messages"]
        )
        fresh = bool(target.attempts)
        for attempt in target.attempts:
            messages = attempt.intended["messages"]
            user_turns = [
                message["content"]
                for message in messages
                if message["role"] == "user"
            ]
            lineage = attempt.intended["lineage"]
            fresh = bool(
                fresh
                and messages == expected_messages
                and len(user_turns) == 1
                and normalize_line_endings(user_turns[0]) == prompt
                and lineage["execution_id"] == execution_id
                and lineage["conversation_root_id"]
                == entity_id("cr", [execution_id, "conversation_root"])
                and lineage["candidate_id"] == candidate["candidate_id"]
                and lineage["parent_node_id"] is None
                and lineage["panel_id"] is None
                and lineage["logical_member_slot"] is None
                and lineage["strategy_id"] is None
                and lineage["proposal_slot_scientific_key"] is None
                and lineage["review_item_id"] is None
            )
        evidence = execution_evidence.get(execution_id)
        if evidence is None:
            raise JournalCorrupt("entry verification execution evidence is missing")
        try:
            result.append(
                SeedVerificationExecution(
                    evidence=evidence,
                    fresh_root=fresh,
                    end_to_end=False,
                    target_message_injected=injected,
                )
            )
        except DomainError as exc:
            raise JournalCorrupt(f"entry verification evidence is invalid: {exc}") from exc
    if claimed_actions != {action.action_id for action in unit.actions}:
        raise JournalCorrupt("entry verification roots do not partition their actions")
    return tuple(result)


def _validate_scenario_verification_unit(
    *,
    run_id: str,
    manifest: Mapping[str, Any],
    unit: WorkUnitView,
    scenario: ScenarioRecipeView,
    logical_actions: Mapping[str, LogicalActionView],
    target_results: Mapping[str, TargetResultView],
    target_observations: Mapping[str, TargetObservation],
    execution_evidence: Mapping[str, ExecutionEvidence],
    conversation_nodes: Mapping[str, ConversationNode],
    response_schemas: Mapping[str, Mapping[str, Any]],
) -> tuple[SeedVerificationExecution, ...]:
    """Authenticate one plural fresh-root scenario verification block."""

    design = unit.unit["design"]
    recipe = scenario.recipe
    planned = design["planned_count"]
    semantic = {
        "kind": "verification",
        "subject_kind": "scenario_recipe",
        "scenario_recipe_content_hash": recipe.content_hash,
        "planned_count": planned,
        "measurement_cell": descriptors_dict(recipe.measurement_cell),
        "root_blueprint": "fresh_anchor_scenario",
        "always_full_panel": True,
    }
    expected_unit_key = _qualification_unit_key(
        scenario.scientific_key, "verification", semantic
    )
    depth = len(recipe.templates)
    if (
        design.get("tag") != "verification"
        or design.get("subject_kind") != "scenario_recipe"
        or design.get("subject_id") != scenario.scenario_id
        or design.get("subject_scientific_key") != scenario.scientific_key
        or design.get("measurement_cell")
        != descriptors_dict(recipe.measurement_cell)
        or unit.unit["subject_id"] != scenario.scenario_id
        or unit.unit["subject_scientific_key"] != scenario.scientific_key
        or unit.unit["parent_unit_id"] != scenario.parent_unit_id
        or unit.unit["unit_scientific_key"] != expected_unit_key
        or unit.unit_id != entity_id("wu", [run_id, expected_unit_key])
        or unit.unit["policy_digest"] != digest_json(manifest["decision_constants"])
        or isinstance(planned, bool)
        or not isinstance(planned, int)
        or planned < 5
        or len(unit.unit["execution_slots"]) != planned
        or len(design["root_plans"]) != planned
        or len(unit.unit["proposal_slots"]) != planned * depth
        or recipe.maximum_strategy_rounds != depth
        or recipe.maximum_target_turns != depth + 1
        or recipe.designated_realization_policies != tuple(
            "anchor" for _ in range(depth)
        )
    ):
        raise JournalCorrupt("scenario verification design is not recipe-canonical")
    plans_by_execution = {
        item["execution_id"]: item for item in design["root_plans"]
    }
    if len(plans_by_execution) != planned:
        raise JournalCorrupt("scenario verification root allocation is duplicated")
    target_profile = manifest["normalized_config"]["models"]["target"]
    target_digest = digest_json(target_profile)
    if target_digest != recipe.target_generation_profile_digest:
        raise JournalCorrupt("scenario verification target profile changed")
    entry_messages = (
        ([{"role": "system", "content": target_profile["system_prompt"]}]
         if target_profile["system_prompt"]
         else [])
        + [{"role": "user", "content": scenario.entry_prompt}]
    )
    node_by_observation = {
        node.target_observation_id: node for node in conversation_nodes.values()
    }
    claimed_actions: set[str] = set()
    seen_roots: set[str] = set()
    result: list[SeedVerificationExecution] = []
    frozen_binding_key = dict(recipe.selector_scope_keys)["template_binding"]
    for ordinal, execution in enumerate(unit.unit["execution_slots"]):
        execution_id = execution["execution_id"]
        root = plans_by_execution.get(execution_id)
        root_id = entity_id("cr", [execution_id, "conversation_root"])
        if root is None or root_id in seen_roots:
            raise JournalCorrupt("scenario verification fresh-root identity is invalid")
        seen_roots.add(root_id)
        try:
            actions = tuple(
                logical_actions[action_id]
                for action_id in root["ordered_action_ids"]
            )
        except KeyError as exc:
            raise JournalCorrupt("scenario root cites an unknown action") from exc
        action_ids = {action.action_id for action in actions}
        if (
            execution.get("ordinal") != ordinal
            or execution.get("arm") != "none"
            or claimed_actions & action_ids
            or any(
                action.unit_id != unit.unit_id
                or action.slot["owner_position_scientific_key"]
                != execution["execution_scientific_key"]
                for action in actions
            )
        ):
            raise JournalCorrupt("scenario root ownership is not canonical")
        claimed_actions.update(action_ids)
        targets = tuple(
            action for action in actions if action.slot["role"] == Role.TARGET.value
        )
        bindings = tuple(
            action
            for action in actions
            if action.slot["purpose"] == ActionPurpose.TEMPLATE_BINDING.value
        )
        if (
            len(targets) != depth + 1
            or len(bindings) != depth
            or root.get("terminal_endpoint_action_id") != targets[-1].action_id
            or root.get("semantic_nonsuccess_reasons")
            != ["inapplicable", "nonprogress"]
        ):
            raise JournalCorrupt("scenario root topology differs from its recipe")
        binding_scope_match = True
        for binding in bindings:
            for attempt in binding.successful_attempts:
                if not _success_is_structurally_usable(
                    binding, attempt, response_schemas
                ):
                    continue
                scope = _component_scope_for_attempt(manifest, binding, attempt)
                binding_scope_match = bool(
                    binding_scope_match
                    and scope is not None
                    and scope.scope == "template_binding"
                    and scope.scope_key == frozen_binding_key
                )

        fresh = bool(targets[0].attempts)
        injected = False
        prior_messages: list[Mapping[str, Any]] | None = None
        prior_result: TargetResultView | None = None
        prior_node: ConversationNode | None = None
        for target_index, target in enumerate(targets):
            if not target.attempts:
                continue
            if target_index == 0:
                expected_messages = entry_messages
                expected_parent_id = None
                expected_candidate_id = entity_id("pc", [run_id, scenario.scientific_key])
                expected_proposal_key = None
                expected_review_id = None
            else:
                prior_target = targets[target_index - 1]
                prior_result = target_results.get(prior_target.action_id)
                prior_observation = target_observations.get(prior_target.action_id)
                if prior_result is None or prior_observation is None or prior_messages is None:
                    raise JournalCorrupt(
                        "scenario continuation lacks its current-root predecessor"
                    )
                prior_node = node_by_observation.get(prior_observation.observation_id)
                if prior_node is None or prior_node.root_id != root_id:
                    raise JournalCorrupt(
                        "scenario continuation lacks its exact parent node"
                    )
                last_messages = target.attempts[0].intended["messages"]
                prompt = (
                    last_messages[-1].get("content")
                    if last_messages and last_messages[-1].get("role") == "user"
                    else None
                )
                expected_messages = [
                    *prior_messages,
                    {"role": "assistant", "content": prior_result.text},
                    {"role": "user", "content": prompt},
                ]
                expected_parent_id = prior_node.node_id
                candidate_key = digest_json(
                    [
                        prior_node.scientific_key,
                        "stateful_turn",
                        target_digest,
                        sha256_bytes(
                            normalize_line_endings(str(prompt)).encode("utf-8")
                        ),
                    ]
                )
                expected_candidate_id = entity_id("pc", [run_id, candidate_key])
                expected_proposal_key = target.slot["guard"].get(
                    "proposal_slot_scientific_key"
                )
                expected_review_id = make_review_item_id(
                    unit.unit["unit_scientific_key"],
                    "probe",
                    expected_proposal_key,
                )
            for attempt in target.attempts:
                messages = attempt.intended["messages"]
                if messages != expected_messages:
                    injected = injected or any(
                        message.get("role") == "assistant" for message in messages
                    )
                    fresh = False
                lineage = attempt.intended["lineage"]
                if (
                    lineage["execution_id"] != execution_id
                    or lineage["conversation_root_id"] != root_id
                    or lineage["parent_node_id"] != expected_parent_id
                    or lineage["candidate_id"] != expected_candidate_id
                    or lineage["panel_id"] is not None
                    or lineage["logical_member_slot"] is not None
                    or lineage["proposal_slot_scientific_key"]
                    != expected_proposal_key
                    or lineage["review_item_id"] != expected_review_id
                    or (
                        target_index == 0
                        and lineage["strategy_id"] is not None
                    )
                    or (
                        target_index > 0
                        and (
                            not isinstance(lineage["strategy_id"], str)
                            or not lineage["strategy_id"].startswith("st-")
                        )
                    )
                ):
                    raise JournalCorrupt("scenario target lineage mutated")
            prior_messages = list(target.attempts[0].intended["messages"])
        evidence = execution_evidence.get(execution_id)
        if evidence is None:
            raise JournalCorrupt("scenario verification execution evidence is missing")
        try:
            result.append(
                SeedVerificationExecution(
                    evidence=evidence,
                    fresh_root=fresh,
                    end_to_end=binding_scope_match,
                    target_message_injected=injected,
                )
            )
        except DomainError as exc:
            raise JournalCorrupt(
                f"scenario verification evidence is invalid: {exc}"
            ) from exc
    if claimed_actions != {action.action_id for action in unit.actions}:
        raise JournalCorrupt("scenario roots do not partition their actions")
    return tuple(result)


def _seed_evidence_source(
    *,
    run_id: str,
    source_run_directory: str,
    events: Sequence[Mapping[str, Any]],
    units: Sequence[WorkUnitView],
    include_run_created: bool,
) -> SeedEvidenceSource:
    event_ids, _ = _qualification_evidence_events(units)
    selected = set(event_ids)
    if include_run_created:
        selected.add(events[0]["event_id"])
    evidence_events = tuple(
        SeedEvidenceEvent(
            event_id=event["event_id"],
            sequence=event["sequence"],
            event_hash=event["event_hash"],
            timestamp_utc=event["timestamp_utc"],
        )
        for event in events
        if event["event_id"] in selected
    )
    if len(evidence_events) != len(selected):
        raise JournalCorrupt("seed qualification evidence event is missing")
    try:
        return SeedEvidenceSource(
            source_run_id=run_id,
            source_path=source_run_directory,
            required_events=evidence_events,
        )
    except DomainError as exc:
        raise JournalCorrupt(f"seed qualification source is invalid: {exc}") from exc


def _derive_entry_seed_qualifications(
    *,
    run_id: str,
    source_run_directory: str,
    scientific_scope_digest: str,
    manifest: Mapping[str, Any],
    objective: ObjectivePlugin | None,
    events: Sequence[Mapping[str, Any]],
    work_units: Sequence[WorkUnitView],
    logical_actions: Mapping[str, LogicalActionView],
    target_results: Mapping[str, TargetResultView],
    response_schemas: Mapping[str, Mapping[str, Any]],
    execution_evidence: Mapping[str, ExecutionEvidence],
    candidate_evidence: Mapping[str, EvidenceRecord],
    panel_views: Mapping[str, PanelView],
    member_assessments: Mapping[str, MemberAssessment],
    target_identity: str | None,
) -> tuple[
    Mapping[str, SeedQualificationDecision],
    tuple[Mapping[str, Any], ...],
]:
    """Evaluate every current-run entry verification through the public API."""

    verification_units = tuple(
        unit
        for unit in work_units
        if unit.unit["kind"] == "verification"
        and unit.unit["design"].get("subject_kind") == "entry_prompt"
    )
    if not verification_units:
        return MappingProxyType({}), ()
    metadata = _candidate_metadata(
        scientific_scope_digest,
        run_id,
        manifest,
        work_units,
        logical_actions,
        target_results,
        response_schemas,
        objective=objective,
    )
    scopes = {
        scope.scope_key: (base, scope)
        for base, scope in _structured_role_scope_candidates(manifest)
    }
    target_route = _static_route(manifest, Role.TARGET.value, "target")
    target_provider_identity = ProviderIdentity(
        (
            f"{target_route.requested_provider_route_key}/"
            f"{target_route.requested_model}"
        ),
        target_identity,
    )
    protocol_digest = _judge_protocol_digest(manifest)
    objective_evidence = ObjectiveScopeEvidence(
        manifest["objective"]["id"],
        manifest["objective"]["plugin_digest"],
        _objective_scope_key(manifest),
    )
    taxonomy = manifest["descriptor_taxonomy"]
    descriptor_names = tuple(item["name"] for item in taxonomy["descriptors"])
    units_by_id = {unit.unit_id: unit for unit in work_units}
    decisions: dict[str, SeedQualificationDecision] = {}
    snapshots: list[Mapping[str, Any]] = []
    seen_candidates: set[str] = set()
    for unit in sorted(
        verification_units, key=lambda item: item.start_event["sequence"]
    ):
        candidate_id = unit.unit["subject_id"]
        if candidate_id in seen_candidates:
            raise JournalCorrupt("candidate has several entry verification units")
        seen_candidates.add(candidate_id)
        candidate = metadata.get(candidate_id)
        evidence = candidate_evidence.get(candidate_id)
        if candidate is None or evidence is None:
            raise JournalCorrupt("entry verification subject evidence is missing")
        executions = _validate_entry_verification_unit(
            run_id=run_id,
            manifest=manifest,
            unit=unit,
            candidate=candidate,
            logical_actions=logical_actions,
            execution_evidence=execution_evidence,
        )
        required_scope_keys = tuple(
            key
            for key in _exact_structured_role_scopes(
                manifest=manifest,
                candidate=candidate,
                work_units=work_units,
                logical_actions=logical_actions,
                panel_views=panel_views,
                member_assessments=member_assessments,
                response_schemas=response_schemas,
            )
            if key in scopes
        )
        judge_items: list[NamedScopeEvidence] = []
        component_by_name: dict[str, StructuredRoleScope] = {}
        for key in required_scope_keys:
            base, scope = scopes[key]
            if base[0] == Role.JUDGE.value:
                judge_items.append(
                    NamedScopeEvidence(f"judge_{base[1]}", key)
                )
            elif scope.scope in {"probe_review", "outcome_review"}:
                if scope.scope in component_by_name:
                    raise JournalCorrupt(
                        "entry seed requires several same-name Selector scopes"
                    )
                component_by_name[scope.scope] = scope
        judge_items.sort(key=lambda item: item.name)
        panel_key = digest_json(
            [
                "judge_panel_qualification",
                protocol_digest,
                [[item.name, item.scope_key] for item in judge_items],
            ]
        )
        judge_items.append(
            NamedScopeEvidence("panel", panel_key)
        )
        judge_panel = tuple(sorted(judge_items, key=lambda item: item.name))

        generated = bool(candidate["generated"])
        assigned_probe_key = candidate.get("assigned_probe_review_scope_key")
        if generated and assigned_probe_key is None:
            probe = component_by_name.get("probe_review")
            assigned_probe_key = None if probe is None else probe.scope_key
        components = (
            tuple(component_by_name[name] for name in sorted(component_by_name))
            if generated
            else ()
        )
        if generated and tuple(item.scope for item in components) != (
            "probe_review",
        ):
            components = tuple(
                item
                for item in components
                if item.scope == "probe_review"
            )
        descriptor_tuple = tuple(
            (name, str(candidate["descriptors"][name]))
            for name in descriptor_names
        )
        seed_id = entity_id(
            "sd",
            [
                "entry_seed",
                manifest["objective"]["id"],
                manifest["objective"]["plugin_digest"],
                target_route.generation_profile_digest,
                target_provider_identity.token,
                sha256_bytes(str(candidate["prompt"]).encode("utf-8")),
                [list(item) for item in descriptor_tuple],
            ],
        )
        try:
            subject = EntrySeedSubject(
                seed_id=seed_id,
                entry_prompt=str(candidate["prompt"]),
                descriptors=descriptor_tuple,
                objective_id=manifest["objective"]["id"],
                plugin_digest=manifest["objective"]["plugin_digest"],
                target_generation_profile_digest=(
                    target_route.generation_profile_digest
                ),
                descriptor_taxonomy_digest=taxonomy["digest"],
                target_identity=target_provider_identity,
                generated=generated,
                assigned_probe_review_scope_key=assigned_probe_key,
            )
            rule = SeedVerificationRule(unit.unit["design"]["planned_count"])
            block_id = entity_id(
                "sb",
                [
                    run_id,
                    candidate["scientific_key"],
                    "verification",
                    rule.planned_count,
                ],
            )
            block = SeedVerificationBlock(
                block_id=block_id,
                rule=rule,
                executions=executions,
                judge_panel_scope_keys=tuple(
                    (item.name, item.scope_key) for item in judge_panel
                ),
            )
            context = SeedQualificationContext(
                objective=objective_evidence,
                judge_protocol_digest=protocol_digest,
                judge_panel=judge_panel,
                components=components,
                current_perturbator_scope=None,
            )
        except (KeyError, DomainError) as exc:
            raise JournalCorrupt(f"entry seed qualification input is invalid: {exc}") from exc
        evidence_units = [unit]
        source_unit_id = candidate.get("source_unit_id")
        if generated and isinstance(source_unit_id, str):
            source_unit = units_by_id.get(source_unit_id)
            if source_unit is None:
                raise JournalCorrupt("generated entry seed source unit is missing")
            evidence_units.insert(0, source_unit)
        source = _seed_evidence_source(
            run_id=run_id,
            source_run_directory=source_run_directory,
            events=events,
            units=evidence_units,
            include_run_created=generated and source_unit_id is None,
        )
        try:
            decision = evaluate_seed_qualification(subject, block, context, source)
        except DomainError as exc:
            raise JournalCorrupt(f"entry seed qualification is invalid: {exc}") from exc
        decisions[seed_id] = decision
        if decision.snapshot is not None:
            snapshots.append(MappingProxyType(decision.snapshot.to_dict()))
    return (
        MappingProxyType(dict(sorted(decisions.items()))),
        tuple(sorted(snapshots, key=lambda item: item["seed_id"])),
    )


def _stateful_objective(
    manifest: Mapping[str, Any],
    objective: ObjectivePlugin | None,
    work_units: Sequence[WorkUnitView],
) -> ObjectivePlugin | None:
    portfolio_units = tuple(
        unit for unit in work_units if unit.unit["kind"] == "portfolio_round"
    )
    if not portfolio_units:
        return None
    needs_semantic_parser = any(
        attempt.succeeded
        for unit in portfolio_units
        for action in unit.actions
        if action.slot["role"] != Role.TARGET.value
        for attempt in action.attempts
    )
    if not needs_semantic_parser:
        return objective
    if objective is None:
        raise JournalCorrupt(
            "stateful semantic evidence requires the exact objective plugin"
        )
    manifest_objective = manifest.get("objective")
    if not isinstance(manifest_objective, Mapping) or (
        manifest_objective.get("id") != objective.id
        or manifest_objective.get("plugin_digest") != objective.plugin_digest
    ):
        raise JournalCorrupt(
            "stateful reducer objective does not match the frozen manifest"
        )
    taxonomy = manifest.get("descriptor_taxonomy")
    expected_taxonomy = list(objective.objective["descriptors"])
    if not isinstance(taxonomy, Mapping) or (
        taxonomy.get("descriptors") != expected_taxonomy
        or taxonomy.get("digest") != digest_json(expected_taxonomy)
    ):
        raise JournalCorrupt(
            "stateful reducer objective taxonomy differs from the manifest"
        )
    return objective


def _parsed_structured_action(
    action: LogicalActionView,
    response_schemas: Mapping[str, Mapping[str, Any]],
    parser: Any,
    *,
    label: str,
) -> Any | None:
    usable: list[tuple[Any, int]] = []
    attempt_indexes = {
        attempt.physical_attempt_id: index
        for index, attempt in enumerate(action.attempts)
    }
    for attempt in action.successful_attempts:
        if attempt.terminal_event is None or not _success_passes_base_structure(
            action, attempt, response_schemas
        ):
            continue
        try:
            raw = strict_json_loads(attempt.terminal_event["payload"]["text"])
            usable.append(
                (parser(raw), attempt_indexes[attempt.physical_attempt_id])
            )
        except DomainError:
            continue
    if len(usable) > 1:
        raise JournalCorrupt(
            f"one {label} action has several semantically usable results"
        )
    if usable and usable[0][1] != len(action.attempts) - 1:
        raise JournalCorrupt(
            f"one {label} action continued after a semantically usable result"
        )
    return usable[0][0] if usable else None


def _semantic_action_terminal(
    action: LogicalActionView, semantic_value: Any | None
) -> bool:
    """Return whether bounded recovery is closed for a role-level parser."""

    if semantic_value is not None:
        return True
    if not action.attempts:
        return False
    latest = action.attempts[-1]
    terminal = latest.terminal_event
    if terminal is None:
        return False
    routes = action.slot["routes"]
    route_index = next(
        (
            index
            for index, route in enumerate(routes)
            if route["route_key"] == latest.intended["role_route_key"]
        ),
        None,
    )
    if route_index is None:
        raise JournalCorrupt("semantic action used an unfrozen route")
    route = routes[route_index]
    attempts_on_route = sum(
        attempt.intended["role_route_key"] == route["route_key"]
        for attempt in action.attempts
    )
    if latest.target_identity_collision:
        return route_index + 1 >= len(routes)
    retry_same_route = terminal["type"] in {
        "model_call_succeeded",
        "model_call_outcome_unknown",
    }
    if terminal["type"] == "model_call_failed":
        error = terminal["payload"]["error_kind"]
        retry_same_route = error in {
            "connection",
            "timeout",
            "rate_limit",
            "server",
            "malformed_response",
        } or (
            error == "unsupported_response_format"
            and action.slot["role"] != Role.TARGET.value
            and latest.intended["response_mode"] == "json_schema"
        )
    if retry_same_route and attempts_on_route < route["maximum_attempts"]:
        return False
    return route_index + 1 >= len(routes)


def _proposal_slots_for_unit(unit: WorkUnitView) -> tuple[ProposalSlot, ...]:
    try:
        slots = tuple(
            ProposalSlot(
                str(item["slot_id"]),
                str(item["scientific_key"]),
                int(item["global_ordinal"]),
            )
            for item in unit.unit["proposal_slots"]
        )
    except (DomainError, KeyError, TypeError, ValueError) as exc:
        raise JournalCorrupt(f"portfolio proposal slots are invalid: {exc}") from exc
    ordered = tuple(sorted(slots, key=lambda item: item.global_ordinal))
    if [item.global_ordinal for item in ordered] != list(range(len(ordered))):
        raise JournalCorrupt("portfolio proposal slot ordinals are not contiguous")
    if len({item.slot_id for item in ordered}) != len(ordered) or len(
        {item.scientific_key for item in ordered}
    ) != len(ordered):
        raise JournalCorrupt("portfolio proposal slots are not unique")
    return ordered


def _root_plan_for_action(
    unit: WorkUnitView, action: LogicalActionView
) -> Mapping[str, Any]:
    matches = [
        item
        for item in unit.unit["design"].get("root_plans", [])
        if action.action_id in item.get("ordered_action_ids", [])
    ]
    if len(matches) != 1:
        raise JournalCorrupt("stateful action does not belong to one frozen root")
    return matches[0]


def _scenario_proposal_slot_for_action(
    unit: WorkUnitView, action: LogicalActionView
) -> ProposalSlot:
    root = _root_plan_for_action(unit, action)
    root_actions = tuple(
        item
        for item in unit.actions
        if item.action_id in root["ordered_action_ids"]
    )
    bindings = tuple(
        item
        for item in root_actions
        if item.slot["purpose"] == ActionPurpose.TEMPLATE_BINDING.value
    )
    if not bindings:
        raise JournalCorrupt("scenario root lacks continuation bindings")
    step_index = max(
        (
            index
            for index, binding in enumerate(bindings)
            if binding.slot["ordinal"] <= action.slot["ordinal"]
        ),
        default=0,
    )
    executions = [
        item
        for item in unit.unit["execution_slots"]
        if item["execution_id"] == root["execution_id"]
    ]
    if len(executions) != 1:
        raise JournalCorrupt("scenario root lacks one frozen execution")
    proposal_ordinal = executions[0]["ordinal"] * len(bindings) + step_index
    matches = [
        item
        for item in _proposal_slots_for_unit(unit)
        if item.global_ordinal == proposal_ordinal
    ]
    if len(matches) != 1:
        raise JournalCorrupt("scenario step lacks one frozen proposal slot")
    return matches[0]


def _proposal_slots_for_action(
    unit: WorkUnitView, action: LogicalActionView
) -> tuple[ProposalSlot, ...]:
    """Resolve the engine-owned slots behind one request-local proposal batch."""

    slots = _proposal_slots_for_unit(unit)
    kind = unit.unit["kind"]
    if kind == "map_exploration":
        return slots
    if kind == "portfolio_round":
        strategy_slots = _strategy_slots_for_unit(unit)
        owners = [
            item
            for item in strategy_slots
            if item.scientific_key
            == action.slot["owner_position_scientific_key"]
        ]
        if len(owners) != 1:
            raise JournalCorrupt("portfolio proposal action names no strategy slot")
        start = 3 * owners[0].ordinal
        owned = slots[start : start + 3]
        if len(owned) != 3:
            raise JournalCorrupt("portfolio strategy lane lacks three proposal slots")
        return owned
    if kind in {"strategy_development", "strategy_holdout"}:
        _, _, _, _, proposal = _qualification_root_position(
            unit, _root_plan_for_action(unit, action)
        )
        if proposal is None:
            raise JournalCorrupt("qualification proposal action lacks a frozen slot")
        return (proposal,)
    if kind == "verification" and unit.unit["design"].get(
        "subject_kind"
    ) == "scenario_recipe":
        return (_scenario_proposal_slot_for_action(unit, action),)
    raise JournalCorrupt("proposal action belongs to no canonical collaboration graph")


def _probe_review_mode(unit: WorkUnitView, action: LogicalActionView) -> str:
    kind = unit.unit["kind"]
    if kind == "map_exploration":
        return "ordinary"
    if kind == "portfolio_round":
        return "strategy"
    if kind in {"strategy_development", "strategy_holdout"}:
        return (
            "strategy"
            if _root_plan_for_action(unit, action)["arm"] == "treatment"
            else "ordinary"
        )
    if kind == "verification" and unit.unit["design"].get(
        "subject_kind"
    ) == "scenario_recipe":
        return "strategy"
    raise JournalCorrupt("probe review belongs to no canonical collaboration graph")


def _completion_review_expectations(
    unit: WorkUnitView,
    action: LogicalActionView,
    actions: Mapping[str, LogicalActionView],
    semantic_value: Any,
) -> tuple[ReviewExpectation, ...]:
    """Correlate one review action only with its frozen proposal dependencies."""

    slots_by_id = {
        item.slot_id: item for item in _proposal_slots_for_unit(unit)
    }
    expectations: dict[str, ReviewExpectation] = {}
    for dependency_id in action.slot["dependency_action_ids"]:
        dependency = actions[dependency_id]
        if (
            dependency.slot["purpose"]
            != ActionPurpose.PERTURBATOR_PROPOSAL.value
        ):
            continue
        batch = semantic_value(dependency_id)
        if not isinstance(batch, ProposalBatch):
            continue
        for proposal in batch.proposed_items:
            slot = slots_by_id.get(proposal.slot_id)
            if slot is None:
                raise JournalCorrupt("proposal batch names no frozen review slot")
            review_id = make_review_item_id(
                unit.unit["unit_scientific_key"], "probe", slot.scientific_key
            )
            if review_id in expectations:
                raise JournalCorrupt("probe review has duplicate proposal ownership")
            expectations[review_id] = ReviewExpectation(review_id, proposal)
    return tuple(
        expectations[review_id] for review_id in sorted(expectations)
    )


def _strategy_slots_for_unit(unit: WorkUnitView) -> tuple[StrategySlot, ...]:
    try:
        slots = tuple(
            StrategySlot(
                str(item["entity_id"]),
                str(item["scientific_key"]),
                int(item["ordinal"]),
            )
            for item in unit.unit["design"]["strategy_slots"]
        )
    except (DomainError, KeyError, TypeError, ValueError) as exc:
        raise JournalCorrupt(f"portfolio strategy slots are invalid: {exc}") from exc
    ordered = tuple(sorted(slots, key=lambda item: item.ordinal))
    if [item.ordinal for item in ordered] != list(range(len(ordered))):
        raise JournalCorrupt("portfolio strategy slot ordinals are not contiguous")
    return ordered


def _frozen_assignments(unit: WorkUnitView) -> tuple[ProposalAssignment, ...]:
    try:
        return tuple(
            ProposalAssignment(
                str(item["proposal_slot_scientific_key"]),
                (
                    None
                    if item["strategy_slot_scientific_key"] is None
                    else str(item["strategy_slot_scientific_key"])
                ),
                None if item["lane_ordinal"] is None else int(item["lane_ordinal"]),
                str(item["intent"]),
            )
            for item in unit.unit["design"]["frozen_assignments"]
        )
    except (DomainError, KeyError, TypeError, ValueError) as exc:
        raise JournalCorrupt(
            f"portfolio frozen assignments are invalid: {exc}"
        ) from exc


def _branch_turn_texts(
    *,
    unit: WorkUnitView,
    conversation_roots: Mapping[str, ConversationRootView],
    conversation_nodes: Mapping[str, ConversationNode],
    target_observations: Mapping[str, TargetObservation],
) -> Mapping[str, str]:
    design = unit.unit["design"]
    parent_id = design["generation_parent_node_id"]
    parent_key = design["generation_parent_node_scientific_key"]
    root_id = design["conversation_root_id"]
    node = conversation_nodes.get(parent_id)
    root = conversation_roots.get(root_id)
    if node is None or root is None:
        raise JournalCorrupt("portfolio generation parent is not durable")
    if (
        node.scientific_key != parent_key
        or node.root_id != root_id
        or root.scientific_key != unit.unit["subject_scientific_key"]
    ):
        raise JournalCorrupt("portfolio generation parent identity mutated")
    observations_by_id = {
        observation.observation_id: observation
        for observation in target_observations.values()
    }
    reverse_result: list[tuple[str, str]] = []
    seen: set[str] = set()
    current: ConversationNode | None = node
    while current is not None:
        if current.node_id in seen:
            raise JournalCorrupt("conversation ancestry contains a cycle")
        if current.root_id != root_id:
            raise JournalCorrupt("portfolio conversation ancestry crosses roots")
        seen.add(current.node_id)
        observation = observations_by_id.get(current.target_observation_id)
        if observation is None:
            raise JournalCorrupt("conversation ancestry lacks its target observation")
        if (
            observation.candidate_id != current.candidate_id
            or observation.parent_node_id != current.parent_node_id
        ):
            raise JournalCorrupt("conversation node/observation lineage disagrees")
        reverse_result.append(
            (observation.scientific_key, observation.response)
        )
        ancestor_id = current.parent_node_id
        if ancestor_id is None:
            current = None
            continue
        current = conversation_nodes.get(ancestor_id)
        if current is None:
            raise JournalCorrupt("conversation ancestry names an unknown parent")
    return MappingProxyType(dict(reversed(reverse_result)))


def strategy_evidence_polarities(
    *,
    turn_texts: Mapping[str, str],
    target_observations: Mapping[str, TargetObservation],
    panel_views: Mapping[str, PanelView],
    member_assessments: Mapping[str, MemberAssessment],
) -> Mapping[tuple[str, str], frozenset[str]]:
    observation_by_id = {
        item.observation_id: item for item in target_observations.values()
    }
    polarities: dict[tuple[str, str], set[str]] = {}
    for panel in panel_views.values():
        observation = observation_by_id.get(panel.observation_id)
        if observation is None or observation.scientific_key not in turn_texts:
            continue
        if panel.details is None:
            members = () if panel.primary is None else (panel.primary,)
        else:
            included = frozenset(panel.details.assessment.member_ids)
            members = tuple(
                sorted(
                    (
                        member
                        for member in member_assessments.values()
                        if member.observation_id == observation.observation_id
                        and member.assessment_id in included
                    ),
                    key=lambda member: (
                        MEMBER_ORDER[member.logical_member_slot],
                        member.assessment_id,
                    ),
                )
            )
        exact_lines = frozenset(
            normalize_line_endings(line)
            for line in evidence_lines(
                turn_texts[observation.scientific_key]
            )
        )
        for member in members:
            for excerpt in member.evidence:
                quote = normalize_line_endings(excerpt.quote)
                if quote not in exact_lines:
                    continue
                polarities.setdefault(
                    (observation.scientific_key, quote), set()
                ).add(excerpt.polarity)
        if portfolio_ownership(panel, member_assessments) == "adopted":
            for member in members:
                for quote in member.procedure.premise_evidence:
                    if normalize_line_endings(quote) in exact_lines:
                        polarities.setdefault((observation.scientific_key, normalize_line_endings(quote)), set()).add("adopted_premise")
    return MappingProxyType(
        {key: frozenset(value) for key, value in polarities.items()}
    )


def portfolio_transition(panel: PanelView, members: Mapping[str, MemberAssessment]) -> str:
    """One conservative semantic transition for reporting and adaptive search."""
    if not panel.resolved:
        return "unresolved"
    if panel.assessment is not None and panel.assessment.confirmed_objective_application:
        return "verify_commitment"
    ownership = portfolio_ownership(panel, members)
    included = (() if panel.primary is None else (panel.primary,)) if panel.details is None else tuple(
        member for member in members.values() if member.assessment_id in panel.assessment.member_ids
    )
    required = 1 if panel.details is None else 2
    if ownership == "conditional":
        return "resolve_ownership"
    if ownership != "adopted":
        return "recover_opening"
    procedures = [member.procedure for member in included]
    if any(item.inference in {"rejected", "conditional", "unclear", "described"} or item.reservation != "none" for item in procedures):
        return "repair_inference"
    if sum(item.inference == "endorsed" for item in procedures) < required:
        return "repair_inference"
    if sum(item.commitment == "adopted" for item in procedures) >= required:
        return "apply_commitment"
    return "complete_commitment"


def portfolio_intent_for_node(
    *,
    node_id: str,
    conversation_nodes: Mapping[str, ConversationNode],
    target_observations: Mapping[str, TargetObservation],
    panel_views: Mapping[str, PanelView],
    member_assessments: Mapping[str, MemberAssessment],
) -> PortfolioIntent | None:
    """Derive adaptive-search intent from the node's usable exact evidence."""

    node = conversation_nodes.get(node_id)
    if node is None:
        raise JournalCorrupt("strategy-intent node is missing")
    observation = next(
        (
            item
            for item in target_observations.values()
            if item.observation_id == node.target_observation_id
        ),
        None,
    )
    if observation is None:
        raise JournalCorrupt("strategy-intent node lacks its observation")
    panels = tuple(
        item
        for item in panel_views.values()
        if item.observation_id == observation.observation_id
    )
    if len(panels) != 1:
        raise JournalCorrupt("strategy-intent node does not own one panel")
    if not panels[0].resolved:
        return None
    if panels[0].assessment is not None and panels[0].assessment.confirmed_objective_application:
        return None
    polarities = strategy_evidence_polarities(
        turn_texts={observation.scientific_key: observation.response},
        target_observations=target_observations,
        panel_views=panel_views,
        member_assessments=member_assessments,
    )
    observed = frozenset(
        polarity
        for values in polarities.values()
        for polarity in values
    )
    if "supports" in observed or "adopted_premise" in observed:
        return PortfolioIntent.ADVANCE
    if "resists" in observed:
        return PortfolioIntent.RECOVER
    return None


def portfolio_node_progress(
    state: RunState, node: ConversationNode
) -> tuple[int, Decimal | None]:
    """Project actual adoption/application and the procedure-bounded score."""
    panels = tuple(
        panel for panel in state.panel_views.values()
        if panel.observation_id == node.target_observation_id
    )
    if len(panels) != 1 or not panels[0].resolved:
        return 0, None
    panel = panels[0]
    if panel.assessment is not None and panel.assessment.confirmed_objective_application:
        return 5, panel.effective_score
    ownership = portfolio_ownership(panel, state.member_assessments)
    if ownership != "adopted":
        return (1 if ownership == "conditional" else 0), panel.effective_score
    included = (() if panel.primary is None else (panel.primary,)) if panel.details is None else tuple(
        member for member in state.member_assessments.values()
        if member.assessment_id in panel.assessment.member_ids
    )
    required = 1 if panel.details is None else 2
    # Reservations decide the next repair, not whether an inference or rule
    # has already been adopted. Count a coherent chain within each member.
    endorsed = [m for m in included if m.procedure.inference == "endorsed"]
    if sum(m.procedure.commitment == "adopted" for m in endorsed) >= required:
        return 4, panel.effective_score
    if len(endorsed) >= required:
        return 3, panel.effective_score
    return 2, panel.effective_score


def portfolio_branch_closed(
    state: RunState, node: ConversationNode, *,
    maximum_followups: int, maximum_stalled_followups: int,
) -> bool:
    """Bound linked pursuit using semantic progress; score drift is not progress."""
    progress, _ = portfolio_node_progress(state, node)
    if progress == 5 or node.depth >= maximum_followups:
        return True
    stalled = 0
    current = node
    while current.parent_node_id is not None:
        parent = state.conversation_nodes[current.parent_node_id]
        before_rank, before_score = portfolio_node_progress(state, parent)
        rank, after_score = portfolio_node_progress(state, current)
        if rank > before_rank:
            break
        stalled += 1
        if stalled >= maximum_stalled_followups:
            return True
        current = parent
    return False


def verified_commitment_units(state: RunState, *, before_sequence: int | None = None) -> tuple[WorkUnitView, ...]:
    """Completed fresh blocks meeting the repeat rule at the commitment endpoint."""
    from .evidence import strict_repeat_passes

    panels = {panel.target_action_id: panel for panel in state.panel_views.values()}
    result: list[WorkUnitView] = []
    for unit in state.work_units:
        if unit.unit["kind"] != "verification" or unit.unit_id not in state.completed_work_unit_ids:
            continue
        if before_sequence is not None and any(
            attempt.terminal_event is not None and attempt.terminal_event["sequence"] >= before_sequence
            for action in unit.actions for attempt in action.attempts
        ):
            continue
        roots = unit.unit["design"]["root_plans"]
        if not all(state.execution_evidence[root["execution_id"]].resolved for root in roots):
            continue
        count = sum(
            root["terminal_endpoint_action_id"] in panels
            and panels[root["terminal_endpoint_action_id"]].assessment is not None
            and panels[root["terminal_endpoint_action_id"]].assessment.confirmed_objective_application
            for root in roots
        )
        if strict_repeat_passes(count, len(roots)):
            result.append(unit)
    return tuple(result)


def validate_search_policy(state: RunState) -> None:
    """Reproduce search limits and protected admission from authoritative prefixes."""
    config = state.manifest["normalized_config"]
    search = config.get("search")
    if search is None:
        return
    plan = funded_work_plan(config, state.manifest["bootstrap_snapshot"]["probes"], state.manifest["seed_snapshot"])
    if funded_call_limits(plan) != _budget_limit(state.manifest):
        raise JournalCorrupt("physical call limits disagree with the funded plan")
    started = {kind: 0 for kind in plan}
    for unit in state.work_units:
        kind = unit.unit["kind"]
        count, maximum = plan[kind]
        started[kind] += 1
        if started[kind] > count:
            raise JournalCorrupt("work unit exceeds its funded allocation")
        if not RoleCallVector(**unit.unit["worst_case_calls"]).fits(maximum):
            raise JournalCorrupt("work unit exceeds its funded worst-case call vector")
        if kind == "portfolio_round":
            node = state.conversation_nodes[unit.unit["design"]["generation_parent_node_id"]]
            if portfolio_branch_closed(state, node, maximum_followups=search["maximum_followups"], maximum_stalled_followups=search["maximum_stalled_followups"]):
                raise JournalCorrupt("portfolio started on a completed, stalled, or depth-limited branch")


def _action_effectively_closed(
    unit: WorkUnitView,
    action: LogicalActionView,
    logical_actions: Mapping[str, LogicalActionView],
    target_results: Mapping[str, TargetResultView],
    response_schemas: Mapping[str, Mapping[str, Any]],
    semantic_cache: dict[str, Any],
) -> bool:
    if _completion_action_closed(
        unit,
        action,
        logical_actions,
        target_results,
        response_schemas,
        semantic_cache,
    ):
        return True
    dependencies_closed = all(
        _action_effectively_closed(
            unit,
            logical_actions[dependency],
            logical_actions,
            target_results,
            response_schemas,
            semantic_cache,
        )
        for dependency in action.slot["dependency_action_ids"]
    )
    return bool(
        dependencies_closed
        and not _completion_guard_allows(
            unit,
            action,
            logical_actions,
            target_results,
            response_schemas,
            semantic_cache,
        )
    )


def _parent_strategy_evidence(
    *,
    parent_node_id: str,
    conversation_nodes: Mapping[str, ConversationNode],
    work_units: Sequence[WorkUnitView],
    execution_evidence: Mapping[str, ExecutionEvidence],
    candidate_evidence: Mapping[str, EvidenceRecord],
    panel_views: Mapping[str, PanelView],
    before_sequence: int | None = None,
) -> tuple[Decimal, SearchStatus]:
    generation_parent = conversation_nodes.get(parent_node_id)
    if generation_parent is None:
        raise JournalCorrupt("strategy parent conversation node is missing")
    node = generation_parent
    seen_nodes: set[str] = set()
    screening_units: tuple[WorkUnitView, ...] = ()
    while True:
        if node.node_id in seen_nodes:
            raise JournalCorrupt("strategy parent ancestry contains a cycle")
        if node.root_id != generation_parent.root_id:
            raise JournalCorrupt("strategy parent ancestry crosses conversation roots")
        seen_nodes.add(node.node_id)
        screening_units = tuple(
            unit
            for unit in work_units
            if unit.unit["kind"] == "screening"
            and unit.unit["subject_id"] == node.candidate_id
            and (
                before_sequence is None
                or unit.start_event["sequence"] < before_sequence
            )
        )
        if len(screening_units) > 1:
            raise JournalCorrupt("strategy parent has several screening blocks")
        if screening_units or node.parent_node_id is None:
            break
        parent = conversation_nodes.get(node.parent_node_id)
        if parent is None:
            raise JournalCorrupt("strategy parent ancestry names an unknown node")
        node = parent
    if screening_units:
        unit = screening_units[0]
        execution_ids = (
            unit.unit["design"]["triggering_execution_id"],
            *(item["execution_id"] for item in unit.unit["execution_slots"]),
        )
        try:
            screening = evaluate_screening(
                tuple(execution_evidence[item] for item in execution_ids)
            )
        except (DomainError, KeyError) as exc:
            raise JournalCorrupt(
                f"strategy parent screening evidence is invalid: {exc}"
            ) from exc
        status = screening.status
    else:
        # Reconstruct the status at admission, not the mutable current screen.
        # A later screen may be pending (and temporarily absent from the archive)
        # or completed; neither may rewrite an earlier portfolio's baseline.
        entry_panels = tuple(panel for panel in panel_views.values()
                             if panel.observation_id == node.target_observation_id)
        if len(entry_panels) != 1 or not entry_panels[0].resolved:
            raise JournalCorrupt("strategy parent lacks resolved entry evidence")
        status = SearchStatus.EXPLORATORY_LEAD if entry_panels[0].disputed_high else SearchStatus.OBSERVED
    parent_panels = tuple(
        panel
        for panel in panel_views.values()
        if panel.observation_id == generation_parent.target_observation_id
    )
    if len(parent_panels) != 1 or not parent_panels[0].resolved:
        raise JournalCorrupt("strategy parent lacks one resolved assessment panel")
    baseline = parent_panels[0].effective_score
    if baseline is None:
        raise JournalCorrupt("strategy parent lacks an effective baseline score")
    return baseline, status


def _parsed_selector_portfolio_action(
    action: LogicalActionView,
    response_schemas: Mapping[str, Mapping[str, Any]],
    objective: ObjectivePlugin,
    *,
    turn_texts: Mapping[str, str],
    evidence_polarities: Mapping[tuple[str, str], frozenset[str]],
    maximum_strategies: int,
    current_verdict: Verdict | None,
    current_application: Application | None,
    portfolio_intent: PortfolioIntent,
    current_ownership: str = "unclear",
    current_transition: str | None = None,
) -> Any | None:
    """Parse one Selector action against its frozen branch evidence."""

    def parse(value: object) -> Any:
        _, evidence_by_alias = build_strategy_evidence_catalog(
            tuple(turn_texts.items()), evidence_polarities
        )
        portfolio = parse_selector_portfolio(
            value,
            objective.descriptor_values,
            objective.diagnosis_categories,
            turn_text_by_ref=turn_texts,
            evidence_by_alias=evidence_by_alias,
            maximum_strategies=maximum_strategies,
            diagnosis_normalizer=objective.normalize_diagnosis,
            evidence_polarities=evidence_polarities,
            current_verdict=current_verdict,
            current_application=current_application,
            current_ownership=current_ownership,
            current_transition=current_transition,
            portfolio_intent=portfolio_intent,
        )
        return portfolio

    return _parsed_structured_action(
        action,
        response_schemas,
        parse,
        label="selector portfolio",
    )


def _derive_stateful_portfolios(
    *,
    run_id: str,
    manifest: Mapping[str, Any],
    objective: ObjectivePlugin | None,
    work_units: Sequence[WorkUnitView],
    logical_actions: Mapping[str, LogicalActionView],
    target_results: Mapping[str, TargetResultView],
    target_observations: Mapping[str, TargetObservation],
    panel_views: Mapping[str, PanelView],
    member_assessments: Mapping[str, MemberAssessment],
    execution_evidence: Mapping[str, ExecutionEvidence],
    conversation_roots: Mapping[str, ConversationRootView],
    conversation_nodes: Mapping[str, ConversationNode],
    candidate_evidence: Mapping[str, EvidenceRecord],
    response_schemas: Mapping[str, Mapping[str, Any]],
    remaining: RoleCallVector,
    terminal_event: Mapping[str, Any] | None,
) -> tuple[
    Mapping[str, PortfolioRoundView],
    Mapping[str, StrategyEpisodeView],
    tuple[str, ...],
]:
    portfolio_units = tuple(
        sorted(
            (
                unit
                for unit in work_units
                if unit.unit["kind"] == "portfolio_round"
            ),
            key=lambda item: (item.start_event["sequence"], item.unit_id),
        )
    )
    if not portfolio_units:
        return MappingProxyType({}), MappingProxyType({}), ()
    plugin = _stateful_objective(manifest, objective, portfolio_units)
    try:
        random_seed = manifest["random_seed"]
        proposals_per_step = manifest["normalized_config"]["search"][
            "proposals_per_step"
        ]
        target_profile_digest = digest_json(
            manifest["normalized_config"]["models"]["target"]
        )
    except (KeyError, TypeError) as exc:
        raise JournalCorrupt("stateful manifest configuration is incomplete") from exc
    if (
        isinstance(random_seed, bool)
        or not isinstance(random_seed, int)
        or isinstance(proposals_per_step, bool)
        or not isinstance(proposals_per_step, int)
        or proposals_per_step < 3
    ):
        raise JournalCorrupt("stateful manifest search values are invalid")

    panel_by_target: dict[str, PanelView] = {}
    for panel in panel_views.values():
        if panel.target_action_id in panel_by_target:
            raise JournalCorrupt("one target action owns several panels")
        panel_by_target[panel.target_action_id] = panel
    observation_by_id = {
        observation.observation_id: observation
        for observation in target_observations.values()
    }

    round_views: dict[str, PortfolioRoundView] = {}
    portfolio_by_parent: dict[str, PortfolioRoundView] = {}
    lane_by_strategy_id: dict[str, BoundPortfolioLane] = {}
    identity_by_strategy_key: dict[str, tuple[str, str, str, str]] = {}
    rounds_by_strategy: dict[str, list[StrategyRoundView]] = {}
    global_fingerprint_owner: dict[str, str] = {}

    for unit in portfolio_units:
        design = unit.unit["design"]
        try:
            portfolio_intent = PortfolioIntent(design["portfolio_intent"])
            round_index = design["round_index"]
            proposal_count = design["proposal_count"]
            maximum_strategy_count = design["maximum_strategy_count"]
            parent_node_id = design["generation_parent_node_id"]
            parent_node_key = design["generation_parent_node_scientific_key"]
            conversation_root_id = design["conversation_root_id"]
            conversation_root_key = unit.unit["subject_scientific_key"]
        except (KeyError, TypeError, ValueError) as exc:
            raise JournalCorrupt("portfolio round design is incomplete") from exc
        if (
            round_index != 0
            or proposal_count != proposals_per_step
            or maximum_strategy_count != proposal_count // 3
        ):
            raise JournalCorrupt("portfolio round design violates its fixed ceiling")
        derived_intent = portfolio_intent_for_node(
            node_id=parent_node_id,
            conversation_nodes=conversation_nodes,
            target_observations=target_observations,
            panel_views=panel_views,
            member_assessments=member_assessments,
        )
        if derived_intent is not portfolio_intent:
            raise JournalCorrupt(
                "portfolio intent disagrees with its parent-node evidence"
            )
        proposal_slots = _proposal_slots_for_unit(unit)
        if len(proposal_slots) != proposal_count:
            raise JournalCorrupt("portfolio proposal count differs from frozen slots")
        turn_texts = _branch_turn_texts(
            unit=unit,
            conversation_roots=conversation_roots,
            conversation_nodes=conversation_nodes,
            target_observations=target_observations,
        )
        evidence_polarities = strategy_evidence_polarities(
            turn_texts=turn_texts,
            target_observations=target_observations,
            panel_views=panel_views,
            member_assessments=member_assessments,
        )
        parent_node = conversation_nodes[parent_node_id]
        parent_observation = observation_by_id.get(
            parent_node.target_observation_id
        )
        if parent_observation is None:
            raise JournalCorrupt("portfolio parent lacks its exact observation")
        parent_panel = next(
            (
                item
                for item in panel_views.values()
                if item.observation_id == parent_observation.observation_id
            ),
            None,
        )
        current_assessment = (
            None
            if parent_panel is None
            else portfolio_current_assessment(parent_panel, member_assessments)
        )
        if current_assessment is None:
            current_verdict = None
            current_application = None
        else:
            current_verdict, current_application = current_assessment
        selector_actions = tuple(
            action
            for action in unit.actions
            if action.slot["purpose"] == ActionPurpose.SELECTOR_PORTFOLIO.value
        )
        selector_terminal = True
        if round_index == 0:
            if parent_node_key in portfolio_by_parent:
                raise JournalCorrupt(
                    "one conversation node has several portfolios"
                )
            if len(selector_actions) != 1:
                raise JournalCorrupt("portfolio contract step lacks one Selector action")
            _parent_strategy_evidence(
                parent_node_id=parent_node_id,
                conversation_nodes=conversation_nodes,
                work_units=work_units,
                execution_evidence=execution_evidence,
                candidate_evidence=candidate_evidence,
                panel_views=panel_views,
                before_sequence=unit.start_event.get("sequence"),
            )
            portfolio = None
            if plugin is not None:
                portfolio = _parsed_selector_portfolio_action(
                    selector_actions[0],
                    response_schemas,
                    plugin,
                    turn_texts=turn_texts,
                    evidence_polarities=evidence_polarities,
                    maximum_strategies=maximum_strategy_count,
                    current_verdict=current_verdict,
                    current_application=current_application,
                    current_ownership="unclear" if parent_panel is None else portfolio_ownership(parent_panel, member_assessments),
                    current_transition=None if parent_panel is None else portfolio_transition(parent_panel, member_assessments),
                    portfolio_intent=portfolio_intent,
                )
            selector_terminal = _semantic_action_terminal(
                selector_actions[0], portfolio
            )
            try:
                strategy_slots = _strategy_slots_for_unit(unit)
                if len(strategy_slots) != maximum_strategy_count:
                    raise JournalCorrupt(
                        "portfolio strategy-slot count differs from its ceiling"
                    )
                lanes = (
                    ()
                    if portfolio is None
                    else bind_portfolio_to_slots(
                        portfolio,
                        strategy_slots,
                        parent_node_key,
                    )
                )
                assignments = allocate_contract_step(
                    proposal_slots,
                    lanes,
                )
            except DomainError as exc:
                raise JournalCorrupt(
                    f"portfolio binding/allocation is invalid: {exc}"
                ) from exc
            if design["active_strategy_ids"] or _frozen_assignments(unit):
                raise JournalCorrupt(
                    "portfolio contract step precommitted content-dependent identities"
                )
            for lane in lanes:
                strategy_id = entity_id(
                    "st", [run_id, lane.strategy_scientific_key]
                )
                if strategy_id in lane_by_strategy_id:
                    raise JournalCorrupt("one bound strategy has duplicate contract owners")
                lane_by_strategy_id[strategy_id] = lane
                identity_by_strategy_key[lane.strategy_scientific_key] = (
                    conversation_root_id,
                    conversation_root_key,
                    parent_node_id,
                    parent_node_key,
                )
        else:
            raise JournalCorrupt("a bound strategy contract is limited to one step")

        lane_for_assignment: dict[str, BoundPortfolioLane] = {}
        for lane in lanes:
            acceptable_keys = {
                lane.strategy_slot.scientific_key,
                lane.strategy_scientific_key,
            }
            for assignment in assignments:
                if assignment.strategy_slot_scientific_key in acceptable_keys:
                    lane_for_assignment[
                        assignment.proposal_slot_scientific_key
                    ] = lane
        if len(lane_for_assignment) != len(assignments):
            raise JournalCorrupt("portfolio assignment names no active bound strategy")

        review_actions = tuple(
            action
            for action in unit.actions
            if action.slot["purpose"] == ActionPurpose.PROBE_REVIEW.value
        )
        outcome_actions = tuple(
            action
            for action in unit.actions
            if action.slot["purpose"] == ActionPurpose.OUTCOME_REVIEW.value
        )
        if len(review_actions) != 1 or len(outcome_actions) != 1:
            raise JournalCorrupt("portfolio collaboration graph has wrong review owners")
        first_review = review_actions[0]
        proposal_actions = tuple(
            action
            for action in unit.actions
            if action.slot["purpose"] == ActionPurpose.PERTURBATOR_PROPOSAL.value
        )
        original_actions = proposal_actions
        active_proposal_action_ids: set[str] = set()
        originals: dict[str, ProposalBatch | None] = {}
        original_action_for_lane: dict[str, LogicalActionView] = {}
        original_terminals: list[bool] = []
        assignment_by_key = {
            item.proposal_slot_scientific_key: item for item in assignments
        }
        slot_by_key = {item.scientific_key: item for item in proposal_slots}
        for lane in lanes:
            owner_keys = {
                lane.strategy_slot.scientific_key,
                lane.strategy_scientific_key,
            }
            matching = tuple(
                action
                for action in original_actions
                if action.slot["owner_position_scientific_key"] in owner_keys
            )
            if len(matching) != 1:
                raise JournalCorrupt("bound strategy lacks one original generator")
            action = matching[0]
            active_proposal_action_ids.add(action.action_id)
            original_action_for_lane[lane.strategy_scientific_key] = action
            lane_slots = tuple(
                slot
                for slot in proposal_slots
                if lane_for_assignment.get(slot.scientific_key) is lane
            )
            originals[lane.strategy_scientific_key] = (
                None
                if plugin is None
                else _parsed_structured_action(
                    action,
                    response_schemas,
                    lambda value, slots=lane_slots: parse_proposal_batch(
                        value, slots
                    ),
                    label="strategy proposal",
                )
            )
            original_terminals.append(
                _semantic_action_terminal(
                    action, originals[lane.strategy_scientific_key]
                )
            )
        if any(
            action.attempts and action.action_id not in active_proposal_action_ids
            for action in original_actions
        ):
            raise JournalCorrupt("an unbound strategy lane generated proposals")

        initial_expectations: list[ReviewExpectation] = []
        initial_contracts: dict[str, Mapping[str, str]] = {}
        original_item_by_slot: dict[str, ProposalItem] = {}
        for assignment in assignments:
            slot = slot_by_key[assignment.proposal_slot_scientific_key]
            lane = lane_for_assignment[slot.scientific_key]
            batch = originals.get(lane.strategy_scientific_key)
            if batch is None:
                continue
            item = next(value for value in batch.items if value.slot_id == slot.slot_id)
            original_item_by_slot[slot.scientific_key] = item
            if item.proposed:
                review_id = make_review_item_id(
                    unit.unit["unit_scientific_key"], "probe", slot.scientific_key
                )
                initial_expectations.append(ReviewExpectation(review_id, item))
                initial_contracts[item.slot_id] = dict(
                    lane.item.contract.move.descriptors
                )
        initial_reviews: tuple[ProbeReview, ...] | None
        if not initial_expectations:
            if first_review.attempts:
                raise JournalCorrupt("probe review ran without proposed items")
            initial_reviews = ()
        elif plugin is None:
            initial_reviews = None
        else:
            initial_reviews = _parsed_structured_action(
                first_review,
                response_schemas,
                lambda value: parse_probe_reviews(
                    value,
                    initial_expectations,
                    plugin.descriptor_values,
                    mode="strategy",
                    contract_descriptors_by_slot=initial_contracts,
                ),
                label="strategy probe review",
            )
        initial_review_terminal = (
            True
            if not initial_expectations
            else _semantic_action_terminal(first_review, initial_reviews)
        )
        initial_review_by_id = {
            item.review_item_id: item for item in (initial_reviews or ())
        }
        collaboration_terminal = bool(
            selector_terminal
            and all(original_terminals)
            and initial_review_terminal
        )

        target_by_proposal: dict[str, LogicalActionView] = {}
        for action in unit.actions:
            if action.slot["role"] != Role.TARGET.value:
                continue
            guard = action.slot["guard"]
            if guard.get("kind") != "if_dispatchable_probe":
                raise JournalCorrupt("portfolio target lacks a proposal-slot guard")
            proposal_key = guard["proposal_slot_scientific_key"]
            if proposal_key in target_by_proposal:
                raise JournalCorrupt("one portfolio slot owns several target actions")
            target_by_proposal[proposal_key] = action
        if frozenset(target_by_proposal) != frozenset(slot_by_key):
            raise JournalCorrupt("portfolio targets do not match proposal slots")
        if any(
            action.attempts and proposal_key not in assignment_by_key
            for proposal_key, action in target_by_proposal.items()
        ):
            raise JournalCorrupt("an unbound portfolio slot reached the target")

        preliminary: list[dict[str, Any]] = []
        semantic_cache: dict[str, Any] = {}
        for assignment in assignments:
            slot = slot_by_key[assignment.proposal_slot_scientific_key]
            lane = lane_for_assignment[slot.scientific_key]
            strategy_id = entity_id("st", [run_id, lane.strategy_scientific_key])
            review_id = make_review_item_id(
                unit.unit["unit_scientific_key"], "probe", slot.scientific_key
            )
            original = original_item_by_slot.get(slot.scientific_key)
            original_review = initial_review_by_id.get(review_id)
            selected: SelectedRealization | None = None
            if original is not None and original.proposed and original_review is not None:
                try:
                    selected = select_strategy_realization(original, original_review)
                except DomainError as exc:
                    raise JournalCorrupt(
                        f"strategy realization ownership is invalid: {exc}"
                    ) from exc
            target_action = target_by_proposal[slot.scientific_key]
            if target_action.attempts and selected is None:
                raise JournalCorrupt("target work was credited to an unselected realization")
            if selected is not None and selected.review.status != "aligned":
                raise JournalCorrupt("selected strategy realization is nondispatchable")
            execution_matches = tuple(
                item
                for item in unit.unit["execution_slots"]
                if item["execution_scientific_key"]
                == target_action.slot["owner_position_scientific_key"]
            )
            if len(execution_matches) != 1:
                raise JournalCorrupt("portfolio target lacks one execution slot")
            execution_id = execution_matches[0]["execution_id"]
            panel = panel_by_target.get(target_action.action_id)
            if panel is None:
                raise JournalCorrupt("portfolio target lacks one assessment panel")

            target_attempt_ids = tuple(
                item.physical_attempt_id for item in target_action.attempts
            )
            dispatched_attempt_ids = tuple(
                item.physical_attempt_id
                for item in target_action.attempts
                if item.dispatched
            )
            candidate_id: str | None = None
            dispatch_fingerprint: str | None = None
            child_key = digest_json(
                [
                    unit.unit["unit_scientific_key"],
                    "strategy_child",
                    slot.scientific_key,
                ]
            )
            candidate_key: str | None = None
            if selected is not None:
                prompt = selected.proposal.user_turn or ""
                candidate_key = digest_json(
                    [
                        parent_node_key,
                        "stateful_turn",
                        target_profile_digest,
                        sha256_bytes(
                            normalize_line_endings(prompt).encode("utf-8")
                        ),
                    ]
                )
                expected_candidate_id = entity_id("pc", [run_id, candidate_key])
                expected_messages = [
                    message.to_dict() for message in parent_observation.messages
                ] + [
                    {"role": "assistant", "content": parent_observation.response},
                    {"role": "user", "content": prompt},
                ]
                for attempt in target_action.attempts:
                    messages = attempt.intended["messages"]
                    if messages != expected_messages:
                        raise JournalCorrupt(
                            "stateful target request differs from its exact parent branch"
                        )
                    lineage = attempt.intended["lineage"]
                    expected_lineage = {
                        "candidate_id": expected_candidate_id,
                        "execution_id": execution_id,
                        "conversation_root_id": conversation_root_id,
                        "parent_node_id": parent_node_id,
                        "strategy_id": strategy_id,
                        "proposal_slot_scientific_key": slot.scientific_key,
                        "review_item_id": review_id,
                    }
                    if any(
                        lineage.get(name) != value
                        for name, value in expected_lineage.items()
                    ):
                        raise JournalCorrupt("stateful target lineage mutated")
                    if attempt.dispatched:
                        route = next(
                            (
                                value
                                for value in target_action.slot["routes"]
                                if value["route_key"]
                                == attempt.intended["role_route_key"]
                            ),
                            None,
                        )
                        if route is None:
                            raise JournalCorrupt(
                                "stateful target dispatch uses no frozen route"
                            )
                        if (
                            route["generation_profile_digest"]
                            != target_profile_digest
                        ):
                            raise JournalCorrupt(
                                "stateful target route changed its generation profile"
                            )
                        try:
                            current_fingerprint = strategy_dispatch_fingerprint(
                                messages[:-1],
                                prompt,
                                route["generation_profile_digest"],
                            )
                        except DomainError as exc:
                            raise JournalCorrupt(
                                f"stateful dispatch fingerprint is invalid: {exc}"
                            ) from exc
                        if (
                            dispatch_fingerprint is not None
                            and dispatch_fingerprint != current_fingerprint
                        ):
                            raise JournalCorrupt(
                                "physical retries changed a strategy dispatch fingerprint"
                            )
                        dispatch_fingerprint = current_fingerprint
                candidate_id = (
                    expected_candidate_id
                    if target_action.attempts
                    else None
                )
            target = target_results.get(target_action.action_id)
            if target is not None:
                candidate_id = target.candidate_id
                if candidate_key is None or candidate_id != entity_id(
                    "pc", [run_id, candidate_key]
                ):
                    raise JournalCorrupt("stateful target result has conflicting ownership")
                if target.execution_id != execution_id:
                    raise JournalCorrupt("stateful target result changed execution")
            observation = target_observations.get(target_action.action_id)
            if observation is not None and observation.candidate_id != candidate_id:
                raise JournalCorrupt("stateful observation changed candidate identity")
            panel_actions = [
                logical_actions[item["judge_action_id"]]
                for plan in unit.unit["panel_plans"]
                if plan["panel_id"] == panel.panel_id
                for item in plan["normal_members"]
            ]
            replacement_actions = [
                logical_actions[plan["replacement"]["judge_action_id"]]
                for plan in unit.unit["panel_plans"]
                if plan["panel_id"] == panel.panel_id and plan["replacement"] is not None
            ]
            terminal = (
                collaboration_terminal
                if selected is None
                else all(
                    _action_effectively_closed(
                        unit,
                        item,
                        logical_actions,
                        target_results,
                        response_schemas,
                        semantic_cache,
                    )
                    for item in (
                        target_action,
                        *panel_actions,
                        *replacement_actions,
                    )
                )
            )
            preliminary.append(
                {
                    "slot": slot,
                    "assignment": assignment,
                    "lane": lane,
                    "strategy_id": strategy_id,
                    "review_id": review_id,
                    "original": original,
                    "original_review": original_review,
                    "selected": selected,
                    "target_action": target_action,
                    "execution_id": execution_id,
                    "panel": panel,
                    "target_attempt_ids": target_attempt_ids,
                    "dispatched_attempt_ids": dispatched_attempt_ids,
                    "candidate_id": candidate_id,
                    "candidate_key": candidate_key,
                    "child_key": child_key,
                    "observation": observation,
                    "terminal": terminal,
                    "dispatch_fingerprint": dispatch_fingerprint,
                }
            )

        outcome_action = outcome_actions[0]
        if outcome_action.attempts and not all(item["terminal"] for item in preliminary):
            raise JournalCorrupt(
                "outcome review ran before every strategy child became terminal"
            )
        expected_outcomes: dict[str, tuple[str, str]] = {}
        for item in preliminary:
            selected = item["selected"]
            panel = item["panel"]
            observation = item["observation"]
            if (
                selected is not None
                and selected.review.status == "aligned"
                and panel.resolved
                and observation is not None
            ):
                outcome_id = make_review_item_id(
                    unit.unit["unit_scientific_key"],
                    "outcome",
                    item["slot"].scientific_key,
                )
                expected_outcomes[outcome_id] = (
                    observation.scientific_key,
                    observation.response,
                )
        outcome_reviews: tuple[StrategyOutcomeReview, ...] | None
        if not expected_outcomes:
            if outcome_action.attempts:
                raise JournalCorrupt(
                    "outcome review ran without aligned resolved children"
                )
            outcome_reviews = ()
        else:
            outcome_reviews = _parsed_structured_action(
                outcome_action,
                response_schemas,
                lambda value: parse_outcome_reviews(value, expected_outcomes),
                label="strategy outcome review",
            )
        outcome_terminal = (
            True
            if not expected_outcomes
            else bool(
                all(item["terminal"] for item in preliminary)
                and _semantic_action_terminal(outcome_action, outcome_reviews)
            )
        )
        outcome_by_id = {
            item.review_item_id: item for item in (outcome_reviews or ())
        }

        children: list[PortfolioChildView] = []
        for item in preliminary:
            selected = item["selected"]
            panel = item["panel"]
            observation = item["observation"]
            aligned = bool(
                selected is not None and selected.review.status == "aligned"
            )
            dispatched = bool(item["dispatched_attempt_ids"])
            resolved = bool(dispatched and panel.resolved)
            review = outcome_by_id.get(
                make_review_item_id(
                    unit.unit["unit_scientific_key"],
                    "outcome",
                    item["slot"].scientific_key,
                )
            )
            if review is not None and not (aligned and resolved):
                raise JournalCorrupt(
                    "strategy credit was attached to an unaligned or unresolved child"
                )
            assessment = panel.assessment if resolved else None
            try:
                outcome = StrategyChildOutcome(
                    child_scientific_key=item["child_key"],
                    aligned=aligned,
                    dispatched=dispatched,
                    resolved=resolved,
                    outcome_result=None if review is None else review.result,
                    response_state=None if review is None else review.response_state,
                    effective_score=panel.effective_score if resolved else None,
                    confirmed_objective_high=bool(
                        assessment is not None
                        and assessment.confirmed_objective_high
                    ),
                    confirmed_objective_exceptional=bool(
                        assessment is not None
                        and assessment.confirmed_objective_exceptional
                    ),
                    confirmed_objective_application=bool(
                        assessment is not None
                        and assessment.confirmed_objective_application
                    ),
                    panel_confirmed_declared_application_transition=bool(
                        review is not None
                        and review.result == "supports"
                        and assessment is not None
                        and assessment.confirmed_objective_application
                    ),
                    dispatch_fingerprint=item["dispatch_fingerprint"],
                )
            except DomainError as exc:
                raise JournalCorrupt(
                    f"strategy child evidence is contradictory: {exc}"
                ) from exc
            if outcome.dispatch_fingerprint is not None:
                prior_owner = global_fingerprint_owner.get(outcome.dispatch_fingerprint)
                if (
                    prior_owner is not None
                    and prior_owner != item["target_action"].action_id
                ):
                    raise JournalCorrupt(
                        "duplicate target-dispatch fingerprint across strategy children"
                    )
                global_fingerprint_owner[outcome.dispatch_fingerprint] = item[
                    "target_action"
                ].action_id
            children.append(
                PortfolioChildView(
                    work_unit_id=unit.unit_id,
                    proposal_slot=item["slot"],
                    assignment=item["assignment"],
                    strategy_id=item["strategy_id"],
                    strategy_scientific_key=item["lane"].strategy_scientific_key,
                    original=item["original"],
                    original_review=item["original_review"],
                    selected=selected,
                    alignment_status=(
                        None if selected is None else selected.review.status
                    ),
                    target_action_id=item["target_action"].action_id,
                    execution_id=item["execution_id"],
                    target_attempt_ids=item["target_attempt_ids"],
                    dispatched_attempt_ids=item["dispatched_attempt_ids"],
                    candidate_id=item["candidate_id"],
                    candidate_scientific_key=item["candidate_key"],
                    child_scientific_key=item["child_key"],
                    observation_id=(
                        None if observation is None else observation.observation_id
                    ),
                    observation_scientific_key=(
                        None if observation is None else observation.scientific_key
                    ),
                    panel_id=panel.panel_id,
                    terminal=item["terminal"],
                    outcome_review=review,
                    dispatch_fingerprint=item["dispatch_fingerprint"],
                    outcome=outcome,
                )
            )

        complete = bool(
            collaboration_terminal
            and all(item["terminal"] for item in preliminary)
            and outcome_terminal
        )
        strategy_round_views: list[StrategyRoundView] = []
        for lane in lanes:
            lane_assignments = tuple(
                assignment
                for assignment in assignments
                if lane_for_assignment[assignment.proposal_slot_scientific_key]
                is lane
            )
            lane_children = tuple(
                child
                for child in children
                if child.strategy_scientific_key == lane.strategy_scientific_key
            )
            try:
                evidence = StrategyRoundEvidence(
                    round_index,
                    tuple(child.outcome for child in lane_children),
                )
                refusal = evaluate_refusal_step(evidence)
            except DomainError as exc:
                raise JournalCorrupt(
                    f"strategy round evidence is invalid: {exc}"
                ) from exc
            strategy_id = entity_id("st", [run_id, lane.strategy_scientific_key])
            strategy_round = StrategyRoundView(
                work_unit_id=unit.unit_id,
                work_unit_scientific_key=unit.unit["unit_scientific_key"],
                portfolio_intent=portfolio_intent,
                round_index=round_index,
                strategy_id=strategy_id,
                strategy_scientific_key=lane.strategy_scientific_key,
                lane=lane,
                assignments=lane_assignments,
                children=lane_children,
                evidence=evidence,
                refusal=refusal,
                complete=complete,
            )
            prior_rounds = rounds_by_strategy.setdefault(
                lane.strategy_scientific_key, []
            )
            if len(prior_rounds) != round_index:
                raise JournalCorrupt("strategy round index changed during reduction")
            prior_rounds.append(strategy_round)
            strategy_round_views.append(strategy_round)

        round_view = PortfolioRoundView(
            work_unit_id=unit.unit_id,
            work_unit_scientific_key=unit.unit["unit_scientific_key"],
            portfolio_intent=portfolio_intent,
            conversation_root_id=conversation_root_id,
            conversation_root_scientific_key=conversation_root_key,
            generation_parent_node_id=parent_node_id,
            generation_parent_node_scientific_key=parent_node_key,
            round_index=round_index,
            lanes=tuple(lanes),
            assignments=tuple(assignments),
            strategy_rounds=tuple(strategy_round_views),
            children=tuple(children),
            outcome_reviews=tuple(outcome_reviews or ()),
            complete=complete,
        )
        round_views[unit.unit_id] = round_view
        if round_index == 0:
            portfolio_by_parent[parent_node_key] = round_view

    episodes: dict[str, StrategyEpisodeView] = {}
    for strategy_key, strategy_round_list in sorted(rounds_by_strategy.items()):
        lane = strategy_round_list[0].lane
        identity = identity_by_strategy_key[strategy_key]
        root_id, root_key, parent_id, parent_key = identity
        parent_baseline, parent_search_status = _parent_strategy_evidence(
            parent_node_id=parent_id,
            conversation_nodes=conversation_nodes,
            work_units=work_units,
            execution_evidence=execution_evidence,
            candidate_evidence=candidate_evidence,
            panel_views=panel_views,
        )
        child_views = tuple(
            child for round_view in strategy_round_list for child in round_view.children
        )
        episode_evidence = StrategyEpisodeEvidence(
            strategy_key,
            parent_baseline,
            strategy_round_list[0].complete,
            tuple(child.outcome for child in child_views),
        )
        preference = evaluate_episode_preference(episode_evidence)
        supportive = tuple(
            sorted(
                (
                    child
                    for child in child_views
                    if child.outcome.aligned
                    and child.outcome.resolved
                    and child.outcome.outcome_result == "supports"
                ),
                key=lambda child: (
                    -int(child.outcome.confirmed_objective_application),
                    -int(child.outcome.confirmed_objective_high),
                    -(child.outcome.effective_score or Decimal(0)),
                    child.child_scientific_key,
                ),
            )
        )
        if len(strategy_round_list) != 1:
            raise JournalCorrupt("a strategy contract must contain exactly one step")
        strategy_id = entity_id("st", [run_id, strategy_key])
        episodes[strategy_id] = StrategyEpisodeView(
            strategy_id=strategy_id,
            strategy_scientific_key=strategy_key,
            portfolio_intent=strategy_round_list[0].portfolio_intent,
            conversation_root_id=root_id,
            conversation_root_scientific_key=root_key,
            generation_parent_node_id=parent_id,
            generation_parent_node_scientific_key=parent_key,
            diagnosis=lane.item.diagnosis.category,
            contract=lane.item.contract,
            rounds=tuple(strategy_round_list),
            children=child_views,
            parent_baseline_score=parent_baseline,
            parent_search_status=parent_search_status,
            evidence=episode_evidence,
            preference=preference,
            supporting_children=tuple(
                child.child_scientific_key for child in supportive
            ),
            dispatch_fingerprints=tuple(
                child.dispatch_fingerprint
                for child in child_views
                if child.dispatch_fingerprint is not None
            ),
            refusal=strategy_round_list[0].refusal,
        )

    def support_rank(episode: StrategyEpisodeView) -> tuple[Any, ...]:
        child_by_key = {
            child.child_scientific_key: child for child in episode.children
        }
        strongest = child_by_key[episode.supporting_children[0]].outcome
        return (
            -int(strongest.confirmed_objective_application),
            -int(strongest.confirmed_objective_high),
            -(strongest.effective_score or Decimal(0)),
            episode.strategy_scientific_key,
        )

    supports_ordering = tuple(
        item.strategy_id
        for item in sorted(
            (
                episode
                for episode in episodes.values()
                if episode.supporting_children
            ),
            key=support_rank,
        )
    )
    return (
        MappingProxyType(dict(sorted(round_views.items()))),
        MappingProxyType(dict(sorted(episodes.items()))),
        supports_ordering,
    )


@dataclass(frozen=True, slots=True)
class _QualificationRootProjection:
    outcome: QualificationRootOutcome
    used_attempt_ids: tuple[str, ...]


def _strategy_template_identity(
    run_id: str,
    scientific_scope_digest: str,
    template: StrategyTemplate,
) -> tuple[str, str]:
    template_key = scientific_key(
        [
            scientific_scope_digest,
            "artifact",
            "strategy_template",
            template.content_hash,
        ]
    )
    return template_key, entity_id("tp", [run_id, template_key])


def _strategy_source_entry(
    episode: StrategyEpisodeView,
    conversation_nodes: Mapping[str, ConversationNode],
    candidate_metadata: Mapping[str, Mapping[str, Any]],
) -> tuple[ConversationNode, Mapping[str, Any]]:
    node = conversation_nodes.get(episode.generation_parent_node_id)
    if (
        node is None
        or node.scientific_key
        != episode.generation_parent_node_scientific_key
        or node.root_id != episode.conversation_root_id
    ):
        raise JournalCorrupt("strategy qualification source conversation is missing")
    seen: set[str] = set()
    while node.parent_node_id is not None:
        if node.node_id in seen:
            raise JournalCorrupt("strategy qualification source ancestry contains a cycle")
        seen.add(node.node_id)
        parent = conversation_nodes.get(node.parent_node_id)
        if parent is None or parent.root_id != episode.conversation_root_id:
            raise JournalCorrupt("strategy qualification source ancestry is incomplete")
        node = parent
    candidate = candidate_metadata.get(node.candidate_id)
    if candidate is None:
        raise JournalCorrupt("strategy qualification source entry candidate is unknown")
    return node, candidate


def _episode_has_qualification_trigger(episode: StrategyEpisodeView) -> bool:
    return bool(
        episode.preferred_in_run
        and episode.rounds
        and episode.rounds[0].lane.item.template_candidate is not None
        and any(
            child.outcome.confirmed_objective_exceptional
            or child.outcome.confirmed_objective_application
            for child in episode.children
        )
    )


def _qualification_source_episode(
    unit: WorkUnitView,
    *,
    run_id: str,
    scientific_scope_digest: str,
    episodes: Mapping[str, StrategyEpisodeView],
    work_units_by_id: Mapping[str, WorkUnitView],
) -> StrategyEpisodeView:
    source_unit_id = unit.unit.get("parent_unit_id")
    if unit.unit["kind"] == "strategy_holdout":
        source_development_id = unit.unit["design"]["source_development_unit_id"]
        if source_unit_id != source_development_id:
            raise JournalCorrupt(
                "strategy holdout parent differs from its source development unit"
            )
        source_development = work_units_by_id.get(source_development_id)
        if source_development is None:
            raise JournalCorrupt("strategy holdout source development unit is missing")
        source_unit_id = source_development.unit.get("parent_unit_id")
    if not isinstance(source_unit_id, str):
        raise JournalCorrupt("strategy qualification lacks a source portfolio unit")
    matches: list[StrategyEpisodeView] = []
    for episode in episodes.values():
        if not episode.rounds:
            continue
        template = episode.rounds[0].lane.item.template_candidate
        if template is None:
            continue
        template_key, _ = _strategy_template_identity(
            run_id, scientific_scope_digest, template
        )
        if (
            template_key == unit.unit["subject_scientific_key"]
            and any(item.work_unit_id == source_unit_id for item in episode.rounds)
        ):
            matches.append(episode)
    if not matches:
        raise JournalCorrupt(
            "strategy qualification cannot recover its frozen source episode"
        )
    return min(matches, key=lambda item: item.strategy_scientific_key)


def _qualification_unit_key(
    subject_key: str, kind: str, semantic_design: Mapping[str, Any]
) -> str:
    return scientific_key(
        [subject_key, "unit", kind, 0, digest_json(semantic_design)]
    )


def _qualification_root_position(
    unit: WorkUnitView,
    root_plan: Mapping[str, Any],
) -> tuple[Mapping[str, Any], int, int, str | None, ProposalSlot | None]:
    executions = [
        item
        for item in unit.unit["execution_slots"]
        if item["execution_id"] == root_plan["execution_id"]
    ]
    if len(executions) != 1:
        raise JournalCorrupt("qualification root plan has no unique execution")
    execution = executions[0]
    if execution["root_plan_id"] != root_plan["root_plan_id"]:
        raise JournalCorrupt("qualification root plan/execution identity disagrees")
    root_ordinal = execution["ordinal"]
    arm = root_plan["arm"]
    proposal_ordinal: int | None
    if unit.unit["kind"] == "strategy_development":
        count = unit.unit["design"]["planned_count_per_arm"]
        offsets = {
            "treatment": 0,
            "parent_baseline": count,
            "strategy_agnostic": 2 * count,
        }
        if arm not in offsets:
            raise JournalCorrupt("development root has an invalid arm")
        arm_local = root_ordinal - offsets[arm]
        if not 0 <= arm_local < count:
            raise JournalCorrupt("development root ordinal disagrees with its arm")
        intent = (
            None
            if arm == "parent_baseline"
            else unit.unit["design"]["realization_intent_assignment"][arm_local]
        )
        proposal_ordinal = (
            None
            if arm == "parent_baseline"
            else arm_local if arm == "treatment" else count + arm_local
        )
    else:
        future = [
            item
            for item in unit.unit["design"]["sealed_holdout_plan"]["future_roots"]
            if item["root_ordinal"] == root_ordinal
        ]
        if len(future) != 1 or future[0]["arm"] != arm:
            raise JournalCorrupt("holdout root differs from its sealed future root")
        arm_local = future[0]["arm_local_ordinal"]
        intent = future[0]["realization_intent"]
        proposal_ordinal = root_ordinal
    proposal = None
    if proposal_ordinal is not None:
        matches = [
            ProposalSlot(
                item["slot_id"], item["scientific_key"], item["global_ordinal"]
            )
            for item in unit.unit["proposal_slots"]
            if item["global_ordinal"] == proposal_ordinal
        ]
        if len(matches) != 1:
            raise JournalCorrupt("qualification root lacks one proposal slot")
        proposal = matches[0]
    return execution, root_ordinal, arm_local, intent, proposal


def _qualification_root_actions(
    root_plan: Mapping[str, Any],
    logical_actions: Mapping[str, LogicalActionView],
) -> tuple[LogicalActionView, ...]:
    try:
        actions = tuple(
            logical_actions[action_id]
            for action_id in root_plan["ordered_action_ids"]
        )
    except KeyError as exc:
        raise JournalCorrupt("qualification root cites an unknown action") from exc
    if tuple(action.action_id for action in sorted(actions, key=lambda item: item.slot["ordinal"])) != tuple(
        root_plan["ordered_action_ids"]
    ):
        raise JournalCorrupt("qualification root action order is not canonical")
    return actions


def _validate_qualification_root_graphs(
    unit: WorkUnitView,
    *,
    run_id: str,
    manifest: Mapping[str, Any],
    logical_actions: Mapping[str, LogicalActionView],
) -> None:
    design = unit.unit["design"]
    plans = tuple(design["root_plans"])
    expected_count = (
        3 * design["planned_count_per_arm"]
        if unit.unit["kind"] == "strategy_development"
        else 2 * design["sealed_holdout_plan"]["planned_count_per_arm"]
    )
    if len(plans) != expected_count or len(unit.unit["execution_slots"]) != expected_count:
        raise JournalCorrupt("qualification root allocation has the wrong arm size")
    if len({item["root_plan_id"] for item in plans}) != len(plans):
        raise JournalCorrupt("qualification root plan identity is duplicated")
    claimed_actions: set[str] = set()
    for plan in plans:
        execution, root_ordinal, _, _, proposal = _qualification_root_position(
            unit, plan
        )
        expected_plan_id = entity_id(
            "rp",
            [run_id, unit.unit["unit_scientific_key"], "root_plan", root_ordinal],
        )
        if plan["root_plan_id"] != expected_plan_id:
            raise JournalCorrupt("qualification root plan ID is not canonical")
        continuation = [] if proposal is None else [proposal.scientific_key]
        execution_key = scientific_key(
            [
                unit.unit["unit_scientific_key"],
                "execution",
                root_ordinal,
                continuation,
            ]
        )
        if (
            execution["execution_scientific_key"] != execution_key
            or execution["execution_id"] != entity_id("ex", [run_id, execution_key])
            or execution["arm"] != plan["arm"]
        ):
            raise JournalCorrupt("qualification execution identity is not canonical")
        actions = _qualification_root_actions(plan, logical_actions)
        if claimed_actions & {item.action_id for item in actions}:
            raise JournalCorrupt("qualification action belongs to several roots")
        claimed_actions.update(item.action_id for item in actions)
        for action in actions:
            if (
                action.unit_id != unit.unit_id
                or action.slot["owner_position_scientific_key"] != execution_key
                or action.slot["action_scientific_key"]
                != scientific_key(
                    [
                        execution_key,
                        "action",
                        action.slot["role"],
                        action.slot["purpose"],
                        action.slot["ordinal"],
                    ]
                )
                or action.action_id
                != entity_id("la", [run_id, action.slot["action_scientific_key"]])
            ):
                raise JournalCorrupt("qualification logical action identity is not canonical")
        purposes = tuple(action.slot["purpose"] for action in actions)
        panel_prefix = (
            ActionPurpose.TARGET_SAMPLE.value,
            *(ActionPurpose.JUDGE_MEMBER.value for _ in configured_judge_slots(manifest)),
        )
        if plan["arm"] == "parent_baseline":
            expected_purposes = panel_prefix
            expected_endpoint = actions[0].action_id if actions else None
            expected_reasons: list[str] = []
        else:
            middle = (
                *((ActionPurpose.TEMPLATE_BINDING.value,) if plan["arm"] == "treatment" else ()),
                ActionPurpose.PERTURBATOR_PROPOSAL.value,
                ActionPurpose.PROBE_REVIEW.value,
                ActionPurpose.PERTURBATOR_PROPOSAL.value,
                ActionPurpose.PROBE_REVIEW.value,
            )
            suffix = (
                *panel_prefix,
                *((ActionPurpose.OUTCOME_REVIEW.value,) if plan["arm"] == "treatment" else ()),
            )
            expected_purposes = (*panel_prefix, *middle, *suffix)
            terminal_targets = [
                action
                for action in actions
                if action.slot["purpose"] == ActionPurpose.TARGET_SAMPLE.value
            ]
            expected_endpoint = (
                terminal_targets[-1].action_id if terminal_targets else None
            )
            expected_reasons = ["inapplicable"] if plan["arm"] == "treatment" else []
        if (
            purposes != expected_purposes
            or plan["terminal_endpoint_action_id"] != expected_endpoint
            or plan["semantic_nonsuccess_reasons"] != expected_reasons
        ):
            raise JournalCorrupt("qualification root action blueprint is not canonical")
    if claimed_actions != {action.action_id for action in unit.actions}:
        raise JournalCorrupt("qualification roots do not partition their action graph")


def _validate_strategy_development_design(
    unit: WorkUnitView,
    *,
    run_id: str,
    scientific_scope_digest: str,
    manifest: Mapping[str, Any],
    objective: ObjectivePlugin,
    template: StrategyTemplate,
    logical_actions: Mapping[str, LogicalActionView],
) -> None:
    design = unit.unit["design"]
    template_key, template_id = _strategy_template_identity(
        run_id, scientific_scope_digest, template
    )
    if (
        unit.unit["subject_id"] != template_id
        or unit.unit["subject_scientific_key"] != template_key
        or design["strategy_template_id"] != template_id
        or design["strategy_template_scientific_key"] != template_key
    ):
        raise JournalCorrupt("strategy development changed its template identity")
    count = design["planned_count_per_arm"]
    expected_dimensions = derive_template_comparison_dimensions(template)
    if not expected_dimensions or tuple(design["comparison_dimensions"]) != expected_dimensions:
        raise JournalCorrupt("strategy development comparison dimensions are not canonical")
    if design["measurement_cell"] != descriptors_dict(template.move.descriptors):
        raise JournalCorrupt("strategy development changed its measurement cell")
    expected_intents = freeze_qualification_intent_assignment(
        count,
        random_seed=manifest["random_seed"],
        qualification_subject_scientific_key=template_key,
        phase="development",
    )
    if tuple(design["realization_intent_assignment"]) != expected_intents:
        raise JournalCorrupt("strategy development intent assignment is not canonical")
    positions = freeze_development_arm_order(
        count,
        random_seed=manifest["random_seed"],
        qualification_subject_scientific_key=template_key,
    )
    plan_by_position: dict[tuple[str, int], Mapping[str, Any]] = {}
    for plan in design["root_plans"]:
        _, _, local, _, _ = _qualification_root_position(unit, plan)
        key = (plan["arm"], local)
        if key in plan_by_position:
            raise JournalCorrupt("strategy development root position is duplicated")
        plan_by_position[key] = plan
    expected_interleave = [
        plan_by_position[(item.arm, item.arm_local_ordinal)]["root_plan_id"]
        for item in positions
    ]
    if design["interleaved_root_order"] != expected_interleave:
        raise JournalCorrupt("strategy development arm order is not canonical")
    try:
        sealed = parse_sealed_holdout_plan(
            design["sealed_holdout_plan"], objective.descriptor_values
        )
        expected_seal = build_sealed_holdout_plan(
            template,
            planned_count_per_arm=manifest["normalized_config"]["samples"]["holdout"],
            random_seed=manifest["random_seed"],
            qualification_subject_scientific_key=template_key,
            measurement_cell=design["measurement_cell"],
            descriptor_vocabulary=objective.descriptor_values,
        )
    except (KeyError, DomainError) as exc:
        raise JournalCorrupt(f"strategy development holdout seal is invalid: {exc}") from exc
    if (
        sealed != expected_seal
        or design["sealed_holdout_digest"] != sealed.content_digest
    ):
        raise JournalCorrupt("strategy development holdout seal is not canonical")
    semantic = {
        "kind": "strategy_development",
        "strategy_template": template.to_dict(),
        "planned_count_per_arm": count,
        "comparison_dimensions": list(expected_dimensions),
        "realization_intent_assignment": list(expected_intents),
        "interleaved_root_order": [
            [item.arm, item.arm_local_ordinal] for item in positions
        ],
        "root_blueprints": [
            "treatment_qualification",
            "parent_baseline",
            "strategy_agnostic_qualification",
        ],
        "measurement_cell": design["measurement_cell"],
        "sealed_holdout_plan": sealed.to_dict(),
    }
    expected_unit_key = _qualification_unit_key(
        template_key, "strategy_development", semantic
    )
    if (
        unit.unit["unit_scientific_key"] != expected_unit_key
        or unit.unit_id != entity_id("wu", [run_id, expected_unit_key])
        or unit.unit["policy_digest"] != digest_json(manifest["decision_constants"])
        or tuple(
            ProposalSlot(item["slot_id"], item["scientific_key"], item["global_ordinal"])
            for item in unit.unit["proposal_slots"]
        )
        != make_proposal_slots(expected_unit_key, 2 * count)
    ):
        raise JournalCorrupt("strategy development work-unit identity is not canonical")
    _validate_qualification_root_graphs(
        unit, run_id=run_id, manifest=manifest, logical_actions=logical_actions
    )


def _validate_strategy_holdout_design(
    unit: WorkUnitView,
    development: WorkUnitView,
    *,
    run_id: str,
    scientific_scope_digest: str,
    manifest: Mapping[str, Any],
    objective: ObjectivePlugin,
    template: StrategyTemplate,
    logical_actions: Mapping[str, LogicalActionView],
) -> None:
    design = unit.unit["design"]
    source_design = development.unit["design"]
    template_key, template_id = _strategy_template_identity(
        run_id, scientific_scope_digest, template
    )
    if (
        unit.unit.get("parent_unit_id") != development.unit_id
        or design["source_development_unit_id"] != development.unit_id
        or unit.unit["subject_id"] != template_id
        or unit.unit["subject_scientific_key"] != template_key
        or design["strategy_template_id"] != template_id
        or design["strategy_template_scientific_key"] != template_key
        or design["source_sealed_holdout_digest"]
        != source_design["sealed_holdout_digest"]
        or design["sealed_holdout_plan"] != source_design["sealed_holdout_plan"]
    ):
        raise JournalCorrupt("strategy holdout changed its development-time seal")
    try:
        sealed = parse_sealed_holdout_plan(
            design["sealed_holdout_plan"], objective.descriptor_values
        )
    except DomainError as exc:
        raise JournalCorrupt(f"strategy holdout seal is invalid: {exc}") from exc
    if sealed.content_digest != design["source_sealed_holdout_digest"]:
        raise JournalCorrupt("strategy holdout seal digest is inconsistent")
    semantic = {
        "kind": "strategy_holdout",
        "strategy_template": template.to_dict(),
        "sealed_holdout_plan": sealed.to_dict(),
    }
    expected_unit_key = _qualification_unit_key(
        template_key, "strategy_holdout", semantic
    )
    count = sealed.planned_count_per_arm
    if (
        unit.unit["unit_scientific_key"] != expected_unit_key
        or unit.unit_id != entity_id("wu", [run_id, expected_unit_key])
        or unit.unit["policy_digest"] != digest_json(manifest["decision_constants"])
        or tuple(
            ProposalSlot(item["slot_id"], item["scientific_key"], item["global_ordinal"])
            for item in unit.unit["proposal_slots"]
        )
        != make_proposal_slots(expected_unit_key, 2 * count)
    ):
        raise JournalCorrupt("strategy holdout work-unit identity is not canonical")
    future_by_ordinal = {
        item["root_ordinal"]: item
        for item in design["sealed_holdout_plan"]["future_roots"]
    }
    plans_by_ordinal = {
        _qualification_root_position(unit, plan)[1]: plan
        for plan in design["root_plans"]
    }
    expected_order = [
        plans_by_ordinal[ordinal]["root_plan_id"]
        for ordinal in design["sealed_holdout_plan"]["interleaved_root_order"]
    ]
    actual_action_order = [
        plan["root_plan_id"]
        for plan in sorted(
            design["root_plans"],
            key=lambda item: min(
                logical_actions[action_id].slot["ordinal"]
                for action_id in item["ordered_action_ids"]
            ),
        )
    ]
    if actual_action_order != expected_order or set(future_by_ordinal) != set(plans_by_ordinal):
        raise JournalCorrupt("strategy holdout arm materialization is not canonical")
    _validate_qualification_root_graphs(
        unit, run_id=run_id, manifest=manifest, logical_actions=logical_actions
    )


def _parsed_qualification_action(
    action: LogicalActionView,
    response_schemas: Mapping[str, Mapping[str, Any]],
    parser: Any,
    *,
    label: str,
) -> tuple[Any | None, AttemptView | None]:
    usable: list[tuple[Any, AttemptView, int]] = []
    attempt_indexes = {
        attempt.physical_attempt_id: index
        for index, attempt in enumerate(action.attempts)
    }
    for attempt in action.successful_attempts:
        if attempt.terminal_event is None or not _success_passes_base_structure(
            action, attempt, response_schemas
        ):
            continue
        try:
            value = strict_json_loads(attempt.terminal_event["payload"]["text"])
            usable.append(
                (
                    parser(value),
                    attempt,
                    attempt_indexes[attempt.physical_attempt_id],
                )
            )
        except DomainError:
            continue
    if len(usable) > 1:
        raise JournalCorrupt(
            f"one {label} action has several semantically usable results"
        )
    if usable and usable[0][2] != len(action.attempts) - 1:
        raise JournalCorrupt(
            f"one {label} action continued after a semantically usable result"
        )
    return (usable[0][0], usable[0][1]) if usable else (None, None)


def _qualification_measurement_cell(
    unit: WorkUnitView,
) -> Mapping[str, str]:
    value = (
        unit.unit["design"]["measurement_cell"]
        if unit.unit["kind"] == "strategy_development"
        else unit.unit["design"]["sealed_holdout_plan"]["measurement_cell"]
    )
    if not isinstance(value, Mapping):
        raise JournalCorrupt("qualification measurement cell is invalid")
    return value


def _qualification_comparison_dimensions(unit: WorkUnitView) -> tuple[str, ...]:
    value = (
        unit.unit["design"]["comparison_dimensions"]
        if unit.unit["kind"] == "strategy_development"
        else unit.unit["design"]["sealed_holdout_plan"]["comparison_dimensions"]
    )
    return tuple(value)


def _qualification_panel_for_target(
    unit: WorkUnitView,
    target_action_id: str,
    panel_views: Mapping[str, PanelView],
) -> PanelView:
    plans = [
        item
        for item in unit.unit["panel_plans"]
        if item["target_action_id"] == target_action_id
    ]
    if len(plans) != 1 or plans[0]["trigger_policy"] != "always_full":
        raise JournalCorrupt("qualification target lacks one full assessment panel")
    panel = panel_views.get(plans[0]["panel_id"])
    if panel is None:
        raise JournalCorrupt("qualification target panel is not reduced")
    return panel


def _qualification_proposal(
    action: LogicalActionView,
    *,
    unit: WorkUnitView,
    proposal_slot: ProposalSlot,
    intent: str,
    arm: str,
    objective: ObjectivePlugin,
    response_schemas: Mapping[str, Mapping[str, Any]],
) -> tuple[ProposalItem | None, AttemptView | None, bool]:
    def parse(value: object) -> ProposalBatch:
        return parse_proposal_batch(value, (proposal_slot,))

    batch, attempt = _parsed_qualification_action(
        action,
        response_schemas,
        parse,
        label="qualification proposal",
    )
    terminal = _semantic_action_terminal(action, batch)
    if not isinstance(batch, ProposalBatch):
        return None, attempt, terminal
    return batch.items[0], attempt, terminal


def _qualification_probe_review(
    action: LogicalActionView,
    *,
    unit: WorkUnitView,
    proposal_slot: ProposalSlot,
    proposal: ProposalItem | None,
    arm: str,
    bound_contract: BoundStrategy | None,
    objective: ObjectivePlugin,
    response_schemas: Mapping[str, Mapping[str, Any]],
) -> tuple[ProbeReview | None, AttemptView | None, bool]:
    if proposal is None or not proposal.proposed:
        if action.attempts:
            raise JournalCorrupt("qualification review ran without a proposed item")
        return None, None, True
    review_id = make_review_item_id(
        unit.unit["unit_scientific_key"], "probe", proposal_slot.scientific_key
    )
    expectation = ReviewExpectation(review_id, proposal)
    mode = "strategy" if arm == "treatment" else "ordinary"
    contracts: Mapping[str, Any] | None = None
    if arm == "treatment":
        if bound_contract is None:
            if action.attempts:
                raise JournalCorrupt(
                    "qualification strategy review ran without a grounded binding"
                )
            return None, None, True
        contracts = {proposal_slot.slot_id: bound_contract.move.descriptors}

    def parse(value: object) -> tuple[ProbeReview, ...]:
        return parse_probe_reviews(
            value,
            (expectation,),
            objective.descriptor_values,
            mode=mode,
            contract_descriptors_by_slot=contracts,
        )

    reviews, attempt = _parsed_qualification_action(
        action,
        response_schemas,
        parse,
        label="qualification probe review",
    )
    terminal = _semantic_action_terminal(action, reviews)
    if not isinstance(reviews, tuple):
        return None, attempt, terminal
    return reviews[0], attempt, terminal


def _derive_qualification_root(
    unit: WorkUnitView,
    root_plan: Mapping[str, Any],
    *,
    template: StrategyTemplate,
    objective: ObjectivePlugin,
    logical_actions: Mapping[str, LogicalActionView],
    target_results: Mapping[str, TargetResultView],
    target_observations: Mapping[str, TargetObservation],
    panel_views: Mapping[str, PanelView],
    response_schemas: Mapping[str, Mapping[str, Any]],
) -> _QualificationRootProjection:
    execution, _, arm_local, intent, proposal_slot = _qualification_root_position(
        unit, root_plan
    )
    actions = _qualification_root_actions(root_plan, logical_actions)
    target_actions = tuple(
        item for item in actions if item.slot["role"] == Role.TARGET.value
    )
    if len(target_actions) != (1 if root_plan["arm"] == "parent_baseline" else 2):
        raise JournalCorrupt("qualification root target endpoint count is invalid")
    entry_action = target_actions[0]
    entry_panel = _qualification_panel_for_target(
        unit, entry_action.action_id, panel_views
    )
    used_attempt_ids: list[str] = []
    semantic_cache: dict[str, Any] = {}
    root_terminal = all(
        _action_effectively_closed(
            unit,
            action,
            logical_actions,
            target_results,
            response_schemas,
            semantic_cache,
        )
        for action in actions
    )

    if root_plan["arm"] == "parent_baseline":
        resolved = bool(root_terminal and entry_panel.resolved)
        assessment = entry_panel.assessment if resolved else None
        try:
            outcome = QualificationRootOutcome(
                execution["execution_scientific_key"],
                "parent_baseline",
                arm_local,
                None,
                root_terminal,
                resolved,
                None,
                bool(assessment and assessment.confirmed_objective_high),
                False,
                False,
                entry_panel.effective_score if resolved else None,
            )
        except DomainError as exc:
            raise JournalCorrupt(f"parent-baseline root is invalid: {exc}") from exc
        return _QualificationRootProjection(outcome, ())

    if proposal_slot is None or intent is None:
        raise JournalCorrupt("qualification continuation root lacks slot or intent")
    binding = None
    bound_contract: BoundStrategy | None = None
    template_applicable: bool | None = None
    if root_plan["arm"] == "treatment":
        binding_actions = tuple(
            item
            for item in actions
            if item.slot["purpose"] == ActionPurpose.TEMPLATE_BINDING.value
        )
        if len(binding_actions) != 1:
            raise JournalCorrupt("treatment root lacks one template binding action")
        binding_action = binding_actions[0]
        entry_observation = target_observations.get(entry_action.action_id)
        if entry_observation is not None:
            binding_id = make_template_binding_item_id(
                unit.unit["unit_scientific_key"],
                execution["execution_scientific_key"],
                unit.unit["design"]["strategy_template_scientific_key"],
                0,
            )
            binding, binding_attempt = _parsed_qualification_action(
                binding_action,
                response_schemas,
                lambda value: parse_template_binding(
                    value,
                    expected_binding_item_id=binding_id,
                    template=template,
                    descriptor_vocabulary=objective.descriptor_values,
                    fresh_turn_text_by_ref={
                        entry_observation.scientific_key: entry_observation.response
                    },
                ),
                label="qualification template binding",
            )
            if binding_attempt is not None:
                used_attempt_ids.append(binding_attempt.physical_attempt_id)
        if binding is not None:
            template_applicable = binding.status == "applicable"
            bound_contract = binding.contract
        if template_applicable is not True:
            later_attempts = [
                attempt
                for action in actions
                if action.slot["ordinal"] > binding_action.slot["ordinal"]
                for attempt in action.attempts
            ]
            if later_attempts:
                raise JournalCorrupt(
                    "treatment continuation ran without an applicable strict binding"
                )
            resolved = bool(
                root_terminal
                and template_applicable is False
                and entry_panel.resolved
            )
            try:
                outcome = QualificationRootOutcome(
                    execution["execution_scientific_key"],
                    "treatment",
                    arm_local,
                    intent,
                    root_terminal,
                    resolved,
                    template_applicable,
                    False,
                    False,
                    False,
                    entry_panel.effective_score if resolved else None,
                )
            except DomainError as exc:
                raise JournalCorrupt(
                    f"inapplicable treatment root is invalid: {exc}"
                ) from exc
            return _QualificationRootProjection(
                outcome, tuple(sorted(used_attempt_ids))
            )

    proposal_actions = tuple(
        item
        for item in actions
        if item.slot["purpose"] == ActionPurpose.PERTURBATOR_PROPOSAL.value
    )
    review_actions = tuple(
        item
        for item in actions
        if item.slot["purpose"] == ActionPurpose.PROBE_REVIEW.value
    )
    if len(proposal_actions) != 2 or len(review_actions) != 2:
        raise JournalCorrupt("qualification continuation collaboration is incomplete")
    original, original_attempt, _ = _qualification_proposal(
        proposal_actions[0],
        unit=unit,
        proposal_slot=proposal_slot,
        intent=intent,
        arm=root_plan["arm"],
        objective=objective,
        response_schemas=response_schemas,
    )
    if original_attempt is not None:
        used_attempt_ids.append(original_attempt.physical_attempt_id)
    original_review, review_attempt, _ = _qualification_probe_review(
        review_actions[0],
        unit=unit,
        proposal_slot=proposal_slot,
        proposal=original,
        arm=root_plan["arm"],
        bound_contract=bound_contract,
        objective=objective,
        response_schemas=response_schemas,
    )
    if review_attempt is not None:
        used_attempt_ids.append(review_attempt.physical_attempt_id)
    repairable_statuses = {"invalid"}
    corrective: ProposalItem | None = None
    corrective_review: ProbeReview | None = None
    if original_review is not None and original_review.status in repairable_statuses:
        corrective, corrective_attempt, _ = _qualification_proposal(
            proposal_actions[1],
            unit=unit,
            proposal_slot=proposal_slot,
            intent=intent,
            arm=root_plan["arm"],
            objective=objective,
            response_schemas=response_schemas,
        )
        if corrective_attempt is not None:
            used_attempt_ids.append(corrective_attempt.physical_attempt_id)
        corrective_review, rereview_attempt, _ = _qualification_probe_review(
            review_actions[1],
            unit=unit,
            proposal_slot=proposal_slot,
            proposal=corrective,
            arm=root_plan["arm"],
            bound_contract=bound_contract,
            objective=objective,
            response_schemas=response_schemas,
        )
        if rereview_attempt is not None:
            used_attempt_ids.append(rereview_attempt.physical_attempt_id)
    elif proposal_actions[1].attempts or review_actions[1].attempts:
        raise JournalCorrupt("qualification correction ran without a repairable review")

    selected: SelectedRealization | None = None
    if original is not None and original.proposed and original_review is not None:
        try:
            if root_plan["arm"] == "treatment":
                selected = select_strategy_realization(
                    original,
                    original_review,
                    corrective if corrective_review is not None else None,
                    corrective_review,
                )
                if selected is not None and selected.review.status != "aligned":
                    selected = None
            else:
                selected = select_strategy_agnostic_realization(
                    original,
                    original_review,
                    corrective if corrective_review is not None else None,
                    corrective_review,
                )
        except DomainError as exc:
            raise JournalCorrupt(
                f"qualification realization ownership is invalid: {exc}"
            ) from exc

    terminal_action = target_actions[-1]
    if terminal_action.attempts and selected is None:
        raise JournalCorrupt(
            "qualification target dispatched without a semantically admissible realization"
        )
    terminal_panel = _qualification_panel_for_target(
        unit, terminal_action.action_id, panel_views
    )
    terminal_observation = target_observations.get(terminal_action.action_id)
    review = None
    if root_plan["arm"] == "treatment":
        outcome_actions = tuple(
            item
            for item in actions
            if item.slot["purpose"] == ActionPurpose.OUTCOME_REVIEW.value
        )
        if len(outcome_actions) != 1:
            raise JournalCorrupt("treatment root lacks one outcome review")
        outcome_action = outcome_actions[0]
        if (
            selected is not None
            and terminal_observation is not None
            and terminal_panel.resolved
        ):
            review_id = make_review_item_id(
                unit.unit["unit_scientific_key"],
                "outcome",
                proposal_slot.scientific_key,
            )
            reviews, outcome_attempt = _parsed_qualification_action(
                outcome_action,
                response_schemas,
                lambda value: parse_outcome_reviews(
                    value,
                    {
                        review_id: (
                            terminal_observation.scientific_key,
                            terminal_observation.response,
                        )
                    },
                ),
                label="qualification outcome review",
            )
            if outcome_attempt is not None:
                used_attempt_ids.append(outcome_attempt.physical_attempt_id)
            if isinstance(reviews, tuple):
                review = reviews[0]
        elif outcome_action.attempts:
            raise JournalCorrupt(
                "qualification outcome review ran without an aligned resolved response"
            )

    resolved = bool(
        root_terminal
        and selected is not None
        and terminal_panel.resolved
        and terminal_observation is not None
        and (root_plan["arm"] != "treatment" or review is not None)
    )
    assessment = terminal_panel.assessment if resolved else None
    objective_success = bool(
        assessment is not None and assessment.confirmed_objective_high
    )
    fidelity = bool(
        root_plan["arm"] == "treatment"
        and objective_success
        and review is not None
        and review.result == "supports"
    )
    exceptional = bool(
        fidelity
        and assessment is not None
        and (
            assessment.confirmed_objective_exceptional
            or assessment.confirmed_objective_application
        )
    )
    try:
        outcome = QualificationRootOutcome(
            execution["execution_scientific_key"],
            root_plan["arm"],
            arm_local,
            intent,
            root_terminal,
            resolved,
            template_applicable,
            objective_success,
            fidelity,
            exceptional,
            terminal_panel.effective_score if resolved else None,
            fidelity,
        )
    except DomainError as exc:
        raise JournalCorrupt(f"qualification root outcome is invalid: {exc}") from exc
    return _QualificationRootProjection(
        outcome, tuple(sorted(set(used_attempt_ids)))
    )


def _measurement_cell_verified_before(
    development: WorkUnitView,
    measurement_cell: Mapping[str, Any],
    *,
    work_units: Sequence[WorkUnitView],
    candidate_metadata: Mapping[str, Mapping[str, Any]],
    candidate_evidence: Mapping[str, EvidenceRecord],
    scenarios: Mapping[str, ScenarioRecipeView],
    scenario_evidence: Mapping[str, EvidenceRecord],
) -> bool:
    freeze_sequence = development.start_event["sequence"]
    for unit in work_units:
        if (
            _work_unit_evidence_cutoff(unit) > freeze_sequence
            or unit.unit["kind"] != "verification"
        ):
            continue
        if unit.unit["design"].get("subject_kind") == "scenario_recipe":
            scenario = scenarios.get(unit.unit["subject_scientific_key"])
            evidence = (
                None
                if scenario is None
                else scenario_evidence.get(scenario.scenario_id)
            )
            if (
                scenario is not None
                and scenario.scenario_id == unit.unit["subject_id"]
                and descriptors_dict(scenario.recipe.measurement_cell)
                == dict(measurement_cell)
                and evidence is not None
                and evidence.verification_status is VerificationStatus.VERIFIED
            ):
                return True
            continue
        candidate = candidate_metadata.get(unit.unit["subject_id"])
        evidence = candidate_evidence.get(unit.unit["subject_id"])
        if (
            candidate is not None
            and candidate.get("descriptors") == dict(measurement_cell)
            and evidence is not None
            and evidence.verification_status is VerificationStatus.VERIFIED
        ):
            return True
    return False


def _objective_scope_key(manifest: Mapping[str, Any]) -> str:
    try:
        artifacts = manifest["objective"]["artifact_digests"]
        if not isinstance(artifacts, Mapping):
            raise TypeError("artifact digests are not an object")
        return digest_json(
            [
                "objective_scope",
                manifest["objective"]["id"],
                manifest["objective"]["plugin_digest"],
                [[name, artifacts[name]] for name in sorted(artifacts)],
                manifest["response_schema_digests"]["judge_output"],
            ]
        )
    except (KeyError, TypeError) as exc:
        raise JournalCorrupt(f"objective scope key is invalid: {exc}") from exc


def _component_scope_for_attempt(
    manifest: Mapping[str, Any],
    action: LogicalActionView,
    attempt: AttemptView,
) -> StructuredRoleScope | None:
    route_bindings = [
        item
        for item in action.slot["routes"]
        if item["route_key"] == attempt.intended["role_route_key"]
    ]
    if len(route_bindings) != 1:
        raise JournalCorrupt("qualification evidence uses an unknown route binding")
    binding = route_bindings[0]
    route = _static_route(
        manifest, action.slot["role"], binding["configured_slot"]
    )
    if route.route_key != binding["route_key"]:
        raise JournalCorrupt("qualification evidence route is not canonical")
    mode = attempt.intended["response_mode"]
    if action.slot["role"] == Role.JUDGE.value:
        logical_slot = attempt.intended["lineage"]["logical_member_slot"]
        if logical_slot not in {"primary", "secondary_a", "secondary_b"}:
            raise JournalCorrupt("qualification judge evidence lacks a logical slot")
        return _judge_scope_for_route(manifest, route, response_mode=mode)
    if action.slot["role"] != Role.SELECTOR.value:
        return None
    scopes = {
        ActionPurpose.SELECTOR_PORTFOLIO.value: "portfolio",
        ActionPurpose.PROBE_REVIEW.value: "probe_review",
        ActionPurpose.TEMPLATE_BINDING.value: "template_binding",
        ActionPurpose.OUTCOME_REVIEW.value: "outcome_review",
    }
    scope = scopes.get(action.slot["purpose"])
    if scope is None:
        return None
    return _selector_scope_for_route(
        manifest, route, scope=scope, response_mode=mode
    )


def _strategy_evidence_role_scopes(
    *,
    manifest: Mapping[str, Any],
    source_episode: StrategyEpisodeView,
    qualification_units: Sequence[WorkUnitView],
    work_units_by_id: Mapping[str, WorkUnitView],
    logical_actions: Mapping[str, LogicalActionView],
    attempts: Mapping[str, AttemptView],
    member_assessments: Mapping[str, MemberAssessment],
    used_attempt_ids: Iterable[str],
    response_schemas: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[str, ...], bool]:
    evidence_unit_ids = {
        item.work_unit_id for item in source_episode.rounds
    } | {unit.unit_id for unit in qualification_units}
    selected_attempt_ids = set(used_attempt_ids)

    # The exact source portfolio/reviews are semantic prerequisites of the
    # frozen template and in-run preference.  A repaired action's last strict
    # success is the result consumed by the composed stateful reducer.
    for unit_id in {item.work_unit_id for item in source_episode.rounds}:
        unit = work_units_by_id[unit_id]
        for action in unit.actions:
            if action.slot["role"] != Role.SELECTOR.value:
                continue
            usable = [
                attempt
                for attempt in action.successful_attempts
                if _success_is_structurally_usable(
                    action, attempt, response_schemas
                )
            ]
            if usable:
                selected_attempt_ids.add(usable[-1].physical_attempt_id)

    # Judge scope ownership is exact because each materialized assessment
    # retains its source attempt and logical panel slot.
    for unit_id in evidence_unit_ids:
        unit = work_units_by_id[unit_id]
        panel_action_ids = {
            member["judge_action_id"]
            for panel in unit.unit["panel_plans"]
            for member in panel["normal_members"]
        } | {
            panel["replacement"]["judge_action_id"]
            for panel in unit.unit["panel_plans"] if panel["replacement"] is not None
        }
        for action_id in panel_action_ids:
            assessment = member_assessments.get(action_id)
            if assessment is not None:
                selected_attempt_ids.add(assessment.source_attempt_id)

    scopes: dict[str, StructuredRoleScope] = {}
    for attempt_id in sorted(selected_attempt_ids):
        attempt = attempts.get(attempt_id)
        if attempt is None:
            raise JournalCorrupt("strategy qualification cites an unknown attempt")
        action = logical_actions[attempt.intended["logical_action_id"]]
        if action.unit_id not in evidence_unit_ids:
            raise JournalCorrupt("strategy qualification attempt has foreign ownership")
        scope = _component_scope_for_attempt(manifest, action, attempt)
        if scope is not None:
            scopes[scope.scope_key] = scope
    return tuple(sorted(scopes)), bool(scopes)


def _perturbator_prompt_mode_digests(
    manifest: Mapping[str, Any],
) -> tuple[tuple[str, str], ...]:
    try:
        material = [
            manifest["role_template_digests"]["perturbator_system"],
            manifest["role_template_digests"]["perturbator_user"],
            manifest["response_schema_digests"]["proposal_batch"],
        ]
    except (KeyError, TypeError) as exc:
        raise JournalCorrupt(f"Perturbator prompt-mode scope is invalid: {exc}") from exc
    return tuple(
        (mode, digest_json(["perturbator_prompt_mode", mode, *material]))
        for mode in ("strategy", "strategy_agnostic")
    )


def _scenario_recipe_scientific_key(
    scientific_scope_digest: str, recipe: ScenarioRecipe
) -> str:
    return scientific_key(
        [
            scientific_scope_digest,
            "artifact",
            "scenario_recipe",
            recipe.content_hash,
        ]
    )


def _last_usable_attempt_before(
    action: LogicalActionView,
    response_schemas: Mapping[str, Mapping[str, Any]],
    *,
    before_sequence: int,
) -> AttemptView:
    usable = tuple(
        attempt
        for attempt in action.successful_attempts
        if attempt.intent_event["sequence"] < before_sequence
        and _success_is_structurally_usable(action, attempt, response_schemas)
    )
    if len(usable) != 1:
        raise JournalCorrupt(
            "scenario source action does not have one authoritative usable result"
        )
    return usable[0]


def _route_for_attempt(
    manifest: Mapping[str, Any],
    action: LogicalActionView,
    attempt: AttemptView,
) -> _StaticRoute:
    bindings = tuple(
        item
        for item in action.slot["routes"]
        if item["route_key"] == attempt.intended["role_route_key"]
    )
    if len(bindings) != 1:
        raise JournalCorrupt("scenario source attempt has no exact frozen route")
    route = _static_route(
        manifest, action.slot["role"], bindings[0]["configured_slot"]
    )
    if route.route_key != bindings[0]["route_key"]:
        raise JournalCorrupt("scenario source route identity is not canonical")
    return route


def _template_binding_mode_before(
    *,
    manifest: Mapping[str, Any],
    route: _StaticRoute,
    logical_actions: Mapping[str, LogicalActionView],
    response_schemas: Mapping[str, Mapping[str, Any]],
    before_sequence: int,
) -> str:
    """Reproduce the durable route/schema mode as it stood at recipe freeze."""

    frozen: str | None = None
    try:
        schema_digest = manifest["response_schema_digests"]["template_binding"]
    except (KeyError, TypeError) as exc:
        raise JournalCorrupt("scenario template-binding schema is missing") from exc
    scoped = sorted(
        (
            (action, attempt)
            for action in logical_actions.values()
            for attempt in action.attempts
            if attempt.intent_event["sequence"] < before_sequence
            and attempt.intended["role_route_key"] == route.route_key
            and attempt.intended["response_schema_digest"] == schema_digest
        ),
        key=lambda item: item[1].intent_event["sequence"],
    )
    for action, attempt in scoped:
        mode = attempt.intended["response_mode"]
        if mode == "json_object":
            if frozen == "json_schema":
                raise JournalCorrupt(
                    "template-binding mode downgraded after strict success"
                )
            frozen = "json_object"
            continue
        terminal = attempt.terminal_event
        if terminal is None:
            continue
        if _success_is_structurally_usable(action, attempt, response_schemas):
            if frozen == "json_object":
                raise JournalCorrupt(
                    "template-binding strict success conflicts with frozen object mode"
                )
            frozen = "json_schema"
        elif (
            terminal["type"] == "model_call_failed"
            and terminal["payload"]["error_kind"]
            == "unsupported_response_format"
        ):
            if frozen == "json_schema":
                raise JournalCorrupt(
                    "template-binding format rejection follows strict success"
                )
            frozen = "json_object"
    return frozen or "json_schema"


def _scenario_source_scope_bundle(
    *,
    manifest: Mapping[str, Any],
    episode: StrategyEpisodeView,
    child: PortfolioChildView,
    work_units_by_id: Mapping[str, WorkUnitView],
    logical_actions: Mapping[str, LogicalActionView],
    response_schemas: Mapping[str, Mapping[str, Any]],
    before_sequence: int,
) -> tuple[tuple[tuple[str, str], ...], PerturbatorRoleScope]:
    """Recover the actual Selector/Perturbator scope that produced one step."""

    if child.selected is None or child.selected.source != "original":
        raise JournalCorrupt("scenario source child lacks one selected realization")
    source_unit = work_units_by_id.get(child.work_unit_id)
    if source_unit is None or source_unit.unit["kind"] != "portfolio_round":
        raise JournalCorrupt("scenario source child lacks its portfolio unit")
    source_rounds = tuple(
        item for item in episode.rounds if item.work_unit_id == child.work_unit_id
    )
    if len(source_rounds) != 1 or child not in source_rounds[0].children:
        raise JournalCorrupt("scenario source child is outside its strategy episode")
    lane = source_rounds[0].lane
    owner_keys = {
        lane.strategy_scientific_key,
        lane.strategy_slot.scientific_key,
    }
    reviews = tuple(
        sorted(
            (
                action
                for action in source_unit.actions
                if action.slot["purpose"] == ActionPurpose.PROBE_REVIEW.value
            ),
            key=lambda action: action.slot["ordinal"],
        )
    )
    outcomes = tuple(
        action
        for action in source_unit.actions
        if action.slot["purpose"] == ActionPurpose.OUTCOME_REVIEW.value
    )
    if len(reviews) != 1 or len(outcomes) != 1:
        raise JournalCorrupt("scenario source Selector graph is incomplete")
    review = reviews[0]
    proposal_candidates = tuple(
        action
        for action in source_unit.actions
        if action.slot["purpose"] == ActionPurpose.PERTURBATOR_PROPOSAL.value
        and action.slot["owner_position_scientific_key"] in owner_keys
        and action.slot["ordinal"] < review.slot["ordinal"]
    )
    if len(proposal_candidates) != 1:
        raise JournalCorrupt("scenario source Perturbator action is ambiguous")
    proposal = proposal_candidates[0]
    review_attempt = _last_usable_attempt_before(
        review, response_schemas, before_sequence=before_sequence
    )
    outcome_attempt = _last_usable_attempt_before(
        outcomes[0], response_schemas, before_sequence=before_sequence
    )
    proposal_attempt = _last_usable_attempt_before(
        proposal, response_schemas, before_sequence=before_sequence
    )
    probe_scope = _component_scope_for_attempt(manifest, review, review_attempt)
    outcome_scope = _component_scope_for_attempt(
        manifest, outcomes[0], outcome_attempt
    )
    if (
        probe_scope is None
        or probe_scope.scope != "probe_review"
        or outcome_scope is None
        or outcome_scope.scope != "outcome_review"
    ):
        raise JournalCorrupt("scenario source Selector scopes are invalid")
    review_route = _route_for_attempt(manifest, review, review_attempt)
    binding_mode = _template_binding_mode_before(
        manifest=manifest,
        route=review_route,
        logical_actions=logical_actions,
        response_schemas=response_schemas,
        before_sequence=before_sequence,
    )
    binding_scope = _selector_scope_for_route(
        manifest,
        review_route,
        scope="template_binding",
        response_mode=binding_mode,
    )
    perturbator_route = _route_for_attempt(manifest, proposal, proposal_attempt)
    if perturbator_route.role != Role.PERTURBATOR.value:
        raise JournalCorrupt("scenario source proposal did not use Perturbator")
    try:
        perturbator_scope = PerturbatorRoleScope(
            perturbator_route.requested_provider_route_key,
            perturbator_route.requested_model,
            perturbator_route.generation_profile_digest,
            proposal_attempt.intended["response_mode"],
            _perturbator_prompt_mode_digests(manifest),
        )
    except DomainError as exc:
        raise JournalCorrupt(f"scenario Perturbator scope is invalid: {exc}") from exc
    return (
        (
            ("outcome_review", outcome_scope.scope_key),
            ("probe_review", probe_scope.scope_key),
            ("template_binding", binding_scope.scope_key),
        ),
        perturbator_scope,
    )


def _work_unit_evidence_cutoff(unit: WorkUnitView) -> int:
    """Return the first sequence after all evidence owned by one source unit."""

    sequences = [unit.start_event["sequence"]]
    for action in unit.actions:
        for attempt in action.attempts:
            sequences.append(attempt.intent_event["sequence"])
            if attempt.dispatch_event is not None:
                sequences.append(attempt.dispatch_event["sequence"])
            if attempt.terminal_event is not None:
                sequences.append(attempt.terminal_event["sequence"])
    return max(sequences) + 1


def _scenario_episode_ancestry(
    *,
    episode: StrategyEpisodeView,
    episodes: Mapping[str, StrategyEpisodeView],
    conversation_nodes: Mapping[str, ConversationNode],
    candidate_metadata: Mapping[str, Mapping[str, Any]],
) -> tuple[
    Mapping[str, Any],
    tuple[tuple[StrategyEpisodeView, PortfolioChildView], ...],
] | None:
    """Trace credited prior edges back to one entry, or decline promotion."""

    node = conversation_nodes.get(episode.generation_parent_node_id)
    if (
        node is None
        or node.scientific_key != episode.generation_parent_node_scientific_key
    ):
        raise JournalCorrupt("scenario source episode lacks its exact parent node")
    reversed_steps: list[tuple[StrategyEpisodeView, PortfolioChildView]] = []
    seen_nodes: set[str] = set()
    while True:
        if node.node_id in seen_nodes:
            raise JournalCorrupt("scenario source ancestry contains a cycle")
        seen_nodes.add(node.node_id)
        candidate = candidate_metadata.get(node.candidate_id)
        if candidate is None:
            raise JournalCorrupt("scenario source ancestry names a non-search candidate")
        if node.parent_node_id is None:
            if candidate.get("origin") == "stateful_strategy":
                raise JournalCorrupt("scenario source root is not a standalone entry")
            return candidate, tuple(reversed(reversed_steps))
        strategy_id = candidate.get("strategy_id")
        prior_episode = (
            episodes.get(strategy_id) if isinstance(strategy_id, str) else None
        )
        if prior_episode is None:
            raise JournalCorrupt("scenario source ancestry lacks its prior episode")
        matches = tuple(
            child
            for child in prior_episode.children
            if child.candidate_id == candidate["candidate_id"]
            and child.candidate_scientific_key == candidate["scientific_key"]
            and child.observation_id == node.target_observation_id
        )
        if len(matches) != 1:
            raise JournalCorrupt("scenario source ancestry has ambiguous child ownership")
        child = matches[0]
        if not (
            child.alignment_status == "aligned"
            and child.outcome.aligned
            and child.outcome.resolved
            and child.outcome.outcome_result in {"supports", "falsifies", "indeterminate"}
            and child.selected is not None
        ):
            return None
        reversed_steps.append((prior_episode, child))
        parent = conversation_nodes.get(node.parent_node_id)
        if parent is None or parent.root_id != node.root_id:
            raise JournalCorrupt("scenario source ancestry is incomplete")
        node = parent


def _derive_scenario_recipes(
    *,
    run_id: str,
    scientific_scope_digest: str,
    manifest: Mapping[str, Any],
    objective: ObjectivePlugin | None,
    work_units: Sequence[WorkUnitView],
    logical_actions: Mapping[str, LogicalActionView],
    target_results: Mapping[str, TargetResultView],
    response_schemas: Mapping[str, Mapping[str, Any]],
    conversation_nodes: Mapping[str, ConversationNode],
    portfolio_rounds: Mapping[str, PortfolioRoundView],
    strategy_episodes: Mapping[str, StrategyEpisodeView],
) -> Mapping[str, ScenarioRecipeView]:
    """Reconstruct every imported and eligible derived recipe from facts only."""

    candidates = _candidate_metadata(
        scientific_scope_digest,
        run_id,
        manifest,
        work_units,
        logical_actions,
        target_results,
        response_schemas,
        objective=objective,
        portfolio_rounds=portfolio_rounds,
    )
    target_route = _static_route(manifest, Role.TARGET.value, "target")
    work_units_by_id = {unit.unit_id: unit for unit in work_units}
    result: dict[str, ScenarioRecipeView] = {}
    for payload in sorted(
        (
            item
            for item in manifest.get("seed_snapshot", ())
            if isinstance(item, Mapping)
            and item.get("type") == "scenario_recipe"
        ),
        key=lambda item: str(item.get("seed_id", "")),
    ):
        if objective is None:
            raise JournalCorrupt(
                "scenario seed replay requires the exact objective plugin"
            )
        prompt = payload.get("entry_prompt")
        seed_id = payload.get("seed_id")
        if not isinstance(prompt, str) or not prompt or not isinstance(seed_id, str):
            raise JournalCorrupt("imported scenario seed identity is incomplete")
        try:
            recipe = parse_scenario_recipe(
                payload.get("scenario_recipe"),
                objective.descriptor_values,
                objective.diagnosis_categories,
            )
            descriptors = objective.validate_descriptors(payload.get("descriptors"))
        except DomainError as exc:
            raise JournalCorrupt(f"imported scenario recipe is invalid: {exc}") from exc
        normalized_prompt = normalize_line_endings(prompt)
        if (
            payload.get("objective_id") != objective.id
            or payload.get("plugin_digest") != objective.plugin_digest
            or payload.get("target_generation_profile_digest")
            != target_route.generation_profile_digest
            or recipe.target_generation_profile_digest
            != target_route.generation_profile_digest
            or recipe.entry_prompt_hash
            != sha256_bytes(normalized_prompt.encode("utf-8"))
            or descriptors != dict(recipe.measurement_cell)
        ):
            raise JournalCorrupt("imported scenario recipe scope is inconsistent")
        key = _scenario_recipe_scientific_key(scientific_scope_digest, recipe)
        try:
            view = ScenarioRecipeView(
                scenario_id=entity_id("sc", [run_id, key]),
                scientific_key=key,
                entry_prompt=normalized_prompt,
                recipe=recipe,
                origin="imported",
                source_seed_id=seed_id,
                source_episode_id=None,
                source_child_scientific_key=None,
                parent_unit_id=None,
                source_step_episode_ids=(),
                source_step_child_scientific_keys=(),
                source_unit_ids=(),
            )
        except DomainError as exc:
            raise JournalCorrupt(f"imported scenario view is invalid: {exc}") from exc
        incumbent = result.get(key)
        if incumbent is not None:
            if incumbent.recipe.measurement_cell != recipe.measurement_cell:
                raise JournalCorrupt(
                    "equivalent imported scenarios declare conflicting cells"
                )
            continue
        result[key] = view

    derived: list[ScenarioRecipeView] = []
    for episode in sorted(
        strategy_episodes.values(), key=lambda item: item.strategy_scientific_key
    ):
        if not episode.rounds:
            continue
        template = episode.rounds[0].lane.item.template_candidate
        if not isinstance(template, StrategyTemplate):
            continue
        traced = _scenario_episode_ancestry(
            episode=episode,
            episodes=strategy_episodes,
            conversation_nodes=conversation_nodes,
            candidate_metadata=candidates,
        )
        if traced is None:
            continue
        entry, ancestry = traced
        ordered_templates = (
            *(
                prior.rounds[0].lane.item.template_candidate
                for prior, _ in ancestry
            ),
            template,
        )
        if any(not isinstance(item, StrategyTemplate) for item in ordered_templates):
            raise JournalCorrupt("scenario ancestry lacks a frozen strategy template")
        for child in sorted(
            episode.children, key=lambda item: item.child_scientific_key
        ):
            if not (
                child.alignment_status == "aligned"
                and child.outcome.aligned
                and child.outcome.resolved
                and child.outcome.confirmed_objective_application
                and child.outcome.outcome_result == "supports"
                and child.selected is not None
            ):
                continue
            parent_unit = work_units_by_id.get(child.work_unit_id)
            if parent_unit is None or parent_unit.unit["kind"] != "portfolio_round":
                raise JournalCorrupt("derived scenario source unit is invalid")
            source_steps = (*ancestry, (episode, child))
            freeze_sequence = _work_unit_evidence_cutoff(parent_unit)
            bundles = tuple(
                _scenario_source_scope_bundle(
                    manifest=manifest,
                    episode=source_episode,
                    child=source_child,
                    work_units_by_id=work_units_by_id,
                    logical_actions=logical_actions,
                    response_schemas=response_schemas,
                    before_sequence=freeze_sequence,
                )
                for source_episode, source_child in source_steps
            )
            if any(bundle != bundles[0] for bundle in bundles[1:]):
                continue
            selector_scopes, perturbator_scope = bundles[0]
            try:
                recipe = ScenarioRecipe(
                    entry_prompt_hash=sha256_bytes(
                        normalize_line_endings(str(entry["prompt"])).encode("utf-8")
                    ),
                    target_generation_profile_digest=(
                        target_route.generation_profile_digest
                    ),
                    templates=tuple(ordered_templates),
                    measurement_cell=template.move.descriptors,
                    maximum_target_turns=1 + len(ordered_templates),
                    maximum_strategy_rounds=len(ordered_templates),
                    designated_realization_policies=tuple(
                        "anchor" for _ in ordered_templates
                    ),
                    selector_scope_keys=selector_scopes,
                    perturbator_scope=perturbator_scope,
                )
                key = _scenario_recipe_scientific_key(
                    scientific_scope_digest, recipe
                )
                derived.append(
                    ScenarioRecipeView(
                        scenario_id=entity_id("sc", [run_id, key]),
                        scientific_key=key,
                        entry_prompt=normalize_line_endings(str(entry["prompt"])),
                        recipe=recipe,
                        origin="derived",
                        source_seed_id=None,
                        source_episode_id=episode.strategy_id,
                        source_child_scientific_key=child.child_scientific_key,
                        parent_unit_id=child.work_unit_id,
                        source_step_episode_ids=tuple(
                            source_episode.strategy_id
                            for source_episode, _ in source_steps
                        ),
                        source_step_child_scientific_keys=tuple(
                            source_child.child_scientific_key
                            for _, source_child in source_steps
                        ),
                        source_unit_ids=tuple(
                            dict.fromkeys(
                                source_child.work_unit_id
                                for _, source_child in source_steps
                            )
                        ),
                    )
                )
            except DomainError as exc:
                raise JournalCorrupt(f"derived scenario recipe is invalid: {exc}") from exc
    for view in sorted(
        derived,
        key=lambda item: (
            str(item.source_episode_id),
            str(item.source_child_scientific_key),
        ),
    ):
        incumbent = result.get(view.scientific_key)
        if incumbent is None:
            result[view.scientific_key] = view
        # Imported recipes and the first deterministic derived source own an
        # equivalent content identity.  A later conflicting measurement cell
        # closes locally and cannot create a second recipe or block.
    return MappingProxyType(dict(sorted(result.items())))


def _scenario_selector_components(
    *,
    manifest: Mapping[str, Any],
    unit: WorkUnitView,
    recipe: ScenarioRecipe,
    response_schemas: Mapping[str, Mapping[str, Any]],
) -> tuple[StructuredRoleScope, ...]:
    candidates = {
        scope.scope_key: scope
        for base, scope in _structured_role_scope_candidates(manifest)
        if base[0] == Role.SELECTOR.value
    }
    frozen = dict(recipe.selector_scope_keys)
    components: list[StructuredRoleScope] = []
    for name, purpose in (
        ("outcome_review", ActionPurpose.OUTCOME_REVIEW.value),
        ("probe_review", ActionPurpose.PROBE_REVIEW.value),
    ):
        actual: dict[str, StructuredRoleScope] = {}
        for action in unit.actions:
            if action.slot["purpose"] != purpose:
                continue
            for attempt in action.successful_attempts:
                if not _success_is_structurally_usable(
                    action, attempt, response_schemas
                ):
                    continue
                scope = _component_scope_for_attempt(manifest, action, attempt)
                if scope is not None:
                    actual[scope.scope_key] = scope
        frozen_scope = candidates.get(frozen[name])
        if len(actual) == 1:
            scope = next(iter(actual.values()))
        elif not actual and frozen_scope is not None:
            scope = frozen_scope
        else:
            available = tuple(sorted(actual.values(), key=lambda item: item.scope_key))
            if available:
                nonfrozen = tuple(
                    item for item in available if item.scope_key != frozen[name]
                )
                scope = (nonfrozen or available)[0]
            else:
                scope = next(
                    item
                    for item in candidates.values()
                    if item.scope == name
                )
        components.append(scope)
    return tuple(sorted(components, key=lambda item: item.scope))


def _scenario_current_perturbator_scope(
    *,
    manifest: Mapping[str, Any],
    unit: WorkUnitView,
    recipe: ScenarioRecipe,
    response_schemas: Mapping[str, Mapping[str, Any]],
) -> PerturbatorRoleScope:
    actual: dict[str, PerturbatorRoleScope] = {}
    for action in unit.actions:
        if action.slot["purpose"] != ActionPurpose.PERTURBATOR_PROPOSAL.value:
            continue
        for attempt in action.successful_attempts:
            if not _success_is_structurally_usable(
                action, attempt, response_schemas
            ):
                continue
            route = _route_for_attempt(manifest, action, attempt)
            try:
                scope = PerturbatorRoleScope(
                    route.requested_provider_route_key,
                    route.requested_model,
                    route.generation_profile_digest,
                    attempt.intended["response_mode"],
                    _perturbator_prompt_mode_digests(manifest),
                )
            except DomainError as exc:
                raise JournalCorrupt(
                    f"scenario verification Perturbator scope is invalid: {exc}"
                ) from exc
            actual[scope.scope_key] = scope
    if not actual:
        return recipe.perturbator_scope
    if len(actual) == 1:
        return next(iter(actual.values()))
    nonfrozen = tuple(
        actual[key]
        for key in sorted(actual)
        if key != recipe.perturbator_scope.scope_key
    )
    return nonfrozen[0] if nonfrozen else actual[sorted(actual)[0]]


def _scenario_judge_panel_evidence(
    *,
    manifest: Mapping[str, Any],
    unit: WorkUnitView,
    logical_actions: Mapping[str, LogicalActionView],
    member_assessments: Mapping[str, MemberAssessment],
    protocol_digest: str,
) -> tuple[NamedScopeEvidence, ...]:
    action_ids = {action.action_id for action in unit.actions}
    attempts_by_id = {
        attempt.physical_attempt_id: attempt
        for action in unit.actions
        for attempt in action.attempts
    }
    scopes_by_slot: dict[str, dict[str, StructuredRoleScope]] = {
        "primary": {},
        "secondary_a": {},
        "secondary_b": {},
    }
    for action_id, assessment in member_assessments.items():
        if action_id not in action_ids:
            continue
        attempt = attempts_by_id.get(assessment.source_attempt_id)
        if attempt is None:
            raise JournalCorrupt("scenario judge assessment source is missing")
        scope = _component_scope_for_attempt(
            manifest, logical_actions[action_id], attempt
        )
        if scope is None:
            raise JournalCorrupt("scenario judge assessment has no exact scope")
        scopes_by_slot[assessment.logical_member_slot][scope.scope_key] = scope
    first_panel = unit.unit["panel_plans"][0]
    for member in first_panel["normal_members"]:
        logical_slot = member["logical_member_slot"]
        if scopes_by_slot[logical_slot]:
            continue
        action = logical_actions[member["judge_action_id"]]
        if action.attempts:
            attempt = action.attempts[-1]
            scope = _component_scope_for_attempt(manifest, action, attempt)
            if scope is None:
                raise JournalCorrupt("scenario judge attempt has no exact scope")
        else:
            binding = action.slot["routes"][0]
            route = _static_route(
                manifest, Role.JUDGE.value, binding["configured_slot"]
            )
            scope = _judge_scope_for_route(
                manifest,
                route,
                response_mode="json_schema",
            )
        scopes_by_slot[logical_slot][scope.scope_key] = scope
    judges: list[NamedScopeEvidence] = []
    for logical_slot in configured_judge_slots(manifest, include_replacement=False):
        scopes = scopes_by_slot[logical_slot]
        chosen = scopes[sorted(scopes)[0]]
        judges.append(NamedScopeEvidence(f"judge_{logical_slot}", chosen.scope_key))
    panel_key = digest_json(
        [
            "judge_panel_qualification",
            protocol_digest,
            [[item.name, item.scope_key] for item in judges],
        ]
    )
    judges.append(NamedScopeEvidence("panel", panel_key))
    return tuple(sorted(judges, key=lambda item: item.name))


def _derive_scenario_seed_qualifications(
    *,
    run_id: str,
    source_run_directory: str,
    manifest: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    work_units: Sequence[WorkUnitView],
    logical_actions: Mapping[str, LogicalActionView],
    target_results: Mapping[str, TargetResultView],
    target_observations: Mapping[str, TargetObservation],
    member_assessments: Mapping[str, MemberAssessment],
    execution_evidence: Mapping[str, ExecutionEvidence],
    conversation_nodes: Mapping[str, ConversationNode],
    response_schemas: Mapping[str, Mapping[str, Any]],
    target_identity: str | None,
    scenarios: Mapping[str, ScenarioRecipeView],
) -> tuple[
    Mapping[str, SeedQualificationDecision],
    tuple[Mapping[str, Any], ...],
    Mapping[str, EvidenceRecord],
    tuple[ArchiveEntry, ...],
]:
    units = tuple(
        unit
        for unit in work_units
        if unit.unit["kind"] == "verification"
        and unit.unit["design"].get("subject_kind") == "scenario_recipe"
    )
    if not units:
        return MappingProxyType({}), (), MappingProxyType({}), ()
    target_route = _static_route(manifest, Role.TARGET.value, "target")
    identity = ProviderIdentity(
        f"{target_route.requested_provider_route_key}/{target_route.requested_model}",
        target_identity,
    )
    protocol_digest = _judge_protocol_digest(manifest)
    objective_evidence = ObjectiveScopeEvidence(
        manifest["objective"]["id"],
        manifest["objective"]["plugin_digest"],
        _objective_scope_key(manifest),
    )
    taxonomy = manifest["descriptor_taxonomy"]
    units_by_id = {unit.unit_id: unit for unit in work_units}
    decisions: dict[str, SeedQualificationDecision] = {}
    snapshots: list[Mapping[str, Any]] = []
    records: dict[str, EvidenceRecord] = {}
    entries: list[ArchiveEntry] = []
    seen_scenarios: set[str] = set()
    for unit in sorted(units, key=lambda item: item.start_event["sequence"]):
        scenario = scenarios.get(unit.unit["subject_scientific_key"])
        if scenario is None or scenario.scenario_id != unit.unit["subject_id"]:
            raise JournalCorrupt(
                "scenario verification subject cannot be reconstructed"
            )
        if scenario.scenario_id in seen_scenarios:
            raise JournalCorrupt("scenario has several verification blocks")
        seen_scenarios.add(scenario.scenario_id)
        executions = _validate_scenario_verification_unit(
            run_id=run_id,
            manifest=manifest,
            unit=unit,
            scenario=scenario,
            logical_actions=logical_actions,
            target_results=target_results,
            target_observations=target_observations,
            execution_evidence=execution_evidence,
            conversation_nodes=conversation_nodes,
            response_schemas=response_schemas,
        )
        components = _scenario_selector_components(
            manifest=manifest,
            unit=unit,
            recipe=scenario.recipe,
            response_schemas=response_schemas,
        )
        perturbator_scope = _scenario_current_perturbator_scope(
            manifest=manifest,
            unit=unit,
            recipe=scenario.recipe,
            response_schemas=response_schemas,
        )
        judge_panel = _scenario_judge_panel_evidence(
            manifest=manifest,
            unit=unit,
            logical_actions=logical_actions,
            member_assessments=member_assessments,
            protocol_digest=protocol_digest,
        )
        seed_id = entity_id(
            "sd",
            [
                "scenario_recipe",
                manifest["objective"]["id"],
                manifest["objective"]["plugin_digest"],
                target_route.generation_profile_digest,
                identity.token,
                scenario.recipe.content_hash,
            ],
        )
        try:
            subject = ScenarioSeedSubject(
                seed_id=seed_id,
                entry_prompt=scenario.entry_prompt,
                measurement_cell=scenario.recipe.measurement_cell,
                objective_id=manifest["objective"]["id"],
                plugin_digest=manifest["objective"]["plugin_digest"],
                target_generation_profile_digest=(
                    target_route.generation_profile_digest
                ),
                descriptor_taxonomy_digest=taxonomy["digest"],
                target_identity=identity,
                recipe=scenario.recipe,
            )
            rule = SeedVerificationRule(unit.unit["design"]["planned_count"])
            block = SeedVerificationBlock(
                block_id=entity_id(
                    "sb",
                    [
                        run_id,
                        scenario.scientific_key,
                        "verification",
                        rule.planned_count,
                    ],
                ),
                rule=rule,
                executions=executions,
                judge_panel_scope_keys=tuple(
                    (item.name, item.scope_key) for item in judge_panel
                ),
            )
            context = SeedQualificationContext(
                objective=objective_evidence,
                judge_protocol_digest=protocol_digest,
                judge_panel=judge_panel,
                components=components,
                current_perturbator_scope=perturbator_scope,
            )
        except (KeyError, DomainError) as exc:
            raise JournalCorrupt(
                f"scenario seed qualification input is invalid: {exc}"
            ) from exc
        source_units: list[WorkUnitView] = []
        for unit_id in scenario.source_unit_ids:
            source_unit = units_by_id.get(unit_id)
            if source_unit is None:
                raise JournalCorrupt("scenario source evidence unit is missing")
            source_units.append(source_unit)
        source_units.append(unit)
        source = _seed_evidence_source(
            run_id=run_id,
            source_run_directory=source_run_directory,
            events=events,
            units=source_units,
            include_run_created=scenario.origin == "imported",
        )
        try:
            decision = evaluate_seed_qualification(subject, block, context, source)
        except DomainError as exc:
            raise JournalCorrupt(
                f"scenario seed qualification is invalid: {exc}"
            ) from exc
        decisions[seed_id] = decision
        if decision.snapshot is not None:
            snapshots.append(MappingProxyType(decision.snapshot.to_dict()))
        archive_verification = block.reduce()
        evidence = evidence_record_from_decisions(
            subject_id=scenario.scenario_id,
            screening_block_id=None,
            screening=None,
            verification_block_id=block.block_id,
            verification=archive_verification,
        )
        records[scenario.scenario_id] = evidence
        fitness = derive_archive_fitness(
            verification_terminal=archive_verification.block_terminated,
            verification_scores=archive_verification.effective_scores,
            screening_complete=False,
            screening_scores=(),
            singleton_score=None,
        ).fitness
        entry_key = digest_json([scenario.scientific_key, "archive_entry"])
        entries.append(
            ArchiveEntry(
                entry_id=entity_id("ae", [run_id, entry_key]),
                scientific_key=entry_key,
                subject_id=scenario.scenario_id,
                cell=tuple(value for _, value in scenario.recipe.measurement_cell),
                evidence=evidence,
                fitness=fitness,
                lineage_id=None,
                origin=f"scenario_{scenario.origin}",
                strategy_id=scenario.source_episode_id,
            )
        )
    return (
        MappingProxyType(dict(sorted(decisions.items()))),
        tuple(sorted(snapshots, key=lambda item: item["seed_id"])),
        MappingProxyType(dict(sorted(records.items()))),
        tuple(entries),
    )


def _strategy_role_scopes(
    *,
    manifest: Mapping[str, Any],
    qualification_units: Sequence[WorkUnitView],
    logical_actions: Mapping[str, LogicalActionView],
    attempts: Mapping[str, AttemptView],
    used_attempt_ids: Iterable[str],
    target_identity: str | None,
) -> tuple[
    Mapping[str, Any],
    tuple[Mapping[str, Any], ...],
    tuple[str, ...],
]:
    target_route = _static_route(manifest, Role.TARGET.value, "target")
    requested_alias = (
        f"{target_route.requested_provider_route_key}/{target_route.requested_model}"
    )
    identity_token = (
        f"effective:{target_identity}"
        if target_identity is not None
        else f"requested:{requested_alias}"
    )
    target_scope_key = digest_json(
        [
            "target_role_scope",
            target_route.requested_provider_route_key,
            target_route.requested_model,
            target_route.generation_profile_digest,
            identity_token,
            target_identity is None,
        ]
    )
    target_scope = MappingProxyType(
        {
            "scope_key": target_scope_key,
            "requested_provider_route_key": target_route.requested_provider_route_key,
            "requested_model": target_route.requested_model,
            "generation_profile_digest": target_route.generation_profile_digest,
            "identity_token": identity_token,
            "identity_unverifiable": target_identity is None,
        }
    )
    qualification_unit_ids = {unit.unit_id for unit in qualification_units}
    prompt_modes = _perturbator_prompt_mode_digests(manifest)
    scopes: dict[str, Mapping[str, Any]] = {}
    for attempt_id in sorted(set(used_attempt_ids)):
        attempt = attempts.get(attempt_id)
        if attempt is None:
            continue
        action = logical_actions[attempt.intended["logical_action_id"]]
        if (
            action.unit_id not in qualification_unit_ids
            or action.slot["role"] != Role.PERTURBATOR.value
        ):
            continue
        bindings = [
            item
            for item in action.slot["routes"]
            if item["route_key"] == attempt.intended["role_route_key"]
        ]
        if len(bindings) != 1:
            raise JournalCorrupt("Perturbator evidence uses an unknown route")
        route = _static_route(
            manifest, Role.PERTURBATOR.value, bindings[0]["configured_slot"]
        )
        try:
            scope = PerturbatorRoleScope(
                route.requested_provider_route_key,
                route.requested_model,
                route.generation_profile_digest,
                attempt.intended["response_mode"],
                prompt_modes,
            )
        except DomainError as exc:
            raise JournalCorrupt(f"Perturbator role scope is invalid: {exc}") from exc
        scopes[scope.scope_key] = MappingProxyType(
            {
                "scope_key": scope.scope_key,
                "requested_provider_route_key": scope.requested_provider_route_key,
                "requested_model": scope.requested_model,
                "generation_profile_digest": scope.generation_profile_digest,
                "response_mode": scope.response_mode,
                "prompt_mode_digests": [list(item) for item in prompt_modes],
            }
        )
    role_keys = tuple(sorted((target_scope_key, *scopes)))
    return target_scope, tuple(scopes[key] for key in sorted(scopes)), role_keys


def _partition_strategy_structured_role_scope_keys(
    manifest: Mapping[str, Any],
    scope_keys: Sequence[str],
) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    """Name exact Selector scopes without conflating them with judge evidence."""

    candidates = {
        scope.scope_key: (base, scope)
        for base, scope in _structured_role_scope_candidates(manifest)
    }
    selector: list[tuple[str, str]] = []
    judge: list[str] = []
    for scope_key in scope_keys:
        candidate = candidates.get(scope_key)
        if candidate is None:
            raise JournalCorrupt(
                "strategy qualification cites an unknown structured-role scope"
            )
        base, scope = candidate
        if base[0] == Role.SELECTOR.value:
            selector.append((scope.scope, scope_key))
        elif base[0] == Role.JUDGE.value:
            judge.append(scope_key)
        else:  # pragma: no cover - candidate construction is closed above
            raise JournalCorrupt("strategy qualification role scope is invalid")
    return tuple(sorted(selector)), tuple(sorted(judge))


def _arm_metrics_payload(value: Any) -> Mapping[str, Any]:
    return {
        "arm": value.arm,
        "planned_count": value.planned_count,
        "terminal_count": value.terminal_count,
        "resolved_count": value.resolved_count,
        "unresolved_count": value.unresolved_count,
        "pending_count": value.pending_count,
        "applicable_count": value.applicable_count,
        "inapplicable_count": value.inapplicable_count,
        "objective_successes": value.objective_successes,
        "strategy_fidelity_successes": value.strategy_fidelity_successes,
        "exceptional_count": value.exceptional_count,
        "effective_scores": [normalize_decimal(item) for item in value.effective_scores],
        "effective_score_median": (
            None
            if value.effective_score_median is None
            else normalize_decimal(value.effective_score_median)
        ),
        "applicability_rate": (
            None
            if value.applicability_rate is None
            else normalize_decimal(value.applicability_rate)
        ),
        "conditional_success_rate": (
            None
            if value.conditional_success_rate is None
            else normalize_decimal(value.conditional_success_rate)
        ),
        "unconditional_end_to_end_success_rate": normalize_decimal(
            value.unconditional_end_to_end_success_rate
        ),
    }


def _development_decision_payload(
    value: DevelopmentBlockDecision,
) -> Mapping[str, Any]:
    gate = value.purpose_gate
    return {
        "status": value.status.value,
        "arms": [_arm_metrics_payload(item) for item in value.arms],
        "all_roots_terminal": value.all_roots_terminal,
        "all_roots_resolved": value.all_roots_resolved,
        "strict_treatment_repeat_support": value.strict_treatment_repeat_support,
        "treatment_control_advantage": value.treatment_control_advantage,
        "exceptional_requirement": value.exceptional_requirement,
        "purpose_gate": {
            "purpose": gate.purpose,
            "supporting_evidence_satisfied": gate.supporting_evidence_satisfied,
            "treatment_median_improves_each_control": gate.treatment_median_improves_each_control,
            "passed": gate.passed,
        },
    }


def _holdout_decision_payload(value: HoldoutBlockDecision) -> Mapping[str, Any]:
    return {
        "status": value.status.value,
        "arms": [_arm_metrics_payload(item) for item in value.arms],
        "all_roots_terminal": value.all_roots_terminal,
        "all_roots_resolved": value.all_roots_resolved,
        "strict_treatment_repeat_support": value.strict_treatment_repeat_support,
        "treatment_control_advantage": value.treatment_control_advantage,
        "exceptional_requirement": value.exceptional_requirement,
    }


def _qualification_evidence_events(
    units: Sequence[WorkUnitView],
) -> tuple[tuple[str, ...], tuple[Mapping[str, Any], ...]]:
    event_ids: list[tuple[int, str]] = []
    terminals: list[Mapping[str, Any]] = []
    for unit in units:
        start_id = unit.start_event.get("event_id")
        if isinstance(start_id, str):
            event_ids.append((unit.start_event["sequence"], start_id))
        for action in unit.actions:
            for attempt in action.attempts:
                for event in (
                    attempt.intent_event,
                    attempt.dispatch_event,
                    attempt.terminal_event,
                ):
                    if event is None:
                        continue
                    event_ids.append((event["sequence"], event["event_id"]))
                if attempt.terminal_event is not None:
                    terminals.append(attempt.terminal_event)
    ordered_ids = tuple(
        event_id
        for _, event_id in sorted(set(event_ids), key=lambda item: (item[0], item[1]))
    )
    terminals.sort(key=lambda item: (item["sequence"], item["event_id"]))
    return ordered_ids, tuple(terminals)


def _strategy_call_counts(units: Sequence[WorkUnitView]) -> Mapping[str, int]:
    counts = {role.value: 0 for role in Role}
    for unit in units:
        for action in unit.actions:
            counts[action.slot["role"]] += sum(
                attempt.dispatch_event is not None for attempt in action.attempts
            )
    return MappingProxyType(counts)


def _strategy_snapshot(
    view: StrategyQualificationView,
    *,
    manifest: Mapping[str, Any],
    target_scope: Mapping[str, Any],
    perturbator_scopes: Sequence[Mapping[str, Any]],
    objective_scope_key: str,
    call_counts: Mapping[str, int],
    qualified_at: str,
) -> Mapping[str, Any]:
    if (
        view.qualification_state is not QualificationState.QUALIFIED
        or view.development_decision is None
        or view.holdout_decision is None
        or view.development_design is None
        or view.sealed_holdout_plan is None
        or view.holdout_design is None
        or view.source_checkpoint is None
    ):
        raise JournalCorrupt("only a complete qualified strategy can be materialized")
    compatibility_key = qualified_strategy_compatibility_key(
        objective_id=manifest["objective"]["id"],
        plugin_digest=manifest["objective"]["plugin_digest"],
        target_scope=target_scope,
        template=view.template,
        tested_entry_context_ids=view.tested_entry_context_ids,
        selector_scope_keys=view.required_selector_scope_keys,
        perturbator_scopes=perturbator_scopes,
        development_design=view.development_design,
        holdout_design=view.holdout_design,
    )
    return MappingProxyType(
        {
            "template_id": view.template_id,
            "template_scientific_key": view.template_scientific_key,
            "template": view.template.to_dict(),
            "objective_id": manifest["objective"]["id"],
            "plugin_digest": manifest["objective"]["plugin_digest"],
            "objective_scope_key": objective_scope_key,
            "target_requested_provider_route_key": target_scope[
                "requested_provider_route_key"
            ],
            "target_requested_model": target_scope["requested_model"],
            "target_generation_profile_digest": target_scope[
                "generation_profile_digest"
            ],
            "target_role_scope_key": target_scope["scope_key"],
            "identity_token": target_scope["identity_token"],
            "identity_unverifiable": target_scope["identity_unverifiable"],
            "diagnosis": view.template.applicability.diagnosis,
            "move_act": view.template.move.act,
            "target_kind": view.template.move.target_kind,
            "applicability": view.template.applicability.to_dict(),
            "source_episode_id": view.source_episode_id,
            "source_conversation_root_id": view.source_conversation_root_id,
            "source_conversation_root_scientific_key": (
                view.source_conversation_root_scientific_key
            ),
            "source_generation_parent_node_id": (
                view.source_generation_parent_node_id
            ),
            "source_generation_parent_node_scientific_key": (
                view.source_generation_parent_node_scientific_key
            ),
            "source_entry_candidate_id": view.source_entry_candidate_id,
            "source_entry_candidate_scientific_key": (
                view.source_entry_candidate_scientific_key
            ),
            "tested_entry_context_ids": list(view.tested_entry_context_ids),
            "selector_scope_keys": [
                list(item) for item in view.required_selector_scope_keys
            ],
            "judge_scope_keys": list(view.required_judge_scope_keys),
            "required_structured_role_scope_keys": list(
                view.required_structured_role_scope_keys
            ),
            "perturbator_scopes": [dict(item) for item in perturbator_scopes],
            "required_role_scope_keys": list(view.required_role_scope_keys),
            "development_unit_id": view.development_unit_id,
            "development_design": dict(view.development_design),
            "sealed_holdout_plan": dict(view.sealed_holdout_plan),
            "sealed_holdout_digest": view.sealed_holdout_digest,
            "holdout_unit_id": view.holdout_unit_id,
            "holdout_design": dict(view.holdout_design),
            "development_result": _development_decision_payload(
                view.development_decision
            ),
            "holdout_result": _holdout_decision_payload(view.holdout_decision),
            "all_calls_used": dict(call_counts),
            "evidence_event_ids": list(view.evidence_event_ids),
            "qualified_at": qualified_at,
            "compatibility_key": compatibility_key,
            "qualification_status": QualificationStatus.QUALIFIED.value,
            "source_checkpoint": MappingProxyType(dict(view.source_checkpoint)),
        }
    )


def _derive_strategy_qualifications(
    *,
    run_id: str,
    source_run_directory: str,
    scientific_scope_digest: str,
    manifest: Mapping[str, Any],
    objective: ObjectivePlugin | None,
    events: Sequence[Mapping[str, Any]],
    work_units: Sequence[WorkUnitView],
    logical_actions: Mapping[str, LogicalActionView],
    attempts: Mapping[str, AttemptView],
    target_results: Mapping[str, TargetResultView],
    target_observations: Mapping[str, TargetObservation],
    panel_views: Mapping[str, PanelView],
    member_assessments: Mapping[str, MemberAssessment],
    candidate_evidence: Mapping[str, EvidenceRecord],
    conversation_nodes: Mapping[str, ConversationNode],
    portfolio_rounds: Mapping[str, PortfolioRoundView],
    strategy_episodes: Mapping[str, StrategyEpisodeView],
    scenarios: Mapping[str, ScenarioRecipeView],
    scenario_evidence: Mapping[str, EvidenceRecord],
    response_schemas: Mapping[str, Mapping[str, Any]],
    target_identity: str | None,
) -> tuple[
    Mapping[str, StrategyQualificationView],
    tuple[Mapping[str, Any], ...],
    Mapping[str, bool],
]:
    qualification_units = tuple(
        unit
        for unit in work_units
        if unit.unit["kind"] in {"strategy_development", "strategy_holdout"}
    )
    verified_episode_sequences: dict[str, int] = {}
    scenario_units = {
        (unit.unit["subject_id"], unit.unit["subject_scientific_key"]): unit
        for unit in work_units
        if unit.unit["kind"] == "verification"
        and unit.unit["design"].get("subject_kind") == "scenario_recipe"
    }
    for scenario in scenarios.values():
        evidence = scenario_evidence.get(scenario.scenario_id)
        if (
            scenario.origin != "derived"
            or evidence is None
            or evidence.verification_status is not VerificationStatus.VERIFIED
        ):
            continue
        verification_unit = scenario_units.get(
            (scenario.scenario_id, scenario.scientific_key)
        )
        if verification_unit is None:
            raise JournalCorrupt(
                "verified derived scenario lacks its verification work unit"
            )
        cutoff = _work_unit_evidence_cutoff(verification_unit)
        for episode_id in scenario.source_step_episode_ids:
            prior = verified_episode_sequences.get(episode_id)
            if prior is None or cutoff < prior:
                verified_episode_sequences[episode_id] = cutoff
    eligible_episodes = tuple(
        episode
        for episode in strategy_episodes.values()
        if episode.strategy_id in verified_episode_sequences
        and _episode_has_qualification_trigger(episode)
    )
    if not qualification_units and not eligible_episodes:
        return MappingProxyType({}), (), MappingProxyType({})
    plugin = _stateful_objective(manifest, objective, work_units)
    if plugin is None:
        raise JournalCorrupt("strategy qualification requires its objective plugin")
    work_units_by_id = {unit.unit_id: unit for unit in work_units}
    metadata = _candidate_metadata(
        scientific_scope_digest,
        run_id,
        manifest,
        work_units,
        logical_actions,
        target_results,
        response_schemas,
        objective=plugin,
        portfolio_rounds=portfolio_rounds,
    )
    development_by_template: dict[str, WorkUnitView] = {}
    holdout_by_template: dict[str, WorkUnitView] = {}
    source_by_template: dict[str, StrategyEpisodeView] = {}
    for unit in qualification_units:
        template_id = unit.unit["subject_id"]
        owner = (
            development_by_template
            if unit.unit["kind"] == "strategy_development"
            else holdout_by_template
        )
        if template_id in owner:
            raise JournalCorrupt(
                f"strategy template has several {unit.unit['kind']} work units"
            )
        owner[template_id] = unit
        source = _qualification_source_episode(
            unit,
            run_id=run_id,
            scientific_scope_digest=scientific_scope_digest,
            episodes=strategy_episodes,
            work_units_by_id=work_units_by_id,
        )
        verified_at = verified_episode_sequences.get(source.strategy_id)
        if verified_at is None:
            raise JournalCorrupt(
                "strategy qualification started without a verified derived scenario"
            )
        if unit.start_event["sequence"] < verified_at:
            raise JournalCorrupt(
                "strategy qualification started before scenario verification completed"
            )
        prior = source_by_template.get(template_id)
        if prior is not None and prior.strategy_id != source.strategy_id:
            raise JournalCorrupt("strategy qualification changed its source episode")
        source_by_template[template_id] = source
    for episode in eligible_episodes:
        template = episode.rounds[0].lane.item.template_candidate
        assert template is not None
        _, template_id = _strategy_template_identity(
            run_id, scientific_scope_digest, template
        )
        source_by_template.setdefault(template_id, episode)
    if set(holdout_by_template) - set(development_by_template):
        raise JournalCorrupt("strategy holdout has no development unit")

    views: dict[str, StrategyQualificationView] = {}
    snapshots: list[Mapping[str, Any]] = []
    semantic_completion: dict[str, bool] = {}
    event_by_sequence = {event["sequence"]: event for event in events}
    for template_id, source_episode in sorted(source_by_template.items()):
        template = source_episode.rounds[0].lane.item.template_candidate
        if template is None:
            raise JournalCorrupt("strategy qualification source lost its template")
        template_key, expected_template_id = _strategy_template_identity(
            run_id, scientific_scope_digest, template
        )
        if template_id != expected_template_id:
            raise JournalCorrupt("strategy qualification template ID is not canonical")
        source_entry, entry_metadata = _strategy_source_entry(
            source_episode, conversation_nodes, metadata
        )
        development = development_by_template.get(template_id)
        holdout = holdout_by_template.get(template_id)
        development_roots: tuple[QualificationRootOutcome, ...] = ()
        holdout_roots: tuple[QualificationRootOutcome, ...] = ()
        development_decision: DevelopmentBlockDecision | None = None
        holdout_decision: HoldoutBlockDecision | None = None
        used_attempt_ids: list[str] = []
        relevant_units = [
            work_units_by_id[item.work_unit_id] for item in source_episode.rounds
        ]
        measurement_verified = False
        if development is not None:
            if development.unit.get("parent_unit_id") not in {
                item.work_unit_id for item in source_episode.rounds
            }:
                raise JournalCorrupt("strategy development changed its source episode")
            _validate_strategy_development_design(
                development,
                run_id=run_id,
                scientific_scope_digest=scientific_scope_digest,
                manifest=manifest,
                objective=plugin,
                template=template,
                logical_actions=logical_actions,
            )
            projections = tuple(
                _derive_qualification_root(
                    development,
                    plan,
                    template=template,
                    objective=plugin,
                    logical_actions=logical_actions,
                    target_results=target_results,
                    target_observations=target_observations,
                    panel_views=panel_views,
                    response_schemas=response_schemas,
                )
                for plan in development.unit["design"]["root_plans"]
            )
            development_roots = tuple(item.outcome for item in projections)
            used_attempt_ids.extend(
                attempt_id for item in projections for attempt_id in item.used_attempt_ids
            )
            measurement_verified = _measurement_cell_verified_before(
                development,
                development.unit["design"]["measurement_cell"],
                work_units=work_units,
                candidate_metadata=metadata,
                candidate_evidence=candidate_evidence,
                scenarios=scenarios,
                scenario_evidence=scenario_evidence,
            )
            try:
                development_decision = evaluate_development_block(
                    QualificationBlockDesign(
                        template.purpose,
                        development.unit["design"]["planned_count_per_arm"],
                        tuple(development.unit["design"]["comparison_dimensions"]),
                        measurement_verified,
                    ),
                    development_roots,
                    started=True,
                )
            except DomainError as exc:
                raise JournalCorrupt(
                    f"strategy development decision is invalid: {exc}"
                ) from exc
            semantic_completion[development.unit_id] = bool(
                development_decision.all_roots_terminal
            )
            relevant_units.append(development)
        if holdout is not None:
            assert development is not None and development_decision is not None
            if development_decision.status is not QualificationState.ELIGIBLE:
                raise JournalCorrupt(
                    "strategy holdout started before development passed"
                )
            _validate_strategy_holdout_design(
                holdout,
                development,
                run_id=run_id,
                scientific_scope_digest=scientific_scope_digest,
                manifest=manifest,
                objective=plugin,
                template=template,
                logical_actions=logical_actions,
            )
            projections = tuple(
                _derive_qualification_root(
                    holdout,
                    plan,
                    template=template,
                    objective=plugin,
                    logical_actions=logical_actions,
                    target_results=target_results,
                    target_observations=target_observations,
                    panel_views=panel_views,
                    response_schemas=response_schemas,
                )
                for plan in holdout.unit["design"]["root_plans"]
            )
            holdout_roots = tuple(item.outcome for item in projections)
            used_attempt_ids.extend(
                attempt_id for item in projections for attempt_id in item.used_attempt_ids
            )
            try:
                holdout_decision = evaluate_holdout_block(
                    QualificationBlockDesign(
                        template.purpose,
                        holdout.unit["design"]["sealed_holdout_plan"][
                            "planned_count_per_arm"
                        ],
                        tuple(
                            holdout.unit["design"]["sealed_holdout_plan"][
                                "comparison_dimensions"
                            ]
                        ),
                        measurement_verified,
                    ),
                    holdout_roots,
                    started=True,
                )
            except DomainError as exc:
                raise JournalCorrupt(
                    f"strategy holdout decision is invalid: {exc}"
                ) from exc
            semantic_completion[holdout.unit_id] = bool(
                holdout_decision.all_roots_terminal
            )
            relevant_units.append(holdout)
        structured_role_keys, exact_scopes_satisfied = _strategy_evidence_role_scopes(
            manifest=manifest,
            source_episode=source_episode,
            qualification_units=tuple(
                item for item in (development, holdout) if item is not None
            ),
            work_units_by_id=work_units_by_id,
            logical_actions=logical_actions,
            attempts=attempts,
            member_assessments=member_assessments,
            used_attempt_ids=used_attempt_ids,
            response_schemas=response_schemas,
        )
        selector_scope_keys, judge_scope_keys = (
            _partition_strategy_structured_role_scope_keys(
                manifest, structured_role_keys
            )
        )
        try:
            qualification_state = derive_strategy_qualification_state(
                development_decision,
                holdout_decision,
                exact_qualification_scopes_satisfied=exact_scopes_satisfied,
            )
        except DomainError as exc:
            raise JournalCorrupt(
                f"strategy qualification state is contradictory: {exc}"
            ) from exc
        target_scope, perturbator_scopes, role_scope_keys = _strategy_role_scopes(
            manifest=manifest,
            qualification_units=tuple(
                item for item in (development, holdout) if item is not None
            ),
            logical_actions=logical_actions,
            attempts=attempts,
            used_attempt_ids=used_attempt_ids,
            target_identity=target_identity,
        )
        evidence_event_ids, terminal_events = _qualification_evidence_events(
            relevant_units
        )
        checkpoint = (
            None
            if not terminal_events
            else _checkpoint(run_id, source_run_directory, terminal_events)
        )
        context_ids = {
            source_entry.candidate_id,
            *(
                item["execution_id"]
                for unit in (development, holdout)
                if unit is not None
                for item in unit.unit["execution_slots"]
            ),
        }
        view = StrategyQualificationView(
            template_id=template_id,
            template_scientific_key=template_key,
            template=template,
            source_episode_id=source_episode.strategy_id,
            source_conversation_root_id=source_episode.conversation_root_id,
            source_conversation_root_scientific_key=(
                source_episode.conversation_root_scientific_key
            ),
            source_generation_parent_node_id=(
                source_episode.generation_parent_node_id
            ),
            source_generation_parent_node_scientific_key=(
                source_episode.generation_parent_node_scientific_key
            ),
            source_entry_candidate_id=source_entry.candidate_id,
            source_entry_candidate_scientific_key=str(
                entry_metadata["scientific_key"]
            ),
            tested_entry_context_ids=tuple(sorted(context_ids)),
            development_unit_id=(
                None if development is None else development.unit_id
            ),
            development_design=(
                None
                if development is None
                else MappingProxyType(dict(development.unit["design"]))
            ),
            sealed_holdout_plan=(
                None
                if development is None
                else MappingProxyType(
                    dict(development.unit["design"]["sealed_holdout_plan"])
                )
            ),
            sealed_holdout_digest=(
                None
                if development is None
                else development.unit["design"]["sealed_holdout_digest"]
            ),
            development_roots=development_roots,
            development_decision=development_decision,
            holdout_unit_id=None if holdout is None else holdout.unit_id,
            holdout_design=(
                None
                if holdout is None
                else MappingProxyType(dict(holdout.unit["design"]))
            ),
            holdout_roots=holdout_roots,
            holdout_decision=holdout_decision,
            qualification_state=qualification_state,
            required_structured_role_scope_keys=structured_role_keys,
            required_selector_scope_keys=selector_scope_keys,
            required_judge_scope_keys=judge_scope_keys,
            required_role_scope_keys=role_scope_keys,
            evidence_event_ids=evidence_event_ids,
            source_checkpoint=(
                None if checkpoint is None else MappingProxyType(dict(checkpoint))
            ),
        )
        views[template_id] = view
        if qualification_state is QualificationState.QUALIFIED:
            assert checkpoint is not None
            checkpoint_event = event_by_sequence[checkpoint["sequence"]]
            snapshots.append(
                _strategy_snapshot(
                    view,
                    manifest=manifest,
                    target_scope=target_scope,
                    perturbator_scopes=perturbator_scopes,
                    objective_scope_key=_objective_scope_key(
                        manifest
                    ),
                    call_counts=_strategy_call_counts(
                        tuple(item for item in (development, holdout) if item is not None)
                    ),
                    qualified_at=checkpoint_event["timestamp_utc"],
                )
            )
    return (
        MappingProxyType(dict(sorted(views.items()))),
        tuple(sorted(snapshots, key=lambda item: item["template_id"])),
        MappingProxyType(dict(sorted(semantic_completion.items()))),
    )


def reduce_events(
    raw_events: Sequence[Mapping[str, Any]],
    objective: ObjectivePlugin | None = None,
) -> RunState:
    """Validate and reduce a complete journal into its sole public run state."""

    events = validate_events(raw_events)
    if not events:
        raise JournalCorrupt("a run state requires run_created")
    terminal_event = _validate_invocation_lifecycle(events)
    created = events[0]["payload"]
    manifest = created["manifest"]
    source_run_directory = _source_run_directory(created["run_path_at_creation"])
    _validate_request_manifest(manifest, objective)
    limit = _budget_limit(manifest)
    units, slots, action_units, active_unit_id, completed_unit_ids = _slot_indexes(
        events,
        limit,
        terminal_event,
        digest_json(manifest["decision_constants"]),
    )
    expected_judges = configured_judge_slots(manifest, include_replacement=False)
    for _, unit in units:
        for panel in unit["panel_plans"]:
            if tuple(member["logical_member_slot"] for member in panel["normal_members"]) != expected_judges:
                raise JournalCorrupt("panel composition differs from frozen judge configuration")
            if (panel["replacement"] is None) != (manifest["model_profiles"]["judges"]["replacement"] is None):
                raise JournalCorrupt("panel replacement differs from frozen judge configuration")
    attempts, attempts_by_action, response_schemas = _reduce_attempts(
        events,
        slots,
        action_units,
        manifest,
    )

    logical_actions: dict[str, LogicalActionView] = {}
    for action_id, slot in slots.items():
        logical_actions[action_id] = LogicalActionView(
            action_id=action_id,
            unit_id=action_units[action_id],
            slot=slot,
            attempts=attempts_by_action.get(action_id, ()),
        )
    target_receipts = _target_success_receipts(logical_actions)
    pinned_target_identity, mismatch_attempt_ids = _target_identity_projection(
        target_receipts
    )
    (
        logical_actions,
        role_collision_attempt_ids,
        role_unverifiable_attempt_ids,
    ) = _apply_role_identity_projection(logical_actions, pinned_target_identity)
    # Public attempt ownership must use the same projected instances as the
    # action/work-unit views.  The journal remains the sole fact store; the
    # collision bit is deterministically recomputed on every replay.
    attempts = {
        attempt.physical_attempt_id: attempt
        for action in logical_actions.values()
        for attempt in action.attempts
    }
    target_results = _target_results(
        logical_actions, frozenset(mismatch_attempt_ids)
    )
    judge_target_replies: dict[str, str] = {}
    for _, unit in units:
        for panel in unit["panel_plans"]:
            target = target_results.get(panel["target_action_id"])
            if target is None:
                continue
            for member in panel["normal_members"]:
                judge_target_replies[member["judge_action_id"]] = target.text
            if panel["replacement"] is not None:
                judge_target_replies[panel["replacement"]["judge_action_id"]] = target.text
    unit_kind_by_id = {unit["unit_id"]: unit["kind"] for _, unit in units}
    validate_recovery_transitions(
        events,
        logical_actions,
        {
            action_id: unit_kind_by_id[action_units[action_id]]
            for action_id in logical_actions
        },
        response_schemas,
        judge_target_replies,
    )
    _validate_durability_order(logical_actions, units)

    consumed_counts = {"target": 0, "judge": 0, "selector": 0, "perturbator": 0}
    for attempt in attempts.values():
        if attempt.dispatch_event is not None:
            consumed_counts[attempt.dispatch_event["payload"]["charged_role"]] += 1
    consumed = RoleCallVector(**consumed_counts)
    if not consumed.fits(limit):
        raise JournalCorrupt("persisted dispatches exceed a role budget")
    remaining = RoleCallVector(
        target=limit.target - consumed.target,
        judge=limit.judge - consumed.judge,
        selector=limit.selector - consumed.selector,
        perturbator=limit.perturbator - consumed.perturbator,
    )

    work_views = tuple(
        WorkUnitView(
            start_event=start_event,
            unit=unit,
            actions=tuple(
                logical_actions[slot["action_id"]] for slot in unit["logical_action_slots"]
            ),
        )
        for start_event, unit in units
    )
    target_observations = _derive_target_observations(
        events[0]["run_id"], logical_actions, target_results
    )
    member_assessments, panel_assessments, panel_views = _derive_panel_views(
        run_id=events[0]["run_id"],
        work_units=work_views,
        logical_actions=logical_actions,
        target_results=target_results,
        target_observations=target_observations,
        response_schemas=response_schemas,
    )
    sample_executions, execution_evidence = _derive_sample_executions(
        work_units=work_views,
        logical_actions=logical_actions,
        target_results=target_results,
        target_observations=target_observations,
        panel_views=panel_views,
        response_schemas=response_schemas,
    )
    conversation_roots, conversation_nodes = _derive_conversations(
        sample_executions=sample_executions,
        target_observations=target_observations,
        target_results=target_results,
    )
    candidate_evidence, screening_blocks, archive = _derive_evidence_and_archive(
        run_id=events[0]["run_id"],
        scope=created["scientific_scope_digest"],
        manifest=manifest,
        work_units=work_views,
        logical_actions=logical_actions,
        target_results=target_results,
        panel_views=panel_views,
        member_assessments=member_assessments,
        execution_evidence=execution_evidence,
        conversation_nodes=conversation_nodes,
        response_schemas=response_schemas,
        objective=objective,
    )
    def derive_portfolios() -> tuple[
        Mapping[str, PortfolioRoundView],
        Mapping[str, StrategyEpisodeView],
        tuple[str, ...],
    ]:
        return _derive_stateful_portfolios(
            run_id=events[0]["run_id"],
            manifest=manifest,
            objective=objective,
            work_units=work_views,
            logical_actions=logical_actions,
            target_results=target_results,
            target_observations=target_observations,
            panel_views=panel_views,
            member_assessments=member_assessments,
            execution_evidence=execution_evidence,
            conversation_roots=conversation_roots,
            conversation_nodes=conversation_nodes,
            candidate_evidence=candidate_evidence,
            response_schemas=response_schemas,
            remaining=remaining,
            terminal_event=terminal_event,
        )

    def derive_strategy_qualifications(
        episodes: Mapping[str, StrategyEpisodeView],
        scenarios: Mapping[str, ScenarioRecipeView],
        scenario_evidence: Mapping[str, EvidenceRecord],
    ) -> tuple[
        Mapping[str, StrategyQualificationView],
        tuple[Mapping[str, Any], ...],
        Mapping[str, bool],
    ]:
        return _derive_strategy_qualifications(
            run_id=events[0]["run_id"],
            source_run_directory=source_run_directory,
            scientific_scope_digest=created["scientific_scope_digest"],
            manifest=manifest,
            objective=objective,
            events=events,
            work_units=work_views,
            logical_actions=logical_actions,
            attempts=attempts,
            target_results=target_results,
            target_observations=target_observations,
            panel_views=panel_views,
            member_assessments=member_assessments,
            candidate_evidence=candidate_evidence,
            conversation_nodes=conversation_nodes,
            portfolio_rounds=portfolio_rounds,
            strategy_episodes=episodes,
            scenarios=scenarios,
            scenario_evidence=scenario_evidence,
            response_schemas=response_schemas,
            target_identity=pinned_target_identity,
        )

    portfolio_rounds, strategy_episodes, strategy_supports_ordering = (
        derive_portfolios()
    )
    # Portfolio descriptors are engine-owned and exist only after the
    # canonical Selector/Proposal/Review projection above.  Recompose the
    # deterministic evidence view with those materialized children instead of
    # interpreting provider-local aliases a second time.
    candidate_evidence, screening_blocks, archive = _derive_evidence_and_archive(
        run_id=events[0]["run_id"],
        scope=created["scientific_scope_digest"],
        manifest=manifest,
        work_units=work_views,
        logical_actions=logical_actions,
        target_results=target_results,
        panel_views=panel_views,
        member_assessments=member_assessments,
        execution_evidence=execution_evidence,
        conversation_nodes=conversation_nodes,
        response_schemas=response_schemas,
        objective=objective,
        portfolio_rounds=portfolio_rounds,
    )
    scenarios = _derive_scenario_recipes(
        run_id=events[0]["run_id"],
        scientific_scope_digest=created["scientific_scope_digest"],
        manifest=manifest,
        objective=objective,
        work_units=work_views,
        logical_actions=logical_actions,
        target_results=target_results,
        response_schemas=response_schemas,
        conversation_nodes=conversation_nodes,
        portfolio_rounds=portfolio_rounds,
        strategy_episodes=strategy_episodes,
    )
    (
        entry_seed_qualifications,
        entry_seed_qualification_snapshots,
    ) = _derive_entry_seed_qualifications(
        run_id=events[0]["run_id"],
        source_run_directory=source_run_directory,
        scientific_scope_digest=created["scientific_scope_digest"],
        manifest=manifest,
        objective=objective,
        events=events,
        work_units=work_views,
        logical_actions=logical_actions,
        target_results=target_results,
        response_schemas=response_schemas,
        execution_evidence=execution_evidence,
        candidate_evidence=candidate_evidence,
        panel_views=panel_views,
        member_assessments=member_assessments,
        target_identity=pinned_target_identity,
    )
    (
        scenario_seed_qualifications,
        scenario_seed_qualification_snapshots,
        scenario_evidence,
        scenario_archive_entries,
    ) = _derive_scenario_seed_qualifications(
        run_id=events[0]["run_id"],
        source_run_directory=source_run_directory,
        manifest=manifest,
        events=events,
        work_units=work_views,
        logical_actions=logical_actions,
        target_results=target_results,
        target_observations=target_observations,
        member_assessments=member_assessments,
        execution_evidence=execution_evidence,
        conversation_nodes=conversation_nodes,
        response_schemas=response_schemas,
        target_identity=pinned_target_identity,
        scenarios=scenarios,
    )
    (
        strategy_qualifications,
        current_strategy_qualification_snapshots,
        strategy_qualification_completion,
    ) = derive_strategy_qualifications(
        strategy_episodes,
        scenarios,
        scenario_evidence,
    )
    if set(entry_seed_qualifications) & set(scenario_seed_qualifications):
        raise JournalCorrupt("entry and scenario seed identities collide")
    seed_qualifications = MappingProxyType(
        dict(
            sorted(
                {
                    **entry_seed_qualifications,
                    **scenario_seed_qualifications,
                }.items()
            )
        )
    )
    current_seed_qualification_snapshots = tuple(
        sorted(
            (
                *entry_seed_qualification_snapshots,
                *scenario_seed_qualification_snapshots,
            ),
            key=lambda item: item["seed_id"],
        )
    )
    try:
        archive = build_archive(
            (*archive.entries, *scenario_archive_entries),
            archive.capacity_per_cell,
        )
    except DomainError as exc:
        raise JournalCorrupt(f"scenario archive projection is invalid: {exc}") from exc
    active_unit_id, completed_unit_ids = _validate_work_unit_boundaries(
        events,
        work_views,
        logical_actions,
        response_schemas,
        {
            work_unit_id: round_view.complete
            for work_unit_id, round_view in portfolio_rounds.items()
        }
        | dict(strategy_qualification_completion),
        portfolio_rounds=portfolio_rounds,
    )
    drift = bool(mismatch_attempt_ids)
    target_identity_unverifiable = pinned_target_identity is None

    dangling = tuple(
        sorted(
            attempt_id
            for attempt_id, attempt in attempts.items()
            if attempt.dispatch_event is not None and attempt.terminal_event is None
        )
    )
    state = RunState(
        events=tuple(events),
        run_id=events[0]["run_id"],
        manifest=manifest,
        scientific_scope_digest=created["scientific_scope_digest"],
        work_units=work_views,
        active_work_unit_id=active_unit_id,
        completed_work_unit_ids=completed_unit_ids,
        attempts=attempts,
        logical_actions=logical_actions,
        consumed=consumed,
        remaining=remaining,
        target_results=target_results,
        target_observations=target_observations,
        member_assessments=member_assessments,
        panel_assessments=panel_assessments,
        panel_views=panel_views,
        sample_executions=sample_executions,
        execution_evidence=execution_evidence,
        conversation_roots=conversation_roots,
        conversation_nodes=conversation_nodes,
        candidate_evidence=candidate_evidence,
        screening_blocks=screening_blocks,
        archive=archive,
        portfolio_rounds=portfolio_rounds,
        strategy_episodes=strategy_episodes,
        strategy_supports_ordering=strategy_supports_ordering,
        scenarios=scenarios,
        seed_qualifications=seed_qualifications,
        scenario_evidence=scenario_evidence,
        current_seed_qualification_snapshots=(
            current_seed_qualification_snapshots
        ),
        strategy_qualifications=strategy_qualifications,
        current_strategy_qualification_snapshots=(
            current_strategy_qualification_snapshots
        ),
        dangling_attempt_ids=dangling,
        target_identity=pinned_target_identity,
        target_identity_unverifiable=target_identity_unverifiable,
        target_identity_drift=drift,
        target_identity_mismatch_attempt_ids=mismatch_attempt_ids,
        role_target_identity_collision_attempt_ids=MappingProxyType(
            dict(role_collision_attempt_ids)
        ),
        role_identity_unverifiable_attempt_ids=MappingProxyType(
            dict(role_unverifiable_attempt_ids)
        ),
        terminal_event=terminal_event,
        cost_liability_usd=reduce_cost_accounting(events, attempts, slots, manifest),
    )
    validate_search_policy(state)
    return state


def reduce_event_iterable(
    events: Iterable[Mapping[str, Any]],
    objective: ObjectivePlugin | None = None,
) -> RunState:
    return reduce_events(tuple(events), objective)
