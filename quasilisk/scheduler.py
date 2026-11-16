"""Pure work-unit admission, ordering, and committed-slot fairness policy.

There is no queue and no resource ledger here.  Every function consumes an
already-derived state snapshot and returns a deterministic value.  The engine
remains the sole owner of state transitions and provider calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

from .domain import (
    RoleCallVector,
    WorkUnit,
    require_digest,
    seeded_rank,
)
from .run_state import (
    SchedulerValidationError,
    action_slot_maximum,
    entry_verification_maximum,
    evaluated_target_maximum,
    formula_maximum,
    initial_observation_maximum,
    map_exploration_maximum,
    parent_baseline_root_maximum,
    portfolio_round_maximum,
    recompute_work_unit_maximum,
    screening_maximum,
    stateful_root_maximum,
    strategy_agnostic_root_maximum,
    strategy_development_maximum,
    strategy_holdout_maximum,
    sum_root_maxima,
    treatment_root_maximum,
    validate_work_unit_maximum,
)


def _scheduler_count(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SchedulerValidationError(f"{label} must be a nonnegative integer")
    return value


def _vector_fits(required: RoleCallVector, remaining: RoleCallVector) -> bool:
    """Private component-wise fit primitive.

    Callers outside this module admit only complete, validated ``WorkUnit``
    records.  Keeping the vector primitive private prevents a hand-written
    maximum from bypassing the frozen design and action graph.
    """

    if not isinstance(required, RoleCallVector):
        raise SchedulerValidationError("required budget must be a RoleCallVector")
    if not isinstance(remaining, RoleCallVector):
        raise SchedulerValidationError("remaining budget must be a RoleCallVector")
    return required.fits(remaining)


def can_start(
    work: WorkUnit,
    remaining: RoleCallVector,
) -> bool:
    """Return true exactly when every worst-case role component fits."""

    required = validate_work_unit_maximum(work)
    return _vector_fits(required, remaining)


def can_start_sequence(
    work: Iterable[WorkUnit],
    remaining: RoleCallVector,
) -> bool:
    """Preflight a fixed sequence atomically without reserving budget."""

    total = RoleCallVector()
    for item in work:
        total = total + validate_work_unit_maximum(item)
    return _vector_fits(total, remaining)


EVIDENCE_TIERS = frozenset({"application", "exceptional", "ordinary"})
ORDINARY_KINDS = frozenset({"initial_observation", "map_exploration"})
STATEFUL_KINDS = frozenset({"portfolio_round"})


@dataclass(frozen=True, slots=True)
class WorkCandidate:
    """A reducer-derived, not-yet-started finite work-unit candidate."""

    work_unit: WorkUnit
    active: bool = False
    eligible: bool = True
    measurement_cell_verified: bool = False
    evidence_tier: str = "ordinary"
    fixed_block_median: Decimal | None = None
    prior_target_calls: int = 0
    slot_ordinal: int = 0
    semantic_progress: int = 0

    def __post_init__(self) -> None:
        validate_work_unit_maximum(self.work_unit)
        require_digest(
            self.work_unit.subject_scientific_key,
            "work subject scientific key",
        )
        require_digest(
            self.work_unit.unit_scientific_key,
            "work candidate scientific key",
        )
        _scheduler_count(self.prior_target_calls, "candidate prior target calls")
        _scheduler_count(self.slot_ordinal, "candidate slot ordinal")
        if self.evidence_tier not in EVIDENCE_TIERS:
            raise SchedulerValidationError("candidate evidence tier is invalid")
        if self.kind in ORDINARY_KINDS | STATEFUL_KINDS and self.target_slots < 1:
            raise SchedulerValidationError("discretionary work must commit at least one target slot")
        if self.fixed_block_median is not None:
            if isinstance(self.fixed_block_median, (bool, float)):
                raise SchedulerValidationError("candidate median cannot use Boolean or binary float")
            try:
                converted = Decimal(str(self.fixed_block_median))
            except (InvalidOperation, ValueError) as exc:
                raise SchedulerValidationError("candidate median is invalid") from exc
            if not converted.is_finite():
                raise SchedulerValidationError("candidate median must be finite")
            object.__setattr__(self, "fixed_block_median", converted)

    @property
    def candidate_key(self) -> str:
        return self.work_unit.unit_scientific_key

    @property
    def kind(self) -> str:
        return self.work_unit.kind

    @property
    def subject_scientific_key(self) -> str:
        return self.work_unit.subject_scientific_key

    @property
    def target_slots(self) -> int:
        if self.kind == "initial_observation":
            return 1
        if self.kind in {"map_exploration", "portfolio_round"}:
            # Work-unit conformance has already proved this closed design field.
            return int(self.work_unit.design["proposal_count"])
        return 0


def priority_class(candidate: WorkCandidate) -> int:
    """Implement exact work priority, including replayed fixture-work kinds."""

    if candidate.active:
        return 1
    if candidate.kind == "verification":
        return 2
    if candidate.kind in {"strategy_development", "strategy_holdout"}:
        return 7
    if candidate.kind == "screening":
        return 4
    if candidate.kind == "portfolio_round":
        return 3 if candidate.work_unit.design.get("portfolio_intent") == "advance" else 6
    if candidate.kind == "initial_observation":
        return 5
    if candidate.kind == "map_exploration":
        return 6
    raise SchedulerValidationError(f"candidate has unknown work kind: {candidate.kind}")


def candidate_order_key(candidate: WorkCandidate) -> tuple[Any, ...]:
    tier_rank = {"application": 0, "exceptional": 1, "ordinary": 2}
    portfolio_intent_rank = (
        {"advance": 0, "recover": 1}.get(
            candidate.work_unit.design.get("portfolio_intent"), 2
        )
        if candidate.kind == "portfolio_round"
        else 0
    )
    median_key: tuple[int, Decimal]
    if candidate.fixed_block_median is None:
        median_key = (1, Decimal(0))
    else:
        median_key = (0, -candidate.fixed_block_median)
    return (
        0 if candidate.active else 1,
        priority_class(candidate),
        portfolio_intent_rank,
        -candidate.semantic_progress,
        1 if candidate.measurement_cell_verified else 0,
        tier_rank[candidate.evidence_tier],
        median_key,
        candidate.prior_target_calls,
        candidate.subject_scientific_key,
        candidate.slot_ordinal,
    )


def order_candidates(candidates: Iterable[WorkCandidate]) -> tuple[WorkCandidate, ...]:
    eligible = tuple(item for item in candidates if item.eligible)
    seen_candidate_keys: set[str] = set()
    seen_order_keys: set[tuple[Any, ...]] = set()
    for item in eligible:
        if item.candidate_key in seen_candidate_keys:
            raise SchedulerValidationError("work candidate key is duplicated")
        seen_candidate_keys.add(item.candidate_key)
        order_key = candidate_order_key(item)
        if order_key in seen_order_keys:
            raise SchedulerValidationError("two work candidates have the same complete policy rank")
        seen_order_keys.add(order_key)
    return tuple(sorted(eligible, key=candidate_order_key))


def fairness_allows_stateful(
    coverage_slots_committed: int,
    stateful_slots_committed: int,
    prospective_stateful_slots: int,
    *,
    ordinary_work_admissible: bool,
) -> bool:
    coverage = _scheduler_count(coverage_slots_committed, "coverage slots committed")
    stateful = _scheduler_count(stateful_slots_committed, "stateful slots committed")
    prospective = _scheduler_count(prospective_stateful_slots, "prospective stateful slots")
    return (not ordinary_work_admissible) or 3 * coverage >= stateful + prospective


def fairness_debt(coverage_slots_committed: int, stateful_slots_committed: int) -> int:
    coverage = _scheduler_count(coverage_slots_committed, "coverage slots committed")
    stateful = _scheduler_count(stateful_slots_committed, "stateful slots committed")
    return max(0, stateful - 3 * coverage)


def choose_next_work(
    candidates: Iterable[WorkCandidate],
    remaining: RoleCallVector,
    *,
    coverage_slots_committed: int,
    stateful_slots_committed: int,
) -> WorkCandidate | None:
    """Choose the first complete admissible unit with the fairness override."""

    ordered = order_candidates(candidates)
    active = tuple(item for item in ordered if item.active)
    if len(active) > 1:
        raise SchedulerValidationError("only one work unit may be active")
    if active:
        # Admission was proven before work_unit_started.  Reapplying the
        # original full-unit maximum after some calls have been consumed would
        # incorrectly strand a fixed block.
        return active[0]
    def admissible(item: WorkCandidate) -> bool:
        return _vector_fits(validate_work_unit_maximum(item.work_unit), remaining)
    feasible = tuple(item for item in ordered if admissible(item))
    ordinary = tuple(item for item in feasible if item.kind in ORDINARY_KINDS)
    for item in feasible:
        if item.kind in STATEFUL_KINDS and item.work_unit.design.get("portfolio_intent") != "advance" and not item.active and not fairness_allows_stateful(
            coverage_slots_committed,
            stateful_slots_committed,
            item.target_slots,
            ordinary_work_admissible=bool(ordinary),
        ):
            # The binding rule starts the highest-ranked admissible ordinary
            # unit immediately; it does not shop for a smaller stateful round.
            return ordinary[0]
        return item
    return None


def deterministic_decision_key(
    random_seed: int,
    decision_kind: str,
    subject_scientific_key: str,
    round_index: int,
    slot_ordinal: int,
) -> str:
    require_digest(subject_scientific_key, "decision subject scientific key")
    if not decision_kind:
        raise SchedulerValidationError("decision kind is empty")
    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise SchedulerValidationError("random seed must be an integer")
    _scheduler_count(round_index, "decision round index")
    _scheduler_count(slot_ordinal, "decision slot ordinal")
    return seeded_rank(
        random_seed,
        [decision_kind, subject_scientific_key, round_index, slot_ordinal],
    )


__all__ = [
    "SchedulerValidationError",
    "WorkCandidate",
    "action_slot_maximum",
    "can_start",
    "can_start_sequence",
    "candidate_order_key",
    "choose_next_work",
    "deterministic_decision_key",
    "entry_verification_maximum",
    "evaluated_target_maximum",
    "fairness_allows_stateful",
    "fairness_debt",
    "formula_maximum",
    "initial_observation_maximum",
    "map_exploration_maximum",
    "order_candidates",
    "parent_baseline_root_maximum",
    "portfolio_round_maximum",
    "priority_class",
    "recompute_work_unit_maximum",
    "screening_maximum",
    "stateful_root_maximum",
    "strategy_agnostic_root_maximum",
    "strategy_development_maximum",
    "strategy_holdout_maximum",
    "sum_root_maxima",
    "treatment_root_maximum",
    "validate_work_unit_maximum",
]
