"""Purpose-aware scheduling for validated perturbation candidates.

The perturbator and validator decide which candidates exist.  This module only
decides which validated candidates consume the remaining target-call budget.
Keeping that policy pure makes budget allocation independently testable and
keeps evaluation side effects out of scheduling decisions.
"""

from __future__ import annotations

import dataclasses
import math
from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from typing import Any, Generic, Protocol, TypeVar

from .models import MutationTask, TaskPurpose


WORK_PURPOSES = (
    TaskPurpose.OPTIMIZATION,
    TaskPurpose.DISCOVERY,
    TaskPurpose.DIAGNOSTIC,
)
PURPOSE_RANK = {
    purpose: index for index, purpose in enumerate(WORK_PURPOSES)
}


class SchedulableCandidate(Protocol):
    """The minimal candidate surface needed by the scheduler."""

    id: str
    parent_id: str | None
    metadata: dict[str, Any]


CandidateT = TypeVar("CandidateT", bound=SchedulableCandidate)


@dataclasses.dataclass(frozen=True)
class SchedulingPolicy:
    """Immutable, serializable candidate-allocation policy.

    Purpose weights apply only to perturbator-generated work.  Replay controls
    are paired separately: at most ``max_controls_per_attempt`` controls are
    admitted after every schedulable attempt has first been allocated one work
    slot.  This prevents controls from starving a parent's only search child.
    """

    purpose_weights: tuple[tuple[TaskPurpose, float], ...] = (
        (TaskPurpose.OPTIMIZATION, 4.0),
        (TaskPurpose.DISCOVERY, 1.0),
        (TaskPurpose.DIAGNOSTIC, 1.0),
    )
    max_controls_per_attempt: int = 1
    max_control_share: float = 0.25

    def __post_init__(self) -> None:
        purposes = [purpose for purpose, _weight in self.purpose_weights]
        if len(purposes) != len(set(purposes)):
            raise ValueError("purpose_weights must not contain duplicate purposes")
        if set(purposes) != set(WORK_PURPOSES):
            expected = ", ".join(purpose.value for purpose in WORK_PURPOSES)
            raise ValueError(f"purpose_weights must define exactly: {expected}")
        for purpose, weight in self.purpose_weights:
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise TypeError(f"weight for {purpose.value} must be numeric")
            if not math.isfinite(float(weight)) or float(weight) <= 0:
                raise ValueError(f"weight for {purpose.value} must be finite and > 0")
        if (
            isinstance(self.max_controls_per_attempt, bool)
            or not isinstance(self.max_controls_per_attempt, int)
        ):
            raise TypeError("max_controls_per_attempt must be an integer")
        if self.max_controls_per_attempt < 0:
            raise ValueError("max_controls_per_attempt must be >= 0")
        if (
            isinstance(self.max_control_share, bool)
            or not isinstance(self.max_control_share, (int, float))
        ):
            raise TypeError("max_control_share must be numeric")
        if (
            not math.isfinite(float(self.max_control_share))
            or not 0 <= float(self.max_control_share) < 1
        ):
            raise ValueError("max_control_share must be finite and in [0, 1)")

    def weight(self, purpose: TaskPurpose) -> float:
        for configured_purpose, weight in self.purpose_weights:
            if configured_purpose is purpose:
                return float(weight)
        raise ValueError(f"No scheduling weight for purpose: {purpose.value}")

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "purpose_weights": {
                purpose.value: weight
                for purpose, weight in self.purpose_weights
            },
            "max_controls_per_attempt": self.max_controls_per_attempt,
            "max_control_share": self.max_control_share,
        }

    @classmethod
    def from_config(cls, data: Any) -> "SchedulingPolicy":
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise TypeError("perturbation_scheduler must be a JSON object")
        unknown = set(data) - {
            "purpose_weights",
            "max_controls_per_attempt",
            "max_control_share",
        }
        if unknown:
            raise ValueError(
                "unknown perturbation_scheduler field(s): "
                + ", ".join(sorted(str(key) for key in unknown))
            )

        defaults = cls()
        configured_weights = data.get("purpose_weights")
        if configured_weights is None:
            weights = defaults.purpose_weights
        else:
            if not isinstance(configured_weights, Mapping):
                raise TypeError("perturbation_scheduler.purpose_weights must be an object")
            unknown_purposes = set(configured_weights) - {
                purpose.value for purpose in WORK_PURPOSES
            }
            if unknown_purposes:
                raise ValueError(
                    "unknown scheduling purpose(s): "
                    + ", ".join(sorted(str(key) for key in unknown_purposes))
                )
            default_by_purpose = dict(defaults.purpose_weights)
            weights = tuple(
                (
                    purpose,
                    configured_weights.get(
                        purpose.value,
                        default_by_purpose[purpose],
                    ),
                )
                for purpose in WORK_PURPOSES
            )

        raw_control_limit = data.get(
            "max_controls_per_attempt",
            defaults.max_controls_per_attempt,
        )
        return cls(
            purpose_weights=weights,
            max_controls_per_attempt=raw_control_limit,
            max_control_share=data.get(
                "max_control_share",
                defaults.max_control_share,
            ),
        )


@dataclasses.dataclass(frozen=True)
class CandidateSchedule(Generic[CandidateT]):
    """One deterministic scheduling decision and its complete audit."""

    scheduled: tuple[CandidateT, ...]
    deferred: tuple[CandidateT, ...]
    audit: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class MutationTaskSchedule:
    """Purpose-aware selection of one attempt's perturbator generation tasks."""

    controls: tuple[MutationTask, ...]
    generation_tasks: tuple[MutationTask, ...]
    deferred: tuple[MutationTask, ...]
    audit: dict[str, Any]

    @property
    def selected(self) -> tuple[MutationTask, ...]:
        return self.controls + self.generation_tasks


@dataclasses.dataclass(frozen=True)
class _CandidateRecord(Generic[CandidateT]):
    candidate: CandidateT
    purpose: TaskPurpose
    attempt_id: str
    input_position: int


def schedule_mutation_tasks(
    tasks: Sequence[MutationTask],
    *,
    offspring_count: int,
    policy: SchedulingPolicy | None = None,
) -> MutationTaskSchedule:
    """Select model tasks by purpose without charging controls to offspring.

    This is the generation-side half of the scheduling boundary.  Candidate
    scheduling later applies target-call limits after model output validation.
    Keeping both decisions under one policy prevents a purpose-aware evaluator
    from receiving a purpose-blind, prefix-truncated task portfolio.
    """

    resolved_policy = policy or SchedulingPolicy()
    if (
        isinstance(offspring_count, bool)
        or not isinstance(offspring_count, int)
        or offspring_count < 0
    ):
        raise ValueError("offspring_count must be a non-negative integer")
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("mutation task ids must be unique within an attempt")

    work = [task for task in tasks if task.purpose is not TaskPurpose.CONTROL]
    ordered_work = _weighted_task_order(work, resolved_policy)
    selected_work = ordered_work[:offspring_count]
    controls = [task for task in tasks if task.purpose is TaskPurpose.CONTROL]
    control_limit = (
        resolved_policy.max_controls_per_attempt
        if selected_work and resolved_policy.max_control_share > 0
        else 0
    )
    selected_controls = controls[:control_limit]
    selected_ids = {
        task.task_id for task in (*selected_controls, *selected_work)
    }
    deferred = [task for task in tasks if task.task_id not in selected_ids]
    deferred_tasks = []
    for task in deferred:
        if task.purpose is not TaskPurpose.CONTROL:
            reason = "offspring_quota"
        elif not selected_work:
            reason = "orphan_control"
        elif resolved_policy.max_control_share == 0:
            reason = "control_disabled"
        else:
            reason = "control_limit"
        deferred_tasks.append(
            {
                "task_id": task.task_id,
                "purpose": task.purpose.value,
                "operator_id": task.operator_id,
                "reason": reason,
            }
        )
    accepted_counts = Counter(task.purpose.value for task in tasks)
    selected_counts = Counter(
        task.purpose.value for task in (*selected_controls, *selected_work)
    )
    audit = {
        "schema_version": 1,
        "policy": resolved_policy.to_jsonable(),
        "offspring_count": offspring_count,
        "available": len(tasks),
        "selected": len(selected_controls) + len(selected_work),
        "selected_control_count": len(selected_controls),
        "selected_generation_count": len(selected_work),
        "by_purpose": {
            purpose.value: {
                "available": accepted_counts[purpose.value],
                "selected": selected_counts[purpose.value],
                "deferred": (
                    accepted_counts[purpose.value]
                    - selected_counts[purpose.value]
                ),
            }
            for purpose in TaskPurpose
        },
        "work_order": [task.task_id for task in ordered_work],
        "selected_task_ids": [
            task.task_id for task in (*selected_controls, *selected_work)
        ],
        "deferred_tasks": deferred_tasks,
        "invariants": {
            "controls_excluded_from_offspring_count": (
                len(selected_work) <= offspring_count
            ),
            "generation_tasks_are_not_controls": all(
                task.purpose is not TaskPurpose.CONTROL
                for task in selected_work
            ),
        },
    }
    return MutationTaskSchedule(
        controls=tuple(selected_controls),
        generation_tasks=tuple(selected_work),
        deferred=tuple(deferred),
        audit=audit,
    )


def schedule_candidates(
    candidates: Sequence[CandidateT],
    *,
    budget: int | None,
    policy: SchedulingPolicy | None = None,
) -> CandidateSchedule[CandidateT]:
    """Allocate validated candidates without allowing FIFO or control starvation.

    The returned order places controls first because the evaluator uses them as
    contemporaneous baselines.  Work selection itself is weighted by purpose,
    round-robin within each purpose, and gives each attempt one candidate before
    any attempt receives a second.  Unused allocations are always borrowed by
    another purpose, so the scheduler is work-conserving.
    """

    resolved_policy = policy or SchedulingPolicy()
    if budget is not None and (
        isinstance(budget, bool) or not isinstance(budget, int) or budget < 0
    ):
        raise ValueError("budget must be a non-negative integer or None")

    records = _candidate_records(candidates)
    work_records = [
        record for record in records if record.purpose is not TaskPurpose.CONTROL
    ]
    control_records = [
        record for record in records if record.purpose is TaskPurpose.CONTROL
    ]
    work_order = _ordered_work(work_records, resolved_policy)
    work_attempts = {record.attempt_id for record in work_records}

    eligible_controls, deferred_reasons = _eligible_controls(
        control_records,
        work_attempts=work_attempts,
        max_per_attempt=resolved_policy.max_controls_per_attempt,
    )
    schedulable_count = len(work_records) + len(eligible_controls)
    slot_count = (
        schedulable_count
        if budget is None
        else min(budget, schedulable_count)
    )

    attempt_count = len(work_attempts)
    minimum_work_slots = min(len(work_records), attempt_count, slot_count)
    control_share_limit = (
        max(1, math.floor(slot_count * resolved_policy.max_control_share))
        if resolved_policy.max_control_share > 0
        else 0
    )
    control_slot_count = min(
        len(eligible_controls),
        max(0, slot_count - minimum_work_slots),
        control_share_limit,
    )
    work_slot_count = min(len(work_order), slot_count - control_slot_count)
    selected_work = work_order[:work_slot_count]
    selected_attempts = {record.attempt_id for record in selected_work}
    selected_controls = [
        record
        for record in eligible_controls
        if record.attempt_id in selected_attempts
    ][:control_slot_count]

    if len(selected_controls) != control_slot_count:
        raise AssertionError(
            "control allocation violated the one-work-slot attempt floor"
        )

    selected_records = selected_controls + selected_work
    selected_ids = {record.candidate.id for record in selected_records}
    for record in eligible_controls:
        if record.candidate.id not in selected_ids:
            deferred_reasons[record.candidate.id] = "control_budget"
    for record in records:
        if record.candidate.id in selected_ids or record.candidate.id in deferred_reasons:
            continue
        deferred_reasons[record.candidate.id] = "budget"

    scheduled = tuple(record.candidate for record in selected_records)
    deferred = tuple(
        record.candidate
        for record in records
        if record.candidate.id not in selected_ids
    )
    audit = _schedule_audit(
        records=records,
        selected_records=selected_records,
        deferred_reasons=deferred_reasons,
        budget=budget,
        policy=resolved_policy,
        work_order=work_order,
        control_share_limit=control_share_limit,
    )
    return CandidateSchedule(
        scheduled=scheduled,
        deferred=deferred,
        audit=audit,
    )


def _candidate_records(
    candidates: Sequence[CandidateT],
) -> list[_CandidateRecord[CandidateT]]:
    records: list[_CandidateRecord[CandidateT]] = []
    seen_ids: set[str] = set()
    for position, candidate in enumerate(candidates):
        candidate_id = str(candidate.id)
        if candidate_id in seen_ids:
            raise ValueError(f"duplicate candidate id in scheduling input: {candidate_id}")
        seen_ids.add(candidate_id)
        metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
        refinement = metadata.get("refinement")
        if isinstance(refinement, Mapping):
            disposition = str(refinement.get("disposition") or "unknown")
            if disposition == "ranked_out":
                raise ValueError(
                    "pre-target ranked-out candidate cannot enter scheduling: "
                    f"{candidate_id}"
                )
            if disposition not in {"selected", "unknown"}:
                raise ValueError(
                    f"unknown refinement disposition for {candidate_id}: "
                    f"{disposition}"
                )
        purpose = _purpose_from_metadata(metadata, candidate_id=candidate_id)
        attempt_id = str(metadata.get("perturbation_attempt_id") or "").strip()
        if not attempt_id:
            attempt_id = str(candidate.parent_id or candidate_id)
        records.append(
            _CandidateRecord(
                candidate=candidate,
                purpose=purpose,
                attempt_id=attempt_id,
                input_position=position,
            )
        )
    return records


def _purpose_from_metadata(
    metadata: Mapping[str, Any],
    *,
    candidate_id: str,
) -> TaskPurpose:
    task_data = metadata.get("mutation_task")
    if isinstance(task_data, dict):
        task_purpose = MutationTask.from_jsonable(task_data).purpose
        explicit_purpose = metadata.get("task_purpose")
        if explicit_purpose is not None:
            try:
                parsed_explicit = _parse_purpose(explicit_purpose)
            except ValueError as exc:
                raise ValueError(
                    f"unknown task purpose for candidate {candidate_id}: "
                    f"{explicit_purpose}"
                ) from exc
            if parsed_explicit is not task_purpose:
                raise ValueError(
                    f"conflicting task purposes for candidate {candidate_id}: "
                    f"{parsed_explicit.value} != {task_purpose.value}"
                )
        return task_purpose

    raw_purpose = metadata.get("task_purpose", TaskPurpose.OPTIMIZATION)
    try:
        return _parse_purpose(raw_purpose)
    except ValueError as exc:
        raise ValueError(
            f"unknown task purpose for candidate {candidate_id}: {raw_purpose}"
        ) from exc


def _parse_purpose(value: Any) -> TaskPurpose:
    return value if isinstance(value, TaskPurpose) else TaskPurpose(str(value))


def _ordered_work(
    records: Sequence[_CandidateRecord[CandidateT]],
    policy: SchedulingPolicy,
) -> list[_CandidateRecord[CandidateT]]:
    """Return a purpose-weighted order with a one-per-attempt first round."""

    by_attempt: dict[str, list[_CandidateRecord[CandidateT]]] = defaultdict(list)
    for record in records:
        by_attempt[record.attempt_id].append(record)

    heads: list[_CandidateRecord[CandidateT]] = []
    remainder: list[_CandidateRecord[CandidateT]] = []
    for attempt_id in sorted(by_attempt):
        attempt_records = by_attempt[attempt_id]
        head = max(
            attempt_records,
            key=lambda record: (
                policy.weight(record.purpose),
                -PURPOSE_RANK[record.purpose],
                -record.input_position,
            ),
        )
        heads.append(head)
        remainder.extend(record for record in attempt_records if record is not head)

    return _weighted_interleave(heads, policy) + _weighted_interleave(
        remainder,
        policy,
    )


def _weighted_task_order(
    tasks: Sequence[MutationTask],
    policy: SchedulingPolicy,
) -> list[MutationTask]:
    queues = {
        purpose: deque(task for task in tasks if task.purpose is purpose)
        for purpose in WORK_PURPOSES
    }
    current = {purpose: 0.0 for purpose in WORK_PURPOSES}
    ordered: list[MutationTask] = []
    while True:
        active = [purpose for purpose in WORK_PURPOSES if queues[purpose]]
        if not active:
            return ordered
        chosen = _next_weighted_purpose(
            active=active,
            current=current,
            policy=policy,
        )
        ordered.append(queues[chosen].popleft())


def _weighted_interleave(
    records: Sequence[_CandidateRecord[CandidateT]],
    policy: SchedulingPolicy,
) -> list[_CandidateRecord[CandidateT]]:
    """Smooth weighted round-robin across purposes and attempts."""

    by_purpose_attempt: dict[
        TaskPurpose,
        dict[str, deque[_CandidateRecord[CandidateT]]],
    ] = defaultdict(lambda: defaultdict(deque))
    for record in records:
        by_purpose_attempt[record.purpose][record.attempt_id].append(record)

    attempt_cycles: dict[TaskPurpose, deque[str]] = {}
    for purpose, by_attempt in by_purpose_attempt.items():
        attempt_cycles[purpose] = deque(sorted(by_attempt))

    current = {purpose: 0.0 for purpose in WORK_PURPOSES}
    ordered: list[_CandidateRecord[CandidateT]] = []
    while True:
        active = [
            purpose
            for purpose in WORK_PURPOSES
            if attempt_cycles.get(purpose)
        ]
        if not active:
            return ordered
        chosen = _next_weighted_purpose(
            active=active,
            current=current,
            policy=policy,
        )

        attempt_id = attempt_cycles[chosen].popleft()
        queue = by_purpose_attempt[chosen][attempt_id]
        ordered.append(queue.popleft())
        if queue:
            attempt_cycles[chosen].append(attempt_id)


def _next_weighted_purpose(
    *,
    active: Sequence[TaskPurpose],
    current: dict[TaskPurpose, float],
    policy: SchedulingPolicy,
) -> TaskPurpose:
    total_weight = sum(policy.weight(purpose) for purpose in active)
    for purpose in active:
        current[purpose] += policy.weight(purpose)
    chosen = max(
        active,
        key=lambda purpose: (current[purpose], -PURPOSE_RANK[purpose]),
    )
    current[chosen] -= total_weight
    return chosen


def _eligible_controls(
    controls: Sequence[_CandidateRecord[CandidateT]],
    *,
    work_attempts: set[str],
    max_per_attempt: int,
) -> tuple[list[_CandidateRecord[CandidateT]], dict[str, str]]:
    by_attempt: dict[str, list[_CandidateRecord[CandidateT]]] = defaultdict(list)
    deferred_reasons: dict[str, str] = {}
    for record in controls:
        if record.attempt_id not in work_attempts:
            deferred_reasons[record.candidate.id] = "orphan_control"
            continue
        by_attempt[record.attempt_id].append(record)

    eligible: list[_CandidateRecord[CandidateT]] = []
    for attempt_id in sorted(by_attempt):
        attempt_controls = by_attempt[attempt_id]
        eligible.extend(attempt_controls[:max_per_attempt])
        for record in attempt_controls[max_per_attempt:]:
            deferred_reasons[record.candidate.id] = "control_limit"
    return eligible, deferred_reasons


def _schedule_audit(
    *,
    records: Sequence[_CandidateRecord[CandidateT]],
    selected_records: Sequence[_CandidateRecord[CandidateT]],
    deferred_reasons: Mapping[str, str],
    budget: int | None,
    policy: SchedulingPolicy,
    work_order: Sequence[_CandidateRecord[CandidateT]],
    control_share_limit: int,
) -> dict[str, Any]:
    selected_ids = {record.candidate.id for record in selected_records}
    accepted_counts = Counter(record.purpose.value for record in records)
    scheduled_counts = Counter(record.purpose.value for record in selected_records)
    deferred_counts = Counter(
        record.purpose.value
        for record in records
        if record.candidate.id not in selected_ids
    )
    by_purpose = {
        purpose.value: {
            "accepted": accepted_counts[purpose.value],
            "scheduled": scheduled_counts[purpose.value],
            "deferred": deferred_counts[purpose.value],
        }
        for purpose in TaskPurpose
    }
    scheduled_work_attempts = {
        record.attempt_id
        for record in selected_records
        if record.purpose is not TaskPurpose.CONTROL
    }
    scheduled_control_attempts = {
        record.attempt_id
        for record in selected_records
        if record.purpose is TaskPurpose.CONTROL
    }
    scheduled_work_count = sum(
        record.purpose is not TaskPurpose.CONTROL
        for record in selected_records
    )
    scheduled_control_count = len(selected_records) - scheduled_work_count
    available_work_attempts = {
        record.attempt_id
        for record in records
        if record.purpose is not TaskPurpose.CONTROL
    }
    fairness_floor = min(scheduled_work_count, len(available_work_attempts))
    return {
        "schema_version": 1,
        "policy": {
            **policy.to_jsonable(),
            "fairness": "one_per_attempt_then_weighted_round_robin",
            "control_admission": "paired_after_first_work_slot",
        },
        "budget": budget,
        "accepted": len(records),
        "scheduled": len(selected_records),
        "deferred": len(records) - len(selected_records),
        "allocation": {
            "available_work_attempts": len(available_work_attempts),
            "scheduled_work_attempts": len(scheduled_work_attempts),
            "scheduled_work": scheduled_work_count,
            "scheduled_controls": scheduled_control_count,
            "control_share_limit": control_share_limit,
            "realized_control_share": (
                round(scheduled_control_count / len(selected_records), 6)
                if selected_records
                else 0.0
            ),
        },
        "by_purpose": by_purpose,
        "work_order": [record.candidate.id for record in work_order],
        "accepted_candidate_ids": [record.candidate.id for record in records],
        "scheduled_candidate_ids": [
            record.candidate.id for record in selected_records
        ],
        "scheduled_candidates": [
            {
                "candidate_id": record.candidate.id,
                "attempt_id": record.attempt_id,
                "purpose": record.purpose.value,
            }
            for record in selected_records
        ],
        "deferred_candidates": [
            {
                "candidate_id": record.candidate.id,
                "attempt_id": record.attempt_id,
                "purpose": record.purpose.value,
                "reason": deferred_reasons.get(record.candidate.id, "budget"),
            }
            for record in records
            if record.candidate.id not in selected_ids
        ],
        "invariants": {
            "within_budget": budget is None or len(selected_records) <= budget,
            "controls_have_scheduled_work": scheduled_control_attempts
            <= scheduled_work_attempts,
            "attempt_fairness_floor_met": (
                len(scheduled_work_attempts) >= fairness_floor
            ),
            "control_share_limit_met": (
                scheduled_control_count <= control_share_limit
            ),
            "candidate_dispositions_reconciled": (
                len(records)
                == len(selected_records)
                + sum(
                    record.candidate.id not in selected_ids
                    for record in records
                )
            ),
            "all_deferred_candidates_have_reasons": all(
                record.candidate.id in deferred_reasons
                for record in records
                if record.candidate.id not in selected_ids
            ),
        },
    }
