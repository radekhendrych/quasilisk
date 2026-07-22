"""Deterministic pre-target refinement of perturbator proposals.

The model may propose several implementations of one mutation task.  This
module ranks those proposals using validator evidence only, nominates at most
one child per task, and identifies tasks that need one bounded revision.  It
does not call models, validate objective semantics, mutate run state, or spend
target-call budget.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from ..utils import short_hash
from .models import MutationTask, TaskPurpose
from .rejection_memory import normalized_prompt_key


MAX_VARIANTS_PER_TASK = 8
TRANSITION_TIER_RANK = {
    "clean_ablation": 4.0,
    "accepted": 3.0,
    "accepted_low_confidence": 2.0,
    "accepted_warning": 1.0,
    "unknown": 0.0,
    "rejected": -1.0,
}
ABSOLUTE_TIER_RANK = {
    "accepted": 2.0,
    "accepted_warning": 1.0,
    "unknown": 0.0,
    "rejected": -1.0,
}


@dataclasses.dataclass(frozen=True)
class RefinementPolicy:
    """Immutable controls for model-side oversampling and bounded revision."""

    variants_per_task: int = 2
    revision_enabled: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.variants_per_task, bool)
            or not isinstance(self.variants_per_task, int)
        ):
            raise TypeError("variants_per_task must be an integer")
        if not 1 <= self.variants_per_task <= MAX_VARIANTS_PER_TASK:
            raise ValueError(
                "variants_per_task must be between 1 and "
                f"{MAX_VARIANTS_PER_TASK}"
            )
        if not isinstance(self.revision_enabled, bool):
            raise TypeError("revision_enabled must be a boolean")

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "variants_per_task": self.variants_per_task,
            "revision_enabled": self.revision_enabled,
        }

    @classmethod
    def from_config(cls, data: Any) -> "RefinementPolicy":
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise TypeError("perturbation_refinement must be a JSON object")
        unknown = set(data) - {"variants_per_task", "revision_enabled"}
        if unknown:
            raise ValueError(
                "unknown perturbation_refinement field(s): "
                + ", ".join(sorted(str(key) for key in unknown))
            )
        defaults = cls()
        return cls(
            variants_per_task=data.get(
                "variants_per_task",
                defaults.variants_per_task,
            ),
            revision_enabled=data.get(
                "revision_enabled",
                defaults.revision_enabled,
            ),
        )


@dataclasses.dataclass(frozen=True)
class VariantAssessment:
    """Pure validation evidence for one parsed perturbator output."""

    output_id: str
    input_position: int
    variant: dict[str, Any]
    task: MutationTask | None
    prompt: str
    normalized_prompt: str
    accepted: bool
    reason: str | None
    validation: dict[str, Any] | None = None
    transition_validation: dict[str, Any] | None = None
    repair_metadata: dict[str, Any] | None = None

    @property
    def task_id(self) -> str:
        return self.task.task_id if self.task is not None else ""


@dataclasses.dataclass(frozen=True)
class AssessmentDisposition:
    """Terminal pre-target disposition for one assessed output."""

    assessment: VariantAssessment
    disposition: str
    reason: str | None
    rank_position: int | None = None
    rank_evidence: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class RefinementDecision:
    """One deterministic nomination decision with complete reconciliation."""

    selected: tuple[VariantAssessment, ...]
    deferred: tuple[VariantAssessment, ...]
    rejected: tuple[AssessmentDisposition, ...]
    dispositions: tuple[AssessmentDisposition, ...]
    missing_task_ids: tuple[str, ...]
    audit: dict[str, Any]

    @property
    def validated(self) -> tuple[VariantAssessment, ...]:
        return self.selected + self.deferred


def select_refined_assessments(
    assessments: Sequence[VariantAssessment],
    *,
    tasks: Sequence[MutationTask],
    seen_prompts: set[str] | None = None,
    rejected_prompts: set[str] | None = None,
) -> RefinementDecision:
    """Nominate one validator-preferred, unique prompt for each mutation task."""

    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("refinement task ids must be unique within an attempt")
    known_task_ids = set(task_ids)
    assessment_ids = [assessment.output_id for assessment in assessments]
    if len(assessment_ids) != len(set(assessment_ids)):
        raise ValueError("refinement assessment output ids must be unique")
    if any(assessment.task is None for assessment in assessments):
        raise ValueError("task-backed refinement does not accept unscoped outputs")
    tasks_by_id = {task.task_id: task for task in tasks}
    foreign_task_ids = sorted(
        {
            assessment.task_id
            for assessment in assessments
            if assessment.task_id and assessment.task_id not in known_task_ids
        }
    )
    if foreign_task_ids:
        raise ValueError(
            "assessment references task(s) outside refinement scope: "
            + ", ".join(foreign_task_ids)
        )
    mismatched_task_ids = sorted(
        {
            assessment.task_id
            for assessment in assessments
            if assessment.task is not None
            and assessment.task != tasks_by_id[assessment.task_id]
        }
    )
    if mismatched_task_ids:
        raise ValueError(
            "assessment task metadata differs from refinement scope: "
            + ", ".join(mismatched_task_ids)
        )

    existing_prompts = {
        normalized_prompt_key(str(prompt))
        for prompt in (seen_prompts or set())
        if str(prompt).strip()
    }
    rejected_prompt_keys = {
        normalized_prompt_key(str(prompt))
        for prompt in (rejected_prompts or set())
        if str(prompt).strip()
    }
    by_task: dict[str, list[VariantAssessment]] = {
        task.task_id: [] for task in tasks
    }
    for assessment in assessments:
        by_task[assessment.task_id].append(assessment)

    selected: list[VariantAssessment] = []
    deferred: list[VariantAssessment] = []
    rejected: list[AssessmentDisposition] = []
    dispositions: list[AssessmentDisposition] = []
    selected_prompt_keys: set[str] = set()

    for task in tasks:
        task_dispositions = _decide_task_assessments(
            task,
            by_task[task.task_id],
            existing_prompts=existing_prompts,
            rejected_prompt_keys=rejected_prompt_keys,
            selected_prompt_keys=selected_prompt_keys,
        )
        for disposition in task_dispositions:
            dispositions.append(disposition)
            if disposition.disposition == "selected":
                selected.append(disposition.assessment)
                if not task.is_control:
                    selected_prompt_keys.add(
                        normalized_prompt_key(
                            disposition.assessment.normalized_prompt
                        )
                    )
            elif disposition.disposition == "ranked_out":
                deferred.append(disposition.assessment)
            else:
                rejected.append(disposition)

    selected_task_ids = {
        assessment.task_id for assessment in selected if assessment.task_id
    }
    missing_task_ids = tuple(
        task.task_id
        for task in tasks
        if not task.is_control and task.task_id not in selected_task_ids
    )
    audit = _decision_audit(
        assessments=assessments,
        tasks=tasks,
        dispositions=dispositions,
        selected=selected,
        deferred=deferred,
        rejected=rejected,
        missing_task_ids=missing_task_ids,
    )
    return RefinementDecision(
        selected=tuple(selected),
        deferred=tuple(deferred),
        rejected=tuple(rejected),
        dispositions=tuple(dispositions),
        missing_task_ids=missing_task_ids,
        audit=audit,
    )


def revision_feedback(
    decision: RefinementDecision,
    *,
    tasks: Sequence[MutationTask],
) -> list[dict[str, Any]]:
    """Build compact validator feedback for tasks without a nominee."""

    missing = set(decision.missing_task_ids)
    feedback: list[dict[str, Any]] = []
    for task in tasks:
        if task.task_id not in missing:
            continue
        task_dispositions = [
            item
            for item in decision.dispositions
            if item.assessment.task_id == task.task_id
        ]
        reasons = Counter(
            str(item.reason or item.assessment.reason or "unknown")
            for item in task_dispositions
        )
        if not task_dispositions:
            reasons["missing_model_output"] += 1
        diagnostic_items = [
            dict(item.assessment.transition_validation.get("diagnostics") or {})
            for item in task_dispositions
            if isinstance(item.assessment.transition_validation, dict)
        ]
        required_values: dict[str, Any] = {}
        for diagnostics in diagnostic_items:
            values = diagnostics.get("protected_factor_requirements")
            if isinstance(values, dict):
                required_values.update(values)
        feedback.append(
            {
                "task_id": task.task_id,
                "intended_factor": task.factor,
                "failure_reasons": dict(sorted(reasons.items())),
                "required_protected_values": dict(sorted(required_values.items())),
                "failed_outputs": [
                    {
                        "output_id": item.assessment.output_id,
                        "prompt_hash": short_hash(item.assessment.prompt, 16),
                        "prompt": item.assessment.prompt,
                        "reason": item.reason or item.assessment.reason,
                        "changed_factors": list(
                            (
                                (
                                    item.assessment.transition_validation or {}
                                ).get("diagnostics")
                                or {}
                            ).get("changed_factors", [])
                        ),
                        "factor_deltas": dict(
                            (
                                (
                                    item.assessment.transition_validation or {}
                                ).get("diagnostics")
                                or {}
                            ).get("factor_deltas", {})
                        ),
                        "first_surface_delta": (
                            (
                                (
                                    item.assessment.transition_validation or {}
                                ).get("diagnostics")
                                or {}
                            ).get("first_surface_delta")
                        ),
                    }
                    for item in task_dispositions[:3]
                ],
                "revision_instruction": (
                    f"Return one distinct replacement that changes the declared "
                    f"factor {task.factor!r}, preserves every required factor value, "
                    "and resolves every listed failure. For answer_shape, change "
                    "the semantic response structure rather than only substituting "
                    "synonyms. Do not repeat any failed prompt."
                ),
            }
        )
    return feedback


def _decide_task_assessments(
    task: MutationTask,
    assessments: Sequence[VariantAssessment],
    *,
    existing_prompts: set[str],
    rejected_prompt_keys: set[str],
    selected_prompt_keys: set[str],
) -> list[AssessmentDisposition]:
    dispositions: list[AssessmentDisposition] = []
    rankable: list[VariantAssessment] = []
    for assessment in assessments:
        if not assessment.accepted:
            dispositions.append(
                AssessmentDisposition(
                    assessment=assessment,
                    disposition="rejected",
                    reason=assessment.reason or "validation_rejected",
                )
            )
            continue
        prompt_key = normalized_prompt_key(assessment.normalized_prompt)
        if not task.is_control and prompt_key in rejected_prompt_keys:
            dispositions.append(
                AssessmentDisposition(
                    assessment=assessment,
                    disposition="rejected",
                    reason="duplicate_rejected",
                )
            )
            continue
        if not task.is_control and prompt_key in existing_prompts:
            dispositions.append(
                AssessmentDisposition(
                    assessment=assessment,
                    disposition="rejected",
                    reason="duplicate_existing",
                )
            )
            continue
        rankable.append(assessment)

    ranked = sorted(rankable, key=_rank_sort_key)
    winner_found = False
    observed_prompt_keys: set[str] = set()
    for rank_position, assessment in enumerate(ranked, start=1):
        rank_evidence = assessment_rank_evidence(assessment)
        prompt_key = normalized_prompt_key(assessment.normalized_prompt)
        if not task.is_control and prompt_key in observed_prompt_keys:
            disposition = AssessmentDisposition(
                assessment=assessment,
                disposition="rejected",
                reason="duplicate_task_output",
                rank_position=rank_position,
                rank_evidence=rank_evidence,
            )
        elif not task.is_control and prompt_key in selected_prompt_keys:
            disposition = AssessmentDisposition(
                assessment=assessment,
                disposition="rejected",
                reason="duplicate_selected_output",
                rank_position=rank_position,
                rank_evidence=rank_evidence,
            )
        elif not winner_found:
            disposition = AssessmentDisposition(
                assessment=assessment,
                disposition="selected",
                reason=None,
                rank_position=rank_position,
                rank_evidence=rank_evidence,
            )
            winner_found = True
        else:
            disposition = AssessmentDisposition(
                assessment=assessment,
                disposition="ranked_out",
                reason="lower_pre_target_rank",
                rank_position=rank_position,
                rank_evidence=rank_evidence,
            )
        dispositions.append(disposition)
        observed_prompt_keys.add(prompt_key)
    return dispositions


def assessment_rank_evidence(
    assessment: VariantAssessment,
) -> dict[str, Any]:
    """Return the transparent evidence used for deterministic ranking."""

    transition = assessment.transition_validation or {}
    transition_rubric = transition.get("rubric") or {}
    validation = assessment.validation or {}
    validation_rubric = validation.get("rubric") or {}
    validation_warnings = validation.get("warnings") or []
    transition_warnings = transition.get("warnings") or []
    return {
        "task_purpose": (
            assessment.task.purpose.value
            if assessment.task is not None
            else "unscoped"
        ),
        "transition_tier": str(transition.get("tier") or "unknown"),
        "transition_tier_rank": TRANSITION_TIER_RANK.get(
            str(transition.get("tier") or "unknown"),
            0.0,
        ),
        "transition_confidence": _finite_float(
            transition_rubric.get("confidence")
        ),
        "contract_fit": _finite_float(transition_rubric.get("contract_fit")),
        "token_similarity": _finite_float(
            transition_rubric.get("token_similarity")
        ),
        "novelty": 1.0
        - _finite_float(transition_rubric.get("token_similarity")),
        "absolute_tier": str(validation.get("tier") or "unknown"),
        "absolute_tier_rank": ABSOLUTE_TIER_RANK.get(
            str(validation.get("tier") or "unknown"),
            0.0,
        ),
        "objective_fit": _finite_float(validation_rubric.get("objective_fit")),
        "directness": _finite_float(validation_rubric.get("directness")),
        "artifact_free": _finite_float(validation_rubric.get("artifact_free")),
        "plan_alignment": _finite_float(validation_rubric.get("plan_alignment")),
        "length_fit": _finite_float(validation_rubric.get("length_fit")),
        "warning_count": len(validation_warnings) + len(transition_warnings),
        "native_output": assessment.repair_metadata is None,
        "stable_prompt_hash": short_hash(assessment.prompt, 16),
    }


def _rank_sort_key(assessment: VariantAssessment) -> tuple[Any, ...]:
    evidence = assessment_rank_evidence(assessment)
    purpose = (
        assessment.task.purpose
        if assessment.task is not None
        else TaskPurpose.OPTIMIZATION
    )
    if purpose is TaskPurpose.DISCOVERY:
        descending = (
            evidence["absolute_tier_rank"],
            evidence["objective_fit"],
            evidence["directness"],
            evidence["artifact_free"],
            evidence["plan_alignment"],
            evidence["length_fit"],
            evidence["contract_fit"],
            evidence["novelty"],
            float(evidence["native_output"]),
            -float(evidence["warning_count"]),
        )
    else:
        descending = (
            evidence["transition_tier_rank"],
            evidence["transition_confidence"],
            evidence["contract_fit"],
            evidence["absolute_tier_rank"],
            evidence["objective_fit"],
            evidence["directness"],
            evidence["artifact_free"],
            evidence["plan_alignment"],
            evidence["length_fit"],
            float(evidence["native_output"]),
            -float(evidence["warning_count"]),
            evidence["token_similarity"],
        )
    return (
        *(-float(value) for value in descending),
        evidence["stable_prompt_hash"],
        assessment.output_id,
    )


def _finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number or number in {float("inf"), float("-inf")}:
        return 0.0
    return number


def _decision_audit(
    *,
    assessments: Sequence[VariantAssessment],
    tasks: Sequence[MutationTask],
    dispositions: Sequence[AssessmentDisposition],
    selected: Sequence[VariantAssessment],
    deferred: Sequence[VariantAssessment],
    rejected: Sequence[AssessmentDisposition],
    missing_task_ids: Sequence[str],
) -> dict[str, Any]:
    disposition_counts = Counter(item.disposition for item in dispositions)
    by_task: dict[str, dict[str, Any]] = {}
    for task in tasks:
        items = [
            item
            for item in dispositions
            if item.assessment.task_id == task.task_id
        ]
        by_task[task.task_id] = {
            "purpose": task.purpose.value,
            "operator_id": task.operator_id,
            "assessed": len(items),
            "selected_output_ids": [
                item.assessment.output_id
                for item in items
                if item.disposition == "selected"
            ],
            "ranked_outputs": [
                {
                    "output_id": item.assessment.output_id,
                    "disposition": item.disposition,
                    "reason": item.reason,
                    "rank_position": item.rank_position,
                    "rank_evidence": item.rank_evidence,
                }
                for item in sorted(
                    items,
                    key=lambda item: (
                        item.rank_position is None,
                        item.rank_position or 0,
                        item.assessment.input_position,
                    ),
                )
            ],
        }
    assessed_ids = [assessment.output_id for assessment in assessments]
    disposition_ids = [item.assessment.output_id for item in dispositions]
    return {
        "schema_version": 1,
        "selection_rule": "one_validator_ranked_nominee_per_task",
        "assessed": len(assessments),
        "validated": len(selected) + len(deferred),
        "selected": len(selected),
        "ranked_out": len(deferred),
        "rejected": len(rejected),
        "by_disposition": dict(sorted(disposition_counts.items())),
        "missing_task_ids": list(missing_task_ids),
        "selected_output_ids": [item.output_id for item in selected],
        "ranked_out_output_ids": [item.output_id for item in deferred],
        "by_task": by_task,
        "invariants": {
            "all_assessments_reconciled": (
                len(assessments) == len(dispositions)
                and Counter(assessed_ids) == Counter(disposition_ids)
            ),
            "validated_reconciled": (
                len(selected) + len(deferred)
                == len(assessments) - len(rejected)
            ),
            "at_most_one_nominee_per_task": all(
                sum(item.task_id == task.task_id for item in selected) <= 1
                for task in tasks
            ),
            "selected_prompts_unique": (
                len(
                    {
                        item.normalized_prompt
                        for item in selected
                        if item.task is None or not item.task.is_control
                    }
                )
                == sum(
                    item.task is None or not item.task.is_control
                    for item in selected
                )
            ),
        },
    }
