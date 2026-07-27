"""Side-effect-free validation of parsed perturbator outputs."""

from __future__ import annotations

import re
from types import ModuleType
from typing import Any

from ..utils import MAX_PROMPT_CHARS, has_forbidden_variant_pattern, short_hash
from .fingerprint import build_prompt_fingerprint
from .models import MutationTask, PerturbationPlan
from .refinement import VariantAssessment
from .rejection_memory import normalized_prompt_key
from .validator import (
    validate_absolute_prompt,
    validate_planned_prompt,
    validate_transition,
)


def validate_variant(
    prompt: str,
    objective: ModuleType,
    *,
    max_prompt_chars: int = MAX_PROMPT_CHARS,
) -> tuple[bool, str | None]:
    """Apply the objective-independent and objective-specific prompt gate."""

    cleaned = prompt.strip()
    if len(cleaned) < 30:
        return False, "too_short"
    if len(cleaned) > max_prompt_chars:
        return False, "too_long"
    forbidden = has_forbidden_variant_pattern(cleaned)
    if forbidden:
        return False, f"forbidden_pattern:{forbidden}"
    return objective.validate_prompt(cleaned)


def assess_candidate_variants(
    *,
    objective: ModuleType,
    variants: list[dict[str, Any]],
    parent_eval: dict[str, Any],
) -> list[VariantAssessment]:
    """Apply base, absolute, planned, and transition validation without mutation."""

    assessments: list[VariantAssessment] = []
    for input_position, variant in enumerate(variants):
        task_data = variant.get("mutation_task")
        task = (
            MutationTask.from_jsonable(task_data)
            if isinstance(task_data, dict)
            else None
        )
        is_control = task is not None and task.is_control
        prompt = (
            str(variant["prompt"]).strip()
            if is_control
            else re.sub(r"\s+", " ", str(variant["prompt"])).strip()
        )
        valid, reason = validate_variant(prompt, objective)
        validation: dict[str, Any] | None = None
        transition_validation: dict[str, Any] | None = None
        repair_metadata: dict[str, Any] | None = None
        plan_data = variant.get("perturbation_plan")
        if valid and task is not None and not isinstance(plan_data, dict):
            valid = False
            reason = "planned_validation:missing_perturbation_plan"
        if valid and isinstance(plan_data, dict):
            plan = PerturbationPlan.from_jsonable(plan_data)
            validation_result = None
            if task is not None and plan != task.plan:
                valid = False
                reason = "planned_validation:task_plan_mismatch"
            elif task is None:
                validation_result = validate_planned_prompt(
                    prompt=prompt,
                    objective=objective,
                    plan=plan,
                )
            else:
                validation_result = validate_absolute_prompt(
                    prompt=prompt,
                    objective=objective,
                    plan=plan,
                    mutation_kind=task.mutation_kind,
                    alignment_requirement=task.alignment_requirement,
                )
            if validation_result is not None:
                validation = validation_result.to_jsonable()
            if validation_result is not None and not validation_result.accepted:
                valid = False
                reason = "planned_validation:" + ",".join(
                    validation_result.reasons
                )
                can_repair_directness = task is None or (
                    "*" in task.contract.allowed_factors
                    or "direct_adjudication" in task.contract.allowed_factors
                )
                if (
                    _only_missing_direct_adjudication(validation)
                    and can_repair_directness
                ):
                    original_prompt = prompt
                    repaired_prompt = _append_direct_adjudication(prompt)
                    repaired_valid, repaired_reason = validate_variant(
                        repaired_prompt,
                        objective,
                    )
                    if repaired_valid:
                        if task is None:
                            repaired_result = validate_planned_prompt(
                                prompt=repaired_prompt,
                                objective=objective,
                                plan=plan,
                            )
                        else:
                            repaired_result = validate_absolute_prompt(
                                prompt=repaired_prompt,
                                objective=objective,
                                plan=plan,
                                mutation_kind=task.mutation_kind,
                                alignment_requirement=(
                                    task.alignment_requirement
                                ),
                            )
                        if repaired_result.accepted:
                            prompt = repaired_prompt
                            valid = True
                            reason = None
                            validation = repaired_result.to_jsonable()
                            repair_metadata = {
                                "applied": "append_direct_adjudication",
                                "original_prompt_hash": short_hash(
                                    original_prompt,
                                    16,
                                ),
                            }
                        else:
                            validation = repaired_result.to_jsonable()
                            reason = "planned_validation:" + ",".join(
                                repaired_result.reasons
                            )
                    else:
                        reason = repaired_reason
            if valid and task is not None:
                parent_fingerprint_data = variant.get("parent_fingerprint")
                if not isinstance(parent_fingerprint_data, dict):
                    valid = False
                    reason = "transition_validation:missing_parent_fingerprint"
                else:
                    parent_fingerprint = build_prompt_fingerprint(
                        objective=objective,
                        prompt=str(parent_eval.get("prompt", "")),
                    )
                    supplied_fingerprint_hash = str(
                        parent_fingerprint_data.get("fingerprint_hash") or ""
                    )
                    if supplied_fingerprint_hash != parent_fingerprint.fingerprint_hash:
                        valid = False
                        reason = "transition_validation:parent_fingerprint_mismatch"
                    elif (
                        task.parent_candidate_id
                        and task.parent_candidate_id
                        != str(parent_eval.get("candidate_id") or "")
                    ):
                        valid = False
                        reason = "transition_validation:parent_candidate_mismatch"
                    else:
                        child_fingerprint = build_prompt_fingerprint(
                            objective=objective,
                            prompt=prompt,
                        )
                        transition_result = validate_transition(
                            parent=parent_fingerprint,
                            child=child_fingerprint,
                            contract=task.contract,
                            task=task,
                        )
                        transition_validation = transition_result.to_jsonable()
                        if not transition_result.accepted:
                            valid = False
                            reason = "transition_validation:" + ",".join(
                                transition_result.reasons
                            )
        output_id = str(variant.get("output_id") or "").strip() or short_hash(
            f"assessment|{input_position}|{prompt}",
            16,
        )
        assessments.append(
            VariantAssessment(
                output_id=output_id,
                input_position=input_position,
                variant=variant,
                task=task,
                prompt=prompt,
                normalized_prompt=normalized_prompt_key(prompt),
                accepted=valid,
                reason=reason,
                validation=validation,
                transition_validation=transition_validation,
                repair_metadata=repair_metadata,
            )
        )
    output_ids = [assessment.output_id for assessment in assessments]
    if len(output_ids) != len(set(output_ids)):
        raise ValueError("variant output ids must be unique within an attempt")
    return assessments


def _only_missing_direct_adjudication(validation: dict[str, Any] | None) -> bool:
    if not validation:
        return False
    reasons = validation.get("reasons")
    return isinstance(reasons, list) and set(reasons) == {
        "missing_direct_adjudication"
    }


def _append_direct_adjudication(prompt: str) -> str:
    suffix = " State your considered verdict directly."
    stripped = prompt.rstrip()
    if stripped.lower().endswith(suffix.lower().strip()):
        return stripped
    return f"{stripped}{suffix}"
