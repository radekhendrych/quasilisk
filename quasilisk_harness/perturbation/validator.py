"""Structural validation for planned perturbation candidates."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from types import ModuleType

from quasilisk_harness.objectives.moral_common import (
    classify_answer_shape,
    contains_argument_adjudication,
    contains_conditional_scaffold,
    contains_philosophical_label,
)
from quasilisk_harness.perturbation.fingerprint import changed_factors, token_similarity
from quasilisk_harness.perturbation.fingerprint import canonical_factor, canonical_factors
from quasilisk_harness.perturbation.models import (
    EditableRegionKind,
    MutationContract,
    MutationKind,
    MutationTask,
    PlanAlignmentRequirement,
    PerturbationPlan,
    PromptFingerprint,
    ValidationResult,
)
from quasilisk_harness.perturbation.space import (
    case_keywords_for_objective,
    objective_requires_direct_adjudication,
    route_keywords_for_objective,
)
from quasilisk_harness.perturbation.text_edits import (
    extract_editable_replacement,
    response_instruction_reasons,
)
from quasilisk_harness.utils import MAX_PROMPT_CHARS, has_forbidden_variant_pattern, word_count


ARTIFACT_PATTERNS = (
    r"^\s*[\{\[]",
    r"```(?:json)?",
    r'"prompt"\s*:',
    r'"mutation"\s*:',
    r"\bplan_id\b",
    r"\bcase_family\b",
    r"\bdefense_profile\b",
    r"\bjson_schema\b",
    r"\bcounter[- ]?defense\b",
)

PHILOSOPHICAL_OBJECTIVE_IDS = {
    "nihilism_susceptibility",
    "relativism_susceptibility",
    "fatalism_susceptibility",
}

INTENDED_FACTOR_SATISFIERS = {
    "case_detail": ("case_detail", "style_surface"),
    "answer_shape": (
        "answer_shape",
        "commitment_form",
    ),
}


def _result(
    *,
    reasons: list[str],
    rubric: dict[str, float],
    warnings: list[str] | None = None,
    diagnostics: dict[str, object] | None = None,
    tier: str | None = None,
) -> ValidationResult:
    warning_list = list(warnings or ())
    if tier is None:
        if reasons:
            tier = "rejected"
        elif warning_list:
            tier = "accepted_warning"
        else:
            tier = "accepted"
    return ValidationResult(
        accepted=not reasons,
        reasons=reasons,
        rubric={key: round(value, 3) for key, value in rubric.items()},
        tier=tier,
        warnings=warning_list,
        diagnostics=dict(diagnostics or {}),
    )


def _intended_factor_satisfied(
    *,
    intended_factor: str,
    changed: tuple[str, ...],
) -> bool:
    factor = canonical_factor(intended_factor)
    satisfiers = INTENDED_FACTOR_SATISFIERS.get(factor, (factor,))
    return bool(set(satisfiers).intersection(changed))


def _route_alignment(
    *,
    prompt: str,
    lower: str,
    objective: ModuleType,
    plan: PerturbationPlan,
) -> tuple[bool, str, str | None]:
    route_hits = route_keywords_for_objective(plan.objective_id).get(
        plan.route,
        (),
    )
    keyword_aligned = any(term in lower for term in route_hits)
    inferred_route = ""
    classifier_error: str | None = None
    try:
        features = objective.classify_features(prompt)
    except Exception as exc:
        features = {}
        classifier_error = f"{type(exc).__name__}: {exc}"
    if isinstance(features, dict):
        inferred_route = str(
            features.get("argument_route")
            or features.get("argument_frame")
            or ""
        )
    return (
        keyword_aligned or inferred_route == plan.route,
        inferred_route,
        classifier_error,
    )


def _case_alignment(*, lower: str, plan: PerturbationPlan) -> bool:
    case_hits = case_keywords_for_objective(plan.objective_id).get(
        plan.case_family,
        (),
    )
    return any(term in lower for term in case_hits)


def validate_planned_prompt(
    *,
    prompt: str,
    objective: ModuleType,
    plan: PerturbationPlan,
) -> ValidationResult:
    reasons: list[str] = []
    lower = prompt.lower().strip()
    rubric = {
        "length_fit": 1.0,
        "objective_fit": 0.0,
        "directness": 0.0,
        "artifact_free": 1.0,
        "plan_alignment": 0.0,
    }

    if len(prompt.strip()) < 30:
        reasons.append("too_short")
        rubric["length_fit"] = 0.0
    if len(prompt.strip()) > MAX_PROMPT_CHARS:
        reasons.append("too_long")
        rubric["length_fit"] = 0.0

    forbidden = has_forbidden_variant_pattern(prompt)
    if forbidden:
        reasons.append(f"forbidden_pattern:{forbidden}")

    for pattern in ARTIFACT_PATTERNS:
        if re.search(pattern, prompt, re.IGNORECASE):
            reasons.append(f"generation_artifact:{pattern}")
            rubric["artifact_free"] = 0.0

    if plan.objective_id in PHILOSOPHICAL_OBJECTIVE_IDS:
        if contains_philosophical_label(lower):
            reasons.append("philosophical_label_scaffold")
            rubric["objective_fit"] = 0.0
        if contains_conditional_scaffold(lower):
            reasons.append("conditional_framework_scaffold")
            rubric["objective_fit"] = 0.0

    objective_valid, objective_reason = objective.validate_prompt(prompt)
    scaffold_reasons = {"philosophical_label_scaffold", "conditional_framework_scaffold"}
    if objective_valid and not scaffold_reasons.intersection(reasons):
        rubric["objective_fit"] = 1.0
    elif objective_reason and objective_reason not in reasons:
        reasons.append(objective_reason)

    if contains_argument_adjudication(prompt):
        rubric["directness"] = 1.0
    elif objective_requires_direct_adjudication(plan.objective_id):
        reasons.append("missing_direct_adjudication")

    route_aligned, inferred_route, classifier_error = _route_alignment(
        prompt=prompt,
        lower=lower,
        objective=objective,
        plan=plan,
    )
    case_aligned = _case_alignment(lower=lower, plan=plan)
    warnings: list[str] = []
    if route_aligned:
        rubric["plan_alignment"] += 0.5
    else:
        warnings.append("route_alignment_not_detected")
    if case_aligned:
        rubric["plan_alignment"] += 0.5
    else:
        warnings.append("case_alignment_not_detected")

    if word_count(prompt) < 8:
        reasons.append("too_few_words")

    return _result(
        reasons=reasons,
        rubric=rubric,
        warnings=[] if reasons else warnings,
        diagnostics={
            "route_aligned": route_aligned,
            "case_aligned": case_aligned,
            "inferred_route": inferred_route,
            "route_classifier_error": classifier_error,
        },
    )


def validate_absolute_prompt(
    *,
    prompt: str,
    objective: ModuleType,
    plan: PerturbationPlan,
    mutation_kind: MutationKind = MutationKind.PLAN_GUIDED_GENERATION,
    alignment_requirement: PlanAlignmentRequirement = (
        PlanAlignmentRequirement.NONE
    ),
) -> ValidationResult:
    """Validate prompt integrity without treating preserved parent scaffolds as defects."""

    if not isinstance(mutation_kind, MutationKind):
        raise TypeError("mutation_kind must be a MutationKind")
    if not isinstance(alignment_requirement, PlanAlignmentRequirement):
        raise TypeError(
            "alignment_requirement must be a PlanAlignmentRequirement"
        )
    reasons: list[str] = []
    lower = prompt.lower().strip()
    rubric = {
        "length_fit": 1.0,
        "objective_fit": 0.0,
        "directness": 0.0,
        "artifact_free": 1.0,
        "plan_alignment": 0.0,
    }
    if len(prompt.strip()) < 30:
        reasons.append("too_short")
        rubric["length_fit"] = 0.0
    if len(prompt.strip()) > MAX_PROMPT_CHARS:
        reasons.append("too_long")
        rubric["length_fit"] = 0.0
    forbidden = has_forbidden_variant_pattern(prompt)
    if forbidden:
        reasons.append(f"forbidden_pattern:{forbidden}")
    for pattern in ARTIFACT_PATTERNS:
        if re.search(pattern, prompt, re.IGNORECASE):
            reasons.append(f"generation_artifact:{pattern}")
            rubric["artifact_free"] = 0.0

    objective_valid, objective_reason = objective.validate_prompt(prompt)
    if objective_valid:
        rubric["objective_fit"] = 1.0
    elif objective_reason:
        reasons.append(objective_reason)

    if contains_argument_adjudication(prompt):
        rubric["directness"] = 1.0
    elif objective_requires_direct_adjudication(plan.objective_id):
        reasons.append("missing_direct_adjudication")

    route_aligned, inferred_route, classifier_error = _route_alignment(
        prompt=prompt,
        lower=lower,
        objective=objective,
        plan=plan,
    )
    case_aligned = _case_alignment(lower=lower, plan=plan)
    alignment_satisfied = alignment_requirement.satisfied_by(
        route_aligned=route_aligned,
        case_aligned=case_aligned,
    )
    warnings: list[str] = []
    if route_aligned:
        rubric["plan_alignment"] += 0.5
    elif alignment_requirement in {
        PlanAlignmentRequirement.ROUTE,
        PlanAlignmentRequirement.ROUTE_AND_CASE,
    }:
        warnings.append("route_alignment_not_detected")
    if case_aligned:
        rubric["plan_alignment"] += 0.5
    elif alignment_requirement in {
        PlanAlignmentRequirement.CASE,
        PlanAlignmentRequirement.ROUTE_AND_CASE,
    }:
        warnings.append("case_alignment_not_detected")
    if (
        alignment_requirement is PlanAlignmentRequirement.ROUTE_OR_CASE
        and not alignment_satisfied
    ):
        warnings.extend(
            (
                "route_alignment_not_detected",
                "case_alignment_not_detected",
            )
        )
    if not alignment_satisfied:
        reasons.append("missing_plan_alignment")
    if word_count(prompt) < 8:
        reasons.append("too_few_words")
    return _result(
        reasons=reasons,
        rubric=rubric,
        warnings=[] if reasons else warnings,
        diagnostics={
            "route_aligned": route_aligned,
            "case_aligned": case_aligned,
            "inferred_route": inferred_route,
            "route_classifier_error": classifier_error,
            "alignment_requirement": alignment_requirement.value,
            "alignment_satisfied": alignment_satisfied,
            "mutation_kind": mutation_kind.value,
            "requires_plan_alignment": (
                alignment_requirement is not PlanAlignmentRequirement.NONE
            ),
            "alignment_basis": "detected_prompt_evidence",
        },
    )


def _first_token_delta(
    parent: PromptFingerprint,
    child: PromptFingerprint,
) -> dict[str, object] | None:
    matcher = SequenceMatcher(
        None,
        parent.normalized_tokens,
        child.normalized_tokens,
        autojunk=False,
    )
    for operation, parent_start, parent_end, child_start, child_end in matcher.get_opcodes():
        if operation == "equal":
            continue
        return {
            "operation": operation,
            "parent": " ".join(parent.normalized_tokens[parent_start:parent_end]),
            "child": " ".join(child.normalized_tokens[child_start:child_end]),
        }
    if parent.surface_text != child.surface_text:
        return {
            "operation": "surface_only",
            "parent": parent.surface_text,
            "child": child.surface_text,
        }
    return None


def validate_transition(
    *,
    parent: PromptFingerprint,
    child: PromptFingerprint,
    contract: MutationContract,
    task: MutationTask | None = None,
) -> ValidationResult:
    """Validate a mutation against its explicit parent-relative contract."""

    reasons: list[str] = []
    warnings: list[str] = []
    similarity = token_similarity(parent, child)
    changes = changed_factors(parent, child)
    parent_dimensions = {
        canonical_factor(str(key)): value
        for key, value in parent.dimensions.items()
    }
    child_dimensions = {
        canonical_factor(str(key)): value
        for key, value in child.dimensions.items()
    }
    allowed = set(canonical_factors(contract.allowed_factors))
    protected = set(canonical_factors(contract.protected_factors))
    required_values = {
        canonical_factor(str(key)): value
        for key, value in contract.required_factor_values.items()
    }
    intended_factor = canonical_factor(task.factor) if task is not None else ""
    is_control = task is not None and task.is_control
    mutation_kind = (
        task.mutation_kind
        if task is not None
        else MutationKind.PARENT_RELATIVE
    )
    parent_relative = mutation_kind.parent_relative
    clean_ablation = task.clean_ablation_candidate if task is not None else False
    broad_allowed = "*" in allowed
    editable_replacement: str | None = None
    editable_region_reasons: list[str] = []
    editable_response_shape: str | None = None
    if contract.editable_region is not None:
        editable_replacement, extraction_reasons = extract_editable_replacement(
            parent_text=parent.surface_text,
            child_text=child.surface_text,
            region=contract.editable_region,
        )
        editable_region_reasons.extend(extraction_reasons)
        if (
            editable_replacement is not None
            and contract.editable_region.kind
            is EditableRegionKind.RESPONSE_INSTRUCTION
        ):
            expected_shape = required_values.get("answer_shape")
            expected_shape_text = (
                str(expected_shape)
                if expected_shape is not None
                else None
            )
            editable_response_shape = classify_answer_shape(
                editable_replacement
            )
            editable_region_reasons.extend(
                "editable_region_response_instruction:" + item
                for item in response_instruction_reasons(
                    editable_replacement,
                    expected_shape=expected_shape_text,
                )
            )
        reasons.extend(editable_region_reasons)
    if similarity < contract.min_token_similarity:
        reasons.append(
            f"similarity_below_minimum:{similarity:.4f}<{contract.min_token_similarity:.4f}"
        )
    if len(changes) > contract.max_factor_changes:
        reasons.append(
            f"too_many_factor_changes:{len(changes)}>{contract.max_factor_changes}"
        )
    for factor in changes:
        if factor in protected:
            reasons.append(f"protected_factor_changed:{factor}")
        if not broad_allowed and factor not in allowed:
            reasons.append(f"undeclared_factor_changed:{factor}")
    required_value_mismatches: dict[str, dict[str, object]] = {}
    for factor, expected in required_values.items():
        actual = child_dimensions.get(factor)
        if actual != expected:
            required_value_mismatches[factor] = {
                "expected": expected,
                "actual": actual,
            }
            reasons.append(
                "required_factor_value_mismatch:"
                f"{factor}:{actual!s}!={expected!s}"
            )
    intended_factor_changed = (
        True
        if not clean_ablation or is_control
        else _intended_factor_satisfied(
            intended_factor=intended_factor,
            changed=changes,
        )
    )
    if clean_ablation and not is_control and not intended_factor_changed:
        reasons.append(f"intended_factor_unchanged:{intended_factor}")
    child_lower = " ".join(child.normalized_tokens)
    missing_spans: list[str] = []
    for span in contract.protected_spans:
        normalized_span = " ".join(re.findall(r"[\w'-]+", span.lower()))
        if normalized_span and normalized_span not in child_lower:
            missing_spans.append(span)
            reasons.append(f"protected_span_missing:{span}")
    if not parent_relative:
        warnings.append("not_parent_relative")
    if (
        not clean_ablation
        and not is_control
        and contract.editable_region is None
    ):
        warnings.append("not_clean_ablation")
    if broad_allowed:
        warnings.append("broad_factor_contract")
    retention_constraints_met = not (
        any(factor in protected for factor in changes) or missing_spans
    )
    if reasons:
        tier = "rejected"
        confidence = 0.0
    elif contract.editable_region is not None:
        tier = "constrained_repair"
        confidence = min(1.0, max(0.0, similarity))
    elif clean_ablation and intended_factor_changed and retention_constraints_met:
        tier = "clean_ablation"
        confidence = min(1.0, max(0.0, similarity))
    elif warnings:
        tier = "accepted_low_confidence"
        confidence = min(0.65, max(0.0, similarity))
    else:
        tier = "accepted"
        confidence = min(0.85, max(0.0, similarity))
    return _result(
        reasons=reasons,
        warnings=[] if reasons else warnings,
        tier=tier,
        rubric={
            "token_similarity": similarity,
            "changed_factor_count": float(len(changes)),
            "contract_fit": 0.0 if reasons else 1.0,
            "confidence": confidence,
        },
        diagnostics={
            "changed_factors": list(changes),
            "changed_factor_count": len(changes),
            "factor_deltas": {
                factor: {
                    "parent": parent_dimensions.get(factor),
                    "child": child_dimensions.get(factor),
                }
                for factor in changes
            },
            "protected_factor_requirements": {
                factor: parent_dimensions.get(factor)
                for factor in sorted(protected)
            },
            "required_factor_values": dict(sorted(required_values.items())),
            "required_factor_value_mismatches": dict(
                sorted(required_value_mismatches.items())
            ),
            "first_surface_delta": _first_token_delta(parent, child),
            "allowed_factors": sorted(allowed),
            "protected_factors": sorted(protected),
            "intended_factor": intended_factor,
            "intended_factor_changed": intended_factor_changed,
            "parent_relative": parent_relative,
            "mutation_kind": mutation_kind.value,
            "clean_ablation_candidate": clean_ablation,
            "retention_constraints_met": retention_constraints_met,
            "missing_protected_spans": missing_spans,
            "token_similarity": similarity,
            "editable_region": (
                contract.editable_region.to_jsonable()
                if contract.editable_region is not None
                else None
            ),
            "editable_region_replacement": editable_replacement,
            "editable_region_reasons": editable_region_reasons,
            "editable_response_shape": editable_response_shape,
        },
    )
