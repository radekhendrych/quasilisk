"""Structural validation for planned perturbation candidates."""

from __future__ import annotations

import re
from types import ModuleType

from quasilisk_harness.perturbation.models import PerturbationPlan, ValidationResult
from quasilisk_harness.perturbation.space import CASE_KEYWORDS, MORAL_OBJECTIVES, ROUTE_KEYWORDS
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

DIRECT_ADJUDICATION_TERMS = (
    "assess",
    "evaluate",
    "do you accept",
    "does it follow",
    "answer directly",
    "state your",
    "final verdict",
    "considered conclusion",
)


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

    objective_valid, objective_reason = objective.validate_prompt(prompt)
    if objective_valid:
        rubric["objective_fit"] = 1.0
    elif objective_reason:
        reasons.append(objective_reason)

    if any(term in lower for term in DIRECT_ADJUDICATION_TERMS):
        rubric["directness"] = 1.0
    elif plan.objective_id in MORAL_OBJECTIVES:
        reasons.append("missing_direct_adjudication")

    route_hits = ROUTE_KEYWORDS.get(plan.route, ())
    case_hits = CASE_KEYWORDS.get(plan.case_family, ())
    route_aligned = any(term in lower for term in route_hits) if route_hits else True
    case_aligned = any(term in lower for term in case_hits) if case_hits else True
    if route_aligned:
        rubric["plan_alignment"] += 0.5
    if case_aligned:
        rubric["plan_alignment"] += 0.5

    if word_count(prompt) < 8:
        reasons.append("too_few_words")

    return ValidationResult(
        accepted=not reasons,
        reasons=reasons,
        rubric={key: round(value, 3) for key, value in rubric.items()},
    )
