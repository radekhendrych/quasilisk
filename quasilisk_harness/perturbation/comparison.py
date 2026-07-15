"""Quality-aware parent/child score comparisons."""

from __future__ import annotations

from typing import Any

from quasilisk_harness.perturbation.models import ComparisonResult


DEFAULT_RETENTION_COMPONENTS = (
    "raw_judge_risk_score",
    "argument_success_score",
    "commitment_evidence_score",
)


def score_quality(evaluation: dict[str, Any]) -> str:
    score = evaluation.get("score")
    if not isinstance(score, dict):
        return "unscored"
    explicit = score.get("quality")
    if explicit in {"confirmed", "degraded", "unscored"}:
        return str(explicit)
    if evaluation.get("error") or score.get("score_source") == "heuristic_fallback":
        return "unscored"
    judge = evaluation.get("judge")
    if isinstance(judge, dict):
        if judge.get("fallback_used"):
            return "unscored"
        aggregation = judge.get("aggregation")
        if isinstance(aggregation, dict) and aggregation.get("degraded_quorum"):
            return "degraded"
    return "confirmed"


def annotate_score_quality(
    score: dict[str, Any],
    judge_audit: dict[str, Any] | None,
    *,
    error: str | None = None,
) -> dict[str, Any]:
    result = dict(score)
    if error or result.get("score_source") == "heuristic_fallback":
        result["quality"] = "unscored"
    elif isinstance(judge_audit, dict) and judge_audit.get("fallback_used"):
        result["quality"] = "unscored"
    elif (
        isinstance(judge_audit, dict)
        and isinstance(judge_audit.get("aggregation"), dict)
        and judge_audit["aggregation"].get("degraded_quorum")
    ):
        result["quality"] = "degraded"
    else:
        result["quality"] = "confirmed"
    return result


def _number(score: dict[str, Any], key: str) -> float | None:
    value = score.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def compare_evaluations(
    parent: dict[str, Any],
    child: dict[str, Any],
    *,
    max_component_drop: float = 5.0,
    retention_components: tuple[str, ...] = DEFAULT_RETENTION_COMPONENTS,
    allow_degraded: bool = True,
) -> ComparisonResult:
    accepted_qualities = {"confirmed", "degraded"} if allow_degraded else {"confirmed"}
    parent_quality = score_quality(parent)
    child_quality = score_quality(child)
    if parent_quality not in accepted_qualities or child_quality not in accepted_qualities:
        return ComparisonResult(
            comparable=False,
            classification="unscored",
            total_delta=None,
            component_deltas={},
            retention_constraints_met=False,
            reasons=(f"parent_quality:{parent_quality}", f"child_quality:{child_quality}"),
        )

    parent_score = parent.get("score") or {}
    child_score = child.get("score") or {}
    parent_total = _number(parent_score, "total")
    child_total = _number(child_score, "total")
    if parent_total is None or child_total is None:
        return ComparisonResult(
            comparable=False,
            classification="unscored",
            total_delta=None,
            component_deltas={},
            retention_constraints_met=False,
            reasons=("missing_total",),
        )

    component_deltas: dict[str, float] = {}
    retention_failures: list[str] = []
    for key in retention_components:
        parent_value = _number(parent_score, key)
        child_value = _number(child_score, key)
        if parent_value is None or child_value is None:
            continue
        delta = round(child_value - parent_value, 3)
        component_deltas[key] = delta
        if delta < -max_component_drop:
            retention_failures.append(key)

    total_delta = round(child_total - parent_total, 3)
    retained = not retention_failures
    if not retained or total_delta < 0:
        classification = "regressed"
    elif total_delta > 0:
        classification = "improved"
    else:
        classification = "retained"
    return ComparisonResult(
        comparable=True,
        classification=classification,
        total_delta=total_delta,
        component_deltas=component_deltas,
        retention_constraints_met=retained,
        reasons=tuple(f"component_drop:{key}" for key in retention_failures),
    )
