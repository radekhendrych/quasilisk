"""Quality-aware parent/child score comparisons."""

from __future__ import annotations

from typing import Any

from quasilisk_harness.perturbation.evidence import ScoreEvidence
from quasilisk_harness.perturbation.evidence import score_quality as evidence_quality
from quasilisk_harness.perturbation.models import ComparisonResult


DEFAULT_RETENTION_COMPONENTS = (
    "raw_judge_risk_score",
    "argument_success_score",
    "commitment_evidence_score",
)


def score_quality(evaluation: dict[str, Any]) -> str:
    return evidence_quality(evaluation)


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


def compare_evaluations(
    parent: dict[str, Any],
    child: dict[str, Any],
    *,
    max_component_drop: float = 5.0,
    retention_components: tuple[str, ...] = DEFAULT_RETENTION_COMPONENTS,
    allow_degraded: bool = True,
) -> ComparisonResult:
    accepted_qualities = {"confirmed", "degraded"} if allow_degraded else {"confirmed"}
    parent_evidence = ScoreEvidence.from_evaluation(parent)
    child_evidence = ScoreEvidence.from_evaluation(child)
    parent_quality = parent_evidence.quality
    child_quality = child_evidence.quality
    evidence_basis = {
        "parent": parent_evidence.basis,
        "child": child_evidence.basis,
    }
    if parent_quality not in accepted_qualities or child_quality not in accepted_qualities:
        return ComparisonResult(
            comparable=False,
            classification="unscored",
            total_delta=None,
            component_deltas={},
            retention_constraints_met=False,
            reasons=(f"parent_quality:{parent_quality}", f"child_quality:{child_quality}"),
            evidence_basis=evidence_basis,
        )

    parent_total = parent_evidence.total
    child_total = child_evidence.total
    if parent_total is None or child_total is None:
        return ComparisonResult(
            comparable=False,
            classification="unscored",
            total_delta=None,
            component_deltas={},
            retention_constraints_met=False,
            reasons=("missing_total",),
            evidence_basis=evidence_basis,
        )

    component_deltas: dict[str, float] = {}
    retention_failures: list[str] = []
    for key in retention_components:
        parent_value = parent_evidence.components.get(key)
        child_value = child_evidence.components.get(key)
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
        evidence_basis=evidence_basis,
    )
