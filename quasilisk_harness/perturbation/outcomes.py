"""Canonical perturbation lifecycle classification and metric policy."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from quasilisk_harness.perturbation.models import (
    TaskPurpose,
    VariantLifecycle,
    VariantOutcome,
    VariantSource,
    VariantStage,
)


class OutcomePolicy:
    """Own all terminal classifications and their denominators."""

    @staticmethod
    def classify_evaluation(evaluation: dict[str, Any]) -> VariantOutcome:
        if evaluation.get("error") or not evaluation.get("response"):
            return VariantOutcome.UNSCORED
        metadata = evaluation.get("metadata")
        comparison = (
            metadata.get("parent_child_comparison")
            if isinstance(metadata, dict)
            else None
        )
        if not isinstance(comparison, dict) or not comparison.get("comparable"):
            return VariantOutcome.UNSCORED
        try:
            return VariantOutcome(str(comparison.get("classification")))
        except ValueError:
            return VariantOutcome.UNSCORED

    @staticmethod
    def metrics(
        variants: Iterable[VariantLifecycle],
        *,
        purpose: TaskPurpose,
        source: VariantSource | None = VariantSource.PERTURBATOR,
    ) -> dict[str, Any]:
        selected = [
            item
            for item in variants
            if item.purpose is purpose and (source is None or item.source is source)
        ]
        stages = Counter(item.stage.value for item in selected)
        outcomes = Counter(
            item.outcome.value for item in selected if item.outcome is not None
        )
        produced = len(selected)
        validated = sum(
            count
            for stage, count in stages.items()
            if _stage_at_least(stage, VariantStage.VALIDATED)
        )
        scheduled = sum(
            count
            for stage, count in stages.items()
            if _stage_at_least(stage, VariantStage.SCHEDULED)
        )
        evaluated = stages[VariantStage.EVALUATED.value]
        comparable = sum(
            outcomes[item.value]
            for item in (
                VariantOutcome.IMPROVED,
                VariantOutcome.RETAINED,
                VariantOutcome.REGRESSED,
            )
        )
        terminal = sum(outcomes.values())
        result: dict[str, Any] = {
            "produced": produced,
            "validated": validated,
            "scheduled": scheduled,
            "evaluated": evaluated,
            "comparable": comparable,
            "terminal": terminal,
            "unresolved": produced - terminal,
            "reconciled": produced == terminal,
            **{outcome.value: outcomes[outcome.value] for outcome in VariantOutcome},
            "valid_rate": _ratio(validated, produced),
            "conditional_improvement_rate": _ratio(
                outcomes[VariantOutcome.IMPROVED.value],
                comparable,
            ),
        }
        result["end_to_end_improvement_rate"] = _ratio(
            outcomes[VariantOutcome.IMPROVED.value],
            produced,
        )
        result["retention_rate"] = _ratio(
            outcomes[VariantOutcome.IMPROVED.value]
            + outcomes[VariantOutcome.RETAINED.value],
            produced,
        )
        return result

    @staticmethod
    def control_variance(
        variants: Iterable[VariantLifecycle],
    ) -> dict[str, Any]:
        controls = [item for item in variants if item.purpose is TaskPurpose.CONTROL]
        comparable_controls = [
            item
            for item in controls
            if item.stage is VariantStage.EVALUATED
            and item.score_delta is not None
        ]
        deltas = [float(item.score_delta) for item in comparable_controls]
        return {
            "produced": len(controls),
            "evaluated": sum(
                item.stage is VariantStage.EVALUATED for item in controls
            ),
            "comparable": len(comparable_controls),
            "nonzero": sum(delta != 0 for delta in deltas),
            "nonzero_rate": _ratio(sum(delta != 0 for delta in deltas), len(deltas)),
            "mean_delta": _mean(deltas),
            "mean_absolute_delta": _mean([abs(delta) for delta in deltas]),
            "min_delta": round(min(deltas), 3) if deltas else None,
            "max_delta": round(max(deltas), 3) if deltas else None,
        }


def _stage_at_least(value: str, minimum: VariantStage) -> bool:
    order = tuple(VariantStage)
    return order.index(VariantStage(value)) >= order.index(minimum)


def _ratio(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 3)
