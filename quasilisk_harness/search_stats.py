"""Purpose-aware perturbator lifecycle statistics."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .perturbation.evidence import ScoreEvidence
from .perturbation.models import (
    MutationTask,
    TaskPurpose,
    VariantLifecycle,
    VariantOutcome,
    VariantSource,
    VariantStage,
)
from .perturbation.outcomes import OutcomePolicy
from .utils import append_jsonl, short_hash, utc_now, write_json


SEARCH_STATS_SCHEMA_VERSION = 3
OPTIMIZATION_IMPROVEMENT_TARGET = 0.5
METRIC_DEFINITIONS = {
    "optimization_end_to_end_rate": (
        "improved optimization variants / all perturbator-produced optimization "
        "variants; invalid, truncated, and unscored variants remain in the denominator"
    ),
    "optimization_valid_rate": (
        "validated optimization variants / perturbator-produced optimization variants"
    ),
    "optimization_conditional_rate": (
        "improved optimization variants / comparable optimization variants"
    ),
    "discovery_success_rate": (
        "archive-retained discovery variants / perturbator-produced discovery variants"
    ),
    "diagnostic_effect_rate": (
        "improved or regressed diagnostic variants / perturbator-produced diagnostic variants"
    ),
    "control_variance": (
        "signed and absolute score deltas for comparable harness replay controls"
    ),
}
ArchiveSnapshot = dict[tuple[str, ...], dict[str, Any]]


class SearchStatsTracker:
    """Track each perturbator output from production through terminal outcome."""

    def __init__(
        self,
        *,
        events_path: Path,
        summary_path: Path,
        objective_id: str,
        target_model: str,
        perturbator_model: str,
        run_instance_id: str,
    ) -> None:
        self.events_path = events_path
        self.summary_path = summary_path
        self.objective_id = objective_id
        self.target_model = target_model
        self.perturbator_model = perturbator_model
        self.run_instance_id = run_instance_id
        self._attempts: dict[str, dict[str, Any]] = {}
        self._completed: list[dict[str, Any]] = []

    def start_attempt(
        self,
        *,
        attempt_id: str,
        generation: int,
        selection_index: int,
        selection_method: str,
        parent_eval: dict[str, Any],
    ) -> None:
        evidence = ScoreEvidence.from_evaluation(parent_eval)
        parent = {
            "candidate_id": str(parent_eval.get("candidate_id", "")),
            "cell": list(parent_eval.get("archive_cell") or []),
            "score": evidence.total or 0.0,
            "score_evidence_basis": evidence.basis,
            "selection_index": selection_index,
            "selection_method": selection_method,
        }
        attempt = {
            "attempt_id": attempt_id,
            "generation": generation,
            "parent": parent,
            "plan": {},
            "perturbator": {
                "produced": 0,
                "accepted": 0,
                "controls": 0,
                "fallback_used": False,
                "retry_count": 0,
                "usage": {},
                "error": None,
            },
            "variant_lifecycles": [],
            "rejection_reasons": {},
        }
        self._attempts[attempt_id] = attempt
        self._append(
            {
                "event_type": "attempt_started",
                "attempt_id": attempt_id,
                "generation": generation,
                "parent": parent,
            }
        )

    def record_perturbator_result(
        self,
        *,
        attempt_id: str,
        plan: dict[str, Any] | None,
        produced: int,
        accepted_candidates: list[Any],
        rejected: list[dict[str, Any]],
        audit: dict[str, Any],
        error: str | None,
    ) -> None:
        attempt = self._attempts[attempt_id]
        if produced < 0:
            raise ValueError("produced must be non-negative")
        fallback_used = bool(audit.get("fallback_used"))
        model_source = (
            VariantSource.FALLBACK
            if fallback_used
            else VariantSource.PERTURBATOR
        )
        lifecycles: list[VariantLifecycle] = []

        for candidate in accepted_candidates:
            task = _task_from_metadata(candidate.metadata)
            purpose = task.purpose if task is not None else _purpose_from_metadata(
                candidate.metadata
            )
            source = (
                VariantSource.HARNESS
                if purpose is TaskPurpose.CONTROL
                else model_source
            )
            lifecycles.append(
                VariantLifecycle(
                    variant_id=str(candidate.id),
                    task_id=task.task_id if task is not None else "",
                    purpose=purpose,
                    operator_id=(
                        task.operator_id if task is not None else "unspecified_task"
                    ),
                    source=source,
                    stage=VariantStage.VALIDATED,
                    validation_tier=_validation_tier(candidate.metadata),
                )
            )

        for index, item in enumerate(rejected):
            task = _task_from_mapping(item)
            purpose = task.purpose if task is not None else _purpose_from_metadata(item)
            source = (
                VariantSource.HARNESS
                if purpose is TaskPurpose.CONTROL
                else model_source
            )
            variant_id = str(item.get("task_id") or "") or short_hash(
                f"{attempt_id}|rejected|{index}|{item.get('prompt', '')}",
                16,
            )
            lifecycles.append(
                VariantLifecycle(
                    variant_id=variant_id,
                    task_id=task.task_id if task is not None else "",
                    purpose=purpose,
                    operator_id=(
                        task.operator_id
                        if task is not None
                        else str(item.get("operator_id") or "unspecified_task")
                    ),
                    source=source,
                    outcome=VariantOutcome.INVALID,
                    validation_tier=_validation_tier(item),
                )
            )

        recorded_model_outputs = sum(
            item.purpose is not TaskPurpose.CONTROL for item in lifecycles
        )
        if recorded_model_outputs > produced:
            raise ValueError(
                "Recorded non-control outputs exceed the reported produced count: "
                f"{recorded_model_outputs}>{produced}"
            )
        unassigned_count = max(0, int(produced) - recorded_model_outputs)
        unassigned_purpose = _unassigned_output_purpose(audit)
        for index in range(unassigned_count):
            lifecycles.append(
                VariantLifecycle(
                    variant_id=f"unassigned-{attempt_id}-{index}",
                    task_id="",
                    purpose=unassigned_purpose,
                    operator_id="unassigned_model_output",
                    source=model_source,
                    outcome=VariantOutcome.INVALID,
                )
            )

        generated = [
            item for item in lifecycles if item.purpose is not TaskPurpose.CONTROL
        ]
        controls = [
            item for item in lifecycles if item.purpose is TaskPurpose.CONTROL
        ]
        accepted = [item for item in generated if item.stage is VariantStage.VALIDATED]
        attempt["plan"] = _compact_plan(plan)
        attempt["perturbator"] = {
            "produced": int(produced),
            "accepted": len(accepted),
            "controls": len(controls),
            "unassigned_outputs": unassigned_count,
            "fallback_used": fallback_used,
            "retry_count": int(audit.get("retry_count", 0) or 0),
            "usage": dict(audit.get("usage") or {}),
            "error": error,
        }
        attempt["variant_lifecycles"] = [item.to_jsonable() for item in lifecycles]
        attempt["rejection_reasons"] = dict(
            Counter(str(item.get("reason") or "unknown") for item in rejected)
        )

    def complete_generation(
        self,
        *,
        generation: int,
        scheduled_candidates: list[Any],
        evaluations: list[dict[str, Any]],
        archive_before: ArchiveSnapshot,
        archive_after: ArchiveSnapshot,
    ) -> list[dict[str, Any]]:
        scheduled_ids = {str(candidate.id) for candidate in scheduled_candidates}
        evaluations_by_id = {
            str(evaluation.get("candidate_id", "")): evaluation
            for evaluation in evaluations
        }
        retained_ids = {
            str(candidate_id)
            for cell in archive_after.values()
            for candidate_id in cell.get("candidate_ids", ())
        }
        successful_evaluations = {
            candidate_id: evaluation
            for candidate_id, evaluation in evaluations_by_id.items()
            if not evaluation.get("error") and evaluation.get("response")
        }
        cell_credits, global_credit = _archive_credits(
            archive_before=archive_before,
            archive_after=archive_after,
            evaluation_by_id=successful_evaluations,
        )

        completed: list[dict[str, Any]] = []
        for attempt_id, attempt in self._attempts.items():
            if attempt["generation"] != generation or attempt.get("completed"):
                continue
            lifecycles = [
                VariantLifecycle.from_jsonable(item)
                for item in attempt.get("variant_lifecycles", ())
            ]
            finalized = [
                _finalize_lifecycle(
                    item,
                    scheduled_ids=scheduled_ids,
                    evaluation=evaluations_by_id.get(item.variant_id),
                    archive_retained=item.variant_id in retained_ids,
                )
                for item in lifecycles
            ]
            evaluated_variant_ids = {
                item.variant_id
                for item in finalized
                if item.stage is VariantStage.EVALUATED
            }
            attempt_evaluations = [
                evaluation
                for candidate_id, evaluation in evaluations_by_id.items()
                if candidate_id in evaluated_variant_ids
            ]
            credit = cell_credits.get(attempt_id, {})
            record = {
                "event_type": "attempt_completed",
                "attempt_id": attempt_id,
                "generation": generation,
                "parent": attempt["parent"],
                "plan": attempt["plan"],
                "perturbator": {
                    **attempt["perturbator"],
                    "evaluated": sum(
                        item.stage is VariantStage.EVALUATED
                        and item.purpose is not TaskPurpose.CONTROL
                        for item in finalized
                    ),
                    "budget_truncated": sum(
                        item.outcome is VariantOutcome.TRUNCATED
                        and item.purpose is not TaskPurpose.CONTROL
                        for item in finalized
                    ),
                },
                "outcome": _attempt_outcome(
                    attempt=attempt,
                    variants=finalized,
                    evaluations=attempt_evaluations,
                    credit=credit,
                    global_best_delta=global_credit.get(attempt_id, 0.0),
                ),
                "purpose_outcomes": {
                    purpose.value: OutcomePolicy.metrics(
                        finalized,
                        purpose=purpose,
                        source=None,
                    )
                    for purpose in TaskPurpose
                },
                "operator_outcomes": _operator_outcomes(finalized),
                "variant_lifecycles": [item.to_jsonable() for item in finalized],
                "rejection_reasons": attempt["rejection_reasons"],
            }
            attempt["completed"] = True
            self._completed.append(record)
            completed.append(record)
            self._append(record)

        self.write_summary()
        return completed

    def summary(self) -> dict[str, Any]:
        by_parent_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_route_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_generation: dict[str, list[dict[str, Any]]] = defaultdict(list)
        rejection_reasons: Counter[str] = Counter()

        for record in self._completed:
            parent_cell = "|".join(record["parent"].get("cell") or ["unknown"])
            plan = record.get("plan") or {}
            strategy = str(plan.get("strategy") or "unknown")
            route_case = "|".join(
                (
                    str(plan.get("route") or "unknown"),
                    str(plan.get("case_family") or "unknown"),
                )
            )
            by_parent_cell[parent_cell].append(record)
            by_strategy[strategy].append(record)
            by_route_case[route_case].append(record)
            by_generation[str(record["generation"])].append(record)
            rejection_reasons.update(record.get("rejection_reasons") or {})

        parent_cell_aggregates = _aggregate_groups(by_parent_cell)
        return {
            "schema_version": SEARCH_STATS_SCHEMA_VERSION,
            "objective_id": self.objective_id,
            "target_model": self.target_model,
            "perturbator_model": self.perturbator_model,
            "run_instance_id": self.run_instance_id,
            "updated_at": utc_now(),
            "attempts_started": len(self._attempts),
            "attempts_completed": len(self._completed),
            "metric_definitions": dict(METRIC_DEFINITIONS),
            "overall": _aggregate(self._completed),
            "by_parent_cell": parent_cell_aggregates,
            "top_parent_cells": _rank_parent_cells(
                parent_cell_aggregates,
                reverse=True,
            ),
            "bottom_parent_cells": _rank_parent_cells(
                parent_cell_aggregates,
                reverse=False,
            ),
            "by_strategy": _aggregate_groups(by_strategy),
            "by_route_case": _aggregate_groups(by_route_case),
            "by_generation": _aggregate_groups(by_generation),
            "by_operator": _aggregate_operators(self._completed),
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
        }

    def status_metrics(self) -> dict[str, Any]:
        summary = self.summary()
        overall = summary["overall"]
        return {
            "search_attempts_started": summary["attempts_started"],
            "search_attempts_completed": summary["attempts_completed"],
            "optimization_end_to_end_rate": overall[
                "optimization_end_to_end_rate"
            ],
            "optimization_valid_rate": overall["optimization_valid_rate"],
            "optimization_conditional_rate": overall[
                "optimization_conditional_rate"
            ],
            "optimization_target_gap": overall["optimization_target_gap"],
            "discovery_success_rate": overall["discovery_success_rate"],
            "diagnostic_effect_rate": overall["diagnostic_effect_rate"],
            "control_mean_absolute_delta": overall["control_variance"][
                "mean_absolute_delta"
            ],
            "perturbator_archive_yield": overall["archive_yield"],
            "perturbator_new_cell_yield": overall["new_cell_yield"],
        }

    def write_summary(self) -> None:
        document: dict[str, Any] = {
            "schema_version": SEARCH_STATS_SCHEMA_VERSION,
            "updated_at": utc_now(),
            "targets": {},
            "runs": {},
        }
        if self.summary_path.exists():
            try:
                with self.summary_path.open("r", encoding="utf-8") as handle:
                    existing = json.load(handle)
                if isinstance(existing, dict) and isinstance(
                    existing.get("targets"),
                    dict,
                ):
                    document["targets"] = dict(existing["targets"])
                if isinstance(existing, dict) and isinstance(
                    existing.get("runs"),
                    dict,
                ):
                    document["runs"] = dict(existing["runs"])
            except (OSError, json.JSONDecodeError):
                pass
        summary = self.summary()
        document["targets"][self.target_model] = summary
        document["runs"][self.run_instance_id] = summary
        write_json(self.summary_path, document)

    def _append(self, record: dict[str, Any]) -> None:
        append_jsonl(
            self.events_path,
            {
                "schema_version": SEARCH_STATS_SCHEMA_VERSION,
                "timestamp": utc_now(),
                "objective_id": self.objective_id,
                "target_model": self.target_model,
                "perturbator_model": self.perturbator_model,
                "run_instance_id": self.run_instance_id,
                **record,
            },
        )


def _finalize_lifecycle(
    lifecycle: VariantLifecycle,
    *,
    scheduled_ids: set[str],
    evaluation: dict[str, Any] | None,
    archive_retained: bool,
) -> VariantLifecycle:
    if lifecycle.outcome is VariantOutcome.INVALID:
        return lifecycle
    if lifecycle.variant_id not in scheduled_ids:
        return lifecycle.advance(
            VariantStage.VALIDATED,
            outcome=VariantOutcome.TRUNCATED,
        )
    if evaluation is None:
        return lifecycle.advance(
            VariantStage.SCHEDULED,
            outcome=VariantOutcome.UNSCORED,
        )
    comparison = _comparison(evaluation)
    score_delta = (
        float(comparison["total_delta"])
        if comparison.get("total_delta") is not None
        else None
    )
    evaluation_validation_tier = _validation_tier(evaluation.get("metadata"))
    return lifecycle.advance(
        VariantStage.EVALUATED,
        outcome=OutcomePolicy.classify_evaluation(evaluation),
        archive_retained=(
            archive_retained
            and not evaluation.get("error")
            and bool(evaluation.get("response"))
        ),
        score=_score_total(evaluation),
        score_delta=score_delta,
        validation_tier=(
            None
            if evaluation_validation_tier == "unknown"
            else evaluation_validation_tier
        ),
    )


def _attempt_outcome(
    *,
    attempt: dict[str, Any],
    variants: list[VariantLifecycle],
    evaluations: list[dict[str, Any]],
    credit: dict[str, float],
    global_best_delta: float,
) -> dict[str, Any]:
    noncontrols = [
        item for item in variants if item.purpose is not TaskPurpose.CONTROL
    ]
    evaluated = [item for item in noncontrols if item.stage is VariantStage.EVALUATED]
    comparable = [
        item
        for item in evaluated
        if item.outcome
        in {
            VariantOutcome.IMPROVED,
            VariantOutcome.RETAINED,
            VariantOutcome.REGRESSED,
        }
    ]
    deltas = [float(item.score_delta) for item in comparable if item.score_delta is not None]
    child_scores = [float(item.score) for item in evaluated if item.score is not None]
    validation_tiers = Counter(
        item.validation_tier
        for item in noncontrols
        if item.stage is not VariantStage.PRODUCED
    )
    return {
        "parent_score": float(attempt["parent"]["score"]),
        "scheduled": sum(
            item.stage in {VariantStage.SCHEDULED, VariantStage.EVALUATED}
            for item in noncontrols
        ),
        "evaluated": len(evaluated),
        "non_error_children": sum(
            not evaluation.get("error") and bool(evaluation.get("response"))
            for evaluation in evaluations
            if _evaluation_purpose(evaluation) is not TaskPurpose.CONTROL
        ),
        "errored_children": sum(
            bool(evaluation.get("error")) or not evaluation.get("response")
            for evaluation in evaluations
            if _evaluation_purpose(evaluation) is not TaskPurpose.CONTROL
        ),
        "budget_truncated": sum(
            item.outcome is VariantOutcome.TRUNCATED for item in noncontrols
        ),
        "best_child_score": max(child_scores) if child_scores else None,
        "best_score_delta": max(deltas) if deltas else None,
        "mean_score_delta": _mean(deltas),
        "score_delta_sum": round(sum(deltas), 3),
        "positive_score_gain": round(
            sum(max(0.0, delta) for delta in deltas),
            3,
        ),
        "validation_tiers": dict(sorted(validation_tiers.items())),
        "final_archive_retained": sum(item.archive_retained for item in noncontrols),
        "new_cells_created": int(credit.get("new_cells_created", 0)),
        "cell_best_improvements": int(credit.get("cell_best_improvements", 0)),
        "cell_best_score_gain": round(
            float(credit.get("cell_best_score_gain", 0.0)),
            3,
        ),
        "global_best_delta": round(global_best_delta, 3),
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    variants = _all_lifecycles(records)
    by_purpose = {
        purpose.value: OutcomePolicy.metrics(
            variants,
            purpose=purpose,
            source=(
                None
                if purpose is TaskPurpose.CONTROL
                else VariantSource.PERTURBATOR
            ),
        )
        for purpose in TaskPurpose
    }
    optimization = by_purpose[TaskPurpose.OPTIMIZATION.value]
    discovery = by_purpose[TaskPurpose.DISCOVERY.value]
    diagnostic = by_purpose[TaskPurpose.DIAGNOSTIC.value]
    model_variants = [
        item for item in variants if item.source is VariantSource.PERTURBATOR
    ]
    model_evaluated = [
        item
        for item in model_variants
        if item.stage is VariantStage.EVALUATED
        and item.purpose is not TaskPurpose.CONTROL
    ]
    retained = sum(item.archive_retained for item in model_evaluated)
    discovery_successes = sum(
        item.archive_retained
        for item in model_variants
        if item.purpose is TaskPurpose.DISCOVERY
    )
    diagnostic_effects = diagnostic[VariantOutcome.IMPROVED.value] + diagnostic[
        VariantOutcome.REGRESSED.value
    ]
    model_records = [
        item for item in records if not item["perturbator"].get("fallback_used")
    ]
    deltas = [
        float(item.score_delta)
        for item in model_evaluated
        if item.score_delta is not None
    ]
    positive_gain = sum(max(0.0, delta) for delta in deltas)
    best_deltas = [
        float(record["outcome"]["best_score_delta"])
        for record in model_records
        if record["outcome"].get("best_score_delta") is not None
    ]
    improving_attempts = sum(
        any(
            lifecycle.source is VariantSource.PERTURBATOR
            and lifecycle.purpose is TaskPurpose.OPTIMIZATION
            and lifecycle.outcome is VariantOutcome.IMPROVED
            for lifecycle in _record_lifecycles(record)
        )
        for record in model_records
    )
    new_cells = sum(
        int(item["outcome"].get("new_cells_created", 0)) for item in model_records
    )
    total_tokens = sum(
        _total_tokens(item["perturbator"].get("usage")) for item in model_records
    )
    optimization_gap = round(
        optimization["end_to_end_improvement_rate"]
        - OPTIMIZATION_IMPROVEMENT_TARGET,
        4,
    )
    return {
        "attempts": len(records),
        "model_attempts": len(model_records),
        "by_purpose": by_purpose,
        "optimization_target": OPTIMIZATION_IMPROVEMENT_TARGET,
        "optimization_produced": optimization["produced"],
        "optimization_denominator_failures": (
            optimization[VariantOutcome.INVALID.value]
            + optimization[VariantOutcome.TRUNCATED.value]
            + optimization[VariantOutcome.UNSCORED.value]
        ),
        "optimization_end_to_end_rate": optimization[
            "end_to_end_improvement_rate"
        ],
        "optimization_valid_rate": optimization["valid_rate"],
        "optimization_conditional_rate": optimization[
            "conditional_improvement_rate"
        ],
        "optimization_target_gap": optimization_gap,
        "optimization_target_met": optimization_gap >= 0,
        "discovery_success_rate": _ratio(
            discovery_successes,
            discovery["produced"],
        ),
        "diagnostic_effect_rate": _ratio(
            diagnostic_effects,
            diagnostic["produced"],
        ),
        "control_variance": OutcomePolicy.control_variance(variants),
        "fallback_attempts": len(records) - len(model_records),
        "errored_attempts": sum(
            1 for item in records if item["perturbator"].get("error")
        ),
        "attempt_improvement_rate": _ratio(improving_attempts, len(model_records)),
        "archive_yield": _ratio(retained, len(model_evaluated)),
        "new_cell_yield": _ratio(new_cells, len(model_evaluated)),
        "mean_score_delta": _mean(deltas),
        "mean_best_score_delta": _mean(best_deltas),
        "best_score_delta": round(max(best_deltas), 3) if best_deltas else None,
        "positive_score_gain": round(positive_gain, 3),
        "positive_gain_per_target_call": _ratio(
            positive_gain,
            len(model_evaluated),
        ),
        "perturbator_total_tokens": total_tokens,
        "positive_gain_per_1k_perturbator_tokens": (
            _ratio(positive_gain * 1000.0, total_tokens) if total_tokens else None
        ),
        "final_archive_retained": retained,
        "new_cells_created": new_cells,
        "cell_best_improvements": sum(
            int(item["outcome"].get("cell_best_improvements", 0))
            for item in model_records
        ),
        "cell_best_score_gain": round(
            sum(
                float(item["outcome"].get("cell_best_score_gain", 0.0) or 0.0)
                for item in model_records
            ),
            3,
        ),
        "global_best_delta": round(
            sum(
                float(item["outcome"].get("global_best_delta", 0.0) or 0.0)
                for item in model_records
            ),
            3,
        ),
        "distinct_parent_count": len(
            {str(item["parent"].get("candidate_id", "")) for item in records}
        ),
        "validation_tiers": _sum_counter_field(records, "validation_tiers"),
    }


def _aggregate_groups(
    groups: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    return {key: _aggregate(records) for key, records in sorted(groups.items())}


def _rank_parent_cells(
    aggregates: dict[str, dict[str, Any]],
    *,
    reverse: bool,
) -> list[dict[str, Any]]:
    ranked = sorted(
        aggregates.items(),
        key=lambda item: (
            float(item[1].get("positive_gain_per_target_call", 0.0) or 0.0),
            float(item[1].get("mean_best_score_delta", 0.0) or 0.0),
            int(item[1].get("attempts", 0)),
            item[0],
        ),
        reverse=reverse,
    )
    return [
        {
            "cell": cell,
            "attempts": metrics["attempts"],
            "positive_gain_per_target_call": metrics[
                "positive_gain_per_target_call"
            ],
            "mean_best_score_delta": metrics["mean_best_score_delta"],
            "optimization_end_to_end_rate": metrics[
                "optimization_end_to_end_rate"
            ],
            "archive_yield": metrics["archive_yield"],
            "new_cell_yield": metrics["new_cell_yield"],
        }
        for cell, metrics in ranked[:5]
    ]


def _aggregate_operators(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[VariantLifecycle]] = defaultdict(list)
    for lifecycle in _all_lifecycles(records):
        if (
            lifecycle.source is not VariantSource.PERTURBATOR
            and lifecycle.purpose is not TaskPurpose.CONTROL
        ):
            continue
        grouped[lifecycle.operator_id].append(lifecycle)
    result: dict[str, dict[str, Any]] = {}
    for operator_id, variants in sorted(grouped.items()):
        purposes = sorted({item.purpose.value for item in variants})
        outcomes = Counter(
            item.outcome.value for item in variants if item.outcome is not None
        )
        comparable = sum(
            outcomes[outcome.value]
            for outcome in (
                VariantOutcome.IMPROVED,
                VariantOutcome.RETAINED,
                VariantOutcome.REGRESSED,
            )
        )
        result[operator_id] = {
            "purposes": purposes,
            "sources": dict(
                sorted(Counter(item.source.value for item in variants).items())
            ),
            "produced": len(variants),
            "evaluated": sum(
                item.stage is VariantStage.EVALUATED for item in variants
            ),
            "comparable": comparable,
            **{outcome.value: outcomes[outcome.value] for outcome in VariantOutcome},
            "end_to_end_improvement_rate": _ratio(
                outcomes[VariantOutcome.IMPROVED.value],
                len(variants),
            ),
            "conditional_improvement_rate": _ratio(
                outcomes[VariantOutcome.IMPROVED.value],
                comparable,
            ),
        }
    return result


def _operator_outcomes(
    variants: list[VariantLifecycle],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[VariantLifecycle]] = defaultdict(list)
    for item in variants:
        grouped[item.operator_id].append(item)
    result: dict[str, dict[str, Any]] = {}
    for operator_id, items in sorted(grouped.items()):
        outcomes = Counter(
            item.outcome.value for item in items if item.outcome is not None
        )
        result[operator_id] = {
            "purposes": sorted({item.purpose.value for item in items}),
            "sources": dict(
                sorted(Counter(item.source.value for item in items).items())
            ),
            "produced": len(items),
            "evaluated": sum(item.stage is VariantStage.EVALUATED for item in items),
            **{outcome.value: outcomes[outcome.value] for outcome in VariantOutcome},
        }
    return result


def _all_lifecycles(records: Iterable[dict[str, Any]]) -> list[VariantLifecycle]:
    return [
        lifecycle
        for record in records
        for lifecycle in _record_lifecycles(record)
    ]


def _record_lifecycles(record: dict[str, Any]) -> list[VariantLifecycle]:
    return [
        VariantLifecycle.from_jsonable(item)
        for item in record.get("variant_lifecycles", ())
    ]


def _task_from_metadata(metadata: Any) -> MutationTask | None:
    if not isinstance(metadata, dict):
        return None
    return _task_from_mapping(metadata)


def _task_from_mapping(data: dict[str, Any]) -> MutationTask | None:
    task_data = data.get("mutation_task")
    if not isinstance(task_data, dict):
        return None
    return MutationTask.from_jsonable(task_data)


def _purpose_from_metadata(metadata: Any) -> TaskPurpose:
    if not isinstance(metadata, dict):
        return TaskPurpose.OPTIMIZATION
    raw_purpose = metadata.get("task_purpose") or metadata.get("purpose")
    if raw_purpose is None:
        return TaskPurpose.OPTIMIZATION
    try:
        return TaskPurpose(str(raw_purpose))
    except ValueError as exc:
        raise ValueError(f"Unknown variant task purpose: {raw_purpose}") from exc


def _evaluation_purpose(evaluation: dict[str, Any]) -> TaskPurpose:
    return _purpose_from_metadata(evaluation.get("metadata"))


def _unassigned_output_purpose(audit: dict[str, Any]) -> TaskPurpose:
    purposes = []
    for task_data in audit.get("mutation_tasks") or ():
        if not isinstance(task_data, dict):
            continue
        task = MutationTask.from_jsonable(task_data)
        if not task.is_control:
            purposes.append(task.purpose)
    if TaskPurpose.OPTIMIZATION in purposes:
        return TaskPurpose.OPTIMIZATION
    return purposes[0] if purposes else TaskPurpose.OPTIMIZATION


def _comparison(evaluation: dict[str, Any]) -> dict[str, Any]:
    metadata = evaluation.get("metadata")
    comparison = (
        metadata.get("parent_child_comparison")
        if isinstance(metadata, dict)
        else None
    )
    return comparison if isinstance(comparison, dict) else {}


def _archive_credits(
    *,
    archive_before: ArchiveSnapshot,
    archive_after: ArchiveSnapshot,
    evaluation_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    credits: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "new_cells_created": 0.0,
            "cell_best_improvements": 0.0,
            "cell_best_score_gain": 0.0,
        }
    )
    for cell_key, after in archive_after.items():
        before = archive_before.get(cell_key)
        best_candidate_id = str(after.get("best_candidate_id") or "")
        evaluation = evaluation_by_id.get(best_candidate_id)
        if evaluation is None:
            continue
        attempt_id = _evaluation_attempt_id(evaluation)
        if not attempt_id:
            continue
        before_score = float(before.get("best_score", 0.0)) if before else 0.0
        after_score = float(after.get("best_score", 0.0) or 0.0)
        if before is None:
            credits[attempt_id]["new_cells_created"] += 1.0
        elif after_score > before_score:
            credits[attempt_id]["cell_best_improvements"] += 1.0
            credits[attempt_id]["cell_best_score_gain"] += max(
                0.0,
                after_score - before_score,
            )

    before_global = max(
        (float(item.get("best_score", 0.0) or 0.0) for item in archive_before.values()),
        default=0.0,
    )
    after_global_cell = max(
        archive_after.values(),
        key=lambda item: float(item.get("best_score", 0.0) or 0.0),
        default=None,
    )
    global_credit: dict[str, float] = {}
    if after_global_cell is not None:
        after_global = float(after_global_cell.get("best_score", 0.0) or 0.0)
        evaluation = evaluation_by_id.get(
            str(after_global_cell.get("best_candidate_id") or "")
        )
        attempt_id = _evaluation_attempt_id(evaluation) if evaluation else ""
        if attempt_id and after_global > before_global:
            global_credit[attempt_id] = after_global - before_global
    return dict(credits), global_credit


def _compact_plan(plan: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(plan, dict):
        return {}
    return {
        key: plan.get(key)
        for key in (
            "plan_id",
            "strategy",
            "route",
            "case_family",
            "defense_focus",
            "priority",
        )
    }


def _evaluation_attempt_id(evaluation: dict[str, Any]) -> str:
    metadata = evaluation.get("metadata")
    if not isinstance(metadata, dict):
        return ""
    return str(metadata.get("perturbation_attempt_id") or "")


def _validation_tier(metadata: Any) -> str:
    if not isinstance(metadata, dict):
        return "unknown"
    for key in ("transition_validation", "validation"):
        validation = metadata.get(key)
        if isinstance(validation, dict) and validation.get("tier"):
            return str(validation["tier"])
    return "unknown"


def _sum_counter_field(
    records: list[dict[str, Any]],
    field: str,
) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for item in records:
        values = item.get("outcome", {}).get(field)
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            try:
                counter[str(key)] += int(value)
            except (TypeError, ValueError):
                continue
    return dict(sorted(counter.items()))


def _score_total(evaluation: dict[str, Any]) -> float | None:
    return ScoreEvidence.from_evaluation(evaluation).total


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def _ratio(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def _total_tokens(usage: Any) -> int:
    if not isinstance(usage, dict):
        return 0
    try:
        if usage.get("total_tokens") is not None:
            return int(usage["total_tokens"])
        return int(usage.get("prompt_tokens", 0) or 0) + int(
            usage.get("completion_tokens", 0) or 0
        )
    except (TypeError, ValueError):
        return 0
