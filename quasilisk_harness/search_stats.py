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
from .perturbation.rejection_memory import normalized_prompt_key
from .utils import append_jsonl, short_hash, utc_now, write_json


SEARCH_STATS_SCHEMA_VERSION = 5
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
    "optimization_nomination_rate": (
        "validator-nominated optimization variants / perturbator-produced "
        "optimization variants; an output-level efficiency diagnostic"
    ),
    "optimization_task_end_to_end_rate": (
        "optimization mutation tasks with an improved nominee / attempted "
        "optimization mutation tasks; this is the improvement target basis"
    ),
    "optimization_task_nomination_rate": (
        "optimization mutation tasks that gained an admissible nominee / "
        "attempted optimization mutation tasks"
    ),
    "perturbator_delivery_rate": (
        "parsed perturbator outputs / requested perturbator outputs"
    ),
    "perturbator_absolute_valid_rate": (
        "perturbator outputs passing objective and absolute prompt validation / "
        "parsed perturbator outputs"
    ),
    "perturbator_transition_contract_valid_rate": (
        "perturbator outputs satisfying their parent-relative mutation contract / "
        "parsed perturbator outputs"
    ),
    "perturbator_unique_output_rate": (
        "globally distinct normalized perturbator prompts / parsed perturbator outputs"
    ),
    "revision_task_success_rate": (
        "failed mutation tasks that gained an admissible nominee after the single "
        "bounded revision / mutation tasks sent for revision"
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
            source = _variant_source_from_metadata(
                candidate.metadata,
                purpose=purpose,
                default=model_source,
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
                    source_output_id=str(
                        candidate.metadata.get("output_id") or ""
                    ),
                    stage=VariantStage.VALIDATED,
                    validation_tier=_validation_tier(candidate.metadata),
                    refinement_disposition=_refinement_disposition(
                        candidate.metadata
                    ),
                )
            )

        for index, item in enumerate(rejected):
            task = _task_from_mapping(item)
            purpose = task.purpose if task is not None else _purpose_from_metadata(item)
            source = _variant_source_from_metadata(
                item,
                purpose=purpose,
                default=model_source,
            )
            variant_id = str(item.get("output_id") or "").strip() or short_hash(
                f"{attempt_id}|rejected|{index}|{item.get('task_id', '')}|"
                f"{item.get('prompt', '')}",
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
                    source_output_id=str(item.get("output_id") or variant_id),
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
        audited_unassigned = _audited_unassigned_outputs(audit)
        if len(audited_unassigned) > unassigned_count:
            raise ValueError(
                "Audited unassigned outputs exceed the unrecorded produced count: "
                f"{len(audited_unassigned)}>{unassigned_count}"
            )
        tasks_by_id = _audit_tasks_by_id(audit)
        unassigned_purpose = _unassigned_output_purpose(audit)
        for index, item in enumerate(audited_unassigned):
            task = tasks_by_id.get(str(item.get("requested_task_id") or ""))
            purpose = task.purpose if task is not None else unassigned_purpose
            source = _variant_source_from_metadata(
                item,
                purpose=purpose,
                default=model_source,
            )
            lifecycles.append(
                VariantLifecycle(
                    variant_id=str(item.get("output_id") or "").strip()
                    or f"unassigned-{attempt_id}-{index}",
                    task_id=task.task_id if task is not None else "",
                    purpose=purpose,
                    operator_id=(
                        task.operator_id
                        if task is not None
                        else "unassigned_model_output"
                    ),
                    source=source,
                    source_output_id=str(item.get("output_id") or ""),
                    outcome=VariantOutcome.INVALID,
                )
            )
        for index in range(len(audited_unassigned), unassigned_count):
            lifecycles.append(
                VariantLifecycle(
                    variant_id=f"unassigned-{attempt_id}-{index}",
                    task_id="",
                    purpose=unassigned_purpose,
                    operator_id="unassigned_model_output",
                    source=model_source,
                    source_output_id=f"unassigned-{attempt_id}-{index}",
                    outcome=VariantOutcome.INVALID,
                )
            )

        lifecycle_ids = [item.variant_id for item in lifecycles]
        if len(lifecycle_ids) != len(set(lifecycle_ids)):
            raise ValueError("variant lifecycle ids must be unique within an attempt")
        source_output_ids = [
            item.source_output_id for item in lifecycles if item.source_output_id
        ]
        if len(source_output_ids) != len(set(source_output_ids)):
            raise ValueError("source output ids must be unique within an attempt")

        generated = [
            item for item in lifecycles if item.purpose is not TaskPurpose.CONTROL
        ]
        controls = [
            item for item in lifecycles if item.purpose is TaskPurpose.CONTROL
        ]
        accepted = [item for item in generated if item.stage is VariantStage.VALIDATED]
        nominated = [
            item
            for item in accepted
            if item.refinement_disposition in {"selected", "unknown"}
        ]
        ranked_out = [
            item
            for item in accepted
            if item.refinement_disposition == "ranked_out"
        ]
        refinement_audit = audit.get("refinement")
        revision_audit = (
            refinement_audit.get("revision")
            if isinstance(refinement_audit, dict)
            else None
        )
        revision_audit = revision_audit if isinstance(revision_audit, dict) else {}
        revision_task_ids = list(revision_audit.get("task_ids") or ())
        revision_resolved_task_ids = list(
            revision_audit.get("resolved_task_ids") or ()
        )
        revision_unresolved_task_ids = list(
            revision_audit.get("unresolved_task_ids") or ()
        )
        if bool(revision_audit.get("attempted")):
            if (
                len(revision_task_ids) != len(set(revision_task_ids))
                or len(revision_resolved_task_ids)
                != len(set(revision_resolved_task_ids))
                or len(revision_unresolved_task_ids)
                != len(set(revision_unresolved_task_ids))
                or set(revision_task_ids)
                != (
                    set(revision_resolved_task_ids)
                    | set(revision_unresolved_task_ids)
                )
                or set(revision_resolved_task_ids)
                & set(revision_unresolved_task_ids)
            ):
                raise ValueError("revision task outcomes do not reconcile")
        elif revision_task_ids:
            raise ValueError("unattempted revision cannot contain requested tasks")
        output_funnel, output_evidence = _build_output_funnel(
            produced=int(produced),
            lifecycles=generated,
            accepted_candidates=accepted_candidates,
            rejected=rejected,
            audit=audit,
        )
        attempt["plan"] = _compact_plan(plan)
        attempt["task_inventory"] = [
            task.to_jsonable()
            for task in tasks_by_id.values()
            if not task.is_control
        ]
        attempt["output_evidence"] = output_evidence
        attempt["perturbator"] = {
            "produced": int(produced),
            "accepted": len(accepted),
            "nominated": len(nominated),
            "ranked_out": len(ranked_out),
            "revision_attempted": bool(revision_audit.get("attempted")),
            "revision_task_count": len(revision_task_ids),
            "revision_resolved_task_count": len(revision_resolved_task_ids),
            "revision_unresolved_task_count": len(revision_unresolved_task_ids),
            "controls": len(controls),
            "unassigned_outputs": unassigned_count,
            "fallback_used": fallback_used,
            "retry_count": int(audit.get("retry_count", 0) or 0),
            "usage": dict(audit.get("usage") or {}),
            "error": error,
            "output_funnel": output_funnel,
        }
        attempt["variant_lifecycles"] = [item.to_jsonable() for item in lifecycles]
        attempt["rejection_reasons"] = dict(
            Counter(
                str(item.get("reason") or "unknown")
                for item in [*rejected, *audited_unassigned]
            )
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
                        and item.refinement_disposition != "ranked_out"
                        for item in finalized
                    ),
                    "pre_target_ranked_out": sum(
                        item.outcome is VariantOutcome.TRUNCATED
                        and item.purpose is not TaskPurpose.CONTROL
                        and item.refinement_disposition == "ranked_out"
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
                "task_outcomes": _build_task_outcomes(
                    task_inventory=attempt.get("task_inventory") or (),
                    variants=finalized,
                    output_evidence=attempt.get("output_evidence") or {},
                ),
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
            "optimization_nomination_rate": overall[
                "optimization_nomination_rate"
            ],
            "optimization_nomination_headroom": overall[
                "optimization_nomination_headroom"
            ],
            "optimization_task_attempted": overall[
                "optimization_task_attempted"
            ],
            "optimization_task_valid_rate": overall[
                "optimization_task_valid_rate"
            ],
            "optimization_task_nomination_rate": overall[
                "optimization_task_nomination_rate"
            ],
            "optimization_task_end_to_end_rate": overall[
                "optimization_task_end_to_end_rate"
            ],
            "optimization_task_conditional_rate": overall[
                "optimization_task_conditional_rate"
            ],
            "optimization_task_nomination_headroom": overall[
                "optimization_task_nomination_headroom"
            ],
            "optimization_target_feasible_from_nominations": overall[
                "optimization_target_feasible_from_nominations"
            ],
            "revision_task_success_rate": overall[
                "revision_task_success_rate"
            ],
            "optimization_conditional_rate": overall[
                "optimization_conditional_rate"
            ],
            "optimization_target_gap": overall["optimization_target_gap"],
            "optimization_target_basis": overall["optimization_target_basis"],
            "perturbator_delivery_rate": overall["perturbator_delivery_rate"],
            "perturbator_absolute_valid_rate": overall[
                "perturbator_absolute_valid_rate"
            ],
            "perturbator_transition_contract_valid_rate": overall[
                "perturbator_transition_contract_valid_rate"
            ],
            "perturbator_unique_output_rate": overall[
                "perturbator_unique_output_rate"
            ],
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
    if (
        lifecycle.refinement_disposition == "ranked_out"
        and lifecycle.variant_id in scheduled_ids
    ):
        raise ValueError("pre-target ranked-out variant cannot be scheduled")
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
            item.outcome is VariantOutcome.TRUNCATED
            and item.refinement_disposition != "ranked_out"
            for item in noncontrols
        ),
        "pre_target_ranked_out": sum(
            item.outcome is VariantOutcome.TRUNCATED
            and item.refinement_disposition == "ranked_out"
            for item in noncontrols
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


def _aggregate_output_funnel(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    funnels = [
        item.get("perturbator", {}).get("output_funnel") or {}
        for item in records
    ]

    def aggregate_items(items: list[dict[str, Any]]) -> dict[str, Any]:
        requested = sum(int(item.get("requested", 0) or 0) for item in items)
        produced = sum(int(item.get("produced", 0) or 0) for item in items)
        absolute_valid = sum(
            int(item.get("absolute_valid", 0) or 0) for item in items
        )
        contract_valid = sum(
            int(item.get("transition_contract_valid", 0) or 0)
            for item in items
        )
        prompt_hashes = {
            str(prompt_hash)
            for item in items
            for prompt_hash in item.get("prompt_hashes") or ()
            if str(prompt_hash)
        }
        return {
            "requested": requested,
            "produced": produced,
            "unique": len(prompt_hashes),
            "absolute_valid": absolute_valid,
            "transition_contract_valid": contract_valid,
            "delivery_rate": _ratio(produced, requested),
            "unique_rate": _ratio(len(prompt_hashes), produced),
            "absolute_valid_rate": _ratio(absolute_valid, produced),
            "transition_contract_valid_rate": _ratio(
                contract_valid,
                produced,
            ),
        }

    overall = aggregate_items(funnels)
    by_purpose: dict[str, dict[str, Any]] = {}
    for purpose in TaskPurpose:
        if purpose is TaskPurpose.CONTROL:
            continue
        purpose_items: list[dict[str, Any]] = []
        for funnel in funnels:
            item = (funnel.get("by_purpose") or {}).get(purpose.value) or {}
            purpose_items.append(
                {
                    **item,
                    "requested": 0,
                }
            )
        by_purpose[purpose.value] = aggregate_items(purpose_items)
        by_purpose[purpose.value].pop("requested", None)
        by_purpose[purpose.value].pop("delivery_rate", None)
    return {**overall, "by_purpose": by_purpose}


def _aggregate_task_funnel(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    tasks = [
        task
        for record in records
        for task in record.get("task_outcomes") or ()
        if isinstance(task, dict)
    ]
    result: dict[str, dict[str, Any]] = {}
    for purpose in TaskPurpose:
        if purpose is TaskPurpose.CONTROL:
            continue
        selected = [
            item for item in tasks if item.get("purpose") == purpose.value
        ]
        attempted = len(selected)
        counts = {
            "with_output": sum(bool(item.get("with_output")) for item in selected),
            "absolute_valid": sum(
                bool(item.get("absolute_valid_output")) for item in selected
            ),
            "transition_contract_valid": sum(
                bool(item.get("transition_contract_valid_output"))
                for item in selected
            ),
            "validated": sum(bool(item.get("validated")) for item in selected),
            "nominated": sum(bool(item.get("nominated")) for item in selected),
            "evaluated": sum(bool(item.get("evaluated")) for item in selected),
            "comparable": sum(bool(item.get("comparable")) for item in selected),
            "improved": sum(bool(item.get("improved")) for item in selected),
            "retained": sum(bool(item.get("retained")) for item in selected),
            "regressed": sum(bool(item.get("regressed")) for item in selected),
        }
        result[purpose.value] = {
            "attempted": attempted,
            **counts,
            "output_delivery_rate": _ratio(counts["with_output"], attempted),
            "absolute_valid_rate": _ratio(counts["absolute_valid"], attempted),
            "transition_contract_valid_rate": _ratio(
                counts["transition_contract_valid"],
                attempted,
            ),
            "valid_rate": _ratio(counts["validated"], attempted),
            "nomination_rate": _ratio(counts["nominated"], attempted),
            "evaluation_rate": _ratio(counts["evaluated"], attempted),
            "end_to_end_improvement_rate": _ratio(
                counts["improved"],
                attempted,
            ),
            "conditional_improvement_rate": _ratio(
                counts["improved"],
                counts["comparable"],
            ),
        }
    return result


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
    output_funnel = _aggregate_output_funnel(model_records)
    task_funnel = _aggregate_task_funnel(model_records)
    optimization_tasks = task_funnel[TaskPurpose.OPTIMIZATION.value]
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
    optimization_output_gap = round(
        optimization["end_to_end_improvement_rate"]
        - OPTIMIZATION_IMPROVEMENT_TARGET,
        4,
    )
    optimization_gap = round(
        optimization_tasks["end_to_end_improvement_rate"]
        - OPTIMIZATION_IMPROVEMENT_TARGET,
        4,
    )
    optimization_nominated = sum(
        item.source is VariantSource.PERTURBATOR
        and item.purpose is TaskPurpose.OPTIMIZATION
        and item.refinement_disposition in {"selected", "unknown"}
        and item.outcome is not VariantOutcome.INVALID
        for item in variants
    )
    optimization_nomination_rate = _ratio(
        optimization_nominated,
        optimization["produced"],
    )
    revision_task_count = sum(
        int(item["perturbator"].get("revision_task_count", 0) or 0)
        for item in model_records
    )
    revision_resolved_task_count = sum(
        int(item["perturbator"].get("revision_resolved_task_count", 0) or 0)
        for item in model_records
    )
    revision_unresolved_task_count = sum(
        int(item["perturbator"].get("revision_unresolved_task_count", 0) or 0)
        for item in model_records
    )
    return {
        "attempts": len(records),
        "model_attempts": len(model_records),
        "by_purpose": by_purpose,
        "task_funnel": task_funnel,
        "output_funnel": output_funnel,
        "optimization_target": OPTIMIZATION_IMPROVEMENT_TARGET,
        "optimization_target_basis": "attempted_optimization_tasks",
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
        "optimization_nomination_rate": optimization_nomination_rate,
        "optimization_nomination_headroom": round(
            optimization_nomination_rate
            - optimization["end_to_end_improvement_rate"],
            4,
        ),
        "optimization_task_attempted": optimization_tasks["attempted"],
        "optimization_task_valid_rate": optimization_tasks["valid_rate"],
        "optimization_task_nomination_rate": optimization_tasks[
            "nomination_rate"
        ],
        "optimization_task_end_to_end_rate": optimization_tasks[
            "end_to_end_improvement_rate"
        ],
        "optimization_task_conditional_rate": optimization_tasks[
            "conditional_improvement_rate"
        ],
        "optimization_task_nomination_headroom": round(
            optimization_tasks["nomination_rate"]
            - optimization_tasks["end_to_end_improvement_rate"],
            4,
        ),
        "optimization_target_feasible_from_nominations": (
            optimization_tasks["nomination_rate"]
            >= OPTIMIZATION_IMPROVEMENT_TARGET
            if optimization_tasks["attempted"]
            else None
        ),
        "optimization_conditional_rate": optimization[
            "conditional_improvement_rate"
        ],
        "optimization_target_gap": optimization_gap,
        "optimization_target_met": optimization_gap >= 0,
        "optimization_output_target_gap": optimization_output_gap,
        "perturbator_delivery_rate": output_funnel["delivery_rate"],
        "perturbator_absolute_valid_rate": output_funnel[
            "absolute_valid_rate"
        ],
        "perturbator_transition_contract_valid_rate": output_funnel[
            "transition_contract_valid_rate"
        ],
        "perturbator_unique_output_rate": output_funnel["unique_rate"],
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
        "revision_attempts": sum(
            bool(item["perturbator"].get("revision_attempted"))
            for item in model_records
        ),
        "revision_tasks": revision_task_count,
        "revision_resolved_tasks": revision_resolved_task_count,
        "revision_unresolved_tasks": revision_unresolved_task_count,
        "revision_task_success_rate": _ratio(
            revision_resolved_task_count,
            revision_task_count,
        ),
        "pre_target_ranked_out": sum(
            int(item["outcome"].get("pre_target_ranked_out", 0) or 0)
            for item in model_records
        ),
        "budget_truncated": sum(
            int(item["outcome"].get("budget_truncated", 0) or 0)
            for item in model_records
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


def _variant_source_from_metadata(
    metadata: Any,
    *,
    purpose: TaskPurpose,
    default: VariantSource,
) -> VariantSource:
    if purpose is TaskPurpose.CONTROL:
        return VariantSource.HARNESS
    if not isinstance(metadata, dict) or metadata.get("variant_source") is None:
        return default
    raw_source = str(metadata["variant_source"])
    try:
        return VariantSource(raw_source)
    except ValueError as exc:
        raise ValueError(f"Unknown variant source: {raw_source}") from exc


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


def _audit_tasks_by_id(audit: dict[str, Any]) -> dict[str, MutationTask]:
    tasks: dict[str, MutationTask] = {}
    for task_data in audit.get("mutation_tasks") or ():
        if not isinstance(task_data, dict):
            continue
        task = MutationTask.from_jsonable(task_data)
        if task.task_id in tasks:
            raise ValueError(f"Duplicate audited mutation task id: {task.task_id}")
        tasks[task.task_id] = task
    return tasks


def _audited_unassigned_outputs(audit: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    assignment = audit.get("variant_assignment")
    if isinstance(assignment, dict):
        records.extend(
            item
            for item in assignment.get("unassigned") or ()
            if isinstance(item, dict)
        )
    for call in audit.get("revision_calls") or ():
        if not isinstance(call, dict):
            continue
        call_assignment = call.get("variant_assignment")
        if not isinstance(call_assignment, dict):
            continue
        records.extend(
            item
            for item in call_assignment.get("unassigned") or ()
            if isinstance(item, dict)
        )
    return records


def _validation_acceptance(value: Any, *, default: bool = False) -> bool:
    if not isinstance(value, dict):
        return default
    return bool(value.get("accepted"))


def _prompt_evidence(
    *,
    prompt: Any,
    validation: Any,
    transition_validation: Any,
    absolute_default: bool = False,
) -> dict[str, Any]:
    normalized = normalized_prompt_key(str(prompt or ""))
    return {
        "prompt_hash": short_hash(normalized, 16) if normalized else "",
        "absolute_valid": _validation_acceptance(
            validation,
            default=absolute_default,
        ),
        "transition_contract_valid": _validation_acceptance(
            transition_validation,
        ),
    }


def _build_output_funnel(
    *,
    produced: int,
    lifecycles: list[VariantLifecycle],
    accepted_candidates: list[Any],
    rejected: list[dict[str, Any]],
    audit: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Separate transport, absolute, transition, uniqueness, and nomination loss."""

    evidence: dict[str, dict[str, Any]] = {}
    for candidate in accepted_candidates:
        metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
        output_id = str(metadata.get("output_id") or candidate.id)
        evidence[output_id] = _prompt_evidence(
            prompt=candidate.prompt,
            validation=metadata.get("validation"),
            transition_validation=metadata.get("transition_validation"),
            absolute_default=True,
        )
    for item in rejected:
        output_id = str(item.get("output_id") or "")
        if not output_id:
            continue
        evidence[output_id] = _prompt_evidence(
            prompt=item.get("prompt"),
            validation=item.get("validation"),
            transition_validation=item.get("transition_validation"),
        )
    for item in _audited_unassigned_outputs(audit):
        output_id = str(item.get("output_id") or "")
        if not output_id:
            continue
        evidence[output_id] = _prompt_evidence(
            prompt=item.get("prompt"),
            validation=None,
            transition_validation=None,
        )

    def lifecycle_evidence(item: VariantLifecycle) -> dict[str, Any]:
        resolved = evidence.get(item.source_output_id) or evidence.get(item.variant_id)
        if resolved is not None:
            return resolved
        return {
            "prompt_hash": short_hash(
                f"unrecorded-output:{item.source_output_id or item.variant_id}",
                16,
            ),
            "absolute_valid": item.stage is not VariantStage.PRODUCED,
            "transition_contract_valid": item.stage is not VariantStage.PRODUCED,
        }

    output_evidence = {
        item.source_output_id or item.variant_id: lifecycle_evidence(item)
        for item in lifecycles
    }
    prompt_hashes = {
        str(item.get("prompt_hash") or "")
        for item in output_evidence.values()
        if item.get("prompt_hash")
    }
    absolute_valid = sum(
        bool(item.get("absolute_valid")) for item in output_evidence.values()
    )
    contract_valid = sum(
        bool(item.get("transition_contract_valid"))
        for item in output_evidence.values()
    )
    variant_counts = audit.get("variant_counts")
    if isinstance(variant_counts, dict):
        requested = int(variant_counts.get("requested_initial", 0) or 0) + int(
            variant_counts.get("requested_revision", 0) or 0
        )
    else:
        requested = int(
            (audit.get("generation_contract") or {}).get("requested", produced)
            or 0
        )
    by_purpose: dict[str, dict[str, Any]] = {}
    for purpose in TaskPurpose:
        if purpose is TaskPurpose.CONTROL:
            continue
        purpose_outputs = [item for item in lifecycles if item.purpose is purpose]
        purpose_evidence = [
            lifecycle_evidence(item) for item in purpose_outputs
        ]
        purpose_hashes = {
            str(item.get("prompt_hash") or "")
            for item in purpose_evidence
            if item.get("prompt_hash")
        }
        purpose_produced = len(purpose_outputs)
        purpose_absolute = sum(
            bool(item.get("absolute_valid")) for item in purpose_evidence
        )
        purpose_contract = sum(
            bool(item.get("transition_contract_valid"))
            for item in purpose_evidence
        )
        by_purpose[purpose.value] = {
            "produced": purpose_produced,
            "unique": len(purpose_hashes),
            "absolute_valid": purpose_absolute,
            "transition_contract_valid": purpose_contract,
            "unique_rate": _ratio(len(purpose_hashes), purpose_produced),
            "absolute_valid_rate": _ratio(purpose_absolute, purpose_produced),
            "transition_contract_valid_rate": _ratio(
                purpose_contract,
                purpose_produced,
            ),
            "prompt_hashes": sorted(purpose_hashes),
        }
    funnel = {
        "requested": requested,
        "produced": produced,
        "unique": len(prompt_hashes),
        "absolute_valid": absolute_valid,
        "transition_contract_valid": contract_valid,
        "delivery_rate": _ratio(produced, requested),
        "unique_rate": _ratio(len(prompt_hashes), produced),
        "absolute_valid_rate": _ratio(absolute_valid, produced),
        "transition_contract_valid_rate": _ratio(contract_valid, produced),
        "prompt_hashes": sorted(prompt_hashes),
        "by_purpose": by_purpose,
    }
    return funnel, output_evidence


def _build_task_outcomes(
    *,
    task_inventory: Iterable[dict[str, Any]],
    variants: list[VariantLifecycle],
    output_evidence: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    for task_data in task_inventory:
        task = MutationTask.from_jsonable(dict(task_data))
        if task.is_control:
            continue
        task_variants = [item for item in variants if item.task_id == task.task_id]
        evidence = [
            output_evidence.get(item.source_output_id or item.variant_id, {})
            for item in task_variants
        ]
        outcomes.append(
            {
                "task_id": task.task_id,
                "purpose": task.purpose.value,
                "operator_id": task.operator_id,
                "output_count": len(task_variants),
                "with_output": bool(task_variants),
                "absolute_valid_output": any(
                    bool(item.get("absolute_valid")) for item in evidence
                ),
                "transition_contract_valid_output": any(
                    bool(item.get("transition_contract_valid"))
                    for item in evidence
                ),
                "validated": any(
                    item.stage is not VariantStage.PRODUCED
                    for item in task_variants
                ),
                "nominated": any(
                    item.refinement_disposition in {"selected", "unknown"}
                    and item.outcome is not VariantOutcome.INVALID
                    for item in task_variants
                ),
                "evaluated": any(
                    item.stage is VariantStage.EVALUATED
                    for item in task_variants
                ),
                "comparable": any(
                    item.outcome
                    in {
                        VariantOutcome.IMPROVED,
                        VariantOutcome.RETAINED,
                        VariantOutcome.REGRESSED,
                    }
                    for item in task_variants
                ),
                "improved": any(
                    item.outcome is VariantOutcome.IMPROVED
                    for item in task_variants
                ),
                "retained": any(
                    item.outcome is VariantOutcome.RETAINED
                    for item in task_variants
                ),
                "regressed": any(
                    item.outcome is VariantOutcome.REGRESSED
                    for item in task_variants
                ),
            }
        )
    return outcomes


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


def _refinement_disposition(metadata: Any) -> str:
    if not isinstance(metadata, dict):
        return "unknown"
    refinement = metadata.get("refinement")
    if not isinstance(refinement, dict):
        return "unknown"
    disposition = str(refinement.get("disposition") or "unknown")
    if disposition not in {"selected", "ranked_out", "unknown"}:
        raise ValueError(f"Unknown refinement disposition: {disposition}")
    return disposition


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
