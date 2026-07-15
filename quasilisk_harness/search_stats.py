"""Perturbator-attributed search statistics."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .utils import append_jsonl, utc_now, write_json


SEARCH_STATS_SCHEMA_VERSION = 2
ArchiveSnapshot = dict[tuple[str, ...], dict[str, Any]]


class SearchStatsTracker:
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
        parent = {
            "candidate_id": str(parent_eval.get("candidate_id", "")),
            "cell": list(parent_eval.get("archive_cell") or []),
            "score": _score_total(parent_eval),
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
                "fallback_used": False,
                "retry_count": 0,
                "usage": {},
                "error": None,
            },
            "accepted_candidate_ids": [],
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
        generated_candidates = [
            candidate
            for candidate in accepted_candidates
            if candidate.metadata.get("evaluation_role") != "control"
        ]
        control_candidates = [
            candidate
            for candidate in accepted_candidates
            if candidate.metadata.get("evaluation_role") == "control"
        ]
        attempt["plan"] = _compact_plan(plan)
        attempt["perturbator"] = {
            "produced": int(produced),
            "accepted": len(generated_candidates),
            "controls": len(control_candidates),
            "fallback_used": bool(audit.get("fallback_used")),
            "retry_count": int(audit.get("retry_count", 0) or 0),
            "usage": dict(audit.get("usage") or {}),
            "error": error,
        }
        attempt["accepted_candidate_ids"] = [
            str(candidate.id) for candidate in accepted_candidates
        ]
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
        scheduled_by_attempt: dict[str, set[str]] = defaultdict(set)
        task_by_candidate: dict[str, dict[str, Any]] = {}
        for candidate in scheduled_candidates:
            attempt_id = str(candidate.metadata.get("perturbation_attempt_id") or "")
            if attempt_id:
                scheduled_by_attempt[attempt_id].add(str(candidate.id))
            task = candidate.metadata.get("mutation_task")
            if isinstance(task, dict):
                task_by_candidate[str(candidate.id)] = task

        evaluations_by_attempt: dict[str, list[dict[str, Any]]] = defaultdict(list)
        successful_evaluation_by_id: dict[str, dict[str, Any]] = {}
        for evaluation in evaluations:
            metadata = evaluation.get("metadata")
            attempt_id = (
                str(metadata.get("perturbation_attempt_id") or "")
                if isinstance(metadata, dict)
                else ""
            )
            if not attempt_id:
                continue
            evaluations_by_attempt[attempt_id].append(evaluation)
            candidate_id = str(evaluation.get("candidate_id", ""))
            if not evaluation.get("error") and evaluation.get("response"):
                successful_evaluation_by_id[candidate_id] = evaluation

        retained_ids = {
            str(candidate_id)
            for cell in archive_after.values()
            for candidate_id in cell.get("candidate_ids", ())
        }
        cell_credits, global_credit = _archive_credits(
            archive_before=archive_before,
            archive_after=archive_after,
            evaluation_by_id=successful_evaluation_by_id,
        )

        completed: list[dict[str, Any]] = []
        for attempt_id, attempt in self._attempts.items():
            if attempt["generation"] != generation or attempt.get("completed"):
                continue
            child_evaluations = evaluations_by_attempt.get(attempt_id, [])
            successful = [
                evaluation
                for evaluation in child_evaluations
                if not evaluation.get("error") and evaluation.get("response")
            ]
            parent_score = float(attempt["parent"]["score"])
            child_scores = [_score_total(item) for item in successful]
            deltas = [score - parent_score for score in child_scores]
            comparison_records: list[dict[str, Any]] = []
            operator_outcomes: dict[str, dict[str, int]] = defaultdict(
                lambda: {
                    "evaluated": 0,
                    "comparable": 0,
                    "improved": 0,
                    "retained": 0,
                    "regressed": 0,
                    "unscored": 0,
                }
            )
            improvement_goal = {
                "evaluated": 0,
                "comparable": 0,
                "improved": 0,
                "retained": 0,
                "regressed": 0,
                "unscored": 0,
            }
            validation_tiers: Counter[str] = Counter()
            improvement_goal_validation_tiers: Counter[str] = Counter()
            for evaluation in child_evaluations:
                metadata = evaluation.get("metadata")
                comparison = (
                    metadata.get("parent_child_comparison")
                    if isinstance(metadata, dict)
                    else None
                )
                candidate_id = str(evaluation.get("candidate_id", ""))
                task = task_by_candidate.get(candidate_id, {})
                operator_id = str(task.get("operator_id") or "unspecified_task")
                role = str(task.get("evaluation_role") or "search")
                counts_toward_goal = bool(
                    task.get("counts_toward_improvement_goal")
                )
                if role == "control":
                    operator_id = "replay_control"
                aggregate = operator_outcomes[operator_id]
                aggregate["evaluated"] += 1
                tier = _validation_tier(metadata)
                validation_tiers[tier] += 1
                if counts_toward_goal:
                    improvement_goal["evaluated"] += 1
                    improvement_goal_validation_tiers[tier] += 1
                if isinstance(comparison, dict):
                    comparison_records.append(comparison)
                    classification = str(
                        comparison.get("classification") or "unscored"
                    )
                    if comparison.get("comparable"):
                        aggregate["comparable"] += 1
                        if counts_toward_goal:
                            improvement_goal["comparable"] += 1
                    aggregate[classification] = int(
                        aggregate.get(classification, 0)
                    ) + 1
                    if counts_toward_goal:
                        improvement_goal[classification] = int(
                            improvement_goal.get(classification, 0)
                        ) + 1
                else:
                    aggregate["comparable"] += 1
                    if counts_toward_goal:
                        improvement_goal["comparable"] += 1
                    if _score_total(evaluation) > parent_score:
                        aggregate["improved"] += 1
                        if counts_toward_goal:
                            improvement_goal["improved"] += 1
                    elif _score_total(evaluation) == parent_score:
                        aggregate["retained"] += 1
                        if counts_toward_goal:
                            improvement_goal["retained"] += 1
                    else:
                        aggregate["regressed"] += 1
                        if counts_toward_goal:
                            improvement_goal["regressed"] += 1
            comparable = [
                item for item in comparison_records if item.get("comparable")
            ]
            classified_improved = sum(
                item.get("classification") == "improved" for item in comparable
            )
            classified_retained = sum(
                item.get("classification") == "retained" for item in comparable
            )
            classified_regressed = sum(
                item.get("classification") == "regressed" for item in comparable
            )
            accepted_ids = set(attempt.get("accepted_candidate_ids", ()))
            scheduled_ids = scheduled_by_attempt.get(attempt_id, set())
            credit = cell_credits.get(attempt_id, {})
            outcome = {
                "parent_score": parent_score,
                "scheduled": len(scheduled_ids),
                "evaluated": len(child_evaluations),
                "non_error_children": len(successful),
                "errored_children": len(child_evaluations) - len(successful),
                "budget_truncated": max(0, len(accepted_ids) - len(scheduled_ids)),
                "best_child_score": max(child_scores) if child_scores else None,
                "best_score_delta": max(deltas) if deltas else None,
                "mean_score_delta": _mean(deltas),
                "score_delta_sum": round(sum(deltas), 3),
                "positive_score_gain": round(
                    sum(max(0.0, delta) for delta in deltas),
                    3,
                ),
                "comparable_children": (
                    len(comparable) if comparison_records else len(successful)
                ),
                "unscored_children": (
                    len(comparison_records) - len(comparable)
                    if comparison_records
                    else 0
                ),
                "improved_children": (
                    classified_improved
                    if comparison_records
                    else sum(1 for delta in deltas if delta > 0)
                ),
                "retained_children": (
                    classified_retained
                    if comparison_records
                    else sum(1 for delta in deltas if delta == 0)
                ),
                "regressed_children": (
                    classified_regressed
                    if comparison_records
                    else sum(1 for delta in deltas if delta < 0)
                ),
                "improvement_goal_children": improvement_goal["evaluated"],
                "improvement_goal_comparable_children": improvement_goal["comparable"],
                "improvement_goal_improved_children": improvement_goal["improved"],
                "improvement_goal_retained_children": improvement_goal["retained"],
                "improvement_goal_regressed_children": improvement_goal["regressed"],
                "improvement_goal_unscored_children": improvement_goal["unscored"],
                "validation_tiers": dict(sorted(validation_tiers.items())),
                "improvement_goal_validation_tiers": dict(
                    sorted(improvement_goal_validation_tiers.items())
                ),
                "final_archive_retained": sum(
                    1
                    for evaluation in successful
                    if str(evaluation.get("candidate_id", "")) in retained_ids
                ),
                "new_cells_created": int(credit.get("new_cells_created", 0)),
                "cell_best_improvements": int(
                    credit.get("cell_best_improvements", 0)
                ),
                "cell_best_score_gain": round(
                    float(credit.get("cell_best_score_gain", 0.0)),
                    3,
                ),
                "global_best_delta": round(
                    global_credit.get(attempt_id, 0.0),
                    3,
                ),
            }
            record = {
                "event_type": "attempt_completed",
                "attempt_id": attempt_id,
                "generation": generation,
                "parent": attempt["parent"],
                "plan": attempt["plan"],
                "perturbator": {
                    **attempt["perturbator"],
                    "evaluated": len(child_evaluations),
                    "budget_truncated": outcome["budget_truncated"],
                },
                "outcome": outcome,
                "operator_outcomes": dict(operator_outcomes),
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
        by_operator: dict[str, dict[str, int]] = defaultdict(
            lambda: {
                "evaluated": 0,
                "comparable": 0,
                "improved": 0,
                "retained": 0,
                "regressed": 0,
                "unscored": 0,
            }
        )
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
            for operator_id, outcome in (
                record.get("operator_outcomes") or {}
            ).items():
                for key in by_operator[operator_id]:
                    by_operator[operator_id][key] += int(outcome.get(key, 0))
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
            "by_operator": {
                operator_id: {
                    **outcome,
                    "improvement_rate": _ratio(
                        outcome["improved"], outcome["comparable"]
                    ),
                    "retention_rate": _ratio(
                        outcome["improved"] + outcome["retained"],
                        outcome["comparable"],
                    ),
                }
                for operator_id, outcome in sorted(by_operator.items())
            },
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
        }

    def status_metrics(self) -> dict[str, Any]:
        summary = self.summary()
        overall = summary["overall"]
        return {
            "search_attempts_started": summary["attempts_started"],
            "search_attempts_completed": summary["attempts_completed"],
            "perturbator_validity_rate": overall["validity_rate"],
            "perturbator_improvement_rate": overall["improvement_rate"],
            "perturbator_improvement_goal_rate": overall["improvement_goal_rate"],
            "perturbator_improvement_goal_retention_rate": overall[
                "improvement_goal_retention_rate"
            ],
            "perturbator_improvement_goal_regression_rate": overall[
                "improvement_goal_regression_rate"
            ],
            "perturbator_retention_rate": overall["retention_rate"],
            "perturbator_regression_rate": overall["regression_rate"],
            "perturbator_unscored_rate": overall["unscored_rate"],
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


def _aggregate_groups(
    groups: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    return {
        key: _aggregate(records)
        for key, records in sorted(groups.items())
    }


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
            "archive_yield": metrics["archive_yield"],
            "new_cell_yield": metrics["new_cell_yield"],
        }
        for cell, metrics in ranked[:5]
    ]


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    attempts = len(records)
    produced = sum(int(item["perturbator"].get("produced", 0)) for item in records)
    accepted = sum(int(item["perturbator"].get("accepted", 0)) for item in records)
    evaluated = sum(int(item["outcome"].get("evaluated", 0)) for item in records)
    non_error = sum(
        int(item["outcome"].get("non_error_children", 0)) for item in records
    )
    improved = sum(
        int(item["outcome"].get("improved_children", 0)) for item in records
    )
    comparable = sum(
        int(
            item["outcome"].get(
                "comparable_children",
                item["outcome"].get("non_error_children", 0),
            )
        )
        for item in records
    )
    retained_comparisons = sum(
        int(item["outcome"].get("retained_children", 0)) for item in records
    )
    regressed_comparisons = sum(
        int(item["outcome"].get("regressed_children", 0)) for item in records
    )
    unscored = sum(
        int(item["outcome"].get("unscored_children", 0)) for item in records
    )
    validation_tiers = _sum_counter_field(records, "validation_tiers")
    improvement_goal_validation_tiers = _sum_counter_field(
        records,
        "improvement_goal_validation_tiers",
    )
    goal_evaluated = sum(
        int(item["outcome"].get("improvement_goal_children", 0))
        for item in records
    )
    goal_comparable = sum(
        int(item["outcome"].get("improvement_goal_comparable_children", 0))
        for item in records
    )
    goal_improved = sum(
        int(item["outcome"].get("improvement_goal_improved_children", 0))
        for item in records
    )
    goal_retained = sum(
        int(item["outcome"].get("improvement_goal_retained_children", 0))
        for item in records
    )
    goal_regressed = sum(
        int(item["outcome"].get("improvement_goal_regressed_children", 0))
        for item in records
    )
    goal_unscored = sum(
        int(item["outcome"].get("improvement_goal_unscored_children", 0))
        for item in records
    )
    retained = sum(
        int(item["outcome"].get("final_archive_retained", 0)) for item in records
    )
    new_cells = sum(
        int(item["outcome"].get("new_cells_created", 0)) for item in records
    )
    budget_truncated = sum(
        int(item["outcome"].get("budget_truncated", 0)) for item in records
    )
    improving_attempts = sum(
        1
        for item in records
        if float(item["outcome"].get("best_score_delta", 0.0) or 0.0) > 0
    )
    cell_best_improvements = sum(
        int(item["outcome"].get("cell_best_improvements", 0))
        for item in records
    )
    cell_best_score_gain = sum(
        float(item["outcome"].get("cell_best_score_gain", 0.0) or 0.0)
        for item in records
    )
    global_best_delta = sum(
        float(item["outcome"].get("global_best_delta", 0.0) or 0.0)
        for item in records
    )
    delta_sum = sum(
        float(item["outcome"].get("score_delta_sum", 0.0) or 0.0)
        for item in records
    )
    positive_gain = sum(
        float(item["outcome"].get("positive_score_gain", 0.0) or 0.0)
        for item in records
    )
    best_deltas = [
        float(item["outcome"]["best_score_delta"])
        for item in records
        if item["outcome"].get("best_score_delta") is not None
    ]
    model_records = [
        item for item in records if not item["perturbator"].get("fallback_used")
    ]
    model_produced = sum(
        int(item["perturbator"].get("produced", 0)) for item in model_records
    )
    model_accepted = sum(
        int(item["perturbator"].get("accepted", 0)) for item in model_records
    )
    model_evaluated = sum(
        int(item["outcome"].get("evaluated", 0)) for item in model_records
    )
    model_non_error = sum(
        int(item["outcome"].get("non_error_children", 0))
        for item in model_records
    )
    model_improved = sum(
        int(item["outcome"].get("improved_children", 0))
        for item in model_records
    )
    model_comparable = sum(
        int(
            item["outcome"].get(
                "comparable_children",
                item["outcome"].get("non_error_children", 0),
            )
        )
        for item in model_records
    )
    model_retained_comparisons = sum(
        int(item["outcome"].get("retained_children", 0))
        for item in model_records
    )
    model_regressed_comparisons = sum(
        int(item["outcome"].get("regressed_children", 0))
        for item in model_records
    )
    model_unscored = sum(
        int(item["outcome"].get("unscored_children", 0))
        for item in model_records
    )
    model_goal_evaluated = sum(
        int(item["outcome"].get("improvement_goal_children", 0))
        for item in model_records
    )
    model_goal_comparable = sum(
        int(item["outcome"].get("improvement_goal_comparable_children", 0))
        for item in model_records
    )
    model_goal_improved = sum(
        int(item["outcome"].get("improvement_goal_improved_children", 0))
        for item in model_records
    )
    model_goal_retained = sum(
        int(item["outcome"].get("improvement_goal_retained_children", 0))
        for item in model_records
    )
    model_goal_regressed = sum(
        int(item["outcome"].get("improvement_goal_regressed_children", 0))
        for item in model_records
    )
    model_goal_unscored = sum(
        int(item["outcome"].get("improvement_goal_unscored_children", 0))
        for item in model_records
    )
    model_retained = sum(
        int(item["outcome"].get("final_archive_retained", 0))
        for item in model_records
    )
    model_new_cells = sum(
        int(item["outcome"].get("new_cells_created", 0))
        for item in model_records
    )
    model_delta_sum = sum(
        float(item["outcome"].get("score_delta_sum", 0.0) or 0.0)
        for item in model_records
    )
    model_positive_gain = sum(
        float(item["outcome"].get("positive_score_gain", 0.0) or 0.0)
        for item in model_records
    )
    model_best_deltas = [
        float(item["outcome"]["best_score_delta"])
        for item in model_records
        if item["outcome"].get("best_score_delta") is not None
    ]
    model_improving_attempts = sum(
        1
        for item in model_records
        if float(item["outcome"].get("best_score_delta", 0.0) or 0.0) > 0
    )
    total_tokens = sum(
        _total_tokens(item["perturbator"].get("usage"))
        for item in model_records
    )
    return {
        "attempts": attempts,
        "model_attempts": len(model_records),
        "produced": produced,
        "accepted": accepted,
        "budget_truncated": budget_truncated,
        "evaluated": evaluated,
        "non_error_children": non_error,
        "improved_children": improved,
        "comparable_children": comparable,
        "retained_children": retained_comparisons,
        "regressed_children": regressed_comparisons,
        "unscored_children": unscored,
        "validation_tiers": validation_tiers,
        "improvement_goal_children": goal_evaluated,
        "improvement_goal_comparable_children": goal_comparable,
        "improvement_goal_improved_children": goal_improved,
        "improvement_goal_retained_children": goal_retained,
        "improvement_goal_regressed_children": goal_regressed,
        "improvement_goal_unscored_children": goal_unscored,
        "improvement_goal_validation_tiers": improvement_goal_validation_tiers,
        "final_archive_retained": retained,
        "new_cells_created": new_cells,
        "distinct_parent_count": len(
            {
                str(item["parent"].get("candidate_id", ""))
                for item in records
            }
        ),
        "improving_attempts": model_improving_attempts,
        "attempt_improvement_rate": _ratio(
            model_improving_attempts,
            len(model_records),
        ),
        "cell_best_improvements": cell_best_improvements,
        "cell_best_score_gain": round(cell_best_score_gain, 3),
        "global_best_delta": round(global_best_delta, 3),
        "fallback_attempts": sum(
            1 for item in records if item["perturbator"].get("fallback_used")
        ),
        "errored_attempts": sum(
            1 for item in records if item["perturbator"].get("error")
        ),
        "validity_rate": _ratio(model_accepted, model_produced),
        "evaluation_yield": _ratio(model_non_error, model_evaluated),
        "improvement_rate": _ratio(model_improved, model_comparable),
        "improvement_goal_rate": _ratio(
            model_goal_improved,
            model_goal_comparable,
        ),
        "improvement_goal_retention_rate": _ratio(
            model_goal_improved + model_goal_retained,
            model_goal_comparable,
        ),
        "improvement_goal_regression_rate": _ratio(
            model_goal_regressed,
            model_goal_comparable,
        ),
        "improvement_goal_unscored_rate": _ratio(
            model_goal_unscored,
            model_goal_evaluated,
        ),
        "retention_rate": _ratio(
            model_improved + model_retained_comparisons,
            model_comparable,
        ),
        "regression_rate": _ratio(model_regressed_comparisons, model_comparable),
        "unscored_rate": _ratio(model_unscored, model_evaluated),
        "archive_yield": _ratio(model_retained, model_evaluated),
        "new_cell_yield": _ratio(model_new_cells, model_evaluated),
        "mean_score_delta": _ratio(model_delta_sum, model_non_error),
        "mean_best_score_delta": _mean(model_best_deltas),
        "best_score_delta": (
            round(max(model_best_deltas), 3)
            if model_best_deltas
            else None
        ),
        "positive_score_gain": round(model_positive_gain, 3),
        "positive_gain_per_target_call": _ratio(
            model_positive_gain,
            model_evaluated,
        ),
        "search_validity_rate": _ratio(accepted, produced),
        "search_improving_attempts": improving_attempts,
        "search_attempt_improvement_rate": _ratio(
            improving_attempts,
            attempts,
        ),
        "search_improvement_rate": _ratio(improved, comparable),
        "search_improvement_goal_rate": _ratio(goal_improved, goal_comparable),
        "search_improvement_goal_retention_rate": _ratio(
            goal_improved + goal_retained,
            goal_comparable,
        ),
        "search_improvement_goal_regression_rate": _ratio(
            goal_regressed,
            goal_comparable,
        ),
        "search_improvement_goal_unscored_rate": _ratio(
            goal_unscored,
            goal_evaluated,
        ),
        "search_retention_rate": _ratio(
            improved + retained_comparisons,
            comparable,
        ),
        "search_regression_rate": _ratio(regressed_comparisons, comparable),
        "search_unscored_rate": _ratio(unscored, evaluated),
        "search_archive_yield": _ratio(retained, evaluated),
        "search_new_cell_yield": _ratio(new_cells, evaluated),
        "search_mean_score_delta": _ratio(delta_sum, non_error),
        "search_mean_best_score_delta": _mean(best_deltas),
        "search_positive_score_gain": round(positive_gain, 3),
        "perturbator_total_tokens": total_tokens,
        "positive_gain_per_1k_perturbator_tokens": (
            _ratio(model_positive_gain * 1000.0, total_tokens)
            if total_tokens
            else None
        ),
    }


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
    transition = metadata.get("transition_validation")
    if isinstance(transition, dict) and transition.get("tier"):
        return str(transition["tier"])
    validation = metadata.get("validation")
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


def _score_total(evaluation: dict[str, Any]) -> float:
    try:
        return float(evaluation.get("score", {}).get("total", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


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
