"""Run-local, quality-aware mutation-operator outcome memory."""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import Any

from quasilisk_harness.perturbation.comparison import compare_evaluations
from quasilisk_harness.perturbation.models import (
    ComparisonResult,
    MutationTask,
    PromptFingerprint,
)
from quasilisk_harness.utils import short_hash


@dataclasses.dataclass(frozen=True)
class OperatorKey:
    parent_fingerprint_hash: str
    operator_id: str
    factor: str
    contract_hash: str

    def label(self) -> str:
        return "|".join(dataclasses.astuple(self))


class OperatorOutcomeLedger:
    def __init__(
        self,
        *,
        enabled: bool,
        max_component_drop: float,
        block_after_regressions: int = 1,
    ) -> None:
        self.enabled = enabled
        self.max_component_drop = float(max_component_drop)
        self.block_after_regressions = int(block_after_regressions)
        self._outcomes: dict[OperatorKey, dict[str, Any]] = {}

    @staticmethod
    def key_for(task: MutationTask, fingerprint: PromptFingerprint) -> OperatorKey:
        return OperatorKey(
            parent_fingerprint_hash=fingerprint.fingerprint_hash,
            operator_id=task.operator_id,
            factor=task.factor,
            contract_hash=short_hash(repr(task.contract.to_jsonable()), 16),
        )

    def is_blocked(
        self,
        task: MutationTask,
        fingerprint: PromptFingerprint,
    ) -> bool:
        if not self.enabled or task.evaluation_role == "control":
            return False
        outcome = self._outcomes.get(self.key_for(task, fingerprint))
        return bool(
            outcome
            and int(outcome.get("regressed", 0)) >= self.block_after_regressions
            and int(outcome.get("improved", 0)) == 0
        )

    def filter_tasks(
        self,
        tasks: list[MutationTask],
        fingerprint: PromptFingerprint,
    ) -> tuple[list[MutationTask], list[dict[str, Any]]]:
        accepted: list[MutationTask] = []
        blocked: list[dict[str, Any]] = []
        for task in tasks:
            if self.is_blocked(task, fingerprint):
                blocked.append(
                    {
                        "task_id": task.task_id,
                        "operator_id": task.operator_id,
                        "factor": task.factor,
                        "reason": "operator_outcome_blocked",
                    }
                )
            else:
                accepted.append(task)
        return accepted, blocked

    def record(
        self,
        *,
        parents: dict[str, dict[str, Any]],
        evaluations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        grouped: dict[OperatorKey, list[dict[str, Any]]] = defaultdict(list)
        for evaluation in evaluations:
            metadata = evaluation.get("metadata")
            task_data = (
                metadata.get("mutation_task") if isinstance(metadata, dict) else None
            )
            fingerprint_data = (
                metadata.get("parent_fingerprint") if isinstance(metadata, dict) else None
            )
            if not isinstance(task_data, dict) or not isinstance(fingerprint_data, dict):
                continue
            task = MutationTask.from_jsonable(task_data)
            if task.operator_id == "legacy_planned_rewrite":
                continue
            fingerprint = PromptFingerprint.from_jsonable(fingerprint_data)
            parent = parents.get(str(evaluation.get("parent_id", "")))
            if parent is None:
                continue
            comparison_data = metadata.get("parent_child_comparison")
            if isinstance(comparison_data, dict):
                comparison = ComparisonResult.from_jsonable(comparison_data)
            else:
                comparison = compare_evaluations(
                    parent,
                    evaluation,
                    max_component_drop=self.max_component_drop,
                )
                evaluation.setdefault("metadata", {})[
                    "parent_child_comparison"
                ] = comparison.to_jsonable()
            if not comparison.comparable or task.evaluation_role == "control":
                continue
            grouped[self.key_for(task, fingerprint)].append(
                comparison.to_jsonable()
            )

        records: list[dict[str, Any]] = []
        for key, comparisons in grouped.items():
            outcome = self._outcomes.setdefault(
                key,
                {
                    "improved": 0,
                    "retained": 0,
                    "regressed": 0,
                    "comparisons": 0,
                },
            )
            for comparison in comparisons:
                classification = str(comparison["classification"])
                outcome[classification] = int(outcome.get(classification, 0)) + 1
                outcome["comparisons"] = int(outcome.get("comparisons", 0)) + 1
            records.append(
                {
                    "operator_key": {
                        "parent_fingerprint_hash": key.parent_fingerprint_hash,
                        "operator_id": key.operator_id,
                        "factor": key.factor,
                        "contract_hash": key.contract_hash,
                    },
                    "outcome": dict(outcome),
                    "blocked": (
                        int(outcome.get("regressed", 0))
                        >= self.block_after_regressions
                        and int(outcome.get("improved", 0)) == 0
                    ),
                }
            )
        return records

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_component_drop": self.max_component_drop,
            "block_after_regressions": self.block_after_regressions,
            "entries": [
                {
                    "key": {
                        "parent_fingerprint_hash": key.parent_fingerprint_hash,
                        "operator_id": key.operator_id,
                        "factor": key.factor,
                        "contract_hash": key.contract_hash,
                    },
                    "outcome": dict(outcome),
                }
                for key, outcome in sorted(
                    self._outcomes.items(), key=lambda item: item[0].label()
                )
            ],
        }
