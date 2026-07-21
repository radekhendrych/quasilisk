"""Run-local memory of parent-plan outcomes."""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import Any

from quasilisk_harness.perturbation.comparison import score_quality
from quasilisk_harness.perturbation.models import PerturbationPlan, TaskPurpose


VALID_PARENT_PLAN_MEMORY_MODES = {"block", "penalize"}


@dataclasses.dataclass(frozen=True)
class ParentPlanKey:
    parent_id: str
    strategy: str
    route: str
    case_family: str

    @classmethod
    def from_plan(cls, plan: PerturbationPlan) -> "ParentPlanKey":
        return cls(
            parent_id=plan.parent_candidate_id,
            strategy=plan.strategy or "explore_coverage",
            route=plan.route,
            case_family=plan.case_family,
        )

    @classmethod
    def from_plan_data(
        cls,
        plan_data: dict[str, Any],
        *,
        fallback_parent_id: str = "",
    ) -> "ParentPlanKey | None":
        route = str(plan_data.get("route") or "").strip()
        case_family = str(plan_data.get("case_family") or "").strip()
        if not route or not case_family:
            return None
        return cls(
            parent_id=str(
                plan_data.get("parent_candidate_id") or fallback_parent_id
            ).strip(),
            strategy=str(plan_data.get("strategy") or "explore_coverage").strip(),
            route=route,
            case_family=case_family,
        )

    def to_jsonable(self) -> dict[str, str]:
        return dataclasses.asdict(self)

    def label(self) -> str:
        return "|".join((self.parent_id, self.strategy, self.route, self.case_family))


@dataclasses.dataclass
class ParentPlanMemoryEntry:
    key: ParentPlanKey
    generation: int
    parent_score: float
    best_child_score: float
    threshold: float
    blocked_until_generation: int
    child_count: int
    mode: str

    def is_active_for(self, generation: int) -> bool:
        return self.generation < generation <= self.blocked_until_generation

    def to_jsonable(self) -> dict[str, Any]:
        return {
            **self.key.to_jsonable(),
            "generation": self.generation,
            "parent_score": round(self.parent_score, 3),
            "best_child_score": round(self.best_child_score, 3),
            "threshold": round(self.threshold, 3),
            "blocked_until_generation": self.blocked_until_generation,
            "child_count": self.child_count,
            "mode": self.mode,
        }


class ParentPlanOutcomeMemory:
    def __init__(
        self,
        *,
        enabled: bool,
        mode: str,
        regression_threshold: float,
        cooldown_generations: int,
        penalty: float,
        min_evaluated_children: int,
    ) -> None:
        if mode not in VALID_PARENT_PLAN_MEMORY_MODES:
            raise ValueError(f"Invalid parent-plan memory mode: {mode}")
        self.enabled = enabled
        self.mode = mode
        self.regression_threshold = float(regression_threshold)
        self.cooldown_generations = int(cooldown_generations)
        self.penalty_amount = float(penalty)
        self.min_evaluated_children = int(min_evaluated_children)
        self._active: dict[ParentPlanKey, ParentPlanMemoryEntry] = {}

    def expire(self, generation: int) -> None:
        expired = [
            key
            for key, entry in self._active.items()
            if entry.blocked_until_generation < generation
        ]
        for key in expired:
            del self._active[key]

    def active_entries(self, generation: int) -> list[dict[str, Any]]:
        return [
            entry.to_jsonable()
            for entry in self._active.values()
            if entry.is_active_for(generation)
        ]

    def is_active(self, key: ParentPlanKey, generation: int) -> bool:
        if not self.enabled:
            return False
        entry = self._active.get(key)
        return bool(entry and entry.is_active_for(generation))

    def is_blocked(self, key: ParentPlanKey, generation: int) -> bool:
        return self.mode == "block" and self.is_active(key, generation)

    def penalty(self, key: ParentPlanKey, generation: int) -> float:
        if self.mode != "penalize" or not self.is_active(key, generation):
            return 0.0
        return self.penalty_amount

    def plan_key_is_blocked(self, plan: PerturbationPlan, generation: int) -> bool:
        return self.is_blocked(ParentPlanKey.from_plan(plan), generation)

    def plan_penalty(self, plan: PerturbationPlan, generation: int) -> float:
        return self.penalty(ParentPlanKey.from_plan(plan), generation)

    def record_generation_outcomes(
        self,
        *,
        generation: int,
        parent_scores: dict[str, float],
        child_evaluations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not self.enabled:
            return []

        grouped: dict[ParentPlanKey, list[float]] = defaultdict(list)
        skipped_children: dict[ParentPlanKey, int] = defaultdict(int)
        for evaluation in child_evaluations:
            metadata = evaluation.get("metadata")
            if (
                isinstance(metadata, dict)
                and metadata.get("task_purpose") == TaskPurpose.CONTROL.value
            ):
                continue
            plan_data = (
                metadata.get("perturbation_plan")
                if isinstance(metadata, dict)
                else None
            )
            if not isinstance(plan_data, dict):
                continue
            key = ParentPlanKey.from_plan_data(
                plan_data,
                fallback_parent_id=str(evaluation.get("parent_id", "")),
            )
            if key is None:
                continue
            if (
                evaluation.get("error")
                or not evaluation.get("response")
                or score_quality(evaluation) == "unscored"
            ):
                skipped_children[key] += 1
                continue
            grouped[key].append(_score_total(evaluation))

        records: list[dict[str, Any]] = []
        all_keys = set(grouped) | set(skipped_children)
        for key in sorted(all_keys, key=lambda item: item.label()):
            child_scores = grouped.get(key, [])
            child_count = len(child_scores)
            parent_score = parent_scores.get(key.parent_id)
            base_record: dict[str, Any] = {
                **key.to_jsonable(),
                "generation": generation,
                "threshold": self.regression_threshold,
                "mode": self.mode,
                "skipped_child_count": skipped_children.get(key, 0),
                "child_count": child_count,
            }
            if parent_score is None:
                records.append({**base_record, "action": "skipped_missing_parent_score"})
                continue
            base_record["parent_score"] = round(parent_score, 3)
            if child_count < self.min_evaluated_children:
                records.append(
                    {
                        **base_record,
                        "action": "skipped_insufficient_evaluated_children",
                    }
                )
                continue
            best_child_score = max(child_scores)
            base_record["best_child_score"] = round(best_child_score, 3)
            if best_child_score < parent_score - self.regression_threshold:
                blocked_until = generation + self.cooldown_generations
                self._active[key] = ParentPlanMemoryEntry(
                    key=key,
                    generation=generation,
                    parent_score=parent_score,
                    best_child_score=best_child_score,
                    threshold=self.regression_threshold,
                    blocked_until_generation=blocked_until,
                    child_count=child_count,
                    mode=self.mode,
                )
                records.append(
                    {
                        **base_record,
                        "action": "blocked" if self.mode == "block" else "penalized",
                        "blocked_until_generation": blocked_until,
                        "penalty": self.penalty_amount if self.mode == "penalize" else 0.0,
                    }
                )
            else:
                self._active.pop(key, None)
                records.append({**base_record, "action": "retained"})
        return records

    def to_jsonable(self, *, generation: int) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "regression_threshold": self.regression_threshold,
            "cooldown_generations": self.cooldown_generations,
            "penalty": self.penalty_amount,
            "min_evaluated_children": self.min_evaluated_children,
            "active": self.active_entries(generation),
        }


def _score_total(evaluation: dict[str, Any]) -> float:
    try:
        return float(evaluation.get("score", {}).get("total", 0) or 0)
    except (TypeError, ValueError):
        return 0.0
