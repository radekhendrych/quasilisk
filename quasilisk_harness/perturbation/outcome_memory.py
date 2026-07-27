"""Run-local memories for parent-plan and cross-parent discovery outcomes."""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import Any

from quasilisk_harness.perturbation.comparison import score_quality
from quasilisk_harness.perturbation.models import (
    MutationTask,
    PerturbationPlan,
    TaskPurpose,
)


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


@dataclasses.dataclass(frozen=True)
class CoverageFailureEntry:
    """One unsuccessful discovery attempt, independent of parent identity."""

    generation: int
    attempt_id: str
    parent_id: str
    route: str
    case_family: str
    failure_stage: str
    child_count: int

    def is_active_for(self, generation: int, cooldown_generations: int) -> bool:
        return (
            self.generation < generation
            <= self.generation + cooldown_generations
        )

    def to_jsonable(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


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


class CoverageNoveltyMemory:
    """Cross-parent memory for low-signal route/case exploration.

    Parent-plan memory answers whether one exact parent/strategy tuple recently
    regressed. This component answers a different question: whether a discovery
    route or route/case combination has already produced an aligned, evaluated
    child without improvement. Generation and validation failures are audited
    but are not route-performance evidence. Keeping the memories separate lets
    either policy be enabled without silently activating the other.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        combo_failure_limit: int = 1,
        route_failure_limit: int = 2,
        cooldown_generations: int = 3,
        distinct_routes_per_generation: bool = True,
    ) -> None:
        if combo_failure_limit <= 0:
            raise ValueError("combo_failure_limit must be > 0")
        if route_failure_limit <= 0:
            raise ValueError("route_failure_limit must be > 0")
        if cooldown_generations < 0:
            raise ValueError("cooldown_generations must be >= 0")
        self.enabled = bool(enabled)
        self.combo_failure_limit = int(combo_failure_limit)
        self.route_failure_limit = int(route_failure_limit)
        self.cooldown_generations = int(cooldown_generations)
        self.distinct_routes_per_generation = bool(
            distinct_routes_per_generation
        )
        self._failures: list[CoverageFailureEntry] = []
        self._reservations: dict[
            int, dict[tuple[str, str], str]
        ] = defaultdict(dict)

    def expire(self, generation: int) -> None:
        self._failures = [
            entry
            for entry in self._failures
            if entry.is_active_for(generation, self.cooldown_generations)
        ]
        self._reservations = defaultdict(
            dict,
            {
                reserved_generation: reservations
                for reserved_generation, reservations in self._reservations.items()
                if reserved_generation >= generation
            },
        )

    def reserve_plan(
        self,
        plan: PerturbationPlan,
        *,
        generation: int,
        attempt_id: str,
    ) -> None:
        """Reserve discovery coverage before another parent is planned."""

        if not self.enabled:
            return
        self._reservations[generation].setdefault(
            (plan.route, plan.case_family),
            attempt_id,
        )

    def block_reason(
        self,
        plan: PerturbationPlan,
        *,
        generation: int,
    ) -> str | None:
        """Return why this discovery candidate should yield to a novel one."""

        if not self.enabled:
            return None
        reservations = self._reservations.get(generation, {})
        combo = (plan.route, plan.case_family)
        if combo in reservations:
            return "same_generation_combo_reserved"
        if (
            self.distinct_routes_per_generation
            and any(route == plan.route for route, _case in reservations)
        ):
            return "same_generation_route_reserved"

        active = self._active_failure_entries(generation)
        combo_failures = sum(
            entry.route == plan.route
            and entry.case_family == plan.case_family
            for entry in active
        )
        if combo_failures >= self.combo_failure_limit:
            return "recent_combo_failures"
        route_failures = sum(entry.route == plan.route for entry in active)
        if route_failures >= self.route_failure_limit:
            return "recent_route_failures"
        return None

    def record_unusable_attempt(
        self,
        *,
        generation: int,
        attempt_id: str,
        plan: PerturbationPlan,
        accepted_discovery_children: int,
        generation_error: str | None = None,
    ) -> dict[str, Any] | None:
        """Audit an untested plan without treating generation failure as route evidence."""

        if not self.enabled or accepted_discovery_children > 0:
            return None
        return {
            "generation": generation,
            "attempt_id": attempt_id,
            "parent_id": plan.parent_candidate_id,
            "route": plan.route,
            "case_family": plan.case_family,
            "failure_stage": (
                "generation" if generation_error else "validation"
            ),
            "child_count": 0,
            "action": "unusable_attempt_not_attributed",
            "counts_toward_cooldown": False,
            "generation_error": generation_error,
        }

    def record_generation_outcomes(
        self,
        *,
        generation: int,
        parent_scores: dict[str, float],
        child_evaluations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Record whether evaluated discovery children improved their parents."""

        if not self.enabled:
            return []
        records: list[dict[str, Any]] = []
        grouped: dict[
            tuple[str, str, str, str],
            list[bool],
        ] = defaultdict(list)
        for evaluation in child_evaluations:
            metadata = evaluation.get("metadata")
            if not isinstance(metadata, dict):
                continue
            task_data = metadata.get("mutation_task")
            if not isinstance(task_data, dict):
                continue
            try:
                task = MutationTask.from_jsonable(task_data)
            except (TypeError, ValueError):
                continue
            if task.purpose is not TaskPurpose.DISCOVERY:
                continue
            if not task.coverage_participation.learns:
                continue
            plan_data = metadata.get("perturbation_plan")
            if not isinstance(plan_data, dict):
                continue
            try:
                evaluation_plan = PerturbationPlan.from_jsonable(plan_data)
            except (TypeError, ValueError):
                continue
            if evaluation_plan != task.plan:
                continue
            route = str(plan_data.get("route") or "").strip()
            case_family = str(plan_data.get("case_family") or "").strip()
            parent_id = str(evaluation.get("parent_id") or "").strip()
            if not route or not case_family:
                continue
            validation = metadata.get("validation")
            diagnostics = (
                validation.get("diagnostics")
                if isinstance(validation, dict)
                and isinstance(validation.get("diagnostics"), dict)
                else {}
            )
            observed_alignment = task.alignment_requirement.satisfied_by(
                route_aligned=diagnostics.get("route_aligned") is True,
                case_aligned=diagnostics.get("case_aligned") is True,
            )
            if (
                diagnostics.get("alignment_satisfied") is not True
                or not observed_alignment
            ):
                records.append(
                    {
                        "generation": generation,
                        "attempt_id": str(
                            metadata.get("perturbation_attempt_id") or ""
                        ),
                        "parent_id": parent_id,
                        "route": route,
                        "case_family": case_family,
                        "failure_stage": "alignment",
                        "child_count": 1,
                        "action": "alignment_not_attributed",
                        "counts_toward_cooldown": False,
                    }
                )
                continue
            if (
                evaluation.get("error")
                or not evaluation.get("response")
                or score_quality(evaluation) == "unscored"
            ):
                continue
            comparison = metadata.get("parent_child_comparison")
            classification = (
                str(comparison.get("classification") or "")
                if isinstance(comparison, dict)
                else ""
            )
            if classification in {"improved", "retained", "regressed"}:
                improved = classification == "improved"
            elif classification:
                continue
            else:
                parent_score = parent_scores.get(parent_id)
                if parent_score is None:
                    continue
                improved = _score_total(evaluation) > parent_score
            attempt_id = str(
                metadata.get("perturbation_attempt_id")
                or f"{parent_id}|{route}|{case_family}"
            )
            grouped[(attempt_id, parent_id, route, case_family)].append(
                improved
            )

        successful_routes = {
            route
            for (
                _attempt_id,
                _parent_id,
                route,
                _case_family,
            ), outcomes in grouped.items()
            if any(outcomes)
        }
        cleared_by_route = {
            route: self._clear_route(route)
            for route in successful_routes
        }
        for (
            attempt_id,
            parent_id,
            route,
            case_family,
        ), outcomes in sorted(grouped.items()):
            if any(outcomes):
                records.append(
                    {
                        "generation": generation,
                        "attempt_id": attempt_id,
                        "parent_id": parent_id,
                        "route": route,
                        "case_family": case_family,
                        "failure_stage": "evaluation",
                        "child_count": len(outcomes),
                        "action": "route_reopened",
                        "cleared_failure_count": cleared_by_route[route],
                    }
                )
                continue
            if route in successful_routes:
                records.append(
                    {
                        "generation": generation,
                        "attempt_id": attempt_id,
                        "parent_id": parent_id,
                        "route": route,
                        "case_family": case_family,
                        "failure_stage": "evaluation",
                        "child_count": len(outcomes),
                        "action": "failure_superseded_by_route_success",
                    }
                )
                continue
            entry = CoverageFailureEntry(
                generation=generation,
                attempt_id=attempt_id,
                parent_id=parent_id,
                route=route,
                case_family=case_family,
                failure_stage="evaluation",
                child_count=len(outcomes),
            )
            self._append_failure(entry)
            records.append(
                {
                    **entry.to_jsonable(),
                    "action": "failure_recorded",
                    "counts_toward_cooldown": True,
                }
            )
        return records

    def active_failures(self, generation: int) -> list[dict[str, Any]]:
        return [
            entry.to_jsonable()
            for entry in self._active_failure_entries(generation)
        ]

    def _active_failure_entries(
        self,
        generation: int,
    ) -> list[CoverageFailureEntry]:
        return [
            entry
            for entry in self._failures
            if entry.is_active_for(generation, self.cooldown_generations)
        ]

    def _append_failure(self, entry: CoverageFailureEntry) -> None:
        key = (
            entry.generation,
            entry.attempt_id,
            entry.failure_stage,
        )
        if any(
            (
                existing.generation,
                existing.attempt_id,
                existing.failure_stage,
            )
            == key
            for existing in self._failures
        ):
            return
        self._failures.append(entry)

    def _clear_route(self, route: str) -> int:
        before = len(self._failures)
        self._failures = [
            entry for entry in self._failures if entry.route != route
        ]
        return before - len(self._failures)

    def to_jsonable(self, *, generation: int) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "combo_failure_limit": self.combo_failure_limit,
            "route_failure_limit": self.route_failure_limit,
            "cooldown_generations": self.cooldown_generations,
            "distinct_routes_per_generation": (
                self.distinct_routes_per_generation
            ),
            "active_failures": self.active_failures(generation),
            "reservations": [
                {
                    "route": route,
                    "case_family": case_family,
                    "attempt_id": attempt_id,
                }
                for (route, case_family), attempt_id in sorted(
                    self._reservations.get(generation, {}).items()
                )
            ],
        }


def _score_total(evaluation: dict[str, Any]) -> float:
    try:
        return float(evaluation.get("score", {}).get("total", 0) or 0)
    except (TypeError, ValueError):
        return 0.0
