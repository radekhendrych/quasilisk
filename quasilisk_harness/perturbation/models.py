"""Typed records for planned perturbation generation."""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class DefenseProfile:
    primary: str
    labels: list[str]
    evidence: list[str]
    source_candidate_id: str
    score_total: float
    score_notes: list[str]

    def to_jsonable(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class PerturbationPlan:
    plan_id: str
    route: str
    case_family: str
    defense_focus: str
    objective_id: str
    parent_candidate_id: str
    priority: float
    priority_components: dict[str, float]
    rationale: str
    strategy: str = "explore_coverage"
    parent_route: str = ""
    parent_pressure_pattern: str = ""
    parent_commitment_form: str = ""
    mutation_scope: str = ""

    def to_jsonable(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    reasons: list[str]
    rubric: dict[str, float]

    def to_jsonable(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
