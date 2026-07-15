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

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> "PerturbationPlan":
        return cls(
            plan_id=str(data.get("plan_id", "")),
            route=str(data.get("route", "")),
            case_family=str(data.get("case_family", "")),
            defense_focus=str(data.get("defense_focus", "")),
            objective_id=str(data.get("objective_id", "")),
            parent_candidate_id=str(data.get("parent_candidate_id", "")),
            priority=float(data.get("priority", 0) or 0),
            priority_components={
                str(key): float(value)
                for key, value in (data.get("priority_components") or {}).items()
            },
            rationale=str(data.get("rationale", "")),
            strategy=str(data.get("strategy", "explore_coverage")),
            parent_route=str(data.get("parent_route", "")),
            parent_pressure_pattern=str(data.get("parent_pressure_pattern", "")),
            parent_commitment_form=str(data.get("parent_commitment_form", "")),
            mutation_scope=str(data.get("mutation_scope", "")),
        )


@dataclasses.dataclass(frozen=True)
class PromptFingerprint:
    dimensions: dict[str, Any]
    protected_spans: tuple[str, ...]
    normalized_tokens: tuple[str, ...]
    fingerprint_hash: str

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "dimensions": self.dimensions,
            "protected_spans": list(self.protected_spans),
            "normalized_tokens": list(self.normalized_tokens),
            "fingerprint_hash": self.fingerprint_hash,
        }

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> "PromptFingerprint":
        return cls(
            dimensions=dict(data.get("dimensions") or {}),
            protected_spans=tuple(str(item) for item in data.get("protected_spans") or ()),
            normalized_tokens=tuple(
                str(item) for item in data.get("normalized_tokens") or ()
            ),
            fingerprint_hash=str(data.get("fingerprint_hash", "")),
        )


@dataclasses.dataclass(frozen=True)
class MutationContract:
    allowed_factors: tuple[str, ...]
    protected_factors: tuple[str, ...]
    protected_spans: tuple[str, ...]
    max_factor_changes: int = 1
    min_token_similarity: float = 0.8
    archive_eligible: bool = True

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "allowed_factors": list(self.allowed_factors),
            "protected_factors": list(self.protected_factors),
            "protected_spans": list(self.protected_spans),
            "max_factor_changes": self.max_factor_changes,
            "min_token_similarity": self.min_token_similarity,
            "archive_eligible": self.archive_eligible,
        }

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> "MutationContract":
        raw_max_changes = data.get("max_factor_changes", 1)
        return cls(
            allowed_factors=tuple(str(item) for item in data.get("allowed_factors") or ()),
            protected_factors=tuple(
                str(item) for item in data.get("protected_factors") or ()
            ),
            protected_spans=tuple(str(item) for item in data.get("protected_spans") or ()),
            max_factor_changes=int(
                1 if raw_max_changes is None else raw_max_changes
            ),
            min_token_similarity=float(data.get("min_token_similarity", 0.8) or 0),
            archive_eligible=bool(data.get("archive_eligible", True)),
        )


@dataclasses.dataclass(frozen=True)
class MutationTask:
    task_id: str
    parent_candidate_id: str
    mode: str
    operator_id: str
    factor: str
    contract: MutationContract
    rationale: str
    plan: PerturbationPlan
    evaluation_role: str = "search"

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "parent_candidate_id": self.parent_candidate_id,
            "mode": self.mode,
            "operator_id": self.operator_id,
            "factor": self.factor,
            "contract": self.contract.to_jsonable(),
            "rationale": self.rationale,
            "plan": self.plan.to_jsonable(),
            "evaluation_role": self.evaluation_role,
        }

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> "MutationTask":
        return cls(
            task_id=str(data.get("task_id", "")),
            parent_candidate_id=str(data.get("parent_candidate_id", "")),
            mode=str(data.get("mode", "explore")),
            operator_id=str(data.get("operator_id", "legacy_planned_rewrite")),
            factor=str(data.get("factor", "legacy_unspecified")),
            contract=MutationContract.from_jsonable(dict(data.get("contract") or {})),
            rationale=str(data.get("rationale", "")),
            plan=PerturbationPlan.from_jsonable(dict(data.get("plan") or {})),
            evaluation_role=str(data.get("evaluation_role", "search")),
        )


@dataclasses.dataclass(frozen=True)
class ComparisonResult:
    comparable: bool
    classification: str
    total_delta: float | None
    component_deltas: dict[str, float]
    retention_constraints_met: bool
    reasons: tuple[str, ...] = ()

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "comparable": self.comparable,
            "classification": self.classification,
            "total_delta": self.total_delta,
            "component_deltas": self.component_deltas,
            "retention_constraints_met": self.retention_constraints_met,
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> "ComparisonResult":
        return cls(
            comparable=bool(data.get("comparable")),
            classification=str(data.get("classification", "unscored")),
            total_delta=(
                float(data["total_delta"])
                if data.get("total_delta") is not None
                else None
            ),
            component_deltas={
                str(key): float(value)
                for key, value in (data.get("component_deltas") or {}).items()
            },
            retention_constraints_met=bool(
                data.get("retention_constraints_met")
            ),
            reasons=tuple(str(item) for item in data.get("reasons") or ()),
        )


@dataclasses.dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    reasons: list[str]
    rubric: dict[str, float]

    def to_jsonable(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
