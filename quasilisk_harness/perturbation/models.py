"""Typed records for planned perturbation generation."""

from __future__ import annotations

import dataclasses
import enum
from typing import Any


class TaskPurpose(str, enum.Enum):
    """Why a mutation task exists and therefore how its outcome is measured."""

    CONTROL = "control"
    DISCOVERY = "discovery"
    OPTIMIZATION = "optimization"
    DIAGNOSTIC = "diagnostic"


class VariantStage(str, enum.Enum):
    """Furthest successfully completed stage in a generated variant's lifecycle."""

    PRODUCED = "produced"
    VALIDATED = "validated"
    SCHEDULED = "scheduled"
    EVALUATED = "evaluated"


class VariantOutcome(str, enum.Enum):
    """Terminal outcome assigned by the canonical outcome policy."""

    IMPROVED = "improved"
    RETAINED = "retained"
    REGRESSED = "regressed"
    INVALID = "invalid"
    UNSCORED = "unscored"
    TRUNCATED = "truncated"


class VariantSource(str, enum.Enum):
    """Origin of a variant, kept orthogonal to its task purpose."""

    PERTURBATOR = "perturbator"
    FALLBACK = "fallback"
    HARNESS = "harness"


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
    surface_text: str = ""

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "dimensions": self.dimensions,
            "protected_spans": list(self.protected_spans),
            "normalized_tokens": list(self.normalized_tokens),
            "fingerprint_hash": self.fingerprint_hash,
            "surface_text": self.surface_text,
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
            surface_text=str(data.get("surface_text", "")),
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
    purpose: TaskPurpose = TaskPurpose.OPTIMIZATION
    parent_relative: bool = True
    clean_ablation_candidate: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.purpose, TaskPurpose):
            raise TypeError("MutationTask purpose must be a TaskPurpose")

    @property
    def is_control(self) -> bool:
        return self.purpose is TaskPurpose.CONTROL

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
            "purpose": self.purpose.value,
            "parent_relative": self.parent_relative,
            "clean_ablation_candidate": self.clean_ablation_candidate,
        }

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> "MutationTask":
        raw_operator_id = str(data.get("operator_id") or "historical_unknown_task")
        operator_id = (
            "historical_legacy_rewrite"
            if raw_operator_id == "legacy_planned_rewrite"
            else raw_operator_id
        )
        historical = operator_id.startswith("historical_")
        broad_exploration = operator_id in {
            "coverage_exploration",
            "low_signal_exploration",
            "historical_legacy_rewrite",
            "historical_unknown_task",
        }
        parent_relative = bool(
            data.get("parent_relative", not broad_exploration)
        )
        clean_ablation = bool(
            data.get(
                "clean_ablation_candidate",
                parent_relative and not broad_exploration,
            )
        )
        purpose = _task_purpose_from_artifact(
            data,
            broad_exploration=broad_exploration,
            clean_ablation=clean_ablation,
        )
        return cls(
            task_id=str(data.get("task_id", "")),
            parent_candidate_id=str(data.get("parent_candidate_id", "")),
            mode=str(data.get("mode", "historical" if historical else "explore")),
            operator_id=operator_id,
            factor=str(
                data.get(
                    "factor",
                    "historical_unspecified" if historical else "unspecified",
                )
            ),
            contract=MutationContract.from_jsonable(dict(data.get("contract") or {})),
            rationale=str(data.get("rationale", "")),
            plan=PerturbationPlan.from_jsonable(dict(data.get("plan") or {})),
            purpose=purpose,
            parent_relative=parent_relative,
            clean_ablation_candidate=clean_ablation,
        )


def _task_purpose_from_artifact(
    data: dict[str, Any],
    *,
    broad_exploration: bool,
    clean_ablation: bool,
) -> TaskPurpose:
    """Translate canonical and pre-purpose task artifacts at the input boundary."""

    if data.get("purpose") is not None:
        try:
            return TaskPurpose(str(data["purpose"]))
        except ValueError as exc:
            raise ValueError(f"Unknown mutation task purpose: {data['purpose']}") from exc
    if str(data.get("evaluation_role") or "search") == "control":
        return TaskPurpose.CONTROL
    if broad_exploration:
        return TaskPurpose.DISCOVERY
    if bool(data.get("counts_toward_improvement_goal", clean_ablation)):
        return TaskPurpose.OPTIMIZATION
    return TaskPurpose.DIAGNOSTIC


@dataclasses.dataclass(frozen=True)
class VariantLifecycle:
    """Canonical, serializable record for one produced variant."""

    variant_id: str
    task_id: str
    purpose: TaskPurpose
    operator_id: str
    source: VariantSource
    source_output_id: str = ""
    stage: VariantStage = VariantStage.PRODUCED
    outcome: VariantOutcome | None = None
    validation_tier: str = "unknown"
    refinement_disposition: str = "unknown"
    archive_retained: bool = False
    score: float | None = None
    score_delta: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.purpose, TaskPurpose):
            raise TypeError("Variant lifecycle purpose must be a TaskPurpose")
        if not isinstance(self.source, VariantSource):
            raise TypeError("Variant lifecycle source must be a VariantSource")
        if not isinstance(self.stage, VariantStage):
            raise TypeError("Variant lifecycle stage must be a VariantStage")
        if self.outcome is not None and not isinstance(self.outcome, VariantOutcome):
            raise TypeError("Variant lifecycle outcome must be a VariantOutcome")
        if self.refinement_disposition not in {
            "unknown",
            "selected",
            "ranked_out",
        }:
            raise ValueError(
                "Unknown lifecycle refinement disposition: "
                f"{self.refinement_disposition}"
            )
        allowed_stages = {
            VariantOutcome.INVALID: {VariantStage.PRODUCED},
            VariantOutcome.TRUNCATED: {VariantStage.VALIDATED},
            VariantOutcome.UNSCORED: {
                VariantStage.SCHEDULED,
                VariantStage.EVALUATED,
            },
            VariantOutcome.IMPROVED: {VariantStage.EVALUATED},
            VariantOutcome.RETAINED: {VariantStage.EVALUATED},
            VariantOutcome.REGRESSED: {VariantStage.EVALUATED},
        }
        if self.outcome is not None and self.stage not in allowed_stages[self.outcome]:
            raise ValueError(
                f"Outcome {self.outcome.value} is invalid at stage {self.stage.value}"
            )
        if self.archive_retained and self.stage is not VariantStage.EVALUATED:
            raise ValueError("Only evaluated variants can be retained in the archive")
        if (
            self.score is not None or self.score_delta is not None
        ) and self.stage is not VariantStage.EVALUATED:
            raise ValueError("Only evaluated variants can carry score evidence")

    def advance(
        self,
        stage: VariantStage,
        *,
        outcome: VariantOutcome | None = None,
        archive_retained: bool | None = None,
        score: float | None = None,
        score_delta: float | None = None,
        validation_tier: str | None = None,
    ) -> "VariantLifecycle":
        order = tuple(VariantStage)
        if order.index(stage) < order.index(self.stage):
            raise ValueError(
                f"Variant lifecycle cannot regress from {self.stage} to {stage}"
            )
        return dataclasses.replace(
            self,
            stage=stage,
            outcome=outcome,
            archive_retained=(
                self.archive_retained
                if archive_retained is None
                else archive_retained
            ),
            score=self.score if score is None else score,
            score_delta=self.score_delta if score_delta is None else score_delta,
            validation_tier=(
                self.validation_tier
                if validation_tier is None
                else validation_tier
            ),
        )

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "task_id": self.task_id,
            "purpose": self.purpose.value,
            "operator_id": self.operator_id,
            "source": self.source.value,
            "source_output_id": self.source_output_id,
            "stage": self.stage.value,
            "outcome": self.outcome.value if self.outcome is not None else None,
            "validation_tier": self.validation_tier,
            "refinement_disposition": self.refinement_disposition,
            "archive_retained": self.archive_retained,
            "score": self.score,
            "score_delta": self.score_delta,
        }

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> "VariantLifecycle":
        for required in ("purpose", "source", "stage"):
            if data.get(required) is None:
                raise ValueError(f"Variant lifecycle is missing required field: {required}")
        raw_outcome = data.get("outcome")
        return cls(
            variant_id=str(data.get("variant_id", "")),
            task_id=str(data.get("task_id", "")),
            purpose=TaskPurpose(str(data["purpose"])),
            operator_id=str(data.get("operator_id") or "unspecified_task"),
            source=VariantSource(str(data["source"])),
            source_output_id=str(data.get("source_output_id") or ""),
            stage=VariantStage(str(data["stage"])),
            outcome=(VariantOutcome(str(raw_outcome)) if raw_outcome else None),
            validation_tier=str(data.get("validation_tier") or "unknown"),
            refinement_disposition=str(
                data.get("refinement_disposition") or "unknown"
            ),
            archive_retained=bool(data.get("archive_retained")),
            score=(float(data["score"]) if data.get("score") is not None else None),
            score_delta=(
                float(data["score_delta"])
                if data.get("score_delta") is not None
                else None
            ),
        )


@dataclasses.dataclass(frozen=True)
class ComparisonResult:
    comparable: bool
    classification: str
    total_delta: float | None
    component_deltas: dict[str, float]
    retention_constraints_met: bool
    reasons: tuple[str, ...] = ()
    evidence_basis: dict[str, str] = dataclasses.field(default_factory=dict)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "comparable": self.comparable,
            "classification": self.classification,
            "total_delta": self.total_delta,
            "component_deltas": self.component_deltas,
            "retention_constraints_met": self.retention_constraints_met,
            "reasons": list(self.reasons),
            "evidence_basis": self.evidence_basis,
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
            evidence_basis={
                str(key): str(value)
                for key, value in (data.get("evidence_basis") or {}).items()
            },
        )


@dataclasses.dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    reasons: list[str]
    rubric: dict[str, float]
    tier: str = "accepted"
    warnings: list[str] = dataclasses.field(default_factory=list)
    diagnostics: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_jsonable(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
