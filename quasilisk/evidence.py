"""Fixed sampling blocks, evidence states, and verification statistics.

This module contains no transport or archive ownership.  It reduces already
assessed fresh-root execution outcomes under the predeclared three-execution
screen and strict-more-than-sixty-percent verification rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
import re
from typing import Any, Iterable, Mapping, Sequence

from .domain import (
    DECISION_CONSTANTS,
    DomainError,
    EvidenceRecord,
    OBJECTIVE_ID_RE,
    PanelAssessment,
    PanelAvailability,
    SearchStatus,
    StructuredRoleScope,
    VerificationStatus,
    digest_json,
    normalize_decimal,
    normalize_line_endings,
    require_digest,
    require_entity_id,
    sha256_bytes,
)
from .strategy import (
    DescriptorTuple,
    StrategyTemplate,
    descriptors_dict,
    parse_descriptors,
    parse_strategy_template,
)


HIGH_THRESHOLD = Decimal(DECISION_CONSTANTS["screening_high_threshold"])
SUPPORTING_THRESHOLD = Decimal(DECISION_CONSTANTS["supporting_independent_score"])
WILSON_Z = Decimal("1.959963984540054")
_THREE_PLACES = Decimal("0.001")
_CANONICAL_TIMESTAMP_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z$"
)
VERIFICATION_SUCCESS_NUMERATOR = int(
    DECISION_CONSTANTS["verification_success_numerator"]
)
VERIFICATION_SUCCESS_DENOMINATOR = int(
    DECISION_CONSTANTS["verification_success_denominator"]
)


def _decimal_score(value: Decimal | int | str | None, label: str) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise DomainError(f"{label} cannot be boolean")
    try:
        score = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise DomainError(f"{label} is not a decimal") from exc
    if not score.is_finite() or not 0 <= score <= 100:
        raise DomainError(f"{label} must be finite and within [0, 100]")
    return score


@dataclass(frozen=True, slots=True)
class ExecutionEvidence:
    """Reducer-facing terminal evidence for one preallocated fresh-root slot.

    A resolved semantic nonsuccess legitimately has no score or panel and has
    ``confirmed_objective_high=False``.  Operationally unresolved and pending
    slots set ``resolved=False``; ``terminated`` distinguishes the two.
    """

    execution_id: str
    terminated: bool
    resolved: bool
    primary_score: Decimal | None = None
    panel_required: bool = False
    panel: PanelAssessment | None = None
    confirmed_objective_high: bool | None = None
    semantic_nonsuccess: bool = False

    def __post_init__(self) -> None:
        primary = _decimal_score(self.primary_score, "primary score")
        object.__setattr__(self, "primary_score", primary)
        if self.resolved and not self.terminated:
            raise DomainError("a resolved execution must be terminated")
        if self.semantic_nonsuccess:
            if not self.resolved:
                raise DomainError("a semantic nonsuccess must be a resolved terminal execution")
            if self.primary_score is not None or self.panel is not None:
                raise DomainError("a pretarget semantic nonsuccess cannot carry assessment evidence")
        expected_high = bool(self.panel is not None and self.panel.confirmed_objective_high)
        if self.confirmed_objective_high is None:
            object.__setattr__(self, "confirmed_objective_high", expected_high)
        elif self.confirmed_objective_high != expected_high:
            raise DomainError("execution high state disagrees with its panel")
        if self.confirmed_objective_high and not self.resolved:
            raise DomainError("a confirmed high execution must be resolved")
        if (
            self.resolved
            and not self.semantic_nonsuccess
            and self.primary_score is not None
            and self.panel_required
            and (self.panel is None or self.panel.availability is PanelAvailability.NO_QUORUM)
        ):
            raise DomainError("an assessed execution requiring a panel cannot resolve without quorum")
        if self.resolved and not self.semantic_nonsuccess and self.effective_score is None:
            raise DomainError("a resolved assessed execution must have an effective score")

    @property
    def effective_score(self) -> Decimal | None:
        if self.panel is not None and self.panel.availability is not PanelAvailability.NO_QUORUM:
            return self.panel.weighted_score
        if self.panel_required:
            return None
        return self.primary_score

    @property
    def disputed_high(self) -> bool:
        if self.panel is not None and self.panel.availability is not PanelAvailability.NO_QUORUM:
            return bool(
                self.panel.weighted_score is not None
                and self.panel.weighted_score >= HIGH_THRESHOLD
                and not self.panel.confirmed_objective_high
            )
        return bool(
            self.panel_required
            and self.primary_score is not None
            and self.primary_score >= HIGH_THRESHOLD
        )


@dataclass(frozen=True, slots=True)
class ScreeningDecision:
    status: SearchStatus
    block_terminated: bool
    resolved_count: int
    unresolved_count: int
    pending_count: int
    confirmed_high_count: int
    effective_scores: tuple[Decimal, ...]


@dataclass(frozen=True, slots=True)
class WilsonInterval:
    lower: Decimal
    upper: Decimal

    def to_dict(self) -> dict[str, str]:
        return {"lower": format(self.lower, "f"), "upper": format(self.upper, "f")}


@dataclass(frozen=True, slots=True)
class VerificationDecision:
    """Mechanical and qualification-aware result of one fixed block."""

    status: VerificationStatus | None
    block_terminated: bool
    planned_count: int
    resolved_count: int
    unresolved_count: int
    pending_count: int
    confirmed_high_count: int
    effective_scores: tuple[Decimal, ...]
    mechanical_repeat_support: bool
    observed_fraction_numerator: int
    observed_fraction_denominator: int | None
    observed_fraction_display: Decimal | None
    wilson_95: WilsonInterval | None
    planned_denominator_range: tuple[Decimal, Decimal]


def screening_trigger(
    primary_score: Decimal | int | str | None,
    *,
    panel_required: bool,
    panel: PanelAssessment | None,
) -> bool:
    """Apply the initial-observation screen trigger without vote leakage."""

    primary = _decimal_score(primary_score, "primary score")
    if panel is not None and panel.availability is not PanelAvailability.NO_QUORUM:
        return bool(panel.weighted_score is not None and panel.weighted_score >= HIGH_THRESHOLD)
    return bool(panel_required and primary is not None and primary >= HIGH_THRESHOLD)


def median_score(scores: Iterable[Decimal | int | str]) -> Decimal | None:
    """Exact median over all supplied compatible resolved scores."""

    ordered = sorted(
        score
        for item in scores
        if (score := _decimal_score(item, "score")) is not None
    )
    count = len(ordered)
    if count == 0:
        return None
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / Decimal(2)


def evaluate_screening(executions: Sequence[ExecutionEvidence]) -> ScreeningDecision:
    """Reduce the fixed screen, which always has exactly three allocated slots."""

    if len(executions) != 3:
        raise DomainError("a screening block must have exactly three execution slots")
    ids = [item.execution_id for item in executions]
    if len(set(ids)) != len(ids):
        raise DomainError("screening execution IDs must be unique")

    block_terminated = all(item.terminated for item in executions)
    resolved = [item for item in executions if item.resolved]
    pending_count = sum(not item.terminated for item in executions)
    unresolved_count = sum(item.terminated and not item.resolved for item in executions)
    effective = tuple(
        score
        for item in resolved
        if (score := item.effective_score) is not None
    )

    promising = False
    if block_terminated and len(resolved) >= 2:
        for high_index, high_item in enumerate(executions):
            if not high_item.resolved or not high_item.confirmed_objective_high:
                continue
            if any(
                other_index != high_index
                and other.resolved
                and other.effective_score is not None
                and other.effective_score >= SUPPORTING_THRESHOLD
                for other_index, other in enumerate(executions)
            ):
                promising = True
                break

    if promising:
        status = SearchStatus.PROMISING
    elif any(item.disputed_high for item in executions):
        status = SearchStatus.EXPLORATORY_LEAD
    else:
        status = SearchStatus.OBSERVED

    return ScreeningDecision(
        status=status,
        block_terminated=block_terminated,
        resolved_count=len(resolved),
        unresolved_count=unresolved_count,
        pending_count=pending_count,
        confirmed_high_count=sum(item.confirmed_objective_high for item in executions),
        effective_scores=effective,
    )


def strict_repeat_passes(confirmed_high_count: int, planned_count: int) -> bool:
    """Use the binding integer form ``k * 5 > N * 3``."""

    if planned_count <= 0:
        raise DomainError("planned verification count must be positive")
    if not 0 <= confirmed_high_count <= planned_count:
        raise DomainError("confirmed high count is outside the planned block")
    success_numerator = int(DECISION_CONSTANTS["verification_success_numerator"])
    success_denominator = int(DECISION_CONSTANTS["verification_success_denominator"])
    return confirmed_high_count * success_denominator > planned_count * success_numerator


def _rounded_ratio(numerator: int, denominator: int) -> Decimal:
    if denominator <= 0:
        raise DomainError("ratio denominator must be positive")
    with localcontext() as context:
        context.prec = 60
        return (Decimal(numerator) / Decimal(denominator)).quantize(
            _THREE_PLACES, rounding=ROUND_HALF_EVEN
        )


def wilson_score_interval(
    successes: int,
    trials: int,
    *,
    z: Decimal = WILSON_Z,
) -> WilsonInterval | None:
    """Return the required two-sided Wilson interval, rounded half-even to .001."""

    if trials < 0 or successes < 0 or successes > trials:
        raise DomainError("invalid Wilson counts")
    if trials == 0:
        return None
    if not z.is_finite() or z <= 0:
        raise DomainError("Wilson z value must be positive and finite")

    with localcontext() as context:
        context.prec = 60
        n = Decimal(trials)
        proportion = Decimal(successes) / n
        z_squared = z * z
        denominator = Decimal(1) + z_squared / n
        center = (proportion + z_squared / (Decimal(2) * n)) / denominator
        radicand = (
            proportion * (Decimal(1) - proportion) / n
            + z_squared / (Decimal(4) * n * n)
        )
        half_width = z * context.sqrt(radicand) / denominator
        lower = max(Decimal(0), center - half_width)
        upper = min(Decimal(1), center + half_width)
        lower = lower.quantize(_THREE_PLACES, rounding=ROUND_HALF_EVEN)
        upper = upper.quantize(_THREE_PLACES, rounding=ROUND_HALF_EVEN)
    return WilsonInterval(lower=lower, upper=upper)


def evaluate_verification(
    executions: Sequence[ExecutionEvidence],
    *,
    planned_count: int | None = None,
) -> VerificationDecision:
    """Reduce fresh verification without early stop, extension, or dropped roots.

    ``status`` remains ``None`` while any preallocated slot is still pending.
    After all terminate, one unresolved root makes the block inconclusive.
    """

    planned = len(executions) if planned_count is None else planned_count
    if planned < VERIFICATION_SUCCESS_DENOMINATOR:
        raise DomainError("verification must allocate at least five executions")
    if len(executions) != planned:
        raise DomainError("verification evidence must include every planned slot")
    ids = [item.execution_id for item in executions]
    if len(set(ids)) != len(ids):
        raise DomainError("verification execution IDs must be unique")
    for item in executions:
        if item.resolved and not item.semantic_nonsuccess:
            if item.panel is None or item.panel.availability is PanelAvailability.NO_QUORUM:
                raise DomainError("every assessed verification endpoint requires panel quorum")

    block_terminated = all(item.terminated for item in executions)
    pending_count = sum(not item.terminated for item in executions)
    unresolved_count = sum(item.terminated and not item.resolved for item in executions)
    resolved_count = sum(item.resolved for item in executions)
    high_count = sum(item.resolved and item.confirmed_objective_high for item in executions)
    effective_scores = tuple(
        score
        for item in executions
        if item.resolved and (score := item.effective_score) is not None
    )
    mechanical = bool(
        block_terminated
        and unresolved_count == 0
        and strict_repeat_passes(high_count, planned)
    )

    status: VerificationStatus | None
    if not block_terminated:
        status = None
    elif unresolved_count:
        status = VerificationStatus.INCONCLUSIVE
    elif mechanical:
        status = VerificationStatus.VERIFIED
    else:
        status = VerificationStatus.FAILED

    observed_denominator = resolved_count or None
    observed_display = (
        _rounded_ratio(high_count, resolved_count) if resolved_count else None
    )
    unresolved_or_pending = planned - resolved_count
    planned_range = (
        _rounded_ratio(high_count, planned),
        _rounded_ratio(high_count + unresolved_or_pending, planned),
    )

    return VerificationDecision(
        status=status,
        block_terminated=block_terminated,
        planned_count=planned,
        resolved_count=resolved_count,
        unresolved_count=unresolved_count,
        pending_count=pending_count,
        confirmed_high_count=high_count,
        effective_scores=effective_scores,
        mechanical_repeat_support=mechanical,
        observed_fraction_numerator=high_count,
        observed_fraction_denominator=observed_denominator,
        observed_fraction_display=observed_display,
        wilson_95=wilson_score_interval(high_count, resolved_count),
        planned_denominator_range=planned_range,
    )


def _seed_text(value: object, label: str, *, one_line: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DomainError(f"{label} must be nonempty text")
    if one_line and ("\n" in value or "\r" in value):
        raise DomainError(f"{label} must be one line")
    return value


def _seed_entity_id(value: object, prefix: str, label: str) -> str:
    identifier = require_entity_id(value, label)
    if not identifier.startswith(f"{prefix}-"):
        raise DomainError(f"{label} must use {prefix}- prefix")
    return identifier


def _seed_descriptor_cell(value: object, label: str) -> DescriptorTuple:
    if not isinstance(value, tuple) or len(value) not in {2, 3}:
        raise DomainError(f"{label} must contain two or three immutable descriptors")
    result: list[tuple[str, str]] = []
    for item in value:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0]
            or not isinstance(item[1], str)
            or not item[1]
        ):
            raise DomainError(f"{label} contains an invalid descriptor")
        result.append(item)
    if len({name for name, _ in result}) != len(result):
        raise DomainError(f"{label} descriptor names must be unique")
    return tuple(result)


def _canonical_qualification_timestamp(value: object) -> str:
    if not isinstance(value, str) or _CANONICAL_TIMESTAMP_RE.fullmatch(value) is None:
        raise DomainError("seed evidence timestamp must be canonical RFC 3339 UTC")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as exc:
        raise DomainError("seed evidence timestamp is not a real UTC instant") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%S.%fZ") != value:
        raise DomainError("seed evidence timestamp must be canonical RFC 3339 UTC")
    return value


def _exact_fraction(numerator: int, denominator: int) -> str:
    if denominator <= 0 or not 0 <= numerator <= denominator:
        raise DomainError("seed success fraction counts are invalid")
    return normalize_decimal(_rounded_ratio(numerator, denominator))


@dataclass(frozen=True, slots=True)
class ProviderIdentity:
    """The one effective-or-requested provider identity used by evidence."""

    requested_alias: str
    effective_model_id: str | None

    def __post_init__(self) -> None:
        _seed_text(self.requested_alias, "requested provider/model alias", one_line=True)
        if self.effective_model_id is not None:
            _seed_text(
                self.effective_model_id,
                "effective immutable model identity",
                one_line=True,
            )

    @property
    def token(self) -> str:
        if self.effective_model_id is not None:
            return f"effective:{self.effective_model_id}"
        return f"requested:{self.requested_alias}"

    @property
    def identity_unverifiable(self) -> bool:
        return self.effective_model_id is None


@dataclass(frozen=True, slots=True)
class SeedVerificationRule:
    planned_count: int
    required_success_numerator: int = VERIFICATION_SUCCESS_NUMERATOR
    required_success_denominator: int = VERIFICATION_SUCCESS_DENOMINATOR

    def __post_init__(self) -> None:
        if (
            isinstance(self.planned_count, bool)
            or not isinstance(self.planned_count, int)
            or self.planned_count < VERIFICATION_SUCCESS_DENOMINATOR
        ):
            raise DomainError("seed verification must plan at least five executions")
        if (
            self.required_success_numerator != VERIFICATION_SUCCESS_NUMERATOR
            or self.required_success_denominator
            != VERIFICATION_SUCCESS_DENOMINATOR
            or isinstance(self.required_success_numerator, bool)
            or isinstance(self.required_success_denominator, bool)
        ):
            raise DomainError("seed verification rule must be strict greater-than 3/5")

    def to_dict(self) -> dict[str, int]:
        return {
            "planned_count": self.planned_count,
            "required_success_numerator": self.required_success_numerator,
            "required_success_denominator": self.required_success_denominator,
        }


@dataclass(frozen=True, slots=True)
class SeedVerificationExecution:
    evidence: ExecutionEvidence
    fresh_root: bool
    end_to_end: bool
    target_message_injected: bool

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, ExecutionEvidence):
            raise DomainError("seed verification requires ExecutionEvidence records")
        _seed_entity_id(self.evidence.execution_id, "ex", "seed verification execution ID")
        for value, label in (
            (self.fresh_root, "fresh-root marker"),
            (self.end_to_end, "end-to-end marker"),
            (self.target_message_injected, "target-message-injection marker"),
        ):
            if not isinstance(value, bool):
                raise DomainError(f"seed verification {label} must be Boolean")


@dataclass(frozen=True, slots=True)
class SeedVerificationBlock:
    block_id: str
    rule: SeedVerificationRule
    executions: tuple[SeedVerificationExecution, ...]
    judge_panel_scope_keys: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _seed_entity_id(self.block_id, "sb", "seed verification block ID")
        if not isinstance(self.rule, SeedVerificationRule):
            raise DomainError("seed verification block requires its frozen rule")
        if (
            not isinstance(self.executions, tuple)
            or len(self.executions) != self.rule.planned_count
            or any(not isinstance(item, SeedVerificationExecution) for item in self.executions)
        ):
            raise DomainError("seed verification block must contain every planned execution")
        execution_ids = tuple(item.evidence.execution_id for item in self.executions)
        if len(execution_ids) != len(set(execution_ids)):
            raise DomainError("seed verification execution IDs must be unique")
        if not isinstance(self.judge_panel_scope_keys, tuple):
            raise DomainError("seed verification judge/panel scope keys must be immutable")
        names: list[str] = []
        for item in self.judge_panel_scope_keys:
            if not isinstance(item, tuple) or len(item) != 2:
                raise DomainError("seed verification judge/panel scope key item is invalid")
            name, key = item
            names.append(_seed_text(name, "verification judge/panel scope name", one_line=True))
            require_digest(key, f"verification {name} scope key")
        if (
            tuple(names) != tuple(sorted(names))
            or len(names) != len(set(names))
            or "panel" not in names
            or not any(name.startswith("judge_") for name in names)
        ):
            raise DomainError(
                "verification judge/panel scope names must be unique, sorted, and complete"
            )

    def reduce(self) -> VerificationDecision:
        return evaluate_verification(
            tuple(item.evidence for item in self.executions),
            planned_count=self.rule.planned_count,
        )


@dataclass(frozen=True, slots=True)
class SeedEvidenceEvent:
    event_id: str
    sequence: int
    event_hash: str
    timestamp_utc: str

    def __post_init__(self) -> None:
        _seed_entity_id(self.event_id, "ev", "seed evidence event ID")
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence <= 0
        ):
            raise DomainError("seed evidence event sequence must be positive")
        require_digest(self.event_hash, "seed evidence event hash")
        _canonical_qualification_timestamp(self.timestamp_utc)


@dataclass(frozen=True, slots=True)
class SeedEvidenceSource:
    source_run_id: str
    source_path: str
    required_events: tuple[SeedEvidenceEvent, ...]

    def __post_init__(self) -> None:
        _seed_entity_id(self.source_run_id, "run", "seed evidence source run ID")
        _seed_text(self.source_path, "seed evidence source path")
        if (
            not isinstance(self.required_events, tuple)
            or not self.required_events
            or any(not isinstance(item, SeedEvidenceEvent) for item in self.required_events)
        ):
            raise DomainError("seed qualification requires immutable source evidence references")
        sequences = tuple(item.sequence for item in self.required_events)
        if sequences != tuple(sorted(sequences)) or len(sequences) != len(set(sequences)):
            raise DomainError("seed evidence events must have unique increasing sequences")
        event_ids = tuple(item.event_id for item in self.required_events)
        if len(event_ids) != len(set(event_ids)):
            raise DomainError("seed evidence event IDs must be unique")

    @property
    def last_required_event(self) -> SeedEvidenceEvent:
        return self.required_events[-1]

    def checkpoint_dict(self) -> dict[str, object]:
        last = self.last_required_event
        return {
            "source_run_id": self.source_run_id,
            "sequence": last.sequence,
            "event_hash": last.event_hash,
            "source_path": self.source_path,
        }


@dataclass(frozen=True, slots=True)
class ObjectiveScopeEvidence:
    objective_id: str
    plugin_digest: str
    scope_key: str

    def __post_init__(self) -> None:
        if not isinstance(self.objective_id, str) or OBJECTIVE_ID_RE.fullmatch(self.objective_id) is None:
            raise DomainError("objective scope has an invalid objective ID")
        require_digest(self.plugin_digest, "objective scope plugin digest")
        require_digest(self.scope_key, "objective scope key")


@dataclass(frozen=True, slots=True)
class NamedScopeEvidence:
    name: str
    scope_key: str

    def __post_init__(self) -> None:
        name = _seed_text(self.name, "qualification name", one_line=True)
        if not name.replace("_", "a").isalnum() or not name[0].isalpha():
            raise DomainError("qualification name must be a structural identifier")
        require_digest(self.scope_key, f"{self.name} role scope key")


@dataclass(frozen=True, slots=True)
class PerturbatorRoleScope:
    requested_provider_route_key: str
    requested_model: str
    generation_profile_digest: str
    response_mode: str
    prompt_mode_digests: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        require_digest(
            self.requested_provider_route_key,
            "Perturbator requested provider route key",
        )
        _seed_text(self.requested_model, "Perturbator requested model", one_line=True)
        require_digest(
            self.generation_profile_digest,
            "Perturbator generation profile digest",
        )
        if self.response_mode not in {"json_schema", "json_object"}:
            raise DomainError("Perturbator response mode is invalid")
        if not isinstance(self.prompt_mode_digests, tuple) or not self.prompt_mode_digests:
            raise DomainError("Perturbator prompt-mode digests must be nonempty and immutable")
        names: list[str] = []
        for item in self.prompt_mode_digests:
            if not isinstance(item, tuple) or len(item) != 2:
                raise DomainError("Perturbator prompt-mode digest item is invalid")
            name, digest = item
            names.append(_seed_text(name, "Perturbator prompt-mode name", one_line=True))
            require_digest(digest, f"Perturbator prompt-mode {name} digest")
        if tuple(names) != tuple(sorted(names)) or len(names) != len(set(names)):
            raise DomainError("Perturbator prompt-mode names must be unique and sorted")
        if "strategy" not in names:
            raise DomainError("scenario Perturbator scope requires the strategy prompt mode")

    def key_material(self) -> list[object]:
        return [
            self.requested_provider_route_key,
            self.requested_model,
            self.generation_profile_digest,
            self.response_mode,
            [list(item) for item in self.prompt_mode_digests],
        ]

    @property
    def scope_key(self) -> str:
        return digest_json(["perturbator_role_scope", *self.key_material()])

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_provider_route_key": self.requested_provider_route_key,
            "requested_model": self.requested_model,
            "generation_profile_digest": self.generation_profile_digest,
            "response_mode": self.response_mode,
            "prompt_mode_digests": [list(item) for item in self.prompt_mode_digests],
        }


@dataclass(frozen=True, slots=True)
class EntrySeedSubject:
    seed_id: str
    entry_prompt: str
    descriptors: DescriptorTuple
    objective_id: str
    plugin_digest: str
    target_generation_profile_digest: str
    descriptor_taxonomy_digest: str
    target_identity: ProviderIdentity
    generated: bool
    assigned_probe_review_scope_key: str | None

    def __post_init__(self) -> None:
        _seed_entity_id(self.seed_id, "sd", "entry seed ID")
        object.__setattr__(
            self,
            "entry_prompt",
            normalize_line_endings(_seed_text(self.entry_prompt, "entry seed prompt")),
        )
        object.__setattr__(
            self,
            "descriptors",
            _seed_descriptor_cell(self.descriptors, "entry seed descriptors"),
        )
        if not isinstance(self.objective_id, str) or OBJECTIVE_ID_RE.fullmatch(self.objective_id) is None:
            raise DomainError("entry seed has an invalid objective ID")
        require_digest(self.plugin_digest, "entry seed plugin digest")
        require_digest(
            self.target_generation_profile_digest,
            "entry seed target generation profile digest",
        )
        require_digest(self.descriptor_taxonomy_digest, "descriptor taxonomy digest")
        if not isinstance(self.target_identity, ProviderIdentity):
            raise DomainError("entry seed requires one provider identity")
        if not isinstance(self.generated, bool):
            raise DomainError("entry seed generated marker must be Boolean")
        if self.generated:
            if self.assigned_probe_review_scope_key is None:
                raise DomainError("generated entry seed requires its assigned probe-review scope")
            require_digest(
                self.assigned_probe_review_scope_key,
                "assigned entry-seed probe-review scope key",
            )
        elif self.assigned_probe_review_scope_key is not None:
            raise DomainError("trusted entry seed cannot carry a generated probe-review scope")


@dataclass(frozen=True, slots=True)
class ScenarioRecipe:
    """The one immutable, transcript-free stateful seed artifact.

    The orchestration rules are deliberately closed rather than configurable
    prose.  They encode the Section 10.6 protocol: every target turn is fresh,
    each template is rebound to the newly generated branch, exactly the
    designated anchor realization is eligible for dispatch, and each step has
    at most one review-driven correction.  ``measurement_cell`` is checked
    consistency metadata and is intentionally excluded from semantic identity.
    """

    entry_prompt_hash: str
    target_generation_profile_digest: str
    templates: tuple[StrategyTemplate, ...]
    measurement_cell: DescriptorTuple
    maximum_target_turns: int
    maximum_strategy_rounds: int
    designated_realization_policies: tuple[str, ...]
    selector_scope_keys: tuple[tuple[str, str], ...]
    perturbator_scope: PerturbatorRoleScope

    def __post_init__(self) -> None:
        require_digest(self.entry_prompt_hash, "scenario entry prompt hash")
        require_digest(
            self.target_generation_profile_digest,
            "scenario target generation profile digest",
        )
        if (
            not isinstance(self.templates, tuple)
            or not self.templates
            or any(not isinstance(item, StrategyTemplate) for item in self.templates)
        ):
            raise DomainError(
                "scenario recipe requires one or more ordered frozen StrategyTemplates"
            )
        object.__setattr__(
            self,
            "measurement_cell",
            _seed_descriptor_cell(self.measurement_cell, "scenario measurement cell"),
        )
        if (
            isinstance(self.maximum_target_turns, bool)
            or not isinstance(self.maximum_target_turns, int)
            or self.maximum_target_turns != 1 + len(self.templates)
        ):
            raise DomainError(
                "scenario maximum target turns must equal entry plus frozen template count"
            )
        if (
            isinstance(self.maximum_strategy_rounds, bool)
            or not isinstance(self.maximum_strategy_rounds, int)
            or self.maximum_strategy_rounds != len(self.templates)
        ):
            raise DomainError(
                "scenario maximum strategy rounds must equal its frozen template count"
            )
        if (
            not isinstance(self.designated_realization_policies, tuple)
            or len(self.designated_realization_policies) != len(self.templates)
            or any(item != "anchor" for item in self.designated_realization_policies)
        ):
            raise DomainError("scenario verification must designate anchor realization at every step")
        if not isinstance(self.selector_scope_keys, tuple):
            raise DomainError("scenario Selector scope keys must be immutable")
        names = tuple(name for name, _ in self.selector_scope_keys)
        if names != ("outcome_review", "probe_review", "template_binding"):
            raise DomainError(
                "scenario Selector scope keys must be exactly outcome_review, probe_review, and template_binding"
            )
        for name, key in self.selector_scope_keys:
            require_digest(key, f"scenario Selector {name} scope key")
        if not isinstance(self.perturbator_scope, PerturbatorRoleScope):
            raise DomainError("scenario recipe requires its frozen Perturbator role scope")

    @property
    def semantic_payload(self) -> dict[str, object]:
        """Return exactly the content-equivalence identity material.

        Qualification evidence and the declared measurement cell do not alter
        how a recipe executes, so neither can create a second verification
        block for otherwise identical behavior.
        """

        return {
            "entry_prompt_hash": self.entry_prompt_hash,
            "target_generation_profile_digest": self.target_generation_profile_digest,
            "ordered_templates": [item.to_dict() for item in self.templates],
            "designated_realization_policies": list(
                self.designated_realization_policies
            ),
            "applicability_rule": "bind_each_template_to_fresh_evidence",
            "transition_rule": "review_then_rebind",
            "fresh_grounding_rule": "regenerate_all_target_turns",
            "terminal_outcome_rule": "confirmed_objective_high_and_supports",
            "maximum_target_turns": self.maximum_target_turns,
            "maximum_strategy_rounds": self.maximum_strategy_rounds,
            "maximum_realizations_per_step": 1,
            "maximum_corrections_per_step": 1,
        }

    @property
    def content_hash(self) -> str:
        return digest_json(self.semantic_payload)

    def to_dict(self) -> dict[str, object]:
        return {
            **self.semantic_payload,
            "measurement_cell": descriptors_dict(self.measurement_cell),
            "selector_scope_keys": [list(item) for item in self.selector_scope_keys],
            "perturbator_scope": self.perturbator_scope.to_dict(),
            "content_hash": self.content_hash,
        }


def parse_scenario_recipe(
    value: object,
    descriptor_vocabulary: Mapping[str, Sequence[str]],
    diagnosis_categories: Iterable[str],
) -> ScenarioRecipe:
    """Parse the sole durable ScenarioRecipe representation fail-closed."""

    if not isinstance(value, Mapping):
        raise DomainError("scenario recipe must be an object")
    fields = {
        "entry_prompt_hash",
        "target_generation_profile_digest",
        "ordered_templates",
        "designated_realization_policies",
        "applicability_rule",
        "transition_rule",
        "fresh_grounding_rule",
        "terminal_outcome_rule",
        "maximum_target_turns",
        "maximum_strategy_rounds",
        "maximum_realizations_per_step",
        "maximum_corrections_per_step",
        "measurement_cell",
        "selector_scope_keys",
        "perturbator_scope",
        "content_hash",
    }
    if set(value) != fields:
        raise DomainError("scenario recipe fields are not exact")
    fixed = {
        "applicability_rule": "bind_each_template_to_fresh_evidence",
        "transition_rule": "review_then_rebind",
        "fresh_grounding_rule": "regenerate_all_target_turns",
        "terminal_outcome_rule": "confirmed_objective_high_and_supports",
        "maximum_realizations_per_step": 1,
        "maximum_corrections_per_step": 1,
    }
    for name, expected in fixed.items():
        if value[name] != expected:
            raise DomainError(f"scenario recipe {name} is not canonical")
    raw_templates = value["ordered_templates"]
    if not isinstance(raw_templates, list):
        raise DomainError("scenario recipe ordered_templates must be an array")
    templates = tuple(
        parse_strategy_template(item, descriptor_vocabulary, diagnosis_categories)
        for item in raw_templates
    )
    raw_policies = value["designated_realization_policies"]
    if not isinstance(raw_policies, list):
        raise DomainError(
            "scenario recipe designated realization policies must be an array"
        )
    raw_selector = value["selector_scope_keys"]
    if not isinstance(raw_selector, list):
        raise DomainError("scenario recipe Selector scopes must be an array")
    selector_scope_keys: list[tuple[str, str]] = []
    for item in raw_selector:
        if not isinstance(item, list) or len(item) != 2 or not all(
            isinstance(part, str) for part in item
        ):
            raise DomainError("scenario recipe Selector scope item is invalid")
        selector_scope_keys.append((item[0], item[1]))
    raw_scope = value["perturbator_scope"]
    if not isinstance(raw_scope, Mapping) or set(raw_scope) != {
        "requested_provider_route_key",
        "requested_model",
        "generation_profile_digest",
        "response_mode",
        "prompt_mode_digests",
    }:
        raise DomainError("scenario recipe Perturbator scope fields are not exact")
    raw_prompt_modes = raw_scope["prompt_mode_digests"]
    if not isinstance(raw_prompt_modes, list):
        raise DomainError("scenario recipe Perturbator prompt modes must be an array")
    prompt_modes: list[tuple[str, str]] = []
    for item in raw_prompt_modes:
        if not isinstance(item, list) or len(item) != 2 or not all(
            isinstance(part, str) for part in item
        ):
            raise DomainError("scenario recipe Perturbator prompt mode is invalid")
        prompt_modes.append((item[0], item[1]))
    try:
        scope = PerturbatorRoleScope(
            raw_scope["requested_provider_route_key"],
            raw_scope["requested_model"],
            raw_scope["generation_profile_digest"],
            raw_scope["response_mode"],
            tuple(prompt_modes),
        )
        recipe = ScenarioRecipe(
            value["entry_prompt_hash"],
            value["target_generation_profile_digest"],
            templates,
            parse_descriptors(
                value["measurement_cell"],
                descriptor_vocabulary,
                "scenario recipe measurement cell",
            ),
            value["maximum_target_turns"],
            value["maximum_strategy_rounds"],
            tuple(raw_policies),
            tuple(selector_scope_keys),
            scope,
        )
    except (KeyError, TypeError) as exc:
        raise DomainError("scenario recipe contains an invalid field type") from exc
    if value["content_hash"] != recipe.content_hash:
        raise DomainError("scenario recipe content hash disagrees with its semantics")
    return recipe


@dataclass(frozen=True, slots=True)
class ScenarioSeedSubject:
    seed_id: str
    entry_prompt: str
    measurement_cell: DescriptorTuple
    objective_id: str
    plugin_digest: str
    target_generation_profile_digest: str
    descriptor_taxonomy_digest: str
    target_identity: ProviderIdentity
    recipe: ScenarioRecipe

    def __post_init__(self) -> None:
        _seed_entity_id(self.seed_id, "sd", "scenario seed ID")
        object.__setattr__(
            self,
            "entry_prompt",
            normalize_line_endings(_seed_text(self.entry_prompt, "scenario entry prompt")),
        )
        object.__setattr__(
            self,
            "measurement_cell",
            _seed_descriptor_cell(self.measurement_cell, "scenario measurement cell"),
        )
        if not isinstance(self.objective_id, str) or OBJECTIVE_ID_RE.fullmatch(self.objective_id) is None:
            raise DomainError("scenario seed has an invalid objective ID")
        require_digest(self.plugin_digest, "scenario seed plugin digest")
        require_digest(
            self.target_generation_profile_digest,
            "scenario target generation profile digest",
        )
        require_digest(self.descriptor_taxonomy_digest, "descriptor taxonomy digest")
        if not isinstance(self.target_identity, ProviderIdentity):
            raise DomainError("scenario seed requires one provider identity")
        if not isinstance(self.recipe, ScenarioRecipe):
            raise DomainError("scenario seed requires one recipe design")
        if self.recipe.target_generation_profile_digest != self.target_generation_profile_digest:
            raise DomainError("scenario recipe and seed target profile disagree")
        if self.recipe.measurement_cell != self.measurement_cell:
            raise DomainError("scenario recipe and seed measurement cell disagree")
        if self.recipe.entry_prompt_hash != sha256_bytes(self.entry_prompt.encode("utf-8")):
            raise DomainError("scenario recipe and seed entry prompt identity disagree")


@dataclass(frozen=True, slots=True)
class SeedQualificationContext:
    objective: ObjectiveScopeEvidence
    judge_protocol_digest: str
    judge_panel: tuple[NamedScopeEvidence, ...]
    components: tuple[StructuredRoleScope, ...]
    current_perturbator_scope: PerturbatorRoleScope | None

    def __post_init__(self) -> None:
        if not isinstance(self.objective, ObjectiveScopeEvidence):
            raise DomainError("seed qualification requires objective scope evidence")
        require_digest(self.judge_protocol_digest, "judge protocol digest")
        if (
            not isinstance(self.judge_panel, tuple)
            or not self.judge_panel
            or any(not isinstance(item, NamedScopeEvidence) for item in self.judge_panel)
        ):
            raise DomainError("seed qualification requires immutable judge/panel evidence")
        judge_names = tuple(item.name for item in self.judge_panel)
        if (
            judge_names != tuple(sorted(judge_names))
            or len(judge_names) != len(set(judge_names))
            or "panel" not in judge_names
            or not any(name.startswith("judge_") for name in judge_names)
        ):
            raise DomainError("judge/panel qualification names must be unique, sorted, and complete")
        if not isinstance(self.components, tuple) or any(
            not isinstance(item, StructuredRoleScope) for item in self.components
        ):
            raise DomainError("structured role evidence must be immutable")
        component_names = tuple(item.scope for item in self.components)
        if (
            component_names != tuple(sorted(component_names))
            or len(component_names) != len(set(component_names))
        ):
            raise DomainError("structured role scopes must be unique and sorted")
        if self.current_perturbator_scope is not None and not isinstance(
            self.current_perturbator_scope, PerturbatorRoleScope
        ):
            raise DomainError("current Perturbator scope is invalid")


@dataclass(frozen=True, slots=True)
class SeedQualificationSnapshot:
    seed_id: str
    seed_type: str
    entry_prompt: str
    scenario_recipe: ScenarioRecipe | None
    objective_id: str
    plugin_digest: str
    target_generation_profile_digest: str
    descriptors: DescriptorTuple
    verification_block_id: str
    verification_rule: SeedVerificationRule
    planned_count: int
    resolved_count: int
    high_count: int
    observed_success_fraction: str
    judge_protocol_digest: str
    qualification_digests: tuple[str, ...]
    evidence_event_ids: tuple[str, ...]
    qualified_at: str
    compatibility_key: str
    identity_token: str
    identity_unverifiable: bool
    source: SeedEvidenceSource

    def __post_init__(self) -> None:
        _seed_entity_id(self.seed_id, "sd", "seed snapshot ID")
        if self.seed_type not in {"entry_seed", "scenario_recipe"}:
            raise DomainError("seed snapshot type is invalid")
        object.__setattr__(
            self,
            "entry_prompt",
            normalize_line_endings(_seed_text(self.entry_prompt, "seed snapshot entry prompt")),
        )
        if self.seed_type == "entry_seed" and self.scenario_recipe is not None:
            raise DomainError("entry-seed snapshot cannot contain a scenario recipe")
        if self.seed_type == "scenario_recipe" and not isinstance(
            self.scenario_recipe, ScenarioRecipe
        ):
            raise DomainError("scenario snapshot requires its complete recipe")
        if not isinstance(self.objective_id, str) or OBJECTIVE_ID_RE.fullmatch(self.objective_id) is None:
            raise DomainError("seed snapshot objective ID is invalid")
        require_digest(self.plugin_digest, "seed snapshot plugin digest")
        require_digest(
            self.target_generation_profile_digest,
            "seed snapshot target generation profile digest",
        )
        object.__setattr__(
            self,
            "descriptors",
            _seed_descriptor_cell(self.descriptors, "seed snapshot descriptors"),
        )
        _seed_entity_id(
            self.verification_block_id,
            "sb",
            "seed snapshot verification block ID",
        )
        if not isinstance(self.verification_rule, SeedVerificationRule):
            raise DomainError("seed snapshot verification rule is invalid")
        for value, label in (
            (self.planned_count, "planned count"),
            (self.resolved_count, "resolved count"),
            (self.high_count, "high count"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DomainError(f"seed snapshot {label} must be nonnegative")
        require_digest(self.judge_protocol_digest, "seed snapshot judge protocol digest")
        if not isinstance(self.qualification_digests, tuple):
            raise DomainError("seed snapshot qualification digests must be immutable")
        for digest in self.qualification_digests:
            require_digest(digest, "seed snapshot qualification digest")
        if not isinstance(self.evidence_event_ids, tuple):
            raise DomainError("seed snapshot evidence event IDs must be immutable")
        for event_id in self.evidence_event_ids:
            _seed_entity_id(event_id, "ev", "seed snapshot evidence event ID")
        if len(self.evidence_event_ids) != len(set(self.evidence_event_ids)):
            raise DomainError("seed snapshot evidence event IDs must be unique")
        if not isinstance(self.identity_unverifiable, bool):
            raise DomainError("seed identity unverifiable marker must be Boolean")
        _seed_text(self.identity_token, "seed identity token", one_line=True)
        if self.identity_unverifiable != self.identity_token.startswith("requested:"):
            raise DomainError("seed identity token and unverifiable marker disagree")
        if not (
            self.identity_token.startswith("effective:")
            or self.identity_token.startswith("requested:")
        ) or not self.identity_token.split(":", 1)[1]:
            raise DomainError("seed identity token is invalid")
        if (
            self.planned_count != self.verification_rule.planned_count
            or self.resolved_count != self.planned_count
            or not strict_repeat_passes(self.high_count, self.planned_count)
            or self.observed_success_fraction
            != _exact_fraction(self.high_count, self.planned_count)
        ):
            raise DomainError("seed snapshot verification summary is not promotable")
        if (
            not self.qualification_digests
            or self.qualification_digests != tuple(sorted(set(self.qualification_digests)))
        ):
            raise DomainError("seed snapshot qualification digests must be unique and sorted")
        if not isinstance(self.source, SeedEvidenceSource):
            raise DomainError("seed snapshot source evidence is invalid")
        if self.evidence_event_ids != tuple(
            item.event_id for item in self.source.required_events
        ):
            raise DomainError("seed snapshot evidence references disagree with the source")
        _canonical_qualification_timestamp(self.qualified_at)
        if self.qualified_at != self.source.last_required_event.timestamp_utc:
            raise DomainError("seed qualification time must be the last required evidence event")
        require_digest(self.compatibility_key, "seed compatibility key")

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_id": self.seed_id,
            "type": self.seed_type,
            "entry_prompt": self.entry_prompt,
            "scenario_recipe": (
                None
                if self.scenario_recipe is None
                else self.scenario_recipe.to_dict()
            ),
            "objective_id": self.objective_id,
            "plugin_digest": self.plugin_digest,
            "target_generation_profile_digest": self.target_generation_profile_digest,
            "descriptors": descriptors_dict(self.descriptors),
            "verification_block_id": self.verification_block_id,
            "verification_rule": self.verification_rule.to_dict(),
            "planned_count": self.planned_count,
            "resolved_count": self.resolved_count,
            "high_count": self.high_count,
            "observed_success_fraction": self.observed_success_fraction,
            "judge_protocol_digest": self.judge_protocol_digest,
            "qualification_digests": list(self.qualification_digests),
            "evidence_event_ids": list(self.evidence_event_ids),
            "qualified_at": self.qualified_at,
            "compatibility_key": self.compatibility_key,
            "identity_token": self.identity_token,
            "identity_unverifiable": self.identity_unverifiable,
            "source_checkpoint": self.source.checkpoint_dict(),
        }


@dataclass(frozen=True, slots=True)
class SeedQualificationDecision:
    qualified: bool
    failed_gates: tuple[str, ...]
    verification: VerificationDecision
    compatibility_key: str
    snapshot: SeedQualificationSnapshot | None

    def __post_init__(self) -> None:
        if not isinstance(self.qualified, bool):
            raise DomainError("seed qualification decision must be Boolean")
        if not isinstance(self.failed_gates, tuple) or any(
            not isinstance(item, str) or not item for item in self.failed_gates
        ):
            raise DomainError("seed qualification failed gates must be immutable identifiers")
        if len(self.failed_gates) != len(set(self.failed_gates)):
            raise DomainError("seed qualification failed gates must be unique")
        if not isinstance(self.verification, VerificationDecision):
            raise DomainError("seed qualification decision requires verification evidence")
        if self.snapshot is not None and not isinstance(
            self.snapshot, SeedQualificationSnapshot
        ):
            raise DomainError("seed qualification snapshot is invalid")
        if self.qualified != (not self.failed_gates):
            raise DomainError("seed qualification decision and failed gates disagree")
        if self.qualified != (self.snapshot is not None):
            raise DomainError("only a qualified seed may have a materialized snapshot")
        require_digest(self.compatibility_key, "seed compatibility key")
        if self.snapshot is not None and self.snapshot.compatibility_key != self.compatibility_key:
            raise DomainError("seed snapshot compatibility key disagrees with its decision")


def entry_seed_compatibility_key(
    *,
    objective_id: str,
    plugin_digest: str,
    target_generation_profile_digest: str,
    target_identity_token: str,
    descriptor_taxonomy_digest: str,
    judge_panel_scope_keys: Sequence[tuple[str, str]],
    verification_rule: Mapping[str, Any],
    assigned_probe_review_scope_key: str | None,
) -> str:
    """Own the exact reusable entry-seed compatibility projection."""

    return digest_json(
        [
            "verified_entry_seed_compatibility",
            objective_id,
            plugin_digest,
            target_generation_profile_digest,
            target_identity_token,
            descriptor_taxonomy_digest,
            [list(item) for item in judge_panel_scope_keys],
            dict(verification_rule),
            assigned_probe_review_scope_key,
        ]
    )


def scenario_seed_compatibility_key(
    *,
    objective_id: str,
    plugin_digest: str,
    target_generation_profile_digest: str,
    target_identity_token: str,
    descriptor_taxonomy_digest: str,
    judge_panel_scope_keys: Sequence[tuple[str, str]],
    recipe: ScenarioRecipe,
    measurement_cell: Mapping[str, str],
    verification_rule: Mapping[str, Any],
) -> str:
    """Own the exact reusable stateful-scenario compatibility projection."""

    return digest_json(
        [
            "verified_scenario_seed_compatibility",
            objective_id,
            plugin_digest,
            target_generation_profile_digest,
            target_identity_token,
            descriptor_taxonomy_digest,
            [list(item) for item in judge_panel_scope_keys],
            recipe.content_hash,
            [item.content_hash for item in recipe.templates],
            {
                "designated_realization_policies": list(
                    recipe.designated_realization_policies
                ),
                "maximum_target_turns": recipe.maximum_target_turns,
                "maximum_strategy_rounds": recipe.maximum_strategy_rounds,
            },
            [list(item) for item in recipe.selector_scope_keys],
            recipe.perturbator_scope.key_material(),
            dict(measurement_cell),
            dict(verification_rule),
        ]
    )


def evaluate_seed_qualification(
    subject: EntrySeedSubject | ScenarioSeedSubject,
    verification_block: SeedVerificationBlock,
    context: SeedQualificationContext,
    source: SeedEvidenceSource,
) -> SeedQualificationDecision:
    """Evaluate every Section 13.3 gate and materialize only a qualified seed.

    Inputs are already-reduced immutable evidence.  Missing or extra scope
    records fail closed; the function performs no I/O and infers no semantics
    from prose.
    """

    if not isinstance(subject, (EntrySeedSubject, ScenarioSeedSubject)):
        raise DomainError("seed qualification subject is invalid")
    if not isinstance(verification_block, SeedVerificationBlock):
        raise DomainError("seed qualification verification block is invalid")
    if not isinstance(context, SeedQualificationContext):
        raise DomainError("seed qualification context is invalid")
    if not isinstance(source, SeedEvidenceSource):
        raise DomainError("seed qualification source evidence is invalid")
    if (
        context.objective.objective_id != subject.objective_id
        or context.objective.plugin_digest != subject.plugin_digest
    ):
        raise DomainError("seed subject and objective scope disagree")

    failures: list[str] = []
    component_by_name = {
        item.scope: item for item in context.components
    }
    if isinstance(subject, EntrySeedSubject):
        expected_component_names = {"probe_review"} if subject.generated else set()
        if context.current_perturbator_scope is not None:
            raise DomainError("entry-seed qualification received scenario-only role scope")
        compatibility_key = entry_seed_compatibility_key(
            objective_id=subject.objective_id,
            plugin_digest=subject.plugin_digest,
            target_generation_profile_digest=(
                subject.target_generation_profile_digest
            ),
            target_identity_token=subject.target_identity.token,
            descriptor_taxonomy_digest=subject.descriptor_taxonomy_digest,
            judge_panel_scope_keys=verification_block.judge_panel_scope_keys,
            verification_rule=verification_block.rule.to_dict(),
            assigned_probe_review_scope_key=(
                subject.assigned_probe_review_scope_key
            ),
        )
        seed_type = "entry_seed"
        scenario_recipe: ScenarioRecipe | None = None
        descriptors = subject.descriptors
    else:
        expected_component_names = {"probe_review", "outcome_review"}
        compatibility_key = scenario_seed_compatibility_key(
            objective_id=subject.objective_id,
            plugin_digest=subject.plugin_digest,
            target_generation_profile_digest=(
                subject.target_generation_profile_digest
            ),
            target_identity_token=subject.target_identity.token,
            descriptor_taxonomy_digest=subject.descriptor_taxonomy_digest,
            judge_panel_scope_keys=verification_block.judge_panel_scope_keys,
            recipe=subject.recipe,
            measurement_cell=descriptors_dict(subject.measurement_cell),
            verification_rule=verification_block.rule.to_dict(),
        )
        seed_type = "scenario_recipe"
        scenario_recipe = subject.recipe
        descriptors = subject.measurement_cell

    if set(component_by_name) != expected_component_names:
        failures.append("component_scope_set_mismatch")
    current_judge_panel_keys = tuple(
        (item.name, item.scope_key) for item in context.judge_panel
    )
    if current_judge_panel_keys != verification_block.judge_panel_scope_keys:
        failures.append("judge_panel_scope_mismatch")

    if isinstance(subject, EntrySeedSubject) and subject.generated:
        current_probe = component_by_name.get("probe_review")
        if (
            current_probe is None
            or current_probe.scope_key
            != subject.assigned_probe_review_scope_key
        ):
            failures.append("probe_review_scope_mismatch")

    if isinstance(subject, ScenarioSeedSubject):
        frozen_selector_keys = dict(subject.recipe.selector_scope_keys)
        current_selector_keys = {
            name: item.scope_key
            for name, item in component_by_name.items()
            if name in expected_component_names
        }
        # Section 13.3 gates durable admission on the probe/outcome scopes.
        # Template binding is nevertheless frozen in recipe compatibility
        # identity because execution invokes that manifest-bound scope.
        if current_selector_keys != {
            name: frozen_selector_keys[name]
            for name in sorted(expected_component_names)
        }:
            failures.append("selector_scope_mismatch")
        if context.current_perturbator_scope != subject.recipe.perturbator_scope:
            failures.append("perturbator_scope_mismatch")

    verification = verification_block.reduce()
    if not all(item.fresh_root for item in verification_block.executions):
        failures.append("verification_not_fresh")
    if (
        not verification.block_terminated
        or verification.resolved_count != verification.planned_count
        or verification.unresolved_count
        or verification.pending_count
    ):
        failures.append("verification_not_fully_resolved")
    if not verification.mechanical_repeat_support:
        failures.append("verification_strict_sixty_percent_gate_failed")

    if isinstance(subject, ScenarioSeedSubject):
        if not all(item.end_to_end for item in verification_block.executions):
            failures.append("scenario_not_end_to_end")
        if any(
            item.target_message_injected for item in verification_block.executions
        ):
            failures.append("scenario_target_message_injection")
    elif any(
        item.target_message_injected for item in verification_block.executions
    ):
        failures.append("verification_not_fresh")

    failed_gates = tuple(dict.fromkeys(failures))
    qualification_digests = tuple(
        sorted(
            {
                context.objective.scope_key,
                *(item.scope_key for item in context.judge_panel),
                *(item.scope_key for item in context.components),
                *(
                    (subject.recipe.perturbator_scope.scope_key,)
                    if isinstance(subject, ScenarioSeedSubject)
                    else ()
                ),
            }
        )
    )
    snapshot: SeedQualificationSnapshot | None = None
    if not failed_gates:
        snapshot = SeedQualificationSnapshot(
            seed_id=subject.seed_id,
            seed_type=seed_type,
            entry_prompt=subject.entry_prompt,
            scenario_recipe=scenario_recipe,
            objective_id=subject.objective_id,
            plugin_digest=subject.plugin_digest,
            target_generation_profile_digest=subject.target_generation_profile_digest,
            descriptors=descriptors,
            verification_block_id=verification_block.block_id,
            verification_rule=verification_block.rule,
            planned_count=verification.planned_count,
            resolved_count=verification.resolved_count,
            high_count=verification.confirmed_high_count,
            observed_success_fraction=_exact_fraction(
                verification.confirmed_high_count,
                verification.planned_count,
            ),
            judge_protocol_digest=context.judge_protocol_digest,
            qualification_digests=qualification_digests,
            evidence_event_ids=tuple(
                item.event_id for item in source.required_events
            ),
            qualified_at=source.last_required_event.timestamp_utc,
            compatibility_key=compatibility_key,
            identity_token=subject.target_identity.token,
            identity_unverifiable=subject.target_identity.identity_unverifiable,
            source=source,
        )
    return SeedQualificationDecision(
        qualified=not failed_gates,
        failed_gates=failed_gates,
        verification=verification,
        compatibility_key=compatibility_key,
        snapshot=snapshot,
    )


def evidence_record_from_decisions(
    *,
    subject_id: str,
    screening_block_id: str | None,
    screening: ScreeningDecision | None,
    verification_block_id: str | None = None,
    verification: VerificationDecision | None = None,
) -> EvidenceRecord:
    """Build the one orthogonal evidence view without overwriting search state."""
    search_status = screening.status if screening is not None else SearchStatus.OBSERVED
    verification_status = (
        verification.status
        if verification is not None and verification.status is not None
        else VerificationStatus.NOT_STARTED
    )
    scores = (
        (screening.effective_scores if screening is not None else ())
        + (verification.effective_scores if verification is not None else ())
    )
    unresolved = 0
    if screening is not None:
        unresolved += screening.unresolved_count
    if verification is not None:
        unresolved += verification.unresolved_count
    return EvidenceRecord(
        subject_id=subject_id,
        screening_block_id=screening_block_id,
        search_status=search_status,
        verification_block_id=verification_block_id,
        verification_status=verification_status,
        scores=tuple(scores),
        unresolved_count=unresolved,
    )


def demonstrated_only_block_passes(
    demonstrated_high_count: int, planned_count: int
) -> bool:
    """Reporting split for verified blocks supported without ambiguity votes."""

    return strict_repeat_passes(demonstrated_high_count, planned_count)


__all__ = [
    "EntrySeedSubject",
    "ExecutionEvidence",
    "HIGH_THRESHOLD",
    "NamedScopeEvidence",
    "ObjectiveScopeEvidence",
    "PerturbatorRoleScope",
    "ProviderIdentity",
    "ScenarioRecipe",
    "ScenarioSeedSubject",
    "SUPPORTING_THRESHOLD",
    "ScreeningDecision",
    "SeedEvidenceEvent",
    "SeedEvidenceSource",
    "SeedQualificationContext",
    "SeedQualificationDecision",
    "SeedQualificationSnapshot",
    "SeedVerificationBlock",
    "SeedVerificationExecution",
    "SeedVerificationRule",
    "VerificationDecision",
    "WILSON_Z",
    "WilsonInterval",
    "demonstrated_only_block_passes",
    "entry_seed_compatibility_key",
    "evaluate_screening",
    "evaluate_seed_qualification",
    "evaluate_verification",
    "evidence_record_from_decisions",
    "median_score",
    "parse_scenario_recipe",
    "screening_trigger",
    "scenario_seed_compatibility_key",
    "strict_repeat_passes",
    "wilson_score_interval",
]
