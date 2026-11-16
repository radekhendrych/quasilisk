"""Strict Perturbator batches and common probe-review correlation.

All three generation modes share the same keyed proposal union.  This module
does not generate prose or call a model; it validates untrusted role output and
derives the deterministic winner for an already preallocated slot.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Mapping, Sequence

from .domain import (
    DomainError,
    ProposalAssignment,
    ProposalSlot,
    entity_id,
    require_digest,
    require_entity_id,
    scientific_key,
    strict_json_loads,
)
from .strategy import DescriptorTuple, DescriptorVocabulary, descriptors_dict, parse_descriptors


class PerturbationValidationError(DomainError):
    """A proposal or common probe-review envelope is unusable."""


def _as_object(value: object, label: str) -> dict[str, Any]:
    if isinstance(value, (str, bytes)):
        value = strict_json_loads(value)
    if not isinstance(value, Mapping):
        raise PerturbationValidationError(f"{label} must be one JSON object")
    if not all(isinstance(key, str) for key in value):
        raise PerturbationValidationError(f"{label} keys must be strings")
    return dict(value)


def _exact_fields(value: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    expected_set = frozenset(expected)
    actual = frozenset(value)
    if actual != expected_set:
        missing = sorted(expected_set - actual)
        unknown = sorted(actual - expected_set)
        parts: list[str] = []
        if missing:
            parts.append("missing=" + ",".join(missing))
        if unknown:
            parts.append("unknown=" + ",".join(unknown))
        raise PerturbationValidationError(f"{label} has wrong fields ({'; '.join(parts)})")


def _array(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise PerturbationValidationError(f"{label} must be an array")
    return value


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PerturbationValidationError(f"{label} must be nonempty text")
    return value


@dataclass(frozen=True, slots=True)
class ProposalItem:
    slot_id: str
    status: str
    user_turn: str | None = None
    reason: str | None = None

    @property
    def proposed(self) -> bool:
        return self.status == "proposed"

    def to_dict(self) -> dict[str, Any]:
        if self.status == "abstained":
            return {"slot_id": self.slot_id, "status": "abstained", "reason": self.reason}
        return {
            "slot_id": self.slot_id,
            "status": "proposed",
            "user_turn": self.user_turn,
        }


@dataclass(frozen=True, slots=True)
class ProposalBatch:
    items: tuple[ProposalItem, ...]

    @property
    def all_abstained(self) -> bool:
        return all(item.status == "abstained" for item in self.items)

    @property
    def proposed_items(self) -> tuple[ProposalItem, ...]:
        return tuple(item for item in self.items if item.proposed)

    def to_dict(self) -> dict[str, Any]:
        return {"items": [item.to_dict() for item in self.items]}


def proposal_slot_aliases(
    proposal_slots: Sequence[ProposalSlot],
) -> dict[str, str]:
    """Map durable proposal-slot IDs to short request-local aliases."""

    ordered = tuple(sorted(proposal_slots, key=lambda item: item.global_ordinal))
    if len({item.global_ordinal for item in ordered}) != len(ordered):
        raise PerturbationValidationError("proposal slot ordinals must be unique")
    if len({item.slot_id for item in ordered}) != len(ordered):
        raise PerturbationValidationError("proposal slot IDs must be unique")
    if len({item.scientific_key for item in ordered}) != len(ordered):
        raise PerturbationValidationError("proposal slot scientific keys must be unique")
    return {slot.slot_id: f"s{ordinal}" for ordinal, slot in enumerate(ordered)}


def parse_proposal_batch(
    value: object,
    proposal_slots: Sequence[ProposalSlot],
) -> ProposalBatch:
    """Parse a minimal Perturbator batch and materialize durable slot IDs."""

    obj = _as_object(value, "proposal batch")
    _exact_fields(obj, {"items"}, "proposal batch")
    raw_items = _array(obj["items"], "proposal batch items")
    ordered_slots = tuple(sorted(proposal_slots, key=lambda item: item.global_ordinal))
    alias_by_id = proposal_slot_aliases(ordered_slots)
    slot_by_id = {slot.slot_id: slot for slot in ordered_slots}
    slot_by_alias = {
        alias: slot_by_id[slot_id] for slot_id, alias in alias_by_id.items()
    }

    by_slot: dict[str, ProposalItem] = {}
    for index, raw_item in enumerate(raw_items):
        item_obj = _as_object(raw_item, f"proposal items[{index}]")
        _exact_fields(item_obj, {"slot", "status", "text"}, f"proposal items[{index}]")
        status = item_obj.get("status")
        if status not in {"proposed", "abstained"}:
            raise PerturbationValidationError("proposal item status must be proposed or abstained")
        alias = item_obj["slot"]
        if not isinstance(alias, str) or alias not in slot_by_alias:
            raise PerturbationValidationError("proposal item has an unknown slot alias")
        slot_id = slot_by_alias[alias].slot_id
        if slot_id in by_slot:
            raise PerturbationValidationError("proposal slot alias is duplicated")
        text = _nonempty(item_obj["text"], "proposal text")
        if status == "abstained":
            by_slot[slot_id] = ProposalItem(slot_id, status, reason=text)
            continue
        by_slot[slot_id] = ProposalItem(slot_id, status, user_turn=text)
    if frozenset(by_slot) != frozenset(alias_by_id):
        raise PerturbationValidationError("proposal batch slot-alias set mismatch")
    return ProposalBatch(tuple(by_slot[slot.slot_id] for slot in ordered_slots))


@dataclass(frozen=True, slots=True)
class ReviewExpectation:
    review_item_id: str
    proposal: ProposalItem

    def __post_init__(self) -> None:
        require_entity_id(self.review_item_id, "review item ID")
        if not self.review_item_id.startswith("ri-"):
            raise PerturbationValidationError("review item ID must use ri- prefix")
        if not self.proposal.proposed:
            raise PerturbationValidationError("only a proposed item may be reviewed")


@dataclass(frozen=True, slots=True)
class ProbeReview:
    review_item_id: str
    slot_id: str
    status: str
    reviewed_descriptors: DescriptorTuple | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "review_item_id": self.review_item_id,
            "slot_id": self.slot_id,
            "status": self.status,
            "reviewed_descriptors": (
                None if self.reviewed_descriptors is None else descriptors_dict(self.reviewed_descriptors)
            ),
            "reason": self.reason,
        }


def probe_review_aliases(
    expected_items: Sequence[ReviewExpectation],
) -> dict[str, str]:
    """Map durable review IDs to deterministic request-local aliases."""

    ids = sorted(item.review_item_id for item in expected_items)
    if len(set(ids)) != len(ids):
        raise PerturbationValidationError("expected review item IDs must be unique")
    return {review_id: f"r{ordinal}" for ordinal, review_id in enumerate(ids)}


def parse_probe_reviews(
    value: object,
    expected_items: Sequence[ReviewExpectation],
    descriptor_vocabulary: DescriptorVocabulary,
    *,
    mode: str,
    contract_descriptors_by_slot: Mapping[str, DescriptorTuple | Mapping[str, str]] | None = None,
) -> tuple[ProbeReview, ...]:
    """Parse the common ordinary/strategy probe-review envelope."""

    decisions = _parse_probe_review_decisions(value, expected_items, mode=mode)
    contracts = dict(contract_descriptors_by_slot or {})
    if mode == "strategy" and frozenset(contracts) != frozenset(
        item.proposal.slot_id for item in expected_items
    ):
        raise PerturbationValidationError("strategy review requires one contract tuple per slot")

    reviews: list[ProbeReview] = []
    for expected, status, raw_descriptors, reason in decisions:
        if status in {"invalid_contract", "invalid"}:
            descriptors = None
        elif status == "aligned":
            raw_contract = contracts[expected.proposal.slot_id]
            descriptors = (
                parse_descriptors(
                    descriptors_dict(raw_contract), descriptor_vocabulary, "contract descriptors"
                )
                if isinstance(raw_contract, tuple)
                else parse_descriptors(raw_contract, descriptor_vocabulary, "contract descriptors")
            )
        else:
            descriptors = parse_descriptors(
                raw_descriptors, descriptor_vocabulary, "reviewed_descriptors"
            )
        reviews.append(
            ProbeReview(
                expected.review_item_id,
                expected.proposal.slot_id,
                status,
                descriptors,
                reason,
            )
        )
    return tuple(reviews)


def _parse_probe_review_decisions(
    value: object,
    expected_items: Sequence[ReviewExpectation],
    *,
    mode: str,
) -> tuple[tuple[ReviewExpectation, str, object, str], ...]:
    """Materialize local review aliases before contextual descriptor binding."""

    if mode not in {"ordinary", "strategy"}:
        raise PerturbationValidationError("probe-review mode must be ordinary or strategy")
    obj = _as_object(value, "probe-review envelope")
    _exact_fields(obj, {"reviews"}, "probe-review envelope")
    raw_reviews = _array(obj["reviews"], "probe reviews")
    expectation_by_id = {item.review_item_id: item for item in expected_items}
    alias_by_id = probe_review_aliases(expected_items)
    expectation_by_alias = {
        alias_by_id[review_id]: expectation
        for review_id, expectation in expectation_by_id.items()
    }
    if len({item.proposal.slot_id for item in expected_items}) != len(expected_items):
        raise PerturbationValidationError("one proposal slot cannot appear twice in a review batch")

    by_id: dict[str, tuple[ReviewExpectation, str, object, str]] = {}
    for index, raw_review in enumerate(raw_reviews):
        review_obj = _as_object(raw_review, f"probe reviews[{index}]")
        _exact_fields(
            review_obj,
            {"review_alias", "status", "reviewed_descriptors", "reason", "procedure"},
            f"probe reviews[{index}]",
        )
        review_alias = review_obj["review_alias"]
        if not isinstance(review_alias, str) or review_alias not in expectation_by_alias:
            expected_aliases = ", ".join(sorted(expectation_by_alias))
            raise PerturbationValidationError(
                "probe review_alias must be exactly one of: " + expected_aliases
            )
        expected = expectation_by_alias[review_alias]
        review_id = expected.review_item_id
        if review_id in by_id:
            raise PerturbationValidationError("probe review_alias is duplicated")
        status = review_obj["status"]
        allowed_statuses = {"valid", "invalid"} if mode == "ordinary" else {
            "aligned",
            "invalid_contract",
            "invalid",
        }
        if not isinstance(status, str) or status not in allowed_statuses:
            raise PerturbationValidationError("probe review status is invalid for its mode")
        raw_descriptors = review_obj["reviewed_descriptors"]
        procedure = _as_object(review_obj["procedure"], "probe review procedure")
        _exact_fields(
            procedure, {"assumption", "assumption_quote", "direction", "novelty"},
            "probe review procedure",
        )
        if not isinstance(procedure["assumption"], str) or procedure["assumption"] not in {
            "none", "scenario_fact", "target_owned", "objective_conditional", "unresolved"
        }:
            raise PerturbationValidationError("invalid assumption check")
        if (
            not isinstance(procedure["direction"], str)
            or procedure["direction"] not in {"forward", "reverse", "neutral"}
            or not isinstance(procedure["novelty"], str)
            or procedure["novelty"] not in {"new", "repeated"}
        ):
            raise PerturbationValidationError("invalid direction or novelty check")
        quote = procedure["assumption_quote"]
        if procedure["assumption"] == "none":
            if quote is not None:
                raise PerturbationValidationError("absent assumption requires null quotation")
        elif not isinstance(quote, str) or not quote or quote not in expected.proposal.user_turn:
            raise PerturbationValidationError("assumption quotation must occur exactly in the proposed turn")
        blocked = (
            procedure["assumption"] in {"objective_conditional", "unresolved"}
            or procedure["direction"] != "forward"
            or procedure["novelty"] != "new"
        )
        if blocked:
            status = "invalid" if mode == "ordinary" or procedure["novelty"] == "repeated" else "invalid_contract"
            raw_descriptors = None
        if status in {"invalid_contract", "invalid"}:
            if raw_descriptors is not None:
                raise PerturbationValidationError("invalid review must have null descriptors")
        elif status == "aligned":
            if raw_descriptors is not None:
                raise PerturbationValidationError(
                    "aligned strategy review must not copy descriptors"
                )
        else:
            if raw_descriptors is None:
                raise PerturbationValidationError(
                    "valid ordinary review requires descriptors"
                )
        by_id[review_id] = (
            expected,
            status,
            raw_descriptors,
            ("Procedure blocks dispatch. " if blocked else "") + _nonempty(review_obj["reason"], "probe review reason"),
        )
    if frozenset(by_id) != frozenset(expectation_by_id):
        raise PerturbationValidationError("probe-review item-ID set mismatch")
    return tuple(by_id[item.review_item_id] for item in expected_items)


def parse_probe_review_statuses(
    value: object,
    expected_items: Sequence[ReviewExpectation],
    *,
    mode: str,
) -> dict[str, str]:
    """Return durable review IDs and admission statuses for frozen guards."""

    return {
        expected.review_item_id: status
        for expected, status, _, _ in _parse_probe_review_decisions(
            value, expected_items, mode=mode
        )
    }


@dataclass(frozen=True, slots=True)
class SelectedRealization:
    proposal: ProposalItem
    review: ProbeReview
    source: str


def select_strategy_realization(
    original: ProposalItem,
    original_review: ProbeReview,
    corrective: ProposalItem | None = None,
    corrective_review: ProbeReview | None = None,
) -> SelectedRealization | None:
    """Apply the exact one-slot post-rereview winner rule."""

    if original.slot_id != original_review.slot_id:
        raise PerturbationValidationError("original proposal/review slot mismatch")
    if original_review.status not in {
        "aligned",
        "invalid_contract",
        "invalid",
    }:
        raise PerturbationValidationError("original review is not a strategy-mode review")
    if (corrective is None) != (corrective_review is None):
        raise PerturbationValidationError("corrective proposal and review must occur together")
    if corrective is not None:
        if original_review.status != "invalid":
            raise PerturbationValidationError("strategy correction was not permitted")
        if corrective.slot_id != original.slot_id or corrective_review.slot_id != original.slot_id:
            raise PerturbationValidationError("correction changed its preallocated slot")
        if corrective_review.status not in {
            "aligned",
            "invalid_contract",
            "invalid",
        }:
            raise PerturbationValidationError("corrective review is not a strategy-mode review")
        if corrective_review.review_item_id != original_review.review_item_id:
            raise PerturbationValidationError("correction changed its stable review item ID")
        if corrective_review.status == "aligned":
            return SelectedRealization(corrective, corrective_review, "corrective")
    if original_review.status == "aligned":
        return SelectedRealization(original, original_review, "original")
    return None


def select_ordinary_realization(
    original: ProposalItem,
    original_review: ProbeReview,
    corrective: ProposalItem | None = None,
    corrective_review: ProbeReview | None = None,
    *,
    correction_permitted: bool = False,
) -> SelectedRealization | None:
    """Select an ordinary prompt; MAP mode has no semantic correction cycle."""

    if original.slot_id != original_review.slot_id:
        raise PerturbationValidationError("ordinary proposal/review slot mismatch")
    if original_review.status not in {"valid", "invalid"}:
        raise PerturbationValidationError("original review is not an ordinary-mode review")
    if (corrective is None) != (corrective_review is None):
        raise PerturbationValidationError("ordinary correction is incomplete")
    if corrective is not None:
        if not correction_permitted:
            raise PerturbationValidationError("ordinary MAP review cannot negotiate a correction")
        if original_review.status != "invalid":
            raise PerturbationValidationError("ordinary correction was not permitted")
        if corrective.slot_id != original.slot_id or corrective_review.slot_id != original.slot_id:
            raise PerturbationValidationError("ordinary correction changed its slot")
        if corrective_review.status not in {"valid", "invalid"}:
            raise PerturbationValidationError("corrective review is not an ordinary-mode review")
        if corrective_review.review_item_id != original_review.review_item_id:
            raise PerturbationValidationError("ordinary correction changed its stable review item ID")
        if corrective_review.status == "valid":
            return SelectedRealization(corrective, corrective_review, "corrective")
    if original_review.status == "valid":
        return SelectedRealization(original, original_review, "original")
    return None


def select_strategy_agnostic_realization(
    original: ProposalItem,
    original_review: ProbeReview,
    corrective: ProposalItem | None = None,
    corrective_review: ProbeReview | None = None,
) -> SelectedRealization | None:
    """Select a masked control realization with its one matched correction."""

    return select_ordinary_realization(
        original,
        original_review,
        corrective,
        corrective_review,
        correction_permitted=True,
    )


def make_proposal_slots(work_unit_scientific_key: str, count: int) -> tuple[ProposalSlot, ...]:
    require_digest(work_unit_scientific_key, "work unit scientific key")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise PerturbationValidationError("proposal slot count must be a nonnegative integer")
    result: list[ProposalSlot] = []
    for ordinal in range(count):
        key = scientific_key([work_unit_scientific_key, "proposal_slot", ordinal])
        slot_id = "ps-" + hashlib.sha256(key.encode("utf-8")).hexdigest()
        result.append(ProposalSlot(slot_id, key, ordinal))
    return tuple(result)


def make_review_item_id(
    work_unit_scientific_key: str,
    review_kind: str,
    proposal_slot_scientific_key: str,
) -> str:
    require_digest(work_unit_scientific_key, "work unit scientific key")
    require_digest(proposal_slot_scientific_key, "proposal slot scientific key")
    if review_kind not in {"probe", "outcome"}:
        raise PerturbationValidationError("review kind must be probe or outcome")
    return entity_id(
        "ri",
        ["review_item", work_unit_scientific_key, review_kind, proposal_slot_scientific_key],
    )


def coverage_assignments(proposal_slots: Sequence[ProposalSlot]) -> tuple[ProposalAssignment, ...]:
    ordered = sorted(proposal_slots, key=lambda item: item.global_ordinal)
    if [item.global_ordinal for item in ordered] != list(range(len(ordered))):
        raise PerturbationValidationError("proposal slot ordinals must be contiguous")
    if len({item.scientific_key for item in ordered}) != len(ordered):
        raise PerturbationValidationError("proposal slot scientific keys must be unique")
    if len({item.slot_id for item in ordered}) != len(ordered):
        raise PerturbationValidationError("proposal slot IDs must be unique")
    return tuple(ProposalAssignment(item.scientific_key, None, None, "coverage") for item in ordered)


__all__ = [
    "PerturbationValidationError",
    "ProbeReview",
    "ProposalBatch",
    "ProposalItem",
    "ReviewExpectation",
    "SelectedRealization",
    "coverage_assignments",
    "make_proposal_slots",
    "make_review_item_id",
    "parse_probe_review_statuses",
    "parse_probe_reviews",
    "parse_proposal_batch",
    "probe_review_aliases",
    "proposal_slot_aliases",
    "select_ordinary_realization",
    "select_strategy_agnostic_realization",
    "select_strategy_realization",
]
