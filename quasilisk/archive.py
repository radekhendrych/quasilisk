"""The single derived MAP-Elites archive and deterministic selection policy."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from .domain import (
    ArchiveEntry,
    DomainError,
    SearchStatus,
    VerificationStatus,
    require_digest,
    require_entity_id,
    seeded_rank,
)


Cell = tuple[str, ...]
ArchiveCells = Mapping[Cell, tuple[ArchiveEntry, ...]]

_VERIFICATION_RANK: Mapping[VerificationStatus, int] = {
    VerificationStatus.VERIFIED: 2,
    VerificationStatus.NOT_STARTED: 0,
    VerificationStatus.FAILED: 0,
    VerificationStatus.INCONCLUSIVE: 0,
}
_SEARCH_RANK: Mapping[SearchStatus, int] = {
    SearchStatus.PROMISING: 2,
    SearchStatus.EXPLORATORY_LEAD: 1,
    SearchStatus.OBSERVED: 0,
}


@dataclass(frozen=True, slots=True)
class ArchiveSnapshot:
    """Immutable current projection of retained entries by descriptor cell."""

    capacity_per_cell: int
    cells: ArchiveCells

    def __post_init__(self) -> None:
        if self.capacity_per_cell < 1:
            raise DomainError("archive capacity must be at least one")
        copied: dict[Cell, tuple[ArchiveEntry, ...]] = {}
        seen_keys: set[str] = set()
        descriptor_arity: int | None = None
        for raw_cell, entries in self.cells.items():
            cell = _validate_cell(raw_cell)
            if descriptor_arity is None:
                descriptor_arity = len(cell)
            elif len(cell) != descriptor_arity:
                raise DomainError("archive cells mix descriptor arities")
            if not entries:
                raise DomainError("archive cannot retain an empty occupied cell")
            if len(entries) > self.capacity_per_cell:
                raise DomainError("archive cell exceeds configured capacity")
            ordered = tuple(sorted(entries, key=archive_entry_sort_key))
            for entry in ordered:
                _validate_entry(entry)
                if entry.cell != cell:
                    raise DomainError("archive entry is stored under a different cell")
                if entry.scientific_key in seen_keys:
                    raise DomainError("archive entry scientific key occurs more than once")
                seen_keys.add(entry.scientific_key)
            copied[cell] = ordered
        object.__setattr__(self, "cells", MappingProxyType(copied))

    @property
    def occupied_cell_count(self) -> int:
        return len(self.cells)

    @property
    def entries(self) -> tuple[ArchiveEntry, ...]:
        return tuple(
            entry
            for cell in sorted(self.cells)
            for entry in self.cells[cell]
        )


@dataclass(frozen=True, slots=True)
class ArchiveUpdate:
    snapshot: ArchiveSnapshot
    retained: bool
    evicted: tuple[ArchiveEntry, ...]


@dataclass(frozen=True, slots=True)
class ArchiveMetric:
    verified_cell_count: int
    occupied_cell_count: int
    total_possible_cells: int
    all_target_attempts: int
    completed_target_responses: int
    rate_numerator: int
    rate_denominator: int
    rate_display: str | None

    def to_dict(self) -> dict[str, int | str | None]:
        return {
            "verified_cell_count": self.verified_cell_count,
            "occupied_cell_count": self.occupied_cell_count,
            "total_possible_cells": self.total_possible_cells,
            "all_target_attempts": self.all_target_attempts,
            "completed_target_responses": self.completed_target_responses,
            "rate_numerator": self.rate_numerator,
            "rate_denominator": self.rate_denominator,
            "rate_display": self.rate_display,
        }


@dataclass(frozen=True, slots=True)
class FitnessDecision:
    fitness: Decimal | None
    source: str


def _score_decimal(value: Decimal | int | str) -> Decimal:
    if isinstance(value, bool):
        raise DomainError("fitness source score cannot be boolean")
    try:
        score = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise DomainError("fitness source score is not a decimal") from exc
    if not score.is_finite() or not 0 <= score <= 100:
        raise DomainError("fitness source score must be finite and within [0, 100]")
    return score


def _median(scores: Iterable[Decimal | int | str]) -> Decimal | None:
    ordered = sorted(_score_decimal(value) for value in scores)
    if not ordered:
        return None
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / Decimal(2)


def _validate_cell(cell: Sequence[str]) -> Cell:
    if isinstance(cell, (str, bytes)):
        raise DomainError("archive cell must be a descriptor tuple")
    result = tuple(cell)
    if len(result) not in {2, 3} or not all(
        isinstance(value, str) and value for value in result
    ):
        raise DomainError("archive cell must contain two or three nonempty descriptor values")
    return result


def _validate_fitness(fitness: Decimal | None) -> None:
    if fitness is None:
        return
    if not isinstance(fitness, Decimal):
        raise DomainError("archive fitness must be Decimal or null")
    if not fitness.is_finite() or not 0 <= fitness <= 100:
        raise DomainError("archive fitness must be finite and within [0, 100]")


def _validate_entry(entry: ArchiveEntry) -> None:
    _validate_cell(entry.cell)
    require_entity_id(entry.entry_id, "archive entry ID")
    if not entry.entry_id.startswith("ae-"):
        raise DomainError("archive entry ID must use ae- prefix")
    require_digest(entry.scientific_key, "archive entry scientific key")
    _validate_fitness(entry.fitness)


def archive_entry_sort_key(entry: ArchiveEntry) -> tuple[object, ...]:
    """Best-first key for the binding status/median/scientific-key ranking."""

    _validate_entry(entry)
    verification_rank = _VERIFICATION_RANK[entry.evidence.verification_status]
    search_rank = _SEARCH_RANK[entry.evidence.search_status]
    # Null sorts below every finite median; scientific key is the final
    # ascending tie-break while every preceding quality component is descending.
    return (
        -verification_rank,
        -search_rank,
        entry.fitness is None,
        Decimal(0) if entry.fitness is None else -entry.fitness,
        entry.scientific_key,
    )


def entry_is_better(left: ArchiveEntry, right: ArchiveEntry) -> bool:
    return archive_entry_sort_key(left) < archive_entry_sort_key(right)


def retain_cell_entries(
    entries: Iterable[ArchiveEntry], capacity_per_cell: int
) -> tuple[ArchiveEntry, ...]:
    """Retain the best distinct entries in one cell."""

    if capacity_per_cell < 1:
        raise DomainError("archive capacity must be at least one")
    entries_tuple = tuple(entries)
    if not entries_tuple:
        return ()
    cell = _validate_cell(entries_tuple[0].cell)
    by_key: dict[str, ArchiveEntry] = {}
    for entry in entries_tuple:
        _validate_entry(entry)
        if entry.cell != cell:
            raise DomainError("cannot rank entries from different cells together")
        existing = by_key.get(entry.scientific_key)
        if existing is not None and existing.entry_id != entry.entry_id:
            raise DomainError("one archive scientific key has conflicting entity IDs")
        if existing is not None and existing != entry:
            raise DomainError("one archive scientific key has multiple current views")
        # Replacing the same stable entry key is how reducer-derived evidence
        # evolves without creating a second archive entry.
        by_key[entry.scientific_key] = entry
    return tuple(sorted(by_key.values(), key=archive_entry_sort_key)[:capacity_per_cell])


def build_archive(
    entries: Iterable[ArchiveEntry], capacity_per_cell: int
) -> ArchiveSnapshot:
    """Build the complete one-archive projection from current entry views."""

    if capacity_per_cell < 1:
        raise DomainError("archive capacity must be at least one")
    grouped: dict[Cell, list[ArchiveEntry]] = {}
    global_keys: dict[str, Cell] = {}
    for entry in entries:
        _validate_entry(entry)
        cell = _validate_cell(entry.cell)
        previous_cell = global_keys.get(entry.scientific_key)
        if previous_cell is not None and previous_cell != cell:
            raise DomainError("one archive entry scientific key cannot change cells")
        global_keys[entry.scientific_key] = cell
        grouped.setdefault(cell, []).append(entry)
    retained = {
        cell: retain_cell_entries(cell_entries, capacity_per_cell)
        for cell, cell_entries in grouped.items()
    }
    return ArchiveSnapshot(capacity_per_cell=capacity_per_cell, cells=retained)


def insert_archive_entry(
    snapshot: ArchiveSnapshot, entry: ArchiveEntry
) -> ArchiveUpdate:
    """Return a new projection after inserting or refreshing one stable entry."""

    _validate_entry(entry)
    for cell, entries in snapshot.cells.items():
        if cell == entry.cell:
            continue
        if any(current.scientific_key == entry.scientific_key for current in entries):
            raise DomainError("one archive entry scientific key cannot change cells")

    current_entries = snapshot.cells.get(entry.cell, ())
    current_by_key = {item.scientific_key: item for item in current_entries}
    combined = [
        current
        for current in current_entries
        if current.scientific_key != entry.scientific_key
    ]
    combined.append(entry)
    retained_entries = retain_cell_entries(combined, snapshot.capacity_per_cell)
    retained_keys = {item.scientific_key for item in retained_entries}
    evicted = tuple(
        current
        for key, current in current_by_key.items()
        if key not in retained_keys and key != entry.scientific_key
    )

    cells = dict(snapshot.cells)
    cells[entry.cell] = retained_entries
    return ArchiveUpdate(
        snapshot=ArchiveSnapshot(snapshot.capacity_per_cell, cells),
        retained=entry.scientific_key in retained_keys,
        evicted=tuple(sorted(evicted, key=archive_entry_sort_key)),
    )


def derive_archive_fitness(
    *,
    verification_terminal: bool = False,
    verification_scores: Iterable[Decimal | int | str] = (),
    screening_complete: bool = False,
    screening_scores: Iterable[Decimal | int | str] = (),
    singleton_score: Decimal | int | str | None = None,
) -> FitnessDecision:
    """Apply verification/screen/singleton source precedence using medians."""

    verification_values = tuple(verification_scores)
    screening_values = tuple(screening_scores)
    if verification_terminal:
        return FitnessDecision(_median(verification_values), "verification")
    if screening_complete:
        return FitnessDecision(_median(screening_values), "screening")
    if singleton_score is None:
        return FitnessDecision(None, "singleton")
    singleton = _median((singleton_score,))
    return FitnessDecision(singleton, "singleton")


def cell_is_verified(entries: Iterable[ArchiveEntry]) -> bool:
    return any(
        entry.evidence.verification_status is VerificationStatus.VERIFIED
        for entry in entries
    )


def verified_cells(snapshot: ArchiveSnapshot) -> tuple[Cell, ...]:
    return tuple(
        cell
        for cell in sorted(snapshot.cells)
        if cell_is_verified(snapshot.cells[cell])
    )


def under_covered_cells(
    declared_grid: Iterable[Sequence[str]],
    snapshot: ArchiveSnapshot,
    *,
    prior_directed_slots: Mapping[Cell, int] | None = None,
    limit: int,
) -> tuple[Cell, ...]:
    """Order the complete grid by the exact coverage tuple and take ``limit``."""

    if limit < 0:
        raise DomainError("under-covered cell limit cannot be negative")
    cells = tuple(_validate_cell(cell) for cell in declared_grid)
    if len(set(cells)) != len(cells):
        raise DomainError("declared descriptor grid contains duplicate cells")
    if len({len(cell) for cell in cells}) > 1:
        raise DomainError("declared descriptor grid mixes descriptor arities")
    unknown_occupied = set(snapshot.cells) - set(cells)
    if unknown_occupied:
        raise DomainError("archive contains a cell outside the declared grid")
    counts = {} if prior_directed_slots is None else dict(prior_directed_slots)
    unknown = set(counts) - set(cells)
    if unknown:
        raise DomainError("prior proposal counts name cells outside the declared grid")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts.values()):
        raise DomainError("prior proposal-slot counts must be nonnegative integers")

    def key(cell: Cell) -> tuple[object, ...]:
        retained = snapshot.cells.get(cell, ())
        occupied = bool(retained)
        return (
            occupied,
            cell_is_verified(retained) if occupied else False,
            len(retained),
            counts.get(cell, 0),
            cell,
        )

    return tuple(sorted(cells, key=key)[:limit])


def _select_parent_cell(
    candidate_cells: Sequence[Cell],
    snapshot: ArchiveSnapshot,
    *,
    ordinary_expansion_counts: Mapping[Cell, int],
    seed: int,
    decision_kind: str,
) -> Cell:
    def base_key(cell: Cell) -> tuple[int, int, int]:
        entries = snapshot.cells[cell]
        return (
            int(cell_is_verified(entries)),
            len(entries),
            ordinary_expansion_counts.get(cell, 0),
        )

    best_base = min(base_key(cell) for cell in candidate_cells)
    tied = [cell for cell in candidate_cells if base_key(cell) == best_base]
    return min(
        tied,
        key=lambda cell: (
            seeded_rank(seed, [decision_kind, "cell", list(cell)]),
            cell,
        ),
    )


def select_archive_parent(
    snapshot: ArchiveSnapshot,
    *,
    expandable_entry_keys: Iterable[str],
    ordinary_expansion_counts: Mapping[Cell, int] | None = None,
    seed: int,
    decision_kind: str = "archive_parent",
) -> ArchiveEntry | None:
    """Choose one expandable retained standalone entry deterministically."""

    expandable = frozenset(expandable_entry_keys)
    if not expandable:
        return None
    counts = {} if ordinary_expansion_counts is None else dict(ordinary_expansion_counts)
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts.values()):
        raise DomainError("ordinary expansion counts must be nonnegative integers")

    candidates_by_cell: dict[Cell, tuple[ArchiveEntry, ...]] = {}
    for cell, entries in snapshot.cells.items():
        eligible = tuple(
            entry for entry in entries if entry.scientific_key in expandable
        )
        if eligible:
            candidates_by_cell[cell] = eligible
    if not candidates_by_cell:
        return None

    selected_cell = _select_parent_cell(
        tuple(candidates_by_cell),
        snapshot,
        ordinary_expansion_counts=counts,
        seed=seed,
        decision_kind=decision_kind,
    )
    # Evidence rank already has the required nonce-free ascending final tie-break.
    return min(candidates_by_cell[selected_cell], key=archive_entry_sort_key)


def archive_metric(
    snapshot: ArchiveSnapshot,
    *,
    total_possible_cells: int,
    all_target_attempts: int,
    completed_target_responses: int,
) -> ArchiveMetric:
    """Compute the primary archive endpoint at one physical-attempt cutoff."""

    if total_possible_cells < 0:
        raise DomainError("total possible cells cannot be negative")
    if snapshot.occupied_cell_count > total_possible_cells:
        raise DomainError("occupied cells exceed the declared grid")
    if all_target_attempts < 0 or completed_target_responses < 0:
        raise DomainError("target counts cannot be negative")
    if completed_target_responses > all_target_attempts:
        raise DomainError("completed target responses exceed physical attempts")
    verified_count = len(verified_cells(snapshot))
    display: str | None = None
    if all_target_attempts:
        with localcontext() as context:
            context.prec = 50
            rate = Decimal(verified_count) / Decimal(all_target_attempts)
            display = format(
                rate.quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN), "f"
            )
    return ArchiveMetric(
        verified_cell_count=verified_count,
        occupied_cell_count=snapshot.occupied_cell_count,
        total_possible_cells=total_possible_cells,
        all_target_attempts=all_target_attempts,
        completed_target_responses=completed_target_responses,
        rate_numerator=verified_count,
        rate_denominator=all_target_attempts,
        rate_display=display,
    )


__all__ = [
    "ArchiveCells",
    "ArchiveMetric",
    "ArchiveSnapshot",
    "ArchiveUpdate",
    "Cell",
    "FitnessDecision",
    "archive_entry_sort_key",
    "archive_metric",
    "build_archive",
    "cell_is_verified",
    "derive_archive_fitness",
    "entry_is_better",
    "insert_archive_entry",
    "retain_cell_entries",
    "select_archive_parent",
    "under_covered_cells",
    "verified_cells",
]
