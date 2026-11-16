"""Binding-contract tests for the one archive and deterministic scheduler."""

from __future__ import annotations

import unittest
from decimal import Decimal
from unittest.mock import patch

from quasilisk.archive import (
    archive_metric,
    build_archive,
    derive_archive_fitness,
    insert_archive_entry,
    select_archive_parent,
    under_covered_cells,
    verified_cells,
)
from quasilisk.domain import (
    ArchiveEntry,
    DomainError,
    EvidenceRecord,
    RoleCallVector,
    SearchStatus,
    VerificationStatus,
    WorkUnit,
    entity_id,
    scientific_key,
)
from quasilisk.scheduler import (
    SchedulerValidationError,
    WorkCandidate,
    can_start,
    can_start_sequence,
    choose_next_work,
    deterministic_decision_key,
    entry_verification_maximum,
    fairness_allows_stateful,
    fairness_debt,
    formula_maximum,
    initial_observation_maximum,
    map_exploration_maximum,
    order_candidates,
    parent_baseline_root_maximum,
    portfolio_round_maximum,
    screening_maximum,
    stateful_root_maximum,
    strategy_agnostic_root_maximum,
    strategy_development_maximum,
    strategy_holdout_maximum,
    treatment_root_maximum,
)


def _entry(
    label: str,
    cell: tuple[str, ...],
    *,
    search: SearchStatus = SearchStatus.OBSERVED,
    verification: VerificationStatus = VerificationStatus.NOT_STARTED,
    fitness: str | None = None,
) -> ArchiveEntry:
    evidence = EvidenceRecord(
        subject_id=entity_id("pc", ["subject", label]),
        search_status=search,
        verification_status=verification,
    )
    return ArchiveEntry(
        entry_id=entity_id("ae", [label]),
        scientific_key=scientific_key(["archive-entry", label]),
        subject_id=evidence.subject_id,
        cell=cell,
        evidence=evidence,
        fitness=None if fitness is None else Decimal(fitness),
    )


def _work(
    label: str,
    kind: str,
    *,
    proposal_count: int = 3,
    active: bool = False,
    verified_cell: bool = False,
    tier: str = "ordinary",
    median: str | None = None,
    prior_calls: int = 0,
    ordinal: int = 0,
    portfolio_intent: str = "advance",
) -> WorkCandidate:
    subject_key = scientific_key(["subject", label])
    if kind == "verification":
        design = {
            "tag": kind,
            "subject_kind": "entry_prompt",
            "planned_count": 1,
        }
    elif kind == "strategy_development":
        design = {"tag": kind, "planned_count_per_arm": 1}
    elif kind == "strategy_holdout":
        design = {
            "tag": kind,
            "sealed_holdout_plan": {"planned_count_per_arm": 1},
        }
    elif kind in {"map_exploration", "portfolio_round"}:
        design = {
            "tag": kind,
            "proposal_count": proposal_count,
        }
        if kind == "portfolio_round":
            design.update(
                round_index=0,
                maximum_strategy_count=proposal_count // 3,
                portfolio_intent=portfolio_intent,
            )
    else:
        design = {"tag": kind}
    maximum = formula_maximum(kind, design)
    unit_key = scientific_key(["work", label])
    unit = WorkUnit(
        unit_id=entity_id("wu", [label]),
        unit_scientific_key=unit_key,
        kind=kind,
        subject_id=entity_id("pc", ["subject", label]),
        subject_scientific_key=subject_key,
        parent_unit_id=None,
        policy_digest=scientific_key(["policy"]),
        design=design,
        worst_case_calls=maximum,
    )
    # These are policy-rank stubs, not frozen WorkUnit conformance fixtures.
    # Conformance is exercised with materialized engine units in its own tests.
    with patch(
        "quasilisk.scheduler.validate_work_unit_maximum",
        return_value=maximum,
    ):
        return WorkCandidate(
            work_unit=unit,
            active=active,
            measurement_cell_verified=verified_cell,
            evidence_tier=tier,
            fixed_block_median=None if median is None else Decimal(median),
            prior_target_calls=prior_calls,
            slot_ordinal=ordinal,
        )


class ArchiveFitnessTests(unittest.TestCase):
    def test_fitness_precedence_uses_terminal_block_medians(self) -> None:
        decision = derive_archive_fitness(
            verification_terminal=True,
            verification_scores=("10", "90", "50"),
            screening_complete=True,
            screening_scores=("99", "100", "100"),
            singleton_score="100",
        )
        self.assertEqual((decision.source, decision.fitness), ("verification", Decimal("50")))
        screen = derive_archive_fitness(
            screening_complete=True,
            screening_scores=("60", "80"),
            singleton_score="100",
        )
        self.assertEqual((screen.source, screen.fitness), ("screening", Decimal("70")))

    def test_status_first_ranking_protects_verified_evidence(self) -> None:
        cell = ("route", "move")
        verified = _entry(
            "verified", cell, search=SearchStatus.OBSERVED,
            verification=VerificationStatus.VERIFIED, fitness="1",
        )
        lucky = _entry(
            "lucky", cell, search=SearchStatus.PROMISING,
            verification=VerificationStatus.NOT_STARTED, fitness="100",
        )
        snapshot = build_archive((lucky, verified), capacity_per_cell=1)
        self.assertEqual(snapshot.cells[cell], (verified,))
        self.assertEqual(verified_cells(snapshot), (cell,))

    def test_search_status_precedes_median_and_null_is_last(self) -> None:
        cell = ("route", "move")
        promising_null = _entry(
            "promising-null", cell, search=SearchStatus.PROMISING, fitness=None
        )
        observed_high = _entry("observed-high", cell, fitness="100")
        snapshot = build_archive((observed_high, promising_null), capacity_per_cell=1)
        self.assertEqual(snapshot.cells[cell], (promising_null,))

        finite = _entry("finite", cell, fitness="0")
        null = _entry("null", cell, fitness=None)
        snapshot = build_archive((null, finite), capacity_per_cell=1)
        self.assertEqual(snapshot.cells[cell], (finite,))

    def test_insert_is_immutable_and_reports_eviction(self) -> None:
        cell = ("route", "move")
        old = _entry("old", cell, fitness="20")
        new = _entry("new", cell, fitness="30")
        original = build_archive((old,), 1)
        update = insert_archive_entry(original, new)
        self.assertEqual(original.cells[cell], (old,))
        self.assertEqual(update.snapshot.cells[cell], (new,))
        self.assertTrue(update.retained)
        self.assertEqual(update.evicted, (old,))

    def test_under_covered_order_is_exact_and_grid_closed(self) -> None:
        empty_first = ("a", "move")
        empty_directed = ("b", "move")
        occupied = ("c", "move")
        verified = ("d", "move")
        snapshot = build_archive(
            (
                _entry("occupied", occupied, fitness="20"),
                _entry("verified", verified, verification=VerificationStatus.VERIFIED, fitness="20"),
            ),
            2,
        )
        result = under_covered_cells(
            (verified, occupied, empty_directed, empty_first),
            snapshot,
            prior_directed_slots={empty_directed: 2},
            limit=4,
        )
        self.assertEqual(result, (empty_first, empty_directed, occupied, verified))
        with self.assertRaises(DomainError):
            under_covered_cells((empty_first,), snapshot, limit=1)

    def test_parent_selection_prefers_unverified_less_expanded_cell(self) -> None:
        cell_a = ("a", "move")
        cell_b = ("b", "move")
        a = _entry("a", cell_a, verification=VerificationStatus.VERIFIED, fitness="99")
        b = _entry("b", cell_b, fitness="1")
        snapshot = build_archive((a, b), 1)
        selected = select_archive_parent(
            snapshot,
            expandable_entry_keys=(a.scientific_key, b.scientific_key),
            ordinary_expansion_counts={cell_a: 0, cell_b: 5},
            seed=7,
        )
        self.assertEqual(selected, b)

    def test_headline_metric_uses_physical_target_attempt_denominator(self) -> None:
        snapshot = build_archive(
            (
                _entry("verified-a", ("a", "move"), verification=VerificationStatus.VERIFIED),
                _entry("verified-b", ("b", "move"), verification=VerificationStatus.VERIFIED),
                _entry("observed", ("c", "move")),
            ),
            1,
        )
        metric = archive_metric(
            snapshot,
            total_possible_cells=20,
            all_target_attempts=8,
            completed_target_responses=6,
        )
        self.assertEqual(metric.verified_cell_count, 2)
        self.assertEqual(metric.rate_display, "0.250000")
        self.assertEqual(metric.rate_denominator, 8)


class SchedulerFormulaTests(unittest.TestCase):
    def test_closed_work_unit_maxima_match_binding_formulas(self) -> None:
        self.assertEqual(initial_observation_maximum(), RoleCallVector(2, 8, 0, 0))
        self.assertEqual(screening_maximum(), RoleCallVector(4, 16, 0, 0))
        self.assertEqual(map_exploration_maximum(6), RoleCallVector(12, 48, 4, 4))
        self.assertEqual(
            portfolio_round_maximum(6, 2),
            RoleCallVector(12, 48, 10, 8),
        )
        self.assertEqual(entry_verification_maximum(5), RoleCallVector(10, 40, 0, 0))
        self.assertEqual(stateful_root_maximum(2), RoleCallVector(6, 24, 32, 16))
        self.assertEqual(treatment_root_maximum(), RoleCallVector(4, 16, 16, 8))
        self.assertEqual(parent_baseline_root_maximum(), RoleCallVector(2, 8, 0, 0))
        self.assertEqual(strategy_agnostic_root_maximum(), RoleCallVector(4, 16, 8, 8))
        self.assertEqual(strategy_development_maximum(5), RoleCallVector(50, 200, 120, 80))
        self.assertEqual(strategy_holdout_maximum(5), RoleCallVector(40, 160, 120, 80))

    def test_maximum_rejects_underallocation_and_bad_strategy_count(self) -> None:
        with self.assertRaises(SchedulerValidationError):
            map_exploration_maximum(2)
        with self.assertRaises(SchedulerValidationError):
            portfolio_round_maximum(7, 1)

    def test_admission_requires_every_role_and_complete_sequence_to_fit(self) -> None:
        initial = _work("initial", "initial_observation").work_unit
        screen = _work("screen", "screening").work_unit
        maximum = lambda unit: unit.worst_case_calls
        with patch("quasilisk.scheduler.validate_work_unit_maximum", side_effect=maximum):
            self.assertTrue(can_start(initial, RoleCallVector(2, 8)))
            self.assertFalse(can_start(initial, RoleCallVector(2, 7, 100, 100)))
            self.assertTrue(
                can_start_sequence(
                    (screen, initial),
                    RoleCallVector(target=6, judge=24),
                )
            )
            self.assertFalse(
                can_start_sequence(
                    (screen, initial),
                    RoleCallVector(target=6, judge=23),
                )
            )
        with self.assertRaises(SchedulerValidationError):
            can_start(RoleCallVector(2, 8), RoleCallVector(2, 8))  # type: ignore[arg-type]


class SchedulerPolicyTests(unittest.TestCase):
    def test_all_next_work_priority_classes_follow_the_binding_order(self) -> None:
        candidates = (
            _work("map", "map_exploration"),
            _work("initial", "initial_observation"),
            _work("portfolio", "portfolio_round"),
            _work("screen", "screening"),
            _work("holdout", "strategy_holdout"),
            _work("development", "strategy_development"),
            _work("verification", "verification"),
        )
        kinds = [item.kind for item in order_candidates(candidates)]
        self.assertEqual(kinds[0], "verification")
        self.assertEqual(set(kinds[-2:]), {"strategy_development", "strategy_holdout"})
        self.assertEqual(
            kinds[1:5],
            ["portfolio_round", "screening", "initial_observation", "map_exploration"],
        )

    def test_priority_and_within_class_order_are_lexicographic(self) -> None:
        verification = _work("verification", "verification")
        screen = _work("screen", "screening")
        ordinary = _work("ordinary", "initial_observation")
        ordered = order_candidates((ordinary, screen, verification))
        self.assertEqual([item.kind for item in ordered], ["verification", "screening", "initial_observation"])

        application = _work(
            "application", "verification", tier="application",
            median="70", verified_cell=False,
        )
        exceptional = _work(
            "exceptional", "verification", tier="exceptional",
            median="100", verified_cell=False,
        )
        verified_cell = _work(
            "verified-cell", "verification", tier="application",
            median="100", verified_cell=True,
        )
        self.assertEqual(
            list(order_candidates((verified_cell, exceptional, application))),
            [application, exceptional, verified_cell],
        )

        recovery = _work(
            "a-recovery",
            "portfolio_round",
            portfolio_intent="recover",
            tier="application",
            median="100",
        )
        advance = _work(
            "z-advance",
            "portfolio_round",
            portfolio_intent="advance",
        )
        self.assertEqual(order_candidates((recovery, advance)), (advance, recovery))

    def test_active_work_always_finishes_without_readmission(self) -> None:
        active = _work("active", "screening", active=True)
        result = choose_next_work(
            (active,), RoleCallVector(), coverage_slots_committed=0, stateful_slots_committed=0
        )
        self.assertEqual(result, active)

    def test_one_to_three_fairness_redirects_to_ordinary_work(self) -> None:
        stateful = _work(
            "stateful", "portfolio_round", proposal_count=3, portfolio_intent="recover"
        )
        ordinary = _work(
            "ordinary", "initial_observation"
        )
        self.assertFalse(
            fairness_allows_stateful(0, 0, 3, ordinary_work_admissible=True)
        )
        with patch(
            "quasilisk.scheduler.validate_work_unit_maximum",
            side_effect=lambda unit: unit.worst_case_calls,
        ):
            selected = choose_next_work(
                (ordinary, stateful),
                RoleCallVector(target=6, judge=24, selector=16, perturbator=8),
                coverage_slots_committed=0,
                stateful_slots_committed=0,
            )
        self.assertEqual(selected, ordinary)
        self.assertTrue(
            fairness_allows_stateful(0, 0, 3, ordinary_work_admissible=False)
        )
        self.assertEqual(fairness_debt(1, 5), 2)

    def test_seeded_decision_has_no_mutable_stream(self) -> None:
        subject = scientific_key(["stable subject"])
        first = deterministic_decision_key(42, "parent", subject, 1, 2)
        self.assertEqual(first, deterministic_decision_key(42, "parent", subject, 1, 2))
        self.assertNotEqual(first, deterministic_decision_key(42, "parent", subject, 1, 3))
        self.assertNotEqual(first, deterministic_decision_key(43, "parent", subject, 1, 2))


if __name__ == "__main__":
    unittest.main()
