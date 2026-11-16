"""Focused report projections for verification and strategy evidence."""

from __future__ import annotations

import ast
from decimal import Decimal
from inspect import signature
from pathlib import Path
from types import SimpleNamespace
import unittest

from quasilisk.domain import (
    Application,
    EvidenceExcerpt,
    MemberAssessment,
    Verdict,
    digest_json,
    entity_id,
)
from quasilisk.judging import aggregate_panel_details
from quasilisk.reporting import (
    _qualification_decision,
    _strategy_experiments,
    _strategy_search_summary,
    archive_projection,
    build_report_bundle,
    write_derived_artifacts,
)
from quasilisk.run_state import PanelView, RunState
from quasilisk.strategy import (
    DevelopmentBlockDecision,
    HoldoutBlockDecision,
    PurposeGateDecision,
    QualificationArmMetrics,
    QualificationState,
)


def _verification_state(planned_count: int) -> tuple[SimpleNamespace, list[str]]:
    subject_id = entity_id("pc", ["reporting-demonstrated", planned_count])
    target_action_ids = [
        entity_id("la", ["reporting-demonstrated", planned_count, ordinal])
        for ordinal in range(planned_count)
    ]
    state = SimpleNamespace(
        work_units=(
            SimpleNamespace(
                unit={
                    "kind": "verification",
                    "subject_id": subject_id,
                    "design": {
                        "planned_count": planned_count,
                        "root_plans": [
                            {"terminal_endpoint_action_id": action_id}
                            for action_id in target_action_ids
                        ],
                    },
                }
            ),
        ),
        panel_views={},
    )
    return state, target_action_ids


def _panel(action_id: str, *, confirmed_high: bool, demonstrated: int) -> SimpleNamespace:
    return SimpleNamespace(
        target_action_id=action_id,
        terminated=True,
        details=SimpleNamespace(
            assessment=SimpleNamespace(
                confirmed_objective_high=confirmed_high
            ),
            counts_by_verdict={"demonstrated": demonstrated},
        ),
    )


def _member(slot: str, score: str, identity: str) -> MemberAssessment:
    material = ["reporting-panel-view", slot, score, identity]
    return MemberAssessment(
        assessment_id=entity_id("ma", material),
        observation_id=entity_id("to", ["reporting-panel-view"]),
        logical_member_slot=slot,
        score=Decimal(score),
        verdict=Verdict.AMBIGUOUS,
        application=Application.ABSTRACT,
        evidence=(EvidenceExcerpt("reply", "supports"),),
        justification="Grounded.",
        warnings=(),
        effective_model_id=identity,
        role_route_key=digest_json(["reporting-panel-route", slot]),
        source_attempt_id=entity_id("pa", material),
     ownership="adopted")


def _outcome(**changes: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "aligned": False,
        "dispatched": False,
        "resolved": False,
        "outcome_result": None,
        "response_state": None,
        "effective_score": None,
        "confirmed_objective_high": False,
        "confirmed_objective_exceptional": False,
        "confirmed_objective_application": False,
        "panel_confirmed_declared_application_transition": False,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _child(label: str, outcome: SimpleNamespace, *, terminal: bool) -> SimpleNamespace:
    return SimpleNamespace(
        child_scientific_key=digest_json(["reporting-child", label]),
        work_unit_id=entity_id("wu", ["reporting-child", label]),
        target_action_id=entity_id("la", ["reporting-child", label]),
        execution_id=entity_id("ex", ["reporting-child", label]),
        candidate_id=entity_id("pc", ["reporting-child", label]),
        candidate_scientific_key=digest_json(
            ["reporting-child", label, "candidate"]
        ),
        observation_id=entity_id("to", ["reporting-child", label]),
        observation_scientific_key=digest_json(
            ["reporting-child", label, "observation"]
        ),
        panel_id=entity_id("pa", ["reporting-child", label]),
        terminal=terminal,
        outcome_review=(
            None
            if outcome.outcome_result is None
            else SimpleNamespace(
                review_item_id=entity_id("ri", ["reporting-child", label]),
                response_state=outcome.response_state,
            )
        ),
        outcome=outcome,
    )


class DemonstratedVerificationEvidenceViewTests(unittest.TestCase):
    def test_low_demonstrated_roots_cannot_support_a_high_ambiguity_block(self) -> None:
        state, action_ids = _verification_state(3)
        panels = (
            _panel(action_ids[0], confirmed_high=False, demonstrated=2),
            _panel(action_ids[1], confirmed_high=False, demonstrated=3),
            _panel(action_ids[2], confirmed_high=True, demonstrated=0),
        )
        state.panel_views = {panel.target_action_id: panel for panel in panels}

        self.assertEqual(
            RunState.demonstrated_repeat_supported_subject_ids(state),
            frozenset(),
        )

    def test_demonstrated_high_roots_still_use_the_exact_repeat_threshold(self) -> None:
        state, action_ids = _verification_state(5)
        three_of_five = tuple(
            _panel(
                action_id,
                confirmed_high=ordinal < 3,
                demonstrated=2 if ordinal < 3 else 0,
            )
            for ordinal, action_id in enumerate(action_ids)
        )
        four_of_five = tuple(
            _panel(
                action_id,
                confirmed_high=ordinal < 4,
                demonstrated=2 if ordinal < 4 else 0,
            )
            for ordinal, action_id in enumerate(action_ids)
        )

        state.panel_views = {
            panel.target_action_id: panel for panel in three_of_five
        }
        self.assertEqual(
            RunState.demonstrated_repeat_supported_subject_ids(state),
            frozenset(),
        )
        state.panel_views = {
            panel.target_action_id: panel for panel in four_of_five
        }
        self.assertEqual(
            RunState.demonstrated_repeat_supported_subject_ids(state),
            frozenset({state.work_units[0].unit["subject_id"]}),
        )


class PanelEvidenceViewTests(unittest.TestCase):
    def test_panel_view_owns_half_even_display_and_disputed_high_status(self) -> None:
        members = (
            _member("primary", "91", "judge-a"),
            _member("secondary_a", "65", "judge-b"),
        )
        details = aggregate_panel_details(
            entity_id("pn", ["reporting-panel-view"]),
            members[0].observation_id,
            members,
        )
        panel = PanelView(
            panel_id=details.assessment.panel_id,
            target_action_id=entity_id("la", ["reporting-panel-view"]),
            observation_id=members[0].observation_id,
            candidate_id=None,
            execution_id=None,
            target_event_id=None,
            triggered=True,
            primary=members[0],
            details=details,
            invalid_member_actions=(),
            member_validation_errors={},
            terminated=True,
        )

        self.assertEqual(panel.half_even_score_display, "80.600000")
        self.assertTrue(panel.disputed_high)


class StrategyExperimentReportingTests(unittest.TestCase):
    def test_recovery_summary_distinguishes_attempt_salvage_and_advance(self) -> None:
        recovery_episode_id = entity_id("st", ["reporting-recovery"])
        supported = _child(
            "recovery-supported",
            _outcome(
                aligned=True,
                dispatched=True,
                resolved=True,
                outcome_result="supports",
                effective_score=Decimal("25"),
            ),
            terminal=True,
        )
        recovery_episode = SimpleNamespace(
            strategy_id=recovery_episode_id,
            portfolio_intent=SimpleNamespace(value="recover"),
            children=(supported,),
            parent_baseline_score=Decimal("10"),
        )
        recovery_round = SimpleNamespace(
            work_unit_id=entity_id("wu", ["reporting-recovery-round"]),
            portfolio_intent=SimpleNamespace(value="recover"),
            lanes=(SimpleNamespace(),),
        )
        advance_parent_id = entity_id("cn", ["reporting-recovery-child"])
        advance_round = SimpleNamespace(
            work_unit_id=entity_id("wu", ["reporting-advance-round"]),
            portfolio_intent=SimpleNamespace(value="advance"),
            generation_parent_node_id=advance_parent_id,
        )
        scenario_id = entity_id("sc", ["reporting-recovery"])
        state = SimpleNamespace(
            portfolio_rounds={"recover": recovery_round, "advance": advance_round},
            work_units=(
                SimpleNamespace(
                    unit_id=recovery_round.work_unit_id,
                    unit={"design": {"maximum_strategy_count": 2}},
                ),
                SimpleNamespace(
                    unit_id=advance_round.work_unit_id,
                    unit={"design": {"maximum_strategy_count": 2}},
                ),
            ),
            strategy_episodes={recovery_episode_id: recovery_episode},
            conversation_nodes={
                advance_parent_id: SimpleNamespace(
                    candidate_id=supported.candidate_id
                )
            },
            scenarios={
                scenario_id: SimpleNamespace(
                    scenario_id=scenario_id,
                    origin="derived",
                    source_step_episode_ids=(recovery_episode_id,),
                )
            },
            scenario_evidence={
                scenario_id: SimpleNamespace(
                    verification_status=SimpleNamespace(value="verified")
                )
            },
        )

        summary = _strategy_search_summary(state)

        self.assertEqual(summary["recovery_portfolio_rounds"], 1)
        self.assertEqual(summary["recovery_distinct_approaches"], 1)
        self.assertEqual(summary["recovery_strategy_capacity"], 2)
        self.assertEqual(summary["recovery_unused_strategy_slots"], 1)
        self.assertEqual(summary["recovery_multi_strategy_portfolio_rounds"], 0)
        self.assertEqual(summary["recovery_supported_children"], 1)
        self.assertEqual(summary["recovery_resolved_children"], 1)
        self.assertEqual(summary["recovery_score_improved_children"], 1)
        self.assertEqual(summary["recovery_supported_score_improved_children"], 1)
        self.assertEqual(summary["best_recovery_score_delta"], "15")
        self.assertEqual(summary["salvaged_recovery_episodes"], 1)
        self.assertEqual(summary["recovery_to_advance_rounds"], 1)
        self.assertEqual(summary["verified_recovery_scenarios"], 1)

    def test_strategy_detail_uses_reducer_owned_success_evidence(self) -> None:
        successful = _child(
            "successful",
            _outcome(
                aligned=True,
                dispatched=True,
                resolved=True,
                outcome_result="supports",
                response_state="complete_commitment",
                confirmed_objective_high=True,
                confirmed_objective_exceptional=True,
                confirmed_objective_application=True,
                panel_confirmed_declared_application_transition=True,
            ),
            terminal=True,
        )
        unaligned = _child(
            "unaligned",
            _outcome(
                dispatched=True,
                resolved=True,
                confirmed_objective_high=True,
            ),
            terminal=True,
        )
        pending = _child(
            "pending",
            _outcome(aligned=True),
            terminal=False,
        )
        episode_id = entity_id("st", ["reporting-episode"])
        episode = SimpleNamespace(
            strategy_id=episode_id,
            diagnosis="missing_application",
            portfolio_intent=SimpleNamespace(value="advance"),
            children=(successful, unaligned, pending),
        )
        applicability = SimpleNamespace(
            diagnosis="missing_application",
            to_dict=lambda: {
                "diagnosis": "missing_application",
                "requires": "accepted premise without application",
                "excludes": "an unresolved objection",
            },
        )
        move_payload = {
            "act": "complete_commitment",
            "target_kind": "claim",
            "descriptors": {"argument_move": "derive_implication"},
            "blueprint": "derive the action-guiding consequence",
        }
        transition_payload = {
            "from": "abstract acceptance",
            "to": "action-governing rule",
        }
        template = SimpleNamespace(
            purpose="completion",
            applicability=applicability,
            move=SimpleNamespace(to_dict=lambda: dict(move_payload)),
            success=SimpleNamespace(
                transition=SimpleNamespace(
                    to_dict=lambda: dict(transition_payload)
                ),
                supports="the reply states the rule",
                falsifies="the reply rejects the implication",
            ),
        )
        template_id = entity_id("tp", ["reporting-template"])
        view = SimpleNamespace(
            template_id=template_id,
            template_scientific_key=digest_json(["reporting-template"]),
            template=template,
            source_episode_id=episode_id,
            tested_entry_context_ids=(),
            qualification_state=SimpleNamespace(value="eligible"),
            development_unit_id=None,
            holdout_unit_id=None,
            development_design=None,
            development_decision=None,
            holdout_decision=None,
            required_structured_role_scope_keys=(),
            required_role_scope_keys=(),
            evidence_event_ids=(entity_id("ev", ["reporting-template"]),),
        )
        state = SimpleNamespace(
            strategy_episodes={episode_id: episode},
            strategy_qualifications={template_id: view},
            logical_actions={},
        )

        experiment = _strategy_experiments(state)[0]

        self.assertEqual(experiment["purpose"], "completion")
        self.assertEqual(experiment["portfolio_intent"], "advance")
        self.assertEqual(experiment["diagnosis"], "missing_application")
        self.assertEqual(experiment["move"], move_payload)
        self.assertEqual(experiment["predicted_transition"], transition_payload)
        self.assertEqual(experiment["applicability"], applicability.to_dict())
        self.assertEqual(
            experiment["success_contract"],
            {
                "supports": "the reply states the rule",
                "falsifies": "the reply rejects the implication",
            },
        )
        self.assertEqual(
            experiment["episode_outcomes"]["aligned"]["allocated_count"],
            2,
        )
        self.assertEqual(
            experiment["episode_outcomes"]["aligned"]["pending_count"],
            1,
        )
        self.assertEqual(
            experiment["episode_outcomes"]["aligned"]["credited_count"],
            1,
        )
        self.assertEqual(
            experiment["episode_outcomes"]["unaligned"]["allocated_count"],
            1,
        )
        self.assertEqual(
            experiment["episode_outcomes"]["unaligned"][
                "confirmed_objective_high_count"
            ],
            1,
        )
        realization = experiment["successful_realizations"][0]
        self.assertEqual(
            realization["success_types"],
            [
                "supports",
                "confirmed_objective_high",
                "confirmed_objective_exceptional",
                "confirmed_objective_application",
                "panel_confirmed_declared_application_transition",
            ],
        )
        self.assertNotIn("changed_dimensions", realization)
        self.assertEqual(
            {
                realization["work_unit_id"],
                realization["target_action_id"],
                realization["execution_id"],
                realization["candidate_id"],
                realization["observation_id"],
                realization["panel_id"],
                realization["outcome_review_item_id"],
            },
            {
                successful.work_unit_id,
                successful.target_action_id,
                successful.execution_id,
                successful.candidate_id,
                successful.observation_id,
                successful.panel_id,
                successful.outcome_review.review_item_id,
            },
        )


class TypedQualificationReportingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metrics = QualificationArmMetrics(
            arm="treatment",
            planned_count=5,
            terminal_count=5,
            resolved_count=4,
            unresolved_count=1,
            pending_count=0,
            applicable_count=4,
            inapplicable_count=1,
            objective_successes=4,
            strategy_fidelity_successes=3,
            exceptional_count=1,
            effective_scores=(Decimal("70"), Decimal("90")),
            effective_score_median=Decimal("80"),
            applicability_rate=Decimal("0.8"),
            conditional_success_rate=Decimal("0.75"),
            unconditional_end_to_end_success_rate=Decimal("0.6"),
        )

    def test_typed_development_and_holdout_views_project_directly(self) -> None:
        gate = PurposeGateDecision(
            purpose="repair",
            supporting_evidence_satisfied=True,
            treatment_median_improves_each_control=True,
            passed=True,
        )
        development = DevelopmentBlockDecision(
            QualificationState.QUALIFIED,
            (self.metrics,),
            True,
            True,
            True,
            True,
            True,
            gate,
        )
        holdout = HoldoutBlockDecision(
            QualificationState.QUALIFIED,
            (self.metrics,),
            True,
            True,
            True,
            True,
            True,
        )

        development_view = _qualification_decision(
            development,
            include_purpose_gate=True,
        )
        holdout_view = _qualification_decision(
            holdout,
            include_purpose_gate=False,
        )

        assert development_view is not None and holdout_view is not None
        self.assertEqual(development_view["status"], "qualified")
        self.assertEqual(
            development_view["arms"][0]["effective_score_distribution"],
            ["70", "90"],
        )
        self.assertEqual(
            development_view["arms"][0]["conditional_success_rate"],
            "0.75",
        )
        self.assertEqual(development_view["purpose_gate"]["purpose"], "repair")
        self.assertNotIn("purpose_gate", holdout_view)

    def test_public_reporting_entry_points_have_no_alternate_state_overrides(self) -> None:
        self.assertEqual(
            tuple(signature(build_report_bundle).parameters),
            ("state",),
        )
        self.assertEqual(
            tuple(signature(write_derived_artifacts).parameters),
            ("run_directory", "state"),
        )
        self.assertEqual(tuple(signature(archive_projection).parameters), ("state",))

    def test_reporting_imports_only_domain_and_run_state_modules(self) -> None:
        source = (
            Path(__file__).resolve().parents[1] / "quasilisk" / "reporting.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        local_imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level == 1
        }
        self.assertEqual(local_imports, {"domain", "run_state"})


if __name__ == "__main__":
    unittest.main()
