"""Semantic admission checks for the one closed WorkUnit contract."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from integration_support import dryrun_config
from quasilisk.domain import PortfolioIntent, RoleCallVector, entity_id, scientific_key
from quasilisk.engine import (
    RunContext,
    _build_candidates,
    _initialize_routes,
    build_initial_observation,
    build_map_exploration,
    build_portfolio_round,
    build_screening,
    build_strategy_development,
    build_strategy_holdout,
    build_verification,
    load_resources,
)
from quasilisk.run_state import SchedulerValidationError, validate_work_unit_maximum
from quasilisk.scheduler import validate_work_unit_maximum as admission_validator
from quasilisk.strategy import parse_strategy_template


class WorkUnitConformanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        resources = load_resources(dryrun_config("fatalism", root / "runs"))
        self.context = RunContext(
            resources=resources,
            run_id=entity_id("run", ["scheduler-conformance"]),
            run_nonce="0" * 32,
            invocation_id=entity_id("iv", ["scheduler-conformance"]),
            run_directory=root / "run",
            candidates={},
            routes={},
        )
        _initialize_routes(self.context)
        self.context.candidates.update(_build_candidates(resources, self.context.run_id))
        self.candidate = next(iter(self.context.candidates.values()))
        self.parent_id = entity_id("wu", ["scheduler-conformance-parent"])
        self.template = parse_strategy_template(
            {
                "purpose": "repair",
                "applicability": {
                    "diagnosis": "conditional_acceptance",
                    "requires": "A stated condition remains unresolved.",
                    "excludes": "No stated condition remains.",
                },
                "move": {
                    "act": "repair_grounded_gap",
                    "target_kind": "condition",
                    "descriptors": {
                        "objective_route": "causal_fixity",
                        "argument_move": "answer_objection",
                        "application_scope": "reasoning_policy",
                    },
                    "blueprint": "Resolve the condition without changing the premise.",
                },
                "hypothesis": "Resolving the condition should permit progress.",
                "preserve": [
                    {"kind": "meaning", "statement": "Preserve the accepted meaning."}
                ],
                "vary": [
                    {
                        "dimension": "framing",
                        "instruction": "Vary only this declared dimension.",
                    }
                ],
                "success": {
                    "transition": {"from": "conditional", "to": "resolved"},
                    "supports": "The target resolves its own stated condition.",
                    "falsifies": "The target retains the condition.",
                },
            },
            self.context.objective.descriptor_values,
            self.context.objective.diagnosis_categories,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_scheduler_uses_the_reducer_owned_validator(self) -> None:
        self.assertIs(admission_validator, validate_work_unit_maximum)

    def _units(self):
        initial = build_initial_observation(self.context, self.candidate)
        development = build_strategy_development(
            self.context, self.template, parent_unit_id=self.parent_id
        )
        return (
            initial,
            build_screening(
                self.context,
                self.candidate,
                initial.design["execution_id"],
                initial.design["execution_scientific_key"],
                parent_unit_id=initial.unit_id,
            ),
            build_map_exploration(
                self.context, self.candidate, parent_unit_id=initial.unit_id
            ),
            build_portfolio_round(
                self.context,
                self.candidate,
                parent_unit_id=initial.unit_id,
                conversation_root_id=entity_id("cr", ["scheduler-conformance"]),
                conversation_root_scientific_key=scientific_key(
                    ["scheduler-conformance-root"]
                ),
                parent_node_id=entity_id("cn", ["scheduler-conformance"]),
                parent_node_scientific_key=scientific_key(
                    ["scheduler-conformance-node"]
                ),
                portfolio_intent=PortfolioIntent.ADVANCE,
            ),
            build_verification(
                self.context, self.candidate, parent_unit_id=initial.unit_id
            ),
            development,
            build_strategy_holdout(
                self.context,
                self.template,
                source_development_unit=development.to_dict(),
            ),
        )

    def test_every_closed_kind_passes_the_same_semantic_validator(self) -> None:
        units = self._units()
        self.assertEqual({unit.kind for unit in units}, {
            "initial_observation",
            "screening",
            "map_exploration",
            "portfolio_round",
            "verification",
            "strategy_development",
            "strategy_holdout",
        })
        for unit in units:
            with self.subTest(kind=unit.kind):
                self.assertEqual(validate_work_unit_maximum(unit), unit.worst_case_calls)

    def test_rejects_identity_route_guard_panel_and_design_drift(self) -> None:
        units = {unit.kind: unit for unit in self._units()}

        map_unit = units["map_exploration"]
        proposals = deepcopy(map_unit.proposal_slots)
        proposals[0]["scientific_key"] = scientific_key(["wrong-proposal"])
        with self.assertRaisesRegex(SchedulerValidationError, "proposal slot identity"):
            validate_work_unit_maximum(replace(map_unit, proposal_slots=tuple(proposals)))

        actions = deepcopy(map_unit.logical_action_slots)
        actions[0]["routes"].reverse()
        with self.assertRaisesRegex(SchedulerValidationError, "configured sequence"):
            validate_work_unit_maximum(replace(map_unit, logical_action_slots=tuple(actions)))

        actions = deepcopy(map_unit.logical_action_slots)
        actions[1]["guard"]["proposal_action_ids"] = [actions[-1]["action_id"]]
        with self.assertRaisesRegex(SchedulerValidationError, "guard"):
            validate_work_unit_maximum(replace(map_unit, logical_action_slots=tuple(actions)))

        initial = units["initial_observation"]
        panels = deepcopy(initial.panel_plans)
        panels[0]["normal_members"][0]["weight"] = 2
        with self.assertRaisesRegex(SchedulerValidationError, "order or weight"):
            validate_work_unit_maximum(replace(initial, panel_plans=tuple(panels)))

        portfolio = units["portfolio_round"]
        design = deepcopy(portfolio.design)
        design["maximum_strategy_count"] += 1
        with self.assertRaisesRegex(SchedulerValidationError, "floor\(M/3\)"):
            validate_work_unit_maximum(replace(portfolio, design=design))

    def test_rejects_root_topology_and_holdout_seal_drift(self) -> None:
        units = {unit.kind: unit for unit in self._units()}
        development = units["strategy_development"]
        design = deepcopy(development.design)
        design["root_plans"][0]["ordered_action_ids"].reverse()
        with self.assertRaisesRegex(SchedulerValidationError, "root-plan action"):
            validate_work_unit_maximum(replace(development, design=design))

        holdout = units["strategy_holdout"]
        design = deepcopy(holdout.design)
        design["sealed_holdout_plan"]["future_roots"][0]["arm"] = "strategy_agnostic"
        with self.assertRaises(SchedulerValidationError):
            validate_work_unit_maximum(replace(holdout, design=design))

        with self.assertRaisesRegex(SchedulerValidationError, "recorded work-unit maximum"):
            validate_work_unit_maximum(
                replace(holdout, worst_case_calls=RoleCallVector())
            )


if __name__ == "__main__":
    unittest.main()
