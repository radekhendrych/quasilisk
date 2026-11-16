"""Engine tests for the simplified Selector portfolio path."""

from __future__ import annotations

from foundation_support import selector_applicability_payload

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from integration_support import dryrun_config
from quasilisk.domain import ActionPurpose, PortfolioIntent, entity_id, scientific_key
from quasilisk.engine import (
    RunContext,
    _initialize_routes,
    _next_route_plan,
    _portfolio_result,
    _portfolio_selector_request,
    _portfolio_strategy_selection_limit,
    load_resources,
)
from quasilisk.run_state import AttemptView, LogicalActionView


class SimplifiedSelectorEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        resources = load_resources(dryrun_config("fatalism", root / "runs"))
        self.context = RunContext(
            resources=resources,
            run_id=entity_id("run", ["simplified-selector-engine"]),
            run_nonce="0" * 32,
            invocation_id=entity_id("iv", ["simplified-selector-engine"]),
            run_directory=root / "run",
            candidates={},
            routes={},
        )
        _initialize_routes(self.context)
        self.turn_ref = scientific_key(["selector", "target-turn"])
        self.turn_text = "I accept the opening, but this objection remains."
        self.unit = SimpleNamespace(
            unit_id=entity_id("wu", ["selector-unit"]),
            unit={
                "kind": "portfolio_round",
                "unit_scientific_key": scientific_key(["selector-unit"]),
                "subject_scientific_key": scientific_key(["selector-subject"]),
                "design": {
                    "round_index": 0,
                    "proposal_count": 6,
                    "maximum_strategy_count": 2,
                    "portfolio_intent": PortfolioIntent.ADVANCE.value,
                },
            },
            actions=(),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _parent_patches(self):
        return (
            patch(
                "quasilisk.engine._portfolio_parent",
                return_value=(SimpleNamespace(), object(), scientific_key(["parent"])),
            ),
            patch(
                "quasilisk.engine._portfolio_candidate_ancestry",
                return_value=SimpleNamespace(linked_steps=()),
            ),
        )

    def _portfolio(self, *, act: str = "overcome_objection") -> dict[str, object]:
        return {
            "items": [
                {
                    "diagnosis": {
                        "category": "counterargument",
                        "summary": "The target retains one explicit objection.",
                        "evidence": [{"evidence_id": "e0"}],
                    },
                    "contract": {
                        "basis": [
                            {
                                "use": "continuation_anchor",
                                "evidence_id": "e0",
                                "reading": "The target supplied a forward opening.",
                            },
                            {
                                "use": "judge_concern",
                                "evidence_id": "e0",
                                "reading": "The target states the obstacle to overcome.",
                            },
                        ],
                        "move": {
                            "operation": act,
                            "target": {
                                "kind": "inference",
                                "evidence_id": "e0",
                                "summary": "the remaining objection",
                            },
                            "descriptors": {
                                "objective_route": "causal_fixity",
                                "application_scope": "abstract",
                            },
                            "blueprint": "Answer the cited objection while holding the opening fixed.",
                        },
                        "hypothesis": "Answering the exact objection may extend the opening.",
                        "preserve": [
                            {
                                "kind": "commitment",
                                "statement": "Preserve the cited opening.",
                                "evidence_id": "e0",
                            }
                        ],
                        "vary": [
                            {
                                "dimension": "wording",
                                "instruction": "Vary wording without changing direction.",
                            }
                        ],
                        "success": {
                            "transition": {"from": "opening", "to": "extended commitment"},
                            "supports": "The target owns the extended commitment.",
                            "falsifies": "The target retains the objection.",
                        },
                    },
                    "applicability": selector_applicability_payload(None),
                }
            ]
        }

    def _attempt(self, label: str, value: dict[str, object]) -> AttemptView:
        return AttemptView(
            entity_id("pa", ["selector", label]),
            {
                "payload": {
                    "role_route_key": scientific_key(["selector-route"]),
                    "response_mode": "json_schema",
                }
            },
            {"type": "model_call_dispatched"},
            {
                "type": "model_call_succeeded",
                "payload": {"finish_status": "complete", "text": json.dumps(value)},
            },
        )

    def _action(self, *attempts: AttemptView) -> LogicalActionView:
        return LogicalActionView(
            entity_id("la", [self.unit.unit_id, "selector"]),
            self.unit.unit_id,
            {
                "role": "selector",
                "purpose": ActionPurpose.SELECTOR_PORTFOLIO.value,
                "routes": [
                    {
                        "route_key": scientific_key(["selector-route"]),
                        "route_label": "primary",
                        "maximum_attempts": 2,
                    }
                ],
            },
            tuple(attempts),
        )

    def _result(self, action: LogicalActionView):
        with (
            patch(
                "quasilisk.engine._portfolio_branch_context",
                return_value=(
                    [],
                    [{"turn_ref": self.turn_ref, "text": self.turn_text}],
                ),
            ),
            patch(
                "quasilisk.engine._portfolio_strategy_evidence",
                return_value=(
                    [],
                    {
                        (self.turn_ref, self.turn_text): frozenset(
                            {"supports", "resists"}
                        ),
                    },
                ),
            ),
            patch(
                "quasilisk.engine._portfolio_strategy_selection_limit",
                return_value=1,
            ),
        ):
            return _portfolio_result(
                self.context, SimpleNamespace(work_units=()), self.unit, action
            )

    def test_strategy_limit_uses_frozen_capacity_for_both_intents(self) -> None:
        with self._parent_patches()[0], self._parent_patches()[1]:
            self.assertEqual(
                _portfolio_strategy_selection_limit(
                    self.context, SimpleNamespace(work_units=()), self.unit
                ),
                2,
            )
        self.unit.unit["design"]["portfolio_intent"] = PortfolioIntent.RECOVER.value
        with self._parent_patches()[0], self._parent_patches()[1]:
            self.assertEqual(
                _portfolio_strategy_selection_limit(
                    self.context, SimpleNamespace(work_units=()), self.unit
                ),
                2,
            )

    def test_request_contains_goal_branch_and_directional_evidence_only(self) -> None:
        evidence = [
            {
                "evidence_id": "e0",
                "text": self.turn_text,
                "polarities": ["resists", "supports"],
            }
        ]
        assessment = {
            "verdict": "demonstrated",
            "application": "abstract",
         "ownership": "adopted"}
        patches = self._parent_patches()
        with (
            patches[0],
            patches[1],
            patch("quasilisk.engine._generation_brief", return_value={"next_transition": "repair_inference"}),
            patch(
                "quasilisk.engine._portfolio_branch_context",
                return_value=(
                    [{"role": "assistant", "content": self.turn_text}],
                    [{"turn_ref": self.turn_ref, "text": self.turn_text}],
                ),
            ),
            patch(
                "quasilisk.engine._portfolio_strategy_evidence",
                return_value=(evidence, {}),
            ),
            patch(
                "quasilisk.engine._portfolio_current_assessment",
                return_value=assessment,
            ),
        ):
            request = _portfolio_selector_request(
                self.context, SimpleNamespace(work_units=()), self.unit
            )
        self.assertEqual(
            set(request),
            {
                "objective_definition",
                "descriptor_vocabulary",
                "diagnosis_categories",
                "portfolio_intent",
                "maximum_strategies",
                "conversation",
                "evidence_catalog",
                "current_assessment",
             "finisher_strategies", "prior_attempts", "generation_brief"},
        )
        self.assertEqual(request["evidence_catalog"], evidence)
        self.assertEqual(request["portfolio_intent"], "advance")
        self.assertEqual(request["current_assessment"], assessment)
        self.assertEqual(
            request["objective_definition"],
            {
                "objective_id": self.context.objective.id,
                "strategy_goal": self.context.objective.strategy_goal,
            },
        )
        serialized = json.dumps(request, sort_keys=True).lower()
        for removed in ("strategy_catalog", "recommended_strategy", "selection_lane", "score"):
            self.assertNotIn(removed, serialized)

    def test_old_direction_ambiguous_operator_is_rejected_then_repairable(self) -> None:
        invalid = self._action(
            self._attempt("old-operator", self._portfolio(act="challenge_inference"))
        )
        result = self._result(invalid)
        self.assertFalse(result.usable)
        self.assertIsNotNone(_next_route_plan(invalid, result))
        self.assertEqual(_next_route_plan(invalid, result)[1], "schema_repair")

        repaired = self._action(
            invalid.attempts[0],
            self._attempt("forward-operator", self._portfolio()),
        )
        recovered = self._result(repaired)
        self.assertTrue(recovered.usable)
        self.assertEqual(recovered.value.items[0].contract.move.act, "overcome_objection")


if __name__ == "__main__":
    unittest.main()
