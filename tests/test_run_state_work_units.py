from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from foundation_support import (
    call_fixture,
    chain_events,
    dispatch_payload,
    failure_payload,
    intended_payload,
    invocation_id,
    reduce_fixture_events as reduce_events,
    run_created_payload,
    success_payload,
)
from quasilisk.domain import digest_json, entity_id
from quasilisk.evidence import ExecutionEvidence
from quasilisk.events import JournalCorrupt
from quasilisk.run_state import _derive_evidence_and_archive


class WorkUnitReducerTests(unittest.TestCase):
    def test_screening_projects_exhausted_unusable_target_without_a_panel(self) -> None:
        run_id = entity_id("run", ["screening-unusable-target"])
        candidate_id = entity_id("pc", [run_id, "candidate"])
        candidate_key = digest_json(["screening-unusable-target", "candidate"])
        initial_unit_id = entity_id("wu", [run_id, "initial"])
        screening_unit_id = entity_id("wu", [run_id, "screening"])
        trigger_execution_id = entity_id("ex", [run_id, "trigger"])
        new_execution_ids = (
            entity_id("ex", [run_id, "new", 0]),
            entity_id("ex", [run_id, "new", 1]),
        )
        new_execution_keys = (
            digest_json(["screening-unusable-target", "new", 0]),
            digest_json(["screening-unusable-target", "new", 1]),
        )
        target_action_ids = tuple(
            entity_id("la", [run_id, "target", ordinal]) for ordinal in range(3)
        )

        def action(action_id: str, unit_id: str, owner_key: str) -> SimpleNamespace:
            return SimpleNamespace(
                action_id=action_id,
                unit_id=unit_id,
                slot={
                    "role": "target",
                    "owner_position_scientific_key": owner_key,
                },
            )

        initial_action = action(
            target_action_ids[0], initial_unit_id, digest_json(["initial-owner"])
        )
        screening_actions = (
            action(target_action_ids[1], screening_unit_id, new_execution_keys[0]),
            action(target_action_ids[2], screening_unit_id, new_execution_keys[1]),
        )
        work_units = (
            SimpleNamespace(
                unit_id=initial_unit_id,
                unit={"kind": "initial_observation", "subject_id": candidate_id},
                actions=(initial_action,),
            ),
            SimpleNamespace(
                unit_id=screening_unit_id,
                unit={
                    "kind": "screening",
                    "subject_id": candidate_id,
                    "design": {
                        "triggering_execution_id": trigger_execution_id,
                        "block_id": entity_id(
                            "sb", [run_id, candidate_key, "screening", 3]
                        ),
                    },
                    "execution_slots": [
                        {
                            "execution_id": execution_id,
                            "execution_scientific_key": execution_key,
                        }
                        for execution_id, execution_key in zip(
                            new_execution_ids, new_execution_keys, strict=True
                        )
                    ],
                },
                actions=screening_actions,
            ),
        )
        logical_actions = {
            item.action_id: item
            for item in (initial_action, *screening_actions)
        }
        panels = {
            entity_id("pn", [run_id, "panel", ordinal]): SimpleNamespace(
                panel_id=entity_id("pn", [run_id, "panel", ordinal]),
                candidate_id=candidate_id,
                target_action_id=target_action_ids[ordinal],
                execution_id=(
                    trigger_execution_id
                    if ordinal == 0
                    else new_execution_ids[ordinal - 1]
                ),
                observation_id=entity_id("to", [run_id, "observation", ordinal]),
            )
            for ordinal in (0, 1)
        }
        target_results = {
            target_action_ids[ordinal]: SimpleNamespace(
                terminal_sequence=ordinal + 1,
            )
            for ordinal in range(2)
        }
        execution_evidence = {
            trigger_execution_id: ExecutionEvidence(
                trigger_execution_id, True, True, primary_score=Decimal("40")
            ),
            new_execution_ids[0]: ExecutionEvidence(
                new_execution_ids[0], True, True, primary_score=Decimal("35")
            ),
            new_execution_ids[1]: ExecutionEvidence(
                new_execution_ids[1], True, False
            ),
        }
        candidate = {
            "scientific_key": candidate_key,
            "descriptors": {},
            "origin": "generated",
        }
        manifest = {
            "normalized_config": {
                "search": {"archive_capacity_per_cell": 1}
            },
            "descriptor_taxonomy": {"descriptors": []},
        }

        with patch(
            "quasilisk.run_state._candidate_metadata",
            return_value={candidate_id: candidate},
        ):
            _, screening_blocks, _ = _derive_evidence_and_archive(
                run_id=run_id,
                scope=digest_json(["screening-unusable-target", "scope"]),
                manifest=manifest,
                work_units=work_units,
                logical_actions=logical_actions,
                target_results=target_results,
                panel_views=panels,
                member_assessments={},
                execution_evidence=execution_evidence,
                conversation_nodes={},
                response_schemas={},
            )

        block = screening_blocks[candidate_id]
        self.assertEqual(block.target_action_ids, target_action_ids)
        self.assertEqual(len(block.observation_ids), 2)
        self.assertTrue(block.block_terminated)
        self.assertEqual(block.decision.resolved_count, 2)
        self.assertEqual(block.decision.unresolved_count, 1)

    def test_recomputes_and_rejects_tampered_worst_case_vector(self) -> None:
        fixture = call_fixture(label="bad-vector", role="target")
        fixture["unit"]["worst_case_calls"]["judge"] = 7
        events = chain_events(
            run_created_payload(label="bad-vector"),
            [("work_unit_started", {"unit": fixture["unit"]})],
        )
        with self.assertRaises(JournalCorrupt):
            reduce_events(events)

    def test_proves_complete_maximum_fits_at_start(self) -> None:
        fixture = call_fixture(label="not-admitted", role="target")
        events = chain_events(
            run_created_payload(label="not-admitted", target_budget=1),
            [("work_unit_started", {"unit": fixture["unit"]})],
        )
        with self.assertRaisesRegex(JournalCorrupt, "did not fit"):
            reduce_events(events)

    def test_rejects_new_unit_while_attempt_is_unsettled(self) -> None:
        first = call_fixture(label="overlap-first", role="target")
        second = call_fixture(label="overlap-second", role="target")
        events = chain_events(
            run_created_payload(label="overlap", target_budget=4),
            [
                ("work_unit_started", {"unit": first["unit"]}),
                ("model_call_intended", intended_payload(first)),
                ("work_unit_started", {"unit": second["unit"]}),
            ],
        )
        with self.assertRaisesRegex(JournalCorrupt, "unsettled"):
            reduce_events(events)

    def test_old_unit_cannot_receive_intent_after_next_start(self) -> None:
        first = call_fixture(label="old-first", role="target")
        second = call_fixture(label="old-second", role="target")
        old_retry = intended_payload(
            first,
            ordinal=1,
            parent_attempt_id=success_payload(first)["physical_attempt_id"],
        )
        events = chain_events(
            run_created_payload(label="old-unit", target_budget=4),
            [
                ("work_unit_started", {"unit": first["unit"]}),
                ("model_call_intended", intended_payload(first)),
                ("model_dispatch_started", dispatch_payload(first)),
                ("model_call_succeeded", success_payload(first)),
                ("work_unit_started", {"unit": second["unit"]}),
                ("model_call_intended", old_retry),
            ],
        )
        with self.assertRaisesRegex(JournalCorrupt, "outside the active"):
            reduce_events(events)

    def test_exposes_active_and_completed_unit_boundaries(self) -> None:
        first = call_fixture(
            label="facts-first", role="selector", maximum_attempts=1
        )
        second = call_fixture(
            label="facts-second", role="selector", maximum_attempts=1
        )
        events = chain_events(
            run_created_payload(label="facts", selector_budget=2),
            [
                ("work_unit_started", {"unit": first["unit"]}),
                ("model_call_intended", intended_payload(first)),
                ("model_dispatch_started", dispatch_payload(first)),
                ("model_call_succeeded", success_payload(first, text="{}")),
                ("work_unit_started", {"unit": second["unit"]}),
            ],
        )
        state = reduce_events(events)
        self.assertEqual(state.active_work_unit_id, second["unit_id"])
        self.assertEqual(state.active_work_unit.unit_id, second["unit_id"])
        self.assertEqual(state.completed_work_unit_ids, (first["unit_id"],))
        self.assertEqual(
            tuple(unit.unit_id for unit in state.completed_work_units),
            (first["unit_id"],),
        )

    def test_rejects_new_unit_after_skipping_an_always_action(self) -> None:
        first = call_fixture(
            label="skipped-always-first", role="selector", maximum_attempts=1
        )
        second = call_fixture(
            label="skipped-always-second", role="selector", maximum_attempts=1
        )
        events = chain_events(
            run_created_payload(label="skipped-always", selector_budget=2),
            [
                ("work_unit_started", {"unit": first["unit"]}),
                ("work_unit_started", {"unit": second["unit"]}),
            ],
        )
        with self.assertRaisesRegex(JournalCorrupt, "action graph completed"):
            reduce_events(events)

    def test_rejects_terminal_that_claims_an_incomplete_active_unit(self) -> None:
        fixture = call_fixture(
            label="false-terminal-completion",
            role="selector",
            maximum_attempts=1,
        )
        events = chain_events(
            run_created_payload(label="false-terminal-completion", selector_budget=1),
            [
                ("work_unit_started", {"unit": fixture["unit"]}),
                (
                    "run_terminated",
                    {
                        "invocation_id": invocation_id(
                            "false-terminal-completion-creation"
                        ),
                        "reason": "completed_plan",
                        "last_completed_work_unit_id": fixture["unit_id"],
                        "resumable": False,
                        "detail": "fabricated completion",
                    },
                ),
            ],
        )
        with self.assertRaisesRegex(JournalCorrupt, "last completed"):
            reduce_events(events)

    def test_rejects_attempt_recorded_under_a_false_guard(self) -> None:
        first = call_fixture(
            label="false-guard-first", role="selector", maximum_attempts=1
        )
        guarded = call_fixture(
            label="false-guard-second", role="selector", maximum_attempts=1
        )
        guarded["unit_id"] = first["unit_id"]
        guarded_slot = guarded["unit"]["logical_action_slots"][0]
        guarded_slot["ordinal"] = 1
        guarded["action_key"] = digest_json(
            [
                guarded["owner_key"],
                "action",
                "selector",
                "selector_portfolio",
                1,
            ]
        )
        guarded_slot["action_scientific_key"] = guarded["action_key"]
        guarded_slot["dependency_action_ids"] = [first["action_id"]]
        guarded_slot["guard"] = {
            "kind": "if_action_usable",
            "action_id": first["action_id"],
        }
        first["unit"]["logical_action_slots"].append(guarded_slot)
        first["unit"]["worst_case_calls"]["selector"] = 2

        events = chain_events(
            run_created_payload(label="false-guard", selector_budget=2),
            [
                ("work_unit_started", {"unit": first["unit"]}),
                ("model_call_intended", intended_payload(first)),
                ("model_dispatch_started", dispatch_payload(first)),
                ("model_call_failed", failure_payload(first)),
                ("model_call_intended", intended_payload(guarded)),
                ("model_dispatch_started", dispatch_payload(guarded)),
                ("model_call_succeeded", success_payload(guarded, text="{}")),
            ],
        )
        with self.assertRaisesRegex(JournalCorrupt, "false frozen guard"):
            reduce_events(events)


if __name__ == "__main__":
    unittest.main()
