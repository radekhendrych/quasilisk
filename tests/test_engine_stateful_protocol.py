from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from integration_support import dryrun_config
from quasilisk.archive import ArchiveSnapshot
from quasilisk.domain import (
    ActionPurpose,
    ChatMessage,
    ConversationNode,
    FinishStatus,
    ProposalAssignment,
    ProposalSlot,
    Role,
    RoleCallVector,
    TargetObservation,
    entity_id,
    scientific_key,
)
from quasilisk.engine import (
    CandidatePlan,
    RunContext,
    _messages_for_action,
    _portfolio_branch_context,
    _portfolio_outcome_result,
    _portfolio_outcome_review_request,
    _portfolio_selector_request,
    _portfolio_stateful_dispatch_status,
    _stateful_candidate_scientific_key,
    load_resources,
)
from quasilisk.perturbation import make_review_item_id


def _delimited_json(text: str, marker: str) -> dict[str, object]:
    opening = f"<{marker}>"
    closing = f"</{marker}>"
    start = text.index(opening) + len(opening)
    end = text.index(closing, start)
    return json.loads(text[start:end].strip())


class StatefulProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        resources = load_resources(dryrun_config("fatalism", root / "runs"))
        self.context = RunContext(
            resources=resources,
            run_id=entity_id("run", ["stateful-protocol"]),
            run_nonce="0" * 32,
            invocation_id=entity_id("iv", ["stateful-protocol"]),
            run_directory=root / "run",
            candidates={},
            routes={},
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _branch(self):
        root_id = entity_id("cr", ["stateful-root"])
        root_key = scientific_key(["stateful-root"])
        first_node_id = entity_id("cn", ["first"])
        first_node_key = scientific_key(["first-node"])
        source_strategy_id = entity_id("st", ["first-to-second"])
        source_strategy_key = scientific_key(["first-to-second"])
        first = CandidatePlan(
            entity_id("pc", ["first"]),
            scientific_key(["first"]),
            "first user turn",
            {
                "objective_route": "causal_fixity",
                "argument_move": "test_inference",
                "application_scope": "abstract",
            },
            "bootstrap",
            None,
        )
        second = CandidatePlan(
            entity_id("pc", ["second"]),
            scientific_key(["second"]),
            "second user turn",
            {
                "objective_route": "causal_fixity",
                "argument_move": "answer_objection",
                "application_scope": "abstract",
            },
            "stateful_strategy",
            None,
            parent_node_id=first_node_id,
            parent_node_scientific_key=first_node_key,
            strategy_id=source_strategy_id,
        )
        first_observation = TargetObservation(
            entity_id("to", ["first"]),
            scientific_key(["first-observation"]),
            first.candidate_id,
            None,
            (ChatMessage("user", first.prompt),),
            scientific_key(["target-profile"]),
            "first target reply",
            FinishStatus.COMPLETE,
            entity_id("pa", ["first"]),
        )
        first_node = ConversationNode(
            first_node_id,
            first_node_key,
            root_id,
            None,
            0,
            first.candidate_id,
            first_observation.observation_id,
        )
        second_observation = TargetObservation(
            entity_id("to", ["second"]),
            scientific_key(["second-observation"]),
            second.candidate_id,
            first_node.node_id,
            (
                ChatMessage("user", first.prompt),
                ChatMessage("assistant", first_observation.response),
                ChatMessage("user", second.prompt),
            ),
            scientific_key(["target-profile"]),
            "second target reply",
            FinishStatus.COMPLETE,
            entity_id("pa", ["second"]),
        )
        second_node = ConversationNode(
            entity_id("cn", ["second"]),
            scientific_key(["second-node"]),
            root_id,
            first_node.node_id,
            1,
            second.candidate_id,
            second_observation.observation_id,
        )
        self.context.candidates.update(
            {first.scientific_key: first, second.scientific_key: second}
        )
        unit_key = scientific_key(["portfolio-unit"])
        unit_id = entity_id("wu", ["portfolio-unit"])
        unit = SimpleNamespace(
            unit_id=unit_id,
            start_event={"sequence": 40},
            unit={
                "kind": "portfolio_round",
                "unit_scientific_key": unit_key,
                "subject_scientific_key": root_key,
                "design": {
                    "conversation_root_id": root_id,
                    "generation_parent_node_id": second_node.node_id,
                    "generation_parent_node_scientific_key": second_node.scientific_key,
                    "proposal_count": 6,
                    "maximum_strategy_count": 2,
                    "round_index": 0,
                    "portfolio_intent": "advance",
                },
                "proposal_slots": [],
                "panel_plans": [],
            },
            actions=(),
        )
        state = SimpleNamespace(
            conversation_nodes={
                first_node.node_id: first_node,
                second_node.node_id: second_node,
            },
            conversation_roots={
                root_id: SimpleNamespace(scientific_key=root_key)
            },
            target_observations={
                "first-action": first_observation,
                "second-action": second_observation,
            },
            # This unrelated turn must never enter the selected branch request.
            target_results={
                "unrelated": SimpleNamespace(text="UNRELATED SECRET TURN")
            },
            panel_views={},
            archive=ArchiveSnapshot(2, {}),
            remaining=RoleCallVector(20, 40, 12, 8),
            work_units=(),
            portfolio_rounds={
                unit_id: SimpleNamespace(
                    round_index=0,
                    complete=False,
                    lanes=(),
                )
            },
            current_strategy_qualification_snapshots=(),
            strategy_episodes={
                source_strategy_id: SimpleNamespace(
                    strategy_id=source_strategy_id,
                    strategy_scientific_key=source_strategy_key,
                    conversation_root_id=root_id,
                    conversation_root_scientific_key=root_key,
                    generation_parent_node_id=first_node.node_id,
                    generation_parent_node_scientific_key=first_node.scientific_key,
                    contract=SimpleNamespace(
                        to_dict=lambda: {"purpose": "repair"}
                    ),
                    children=(
                        SimpleNamespace(
                            candidate_id=second.candidate_id,
                            candidate_scientific_key=second.scientific_key,
                            strategy_id=source_strategy_id,
                            strategy_scientific_key=source_strategy_key,
                            alignment_status="aligned",
                            observation_id=second_observation.observation_id,
                            outcome=SimpleNamespace(
                                aligned=True,
                                resolved=True,
                                outcome_result="supports",
                            ),
                        ),
                    ),
                )
            },
        )
        unit.start_event = {"sequence": 10}
        state.member_assessments = {}
        state.work_units = (unit,)
        return unit, state, first, second

    def test_selector_and_target_use_only_complete_exact_branch(self) -> None:
        unit, state, first, second = self._branch()
        conversation, anchors = _portfolio_branch_context(
            self.context, state, unit
        )
        self.assertEqual(
            conversation,
            [
                {"role": "user", "content": first.prompt},
                {"role": "assistant", "content": "first target reply"},
                {"role": "user", "content": second.prompt},
                {"role": "assistant", "content": "second target reply"},
            ],
        )
        self.assertEqual(len(anchors), 2)

        with patch(
            "quasilisk.engine._portfolio_strategy_selection_limit",
            return_value=unit.unit["design"]["maximum_strategy_count"],
        ):
            request = _portfolio_selector_request(self.context, state, unit)
        serialized = json.dumps(request, sort_keys=True)
        self.assertNotIn("UNRELATED SECRET TURN", serialized)
        self.assertEqual(
            request["objective_definition"],
            {
                "objective_id": self.context.objective.id,
                "strategy_goal": self.context.objective.strategy_goal,
            },
        )
        self.assertEqual(request["conversation"], conversation)
        self.assertNotIn("target_response_anchors", request)
        self.assertEqual(
            request["evidence_catalog"],
            [
                {
                    "evidence_id": "e0",
                    "text": "first target reply",
                    "polarities": [],
                },
                {
                    "evidence_id": "e1",
                    "text": "second target reply",
                    "polarities": [],
                },
            ],
        )
        self.assertIsNone(request["current_assessment"])
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
                "linked_parent_strategy",
             "finisher_strategies", "prior_attempts", "generation_brief"},
        )
        self.assertEqual(request["portfolio_intent"], "advance")
        self.assertNotIn("archive", serialized.lower())
        self.assertNotIn("budget", serialized.lower())

        child = CandidatePlan(
            entity_id("pc", ["child"]),
            scientific_key(["child"]),
            "third user turn",
            dict(second.descriptors),
            "stateful_strategy",
            None,
        )
        target_action = SimpleNamespace(
            action_id=entity_id("la", ["child-target"]),
            slot={
                "role": Role.TARGET.value,
                "purpose": ActionPurpose.TARGET_SAMPLE.value,
            },
            attempts=(),
        )
        with patch(
            "quasilisk.engine._candidate_for_target_action", return_value=child
        ):
            messages = _messages_for_action(
                self.context, state, unit, target_action
            )
        visible = [
            {"role": item.role, "content": item.content}
            for item in messages
            if item.role != "system"
        ]
        self.assertEqual(
            visible,
            [*conversation, {"role": "user", "content": child.prompt}],
        )

    def test_perturbator_request_has_three_slots_and_no_correction_protocol(self) -> None:
        unit, state, _, _ = self._branch()
        unit_key = unit.unit["unit_scientific_key"]
        slots = tuple(
            ProposalSlot(
                entity_id("ps", [unit_key, ordinal]),
                scientific_key([unit_key, "slot", ordinal]),
                ordinal,
            )
            for ordinal in range(6)
        )
        unit.unit["proposal_slots"] = [item.to_dict() for item in slots]
        strategy_key = scientific_key([unit_key, "strategy", 0])
        proposal_action = SimpleNamespace(
            action_id=entity_id("la", [unit_key, "proposal"]),
            slot={
                "role": Role.PERTURBATOR.value,
                "purpose": ActionPurpose.PERTURBATOR_PROPOSAL.value,
                "ordinal": 1,
                "action_scientific_key": scientific_key(
                    [unit_key, "proposal"]
                ),
                "owner_position_scientific_key": strategy_key,
            },
            attempts=(),
        )
        unit.actions = (proposal_action,)
        assignments = tuple(
            ProposalAssignment(
                slot.scientific_key,
                strategy_key if slot.global_ordinal < 3 else scientific_key(
                    [unit_key, "strategy", 1]
                ),
                slot.global_ordinal % 3,
                ("anchor", "close", "divergent")[slot.global_ordinal % 3],
            )
            for slot in slots
        )
        descriptors = (
            ("objective_route", "causal_fixity"),
            ("argument_move", "answer_objection"),
            ("application_scope", "abstract"),
        )
        contract = SimpleNamespace(
            vary=(SimpleNamespace(dimension="wording"),),
            move=SimpleNamespace(descriptors=descriptors),
            to_dict=lambda: {"purpose": "repair"},
        )
        lane = SimpleNamespace(
            strategy_slot=SimpleNamespace(scientific_key=strategy_key),
            item=SimpleNamespace(contract=contract),
        )
        with (
            patch("quasilisk.engine._portfolio_lane_for_action", return_value=lane),
            patch(
                "quasilisk.engine._portfolio_slots_for_lane",
                return_value=slots[:3],
            ),
            patch(
                "quasilisk.engine._portfolio_assignments",
                return_value=assignments,
            ),
        ):
            messages = _messages_for_action(
                self.context, state, unit, proposal_action
            )
        request = _delimited_json(
            messages[-1].content, "UNTRUSTED_PERTURBATOR_REQUEST_JSON"
        )
        self.assertEqual(
            request["proposal_slots"],
            [
                {"slot": "s0", "intent": "anchor"},
                {"slot": "s1", "intent": "close"},
                {"slot": "s2", "intent": "divergent"},
            ],
        )
        self.assertNotIn("fixed_output_fields", request)
        self.assertEqual(
            request["objective_definition"],
            {
                "objective_id": self.context.objective.id,
                "strategy_goal": self.context.objective.strategy_goal,
            },
        )
        self.assertEqual(request["allowed_dimensions"], ["wording"])
        self.assertNotIn("correction_reviews", request)
        self.assertNotIn("assignments", request)
        self.assertNotIn("work_unit", request)
        self.assertEqual(len(request["conversation"]), 4)

    def test_stateful_duplicate_owner_and_descriptor_conflicts_close_predispatch(self) -> None:
        unit_key = scientific_key(["stateful-dedup-unit"])
        parent_key = scientific_key(["stateful-dedup-parent"])
        actions = tuple(
            SimpleNamespace(
                action_id=entity_id("la", [unit_key, ordinal]),
                slot={"role": Role.TARGET.value},
            )
            for ordinal in range(2)
        )
        slots = tuple(
            ProposalSlot(
                entity_id("ps", [unit_key, ordinal]),
                scientific_key([unit_key, "slot", ordinal]),
                ordinal,
            )
            for ordinal in range(2)
        )
        unit = SimpleNamespace(
            unit={
                "design": {
                    "generation_parent_node_scientific_key": parent_key
                }
            },
            actions=actions,
        )
        descriptors = (
            ("objective_route", "causal_fixity"),
            ("argument_move", "answer_objection"),
            ("application_scope", "abstract"),
        )

        def selection(slot, reviewed=descriptors):
            return (
                slot,
                SimpleNamespace(),
                SimpleNamespace(
                    proposal=SimpleNamespace(user_turn="identical next turn"),
                    review=SimpleNamespace(
                        status="aligned", reviewed_descriptors=reviewed
                    ),
                ),
            )

        selected = {
            actions[0].action_id: selection(slots[0]),
            actions[1].action_id: selection(slots[1]),
        }
        with patch(
            "quasilisk.engine._portfolio_selected_realization",
            side_effect=lambda context, state, unit, action: selected[
                action.action_id
            ],
        ):
            statuses = {
                action.action_id: _portfolio_stateful_dispatch_status(
                    self.context, SimpleNamespace(), unit, action
                )
                for action in actions
            }
        owner_action = actions[
            0 if slots[0].scientific_key < slots[1].scientific_key else 1
        ]
        duplicate_action = actions[
            1 if owner_action is actions[0] else 0
        ]
        self.assertEqual(statuses[owner_action.action_id], "dispatch")
        self.assertEqual(
            statuses[duplicate_action.action_id], "duplicate_candidate"
        )

        conflicting = (
            ("objective_route", "absent_alternatives"),
            ("argument_move", "answer_objection"),
            ("application_scope", "abstract"),
        )
        selected[actions[1].action_id] = selection(slots[1], conflicting)
        with patch(
            "quasilisk.engine._portfolio_selected_realization",
            side_effect=lambda context, state, unit, action: selected[
                action.action_id
            ],
        ):
            self.assertEqual(
                _portfolio_stateful_dispatch_status(
                    self.context, SimpleNamespace(), unit, actions[0]
                ),
                "descriptor_conflict",
            )

        selected[actions[1].action_id] = selection(slots[1])
        candidate_key = _stateful_candidate_scientific_key(
            self.context, parent_key, "identical next turn"
        )
        self.context.candidates[candidate_key] = CandidatePlan(
            entity_id("pc", [candidate_key]),
            candidate_key,
            "identical next turn",
            dict(descriptors),
            "stateful_strategy",
            None,
            parent_node_id=entity_id("cn", ["same-parent"]),
            parent_node_scientific_key=parent_key,
            proposal_slot_scientific_key=scientific_key(["prior-slot"]),
        )
        with patch(
            "quasilisk.engine._portfolio_selected_realization",
            side_effect=lambda context, state, unit, action: selected[
                action.action_id
            ],
        ):
            self.assertEqual(
                _portfolio_stateful_dispatch_status(
                    self.context, SimpleNamespace(), unit, owner_action
                ),
                "duplicate_candidate",
            )

    def test_outcome_request_excludes_unaligned_and_unresolved_children(self) -> None:
        unit_key = scientific_key(["outcome-unit"])
        before = SimpleNamespace(
            scientific_key=scientific_key(["before"]),
            response="before target reply",
        )
        target_actions = tuple(
            SimpleNamespace(
                action_id=entity_id("la", [unit_key, label]),
                slot={
                    "role": Role.TARGET.value,
                    "purpose": ActionPurpose.TARGET_SAMPLE.value,
                },
            )
            for label in ("aligned", "unaligned", "unresolved", "unusable")
        )
        outcome_action = SimpleNamespace(
            action_id=entity_id("la", [unit_key, "outcome"]),
            slot={
                "role": Role.SELECTOR.value,
                "purpose": ActionPurpose.OUTCOME_REVIEW.value,
                "guard": {
                    "kind": "if_any_aligned_response",
                    "target_action_ids": [
                        item.action_id for item in target_actions
                    ],
                },
            },
        )
        panels = [
            {
                "panel_id": entity_id("pn", [item.action_id]),
                "target_action_id": item.action_id,
                "normal_members": [],
                "replacement": {"judge_action_id": entity_id("la", [item.action_id, "r"])},
            }
            for item in target_actions
        ]
        unit = SimpleNamespace(
            unit={
                "unit_scientific_key": unit_key,
                "design": {"round_index": 0},
                "panel_plans": panels,
            },
            actions=(*target_actions, outcome_action),
        )
        slots = tuple(
            ProposalSlot(
                entity_id("ps", [unit_key, ordinal]),
                scientific_key([unit_key, "slot", ordinal]),
                ordinal,
            )
            for ordinal in range(4)
        )

        class Success:
            def to_dict(self):
                return {
                    "transition": {"from": "premise", "to": "commitment"},
                    "supports": "The transition is made.",
                    "falsifies": "The transition is rejected.",
                }

        lane = SimpleNamespace(
            item=SimpleNamespace(contract=SimpleNamespace(success=Success()))
        )
        statuses = ("aligned", "invalid_contract", "aligned", "aligned")
        selected_by_action = {
            action.action_id: (
                slot,
                lane,
                SimpleNamespace(
                    review=SimpleNamespace(status=status),
                    proposal=SimpleNamespace(user_turn=f"user {ordinal}"),
                ),
            )
            for ordinal, (action, slot, status) in enumerate(
                zip(target_actions, slots, statuses, strict=True)
            )
        }
        observations = {
            action.action_id: SimpleNamespace(
                scientific_key=scientific_key([action.action_id, "observation"]),
                response=f"after {ordinal}",
            )
            for ordinal, action in enumerate(target_actions)
        }
        state = SimpleNamespace(
            target_results={
                action.action_id: SimpleNamespace(
                    usable=ordinal != 3, text=f"after {ordinal}"
                )
                for ordinal, action in enumerate(target_actions)
            },
            target_observations=observations,
            panel_views={
                panel["panel_id"]: SimpleNamespace(resolved=index != 2)
                for index, panel in enumerate(panels)
            },
        )
        with (
            patch(
                "quasilisk.engine._portfolio_branch_turns",
                return_value=((SimpleNamespace(), SimpleNamespace(), before),),
            ),
            patch(
                "quasilisk.engine._portfolio_selected_realization",
                side_effect=lambda context, state, unit, action: selected_by_action[
                    action.action_id
                ],
            ),
        ):
            request, expected = _portfolio_outcome_review_request(
                self.context, state, unit
            )
        self.assertEqual(len(request["items"]), 1)
        item = request["items"][0]
        self.assertEqual(
            set(item),
            {
                "review_alias",
                "transition_criterion",
                "before_target",
                "intervening_user",
                "after_target",
            },
        )
        expected_id = make_review_item_id(
            unit_key, "outcome", slots[0].scientific_key
        )
        self.assertEqual(item["review_alias"], "r0")
        self.assertEqual(
            expected,
            {
                expected_id: (
                    observations[target_actions[0].action_id].scientific_key,
                    "after 0",
                )
            },
        )
        serialized = json.dumps(request, sort_keys=True)
        for forbidden in ("score", "judge", "diagnosis", "archive", "promotion"):
            self.assertNotIn(forbidden, serialized)

    def test_outcome_result_uses_keyed_provenance_parser(self) -> None:
        review_id = make_review_item_id(
            scientific_key(["unit"]), "outcome", scientific_key(["slot"])
        )
        turn_ref = scientific_key(["after"])
        expected = {review_id: (turn_ref, "exact response evidence")}
        good = {
            "reviews": [
                {
                    "review_alias": "r0",
                    "result": "supports",
                    "response_state": "engaged",
                    "justification": "The frozen transition occurred.",
                }
            ]
        }
        action = SimpleNamespace()
        with (
            patch(
                "quasilisk.engine._portfolio_outcome_review_request",
                return_value=({"items": []}, expected),
            ),
            patch(
                "quasilisk.engine._schema_validated_payloads",
                return_value=(((None, None, good),), ()),
            ),
        ):
            result = _portfolio_outcome_result(
                self.context, SimpleNamespace(), SimpleNamespace(), action
            )
        self.assertTrue(result.usable)
        self.assertEqual(result.value[0].review_item_id, review_id)

        bad = {"reviews": [{**good["reviews"][0], "review_alias": "r9"}]}
        with (
            patch(
                "quasilisk.engine._portfolio_outcome_review_request",
                return_value=({"items": []}, expected),
            ),
            patch(
                "quasilisk.engine._schema_validated_payloads",
                return_value=(((None, None, bad),), ()),
            ),
        ):
            result = _portfolio_outcome_result(
                self.context, SimpleNamespace(), SimpleNamespace(), action
            )
        self.assertFalse(result.usable)
        self.assertTrue(
            any("review_alias must be exactly one of: r0" in item for item in result.errors)
        )


if __name__ == "__main__":
    unittest.main()
