"""Keep novelty history separate from the target's actual conversation."""

from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from quasilisk.domain import scientific_key
from quasilisk.engine import (
    _messages_for_action, _prior_attempts,
    _qualification_perturbator_request, _scenario_perturbator_request,
)
from quasilisk.strategy import TemplateBindingResult
from quasilisk.judging import judge_objective_context
from quasilisk.perturbation import (
    ReviewExpectation, make_proposal_slots, make_review_item_id, probe_objective_context,
    parse_probe_review_statuses, parse_proposal_batch,
)

ROOT = Path(__file__).resolve().parents[1]


class SemanticContextHygieneTests(unittest.TestCase):
    def setUp(self):
        self.key = scientific_key(["context-hygiene"])
        self.slots = make_proposal_slots(self.key, 1)
        self.batch = parse_proposal_batch({"items": [{
            "slot": "s0", "status": "proposed",
            "text": "On your view, the warning does not bind. State your position.",
        }]}, self.slots)

    def test_novelty_history_cannot_supply_other_target_replies_or_assessments(self):
        def previous(sequence, prompt):
            action = SimpleNamespace(
                slot={"role": "target", "action_scientific_key": str(sequence)},
                attempts=[SimpleNamespace(intended={"messages": [
                    {"role": "assistant", "content": "UNRELATED EARLIER CONCESSION"},
                    {"role": "user", "content": prompt},
                ]})],
            )
            return SimpleNamespace(unit_id=str(sequence),
                unit={"kind": "initial_observation"},
                start_event={"sequence": sequence}, actions=[action])

        current = SimpleNamespace(unit_id="current", start_event={"sequence": 10})
        state = SimpleNamespace(work_units=[previous(2, "warning question"),
            previous(3, "other question"), previous(11, "future warning")])
        context = SimpleNamespace(config=SimpleNamespace(
            search=SimpleNamespace(proposals_per_step=2)))
        # No response store is needed or consulted by this projection.
        result = _prior_attempts(context, state, current, "warning")
        self.assertEqual([item["user_turn"] for item in result],
                         ["warning question", "other question"])
        for item in result:
            self.assertEqual(set(item), {"attempt_ref", "context_role", "user_turn"})
            self.assertEqual(item["context_role"], "novelty_only")
        self.assertNotIn("UNRELATED", json.dumps(result))

    def test_ordinary_review_does_not_receive_generation_parent_evidence(self):
        unit = SimpleNamespace(unit_id="current", unit={
            "kind": "map_exploration", "unit_scientific_key": self.key},
            actions=[SimpleNamespace(slot={"purpose": "perturbator_proposal"})])
        action = SimpleNamespace(slot={"role": "selector", "purpose": "probe_review"})
        context = SimpleNamespace(
            objective=SimpleNamespace(objective=json.loads((ROOT /
                "objectives/fatalism_susceptibility/objective.json").read_text())),
            resources=SimpleNamespace(templates={name: (ROOT / "templates" / name).read_text()
                for name in ("probe_review_system.md", "probe_review_user.md")}),
        )
        with patch("quasilisk.engine._schema_for_action", return_value=None), \
             patch("quasilisk.engine._map_proposal_result", return_value=SimpleNamespace(value=self.batch)), \
             patch("quasilisk.engine._proposal_slots", return_value=self.slots), \
             patch("quasilisk.engine._prior_attempts", return_value=[]), \
             patch("quasilisk.engine._generation_brief", side_effect=AssertionError("historical ownership leak")):
            messages = _messages_for_action(context, SimpleNamespace(), unit, action)
        text = messages[-1].content
        request = json.loads(text.split("<UNTRUSTED_PROBE_REVIEW_REQUEST_JSON>")[1]
                            .split("</UNTRUSTED_PROBE_REVIEW_REQUEST_JSON>")[0])
        self.assertEqual(request["mode"], "ordinary")
        self.assertEqual(request["objective"], probe_objective_context(context.objective.objective))
        self.assertNotIn("generation_brief", request)
        self.assertNotIn("conversation", request)
        self.assertEqual(request["items"][0]["proposal"]["user_turn"], self.batch.items[0].user_turn)

    def test_claimed_target_ownership_is_impossible_only_in_fresh_mode(self):
        proposal = self.batch.items[0]
        review_id = make_review_item_id(self.key, "probe", self.slots[0].scientific_key)
        expected = (ReviewExpectation(review_id, proposal),)
        for mode, status, want in (("ordinary", "valid", "invalid"),
                                   ("strategy", "aligned", "aligned")):
            with self.subTest(mode=mode):
                value = {"reviews": [{"review_alias": "r0", "status": status,
                    "reviewed_descriptors": {} if mode == "ordinary" else None,
                    "reason": "The reviewer claims prior ownership.",
                    "procedure": {"assumption": "target_owned",
                        "assumption_quote": "On your view", "direction": "forward", "novelty": "new"}}]}
                self.assertEqual(parse_probe_review_statuses(value, expected, mode=mode), {review_id: want})
        value["reviews"][0].update(status="valid", reviewed_descriptors={})
        value["reviews"][0]["procedure"].update(assumption="scenario_fact")
        self.assertEqual(parse_probe_review_statuses(value, expected, mode="ordinary"), {review_id: "valid"})

    def test_replay_and_qualification_treatments_receive_goal_without_control_leak(self):
        objective = SimpleNamespace(id="test", strategy_goal="Destination, not just the next step.",
            objective={"scientific_question": "Research question"},
            descriptor_names=("argument_move",), descriptor_values={"argument_move": ("test_inference",)})
        context = SimpleNamespace(objective=objective)
        conversation = [{"role": "assistant", "content": "Current branch only"}]
        frozen = {"success": {"supports": "Owns the missing inference"}}
        binding = TemplateBindingResult(
            "b0", "applicable", (), "Exact branch",
            SimpleNamespace(to_dict=lambda: frozen),
        )
        unit = SimpleNamespace(unit={
            "kind": "strategy_development",
            "design": {"comparison_dimensions": ["wording"]},
        })
        slot = self.slots[0]
        for path in ("treatment", "strategy_agnostic", "scenario"):
            with self.subTest(path=path), ExitStack() as stack:
                if path == "scenario":
                    mocks = {
                        "_scenario_for_unit": SimpleNamespace(recipe=SimpleNamespace(templates=[
                            SimpleNamespace(vary=[SimpleNamespace(dimension="wording")])])),
                        "_scenario_root_context": ({}, 0, slot),
                        "_scenario_step_index": 0,
                        "_scenario_fresh_branch": (conversation, None, None, None),
                        "_scenario_binding_action": None,
                        "_scenario_binding_result": SimpleNamespace(value=binding),
                        "_scenario_correction_review": None,
                    }
                    build = _scenario_perturbator_request
                else:
                    mocks = {
                        "_qualification_root_context": ({"arm": path}, {}, 0, 0, slot, "anchor"),
                        "_qualification_fresh_branch": (conversation, None, None, None),
                        "_qualification_correction_review": None,
                        "_qualification_binding_action": None,
                        "_qualification_binding_result": SimpleNamespace(value=binding),
                    }
                    build = _qualification_perturbator_request
                for name, value in mocks.items():
                    stack.enter_context(patch("quasilisk.engine." + name, return_value=value))
                request = build(context, SimpleNamespace(), unit, SimpleNamespace())
                if path == "strategy_agnostic":
                    self.assertNotIn("objective_definition", request)
                    self.assertNotIn("bound_strategy", request)
                    self.assertNotIn(objective.strategy_goal, json.dumps(request))
                    self.assertEqual(request["fresh_branch_transcript"], conversation)
                else:
                    self.assertEqual(request["objective_definition"], {
                        "objective_id": objective.id, "strategy_goal": objective.strategy_goal})
                    self.assertEqual(request["bound_strategy"], frozen)
                    self.assertEqual(request["conversation"], conversation)
                self.assertEqual(request["proposal_slots"], [{"slot": "s0", "intent": "anchor"}])

    def test_probe_projection_keeps_descriptor_meanings_without_judge_or_diagnosis_policy(self):
        for path in (ROOT / "objectives").glob("*/objective.json"):
            with self.subTest(objective=path.parent.name):
                objective = json.loads(path.read_text())
                projected = probe_objective_context(objective)
                self.assertEqual(set(projected), {
                    "id", "strategy_goal", "non_failure_boundaries", "descriptors"})
                for name in projected:
                    self.assertEqual(projected[name], objective[name])
                projected["descriptors"][0]["definition"] = "mutated"
                self.assertNotEqual(projected["descriptors"], objective["descriptors"])

    def test_judge_projection_preserves_contract_and_excludes_search_instructions(self):
        for path in (ROOT / "objectives").glob("*/objective.json"):
            with self.subTest(objective=path.parent.name):
                objective = json.loads(path.read_text())
                projected = judge_objective_context(objective)
                self.assertEqual(set(projected), {"id", "scientific_question", "failure_definition",
                    "non_failure_boundaries", "score_anchors"})
                for name in projected:
                    self.assertEqual(projected[name], objective[name])
                projected["score_anchors"][0]["meaning"] = "mutated"
                self.assertNotEqual(projected["score_anchors"], objective["score_anchors"])


if __name__ == "__main__":
    unittest.main()
