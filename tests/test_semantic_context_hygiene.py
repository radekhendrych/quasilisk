"""Keep novelty history separate from the target's actual conversation."""

import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from quasilisk.domain import scientific_key
from quasilisk.engine import _messages_for_action, _prior_attempts
from quasilisk.judging import judge_objective_context
from quasilisk.perturbation import (
    ReviewExpectation, make_proposal_slots, make_review_item_id,
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
            objective=SimpleNamespace(objective={"id": "test"}),
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
