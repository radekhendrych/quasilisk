"""Pragmatic Perturbator parsing at the Selector admission boundary."""

from __future__ import annotations

import unittest

from quasilisk.domain import scientific_key
from quasilisk.perturbation import (
    PerturbationValidationError,
    ReviewExpectation,
    make_proposal_slots,
    make_review_item_id,
    parse_probe_review_statuses,
    parse_proposal_batch,
)


class StrategyPerturbationPracticalityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.slot = make_proposal_slots(
            scientific_key(["strategy-perturbation-practicality", "unit"]), 1
        )[0]
        self.item = {
            "slot": "s0",
            "status": "proposed",
            "text": "Please carry the cited opening through the exact gap.",
        }

    def test_minimal_proposal_materializes_the_durable_slot(self) -> None:
        parsed = parse_proposal_batch({"items": [self.item]}, (self.slot,))
        self.assertEqual(parsed.items[0].slot_id, self.slot.slot_id)
        self.assertEqual(parsed.items[0].user_turn, self.item["text"])

    def test_removed_generator_bookkeeping_is_rejected(self) -> None:
        self.item["changed_dimensions"] = []
        with self.assertRaisesRegex(PerturbationValidationError, "unknown"):
            parse_proposal_batch({"items": [self.item]}, (self.slot,))

    def test_abstention_uses_the_same_small_shape(self) -> None:
        self.item.update(
            status="abstained", text="No faithful distinct turn exists.\nAbstain."
        )
        parsed = parse_proposal_batch({"items": [self.item]}, (self.slot,))
        self.assertTrue(parsed.all_abstained)
        self.assertIn("Abstain", parsed.items[0].reason or "")

    def test_review_alias_materializes_the_durable_guard_status(self) -> None:
        proposal = parse_proposal_batch({"items": [self.item]}, (self.slot,)).items[0]
        review_id = make_review_item_id(
            scientific_key(["strategy-perturbation-practicality", "unit"]),
            "probe",
            self.slot.scientific_key,
        )
        statuses = parse_probe_review_statuses(
            {
                "reviews": [
                    {
                        "review_alias": "r0",
                        "status": "aligned",
                        "reviewed_descriptors": None,
                        "reason": "The turn advances the frozen strategy.",
                     "procedure": {"assumption": "none", "assumption_quote": None, "direction": "forward", "novelty": "new"}}
                ]
            },
            (ReviewExpectation(review_id, proposal),),
            mode="strategy",
        )
        self.assertEqual(statuses, {review_id: "aligned"})

    def test_ordinary_valid_status_still_requires_descriptors(self) -> None:
        proposal = parse_proposal_batch({"items": [self.item]}, (self.slot,)).items[0]
        review_id = make_review_item_id(
            scientific_key(["strategy-perturbation-practicality", "unit"]),
            "probe",
            self.slot.scientific_key,
        )
        with self.assertRaisesRegex(
            PerturbationValidationError, "requires descriptors"
        ):
            parse_probe_review_statuses(
                {
                    "reviews": [
                        {
                            "review_alias": "r0",
                            "status": "valid",
                            "reviewed_descriptors": None,
                            "reason": "Missing required descriptors.",
                         "procedure": {"assumption": "none", "assumption_quote": None, "direction": "forward", "novelty": "new"}}
                    ]
                },
                (ReviewExpectation(review_id, proposal),),
                mode="ordinary",
            )


if __name__ == "__main__":
    unittest.main()
