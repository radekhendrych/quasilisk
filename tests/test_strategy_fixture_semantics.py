"""Strategy-owned semantic conformance for checked-in Selector fixtures."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from quasilisk.domain import strict_json_loads
from quasilisk.objective import ObjectiveError, load_objective
from quasilisk.strategy import validate_selector_fixture_semantics

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OBJECTIVE_ROOT = REPOSITORY_ROOT / "objectives"
OBJECTIVE_ID = "fatalism_susceptibility"

class StrategyFixtureSemanticTests(unittest.TestCase):

    def test_recovery_fixture_suite_covers_direction_not_just_engagement(self) -> None:
        for fixture_path in sorted(OBJECTIVE_ROOT.glob("*/selector_fixtures.json")):
            data = strict_json_loads(fixture_path.read_bytes())
            recovery_reviews = [
                item
                for item in data["strategy_probe_review"]
                if item["contract"]["purpose"] == "recovery"
            ]
            self.assertEqual(
                {item["expected"]["status"] for item in recovery_reviews},
                {"aligned", "invalid_contract"},
            )
            recovery_outcomes = [
                item
                for item in data["outcome_review"]
                if item["contract"]["purpose"] == "recovery"
            ]
            self.assertEqual(
                {item["expected"]["result"] for item in recovery_outcomes},
                {"supports", "falsifies", "indeterminate"},
            )
            neutral = next(
                item
                for item in recovery_outcomes
                if item["expected"]["result"] == "indeterminate"
            )
            self.assertEqual(neutral["expected"]["response_state"], "engaged")
            supported = next(
                item
                for item in recovery_outcomes
                if item["expected"]["result"] == "supports"
            )
            self.assertRegex(
                supported["after_target"]["content"], r"^I (accept|conclude)\b"
            )

    def test_conditional_selector_fixtures_require_condition_repair(self) -> None:
        for fixture_path in sorted(OBJECTIVE_ROOT.glob("*/selector_fixtures.json")):
            data = strict_json_loads(fixture_path.read_bytes())
            conditional = next(
                item
                for item in data["portfolio"]
                if item["fixture_id"] == "selector_portfolio_conditional"
            )
            expected = conditional["expected"]
            self.assertEqual(expected["allowed_purposes"], ["repair"])
            self.assertEqual(expected["allowed_move_acts"], ["repair_grounded_gap"])
            self.assertEqual(expected["allowed_target_kinds"], ["condition"])

    def test_portfolio_support_quotes_retain_their_conditional_scope(self) -> None:
        for fixture_path in sorted(OBJECTIVE_ROOT.glob("*/selector_fixtures.json")):
            data = strict_json_loads(fixture_path.read_bytes())
            for fixture in data["portfolio"]:
                supports = [
                    item["quote"]
                    for item in fixture["judge_context"]["evidence"]
                    if item["polarity"] == "supports"
                ]
                self.assertTrue(supports, fixture["fixture_id"])
                self.assertTrue(
                    all(quote.startswith("If ") for quote in supports),
                    fixture["fixture_id"],
                )

    def _load_mutated(self, mutate) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / OBJECTIVE_ID
            shutil.copytree(OBJECTIVE_ROOT / OBJECTIVE_ID, destination)
            fixture_path = destination / "selector_fixtures.json"
            data = strict_json_loads(fixture_path.read_bytes())
            mutate(data)
            fixture_path.write_text(
                json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            load_objective(
                OBJECTIVE_ID,
                root,
                selector_fixture_semantic_validator=validate_selector_fixture_semantics,
            )

    def test_aligned_expectation_must_match_contract_and_realization(self) -> None:
        def mutate(data) -> None:
            data["strategy_probe_review"][0]["expected"]["reviewed_descriptors"][
                "objective_route"
            ] = "absent_alternatives"

        with self.assertRaisesRegex(
            ObjectiveError,
            "aligned fixture descriptors must equal the contract declaration",
        ):
            self._load_mutated(mutate)

    def test_realization_uses_the_canonical_proposal_parser(self) -> None:
        def mutate(data) -> None:
            data["strategy_probe_review"][0]["realization"]["status"] = "abstained"

        with self.assertRaisesRegex(
            ObjectiveError, "only a proposed item may be reviewed"
        ):
            self._load_mutated(mutate)

    def test_outcome_contract_cannot_depend_on_the_future_target_reply(self) -> None:
        def mutate(data) -> None:
            fixture = data["outcome_review"][0]
            contract = fixture["contract"]
            after = fixture["after_target"]
            future_ref = f"fixture:{fixture['fixture_id']}:{after['turn_id']}"
            contract["basis"][0]["turn_ref"] = future_ref
            contract["basis"][0]["quote"] = "standing decision rule"
            contract["move"]["target"]["anchor_ref"] = future_ref
            contract["preserve"][0]["anchor_ref"] = future_ref

        with self.assertRaisesRegex(
            ObjectiveError, "references an unknown target turn"
        ):
            self._load_mutated(mutate)

if __name__ == "__main__":
    unittest.main()
