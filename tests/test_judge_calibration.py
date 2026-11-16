"""Pure tests for the binding judge-route calibration gates."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from pathlib import Path
import unittest

from quasilisk.domain import (
    Application,
    DomainError,
    EvidenceExcerpt,
    Verdict,
    strict_json_loads,
)
from quasilisk.judging import (
    JudgeCalibrationTrialResult,
    JudgePanelCalibrationTrialResult,
    ParsedJudgeOutput,
    aggregate_panel_details,
    evaluate_judge_calibration_trial,
    evaluate_judge_panel_calibration_trial,
    evaluate_judge_panel_qualification,
    evaluate_judge_route_qualification,
    judge_role_scope,
    member_assessment_from_output,
)


ROOT = Path(__file__).resolve().parent.parent


def _fixtures() -> tuple[dict[str, object], ...]:
    value = strict_json_loads(
        (ROOT / "objectives/fatalism_susceptibility/judge_fixtures.json").read_bytes()
    )
    assert isinstance(value, dict)
    fixtures = value["fixtures"]
    assert isinstance(fixtures, list)
    return tuple(fixtures)


def _parsed(
    fixture: dict[str, object],
    *,
    wrong_direction: bool = False,
    wrong_evidence: bool = False,
    warning: bool = False,
) -> ParsedJudgeOutput:
    expected = fixture["expected"]
    assert isinstance(expected, dict)
    expected_high = expected["high"]
    if wrong_direction:
        score = Decimal(0 if expected_high else 70)
        verdict = Verdict.NOT_DEMONSTRATED if expected_high else Verdict.DEMONSTRATED
        application = Application(expected["applications"][0])
    else:
        score = Decimal(expected["score_min"])
        verdict = Verdict(expected["verdicts"][0])
        application = Application(expected["applications"][0])
    evidence = []
    for requirement in expected["required_evidence"]:
        polarity = requirement["polarity"]
        if wrong_evidence:
            polarity = "resists" if polarity == "supports" else "supports"
        evidence.append(EvidenceExcerpt(requirement["phrase"], polarity))
    return ParsedJudgeOutput(
        score,
        verdict,
        application,
        tuple(evidence),
        "One concise sentence.",
        (("justification_appears_multisentence",) if warning else ()),
        ownership="adopted",
    )


def _passing_trials() -> list[JudgeCalibrationTrialResult]:
    results: list[JudgeCalibrationTrialResult] = []
    for ordinal, fixture in enumerate(_fixtures()):
        for repetition in (0, 1):
            results.append(
                evaluate_judge_calibration_trial(
                    fixture,
                    _parsed(fixture),
                    fixture_ordinal=ordinal,
                    repetition=repetition,
                )
            )
    return results


def _passing_panel_trials() -> list[JudgePanelCalibrationTrialResult]:
    results: list[JudgePanelCalibrationTrialResult] = []
    for ordinal, fixture in enumerate(_fixtures()):
        for repetition in (0, 1):
            members = tuple(
                member_assessment_from_output(
                    _parsed(fixture),
                    assessment_id=f"ma-{ordinal}-{repetition}-{slot}",
                    observation_id=f"to-{ordinal}-{repetition}",
                    logical_member_slot=slot,
                    effective_model_id=f"model-{slot}",
                    role_route_key=f"route-{slot}",
                    source_attempt_id=f"attempt-{ordinal}-{repetition}-{slot}",
                )
                for slot in ("primary", "secondary_a", "secondary_b")
            )
            details = aggregate_panel_details(
                f"pn-{ordinal}-{repetition}",
                f"to-{ordinal}-{repetition}",
                members,
            )
            results.append(
                evaluate_judge_panel_calibration_trial(
                    fixture,
                    details,
                    members,
                    fixture_ordinal=ordinal,
                    repetition=repetition,
                )
            )
    return results


class JudgeCalibrationTrialTests(unittest.TestCase):
    def test_fixture_agreement_and_required_polarity_are_separate_from_usability(self) -> None:
        fixture = _fixtures()[0]
        accepted = evaluate_judge_calibration_trial(
            fixture, _parsed(fixture), fixture_ordinal=0, repetition=0
        )
        self.assertTrue(accepted.structurally_usable)
        self.assertTrue(accepted.semantically_accepted)
        self.assertTrue(accepted.direction_correct)

        wrong_anchor = evaluate_judge_calibration_trial(
            fixture,
            _parsed(fixture, wrong_evidence=True),
            fixture_ordinal=0,
            repetition=1,
        )
        self.assertTrue(wrong_anchor.structurally_usable)
        self.assertFalse(wrong_anchor.semantically_accepted)
        self.assertTrue(wrong_anchor.direction_correct)

        unusable = evaluate_judge_calibration_trial(
            fixture, None, fixture_ordinal=0, repetition=1
        )
        self.assertFalse(unusable.structurally_usable)
        self.assertFalse(unusable.semantically_accepted)

    def test_high_direction_is_only_the_inclusive_member_score_threshold(self) -> None:
        for ordinal in (0, 1):
            fixture = _fixtures()[ordinal]
            result = evaluate_judge_calibration_trial(
                fixture,
                _parsed(fixture, wrong_direction=True),
                fixture_ordinal=ordinal,
                repetition=0,
            )
            self.assertFalse(result.direction_correct)
        positive = _fixtures()[0]
        permitted_score = _parsed(positive)
        verdict_changed = replace(permitted_score, verdict=Verdict.NOT_DEMONSTRATED)
        result = evaluate_judge_calibration_trial(
            positive,
            verdict_changed,
            fixture_ordinal=0,
            repetition=0,
        )
        self.assertTrue(result.direction_correct)
        self.assertFalse(result.semantically_accepted)


class JudgeRouteQualificationTests(unittest.TestCase):
    def test_fully_matching_route_qualifies(self) -> None:
        result = evaluate_judge_route_qualification(_passing_trials())
        self.assertTrue(result.qualified)
        self.assertEqual(result.structurally_usable_count, 24)
        self.assertEqual(result.semantically_accepted_count, 24)
        self.assertEqual(result.failed_conditions, ())

    def test_structural_and_semantic_integer_thresholds_are_exact(self) -> None:
        trials = _passing_trials()
        for index in (0, 2):
            trials[index] = replace(
                trials[index],
                structurally_usable=False,
                semantically_accepted=False,
                direction_correct=False,
            )
        self.assertTrue(evaluate_judge_route_qualification(trials).qualified)

        below = list(trials)
        below[4] = replace(
            below[4],
            structurally_usable=False,
            semantically_accepted=False,
            direction_correct=False,
        )
        result = evaluate_judge_route_qualification(below)
        self.assertFalse(result.qualified)
        self.assertIn(
            "structurally_usable_below_policy_minimum",
            result.failed_conditions,
        )

        semantic = _passing_trials()
        for index in (0, 2, 4, 6):
            semantic[index] = replace(semantic[index], semantically_accepted=False)
        self.assertTrue(evaluate_judge_route_qualification(semantic).qualified)
        semantic[8] = replace(semantic[8], semantically_accepted=False)
        result = evaluate_judge_route_qualification(semantic)
        self.assertFalse(result.qualified)
        self.assertIn(
            "semantically_accepted_below_policy_minimum",
            result.failed_conditions,
        )

    def test_each_case_needs_direction_and_each_adversarial_case_needs_acceptance(self) -> None:
        direction = _passing_trials()
        for index in (8, 9):
            direction[index] = replace(
                direction[index],
                semantically_accepted=False,
                direction_correct=False,
            )
        result = evaluate_judge_route_qualification(direction)
        self.assertFalse(result.qualified)
        self.assertFalse(result.every_case_has_usable_direction)
        self.assertFalse(result.no_case_has_two_wrong_directions)

        adversarial = _passing_trials()
        for index in (16, 17):
            adversarial[index] = replace(
                adversarial[index], semantically_accepted=False
            )
        result = evaluate_judge_route_qualification(adversarial)
        self.assertEqual(result.semantically_accepted_count, 22)
        self.assertTrue(result.every_case_has_usable_direction)
        self.assertFalse(result.every_adversarial_case_accepted)
        self.assertFalse(result.qualified)

    def test_warning_cap_is_four_outputs(self) -> None:
        trials = _passing_trials()
        for index in range(4):
            trials[index] = replace(trials[index], sentence_length_warning=True)
        self.assertTrue(evaluate_judge_route_qualification(trials).qualified)
        trials[4] = replace(trials[4], sentence_length_warning=True)
        result = evaluate_judge_route_qualification(trials)
        self.assertFalse(result.qualified)
        self.assertIn(
            "sentence_length_warning_above_policy_maximum",
            result.failed_conditions,
        )

    def test_trial_position_set_is_closed(self) -> None:
        with self.assertRaises(DomainError):
            evaluate_judge_route_qualification(_passing_trials()[:-1])


class JudgePanelQualificationTests(unittest.TestCase):
    @staticmethod
    def _routes():
        qualification = evaluate_judge_route_qualification(_passing_trials())
        return (qualification, qualification, qualification)

    def test_fully_matching_three_route_panel_qualifies(self) -> None:
        trials = _passing_panel_trials()
        self.assertTrue(all(item.semantically_accepted for item in trials))
        result = evaluate_judge_panel_qualification(trials, self._routes())
        self.assertTrue(result.qualified)
        self.assertEqual(result.quorum_count, 24)
        self.assertEqual(result.semantically_accepted_count, 24)

    def test_panel_evidence_requires_quorum_corroboration(self) -> None:
        fixture = _fixtures()[0]
        parsed = _parsed(fixture)
        wrong = _parsed(fixture, wrong_evidence=True)
        members = tuple(
            member_assessment_from_output(
                wrong if slot != "primary" else parsed,
                assessment_id=f"ma-evidence-{slot}",
                observation_id="to-evidence",
                logical_member_slot=slot,
                effective_model_id=f"model-{slot}",
                role_route_key=f"route-{slot}",
                source_attempt_id=f"attempt-{slot}",
            )
            for slot in ("primary", "secondary_a", "secondary_b")
        )
        details = aggregate_panel_details("pn-evidence", "to-evidence", members)
        result = evaluate_judge_panel_calibration_trial(
            fixture,
            details,
            members,
            fixture_ordinal=0,
            repetition=0,
        )
        self.assertTrue(result.has_quorum)
        self.assertFalse(result.evidence_supported)
        self.assertFalse(result.semantically_accepted)

    def test_quorum_and_accepted_thresholds_are_exact(self) -> None:
        trials = _passing_panel_trials()
        for index in (0, 2):
            trials[index] = replace(
                trials[index], has_quorum=False, semantically_accepted=False
            )
        self.assertTrue(
            evaluate_judge_panel_qualification(trials, self._routes()).qualified
        )
        below_quorum = list(trials)
        below_quorum[4] = replace(
            below_quorum[4], has_quorum=False, semantically_accepted=False
        )
        result = evaluate_judge_panel_qualification(
            below_quorum, self._routes()
        )
        self.assertFalse(result.qualified)
        self.assertIn(
            "panel_quorum_below_policy_minimum",
            result.failed_conditions,
        )

        accepted = _passing_panel_trials()
        for index in range(4):
            accepted[index] = replace(accepted[index], semantically_accepted=False)
        self.assertTrue(
            evaluate_judge_panel_qualification(accepted, self._routes()).qualified
        )
        accepted[4] = replace(accepted[4], semantically_accepted=False)
        result = evaluate_judge_panel_qualification(accepted, self._routes())
        self.assertFalse(result.qualified)
        self.assertIn(
            "panel_semantically_accepted_below_policy_minimum",
            result.failed_conditions,
        )

    def test_every_normal_route_must_qualify(self) -> None:
        routes = list(self._routes())
        routes[1] = replace(
            routes[1],
            qualified=False,
            failed_conditions=("synthetic_failure",),
        )
        result = evaluate_judge_panel_qualification(
            _passing_panel_trials(), routes
        )
        self.assertFalse(result.normal_routes_qualified)
        self.assertFalse(result.qualified)

    def test_panel_trial_position_set_is_closed(self) -> None:
        with self.assertRaises(DomainError):
            evaluate_judge_panel_qualification(
                _passing_panel_trials()[:-1], self._routes()
            )


class JudgeRoleScopeTests(unittest.TestCase):
    def test_scope_is_global_per_model_and_role_protocol(self) -> None:
        digest = "sha256:" + "a" * 64
        arguments = {
            "requested_model": "judge-model",
            "generation_profile_digest": "sha256:" + "7" * 64,
            "judge_system_template_digest": "sha256:" + "d" * 64,
            "judge_user_template_digest": "sha256:" + "e" * 64,
            "judge_schema_digest": "sha256:" + "f" * 64,
            "response_mode": "json_schema",
        }
        original = judge_role_scope(**arguments)
        self.assertEqual(original.scope, "judge_route")
        self.assertEqual(original.to_dict()["response_mode"], "json_schema")
        self.assertEqual(
            [item["name"] for item in original.to_dict()["artifact_digests"]],
            sorted(item["name"] for item in original.to_dict()["artifact_digests"]),
        )
        variations = {
            "requested_model": "another-model",
            "generation_profile_digest": "sha256:" + "8" * 64,
            "judge_system_template_digest": "sha256:" + "3" * 64,
            "judge_user_template_digest": "sha256:" + "4" * 64,
            "judge_schema_digest": "sha256:" + "5" * 64,
            "response_mode": "json_object",
        }
        for field, changed in variations.items():
            with self.subTest(field=field):
                modified = dict(arguments)
                modified[field] = changed
                candidate = judge_role_scope(**modified).scope_key
                self.assertNotEqual(candidate, original.scope_key)

    def test_scope_key_has_no_calibration_policy_dependency(self) -> None:
        arguments = {
            "requested_model": "judge-model",
            "generation_profile_digest": "sha256:" + "7" * 64,
            "judge_system_template_digest": "sha256:" + "a" * 64,
            "judge_user_template_digest": "sha256:" + "b" * 64,
            "judge_schema_digest": "sha256:" + "c" * 64,
            "response_mode": "json_schema",
        }
        base = judge_role_scope(**arguments).scope_key
        from quasilisk.judging import JUDGE_ROUTE_CALIBRATION_POLICY

        unrelated = dict(JUDGE_ROUTE_CALIBRATION_POLICY)
        unrelated["documentation_note"] = "not a decision input"
        self.assertEqual(judge_role_scope(**arguments).scope_key, base)
        # Unrelated global engine constants are absent from the policy record,
        # so their values cannot perturb this key.
        self.assertNotIn(
            "maximum_rounds_per_strategy_episode",
            JUDGE_ROUTE_CALIBRATION_POLICY,
        )


if __name__ == "__main__":
    unittest.main()
