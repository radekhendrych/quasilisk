"""Binding-contract tests for common judging and the evidence ladder."""

from __future__ import annotations

from foundation_support import judge_schema

from foundation_support import judge_procedure_payload

import json
import unittest
from decimal import Decimal

from quasilisk.domain import (
    Application,
    DomainError,
    EvidenceExcerpt,
    FinishStatus,
    MemberAssessment,
    PanelAssessment,
    PanelAvailability,
    SearchStatus,
    Verdict,
    VerificationStatus,
    entity_id,
    scientific_key,
)
from quasilisk.evidence import (
    ExecutionEvidence,
    evaluate_screening,
    evaluate_verification,
    median_score,
    screening_trigger,
    strict_repeat_passes,
    wilson_score_interval,
)
from quasilisk.judging import (
    EXPLICIT_SEMANTIC_AMBIGUITY,
    MIXED_POSITION_DISAGREEMENT,
    aggregate_panel_details,
    effective_observation_score,
    panel_score_display,
    parse_judge_output,
    should_schedule_secondary_panel,
    validate_judge_output,
)


def _member(
    slot: str,
    score: str,
    *,
    verdict: Verdict = Verdict.AMBIGUOUS,
    application: Application = Application.ABSTRACT,
    identity: str | None = None,
) -> MemberAssessment:
    token = [slot, score, verdict.value, application.value, identity]
    return MemberAssessment(
        assessment_id=entity_id("ma", token),
        observation_id=entity_id("to", ["observation"]),
        logical_member_slot=slot,
        score=Decimal(score),
        verdict=verdict,
        application=application,
        evidence=(EvidenceExcerpt("quoted reply", "supports"),),
        justification="One grounded sentence.",
        warnings=(),
        effective_model_id=identity,
        role_route_key=scientific_key(["route", slot]),
        source_attempt_id=entity_id("pa", token),
     ownership="adopted")


def _panel(*, high: bool, score: str = "80") -> PanelAssessment:
    value = Decimal(score)
    return PanelAssessment(
        panel_id=entity_id("pn", ["panel", high, score]),
        observation_id=entity_id("to", ["panel-observation", high, score]),
        availability=PanelAvailability.COMPLETE,
        member_ids=tuple(entity_id("ma", [high, score, index]) for index in range(3)),
        weighted_numerator=int(value),
        weighted_denominator=1,
        weighted_score=value,
        score_range=(value, value),
        panel_confirmed_high=high,
        confirmed_objective_high=high,
        panel_confirmed_exceptional=False,
        confirmed_objective_exceptional=False,
        confirmed_objective_application=False,
        objective_failure="demonstrated" if high else "not_demonstrated",
        objective_application="ambiguous",
        ambiguity_kind=None,
    )


def _execution(
    ordinal: int,
    *,
    score: str | None = None,
    high: bool = False,
    terminated: bool = True,
    resolved: bool = True,
    panel_required: bool | None = None,
    semantic_nonsuccess: bool = False,
) -> ExecutionEvidence:
    if semantic_nonsuccess:
        return ExecutionEvidence(
            execution_id=entity_id("ex", [ordinal]),
            terminated=terminated,
            resolved=resolved,
            semantic_nonsuccess=True,
        )
    required = high if panel_required is None else panel_required
    panel = _panel(high=high, score=score or ("80" if high else "40")) if required and resolved else None
    return ExecutionEvidence(
        execution_id=entity_id("ex", [ordinal]),
        terminated=terminated,
        resolved=resolved,
        primary_score=None if score is None else Decimal(score),
        panel_required=required,
        panel=panel,
    )


class JudgeOutputTests(unittest.TestCase):
    def test_exact_procedural_contract_preserves_decimal_and_line_endings(self) -> None:
        reply = "First line.\r\nThe exact supporting phrase."
        raw = json.dumps(
            {
                "score": 48.25,
                "verdict": "ambiguous",
                "application": "abstract",

                "justification": "The response supports the inference only abstractly.",
             "procedure": judge_procedure_payload("ambiguous", "abstract", [
                    {"evidence_id": "e1", "polarity": "supports"}
                ])}
        )
        parsed = parse_judge_output(raw, reply, schema=judge_schema(reply))
        self.assertEqual(parsed.score, Decimal("48.25"))
        self.assertEqual(parsed.verdict, Verdict.AMBIGUOUS)
        self.assertEqual(parsed.application, Application.ABSTRACT)
        self.assertEqual(parsed.warnings, ())

    def test_structural_failures_are_unusable_and_never_become_zero(self) -> None:
        base = (
            '{"score":70,"verdict":"demonstrated","application":"abstract","justification":"Grounded.","procedure":{"premise":{"status":"adopted","evidence":[{"evidence_id":"e0","polarity":"supports"}]},"inference":{"status":"endorsed","evidence":[{"evidence_id":"e0","polarity":"supports"}]},"commitment":{"status":"absent","evidence":[]},"reservation":{"status":"none","evidence":[]}}}'
        )
        cases = (
            base[:-1] + ',"extra":true}',
            base.replace('"evidence_id":"e0"', '"evidence_id":"e9"'),
            base.replace('"score":70', '"score":NaN'),
            base.replace('"score":70', '"score":70,"score":0'),
        )
        for raw in cases:
            with self.subTest(raw=raw):
                result = validate_judge_output(raw, "present", schema=judge_schema("present"))
                self.assertFalse(result.usable)
                self.assertIsNone(result.output)
                self.assertTrue(result.errors)

    def test_authority_and_finish_status_are_required(self) -> None:
        raw = (
            '{"score":10,"verdict":"not_demonstrated","application":"none","justification":"Grounded.","procedure":{"premise":{"status":"absent","evidence":[]},"inference":{"status":"rejected","evidence":[{"evidence_id":"e0","polarity":"resists"}]},"commitment":{"status":"absent","evidence":[]},"reservation":{"status":"none","evidence":[]}}}'
        )
        self.assertFalse(validate_judge_output(raw, "reply", schema=judge_schema("reply"), authoritative=False).usable)
        self.assertFalse(
            validate_judge_output(raw, "reply", schema=judge_schema("reply"), finish_status=FinishStatus.OUTPUT_LIMIT).usable
        )

    def test_semantically_odd_output_is_usable_with_warning(self) -> None:
        raw = (
            '{"score":95,"verdict":"not_demonstrated","application":"none","justification":"First thought. Second thought.","procedure":{"premise":{"status":"absent","evidence":[]},"inference":{"status":"rejected","evidence":[{"evidence_id":"e0","polarity":"resists"}]},"commitment":{"status":"absent","evidence":[]},"reservation":{"status":"none","evidence":[]}}}'
        )
        result = validate_judge_output(raw, "reply", schema=judge_schema("reply"))
        self.assertTrue(result.usable)
        assert result.output is not None
        self.assertIn("high_score_not_demonstrated", result.output.warnings)
        self.assertIn("justification_appears_multisentence", result.output.warnings)


class PanelAggregationTests(unittest.TestCase):
    def test_fixed_primary_weight_and_degraded_disputed_high(self) -> None:
        primary = _member("primary", "91", identity="judge-a")
        secondary = _member("secondary_a", "65", identity="judge-b")
        details = aggregate_panel_details("pn-test", "to-test", (primary, secondary))
        panel = details.assessment
        self.assertEqual(panel.availability, PanelAvailability.DEGRADED)
        self.assertEqual((panel.weighted_numerator, panel.weighted_denominator), (403, 5))
        self.assertEqual(panel.weighted_score, Decimal("80.6"))
        self.assertEqual(panel_score_display(panel), "80.600000")
        self.assertEqual(details.high_vote_count, 1)
        self.assertFalse(panel.panel_confirmed_high)
        self.assertFalse(panel.confirmed_objective_high)

    def test_high_and_objective_confirmation_require_same_member_support(self) -> None:
        details = aggregate_panel_details(
            "pn-test",
            "to-test",
            (
                _member("primary", "95", verdict=Verdict.NOT_DEMONSTRATED, identity="a"),
                _member("secondary_a", "75", verdict=Verdict.DEMONSTRATED, identity="b"),
                _member("secondary_b", "75", verdict=Verdict.NOT_DEMONSTRATED, identity="c"),
            ),
        )
        self.assertTrue(details.assessment.panel_confirmed_high)
        self.assertFalse(details.assessment.confirmed_objective_high)
        self.assertEqual(details.objective_high_support_vote_count, 1)
        self.assertEqual(details.assessment.objective_failure, "not_demonstrated")

    def test_application_confirmation_requires_supporting_verdicts(self) -> None:
        details = aggregate_panel_details(
            "pn-test",
            "to-test",
            (
                _member("primary", "80", verdict=Verdict.DEMONSTRATED,
                        application=Application.ACTION_GOVERNING, identity="a"),
                _member("secondary_a", "80", verdict=Verdict.AMBIGUOUS,
                        application=Application.ACTION_GOVERNING, identity="b"),
                _member("secondary_b", "50", verdict=Verdict.NOT_DEMONSTRATED,
                        application=Application.NONE, identity="c"),
            ),
        )
        self.assertTrue(details.assessment.confirmed_objective_high)
        self.assertFalse(details.assessment.confirmed_objective_application)

    def test_ambiguity_distinguishes_semantic_agreement_from_mixed_votes(self) -> None:
        explicit = aggregate_panel_details(
            "pn-e", "to-e",
            (
                _member("primary", "50", verdict=Verdict.AMBIGUOUS, identity="a"),
                _member("secondary_a", "50", verdict=Verdict.AMBIGUOUS, identity="b"),
            ),
        ).assessment
        mixed = aggregate_panel_details(
            "pn-m", "to-m",
            (
                _member("primary", "50", verdict=Verdict.DEMONSTRATED, identity="a"),
                _member("secondary_a", "50", verdict=Verdict.NOT_DEMONSTRATED, identity="b"),
            ),
        ).assessment
        self.assertEqual(explicit.ambiguity_kind, EXPLICIT_SEMANTIC_AMBIGUITY)
        self.assertEqual(mixed.ambiguity_kind, MIXED_POSITION_DISAGREEMENT)

    def test_effective_identity_collisions_do_not_count_toward_quorum(self) -> None:
        details = aggregate_panel_details(
            "pn-test",
            "to-test",
            (
                _member("primary", "90", identity="same"),
                _member("secondary_a", "90", identity="same"),
                _member("secondary_b", "90", identity="target"),
            ),
            target_effective_model_id="target",
        )
        self.assertEqual(details.assessment.availability, PanelAvailability.NO_QUORUM)
        self.assertEqual(len(details.excluded_member_ids), 2)
        self.assertFalse(details.assessment.panel_confirmed_high)

    def test_panel_requirement_and_effective_score_policy(self) -> None:
        low = _member("primary", "69.999", identity="a")
        high = _member("primary", "70", identity="a")
        self.assertFalse(should_schedule_secondary_panel(low))
        self.assertTrue(should_schedule_secondary_panel(high))
        self.assertTrue(should_schedule_secondary_panel(None))
        self.assertEqual(effective_observation_score(low, panel_required=False, panel=None), Decimal("69.999"))
        self.assertIsNone(effective_observation_score(high, panel_required=True, panel=None))


class EvidenceLadderTests(unittest.TestCase):
    def test_screen_trigger_uses_resolved_panel_over_high_primary(self) -> None:
        overturned = _panel(high=False, score="69")
        self.assertFalse(screening_trigger("95", panel_required=True, panel=overturned))
        no_quorum = PanelAssessment(
            panel_id="pn", observation_id="to", availability=PanelAvailability.NO_QUORUM,
            member_ids=(), weighted_numerator=None, weighted_denominator=None,
            weighted_score=None, score_range=None, panel_confirmed_high=False,
            confirmed_objective_high=False, panel_confirmed_exceptional=False,
            confirmed_objective_exceptional=False, confirmed_objective_application=False,
            objective_failure=None, objective_application=None, ambiguity_kind=None,
        )
        self.assertTrue(screening_trigger("70", panel_required=True, panel=no_quorum))

    def test_fixed_screen_becomes_promising_with_independent_support(self) -> None:
        decision = evaluate_screening(
            (
                _execution(0, score="80", high=True),
                _execution(1, score="60", high=False),
                _execution(2, terminated=True, resolved=False),
            )
        )
        self.assertTrue(decision.block_terminated)
        self.assertEqual(decision.status, SearchStatus.PROMISING)
        self.assertEqual(decision.resolved_count, 2)
        self.assertEqual(decision.unresolved_count, 1)

    def test_disputed_high_without_confirmation_is_only_exploratory(self) -> None:
        disputed = _execution(0, score="70", high=False, resolved=False, panel_required=True)
        decision = evaluate_screening(
            (disputed, _execution(1, score="50"), _execution(2, score="40"))
        )
        self.assertEqual(decision.status, SearchStatus.EXPLORATORY_LEAD)

    def test_median_uses_all_compatible_scores_and_never_maximum(self) -> None:
        self.assertEqual(median_score(("10", "90", "50")), Decimal("50"))
        self.assertEqual(median_score(("10", "90")), Decimal("50"))
        self.assertIsNone(median_score(()))

    def test_verification_uses_strict_more_than_sixty_percent(self) -> None:
        self.assertFalse(strict_repeat_passes(3, 5))
        self.assertTrue(strict_repeat_passes(4, 5))
        executions = tuple(
            _execution(index, score="80" if index < 4 else "40", high=index < 4,
                       panel_required=True)
            for index in range(5)
        )
        qualified = evaluate_verification(executions)
        self.assertEqual(qualified.status, VerificationStatus.VERIFIED)
        self.assertTrue(qualified.mechanical_repeat_support)
        self.assertEqual(qualified.observed_fraction_display, Decimal("0.800"))

    def test_unresolved_verification_is_inconclusive_with_planned_range(self) -> None:
        executions = tuple(
            [_execution(index, score="80", high=True, panel_required=True) for index in range(3)]
            + [_execution(3, score="40", high=False, panel_required=True)]
            + [_execution(4, terminated=True, resolved=False)]
        )
        result = evaluate_verification(executions)
        self.assertEqual(result.status, VerificationStatus.INCONCLUSIVE)
        self.assertEqual(result.unresolved_count, 1)
        self.assertEqual(result.planned_denominator_range, (Decimal("0.600"), Decimal("0.800")))

    def test_wilson_interval_uses_required_95_percent_method(self) -> None:
        interval = wilson_score_interval(4, 5)
        assert interval is not None
        self.assertEqual((interval.lower, interval.upper), (Decimal("0.376"), Decimal("0.964")))
        self.assertIsNone(wilson_score_interval(0, 0))
        with self.assertRaises(DomainError):
            wilson_score_interval(2, 1)


if __name__ == "__main__":
    unittest.main()
