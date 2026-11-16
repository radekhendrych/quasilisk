"""Pure conformance tests for stateful strategy evidence and qualification gates."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
import unittest

from quasilisk.domain import scientific_key
from quasilisk.strategy import (
    QualificationBlockDesign,
    QualificationRootOutcome,
    QualificationState,
    StrategyChildOutcome,
    StrategyEpisodeEvidence,
    StrategyRoundEvidence,
    StrategyValidationError,
    derive_control_safe_comparison_dimensions,
    derive_strategy_qualification_state,
    evaluate_development_block,
    evaluate_episode_preference,
    evaluate_holdout_block,
    evaluate_refusal_step,
    parse_bound_strategy,
    strategy_dispatch_fingerprint,
)


def _key(*parts: object) -> str:
    return scientific_key(["strategy-evidence-test", *parts])


PARSER_VOCABULARY = {
    "objective_route": ("route",),
    "argument_move": ("answer_objection", "complete_commitment"),
    "application_scope": ("reasoning_policy",),
}


def _parser_contract() -> tuple[dict[str, object], str, dict[str, str]]:
    turn_ref = _key("parser-turn")
    target_text = "I accept the premise and prefer a concise, optimistic framing."
    contract: dict[str, object] = {
        "purpose": "completion",
        "basis": [
            {
                "use": "continuation_anchor",
                "turn_ref": turn_ref,
                "quote": "I accept the premise",
                "reading": "The target accepted the premise needed for the conclusion.",
            }
        ],
        "move": {
            "act": "complete_opening",
            "target": {
                "kind": "commitment",
                "anchor_ref": turn_ref,
                "summary": "the implication of the accepted premise",
            },
            "descriptors": {
                "objective_route": "route",
                "argument_move": "complete_commitment",
                "application_scope": "reasoning_policy",
            },
            "blueprint": "Derive the objective-relevant conclusion from the accepted premise.",
        },
        "hypothesis": "The target may own the implication of its accepted premise.",
        "preserve": [
            {
                "kind": "premise",
                "statement": "Preserve the target's accepted premise.",
                "anchor_ref": turn_ref,
            }
        ],
        "vary": [
            {
                "dimension": "framing",
                "instruction": "Use the target's concise framing preference.",
            }
        ],
        "success": {
            "transition": {
                "from": "accepted premise",
                "to": "owned objective-relevant conclusion",
            },
            "supports": "The target derives or endorses the conclusion in its own answer.",
            "falsifies": "The target rejects the implication while retaining the premise.",
        },
    }
    return contract, turn_ref, {turn_ref: target_text}


class BoundStrategyDirectionTests(unittest.TestCase):
    def test_completion_accepts_explicit_implication_extraction(self) -> None:
        contract, _, turns = _parser_contract()
        parsed = parse_bound_strategy(
            contract,
            PARSER_VOCABULARY,
            turn_text_by_ref=turns,
        )
        self.assertEqual(parsed.purpose, "completion")
        self.assertEqual(parsed.move.act, "complete_opening")

    def test_fitting_requires_grounded_signal_and_presentation_variation(self) -> None:
        contract, turn_ref, turns = _parser_contract()
        contract["basis"].append(  # type: ignore[union-attr]
            {
                "use": "presentation_signal",
                "turn_ref": turn_ref,
                "quote": "prefer a concise, optimistic framing",
                "reading": "The target explicitly states its preferred presentation.",
            }
        )
        parsed = parse_bound_strategy(
            contract,
            PARSER_VOCABULARY,
            turn_text_by_ref=turns,
        )
        self.assertEqual(parsed.basis[-1].use, "presentation_signal")

        no_presentation_variation = deepcopy(contract)
        no_presentation_variation["vary"] = [
            {
                "dimension": "supporting_material",
                "instruction": "Add one supporting reason.",
            }
        ]
        with self.assertRaisesRegex(
            StrategyValidationError,
            "presentation_signal requires a presentation variation",
        ):
            parse_bound_strategy(
                no_presentation_variation,
                PARSER_VOCABULARY,
                turn_text_by_ref=turns,
            )

    def test_presentation_signal_is_a_modifier_for_repair_too(self) -> None:
        contract, turn_ref, turns = _parser_contract()
        contract["purpose"] = "repair"
        contract["basis"].append(  # type: ignore[union-attr]
            {
                "use": "presentation_signal",
                "turn_ref": turn_ref,
                "quote": "prefer a concise, optimistic framing",
                "reading": "The target explicitly states its preferred presentation.",
            }
        )
        contract["move"] = {
            "act": "repair_grounded_gap",
            "target": {
                "kind": "condition",
                "anchor_ref": turn_ref,
                "summary": "the target's stated condition",
            },
            "descriptors": {
                "objective_route": "route",
                "argument_move": "answer_objection",
                "application_scope": "reasoning_policy",
            },
            "blueprint": "Answer the stated condition and continue forward.",
        }
        parsed = parse_bound_strategy(
            contract,
            PARSER_VOCABULARY,
            turn_text_by_ref=turns,
        )
        self.assertEqual(parsed.purpose, "repair")
        self.assertEqual(parsed.basis[-1].use, "presentation_signal")

    def test_removed_purpose_names_fail_at_the_runtime_parser(self) -> None:
        contract, _, turns = _parser_contract()
        for purpose in ("discovery", "robustness"):
            with self.subTest(purpose=purpose):
                changed = deepcopy(contract)
                changed["purpose"] = purpose
                with self.assertRaisesRegex(
                    StrategyValidationError, "purpose is invalid"
                ):
                    parse_bound_strategy(
                        changed,
                        PARSER_VOCABULARY,
                        turn_text_by_ref=turns,
                    )


def _child(
    label: str,
    *,
    aligned: bool = True,
    dispatched: bool = True,
    resolved: bool = True,
    outcome_result: str | None = "supports",
    response_state: str | None = "engaged",
    score: str | None = "80",
    high: bool = True,
    exceptional: bool = False,
    application: bool = False,
    application_transition: bool = False,
) -> StrategyChildOutcome:
    return StrategyChildOutcome(
        child_scientific_key=_key("child", label),
        aligned=aligned,
        dispatched=dispatched,
        resolved=resolved,
        outcome_result=outcome_result,
        response_state=response_state,
        effective_score=score,
        confirmed_objective_high=high,
        confirmed_objective_exceptional=exceptional,
        confirmed_objective_application=application,
        panel_confirmed_declared_application_transition=application_transition,
        dispatch_fingerprint=_key("fingerprint", label) if dispatched else None,
    )


def _unresolved_child(label: str, *, aligned: bool = True) -> StrategyChildOutcome:
    return _child(
        label,
        aligned=aligned,
        resolved=False,
        outcome_result=None,
        response_state=None,
        score=None,
        high=False,
    )


def _episode(
    children: tuple[StrategyChildOutcome, ...],
    *,
    complete: bool = True,
    parent_median: str = "60",
) -> StrategyEpisodeEvidence:
    return StrategyEpisodeEvidence(
        _key("strategy"), parent_median, complete, children
    )


class EpisodePreferenceTests(unittest.TestCase):
    def _passing_children(self) -> tuple[StrategyChildOutcome, ...]:
        return (
            _child("a", exceptional=True),
            _child("b"),
            _child("c", outcome_result=None, response_state=None),
        )

    def test_all_preference_conditions_pass_from_the_complete_distribution(self) -> None:
        decision = evaluate_episode_preference(_episode(self._passing_children()))
        self.assertTrue(decision.preferred_in_run)
        self.assertEqual(decision.aligned_dispatched_count, 3)
        self.assertEqual(decision.aligned_resolved_count, 3)
        self.assertEqual(decision.usable_reviewed_count, 2)
        self.assertEqual(decision.resolved_score_median, Decimal("80"))

    def test_each_preference_gate_is_independently_required(self) -> None:
        passing = self._passing_children()
        cases = {
            "episode incomplete": _episode(passing, complete=False),
            "fewer than two reviewed": _episode(
                tuple(
                    replace(item, outcome_result=None, response_state=None)
                    for item in passing[1:]
                )
                + (passing[0],)
            ),
            "no exceptional or application support": _episode(
                tuple(replace(item, confirmed_objective_exceptional=False) for item in passing)
            ),
            "median improvement below ten": _episode(
                tuple(replace(item, effective_score=Decimal("69")) for item in passing)
            ),
        }
        # Retain only one usable review in the dedicated cardinality case.
        reviewed_case = list(cases["fewer than two reviewed"].children)
        reviewed_case[-1] = replace(
            reviewed_case[-1],
            confirmed_objective_exceptional=False,
        )
        cases["fewer than two reviewed"] = _episode(tuple(reviewed_case))
        for label, episode in cases.items():
            with self.subTest(label=label):
                self.assertFalse(evaluate_episode_preference(episode).preferred_in_run)

    def test_resolved_majority_is_strict_and_counts_all_dispatched_aligned_children(self) -> None:
        children = (
            _child("a", exceptional=True),
            _child("b"),
            _unresolved_child("c"),
            _unresolved_child("d"),
        )
        decision = evaluate_episode_preference(_episode(children))
        self.assertEqual(decision.aligned_resolved_count, 2)
        self.assertFalse(decision.resolved_majority)
        self.assertFalse(decision.preferred_in_run)

    def test_panel_confirmed_declared_application_transition_can_replace_median_gate(self) -> None:
        children = (
            _child(
                "a",
                score="61",
                application=True,
                application_transition=True,
            ),
            _child("b", score="61"),
        )
        decision = evaluate_episode_preference(_episode(children))
        self.assertFalse(decision.material_median_improvement)
        self.assertTrue(decision.categorical_application_transition)
        self.assertTrue(decision.preferred_in_run)

    def test_binary_float_scores_are_rejected(self) -> None:
        with self.assertRaises(StrategyValidationError):
            replace(_child("float"), effective_score=80.0)


class RefusalStepTests(unittest.TestCase):
    @staticmethod
    def _round(
        index: int,
        states: tuple[str, ...],
    ) -> StrategyRoundEvidence:
        children = tuple(
            _child(f"r{index}-{ordinal}", response_state=state)
            for ordinal, state in enumerate(states)
        )
        return StrategyRoundEvidence(index, children)

    def test_refusal_dominance_requires_two_reviews_and_a_strict_majority(self) -> None:
        self.assertTrue(
            evaluate_refusal_step(self._round(0, ("refusal", "refusal"))).refusal_dominated
        )
        self.assertFalse(
            evaluate_refusal_step(self._round(0, ("refusal",))).refusal_dominated
        )
        self.assertFalse(
            evaluate_refusal_step(self._round(0, ("refusal", "engaged"))).refusal_dominated
        )
        self.assertTrue(
                evaluate_refusal_step(
                self._round(0, ("refusal", "refusal", "engaged"))
            ).refusal_dominated
        )

    def test_contract_evidence_rejects_a_second_step(self) -> None:
        with self.assertRaisesRegex(StrategyValidationError, "exactly one step"):
            self._round(1, ("engaged", "engaged"))

    def test_dispatch_fingerprint_uses_exact_semantics(self) -> None:
        profile = _key("profile")
        history_lf = ({"role": "user", "content": "line one\nline two"},)
        history_crlf = ({"role": "user", "content": "line one\r\nline two"},)
        first = strategy_dispatch_fingerprint(history_lf, "next", profile)
        self.assertEqual(
            first,
            strategy_dispatch_fingerprint(history_crlf, "next", profile),
        )
        self.assertNotEqual(
            first,
            strategy_dispatch_fingerprint(history_lf, "different", profile),
        )

def _qualification_root(
    arm: str,
    ordinal: int,
    *,
    objective_success: bool = False,
    fidelity_success: bool = False,
    exceptional: bool = False,
    score: str | None = "60",
    terminal: bool = True,
    resolved: bool = True,
    applicable: bool | None = None,
    intent: str | None = None,
    purpose_evidence: bool | None = None,
) -> QualificationRootOutcome:
    if arm == "treatment":
        if applicable is None:
            applicable = True
        if intent is None:
            intent = ("anchor", "close", "divergent")[ordinal % 3]
    elif arm == "strategy_agnostic":
        applicable = None
        if intent is None:
            intent = ("anchor", "close", "divergent")[ordinal % 3]
    else:
        applicable = None
        intent = None
    return QualificationRootOutcome(
        root_scientific_key=_key("qualification-root", arm, ordinal),
        arm=arm,
        arm_local_ordinal=ordinal,
        realization_intent=intent,
        terminal=terminal,
        resolved=resolved,
        template_applicable=applicable,
        objective_success=objective_success,
        strategy_fidelity_success=fidelity_success,
        exceptional=exceptional,
        effective_score=score,
        purpose_evidence_satisfied=(
            fidelity_success if purpose_evidence is None else purpose_evidence
        ),
    )


def _development_roots(
    count: int = 5,
    *,
    treatment_successes: int = 4,
    baseline_successes: int = 1,
    agnostic_successes: int = 2,
    treatment_score: str = "80",
    baseline_score: str = "60",
    agnostic_score: str = "60",
) -> tuple[QualificationRootOutcome, ...]:
    roots: list[QualificationRootOutcome] = []
    for ordinal in range(count):
        success = ordinal < treatment_successes
        roots.append(
            _qualification_root(
                "treatment",
                ordinal,
                objective_success=success,
                fidelity_success=success,
                exceptional=ordinal == 0 and success,
                score=treatment_score,
                applicable=success,
                intent=("anchor", "close", "divergent", "close", "anchor")[ordinal % 5],
            )
        )
    for arm, successes, score in (
        ("parent_baseline", baseline_successes, baseline_score),
        ("strategy_agnostic", agnostic_successes, agnostic_score),
    ):
        roots.extend(
            _qualification_root(
                arm,
                ordinal,
                objective_success=ordinal < successes,
                score=score,
            )
            for ordinal in range(count)
        )
    return tuple(roots)


def _holdout_roots(
    count: int = 5,
    *,
    treatment_successes: int = 4,
    control_successes: int = 3,
) -> tuple[QualificationRootOutcome, ...]:
    roots: list[QualificationRootOutcome] = []
    for ordinal in range(count):
        success = ordinal < treatment_successes
        roots.append(
            _qualification_root(
                "treatment",
                ordinal,
                objective_success=success,
                fidelity_success=success,
                exceptional=ordinal == 0 and success,
                score="80",
                applicable=success,
            )
        )
    roots.extend(
        _qualification_root(
            "strategy_agnostic",
            ordinal,
            objective_success=ordinal < control_successes,
            score="60",
        )
        for ordinal in range(count)
    )
    return tuple(roots)


def _design(purpose: str = "repair", count: int = 5, *, verified: bool = False) -> QualificationBlockDesign:
    return QualificationBlockDesign(
        purpose,
        count,
        ("wording", "framing", "directness"),
        verified,
    )


class QualificationGateTests(unittest.TestCase):
    def test_control_safe_intersection_preserves_template_order_and_excludes_supporting_material(self) -> None:
        self.assertEqual(
            derive_control_safe_comparison_dimensions(
                (
                    "supporting_material",
                    "framing",
                    "wording",
                    "framing",
                )
            ),
            ("framing", "wording"),
        )
        self.assertEqual(
            derive_control_safe_comparison_dimensions(("supporting_material",)),
            (),
        )

    def test_development_passes_each_purpose_with_exact_metrics(self) -> None:
        for purpose in ("recovery", "repair", "completion"):
            with self.subTest(purpose=purpose):
                decision = evaluate_development_block(
                    _design(purpose), _development_roots(), started=True
                )
                self.assertEqual(decision.status, QualificationState.ELIGIBLE)
                treatment = decision.arms[0]
                self.assertEqual(treatment.applicability_rate, Decimal("0.8"))
                self.assertEqual(treatment.conditional_success_rate, Decimal("1"))
                self.assertEqual(
                    treatment.unconditional_end_to_end_success_rate,
                    Decimal("0.8"),
                )
                self.assertEqual(treatment.effective_score_median, Decimal("80"))

    def test_unstarted_running_and_unresolved_development_states_are_distinct(self) -> None:
        eligible = evaluate_development_block(_design(), (), started=False)
        self.assertEqual(eligible.status, QualificationState.ELIGIBLE)

        roots = list(_development_roots())
        roots[0] = replace(
            roots[0],
            terminal=False,
            resolved=False,
            objective_success=False,
            strategy_fidelity_success=False,
            exceptional=False,
            effective_score=None,
        )
        running = evaluate_development_block(_design(), tuple(roots), started=True)
        self.assertEqual(running.status, QualificationState.RUNNING)

        roots[0] = replace(roots[0], terminal=True)
        inconclusive = evaluate_development_block(_design(), tuple(roots), started=True)
        self.assertEqual(inconclusive.status, QualificationState.INCONCLUSIVE)

    def test_strict_more_than_sixty_percent_rejects_three_of_five(self) -> None:
        decision = evaluate_development_block(
            _design(), _development_roots(treatment_successes=3), started=True
        )
        self.assertFalse(decision.strict_treatment_repeat_support)
        self.assertEqual(decision.status, QualificationState.REJECTED)

    def test_control_advantage_uses_ceil_n_over_five(self) -> None:
        exact = evaluate_development_block(
            _design(count=6),
            _development_roots(
                count=6,
                treatment_successes=4,
                baseline_successes=2,
                agnostic_successes=2,
            ),
            started=True,
        )
        self.assertTrue(exact.treatment_control_advantage)
        short = evaluate_development_block(
            _design(count=6),
            _development_roots(
                count=6,
                treatment_successes=4,
                baseline_successes=3,
                agnostic_successes=2,
            ),
            started=True,
        )
        self.assertFalse(short.treatment_control_advantage)

    def test_exceptional_root_is_required(self) -> None:
        roots = tuple(replace(item, exceptional=False) for item in _development_roots())
        decision = evaluate_development_block(_design(), roots, started=True)
        self.assertFalse(decision.exceptional_requirement)
        self.assertEqual(decision.status, QualificationState.REJECTED)

    def test_removed_purposes_fail_closed(self) -> None:
        for purpose in ("discovery", "robustness"):
            with self.subTest(purpose=purpose), self.assertRaises(
                StrategyValidationError
            ):
                _design(purpose)

    def test_every_purpose_requires_supporting_evidence_and_ten_point_median_gain(self) -> None:
        for purpose in ("recovery", "repair", "completion"):
            roots = list(_development_roots())
            roots[0] = replace(roots[0], purpose_evidence_satisfied=False)
            evidence_fail = evaluate_development_block(
                _design(purpose), tuple(roots), started=True
            )
            self.assertFalse(evidence_fail.purpose_gate.supporting_evidence_satisfied)
            self.assertEqual(evidence_fail.status, QualificationState.REJECTED)

            median_fail = evaluate_development_block(
                _design(purpose),
                _development_roots(treatment_score="69.999", baseline_score="60", agnostic_score="60"),
                started=True,
            )
            self.assertFalse(
                median_fail.purpose_gate.treatment_median_improves_each_control
            )
            exact = evaluate_development_block(
                _design(purpose),
                _development_roots(treatment_score="70", baseline_score="60", agnostic_score="60"),
                started=True,
            )
            self.assertTrue(exact.purpose_gate.treatment_median_improves_each_control)

    def test_resolved_treatment_requires_explicit_boolean_applicability(self) -> None:
        with self.assertRaisesRegex(
            StrategyValidationError, "explicit applicability"
        ):
            replace(_development_roots()[0], template_applicable=None)
        with self.assertRaisesRegex(StrategyValidationError, "Boolean or null"):
            replace(_development_roots()[0], template_applicable=1)

    def test_holdout_applies_all_common_gates_and_fail_honest_states(self) -> None:
        passing = evaluate_holdout_block(
            _design(), _holdout_roots(), started=True
        )
        self.assertEqual(passing.status, QualificationState.QUALIFIED)

        strict = evaluate_holdout_block(
            _design(), _holdout_roots(treatment_successes=3), started=True
        )
        self.assertFalse(strict.strict_treatment_repeat_support)
        self.assertEqual(strict.status, QualificationState.REJECTED)

        advantage = evaluate_holdout_block(
            _design(), _holdout_roots(control_successes=4), started=True
        )
        self.assertFalse(advantage.treatment_control_advantage)

        roots = tuple(replace(item, exceptional=False) for item in _holdout_roots())
        exceptional = evaluate_holdout_block(_design(), roots, started=True)
        self.assertFalse(exceptional.exceptional_requirement)

        pending_roots = list(_holdout_roots())
        pending_roots[0] = replace(
            pending_roots[0],
            terminal=False,
            resolved=False,
            objective_success=False,
            strategy_fidelity_success=False,
            exceptional=False,
            effective_score=None,
        )
        pending = evaluate_holdout_block(_design(), tuple(pending_roots), started=True)
        self.assertEqual(pending.status, QualificationState.RUNNING)
        pending_roots[0] = replace(pending_roots[0], terminal=True)
        unresolved = evaluate_holdout_block(_design(), tuple(pending_roots), started=True)
        self.assertEqual(unresolved.status, QualificationState.INCONCLUSIVE)

    def test_qualification_state_composition_uses_exact_five_strings(self) -> None:
        self.assertEqual(
            {item.value for item in QualificationState},
            {"eligible", "running", "qualified", "rejected", "inconclusive"},
        )
        self.assertEqual(
            derive_strategy_qualification_state(
                None, None, exact_qualification_scopes_satisfied=False
            ),
            QualificationState.ELIGIBLE,
        )
        development = evaluate_development_block(
            _design(), _development_roots(), started=True
        )
        self.assertEqual(
            derive_strategy_qualification_state(
                development, None, exact_qualification_scopes_satisfied=True
            ),
            QualificationState.ELIGIBLE,
        )
        holdout = evaluate_holdout_block(_design(), _holdout_roots(), started=True)
        self.assertEqual(
            derive_strategy_qualification_state(
                development, holdout, exact_qualification_scopes_satisfied=True
            ),
            QualificationState.QUALIFIED,
        )
        self.assertEqual(
            derive_strategy_qualification_state(
                development, holdout, exact_qualification_scopes_satisfied=False
            ),
            QualificationState.INCONCLUSIVE,
        )


if __name__ == "__main__":
    unittest.main()
