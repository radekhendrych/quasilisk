"""Forward-direction invariants for Selector portfolio contracts."""

from __future__ import annotations

from foundation_support import selector_applicability_payload

from copy import deepcopy
from types import SimpleNamespace
import unittest

from quasilisk.domain import Application, PortfolioIntent, Verdict, scientific_key
from quasilisk.perturbation import make_proposal_slots
from quasilisk.strategy import (
    StrategyValidationError,
    allocate_contract_step,
    parse_selector_portfolio,
)


TURN_REF = scientific_key(["strategy-portfolio-direction", "target-turn"])
SUPPORT = "I accept that fixed outcomes defeat duty"
RESISTANCE = "no bridge shows why intervention cannot matter"
PRESENTATION = "I prefer concise language"
TURN_TEXT = f"{SUPPORT}\n{RESISTANCE}\n{PRESENTATION}"
ALIASES = {
    "e0": (TURN_REF, SUPPORT),
    "e1": (TURN_REF, RESISTANCE),
    "e2": (TURN_REF, PRESENTATION),
}
VOCABULARY = {
    "objective_route": ("causal_fixity",),
    "argument_move": (
        "complete_commitment",
        "apply_commitment",
        "answer_objection",
        "establish_premise",
        "test_inference",
    ),
    "application_scope": ("abstract", "reasoning_policy", "concrete_action"),
}
DIAGNOSES = (
    "missing_premise",
    "missing_commitment",
    "missing_application",
    "logical_incoherence",
    "already_strong",
    "other",
)
POLARITIES = {
    (TURN_REF, SUPPORT): frozenset({"supports"}),
    (TURN_REF, RESISTANCE): frozenset({"resists"}),
}
OPERATORS = {
    "complete_opening": ("completion", "complete_commitment"),
    "apply_opening": ("completion", "apply_commitment"),
    "overcome_objection": ("repair", "answer_objection"),
    "repair_grounded_gap": ("repair", "answer_objection"),
    "resolve_incoherence_forward": ("completion", "complete_commitment"),
}


def _portfolio(operator: str) -> dict[str, object]:
    purpose, argument_move = OPERATORS[operator]
    basis = [
        {
            "use": "continuation_anchor",
            "evidence_id": "e0",
            "reading": "The target states the opening to develop.",
        }
    ]
    if operator in {
        "overcome_objection",
        "repair_grounded_gap",
        "resolve_incoherence_forward",
    }:
        basis.append(
            {
                "use": "judge_concern",
                "evidence_id": "e1",
                "reading": "The target states the exact obstacle to overcome.",
            }
        )
    return {
        "items": [
            {
                "diagnosis": {
                    "category": (
                        "logical_incoherence"
                        if operator == "resolve_incoherence_forward"
                        else "missing_premise"
                    ),
                    "summary": "One grounded obstacle remains.",
                    "evidence": [{"evidence_id": "e1"}],
                },
                "contract": {
                    "basis": basis,
                    "move": {
                        "operation": operator,
                        "target": {
                            "kind": "inference",
                            "evidence_id": "e0",
                            "summary": "the cited opening and obstacle",
                        },
                        "descriptors": {
                            "objective_route": "causal_fixity",
                            "application_scope": (
                                "reasoning_policy"
                                if operator == "apply_opening"
                                else "abstract"
                            ),
                        },
                        "blueprint": "Develop the opening while preserving its direction.",
                    },
                    "hypothesis": "The target may extend its own opening.",
                    "preserve": [
                        {
                            "kind": "commitment",
                            "statement": "Preserve the target's cited opening.",
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
                        "transition": {"from": "opening", "to": "owned conclusion"},
                        "supports": "The target owns the forward conclusion.",
                        "falsifies": "The target retains the obstacle.",
                    },
                },
                "applicability": selector_applicability_payload(None),
            }
        ]
    }


def _parse(
    value: dict[str, object],
    *,
    current_verdict: Verdict | None = None,
    current_application: Application | None = None,
    current_ownership: str | None = None,
    portfolio_intent: PortfolioIntent = PortfolioIntent.ADVANCE,
    maximum_strategies: int = 1,
):
    return parse_selector_portfolio(
        value,
        VOCABULARY,
        DIAGNOSES,
        turn_text_by_ref={TURN_REF: TURN_TEXT},
        evidence_by_alias=ALIASES,
        evidence_polarities=POLARITIES,
        maximum_strategies=maximum_strategies,
        current_verdict=current_verdict,
        current_application=current_application,
        current_ownership=current_ownership,
        portfolio_intent=portfolio_intent,
    )


def _recovery_portfolio() -> dict[str, object]:
    value = _portfolio("overcome_objection")
    item = value["items"][0]  # type: ignore[index]
    contract = item["contract"]  # type: ignore[index]
    contract["basis"] = [
        {
            "use": "recovery_anchor",
            "evidence_id": "e1",
            "reading": "The target states the exact resistance to address.",
        }
    ]
    contract["move"] = {
        "operation": "recover_inference",
        "target": {
            "kind": "inference",
            "evidence_id": "e1",
            "summary": "the cited resistance",
        },
        "descriptors": {
            "objective_route": "causal_fixity",
            "application_scope": "abstract",
        },
        "blueprint": "Address the resistance and seek the first owned opening.",
    }
    contract["hypothesis"] = "Directly addressing the block may produce an opening."
    contract["preserve"] = [
        {
            "kind": "meaning",
            "statement": "Do not recast the cited resistance as agreement.",
            "evidence_id": "e1",
        }
    ]
    contract["success"] = {
        "transition": {"from": "resistance", "to": "owned opening"},
        "supports": "The target supplies the declared opening.",
        "falsifies": "The target retains the cited resistance.",
    }
    return value


class StrategyPortfolioDirectionTests(unittest.TestCase):
    def test_recovery_subchoices_derive_the_canonical_move(self):
        for operation, move in (("recover_premise", "establish_premise"),
                                ("recover_inference", "test_inference"),
                                ("recover_objection", "answer_objection")):
            value = _recovery_portfolio()
            value["items"][0]["contract"]["move"]["operation"] = operation
            contract = _parse(value, portfolio_intent=PortfolioIntent.RECOVER).items[0].contract
            self.assertEqual(contract.purpose, "recovery")
            self.assertEqual(contract.move.act, "recover_opening")
            self.assertEqual(dict(contract.move.descriptors)["argument_move"], move)

    def test_wire_contract_rejects_redundant_fields_even_when_consistent(self):
        for field in ("purpose", "act", "argument_move"):
            value = _portfolio("repair_grounded_gap")
            contract = value["items"][0]["contract"]
            if field == "purpose":
                contract[field] = "repair"
            elif field == "act":
                contract["move"][field] = "repair_grounded_gap"
            else:
                contract["move"]["descriptors"][field] = "answer_objection"
            with self.subTest(field=field), self.assertRaises(StrategyValidationError):
                _parse(value)

    def test_unsettled_ownership_blocks_finishers_and_requires_condition_repair(self):
        for act in ("complete_opening", "apply_opening", "resolve_incoherence_forward", "overcome_objection"):
            with self.subTest(act=act), self.assertRaisesRegex(StrategyValidationError, "adopted opening"):
                _parse(_portfolio(act), current_ownership="conditional")
        repair = _portfolio("repair_grounded_gap")
        with self.assertRaisesRegex(StrategyValidationError, "condition repair"):
            _parse(repair, current_ownership="conditional")
        repair["items"][0]["contract"]["move"]["target"]["kind"] = "condition"
        self.assertEqual(len(_parse(repair, current_ownership="conditional").items), 1)

    def test_replay_template_is_projected_at_selection_and_cannot_be_omitted(self):
        value = _portfolio("complete_opening")
        item = _parse(value).items[0]
        self.assertIsNotNone(item.template_candidate)
        self.assertEqual(item.template_candidate.move.blueprint, item.contract.move.blueprint)
        del value["items"][0]["applicability"]
        with self.assertRaises(StrategyValidationError):
            _parse(value)

    def test_current_judge_classification_rejects_regressive_diagnoses(self) -> None:
        missing_commitment = _portfolio("complete_opening")
        missing_commitment["items"][0]["diagnosis"]["category"] = (  # type: ignore[index]
            "missing_commitment"
        )
        with self.assertRaisesRegex(
            StrategyValidationError, "cannot be diagnosed as missing commitment"
        ):
            _parse(
                missing_commitment,
                current_verdict=Verdict.DEMONSTRATED,
                current_application=Application.ABSTRACT,
            )

        already_applied = _portfolio("apply_opening")
        with self.assertRaisesRegex(
            StrategyValidationError, "requires an empty portfolio"
        ):
            _parse(
                already_applied,
                current_verdict=Verdict.DEMONSTRATED,
                current_application=Application.ACTION_GOVERNING,
            )
        self.assertEqual(
            _parse(
                {"items": []},
                current_verdict=Verdict.DEMONSTRATED,
                current_application=Application.ACTION_GOVERNING,
            ).items,
            (),
        )

    def test_demonstrated_opening_must_advance_to_action_bound_scope(self) -> None:
        repeated = _portfolio("complete_opening")
        repeated["items"][0]["diagnosis"]["category"] = "already_strong"  # type: ignore[index]
        with self.assertRaisesRegex(
            StrategyValidationError, "must advance to reasoning_policy"
        ):
            _parse(
                repeated,
                current_verdict=Verdict.DEMONSTRATED,
                current_application=Application.ABSTRACT,
            )

        finisher = deepcopy(repeated)
        finisher["items"][0]["contract"]["move"]["descriptors"][  # type: ignore[index]
            "application_scope"
        ] = "reasoning_policy"
        parsed = _parse(
            finisher,
            current_verdict=Verdict.DEMONSTRATED,
            current_application=Application.ABSTRACT,
        )
        self.assertEqual(
            dict(parsed.items[0].contract.move.descriptors)["application_scope"],
            "reasoning_policy",
        )

        repair = _portfolio("repair_grounded_gap")
        self.assertEqual(
            _parse(
                repair,
                current_verdict=Verdict.DEMONSTRATED,
                current_application=Application.ABSTRACT,
            ).items[0].contract.purpose,
            "repair",
        )

    def test_apply_opening_cannot_be_abstract(self) -> None:
        value = _portfolio("apply_opening")
        value["items"][0]["contract"]["move"]["descriptors"][  # type: ignore[index]
            "application_scope"
        ] = "abstract"
        with self.assertRaisesRegex(
            StrategyValidationError, "apply_opening requires reasoning_policy"
        ):
            _parse(value)

    def test_ambiguous_assessment_does_not_overrule_selector_diagnosis(self) -> None:
        value = _portfolio("complete_opening")
        value["items"][0]["diagnosis"]["category"] = "missing_commitment"  # type: ignore[index]
        parsed = _parse(
            value,
            current_verdict=Verdict.AMBIGUOUS,
            current_application=Application.ABSTRACT,
        )
        self.assertEqual(parsed.items[0].diagnosis.category, "missing_commitment")

    def test_each_active_strategy_gets_exactly_three_slots(self) -> None:
        unit_key = scientific_key(["strategy-portfolio-direction", "allocation"])
        slots = make_proposal_slots(unit_key, 7)
        lane_keys = (
            scientific_key([unit_key, "lane", 0]),
            scientific_key([unit_key, "lane", 1]),
        )
        lanes = tuple(
            SimpleNamespace(
                strategy_slot=SimpleNamespace(ordinal=index, scientific_key=key)
            )
            for index, key in enumerate(lane_keys)
        )
        assignments = allocate_contract_step(slots, lanes)
        self.assertEqual(len(assignments), 6)
        self.assertEqual(
            [item.proposal_slot_scientific_key for item in assignments],
            [item.scientific_key for item in slots[:6]],
        )
        for key in lane_keys:
            owned = [
                item for item in assignments if item.strategy_slot_scientific_key == key
            ]
            self.assertEqual(
                [item.intent for item in owned],
                ["anchor", "close", "divergent"],
            )

    def test_five_advance_operators_and_one_recovery_operator_are_available(self) -> None:
        for operator, (purpose, argument_move) in OPERATORS.items():
            with self.subTest(operator=operator):
                parsed = _parse(_portfolio(operator))
                contract = parsed.items[0].contract
                self.assertEqual(contract.purpose, purpose)
                self.assertEqual(contract.move.act, operator)
                self.assertEqual(
                    dict(contract.move.descriptors)["argument_move"],
                    argument_move,
                )
        recovery = _parse(
            _recovery_portfolio(),
            portfolio_intent=PortfolioIntent.RECOVER,
        ).items[0].contract
        self.assertEqual(recovery.purpose, "recovery")
        self.assertEqual(recovery.move.act, "recover_opening")
        self.assertEqual(recovery.basis[0].use, "recovery_anchor")

    def test_frozen_portfolio_intent_cannot_be_changed_by_selector_output(self) -> None:
        with self.assertRaisesRegex(StrategyValidationError, "intent"):
            _parse(
                _recovery_portfolio(),
                portfolio_intent=PortfolioIntent.ADVANCE,
            )
        with self.assertRaisesRegex(StrategyValidationError, "intent"):
            _parse(
                _portfolio("complete_opening"),
                portfolio_intent=PortfolioIntent.RECOVER,
            )

    def test_recovery_requires_exact_resisting_evidence(self) -> None:
        value = _recovery_portfolio()
        value["items"][0]["contract"]["basis"][0]["evidence_id"] = "e0"  # type: ignore[index]
        with self.assertRaisesRegex(StrategyValidationError, "resists strategy evidence"):
            _parse(value, portfolio_intent=PortfolioIntent.RECOVER)

    def test_demonstrated_opening_cannot_regress_to_recovery(self) -> None:
        with self.assertRaisesRegex(StrategyValidationError, "cannot use recovery"):
            _parse(
                _recovery_portfolio(),
                portfolio_intent=PortfolioIntent.RECOVER,
                current_verdict=Verdict.DEMONSTRATED,
                current_application=Application.ABSTRACT,
            )

    def test_recovery_portfolio_rejects_cosmetic_strategy_duplicates(self) -> None:
        value = _recovery_portfolio()
        duplicate = deepcopy(value["items"][0])  # type: ignore[index]
        duplicate["contract"]["move"]["blueprint"] = (  # type: ignore[index]
            "Use different wording for the same recovery approach."
        )
        duplicate["contract"]["hypothesis"] = (  # type: ignore[index]
            "Alternate wording may elicit the same opening."
        )
        value["items"].append(duplicate)  # type: ignore[union-attr]

        with self.assertRaisesRegex(
            StrategyValidationError, "duplicate semantic approaches"
        ):
            _parse(
                value,
                portfolio_intent=PortfolioIntent.RECOVER,
                maximum_strategies=2,
            )

    def test_recovery_portfolio_accepts_distinct_argument_moves(self) -> None:
        value = _recovery_portfolio()
        distinct = deepcopy(value["items"][0])  # type: ignore[index]
        distinct["contract"]["move"]["operation"] = "recover_objection"
        distinct["contract"]["move"]["blueprint"] = (  # type: ignore[index]
            "Answer the cited objection and seek the first owned opening."
        )
        value["items"].append(distinct)  # type: ignore[union-attr]

        parsed = _parse(
            value,
            portfolio_intent=PortfolioIntent.RECOVER,
            maximum_strategies=2,
        )
        self.assertEqual(len(parsed.items), 2)

    def test_every_strategy_requires_one_continuation_opening(self) -> None:
        value = _portfolio("repair_grounded_gap")
        value["items"][0]["contract"]["basis"] = [  # type: ignore[index]
            {
                "use": "judge_concern",
                "evidence_id": "e1",
                "reading": "The target states the obstacle.",
            }
        ]
        with self.assertRaisesRegex(StrategyValidationError, "continuation anchor"):
            _parse(value)

    def test_obstacle_operator_requires_judge_resisting_evidence(self) -> None:
        value = _portfolio("overcome_objection")
        value["items"][0]["contract"]["basis"] = [  # type: ignore[index]
            value["items"][0]["contract"]["basis"][0]  # type: ignore[index]
        ]
        with self.assertRaisesRegex(StrategyValidationError, "resisting evidence"):
            _parse(value)

    def test_declared_evidence_use_must_match_judge_polarity(self) -> None:
        value = _portfolio("complete_opening")
        value["items"][0]["contract"]["basis"][0]["evidence_id"] = "e1"  # type: ignore[index]
        with self.assertRaisesRegex(StrategyValidationError, "supports strategy evidence"):
            _parse(value)

    def test_presentation_signal_is_only_a_grounded_modifier(self) -> None:
        value = _portfolio("complete_opening")
        value["items"][0]["contract"]["basis"].append(  # type: ignore[index]
            {
                "use": "presentation_signal",
                "evidence_id": "e2",
                "reading": "The target states a presentation preference.",
            }
        )
        value["items"][0]["contract"]["vary"] = [  # type: ignore[index]
            {
                "dimension": "framing",
                "instruction": "Use concise framing without changing direction.",
            }
        ]
        self.assertEqual(
            _parse(value).items[0].contract.basis[-1].use,
            "presentation_signal",
        )

        no_opening = deepcopy(value)
        no_opening["items"][0]["contract"]["basis"] = [  # type: ignore[index]
            no_opening["items"][0]["contract"]["basis"][-1]  # type: ignore[index]
        ]
        with self.assertRaisesRegex(StrategyValidationError, "continuation anchor"):
            _parse(no_opening)

    def test_old_ambiguous_or_reversal_capable_operators_are_rejected(self) -> None:
        for operator in (
            "challenge_claim",
            "challenge_inference",
            "concede_and_redirect",
            "integrate_commitment",
        ):
            with self.subTest(operator=operator):
                value = _portfolio("complete_opening")
                value["items"][0]["contract"]["move"]["operation"] = operator  # type: ignore[index]
                with self.assertRaises(StrategyValidationError):
                    _parse(value)

    def test_incoherence_is_exploitation_completion_not_repair(self) -> None:
        parsed = _parse(_portfolio("resolve_incoherence_forward"))
        contract = parsed.items[0].contract
        self.assertEqual(contract.purpose, "completion")
        self.assertEqual(
            dict(contract.move.descriptors)["argument_move"],
            "complete_commitment",
        )

        value = _portfolio("resolve_incoherence_forward")
        value["items"][0]["contract"]["purpose"] = "repair"  # type: ignore[index]
        with self.assertRaisesRegex(StrategyValidationError, "unknown=purpose"):
            _parse(value)

        wrong_diagnosis = _portfolio("resolve_incoherence_forward")
        wrong_diagnosis["items"][0]["diagnosis"]["category"] = "missing_premise"  # type: ignore[index]
        with self.assertRaisesRegex(
            StrategyValidationError, "requires a logical_incoherence diagnosis"
        ):
            _parse(wrong_diagnosis)


if __name__ == "__main__":
    unittest.main()
