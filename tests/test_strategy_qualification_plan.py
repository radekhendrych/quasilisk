"""Conformance tests for the ID-free strategy-qualification holdout seal."""

from __future__ import annotations

from copy import deepcopy
import inspect
import re
import unittest

from quasilisk.domain import digest_json, entity_id, scientific_key
from quasilisk.strategy import (
    DEVELOPMENT_ARMS,
    HOLDOUT_ARMS,
    QUALIFICATION_ENDPOINT_RULE,
    FutureRootSpec,
    HoldoutInformationMasks,
    SealedHoldoutPlan,
    StrategyValidationError,
    build_sealed_holdout_plan,
    derive_template_comparison_dimensions,
    freeze_development_arm_order,
    freeze_holdout_arm_order,
    freeze_qualification_intent_assignment,
    parse_sealed_holdout_plan,
    parse_strategy_template,
    project_materialized_holdout_plan,
    validate_materialized_holdout_plan,
)


VOCABULARY = {
    "objective_route": ("route", "alternate"),
    "argument_move": ("answer_objection", "test_inference"),
    "application_scope": ("reasoning_policy", "action_policy"),
}
MEASUREMENT_CELL = {
    "objective_route": "route",
    "argument_move": "answer_objection",
    "application_scope": "reasoning_policy",
}


def _key(*parts: object) -> str:
    return scientific_key(["qualification-plan-test", *parts])


def _template(*, safe_dimensions: bool = True):
    vary = (
        [
            {"dimension": "supporting_material", "instruction": "Vary supporting context."},
            {"dimension": "framing", "instruction": "Vary framing."},
            {"dimension": "wording", "instruction": "Vary wording."},
        ]
        if safe_dimensions
        else [
            {"dimension": "supporting_material", "instruction": "Vary supporting context."}
        ]
    )
    return parse_strategy_template(
        {
            "purpose": "repair",
            "applicability": {
                "diagnosis": "conditional_acceptance",
                "requires": "A stated condition remains unresolved.",
                "excludes": "No condition remains.",
            },
            "move": {
                "act": "repair_grounded_gap",
                "target_kind": "condition",
                "descriptors": MEASUREMENT_CELL,
                "blueprint": "Resolve the cited condition without changing the premise.",
            },
            "hypothesis": "Resolving the condition should permit progress.",
            "preserve": [{"kind": "meaning", "statement": "Preserve the accepted meaning."}],
            "vary": vary,
            "success": {
                "transition": {"from": "conditional", "to": "resolved"},
                "supports": "The target resolves its stated condition.",
                "falsifies": "The condition remains unresolved.",
            },
        },
        VOCABULARY,
        ("conditional_acceptance", "other"),
    )


def _seal(*, count: int = 5, seed: int = 29, subject: str | None = None) -> SealedHoldoutPlan:
    return build_sealed_holdout_plan(
        _template(),
        planned_count_per_arm=count,
        random_seed=seed,
        qualification_subject_scientific_key=subject or _key("qualification-subject"),
        measurement_cell=MEASUREMENT_CELL,
        descriptor_vocabulary=VOCABULARY,
    )


def _materialized(plan: SealedHoldoutPlan) -> dict[str, object]:
    semantic = plan.to_dict()
    roots: list[dict[str, object]] = []
    root_plan_id_by_ordinal: dict[int, str] = {}
    for item in semantic["future_roots"]:
        assert isinstance(item, dict)
        ordinal = item["root_ordinal"]
        assert isinstance(ordinal, int)
        execution_key = _key("execution", ordinal)
        root_plan_id = entity_id("rp", ["qualification-plan-test", ordinal])
        root_plan_id_by_ordinal[ordinal] = root_plan_id
        roots.append(
            {
                "execution_id": entity_id("ex", ["qualification-plan-test", ordinal]),
                "execution_scientific_key": execution_key,
                "root_plan_id": root_plan_id,
                **item,
            }
        )
    return {
        "source_sealed_holdout_digest": plan.content_digest,
        "planned_count_per_arm": semantic["planned_count_per_arm"],
        "comparison_dimensions": semantic["comparison_dimensions"],
        "future_roots": roots,
        "interleaved_root_order": [
            root_plan_id_by_ordinal[ordinal]
            for ordinal in semantic["interleaved_root_order"]
        ],
        "information_masks": semantic["information_masks"],
        "endpoint_rule": semantic["endpoint_rule"],
        "measurement_cell": semantic["measurement_cell"],
    }


class QualificationFreezeTests(unittest.TestCase):
    def test_template_intersection_is_ordered_distinct_and_control_safe(self) -> None:
        self.assertEqual(
            derive_template_comparison_dimensions(_template()),
            ("framing", "wording"),
        )
        with self.assertRaisesRegex(StrategyValidationError, "control-safe"):
            build_sealed_holdout_plan(
                _template(safe_dimensions=False),
                planned_count_per_arm=5,
                random_seed=1,
                qualification_subject_scientific_key=_key("subject"),
                measurement_cell=MEASUREMENT_CELL,
                descriptor_vocabulary=VOCABULARY,
            )

    def test_seeded_round_robin_intents_are_balanced_shared_and_repeatable(self) -> None:
        first = freeze_qualification_intent_assignment(
            8,
            random_seed=31,
            qualification_subject_scientific_key=_key("subject"),
            phase="development",
        )
        second = freeze_qualification_intent_assignment(
            8,
            random_seed=31,
            qualification_subject_scientific_key=_key("subject"),
            phase="development",
        )
        self.assertEqual(first, second)
        self.assertEqual(set(first[:3]), {"anchor", "close", "divergent"})
        self.assertEqual(first[3:], first[:5])
        counts = [first.count(intent) for intent in ("anchor", "close", "divergent")]
        self.assertLessEqual(max(counts) - min(counts), 1)

        plan = _seal(count=8)
        treatment = [item for item in plan.future_roots if item.arm == "treatment"]
        control = [item for item in plan.future_roots if item.arm == "strategy_agnostic"]
        self.assertEqual(
            [item.realization_intent for item in treatment],
            [item.realization_intent for item in control],
        )

    def test_arm_orders_are_seeded_balanced_and_preserve_local_order(self) -> None:
        for phase, arms, freeze in (
            ("development", DEVELOPMENT_ARMS, freeze_development_arm_order),
            ("holdout", HOLDOUT_ARMS, freeze_holdout_arm_order),
        ):
            with self.subTest(phase=phase):
                first = freeze(
                    7,
                    random_seed=11,
                    qualification_subject_scientific_key=_key("subject"),
                )
                self.assertEqual(
                    first,
                    freeze(
                        7,
                        random_seed=11,
                        qualification_subject_scientific_key=_key("subject"),
                    ),
                )
                self.assertEqual(len(first), len(arms) * 7)
                for wave_start in range(0, len(first), len(arms)):
                    wave = first[wave_start : wave_start + len(arms)]
                    self.assertEqual({item.arm for item in wave}, set(arms))
                    self.assertEqual(
                        {item.arm_local_ordinal for item in wave},
                        {wave_start // len(arms)},
                    )
                for arm in arms:
                    self.assertEqual(
                        [item.arm_local_ordinal for item in first if item.arm == arm],
                        list(range(7)),
                    )

    def test_seal_is_id_free_canonical_and_deterministic(self) -> None:
        first = _seal()
        second = _seal()
        self.assertEqual(first, second)
        self.assertEqual(first.content_digest, second.content_digest)
        self.assertEqual(first.content_digest, digest_json(first.to_dict()))
        self.assertEqual(len(first.future_roots), 2 * first.planned_count_per_arm)
        self.assertEqual(first.endpoint_rule, QUALIFICATION_ENDPOINT_RULE)

        entity_pattern = re.compile(
            r"^(?:run|ev|iv|wu|la|pa|pc|to|ex|cr|cn|ma|pn|sb|ss|st|tp|rz|ep|sc|ae|sd|qf|rp|ps|ri|bi)-[0-9a-f]{64}$"
        )

        def inspect_value(value: object) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    self.assertFalse(key.endswith("_id"), key)
                    self.assertFalse(key.endswith("_scientific_key"), key)
                    inspect_value(child)
            elif isinstance(value, list):
                for child in value:
                    inspect_value(child)
            elif isinstance(value, str):
                self.assertIsNone(entity_pattern.fullmatch(value))

        inspect_value(first.to_dict())
        self.assertEqual(
            parse_sealed_holdout_plan(first.to_dict(), VOCABULARY), first
        )
        injected = first.to_dict()
        injected["future_roots"][0]["root_plan_id"] = entity_id("rp", ["future"])
        with self.assertRaisesRegex(StrategyValidationError, "unknown"):
            parse_sealed_holdout_plan(injected, VOCABULARY)

    def test_seal_has_exact_arm_mapping_blueprints_and_nonsuccess_rules(self) -> None:
        plan = _seal(count=6)
        for arm in HOLDOUT_ARMS:
            roots = [item for item in plan.future_roots if item.arm == arm]
            self.assertEqual(len(roots), 6)
            self.assertEqual([item.arm_local_ordinal for item in roots], list(range(6)))
            self.assertEqual(
                {item.blueprint for item in roots},
                {
                    "treatment_qualification"
                    if arm == "treatment"
                    else "strategy_agnostic_qualification"
                },
            )
            self.assertEqual(
                {item.semantic_nonsuccess_reasons for item in roots},
                {("inapplicable",) if arm == "treatment" else ()},
            )

    def test_freeze_api_has_no_outcome_input_and_closed_parsers_reject_one(self) -> None:
        self.assertNotIn(
            "outcome", " ".join(inspect.signature(build_sealed_holdout_plan).parameters)
        )
        with self.assertRaises(TypeError):
            build_sealed_holdout_plan(
                _template(),
                planned_count_per_arm=5,
                random_seed=1,
                qualification_subject_scientific_key=_key("subject"),
                measurement_cell=MEASUREMENT_CELL,
                descriptor_vocabulary=VOCABULARY,
                development_outcomes=(),  # type: ignore[call-arg]
            )
        materialized = _materialized(_seal())
        materialized["development_outcomes"] = []
        with self.assertRaisesRegex(StrategyValidationError, "unknown"):
            validate_materialized_holdout_plan(_seal(), materialized, VOCABULARY)


class MaterializedHoldoutValidationTests(unittest.TestCase):
    def test_valid_envelope_projects_exactly_and_validates(self) -> None:
        plan = _seal()
        materialized = _materialized(plan)
        source_digest, projected = project_materialized_holdout_plan(
            materialized, VOCABULARY
        )
        self.assertEqual(source_digest, plan.content_digest)
        self.assertEqual(projected, plan)
        self.assertIsNone(
            validate_materialized_holdout_plan(plan, materialized, VOCABULARY)
        )

    def test_every_top_level_seal_field_flip_is_rejected(self) -> None:
        plan = _seal()
        base = _materialized(plan)
        mutations = {}

        wrong_digest = deepcopy(base)
        wrong_digest["source_sealed_holdout_digest"] = _key("wrong-seal")
        mutations["source digest"] = wrong_digest

        count = deepcopy(base)
        count["planned_count_per_arm"] = 6
        mutations["H"] = count

        dimensions = deepcopy(base)
        dimensions["comparison_dimensions"] = [
            "wording",
            "framing",
        ]
        mutations["comparison dimensions"] = dimensions

        order = deepcopy(base)
        order["interleaved_root_order"][0:2] = reversed(
            order["interleaved_root_order"][0:2]
        )
        mutations["interleaved order"] = order

        masks = deepcopy(base)
        masks["information_masks"]["treatment"] = "response_aware_strategy_blind_request"
        mutations["information masks"] = masks

        control_mask = deepcopy(base)
        control_mask["information_masks"]["strategy_agnostic"] = "bound_strategy_request"
        mutations["control information mask"] = control_mask

        endpoint = deepcopy(base)
        endpoint["endpoint_rule"] = "multiple_responses"
        mutations["endpoint"] = endpoint

        cell = deepcopy(base)
        cell["measurement_cell"]["application_scope"] = "action_policy"
        mutations["measurement cell"] = cell

        for label, mutation in mutations.items():
            with self.subTest(label=label):
                with self.assertRaises(StrategyValidationError):
                    validate_materialized_holdout_plan(plan, mutation, VOCABULARY)

    def test_every_future_root_field_flip_is_rejected(self) -> None:
        plan = _seal()
        base = _materialized(plan)
        flips = {
            "root_ordinal": 1,
            "arm_local_ordinal": 1,
            "arm": "strategy_agnostic",
            "realization_intent": next(
                intent
                for intent in ("anchor", "close", "divergent")
                if intent != base["future_roots"][0]["realization_intent"]
            ),
            "blueprint": "strategy_agnostic_qualification",
            "semantic_nonsuccess_reasons": [],
        }
        for field, changed in flips.items():
            with self.subTest(field=field):
                mutation = deepcopy(base)
                mutation["future_roots"][0][field] = changed
                with self.assertRaises(StrategyValidationError):
                    validate_materialized_holdout_plan(plan, mutation, VOCABULARY)

    def test_substitution_reorder_extension_and_wrong_h_are_rejected(self) -> None:
        plan = _seal()
        base = _materialized(plan)

        reordered = deepcopy(base)
        reordered["future_roots"][0], reordered["future_roots"][1] = (
            reordered["future_roots"][1],
            reordered["future_roots"][0],
        )
        extended = deepcopy(base)
        extra = deepcopy(extended["future_roots"][-1])
        extra["execution_id"] = entity_id("ex", ["extra"])
        extra["execution_scientific_key"] = _key("extra")
        extra["root_plan_id"] = entity_id("rp", ["extra"])
        extra["root_ordinal"] = len(extended["future_roots"])
        extra["arm_local_ordinal"] = plan.planned_count_per_arm
        extended["future_roots"].append(extra)
        substituted = deepcopy(base)
        substituted["future_roots"][0]["realization_intent"] = substituted[
            "future_roots"
        ][1]["realization_intent"]

        for label, mutation in (
            ("reordered", reordered),
            ("extended", extended),
            ("substituted", substituted),
        ):
            with self.subTest(label=label):
                with self.assertRaises(StrategyValidationError):
                    validate_materialized_holdout_plan(plan, mutation, VOCABULARY)

    def test_only_exact_materialization_identity_fields_may_be_stripped(self) -> None:
        plan = _seal()
        for field, value in (
            ("action_id", entity_id("la", ["injected"])),
            ("root_plan_scientific_key", _key("injected")),
        ):
            with self.subTest(field=field):
                mutation = _materialized(plan)
                mutation["future_roots"][0][field] = value
                with self.assertRaisesRegex(StrategyValidationError, "unknown"):
                    validate_materialized_holdout_plan(plan, mutation, VOCABULARY)

        top_level = _materialized(plan)
        top_level["unit_id"] = entity_id("wu", ["injected"])
        with self.assertRaisesRegex(StrategyValidationError, "unknown"):
            validate_materialized_holdout_plan(plan, top_level, VOCABULARY)

        wrong_prefix = _materialized(plan)
        wrong_prefix["future_roots"][0]["execution_id"] = entity_id("rp", ["wrong"])
        with self.assertRaisesRegex(StrategyValidationError, "ex- prefix"):
            validate_materialized_holdout_plan(plan, wrong_prefix, VOCABULARY)

    def test_direct_records_reject_each_closed_gate(self) -> None:
        with self.assertRaises(StrategyValidationError):
            HoldoutInformationMasks("wrong", "response_aware_strategy_blind_request")
        with self.assertRaises(StrategyValidationError):
            FutureRootSpec(0, 0, "treatment", "anchor", "strategy_agnostic_qualification", ("inapplicable",))
        with self.assertRaises(StrategyValidationError):
            FutureRootSpec(0, 0, "strategy_agnostic", "anchor", "strategy_agnostic_qualification", ("nonprogress",))

        plan = _seal()
        with self.assertRaisesRegex(StrategyValidationError, "at least five"):
            build_sealed_holdout_plan(
                _template(),
                planned_count_per_arm=4,
                random_seed=1,
                qualification_subject_scientific_key=_key("subject"),
                measurement_cell=MEASUREMENT_CELL,
                descriptor_vocabulary=VOCABULARY,
            )
        with self.assertRaisesRegex(StrategyValidationError, "2H"):
            SealedHoldoutPlan(
                6,
                plan.comparison_dimensions,
                plan.future_roots,
                plan.interleaved_root_order,
                plan.information_masks,
                plan.endpoint_rule,
                plan.measurement_cell,
            )


if __name__ == "__main__":
    unittest.main()
