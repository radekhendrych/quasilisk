"""Canonical plural stateful-scenario verification conformance."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from decimal import Decimal
import tempfile
import unittest
from unittest.mock import patch

from integration_support import dryrun_config
from quasilisk.domain import (
    ActionPurpose,
    RoleCallVector,
    digest_json,
    entity_id,
    scientific_key,
    sha256_bytes,
    ExecutionTerminalStatus,
)
from quasilisk.engine import (
    EngineError,
    RunContext,
    ScenarioPlan,
    _hydrate_scenarios,
    _initialize_routes,
    _scenario_binding_request,
    build_scenario_verification,
    load_resources,
)
from quasilisk.evidence import (
    PerturbatorRoleScope,
    ScenarioRecipe,
    parse_scenario_recipe,
)
from quasilisk.run_state import (
    AttemptView,
    LogicalActionView,
    TargetResultView,
    WorkUnitView,
    _derive_sample_executions,
)
from quasilisk.scheduler import SchedulerValidationError, validate_work_unit_maximum
from quasilisk.strategy import parse_strategy_template


class ScenarioVerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        resources = load_resources(dryrun_config("fatalism", root / "runs"))
        self.context = RunContext(
            resources=resources,
            run_id=entity_id("run", ["scenario-verification"]),
            run_nonce="0" * 32,
            invocation_id=entity_id("iv", ["scenario-verification"]),
            run_directory=root / "run",
            candidates={},
            routes={},
        )
        _initialize_routes(self.context)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _template(self, label: str):
        return parse_strategy_template(
            {
                "purpose": "repair",
                "applicability": {
                    "diagnosis": "conditional_acceptance",
                    "requires": f"The {label} condition remains unresolved.",
                    "excludes": f"The {label} condition is absent.",
                },
                "move": {
                    "act": "repair_grounded_gap",
                    "target_kind": "condition",
                    "descriptors": {
                        "objective_route": "causal_fixity",
                        "argument_move": "answer_objection",
                        "application_scope": "reasoning_policy",
                    },
                    "blueprint": f"Resolve the {label} condition without changing the premise.",
                },
                "hypothesis": f"Resolving {label} should permit progress.",
                "preserve": [
                    {"kind": "meaning", "statement": "Preserve the accepted meaning."}
                ],
                "vary": [
                    {"dimension": "framing", "instruction": "Vary only framing."}
                ],
                "success": {
                    "transition": {"from": "conditional", "to": "resolved"},
                    "supports": f"The target resolves {label}.",
                    "falsifies": f"The target retains {label}.",
                },
            },
            self.context.objective.descriptor_values,
            self.context.objective.diagnosis_categories,
        )

    @staticmethod
    def _digest(label: str) -> str:
        return digest_json(["scenario-verification", label])

    def _recipe(self) -> ScenarioRecipe:
        templates = (self._template("first"), self._template("second"))
        return ScenarioRecipe(
            entry_prompt_hash=sha256_bytes(b"Fresh entry prompt."),
            target_generation_profile_digest=self.context.config.target.digest,
            templates=templates,
            measurement_cell=templates[-1].move.descriptors,
            maximum_target_turns=3,
            maximum_strategy_rounds=2,
            designated_realization_policies=("anchor", "anchor"),
            selector_scope_keys=(
                ("outcome_review", self._digest("outcome")),
                ("probe_review", self._digest("probe")),
                ("template_binding", self._digest("binding")),
            ),
            perturbator_scope=PerturbatorRoleScope(
                self._digest("provider"),
                "provider/model",
                self._digest("generation"),
                "json_schema",
                (
                    ("strategy", self._digest("strategy-prompt")),
                    ("strategy_agnostic", self._digest("control-prompt")),
                ),
            ),
        )

    def _scenario(self) -> ScenarioPlan:
        recipe = self._recipe()
        key = scientific_key(
            [self.context.scope, "artifact", "scenario_recipe", recipe.content_hash]
        )
        return ScenarioPlan(
            scenario_id=entity_id("sc", [self.context.run_id, key]),
            scientific_key=key,
            entry_prompt="Fresh entry prompt.",
            recipe=recipe,
            origin="imported",
        )

    def _one_step_scenario(self) -> ScenarioPlan:
        plural = self._recipe()
        recipe = ScenarioRecipe(
            plural.entry_prompt_hash,
            plural.target_generation_profile_digest,
            plural.templates[:1],
            plural.templates[0].move.descriptors,
            2,
            1,
            ("anchor",),
            plural.selector_scope_keys,
            plural.perturbator_scope,
        )
        key = scientific_key(
            [self.context.scope, "artifact", "scenario_recipe", recipe.content_hash]
        )
        return ScenarioPlan(
            entity_id("sc", [self.context.run_id, key]),
            key,
            "Fresh entry prompt.",
            recipe,
            "imported",
        )

    def _reduce_first_root(
        self,
        values_by_purpose,
        *,
        closed_purposes: set[str] = frozenset(),
        terminal_target: bool = False,
    ):
        scenario = self._one_step_scenario()
        unit = build_scenario_verification(self.context, scenario)
        root = unit.design["root_plans"][0]
        slots = {item["action_id"]: item for item in unit.logical_action_slots}
        root_slots = [slots[item] for item in root["ordered_action_ids"]]
        targets = [item for item in root_slots if item["role"] == "target"]
        execution_id = unit.execution_slots[0]["execution_id"]
        fresh_root = entity_id("cr", [execution_id, "conversation_root"])

        def target_attempt(label: str, messages, parent):
            return AttemptView(
                entity_id("pa", ["scenario-reducer", label]),
                {
                    "payload": {
                        "messages": messages,
                        "lineage": {
                            "execution_id": execution_id,
                            "conversation_root_id": fresh_root,
                            "parent_node_id": parent,
                        },
                    }
                },
                None,
                None,
            )

        entry_messages = [{"role": "user", "content": "Fresh entry prompt."}]
        terminal_messages = [
            *entry_messages,
            {"role": "assistant", "content": "fresh root reply"},
            {"role": "user", "content": "aligned anchor"},
        ]
        action_views = []
        for slot in unit.logical_action_slots:
            attempts = ()
            if slot["action_id"] == targets[0]["action_id"]:
                attempts = (target_attempt("entry", entry_messages, None),)
            elif terminal_target and slot["action_id"] == targets[1]["action_id"]:
                attempts = (
                    target_attempt(
                        "terminal", terminal_messages, entity_id("cn", ["entry"])
                    ),
                )
            action_views.append(
                LogicalActionView(slot["action_id"], unit.unit_id, slot, attempts)
            )
        view = WorkUnitView({}, unit.to_dict(), tuple(action_views))
        logical = {item.action_id: item for item in action_views}
        results = {
            targets[0]["action_id"]: TargetResultView(
                targets[0]["action_id"], scenario.scenario_id, execution_id,
                fresh_root, None, "fresh root reply", "complete", "target/revision",
                entity_id("pa", ["scenario-reducer", "entry"]),
                entity_id("ev", ["scenario-reducer", "entry"]), 1,
            )
        }
        observations = {
            targets[0]["action_id"]: SimpleNamespace(
                observation_id=entity_id("to", ["scenario-reducer", "entry"])
            )
        }
        if terminal_target:
            results[targets[1]["action_id"]] = TargetResultView(
                targets[1]["action_id"], entity_id("pc", ["terminal"]), execution_id,
                fresh_root, entity_id("cn", ["entry"]), "fresh terminal reply",
                "complete", "target/revision",
                entity_id("pa", ["scenario-reducer", "terminal"]),
                entity_id("ev", ["scenario-reducer", "terminal"]), 2,
            )
            observations[targets[1]["action_id"]] = SimpleNamespace(
                observation_id=entity_id("to", ["scenario-reducer", "terminal"]),
                scientific_key=self._digest("terminal-observation"),
                response="fresh terminal reply",
            )
        panels = {}
        for index, target in enumerate(targets):
            if index == 1 and not terminal_target:
                continue
            plan = next(
                item for item in unit.panel_plans if item["target_action_id"] == target["action_id"]
            )
            panels[plan["panel_id"]] = SimpleNamespace(
                target_action_id=target["action_id"],
                terminated=True,
                resolved=True,
                primary=SimpleNamespace(score=Decimal("80")),
                panel_required=False,
                assessment=None,
            )

        purpose_actions: dict[str, list[str]] = {}
        for slot in root_slots:
            purpose_actions.setdefault(slot["purpose"], []).append(slot["action_id"])
        if callable(values_by_purpose):
            values_by_purpose = values_by_purpose(unit, root_slots)
        semantic: dict[str, object] = {}
        for purpose, value in values_by_purpose.items():
            if purpose.endswith("_corrective"):
                base = purpose.removesuffix("_corrective")
                semantic[purpose_actions[base][1]] = value
            else:
                semantic[purpose_actions[purpose][0]] = value
        closed = {
            action_id
            for purpose in closed_purposes
            for action_id in purpose_actions[purpose]
        }
        with (
            patch(
                "quasilisk.run_state._structured_action_value",
                side_effect=lambda action, _schemas: semantic.get(action.action_id),
            ),
            patch(
                "quasilisk.run_state._parsed_structured_action",
                side_effect=lambda action, _schemas, parser, **_kwargs: (
                    None
                    if action.action_id not in semantic
                    else parser(semantic[action.action_id])
                ),
            ),
            patch(
                "quasilisk.run_state._completion_action_closed",
                side_effect=lambda _unit, action, *_args: action.action_id in closed,
            ),
        ):
            samples, evidence = _derive_sample_executions(
                work_units=(view,),
                logical_actions=logical,
                target_results=results,
                target_observations=observations,
                panel_views=panels,
                response_schemas={},
            )
        return samples[execution_id], evidence[execution_id], unit, root_slots

    @staticmethod
    def _step_values(unit, root_slots, *, binding="applicable", review="aligned", rereview="aligned", outcome="supports"):
        del unit, root_slots
        proposed = {
            "items": [
                {
                    "slot": "s0",
                    "status": "proposed",
                    "text": "aligned anchor",
                }
            ]
        }
        reviewed = {
            "reviews": [{
                "review_alias": "r0",
                "status": review,
                "reviewed_descriptors": None,
                "reason": "Review the original realization.",
             "procedure": {"assumption": "none", "assumption_quote": None, "direction": "forward", "novelty": "new"}}]
        }
        rereviewed = {
            "reviews": [{
                "review_alias": "r0",
                "status": rereview,
                "reviewed_descriptors": None,
                "reason": "Review the corrective realization.",
             "procedure": {"assumption": "none", "assumption_quote": None, "direction": "forward", "novelty": "new"}}]
        }
        return {
            "template_binding": {"status": binding},
            "perturbator_proposal": proposed,
            "probe_review": reviewed,
            "perturbator_proposal_corrective": proposed,
            "probe_review_corrective": rereviewed,
            "outcome_review": {
                "reviews": [
                    {
                        "review_alias": "r0",
                        "result": outcome,
                        "response_state": "engaged",
                        "justification": "Assess the observed transition.",
                    }
                ]
            },
        }

    def test_valid_binding_inapplicability_is_a_resolved_negative(self) -> None:
        sample, evidence, _, _ = self._reduce_first_root(
            lambda unit, root: self._step_values(
                unit, root, binding="inapplicable"
            )
        )
        self.assertEqual(
            sample.terminal_status, ExecutionTerminalStatus.SEMANTIC_NONSUCCESS
        )
        self.assertTrue(evidence.terminated)
        self.assertTrue(evidence.resolved)
        self.assertTrue(evidence.semantic_nonsuccess)

    def test_binding_recovery_exhaustion_is_operationally_unresolved(self) -> None:
        sample, evidence, _, _ = self._reduce_first_root(
            {}, closed_purposes={"template_binding"}
        )
        self.assertEqual(
            sample.terminal_status,
            ExecutionTerminalStatus.OPERATIONALLY_UNRESOLVED,
        )
        self.assertTrue(evidence.terminated)
        self.assertFalse(evidence.resolved)

    def test_generator_and_review_exhaustion_are_operationally_unresolved(self) -> None:
        for stopped, values in (
            (
                "perturbator_proposal",
                {"template_binding": {"status": "applicable"}},
            ),
            (
                "probe_review",
                lambda unit, root: {
                    key: value
                    for key, value in self._step_values(unit, root).items()
                    if key in {"template_binding", "perturbator_proposal"}
                },
            ),
        ):
            with self.subTest(stopped=stopped):
                sample, evidence, _, _ = self._reduce_first_root(
                    values, closed_purposes={stopped}
                )
                self.assertEqual(
                    sample.terminal_status,
                    ExecutionTerminalStatus.OPERATIONALLY_UNRESOLVED,
                )
                self.assertTrue(evidence.terminated)
                self.assertFalse(evidence.resolved)

    def test_direction_reversal_in_qualification_treatment_never_dispatches(self) -> None:
        sample, evidence, _, _ = self._reduce_first_root(
            lambda unit, root: self._step_values(
                unit, root, review="invalid_contract", rereview="invalid_contract"
            )
        )
        self.assertEqual(
            sample.terminal_status,
            ExecutionTerminalStatus.OPERATIONALLY_UNRESOLVED,
        )
        self.assertTrue(evidence.terminated)
        self.assertFalse(evidence.resolved)

    def test_final_reviewed_nonprogress_is_a_resolved_negative(self) -> None:
        sample, evidence, _, _ = self._reduce_first_root(
            lambda unit, root: self._step_values(
                unit, root, outcome="falsifies"
            ),
            terminal_target=True,
        )
        self.assertEqual(
            sample.terminal_status, ExecutionTerminalStatus.SEMANTIC_NONSUCCESS
        )
        self.assertTrue(evidence.terminated)
        self.assertTrue(evidence.resolved)
        self.assertFalse(evidence.confirmed_objective_high)

    def test_plural_recipe_round_trips_and_measurement_cell_is_not_identity(self) -> None:
        recipe = self._recipe()
        self.assertEqual(
            parse_scenario_recipe(
                recipe.to_dict(),
                self.context.objective.descriptor_values,
                self.context.objective.diagnosis_categories,
            ),
            recipe,
        )
        alternate_cell = tuple(
            (name, values[-1])
            for name, values in self.context.objective.descriptor_values.items()
        )
        changed = ScenarioRecipe(
            recipe.entry_prompt_hash,
            recipe.target_generation_profile_digest,
            recipe.templates,
            alternate_cell,
            recipe.maximum_target_turns,
            recipe.maximum_strategy_rounds,
            recipe.designated_realization_policies,
            recipe.selector_scope_keys,
            recipe.perturbator_scope,
        )
        self.assertEqual(changed.content_hash, recipe.content_hash)

    def test_hydration_consumes_only_reducer_owned_scenario_views(self) -> None:
        scenario = self._scenario()
        view = SimpleNamespace(
            scenario_id=scenario.scenario_id,
            scientific_key=scenario.scientific_key,
            entry_prompt=scenario.entry_prompt,
            recipe=scenario.recipe,
            origin="derived",
            source_seed_id=None,
            source_episode_id=entity_id("st", ["scenario-source"]),
            source_child_scientific_key=self._digest("scenario-child"),
            parent_unit_id=entity_id("wu", ["scenario-source"]),
            source_step_episode_ids=(entity_id("st", ["scenario-source"]),),
            source_step_child_scientific_keys=(self._digest("scenario-child"),),
            source_unit_ids=(entity_id("wu", ["scenario-source"]),),
        )
        state = SimpleNamespace(scenarios={scenario.scientific_key: view})
        _hydrate_scenarios(self.context, state)
        hydrated = self.context.scenarios[scenario.scientific_key]
        self.assertEqual(hydrated.recipe, scenario.recipe)
        self.assertEqual(hydrated.source_episode_id, view.source_episode_id)

        view.entry_prompt = "changed after reducer projection"
        with self.assertRaisesRegex(EngineError, "reducer ownership"):
            _hydrate_scenarios(self.context, state)

    def test_builder_freezes_every_ordered_step_and_exact_maximum(self) -> None:
        unit = build_scenario_verification(self.context, self._scenario())
        self.context.resources.authoritative_schemas.validate_work_unit(unit.to_dict())
        self.assertEqual(
            validate_work_unit_maximum(unit),
            RoleCallVector(target=30, judge=120, selector=160, perturbator=80),
        )
        self.assertEqual(len(unit.proposal_slots), 10)
        self.assertEqual(len(unit.execution_slots), 5)
        self.assertEqual(len(unit.panel_plans), 15)
        self.assertEqual(len(unit.logical_action_slots), 135)
        by_id = {item["action_id"]: item for item in unit.logical_action_slots}
        for root in unit.design["root_plans"]:
            actions = [by_id[item] for item in root["ordered_action_ids"]]
            self.assertEqual(len(actions), 27)
            self.assertEqual(
                actions[5]["guard"],
                {"kind": "if_panel_resolved", "panel_id": next(
                    p["panel_id"] for p in unit.panel_plans
                    if p["target_action_id"] == actions[0]["action_id"]
                )},
            )
            self.assertEqual(actions[5]["dependency_action_ids"], root["ordered_action_ids"][:5])
            self.assertEqual(
                actions[16]["guard"],
                {"kind": "if_action_usable", "action_id": actions[15]["action_id"]},
            )
            self.assertEqual(actions[16]["dependency_action_ids"], [actions[15]["action_id"]])
            self.assertEqual(
                [item["purpose"] for item in (actions[5], actions[16])],
                [ActionPurpose.TEMPLATE_BINDING.value] * 2,
            )
            self.assertEqual(root["terminal_endpoint_action_id"], actions[21]["action_id"])

        tampered = deepcopy(unit.to_dict())
        terminal = next(
            item
            for item in tampered["logical_action_slots"]
            if item["guard"].get("kind") == "if_dispatchable_probe"
        )
        terminal["guard"]["allowed"] = ["valid"]
        from quasilisk.domain import WorkUnit

        invalid = WorkUnit(
            unit_id=tampered["unit_id"],
            unit_scientific_key=tampered["unit_scientific_key"],
            kind=tampered["kind"],
            subject_id=tampered["subject_id"],
            subject_scientific_key=tampered["subject_scientific_key"],
            parent_unit_id=tampered["parent_unit_id"],
            policy_digest=tampered["policy_digest"],
            design=tampered["design"],
            proposal_slots=tuple(tampered["proposal_slots"]),
            execution_slots=tuple(tampered["execution_slots"]),
            panel_plans=tuple(tampered["panel_plans"]),
            logical_action_slots=tuple(tampered["logical_action_slots"]),
            worst_case_calls=RoleCallVector(**tampered["worst_case_calls"]),
        )
        with self.assertRaises(SchedulerValidationError):
            validate_work_unit_maximum(invalid)

    def test_second_binding_receives_only_the_exact_fresh_two_step_branch(self) -> None:
        scenario = self._scenario()
        unit = build_scenario_verification(self.context, scenario)
        root = unit.design["root_plans"][0]
        slots = {item["action_id"]: item for item in unit.logical_action_slots}
        target_ids = [
            item
            for item in root["ordered_action_ids"]
            if slots[item]["role"] == "target"
        ]
        binding_ids = [
            item
            for item in root["ordered_action_ids"]
            if slots[item]["purpose"] == "template_binding"
        ]
        first_messages = [{"role": "user", "content": "Fresh entry prompt."}]
        second_messages = [
            *first_messages,
            {"role": "assistant", "content": "fresh root reply"},
            {"role": "user", "content": "first aligned anchor"},
        ]

        def attempt(label: str, messages: list[dict[str, str]]) -> AttemptView:
            return AttemptView(
                entity_id("pa", [label]),
                {"payload": {"messages": messages}},
                None,
                None,
            )

        action_views: list[LogicalActionView] = []
        for action_id in root["ordered_action_ids"]:
            attempts = ()
            if action_id == target_ids[0]:
                attempts = (attempt("entry", first_messages),)
            elif action_id == target_ids[1]:
                attempts = (attempt("first-step", second_messages),)
            action_views.append(
                LogicalActionView(action_id, unit.unit_id, slots[action_id], attempts)
            )
        view = WorkUnitView({}, unit.to_dict(), tuple(action_views))
        logical = {item.action_id: item for item in action_views}
        execution_id = unit.execution_slots[0]["execution_id"]
        fresh_root = entity_id("cr", [execution_id, "conversation_root"])
        results = {
            target_ids[0]: TargetResultView(
                target_ids[0], scenario.scenario_id, execution_id, fresh_root, None,
                "fresh root reply", "complete", "target/revision", entity_id("pa", ["entry"]),
                entity_id("ev", ["entry-result"]), 1,
            ),
            target_ids[1]: TargetResultView(
                target_ids[1], entity_id("pc", ["first-step"]), execution_id, fresh_root,
                entity_id("cn", ["entry-node"]), "fresh first-step reply", "complete",
                "target/revision", entity_id("pa", ["first-step"]),
                entity_id("ev", ["first-step-result"]), 2,
            ),
        }
        state = SimpleNamespace(target_results=results, logical_actions=logical)
        self.context.scenarios[scenario.scientific_key] = scenario
        request = _scenario_binding_request(
            self.context, state, view, logical[binding_ids[1]]
        )
        self.assertEqual(
            request["fresh_conversation"],
            [
                *second_messages,
                {"role": "assistant", "content": "fresh first-step reply"},
            ],
        )
        self.assertEqual(
            request["strategy_template"], scenario.recipe.templates[1].to_dict()
        )
        self.assertNotIn("recorded source reply", str(request))


if __name__ == "__main__":
    unittest.main()
