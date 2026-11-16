"""Reducer-owned scenario promotion and archive projection regressions."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from integration_support import dryrun_config
from test_seed_qualification import _execution, _scenario, _source
from quasilisk.domain import (
    Role,
    VerificationStatus,
    SearchStatus,
    digest_json,
    entity_id,
    sha256_bytes,
)
from quasilisk.engine import (
    RunContext,
    ScenarioPlan,
    _initialize_routes,
    _manifest,
    build_scenario_verification,
    load_resources,
)
from quasilisk.evidence import (
    NamedScopeEvidence,
    PerturbatorRoleScope,
    ScenarioRecipe,
)
from quasilisk.events import JournalCorrupt
from quasilisk.run_state import (
    AttemptView,
    LogicalActionView,
    ScenarioRecipeView,
    TargetResultView,
    WorkUnitView,
    _candidate_metadata,
    _structured_role_scope_candidates,
    _derive_scenario_seed_qualifications,
    _derive_scenario_recipes,
    _perturbator_prompt_mode_digests,
    _scenario_episode_ancestry,
    _scenario_recipe_scientific_key,
    _static_route,
    _template_binding_mode_before,
    _validate_scenario_verification_unit,
)


class ScenarioPromotionProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        resources = load_resources(dryrun_config("fatalism", root / "runs"))
        self.context = RunContext(
            resources=resources,
            run_id=entity_id("run", ["scenario-promotion"]),
            run_nonce="0" * 32,
            invocation_id=entity_id("iv", ["scenario-promotion"]),
            run_directory=root / "run",
            candidates={},
            scenarios={},
            routes={},
        )
        _initialize_routes(self.context)
        self.manifest = _manifest(self.context)
        candidates = _structured_role_scope_candidates(self.manifest)

        def selector_scope(name: str, configured_slot: str = "selector_primary"):
            return next(
                scope
                for base, scope in candidates
                if base == (Role.SELECTOR.value, configured_slot, name)
                and scope.response_mode == "json_schema"
            )

        self.outcome_scope = selector_scope("outcome_review")
        self.probe_scope = selector_scope("probe_review")
        self.binding_scope = selector_scope("template_binding")
        self.alternate_probe_scope = selector_scope(
            "probe_review", "selector_fallback"
        )
        perturbator_route = _static_route(
            self.manifest, Role.PERTURBATOR.value, "perturbator_primary"
        )
        self.perturbator_scope = PerturbatorRoleScope(
            perturbator_route.requested_provider_route_key,
            perturbator_route.requested_model,
            perturbator_route.generation_profile_digest,
            "json_schema",
            _perturbator_prompt_mode_digests(self.manifest),
        )
        template = _scenario().recipe.templates[0]
        prompt = "Reducer-owned scenario entry."
        self.recipe = ScenarioRecipe(
            entry_prompt_hash=sha256_bytes(prompt.encode("utf-8")),
            target_generation_profile_digest=self.context.config.target.digest,
            templates=(template,),
            measurement_cell=template.move.descriptors,
            maximum_target_turns=2,
            maximum_strategy_rounds=1,
            designated_realization_policies=("anchor",),
            selector_scope_keys=(
                ("outcome_review", self.outcome_scope.scope_key),
                ("probe_review", self.probe_scope.scope_key),
                ("template_binding", self.binding_scope.scope_key),
            ),
            perturbator_scope=self.perturbator_scope,
        )
        key = _scenario_recipe_scientific_key(self.context.scope, self.recipe)
        self.scenario = ScenarioRecipeView(
            scenario_id=entity_id("sc", [self.context.run_id, key]),
            scientific_key=key,
            entry_prompt=prompt,
            recipe=self.recipe,
            origin="imported",
            source_seed_id=entity_id("sd", ["imported-scenario"]),
            source_episode_id=None,
            source_child_scientific_key=None,
            parent_unit_id=None,
            source_step_episode_ids=(),
            source_step_child_scientific_keys=(),
            source_unit_ids=(),
        )
        self.unit = WorkUnitView(
            {
                "sequence": 1,
                "event_id": entity_id("ev", ["scenario-unit-start"]),
            },
            {
                "unit_id": entity_id("wu", ["scenario-unit"]),
                "kind": "verification",
                "subject_id": self.scenario.scenario_id,
                "subject_scientific_key": self.scenario.scientific_key,
                "design": {
                    "subject_kind": "scenario_recipe",
                    "planned_count": 5,
                },
                "execution_slots": [],
            },
            (),
        )
        self.judge_panel = tuple(
            sorted(
                (
                    NamedScopeEvidence(
                        "judge_primary",
                        digest_json(["scenario-promotion", "judge"]),
                    ),
                    NamedScopeEvidence(
                        "panel",
                        digest_json(["scenario-promotion", "panel"]),
                    ),
                ),
                key=lambda item: item.name,
            )
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _project(
        self,
        *,
        high_count: int,
        unresolved_index: int | None = None,
        probe_scope=None,
        injected_index: int | None = None,
    ):
        executions = tuple(
            _execution(
                index,
                high=index < high_count,
                end_to_end=True,
                injected=index == injected_index,
                resolved=index != unresolved_index,
            )
            for index in range(5)
        )
        chosen_probe = probe_scope or self.probe_scope
        components = (self.outcome_scope, chosen_probe)
        with (
            patch(
                "quasilisk.run_state._validate_scenario_verification_unit",
                return_value=executions,
            ),
            patch(
                "quasilisk.run_state._scenario_selector_components",
                return_value=components,
            ),
            patch(
                "quasilisk.run_state._scenario_current_perturbator_scope",
                return_value=self.perturbator_scope,
            ),
            patch(
                "quasilisk.run_state._scenario_judge_panel_evidence",
                return_value=self.judge_panel,
            ),
            patch(
                "quasilisk.run_state._seed_evidence_source",
                return_value=_source(),
            ),
        ):
            return _derive_scenario_seed_qualifications(
                run_id=self.context.run_id,
                source_run_directory=self.context.run_directory.as_posix(),
                manifest=self.manifest,
                events=(),
                work_units=(self.unit,),
                logical_actions={},
                target_results={},
                target_observations={},
                member_assessments={},
                execution_evidence={},
                conversation_nodes={},
                response_schemas={},
                target_identity="scenario-target-revision",
                scenarios={self.scenario.scientific_key: self.scenario},
            )

    def test_current_run_success_materializes_snapshot_evidence_and_one_entry(self) -> None:
        decisions, snapshots, evidence, entries = self._project(high_count=4)
        decision = next(iter(decisions.values()))
        self.assertTrue(decision.qualified)
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].subject_id, self.scenario.scenario_id)
        self.assertEqual(entries[0].cell, tuple(dict(self.recipe.measurement_cell).values()))
        self.assertIs(
            entries[0].evidence.verification_status, VerificationStatus.VERIFIED
        )

    def test_three_of_five_and_two_of_five_do_not_promote(self) -> None:
        for high_count in (3, 2):
            with self.subTest(high_count=high_count):
                decisions, snapshots, _, entries = self._project(
                    high_count=high_count
                )
                self.assertFalse(next(iter(decisions.values())).qualified)
                self.assertEqual(snapshots, ())
                self.assertIs(
                    entries[0].evidence.verification_status,
                    VerificationStatus.FAILED,
                )

    def test_unresolved_and_scope_drift_never_promote(self) -> None:
        decisions, snapshots, _, entries = self._project(
            high_count=4, unresolved_index=4
        )
        self.assertFalse(next(iter(decisions.values())).qualified)
        self.assertEqual(snapshots, ())
        self.assertIs(
            entries[0].evidence.verification_status,
            VerificationStatus.INCONCLUSIVE,
        )

        decisions, snapshots, _, entries = self._project(
            high_count=4, probe_scope=self.alternate_probe_scope
        )
        self.assertFalse(next(iter(decisions.values())).qualified)
        self.assertEqual(snapshots, ())
        self.assertIs(
            entries[0].evidence.verification_status,
            VerificationStatus.VERIFIED,
        )

    def test_projection_is_replay_stable_and_injection_blocks_promotion(self) -> None:
        first = self._project(high_count=4)
        second = self._project(high_count=4)
        self.assertEqual(first, second)
        decisions, snapshots, _, entries = self._project(
            high_count=4, injected_index=0
        )
        self.assertFalse(next(iter(decisions.values())).qualified)
        self.assertEqual(snapshots, ())
        self.assertIs(
            entries[0].evidence.verification_status,
            VerificationStatus.VERIFIED,
        )

    def test_scenario_seed_and_evaluation_targets_do_not_enter_candidate_map(self) -> None:
        manifest = deepcopy(self.manifest)
        manifest["seed_snapshot"] = [
            {
                "type": "scenario_recipe",
                "entry_prompt": self.scenario.entry_prompt,
                "descriptors": dict(self.recipe.measurement_cell),
            }
        ]
        action_id = entity_id("la", ["scenario-evaluation-target"])
        unit_id = entity_id("wu", ["scenario-evaluation"])
        attempt = AttemptView(
            entity_id("pa", ["scenario-evaluation"]),
            {
                "payload": {
                    "messages": [{"role": "user", "content": "evaluation only"}],
                    "lineage": {"review_item_id": None},
                }
            },
            None,
            None,
        )
        action = LogicalActionView(action_id, unit_id, {}, (attempt,))
        work = WorkUnitView(
            {}, {"unit_id": unit_id, "kind": "verification"}, (action,)
        )
        target = TargetResultView(
            action_id,
            entity_id("pc", ["evaluation-only"]),
            None,
            None,
            None,
            "response",
            "complete",
            None,
            attempt.physical_attempt_id,
            entity_id("ev", ["evaluation-result"]),
            1,
        )
        baseline = deepcopy(manifest)
        baseline["seed_snapshot"] = []
        self.assertEqual(
            _candidate_metadata(
                self.context.scope,
                self.context.run_id,
                manifest,
                (work,),
                {action_id: action},
                {action_id: target},
                {},
            ),
            _candidate_metadata(
                self.context.scope,
                self.context.run_id,
                baseline,
                (),
                {},
                {},
                {},
            ),
        )

    def test_tampered_recipe_unit_linkage_is_rejected(self) -> None:
        plan = ScenarioPlan(
            self.scenario.scenario_id,
            self.scenario.scientific_key,
            self.scenario.entry_prompt,
            self.recipe,
            "imported",
        )
        unit = build_scenario_verification(self.context, plan)
        action_views = tuple(
            LogicalActionView(slot["action_id"], unit.unit_id, slot, ())
            for slot in unit.logical_action_slots
        )
        view = WorkUnitView(
            {"sequence": 1}, unit.to_dict(), action_views
        )
        alternate_cell = tuple(
            (name, value + "_tampered") for name, value in self.recipe.measurement_cell
        )
        changed_recipe = ScenarioRecipe(
            self.recipe.entry_prompt_hash,
            self.recipe.target_generation_profile_digest,
            self.recipe.templates,
            alternate_cell,
            self.recipe.maximum_target_turns,
            self.recipe.maximum_strategy_rounds,
            self.recipe.designated_realization_policies,
            self.recipe.selector_scope_keys,
            self.recipe.perturbator_scope,
        )
        changed = ScenarioRecipeView(
            self.scenario.scenario_id,
            self.scenario.scientific_key,
            self.scenario.entry_prompt,
            changed_recipe,
            "imported",
            self.scenario.source_seed_id,
            None,
            None,
            None,
            (),
            (),
            (),
        )
        with self.assertRaisesRegex(JournalCorrupt, "recipe-canonical"):
            _validate_scenario_verification_unit(
                run_id=self.context.run_id,
                manifest=self.manifest,
                unit=view,
                scenario=changed,
                logical_actions={item.action_id: item for item in action_views},
                target_results={},
                target_observations={},
                execution_evidence={},
                conversation_nodes={},
                response_schemas={},
            )

    def test_template_binding_mode_uses_its_own_schema_history(self) -> None:
        selector_route = _static_route(
            self.manifest, Role.SELECTOR.value, "selector_primary"
        )

        def action(schema_name: str, mode: str, sequence: int, terminal):
            attempt = AttemptView(
                entity_id("pa", [schema_name, mode, sequence]),
                {
                    "sequence": sequence,
                    "payload": {
                        "role_route_key": selector_route.route_key,
                        "response_schema_digest": self.manifest[
                            "response_schema_digests"
                        ][schema_name],
                        "response_mode": mode,
                    },
                },
                None,
                terminal,
            )
            view = LogicalActionView(
                entity_id("la", [schema_name, mode, sequence]),
                entity_id("wu", [schema_name, mode, sequence]),
                {"role": Role.SELECTOR.value},
                (attempt,),
            )
            return view

        unrelated = action("probe_review", "json_object", 1, None)
        self.assertEqual(
            _template_binding_mode_before(
                manifest=self.manifest,
                route=selector_route,
                logical_actions={unrelated.action_id: unrelated},
                response_schemas={},
                before_sequence=10,
            ),
            "json_schema",
        )
        binding = action(
            "template_binding",
            "json_schema",
            2,
            {
                "type": "model_call_failed",
                "payload": {"error_kind": "unsupported_response_format"},
            },
        )
        self.assertEqual(
            _template_binding_mode_before(
                manifest=self.manifest,
                route=selector_route,
                logical_actions={
                    unrelated.action_id: unrelated,
                    binding.action_id: binding,
                },
                response_schemas={},
                before_sequence=10,
            ),
            "json_object",
        )

    def test_derived_recipe_requires_one_identical_scope_bundle_across_steps(self) -> None:
        template = self.recipe.templates[0]
        unit_id = entity_id("wu", ["derived-scenario-source"])
        unit = WorkUnitView(
            {"sequence": 2},
            {"unit_id": unit_id, "kind": "portfolio_round"},
            (),
        )

        def episode(label: str):
            return SimpleNamespace(
                strategy_id=entity_id("st", [label]),
                strategy_scientific_key=digest_json([label]),
                parent_search_status=SearchStatus.PROMISING,
                rounds=(
                    SimpleNamespace(
                        lane=SimpleNamespace(
                            item=SimpleNamespace(template_candidate=template)
                        )
                    ),
                ),
                children=(),
            )

        prior = episode("prior")
        current = episode("current")
        prior_child = SimpleNamespace(
            child_scientific_key=digest_json(["prior-child"]),
            work_unit_id=unit_id,
        )
        current_child = SimpleNamespace(
            child_scientific_key=digest_json(["current-child"]),
            work_unit_id=unit_id,
            alignment_status="aligned",
            outcome=SimpleNamespace(
                aligned=True,
                resolved=True,
                confirmed_objective_high=True,
                confirmed_objective_application=True,
                outcome_result="supports",
            ),
            selected=SimpleNamespace(),
        )
        current.children = (current_child,)
        entry = {
            "prompt": "Reducer-owned derived entry.",
            "origin": "bootstrap",
        }
        frozen_bundle = (
            self.recipe.selector_scope_keys,
            self.recipe.perturbator_scope,
        )
        mismatched_bundle = (
            (
                self.recipe.selector_scope_keys[0],
                ("probe_review", self.alternate_probe_scope.scope_key),
                self.recipe.selector_scope_keys[2],
            ),
            self.recipe.perturbator_scope,
        )

        def derive(bundles):
            with (
                patch(
                    "quasilisk.run_state._scenario_episode_ancestry",
                    return_value=(entry, ((prior, prior_child),)),
                ),
                patch(
                    "quasilisk.run_state._scenario_source_scope_bundle",
                    side_effect=bundles,
                ),
            ):
                return _derive_scenario_recipes(
                    run_id=self.context.run_id,
                    scientific_scope_digest=self.context.scope,
                    manifest=self.manifest,
                    objective=self.context.objective,
                    work_units=(unit,),
                    logical_actions={},
                    target_results={},
                    response_schemas={},
                    conversation_nodes={},
                    portfolio_rounds={},
                    strategy_episodes={current.strategy_id: current},
                )

        self.assertEqual(
            derive((frozen_bundle, mismatched_bundle)),
            {},
        )
        self.assertEqual(len(derive((frozen_bundle, frozen_bundle))), 1)

    def test_replanning_ancestry_keeps_failure_without_fabricating_outcome_support(self) -> None:
        root_candidate_id = entity_id("pc", ["scenario-ancestry-root"])
        child_candidate_id = entity_id("pc", ["scenario-ancestry-child"])
        root_node_id = entity_id("cn", ["scenario-ancestry-root"])
        child_node_id = entity_id("cn", ["scenario-ancestry-child"])
        observation_id = entity_id("to", ["scenario-ancestry-child"])
        prior_strategy_id = entity_id("st", ["scenario-ancestry-prior"])
        root_node = SimpleNamespace(
            node_id=root_node_id,
            scientific_key=digest_json(["scenario-ancestry-root-node"]),
            candidate_id=root_candidate_id,
            parent_node_id=None,
            root_id=root_node_id,
        )
        child_node = SimpleNamespace(
            node_id=child_node_id,
            scientific_key=digest_json(["scenario-ancestry-child-node"]),
            candidate_id=child_candidate_id,
            parent_node_id=root_node_id,
            target_observation_id=observation_id,
            root_id=root_node_id,
        )
        outcome = SimpleNamespace(
            aligned=True,
            resolved=True,
            confirmed_objective_high=True,
            outcome_result=None,
        )
        prior_child = SimpleNamespace(
            candidate_id=child_candidate_id,
            candidate_scientific_key=digest_json(["scenario-ancestry-child-candidate"]),
            observation_id=observation_id,
            alignment_status="aligned",
            outcome=outcome,
            selected=SimpleNamespace(),
        )
        prior = SimpleNamespace(
            strategy_id=prior_strategy_id,
            children=(prior_child,),
        )
        current = SimpleNamespace(
            generation_parent_node_id=child_node_id,
            generation_parent_node_scientific_key=child_node.scientific_key,
        )
        candidates = {
            root_candidate_id: {
                "candidate_id": root_candidate_id,
                "scientific_key": digest_json(["scenario-ancestry-root-candidate"]),
                "origin": "bootstrap",
            },
            child_candidate_id: {
                "candidate_id": child_candidate_id,
                "scientific_key": prior_child.candidate_scientific_key,
                "origin": "stateful_strategy",
                "strategy_id": prior_strategy_id,
            },
        }
        arguments = {
            "episode": current,
            "episodes": {prior_strategy_id: prior},
            "conversation_nodes": {
                root_node_id: root_node,
                child_node_id: child_node,
            },
            "candidate_metadata": candidates,
        }

        self.assertIsNone(_scenario_episode_ancestry(**arguments))
        outcome.outcome_result = "supports"
        entry, ancestry = _scenario_episode_ancestry(**arguments)
        self.assertEqual(entry["candidate_id"], root_candidate_id)
        self.assertEqual(ancestry, ((prior, prior_child),))
        outcome.outcome_result = "falsifies"
        entry, ancestry = _scenario_episode_ancestry(**arguments)
        self.assertEqual(ancestry[0][1].outcome.outcome_result, "falsifies")


if __name__ == "__main__":
    unittest.main()
