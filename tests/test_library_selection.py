from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from integration_support import dryrun_config
from quasilisk.domain import (
    Role,
    canonical_json,
    digest_json,
    entity_id,
    normalize_line_endings,
    sha256_bytes,
)
from quasilisk.engine import (
    EngineCompatibilityError,
    EngineError,
    _current_structured_role_scope_candidates,
    _current_judge_protocol_digest,
    _current_objective_scope_key,
    _current_perturbator_scopes,
    _load_library_snapshots,
    _seed_prior_static_compatible,
    _select_verified_seed_snapshot,
    _strategy_prior_static_compatible,
    _target_static_scope,
    load_resources,
)
from quasilisk.evidence import (
    PerturbatorRoleScope,
    ScenarioRecipe,
    SeedVerificationRule,
    parse_scenario_recipe,
)
from quasilisk.strategy import parse_strategy_template


class VerifiedSeedSelectionTests(unittest.TestCase):
    def test_duplicate_index_identity_is_rejected_before_source_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resources = load_resources(
                dryrun_config("fatalism", root / "initial-runs")
            )
            seed_directory = root / "library" / "seeds"
            seed_directory.mkdir(parents=True)
            seed_id = entity_id("sd", ["duplicate-index-identity"])
            entry = {
                "qualification_summary": {},
                "source_checkpoint": {},
                "subject_payload": {"seed_id": seed_id},
            }
            (seed_directory / "index.json").write_text(
                canonical_json({"entries": [entry, entry]}),
                encoding="utf-8",
            )
            config = replace(
                resources.config,
                repository_root=root,
                seed_library_directory=seed_directory,
            )
            with self.assertRaisesRegex(EngineError, "duplicate seed_id"):
                _load_library_snapshots(config, resources.objective)

    def test_cells_use_declared_order_and_capacity_uses_fraction_then_median(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            resources = load_resources(
                dryrun_config("fatalism", Path(temporary) / "runs")
            )
        objective = resources.objective
        config = resources.config
        first_probe = objective.bootstrap["probes"][0]
        first_descriptors = dict(first_probe["descriptors"])
        second_descriptors = dict(first_descriptors)
        first_axis = objective.descriptor_names[0]
        values = objective.descriptor_values[first_axis]
        second_descriptors[first_axis] = next(
            value for value in values if value != first_descriptors[first_axis]
        )

        def prior(
            label: str,
            descriptors: dict[str, str],
            fraction: str,
            median: str,
        ) -> dict[str, object]:
            subject = {
                "seed_id": entity_id("sd", [label]),
                "type": "entry_seed",
                "entry_prompt": f"Prompt {label}",
                "scenario_recipe": None,
                "objective_id": objective.id,
                "plugin_digest": objective.plugin_digest,
                "target_generation_profile_digest": config.target.digest,
                "descriptors": descriptors,
                "observed_success_fraction": fraction,
            }
            return {
                "subject_payload": subject,
                "qualification_summary": {"verification_median": median},
            }

        priors = (
            prior("lower-fraction", first_descriptors, "0.7", "99"),
            prior("lower-median", first_descriptors, "0.8", "70"),
            prior("winner", first_descriptors, "0.8", "80"),
            prior("other-cell", second_descriptors, "0.9", "90"),
        )
        selected = _select_verified_seed_snapshot(
            priors,
            config=config,
            objective=objective,
        )

        self.assertEqual(len(selected), 2)
        by_cell = {tuple(item["descriptors"].values()): item for item in selected}
        self.assertEqual(
            {item["entry_prompt"] for item in selected},
            {"Prompt winner", "Prompt other-cell"},
        )
        self.assertEqual(len(by_cell), 2)

    def test_exact_prompt_duplicate_is_retained_only_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            resources = load_resources(
                dryrun_config("fatalism", Path(temporary) / "runs")
            )
        objective = resources.objective
        config = resources.config
        descriptors = deepcopy(objective.bootstrap["probes"][0]["descriptors"])

        def prior(label: str, fraction: str) -> dict[str, object]:
            return {
                "subject_payload": {
                    "seed_id": entity_id("sd", [label]),
                    "type": "entry_seed",
                    "entry_prompt": "Same exact prompt",
                    "scenario_recipe": None,
                    "objective_id": objective.id,
                    "plugin_digest": objective.plugin_digest,
                    "target_generation_profile_digest": config.target.digest,
                    "descriptors": descriptors,
                    "observed_success_fraction": fraction,
                },
                "qualification_summary": {"verification_median": "50"},
            }

        selected = _select_verified_seed_snapshot(
            (prior("lower", "0.7"), prior("higher", "0.8")),
            config=config,
            objective=objective,
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["seed_id"], entity_id("sd", ["higher"]))


class StaticLibraryAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config_source = dryrun_config("fatalism", self.root / "runs")
        self.resources = load_resources(self.config_source)
        self.structured_role_candidates = _current_structured_role_scope_candidates(
            config=self.resources.config,
            template_digests=self.resources.template_digests,
            schema_digests=self.resources.schema_digests,
        )
        self.perturbator_scopes = _current_perturbator_scopes(
            config=self.resources.config,
            template_digests=self.resources.template_digests,
            schema_digests=self.resources.schema_digests,
        )
        self.objective_key = _current_objective_scope_key(
            self.resources.objective, self.resources.schema_digests
        )
        self.judge_protocol = _current_judge_protocol_digest(
            self.resources.template_digests, self.resources.schema_digests
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _selector_scope(self, name: str):
        return next(
            scope
            for base, scope in self.structured_role_candidates
            if base[0] == Role.SELECTOR.value
            and base[1] == "selector_primary"
            and scope.scope == name
            and scope.response_mode == "json_schema"
        )

    def _judge_context(self):
        judges = tuple(
            sorted(
                (
                    f"judge_{base[1]}",
                    scope.scope_key,
                )
                for base, scope in self.structured_role_candidates
                if base[0] == Role.JUDGE.value
                and base[1] in {"primary", "secondary_a", "secondary_b"}
                and scope.response_mode == "json_schema"
            )
        )
        panel_key = digest_json(
            [
                "judge_panel_qualification",
                self.judge_protocol,
                [list(item) for item in judges],
            ]
        )
        pairs = tuple(sorted((*judges, ("panel", panel_key))))
        return judges, panel_key, pairs

    def _perturbator_payload(self) -> dict[str, object]:
        return deepcopy(
            next(
                payload
                for payload in self.perturbator_scopes.values()
                if payload["requested_model"]
                == self.resources.config.perturbator.primary().model
                and payload["response_mode"] == "json_schema"
            )
        )

    def _template(self):
        descriptors = dict(
            self.resources.objective.bootstrap["probes"][0]["descriptors"]
        )
        descriptors["argument_move"] = "answer_objection"
        return parse_strategy_template(
            {
                "purpose": "repair",
                "applicability": {
                    "diagnosis": "conditional_acceptance",
                    "requires": "A stated condition remains unresolved.",
                    "excludes": "No stated condition remains.",
                },
                "move": {
                    "act": "repair_grounded_gap",
                    "target_kind": "condition",
                    "descriptors": descriptors,
                    "blueprint": "Resolve the condition without changing the premise.",
                },
                "hypothesis": "Resolving the condition should permit progress.",
                "preserve": [
                    {
                        "kind": "meaning",
                        "statement": "Preserve the accepted meaning.",
                    }
                ],
                "vary": [
                    {
                        "dimension": "framing",
                        "instruction": "Vary only this declared dimension.",
                    }
                ],
                "success": {
                    "transition": {"from": "conditional", "to": "resolved"},
                    "supports": "The target resolves its stated condition.",
                    "falsifies": "The target retains the condition.",
                },
            },
            self.resources.objective.descriptor_values,
            self.resources.objective.diagnosis_categories,
        )

    def _refresh_strategy_key(self, subject: dict[str, object]) -> None:
        template = parse_strategy_template(
            subject["template"],
            self.resources.objective.descriptor_values,
            self.resources.objective.diagnosis_categories,
        )
        target_scope = {
            "scope_key": subject["target_role_scope_key"],
            "requested_provider_route_key": subject[
                "target_requested_provider_route_key"
            ],
            "requested_model": subject["target_requested_model"],
            "generation_profile_digest": subject[
                "target_generation_profile_digest"
            ],
            "identity_token": subject["identity_token"],
            "identity_unverifiable": subject["identity_unverifiable"],
        }
        subject["compatibility_key"] = digest_json(
            [
                "qualified_strategy_compatibility",
                self.resources.objective.id,
                self.resources.objective.plugin_digest,
                target_scope,
                template.content_hash,
                template.applicability.diagnosis,
                template.move.act,
                template.move.target_kind,
                list(subject["tested_entry_context_ids"]),
                template.applicability.to_dict(),
                [list(item) for item in subject["selector_scope_keys"]],
                [dict(item) for item in subject["perturbator_scopes"]],
                dict(subject["development_design"]),
                dict(subject["holdout_design"]),
                subject["development_design"]["planned_count_per_arm"],
                subject["holdout_design"]["sealed_holdout_plan"][
                    "planned_count_per_arm"
                ],
            ]
        )

    def _strategy_subject(self) -> dict[str, object]:
        template = self._template()
        target = _target_static_scope(self.resources.config)
        identity_token = (
            "requested:"
            f"{target['requested_provider_route_key']}/{target['requested_model']}"
        )
        target_scope_key = digest_json(
            [
                "target_role_scope",
                target["requested_provider_route_key"],
                target["requested_model"],
                target["generation_profile_digest"],
                identity_token,
                True,
            ]
        )
        selector = self._selector_scope("probe_review")
        perturbator = self._perturbator_payload()
        judge = next(
            scope.scope_key
            for base, scope in self.structured_role_candidates
            if base[:2] == (Role.JUDGE.value, "primary")
            and scope.response_mode == "json_schema"
        )
        subject: dict[str, object] = {
            "template_id": entity_id("tp", ["library-admission"]),
            "qualification_status": "qualified",
            "objective_id": self.resources.objective.id,
            "plugin_digest": self.resources.objective.plugin_digest,
            "objective_scope_key": self.objective_key,
            "target_requested_provider_route_key": target[
                "requested_provider_route_key"
            ],
            "target_requested_model": target["requested_model"],
            "target_generation_profile_digest": target[
                "generation_profile_digest"
            ],
            "target_role_scope_key": target_scope_key,
            "identity_token": identity_token,
            "identity_unverifiable": True,
            "template": template.to_dict(),
            "diagnosis": template.applicability.diagnosis,
            "move_act": template.move.act,
            "target_kind": template.move.target_kind,
            "applicability": template.applicability.to_dict(),
            "tested_entry_context_ids": [entity_id("pc", ["library-entry"])],
            "selector_scope_keys": [[selector.scope, selector.scope_key]],
            "judge_scope_keys": [judge],
            "required_structured_role_scope_keys": sorted(
                (selector.scope_key, judge)
            ),
            "perturbator_scopes": [perturbator],
            "required_role_scope_keys": sorted(
                (target_scope_key, str(perturbator["scope_key"]))
            ),
            "development_design": {"planned_count_per_arm": 5},
            "holdout_design": {
                "sealed_holdout_plan": {"planned_count_per_arm": 5}
            },
        }
        self._refresh_strategy_key(subject)
        return subject

    def _refresh_entry_seed_key(
        self,
        subject: dict[str, object],
        *,
        taxonomy_digest: str | None = None,
    ) -> None:
        _, _, judge_pairs = self._judge_context()
        subject["compatibility_key"] = digest_json(
            [
                "verified_entry_seed_compatibility",
                self.resources.objective.id,
                self.resources.objective.plugin_digest,
                self.resources.config.target.digest,
                subject["identity_token"],
                taxonomy_digest
                or digest_json(list(self.resources.objective.objective["descriptors"])),
                [list(item) for item in judge_pairs],
                dict(subject["verification_rule"]),
                None,
            ]
        )

    def _refresh_scenario_seed_key(self, subject: dict[str, object]) -> None:
        recipe = self._parse_scenario(subject)
        _, _, judge_pairs = self._judge_context()
        subject["compatibility_key"] = digest_json(
            [
                "verified_scenario_seed_compatibility",
                self.resources.objective.id,
                self.resources.objective.plugin_digest,
                self.resources.config.target.digest,
                subject["identity_token"],
                digest_json(list(self.resources.objective.objective["descriptors"])),
                [list(item) for item in judge_pairs],
                recipe.content_hash,
                [item.content_hash for item in recipe.templates],
                {
                    "designated_realization_policies": list(
                        recipe.designated_realization_policies
                    ),
                    "maximum_target_turns": recipe.maximum_target_turns,
                    "maximum_strategy_rounds": recipe.maximum_strategy_rounds,
                },
                [list(item) for item in recipe.selector_scope_keys],
                recipe.perturbator_scope.key_material(),
                dict(recipe.measurement_cell),
                dict(subject["verification_rule"]),
            ]
        )

    def _parse_scenario(self, subject: dict[str, object]) -> ScenarioRecipe:
        return parse_scenario_recipe(
            subject["scenario_recipe"],
            self.resources.objective.descriptor_values,
            self.resources.objective.diagnosis_categories,
        )

    def _entry_seed_subject(self, label: str = "compatible") -> dict[str, object]:
        target = _target_static_scope(self.resources.config)
        judges, panel_key, _ = self._judge_context()
        identity_token = (
            "requested:"
            f"{target['requested_provider_route_key']}/{target['requested_model']}"
        )
        subject: dict[str, object] = {
            "seed_id": entity_id("sd", ["library", label]),
            "type": "entry_seed",
            "entry_prompt": f"Library seed {label}",
            "scenario_recipe": None,
            "objective_id": self.resources.objective.id,
            "plugin_digest": self.resources.objective.plugin_digest,
            "target_generation_profile_digest": self.resources.config.target.digest,
            "descriptors": dict(
                self.resources.objective.bootstrap["probes"][0]["descriptors"]
            ),
            "verification_rule": SeedVerificationRule(
                self.resources.config.samples.verification
            ).to_dict(),
            "planned_count": self.resources.config.samples.verification,
            "resolved_count": self.resources.config.samples.verification,
            "high_count": 4,
            "observed_success_fraction": "0.8",
            "judge_protocol_digest": self.judge_protocol,
            "qualification_digests": sorted(
                {
                    self.objective_key,
                    *(key for _, key in judges),
                    panel_key,
                }
            ),
            "identity_token": identity_token,
            "identity_unverifiable": True,
        }
        self._refresh_entry_seed_key(subject)
        return subject

    def _scenario_seed_subject(self) -> dict[str, object]:
        prompt = "Library stateful scenario."
        template = self._template()
        selectors = tuple(
            (name, self._selector_scope(name).scope_key)
            for name in ("outcome_review", "probe_review", "template_binding")
        )
        perturbator_payload = self._perturbator_payload()
        perturbator = PerturbatorRoleScope(
            str(perturbator_payload["requested_provider_route_key"]),
            str(perturbator_payload["requested_model"]),
            str(perturbator_payload["generation_profile_digest"]),
            str(perturbator_payload["response_mode"]),
            tuple(
                (str(item[0]), str(item[1]))
                for item in perturbator_payload["prompt_mode_digests"]
            ),
        )
        descriptors = dict(
            self.resources.objective.bootstrap["probes"][0]["descriptors"]
        )
        recipe = ScenarioRecipe(
            sha256_bytes(normalize_line_endings(prompt).encode("utf-8")),
            self.resources.config.target.digest,
            (template,),
            tuple(
                (name, descriptors[name])
                for name in self.resources.objective.descriptor_names
            ),
            2,
            1,
            ("anchor",),
            selectors,
            perturbator,
        )
        subject = self._entry_seed_subject("scenario")
        subject.update(
            {
                "type": "scenario_recipe",
                "entry_prompt": prompt,
                "scenario_recipe": recipe.to_dict(),
                "descriptors": descriptors,
            }
        )
        judges, panel_key, _ = self._judge_context()
        subject["qualification_digests"] = sorted(
            {
                self.objective_key,
                *(key for _, key in judges),
                panel_key,
                dict(selectors)["probe_review"],
                dict(selectors)["outcome_review"],
                perturbator.scope_key,
            }
        )
        self._refresh_scenario_seed_key(subject)
        return subject

    def _strategy_compatible(self, subject: dict[str, object]) -> bool:
        return _strategy_prior_static_compatible(
            subject,
            config=self.resources.config,
            objective=self.resources.objective,
            objective_scope_key=self.objective_key,
            structured_role_candidates=self.structured_role_candidates,
            perturbator_scopes=self.perturbator_scopes,
        )

    def _seed_compatible(self, subject: dict[str, object]) -> bool:
        return _seed_prior_static_compatible(
            subject,
            config=self.resources.config,
            objective=self.resources.objective,
            objective_scope_key=self.objective_key,
            judge_protocol_digest=self.judge_protocol,
            structured_role_candidates=self.structured_role_candidates,
            perturbator_scopes=self.perturbator_scopes,
        )

    def _load_frozen(
        self,
        *,
        seeds,
        strategies,
    ):
        return load_resources(
            self.config_source,
            frozen_bootstrap=self.resources.bootstrap_payloads,
            frozen_seeds=seeds,
            frozen_strategies=strategies,
            frozen_resume=True,
        )

    def test_strategy_requires_exact_target_selector_and_perturbator_scope(self) -> None:
        subject = self._strategy_subject()
        self.assertTrue(self._strategy_compatible(subject))

        cases: dict[str, dict[str, object]] = {}
        wrong_target = deepcopy(subject)
        wrong_target["target_requested_provider_route_key"] = digest_json(
            ["other-target-route"]
        )
        cases["target"] = wrong_target

        wrong_selector = deepcopy(subject)
        wrong_selector["selector_scope_keys"][0][1] = digest_json(
            ["other-selector-scope"]
        )
        wrong_selector["required_structured_role_scope_keys"] = sorted(
            (
                wrong_selector["selector_scope_keys"][0][1],
                *wrong_selector["judge_scope_keys"],
            )
        )
        self._refresh_strategy_key(wrong_selector)
        cases["selector"] = wrong_selector

        wrong_perturbator = deepcopy(subject)
        payload = wrong_perturbator["perturbator_scopes"][0]
        payload["prompt_mode_digests"][0][1] = digest_json(
            ["other-perturbator-prompt"]
        )
        changed_scope = PerturbatorRoleScope(
            payload["requested_provider_route_key"],
            payload["requested_model"],
            payload["generation_profile_digest"],
            payload["response_mode"],
            tuple(tuple(item) for item in payload["prompt_mode_digests"]),
        )
        payload["scope_key"] = changed_scope.scope_key
        wrong_perturbator["required_role_scope_keys"] = sorted(
            (wrong_perturbator["target_role_scope_key"], changed_scope.scope_key)
        )
        self._refresh_strategy_key(wrong_perturbator)
        cases["perturbator"] = wrong_perturbator

        inconsistent = deepcopy(subject)
        inconsistent["move_act"] = "other_move"
        cases["redundant semantics"] = inconsistent

        invalid_template = deepcopy(subject)
        invalid_template["template"]["shadow"] = "not canonical"
        cases["strict template"] = invalid_template

        for label, candidate in cases.items():
            with self.subTest(label=label):
                self.assertFalse(self._strategy_compatible(candidate))

        changed_judge = deepcopy(subject)
        changed_judge["judge_scope_keys"] = [digest_json(["new-source-judge"])]
        changed_judge["required_structured_role_scope_keys"] = sorted(
            (
                changed_judge["selector_scope_keys"][0][1],
                *changed_judge["judge_scope_keys"],
            )
        )
        self.assertTrue(self._strategy_compatible(changed_judge))

    def test_seed_static_key_covers_taxonomy_protocol_rule_and_recipe_scopes(self) -> None:
        entry = self._entry_seed_subject()
        scenario = self._scenario_seed_subject()
        self.assertTrue(self._seed_compatible(entry))
        self.assertTrue(self._seed_compatible(scenario))

        stale_taxonomy = deepcopy(entry)
        self._refresh_entry_seed_key(
            stale_taxonomy, taxonomy_digest=digest_json(["old-taxonomy"])
        )
        stale_protocol = deepcopy(entry)
        stale_protocol["judge_protocol_digest"] = digest_json(["old-protocol"])
        stale_rule = deepcopy(entry)
        stale_rule["verification_rule"]["planned_count"] += 1

        recipe = self._parse_scenario(scenario)
        stale_selector = deepcopy(scenario)
        selector_keys = list(recipe.selector_scope_keys)
        old_probe_key = selector_keys[1][1]
        selector_keys[1] = (
            "probe_review",
            digest_json(["old-probe-review-scope"]),
        )
        selector_recipe = ScenarioRecipe(
            recipe.entry_prompt_hash,
            recipe.target_generation_profile_digest,
            recipe.templates,
            recipe.measurement_cell,
            recipe.maximum_target_turns,
            recipe.maximum_strategy_rounds,
            recipe.designated_realization_policies,
            tuple(selector_keys),
            recipe.perturbator_scope,
        )
        stale_selector["scenario_recipe"] = selector_recipe.to_dict()
        stale_selector["qualification_digests"] = sorted(
            selector_keys[1][1] if item == old_probe_key else item
            for item in stale_selector["qualification_digests"]
        )
        self._refresh_scenario_seed_key(stale_selector)

        stale_perturbator = deepcopy(scenario)
        prompt_modes = list(recipe.perturbator_scope.prompt_mode_digests)
        prompt_modes[0] = (
            prompt_modes[0][0],
            digest_json(["old-perturbator-prompt-mode"]),
        )
        perturbator = PerturbatorRoleScope(
            recipe.perturbator_scope.requested_provider_route_key,
            recipe.perturbator_scope.requested_model,
            recipe.perturbator_scope.generation_profile_digest,
            recipe.perturbator_scope.response_mode,
            tuple(prompt_modes),
        )
        perturbator_recipe = ScenarioRecipe(
            recipe.entry_prompt_hash,
            recipe.target_generation_profile_digest,
            recipe.templates,
            recipe.measurement_cell,
            recipe.maximum_target_turns,
            recipe.maximum_strategy_rounds,
            recipe.designated_realization_policies,
            recipe.selector_scope_keys,
            perturbator,
        )
        stale_perturbator["scenario_recipe"] = perturbator_recipe.to_dict()
        stale_perturbator["qualification_digests"] = sorted(
            perturbator.scope_key
            if item == recipe.perturbator_scope.scope_key
            else item
            for item in stale_perturbator["qualification_digests"]
        )
        self._refresh_scenario_seed_key(stale_perturbator)

        for label, candidate in (
            ("taxonomy", stale_taxonomy),
            ("judge protocol", stale_protocol),
            ("verification rule", stale_rule),
            ("scenario selector", stale_selector),
            ("scenario perturbator", stale_perturbator),
        ):
            with self.subTest(label=label):
                self.assertFalse(self._seed_compatible(candidate))

    def test_load_resources_filters_priors_before_scope_materialization(self) -> None:
        seed = self._entry_seed_subject()
        stale_seed = deepcopy(seed)
        stale_seed["seed_id"] = entity_id("sd", ["library", "stale"])
        stale_seed["entry_prompt"] = "Stale but higher-ranked seed"
        stale_seed["observed_success_fraction"] = "1"
        self._refresh_entry_seed_key(
            stale_seed, taxonomy_digest=digest_json(["stale-taxonomy"])
        )
        strategy = self._strategy_subject()
        stale_strategy = deepcopy(strategy)
        stale_strategy["selector_scope_keys"][0][1] = digest_json(
            ["stale-selector"]
        )
        stale_strategy["required_structured_role_scope_keys"] = sorted(
            (
                stale_strategy["selector_scope_keys"][0][1],
                *stale_strategy["judge_scope_keys"],
            )
        )
        self._refresh_strategy_key(stale_strategy)
        wrapper = lambda subject, summary=None: {
            "subject_payload": subject,
            "qualification_summary": summary or {},
        }
        with patch(
            "quasilisk.engine._load_library_snapshots",
            return_value=(
                (
                    wrapper(seed, {"verification_median": "80"}),
                    wrapper(stale_seed, {"verification_median": "100"}),
                ),
                (wrapper(strategy), wrapper(stale_strategy)),
            ),
        ):
            loaded = load_resources(self.config_source)

        self.assertEqual(
            tuple(item["seed_id"] for item in loaded.seed_snapshot),
            (seed["seed_id"],),
        )
        self.assertEqual(len(loaded.strategy_snapshot), 1)
        self.assertEqual(
            loaded.strategy_snapshot[0]["qualification_status"],
            "provisionally_compatible",
        )

    def test_frozen_resume_recomputes_static_material_and_rejects_duplicates(self) -> None:
        seed = self._entry_seed_subject()
        strategy = self._strategy_subject()
        strategy["qualification_status"] = "provisionally_compatible"
        loaded = self._load_frozen(
            seeds=(seed,),
            strategies=(strategy,),
        )
        self.assertEqual(loaded.seed_snapshot, (seed,))
        self.assertEqual(loaded.strategy_snapshot, (strategy,))

        stale_seed = deepcopy(seed)
        stale_seed["compatibility_key"] = digest_json(["tampered-seed-key"])
        stale_strategy = deepcopy(strategy)
        stale_strategy["move_act"] = "tampered_move"
        qualified_frozen_strategy = deepcopy(strategy)
        qualified_frozen_strategy["qualification_status"] = "qualified"
        cases = (
            ("seed", (stale_seed,), (strategy,)),
            ("strategy", (seed,), (stale_strategy,)),
            (
                "strategy status",
                (seed,),
                (qualified_frozen_strategy,),
            ),
            ("duplicate seed", (seed, seed), (strategy,)),
            ("duplicate strategy", (seed,), (strategy, strategy)),
        )
        for label, seeds, strategies in cases:
            with self.subTest(label=label), self.assertRaises(
                EngineCompatibilityError
            ):
                self._load_frozen(
                    seeds=seeds,
                    strategies=strategies,
                )


if __name__ == "__main__":
    unittest.main()
