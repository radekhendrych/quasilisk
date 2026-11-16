from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from integration_support import dryrun_config
from foundation_support import (
    FIXED_TIMESTAMP,
    call_fixture,
    failure_payload,
    intended_payload,
    run_created_payload,
    success_payload,
)
from quasilisk.domain import (
    JsonSchemaValidationError,
    RoleCallVector,
    canonical_json,
    entity_id,
    scientific_key,
    strict_json_loads,
    validate_json_schema_contract,
    validate_json_schema_instance,
)
from quasilisk.engine import (
    RunContext,
    _initialize_routes,
    _manifest,
    build_strategy_development,
    build_strategy_holdout,
    load_resources,
)
from quasilisk.events import (
    AuthoritativeSchemas,
    JournalCorrupt,
    ZERO_HASH,
    build_event,
    validate_payload,
)
from quasilisk.run_state import _strategy_snapshot
from quasilisk.strategy import QualificationState, parse_strategy_template


class SchemaCompositionTests(unittest.TestCase):
    def test_prefix_items_are_checked_positionally_and_close_the_tail(self) -> None:
        schema = {
            "type": "array",
            "prefixItems": [
                {"type": "string", "const": "first"},
                {"type": "integer"},
            ],
            "items": False,
            "minItems": 2,
            "maxItems": 2,
        }
        validate_json_schema_contract(schema)
        validate_json_schema_instance(["first", 2], schema)
        for invalid in (["first", "2"], ["first", 2, 3]):
            with self.subTest(invalid=invalid), self.assertRaises(
                JsonSchemaValidationError
            ):
                validate_json_schema_instance(invalid, schema)

    def test_external_refs_use_only_the_explicit_registry(self) -> None:
        work_unit = {
            "$id": "https://quasilisk.local/schemas/work_unit.json",
            "type": "object",
            "properties": {"tag": {"const": "unit"}},
            "required": ["tag"],
            "additionalProperties": False,
        }
        event = {
            "$id": "https://quasilisk.local/schemas/event.json",
            "type": "object",
            "properties": {"unit": {"$ref": "work_unit.json"}},
            "required": ["unit"],
            "additionalProperties": False,
        }
        registry = {"work_unit.json": work_unit}
        validate_json_schema_contract(event, registry=registry)
        validate_json_schema_instance(
            {"unit": {"tag": "unit"}}, event, registry=registry
        )
        with self.assertRaises(JsonSchemaValidationError):
            validate_json_schema_instance(
                {"unit": {"tag": "unit", "shadow": True}},
                event,
                registry=registry,
            )


class AuthoritativeNestedContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = call_fixture(label="schema-contract")

    def test_work_unit_schema_closes_nested_design(self) -> None:
        work_unit_schema = {
            "$id": "https://quasilisk.local/schemas/work_unit.json",
            "type": "object",
            "properties": {
                "design": {
                    "type": "object",
                    "properties": {"tag": {"const": "initial_observation"}},
                    "required": ["tag"],
                    "additionalProperties": False,
                }
            },
            "required": ["design"],
        }
        event_schema = {
            "$id": "https://quasilisk.local/schemas/event.json",
            "type": "object",
            "properties": {
                "unit": {"$ref": "work_unit.json"},
            },
            "required": ["unit"],
        }
        schemas = AuthoritativeSchemas(event_schema, work_unit_schema)
        schemas.validate_event({"unit": self.fixture["unit"]})

        invalid = deepcopy(self.fixture["unit"])
        invalid["design"]["shadow"] = "not frozen"
        with self.assertRaisesRegex(JournalCorrupt, "authoritative schema"):
            schemas.validate_work_unit(invalid)

    def test_decimal_strings_and_nullable_ids_are_canonical(self) -> None:
        intended = intended_payload(self.fixture)
        intended["generation_parameters"]["temperature"] = "1.0"
        with self.assertRaisesRegex(JournalCorrupt, "canonical"):
            validate_payload("model_call_intended", intended)

        succeeded = success_payload(self.fixture)
        succeeded["cost"] = "0.00"
        with self.assertRaisesRegex(JournalCorrupt, "canonical"):
            validate_payload("model_call_succeeded", succeeded)
        succeeded = success_payload(self.fixture)
        succeeded["provider_request_id"] = ""
        with self.assertRaisesRegex(JournalCorrupt, "nonempty"):
            validate_payload("model_call_succeeded", succeeded)

        failed = failure_payload(self.fixture)
        failed["retry_after_seconds"] = "01"
        with self.assertRaisesRegex(JournalCorrupt, "canonical"):
            validate_payload("model_call_failed", failed)

    def test_encoded_base64_must_be_canonical(self) -> None:
        succeeded = success_payload(self.fixture)
        succeeded["raw_body"] = {"encoding": "base64", "data": "YQ"}
        with self.assertRaisesRegex(JournalCorrupt, "base64"):
            validate_payload("model_call_succeeded", succeeded)

    def test_timestamp_must_name_a_real_calendar_instant(self) -> None:
        payload = run_created_payload(label="invalid-calendar-time")
        with self.assertRaisesRegex(JournalCorrupt, "calendar"):
            build_event(
                run_id=payload["run_id"],
                sequence=1,
                event_type="run_created",
                payload=payload,
                previous_hash=ZERO_HASH,
                timestamp_utc="2026-99-18T12:00:00.000000Z",
            )
        build_event(
            run_id=payload["run_id"],
            sequence=1,
            event_type="run_created",
            payload=payload,
            previous_hash=ZERO_HASH,
            timestamp_utc=FIXED_TIMESTAMP,
        )

class StrategyTemplateSnapshotContractTests(unittest.TestCase):
    @staticmethod
    def _metric(arm: str) -> SimpleNamespace:
        treatment = arm == "treatment"
        return SimpleNamespace(
            arm=arm,
            planned_count=5,
            terminal_count=5,
            resolved_count=5,
            unresolved_count=0,
            pending_count=0,
            applicable_count=5 if treatment else None,
            inapplicable_count=0 if treatment else None,
            objective_successes=5,
            strategy_fidelity_successes=5 if treatment else 0,
            exceptional_count=1,
            effective_scores=(Decimal("80"),),
            effective_score_median=Decimal("80"),
            applicability_rate=Decimal("1") if treatment else None,
            conditional_success_rate=Decimal("1") if treatment else None,
            unconditional_end_to_end_success_rate=Decimal("1"),
        )

    def _materialized_snapshot(
        self,
        *,
        selector_scope_material: str = "selector-scope",
        judge_scope_material: str = "judge-scope",
    ) -> tuple[dict, dict, dict]:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resources = load_resources(dryrun_config("fatalism", root / "runs"))
            run_id = entity_id("run", ["strategy-snapshot-contract"])
            context = RunContext(
                resources=resources,
                run_id=run_id,
                run_nonce="0" * 32,
                invocation_id=entity_id("iv", ["strategy-snapshot-contract"]),
                run_directory=root / "run",
                candidates={},
                scenarios={},
                routes={},
            )
            _initialize_routes(context)
            template = parse_strategy_template(
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
                        "descriptors": {
                            "objective_route": "causal_fixity",
                            "argument_move": "answer_objection",
                            "application_scope": "reasoning_policy",
                        },
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
                context.objective.descriptor_values,
                context.objective.diagnosis_categories,
            )
            development = build_strategy_development(
                context,
                template,
                parent_unit_id=entity_id("wu", ["strategy-snapshot-parent"]),
            )
            holdout = build_strategy_holdout(
                context,
                template,
                source_development_unit=development.to_dict(),
            )
            development_decision = SimpleNamespace(
                status=QualificationState.ELIGIBLE,
                arms=tuple(
                    self._metric(arm)
                    for arm in (
                        "treatment",
                        "parent_baseline",
                        "strategy_agnostic",
                    )
                ),
                all_roots_terminal=True,
                all_roots_resolved=True,
                strict_treatment_repeat_support=True,
                treatment_control_advantage=True,
                exceptional_requirement=True,
                purpose_gate=SimpleNamespace(
                    purpose="repair",
                    supporting_evidence_satisfied=True,
                    treatment_median_improves_each_control=True,
                    passed=True,
                ),
            )
            holdout_decision = SimpleNamespace(
                status=QualificationState.QUALIFIED,
                arms=tuple(
                    self._metric(arm)
                    for arm in ("treatment", "strategy_agnostic")
                ),
                all_roots_terminal=True,
                all_roots_resolved=True,
                strict_treatment_repeat_support=True,
                treatment_control_advantage=True,
                exceptional_requirement=True,
            )
            digest = scientific_key(["strategy-snapshot-scope"])
            perturbator_scope = {
                "scope_key": scientific_key(["perturbator-scope"]),
                "requested_provider_route_key": scientific_key(
                    ["perturbator-provider-route"]
                ),
                "requested_model": "scripted-perturbator",
                "generation_profile_digest": scientific_key(
                    ["perturbator-generation-profile"]
                ),
                "response_mode": "json_schema",
                "prompt_mode_digests": [["strategy", digest]],
            }
            target_binding = next(
                binding
                for binding in context.routes.values()
                if binding.configured_slot == "target"
            )
            target_scope = {
                "scope_key": scientific_key(["target-role-scope"]),
                "requested_provider_route_key": (
                    target_binding.requested_provider_route_key
                ),
                "requested_model": target_binding.requested_model,
                "generation_profile_digest": (
                    target_binding.generation_profile_digest
                ),
                "identity_token": "requested:scripted-target",
                "identity_unverifiable": True,
            }
            selector_scope = scientific_key([selector_scope_material])
            judge_scope = scientific_key([judge_scope_material])
            view = SimpleNamespace(
                qualification_state=QualificationState.QUALIFIED,
                template_id=development.design["strategy_template_id"],
                template_scientific_key=development.design[
                    "strategy_template_scientific_key"
                ],
                template=template,
                source_episode_id=entity_id("st", ["source-strategy"]),
                source_conversation_root_id=entity_id("cr", ["source-root"]),
                source_conversation_root_scientific_key=scientific_key(
                    ["source-root"]
                ),
                source_generation_parent_node_id=entity_id(
                    "cn", ["source-parent-node"]
                ),
                source_generation_parent_node_scientific_key=scientific_key(
                    ["source-parent-node"]
                ),
                source_entry_candidate_id=entity_id("pc", ["source-entry"]),
                source_entry_candidate_scientific_key=scientific_key(
                    ["source-entry"]
                ),
                tested_entry_context_ids=(
                    entity_id("pc", ["source-entry"]),
                    development.execution_slots[0]["execution_id"],
                ),
                required_selector_scope_keys=(("probe_review", selector_scope),),
                required_judge_scope_keys=(judge_scope,),
                required_structured_role_scope_keys=(selector_scope, judge_scope),
                required_role_scope_keys=(
                    target_scope["scope_key"],
                    perturbator_scope["scope_key"],
                ),
                development_unit_id=development.unit_id,
                development_design=development.design,
                sealed_holdout_plan=development.design["sealed_holdout_plan"],
                sealed_holdout_digest=development.design["sealed_holdout_digest"],
                development_decision=development_decision,
                holdout_unit_id=holdout.unit_id,
                holdout_design=holdout.design,
                holdout_decision=holdout_decision,
                evidence_event_ids=(entity_id("ev", ["strategy-evidence"]),),
                source_checkpoint={
                    "source_run_id": run_id,
                    "sequence": 1,
                    "event_hash": scientific_key(["strategy-checkpoint"]),
                    "source_path": (root / "run" / "events.jsonl").as_posix(),
                },
            )
            snapshot = strict_json_loads(
                canonical_json(
                    _strategy_snapshot(
                        view,
                        manifest=_manifest(context),
                        target_scope=target_scope,
                        perturbator_scopes=(perturbator_scope,),
                        objective_scope_key=scientific_key(
                            ["objective-scope"]
                        ),
                        call_counts=RoleCallVector().to_dict(),
                        qualified_at=FIXED_TIMESTAMP,
                    )
                )
            )
            return (
                snapshot,
                dict(resources.schemas["event.json"]),
                dict(resources.schemas["work_unit.json"]),
            )

    def test_materialized_qualified_snapshot_is_exactly_schema_valid(self) -> None:
        snapshot, event_schema, work_unit_schema = self._materialized_snapshot()
        snapshot_schema = {
            "$ref": "#/$defs/StrategyTemplateSnapshot",
            "$defs": event_schema["$defs"],
        }
        registry = {
            "work_unit.json": work_unit_schema,
            work_unit_schema["$id"]: work_unit_schema,
        }
        validate_json_schema_instance(snapshot, snapshot_schema, registry=registry)

        unknown = deepcopy(snapshot)
        unknown["shadow_status"] = "qualified"
        with self.assertRaises(JsonSchemaValidationError):
            validate_json_schema_instance(unknown, snapshot_schema, registry=registry)

        missing = deepcopy(snapshot)
        del missing["target_role_scope_key"]
        with self.assertRaises(JsonSchemaValidationError):
            validate_json_schema_instance(missing, snapshot_schema, registry=registry)

    def test_strategy_compatibility_uses_selector_not_judge_scopes(self) -> None:
        baseline, _, _ = self._materialized_snapshot()
        changed_judge, _, _ = self._materialized_snapshot(
            judge_scope_material="changed-judge-scope"
        )
        changed_selector, _, _ = self._materialized_snapshot(
            selector_scope_material="changed-selector-scope"
        )

        self.assertEqual(
            baseline["compatibility_key"], changed_judge["compatibility_key"]
        )
        self.assertNotEqual(
            baseline["compatibility_key"], changed_selector["compatibility_key"]
        )

    def test_run_manifest_accepts_only_explicitly_provisional_strategy_priors(self) -> None:
        snapshot, event_schema, work_unit_schema = self._materialized_snapshot()
        manifest_array_schema = {
            "type": "array",
            "items": event_schema["$defs"]["Manifest"]["properties"][
                "strategy_template_snapshot"
            ]["items"],
            "$defs": event_schema["$defs"],
        }
        registry = {
            "work_unit.json": work_unit_schema,
            work_unit_schema["$id"]: work_unit_schema,
        }
        with self.assertRaises(JsonSchemaValidationError):
            validate_json_schema_instance(
                [snapshot], manifest_array_schema, registry=registry
            )

        provisional = deepcopy(snapshot)
        provisional["qualification_status"] = "provisionally_compatible"
        validate_json_schema_instance(
            [provisional], manifest_array_schema, registry=registry
        )


if __name__ == "__main__":
    unittest.main()
