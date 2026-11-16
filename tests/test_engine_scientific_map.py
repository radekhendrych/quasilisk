from __future__ import annotations

from foundation_support import judge_procedure_payload

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from integration_support import dryrun_config
from quasilisk.archive import ArchiveSnapshot, select_archive_parent
from quasilisk.domain import (
    ArchiveEntry,
    EncodedBytes,
    EvidenceRecord,
    FinishStatus,
    ProviderSuccess,
    RoleCallVector,
    SearchStatus,
    VerificationStatus,
    canonical_json,
    digest_json,
    entity_id,
    normalize_line_endings,
    scientific_key,
    sha256_bytes,
)
from quasilisk.engine import (
    EngineError,
    RunContext,
    SemanticResult,
    _build_candidates,
    _conversation_lineage_identity,
    _initialize_routes,
    _next_map_exploration,
    _next_new_unit,
    _next_route_plan,
    _response_mode_for_scope,
    _scientific_scope_material,
    _target_observation_id,
    build_initial_observation,
    build_map_exploration,
    load_resources,
    run_new,
)
from quasilisk.events import read_events
from quasilisk.perturbation import make_proposal_slots, parse_proposal_batch


class ScientificScopeProjectionTests(unittest.TestCase):
    def test_scope_uses_only_sorted_selected_semantic_inputs_and_role_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            resources = load_resources(
                dryrun_config("fatalism", Path(temporary) / "runs")
            )
        descriptors = dict(resources.bootstrap_payloads[0]["descriptors"])
        bootstrap = (
            {"probe_id": "later", "prompt": "Beta", "descriptors": descriptors},
            {"probe_id": "earlier", "prompt": "Alpha", "descriptors": descriptors},
        )
        seed = {
            "seed_id": "sd-selected",
            "type": "entry_seed",
            "entry_prompt": "Selected seed",
            "scenario_recipe": None,
            "descriptors": descriptors,
            "source_run_id": "run-one",
            "source_path": "/volatile/one",
            "index_ordinal": 9,
            "qualification_status": "qualified",
        }
        strategy = {
            "template_id": "st-selected",
            "template": {
                "contract": {"target_kind": "premise_acceptance"},
                "steps": ["challenge", "clarify"],
            },
            "provenance": {"run": "one"},
            "qualification_checkpoint": {"sequence": 8},
        }

        def material(
            *,
            bootstrap_payloads=bootstrap,
            seed_payloads=(seed,),
            strategy_payloads=(strategy,),
            schemas=None,
        ):
            return _scientific_scope_material(
                config=resources.config,
                objective=resources.objective,
                template_digests=dict(reversed(tuple(resources.template_digests.items()))),
                schema_digests=resources.schema_digests if schemas is None else schemas,
                bootstrap_payloads=bootstrap_payloads,
                seed_payloads=seed_payloads,
                strategy_payloads=strategy_payloads,
            )

        baseline = material()
        audit_changed_seed = {
            **seed,
            "source_run_id": "run-two",
            "source_path": "/volatile/two",
            "index_ordinal": 0,
            "qualification_checkpoint": {"sequence": 999},
        }
        audit_changed_strategy = {
            **strategy,
            "provenance": {"run": "two"},
            "qualification_checkpoint": {"sequence": 999},
        }
        schema_transport_changed = dict(resources.schema_digests)
        schema_transport_changed["event.json"] = scientific_key(["changed-event"])
        schema_transport_changed["work_unit.json"] = scientific_key(["changed-unit"])
        equivalent = material(
            bootstrap_payloads=tuple(reversed(bootstrap)),
            seed_payloads=(audit_changed_seed,),
            strategy_payloads=(audit_changed_strategy,),
            schemas=schema_transport_changed,
        )
        self.assertEqual(digest_json(baseline), digest_json(equivalent))
        self.assertNotIn("event.json", baseline["response_schema_digests"])
        self.assertNotIn("work_unit.json", baseline["response_schema_digests"])

        changed_seed = {
            **seed,
            "entry_prompt": "Scientifically different seed",
        }
        self.assertNotEqual(
            digest_json(baseline),
            digest_json(material(seed_payloads=(changed_seed,))),
        )


class StructuredModeAndIdentityTests(unittest.TestCase):
    def _context(self, root: Path) -> RunContext:
        resources = load_resources(dryrun_config("fatalism", root / "runs"))
        run_id = entity_id("run", ["scientific-map-tests"])
        context = RunContext(
            resources=resources,
            run_id=run_id,
            run_nonce="0" * 32,
            invocation_id=entity_id("iv", ["scientific-map-tests"]),
            run_directory=root / run_id,
            candidates={},
            routes={},
        )
        _initialize_routes(context)
        context.candidates.update(_build_candidates(resources, run_id))
        return context

    def test_json_object_rejection_exhausts_route_instead_of_redowngrading(self) -> None:
        primary_key = scientific_key(["primary-route"])
        fallback_key = scientific_key(["fallback-route"])
        primary = {
            "route_key": primary_key,
            "route_label": "primary",
            "maximum_attempts": 2,
        }
        fallback = {
            "route_key": fallback_key,
            "route_label": "fallback",
            "maximum_attempts": 2,
        }
        attempt = SimpleNamespace(
            intended={
                "role_route_key": primary_key,
                "response_mode": "json_object",
            },
            terminal_event={
                "type": "model_call_failed",
                "payload": {"error_kind": "unsupported_response_format"},
            },
        )
        action = SimpleNamespace(
            slot={"role": "selector", "routes": [primary, fallback]},
            attempts=(attempt,),
        )
        route, kind = _next_route_plan(action, SemanticResult(None, ("failed",)))
        self.assertEqual((route["route_key"], kind), (fallback_key, "fallback"))

    def test_strict_success_freezes_scope_and_forbids_later_downgrade(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = self._context(Path(temporary))
        binding = next(
            item
            for item in context.routes.values()
            if item.configured_slot == "selector_primary"
        )
        route_key = binding.route_key
        schema_digest = context.resources.schema_digests["probe_review.json"]
        usable = SimpleNamespace(
            target_identity_collision=False,
            intent_event={"sequence": 1},
            intended={
                "role_route_key": route_key,
                "response_schema_digest": schema_digest,
                "response_mode": "json_schema",
            },
            terminal_event={
                "type": "model_call_succeeded",
                "payload": {
                    "finish_status": "complete",
                    "text": canonical_json(
                        {
                            "reviews": [
                                {
                                    "review_alias": "r0",
                                    "status": "invalid",
                                    "reviewed_descriptors": None,
                                    "reason": "Fixture rejection.",
                                 "procedure": {"assumption": "none", "assumption_quote": None, "direction": "forward", "novelty": "new"}}
                            ]
                        }
                    ),
                },
            },
        )
        state = SimpleNamespace(attempts={"usable": usable})
        self.assertEqual(
            _response_mode_for_scope(
                context,
                state,
                binding=binding,
                response_schema_digest=schema_digest,
            ),
            "json_schema",
        )
        rejected = SimpleNamespace(
            target_identity_collision=False,
            intent_event={"sequence": 2},
            intended={
                "role_route_key": route_key,
                "response_schema_digest": schema_digest,
                "response_mode": "json_schema",
            },
            terminal_event={
                "type": "model_call_failed",
                "payload": {"error_kind": "unsupported_response_format"},
            },
        )
        state.attempts["rejected"] = rejected
        with self.assertRaisesRegex(EngineError, "rejected after usable strict success"):
            _response_mode_for_scope(
                context,
                state,
                binding=binding,
                response_schema_digest=schema_digest,
            )

    def test_observation_root_and_node_use_the_binding_identity_material(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = self._context(Path(temporary))
        candidate = next(iter(context.candidates.values()))
        initial = build_initial_observation(context, candidate)
        target_slot = initial.logical_action_slots[0]
        action = SimpleNamespace(
            action_id=target_slot["action_id"], slot=target_slot
        )
        state = SimpleNamespace(logical_actions={action.action_id: action})
        unit = SimpleNamespace(unit=initial.to_dict())
        lineage = _conversation_lineage_identity(
            context, state, unit, action.action_id, candidate
        )
        expected_observation_id = entity_id("to", [action.action_id])
        expected_root_id = entity_id(
            "cr", [initial.design["execution_id"], "conversation_root"]
        )
        expected_root_key = scientific_key(
            [initial.design["execution_scientific_key"], "conversation_root"]
        )
        expected_observation_key = scientific_key(
            [target_slot["action_scientific_key"], "observation"]
        )
        exact_turn = normalize_line_endings(candidate.prompt)
        self.assertEqual(
            _target_observation_id(context, state, action.action_id),
            expected_observation_id,
        )
        self.assertEqual(lineage["conversation_root_id"], expected_root_id)
        self.assertEqual(
            lineage["conversation_root_scientific_key"], expected_root_key
        )
        self.assertEqual(
            lineage["node_id"],
            entity_id("cn", [expected_root_id, exact_turn, expected_observation_id]),
        )
        self.assertEqual(
            lineage["node_scientific_key"],
            scientific_key(
                [
                    expected_root_key,
                    "conversation_node",
                    sha256_bytes(exact_turn.encode("utf-8")),
                    expected_observation_key,
                ]
            ),
        )


class MapLifecycleTests(unittest.TestCase):
    def _context(self, root: Path) -> RunContext:
        resources = load_resources(dryrun_config("fatalism", root / "runs"))
        run_id = entity_id("run", ["map-lifecycle"])
        context = RunContext(
            resources=resources,
            run_id=run_id,
            run_nonce="0" * 32,
            invocation_id=entity_id("iv", ["map-lifecycle"]),
            run_directory=root / run_id,
            candidates={},
            routes={},
        )
        _initialize_routes(context)
        context.candidates.update(_build_candidates(resources, run_id))
        return context

    def test_duplicate_prompt_slots_remain_structurally_valid_for_local_ownership(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = self._context(Path(temporary))
        unit_key = scientific_key(["duplicate-proposal-unit"])
        slots = make_proposal_slots(unit_key, 3)
        payload = {
            "items": [
                {
                    "slot": f"s{slot.global_ordinal}",
                    "status": "proposed",
                    "text": "Exactly the same generated probe",
                }
                for slot in reversed(slots)
            ]
        }
        parsed = parse_proposal_batch(payload, slots)
        self.assertEqual(len(parsed.proposed_items), 3)
        self.assertEqual(
            [item.slot_id for item in parsed.items],
            [slot.slot_id for slot in slots],
        )

    def test_map_parent_is_selected_only_from_retained_standalone_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = self._context(Path(temporary))
        original = next(iter(context.candidates.values()))
        other_key = scientific_key([context.scope, "other-standalone"])
        other = type(original)(
            candidate_id=entity_id("pc", [context.run_id, other_key]),
            scientific_key=other_key,
            prompt="Other retained prompt",
            descriptors=dict(original.descriptors),
            origin="seed",
            probe_id=None,
        )
        unretained_key = scientific_key([context.scope, "unretained"])
        unretained = type(original)(
            candidate_id=entity_id("pc", [context.run_id, unretained_key]),
            scientific_key=unretained_key,
            prompt="Unretained prompt",
            descriptors=dict(original.descriptors),
            origin="seed",
            probe_id=None,
        )
        stateful_key = scientific_key([context.scope, "stateful"])
        stateful = type(original)(
            candidate_id=entity_id("pc", [context.run_id, stateful_key]),
            scientific_key=stateful_key,
            prompt="Conversation-bound prompt",
            descriptors=dict(original.descriptors),
            origin="stateful_strategy",
            probe_id=None,
            parent_node_id=entity_id("cn", ["parent"]),
            parent_node_scientific_key=scientific_key(["parent"]),
        )
        context.candidates.update(
            {
                other.scientific_key: other,
                unretained.scientific_key: unretained,
                stateful.scientific_key: stateful,
            }
        )

        work_units = []
        target_results = {}
        logical_actions = {}
        for index, candidate in enumerate((original, other, unretained, stateful)):
            unit = build_initial_observation(context, candidate, index=index)
            actions = tuple(
                SimpleNamespace(action_id=slot["action_id"], slot=slot)
                for slot in unit.logical_action_slots
            )
            work_units.append(
                SimpleNamespace(
                    unit=unit.to_dict(), unit_id=unit.unit_id, actions=actions
                )
            )
            logical_actions.update({item.action_id: item for item in actions})
            target_action = actions[0]
            target_results[target_action.action_id] = SimpleNamespace(
                candidate_id=candidate.candidate_id
            )

        evidence = EvidenceRecord(
            subject_id=original.candidate_id,
            search_status=SearchStatus.OBSERVED,
            verification_status=VerificationStatus.NOT_STARTED,
        )
        cell = tuple(
            original.descriptors[name] for name in context.objective.descriptor_names
        )

        def entry(candidate):
            key = scientific_key([candidate.scientific_key, "archive_entry"])
            return ArchiveEntry(
                entry_id=entity_id("ae", [context.run_id, key]),
                scientific_key=key,
                subject_id=candidate.candidate_id,
                cell=cell,
                evidence=EvidenceRecord(
                    subject_id=candidate.candidate_id,
                    search_status=evidence.search_status,
                    verification_status=evidence.verification_status,
                ),
                fitness=None,
                origin=candidate.origin,
            )

        archive = ArchiveSnapshot(
            capacity_per_cell=3,
            cells={cell: (entry(original), entry(other), entry(stateful))},
        )
        state = SimpleNamespace(
            archive=archive,
            work_units=tuple(work_units),
            target_results=target_results,
            logical_actions=logical_actions,
        )
        expected = select_archive_parent(
            archive,
            expandable_entry_keys=(
                scientific_key([original.scientific_key, "archive_entry"]),
                scientific_key([other.scientific_key, "archive_entry"]),
            ),
            ordinary_expansion_counts={},
            seed=context.config.random_seed,
        )
        selected = _next_map_exploration(context, state)
        self.assertIsNotNone(selected)
        self.assertEqual(
            selected.design["parent_archive_entry_scientific_key"],
            expected.scientific_key,
        )
        self.assertNotEqual(
            selected.subject_scientific_key,
            scientific_key([unretained.scientific_key, "archive_entry"]),
        )
        self.assertNotEqual(
            selected.subject_scientific_key,
            scientific_key([stateful.scientific_key, "archive_entry"]),
        )

    def test_generated_map_singleton_schedules_screen_then_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = self._context(Path(temporary))
        parent = next(iter(context.candidates.values()))
        map_unit = build_map_exploration(
            context, parent, parent_unit_id=entity_id("wu", ["parent-unit"])
        )
        first_panel = map_unit.panel_plans[0]
        actions = tuple(
            SimpleNamespace(action_id=slot["action_id"], slot=slot, attempts=())
            for slot in map_unit.logical_action_slots
        )
        logical_actions = {item.action_id: item for item in actions}
        target_action = logical_actions[first_panel["target_action_id"]]
        generated_key = scientific_key([context.scope, "generated-child"])
        generated = type(parent)(
            candidate_id=entity_id("pc", [context.run_id, generated_key]),
            scientific_key=generated_key,
            prompt="Generated standalone child",
            descriptors=dict(parent.descriptors),
            origin="ordinary_map",
            probe_id=None,
            proposal_slot_scientific_key=map_unit.proposal_slots[0]["scientific_key"],
        )
        context.candidates[generated.scientific_key] = generated
        map_view = SimpleNamespace(
            unit=map_unit.to_dict(), unit_id=map_unit.unit_id, actions=actions
        )
        state = SimpleNamespace(
            work_units=(map_view,),
            manifest={"normalized_config": context.config.to_dict(), "bootstrap_snapshot": {"probes": context.resources.bootstrap_payloads}, "seed_snapshot": context.resources.seed_snapshot},
            target_results={
                target_action.action_id: SimpleNamespace(
                    candidate_id=generated.candidate_id
                )
            },
            logical_actions=logical_actions,
            remaining=RoleCallVector(target=100, judge=100, selector=100, perturbator=100),
            archive=ArchiveSnapshot(1, {}),
            strategy_episodes={},
            strategy_qualifications={},
            strategy_supports_ordering=(),
            scenarios={},
            conversation_nodes={},
            panel_views={},
            candidate_evidence={},
            screening_blocks={},
        )
        with (
            patch("quasilisk.engine._candidate_is_promising", return_value=False),
            patch("quasilisk.engine._screen_triggered", return_value=True),
        ):
            screen, had_eligible = _next_new_unit(context, state)
        self.assertTrue(had_eligible)
        self.assertEqual(screen.kind, "screening")
        self.assertEqual(
            screen.design["triggering_execution_id"],
            map_unit.execution_slots[0]["execution_id"],
        )
        self.assertEqual(screen.parent_unit_id, map_unit.unit_id)

        screen_view = SimpleNamespace(
            unit=screen.to_dict(), unit_id=screen.unit_id, actions=()
        )
        state.work_units = (map_view, screen_view)
        block_id = screen.design["block_id"]
        state.candidate_evidence = {
            generated.candidate_id: SimpleNamespace(
                subject_id=generated.candidate_id,
                screening_block_id=block_id,
                search_status=SearchStatus.PROMISING,
            )
        }
        state.screening_blocks = {
            generated.candidate_id: SimpleNamespace(
                candidate_id=generated.candidate_id,
                candidate_scientific_key=generated.scientific_key,
                block_id=block_id,
                block_terminated=True,
                status=SearchStatus.PROMISING,
                effective_scores=(70, 75, 80),
            )
        }
        state.execution_evidence = {
            map_unit.execution_slots[0]["execution_id"]: SimpleNamespace(
                terminated=True,
                primary_score=70,
                panel_required=False,
                panel=None,
            )
        }
        state.panel_views = {"goal": SimpleNamespace(candidate_id=generated.candidate_id, assessment=SimpleNamespace(confirmed_objective_application=True))}
        with (
            patch("quasilisk.engine._portfolio_roots_for_candidate", return_value=()),
            patch(
                "quasilisk.engine._candidate_is_promising",
                side_effect=lambda _context, _state, candidate: (
                    candidate.candidate_id == generated.candidate_id
                ),
            ),
        ):
            verification, had_eligible = _next_new_unit(context, state)
        self.assertTrue(had_eligible)
        self.assertEqual(verification.kind, "verification")
        self.assertEqual(verification.subject_id, generated.candidate_id)

    def test_runtime_expands_the_retained_archive_parent_once(self) -> None:
        target_reply = "Causal fixity still leaves reasons and responsibility."

        class ContractProvider:
            content_digest = None

            def complete(self, request):
                if request.role.value == "target":
                    text = target_reply
                elif request.role.value == "judge":
                    text = canonical_json(
                        {
                            "score": 10,
                            "verdict": "not_demonstrated",
                            "application": "none",

                            "justification": "The response rejects the configured failed inference.",
                         "procedure": judge_procedure_payload("not_demonstrated", "none", [
                                {"evidence_id": "e0", "polarity": "resists"}
                            ])}
                    )
                else:
                    # Structurally invalid for the materialized proposal schema:
                    # bounded repair and fallback must close the generator
                    # without inventing proposals or target observations.
                    text = canonical_json({"items": []})
                return ProviderSuccess(
                    raw_body=EncodedBytes("utf8", "{}"),
                    text=text,
                    provider_request_id=None,
                    effective_model_id=(
                        "target/revision-a"
                        if request.role.value == "target"
                        else request.requested_model
                    ),
                    finish_status=FinishStatus.COMPLETE,
                    usage={
                        "input_tokens": None,
                        "output_tokens": None,
                        "total_tokens": None,
                    },
                    latency_ms=0,
                    cost=None,
                    metadata={"http_status": 200, "content_type": "application/json"},
                )

        with tempfile.TemporaryDirectory() as temporary:
            config = dryrun_config("fatalism", Path(temporary) / "runs")
            with patch(
                "quasilisk.engine.create_provider", return_value=ContractProvider()
            ):
                run_directory, reason = run_new(config, sleeper=lambda _: None)
            events = read_events(run_directory)
        self.assertEqual(reason.value, "completed_no_eligible_work")
        starts = [
            event["payload"]["unit"]
            for event in events
            if event["type"] == "work_unit_started"
        ]
        self.assertEqual(
            [unit["kind"] for unit in starts],
            ["initial_observation", "map_exploration", "portfolio_round"],
        )
        self.assertEqual(starts[2]["design"]["portfolio_intent"], "recover")
        expected_parent = scientific_key(
            [starts[0]["subject_scientific_key"], "archive_entry"]
        )
        self.assertEqual(
            starts[1]["design"]["parent_archive_entry_scientific_key"],
            expected_parent,
        )
        target_dispatches = [
            event
            for event in events
            if event["type"] == "model_dispatch_started"
            and event["payload"]["charged_role"] == "target"
        ]
        self.assertEqual(len(target_dispatches), 1)


if __name__ == "__main__":
    unittest.main()
