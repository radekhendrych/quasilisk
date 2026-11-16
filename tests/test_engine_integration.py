from __future__ import annotations

from foundation_support import judge_schema

from foundation_support import judge_procedure_payload

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from integration_support import dryrun_config, parsed_json, read_validated_events
from quasilisk.domain import (
    EncodedBytes,
    FinishStatus,
    ProviderErrorKind,
    ProviderFailure,
    ProviderSuccess,
    PortfolioIntent,
    RoleCallVector,
    RunTerminalReason,
    canonical_json,
    entity_id,
    scientific_key,
)
from quasilisk.engine import (
    EngineError,
    RunContext,
    _build_candidates,
    _initialize_routes,
    _judge_result,
    _messages_for_action,
    _provider_circuit_outcomes,
    _target_context_limit_trip_sequence,
    build_initial_observation,
    build_map_exploration,
    build_portfolio_round,
    build_screening,
    build_verification,
    load_resources,
    run_new,
)
from quasilisk.events import validate_payload
from quasilisk.judging import validate_judge_output
from quasilisk.provider import ScriptedFixtureMissing
from quasilisk.run_state import reduce_events
from quasilisk.scheduler import validate_work_unit_maximum


class WorkUnitBuilderTests(unittest.TestCase):
    def test_every_engine_builder_freezes_a_conformant_exact_maximum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resources = load_resources(dryrun_config("fatalism", root / "runs"))
            run_id = entity_id("run", ["builder-conformance"])
            context = RunContext(
                resources=resources,
                run_id=run_id,
                run_nonce="0" * 32,
                invocation_id=entity_id("iv", ["builder-conformance"]),
                run_directory=root / run_id,
                candidates={},
                routes={},
            )
            _initialize_routes(context)
            context.candidates.update(_build_candidates(resources, run_id))
            candidate = next(iter(context.candidates.values()))
            initial = build_initial_observation(context, candidate)
            units = (
                initial,
                build_screening(
                    context,
                    candidate,
                    initial.design["execution_id"],
                    initial.design["execution_scientific_key"],
                    parent_unit_id=initial.unit_id,
                ),
                build_verification(
                    context, candidate, parent_unit_id=initial.unit_id
                ),
                build_map_exploration(
                    context, candidate, parent_unit_id=initial.unit_id
                ),
                build_portfolio_round(
                    context,
                    candidate,
                    parent_unit_id=initial.unit_id,
                    conversation_root_id=entity_id(
                        "cr", [initial.design["execution_id"], "conversation_root"]
                    ),
                    conversation_root_scientific_key=scientific_key(
                        [
                            initial.design["execution_scientific_key"],
                            "conversation_root",
                        ]
                    ),
                    parent_node_id=entity_id("cn", ["builder-parent"]),
                    parent_node_scientific_key=scientific_key(["builder-parent"]),
                    portfolio_intent=PortfolioIntent.ADVANCE,
                ),
            )
            expected = {
                "initial_observation": RoleCallVector(2, 8, 0, 0),
                "screening": RoleCallVector(4, 16, 0, 0),
                "verification": RoleCallVector(10, 40, 0, 0),
                "map_exploration": RoleCallVector(6, 24, 4, 4),
                "portfolio_round": RoleCallVector(6, 24, 10, 4),
            }
            for unit in units:
                with self.subTest(kind=unit.kind):
                    self.assertEqual(unit.design["tag"], unit.kind)
                    self.assertEqual(unit.worst_case_calls, expected[unit.kind])
                    self.assertEqual(validate_work_unit_maximum(unit), expected[unit.kind])
                    validate_payload("work_unit_started", {"unit": unit.to_dict()})
                    action_ids = [slot["action_id"] for slot in unit.logical_action_slots]
                    self.assertEqual(len(action_ids), len(set(action_ids)))
                    for ordinal, slot in enumerate(unit.logical_action_slots):
                        self.assertEqual(slot["ordinal"], ordinal)
                        self.assertLessEqual(
                            set(slot["dependency_action_ids"]), set(action_ids[:ordinal])
                        )

            portfolio = units[-1]
            purposes = [item["purpose"] for item in portfolio.logical_action_slots]
            self.assertEqual(purposes.count("selector_portfolio"), 1)
            self.assertEqual(purposes.count("perturbator_proposal"), 1)
            self.assertEqual(purposes.count("probe_review"), 1)
            self.assertEqual(purposes.count("outcome_review"), 1)
            admission = next(
                item
                for item in portfolio.logical_action_slots
                if item["purpose"] == "probe_review"
            )
            self.assertEqual(
                [item["configured_slot"] for item in admission["routes"]],
                ["selector_primary"],
            )
            self.assertEqual(admission["maximum_physical_attempts"], 2)
            self.assertFalse(
                {
                    "if_repairable_proposal_exists",
                    "if_any_corrective_proposal",
                }
                & {item["guard"]["kind"] for item in portfolio.logical_action_slots}
            )
            target_guards = [
                item["guard"]
                for item in portfolio.logical_action_slots
                if item["guard"]["kind"] == "if_dispatchable_probe"
            ]
            self.assertEqual(len(target_guards), 3)
            self.assertTrue(all(item["allowed"] == ["aligned"] for item in target_guards))


class StructuredRoleBoundaryTests(unittest.TestCase):
    def _judge_case(self, root: Path) -> tuple[object, object, object, object, str]:
        resources = load_resources(dryrun_config("fatalism", root / "runs"))
        run_id = entity_id("run", ["structured-boundary"])
        context = RunContext(
            resources=resources,
            run_id=run_id,
            run_nonce="0" * 32,
            invocation_id=entity_id("iv", ["structured-boundary"]),
            run_directory=root / run_id,
            candidates={},
            routes={},
        )
        _initialize_routes(context)
        context.candidates.update(_build_candidates(resources, run_id))
        candidate = next(iter(context.candidates.values()))
        frozen = build_initial_observation(context, candidate).to_dict()
        panel = frozen["panel_plans"][0]
        target_id = panel["target_action_id"]
        judge_id = panel["normal_members"][0]["judge_action_id"]
        slots = {item["action_id"]: item for item in frozen["logical_action_slots"]}
        target_reply = "The claimed conclusion does not follow."
        excerpt = {"evidence_id": "e0", "polarity": "resists"}
        invalid_output = canonical_json(
            {
                "score": 20,
                "verdict": "not_demonstrated",
                "application": "none",

                "justification": "The response resists the failed inference.",
             "procedure": judge_procedure_payload("not_demonstrated", "none", [{**excerpt, "extra": True}])}
        )
        attempt = SimpleNamespace(
            succeeded=True,
            target_identity_collision=False,
            physical_attempt_id=entity_id("pa", ["structured-boundary"]),
            intended={"role_route_key": scientific_key(["judge-route"])},
            terminal_event={
                "payload": {
                    "finish_status": "complete",
                    "text": invalid_output,
                    "effective_model_id": "scripted/judge-primary",
                }
            },
        )
        target_action = SimpleNamespace(
            action_id=target_id,
            slot=slots[target_id],
            attempts=(),
        )
        judge_action = SimpleNamespace(
            action_id=judge_id,
            slot=slots[judge_id],
            attempts=(attempt,),
        )
        actions = tuple(
            judge_action
            if item["action_id"] == judge_id
            else target_action
            if item["action_id"] == target_id
            else SimpleNamespace(
                action_id=item["action_id"], slot=item, attempts=()
            )
            for item in frozen["logical_action_slots"]
        )
        unit = SimpleNamespace(unit=frozen, actions=actions)
        state = SimpleNamespace(
            target_results={target_id: SimpleNamespace(text=target_reply)},
            logical_actions={target_id: target_action, judge_id: judge_action},
        )
        return context, state, unit, judge_action, invalid_output

    def test_schema_gate_runs_before_judge_semantic_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context, state, unit, action, invalid_output = self._judge_case(
                Path(temporary)
            )
            target_reply = next(iter(state.target_results.values())).text
            self.assertFalse(validate_judge_output(invalid_output, target_reply, schema=judge_schema(target_reply)).usable)

            result = _judge_result(context, state, unit, action)
            self.assertIsNone(result.value)
            self.assertIn(
                "$.procedure.inference: must match at least one anyOf branch",
                result.errors,
            )

    def test_schema_repair_context_is_rendered_by_the_canonical_role_template(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context, state, unit, action, invalid_output = self._judge_case(
                Path(temporary)
            )
            initial = _messages_for_action(context, state, unit, action)
            errors = (
                "$.procedure.inference: must match at least one anyOf branch",
            )
            repair = _messages_for_action(
                context, state, unit, action, repair_errors=errors
            )

            self.assertEqual(len(initial), 2)
            self.assertEqual(len(repair), 2)
            self.assertEqual(initial[0], repair[0])
            self.assertIn("<SCHEMA_REPAIR_CONTEXT_JSON>\nnull", initial[1].content)
            self.assertIn(
                canonical_json(
                    {
                        "invalid_output": invalid_output,
                        "structural_errors": list(errors),
                    }
                ),
                repair[1].content,
            )
            self.assertNotIn("Return a corrected object only", repair[1].content)


class ScriptedFailureTests(unittest.TestCase):
    def test_operational_retry_delay_is_bounded_and_immediately_precedes_dispatch(self) -> None:
        for retry_after, expected_delay in (("25", 10.0), ("0", 0.0), (None, 1.0)):
            with self.subTest(retry_after=retry_after), tempfile.TemporaryDirectory() as temporary:
                trace: list[tuple[str, float | int]] = []

                class RetryThenMissingProvider:
                    content_digest = None

                    def complete(self, request):
                        del request
                        trace.append(("dispatch", len(trace)))
                        if sum(item[0] == "dispatch" for item in trace) == 1:
                            return ProviderFailure(
                                error_kind=ProviderErrorKind.RATE_LIMIT,
                                delivery_state="known_no_output",
                                detail="rate_limited",
                                provider_request_id=None,
                                raw_body=None,
                                http_status=429,
                                retry_after_seconds=retry_after,
                            )
                        raise ScriptedFixtureMissing("scripted_fixture_missing: test")

                config = dryrun_config("fatalism", Path(temporary) / "runs")
                with patch(
                    "quasilisk.engine.create_provider",
                    return_value=RetryThenMissingProvider(),
                ):
                    _, reason = run_new(
                        config,
                        sleeper=lambda seconds: trace.append(("sleep", seconds)),
                    )
                self.assertEqual(reason, RunTerminalReason.IMPOSSIBLE_INTERNAL_STATE)
                self.assertEqual(
                    [item[0] for item in trace],
                    ["dispatch", "sleep", "dispatch"],
                )
                self.assertEqual(trace[1], ("sleep", expected_delay))

    def test_missing_semantic_fixture_stops_fail_closed_without_fabricated_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            empty_script = root / "missing-all-keys.jsonl"
            empty_script.write_bytes(b"")
            config = dryrun_config("fatalism", root / "runs")
            config["provider"]["script_path"] = empty_script.as_posix()

            run_directory, reason = run_new(config)
            self.assertEqual(reason, RunTerminalReason.IMPOSSIBLE_INTERNAL_STATE)
            events = read_validated_events(run_directory)
            self.assertEqual(
                [event["type"] for event in events],
                [
                    "run_created",
                    "work_unit_started",
                    "model_call_intended",
                    "model_dispatch_started",
                    "model_call_failed",
                    "run_terminated",
                ],
            )
            self.assertEqual(events[-1]["payload"]["detail"], "scripted_fixture_missing")
            self.assertFalse(events[-1]["payload"]["resumable"])
            self.assertEqual(events[-3]["payload"]["charged_role"], "target")
            self.assertEqual(events[-2]["payload"]["detail"], "scripted_fixture_missing")
            state = reduce_events(events)
            self.assertEqual(state.consumed, RoleCallVector(target=1))
            self.assertEqual(len(state.dangling_attempt_ids), 0)
            self.assertEqual(state.completed_target_responses, 0)
            summary = parsed_json(run_directory / "summary.json")
            self.assertEqual(summary["headline"]["physical_target_attempts_consumed"], 1)
            self.assertEqual(summary["headline"]["completed_target_responses"], 0)
            self.assertEqual(summary["role_accounting"]["target"]["dangling_dispatches"], 0)

    def test_empty_target_success_uses_exact_retry_not_schema_repair(self) -> None:
        class EmptyThenMissingProvider:
            content_digest = None

            def __init__(self) -> None:
                self.calls = 0

            def complete(self, request):
                self.calls += 1
                if self.calls == 1:
                    return ProviderSuccess(
                        raw_body=EncodedBytes("utf8", "{}"),
                        text="",
                        provider_request_id="empty-target",
                        effective_model_id="target/revision-a",
                        finish_status=FinishStatus.COMPLETE,
                        usage={
                            "input_tokens": None,
                            "output_tokens": None,
                            "total_tokens": None,
                        },
                        latency_ms=0,
                        cost=None,
                        metadata={
                            "http_status": 200,
                            "content_type": "application/json",
                        },
                    )
                raise ScriptedFixtureMissing("scripted_fixture_missing: test")

        with tempfile.TemporaryDirectory() as temporary:
            provider = EmptyThenMissingProvider()
            config = dryrun_config("fatalism", Path(temporary) / "runs")
            with patch("quasilisk.engine.create_provider", return_value=provider):
                run_directory, reason = run_new(config)
            self.assertEqual(reason, RunTerminalReason.IMPOSSIBLE_INTERNAL_STATE)
            intents = [
                event["payload"]
                for event in read_validated_events(run_directory)
                if event["type"] == "model_call_intended"
            ]
            self.assertEqual(
                [item["route_kind"] for item in intents],
                ["primary", "exact_retry"],
            )

    def test_target_response_format_failure_is_impossible_without_downgrade(self) -> None:
        class UnsupportedProvider:
            content_digest = None

            def complete(self, request):
                return ProviderFailure(
                    error_kind=ProviderErrorKind.UNSUPPORTED_RESPONSE_FORMAT,
                    delivery_state="known_no_output",
                    detail="response_format_not_supported",
                    provider_request_id=None,
                    raw_body=None,
                    http_status=400,
                    retry_after_seconds=None,
                )

        with tempfile.TemporaryDirectory() as temporary:
            config = dryrun_config("fatalism", Path(temporary) / "runs")
            with patch(
                "quasilisk.engine.create_provider", return_value=UnsupportedProvider()
            ):
                run_directory, reason = run_new(config)
            events = read_validated_events(run_directory)
            self.assertEqual(reason, RunTerminalReason.IMPOSSIBLE_INTERNAL_STATE)
            self.assertEqual(
                [
                    event["payload"]["route_kind"]
                    for event in events
                    if event["type"] == "model_call_intended"
                ],
                ["primary"],
            )
            self.assertEqual(
                events[-1]["payload"]["detail"],
                "target_unsupported_response_format",
            )

    def test_adapter_exceptions_retry_then_trip_after_three_exhausted_actions(self) -> None:
        class RaisingProvider:
            content_digest = None

            def complete(self, request):
                raise RuntimeError("transport implementation failed")

        with tempfile.TemporaryDirectory() as temporary:
            config = dryrun_config("fatalism", Path(temporary) / "runs")
            config["seed"]["bootstrap_limit"] = 3
            delays: list[float] = []
            with patch("quasilisk.engine.create_provider", return_value=RaisingProvider()):
                run_directory, reason = run_new(config, sleeper=delays.append)
            events = read_validated_events(run_directory)
            self.assertEqual(reason, RunTerminalReason.SYSTEMIC_PROVIDER_FAILURE)
            target_intents = [
                event["payload"]
                for event in events
                if event["type"] == "model_call_intended"
                and event["payload"]["role"] == "target"
            ]
            self.assertEqual(len(target_intents), 6)
            self.assertEqual(
                [item["route_kind"] for item in target_intents],
                ["primary", "exact_retry"] * 3,
            )
            unknowns = [
                event
                for event in events
                if event["type"] == "model_call_outcome_unknown"
            ]
            self.assertEqual(len(unknowns), 6)
            self.assertEqual(
                events[-1]["payload"]["detail"],
                "target_provider_logical_failure_circuit",
            )
            self.assertTrue(events[-1]["payload"]["resumable"])
            self.assertEqual(delays, [1.0, 1.0, 1.0])

    def test_exhausted_empty_and_truncated_targets_increment_the_target_circuit(self) -> None:
        for text, finish_status in (
            ("", FinishStatus.COMPLETE),
            ("partial", FinishStatus.OUTPUT_LIMIT),
        ):
            with self.subTest(finish_status=finish_status.value), tempfile.TemporaryDirectory() as temporary:
                class UnusableTargetProvider:
                    content_digest = None

                    def complete(self, request):
                        del request
                        return ProviderSuccess(
                            raw_body=EncodedBytes("utf8", "{}"),
                            text=text,
                            provider_request_id="unusable-target",
                            effective_model_id="target/revision-a",
                            finish_status=finish_status,
                            usage={
                                "input_tokens": None,
                                "output_tokens": None,
                                "total_tokens": None,
                            },
                            latency_ms=0,
                            cost=None,
                            metadata={
                                "http_status": 200,
                                "content_type": "application/json",
                            },
                        )

                config = dryrun_config("fatalism", Path(temporary) / "runs")
                config["seed"]["bootstrap_limit"] = 3
                with patch(
                    "quasilisk.engine.create_provider",
                    return_value=UnusableTargetProvider(),
                ):
                    run_directory, reason = run_new(config, sleeper=lambda _: None)
                events = read_validated_events(run_directory)
                self.assertEqual(reason, RunTerminalReason.SYSTEMIC_PROVIDER_FAILURE)
                self.assertEqual(
                    len(
                        [
                            event
                            for event in events
                            if event["type"] == "model_call_intended"
                            and event["payload"]["role"] == "target"
                        ]
                    ),
                    6,
                )
                self.assertEqual(
                    events[-1]["payload"]["detail"],
                    "target_provider_logical_failure_circuit",
                )

    def test_context_limit_trips_only_after_three_fresh_execution_roots(self) -> None:
        class ContextLimitedProvider:
            content_digest = None

            def complete(self, request):
                del request
                return ProviderFailure(
                    error_kind=ProviderErrorKind.CONTEXT_LIMIT,
                    delivery_state="known_no_output",
                    detail="context_window_exceeded",
                    provider_request_id=None,
                    raw_body=None,
                    http_status=400,
                    retry_after_seconds=None,
                )

        with tempfile.TemporaryDirectory() as temporary:
            config = dryrun_config("fatalism", Path(temporary) / "runs")
            config["seed"]["bootstrap_limit"] = 3
            with patch(
                "quasilisk.engine.create_provider",
                return_value=ContextLimitedProvider(),
            ):
                run_directory, reason = run_new(config, sleeper=lambda _: None)
            events = read_validated_events(run_directory)
            target_intents = [
                event["payload"]
                for event in events
                if event["type"] == "model_call_intended"
                and event["payload"]["role"] == "target"
            ]
            self.assertEqual(reason, RunTerminalReason.SYSTEMIC_PROVIDER_FAILURE)
            self.assertEqual(len(target_intents), 3)
            self.assertEqual(
                len(
                    {
                        item["lineage"]["execution_id"]
                        for item in target_intents
                    }
                ),
                3,
            )
            self.assertEqual(
                [item["route_kind"] for item in target_intents],
                ["primary", "primary", "primary"],
            )
            self.assertEqual(
                events[-1]["payload"]["detail"],
                "target_provider_logical_failure_circuit",
            )


class ProviderCircuitClassificationTests(unittest.TestCase):
    @staticmethod
    def _context_action(label: str, sequence: int, root_id: str):
        return SimpleNamespace(
            action_id=label,
            slot={"role": "target"},
            attempts=(
                SimpleNamespace(
                    intended={
                        "lineage": {
                            "conversation_root_id": root_id,
                            "execution_id": f"execution-{label}",
                        }
                    },
                    terminal_event={
                        "sequence": sequence,
                        "type": "model_call_failed",
                        "payload": {"error_kind": "context_limit"},
                    },
                ),
            ),
        )

    def test_context_limit_requires_three_distinct_fresh_roots(self) -> None:
        actions = (
            self._context_action("first", 1, "root-a"),
            self._context_action("same-root", 2, "root-a"),
            self._context_action("second", 3, "root-b"),
        )
        unit = SimpleNamespace(
            unit={"kind": "portfolio_round"}, actions=actions
        )
        state = SimpleNamespace(work_units=(unit,), target_results={})
        self.assertIsNone(_target_context_limit_trip_sequence(state))

        third = self._context_action("third", 4, "root-c")
        unit.actions = (*actions, third)
        self.assertEqual(_target_context_limit_trip_sequence(state), 4)

        repeated_middle = SimpleNamespace(
            unit={"kind": "portfolio_round"},
            actions=(
                self._context_action("a", 1, "root-a"),
                self._context_action("b-first", 2, "root-b"),
                self._context_action("b-second", 3, "root-b"),
                self._context_action("c", 4, "root-c"),
            ),
        )
        repeated_state = SimpleNamespace(
            work_units=(repeated_middle,), target_results={}
        )
        self.assertIsNone(_target_context_limit_trip_sequence(repeated_state))

    def test_usable_target_clears_context_limit_sequence(self) -> None:
        first = self._context_action("first", 1, "root-a")
        success = SimpleNamespace(action_id="success", slot={"role": "target"}, attempts=())
        second = self._context_action("second", 3, "root-b")
        third = self._context_action("third", 4, "root-c")
        unit = SimpleNamespace(
            unit={"kind": "portfolio_round"},
            actions=(first, success, second, third),
        )
        state = SimpleNamespace(
            work_units=(unit,),
            target_results={
                "success": SimpleNamespace(usable=True, terminal_sequence=2)
            },
        )
        self.assertIsNone(_target_context_limit_trip_sequence(state))

if __name__ == "__main__":
    unittest.main()
