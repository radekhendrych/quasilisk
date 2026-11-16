from __future__ import annotations

import unittest

from foundation_support import (
    FOUNDATION_RESPONSE_SCHEMA,
    FOUNDATION_RESPONSE_SCHEMA_DIGEST,
    FIXED_TIMESTAMP,
    attempt_id,
    call_fixture,
    chain_events,
    digest,
    dispatch_payload,
    failure_payload,
    intended_payload,
    invocation_id,
    run_created_payload,
    reduce_fixture_events as reduce_events,
    success_payload,
    unknown_payload,
)
from quasilisk.domain import (
    ExecutionTerminalStatus,
    RunTerminalReason,
    digest_json,
    entity_id,
)
from quasilisk.events import JournalCorrupt, build_event, validate_events
from quasilisk.reporting import archive_projection
from quasilisk.run_state import (
    ProviderCircuitOutcome,
    _validate_work_unit_boundaries,
    reduce_provider_circuit,
)


def with_unit(fixture: dict, records: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    return [("work_unit_started", {"unit": fixture["unit"]}), *records]


def append_record(
    events: list[dict],
    event_type: str,
    payload: dict,
    *,
    causes: tuple[str, ...] | None = None,
) -> list[dict]:
    if causes is None:
        if event_type in {"run_resumed", "run_terminated", "journal_tail_recovered"}:
            causes = (events[-1]["event_id"],)
        elif event_type == "model_dispatch_started":
            attempt_id_value = payload["physical_attempt_id"]
            intent = next(
                event
                for event in reversed(events)
                if event["type"] == "model_call_intended"
                and event["payload"]["physical_attempt_id"] == attempt_id_value
            )
            causes = (intent["event_id"],)
        elif event_type in {
            "model_call_succeeded",
            "model_call_failed",
            "model_call_outcome_unknown",
        }:
            attempt_id_value = payload["physical_attempt_id"]
            dispatch = next(
                event
                for event in reversed(events)
                if event["type"] == "model_dispatch_started"
                and event["payload"]["physical_attempt_id"] == attempt_id_value
            )
            causes = (dispatch["event_id"],)
        else:
            causes = ()
    event = build_event(
        run_id=events[0]["run_id"],
        sequence=len(events) + 1,
        event_type=event_type,
        payload=payload,
        previous_hash=events[-1]["event_hash"],
        causes=causes,
        timestamp_utc=FIXED_TIMESTAMP,
    )
    return [*events, event]


def resume_payload(
    events: list[dict],
    resume_id: str,
    *,
    prior_reason: str | None,
    recovery_actions: list[str] | None = None,
    dangling_attempt_ids: list[str] | None = None,
) -> dict:
    return {
        "invocation_id": resume_id,
        "prior_recorded_run_path": "/old/run",
        "current_run_path": "/current/run",
        "validated_previous_hash": events[-1]["event_hash"],
        "prior_terminal_reason": prior_reason,
        "recovery_actions": recovery_actions or ["none"],
        "dangling_attempt_ids": dangling_attempt_ids or [],
    }


def terminal_payload(
    invocation: str,
    reason: str,
    resumable: bool,
) -> dict:
    return {
        "invocation_id": invocation,
        "reason": reason,
        "last_completed_work_unit_id": None,
        "resumable": resumable,
        "detail": "test terminal",
    }


class BudgetReductionTests(unittest.TestCase):
    def test_empty_run_has_full_budget_and_no_calls(self) -> None:
        events = chain_events(
            run_created_payload(
                label="empty",
                target_budget=2,
                judge_budget=8,
                selector_budget=3,
                perturbator_budget=4,
            ),
            [],
        )
        state = reduce_events(events)
        self.assertEqual(state.consumed.to_dict(), {
            "target": 0,
            "judge": 0,
            "selector": 0,
            "perturbator": 0,
        })
        self.assertEqual(state.remaining.to_dict(), {
            "target": 2,
            "judge": 8,
            "selector": 3,
            "perturbator": 4,
        })
        self.assertEqual(state.dangling_attempt_ids, ())
        self.assertEqual(state.verified_archive_cells(), ())
        self.assertEqual(
            state.archive_metric(total_possible_cells=0).to_dict(),
            {
                "verified_cell_count": 0,
                "occupied_cell_count": 0,
                "total_possible_cells": 0,
                "all_target_attempts": 0,
                "completed_target_responses": 0,
                "rate_numerator": 0,
                "rate_denominator": 0,
                "rate_display": None,
            },
        )
        report_archive = archive_projection(state)
        self.assertTrue(report_archive["projection_complete"])
        self.assertEqual(report_archive["cells"], [])
        self.assertEqual(
            report_archive["capacity_per_cell"], state.archive.capacity_per_cell
        )

    def test_intent_does_not_charge_but_dispatch_does(self) -> None:
        fixture = call_fixture(label="intent-charge", role="target")
        created = run_created_payload(label="intent-charge", target_budget=2)
        intent_events = chain_events(
            created,
            with_unit(fixture, [("model_call_intended", intended_payload(fixture))]),
        )
        before_dispatch = reduce_events(intent_events)
        self.assertEqual(before_dispatch.consumed.target, 0)
        self.assertEqual(before_dispatch.remaining.target, 2)
        self.assertFalse(before_dispatch.attempt(attempt_id(fixture, 0)).dispatched)

        dispatched_events = chain_events(
            created,
            with_unit(
                fixture,
                [
                    ("model_call_intended", intended_payload(fixture)),
                    ("model_dispatch_started", dispatch_payload(fixture)),
                ],
            ),
        )
        after_dispatch = reduce_events(dispatched_events)
        self.assertEqual(after_dispatch.consumed.target, 1)
        self.assertEqual(after_dispatch.remaining.target, 1)
        self.assertEqual(after_dispatch.all_target_attempts, 1)
        self.assertEqual(after_dispatch.dangling_attempt_ids, (attempt_id(fixture, 0),))

    def test_each_dispatch_charges_exactly_its_frozen_role(self) -> None:
        for role in ("target", "judge", "selector", "perturbator"):
            fixture = call_fixture(label=f"role-{role}", role=role)
            events = chain_events(
                run_created_payload(label=f"role-{role}"),
                with_unit(
                    fixture,
                    [
                        ("model_call_intended", intended_payload(fixture)),
                        ("model_dispatch_started", dispatch_payload(fixture)),
                    ],
                ),
            )
            with self.subTest(role=role):
                state = reduce_events(events)
                expected = {name: 0 for name in ("target", "judge", "selector", "perturbator")}
                expected[role] = 1
                self.assertEqual(state.consumed.to_dict(), expected)

    def test_unknown_post_dispatch_outcome_remains_spent_and_visible(self) -> None:
        fixture = call_fixture(label="unknown", role="target")
        events = chain_events(
            run_created_payload(label="unknown", target_budget=2),
            with_unit(
                fixture,
                [
                    ("model_call_intended", intended_payload(fixture)),
                    ("model_dispatch_started", dispatch_payload(fixture)),
                    ("model_call_outcome_unknown", unknown_payload(fixture)),
                ],
            ),
        )
        state = reduce_events(events)
        attempt = state.attempt(attempt_id(fixture, 0))
        self.assertEqual(attempt.terminal_type, "model_call_outcome_unknown")
        self.assertEqual(state.consumed.target, 1)
        self.assertEqual(state.dangling_attempt_ids, ())
        self.assertEqual(state.completed_target_responses, 0)

    def test_persisted_dispatches_cannot_exceed_role_budget(self) -> None:
        fixture = call_fixture(label="over-budget", role="target", maximum_attempts=2)
        first_attempt = attempt_id(fixture, 0)
        events = chain_events(
            run_created_payload(label="over-budget", target_budget=1),
            with_unit(
                fixture,
                [
                    ("model_call_intended", intended_payload(fixture)),
                    ("model_dispatch_started", dispatch_payload(fixture)),
                    ("model_call_failed", failure_payload(fixture)),
                    (
                        "model_call_intended",
                        intended_payload(
                            fixture, ordinal=1, parent_attempt_id=first_attempt
                        ),
                    ),
                    ("model_dispatch_started", dispatch_payload(fixture, ordinal=1)),
                ],
            ),
        )
        with self.assertRaises(JournalCorrupt):
            reduce_events(events)


class AttemptInvariantTests(unittest.TestCase):
    def test_allocated_execution_is_visible_before_any_dispatch(self) -> None:
        fixture = call_fixture(label="allocated-execution", role="target")
        execution_id = entity_id("ex", ["allocated-execution"])
        fixture["unit"]["execution_slots"] = [
            {
                "execution_id": execution_id,
                "execution_scientific_key": fixture["owner_key"],
                "ordinal": 0,
                "arm": "none",
                "root_plan_id": None,
            }
        ]
        state = reduce_events(
            chain_events(
                run_created_payload(label="allocated-execution"),
                [("work_unit_started", {"unit": fixture["unit"]})],
            )
        )
        execution = state.sample_executions[execution_id]
        self.assertEqual(
            execution.terminal_status,
            ExecutionTerminalStatus.OPERATIONALLY_UNRESOLVED,
        )
        self.assertEqual(execution.observation_ids, ())
        self.assertFalse(state.execution_evidence[execution_id].terminated)
        self.assertFalse(state.execution_evidence[execution_id].resolved)

    def test_successful_retry_has_one_action_result_and_two_physical_charges(self) -> None:
        fixture = call_fixture(label="retry", role="target", maximum_attempts=2)
        first_attempt = attempt_id(fixture, 0)
        events = chain_events(
            run_created_payload(label="retry", target_budget=2),
            with_unit(
                fixture,
                [
                    ("model_call_intended", intended_payload(fixture)),
                    ("model_dispatch_started", dispatch_payload(fixture)),
                    ("model_call_failed", failure_payload(fixture)),
                    (
                        "model_call_intended",
                        intended_payload(
                            fixture, ordinal=1, parent_attempt_id=first_attempt
                        ),
                    ),
                    ("model_dispatch_started", dispatch_payload(fixture, ordinal=1)),
                    (
                        "model_call_succeeded",
                        success_payload(fixture, ordinal=1, text="retry output"),
                    ),
                ],
            ),
        )
        state = reduce_events(events)
        action = state.action(fixture["action_id"])
        self.assertEqual(action.dispatched_attempt_count, 2)
        self.assertEqual(state.consumed.target, 2)
        self.assertEqual(action.successful_attempt.physical_attempt_id, attempt_id(fixture, 1))
        self.assertEqual(state.latest_success(fixture["action_id"]).output_text, "retry output")
        self.assertEqual(state.completed_target_responses, 1)
        self.assertEqual(
            state.target_results[fixture["action_id"]].physical_attempt_id,
            attempt_id(fixture, 1),
        )

    def test_success_projection_preserves_target_evidence_and_identity(self) -> None:
        fixture = call_fixture(label="success", role="target")
        events = chain_events(
            run_created_payload(label="success"),
            with_unit(
                fixture,
                [
                    ("model_call_intended", intended_payload(fixture)),
                    ("model_dispatch_started", dispatch_payload(fixture)),
                    (
                        "model_call_succeeded",
                        success_payload(
                            fixture,
                            text="durable exact text",
                            effective_model_id="immutable/model-revision",
                        ),
                    ),
                ],
            ),
        )
        state = reduce_events(events)
        result = state.target_results[fixture["action_id"]]
        self.assertEqual(result.text, "durable exact text")
        self.assertTrue(result.usable)
        self.assertEqual(state.target_identity, "immutable/model-revision")
        self.assertFalse(state.target_identity_unverifiable)
        self.assertFalse(state.target_identity_drift)
        observation = state.target_observations[fixture["action_id"]]
        self.assertEqual(
            observation.observation_id,
            entity_id("to", [fixture["action_id"]]),
        )
        self.assertEqual(
            observation.scientific_key,
            digest_json([fixture["action_key"], "observation"]),
        )
        self.assertEqual(observation.successful_attempt_id, attempt_id(fixture, 0))
        self.assertEqual(observation.response, result.text)

    def test_every_successful_target_receipt_pins_identity_and_drift_is_excluded(self) -> None:
        fixture = call_fixture(label="identity-retry", role="target", maximum_attempts=2)
        first_attempt = attempt_id(fixture, 0)
        partial = success_payload(
            fixture,
            text="partial output",
            effective_model_id="immutable/model-a",
        )
        partial["finish_status"] = "output_limit"
        events = chain_events(
            run_created_payload(label="identity-retry", target_budget=2),
            with_unit(
                fixture,
                [
                    ("model_call_intended", intended_payload(fixture)),
                    ("model_dispatch_started", dispatch_payload(fixture)),
                    ("model_call_succeeded", partial),
                    (
                        "model_call_intended",
                        intended_payload(
                            fixture, ordinal=1, parent_attempt_id=first_attempt
                        ),
                    ),
                    ("model_dispatch_started", dispatch_payload(fixture, ordinal=1)),
                    (
                        "model_call_succeeded",
                        success_payload(
                            fixture,
                            ordinal=1,
                            text="complete but drifted",
                            effective_model_id="immutable/model-b",
                        ),
                    ),
                ],
            ),
        )
        state = reduce_events(events)
        self.assertEqual(state.target_identity, "immutable/model-a")
        self.assertTrue(state.target_identity_drift)
        self.assertEqual(
            state.target_identity_mismatch_attempt_ids,
            (attempt_id(fixture, 1),),
        )
        self.assertNotIn(fixture["action_id"], state.target_results)
        self.assertEqual(state.completed_target_responses, 0)

    def test_terminal_without_dispatch_and_duplicate_terminal_are_corrupt(self) -> None:
        fixture = call_fixture(label="terminality", role="target")
        created = run_created_payload(label="terminality")
        without_dispatch = chain_events(
            created,
            with_unit(
                fixture,
                [
                    ("model_call_intended", intended_payload(fixture)),
                    ("model_call_failed", failure_payload(fixture)),
                ],
            ),
        )
        with self.assertRaises(JournalCorrupt):
            validate_events(without_dispatch)

        duplicate = chain_events(
            created,
            with_unit(
                fixture,
                [
                    ("model_call_intended", intended_payload(fixture)),
                    ("model_dispatch_started", dispatch_payload(fixture)),
                    ("model_call_failed", failure_payload(fixture)),
                    ("model_call_outcome_unknown", unknown_payload(fixture)),
                ],
            ),
        )
        with self.assertRaises(JournalCorrupt):
            validate_events(duplicate)

    def test_attempts_must_match_frozen_slot_route_role_and_retry_chain(self) -> None:
        fixture = call_fixture(label="frozen", role="target", maximum_attempts=2)
        created = run_created_payload(label="frozen")

        wrong_action_key = intended_payload(fixture)
        wrong_action_key["logical_action_scientific_key"] = (
            "sha256:" + "f" * 64
        )
        wrong_intent = chain_events(
            created,
            with_unit(fixture, [("model_call_intended", wrong_action_key)]),
        )

        wrong_charge = dispatch_payload(fixture)
        wrong_charge["charged_role"] = "judge"
        wrong_dispatch = chain_events(
            created,
            with_unit(
                fixture,
                [
                    ("model_call_intended", intended_payload(fixture)),
                    ("model_dispatch_started", wrong_charge),
                ],
            ),
        )

        noncontiguous = chain_events(
            created,
            with_unit(
                fixture,
                [
                    (
                        "model_call_intended",
                        intended_payload(fixture, ordinal=1, parent_attempt_id=None),
                    )
                ],
            ),
        )
        for label, events in (
            ("intent", wrong_intent),
            ("dispatch", wrong_dispatch),
            ("ordinal", noncontiguous),
        ):
            with self.subTest(label=label), self.assertRaises(JournalCorrupt):
                reduce_events(events)

    def test_terminal_state_is_a_reduced_event_fact(self) -> None:
        created = run_created_payload(label="terminated")
        events = chain_events(
            created,
            [
                (
                    "run_terminated",
                    {
                        "invocation_id": invocation_id("terminated-creation"),
                        "reason": "completed_plan",
                        "last_completed_work_unit_id": None,
                        "resumable": False,
                        "detail": "no admissible unit fits",
                    },
                )
            ],
        )
        state = reduce_events(events)
        self.assertEqual(state.terminal_reason, RunTerminalReason.COMPLETED_PLAN)
        self.assertFalse(state.resumable)
        self.assertIsNone(state.last_completed_work_unit_id)


class InvocationLifecycleTests(unittest.TestCase):
    def test_every_terminal_reason_has_one_exact_resumability(self) -> None:
        expected = {
            "completed_no_eligible_work": False,
            "completed_plan": False,
            "operator_interrupted": True,
            "systemic_provider_failure": True,
            "provider_identity_drift": False,
            "evidence_unwritable": True,
            "evidence_corrupt": False,
            "impossible_internal_state": False,
        }
        for reason, resumable in expected.items():
            label = f"terminal-map-{reason}"
            created = run_created_payload(label=label)
            creation_id = created["creation_invocation_id"]
            base = chain_events(created, [])
            valid = append_record(
                base,
                "run_terminated",
                terminal_payload(creation_id, reason, resumable),
            )
            with self.subTest(reason=reason, variant="valid"):
                self.assertEqual(reduce_events(valid).resumable, resumable)

            invalid = append_record(
                base,
                "run_terminated",
                terminal_payload(creation_id, reason, not resumable),
            )
            with self.subTest(reason=reason, variant="invalid"):
                with self.assertRaises(JournalCorrupt):
                    reduce_events(invalid)

    def test_resume_supersedes_resumable_terminal_and_must_close_current_invocation(self) -> None:
        created = run_created_payload(label="resume-terminal")
        creation_id = created["creation_invocation_id"]
        resumed_id = invocation_id("resume-terminal-second")
        events = chain_events(created, [])
        events = append_record(
            events,
            "run_terminated",
            terminal_payload(
                creation_id, "systemic_provider_failure", resumable=True
            ),
        )
        events = append_record(
            events,
            "run_resumed",
            resume_payload(
                events,
                resumed_id,
                prior_reason="systemic_provider_failure",
            ),
        )
        open_state = reduce_events(events)
        self.assertIsNone(open_state.terminal_event)
        self.assertFalse(open_state.resumable)

        closed = append_record(
            events,
            "run_terminated",
            terminal_payload(resumed_id, "completed_plan", resumable=False),
        )
        self.assertEqual(
            reduce_events(closed).terminal_reason,
            RunTerminalReason.COMPLETED_PLAN,
        )

        foreign = append_record(
            events,
            "run_terminated",
            terminal_payload(creation_id, "completed_plan", resumable=False),
        )
        with self.assertRaises(JournalCorrupt):
            reduce_events(foreign)

        duplicate_resume = append_record(
            events,
            "run_resumed",
            resume_payload(events, resumed_id, prior_reason=None),
        )
        with self.assertRaises(JournalCorrupt):
            reduce_events(duplicate_resume)

    def test_resumed_semantic_unit_progress_does_not_rewrite_prior_terminal_boundary(self) -> None:
        first = call_fixture(
            label="resume-semantic-first",
            role="selector",
            maximum_attempts=1,
        )
        resumed_unit = call_fixture(
            label="resume-semantic-active",
            role="selector",
            maximum_attempts=1,
        )
        created = run_created_payload(
            label="resume-semantic-boundary",
            selector_budget=2,
        )
        events = chain_events(
            created,
            [
                ("work_unit_started", {"unit": first["unit"]}),
                ("model_call_intended", intended_payload(first)),
                ("model_dispatch_started", dispatch_payload(first)),
                ("model_call_succeeded", success_payload(first, text="{}")),
                ("work_unit_started", {"unit": resumed_unit["unit"]}),
            ],
        )
        events = append_record(
            events,
            "run_terminated",
            {
                "invocation_id": created["creation_invocation_id"],
                "reason": "operator_interrupted",
                "last_completed_work_unit_id": first["unit_id"],
                "resumable": True,
                "detail": "interrupted with one active semantic unit",
            },
        )
        resumed_id = invocation_id("resume-semantic-boundary-second")
        events = append_record(
            events,
            "run_resumed",
            resume_payload(
                events,
                resumed_id,
                prior_reason="operator_interrupted",
            ),
        )
        for event_type, payload in (
            ("model_call_intended", intended_payload(resumed_unit)),
            ("model_dispatch_started", dispatch_payload(resumed_unit)),
            (
                "model_call_succeeded",
                success_payload(resumed_unit, text="{}"),
            ),
        ):
            events = append_record(events, event_type, payload)
        events = append_record(
            events,
            "run_terminated",
            {
                "invocation_id": resumed_id,
                "reason": "completed_plan",
                "last_completed_work_unit_id": resumed_unit["unit_id"],
                "resumable": False,
                "detail": "resumed unit completed cleanly",
            },
        )

        state = reduce_events(events)
        active_unit_id, completed_unit_ids = _validate_work_unit_boundaries(
            events,
            state.work_units,
            state.logical_actions,
            {
                FOUNDATION_RESPONSE_SCHEMA_DIGEST: FOUNDATION_RESPONSE_SCHEMA,
            },
            # This is the final-journal stateful projection that previously
            # leaked backward into the first run_terminated checkpoint.
            {resumed_unit["unit_id"]: True},
        )

        self.assertIsNone(active_unit_id)
        self.assertEqual(
            completed_unit_ids,
            (first["unit_id"], resumed_unit["unit_id"]),
        )

    def test_crash_resume_requires_exact_predecessor_hash_reason_and_cause(self) -> None:
        created = run_created_payload(label="crash-resume")
        base = chain_events(created, [])
        resumed_id = invocation_id("crash-resume-second")
        payload = resume_payload(base, resumed_id, prior_reason=None)
        valid = append_record(base, "run_resumed", payload)
        self.assertIsNone(reduce_events(valid).terminal_event)

        wrong_hash = dict(payload)
        wrong_hash["validated_previous_hash"] = digest("wrong-previous")
        wrong_reason = dict(payload)
        wrong_reason["prior_terminal_reason"] = "operator_interrupted"
        bad_cause = append_record(base, "run_resumed", payload, causes=())
        for label, events in (
            ("hash", append_record(base, "run_resumed", wrong_hash)),
            ("reason", append_record(base, "run_resumed", wrong_reason)),
            ("cause", bad_cause),
        ):
            with self.subTest(label=label), self.assertRaises(JournalCorrupt):
                reduce_events(events)

    def test_nonresumable_terminal_cannot_resume_or_continue_scientifically(self) -> None:
        created = run_created_payload(label="closed-terminal")
        creation_id = created["creation_invocation_id"]
        terminal = append_record(
            chain_events(created, []),
            "run_terminated",
            terminal_payload(
                creation_id, "completed_no_eligible_work", resumable=False
            ),
        )
        resumed_id = invocation_id("closed-terminal-second")
        resumed = append_record(
            terminal,
            "run_resumed",
            resume_payload(
                terminal,
                resumed_id,
                prior_reason="completed_no_eligible_work",
            ),
        )
        continued = append_record(
            terminal,
            "work_unit_started",
            {"unit": call_fixture(label="closed-continuation")["unit"]},
        )
        for label, events in (("resume", resumed), ("continue", continued)):
            with self.subTest(label=label), self.assertRaises(JournalCorrupt):
                reduce_events(events)

    def test_resume_declares_and_closes_exact_dangling_dispatches(self) -> None:
        fixture = call_fixture(label="resume-dangling", role="target")
        created = run_created_payload(label="resume-dangling")
        base = chain_events(
            created,
            with_unit(
                fixture,
                [
                    ("model_call_intended", intended_payload(fixture)),
                    ("model_dispatch_started", dispatch_payload(fixture)),
                ],
            ),
        )
        resumed_id = invocation_id("resume-dangling-second")
        dangling_id = attempt_id(fixture, 0)
        resumed = append_record(
            base,
            "run_resumed",
            resume_payload(
                base,
                resumed_id,
                prior_reason=None,
                recovery_actions=["dangling_dispatch_closed"],
                dangling_attempt_ids=[dangling_id],
            ),
        )
        closed = append_record(
            resumed,
            "model_call_outcome_unknown",
            unknown_payload(
                fixture,
                discovery_invocation_id=resumed_id,
            ),
        )
        self.assertEqual(reduce_events(closed).dangling_attempt_ids, ())

        undeclared = append_record(
            base,
            "run_resumed",
            resume_payload(base, resumed_id, prior_reason=None),
        )
        foreign_closure = append_record(
            resumed,
            "model_call_outcome_unknown",
            unknown_payload(
                fixture,
                discovery_invocation_id=created["creation_invocation_id"],
            ),
        )
        terminal_with_dangling = append_record(
            base,
            "run_terminated",
            terminal_payload(
                created["creation_invocation_id"],
                "operator_interrupted",
                resumable=True,
            ),
        )
        for label, events in (
            ("undeclared", undeclared),
            ("foreign-closure", foreign_closure),
            ("terminal-with-dangling", terminal_with_dangling),
        ):
            with self.subTest(label=label), self.assertRaises(JournalCorrupt):
                reduce_events(events)

    def test_tail_recovery_must_match_preceding_hash_and_resume_actions(self) -> None:
        created = run_created_payload(label="tail-recovery")
        base = chain_events(created, [])
        resumed_id = invocation_id("tail-recovery-second")
        recovery_payload = {
            "recovery_invocation_id": resumed_id,
            "pending_event_hash": digest("pending-event"),
            "original_journal_bytes": 10,
            "original_pending_bytes": 20,
            "action": "pending_appended",
            "resulting_terminal_hash": base[-1]["event_hash"],
        }
        recovered = append_record(
            base, "journal_tail_recovered", recovery_payload
        )
        resumed = append_record(
            recovered,
            "run_resumed",
            resume_payload(
                recovered,
                resumed_id,
                prior_reason=None,
                recovery_actions=["pending_appended"],
            ),
        )
        self.assertIsNone(reduce_events(resumed).terminal_event)

        bad_hash_payload = dict(recovery_payload)
        bad_hash_payload["resulting_terminal_hash"] = digest("wrong-tail")
        bad_hash = append_record(
            base, "journal_tail_recovered", bad_hash_payload
        )
        unsupported_claim_id = invocation_id("tail-recovery-unsupported")
        unsupported_claim = append_record(
            base,
            "run_resumed",
            resume_payload(
                base,
                unsupported_claim_id,
                prior_reason=None,
                recovery_actions=["partial_tail_replaced"],
            ),
        )
        for label, events in (
            ("hash", bad_hash),
            ("unsupported-claim", unsupported_claim),
        ):
            with self.subTest(label=label), self.assertRaises(JournalCorrupt):
                reduce_events(events)

    def test_terminal_requires_immediate_cause(self) -> None:
        created = run_created_payload(label="terminal-cause")
        base = chain_events(created, [])
        invalid = append_record(
            base,
            "run_terminated",
            terminal_payload(
                created["creation_invocation_id"],
                "completed_plan",
                resumable=False,
            ),
            causes=(),
        )
        with self.assertRaises(JournalCorrupt):
            reduce_events(invalid)


class ProviderCircuitReductionTests(unittest.TestCase):
    def test_three_consecutive_failures_trip_and_explicit_reset_starts_new_window(self) -> None:
        outcomes = tuple(
            ProviderCircuitOutcome(
                role="target",
                sequence=index,
                scientific_key="sha256:" + f"{index:064x}",
                status="failure",
            )
            for index in range(1, 4)
        )
        tripped = reduce_provider_circuit(outcomes)
        self.assertEqual(tripped.target, 3)
        self.assertEqual(tripped.tripped_role, "target")
        self.assertEqual(tripped.tripped_sequence, 3)

        reset = reduce_provider_circuit(outcomes, reset_after={"target": 3})
        self.assertEqual(reset.target, 0)
        self.assertIsNone(reset.tripped_role)


if __name__ == "__main__":
    unittest.main()
