from __future__ import annotations

from types import SimpleNamespace
import unittest

from quasilisk.domain import ActionPurpose, Role, digest_json
from quasilisk.events import JournalCorrupt
from quasilisk.engine import (
    ROLE_SCHEMA_NAMES,
    SemanticResult,
    _next_route_plan,
    _provider_circuit_outcomes,
    _response_mode_for_scope,
    _successful_payloads,
)
from quasilisk.run_state import reduce_provider_circuit
from quasilisk.run_state import (
    AttemptView,
    LogicalActionView,
    WorkUnitView,
    _apply_role_identity_projection,
    _completion_snapshot,
    _prefix_target_collision_attempt_ids,
    validate_recovery_transitions,
)
from quasilisk.reporting import render_summary_markdown


def _success_attempt(
    *,
    attempt_id: str,
    route_key: str,
    sequence: int,
    collision: bool,
    effective_model_id: str | None = "target/revision",
) -> SimpleNamespace:
    return SimpleNamespace(
        physical_attempt_id=attempt_id,
        succeeded=True,
        target_identity_collision=collision,
        intent_event={"sequence": sequence - 2},
        intended={
            "role_route_key": route_key,
            "response_schema_digest": digest_json(["collision-test", "schema"]),
            "response_mode": "json_schema",
            "route_kind": "primary",
        },
        terminal_event={
            "sequence": sequence,
            "type": "model_call_succeeded",
            "payload": {
                "finish_status": "complete",
                "text": "{}",
                "effective_model_id": effective_model_id,
            },
        },
    )


def _attempt_view(
    *,
    attempt_id: str,
    role: Role,
    intent_sequence: int,
    terminal_sequence: int,
    effective_model_id: str | None,
) -> AttemptView:
    return AttemptView(
        physical_attempt_id=attempt_id,
        intent_event={
            "sequence": intent_sequence,
            "payload": {
                "response_mode": (
                    "text" if role is Role.TARGET else "json_schema"
                ),
                "lineage": {
                    "candidate_id": "candidate" if role is Role.TARGET else None,
                    "execution_id": None,
                    "conversation_root_id": None,
                    "parent_node_id": None,
                },
            },
        },
        dispatch_event={"sequence": intent_sequence + 1},
        terminal_event={
            "event_id": f"event-{attempt_id}",
            "sequence": terminal_sequence,
            "type": "model_call_succeeded",
            "payload": {
                "finish_status": "complete",
                "text": "target text" if role is Role.TARGET else "{}",
                "effective_model_id": effective_model_id,
            },
        },
    )


def _action_view(
    *, action_id: str, role: Role, attempt: AttemptView
) -> LogicalActionView:
    return LogicalActionView(
        action_id=action_id,
        unit_id="identity-unit",
        slot={"ordinal": 0, "role": role.value},
        attempts=(attempt,),
    )


class RouteCollisionPolicyTests(unittest.TestCase):
    def _action(self, attempts: tuple[SimpleNamespace, ...]) -> SimpleNamespace:
        primary_key = digest_json(["collision-test", "primary"])
        fallback_key = digest_json(["collision-test", "fallback"])
        return SimpleNamespace(
            action_id="selector-action",
            slot={
                "role": Role.SELECTOR.value,
                "action_scientific_key": digest_json(
                    ["collision-test", "selector-action"]
                ),
                "routes": [
                    {
                        "route_key": primary_key,
                        "route_label": "primary",
                        "maximum_attempts": 2,
                    },
                    {
                        "route_key": fallback_key,
                        "route_label": "fallback",
                        "maximum_attempts": 2,
                    },
                ],
            },
            attempts=attempts,
        )

    def test_primary_collision_exhausts_route_and_selects_fallback(self) -> None:
        primary_key = digest_json(["collision-test", "primary"])
        action = self._action(
            (
                _success_attempt(
                    attempt_id="primary-collision",
                    route_key=primary_key,
                    sequence=3,
                    collision=True,
                ),
            )
        )
        route, kind = _next_route_plan(
            action, SemanticResult(None, ("target identity collision",))
        )
        self.assertEqual(kind, "fallback")
        self.assertEqual(route["route_label"], "fallback")

    def test_fallback_collision_closes_without_same_route_repair(self) -> None:
        primary_key = digest_json(["collision-test", "primary"])
        fallback_key = digest_json(["collision-test", "fallback"])
        primary = _success_attempt(
            attempt_id="primary-collision",
            route_key=primary_key,
            sequence=3,
            collision=True,
        )
        fallback = _success_attempt(
            attempt_id="fallback-collision",
            route_key=fallback_key,
            sequence=6,
            collision=True,
        )
        fallback.intended["route_kind"] = "fallback"
        action = self._action((primary, fallback))
        self.assertIsNone(
            _next_route_plan(
                action, SemanticResult(None, ("target identity collision",))
            )
        )

    def test_null_effective_identity_remains_usable_input(self) -> None:
        primary_key = digest_json(["collision-test", "primary"])
        attempt = _success_attempt(
            attempt_id="null-identity",
            route_key=primary_key,
            sequence=3,
            collision=False,
            effective_model_id=None,
        )
        action = self._action((attempt,))
        self.assertEqual(_successful_payloads(action), ((attempt, attempt.terminal_event["payload"]),))


class ResponseModeCollisionTests(unittest.TestCase):
    def test_colliding_strict_success_does_not_block_later_downgrade(self) -> None:
        schema_digest = digest_json(["collision-test", "schema"])
        route_key = digest_json(["collision-test", "mode-route"])
        strict_collision = _success_attempt(
            attempt_id="strict-collision",
            route_key=route_key,
            sequence=3,
            collision=True,
        )
        strict_collision.intended["response_schema_digest"] = schema_digest
        rejected = SimpleNamespace(
            physical_attempt_id="later-rejection",
            target_identity_collision=False,
            intent_event={"sequence": 4},
            intended={
                "role_route_key": route_key,
                "response_schema_digest": schema_digest,
                "response_mode": "json_schema",
            },
            terminal_event={
                "sequence": 6,
                "type": "model_call_failed",
                "payload": {"error_kind": "unsupported_response_format"},
            },
        )
        context = SimpleNamespace(
            resources=SimpleNamespace(
                schema_digests={name: schema_digest for name in ROLE_SCHEMA_NAMES},
                schemas={name: {"type": "object"} for name in ROLE_SCHEMA_NAMES},
            )
        )
        state = SimpleNamespace(
            attempts={
                strict_collision.physical_attempt_id: strict_collision,
                rejected.physical_attempt_id: rejected,
            },
        )
        binding = SimpleNamespace(route_key=route_key)
        self.assertEqual(
            _response_mode_for_scope(
                context,
                state,
                binding=binding,
                response_schema_digest=schema_digest,
            ),
            "json_object",
        )


class ReducerIdentityProjectionTests(unittest.TestCase):
    def test_final_replay_retroactively_disqualifies_pre_pin_role_receipt(self) -> None:
        selector = _action_view(
            action_id="selector",
            role=Role.SELECTOR,
            attempt=_attempt_view(
                attempt_id="selector-before-pin",
                role=Role.SELECTOR,
                intent_sequence=1,
                terminal_sequence=3,
                effective_model_id="target/revision",
            ),
        )
        target = _action_view(
            action_id="target",
            role=Role.TARGET,
            attempt=_attempt_view(
                attempt_id="target-pin",
                role=Role.TARGET,
                intent_sequence=4,
                terminal_sequence=6,
                effective_model_id="target/revision",
            ),
        )
        raw = {selector.action_id: selector, target.action_id: target}

        projected, collisions, unverifiable = _apply_role_identity_projection(
            raw, "target/revision"
        )
        self.assertEqual(collisions[Role.SELECTOR.value], ("selector-before-pin",))
        self.assertEqual(unverifiable[Role.SELECTOR.value], ())
        self.assertTrue(
            projected["selector"].attempts[0].target_identity_collision
        )
        # Reprojection is deterministic, matching fresh replay after resume.
        replayed, replay_collisions, replay_unverifiable = (
            _apply_role_identity_projection(projected, "target/revision")
        )
        self.assertEqual(replayed, projected)
        self.assertEqual(replay_collisions, collisions)
        self.assertEqual(replay_unverifiable, unverifiable)

        # Historical recovery at the Selector receipt's own prefix had no pin.
        self.assertNotIn(
            "selector-before-pin", _prefix_target_collision_attempt_ids(raw)
        )
        unit = WorkUnitView(
            start_event={"sequence": 1},
            unit={"unit_id": "identity-unit"},
            actions=(selector, target),
        )
        _, before_actions, _ = _completion_snapshot((unit,), raw, 3)
        _, after_actions, _ = _completion_snapshot((unit,), raw, 6)
        self.assertFalse(
            before_actions["selector"].attempts[0].target_identity_collision
        )
        self.assertTrue(
            after_actions["selector"].attempts[0].target_identity_collision
        )

    def test_collision_known_at_receipt_prefix_and_null_identity_is_unverifiable(self) -> None:
        target = _action_view(
            action_id="target",
            role=Role.TARGET,
            attempt=_attempt_view(
                attempt_id="target-pin",
                role=Role.TARGET,
                intent_sequence=1,
                terminal_sequence=3,
                effective_model_id="target/revision",
            ),
        )
        colliding = _action_view(
            action_id="perturbator",
            role=Role.PERTURBATOR,
            attempt=_attempt_view(
                attempt_id="perturbator-after-pin",
                role=Role.PERTURBATOR,
                intent_sequence=4,
                terminal_sequence=6,
                effective_model_id="target/revision",
            ),
        )
        null_selector = _action_view(
            action_id="selector-null",
            role=Role.SELECTOR,
            attempt=_attempt_view(
                attempt_id="selector-null",
                role=Role.SELECTOR,
                intent_sequence=7,
                terminal_sequence=9,
                effective_model_id=None,
            ),
        )
        raw = {
            target.action_id: target,
            colliding.action_id: colliding,
            null_selector.action_id: null_selector,
        }
        self.assertEqual(
            _prefix_target_collision_attempt_ids(raw),
            frozenset({"perturbator-after-pin"}),
        )
        projected, collisions, unverifiable = _apply_role_identity_projection(
            raw, "target/revision"
        )
        self.assertEqual(
            collisions[Role.PERTURBATOR.value], ("perturbator-after-pin",)
        )
        self.assertEqual(
            unverifiable[Role.SELECTOR.value], ("selector-null",)
        )
        self.assertFalse(
            projected["selector-null"].attempts[0].target_identity_collision
        )

    def test_markdown_report_names_collision_and_unverifiable_attempts(self) -> None:
        markdown = render_summary_markdown(
            {
                "role_identity_integrity": {
                    "selector": {
                        "target_collision_attempt_ids": ["selector-collision"],
                        "identity_unverifiable_attempt_ids": ["selector-null"],
                    },
                    "perturbator": {
                        "target_collision_attempt_ids": ["perturbator-collision"],
                        "identity_unverifiable_attempt_ids": [],
                    },
                }
            }
        )
        self.assertIn("selector-collision", markdown)
        self.assertIn("selector-null", markdown)
        self.assertIn("perturbator-collision", markdown)

    def test_recovery_validator_allows_only_immediate_fallback_after_collision(
        self,
    ) -> None:
        schema_digest = digest_json({"type": "object"})
        messages = [{"role": "user", "content": "request"}]

        def intent(
            attempt_id: str,
            *,
            sequence: int,
            role: str,
            route_key: str,
            route_kind: str,
            schema: str | None,
            mode: str,
            request_messages: list[dict[str, str]],
        ) -> dict:
            return {
                "sequence": sequence,
                "type": "model_call_intended",
                "payload": {
                    "physical_attempt_id": attempt_id,
                    "role": role,
                    "role_route_key": route_key,
                    "route_kind": route_kind,
                    "response_schema_digest": schema,
                    "response_mode": mode,
                    "messages": request_messages,
                    "semantic_request_hash": f"semantic-{attempt_id}",
                },
            }

        def success(attempt_id: str, sequence: int, text: str) -> dict:
            return {
                "sequence": sequence,
                "type": "model_call_succeeded",
                "payload": {
                    "physical_attempt_id": attempt_id,
                    "finish_status": "complete",
                    "text": text,
                    "effective_model_id": "target/revision",
                },
            }

        target_intent = intent(
            "target-pin",
            sequence=1,
            role="target",
            route_key="target-route",
            route_kind="primary",
            schema=None,
            mode="text",
            request_messages=messages,
        )
        target_terminal = success("target-pin", 3, "target text")
        target = LogicalActionView(
            "target",
            "unit",
            {
                "role": "target",
                "routes": [
                    {
                        "route_key": "target-route",
                        "route_label": "primary",
                        "maximum_attempts": 2,
                    }
                ],
            },
            (AttemptView("target-pin", target_intent, None, target_terminal),),
        )
        primary_intent = intent(
            "selector-primary",
            sequence=4,
            role="selector",
            route_key="selector-primary-route",
            route_kind="primary",
            schema=schema_digest,
            mode="json_schema",
            request_messages=messages,
        )
        primary_terminal = success("selector-primary", 6, "{}")
        fallback_intent = intent(
            "selector-fallback",
            sequence=7,
            role="selector",
            route_key="selector-fallback-route",
            route_kind="fallback",
            schema=schema_digest,
            mode="json_schema",
            request_messages=messages,
        )
        slot = {
            "role": "selector",
            "routes": [
                {
                    "route_key": "selector-primary-route",
                    "route_label": "primary",
                    "maximum_attempts": 2,
                },
                {
                    "route_key": "selector-fallback-route",
                    "route_label": "fallback",
                    "maximum_attempts": 2,
                },
            ],
        }
        fallback_action = LogicalActionView(
            "selector",
            "unit",
            slot,
            (
                AttemptView(
                    "selector-primary", primary_intent, None, primary_terminal
                ),
                AttemptView("selector-fallback", fallback_intent, None, None),
            ),
        )
        events = [
            target_intent,
            target_terminal,
            primary_intent,
            primary_terminal,
            fallback_intent,
        ]
        validate_recovery_transitions(
            events,
            {"target": target, "selector": fallback_action},
            {"target": "map_exploration", "selector": "map_exploration"},
            {schema_digest: {"type": "object"}},
        )

        repair_intent = intent(
            "selector-repair",
            sequence=7,
            role="selector",
            route_key="selector-primary-route",
            route_kind="schema_repair",
            schema=schema_digest,
            mode="json_schema",
            request_messages=[*messages, {"role": "user", "content": "repair"}],
        )
        repair_action = LogicalActionView(
            "selector",
            "unit",
            slot,
            (
                AttemptView(
                    "selector-primary", primary_intent, None, primary_terminal
                ),
                AttemptView("selector-repair", repair_intent, None, None),
            ),
        )
        with self.assertRaisesRegex(JournalCorrupt, "no legal prior terminal"):
            validate_recovery_transitions(
                [*events[:-1], repair_intent],
                {"target": target, "selector": repair_action},
                {"target": "map_exploration", "selector": "map_exploration"},
                {schema_digest: {"type": "object"}},
            )


class CollisionCircuitTests(unittest.TestCase):
    def _state_for_role(self, role: Role) -> tuple[SimpleNamespace, SimpleNamespace]:
        schema_name = (
            "template_binding.json"
            if role is Role.SELECTOR
            else "proposal_batch.json"
        )
        purpose = (
            ActionPurpose.TEMPLATE_BINDING.value
            if role is Role.SELECTOR
            else ActionPurpose.PERTURBATOR_PROPOSAL.value
        )
        schema_digest = digest_json(["collision-test", schema_name])
        actions = []
        for index in range(3):
            primary_key = digest_json(
                ["collision-test", role.value, index, "primary"]
            )
            fallback_key = digest_json(
                ["collision-test", role.value, index, "fallback"]
            )
            primary = _success_attempt(
                attempt_id=f"{role.value}-primary-collision-{index}",
                route_key=primary_key,
                sequence=6 * index + 3,
                collision=True,
            )
            fallback = _success_attempt(
                attempt_id=f"{role.value}-fallback-collision-{index}",
                route_key=fallback_key,
                sequence=6 * index + 6,
                collision=True,
            )
            fallback.intended["route_kind"] = "fallback"
            primary.intended["response_schema_digest"] = schema_digest
            fallback.intended["response_schema_digest"] = schema_digest
            actions.append(
                SimpleNamespace(
                    action_id=f"{role.value}-action-{index}",
                    slot={
                        "role": role.value,
                        "purpose": purpose,
                        "action_scientific_key": digest_json(
                            ["collision-test", role.value, "action", index]
                        ),
                        "routes": [
                            {
                                "route_key": primary_key,
                                "route_label": "primary",
                                "maximum_attempts": 2,
                            },
                            {
                                "route_key": fallback_key,
                                "route_label": "fallback",
                                "maximum_attempts": 2,
                            },
                        ],
                    },
                    attempts=(primary, fallback),
                )
            )
        unit = SimpleNamespace(
            unit={"kind": "initial_observation", "panel_plans": []},
            actions=tuple(actions),
        )
        state = SimpleNamespace(
            work_units=(unit,),
            target_results={},
            logical_actions={item.action_id: item for item in actions},
        )
        context = SimpleNamespace(
            validated_payloads={},
            resources=SimpleNamespace(
                schema_digests={schema_name: schema_digest},
                schemas={schema_name: {"type": "object"}},
            )
        )
        return context, state

    def test_three_collision_exhausted_actions_trip_each_role_independently(self) -> None:
        for role in (Role.SELECTOR, Role.PERTURBATOR):
            with self.subTest(role=role.value):
                context, state = self._state_for_role(role)
                outcomes = _provider_circuit_outcomes(context, state)
                self.assertEqual([item.status for item in outcomes], ["failure"] * 3)
                circuit = reduce_provider_circuit(outcomes)
                self.assertEqual(circuit.streak(role.value), 3)
                self.assertEqual(circuit.tripped_role, role.value)


if __name__ == "__main__":
    unittest.main()
