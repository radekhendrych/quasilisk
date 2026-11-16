from __future__ import annotations

from foundation_support import judge_schema

from copy import deepcopy
import unittest

from foundation_support import (
    attempt_id,
    call_fixture,
    chain_events,
    dispatch_payload,
    failure_payload,
    intended_payload,
    manifest as foundation_manifest,
    reduce_fixture_events as reduce_events,
    run_created_payload,
    success_payload,
)
from quasilisk.domain import (
    canonical_json_bytes,
    digest_json,
    DomainError,
    entity_id,
    strict_json_loads,
)
from quasilisk.events import JournalCorrupt
from quasilisk.provider import encode_bytes
from quasilisk.run_state import (
    AttemptView,
    LogicalActionView,
    _parsed_structured_action,
    _validate_request_manifest,
    validate_recovery_transitions,
)


def with_unit(fixture: dict, records: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    return [("work_unit_started", {"unit": fixture["unit"]}), *records]


class ReplayedRequestAuthenticationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = call_fixture(label="replay-auth", role="target")
        self.created = run_created_payload(label="replay-auth")

    def _reduce_intent(self, intent: dict) -> None:
        reduce_events(
            chain_events(
                self.created,
                with_unit(self.fixture, [("model_call_intended", intent)]),
            )
        )

    def test_rejects_forged_semantic_hash_attempt_id_and_frozen_profile(self) -> None:
        variants: list[tuple[str, dict]] = []

        semantic = intended_payload(self.fixture)
        semantic["semantic_request_hash"] = "sha256:" + "f" * 64
        variants.append(("semantic request hash", semantic))

        physical = intended_payload(self.fixture)
        physical["physical_attempt_id"] = entity_id("pa", ["forged"])
        variants.append(("physical attempt ID", physical))

        timeout = intended_payload(self.fixture)
        timeout["timeout_seconds"] = 2
        variants.append(("timeout", timeout))

        generation = intended_payload(self.fixture)
        generation["generation_parameters"]["max_output_tokens"] = 31
        variants.append(("generation parameters", generation))

        for label, intent in variants:
            with self.subTest(label=label), self.assertRaises(JournalCorrupt):
                self._reduce_intent(intent)

    def test_rejects_noncanonical_or_schema_substituted_request_body(self) -> None:
        noncanonical = intended_payload(self.fixture)
        noncanonical["request_body"]["data"] += " "
        with self.assertRaisesRegex(JournalCorrupt, "request_body"):
            self._reduce_intent(noncanonical)

        judge = call_fixture(label="embedded-schema", role="judge")
        created = run_created_payload(label="embedded-schema")
        substituted = intended_payload(judge)
        body = strict_json_loads(substituted["request_body"]["data"])
        body["response_format"]["json_schema"]["schema"] = {"type": "array"}
        substituted["request_body"] = encode_bytes(canonical_json_bytes(body)).to_dict()
        with self.assertRaisesRegex(JournalCorrupt, "embedded response schema"):
            reduce_events(
                chain_events(
                    created,
                    with_unit(judge, [("model_call_intended", substituted)]),
                )
            )

    def test_rejects_initial_nonprimary_and_retry_after_valid_target(self) -> None:
        nonprimary = intended_payload(self.fixture)
        nonprimary["route_kind"] = "exact_retry"
        nonprimary["physical_attempt_id"] = entity_id(
            "pa", [self.fixture["action_id"], "exact_retry", 0]
        )
        with self.assertRaisesRegex(JournalCorrupt, "initial route kind"):
            self._reduce_intent(nonprimary)

        first = attempt_id(self.fixture, 0)
        records = with_unit(
            self.fixture,
            [
                ("model_call_intended", intended_payload(self.fixture)),
                ("model_dispatch_started", dispatch_payload(self.fixture)),
                ("model_call_succeeded", success_payload(self.fixture, text="valid")),
                (
                    "model_call_intended",
                    intended_payload(
                        self.fixture, ordinal=1, parent_attempt_id=first
                    ),
                ),
            ],
        )
        with self.assertRaisesRegex(JournalCorrupt, "prior terminal trigger"):
            reduce_events(chain_events(self.created, records))

    def test_rejects_retry_after_filtered_target(self) -> None:
        first = attempt_id(self.fixture, 0)
        filtered = failure_payload(self.fixture)
        filtered["error_kind"] = "filtered"
        target_records = with_unit(
            self.fixture,
            [
                ("model_call_intended", intended_payload(self.fixture)),
                ("model_dispatch_started", dispatch_payload(self.fixture)),
                ("model_call_failed", filtered),
                (
                    "model_call_intended",
                    intended_payload(
                        self.fixture, ordinal=1, parent_attempt_id=first
                    ),
                ),
            ],
        )
        with self.assertRaisesRegex(JournalCorrupt, "prior terminal trigger"):
            reduce_events(chain_events(self.created, target_records))

class ManifestAuthenticationTests(unittest.TestCase):
    def test_library_links_exactly_cover_selected_snapshots(self) -> None:
        value = foundation_manifest()
        checkpoint = {
            "source_run_id": entity_id("run", ["library-source"]),
            "sequence": 7,
            "event_hash": digest_json(["library-source", "event"]),
            "source_path": "/validated/source/run",
        }
        seed_id = entity_id("sd", ["library-seed"])
        compatibility_key = digest_json(["library-seed", "compatibility"])
        value["seed_snapshot"] = [
            {
                "seed_id": seed_id,
                "compatibility_key": compatibility_key,
                "source_checkpoint": checkpoint,
            }
        ]
        value["library_source_checkpoints"] = [
            {
                "kind": "seed",
                "subject_id": seed_id,
                "compatibility_key": compatibility_key,
                "source": checkpoint,
            }
        ]
        _validate_request_manifest(value, None)

        missing = deepcopy(value)
        missing["library_source_checkpoints"] = []
        with self.assertRaisesRegex(JournalCorrupt, "exactly cover"):
            _validate_request_manifest(missing, None)

        duplicate = deepcopy(value)
        duplicate["seed_snapshot"].append(deepcopy(duplicate["seed_snapshot"][0]))
        with self.assertRaisesRegex(JournalCorrupt, "repeats one seed"):
            _validate_request_manifest(duplicate, None)

    def test_manifest_owns_frozen_policy_and_authenticates_taxonomy(self) -> None:
        policy = foundation_manifest()
        policy["decision_constants"]["target_attempt_cap"] += 1
        _validate_request_manifest(policy, None)

        invalid_policy = foundation_manifest()
        invalid_policy["decision_constants"]["target_attempt_cap"] = 0
        with self.assertRaisesRegex(JournalCorrupt, "decision constant"):
            _validate_request_manifest(invalid_policy, None)

        taxonomy = foundation_manifest()
        taxonomy["descriptor_taxonomy"]["digest"] = digest_json(["forged"])
        with self.assertRaisesRegex(JournalCorrupt, "taxonomy digest"):
            _validate_request_manifest(taxonomy, None)


class PureRecoveryTransitionTests(unittest.TestCase):
    def _selector_action(self, terminal: dict) -> tuple[list[dict], LogicalActionView, str]:
        schema_digest = digest_json({"type": "object"})
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "request"},
        ]
        first_intent = {
            "sequence": 1,
            "type": "model_call_intended",
            "payload": {
                "physical_attempt_id": "pa-primary",
                "role": "selector",
                "role_route_key": "primary-route",
                "route_kind": "primary",
                "response_schema_digest": schema_digest,
                "response_mode": "json_schema",
                "messages": messages,
                "semantic_request_hash": "primary-semantic",
            },
        }
        first_terminal = {
            "sequence": 2,
            "type": terminal["type"],
            "payload": {"physical_attempt_id": "pa-primary", **terminal["payload"]},
        }
        fallback_intent = {
            "sequence": 3,
            "type": "model_call_intended",
            "payload": {
                "physical_attempt_id": "pa-fallback",
                "role": "selector",
                "role_route_key": "fallback-route",
                "route_kind": "fallback",
                "response_schema_digest": schema_digest,
                "response_mode": "json_schema",
                "messages": messages,
                "semantic_request_hash": "fallback-semantic",
            },
        }
        slot = {
            "role": "selector",
            "routes": [
                {
                    "route_key": "primary-route",
                    "route_label": "primary",
                    "maximum_attempts": 2,
                },
                {
                    "route_key": "fallback-route",
                    "route_label": "fallback",
                    "maximum_attempts": 2,
                },
            ],
        }
        action = LogicalActionView(
            action_id="selector-action",
            unit_id="unit",
            slot=slot,
            attempts=(
                AttemptView("pa-primary", first_intent, None, first_terminal),
                AttemptView("pa-fallback", fallback_intent, None, None),
            ),
        )
        return [first_intent, first_terminal, fallback_intent], action, schema_digest

    def test_fallback_requires_primary_exhaustion_and_never_follows_valid_result(self) -> None:
        early_events, early_action, schema_digest = self._selector_action(
            {
                "type": "model_call_failed",
                "payload": {"error_kind": "timeout"},
            }
        )
        with self.assertRaisesRegex(JournalCorrupt, "before primary route exhaustion"):
            validate_recovery_transitions(
                early_events,
                {early_action.action_id: early_action},
                {early_action.action_id: "map_exploration"},
                {schema_digest: {"type": "object"}},
            )

        valid_events, valid_action, schema_digest = self._selector_action(
            {
                "type": "model_call_succeeded",
                "payload": {"finish_status": "complete", "text": "{}"},
            }
        )
        with self.assertRaisesRegex(JournalCorrupt, "before primary route exhaustion"):
            validate_recovery_transitions(
                valid_events,
                {valid_action.action_id: valid_action},
                {valid_action.action_id: "map_exploration"},
                {schema_digest: {"type": "object"}},
            )

        exhausted_events, exhausted_action, schema_digest = self._selector_action(
            {
                "type": "model_call_failed",
                "payload": {"error_kind": "filtered"},
            }
        )
        validate_recovery_transitions(
            exhausted_events,
            {exhausted_action.action_id: exhausted_action},
            {exhausted_action.action_id: "map_exploration"},
            {schema_digest: {"type": "object"}},
        )

    def test_judge_quote_mismatch_is_a_replayable_structural_repair_trigger(self) -> None:
        schema = judge_schema("current reply")
        schema_digest = digest_json(schema)
        initial_messages = [{"role": "user", "content": "assess"}]
        repair_messages = [
            *initial_messages,
            {"role": "user", "content": "repair the evidence quote"},
        ]
        invalid_text = (
            '{"score":50,"verdict":"ambiguous","application":"abstract","justification":"The position is ambiguous.","procedure":{"premise":{"status":"adopted","evidence":[{"evidence_id":"e9","polarity":"supports"}]},"inference":{"status":"endorsed","evidence":[{"evidence_id":"e9","polarity":"supports"}]},"commitment":{"status":"absent","evidence":[]},"reservation":{"status":"none","evidence":[]}}}'
        )
        first_intent = {
            "sequence": 1,
            "type": "model_call_intended",
            "payload": {
                "physical_attempt_id": "pa-judge-primary",
                "role": "judge",
                "role_route_key": "judge-route",
                "route_kind": "primary",
                "response_schema_digest": schema_digest,
                "response_mode": "json_schema",
                "messages": initial_messages,
                "semantic_request_hash": "judge-primary-semantic",
            },
        }
        first_terminal = {
            "sequence": 2,
            "type": "model_call_succeeded",
            "payload": {
                "physical_attempt_id": "pa-judge-primary",
                "finish_status": "complete",
                "text": invalid_text,
            },
        }
        repair_intent = {
            "sequence": 3,
            "type": "model_call_intended",
            "payload": {
                "physical_attempt_id": "pa-judge-repair",
                "role": "judge",
                "role_route_key": "judge-route",
                "route_kind": "schema_repair",
                "response_schema_digest": schema_digest,
                "response_mode": "json_schema",
                "messages": repair_messages,
                "semantic_request_hash": "judge-repair-semantic",
            },
        }
        action = LogicalActionView(
            action_id="judge-action",
            unit_id="unit",
            slot={
                "role": "judge",
                "routes": [
                    {
                        "route_key": "judge-route",
                        "route_label": "primary",
                        "maximum_attempts": 2,
                    }
                ],
            },
            attempts=(
                AttemptView(
                    "pa-judge-primary", first_intent, None, first_terminal
                ),
                AttemptView("pa-judge-repair", repair_intent, None, None),
            ),
        )
        validate_recovery_transitions(
            [first_intent, first_terminal, repair_intent],
            {action.action_id: action},
            {action.action_id: "bootstrap_screening"},
            {schema_digest: schema},
            {action.action_id: "the actual target reply"},
        )

    def test_contextual_schema_repair_supersedes_only_the_rejected_result(self) -> None:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"mode": {"type": "string"}},
            "required": ["mode"],
        }
        schema_digest = digest_json(schema)
        initial_messages = [{"role": "user", "content": "strategy request"}]
        repair_messages = [
            *initial_messages,
            {"role": "user", "content": "repair the contextual structure"},
        ]
        first_intent = {
            "sequence": 1,
            "type": "model_call_intended",
            "payload": {
                "physical_attempt_id": "pa-context-primary",
                "parent_attempt_id": None,
                "role": "perturbator",
                "role_route_key": "perturbator-route",
                "route_kind": "primary",
                "response_schema_digest": schema_digest,
                "response_mode": "json_schema",
                "messages": initial_messages,
                "semantic_request_hash": "context-primary-semantic",
            },
        }
        first_terminal = {
            "sequence": 2,
            "type": "model_call_succeeded",
            "payload": {
                "physical_attempt_id": "pa-context-primary",
                "finish_status": "complete",
                "text": '{"mode":"ordinary"}',
            },
        }
        repair_intent = {
            "sequence": 3,
            "type": "model_call_intended",
            "payload": {
                "physical_attempt_id": "pa-context-repair",
                "parent_attempt_id": "pa-context-primary",
                "role": "perturbator",
                "role_route_key": "perturbator-route",
                "route_kind": "schema_repair",
                "response_schema_digest": schema_digest,
                "response_mode": "json_schema",
                "messages": repair_messages,
                "semantic_request_hash": "context-repair-semantic",
            },
        }
        repair_terminal = {
            "sequence": 4,
            "type": "model_call_succeeded",
            "payload": {
                "physical_attempt_id": "pa-context-repair",
                "finish_status": "complete",
                "text": '{"mode":"strategy"}',
            },
        }
        action = LogicalActionView(
            action_id="contextual-perturbator-action",
            unit_id="unit",
            slot={
                "role": "perturbator",
                "routes": [
                    {
                        "route_key": "perturbator-route",
                        "route_label": "primary",
                        "maximum_attempts": 2,
                    }
                ],
            },
            attempts=(
                AttemptView(
                    "pa-context-primary", first_intent, None, first_terminal
                ),
                AttemptView(
                    "pa-context-repair", repair_intent, None, repair_terminal
                ),
            ),
        )

        validate_recovery_transitions(
            [first_intent, first_terminal, repair_intent, repair_terminal],
            {action.action_id: action},
            {action.action_id: "portfolio_round"},
            {schema_digest: schema},
        )

        def strategy_only(value: object) -> object:
            if not isinstance(value, dict) or value.get("mode") != "strategy":
                raise DomainError("context requires strategy mode")
            return value

        self.assertEqual(
            _parsed_structured_action(
                action,
                {schema_digest: schema},
                strategy_only,
                label="contextual proposal",
            ),
            {"mode": "strategy"},
        )
        with self.assertRaisesRegex(
            JournalCorrupt, "several semantically usable results"
        ):
            _parsed_structured_action(
                action,
                {schema_digest: schema},
                lambda value: value,
                label="contextual proposal",
            )

    def test_contextual_rejection_can_exhaust_primary_into_fallback(self) -> None:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"mode": {"type": "string"}},
            "required": ["mode"],
        }
        schema_digest = digest_json(schema)
        messages = [{"role": "user", "content": "strategy request"}]
        first_intent = {
            "sequence": 1,
            "type": "model_call_intended",
            "payload": {
                "physical_attempt_id": "pa-fallback-primary",
                "parent_attempt_id": None,
                "role": "perturbator",
                "role_route_key": "perturbator-primary",
                "route_kind": "primary",
                "response_schema_digest": schema_digest,
                "response_mode": "json_schema",
                "messages": messages,
                "semantic_request_hash": "primary-semantic",
            },
        }
        first_terminal = {
            "sequence": 2,
            "type": "model_call_outcome_unknown",
            "payload": {
                "physical_attempt_id": "pa-fallback-primary",
                "error_kind": "malformed_response",
            },
        }
        retry_intent = {
            "sequence": 3,
            "type": "model_call_intended",
            "payload": {
                "physical_attempt_id": "pa-fallback-retry",
                "parent_attempt_id": "pa-fallback-primary",
                "role": "perturbator",
                "role_route_key": "perturbator-primary",
                "route_kind": "exact_retry",
                "response_schema_digest": schema_digest,
                "response_mode": "json_schema",
                "messages": messages,
                "semantic_request_hash": "primary-semantic",
            },
        }
        retry_terminal = {
            "sequence": 4,
            "type": "model_call_succeeded",
            "payload": {
                "physical_attempt_id": "pa-fallback-retry",
                "finish_status": "complete",
                "text": '{"mode":"ordinary"}',
            },
        }
        fallback_intent = {
            "sequence": 5,
            "type": "model_call_intended",
            "payload": {
                "physical_attempt_id": "pa-fallback-route",
                "parent_attempt_id": "pa-fallback-retry",
                "role": "perturbator",
                "role_route_key": "perturbator-fallback",
                "route_kind": "fallback",
                "response_schema_digest": schema_digest,
                "response_mode": "json_schema",
                "messages": messages,
                "semantic_request_hash": "fallback-semantic",
            },
        }
        fallback_terminal = {
            "sequence": 6,
            "type": "model_call_succeeded",
            "payload": {
                "physical_attempt_id": "pa-fallback-route",
                "finish_status": "complete",
                "text": '{"mode":"strategy"}',
            },
        }
        action = LogicalActionView(
            action_id="contextual-fallback-action",
            unit_id="unit",
            slot={
                "role": "perturbator",
                "routes": [
                    {
                        "route_key": "perturbator-primary",
                        "route_label": "primary",
                        "maximum_attempts": 2,
                    },
                    {
                        "route_key": "perturbator-fallback",
                        "route_label": "fallback",
                        "maximum_attempts": 2,
                    },
                ],
            },
            attempts=(
                AttemptView(
                    "pa-fallback-primary", first_intent, None, first_terminal
                ),
                AttemptView(
                    "pa-fallback-retry", retry_intent, None, retry_terminal
                ),
                AttemptView(
                    "pa-fallback-route",
                    fallback_intent,
                    None,
                    fallback_terminal,
                ),
            ),
        )
        events = [
            first_intent,
            first_terminal,
            retry_intent,
            retry_terminal,
            fallback_intent,
            fallback_terminal,
        ]
        validate_recovery_transitions(
            events,
            {action.action_id: action},
            {action.action_id: "portfolio_round"},
            {schema_digest: schema},
        )

        def strategy_only(value: object) -> object:
            if not isinstance(value, dict) or value.get("mode") != "strategy":
                raise DomainError("context requires strategy mode")
            return value

        self.assertEqual(
            _parsed_structured_action(
                action,
                {schema_digest: schema},
                strategy_only,
                label="contextual proposal",
            ),
            {"mode": "strategy"},
        )
        with self.assertRaisesRegex(
            JournalCorrupt, "several semantically usable results"
        ):
            _parsed_structured_action(
                action,
                {schema_digest: schema},
                lambda value: value,
                label="contextual proposal",
            )


if __name__ == "__main__":
    unittest.main()
