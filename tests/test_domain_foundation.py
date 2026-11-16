from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import unittest

from quasilisk.domain import (
    ActionPurpose,
    ChatMessage,
    ChatRequest,
    DECISION_CONSTANTS,
    DomainError,
    DuplicateKeyError,
    STRATEGY_OPERATIONS,
    Role,
    RunTerminalReason,
    TERMINAL_EXIT_CODES,
    canonical_json,
    canonical_json_bytes,
    digest_json,
    entity_id,
    normalize_decimal,
    normalize_line_endings,
    require_decimal_string,
    require_digest,
    require_entity_id,
    scientific_key,
    strict_json_loads,
)


class StrictJsonAndCanonicalizationTests(unittest.TestCase):
    def test_strict_json_preserves_exact_decimals_and_rejects_unsafe_inputs(self) -> None:
        parsed = strict_json_loads(b'{"integer":2,"number":1.2500}')
        self.assertEqual(parsed["integer"], 2)
        self.assertIsInstance(parsed["number"], Decimal)
        self.assertEqual(parsed["number"], Decimal("1.2500"))

        with self.assertRaises(DuplicateKeyError):
            strict_json_loads('{"same":1,"same":2}')
        for raw in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(raw=raw), self.assertRaises(DomainError):
                strict_json_loads(raw)
        with self.assertRaises(DomainError):
            strict_json_loads(b'\xff')

    def test_canonical_json_is_compact_sorted_utf8_and_float_free(self) -> None:
        value = {"z": [True, None], "é": "snowman ☃", "a": 2}
        self.assertEqual(
            canonical_json_bytes(value),
            '{"a":2,"z":[true,null],"é":"snowman ☃"}'.encode(),
        )
        self.assertEqual(canonical_json(value), canonical_json_bytes(value).decode())
        with self.assertRaises(DomainError):
            canonical_json_bytes({"binary_float": 0.1})
        with self.assertRaises(DomainError):
            canonical_json_bytes({1: "non-string key"})
        with self.assertRaises(DomainError):
            canonical_json_bytes("\ud800")

    def test_decimal_normalization_has_one_spelling(self) -> None:
        cases = {
            Decimal("0.000"): "0",
            Decimal("-0"): "0",
            Decimal("12.3400"): "12.34",
            Decimal("1E+3"): "1000",
            "0.0100": "0.01",
            7: "7",
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(normalize_decimal(value), expected)
        self.assertEqual(canonical_json_bytes(Decimal("1.20")), b'"1.2"')
        self.assertEqual(require_decimal_string("1.2"), Decimal("1.2"))
        for invalid in ("1.20", "1e2", "-0", True, "NaN"):
            with self.subTest(invalid=invalid), self.assertRaises(DomainError):
                require_decimal_string(invalid)

    def test_line_ending_normalization_does_not_change_other_text_identity(self) -> None:
        self.assertEqual(normalize_line_endings("a\r\nb\rc\n"), "a\nb\nc\n")
        composed = "é"
        decomposed = "e\u0301"
        self.assertNotEqual(digest_json(composed), digest_json(decomposed))


class IdentityTests(unittest.TestCase):
    def test_experiment_policy_constants_are_immutable(self) -> None:
        with self.assertRaises(TypeError):
            DECISION_CONSTANTS["screening_high_threshold"] = 1  # type: ignore[index]
        with self.assertRaises(TypeError):
            TERMINAL_EXIT_CODES[RunTerminalReason.COMPLETED_PLAN] = 5  # type: ignore[index]
        with self.assertRaises(TypeError):
            STRATEGY_OPERATIONS["complete_opening"] = "other"  # type: ignore[index]

    def test_digests_and_entity_ids_are_full_lowercase_sha256_values(self) -> None:
        digest = digest_json({"b": 2, "a": 1})
        self.assertRegex(digest, r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(digest, digest_json({"a": 1, "b": 2}))
        self.assertEqual(require_digest(digest), digest)

        identifier = entity_id("pc", ["candidate", 1])
        self.assertRegex(identifier, r"^pc-[0-9a-f]{64}$")
        self.assertEqual(require_entity_id(identifier), identifier)
        for invalid in (digest[:-1], digest.upper(), "sha256:" + "g" * 64):
            with self.subTest(invalid=invalid), self.assertRaises(DomainError):
                require_digest(invalid)
        for invalid in (identifier[:-1], identifier.upper(), "xx-" + "0" * 64):
            with self.subTest(invalid=invalid), self.assertRaises(DomainError):
                require_entity_id(invalid)
        with self.assertRaises(DomainError):
            entity_id("unknown", [])

    def test_scientific_keys_are_deterministic_canonical_array_hashes(self) -> None:
        material = ["scope", {"b": 2, "a": 1}, 3]
        self.assertEqual(scientific_key(material), digest_json(material))
        self.assertEqual(
            scientific_key(material),
            scientific_key(["scope", {"a": 1, "b": 2}, 3]),
        )
        self.assertNotEqual(scientific_key(material), scientific_key([*material, 4]))


class SemanticRequestIdentityTests(unittest.TestCase):
    def make_request(self) -> ChatRequest:
        return ChatRequest(
            logical_action_id=entity_id("la", ["action", 1]),
            logical_action_scientific_key=digest_json(["action-key", 1]),
            physical_attempt_id=entity_id("pa", ["attempt", 1]),
            attempt_ordinal=0,
            role=Role.TARGET,
            purpose=ActionPurpose.TARGET_SAMPLE,
            route_kind="primary",
            role_route_key=digest_json(["role-route", 1]),
            requested_provider_route_key=digest_json(["provider-route", 1]),
            messages=(ChatMessage("system", "policy"), ChatMessage("user", "probe")),
            requested_model="provider/target",
            generation_parameters={"temperature": "0.2", "max_output_tokens": 64},
            response_mode="text",
            response_schema=None,
            response_schema_digest=None,
            timeout_seconds=30,
            idempotency_key="idem-a",
        )

    def test_semantic_hash_excludes_only_operational_and_run_scoped_fields(self) -> None:
        request = self.make_request()
        expected = digest_json(
            [
                "target",
                "target_sample",
                [message.to_dict() for message in request.messages],
                request.role_route_key,
                request.requested_provider_route_key,
                request.requested_model,
                dict(request.generation_parameters),
                "text",
                None,
            ]
        )
        self.assertEqual(request.semantic_request_hash, expected)

        operational_variants = (
            replace(request, logical_action_id=entity_id("la", ["action", 2])),
            replace(
                request,
                logical_action_scientific_key=digest_json(["action-key", 2]),
            ),
            replace(request, physical_attempt_id=entity_id("pa", ["attempt", 2])),
            replace(request, attempt_ordinal=1),
            replace(request, route_kind="exact_retry"),
            replace(request, timeout_seconds=90),
            replace(request, idempotency_key="idem-b"),
        )
        for variant in operational_variants:
            with self.subTest(variant=variant):
                self.assertEqual(variant.semantic_request_hash, expected)

    def test_every_declared_semantic_field_changes_the_hash(self) -> None:
        request = self.make_request()
        variants = {
            "role": replace(request, role=Role.SELECTOR),
            "purpose": replace(request, purpose=ActionPurpose.PROBE_REVIEW),
            "messages": replace(request, messages=(ChatMessage("user", "different"),)),
            "role_route_key": replace(
                request, role_route_key=digest_json(["role-route", 2])
            ),
            "provider_route_key": replace(
                request,
                requested_provider_route_key=digest_json(["provider-route", 2]),
            ),
            "model": replace(request, requested_model="provider/other"),
            "generation": replace(
                request,
                generation_parameters={"temperature": "0.3", "max_output_tokens": 64},
            ),
            "response_contract": replace(
                request,
                response_mode="json_object",
                response_schema={"type": "object"},
                response_schema_digest=digest_json({"type": "object"}),
            ),
        }
        for field, variant in variants.items():
            with self.subTest(field=field):
                self.assertNotEqual(
                    variant.semantic_request_hash, request.semantic_request_hash
                )

    def test_request_takes_immutable_semantic_input_snapshots(self) -> None:
        request = self.make_request()
        with self.assertRaises(TypeError):
            request.generation_parameters["temperature"] = "0.9"  # type: ignore[index]

        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        schema_digest = digest_json(schema)
        structured = replace(
            request,
            response_mode="json_schema",
            response_schema=schema,
            response_schema_digest=schema_digest,
        )
        schema["properties"]["answer"]["type"] = "integer"
        self.assertEqual(digest_json(structured.response_schema), schema_digest)
        with self.assertRaises(TypeError):
            structured.response_schema["properties"]["answer"]["type"] = "integer"  # type: ignore[index]

    def test_request_rejects_transport_inputs_outside_the_closed_contract(self) -> None:
        request = self.make_request()
        with self.assertRaises(DomainError):
            replace(
                request,
                generation_parameters={
                    "temperature": "0.2",
                    "max_output_tokens": 64,
                    "provider_extension": True,
                },
            )
        with self.assertRaises(DomainError):
            replace(
                request,
                response_mode="text",
                response_schema={"type": "object"},
            )


if __name__ == "__main__":
    unittest.main()
