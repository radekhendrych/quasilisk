from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import io
import os
from pathlib import Path
import signal
import tempfile
import time
import unittest
from unittest.mock import patch
from urllib import error as urlerror

from quasilisk.config import ProviderConfig
from quasilisk.domain import (
    ActionPurpose,
    ChatMessage,
    ChatRequest,
    FinishStatus,
    ProviderErrorKind,
    ProviderFailure,
    ProviderSuccess,
    Role,
    canonical_json_bytes,
    digest_json,
    entity_id,
    sha256_bytes,
    strict_json_loads,
)
from quasilisk.provider import (
    ProviderProtocolError,
    OpenAICompatibleProvider,
    ScriptedFixtureMissing,
    ScriptedProvider,
    decode_bytes,
    encode_bytes,
    render_request_body,
    requested_provider_route_key,
    scripted_lookup_key,
)


def make_request(*, label: str = "one", ordinal: int = 0) -> ChatRequest:
    return ChatRequest(
        logical_action_id=entity_id("la", ["provider-test", label, "action"]),
        logical_action_scientific_key=digest_json(
            ["provider-test", label, "action-key"]
        ),
        physical_attempt_id=entity_id(
            "pa", ["provider-test", label, "attempt", ordinal]
        ),
        attempt_ordinal=ordinal,
        role=Role.TARGET,
        purpose=ActionPurpose.TARGET_SAMPLE,
        route_kind="primary" if ordinal == 0 else "exact_retry",
        role_route_key=digest_json(["provider-test", label, "role-route"]),
        requested_provider_route_key=requested_provider_route_key(
            "scripted", Role.TARGET, "primary"
        ),
        messages=(ChatMessage("system", "system"), ChatMessage("user", label)),
        requested_model="models/target",
        generation_parameters={"temperature": "0.2", "max_output_tokens": 64},
        response_mode="text",
        response_schema=None,
        response_schema_digest=None,
        timeout_seconds=1,
        idempotency_key="idempotency-value",
    )


def success_record(key: str, *, text: str = "success") -> dict:
    return {
        "key": key,
        "outcome": "success",
        "raw_body": {"encoding": "utf8", "data": '{"ok":true}'},
        "text": text,
        "provider_request_id": "request-id",
        "effective_model_id": "effective/model",
        "finish_status": "complete",
        "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
        "latency_ms": 4,
        "cost": "0.01",
        "metadata": {"http_status": 200, "content_type": "application/json"},
    }


def failure_record(key: str) -> dict:
    return {
        "key": key,
        "outcome": "failure",
        "error_kind": "rate_limit",
        "delivery_state": "known_no_output",
        "detail": "provider_rate_limit",
        "provider_request_id": None,
        "raw_body": None,
        "http_status": 429,
        "retry_after_seconds": "1.5",
    }


class ProviderKeyingTests(unittest.TestCase):
    def test_provider_route_key_has_exact_declared_inputs(self) -> None:
        key = requested_provider_route_key("scripted", Role.TARGET, "primary")
        self.assertEqual(key, digest_json(["scripted", "target", "primary"]))
        self.assertNotEqual(
            key, requested_provider_route_key("scripted", Role.TARGET, "fallback")
        )
        self.assertNotEqual(
            key,
            requested_provider_route_key("openai_compatible", Role.TARGET, "primary"),
        )
        with self.assertRaises(ProviderProtocolError):
            requested_provider_route_key("unknown", Role.TARGET, "primary")
        with self.assertRaises(ProviderProtocolError):
            requested_provider_route_key("scripted", "not-a-role", "primary")

    def test_script_lookup_key_excludes_run_scoped_transport_fields(self) -> None:
        request = make_request()
        expected = digest_json(
            [
                request.logical_action_scientific_key,
                "target",
                0,
                "primary",
                request.semantic_request_hash,
            ]
        )
        self.assertEqual(scripted_lookup_key(request), expected)
        variants = (
            replace(request, logical_action_id=entity_id("la", ["other-run"])),
            replace(request, physical_attempt_id=entity_id("pa", ["other-run"])),
            replace(request, timeout_seconds=2),
            replace(request, idempotency_key="other"),
        )
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertEqual(scripted_lookup_key(variant), expected)

        scientific_changes = (
            replace(
                request,
                logical_action_scientific_key=digest_json(["different-action-key"]),
            ),
            replace(request, attempt_ordinal=1),
            replace(request, route_kind="exact_retry"),
            replace(request, messages=(ChatMessage("user", "different"),)),
        )
        for variant in scientific_changes:
            with self.subTest(variant=variant):
                self.assertNotEqual(scripted_lookup_key(variant), expected)

    def test_request_body_renders_exact_decimal_as_json_number(self) -> None:
        body = render_request_body(make_request())
        self.assertIn(b'"temperature":0.2', body)
        self.assertNotIn(b'"temperature":"0.2"', body)
        parsed = strict_json_loads(body)
        self.assertEqual(str(parsed["temperature"]), "0.2")
        self.assertEqual(parsed["max_tokens"], 64)

    def test_request_renderer_has_no_prompt_marker_collision_and_keeps_schema_numbers_numeric(self) -> None:
        marker = "__QUASILISK_EXACT_TEMPERATURE__"
        request = replace(
            make_request(),
            messages=(ChatMessage("user", marker),),
        )
        parsed = strict_json_loads(render_request_body(request))
        self.assertEqual(parsed["messages"][0]["content"], marker)
        self.assertEqual(parsed["temperature"], Decimal("0.2"))

        schema = {"type": "number", "multipleOf": Decimal("0.1")}
        structured = replace(
            request,
            response_mode="json_schema",
            response_schema=schema,
            response_schema_digest=digest_json(schema),
        )
        structured_body = strict_json_loads(render_request_body(structured))
        self.assertEqual(
            structured_body["response_format"]["json_schema"]["schema"][
                "multipleOf"
            ],
            Decimal("0.1"),
        )


class ScriptedProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "script.jsonl"

    def write_records(self, *records: dict) -> bytes:
        raw = b"".join(canonical_json_bytes(record) + b"\n" for record in records)
        self.path.write_bytes(raw)
        return raw

    def test_records_are_stateless_order_independent_semantic_lookups(self) -> None:
        first = make_request(label="first")
        second = make_request(label="second")
        raw = self.write_records(
            failure_record(scripted_lookup_key(second)),
            success_record(scripted_lookup_key(first), text="first output"),
        )
        provider = ScriptedProvider(self.path)
        self.assertEqual(provider.content_digest, sha256_bytes(raw))

        first_result = provider.complete(first)
        self.assertIsInstance(first_result, ProviderSuccess)
        assert isinstance(first_result, ProviderSuccess)
        self.assertEqual(first_result.text, "first output")
        self.assertEqual(first_result.finish_status, FinishStatus.COMPLETE)

        second_result = provider.complete(second)
        self.assertIsInstance(second_result, ProviderFailure)
        assert isinstance(second_result, ProviderFailure)
        self.assertEqual(second_result.error_kind, ProviderErrorKind.RATE_LIMIT)
        self.assertEqual(second_result.retry_after_seconds, "1.5")

        reordered_path = Path(self.temporary.name) / "reordered.jsonl"
        reordered_path.write_bytes(
            canonical_json_bytes(success_record(scripted_lookup_key(first)))
            + b"\n"
            + canonical_json_bytes(failure_record(scripted_lookup_key(second)))
            + b"\n"
        )
        reordered = ScriptedProvider(reordered_path)
        self.assertIsInstance(reordered.complete(first), ProviderSuccess)
        self.assertIsInstance(reordered.complete(second), ProviderFailure)

    def test_missing_key_fails_loudly_instead_of_fabricating_failure(self) -> None:
        present = make_request(label="present")
        self.write_records(success_record(scripted_lookup_key(present)))
        provider = ScriptedProvider(self.path)
        with self.assertRaisesRegex(
            ScriptedFixtureMissing, r"^scripted_fixture_missing: sha256:"
        ):
            provider.complete(make_request(label="absent"))

    def test_success_and_failure_records_reject_unknown_or_wrong_variant_fields(self) -> None:
        key = scripted_lookup_key(make_request())
        variants = []
        unknown = success_record(key)
        unknown["unknown"] = True
        variants.append(unknown)
        wrong_variant = success_record(key)
        wrong_variant["error_kind"] = "timeout"
        variants.append(wrong_variant)
        missing = failure_record(key)
        del missing["detail"]
        variants.append(missing)
        for index, record in enumerate(variants):
            with self.subTest(index=index):
                self.write_records(record)
                with self.assertRaises(ProviderProtocolError):
                    ScriptedProvider(self.path)

    def test_record_value_invariants_are_validated(self) -> None:
        key = scripted_lookup_key(make_request())
        variants = []
        token_mismatch = success_record(key)
        token_mismatch["usage"]["total_tokens"] = 4
        variants.append(token_mismatch)
        noncanonical_cost = success_record(key)
        noncanonical_cost["cost"] = "1.00"
        variants.append(noncanonical_cost)
        invalid_base64 = success_record(key)
        invalid_base64["raw_body"] = {"encoding": "base64", "data": "%%%"}
        variants.append(invalid_base64)
        bad_retry_after = failure_record(key)
        bad_retry_after["retry_after_seconds"] = "01.5"
        variants.append(bad_retry_after)
        bool_token = success_record(key)
        bool_token["usage"]["input_tokens"] = True
        variants.append(bool_token)
        for index, record in enumerate(variants):
            with self.subTest(index=index):
                self.write_records(record)
                with self.assertRaises(ProviderProtocolError):
                    ScriptedProvider(self.path)

    def test_duplicate_fixture_keys_json_keys_utf8_and_blank_lines_are_rejected(self) -> None:
        request = make_request()
        record = success_record(scripted_lookup_key(request))
        self.write_records(record, record)
        with self.assertRaises(ProviderProtocolError):
            ScriptedProvider(self.path)

        self.path.write_bytes(b'{"key":"x","key":"y"}\n')
        with self.assertRaises(ProviderProtocolError):
            ScriptedProvider(self.path)

        self.path.write_bytes(b"\xff\n")
        with self.assertRaises(ProviderProtocolError):
            ScriptedProvider(self.path)

        self.path.write_bytes(canonical_json_bytes(record) + b"\n\n")
        with self.assertRaises(ProviderProtocolError):
            ScriptedProvider(self.path)

    def test_byte_envelopes_use_one_canonical_encoding(self) -> None:
        utf8 = "snowman ☃".encode()
        self.assertEqual(encode_bytes(utf8).encoding, "utf8")
        self.assertEqual(decode_bytes(encode_bytes(utf8).to_dict()), utf8)
        binary = b"\xff\x00"
        encoded = encode_bytes(binary)
        self.assertEqual(encoded.encoding, "base64")
        self.assertEqual(decode_bytes(encoded.to_dict()), binary)
        with self.assertRaises(ProviderProtocolError):
            decode_bytes({"encoding": "base64", "data": "not base64!"})


class NetworkProviderOutcomeTests(unittest.TestCase):
    def make_provider(self) -> OpenAICompatibleProvider:
        return OpenAICompatibleProvider(
            ProviderConfig(
                adapter="openai_compatible",
                base_url="https://provider.example/v1",
                api_key_env="QUASILISK_TEST_KEY",
                timeout_seconds=1,
            )
        )

    def complete_with_http_error(self, status: int, body: bytes) -> ProviderFailure:
        error = urlerror.HTTPError(
            "https://provider.example/v1/chat/completions",
            status,
            "provider error",
            {},
            io.BytesIO(body),
        )
        with patch.dict(os.environ, {"QUASILISK_TEST_KEY": "secret"}), patch(
            "quasilisk.provider.urlrequest.urlopen", side_effect=error
        ):
            result = self.make_provider().complete(make_request())
        self.assertIsInstance(result, ProviderFailure)
        assert isinstance(result, ProviderFailure)
        return result

    def test_typed_content_policy_errors_are_filtered(self) -> None:
        cases = (
            (400, {"error": {"code": "content_filter", "message": "blocked"}}),
            (
                422,
                {
                    "error": {
                        "type": "content_policy_violation",
                        "message": "blocked",
                    }
                },
            ),
            (
                403,
                {
                    "error": {
                        "code": "bad_argument",
                        "inner_error": {"code": "ResponsibleAIPolicyViolation"},
                    }
                },
            ),
        )
        for status, body in cases:
            with self.subTest(status=status, body=body):
                result = self.complete_with_http_error(
                    status, canonical_json_bytes(body)
                )
                self.assertEqual(result.error_kind, ProviderErrorKind.FILTERED)
                self.assertEqual(result.delivery_state, "known_no_output")

    def test_filter_words_without_allowlisted_type_remain_invalid_request(self) -> None:
        result = self.complete_with_http_error(
            400,
            canonical_json_bytes(
                {
                    "error": {
                        "code": "invalid_request_error",
                        "message": "request was filtered by a content policy",
                    }
                }
            ),
        )
        self.assertEqual(result.error_kind, ProviderErrorKind.INVALID_REQUEST)

    def test_untyped_forbidden_response_remains_authentication_failure(self) -> None:
        result = self.complete_with_http_error(
            403, canonical_json_bytes({"error": {"message": "forbidden"}})
        )
        self.assertEqual(result.error_kind, ProviderErrorKind.AUTHENTICATION)

    def test_only_typed_discriminators_select_format_and_context_failures(self) -> None:
        typed_cases = (
            ("unsupported_response_format", ProviderErrorKind.UNSUPPORTED_RESPONSE_FORMAT),
            ("context_length_exceeded", ProviderErrorKind.CONTEXT_LIMIT),
        )
        for code, expected in typed_cases:
            with self.subTest(code=code):
                result = self.complete_with_http_error(
                    400,
                    canonical_json_bytes({"error": {"code": code}}),
                )
                self.assertEqual(result.error_kind, expected)

        prose_only = self.complete_with_http_error(
            400,
            canonical_json_bytes(
                {
                    "error": {
                        "code": "invalid_request_error",
                        "message": "response_format unsupported; context length limit",
                    }
                }
            ),
        )
        self.assertEqual(prose_only.error_kind, ProviderErrorKind.INVALID_REQUEST)

    def test_reasoning_only_length_reply_preserves_known_usage(self) -> None:
        raw = canonical_json_bytes({"model": "effective/model", "choices": [{"message": {"content": None}, "finish_reason": "length"}],
                                    "usage": {"prompt_tokens": 20, "completion_tokens": 16384, "total_tokens": 16404}})
        class Response:
            status = 200
            headers = {"Content-Type": "application/json"}
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return raw
        with patch.dict(os.environ, {"QUASILISK_TEST_KEY": "secret"}), patch("quasilisk.provider.urlrequest.urlopen", return_value=Response()):
            result = self.make_provider().complete(make_request())
        self.assertIsInstance(result, ProviderSuccess)
        self.assertEqual(result.finish_status, FinishStatus.OUTPUT_LIMIT)
        self.assertEqual(result.text, "")
        self.assertEqual(result.usage["output_tokens"], 16384)
        self.assertEqual(decode_bytes(result.raw_body.to_dict()), raw)

    def test_malformed_http_success_is_outcome_unknown(self) -> None:
        class Response:
            status = 200
            headers = {"Content-Type": "application/json"}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self) -> bytes:
                return b'{"choices":['

        with patch.dict(os.environ, {"QUASILISK_TEST_KEY": "secret"}), patch(
            "quasilisk.provider.urlrequest.urlopen", return_value=Response()
        ):
            result = self.make_provider().complete(make_request())
        self.assertIsInstance(result, ProviderFailure)
        assert isinstance(result, ProviderFailure)
        self.assertEqual(result.error_kind, ProviderErrorKind.MALFORMED_RESPONSE)
        self.assertEqual(result.delivery_state, "outcome_unknown")
        self.assertEqual(decode_bytes(result.raw_body.to_dict()), b'{"choices":[')

    def test_trickling_success_body_cannot_extend_wall_clock_deadline(self) -> None:
        class TricklingResponse:
            status = 200
            headers = {"Content-Type": "application/json"}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self) -> bytes:
                # Model a peer that makes progress more frequently than the
                # socket inactivity timeout but never completes in time.
                for _ in range(40):
                    time.sleep(0.05)
                return b'{"choices":[]}'

        previous_handler = signal.getsignal(signal.SIGALRM)
        started = time.monotonic()
        with patch.dict(os.environ, {"QUASILISK_TEST_KEY": "secret"}), patch(
            "quasilisk.provider.urlrequest.urlopen",
            return_value=TricklingResponse(),
        ) as opener:
            result = self.make_provider().complete(make_request())
        elapsed = time.monotonic() - started

        self.assertIsInstance(result, ProviderFailure)
        assert isinstance(result, ProviderFailure)
        self.assertEqual(result.error_kind, ProviderErrorKind.TIMEOUT)
        self.assertEqual(result.delivery_state, "outcome_unknown")
        self.assertEqual(result.detail, "transport_timeout")
        self.assertIsNone(result.raw_body)
        self.assertLess(elapsed, 1.75)
        self.assertEqual(opener.call_args.kwargs["timeout"], 1)
        self.assertIs(signal.getsignal(signal.SIGALRM), previous_handler)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))

    def test_deadline_also_bounds_http_error_body_and_restores_handler(self) -> None:
        previous_handler = signal.getsignal(signal.SIGALRM)

        class DeadlineBody(io.BytesIO):
            closed_by_provider = False

            def read(self, size: int = -1) -> bytes:
                signal.raise_signal(signal.SIGALRM)
                raise AssertionError("deadline signal did not interrupt the read")

            def close(self) -> None:
                self.closed_by_provider = True
                super().close()

        body = DeadlineBody()
        error = urlerror.HTTPError(
            "https://provider.example/v1/chat/completions",
            429,
            "provider error",
            {"x-request-id": "request-id", "Retry-After": "1.5"},
            body,
        )
        with patch.dict(os.environ, {"QUASILISK_TEST_KEY": "secret"}), patch(
            "quasilisk.provider.urlrequest.urlopen", side_effect=error
        ):
            result = self.make_provider().complete(make_request())

        self.assertIsInstance(result, ProviderFailure)
        assert isinstance(result, ProviderFailure)
        self.assertEqual(result.error_kind, ProviderErrorKind.TIMEOUT)
        self.assertEqual(result.delivery_state, "known_no_output")
        self.assertEqual(result.provider_request_id, "request-id")
        self.assertEqual(result.http_status, 429)
        self.assertEqual(result.retry_after_seconds, "1.5")
        self.assertIsNone(result.raw_body)
        self.assertTrue(body.closed_by_provider)
        self.assertIs(signal.getsignal(signal.SIGALRM), previous_handler)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))

    def test_url_error_wrapping_socket_timeout_remains_a_timeout(self) -> None:
        error = urlerror.URLError(TimeoutError("timed out"))
        with patch.dict(os.environ, {"QUASILISK_TEST_KEY": "secret"}), patch(
            "quasilisk.provider.urlrequest.urlopen", side_effect=error
        ):
            result = self.make_provider().complete(make_request())

        self.assertIsInstance(result, ProviderFailure)
        assert isinstance(result, ProviderFailure)
        self.assertEqual(result.error_kind, ProviderErrorKind.TIMEOUT)
        self.assertEqual(result.delivery_state, "outcome_unknown")
        self.assertEqual(result.detail, "transport_timeout")

    def test_expired_setup_budget_never_enters_network_io(self) -> None:
        with patch.dict(os.environ, {"QUASILISK_TEST_KEY": "secret"}), patch(
            "quasilisk.provider.time.monotonic_ns",
            side_effect=(0, 0, 1_000_000_001),
        ), patch("quasilisk.provider.urlrequest.urlopen") as opener:
            result = self.make_provider().complete(make_request())

        self.assertIsInstance(result, ProviderFailure)
        assert isinstance(result, ProviderFailure)
        self.assertEqual(result.error_kind, ProviderErrorKind.TIMEOUT)
        self.assertEqual(result.delivery_state, "outcome_unknown")
        opener.assert_not_called()

    def test_alarm_that_becomes_pending_during_setup_is_not_misattributed(self) -> None:
        timer_checked = False

        def getitimer(_which: int) -> tuple[float, float]:
            nonlocal timer_checked
            timer_checked = True
            return (0.0, 0.0)

        def sigpending() -> set[signal.Signals]:
            return {signal.SIGALRM} if timer_checked else set()

        with patch.dict(os.environ, {"QUASILISK_TEST_KEY": "secret"}), patch(
            "quasilisk.provider.signal.getitimer", side_effect=getitimer
        ), patch(
            "quasilisk.provider.signal.sigpending", side_effect=sigpending
        ), patch("quasilisk.provider.urlrequest.urlopen") as opener:
            with self.assertRaisesRegex(
                ProviderProtocolError, "requires no pending alarm signal"
            ):
                self.make_provider().complete(make_request())
        opener.assert_not_called()

    def test_connection_reset_reading_http_error_keeps_known_evidence(self) -> None:
        class ResetBody(io.BytesIO):
            closed_by_provider = False

            def read(self, size: int = -1) -> bytes:
                raise ConnectionResetError("connection reset")

            def close(self) -> None:
                self.closed_by_provider = True
                super().close()

        body = ResetBody()
        error = urlerror.HTTPError(
            "https://provider.example/v1/chat/completions",
            503,
            "provider error",
            {"x-request-id": "request-id", "Retry-After": "2"},
            body,
        )
        with patch.dict(os.environ, {"QUASILISK_TEST_KEY": "secret"}), patch(
            "quasilisk.provider.urlrequest.urlopen", side_effect=error
        ):
            result = self.make_provider().complete(make_request())

        self.assertIsInstance(result, ProviderFailure)
        assert isinstance(result, ProviderFailure)
        self.assertEqual(result.error_kind, ProviderErrorKind.CONNECTION)
        self.assertEqual(result.delivery_state, "known_no_output")
        self.assertEqual(result.provider_request_id, "request-id")
        self.assertEqual(result.http_status, 503)
        self.assertEqual(result.retry_after_seconds, "2")
        self.assertIsNone(result.raw_body)
        self.assertTrue(body.closed_by_provider)


if __name__ == "__main__":
    unittest.main()
