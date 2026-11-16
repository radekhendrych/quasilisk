"""Role-neutral provider transport.

Adapters perform exactly one physical request or one stateless scripted lookup.
Retry, repair, fallback, judging, and role semantics belong to callers.
"""

from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
import json
import os
from pathlib import Path
import signal
import socket
import threading
import time
from typing import Any, Iterator, Mapping, Protocol
from urllib import error as urlerror
from urllib import request as urlrequest

from .config import ProviderConfig
from .domain import (
    ChatRequest,
    DomainError,
    EncodedBytes,
    FinishStatus,
    ProviderErrorKind,
    ProviderFailure,
    ProviderSuccess,
    Role,
    canonical_json_bytes,
    digest_json,
    normalize_decimal,
    require_decimal_string,
    require_digest,
    sha256_bytes,
    strict_json_loads,
)


class ProviderProtocolError(DomainError):
    """A provider configuration/fixture violates the transport contract."""


class ScriptedFixtureMissing(ProviderProtocolError):
    """The deterministic plan requested a key absent from its frozen script."""


class _ProviderDeadlineExpired(Exception):
    """Escape urllib's broad OSError handling when the absolute deadline expires."""


class Provider(Protocol):
    content_digest: str | None

    def complete(self, request: ChatRequest) -> ProviderSuccess | ProviderFailure:
        ...


def encode_bytes(value: bytes) -> EncodedBytes:
    try:
        return EncodedBytes("utf8", value.decode("utf-8", errors="strict"))
    except UnicodeDecodeError:
        return EncodedBytes("base64", base64.b64encode(value).decode("ascii"))


def decode_bytes(value: Mapping[str, Any], label: str = "raw_body") -> bytes:
    if not isinstance(value, dict) or set(value) != {"encoding", "data"}:
        raise ProviderProtocolError(f"{label} must be an EncodedBytes object")
    encoding, data = value["encoding"], value["data"]
    if encoding not in {"utf8", "base64"} or not isinstance(data, str):
        raise ProviderProtocolError(f"{label} has an invalid encoding or data")
    if encoding == "utf8":
        try:
            return data.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise ProviderProtocolError(
                f"{label} contains text that is not valid UTF-8"
            ) from exc
    try:
        return base64.b64decode(data, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ProviderProtocolError(f"{label} contains invalid RFC 4648 base64") from exc


def requested_provider_route_key(adapter_kind: str, role: Role | str, route_label: str) -> str:
    if not isinstance(adapter_kind, str) or adapter_kind not in {
        "openai_compatible",
        "scripted",
    }:
        raise ProviderProtocolError("invalid adapter kind")
    role_value = role.value if isinstance(role, Role) else role
    if not isinstance(role_value, str) or role_value not in {
        item.value for item in Role
    }:
        raise ProviderProtocolError("invalid role for provider route")
    if not isinstance(route_label, str) or route_label not in {
        "primary",
        "replacement",
        "fallback",
    }:
        raise ProviderProtocolError("invalid provider route label")
    return digest_json([adapter_kind, role_value, route_label])


def scripted_lookup_key(request: ChatRequest) -> str:
    return digest_json(
        [
            request.logical_action_scientific_key,
            request.role.value,
            request.attempt_ordinal,
            request.route_kind,
            request.semantic_request_hash,
        ]
    )


def _canonical_request_object(request: ChatRequest) -> dict[str, Any]:
    temperature = request.generation_parameters.get("temperature")
    max_tokens = request.generation_parameters.get("max_output_tokens")
    if temperature is not None and not isinstance(temperature, str):
        raise ProviderProtocolError("generation temperature must be a normalized decimal string")
    if temperature is not None:
        require_decimal_string(temperature)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ProviderProtocolError("max_output_tokens must be a positive integer")
    body: dict[str, Any] = {
        "model": request.requested_model,
        "messages": [message.to_dict() for message in request.messages],
        # Decimal is rendered as a JSON number below, not as a scientific hash input.
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        body["temperature"] = Decimal(temperature)
    effort = request.generation_parameters.get("reasoning_effort")
    if effort is not None:
        body["reasoning"] = {"effort": effort}
    provider_only = request.generation_parameters.get("provider_only")
    prices = request.generation_parameters.get("max_price")
    if provider_only is not None or prices is not None:
        body["provider"] = {"require_parameters": True, "allow_fallbacks": False, "sort": "price"}
        if provider_only is not None:
            body["provider"]["only"] = [provider_only]
        if prices is not None:
            body["provider"]["max_price"] = {name: Decimal(value) for name, value in prices.items()}
    if request.response_mode == "json_schema":
        if request.response_schema is None:
            raise ProviderProtocolError("json_schema mode requires a response schema")
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "quasilisk_response",
                "strict": True,
                "schema": request.response_schema,
            },
        }
    elif request.response_mode == "json_object":
        body["response_format"] = {"type": "json_object"}
    return body


def render_request_body(request: ChatRequest) -> bytes:
    """Render request bytes deterministically while keeping decimals numeric."""

    def render(value: Any) -> str:
        if value is None or isinstance(value, (str, bool, int)):
            return json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        if isinstance(value, Decimal):
            return normalize_decimal(value)
        if isinstance(value, Mapping):
            if any(not isinstance(key, str) for key in value):
                raise ProviderProtocolError("request object keys must be strings")
            return "{" + ",".join(
                f"{render(key)}:{render(value[key])}" for key in sorted(value)
            ) + "}"
        if isinstance(value, (list, tuple)):
            return "[" + ",".join(render(item) for item in value) + "]"
        raise ProviderProtocolError(
            f"request value is not JSON-compatible: {type(value).__name__}"
        )

    try:
        return render(_canonical_request_object(request)).encode(
            "utf-8", errors="strict"
        )
    except UnicodeEncodeError as exc:
        raise ProviderProtocolError("request contains text that is not valid UTF-8") from exc


def _closed(record: Mapping[str, Any], fields: set[str], label: str) -> None:
    if set(record) != fields:
        missing = sorted(fields - set(record))
        extra = sorted(set(record) - fields)
        raise ProviderProtocolError(
            f"{label} fields mismatch; missing={missing}, unknown={extra}"
        )


def _nullable_string(value: Any, label: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ProviderProtocolError(f"{label} must be string or null")
    return value


def _nullable_nonnegative_integer(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProviderProtocolError(f"{label} must be a nonnegative integer or null")
    return value


@dataclass(frozen=True, slots=True)
class ScriptedRecord:
    key: str
    result: ProviderSuccess | ProviderFailure


def _parse_script_record(record: Any, line_number: int) -> ScriptedRecord:
    label = f"script line {line_number}"
    if not isinstance(record, dict):
        raise ProviderProtocolError(f"{label} must be an object")
    outcome = record.get("outcome")
    key = record.get("key")
    require_digest(key, f"{label} key")
    if outcome == "success":
        _closed(
            record,
            {
                "key",
                "outcome",
                "raw_body",
                "text",
                "provider_request_id",
                "effective_model_id",
                "finish_status",
                "usage",
                "latency_ms",
                "cost",
                "metadata",
            },
            label,
        )
        raw_bytes = decode_bytes(record["raw_body"], f"{label}.raw_body")
        encoded = encode_bytes(raw_bytes)
        if encoded.to_dict() != record["raw_body"]:
            # UTF-8 bodies must use utf8, non-UTF-8 bodies base64: one canonical representation.
            raise ProviderProtocolError(f"{label}.raw_body is not canonically encoded")
        text = record["text"]
        if not isinstance(text, str):
            raise ProviderProtocolError(f"{label}.text must be a string")
        try:
            finish = FinishStatus(record["finish_status"])
        except (ValueError, TypeError) as exc:
            raise ProviderProtocolError(f"{label}.finish_status is invalid") from exc
        usage = record["usage"]
        if not isinstance(usage, dict):
            raise ProviderProtocolError(f"{label}.usage must be an object")
        _closed(usage, {"input_tokens", "output_tokens", "total_tokens"}, f"{label}.usage")
        parsed_usage = {
            name: _nullable_nonnegative_integer(usage[name], f"{label}.usage.{name}")
            for name in ("input_tokens", "output_tokens", "total_tokens")
        }
        if all(value is not None for value in parsed_usage.values()) and parsed_usage["total_tokens"] != parsed_usage["input_tokens"] + parsed_usage["output_tokens"]:  # type: ignore[operator]
            raise ProviderProtocolError(f"{label} token total is inconsistent")
        latency = _nullable_nonnegative_integer(record["latency_ms"], f"{label}.latency_ms")
        if latency is None:
            raise ProviderProtocolError(f"{label}.latency_ms cannot be null")
        cost = record["cost"]
        if cost is not None:
            try:
                require_decimal_string(cost, minimum=Decimal("0"))
            except DomainError as exc:
                raise ProviderProtocolError(f"{label}.cost is invalid") from exc
        metadata = record["metadata"]
        if not isinstance(metadata, dict):
            raise ProviderProtocolError(f"{label}.metadata must be an object")
        _closed(metadata, {"http_status", "content_type"} | ({"reasoning_tokens"} if "reasoning_tokens" in metadata else set()), f"{label}.metadata")
        http_status = metadata["http_status"]
        if http_status is not None and (isinstance(http_status, bool) or not isinstance(http_status, int)):
            raise ProviderProtocolError(f"{label}.metadata.http_status must be integer or null")
        content_type = _nullable_string(metadata["content_type"], f"{label}.metadata.content_type")
        return ScriptedRecord(
            key,
            ProviderSuccess(
                raw_body=encoded,
                text=text,
                provider_request_id=_nullable_string(
                    record["provider_request_id"], f"{label}.provider_request_id"
                ),
                effective_model_id=_nullable_string(
                    record["effective_model_id"], f"{label}.effective_model_id"
                ),
                finish_status=finish,
                usage=parsed_usage,
                latency_ms=latency,
                cost=cost,
                metadata={**metadata, "http_status": http_status, "content_type": content_type},
            ),
        )
    if outcome == "failure":
        _closed(
            record,
            {
                "key",
                "outcome",
                "error_kind",
                "delivery_state",
                "detail",
                "provider_request_id",
                "raw_body",
                "http_status",
                "retry_after_seconds",
            },
            label,
        )
        try:
            error_kind = ProviderErrorKind(record["error_kind"])
        except (ValueError, TypeError) as exc:
            raise ProviderProtocolError(f"{label}.error_kind is invalid") from exc
        if record["delivery_state"] not in {"known_no_output", "outcome_unknown"}:
            raise ProviderProtocolError(f"{label}.delivery_state is invalid")
        detail = record["detail"]
        if not isinstance(detail, str) or not detail:
            raise ProviderProtocolError(f"{label}.detail must be nonempty")
        raw_body = record["raw_body"]
        encoded_body: EncodedBytes | None = None
        if raw_body is not None:
            raw_bytes = decode_bytes(raw_body, f"{label}.raw_body")
            encoded_body = encode_bytes(raw_bytes)
            if encoded_body.to_dict() != raw_body:
                raise ProviderProtocolError(f"{label}.raw_body is not canonically encoded")
        http_status = record["http_status"]
        if http_status is not None and (isinstance(http_status, bool) or not isinstance(http_status, int)):
            raise ProviderProtocolError(f"{label}.http_status must be integer or null")
        retry_after = record["retry_after_seconds"]
        if retry_after is not None:
            try:
                require_decimal_string(retry_after, minimum=Decimal("0"))
            except DomainError as exc:
                raise ProviderProtocolError(f"{label}.retry_after_seconds is invalid") from exc
        return ScriptedRecord(
            key,
            ProviderFailure(
                error_kind=error_kind,
                delivery_state=record["delivery_state"],
                detail=detail,
                provider_request_id=_nullable_string(
                    record["provider_request_id"], f"{label}.provider_request_id"
                ),
                raw_body=encoded_body,
                http_status=http_status,
                retry_after_seconds=retry_after,
            ),
        )
    raise ProviderProtocolError(f"{label}.outcome must be success or failure")


class ScriptedProvider:
    def __init__(self, path: Path):
        self.path = path
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ProviderProtocolError(f"cannot read scripted fixture: {exc}") from exc
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ProviderProtocolError("scripted fixture is not UTF-8") from exc
        records: dict[str, ProviderSuccess | ProviderFailure] = {}
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                raise ProviderProtocolError(
                    f"scripted fixture line {line_number} is blank"
                )
            try:
                value = strict_json_loads(line)
            except DomainError as exc:
                raise ProviderProtocolError(
                    f"scripted fixture line {line_number} is invalid strict JSON: {exc}"
                ) from exc
            try:
                parsed = _parse_script_record(value, line_number)
            except ProviderProtocolError:
                raise
            except DomainError as exc:
                raise ProviderProtocolError(
                    f"scripted fixture line {line_number} violates its contract: {exc}"
                ) from exc
            if parsed.key in records:
                raise ProviderProtocolError(f"duplicate scripted key: {parsed.key}")
            records[parsed.key] = parsed.result
        self._records = records
        self.content_digest = sha256_bytes(raw)

    def complete(self, request: ChatRequest) -> ProviderSuccess | ProviderFailure:
        key = scripted_lookup_key(request)
        try:
            return self._records[key]
        except KeyError as exc:
            raise ScriptedFixtureMissing(
                f"scripted_fixture_missing: {key}"
            ) from exc


def _redact_detail(value: str) -> str:
    # Exception text can echo URLs or headers.  Persist a deterministic class
    # label rather than volatile or secret-bearing implementation details.
    lowered = value.lower()
    if "timed out" in lowered or "timeout" in lowered:
        return "transport_timeout"
    if "name or service" in lowered or "dns" in lowered:
        return "transport_name_resolution_failed"
    return "transport_error"


_FILTER_ERROR_DISCRIMINATORS = frozenset(
    {
        "content_filter",
        "content_policy_violation",
        "responsibleaipolicyviolation",
    }
)

_UNSUPPORTED_FORMAT_DISCRIMINATORS = frozenset(
    {
        "response_format_unsupported",
        "unsupported_response_format",
    }
)

_CONTEXT_LIMIT_DISCRIMINATORS = frozenset(
    {
        "context_length_exceeded",
        "context_limit_exceeded",
        "max_context_length_exceeded",
    }
)


def _typed_error_discriminators(body: bytes) -> tuple[str, ...]:
    """Return allowlistable error codes/types without inspecting message prose."""

    try:
        envelope = strict_json_loads(body)
    except DomainError:
        return ()
    if not isinstance(envelope, Mapping):
        return ()
    error = envelope.get("error", envelope)
    if not isinstance(error, Mapping):
        return ()

    containers: list[Mapping[str, Any]] = [error]
    for name in ("innererror", "inner_error"):
        nested = error.get(name)
        if isinstance(nested, Mapping):
            containers.append(nested)
    details = error.get("details")
    if isinstance(details, list):
        containers.extend(item for item in details if isinstance(item, Mapping))

    values: list[str] = []
    for container in containers:
        for name in ("code", "type"):
            value = container.get(name)
            if isinstance(value, str):
                values.append(value.casefold())
    return tuple(values)


def _http_error_kind(status: int, body: bytes) -> ProviderErrorKind:
    discriminators = _typed_error_discriminators(body)
    if status == 401:
        return ProviderErrorKind.AUTHENTICATION
    if status == 429:
        return ProviderErrorKind.RATE_LIMIT
    if status >= 500:
        return ProviderErrorKind.SERVER
    if any(value in _FILTER_ERROR_DISCRIMINATORS for value in discriminators):
        return ProviderErrorKind.FILTERED
    if status == 403:
        return ProviderErrorKind.AUTHENTICATION
    if any(
        value in _UNSUPPORTED_FORMAT_DISCRIMINATORS
        for value in discriminators
    ):
        return ProviderErrorKind.UNSUPPORTED_RESPONSE_FORMAT
    if any(value in _CONTEXT_LIMIT_DISCRIMINATORS for value in discriminators):
        return ProviderErrorKind.CONTEXT_LIMIT
    if status in {400, 404, 405, 409, 422}:
        return ProviderErrorKind.INVALID_REQUEST
    return ProviderErrorKind.UNKNOWN


@contextmanager
def _strict_wall_clock_deadline(timeout_seconds: int) -> Iterator[None]:
    """Bound one synchronous exchange even when the peer keeps bytes flowing.

    ``urlopen(timeout=...)`` applies an inactivity timeout to individual socket
    operations.  A peer can therefore keep an unbounded ``read()`` alive by
    sending data frequently enough.  The canonical engine is serial, so a
    process real-time timer gives its network adapter one absolute deadline
    without introducing an uncancellable background request.
    """

    if (
        threading.current_thread() is not threading.main_thread()
        or not hasattr(signal, "SIGALRM")
        or not hasattr(signal, "ITIMER_REAL")
        or not hasattr(signal, "setitimer")
        or not hasattr(signal, "pthread_sigmask")
        or not hasattr(signal, "sigpending")
        or not hasattr(signal, "sigwait")
    ):
        raise ProviderProtocolError(
            "strict network provider deadline requires a POSIX main thread"
        )

    def expire(_signum: int, _frame: Any) -> None:
        # Do not subclass OSError: urllib catches it broadly and could otherwise
        # misclassify this adapter-owned deadline as a URL error.
        raise _ProviderDeadlineExpired

    deadline_ns = time.monotonic_ns() + timeout_seconds * 1_000_000_000
    alarm_set = {signal.SIGALRM}
    entry_mask = signal.pthread_sigmask(signal.SIG_BLOCK, alarm_set)
    previous_handler: Any = None
    try:
        if signal.SIGALRM in entry_mask:
            raise ProviderProtocolError(
                "strict network provider deadline requires an unmasked alarm signal"
            )
        previous_timer = signal.getitimer(signal.ITIMER_REAL)
        if previous_timer[0] > 0:
            raise ProviderProtocolError(
                "strict network provider deadline cannot replace an active real-time timer"
            )
        if signal.SIGALRM in signal.sigpending():
            raise ProviderProtocolError(
                "strict network provider deadline requires no pending alarm signal"
            )
        previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, expire)
        remaining_ns = deadline_ns - time.monotonic_ns()
        if remaining_ns <= 0:
            raise _ProviderDeadlineExpired
        signal.setitimer(signal.ITIMER_REAL, remaining_ns / 1_000_000_000)
        if time.monotonic_ns() >= deadline_ns:
            raise _ProviderDeadlineExpired
    except BaseException:
        if previous_handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if signal.SIGALRM in signal.sigpending():
                signal.sigwait(alarm_set)
            signal.signal(signal.SIGALRM, previous_handler)
        signal.pthread_sigmask(signal.SIG_SETMASK, entry_mask)
        raise

    try:
        signal.pthread_sigmask(signal.SIG_SETMASK, entry_mask)
        yield
        if time.monotonic_ns() >= deadline_ns:
            raise _ProviderDeadlineExpired
    finally:
        exit_mask = signal.pthread_sigmask(signal.SIG_BLOCK, alarm_set)
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if signal.SIGALRM in signal.sigpending():
                signal.sigwait(alarm_set)
            signal.signal(signal.SIGALRM, previous_handler)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, exit_mask)


def _http_error_evidence(
    exc: urlerror.HTTPError,
) -> tuple[str | None, int, str | None]:
    headers = exc.headers or {}
    retry_after = headers.get("Retry-After")
    normalized_retry: str | None = None
    if retry_after is not None:
        try:
            normalized_retry = normalize_decimal(retry_after)
            if Decimal(normalized_retry) < 0:
                normalized_retry = None
        except DomainError:
            normalized_retry = None
    return headers.get("x-request-id"), exc.code, normalized_retry


def _transport_timeout_failure(
    exc: urlerror.HTTPError | None = None,
    *,
    detail: str = "transport_timeout",
) -> ProviderFailure:
    if exc is None:
        return ProviderFailure(
            ProviderErrorKind.TIMEOUT,
            "outcome_unknown",
            detail,
            None,
            None,
            None,
            None,
        )
    request_id, status, retry_after = _http_error_evidence(exc)
    return ProviderFailure(
        ProviderErrorKind.TIMEOUT,
        "known_no_output",
        detail,
        request_id,
        None,
        status,
        retry_after,
    )


def _transport_connection_failure(
    detail: str,
    exc: urlerror.HTTPError | None = None,
) -> ProviderFailure:
    if exc is None:
        return ProviderFailure(
            ProviderErrorKind.CONNECTION,
            "outcome_unknown",
            detail,
            None,
            None,
            None,
            None,
        )
    request_id, status, retry_after = _http_error_evidence(exc)
    return ProviderFailure(
        ProviderErrorKind.CONNECTION,
        "known_no_output",
        detail,
        request_id,
        None,
        status,
        retry_after,
    )


class OpenAICompatibleProvider:
    content_digest: str | None = None

    def __init__(self, config: ProviderConfig):
        if config.adapter != "openai_compatible" or not config.base_url or not config.api_key_env:
            raise ProviderProtocolError("network provider requires openai_compatible config")
        self.config = config

    def complete(self, request: ChatRequest) -> ProviderSuccess | ProviderFailure:
        credential = os.environ.get(self.config.api_key_env or "")
        if not credential:
            return ProviderFailure(
                ProviderErrorKind.AUTHENTICATION,
                "known_no_output",
                "credential_environment_variable_missing",
                None,
                None,
                None,
                None,
            )
        body = render_request_body(request)
        url = (self.config.base_url or "").rstrip("/") + "/chat/completions"
        transport_request = urlrequest.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {credential}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        started = time.monotonic_ns()
        http_error: urlerror.HTTPError | None = None
        completed_http_error: tuple[bytes, urlerror.HTTPError] | None = None
        try:
            with _strict_wall_clock_deadline(request.timeout_seconds):
                try:
                    with urlrequest.urlopen(
                        transport_request, timeout=request.timeout_seconds
                    ) as response:
                        raw = response.read()
                        status = response.status
                        content_type = response.headers.get("Content-Type")
                except urlerror.HTTPError as exc:
                    http_error = exc
                    with exc:
                        raw = exc.read()
                    completed_http_error = (raw, exc)
        except _ProviderDeadlineExpired:
            return _transport_timeout_failure(http_error)
        except (TimeoutError, socket.timeout) as exc:
            return _transport_timeout_failure(
                http_error, detail=_redact_detail(str(exc))
            )
        except urlerror.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                return _transport_timeout_failure(
                    http_error,
                    detail=_redact_detail(str(exc.reason)),
                )
            return _transport_connection_failure(
                _redact_detail(str(exc.reason)), http_error
            )
        except OSError as exc:
            return _transport_connection_failure(
                _redact_detail(str(exc)), http_error
            )
        if completed_http_error is not None:
            raw, exc = completed_http_error
            request_id, status, retry_after = _http_error_evidence(exc)
            return ProviderFailure(
                error_kind=_http_error_kind(status, raw),
                delivery_state="known_no_output",
                detail=f"http_status_{status}",
                provider_request_id=request_id,
                raw_body=encode_bytes(raw),
                http_status=status,
                retry_after_seconds=retry_after,
            )
        latency_ms = max(0, (time.monotonic_ns() - started) // 1_000_000)
        cost: str | None = None
        metadata: dict[str, Any] = {"http_status": status, "content_type": content_type}
        try:
            envelope = strict_json_loads(raw)
            if not isinstance(envelope, dict):
                raise DomainError("response envelope is not an object")
            choices = envelope.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise DomainError("response has no first choice")
            choice = choices[0]
            message = choice.get("message")
            # Reasoning-only length completions are known truncated outputs,
            # not malformed/unknown deliveries. Preserve their usage receipt.
            if isinstance(message, dict) and message.get("content") is None and choice.get("finish_reason") == "length":
                message = {**message, "content": ""}
            if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                raise DomainError("response choice has no text content")
            text = message["content"]
            finish_raw = choice.get("finish_reason")
            finish = {
                "stop": FinishStatus.COMPLETE,
                "length": FinishStatus.OUTPUT_LIMIT,
                "content_filter": FinishStatus.CONTENT_FILTER,
            }.get(finish_raw, FinishStatus.OTHER)
            usage_raw = envelope.get("usage")
            usage: dict[str, int | None] = {
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
            }
            if isinstance(usage_raw, dict):
                for source, target in (
                    ("prompt_tokens", "input_tokens"),
                    ("completion_tokens", "output_tokens"),
                    ("total_tokens", "total_tokens"),
                ):
                    value = usage_raw.get(source)
                    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                        usage[target] = value
                # Optional billing data must not invalidate an otherwise usable
                # completion. Preserve exact nonnegative receipts, never infer
                # that a missing/invalid receipt means a zero-dollar dispatch.
                value = usage_raw.get("cost")
                if isinstance(value, (Decimal, int, str)) and not isinstance(value, bool):
                    try:
                        normalized = normalize_decimal(value)
                        if Decimal(normalized) >= 0:
                            cost = normalized
                    except DomainError:
                        pass
                details = usage_raw.get("completion_tokens_details")
                if isinstance(details, dict):
                    reasoning = details.get("reasoning_tokens")
                    if isinstance(reasoning, int) and not isinstance(reasoning, bool) and reasoning >= 0:
                        metadata["reasoning_tokens"] = reasoning
            provider_request_id = envelope.get("id") if isinstance(envelope.get("id"), str) else None
            effective_model = envelope.get("model") if isinstance(envelope.get("model"), str) else None
        except DomainError:
            return ProviderFailure(
                ProviderErrorKind.MALFORMED_RESPONSE,
                # A successful HTTP response may already represent an
                # authoritative completion even when its envelope cannot be
                # decoded.  The adapter therefore cannot prove no output was
                # created; preserve the body and charge it as outcome-unknown.
                "outcome_unknown",
                "authoritative_envelope_malformed",
                None,
                encode_bytes(raw),
                status,
                None,
            )
        return ProviderSuccess(
            raw_body=encode_bytes(raw),
            text=text,
            provider_request_id=provider_request_id,
            effective_model_id=effective_model,
            finish_status=finish,
            usage=usage,
            latency_ms=latency_ms,
            cost=cost,
            metadata=metadata,
        )


def create_provider(config: ProviderConfig) -> Provider:
    if config.adapter == "scripted":
        assert config.script_path is not None
        return ScriptedProvider(config.script_path)
    if config.adapter == "openai_compatible":
        return OpenAICompatibleProvider(config)
    raise ProviderProtocolError(f"unsupported provider adapter: {config.adapter}")
