"""Durable, hash-chained, append-only event journal.

This module owns bytes and integrity only.  It deliberately knows no search,
budget, judging, or scheduling policy.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
import base64
import binascii
import fcntl
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, Sequence
from datetime import datetime, timezone

from .domain import (
    CALL_ROUTE_KINDS,
    DomainError,
    ENTITY_RE,
    JsonSchemaContractError,
    JsonSchemaValidationError,
    ModelProfile,
    RunTerminalReason,
    canonical_json_bytes,
    digest_json,
    entity_id,
    require_decimal_string,
    require_digest as _require_digest,
    require_entity_id as _require_entity_id,
    sha256_bytes,
    strict_json_loads,
    validate_json_schema_contract,
    validate_json_schema_instance,
)


SCHEMA_REVISION = 1
ZERO_HASH = "sha256:" + "0" * 64
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")
RUN_NONCE_RE = re.compile(r"^[0-9a-f]{32}$")

EVENT_TYPES = (
    "run_created",
    "run_resumed",
    "work_unit_started",
    "model_call_intended",
    "model_dispatch_started",
    "model_call_succeeded",
    "model_call_failed",
    "model_call_outcome_unknown",
    "run_terminated",
    "journal_tail_recovered",
)

PAYLOAD_FIELDS: Mapping[str, frozenset[str]] = MappingProxyType({
    "run_created": frozenset(
        {
            "manifest",
            "scientific_scope_digest",
            "run_nonce_hex",
            "run_id",
            "output_base",
            "run_path_at_creation",
            "creation_invocation_id",
        }
    ),
    "run_resumed": frozenset(
        {
            "invocation_id",
            "prior_recorded_run_path",
            "current_run_path",
            "validated_previous_hash",
            "prior_terminal_reason",
            "recovery_actions",
            "dangling_attempt_ids",
        }
    ),
    "work_unit_started": frozenset({"unit"}),
    "model_call_intended": frozenset(
        {
            "logical_action_id",
            "logical_action_scientific_key",
            "physical_attempt_id",
            "attempt_ordinal",
            "unit_id",
            "owner_position_scientific_key",
            "role",
            "purpose",
            "role_route_key",
            "route_kind",
            "parent_attempt_id",
            "messages",
            "request_body",
            "semantic_request_hash",
            "requested_provider_route_key",
            "requested_model",
            "generation_parameters",
            "response_mode",
            "response_schema_digest",
            "timeout_seconds",
            "idempotency_key",
            "lineage",
        }
    ),
    "model_dispatch_started": frozenset(
        {
            "physical_attempt_id",
            "logical_action_id",
            "role_route_key",
            "route_kind",
            "provider_route_key",
            "requested_model",
            "charged_role",
            "charge",
        }
    ),
    "model_call_succeeded": frozenset(
        {
            "physical_attempt_id",
            "raw_body",
            "text",
            "finish_status",
            "effective_model_id",
            "provider_request_id",
            "usage",
            "latency_ms",
            "cost",
            "metadata",
        }
    ),
    "model_call_failed": frozenset(
        {
            "physical_attempt_id",
            "error_kind",
            "delivery_state",
            "detail",
            "provider_request_id",
            "raw_body",
            "http_status",
            "retry_after_seconds",
        }
    ),
    "model_call_outcome_unknown": frozenset(
        {
            "physical_attempt_id",
            "discovery_invocation_id",
            "error_kind",
            "detail",
            "provider_request_id",
            "raw_body",
        }
    ),
    "run_terminated": frozenset(
        {"invocation_id", "reason", "last_completed_work_unit_id", "resumable", "detail"}
    ),
    "journal_tail_recovered": frozenset(
        {
            "recovery_invocation_id",
            "pending_event_hash",
            "original_journal_bytes",
            "original_pending_bytes",
            "action",
            "resulting_terminal_hash",
        }
    ),
})

MANIFEST_FIELDS = frozenset(
    {
        "normalized_config",
        "physical_call_limits",
        "config_digest",
        "scientific_scope_digest",
        "engine",
        "objective",
        "bootstrap_snapshot",
        "seed_snapshot",
        "strategy_template_snapshot",
        "library_source_checkpoints",
        "model_profiles",
        "role_template_digests",
        "response_schema_digests",
        "judge_output_schema",
        "descriptor_taxonomy",
        "decision_constants",
        "event_schema_digest",
        "work_unit_schema_digest",
        "reducer_digest",
        "random_seed",
        "scripted_fixture_digest",
    }
)

PROVIDER_ERROR_KINDS = frozenset(
    {
        "connection",
        "timeout",
        "rate_limit",
        "server",
        "authentication",
        "invalid_request",
        "unsupported_response_format",
        "context_limit",
        "filtered",
        "malformed_response",
        "unknown",
    }
)

ACTION_PURPOSES = frozenset(
    {
        "target_sample",
        "judge_member",
        "selector_portfolio",
        "probe_review",
        "template_binding",
        "outcome_review",
        "perturbator_proposal",
    }
)


class JournalError(RuntimeError):
    pass


class JournalCorrupt(JournalError):
    pass


class JournalUnwritable(JournalError):
    pass


class JournalBusy(JournalError):
    pass


def _freeze_schema_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_schema_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_schema_value(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class AuthoritativeSchemas:
    """The materialized event/work-unit contracts for one objective.

    The registry is explicit so validation can never resolve a schema from the
    network or an ambient filesystem location.
    """

    event: Mapping[str, Any]
    work_unit: Mapping[str, Any]

    def __post_init__(self) -> None:
        event = _freeze_schema_value(self.event)
        work_unit = _freeze_schema_value(self.work_unit)
        if not isinstance(event, Mapping) or not isinstance(work_unit, Mapping):
            raise JournalCorrupt("authoritative schemas must be objects")
        registry: dict[str, Mapping[str, Any]] = {"work_unit.json": work_unit}
        schema_id = work_unit.get("$id")
        if isinstance(schema_id, str):
            registry[schema_id] = work_unit
        try:
            canonical_json_bytes(work_unit)
            canonical_json_bytes(event)
            validate_json_schema_contract(work_unit)
            validate_json_schema_contract(event, registry=registry)
        except (DomainError, JsonSchemaContractError) as exc:
            raise JournalCorrupt(f"authoritative schema contract is invalid: {exc}") from exc
        object.__setattr__(self, "event", event)
        object.__setattr__(self, "work_unit", work_unit)

    @property
    def registry(self) -> Mapping[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {"work_unit.json": self.work_unit}
        schema_id = self.work_unit.get("$id")
        if isinstance(schema_id, str):
            result[schema_id] = self.work_unit
        return result

    def validate_event(self, event: Mapping[str, Any]) -> None:
        try:
            validate_json_schema_instance(event, self.event, registry=self.registry)
        except (JsonSchemaContractError, JsonSchemaValidationError) as exc:
            raise JournalCorrupt(f"event violates authoritative schema: {exc}") from exc

    def validate_work_unit(self, unit: Mapping[str, Any]) -> None:
        try:
            validate_json_schema_instance(unit, self.work_unit)
        except (JsonSchemaContractError, JsonSchemaValidationError) as exc:
            raise JournalCorrupt(f"work unit violates authoritative schema: {exc}") from exc


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _canonical_timestamp(value: object) -> str:
    if not isinstance(value, str) or TIMESTAMP_RE.fullmatch(value) is None:
        raise JournalCorrupt("timestamp is not canonical RFC 3339 UTC")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as exc:
        raise JournalCorrupt("timestamp is not a real UTC calendar instant") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%S.%fZ") != value:
        raise JournalCorrupt("timestamp is not canonical RFC 3339 UTC")
    return value


def require_digest(value: object, label: str = "digest") -> str:
    try:
        return _require_digest(value, label)
    except DomainError as exc:
        raise JournalCorrupt(str(exc)) from exc


def require_entity_id(value: object, label: str = "entity ID") -> str:
    try:
        return _require_entity_id(value, label)
    except DomainError as exc:
        raise JournalCorrupt(str(exc)) from exc


def _entity_with_prefix(value: object, prefix: str, label: str) -> str:
    result = require_entity_id(value, label)
    if not result.startswith(prefix + "-"):
        raise JournalCorrupt(f"{label} must use the {prefix}- prefix")
    return result


def _decimal_string(value: object, label: str) -> str:
    try:
        require_decimal_string(value, minimum=Decimal(0))
    except DomainError as exc:
        raise JournalCorrupt(f"{label} must be a canonical nonnegative decimal string") from exc
    assert isinstance(value, str)
    return value


def _exact_keys(value: Any, fields: Iterable[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise JournalCorrupt(f"{label} must be an object")
    expected = set(fields)
    if set(value) != expected:
        raise JournalCorrupt(
            f"{label} fields mismatch; missing={sorted(expected-set(value))}, "
            f"unknown={sorted(set(value)-expected)}"
        )
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise JournalCorrupt(f"{label} must be a nonnegative integer")
    return value


def _positive_int(value: Any, label: str) -> int:
    result = _nonnegative_int(value, label)
    if result == 0:
        raise JournalCorrupt(f"{label} must be positive")
    return result


def _nullable_text(value: Any, label: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise JournalCorrupt(f"{label} must be string or null")
    return value


def _nullable_nonempty_text(value: Any, label: str) -> str | None:
    if value is not None:
        return _nonempty_text(value, label)
    return None


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise JournalCorrupt(f"{label} must be a nonempty string")
    return value


def _validate_no_noninteger_numbers(value: Any, label: str = "event") -> None:
    if isinstance(value, (float, Decimal)):
        raise JournalCorrupt(f"{label} contains a noninteger JSON number")
    if isinstance(value, list):
        for item in value:
            _validate_no_noninteger_numbers(item, label)
    elif isinstance(value, dict):
        for item in value.values():
            _validate_no_noninteger_numbers(item, label)


def _validate_encoded_bytes(value: Any, label: str) -> None:
    obj = _exact_keys(value, {"encoding", "data"}, label)
    if obj["encoding"] not in {"utf8", "base64"} or not isinstance(obj["data"], str):
        raise JournalCorrupt(f"{label} is invalid")
    if obj["encoding"] == "base64":
        try:
            decoded = base64.b64decode(obj["data"].encode("ascii"), validate=True)
        except (UnicodeEncodeError, binascii.Error) as exc:
            raise JournalCorrupt(f"{label} contains invalid base64") from exc
        if base64.b64encode(decoded).decode("ascii") != obj["data"]:
            raise JournalCorrupt(f"{label} base64 is not canonical")
        try:
            decoded.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            pass
        else:
            raise JournalCorrupt(
                f"{label} must use utf8 encoding for valid UTF-8 bytes"
            )


def _validate_messages(value: Any) -> None:
    if not isinstance(value, list) or not value:
        raise JournalCorrupt("messages must be a nonempty array")
    for index, message in enumerate(value):
        obj = _exact_keys(message, {"role", "content"}, f"messages[{index}]")
        if obj["role"] not in {"system", "user", "assistant"} or not isinstance(obj["content"], str):
            raise JournalCorrupt("invalid chat message")


def _validate_work_unit(
    value: Any, schemas: AuthoritativeSchemas | None = None
) -> None:
    obj = _exact_keys(
        value,
        {
            "unit_id",
            "unit_scientific_key",
            "kind",
            "subject_id",
            "subject_scientific_key",
            "parent_unit_id",
            "policy_digest",
            "design",
            "proposal_slots",
            "execution_slots",
            "panel_plans",
            "logical_action_slots",
            "worst_case_calls",
        },
        "work unit",
    )
    _entity_with_prefix(obj["unit_id"], "wu", "work unit ID")
    require_digest(obj["unit_scientific_key"], "work unit scientific key")
    if obj["kind"] not in {
        "initial_observation",
        "screening",
        "map_exploration",
        "portfolio_round",
        "verification",
        "strategy_development",
        "strategy_holdout",
    }:
        raise JournalCorrupt("invalid work-unit kind")
    require_entity_id(obj["subject_id"], "work unit subject ID")
    require_digest(obj["subject_scientific_key"], "work unit subject key")
    if obj["parent_unit_id"] is not None:
        _entity_with_prefix(obj["parent_unit_id"], "wu", "parent unit ID")
    require_digest(obj["policy_digest"], "work-unit policy digest")
    if not isinstance(obj["design"], dict) or obj["design"].get("tag") != obj["kind"]:
        raise JournalCorrupt("work-unit design tag mismatch")
    for field in ("proposal_slots", "execution_slots", "panel_plans", "logical_action_slots"):
        if not isinstance(obj[field], list):
            raise JournalCorrupt(f"work-unit {field} must be an array")
    calls = _exact_keys(
        obj["worst_case_calls"],
        {"target", "judge", "selector", "perturbator"},
        "worst_case_calls",
    )
    for role, count in calls.items():
        _nonnegative_int(count, f"worst_case_calls.{role}")
    if schemas is not None:
        schemas.validate_work_unit(obj)


def _validate_payload(
    event_type: str,
    payload: Any,
    *,
    sequence: int | None = None,
    schemas: AuthoritativeSchemas | None = None,
) -> None:
    if event_type not in PAYLOAD_FIELDS:
        raise JournalCorrupt(f"unknown event type: {event_type}")
    obj = _exact_keys(payload, PAYLOAD_FIELDS[event_type], f"{event_type} payload")
    _validate_no_noninteger_numbers(obj, f"{event_type} payload")
    if event_type == "run_created":
        if sequence is not None and sequence != 1:
            raise JournalCorrupt("run_created must be sequence 1")
        manifest = _exact_keys(obj["manifest"], MANIFEST_FIELDS, "manifest")
        normalized_config = manifest["normalized_config"]
        if not isinstance(normalized_config, dict):
            raise JournalCorrupt("manifest normalized_config must be an object")
        config_identity = dict(normalized_config)
        config_identity.pop("output_directory", None)
        if manifest["config_digest"] != digest_json(config_identity):
            raise JournalCorrupt("manifest configuration digest mismatch")
        require_digest(obj["scientific_scope_digest"], "scientific scope digest")
        if manifest["scientific_scope_digest"] != obj["scientific_scope_digest"]:
            raise JournalCorrupt("manifest scientific scope digest mismatch")
        if not isinstance(obj["run_nonce_hex"], str) or RUN_NONCE_RE.fullmatch(obj["run_nonce_hex"]) is None:
            raise JournalCorrupt("run nonce must be exactly 128 lowercase-hex bits")
        _entity_with_prefix(obj["run_id"], "run", "payload run ID")
        expected_run_id = entity_id(
            "run",
            [
                obj["run_nonce_hex"],
                obj["scientific_scope_digest"],
                manifest["config_digest"],
            ],
        )
        if obj["run_id"] != expected_run_id:
            raise JournalCorrupt("run ID does not match its frozen identity material")
        _entity_with_prefix(obj["creation_invocation_id"], "iv", "creation invocation ID")
        _nonempty_text(obj["output_base"], "output base")
        _nonempty_text(obj["run_path_at_creation"], "run path")
        return
    if event_type == "run_resumed":
        _entity_with_prefix(obj["invocation_id"], "iv", "resume invocation ID")
        _nonempty_text(obj["prior_recorded_run_path"], "prior recorded run path")
        _nonempty_text(obj["current_run_path"], "current run path")
        require_digest(obj["validated_previous_hash"], "validated previous hash")
        if obj["prior_terminal_reason"] is not None and obj["prior_terminal_reason"] not in {item.value for item in RunTerminalReason}:
            raise JournalCorrupt("invalid prior terminal reason")
        actions = obj["recovery_actions"]
        allowed = {
            "none",
            "pending_removed",
            "pending_appended",
            "partial_tail_replaced",
            "dangling_dispatch_closed",
        }
        if not isinstance(actions, list) or not actions or len(actions) != len(set(actions)) or not set(actions) <= allowed:
            raise JournalCorrupt("invalid resume recovery_actions")
        if (actions == ["none"]) != ("none" in actions):
            raise JournalCorrupt("none must be the sole recovery action")
        dangling = obj["dangling_attempt_ids"]
        if not isinstance(dangling, list) or dangling != sorted(set(dangling)):
            raise JournalCorrupt("dangling attempt IDs must be sorted and unique")
        if bool(dangling) != ("dangling_dispatch_closed" in actions):
            raise JournalCorrupt("dangling recovery action/list mismatch")
        for attempt_id in dangling:
            _entity_with_prefix(attempt_id, "pa", "dangling attempt ID")
        return
    if event_type == "work_unit_started":
        _validate_work_unit(obj["unit"], schemas)
        return
    if event_type == "model_call_intended":
        _entity_with_prefix(obj["logical_action_id"], "la", "logical action ID")
        require_digest(obj["logical_action_scientific_key"], "logical action key")
        _entity_with_prefix(obj["physical_attempt_id"], "pa", "physical attempt ID")
        _nonnegative_int(obj["attempt_ordinal"], "attempt ordinal")
        _entity_with_prefix(obj["unit_id"], "wu", "unit ID")
        require_digest(obj["owner_position_scientific_key"], "owner position key")
        if obj["role"] not in {"target", "judge", "selector", "perturbator"}:
            raise JournalCorrupt("invalid call role")
        if obj["purpose"] not in ACTION_PURPOSES:
            raise JournalCorrupt("invalid action purpose")
        require_digest(obj["role_route_key"], "role route key")
        if obj["route_kind"] not in CALL_ROUTE_KINDS:
            raise JournalCorrupt("invalid route kind")
        if obj["parent_attempt_id"] is not None:
            _entity_with_prefix(obj["parent_attempt_id"], "pa", "parent attempt ID")
        _validate_messages(obj["messages"])
        _validate_encoded_bytes(obj["request_body"], "request_body")
        require_digest(obj["semantic_request_hash"], "semantic request hash")
        require_digest(obj["requested_provider_route_key"], "provider route key")
        _nonempty_text(obj["requested_model"], "requested model")
        generation = _exact_keys(
            obj["generation_parameters"],
            {"temperature", "max_output_tokens"} | (set(obj["generation_parameters"]) & {"reasoning_effort", "provider_only", "max_price"}),
            "generation parameters",
        )
        if generation["temperature"] is not None:
            _decimal_string(generation["temperature"], "temperature")
        ModelProfile(model=obj["requested_model"], **generation)
        _positive_int(generation["max_output_tokens"], "max output tokens")
        if obj["response_mode"] not in {"text", "json_schema", "json_object"}:
            raise JournalCorrupt("invalid response mode")
        if obj["response_schema_digest"] is not None:
            require_digest(obj["response_schema_digest"], "response schema digest")
        _positive_int(obj["timeout_seconds"], "timeout seconds")
        _nullable_nonempty_text(obj["idempotency_key"], "idempotency key")
        lineage = _exact_keys(
            obj["lineage"],
            {
                "execution_id",
                "conversation_root_id",
                "parent_node_id",
                "candidate_id",
                "panel_id",
                "logical_member_slot",
                "strategy_id",
                "proposal_slot_scientific_key",
                "review_item_id",
            },
            "lineage",
        )
        for field in (
            "execution_id",
            "conversation_root_id",
            "parent_node_id",
            "candidate_id",
            "panel_id",
            "strategy_id",
        ):
            if lineage[field] is not None:
                expected_prefix = {
                    "execution_id": "ex",
                    "conversation_root_id": "cr",
                    "parent_node_id": "cn",
                    "candidate_id": "pc",
                    "panel_id": "pn",
                    "strategy_id": "st",
                }[field]
                _entity_with_prefix(lineage[field], expected_prefix, f"lineage {field}")
        if lineage["logical_member_slot"] not in {None, "primary", "secondary_a", "secondary_b"}:
            raise JournalCorrupt("invalid logical member slot")
        if lineage["proposal_slot_scientific_key"] is not None:
            require_digest(lineage["proposal_slot_scientific_key"], "proposal slot key")
        review_item_id = _nullable_nonempty_text(lineage["review_item_id"], "review item ID")
        if review_item_id is not None and not review_item_id.startswith(("ri-", "bi-")):
            raise JournalCorrupt("review item ID must use the ri- or bi- prefix")
        return
    if event_type == "model_dispatch_started":
        _entity_with_prefix(obj["physical_attempt_id"], "pa", "physical attempt ID")
        _entity_with_prefix(obj["logical_action_id"], "la", "logical action ID")
        require_digest(obj["role_route_key"], "role route key")
        if obj["route_kind"] not in CALL_ROUTE_KINDS:
            raise JournalCorrupt("invalid dispatch route kind")
        require_digest(obj["provider_route_key"], "provider route key")
        _nonempty_text(obj["requested_model"], "requested model")
        if obj["charged_role"] not in {"target", "judge", "selector", "perturbator"} or obj["charge"] != 1:
            raise JournalCorrupt("invalid dispatch charge")
        return
    if event_type == "model_call_succeeded":
        _entity_with_prefix(obj["physical_attempt_id"], "pa", "physical attempt ID")
        _validate_encoded_bytes(obj["raw_body"], "raw body")
        if not isinstance(obj["text"], str):
            raise JournalCorrupt("model text must be a string")
        if obj["finish_status"] not in {"complete", "output_limit", "content_filter", "other"}:
            raise JournalCorrupt("invalid finish status")
        _nullable_nonempty_text(obj["effective_model_id"], "effective model ID")
        _nullable_nonempty_text(obj["provider_request_id"], "provider request ID")
        usage = _exact_keys(
            obj["usage"], {"input_tokens", "output_tokens", "total_tokens"}, "usage"
        )
        for name, count in usage.items():
            if count is not None:
                _nonnegative_int(count, f"usage.{name}")
        if all(value is not None for value in usage.values()) and usage["total_tokens"] != usage["input_tokens"] + usage["output_tokens"]:
            raise JournalCorrupt("token total mismatch")
        _nonnegative_int(obj["latency_ms"], "latency_ms")
        if obj["cost"] is not None:
            _decimal_string(obj["cost"], "cost")
        metadata_keys = {"http_status", "content_type"}
        if isinstance(obj["metadata"], dict) and "reasoning_tokens" in obj["metadata"]:
            metadata_keys.add("reasoning_tokens")
        metadata = _exact_keys(obj["metadata"], metadata_keys, "metadata")
        if "reasoning_tokens" in metadata:
            _nonnegative_int(metadata["reasoning_tokens"], "metadata.reasoning_tokens")
        if metadata["http_status"] is not None and (
            isinstance(metadata["http_status"], bool)
            or not isinstance(metadata["http_status"], int)
        ):
            raise JournalCorrupt("metadata.http_status must be integer or null")
        _nullable_text(metadata["content_type"], "metadata.content_type")
        return
    if event_type in {"model_call_failed", "model_call_outcome_unknown"}:
        _entity_with_prefix(obj["physical_attempt_id"], "pa", "physical attempt ID")
        if obj["error_kind"] not in PROVIDER_ERROR_KINDS:
            raise JournalCorrupt("invalid provider error kind")
        detail = _nonempty_text(obj["detail"], "provider error detail")
        if "\r" in detail or "\n" in detail:
            raise JournalCorrupt("provider error detail must be one line")
        _nullable_nonempty_text(obj["provider_request_id"], "provider request ID")
        if obj["raw_body"] is not None:
            _validate_encoded_bytes(obj["raw_body"], "provider error raw body")
        if event_type == "model_call_failed":
            if obj["delivery_state"] != "known_no_output":
                raise JournalCorrupt("model_call_failed requires known_no_output")
            if obj["http_status"] is not None and (
                isinstance(obj["http_status"], bool)
                or not isinstance(obj["http_status"], int)
            ):
                raise JournalCorrupt("http status must be integer or null")
            if obj["retry_after_seconds"] is not None:
                _decimal_string(obj["retry_after_seconds"], "Retry-After")
        else:
            _entity_with_prefix(obj["discovery_invocation_id"], "iv", "discovery invocation ID")
        return
    if event_type == "run_terminated":
        _entity_with_prefix(obj["invocation_id"], "iv", "invocation ID")
        if obj["reason"] not in {item.value for item in RunTerminalReason}:
            raise JournalCorrupt("invalid terminal reason")
        if obj["last_completed_work_unit_id"] is not None:
            _entity_with_prefix(obj["last_completed_work_unit_id"], "wu", "last work-unit ID")
        if not isinstance(obj["resumable"], bool):
            raise JournalCorrupt("run terminal resumable must be boolean")
        detail = _nonempty_text(obj["detail"], "terminal detail")
        if "\r" in detail or "\n" in detail:
            raise JournalCorrupt("terminal detail must be one line")
        return
    if event_type == "journal_tail_recovered":
        _entity_with_prefix(obj["recovery_invocation_id"], "iv", "recovery invocation ID")
        require_digest(obj["pending_event_hash"], "pending event hash")
        _nonnegative_int(obj["original_journal_bytes"], "original journal bytes")
        _nonnegative_int(obj["original_pending_bytes"], "original pending bytes")
        if obj["action"] not in {"pending_removed", "pending_appended", "partial_tail_replaced"}:
            raise JournalCorrupt("invalid tail recovery action")
        require_digest(obj["resulting_terminal_hash"], "resulting terminal hash")


def validate_payload(
    event_type: str,
    payload: Any,
    *,
    sequence: int | None = None,
    schemas: AuthoritativeSchemas | None = None,
) -> None:
    """Validate one payload and translate malformed container operations."""

    try:
        _validate_payload(
            event_type,
            payload,
            sequence=sequence,
            schemas=schemas,
        )
    except JournalCorrupt:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise JournalCorrupt(
            f"{event_type!r} payload has malformed field types: {exc}"
        ) from exc


def event_id_for(
    run_id: str,
    sequence: int,
    event_type: str,
    causes: Sequence[str],
    payload: Mapping[str, Any],
) -> str:
    return entity_id("ev", [run_id, sequence, event_type, list(causes), dict(payload)])


def event_hash_for(event_without_hash: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(event_without_hash))


def build_event(
    *,
    run_id: str,
    sequence: int,
    event_type: str,
    payload: Mapping[str, Any],
    previous_hash: str,
    causes: Iterable[str] = (),
    timestamp_utc: str | None = None,
    schemas: AuthoritativeSchemas | None = None,
) -> dict[str, Any]:
    _entity_with_prefix(run_id, "run", "run ID")
    if sequence <= 0:
        raise JournalCorrupt("event sequence must be positive")
    if event_type not in EVENT_TYPES:
        raise JournalCorrupt(f"unknown event type: {event_type}")
    payload_snapshot = deepcopy(dict(payload))
    validate_payload(
        event_type,
        payload_snapshot,
        sequence=sequence,
        schemas=schemas,
    )
    supplied_causes = tuple(causes)
    cause_list = sorted(set(supplied_causes))
    if len(cause_list) != len(supplied_causes):
        raise JournalCorrupt("event causes must be duplicate-free")
    for cause in cause_list:
        _entity_with_prefix(cause, "ev", "event cause")
    require_digest(previous_hash, "previous event hash")
    timestamp = timestamp_utc or utc_timestamp()
    _canonical_timestamp(timestamp)
    event_without_hash: dict[str, Any] = {
        "schema_revision": SCHEMA_REVISION,
        "run_id": run_id,
        "sequence": sequence,
        "event_id": event_id_for(
            run_id,
            sequence,
            event_type,
            cause_list,
            payload_snapshot,
        ),
        "type": event_type,
        "timestamp_utc": timestamp,
        "causes": cause_list,
        "payload": payload_snapshot,
        "previous_hash": previous_hash,
    }
    event = {**event_without_hash, "event_hash": event_hash_for(event_without_hash)}
    if schemas is not None:
        schemas.validate_event(event)
    return event


def validate_events(
    events: Sequence[Mapping[str, Any]],
    *,
    schemas: AuthoritativeSchemas | None = None,
) -> list[dict[str, Any]]:
    if not events:
        return []
    result: list[dict[str, Any]] = []
    event_ids: set[str] = set()
    run_id: str | None = None
    previous_hash = ZERO_HASH
    attempt_state: dict[str, dict[str, int]] = {}
    for expected_sequence, raw_event in enumerate(events, 1):
        event = _exact_keys(
            raw_event,
            {
                "schema_revision",
                "run_id",
                "sequence",
                "event_id",
                "type",
                "timestamp_utc",
                "causes",
                "payload",
                "previous_hash",
                "event_hash",
            },
            f"event {expected_sequence}",
        )
        _validate_no_noninteger_numbers(event)
        if event["schema_revision"] != SCHEMA_REVISION:
            raise JournalCorrupt("unsupported event schema revision")
        _entity_with_prefix(event["run_id"], "run", "event run ID")
        if run_id is None:
            run_id = event["run_id"]
        elif event["run_id"] != run_id:
            raise JournalCorrupt("event run ID changed inside journal")
        if event["sequence"] != expected_sequence:
            raise JournalCorrupt("event sequence is not contiguous")
        if event["type"] not in EVENT_TYPES:
            raise JournalCorrupt("unknown event type")
        _canonical_timestamp(event["timestamp_utc"])
        causes = event["causes"]
        if not isinstance(causes, list) or causes != sorted(set(causes)):
            raise JournalCorrupt("event causes must be sorted and duplicate-free")
        if any(cause not in event_ids for cause in causes):
            raise JournalCorrupt("event cause does not reference an earlier event")
        validate_payload(
            event["type"],
            event["payload"],
            sequence=expected_sequence,
            schemas=schemas,
        )
        if schemas is not None:
            schemas.validate_event(event)
        expected_id = event_id_for(run_id, expected_sequence, event["type"], causes, event["payload"])
        if event["event_id"] != expected_id:
            raise JournalCorrupt("event ID mismatch")
        if event["event_id"] in event_ids:
            raise JournalCorrupt("duplicate event ID")
        if event["previous_hash"] != previous_hash:
            raise JournalCorrupt("event hash chain is broken")
        without_hash = dict(event)
        event_hash = without_hash.pop("event_hash")
        if event_hash != event_hash_for(without_hash):
            raise JournalCorrupt("event hash mismatch")
        event_ids.add(event["event_id"])
        previous_hash = event_hash

        event_type = event["type"]
        payload = event["payload"]
        if event_type == "model_call_intended":
            attempt_id = payload["physical_attempt_id"]
            if attempt_id in attempt_state:
                raise JournalCorrupt("physical attempt has more than one intent")
            attempt_state[attempt_id] = {"dispatch": 0, "terminal": 0}
        elif event_type == "model_dispatch_started":
            attempt_id = payload["physical_attempt_id"]
            if attempt_id not in attempt_state or attempt_state[attempt_id]["dispatch"]:
                raise JournalCorrupt("dispatch lacks exactly one prior intent")
            attempt_state[attempt_id]["dispatch"] = 1
        elif event_type in {
            "model_call_succeeded",
            "model_call_failed",
            "model_call_outcome_unknown",
        }:
            attempt_id = payload["physical_attempt_id"]
            if (
                attempt_id not in attempt_state
                or not attempt_state[attempt_id]["dispatch"]
                or attempt_state[attempt_id]["terminal"]
            ):
                raise JournalCorrupt("attempt terminality invariant violated")
            attempt_state[attempt_id]["terminal"] = 1
        result.append(dict(event))
    if result[0]["type"] != "run_created" or result[0]["payload"]["run_id"] != run_id:
        raise JournalCorrupt("journal must begin with matching run_created")
    return result


def parse_journal_bytes(
    data: bytes,
    *,
    require_final_newline: bool = True,
    schemas: AuthoritativeSchemas | None = None,
) -> list[dict[str, Any]]:
    if not data:
        return []
    if require_final_newline and not data.endswith(b"\n"):
        raise JournalCorrupt("journal is not newline-terminated")
    content = data[:-1] if data.endswith(b"\n") else data
    lines = content.split(b"\n")
    events: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line:
            raise JournalCorrupt(f"journal line {line_number} is blank")
        try:
            value = strict_json_loads(line)
        except DomainError as exc:
            raise JournalCorrupt(
                f"journal line {line_number} is invalid JSON: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise JournalCorrupt(f"journal line {line_number} is not an object")
        if canonical_json_bytes(value) != line:
            raise JournalCorrupt(f"journal line {line_number} is not canonical JSON")
        events.append(value)
    return validate_events(events, schemas=schemas)


@dataclass(frozen=True, slots=True)
class TailRecovery:
    action: str
    pending_event_hash: str
    original_journal_bytes: int
    original_pending_bytes: int
    resulting_terminal_hash: str


class EventJournal(AbstractContextManager["EventJournal"]):
    """A single-writer journal held under an OS advisory lock."""

    def __init__(
        self,
        run_directory: Path,
        *,
        expected_run_id: str | None = None,
        recovery_invocation_id: str | None = None,
        recovery_invocation_nonce: str | None = None,
        schemas: AuthoritativeSchemas | None = None,
    ) -> None:
        self.run_directory = run_directory
        self.events_path = run_directory / "events.jsonl"
        self.pending_path = run_directory / ".event.pending"
        self.pending_tmp_path = run_directory / ".event.pending.tmp"
        self.lock_path = run_directory / ".events.lock"
        self.expected_run_id = expected_run_id
        self.recovery_invocation_id = recovery_invocation_id
        self.recovery_invocation_nonce = recovery_invocation_nonce
        self.schemas = schemas
        self._lock_handle: Any = None
        self._events: list[dict[str, Any]] = []
        self._tail_recovery_deferred = False
        self._deferred_recovery_requires_audit = False

    @property
    def events(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(deepcopy(self._events))

    @property
    def run_id(self) -> str | None:
        if self._events:
            return self._events[0]["run_id"]
        return self.expected_run_id

    @property
    def terminal_hash(self) -> str:
        return self._events[-1]["event_hash"] if self._events else ZERO_HASH

    def bind_schemas(self, schemas: AuthoritativeSchemas) -> None:
        """Bind one materialized contract after a resume bootstrap read.

        Resume must first recover and read sequence one to discover the frozen
        objective.  It then materializes that objective's schemas and calls
        this method before any new event can be appended.
        """

        if self._lock_handle is None:
            raise JournalUnwritable("schema binding requires the held writer lock")
        if self.schemas is not None and self.schemas is not schemas:
            raise JournalCorrupt("journal schemas are already bound")
        if self._tail_recovery_deferred:
            # The prefix/sequence-one bootstrap was read without mutation so
            # the frozen objective could be discovered.  Validate and recover
            # the pending event only after its materialized contract is bound.
            validate_events(self._events, schemas=schemas)
            if (
                self._deferred_recovery_requires_audit
                and self.recovery_invocation_id is None
                and self.recovery_invocation_nonce is None
            ):
                raise JournalCorrupt("tail recovery requires an invocation ID")
            self.schemas = schemas
            recovery = self._reconcile_tail()
            data = self.events_path.read_bytes() if self.events_path.exists() else b""
            self._events = parse_journal_bytes(data, schemas=schemas)
            self._validate_expected_run_id()
            self._tail_recovery_deferred = False
            self._deferred_recovery_requires_audit = False
            if recovery is not None:
                self._record_recovery(recovery)
            return
        validate_events(self._events, schemas=schemas)
        self.schemas = schemas

    def __enter__(self) -> "EventJournal":
        try:
            self.run_directory.mkdir(parents=False, exist_ok=True)
            self._lock_handle = self.lock_path.open("a+b")
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            if self._lock_handle is not None:
                self._lock_handle.close()
                self._lock_handle = None
            raise JournalBusy(f"run journal is already locked: {self.run_directory}") from exc
        except OSError as exc:
            if self._lock_handle is not None:
                self._lock_handle.close()
                self._lock_handle = None
            raise JournalUnwritable(f"cannot acquire journal lock: {exc}") from exc
        try:
            bootstrap = (
                self._deferred_tail_bootstrap()
                if self.schemas is None
                else None
            )
            if bootstrap is not None:
                self._events, self._deferred_recovery_requires_audit = bootstrap
                self._tail_recovery_deferred = True
                recovery = None
            else:
                recovery = self._reconcile_tail()
                data = self.events_path.read_bytes() if self.events_path.exists() else b""
                self._events = parse_journal_bytes(data, schemas=self.schemas)
            self._validate_expected_run_id()
            if recovery is not None:
                self._record_recovery(recovery)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._lock_handle is not None:
            try:
                fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
            finally:
                self._lock_handle.close()
                self._lock_handle = None

    def _fsync_directory(self) -> None:
        fd = os.open(self.run_directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _append_exact_bytes(self, event_bytes: bytes) -> None:
        try:
            with self.events_path.open("ab", buffering=0) as handle:
                self._write_all(handle, event_bytes)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise JournalUnwritable(f"cannot append event: {exc}") from exc

    @staticmethod
    def _write_all(handle: Any, content: bytes) -> None:
        remaining = memoryview(content)
        while remaining:
            written = handle.write(remaining)
            if written is None or written <= 0:
                raise OSError("short write while persisting journal evidence")
            remaining = remaining[written:]

    def _validate_expected_run_id(self) -> None:
        if (
            self.expected_run_id is not None
            and self._events
            and self.run_id != self.expected_run_id
        ):
            raise JournalCorrupt("journal run ID differs from expected run ID")

    def _record_recovery(self, recovery: TailRecovery) -> None:
        recovery_invocation_id = self.recovery_invocation_id
        if recovery_invocation_id is None and self.recovery_invocation_nonce is not None:
            if self.run_id is None:
                raise JournalCorrupt("recovered journal has no run ID")
            recovery_invocation_id = entity_id(
                "iv", [self.run_id, self.recovery_invocation_nonce]
            )
        if recovery_invocation_id is None:
            raise JournalCorrupt("tail recovery requires an invocation ID")
        self.append(
            "journal_tail_recovered",
            {
                "recovery_invocation_id": recovery_invocation_id,
                "pending_event_hash": recovery.pending_event_hash,
                "original_journal_bytes": recovery.original_journal_bytes,
                "original_pending_bytes": recovery.original_pending_bytes,
                "action": recovery.action,
                "resulting_terminal_hash": recovery.resulting_terminal_hash,
            },
            causes=(self._events[-1]["event_id"],) if self._events else (),
        )

    def _deferred_tail_bootstrap(
        self,
    ) -> tuple[list[dict[str, Any]], bool] | None:
        """Validate a pending relationship without changing authoritative bytes."""

        if not self.pending_path.exists():
            return None
        journal = self.events_path.read_bytes() if self.events_path.exists() else b""
        pending = self.pending_path.read_bytes()
        if not pending.endswith(b"\n") or pending.count(b"\n") != 1:
            raise JournalCorrupt("pending sidecar is not exactly one complete event")
        pending_line = pending[:-1]
        try:
            pending_value = strict_json_loads(pending_line)
        except DomainError as exc:
            raise JournalCorrupt(f"pending event is invalid JSON: {exc}") from exc
        if (
            not isinstance(pending_value, dict)
            or canonical_json_bytes(pending_value) != pending_line
        ):
            raise JournalCorrupt("pending event is not canonical")

        if journal.endswith(b"\n") or not journal:
            events = parse_journal_bytes(journal)
            if events and canonical_json_bytes(events[-1]) + b"\n" == pending:
                return events, False
            combined_events = parse_journal_bytes(journal + pending)
            if len(combined_events) != len(events) + 1:
                raise JournalCorrupt("pending event is not the exact next journal event")
            return events or combined_events[:1], True

        last_newline = journal.rfind(b"\n")
        prefix = journal[: last_newline + 1] if last_newline >= 0 else b""
        fragment = journal[last_newline + 1 :]
        prefix_events = parse_journal_bytes(prefix)
        if not fragment or not pending_line.startswith(fragment):
            raise JournalCorrupt("partial journal tail is not a prefix of pending event")
        combined_events = parse_journal_bytes(prefix + pending)
        if len(combined_events) != len(prefix_events) + 1:
            raise JournalCorrupt("pending event does not extend the validated journal prefix")
        return prefix_events or combined_events[:1], True

    def _reconcile_tail(self) -> TailRecovery | None:
        journal = self.events_path.read_bytes() if self.events_path.exists() else b""
        pending = self.pending_path.read_bytes() if self.pending_path.exists() else None
        original_journal_bytes = len(journal)
        original_pending_bytes = len(pending) if pending is not None else 0

        if pending is None:
            # A temp file was never promoted and is not authoritative.  Remove
            # it only after proving the journal itself is clean.
            parse_journal_bytes(journal, schemas=self.schemas)
            if self.pending_tmp_path.exists():
                self.pending_tmp_path.unlink()
                self._fsync_directory()
            return None

        if not pending.endswith(b"\n") or pending.count(b"\n") != 1:
            raise JournalCorrupt("pending sidecar is not exactly one complete event")
        pending_line = pending[:-1]
        try:
            pending_value = strict_json_loads(pending_line)
        except DomainError as exc:
            raise JournalCorrupt(f"pending event is invalid JSON: {exc}") from exc
        if not isinstance(pending_value, dict) or canonical_json_bytes(pending_value) != pending_line:
            raise JournalCorrupt("pending event is not canonical")

        if journal.endswith(b"\n") or not journal:
            events = parse_journal_bytes(journal, schemas=self.schemas)
            if events and canonical_json_bytes(events[-1]) + b"\n" == pending:
                self.pending_path.unlink()
                self._fsync_directory()
                # The event was already durable. Removing its leftover
                # sidecar is cleanup, not an authoritative journal repair;
                # in particular a nonresumable run remains byte-for-byte
                # read-only under ``--resume``.
                return None
            combined = journal + pending
            combined_events = parse_journal_bytes(combined, schemas=self.schemas)
            if len(combined_events) != len(events) + 1:
                raise JournalCorrupt("pending event is not the exact next journal event")
            self._append_exact_bytes(pending)
            self.pending_path.unlink()
            self._fsync_directory()
            return TailRecovery(
                "pending_appended",
                pending_value["event_hash"],
                original_journal_bytes,
                original_pending_bytes,
                pending_value["event_hash"],
            )

        last_newline = journal.rfind(b"\n")
        prefix = journal[: last_newline + 1] if last_newline >= 0 else b""
        fragment = journal[last_newline + 1 :]
        prefix_events = parse_journal_bytes(prefix, schemas=self.schemas)
        if not fragment or not pending_line.startswith(fragment):
            raise JournalCorrupt("partial journal tail is not a prefix of pending event")
        combined = prefix + pending
        combined_events = parse_journal_bytes(combined, schemas=self.schemas)
        if len(combined_events) != len(prefix_events) + 1:
            raise JournalCorrupt("pending event does not extend the validated journal prefix")
        try:
            with self.events_path.open("r+b", buffering=0) as handle:
                handle.truncate(len(prefix))
                handle.flush()
                os.fsync(handle.fileno())
                handle.seek(0, os.SEEK_END)
                self._write_all(handle, pending)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise JournalUnwritable(f"cannot recover partial tail: {exc}") from exc
        self.pending_path.unlink()
        self._fsync_directory()
        return TailRecovery(
            "partial_tail_replaced",
            pending_value["event_hash"],
            original_journal_bytes,
            original_pending_bytes,
            pending_value["event_hash"],
        )

    def append(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        causes: Iterable[str] = (),
        timestamp_utc: str | None = None,
    ) -> dict[str, Any]:
        if self._lock_handle is None:
            raise JournalUnwritable("journal append requires the held writer lock")
        if self._tail_recovery_deferred:
            raise JournalUnwritable(
                "journal append requires materialized schema binding after pending recovery"
            )
        run_id = self.run_id
        if run_id is None:
            if event_type != "run_created":
                raise JournalCorrupt("first event must be run_created")
            run_id = payload.get("run_id")
        if not isinstance(run_id, str):
            raise JournalCorrupt("run ID is unavailable")
        event = build_event(
            run_id=run_id,
            sequence=len(self._events) + 1,
            event_type=event_type,
            payload=payload,
            previous_hash=self.terminal_hash,
            causes=tuple(causes),
            timestamp_utc=timestamp_utc,
            schemas=self.schemas,
        )
        event_bytes = canonical_json_bytes(event) + b"\n"
        try:
            with self.pending_tmp_path.open("xb", buffering=0) as temp:
                self._write_all(temp, event_bytes)
                temp.flush()
                os.fsync(temp.fileno())
            os.replace(self.pending_tmp_path, self.pending_path)
            self._fsync_directory()
            self._append_exact_bytes(event_bytes)
            self.pending_path.unlink()
            self._fsync_directory()
        except FileExistsError as exc:
            raise JournalCorrupt("an unexpected pending temp file exists") from exc
        except OSError as exc:
            raise JournalUnwritable(f"durable event append failed: {exc}") from exc
        self._events.append(event)
        return deepcopy(event)


def read_events(
    run_directory: Path,
    *,
    schemas: AuthoritativeSchemas | None = None,
    through_sequence: int | None = None,
) -> list[dict[str, Any]]:
    """Read and validate a complete journal or one authoritative prefix.

    A prefix read stops after ``through_sequence`` physical events.  It still
    validates canonical bytes, schemas, sequence numbers, and the hash chain
    for every returned event.  Callers that cite an immutable checkpoint need
    not read unrelated history appended after that checkpoint.
    """

    if through_sequence is not None and (
        isinstance(through_sequence, bool)
        or not isinstance(through_sequence, int)
        or through_sequence < 1
    ):
        raise ValueError("through_sequence must be a positive integer")
    path = run_directory / "events.jsonl"
    try:
        if through_sequence is None:
            data = path.read_bytes()
        else:
            chunks: list[bytes] = []
            with path.open("rb") as handle:
                for _ in range(through_sequence):
                    line = handle.readline()
                    if not line:
                        break
                    chunks.append(line)
            data = b"".join(chunks)
        return parse_journal_bytes(data, schemas=schemas)
    except OSError as exc:
        raise JournalCorrupt(f"cannot read events.jsonl: {exc}") from exc
