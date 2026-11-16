"""Small, engine-independent fixtures shared by foundation unit tests."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping
from unittest.mock import patch

from quasilisk.domain import (
    ActionPurpose,
    ChatMessage,
    ChatRequest,
    DECISION_CONSTANTS,
    Role,
    digest_json,
    entity_id,
    sha256_bytes,
)
from quasilisk.events import ZERO_HASH, build_event
from quasilisk.provider import encode_bytes, render_request_body, requested_provider_route_key


FIXED_TIMESTAMP = "2026-08-18T12:00:00.000000Z"
FOUNDATION_SCOPE = digest_json(["foundation-test", "scope"])
FOUNDATION_RESPONSE_SCHEMA = {"type": "object"}
FOUNDATION_RESPONSE_SCHEMA_DIGEST = digest_json(FOUNDATION_RESPONSE_SCHEMA)
FOUNDATION_SYSTEM_TEXT = {
    "judge_system": "foundation judge system",
    "selector_system": "foundation selector system",
    "perturbator_system": "foundation perturbator system",
    "probe_review_system": "foundation probe review system",
    "template_binding_system": "foundation template binding system",
    "outcome_review_system": "foundation outcome review system",
}
FOUNDATION_MODELS = {
    "target": {
        "model": "model/target",
        "temperature": "0",
        "max_output_tokens": 32,
        "system_prompt": "",
    },
    "judges": {
        "primary": {
            "model": "model/judge",
            "temperature": "0",
            "max_output_tokens": 32,
        },
        "secondaries": [
            {
                "model": "model/judge-secondary-a",
                "temperature": "0",
                "max_output_tokens": 32,
            },
            {
                "model": "model/judge-secondary-b",
                "temperature": "0",
                "max_output_tokens": 32,
            },
        ],
        "replacement": {
            "model": "model/judge-replacement",
            "temperature": "0",
            "max_output_tokens": 32,
        },
    },
    "selector": {
        "model": "model/selector",
        "fallback_model": "model/selector-fallback",
        "temperature": "0",
        "max_output_tokens": 32,
    },
    "perturbator": {
        "model": "model/perturbator",
        "fallback_model": "model/perturbator-fallback",
        "temperature": "0",
        "max_output_tokens": 32,
    },
}


def digest(label: str) -> str:
    return digest_json(["foundation-test", label])


def run_id(label: str = "run") -> str:
    return entity_id("run", ["foundation-test", label])


def invocation_id(label: str = "invocation") -> str:
    return entity_id("iv", ["foundation-test", label])


def manifest(
    *,
    target_budget: int = 4,
    judge_budget: int = 8,
    selector_budget: int = 4,
    perturbator_budget: int = 4,
    scope_label: str = "scope",
) -> dict[str, Any]:
    del scope_label
    scope = FOUNDATION_SCOPE
    normalized_config = {
        "provider": {
            "adapter": "scripted",
            "script_path": "foundation.jsonl",
            "timeout_seconds": 1,
        },
        "models": deepcopy(FOUNDATION_MODELS),

    }
    return {
        "normalized_config": normalized_config,
        "physical_call_limits": {"target": target_budget, "judge": judge_budget, "selector": selector_budget, "perturbator": perturbator_budget},
        "config_digest": digest_json(normalized_config),
        "scientific_scope_digest": scope,
        "engine": {},
        "objective": {},
        "bootstrap_snapshot": [],
        "seed_snapshot": [],
        "strategy_template_snapshot": [],
        "library_source_checkpoints": [],
        "model_profiles": deepcopy(FOUNDATION_MODELS),
        "role_template_digests": {
            name: sha256_bytes(
                FOUNDATION_SYSTEM_TEXT.get(name, f"foundation {name}").encode("utf-8")
            )
            for name in (
                "judge_system",
                "judge_user",
                "selector_system",
                "selector_user",
                "perturbator_system",
                "perturbator_user",
                "probe_review_system",
                "probe_review_user",
                "template_binding_system",
                "template_binding_user",
                "outcome_review_system",
                "outcome_review_user",
            )
        },
        "response_schema_digests": {
            name: FOUNDATION_RESPONSE_SCHEMA_DIGEST
            for name in (
                "judge_output",
                "selector_portfolio",
                "probe_review",
                "template_binding",
                "outcome_review",
                "proposal_batch",
            )
        },
        "judge_output_schema": deepcopy(FOUNDATION_RESPONSE_SCHEMA),
        "descriptor_taxonomy": {
            "descriptors": [],
            "digest": digest_json([]),
        },
        "decision_constants": dict(DECISION_CONSTANTS),
        "event_schema_digest": digest("event-schema"),
        "work_unit_schema_digest": digest("work-unit-schema"),
        "reducer_digest": digest("reducer"),
        "random_seed": 7,
        "scripted_fixture_digest": None,
    }


def run_created_payload(
    *,
    label: str = "run",
    target_budget: int = 4,
    judge_budget: int = 8,
    selector_budget: int = 4,
    perturbator_budget: int = 4,
) -> dict[str, Any]:
    frozen_manifest = manifest(
        target_budget=target_budget,
        judge_budget=judge_budget,
        selector_budget=selector_budget,
        perturbator_budget=perturbator_budget,
        scope_label=f"{label}-scope",
    )
    identifier = entity_id(
        "run",
        [
            "0" * 32,
            frozen_manifest["scientific_scope_digest"],
            frozen_manifest["config_digest"],
        ],
    )
    return {
        "manifest": frozen_manifest,
        "scientific_scope_digest": frozen_manifest["scientific_scope_digest"],
        "run_nonce_hex": "0" * 32,
        "run_id": identifier,
        "output_base": "/tmp/quasilisk-foundation-tests",
        "run_path_at_creation": f"/tmp/quasilisk-foundation-tests/{identifier}",
        "creation_invocation_id": invocation_id(f"{label}-creation"),
    }


def call_fixture(
    *,
    label: str = "call",
    role: str = "target",
    maximum_attempts: int = 2,
) -> dict[str, Any]:
    purpose_by_role = {
        "target": "target_sample",
        "judge": "judge_member",
        "selector": "selector_portfolio",
        "perturbator": "perturbator_proposal",
    }
    unit_identifier = entity_id("wu", ["foundation-test", label, "unit"])
    action_identifier = entity_id("la", ["foundation-test", label, "action"])
    subject_identifier = entity_id("pc", ["foundation-test", label, "subject"])
    candidate_identifier = entity_id("pc", ["foundation-test", label, "candidate"])
    owner_key = digest(f"{label}-owner")
    action_key = digest_json([owner_key, "action", role, purpose_by_role[role], 0])
    configured_slot = {
        "target": "target",
        "judge": "primary",
        "selector": "selector_primary",
        "perturbator": "perturbator_primary",
    }[role]
    route_label = "primary"
    provider_key = requested_provider_route_key("scripted", role, route_label)
    requested_model = f"model/{role}"
    generation_profile = {
        "model": requested_model,
        "temperature": "0",
        "max_output_tokens": 32,
        "system_prompt": "",
    }
    generation_digest = digest_json(generation_profile)
    route_key = digest_json(
        [
            FOUNDATION_SCOPE,
            "role_route",
            role,
            configured_slot,
            route_label,
            provider_key,
            requested_model,
            generation_digest,
        ]
    )
    slot = {
        "action_id": action_identifier,
        "action_scientific_key": action_key,
        "ordinal": 0,
        "owner_position_scientific_key": owner_key,
        "role": role,
        "purpose": purpose_by_role[role],
        "dependency_action_ids": [],
        "maximum_physical_attempts": maximum_attempts,
        "routes": [
            {
                "route_key": route_key,
                "route_label": route_label,
                "configured_slot": configured_slot,
                "requested_provider_route_key": provider_key,
                "requested_model": requested_model,
                "generation_profile_digest": generation_digest,
                "maximum_attempts": maximum_attempts,
            }
        ],
        "guard": {"kind": "always"},
    }
    action_slots = [slot]
    kind = "initial_observation"
    design: dict[str, Any] = {"tag": kind}
    worst = {
        "target": maximum_attempts if role == "target" else 0,
        "judge": maximum_attempts if role == "judge" else 0,
        "selector": maximum_attempts if role == "selector" else 0,
        "perturbator": maximum_attempts if role == "perturbator" else 0,
    }
    if role == "target":
        # Initial-observation conformance has one target and a conservative
        # four-slot judge panel, whether or not its conditional slots dispatch.
        for ordinal in range(1, 5):
            filler_route = digest(f"{label}-judge-route-{ordinal}")
            action_slots.append(
                {
                    "action_id": entity_id("la", ["foundation-test", label, "judge", ordinal]),
                    "action_scientific_key": digest_json(
                        [
                            owner_key,
                            "action",
                            "judge",
                            "judge_member",
                            ordinal,
                        ]
                    ),
                    "ordinal": ordinal,
                    "owner_position_scientific_key": owner_key,
                    "role": "judge",
                    "purpose": "judge_member",
                    "dependency_action_ids": [action_identifier] if ordinal == 1 else [],
                    "maximum_physical_attempts": 2,
                    "routes": [{
                        "route_key": filler_route,
                        "route_label": "primary" if ordinal < 4 else "replacement",
                        "configured_slot": ("primary", "secondary_a", "secondary_b", "replacement")[ordinal - 1],
                        "requested_provider_route_key": digest(f"{label}-judge-provider-{ordinal}"),
                        "requested_model": f"model/judge-{ordinal}",
                        "generation_profile_digest": digest(f"{label}-judge-generation-{ordinal}"),
                        "maximum_attempts": 2,
                    }],
                    "guard": {"kind": "always"},
                }
            )
        worst = {"target": 2, "judge": 8, "selector": 0, "perturbator": 0}
    else:
        # These reducer fixtures deliberately reuse the ordinary work-unit
        # envelope. Semantic conformance of real units is covered separately;
        # no synthetic calibration work kind is needed for role accounting.
        design = {"tag": kind}
    unit = {
        "unit_id": unit_identifier,
        "unit_scientific_key": digest(f"{label}-unit"),
        "kind": kind,
        "subject_id": subject_identifier,
        "subject_scientific_key": digest(f"{label}-subject"),
        "parent_unit_id": None,
        "policy_digest": digest(f"{label}-policy"),
        "design": design,
        "proposal_slots": [],
        "execution_slots": [],
        "panel_plans": [],
        "logical_action_slots": action_slots,
        "worst_case_calls": worst,
    }
    return {
        "label": label,
        "role": role,
        "purpose": purpose_by_role[role],
        "unit": unit,
        "unit_id": unit_identifier,
        "action_id": action_identifier,
        "action_key": action_key,
        "owner_key": owner_key,
        "route_key": route_key,
        "provider_key": provider_key,
        "requested_model": requested_model,
        "candidate_id": candidate_identifier,
    }


def reduce_fixture_events(events: Iterable[Mapping[str, Any]], *args: Any, **kwargs: Any) -> Any:
    """Reduce minimal protocol fixtures without treating them as product designs.

    Foundation reducer tests intentionally freeze one-action units so they can
    isolate dispatch/replay invariants.  Closed WorkUnit semantic conformance is
    covered against real engine builders in ``test_scheduler_conformance``.
    The arithmetic recomputation remains active here.
    """

    from quasilisk.run_state import (
        SchedulerValidationError,
        action_slot_maximum,
        reduce_events,
    )

    def validate_fixture_maximum(
        unit: Any, *, expected_policy_digest: str | None = None
    ) -> Any:
        del expected_policy_digest
        calculated = action_slot_maximum(unit.logical_action_slots)
        if calculated != unit.worst_case_calls:
            raise SchedulerValidationError(
                "recorded work-unit maximum is incorrect"
            )
        return calculated

    with patch(
        "quasilisk.run_state.validate_work_unit_maximum",
        side_effect=validate_fixture_maximum,
    ):
        return reduce_events(events, *args, **kwargs)


def attempt_id(fixture: Mapping[str, Any], ordinal: int) -> str:
    route_kind = "primary" if ordinal == 0 else "exact_retry"
    return entity_id("pa", [fixture["action_id"], route_kind, ordinal])


def intended_payload(
    fixture: Mapping[str, Any],
    *,
    ordinal: int = 0,
    parent_attempt_id: str | None = None,
    physical_attempt_id: str | None = None,
) -> dict[str, Any]:
    attempt = physical_attempt_id or attempt_id(fixture, ordinal)
    if fixture["role"] == "target":
        messages = [{"role": "user", "content": "test prompt"}]
        response_mode = "text"
        response_schema = None
        response_schema_digest = None
    else:
        template_name = {
            "judge": "judge_system",
            "selector": "selector_system",
            "perturbator": "perturbator_system",
        }[fixture["role"]]
        messages = [
            {"role": "system", "content": FOUNDATION_SYSTEM_TEXT[template_name]},
            {"role": "user", "content": "test prompt"},
        ]
        response_mode = "json_schema"
        response_schema = FOUNDATION_RESPONSE_SCHEMA
        response_schema_digest = FOUNDATION_RESPONSE_SCHEMA_DIGEST
    generation = {"temperature": "0", "max_output_tokens": 32}
    request = ChatRequest(
        logical_action_id=fixture["action_id"],
        logical_action_scientific_key=fixture["action_key"],
        physical_attempt_id=attempt,
        attempt_ordinal=ordinal,
        role=Role(fixture["role"]),
        purpose=ActionPurpose(fixture["purpose"]),
        route_kind="primary" if ordinal == 0 else "exact_retry",
        role_route_key=fixture["route_key"],
        requested_provider_route_key=fixture["provider_key"],
        messages=tuple(ChatMessage(item["role"], item["content"]) for item in messages),
        requested_model=fixture["requested_model"],
        generation_parameters=generation,
        response_mode=response_mode,
        response_schema=response_schema,
        response_schema_digest=response_schema_digest,
        timeout_seconds=1,
        idempotency_key=None,
    )
    return {
        "logical_action_id": fixture["action_id"],
        "logical_action_scientific_key": fixture["action_key"],
        "physical_attempt_id": attempt,
        "attempt_ordinal": ordinal,
        "unit_id": fixture["unit_id"],
        "owner_position_scientific_key": fixture["owner_key"],
        "role": fixture["role"],
        "purpose": fixture["purpose"],
        "role_route_key": fixture["route_key"],
        "route_kind": "primary" if ordinal == 0 else "exact_retry",
        "parent_attempt_id": parent_attempt_id,
        "messages": messages,
        "request_body": encode_bytes(render_request_body(request)).to_dict(),
        "semantic_request_hash": request.semantic_request_hash,
        "requested_provider_route_key": fixture["provider_key"],
        "requested_model": fixture["requested_model"],
        "generation_parameters": generation,
        "response_mode": response_mode,
        "response_schema_digest": response_schema_digest,
        "timeout_seconds": 1,
        "idempotency_key": None,
        "lineage": {
            "execution_id": None,
            "conversation_root_id": None,
            "parent_node_id": None,
            "candidate_id": fixture["candidate_id"]
            if fixture["role"] == "target"
            else None,
            "panel_id": None,
            "logical_member_slot": None,
            "strategy_id": None,
            "proposal_slot_scientific_key": None,
            "review_item_id": None,
        },
    }


def dispatch_payload(
    fixture: Mapping[str, Any], *, ordinal: int = 0
) -> dict[str, Any]:
    return {
        "physical_attempt_id": attempt_id(fixture, ordinal),
        "logical_action_id": fixture["action_id"],
        "role_route_key": fixture["route_key"],
        "route_kind": "primary" if ordinal == 0 else "exact_retry",
        "provider_route_key": fixture["provider_key"],
        "requested_model": fixture["requested_model"],
        "charged_role": fixture["role"],
        "charge": 1,
    }


def success_payload(
    fixture: Mapping[str, Any],
    *,
    ordinal: int = 0,
    text: str = "model output",
    effective_model_id: str | None = "effective/model",
) -> dict[str, Any]:
    return {
        "physical_attempt_id": attempt_id(fixture, ordinal),
        "raw_body": {"encoding": "utf8", "data": "{}"},
        "text": text,
        "finish_status": "complete",
        "effective_model_id": effective_model_id,
        "provider_request_id": "provider-request",
        "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
        "latency_ms": 1,
        "cost": "0",
        "metadata": {"http_status": 200, "content_type": "application/json"},
    }


def failure_payload(
    fixture: Mapping[str, Any], *, ordinal: int = 0
) -> dict[str, Any]:
    return {
        "physical_attempt_id": attempt_id(fixture, ordinal),
        "error_kind": "timeout",
        "delivery_state": "known_no_output",
        "detail": "transport_timeout",
        "provider_request_id": None,
        "raw_body": None,
        "http_status": None,
        "retry_after_seconds": None,
    }


def unknown_payload(
    fixture: Mapping[str, Any],
    *,
    ordinal: int = 0,
    discovery_invocation_id: str | None = None,
) -> dict[str, Any]:
    return {
        "physical_attempt_id": attempt_id(fixture, ordinal),
        "discovery_invocation_id": discovery_invocation_id
        or invocation_id(f"{fixture['label']}-creation"),
        "error_kind": "timeout",
        "detail": "outcome_unknown_after_dispatch",
        "provider_request_id": None,
        "raw_body": None,
    }


def chain_events(
    created_payload: Mapping[str, Any],
    records: Iterable[tuple[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    identifier = created_payload["run_id"]
    events = [
        build_event(
            run_id=identifier,
            sequence=1,
            event_type="run_created",
            payload=created_payload,
            previous_hash=ZERO_HASH,
            timestamp_utc=FIXED_TIMESTAMP,
        )
    ]
    for event_type, payload in records:
        causes: tuple[str, ...] = ()
        if event_type == "model_dispatch_started":
            attempt = payload["physical_attempt_id"]
            source = next(
                (
                    event
                    for event in reversed(events)
                    if event["type"] == "model_call_intended"
                    and event["payload"]["physical_attempt_id"] == attempt
                ),
                None,
            )
            causes = () if source is None else (source["event_id"],)
        elif event_type in {
            "model_call_succeeded",
            "model_call_failed",
            "model_call_outcome_unknown",
        }:
            attempt = payload["physical_attempt_id"]
            source = next(
                (
                    event
                    for event in reversed(events)
                    if event["type"] == "model_dispatch_started"
                    and event["payload"]["physical_attempt_id"] == attempt
                ),
                None,
            )
            causes = () if source is None else (source["event_id"],)
        elif event_type in {"run_resumed", "run_terminated", "journal_tail_recovered"}:
            causes = (events[-1]["event_id"],)
        events.append(
            build_event(
                run_id=identifier,
                sequence=len(events) + 1,
                event_type=event_type,
                payload=payload,
                previous_hash=events[-1]["event_hash"],
                causes=causes,
                timestamp_utc=FIXED_TIMESTAMP,
            )
        )
    return events


def judge_procedure_payload(verdict, application, evidence):
    """Explicit proof objects for trusted scripted assessments, not runtime fallback."""
    from copy import deepcopy
    positive = verdict != "not_demonstrated"
    selected = deepcopy(evidence[:1])
    return {
        "premise": {"status": "adopted" if positive else "absent", "evidence": deepcopy(selected) if positive else []},
        "inference": {"status": "endorsed" if positive else "rejected", "evidence": deepcopy(selected)},
        "commitment": {"status": "adopted" if positive and application == "action_governing" else "absent", "evidence": deepcopy(selected) if positive and application == "action_governing" else []},
        "reservation": {"status": "none", "evidence": []},
    }


def judge_schema(reply):
    import json
    from pathlib import Path
    from quasilisk.judging import materialize_judge_schema
    template = json.loads((Path(__file__).resolve().parents[1]/"schemas/judge_output.json").read_text())
    return materialize_judge_schema(template, reply)


def selector_applicability_payload(candidate=None):
    if isinstance(candidate, dict) and isinstance(candidate.get("applicability"), dict):
        return {key: candidate["applicability"][key] for key in ("requires", "excludes")}
    return {"requires": "Fresh evidence for the declared source transition.", "excludes": "The declared source opening is absent or withdrawn."}
