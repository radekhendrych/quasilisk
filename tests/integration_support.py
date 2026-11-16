"""Helpers for bounded end-to-end tests of the canonical engine."""

from __future__ import annotations

from collections import Counter
from decimal import Decimal
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from quasilisk.domain import strict_json_loads
from quasilisk.events import read_events


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
PUBLIC_RUN_FILES = frozenset(
    {"events.jsonl", "manifest.json", "archive.json", "summary.json", "summary.md"}
)
TERMINAL_CALL_EVENT_TYPES = frozenset(
    {"model_call_succeeded", "model_call_failed", "model_call_outcome_unknown"}
)


def dryrun_config(name: str, output_base: Path) -> dict[str, Any]:
    """Load a checked-in config without introducing binary floats."""

    value = strict_json_loads(
        (REPOSITORY_ROOT / "configs" / f"dryrun_{name}.json").read_bytes()
    )
    assert isinstance(value, dict)
    value["output_directory"] = output_base.as_posix()
    return value


def write_config(path: Path, value: Mapping[str, Any]) -> None:
    def source_json(item: Any) -> Any:
        if isinstance(item, Decimal):
            return int(item) if item == item.to_integral() else float(item)
        if isinstance(item, Mapping):
            return {key: source_json(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [source_json(child) for child in item]
        return item

    path.write_text(
        json.dumps(
            source_json(dict(value)),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def parsed_json(path: Path) -> Mapping[str, Any]:
    value = strict_json_loads(path.read_bytes())
    assert isinstance(value, dict)
    return value


def assert_complete_call_audit(testcase: Any, events: Sequence[Mapping[str, Any]]) -> None:
    """Assert the physical-call ordering and one-charge/one-terminal contract."""

    testcase.assertGreaterEqual(len(events), 2)
    testcase.assertEqual(events[0]["type"], "run_created")
    testcase.assertEqual(events[-1]["type"], "run_terminated")

    positions = {event["event_id"]: index for index, event in enumerate(events)}
    intents: dict[str, Mapping[str, Any]] = {}
    dispatches: dict[str, Mapping[str, Any]] = {}
    terminals: dict[str, list[Mapping[str, Any]]] = {}
    work_starts: dict[str, Mapping[str, Any]] = {}
    for event in events:
        payload = event["payload"]
        if event["type"] == "work_unit_started":
            work_starts[payload["unit"]["unit_id"]] = event
        elif event["type"] == "model_call_intended":
            attempt_id = payload["physical_attempt_id"]
            testcase.assertNotIn(attempt_id, intents)
            intents[attempt_id] = event
        elif event["type"] == "model_dispatch_started":
            attempt_id = payload["physical_attempt_id"]
            testcase.assertNotIn(attempt_id, dispatches)
            dispatches[attempt_id] = event
        elif event["type"] in TERMINAL_CALL_EVENT_TYPES:
            terminals.setdefault(payload["physical_attempt_id"], []).append(event)

    testcase.assertEqual(set(intents), set(dispatches))
    testcase.assertEqual(set(intents), set(terminals))
    for attempt_id, intent in intents.items():
        dispatch = dispatches[attempt_id]
        terminal_events = terminals[attempt_id]
        testcase.assertEqual(len(terminal_events), 1)
        terminal = terminal_events[0]
        testcase.assertLess(positions[intent["event_id"]], positions[dispatch["event_id"]])
        testcase.assertLess(positions[dispatch["event_id"]], positions[terminal["event_id"]])
        testcase.assertEqual(dispatch["causes"], [intent["event_id"]])
        testcase.assertEqual(terminal["causes"], [dispatch["event_id"]])
        testcase.assertEqual(dispatch["payload"]["charge"], 1)
        testcase.assertEqual(
            dispatch["payload"]["charged_role"], intent["payload"]["role"]
        )
        testcase.assertEqual(
            dispatch["payload"]["logical_action_id"],
            intent["payload"]["logical_action_id"],
        )
        unit_start = work_starts[intent["payload"]["unit_id"]]
        testcase.assertLess(positions[unit_start["event_id"]], positions[intent["event_id"]])

    # A panel member can only be intended after its corresponding target result
    # has become durable.  Resolve the relationship through the frozen panel
    # plan rather than assuming adjacent events.
    judge_to_target: dict[str, str] = {}
    for start in work_starts.values():
        for panel in start["payload"]["unit"]["panel_plans"]:
            target_action = panel["target_action_id"]
            for member in panel["normal_members"]:
                judge_to_target[member["judge_action_id"]] = target_action
                if panel["replacement"] is not None:
                    judge_to_target[panel["replacement"]["judge_action_id"]] = target_action
    terminal_by_action: dict[str, Mapping[str, Any]] = {}
    for attempt_id, terminal_events in terminals.items():
        action_id = intents[attempt_id]["payload"]["logical_action_id"]
        terminal_by_action[action_id] = terminal_events[0]
    for intent in intents.values():
        if intent["payload"]["role"] != "judge":
            continue
        target_action = judge_to_target[intent["payload"]["logical_action_id"]]
        target_terminal = terminal_by_action[target_action]
        testcase.assertLess(
            positions[target_terminal["event_id"]], positions[intent["event_id"]]
        )


def dispatch_role_counts(events: Sequence[Mapping[str, Any]]) -> Counter[str]:
    return Counter(
        event["payload"]["charged_role"]
        for event in events
        if event["type"] == "model_dispatch_started"
    )


def read_validated_events(run_directory: Path) -> list[dict[str, Any]]:
    return read_events(run_directory)
