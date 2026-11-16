"""Trusted in-repository objective discovery and conformance.

Objective plugins are code-reviewed Python, not sandboxed extensions.  This
module nevertheless keeps their interface narrow: discovery never imports all
plugins, and only the selected plugin's two pure callbacks are loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import inspect
from pathlib import Path
import re
import sys
from typing import Any, Callable, Iterable, Mapping

from .domain import (
    STRATEGY_MOVE_ACTS,
    STRATEGY_OPERATIONS,
    OBJECTIVE_ID_RE,
    STRATEGY_PURPOSES,
    STRATEGY_TARGET_KINDS,
    DomainError,
    canonical_json_bytes,
    digest_json,
    sha256_bytes,
    strict_json_loads,
    normalize_line_endings,
)


OBJECTIVE_FILES = (
    "plugin.py",
    "objective.json",
    "bootstrap.json",
    "judge_fixtures.json",
    "descriptor_fixtures.json",
    "diagnosis_fixtures.json",
    "selector_fixtures.json",
)

COMMON_ARGUMENT_MOVES = (
    "establish_premise",
    "test_inference",
    "answer_objection",
    "complete_commitment",
    "apply_commitment",
    "other",
)
COMMON_APPLICATION_SCOPES = (
    "abstract",
    "reasoning_policy",
    "concrete_action",
    "other",
)

class ObjectiveError(DomainError):
    """An objective artifact or callback violated the generic protocol."""


SelectorFixtureSemanticValidator = Callable[
    [Mapping[str, Any], Mapping[str, tuple[str, ...]], tuple[str, ...]],
    None,
]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = strict_json_loads(path.read_bytes())
    except OSError as exc:
        raise ObjectiveError(f"cannot read objective artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ObjectiveError(f"{path.name} must contain one JSON object")
    return value


def _closed(value: Mapping[str, Any], fields: Iterable[str], label: str) -> None:
    expected = set(fields)
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        bits: list[str] = []
        if missing:
            bits.append("missing " + ", ".join(missing))
        if extra:
            bits.append("unknown " + ", ".join(extra))
        raise ObjectiveError(f"{label}: {'; '.join(bits)}")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ObjectiveError(f"{label} must be a nonempty string")
    return value


def _identifier(value: Any, label: str) -> str:
    text = _text(value, label)
    if OBJECTIVE_ID_RE.fullmatch(text) is None:
        raise ObjectiveError(f"{label} is not a valid identifier")
    return text


def _list(value: Any, label: str, *, nonempty: bool = True) -> list[Any]:
    if not isinstance(value, list) or (nonempty and not value):
        raise ObjectiveError(f"{label} must be {'a nonempty ' if nonempty else 'a '}array")
    return value


def _unique(values: Iterable[Any], label: str) -> None:
    seen: set[Any] = set()
    for value in values:
        if value in seen:
            raise ObjectiveError(f"{label} contains duplicate {value!r}")
        seen.add(value)


def discover_objectives(root: Path) -> dict[str, Path]:
    """Discover objective IDs without importing any plugin code."""

    result: dict[str, Path] = {}
    if not root.is_dir():
        raise ObjectiveError(f"objective root is missing: {root}")
    for directory in sorted(root.iterdir(), key=lambda item: item.name):
        if not directory.is_dir() or directory.name.startswith("."):
            continue
        objective_file = directory / "objective.json"
        if not objective_file.is_file():
            continue
        data = _read_json(objective_file)
        objective_id = _identifier(data.get("id"), f"{directory.name}/objective.json.id")
        if objective_id != directory.name:
            raise ObjectiveError(
                f"objective directory {directory.name!r} does not match ID {objective_id!r}"
            )
        if objective_id in result:
            raise ObjectiveError(f"duplicate objective ID: {objective_id}")
        result[objective_id] = directory
    return result


def _validate_descriptor_object(
    value: Any,
    descriptor_names: tuple[str, ...],
    descriptor_values: Mapping[str, tuple[str, ...]],
    label: str,
) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ObjectiveError(f"{label} must be an object")
    _closed(value, descriptor_names, label)
    result: dict[str, str] = {}
    for name in descriptor_names:
        selected = value[name]
        if not isinstance(selected, str) or selected not in descriptor_values[name]:
            raise ObjectiveError(f"{label}.{name} is undeclared")
        result[name] = selected
    return result


def _validate_objective_data(data: dict[str, Any], directory_name: str) -> tuple[
    tuple[str, ...], dict[str, tuple[str, ...]], tuple[str, ...]
]:
    _closed(
        data,
        {
            "id",
            "title",
            "scientific_question",
            "failure_definition",
            "strategy_goal",
            "non_failure_boundaries",
            "score_anchors",
            "descriptors",
            "diagnosis_categories",
        },
        "objective.json",
    )
    objective_id = _identifier(data["id"], "objective.json.id")
    if objective_id != directory_name:
        raise ObjectiveError("objective ID must equal directory name")
    for field in (
        "title",
        "scientific_question",
        "failure_definition",
        "strategy_goal",
    ):
        _text(data[field], f"objective.json.{field}")
    for index, boundary in enumerate(_list(data["non_failure_boundaries"], "non_failure_boundaries")):
        _text(boundary, f"non_failure_boundaries[{index}]")

    anchors = _list(data["score_anchors"], "score_anchors")
    if len(anchors) != 6:
        raise ObjectiveError("score_anchors must contain exactly six anchors")
    expected_scores = (0, 25, 50, 70, 90, 100)
    for index, (anchor, expected) in enumerate(zip(anchors, expected_scores, strict=True)):
        if not isinstance(anchor, dict):
            raise ObjectiveError("score anchor must be an object")
        _closed(anchor, {"score", "meaning"}, f"score_anchors[{index}]")
        if anchor["score"] != expected:
            raise ObjectiveError("score anchors must be 0,25,50,70,90,100 in order")
        _text(anchor["meaning"], f"score_anchors[{index}].meaning")

    descriptors = _list(data["descriptors"], "descriptors")
    if len(descriptors) not in {2, 3}:
        raise ObjectiveError("an objective must declare exactly two or three descriptors")
    names = tuple(item.get("name") if isinstance(item, dict) else None for item in descriptors)
    required_names = ("objective_route", "argument_move")
    if names[:2] != required_names or (len(names) == 3 and names[2] != "application_scope"):
        raise ObjectiveError(
            "descriptor order must be objective_route, argument_move, optional application_scope"
        )
    descriptor_values: dict[str, tuple[str, ...]] = {}
    for index, descriptor in enumerate(descriptors):
        if not isinstance(descriptor, dict):
            raise ObjectiveError("descriptor must be an object")
        _closed(
            descriptor,
            {"name", "definition", "values", "other_explanation"},
            f"descriptors[{index}]",
        )
        name = descriptor["name"]
        _text(descriptor["definition"], f"descriptors[{index}].definition")
        _text(descriptor["other_explanation"], f"descriptors[{index}].other_explanation")
        raw_values = _list(descriptor["values"], f"descriptors[{index}].values")
        values: list[str] = []
        for value_index, item in enumerate(raw_values):
            if not isinstance(item, dict):
                raise ObjectiveError("descriptor value must be an object")
            _closed(item, {"value", "definition"}, "descriptor value")
            values.append(_identifier(item["value"], "descriptor value"))
            _text(item["definition"], "descriptor value definition")
        _unique(values, f"descriptor {name} values")
        if values[-1] != "other" or values.count("other") != 1:
            raise ObjectiveError(f"descriptor {name} must end with exactly one other")
        descriptor_values[name] = tuple(values)
    if descriptor_values["argument_move"] != COMMON_ARGUMENT_MOVES:
        raise ObjectiveError("argument_move must use the exact common enum")
    if "application_scope" in descriptor_values and descriptor_values["application_scope"] != COMMON_APPLICATION_SCOPES:
        raise ObjectiveError("application_scope must use the exact common enum")
    value_sets = [frozenset(values) for values in descriptor_values.values()]
    if len(value_sets) != len(set(value_sets)):
        raise ObjectiveError("descriptor value sets must not duplicate one another")

    diagnosis_items = _list(data["diagnosis_categories"], "diagnosis_categories")
    diagnoses: list[str] = []
    for index, item in enumerate(diagnosis_items):
        if not isinstance(item, dict):
            raise ObjectiveError("diagnosis category must be an object")
        _closed(item, {"category", "guidance"}, f"diagnosis_categories[{index}]")
        diagnoses.append(_identifier(item["category"], "diagnosis category"))
        _text(item["guidance"], "diagnosis guidance")
    _unique(diagnoses, "diagnosis categories")
    if "other" not in diagnoses:
        raise ObjectiveError("diagnosis categories must include other")

    return tuple(names), descriptor_values, tuple(diagnoses)


def _validate_bootstrap(
    data: dict[str, Any],
    objective_id: str,
    descriptor_names: tuple[str, ...],
    descriptor_values: Mapping[str, tuple[str, ...]],
) -> set[str]:
    _closed(data, {"objective_id", "probes"}, "bootstrap.json")
    if data["objective_id"] != objective_id:
        raise ObjectiveError("bootstrap objective_id mismatch")
    ids: list[str] = []
    for index, probe in enumerate(_list(data["probes"], "bootstrap probes")):
        if not isinstance(probe, dict):
            raise ObjectiveError("bootstrap probe must be an object")
        _closed(probe, {"probe_id", "prompt", "descriptors"}, f"bootstrap.probes[{index}]")
        ids.append(_identifier(probe["probe_id"], "bootstrap probe_id"))
        _text(probe["prompt"], "bootstrap prompt")
        _validate_descriptor_object(
            probe["descriptors"], descriptor_names, descriptor_values, "bootstrap descriptors"
        )
    _unique(ids, "bootstrap probe IDs")
    return set(ids)


def load_seed_prompts(path: Path, objective: ObjectivePlugin) -> dict[str, Any]:
    """Read the same closed bootstrap format from an explicitly selected file."""
    try:
        value = _read_json(path)
    except DomainError as exc:
        raise ObjectiveError(f"invalid seed prompts: {exc}") from exc
    _validate_bootstrap(value, objective.id, objective.descriptor_names, objective.descriptor_values)
    if not value["probes"]:
        raise ObjectiveError("selected seed prompts must not be empty")
    prompts = [normalize_line_endings(item["prompt"]) for item in value["probes"]]
    _unique(prompts, "seed prompt texts")
    return value


def _validate_messages(value: Any, label: str) -> None:
    messages = _list(value, label, nonempty=False)
    expected = "user"
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ObjectiveError(f"{label}[{index}] must be an object")
        _closed(message, {"role", "content"}, f"{label}[{index}]")
        if message["role"] != expected:
            raise ObjectiveError(f"{label} must alternate from user")
        _text(message["content"], f"{label}[{index}].content")
        expected = "assistant" if expected == "user" else "user"
    if messages and messages[-1]["role"] != "assistant":
        raise ObjectiveError(f"{label} must end in assistant before the current prompt")


def _validate_judge_fixtures(
    data: dict[str, Any],
    objective_id: str,
    descriptor_names: tuple[str, ...],
    descriptor_values: Mapping[str, tuple[str, ...]],
) -> set[str]:
    _closed(data, {"objective_id", "fixtures"}, "judge_fixtures.json")
    if data["objective_id"] != objective_id:
        raise ObjectiveError("judge fixture objective_id mismatch")
    fixtures = _list(data["fixtures"], "judge fixtures")
    if len(fixtures) != 12:
        raise ObjectiveError("judge_fixtures.json must contain exactly 12 fixtures")
    ids: list[str] = []
    routes_seen: set[str] = set()
    for index, fixture in enumerate(fixtures):
        if not isinstance(fixture, dict):
            raise ObjectiveError("judge fixture must be an object")
        _closed(
            fixture,
            {"probe_id", "prior_messages", "prompt", "reply", "descriptors", "expected"},
            f"judge_fixtures.fixtures[{index}]",
        )
        ids.append(_identifier(fixture["probe_id"], "judge fixture probe_id"))
        _validate_messages(fixture["prior_messages"], "judge fixture prior_messages")
        prompt = _text(fixture["prompt"], "judge fixture prompt")
        reply = _text(fixture["reply"], "judge fixture reply")
        descriptors = _validate_descriptor_object(
            fixture["descriptors"], descriptor_names, descriptor_values, "judge descriptors"
        )
        routes_seen.add(descriptors["objective_route"])
        expected = fixture["expected"]
        if not isinstance(expected, dict):
            raise ObjectiveError("judge fixture expected must be an object")
        _closed(
            expected,
            {
                "score_min",
                "score_max",
                "verdicts",
                "applications",
                "high",
                "aggregate_classifications",
                "required_evidence",
            },
            "judge fixture expected",
        )
        low, high = expected["score_min"], expected["score_max"]
        if (
            isinstance(low, bool)
            or isinstance(high, bool)
            or not isinstance(low, int)
            or not isinstance(high, int)
            or not 0 <= low <= high <= 100
        ):
            raise ObjectiveError("invalid judge score range")
        verdicts = _list(expected["verdicts"], "expected verdicts")
        applications = _list(expected["applications"], "expected applications")
        if not set(verdicts) <= {"not_demonstrated", "ambiguous", "demonstrated"}:
            raise ObjectiveError("invalid expected verdict")
        if not set(applications) <= {"none", "abstract", "action_governing"}:
            raise ObjectiveError("invalid expected application")
        _unique(verdicts, "expected verdicts")
        _unique(applications, "expected applications")
        if not isinstance(expected["high"], bool):
            raise ObjectiveError("expected.high must be boolean")
        classifications = _list(
            expected["aggregate_classifications"], "aggregate classifications"
        )
        classification_pairs: list[tuple[str, str]] = []
        for item in classifications:
            if not isinstance(item, dict):
                raise ObjectiveError("aggregate classification must be an object")
            _closed(item, {"failure", "application"}, "aggregate classification")
            if item["failure"] not in {"not_demonstrated", "ambiguous", "demonstrated"}:
                raise ObjectiveError("invalid aggregate failure classification")
            if item["application"] not in {"not_demonstrated", "ambiguous", "demonstrated"}:
                raise ObjectiveError("invalid aggregate application classification")
            classification_pairs.append((item["failure"], item["application"]))
        _unique(classification_pairs, "aggregate classifications")
        required = _list(expected["required_evidence"], "required evidence")
        if not 1 <= len(required) <= 3:
            raise ObjectiveError("required_evidence must contain one through three items")
        for item in required:
            if not isinstance(item, dict):
                raise ObjectiveError("required evidence item must be an object")
            _closed(item, {"polarity", "phrase"}, "required evidence item")
            if item["polarity"] not in {"supports", "resists"}:
                raise ObjectiveError("invalid required evidence polarity")
            phrase = _text(item["phrase"], "required evidence phrase")
            if phrase not in reply:
                raise ObjectiveError("required evidence phrase does not occur in reply")
        _unique(
            ((item["polarity"], item["phrase"]) for item in required),
            "required evidence",
        )
        del prompt
    _unique(ids, "judge fixture probe IDs")
    named_routes = set(descriptor_values["objective_route"]) - {"other"}
    if not named_routes <= routes_seen:
        raise ObjectiveError("semantic judge fixtures must cover every named objective route")
    return set(ids)


def _validate_descriptor_fixtures(
    data: dict[str, Any],
    objective_id: str,
    descriptor_names: tuple[str, ...],
    descriptor_values: Mapping[str, tuple[str, ...]],
) -> set[str]:
    _closed(data, {"objective_id", "fixtures"}, "descriptor_fixtures.json")
    if data["objective_id"] != objective_id:
        raise ObjectiveError("descriptor fixture objective_id mismatch")
    fixtures = _list(data["fixtures"], "descriptor fixtures")
    ids: list[str] = []
    seen: dict[str, set[str]] = {name: set() for name in descriptor_names}
    for fixture in fixtures:
        if not isinstance(fixture, dict):
            raise ObjectiveError("descriptor fixture must be an object")
        _closed(fixture, {"probe_id", "prompt", "descriptors"}, "descriptor fixture")
        ids.append(_identifier(fixture["probe_id"], "descriptor fixture probe_id"))
        _text(fixture["prompt"], "descriptor fixture prompt")
        descriptors = _validate_descriptor_object(
            fixture["descriptors"], descriptor_names, descriptor_values, "descriptor fixture tuple"
        )
        for name, value in descriptors.items():
            seen[name].add(value)
    _unique(ids, "descriptor fixture IDs")
    for name, values in descriptor_values.items():
        if not set(values) <= seen[name]:
            raise ObjectiveError(f"descriptor fixtures do not cover every {name} value")
    return set(ids)


def _validate_diagnosis_fixtures(
    data: dict[str, Any], objective_id: str, diagnoses: tuple[str, ...]
) -> None:
    _closed(data, {"objective_id", "fixtures"}, "diagnosis_fixtures.json")
    if data["objective_id"] != objective_id:
        raise ObjectiveError("diagnosis fixture objective_id mismatch")
    fixture_ids: list[str] = []
    covered: set[str] = set()
    for fixture in _list(data["fixtures"], "diagnosis fixtures"):
        if not isinstance(fixture, dict):
            raise ObjectiveError("diagnosis fixture must be an object")
        _closed(
            fixture,
            {"fixture_id", "source_turn_ref", "source_text", "diagnosis", "expected_category"},
            "diagnosis fixture",
        )
        fixture_id = _identifier(fixture["fixture_id"], "diagnosis fixture_id")
        fixture_ids.append(fixture_id)
        if fixture["source_turn_ref"] != f"fixture:{fixture_id}":
            raise ObjectiveError("diagnosis source_turn_ref mismatch")
        source_text = _text(fixture["source_text"], "diagnosis source_text")
        diagnosis = fixture["diagnosis"]
        if not isinstance(diagnosis, dict):
            raise ObjectiveError("diagnosis must be an object")
        _closed(diagnosis, {"category", "summary", "evidence"}, "diagnosis")
        category = diagnosis["category"]
        if category not in diagnoses or fixture["expected_category"] not in diagnoses:
            raise ObjectiveError("diagnosis fixture category is undeclared")
        if category != fixture["expected_category"]:
            raise ObjectiveError("diagnosis fixture expected category mismatch")
        covered.add(category)
        _text(diagnosis["summary"], "diagnosis summary")
        evidence = _list(diagnosis["evidence"], "diagnosis evidence")
        if not 1 <= len(evidence) <= 3:
            raise ObjectiveError("diagnosis evidence must contain one through three items")
        for item in evidence:
            if not isinstance(item, dict):
                raise ObjectiveError("diagnosis evidence item must be an object")
            _closed(item, {"turn_ref", "quote"}, "diagnosis evidence item")
            if item["turn_ref"] != f"fixture:{fixture_id}":
                raise ObjectiveError("diagnosis evidence turn_ref mismatch")
            if _text(item["quote"], "diagnosis quote") not in source_text:
                raise ObjectiveError("diagnosis quote does not occur in source text")
        _unique(
            ((item["turn_ref"], item["quote"]) for item in evidence),
            "diagnosis evidence",
        )
    _unique(fixture_ids, "diagnosis fixture IDs")
    if set(diagnoses) != covered:
        raise ObjectiveError("diagnosis fixtures must cover every declared category")


def _validate_fixture_turns(turns: Any, label: str) -> dict[str, tuple[str, str]]:
    values = _list(turns, label)
    result: dict[str, tuple[str, str]] = {}
    expected_role = "user"
    for index, turn in enumerate(values):
        if not isinstance(turn, dict):
            raise ObjectiveError(f"{label}[{index}] must be an object")
        _closed(turn, {"turn_id", "role", "content"}, f"{label}[{index}]")
        turn_id = _identifier(turn["turn_id"], "fixture turn_id")
        if turn_id in result:
            raise ObjectiveError("duplicate fixture turn ID")
        if turn["role"] != expected_role:
            raise ObjectiveError(f"{label} must alternate from user")
        content = _text(turn["content"], "fixture turn content")
        result[turn_id] = (turn["role"], content)
        expected_role = "assistant" if expected_role == "user" else "user"
    if values[-1]["role"] != "assistant":
        raise ObjectiveError(f"{label} must end in assistant")
    return result


def _validate_fixture_evidence(
    values: Any,
    turns: Mapping[str, tuple[str, str]],
    label: str,
    *,
    required_turn_id: str | None = None,
    required_role: str | None = None,
) -> None:
    evidence = _list(values, label)
    if not 1 <= len(evidence) <= 3:
        raise ObjectiveError(f"{label} must contain one through three items")
    for item in evidence:
        if not isinstance(item, dict):
            raise ObjectiveError(f"{label} item must be an object")
        _closed(item, {"turn_id", "quote"}, f"{label} item")
        turn_id = item["turn_id"]
        if turn_id not in turns:
            raise ObjectiveError(f"{label} references an unknown turn")
        if required_turn_id is not None and turn_id != required_turn_id:
            raise ObjectiveError(f"{label} must reference the final target turn")
        if required_role is not None and turns[turn_id][0] != required_role:
            raise ObjectiveError(f"{label} must reference an {required_role} turn")
        if _text(item["quote"], f"{label} quote") not in turns[turn_id][1]:
            raise ObjectiveError(f"{label} quote is not verbatim")
    _unique(
        ((item["turn_id"], item["quote"]) for item in evidence),
        label,
    )


def _validate_selector_fixtures(
    data: dict[str, Any],
    objective_id: str,
    descriptor_names: tuple[str, ...],
    descriptor_values: Mapping[str, tuple[str, ...]],
    diagnoses: tuple[str, ...],
) -> None:
    _closed(
        data,
        {
            "objective_id",
            "portfolio",
            "strategy_probe_review",
            "template_binding",
            "outcome_review",
        },
        "selector_fixtures.json",
    )
    if data["objective_id"] != objective_id:
        raise ObjectiveError("selector fixture objective_id mismatch")
    groups = {
        "portfolio": 3,
        "strategy_probe_review": 5,
        "template_binding": 3,
        "outcome_review": 7,
    }
    all_ids: list[str] = []
    for name, count in groups.items():
        records = _list(data[name], f"selector {name} fixtures")
        if len(records) != count:
            raise ObjectiveError(f"selector {name} requires exactly {count} fixtures")
        for record in records:
            if not isinstance(record, dict):
                raise ObjectiveError(f"selector {name} fixture must be an object")
            all_ids.append(_identifier(record.get("fixture_id"), f"selector {name} fixture_id"))
    _unique(all_ids, "selector fixture IDs")

    portfolio_categories: set[str] = set()
    injection_count = 0
    required_templates = 0
    for record in data["portfolio"]:
        _closed(
            record,
            {"fixture_id", "conversation", "judge_context", "archive_context", "expected", "injection_case"},
            "portfolio fixture",
        )
        turns = _validate_fixture_turns(record["conversation"], "portfolio conversation")
        judge = record["judge_context"]
        if not isinstance(judge, dict):
            raise ObjectiveError("portfolio judge_context must be an object")
        _closed(judge, {"verdict", "application", "evidence", "grounded_concerns"}, "judge_context")
        if judge["verdict"] not in {"not_demonstrated", "ambiguous", "demonstrated"}:
            raise ObjectiveError("invalid portfolio verdict")
        if judge["application"] not in {"none", "abstract", "action_governing"}:
            raise ObjectiveError("invalid portfolio application")
        # Judge evidence adds polarity to the ordinary fixture-evidence pair.
        evidence = _list(judge["evidence"], "portfolio judge evidence")
        for item in evidence:
            if not isinstance(item, dict):
                raise ObjectiveError("portfolio judge evidence must be an object")
            _closed(item, {"polarity", "turn_id", "quote"}, "portfolio judge evidence")
            if item["polarity"] not in {"supports", "resists"}:
                raise ObjectiveError("invalid portfolio judge evidence polarity")
            if (
                item["turn_id"] not in turns
                or turns[item["turn_id"]][0] != "assistant"
                or not isinstance(item["quote"], str)
                or not item["quote"]
                or item["quote"] not in turns[item["turn_id"]][1]
            ):
                raise ObjectiveError("portfolio judge evidence is ungrounded")
        _unique(
            ((item["polarity"], item["turn_id"], item["quote"]) for item in evidence),
            "portfolio judge evidence",
        )
        _validate_fixture_evidence(
            judge["grounded_concerns"],
            turns,
            "portfolio grounded concerns",
            required_role="assistant",
        )
        archive = record["archive_context"]
        if not isinstance(archive, dict):
            raise ObjectiveError("portfolio archive_context must be an object")
        _closed(archive, {"current_descriptors", "under_covered_cells"}, "archive_context")
        _validate_descriptor_object(archive["current_descriptors"], descriptor_names, descriptor_values, "current descriptors")
        under_covered: list[tuple[tuple[str, str], ...]] = []
        for cell in _list(archive["under_covered_cells"], "under-covered cells"):
            validated_cell = _validate_descriptor_object(
                cell, descriptor_names, descriptor_values, "under-covered cell"
            )
            under_covered.append(tuple(validated_cell.items()))
        _unique(under_covered, "under-covered cells")
        expected = record["expected"]
        if not isinstance(expected, dict):
            raise ObjectiveError("portfolio expected must be an object")
        _closed(
            expected,
            {
                "diagnosis_category",
                "required_diagnosis_evidence",
                "allowed_purposes",
                "allowed_move_acts",
                "allowed_target_kinds",
                "descriptors",
                "template_candidate",
            },
            "portfolio expected",
        )
        category = expected["diagnosis_category"]
        if category not in diagnoses:
            raise ObjectiveError("portfolio expected diagnosis is undeclared")
        portfolio_categories.add(category)
        _validate_fixture_evidence(
            expected["required_diagnosis_evidence"],
            turns,
            "required diagnosis evidence",
            required_role="assistant",
        )
        enum_lists = (
            ("allowed_purposes", STRATEGY_PURPOSES),
            ("allowed_move_acts", STRATEGY_MOVE_ACTS),
            ("allowed_target_kinds", STRATEGY_TARGET_KINDS),
        )
        for field, allowed in enum_lists:
            values = _list(expected[field], f"portfolio expected {field}")
            if any(not isinstance(value, str) or value not in allowed for value in values):
                raise ObjectiveError(f"portfolio expected {field} contains an invalid value")
            _unique(values, f"portfolio expected {field}")
        if expected["template_candidate"] not in {"required", "optional", "forbidden"}:
            raise ObjectiveError("invalid template_candidate expectation")
        required_templates += expected["template_candidate"] == "required"
        _validate_descriptor_object(expected["descriptors"], descriptor_names, descriptor_values, "portfolio expected descriptors")
        if not isinstance(record["injection_case"], bool):
            raise ObjectiveError("injection_case must be boolean")
        injection_count += record["injection_case"]
    if len(portfolio_categories - {"other"}) < 2 or injection_count != 1 or required_templates < 2:
        raise ObjectiveError("portfolio fixtures lack required semantic/injection/template coverage")

    statuses: set[str] = set()
    recovery_statuses: set[str] = set()
    for record in data["strategy_probe_review"]:
        _closed(record, {"fixture_id", "conversation", "contract", "realization", "expected"}, "strategy review fixture")
        turns = _validate_fixture_turns(record["conversation"], "strategy review conversation")
        if not isinstance(record["contract"], dict):
            raise ObjectiveError("strategy review contract must be an object")
        realization = record["realization"]
        if not isinstance(realization, dict):
            raise ObjectiveError("strategy review realization must be an object")
        _closed(
            realization,
            {
                "slot_id",
                "status",
                "user_turn",
            },
            "strategy review realization",
        )
        _identifier(realization["slot_id"], "strategy review realization slot_id")
        expected = record["expected"]
        if not isinstance(expected, dict):
            raise ObjectiveError("strategy review expected must be an object")
        _closed(expected, {"status", "reviewed_descriptors"}, "strategy review expected")
        status = expected["status"]
        if status not in {
            "aligned",
            "invalid_contract",
            "invalid",
        }:
            raise ObjectiveError("invalid strategy-review status")
        statuses.add(status)
        if record["contract"].get("purpose") == "recovery":
            recovery_statuses.add(status)
        if status in {"invalid_contract", "invalid"}:
            if expected["reviewed_descriptors"] is not None:
                raise ObjectiveError("invalid strategy review must have null descriptors")
        else:
            _validate_descriptor_object(expected["reviewed_descriptors"], descriptor_names, descriptor_values, "strategy reviewed descriptors")
    if not {"aligned", "invalid_contract", "invalid"}.issubset(statuses):
        raise ObjectiveError(
            "strategy review fixtures must cover aligned, invalid_contract, and invalid"
        )
    if not {"aligned", "invalid_contract"}.issubset(recovery_statuses):
        raise ObjectiveError(
            "strategy review fixtures must cover aligned and invalid-contract recovery"
        )

    binding_statuses: set[str] = set()
    binding_injections = 0
    for record in data["template_binding"]:
        _closed(record, {"fixture_id", "template", "conversation", "expected", "injection_case"}, "binding fixture")
        turns = _validate_fixture_turns(record["conversation"], "binding conversation")
        if not isinstance(record["template"], dict):
            raise ObjectiveError("binding template must be an object")
        expected = record["expected"]
        if not isinstance(expected, dict):
            raise ObjectiveError("binding expected must be an object")
        _closed(expected, {"status", "required_evidence"}, "binding expected")
        if expected["status"] not in {"applicable", "inapplicable"}:
            raise ObjectiveError("invalid binding status")
        binding_statuses.add(expected["status"])
        _validate_fixture_evidence(
            expected["required_evidence"],
            turns,
            "binding expected evidence",
            required_role="assistant",
        )
        if not isinstance(record["injection_case"], bool):
            raise ObjectiveError("binding injection_case must be boolean")
        binding_injections += record["injection_case"]
        if record["injection_case"] and expected["status"] != "inapplicable":
            raise ObjectiveError("binding injection fixture must be inapplicable")
    if binding_statuses != {"applicable", "inapplicable"} or binding_injections != 1:
        raise ObjectiveError("binding fixtures lack required status/injection coverage")

    outcome_results: set[str] = set()
    recovery_outcome_results: set[str] = set()
    response_states: set[str] = set()
    for record in data["outcome_review"]:
        _closed(
            record,
            {"fixture_id", "contract", "before_target", "intervening_user", "after_target", "expected"},
            "outcome fixture",
        )
        turns: dict[str, tuple[str, str]] = {}
        expected_roles = ("assistant", "user", "assistant")
        turn_objects = (record["before_target"], record["intervening_user"], record["after_target"])
        for turn, role in zip(turn_objects, expected_roles, strict=True):
            if not isinstance(turn, dict):
                raise ObjectiveError("outcome fixture turn must be an object")
            _closed(turn, {"turn_id", "role", "content"}, "outcome fixture turn")
            turn_id = _identifier(turn["turn_id"], "outcome turn_id")
            if turn_id in turns or turn["role"] != role:
                raise ObjectiveError("invalid outcome turn namespace or role")
            turns[turn_id] = (role, _text(turn["content"], "outcome turn content"))
        expected = record["expected"]
        if not isinstance(expected, dict):
            raise ObjectiveError("outcome expected must be an object")
        _closed(expected, {"result", "response_state", "required_evidence"}, "outcome expected")
        if expected["result"] not in {"supports", "falsifies", "indeterminate"}:
            raise ObjectiveError("invalid outcome expected result")
        if expected["response_state"] not in {"engaged", "refusal", "nonanswer"}:
            raise ObjectiveError("invalid outcome response state")
        if not isinstance(record["contract"], dict):
            raise ObjectiveError("outcome contract must be an object")
        outcome_results.add(expected["result"])
        if record["contract"].get("purpose") == "recovery":
            recovery_outcome_results.add(expected["result"])
        response_states.add(expected["response_state"])
        after_id = record["after_target"]["turn_id"]
        _validate_fixture_evidence(
            expected["required_evidence"],
            turns,
            "outcome required evidence",
            required_turn_id=after_id,
        )
    if outcome_results != {"supports", "falsifies", "indeterminate"} or response_states != {"engaged", "refusal", "nonanswer"}:
        raise ObjectiveError("outcome fixtures lack required result/response-state coverage")
    if recovery_outcome_results != {"supports", "falsifies", "indeterminate"}:
        raise ObjectiveError(
            "outcome fixtures lack recovery direction coverage"
        )


def _load_plugin(path: Path, objective_id: str) -> tuple[Callable[..., Any], Callable[..., Any]]:
    module_name = f"_quasilisk_objective_{objective_id}_{sha256_bytes(path.read_bytes())[7:19]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ObjectiveError(f"cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # trusted code may still fail conformance
        raise ObjectiveError(f"selected objective plugin import failed: {exc}") from exc
    describe = getattr(module, "describe_probe", None)
    normalize = getattr(module, "normalize_diagnosis", None)
    for name, callback in (("describe_probe", describe), ("normalize_diagnosis", normalize)):
        if not callable(callback):
            raise ObjectiveError(f"plugin is missing callable {name}")
        signature = inspect.signature(callback)
        if len(signature.parameters) != 1:
            raise ObjectiveError(f"{name} must accept exactly one argument")
    return describe, normalize


@dataclass(frozen=True, slots=True)
class ObjectivePlugin:
    directory: Path
    objective: Mapping[str, Any]
    bootstrap: Mapping[str, Any]
    judge_fixtures: Mapping[str, Any]
    descriptor_fixtures: Mapping[str, Any]
    diagnosis_fixtures: Mapping[str, Any]
    selector_fixtures: Mapping[str, Any]
    artifact_digests: Mapping[str, str]
    plugin_digest: str
    descriptor_names: tuple[str, ...]
    descriptor_values: Mapping[str, tuple[str, ...]]
    diagnosis_categories: tuple[str, ...]
    describe_probe_callback: Callable[[Mapping[str, Any]], Any]
    normalize_diagnosis_callback: Callable[[Mapping[str, Any]], Any]

    @property
    def id(self) -> str:
        return str(self.objective["id"])

    @property
    def strategy_goal(self) -> str:
        return str(self.objective["strategy_goal"])

    @property
    def total_cells(self) -> int:
        result = 1
        for name in self.descriptor_names:
            result *= len(self.descriptor_values[name])
        return result

    def validate_descriptors(self, value: Mapping[str, Any]) -> dict[str, str]:
        return _validate_descriptor_object(
            value, self.descriptor_names, self.descriptor_values, "descriptor object"
        )

    def describe_probe(self, metadata: Mapping[str, Any]) -> dict[str, str]:
        result = self.describe_probe_callback(dict(metadata))
        if not isinstance(result, Mapping):
            raise ObjectiveError("describe_probe did not return a descriptor mapping")
        return self.validate_descriptors(result)

    def normalize_diagnosis(self, diagnosis: Mapping[str, Any]) -> str:
        result = self.normalize_diagnosis_callback(dict(diagnosis))
        category = result.get("category") if isinstance(result, Mapping) else result
        if not isinstance(category, str) or category not in self.diagnosis_categories:
            raise ObjectiveError("normalize_diagnosis returned an undeclared category")
        return category


def load_objective(
    objective_id: str,
    root: Path,
    *,
    selector_fixture_semantic_validator: SelectorFixtureSemanticValidator,
) -> ObjectivePlugin:
    """Load one objective after generic and strategy-owned conformance checks."""

    if not callable(selector_fixture_semantic_validator):
        raise ObjectiveError("selector fixture semantic validator must be callable")
    discovered = discover_objectives(root)
    if objective_id not in discovered:
        raise ObjectiveError(f"unknown objective: {objective_id}")
    directory = discovered[objective_id]
    missing = [name for name in OBJECTIVE_FILES if not (directory / name).is_file()]
    if missing:
        raise ObjectiveError(f"objective {objective_id} is missing: {', '.join(missing)}")
    artifacts = {
        name: _read_json(directory / name)
        for name in OBJECTIVE_FILES
        if name.endswith(".json")
    }
    descriptor_names, descriptor_values, diagnoses = _validate_objective_data(
        artifacts["objective.json"], objective_id
    )
    probe_ids: set[str] = set()
    for ids in (
        _validate_bootstrap(
            artifacts["bootstrap.json"],
            objective_id,
            descriptor_names,
            descriptor_values,
        ),
        _validate_judge_fixtures(
            artifacts["judge_fixtures.json"],
            objective_id,
            descriptor_names,
            descriptor_values,
        ),
        _validate_descriptor_fixtures(
            artifacts["descriptor_fixtures.json"],
            objective_id,
            descriptor_names,
            descriptor_values,
        ),
    ):
        overlap = probe_ids & ids
        if overlap:
            raise ObjectiveError(f"plugin-wide duplicate ProbeId: {sorted(overlap)[0]}")
        probe_ids.update(ids)
    _validate_diagnosis_fixtures(
        artifacts["diagnosis_fixtures.json"], objective_id, diagnoses
    )
    _validate_selector_fixtures(
        artifacts["selector_fixtures.json"],
        objective_id,
        descriptor_names,
        descriptor_values,
        diagnoses,
    )
    try:
        selector_fixture_semantic_validator(
            artifacts["selector_fixtures.json"], descriptor_values, diagnoses
        )
    except DomainError as exc:
        raise ObjectiveError(
            f"selector fixture semantic validation failed: {exc}"
        ) from exc

    artifact_digests = {
        name: sha256_bytes((directory / name).read_bytes()) for name in OBJECTIVE_FILES
    }
    plugin_digest = digest_json(
        [[name, artifact_digests[name]] for name in sorted(OBJECTIVE_FILES)]
    )
    describe, normalize = _load_plugin(directory / "plugin.py", objective_id)

    # Execute the two pure callbacks over every local conformance fixture.
    for source in (
        artifacts["bootstrap.json"]["probes"],
        artifacts["judge_fixtures.json"]["fixtures"],
        artifacts["descriptor_fixtures.json"]["fixtures"],
    ):
        for probe in source:
            metadata = {"descriptors": probe["descriptors"]}
            result = describe(metadata)
            validated = _validate_descriptor_object(
                result,
                descriptor_names,
                descriptor_values,
                "describe_probe result",
            )
            if validated != probe["descriptors"]:
                raise ObjectiveError("describe_probe changed expected descriptor metadata")
    for fixture in artifacts["diagnosis_fixtures.json"]["fixtures"]:
        result = normalize(fixture["diagnosis"])
        category = result.get("category") if isinstance(result, Mapping) else result
        if category != fixture["expected_category"]:
            raise ObjectiveError("normalize_diagnosis failed a local fixture")

    return ObjectivePlugin(
        directory=directory,
        objective=artifacts["objective.json"],
        bootstrap=artifacts["bootstrap.json"],
        judge_fixtures=artifacts["judge_fixtures.json"],
        descriptor_fixtures=artifacts["descriptor_fixtures.json"],
        diagnosis_fixtures=artifacts["diagnosis_fixtures.json"],
        selector_fixtures=artifacts["selector_fixtures.json"],
        artifact_digests=artifact_digests,
        plugin_digest=plugin_digest,
        descriptor_names=descriptor_names,
        descriptor_values=descriptor_values,
        diagnosis_categories=diagnoses,
        describe_probe_callback=describe,
        normalize_diagnosis_callback=normalize,
    )


def materialize_schema(schema: Mapping[str, Any], plugin: ObjectivePlugin) -> dict[str, Any]:
    """Materialize the canonical descriptor, diagnosis and operation markers."""

    import copy

    result = copy.deepcopy(dict(schema))
    defs = result.get("$defs")
    for marker in ("DescriptorObject", "StrategyDescriptorObject"):
        if not isinstance(defs, dict) or marker not in defs:
            continue
        names = [name for name in plugin.descriptor_names
                 if marker == "DescriptorObject" or name != "argument_move"]
        defs[marker] = {
            "type": "object",
            "properties": {
                name: {"type": "string", "enum": list(plugin.descriptor_values[name])}
                for name in names
            },
            "required": names,
            "additionalProperties": False,
        }
    if isinstance(defs, dict) and "StrategyOperation" in defs:
        defs["StrategyOperation"] = {"type": "string", "enum": list(STRATEGY_OPERATIONS)}
    if isinstance(defs, dict) and "DiagnosisCategory" in defs:
        defs["DiagnosisCategory"] = {
            "type": "string",
            "enum": list(plugin.diagnosis_categories),
        }
    return result
