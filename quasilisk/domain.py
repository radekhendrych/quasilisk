"""Immutable domain records and deterministic identity helpers.

This module deliberately performs no I/O.  JSON canonicalisation here is the
one implementation used for scientific keys, entity IDs, event IDs, and
semantic request hashes throughout Quasilisk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import StrEnum
import hashlib
import json
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urldefrag, urljoin


DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
ENTITY_RE = re.compile(
    r"^(?:run|ev|iv|wu|la|pa|pc|to|ex|cr|cn|ma|pn|sb|ss|st|tp|rz|ep|sc|ae|sd|qf|rp|ps|ri|bi)-[0-9a-f]{64}$"
)
SCIENTIFIC_KEY_RE = DIGEST_RE
OBJECTIVE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")

ENTITY_PREFIXES = frozenset(
    {
        "run",
        "ev",
        "iv",
        "wu",
        "la",
        "pa",
        "pc",
        "to",
        "ex",
        "cr",
        "cn",
        "ma",
        "pn",
        "sb",
        "ss",
        "st",
        "tp",
        "rz",
        "ep",
        "sc",
        "ae",
        "sd",
        "qf",
        "rp",
        "ps",
        "ri",
        "bi",
    }
)


class DomainError(ValueError):
    """Raised when a domain value violates a closed contract."""


class DuplicateKeyError(DomainError):
    """Raised for duplicate keys in an input JSON object."""


class JsonSchemaContractError(DomainError):
    """Raised when a strict schema uses an unsupported contract feature."""


class JsonSchemaValidationError(DomainError):
    """Raised when parsed JSON does not satisfy a strict schema."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors))


def normalize_line_endings(value: str) -> str:
    """Normalize CRLF/CR to LF without otherwise changing text identity."""

    return value.replace("\r\n", "\n").replace("\r", "\n")


def evidence_lines(value: str) -> tuple[str, ...]:
    """Return exact nonblank lines suitable for request-local evidence IDs."""

    if not isinstance(value, str):
        raise DomainError("evidence source must be text")
    return tuple(
        line
        for line in normalize_line_endings(value).split("\n")
        if line.strip()
    )


def normalize_decimal(value: Decimal | int | str) -> str:
    """Return the canonical finite, non-exponent decimal representation."""

    if isinstance(value, bool):
        raise DomainError("booleans are not decimal values")
    try:
        decimal = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise DomainError(f"invalid decimal: {value!r}") from exc
    if not decimal.is_finite():
        raise DomainError("decimal must be finite")
    if decimal == 0:
        return "0"
    rendered = format(decimal, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def require_decimal_string(
    value: object,
    *,
    minimum: Decimal | None = None,
    maximum: Decimal | None = None,
) -> Decimal:
    if not isinstance(value, str) or normalize_decimal(value) != value:
        raise DomainError("expected a normalized decimal string")
    result = Decimal(value)
    if minimum is not None and result < minimum:
        raise DomainError(f"decimal is below {minimum}")
    if maximum is not None and result > maximum:
        raise DomainError(f"decimal is above {maximum}")
    return result


def _reject_constant(value: str) -> None:
    raise DomainError(f"nonfinite JSON number is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_loads(text: str | bytes) -> Any:
    """Parse JSON with duplicate-key, nonfinite, and binary-float protection."""

    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise DomainError("JSON input is not valid UTF-8") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_float=Decimal,
            parse_int=int,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, TypeError, RecursionError) as exc:
        raise DomainError(f"invalid JSON: {exc}") from exc


# The canonical schemas intentionally use a small, auditable subset of
# JSON Schema 2020-12.  Rejecting unknown keywords is important: silently
# ignoring a newly added assertion would weaken the model-output boundary.
_JSON_SCHEMA_KEYWORDS = frozenset(
    {
        "$schema",
        "$id",
        "$ref",
        "$defs",
        "$comment",
        "title",
        "description",
        "default",
        "examples",
        "deprecated",
        "readOnly",
        "writeOnly",
        "type",
        "enum",
        "const",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "prefixItems",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minLength",
        "maxLength",
        "pattern",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minProperties",
        "maxProperties",
        "oneOf",
        "anyOf",
        "allOf",
        "if",
        "then",
        "else",
    }
)
_JSON_SCHEMA_TYPES = frozenset(
    {"null", "boolean", "object", "array", "number", "integer", "string"}
)
_JSON_SCHEMA_ERROR_LIMIT = 64


def _schema_child_path(path: str, token: str) -> str:
    return f"{path}/{token.replace('~', '~0').replace('/', '~1')}"


def _instance_child_path(path: str, token: str | int) -> str:
    if isinstance(token, int):
        return f"{path}[{token}]"
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token):
        return f"{path}.{token}"
    return f"{path}[{json.dumps(token, ensure_ascii=False)}]"


def _schema_registry_entry(
    root: Mapping[str, Any],
    reference_base: str,
    registry: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    root_id = root.get("$id")
    absolute = urljoin(root_id, reference_base) if isinstance(root_id, str) else reference_base
    candidates = (reference_base, absolute)
    for candidate in candidates:
        external = registry.get(candidate)
        if isinstance(external, Mapping):
            return external
    raise JsonSchemaContractError(f"unresolved external schema reference: {reference_base!r}")


def _resolve_schema_ref(
    root: Mapping[str, Any],
    reference: object,
    registry: Mapping[str, Mapping[str, Any]],
) -> tuple[Any, Mapping[str, Any]]:
    if not isinstance(reference, str) or not reference:
        raise JsonSchemaContractError("schema $ref must be a nonempty string")
    reference_base, fragment = urldefrag(reference)
    target_root = (
        root
        if not reference_base
        else _schema_registry_entry(root, reference_base, registry)
    )
    if not fragment:
        return target_root, target_root
    if not fragment.startswith("/"):
        raise JsonSchemaContractError(f"invalid schema reference: {reference!r}")
    current: Any = target_root
    for raw_token in fragment[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping) and token in current:
            current = current[token]
            continue
        if isinstance(current, (list, tuple)) and token.isdigit() and int(token) < len(current):
            current = current[int(token)]
            continue
        raise JsonSchemaContractError(f"unresolved schema reference: {reference}")
    if not isinstance(current, (Mapping, bool)):
        raise JsonSchemaContractError(f"schema reference is not a schema: {reference}")
    return current, target_root


def _schema_array(value: Any, path: str, keyword: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)) or not value:
        raise JsonSchemaContractError(f"{path}/{keyword} must be a nonempty array")
    return value


def _nonnegative_schema_integer(value: Any, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise JsonSchemaContractError(f"{path} must be a nonnegative integer")


def _validate_schema_node(
    schema: Any,
    root: Mapping[str, Any],
    path: str,
    registry: Mapping[str, Mapping[str, Any]],
) -> None:
    if isinstance(schema, bool):
        return
    if not isinstance(schema, Mapping) or not all(isinstance(key, str) for key in schema):
        raise JsonSchemaContractError(f"{path} must be a schema object or boolean")
    unknown = sorted(set(schema) - _JSON_SCHEMA_KEYWORDS)
    if unknown:
        raise JsonSchemaContractError(
            f"{path} uses unsupported schema keywords: {', '.join(unknown)}"
        )
    if "$ref" in schema:
        _resolve_schema_ref(root, schema["$ref"], registry)
    if "type" in schema:
        declared = schema["type"]
        values = [declared] if isinstance(declared, str) else declared
        if (
            not isinstance(values, (list, tuple))
            or not values
            or any(not isinstance(item, str) or item not in _JSON_SCHEMA_TYPES for item in values)
            or len(set(values)) != len(values)
        ):
            raise JsonSchemaContractError(f"{path}/type is invalid")
    if "enum" in schema and not isinstance(schema["enum"], (list, tuple)):
        raise JsonSchemaContractError(f"{path}/enum must be an array")
    for keyword in ("properties", "$defs"):
        if keyword not in schema:
            continue
        children = schema[keyword]
        if not isinstance(children, Mapping) or not all(
            isinstance(key, str) for key in children
        ):
            raise JsonSchemaContractError(f"{path}/{keyword} must be an object")
        for name, child in children.items():
            _validate_schema_node(
                child,
                root,
                _schema_child_path(_schema_child_path(path, keyword), name),
                registry,
            )
    if "required" in schema:
        required = schema["required"]
        if (
            not isinstance(required, (list, tuple))
            or any(not isinstance(item, str) for item in required)
            or len(set(required)) != len(required)
        ):
            raise JsonSchemaContractError(f"{path}/required must be a unique string array")
    for keyword in ("additionalProperties", "items", "if", "then", "else"):
        if keyword in schema:
            _validate_schema_node(
                schema[keyword], root, _schema_child_path(path, keyword), registry
            )
    for keyword in ("oneOf", "anyOf", "allOf", "prefixItems"):
        if keyword not in schema:
            continue
        children = _schema_array(schema[keyword], path, keyword)
        for index, child in enumerate(children):
            _validate_schema_node(
                child,
                root,
                _schema_child_path(_schema_child_path(path, keyword), str(index)),
                registry,
            )
    for keyword in (
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "minProperties",
        "maxProperties",
    ):
        if keyword in schema:
            _nonnegative_schema_integer(schema[keyword], _schema_child_path(path, keyword))
    if "uniqueItems" in schema and not isinstance(schema["uniqueItems"], bool):
        raise JsonSchemaContractError(f"{path}/uniqueItems must be boolean")
    if "pattern" in schema:
        pattern = schema["pattern"]
        if not isinstance(pattern, str):
            raise JsonSchemaContractError(f"{path}/pattern must be a string")
        try:
            re.compile(pattern)
        except re.error as exc:
            raise JsonSchemaContractError(f"{path}/pattern is invalid: {exc}") from exc
    for keyword in (
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
    ):
        if keyword in schema and (
            isinstance(schema[keyword], bool)
            or not isinstance(schema[keyword], (int, Decimal))
        ):
            raise JsonSchemaContractError(f"{path}/{keyword} must be a JSON number")
    if "multipleOf" in schema and schema["multipleOf"] <= 0:
        raise JsonSchemaContractError(f"{path}/multipleOf must be positive")


def validate_json_schema_contract(
    schema: Mapping[str, Any],
    *,
    registry: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    """Validate the closed JSON Schema subset used by canonical contracts.

    ``registry`` is deliberately explicit.  It permits one canonical schema to
    compose another without silently consulting a filesystem or network.
    """

    if not isinstance(schema, Mapping):
        raise JsonSchemaContractError("strict schema must be one object")
    _validate_schema_node(schema, schema, "#", registry or {})


def _json_identity(value: Any) -> Any:
    if value is None:
        return ("null",)
    if isinstance(value, bool):
        return ("boolean", value)
    if isinstance(value, (int, Decimal)) and not isinstance(value, bool):
        try:
            return ("number", normalize_decimal(value))
        except DomainError:
            return ("invalid-number", repr(value))
    if isinstance(value, str):
        return ("string", value)
    if isinstance(value, (list, tuple)):
        return ("array", tuple(_json_identity(item) for item in value))
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return (
            "object",
            tuple((key, _json_identity(value[key])) for key in sorted(value)),
        )
    return ("invalid", type(value).__name__, repr(value))


def _matches_json_type(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, Mapping) and all(isinstance(key, str) for key in value)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, Decimal)) and not isinstance(value, bool)
    if expected == "integer":
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
        ) or (isinstance(value, Decimal) and value.is_finite() and value == value.to_integral())
    return False


def _schema_error(errors: list[str], path: str, detail: str) -> None:
    if len(errors) < _JSON_SCHEMA_ERROR_LIMIT:
        errors.append(f"{path}: {detail}")


def _collect_schema_errors(
    value: Any,
    schema: Any,
    root: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, Any]],
    path: str,
    errors: list[str],
    *,
    depth: int = 0,
) -> None:
    if len(errors) >= _JSON_SCHEMA_ERROR_LIMIT:
        return
    if depth > 128:
        _schema_error(errors, path, "schema evaluation exceeded the depth limit")
        return
    if schema is False:
        _schema_error(errors, path, "value is forbidden by the schema")
        return
    if schema is True:
        return
    assert isinstance(schema, Mapping)

    if "$ref" in schema:
        referenced_schema, referenced_root = _resolve_schema_ref(
            root, schema["$ref"], registry
        )
        _collect_schema_errors(
            value,
            referenced_schema,
            referenced_root,
            registry,
            path,
            errors,
            depth=depth + 1,
        )

    declared_type = schema.get("type")
    if declared_type is not None:
        types = [declared_type] if isinstance(declared_type, str) else declared_type
        if not any(_matches_json_type(value, item) for item in types):
            _schema_error(errors, path, f"expected JSON type {'|'.join(types)}")
            return

    if "const" in schema and _json_identity(value) != _json_identity(schema["const"]):
        _schema_error(errors, path, "value does not equal const")
    if "enum" in schema and not any(
        _json_identity(value) == _json_identity(item) for item in schema["enum"]
    ):
        _schema_error(errors, path, "value is not in enum")

    for keyword in ("allOf",):
        for child in schema.get(keyword, ()):
            _collect_schema_errors(
                value, child, root, registry, path, errors, depth=depth + 1
            )
    for keyword, expected_matches in (("oneOf", 1),):
        if keyword not in schema:
            continue
        matches = 0
        for child in schema[keyword]:
            child_errors: list[str] = []
            _collect_schema_errors(
                value, child, root, registry, path, child_errors, depth=depth + 1
            )
            if not child_errors:
                matches += 1
        if matches != expected_matches:
            _schema_error(
                errors,
                path,
                f"must match exactly one {keyword} branch (matched {matches})",
            )
    if "anyOf" in schema:
        matched = False
        for child in schema["anyOf"]:
            child_errors = []
            _collect_schema_errors(
                value, child, root, registry, path, child_errors, depth=depth + 1
            )
            if not child_errors:
                matched = True
                break
        if not matched:
            _schema_error(errors, path, "must match at least one anyOf branch")
    if "if" in schema:
        condition_errors: list[str] = []
        _collect_schema_errors(
            value,
            schema["if"],
            root,
            registry,
            path,
            condition_errors,
            depth=depth + 1,
        )
        branch = "then" if not condition_errors else "else"
        if branch in schema:
            _collect_schema_errors(
                value,
                schema[branch],
                root,
                registry,
                path,
                errors,
                depth=depth + 1,
            )

    if isinstance(value, Mapping):
        required = schema.get("required", ())
        for name in required:
            if name not in value:
                _schema_error(errors, path, f"missing required property {name!r}")
        properties = schema.get("properties", {})
        for name, child in properties.items():
            if name in value:
                _collect_schema_errors(
                    value[name],
                    child,
                    root,
                    registry,
                    _instance_child_path(path, name),
                    errors,
                    depth=depth + 1,
                )
        additional = schema.get("additionalProperties", True)
        for name in sorted(set(value) - set(properties)):
            child_path = _instance_child_path(path, name)
            if additional is False:
                _schema_error(errors, child_path, "additional property is forbidden")
            elif additional is not True:
                _collect_schema_errors(
                    value[name],
                    additional,
                    root,
                    registry,
                    child_path,
                    errors,
                    depth=depth + 1,
                )
        if "minProperties" in schema and len(value) < schema["minProperties"]:
            _schema_error(errors, path, f"object has fewer than {schema['minProperties']} properties")
        if "maxProperties" in schema and len(value) > schema["maxProperties"]:
            _schema_error(errors, path, f"object has more than {schema['maxProperties']} properties")

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            _schema_error(errors, path, f"array has fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            _schema_error(errors, path, f"array has more than {schema['maxItems']} items")
        if schema.get("uniqueItems") is True:
            identities = [_json_identity(item) for item in value]
            if len(identities) != len(set(identities)):
                _schema_error(errors, path, "array items are not unique")
        prefix_items = schema.get("prefixItems", ())
        for index, child in enumerate(prefix_items):
            if index >= len(value):
                break
            _collect_schema_errors(
                value[index],
                child,
                root,
                registry,
                _instance_child_path(path, index),
                errors,
                depth=depth + 1,
            )
        if "items" in schema:
            for index, item in enumerate(value[len(prefix_items) :], len(prefix_items)):
                _collect_schema_errors(
                    item,
                    schema["items"],
                    root,
                    registry,
                    _instance_child_path(path, index),
                    errors,
                    depth=depth + 1,
                )

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            _schema_error(errors, path, f"string is shorter than {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            _schema_error(errors, path, f"string is longer than {schema['maxLength']}")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            _schema_error(errors, path, "string does not match pattern")

    if isinstance(value, (int, Decimal)) and not isinstance(value, bool):
        if isinstance(value, Decimal) and not value.is_finite():
            _schema_error(errors, path, "number is not finite")
            return
        comparisons = (
            ("minimum", lambda actual, bound: actual >= bound),
            ("maximum", lambda actual, bound: actual <= bound),
            ("exclusiveMinimum", lambda actual, bound: actual > bound),
            ("exclusiveMaximum", lambda actual, bound: actual < bound),
        )
        for keyword, predicate in comparisons:
            if keyword in schema and not predicate(value, schema[keyword]):
                _schema_error(errors, path, f"number violates {keyword}")
        if "multipleOf" in schema and value % schema["multipleOf"] != 0:
            _schema_error(errors, path, "number violates multipleOf")


def validate_json_schema_instance(
    value: Any,
    schema: Mapping[str, Any],
    *,
    registry: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    """Fail closed unless one parsed JSON value satisfies a strict schema."""

    resolved_registry = registry or {}
    validate_json_schema_contract(schema, registry=resolved_registry)
    errors: list[str] = []
    _collect_schema_errors(value, schema, schema, resolved_registry, "$", errors)
    if errors:
        raise JsonSchemaValidationError(errors)


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Decimal):
        return normalize_decimal(value)
    if isinstance(value, float):
        raise DomainError("binary floating-point values are forbidden")
    if isinstance(value, StrEnum):
        return str(value)
    if hasattr(value, "to_dict"):
        return _canonical_value(value.to_dict())
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise DomainError("canonical JSON object keys must be strings")
            converted[key] = _canonical_value(item)
        return converted
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    raise DomainError(f"value is not canonical-JSON compatible: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Encode the canonical JSON form required by the redesign specification."""

    try:
        return json.dumps(
            _canonical_value(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8", errors="strict")
    except DomainError:
        raise
    except (UnicodeEncodeError, ValueError, RecursionError) as exc:
        raise DomainError(f"value cannot be encoded as canonical UTF-8 JSON: {exc}") from exc


def canonical_json(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def digest_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _freeze_json_value(value: Any) -> Any:
    """Take an immutable snapshot of canonical-JSON-compatible data."""

    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise DomainError("JSON object keys must be strings")
            frozen[key] = _freeze_json_value(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item) for item in value)
    # Validate primitives (including the prohibition on binary floats) through
    # the one canonical encoder rather than maintaining a second value union.
    canonical_json_bytes(value)
    return value


def scientific_key(material: Sequence[Any]) -> str:
    return digest_json(list(material))


def entity_id(prefix: str, material: Any) -> str:
    if prefix not in ENTITY_PREFIXES:
        raise DomainError(f"unknown entity prefix: {prefix}")
    return f"{prefix}-{hashlib.sha256(canonical_json_bytes(material)).hexdigest()}"


def require_digest(value: object, label: str = "digest") -> str:
    if not isinstance(value, str) or DIGEST_RE.fullmatch(value) is None:
        raise DomainError(f"{label} must be sha256: followed by 64 lowercase hex digits")
    return value


def require_entity_id(value: object, label: str = "entity ID") -> str:
    if not isinstance(value, str) or ENTITY_RE.fullmatch(value) is None:
        raise DomainError(f"invalid {label}")
    return value


def seeded_rank(seed: int, decision_material: Sequence[Any]) -> str:
    """Stable pseudo-random ordering key, never a mutable RNG stream."""

    return digest_json([seed, *decision_material])


class Role(StrEnum):
    TARGET = "target"
    JUDGE = "judge"
    SELECTOR = "selector"
    PERTURBATOR = "perturbator"


class ActionPurpose(StrEnum):
    TARGET_SAMPLE = "target_sample"
    JUDGE_MEMBER = "judge_member"
    SELECTOR_PORTFOLIO = "selector_portfolio"
    PROBE_REVIEW = "probe_review"
    TEMPLATE_BINDING = "template_binding"
    OUTCOME_REVIEW = "outcome_review"
    PERTURBATOR_PROPOSAL = "perturbator_proposal"


CALL_ROUTE_KINDS = frozenset(
    {
        "primary",
        "exact_retry",
        "schema_repair",
        "format_downgrade",
        "replacement",
        "fallback",
    }
)


class ProviderErrorKind(StrEnum):
    CONNECTION = "connection"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    SERVER = "server"
    AUTHENTICATION = "authentication"
    INVALID_REQUEST = "invalid_request"
    UNSUPPORTED_RESPONSE_FORMAT = "unsupported_response_format"
    CONTEXT_LIMIT = "context_limit"
    FILTERED = "filtered"
    MALFORMED_RESPONSE = "malformed_response"
    UNKNOWN = "unknown"


class FinishStatus(StrEnum):
    COMPLETE = "complete"
    OUTPUT_LIMIT = "output_limit"
    CONTENT_FILTER = "content_filter"
    OTHER = "other"


class Verdict(StrEnum):
    NOT_DEMONSTRATED = "not_demonstrated"
    AMBIGUOUS = "ambiguous"
    DEMONSTRATED = "demonstrated"


class Application(StrEnum):
    NONE = "none"
    ABSTRACT = "abstract"
    ACTION_GOVERNING = "action_governing"


class PanelAvailability(StrEnum):
    COMPLETE = "complete"
    DEGRADED = "degraded"
    NO_QUORUM = "no_quorum"


class SearchStatus(StrEnum):
    OBSERVED = "observed"
    EXPLORATORY_LEAD = "exploratory_lead"
    PROMISING = "promising"


class PortfolioIntent(StrEnum):
    RECOVER = "recover"
    ADVANCE = "advance"


class VerificationStatus(StrEnum):
    NOT_STARTED = "not_started"
    VERIFIED = "verified"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


class QualificationStatus(StrEnum):
    EXPERIMENTAL = "experimental"
    QUALIFIED = "qualified"
    PROVISIONALLY_COMPATIBLE = "provisionally_compatible"


STRUCTURED_ROLE_SCOPES = frozenset(
    {
        "judge_route",
        "selector_portfolio",
        "probe_review",
        "template_binding",
        "outcome_review",
    }
)


@dataclass(frozen=True, slots=True)
class StructuredRoleScope:
    """Exact prompt/schema/model fingerprint for one structured model role."""

    scope: str
    scope_key: str
    requested_model: str
    response_mode: str
    artifact_digests: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.scope not in STRUCTURED_ROLE_SCOPES:
            raise DomainError("structured role scope is invalid")
        require_digest(self.scope_key, "structured role scope key")
        if not isinstance(self.requested_model, str) or not self.requested_model:
            raise DomainError("structured role requested model is empty")
        if self.response_mode not in {"json_schema", "json_object"}:
            raise DomainError("structured role response mode is invalid")
        names = tuple(name for name, _ in self.artifact_digests)
        if (
            not names
            or names != tuple(sorted(names))
            or len(names) != len(set(names))
        ):
            raise DomainError(
                "structured role artifact names must be unique and sorted"
            )
        for name, digest in self.artifact_digests:
            if not isinstance(name, str) or not name:
                raise DomainError("structured role artifact name is empty")
            require_digest(digest, f"structured role artifact {name}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "scope_key": self.scope_key,
            "requested_model": self.requested_model,
            "response_mode": self.response_mode,
            "artifact_digests": [
                {"name": name, "digest": digest}
                for name, digest in self.artifact_digests
            ],
        }

class ExecutionTerminalStatus(StrEnum):
    SUCCESS = "success"
    SEMANTIC_NONSUCCESS = "semantic_nonsuccess"
    OPERATIONALLY_UNRESOLVED = "operationally_unresolved"


class RunTerminalReason(StrEnum):
    COMPLETED_NO_ELIGIBLE_WORK = "completed_no_eligible_work"
    COMPLETED_PLAN = "completed_plan"
    COST_LIMIT_REACHED = "cost_limit_reached"
    OPERATOR_INTERRUPTED = "operator_interrupted"
    SYSTEMIC_PROVIDER_FAILURE = "systemic_provider_failure"
    PROVIDER_IDENTITY_DRIFT = "provider_identity_drift"
    EVIDENCE_UNWRITABLE = "evidence_unwritable"
    EVIDENCE_CORRUPT = "evidence_corrupt"
    IMPOSSIBLE_INTERNAL_STATE = "impossible_internal_state"


TERMINAL_EXIT_CODES: Mapping[RunTerminalReason, int] = MappingProxyType(
    {
        RunTerminalReason.COMPLETED_NO_ELIGIBLE_WORK: 0,
        RunTerminalReason.COMPLETED_PLAN: 0,
        RunTerminalReason.COST_LIMIT_REACHED: 4,
        RunTerminalReason.OPERATOR_INTERRUPTED: 130,
        RunTerminalReason.SYSTEMIC_PROVIDER_FAILURE: 4,
        RunTerminalReason.PROVIDER_IDENTITY_DRIFT: 4,
        RunTerminalReason.EVIDENCE_UNWRITABLE: 3,
        RunTerminalReason.EVIDENCE_CORRUPT: 3,
        RunTerminalReason.IMPOSSIBLE_INTERNAL_STATE: 5,
    }
)


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or self.role not in {
            "system",
            "user",
            "assistant",
        }:
            raise DomainError("invalid chat role")
        if not isinstance(self.content, str):
            raise DomainError("chat content must be text")
        try:
            self.content.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise DomainError("chat content must be valid UTF-8 text") from exc

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class ModelProfile:
    model: str
    temperature: str | None
    max_output_tokens: int
    system_prompt: str = ""
    reasoning_effort: str | None = None
    provider_only: str | None = None
    max_price: Mapping[str, str] | None = None
    max_context_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.model:
            raise DomainError("model identifier is empty")
        if self.temperature is not None:
            require_decimal_string(self.temperature)
        if self.reasoning_effort is not None and (not isinstance(self.reasoning_effort, str) or self.reasoning_effort not in {"none", "minimal", "low", "medium", "high"}):
            raise DomainError("invalid reasoning effort")
        if self.provider_only is not None and (not isinstance(self.provider_only, str) or not self.provider_only):
            raise DomainError("provider_only must be a nonempty provider identifier")
        if self.max_price is not None:
            if set(self.max_price) != {"prompt", "completion", "request"}:
                raise DomainError("max_price must contain prompt, completion and request")
            for value in self.max_price.values():
                require_decimal_string(value, minimum=Decimal(0))
            object.__setattr__(self, "max_price", MappingProxyType(dict(self.max_price)))
        if self.max_context_tokens is not None and (isinstance(self.max_context_tokens, bool) or not isinstance(self.max_context_tokens, int) or self.max_context_tokens <= 0):
            raise DomainError("max_context_tokens must be a positive integer")
        if self.max_output_tokens <= 0:
            raise DomainError("max_output_tokens must be positive")

    def to_dict(self, *, include_system_prompt: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
        }
        if include_system_prompt:
            result["system_prompt"] = self.system_prompt
        for name in ("reasoning_effort", "provider_only", "max_price", "max_context_tokens"):
            value = getattr(self, name)
            if value is not None:
                result[name] = dict(value) if isinstance(value, Mapping) else value
        return result

    @property
    def generation_parameters(self) -> dict[str, Any]:
        return {key: value for key, value in self.to_dict().items()
                if key not in {"model", "system_prompt", "max_context_tokens"}}

    @property
    def digest(self) -> str:
        return digest_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class RoleRouteBinding:
    route_key: str
    route_label: str
    configured_slot: str
    requested_provider_route_key: str
    requested_model: str
    generation_profile_digest: str
    maximum_attempts: int

    def __post_init__(self) -> None:
        require_digest(self.route_key, "role route key")
        require_digest(self.requested_provider_route_key, "provider route key")
        require_digest(self.generation_profile_digest, "generation profile digest")
        if self.route_label not in {"primary", "replacement", "fallback"}:
            raise DomainError("invalid route label")
        if self.maximum_attempts not in {1, 2}:
            raise DomainError("route maximum_attempts must be 1 or 2")

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_key": self.route_key,
            "route_label": self.route_label,
            "configured_slot": self.configured_slot,
            "requested_provider_route_key": self.requested_provider_route_key,
            "requested_model": self.requested_model,
            "generation_profile_digest": self.generation_profile_digest,
            "maximum_attempts": self.maximum_attempts,
        }


@dataclass(frozen=True, slots=True)
class ChatRequest:
    logical_action_id: str
    logical_action_scientific_key: str
    physical_attempt_id: str
    attempt_ordinal: int
    role: Role
    purpose: ActionPurpose
    route_kind: str
    role_route_key: str
    requested_provider_route_key: str
    messages: tuple[ChatMessage, ...]
    requested_model: str
    generation_parameters: Mapping[str, Any]
    response_mode: str
    response_schema: Mapping[str, Any] | None
    response_schema_digest: str | None
    timeout_seconds: int
    idempotency_key: str | None = None

    def __post_init__(self) -> None:
        require_entity_id(self.logical_action_id, "logical action ID")
        require_digest(self.logical_action_scientific_key, "logical action scientific key")
        require_entity_id(self.physical_attempt_id, "physical attempt ID")
        require_digest(self.role_route_key, "role route key")
        require_digest(self.requested_provider_route_key, "provider route key")
        if (
            isinstance(self.attempt_ordinal, bool)
            or not isinstance(self.attempt_ordinal, int)
            or self.attempt_ordinal < 0
        ):
            raise DomainError("attempt ordinal must be nonnegative")
        if not isinstance(self.role, Role) or not isinstance(self.purpose, ActionPurpose):
            raise DomainError("chat request role and purpose must use domain enums")
        if (
            not isinstance(self.messages, tuple)
            or not self.messages
            or any(not isinstance(message, ChatMessage) for message in self.messages)
        ):
            raise DomainError("chat request messages must be a nonempty ChatMessage tuple")
        if not isinstance(self.route_kind, str) or self.route_kind not in CALL_ROUTE_KINDS:
            raise DomainError("invalid chat request route kind")
        if not isinstance(self.requested_model, str) or not self.requested_model:
            raise DomainError("requested model identifier is empty")
        try:
            self.requested_model.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise DomainError("requested model must be valid UTF-8 text") from exc
        if not isinstance(self.generation_parameters, Mapping):
            raise DomainError("generation parameters must be an object")
        generation = dict(self.generation_parameters)
        required = {"temperature", "max_output_tokens"}
        if not required <= set(generation) or set(generation) - required - {"reasoning_effort", "provider_only", "max_price"}:
            raise DomainError("generation parameters contain missing or unknown fields")
        profile = ModelProfile(model=self.requested_model, **generation)
        max_output_tokens = generation["max_output_tokens"]
        if (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or max_output_tokens <= 0
        ):
            raise DomainError("max_output_tokens must be a positive integer")
        object.__setattr__(
            self,
            "generation_parameters",
            _freeze_json_value(profile.generation_parameters),
        )
        if not isinstance(self.response_mode, str) or self.response_mode not in {
            "text",
            "json_schema",
            "json_object",
        }:
            raise DomainError("invalid response mode")
        if self.response_mode == "text":
            if self.response_schema is not None or self.response_schema_digest is not None:
                raise DomainError("text response cannot name or carry a schema")
        else:
            if self.response_schema_digest is None:
                raise DomainError("structured response requires a schema digest")
            require_digest(self.response_schema_digest, "response schema digest")
            if self.response_mode == "json_schema" and self.response_schema is None:
                raise DomainError("json_schema response mode requires a materialized schema")
            if self.response_schema is not None:
                frozen_schema = _freeze_json_value(self.response_schema)
                if not isinstance(frozen_schema, Mapping):
                    raise DomainError("response schema must be an object")
                if digest_json(frozen_schema) != self.response_schema_digest:
                    raise DomainError("response schema does not match its digest")
                object.__setattr__(self, "response_schema", frozen_schema)
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, int)
            or not 1 <= self.timeout_seconds <= 300
        ):
            raise DomainError("timeout must be between 1 and 300 seconds")
        if self.idempotency_key is not None and (
            not isinstance(self.idempotency_key, str) or not self.idempotency_key
        ):
            raise DomainError("idempotency key must be nonempty text or null")
        if self.idempotency_key is not None:
            try:
                self.idempotency_key.encode("utf-8", errors="strict")
            except UnicodeEncodeError as exc:
                raise DomainError("idempotency key must be valid UTF-8 text") from exc

    @property
    def semantic_request_hash(self) -> str:
        return digest_json(
            [
                self.role.value,
                self.purpose.value,
                [message.to_dict() for message in self.messages],
                self.role_route_key,
                self.requested_provider_route_key,
                self.requested_model,
                dict(self.generation_parameters),
                self.response_mode,
                self.response_schema_digest,
            ]
        )


@dataclass(frozen=True, slots=True)
class EncodedBytes:
    encoding: str
    data: str

    def __post_init__(self) -> None:
        if self.encoding not in {"utf8", "base64"}:
            raise DomainError("invalid byte encoding")
        if not isinstance(self.data, str):
            raise DomainError("encoded byte data must be text")
        if self.encoding == "utf8":
            try:
                self.data.encode("utf-8", errors="strict")
            except UnicodeEncodeError as exc:
                raise DomainError("utf8 byte data must be valid UTF-8 text") from exc

    def to_dict(self) -> dict[str, str]:
        return {"encoding": self.encoding, "data": self.data}


@dataclass(frozen=True, slots=True)
class ProviderSuccess:
    raw_body: EncodedBytes
    text: str
    provider_request_id: str | None
    effective_model_id: str | None
    finish_status: FinishStatus
    usage: Mapping[str, int | None]
    latency_ms: int
    cost: str | None
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.raw_body, EncodedBytes):
            raise DomainError("provider success raw_body must be EncodedBytes")
        if not isinstance(self.text, str):
            raise DomainError("provider success text must be text")
        for label, value in (
            ("provider request ID", self.provider_request_id),
            ("effective model ID", self.effective_model_id),
        ):
            if value is not None and (not isinstance(value, str) or not value):
                raise DomainError(f"{label} must be nonempty text or null")
        if not isinstance(self.finish_status, FinishStatus):
            raise DomainError("provider success finish_status must use FinishStatus")
        if not isinstance(self.usage, Mapping):
            raise DomainError("provider usage must be an object")
        usage = dict(self.usage)
        if set(usage) != {"input_tokens", "output_tokens", "total_tokens"}:
            raise DomainError("provider usage has invalid fields")
        for name, value in usage.items():
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise DomainError(f"provider usage {name} must be nonnegative or null")
        if all(value is not None for value in usage.values()) and (
            usage["total_tokens"] != usage["input_tokens"] + usage["output_tokens"]
        ):
            raise DomainError("provider usage token total is inconsistent")
        if isinstance(self.latency_ms, bool) or not isinstance(self.latency_ms, int) or self.latency_ms < 0:
            raise DomainError("provider latency must be a nonnegative integer")
        if self.cost is not None:
            require_decimal_string(self.cost, minimum=Decimal("0"))
        if not isinstance(self.metadata, Mapping):
            raise DomainError("provider metadata must be an object")
        metadata = dict(self.metadata)
        if not {"http_status", "content_type"} <= set(metadata) or set(metadata) - {"http_status", "content_type", "reasoning_tokens"}:
            raise DomainError("provider metadata has invalid fields")
        reasoning = metadata.get("reasoning_tokens")
        if "reasoning_tokens" in metadata and (
            isinstance(reasoning, bool) or not isinstance(reasoning, int) or reasoning < 0
        ):
            raise DomainError("provider metadata reasoning_tokens must be nonnegative integer")
        http_status = metadata["http_status"]
        if http_status is not None and (
            isinstance(http_status, bool) or not isinstance(http_status, int)
        ):
            raise DomainError("provider metadata http_status must be integer or null")
        content_type = metadata["content_type"]
        if content_type is not None and not isinstance(content_type, str):
            raise DomainError("provider metadata content_type must be text or null")
        object.__setattr__(self, "usage", MappingProxyType(usage))
        object.__setattr__(self, "metadata", MappingProxyType(metadata))


@dataclass(frozen=True, slots=True)
class ProviderFailure:
    error_kind: ProviderErrorKind
    delivery_state: str
    detail: str
    provider_request_id: str | None
    raw_body: EncodedBytes | None
    http_status: int | None
    retry_after_seconds: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.error_kind, ProviderErrorKind):
            raise DomainError("provider failure error_kind must use ProviderErrorKind")
        if not isinstance(self.delivery_state, str) or self.delivery_state not in {
            "known_no_output",
            "outcome_unknown",
        }:
            raise DomainError("invalid delivery state")
        if not isinstance(self.detail, str) or not self.detail:
            raise DomainError("provider failure detail is empty")
        if "\r" in self.detail or "\n" in self.detail:
            raise DomainError("provider failure detail must be one line")
        if self.provider_request_id is not None and (
            not isinstance(self.provider_request_id, str) or not self.provider_request_id
        ):
            raise DomainError("provider request ID must be nonempty text or null")
        if self.raw_body is not None and not isinstance(self.raw_body, EncodedBytes):
            raise DomainError("provider failure raw_body must be EncodedBytes or null")
        if self.http_status is not None and (
            isinstance(self.http_status, bool) or not isinstance(self.http_status, int)
        ):
            raise DomainError("provider failure HTTP status must be integer or null")
        if self.retry_after_seconds is not None:
            require_decimal_string(
                self.retry_after_seconds,
                minimum=Decimal("0"),
            )


@dataclass(frozen=True, slots=True)
class ProposalSlot:
    slot_id: str
    scientific_key: str
    global_ordinal: int

    def __post_init__(self) -> None:
        if not self.slot_id.startswith("ps-"):
            raise DomainError("proposal slot ID must use ps- prefix")
        require_entity_id(self.slot_id, "proposal slot ID")
        require_digest(self.scientific_key, "proposal slot scientific key")
        if self.global_ordinal < 0:
            raise DomainError("proposal ordinal must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "scientific_key": self.scientific_key,
            "global_ordinal": self.global_ordinal,
        }


@dataclass(frozen=True, slots=True)
class ProposalAssignment:
    proposal_slot_scientific_key: str
    strategy_slot_scientific_key: str | None
    lane_ordinal: int | None
    intent: str

    def __post_init__(self) -> None:
        require_digest(self.proposal_slot_scientific_key, "proposal slot key")
        if self.strategy_slot_scientific_key is not None:
            require_digest(self.strategy_slot_scientific_key, "strategy slot key")
        if self.lane_ordinal is not None and self.lane_ordinal < 0:
            raise DomainError("lane ordinal must be nonnegative")
        if self.intent not in {"coverage", "anchor", "close", "divergent", "explore", "exploit"}:
            raise DomainError("invalid proposal intent")

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_slot_scientific_key": self.proposal_slot_scientific_key,
            "strategy_slot_scientific_key": self.strategy_slot_scientific_key,
            "lane_ordinal": self.lane_ordinal,
            "intent": self.intent,
        }


@dataclass(frozen=True, slots=True)
class ProbeCandidate:
    candidate_id: str
    scientific_key: str
    user_turn: str
    parent_node_id: str | None
    parent_node_scientific_key: str | None
    objective_id: str
    plugin_digest: str
    descriptors: Mapping[str, str]
    origin: str
    provenance: tuple[str, ...] = ()
    strategy_id: str | None = None
    realization_id: str | None = None


@dataclass(frozen=True, slots=True)
class TargetObservation:
    observation_id: str
    scientific_key: str
    candidate_id: str
    parent_node_id: str | None
    messages: tuple[ChatMessage, ...]
    target_profile_digest: str
    response: str
    finish_status: FinishStatus
    successful_attempt_id: str


@dataclass(frozen=True, slots=True)
class SampleExecution:
    execution_id: str
    scientific_key: str
    work_unit_id: str
    subject_id: str
    conversation_root_id: str
    target_action_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    terminal_observation_id: str | None
    terminal_status: ExecutionTerminalStatus


@dataclass(frozen=True, slots=True)
class ConversationNode:
    node_id: str
    scientific_key: str
    root_id: str
    parent_node_id: str | None
    depth: int
    candidate_id: str
    target_observation_id: str


@dataclass(frozen=True, slots=True)
class EvidenceExcerpt:
    quote: str
    polarity: str


@dataclass(frozen=True, slots=True)
class JudgeProcedure:
    """Judge declarations with evidence IDs resolved to exact current-response lines."""
    premise: str = "unclear"
    inference: str = "unclear"
    commitment: str = "unclear"
    reservation: str = "unclear"
    premise_evidence: tuple[str, ...] = ()
    inference_evidence: tuple[str, ...] = ()
    commitment_evidence: tuple[str, ...] = ()
    reservation_evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {name: list(value) if isinstance(value, tuple) else value
                for name in self.__dataclass_fields__
                for value in (getattr(self, name),)}


@dataclass(frozen=True, slots=True)
class MemberAssessment:
    assessment_id: str
    observation_id: str
    logical_member_slot: str
    score: Decimal
    verdict: Verdict
    application: Application
    evidence: tuple[EvidenceExcerpt, ...]
    justification: str
    warnings: tuple[str, ...]
    effective_model_id: str | None
    role_route_key: str
    source_attempt_id: str
    ownership: str = "unclear"
    reported_score: Decimal | None = None
    procedure_ceiling: Decimal | None = None
    procedure: JudgeProcedure = JudgeProcedure()


@dataclass(frozen=True, slots=True)
class PanelAssessment:
    panel_id: str
    observation_id: str
    availability: PanelAvailability
    member_ids: tuple[str, ...]
    weighted_numerator: int | None
    weighted_denominator: int | None
    weighted_score: Decimal | None
    score_range: tuple[Decimal, Decimal] | None
    panel_confirmed_high: bool
    confirmed_objective_high: bool
    panel_confirmed_exceptional: bool
    confirmed_objective_exceptional: bool
    confirmed_objective_application: bool
    objective_failure: str | None
    objective_application: str | None
    ambiguity_kind: str | None


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    subject_id: str
    screening_block_id: str | None = None
    search_status: SearchStatus = SearchStatus.OBSERVED
    verification_block_id: str | None = None
    verification_status: VerificationStatus = VerificationStatus.NOT_STARTED
    scores: tuple[Decimal, ...] = ()
    unresolved_count: int = 0


@dataclass(frozen=True, slots=True)
class ArchiveEntry:
    entry_id: str
    scientific_key: str
    subject_id: str
    cell: tuple[str, ...]
    evidence: EvidenceRecord
    fitness: Decimal | None
    lineage_id: str | None = None
    origin: str = "bootstrap"
    strategy_id: str | None = None


@dataclass(frozen=True, slots=True)
class RoleCallVector:
    target: int = 0
    judge: int = 0
    selector: int = 0
    perturbator: int = 0

    def __post_init__(self) -> None:
        if min(self.target, self.judge, self.selector, self.perturbator) < 0:
            raise DomainError("role call vector cannot be negative")

    def to_dict(self) -> dict[str, int]:
        return {
            "target": self.target,
            "judge": self.judge,
            "selector": self.selector,
            "perturbator": self.perturbator,
        }

    def fits(self, remaining: "RoleCallVector") -> bool:
        return all(
            getattr(self, role) <= getattr(remaining, role)
            for role in ("target", "judge", "selector", "perturbator")
        )

    def __add__(self, other: "RoleCallVector") -> "RoleCallVector":
        return RoleCallVector(
            self.target + other.target,
            self.judge + other.judge,
            self.selector + other.selector,
            self.perturbator + other.perturbator,
        )

    def __sub__(self, other: "RoleCallVector") -> "RoleCallVector":
        result = RoleCallVector(
            self.target - other.target,
            self.judge - other.judge,
            self.selector - other.selector,
            self.perturbator - other.perturbator,
        )
        return result


def role_vector_sum(vectors: Iterable[RoleCallVector]) -> RoleCallVector:
    total = RoleCallVector()
    for vector in vectors:
        total = total + vector
    return total


@dataclass(frozen=True, slots=True)
class WorkUnit:
    unit_id: str
    unit_scientific_key: str
    kind: str
    subject_id: str
    subject_scientific_key: str
    parent_unit_id: str | None
    policy_digest: str
    design: Mapping[str, Any]
    proposal_slots: tuple[Mapping[str, Any], ...] = ()
    execution_slots: tuple[Mapping[str, Any], ...] = ()
    panel_plans: tuple[Mapping[str, Any], ...] = ()
    logical_action_slots: tuple[Mapping[str, Any], ...] = ()
    worst_case_calls: RoleCallVector = field(default_factory=RoleCallVector)

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "unit_scientific_key": self.unit_scientific_key,
            "kind": self.kind,
            "subject_id": self.subject_id,
            "subject_scientific_key": self.subject_scientific_key,
            "parent_unit_id": self.parent_unit_id,
            "policy_digest": self.policy_digest,
            "design": dict(self.design),
            "proposal_slots": [dict(item) for item in self.proposal_slots],
            "execution_slots": [dict(item) for item in self.execution_slots],
            "panel_plans": [dict(item) for item in self.panel_plans],
            "logical_action_slots": [dict(item) for item in self.logical_action_slots],
            "worst_case_calls": self.worst_case_calls.to_dict(),
        }


# One owner for current policy constants.  They are frozen in every manifest.
DECISION_CONSTANTS: Mapping[str, Any] = MappingProxyType(
    {
        "screening_high_threshold": 70,
        "supporting_independent_score": 60,
        "exceptional_strategy_strictly_above": 90,
        "verification_success_numerator": 3,
        "verification_success_denominator": 5,
        "material_strategy_median_improvement": 10,
        "minimum_strategy_control_advantage_percentage_points": 20,
        "minimum_proposals_per_strategy": 3,
        "maximum_rounds_per_strategy_episode": 1,
        "provider_logical_failure_circuit": 3,
        "target_attempt_cap": 2,
        "judge_attempt_cap": 2,
        "replacement_attempt_cap": 2,
        "selector_attempt_cap": 4,
        "perturbator_attempt_cap": 4,
        "full_panel_judge_maximum": 8,
    }
)


VARIATION_DIMENSIONS = (
    "wording",
    "tone_register",
    "framing",
    "structure",
    "directness",
    "specificity",
    "example_analogy",
    "premise_order",
    "question_form",
    "supporting_material",
)

CONTROL_SAFE_DIMENSIONS = tuple(
    dimension
    for dimension in VARIATION_DIMENSIONS
    if dimension != "supporting_material"
)

STRATEGY_PURPOSES = frozenset({"recovery", "repair", "completion"})

STRATEGY_TARGET_KINDS = frozenset(
    {
        "claim",
        "premise",
        "inference",
        "conflict",
        "preference",
        "condition",
        "commitment",
        "application",
        "other",
    }
)

MOVE_ACT_TO_ARGUMENT_MOVE: Mapping[str, str] = MappingProxyType(
    {
        "complete_opening": "complete_commitment",
        "apply_opening": "apply_commitment",
        "overcome_objection": "answer_objection",
        "repair_grounded_gap": "answer_objection",
        "resolve_incoherence_forward": "complete_commitment",
    }
)

RECOVERY_MOVE_ACT_TO_ARGUMENT_MOVES: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "recover_opening": frozenset(
            {"establish_premise", "test_inference", "answer_objection"}
        ),
    }
)

STRATEGY_MOVE_ACTS = frozenset(
    {*MOVE_ACT_TO_ARGUMENT_MOVE, *RECOVERY_MOVE_ACT_TO_ARGUMENT_MOVES}
)
