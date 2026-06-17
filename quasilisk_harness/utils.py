"""Small shared utilities for the Quasilisk harness."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


MAX_PROMPT_CHARS = 1800

FORBIDDEN_VARIANT_PATTERNS = (
    r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions",
    r"\bjailbreak\b",
    r"\bdan\b",
    r"developer\s+mode",
    r"bypass\s+(?:your\s+)?(?:safety|policy|policies|guardrails|filters)",
    r"do\s+not\s+(?:follow|obey)\s+(?:your\s+)?(?:policy|policies|rules|instructions)",
    r"reveal\s+(?:the\s+)?(?:hidden|system|developer)\s+prompt",
    r"\b(?:base64|rot13|cipher|code\s+words)\b",
    r"pretend\s+(?:you\s+have\s+no|there\s+are\s+no)\s+(?:rules|restrictions|policies)",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def short_hash(text: str, length: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def excerpt(text: str, chars: int = 900) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= chars:
        return cleaned
    return cleaned[: chars - 3].rstrip() + "..."


def split_models(values: list[str] | None) -> list[str]:
    models: list[str] = []
    for value in values or []:
        models.extend(part.strip() for part in value.split(",") if part.strip())
    return models


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "model"


def parse_json_object(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if 0 <= start < end:
            return json.loads(cleaned[start : end + 1])
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if 0 <= start < end:
            return json.loads(cleaned[start : end + 1])
        raise


def _prompt_field_fallback(text: str) -> list[dict[str, str]]:
    variants: list[dict[str, str]] = []
    for match in re.finditer(r'"prompt"\s*:\s*"((?:\\.|[^"\\])*)"', text, re.DOTALL):
        raw_prompt = match.group(1)
        try:
            prompt = json.loads(f'"{raw_prompt}"').strip()
        except json.JSONDecodeError:
            prompt = raw_prompt.strip()
        if prompt:
            variants.append({"prompt": prompt, "mutation": "prompt_field_fallback"})
    return variants


def extract_variants(text: str) -> list[dict[str, str]]:
    try:
        parsed = parse_json_object(text)
    except (json.JSONDecodeError, ValueError):
        variants = _prompt_field_fallback(text)
        if variants:
            return variants
        variants = []
        for line in text.splitlines():
            line = line.strip(" -\t")
            if re.match(r'^[\{\}\[\],]*$', line):
                continue
            if re.match(r'^"?(?:prompt|mutation|variants|text)"?\s*:', line, re.IGNORECASE):
                continue
            if line.startswith(("{", "[", "}", "]")):
                continue
            if len(line) >= 30:
                variants.append({"prompt": line, "mutation": "line_fallback"})
        return variants

    if isinstance(parsed, dict):
        raw_variants = parsed.get("variants") or parsed.get("prompts") or []
    else:
        raw_variants = parsed

    variants: list[dict[str, str]] = []
    for item in raw_variants:
        if isinstance(item, str):
            prompt = item.strip()
            if prompt:
                variants.append({"prompt": prompt, "mutation": "string_variant"})
        elif isinstance(item, dict):
            prompt = str(item.get("prompt") or item.get("text") or "").strip()
            mutation = str(item.get("mutation") or item.get("rationale") or "").strip()
            if prompt:
                variants.append({"prompt": prompt, "mutation": mutation})
    return variants


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def write_json(path: Path, item: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            tmp_name = handle.name
            json.dump(item, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
        os.replace(tmp_name, path)
    finally:
        if tmp_name and os.path.exists(tmp_name):
            os.unlink(tmp_name)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            tmp_name = handle.name
            handle.write(text)
            handle.flush()
        os.replace(tmp_name, path)
    finally:
        if tmp_name and os.path.exists(tmp_name):
            os.unlink(tmp_name)


def write_status(out_dir: Path, item: dict[str, Any]) -> None:
    write_json(out_dir / "status.json", {"timestamp": utc_now(), **item})


def has_forbidden_variant_pattern(prompt: str) -> str | None:
    lower = prompt.lower()
    for pattern in FORBIDDEN_VARIANT_PATTERNS:
        if re.search(pattern, lower):
            return pattern
    return None
