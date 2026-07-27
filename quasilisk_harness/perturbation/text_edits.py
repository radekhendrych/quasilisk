"""Exact-span helpers for semantically constrained prompt repairs."""

from __future__ import annotations

import re

from quasilisk_harness.objectives.moral_common import classify_answer_shape
from quasilisk_harness.perturbation.fingerprint import normalized_surface
from quasilisk_harness.perturbation.models import (
    EditableRegion,
    EditableRegionKind,
)


_RESPONSE_IMPERATIVE = re.compile(
    r"^(?:answer|respond|state|give|provide|present|offer|lead)\b",
    re.IGNORECASE,
)
_SENTENCE = re.compile(r"[^.!?]+(?:[.!?]+|$)")
_RESPONSE_SHAPES = {
    "binary_with_explanation",
    "binary_verdict",
    "direct_verdict",
    "single_verdict",
    "verdict_with_explanation",
}
_RESPONSE_FORMAT_WORDS = {
    "a",
    "and",
    "answer",
    "basis",
    "brief",
    "briefly",
    "by",
    "clear",
    "clearly",
    "concise",
    "concisely",
    "conclusion",
    "considered",
    "definitive",
    "direct",
    "directly",
    "explain",
    "explanation",
    "final",
    "first",
    "follow",
    "followed",
    "following",
    "give",
    "it",
    "justification",
    "lead",
    "no",
    "offer",
    "one",
    "or",
    "own",
    "present",
    "provide",
    "reason",
    "reasoning",
    "reasons",
    "respond",
    "response",
    "single",
    "state",
    "supporting",
    "then",
    "unambiguous",
    "verdict",
    "with",
    "yes",
    "your",
}


def response_instruction_reasons(
    text: str,
    *,
    expected_shape: str | None = None,
) -> tuple[str, ...]:
    """Return why text is not a pure response-format instruction."""

    cleaned = normalized_surface(text)
    reasons: list[str] = []
    if not cleaned or not _RESPONSE_IMPERATIVE.search(cleaned):
        reasons.append("missing_response_imperative")
    shape = classify_answer_shape(cleaned)
    if shape not in _RESPONSE_SHAPES:
        reasons.append(f"unrecognized_response_shape:{shape}")
    if expected_shape is not None and shape != expected_shape:
        reasons.append(
            f"response_shape_mismatch:{shape}!={expected_shape}"
        )
    words = re.findall(r"[A-Za-z]+", cleaned.lower())
    if (
        expected_shape == "verdict_with_explanation"
        and {"yes", "no"}.intersection(words)
    ):
        reasons.append("binary_token_in_nonbinary_response_contract")
    if len(words) > 22:
        reasons.append(f"response_instruction_too_long:{len(words)}>22")
    unexpected = sorted(set(words) - _RESPONSE_FORMAT_WORDS)
    if unexpected:
        reasons.append(
            "nonformat_response_content:" + ",".join(unexpected)
        )
    without_terminal = cleaned.rstrip(".!")
    if re.search(r"[.!?;:]", without_terminal):
        reasons.append("compound_response_instruction")
    if any(mark in cleaned for mark in ('"', "'", "“", "”")):
        reasons.append("quoted_response_content")
    return tuple(dict.fromkeys(reasons))


def is_response_instruction(
    text: str,
    *,
    expected_shape: str | None = None,
) -> bool:
    return not response_instruction_reasons(
        text,
        expected_shape=expected_shape,
    )


def find_response_instruction_regions(
    prompt: str,
) -> tuple[EditableRegion, ...]:
    """Find standalone, format-only response instructions in normalized text."""

    surface = normalized_surface(prompt)
    regions: list[EditableRegion] = []
    for match in _SENTENCE.finditer(surface):
        raw = match.group(0)
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw) - len(raw.rstrip())
        start = match.start() + leading
        end = match.end() - trailing
        instruction = surface[start:end]
        if not is_response_instruction(instruction):
            continue
        regions.append(
            EditableRegion(
                kind=EditableRegionKind.RESPONSE_INSTRUCTION,
                prefix=surface[:start],
                original_text=instruction,
                suffix=surface[end:],
            )
        )
    return tuple(regions)


def replace_editable_region(
    parent_text: str,
    region: EditableRegion,
    replacement: str,
) -> str:
    """Replace one authorized span, rejecting stale region descriptors."""

    normalized_parent = normalized_surface(parent_text)
    if normalized_parent != region.parent_text:
        raise ValueError("editable region does not describe the supplied parent")
    cleaned_replacement = normalized_surface(replacement)
    if not cleaned_replacement:
        raise ValueError("editable region replacement must not be empty")
    return (
        f"{region.prefix}{cleaned_replacement}{region.suffix}"
    )


def extract_editable_replacement(
    *,
    parent_text: str,
    child_text: str,
    region: EditableRegion,
) -> tuple[str | None, tuple[str, ...]]:
    """Extract a replacement only when normalized text outside it is unchanged."""

    parent_surface = normalized_surface(parent_text)
    child_surface = normalized_surface(child_text)
    reasons: list[str] = []
    if parent_surface != region.parent_text:
        reasons.append("editable_region_parent_mismatch")
        return None, tuple(reasons)
    if not child_surface.startswith(region.prefix):
        reasons.append("editable_region_prefix_changed")
    if region.suffix and not child_surface.endswith(region.suffix):
        reasons.append("editable_region_suffix_changed")
    replacement_end = (
        len(child_surface) - len(region.suffix)
        if region.suffix
        else len(child_surface)
    )
    replacement_start = len(region.prefix)
    if replacement_end < replacement_start:
        reasons.append("editable_region_boundary_overlap")
    if reasons:
        return None, tuple(reasons)
    replacement = child_surface[replacement_start:replacement_end]
    if replacement != replacement.strip():
        reasons.append("editable_region_boundary_whitespace_changed")
    if not replacement:
        reasons.append("editable_region_empty")
    if replacement == region.original_text:
        reasons.append("editable_region_unchanged")
    return replacement or None, tuple(reasons)
