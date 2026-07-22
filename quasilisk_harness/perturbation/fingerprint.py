"""Parent-relative prompt fingerprints for constrained mutations."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from types import ModuleType
from typing import Any

from quasilisk_harness.objectives.moral_common import (
    classify_answer_shape,
    contains_argument_adjudication,
    contains_conditional_scaffold,
    contains_philosophical_label,
)
from quasilisk_harness.perturbation.models import PromptFingerprint
from quasilisk_harness.perturbation.space import case_keywords_for_objective
from quasilisk_harness.utils import short_hash


_HARM_TERMS = (
    "harm",
    "suffering",
    "injury",
    "coercion",
    "abuse",
    "neglect",
    "patient",
    "emergency",
    "violence",
)

FEATURE_ALIASES = {
    "argument_frame": "argument_route",
    "pressure_style": "pressure_pattern",
    "output_shape": "commitment_form",
    "direct_adjudication_form": "answer_shape",
}


def normalized_tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[\w'-]+", text.lower()))


def normalized_surface(text: str) -> str:
    """Normalize whitespace while preserving punctuation and letter case."""

    return re.sub(r"\s+", " ", str(text)).strip()


def canonical_factor(factor: str) -> str:
    return FEATURE_ALIASES.get(str(factor), str(factor))


def canonical_dimensions(dimensions: dict[str, Any]) -> dict[str, Any]:
    canonical: dict[str, Any] = {}
    for key, value in dimensions.items():
        if str(key) in FEATURE_ALIASES:
            continue
        canonical[canonical_factor(str(key))] = value
    for key, value in dimensions.items():
        canonical_key = canonical_factor(str(key))
        if canonical_key in canonical and canonical[canonical_key] is not None:
            continue
        canonical[canonical_key] = value
    return canonical


def canonical_factors(factors: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(canonical_factor(str(item)) for item in factors))


def token_similarity(
    parent: PromptFingerprint,
    child: PromptFingerprint,
) -> float:
    return round(
        SequenceMatcher(
            None,
            parent.normalized_tokens,
            child.normalized_tokens,
            autojunk=False,
        ).ratio(),
        4,
    )


def _infer_case_family(objective_id: str, prompt: str) -> str:
    lower = prompt.lower()
    for family, keywords in case_keywords_for_objective(objective_id).items():
        if any(keyword.lower() in lower for keyword in keywords):
            return family
    return "objective_anchor"


def _case_detail_signature(objective_id: str, prompt: str) -> str:
    lower = prompt.lower()
    hits: list[str] = []
    for keywords in case_keywords_for_objective(objective_id).values():
        for keyword in keywords:
            normalized = str(keyword).strip().lower()
            if normalized and normalized in lower and normalized not in hits:
                hits.append(normalized)
    if not hits:
        return "none"
    return "|".join(sorted(hits))


def _semantic_anchors(objective: ModuleType, prompt: str) -> tuple[str, ...]:
    lower = prompt.lower()
    anchors = getattr(objective, "SEMANTIC_ANCHORS", ())
    matched: list[str] = []
    for anchor in anchors:
        text = str(anchor).strip()
        if text and text.lower() in lower and text.lower() not in matched:
            matched.append(text.lower())
        if len(matched) >= 10:
            break
    return tuple(matched)


def build_prompt_fingerprint(
    *,
    objective: ModuleType,
    prompt: str,
) -> PromptFingerprint:
    tokens = normalized_tokens(prompt)
    surface_text = normalized_surface(prompt)
    lower = prompt.lower()
    try:
        features = {
            str(key): value
            for key, value in objective.classify_features(prompt).items()
        }
    except Exception:
        features = {}
    dimensions: dict[str, Any] = {
        **features,
        "case_family": _infer_case_family(objective.OBJECTIVE_ID, prompt),
        "conditional_scaffold": bool(contains_conditional_scaffold(lower)),
        "category_label": bool(contains_philosophical_label(lower)),
        "direct_adjudication": contains_argument_adjudication(prompt),
        "answer_shape": classify_answer_shape(prompt),
        "harm_application": any(term in lower for term in _HARM_TERMS),
        "case_detail": _case_detail_signature(objective.OBJECTIVE_ID, prompt),
    }
    spans = list(_semantic_anchors(objective, prompt))
    fingerprint_hook = getattr(objective, "fingerprint_prompt", None)
    if callable(fingerprint_hook):
        custom = fingerprint_hook(prompt)
        if isinstance(custom, dict):
            custom_dimensions = custom.get("dimensions")
            if isinstance(custom_dimensions, dict):
                dimensions.update(
                    {str(key): value for key, value in custom_dimensions.items()}
                )
            custom_spans = custom.get("protected_spans")
            if isinstance(custom_spans, (list, tuple)):
                for span in custom_spans:
                    text = str(span).strip().lower()
                    if text and text not in spans:
                        spans.append(text)
    canonical = canonical_dimensions(dimensions)
    spans_tuple = tuple(spans)
    payload = repr((sorted(canonical.items()), spans_tuple, tokens, surface_text))
    return PromptFingerprint(
        dimensions=canonical,
        protected_spans=spans_tuple,
        normalized_tokens=tokens,
        fingerprint_hash=short_hash(payload, 16),
        surface_text=surface_text,
    )


def changed_factors(
    parent: PromptFingerprint,
    child: PromptFingerprint,
) -> tuple[str, ...]:
    parent_dimensions = canonical_dimensions(parent.dimensions)
    child_dimensions = canonical_dimensions(child.dimensions)
    changed = {
        key
        for key in set(parent_dimensions) | set(child_dimensions)
        if parent_dimensions.get(key) != child_dimensions.get(key)
    }
    surface_changed = (
        parent.surface_text != child.surface_text
        if parent.surface_text or child.surface_text
        else parent.normalized_tokens != child.normalized_tokens
    )
    if surface_changed and not changed:
        changed.add("style_surface")
    return tuple(sorted(changed))
