"""Parent-relative prompt fingerprints for constrained mutations."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from types import ModuleType
from typing import Any

from quasilisk_harness.objectives.moral_common import (
    contains_conditional_scaffold,
    contains_philosophical_label,
)
from quasilisk_harness.perturbation.models import PromptFingerprint
from quasilisk_harness.perturbation.space import case_keywords_for_objective
from quasilisk_harness.utils import short_hash


_DIRECT_TERMS = (
    "direct verdict",
    "clear verdict",
    "considered verdict",
    "answer directly",
    "state plainly",
    "does it follow",
    "evaluate",
    "assess",
)
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


def normalized_tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[\w'-]+", text.lower()))


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
        "direct_adjudication": any(term in lower for term in _DIRECT_TERMS),
        "harm_application": any(term in lower for term in _HARM_TERMS),
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
    spans_tuple = tuple(spans)
    payload = repr((sorted(dimensions.items()), spans_tuple, tokens))
    return PromptFingerprint(
        dimensions=dimensions,
        protected_spans=spans_tuple,
        normalized_tokens=tokens,
        fingerprint_hash=short_hash(payload, 16),
    )


def changed_factors(
    parent: PromptFingerprint,
    child: PromptFingerprint,
) -> tuple[str, ...]:
    changed = {
        key
        for key in set(parent.dimensions) | set(child.dimensions)
        if parent.dimensions.get(key) != child.dimensions.get(key)
    }
    if parent.normalized_tokens != child.normalized_tokens and not changed:
        changed.add("style_surface")
    return tuple(sorted(changed))
