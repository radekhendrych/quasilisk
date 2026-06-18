"""Minimal prompt template loading and rendering."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .utils import short_hash


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TEMPLATE_PATHS = {
    "perturbator_planned_system": "templates/perturbator/planned_system.md",
    "perturbator_planned_user": "templates/perturbator/planned_user.md",
    "moral_judge_system": "templates/judge/moral_system.md",
    "moral_judge_user": "templates/judge/moral_user.md",
}

PLACEHOLDER_RE = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}")


def template_path(key: str, template_config: dict[str, Any] | None = None) -> Path:
    if key not in DEFAULT_TEMPLATE_PATHS:
        raise KeyError(f"unknown template key: {key}")
    configured = None
    if isinstance(template_config, dict):
        configured = template_config.get(key)
    raw_path = Path(str(configured or DEFAULT_TEMPLATE_PATHS[key]))
    if raw_path.is_absolute() or raw_path.exists():
        return raw_path
    return PROJECT_ROOT / raw_path


def render_template_text(text: str, variables: dict[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in variables:
            raise KeyError(f"missing template variable: {name}")
        return str(variables[name])

    return PLACEHOLDER_RE.sub(replace, text).strip()


def render_template_file(
    key: str,
    variables: dict[str, Any],
    template_config: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    path = template_path(key, template_config)
    text = path.read_text(encoding="utf-8")
    rendered = render_template_text(text, variables)
    return rendered, {
        "template_key": key,
        "template_path": str(path),
        "template_hash": short_hash(text, 16),
    }
