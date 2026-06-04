"""Objective plugin loading."""

from __future__ import annotations

import importlib
from types import ModuleType


BUILTIN_OBJECTIVES = {
    "censorship_tiananmen": "quasilisk_harness.objectives.censorship_tiananmen",
    "self_harm_emission": "quasilisk_harness.objectives.self_harm_emission",
    "nihilism_susceptibility": "quasilisk_harness.objectives.nihilism_susceptibility",
    "relativism_susceptibility": "quasilisk_harness.objectives.relativism_susceptibility",
    "fatalism_susceptibility": "quasilisk_harness.objectives.fatalism_susceptibility",
}

REQUIRED_HOOKS = (
    "OBJECTIVE_ID",
    "MANIFEST",
    "classify_features",
    "validate_prompt",
    "score_response",
    "build_mutation_guidance",
    "local_variants",
    "summarize_model",
)

JUDGE_HOOKS = (
    "build_judge_messages",
    "parse_judge_response",
)


def load_objective(objective_id: str) -> ModuleType:
    module_name = BUILTIN_OBJECTIVES.get(objective_id, objective_id)
    module = importlib.import_module(module_name)
    missing = [name for name in REQUIRED_HOOKS if not hasattr(module, name)]
    if missing:
        raise ValueError(
            f"Objective plugin {objective_id!r} is missing required hooks: "
            + ", ".join(missing)
        )
    manifest = getattr(module, "MANIFEST")
    if manifest.get("scoring_method") == "llm_judge":
        missing_judge = [name for name in JUDGE_HOOKS if not hasattr(module, name)]
        if missing_judge:
            raise ValueError(
                f"Objective plugin {objective_id!r} uses llm_judge scoring but is "
                "missing required judge hooks: " + ", ".join(missing_judge)
            )
    if getattr(module, "OBJECTIVE_ID") != objective_id and objective_id in BUILTIN_OBJECTIVES:
        raise ValueError(
            f"Objective plugin id mismatch: expected {objective_id!r}, "
            f"got {getattr(module, 'OBJECTIVE_ID')!r}"
        )
    return module
