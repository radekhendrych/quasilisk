"""CLI/config handling for the generic Quasilisk harness."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

from .clients import DEFAULT_BASE_URL, DryRunClient, OpenRouterClient
from .core import (
    load_seed_prompts_for_target,
    objective_uses_llm_judge,
    run_for_target,
    write_summary,
)
from .objectives.loader import BUILTIN_OBJECTIVES, load_objective
from .perturbation.refinement import RefinementPolicy
from .perturbation.scheduler import SchedulingPolicy
from .utils import safe_slug, split_models, write_json, write_status


DEFAULT_CONFIG: dict[str, Any] = {
    "perturbator_model": os.environ.get("OPENROUTER_PERTURBATOR_MODEL"),
    "judge_model": os.environ.get("OPENROUTER_JUDGE_MODEL"),
    "judge_models": None,
    "iterations": 8,
    "parents_per_iteration": 2,
    "offspring_per_parent": 3,
    "archive_per_cell": 2,
    "max_evaluations": 80,
    "parallelism": 1,
    "judge_parallelism": 3,
    "judge_disagreement_threshold": 20.0,
    "target_temperature": 0.2,
    "perturbator_temperature": 0.8,
    "judge_temperature": 0.0,
    "target_max_tokens": None,
    "perturbator_max_tokens": None,
    "judge_max_tokens": None,
    "target_system_prompt": "",
    "base_url": os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL),
    "api_key_env": "OPENROUTER_API_KEY",
    "http_referer": os.environ.get("OPENROUTER_HTTP_REFERER"),
    "app_title": os.environ.get("OPENROUTER_APP_TITLE", "quasilisk-generic-harness"),
    "timeout": 90.0,
    "retries": 3,
    "random_seed": 7,
    "dry_run": False,
    "summary_response_chars": 900,
    "artifact_policy": "complete_audit_unredacted",
    "objective_config": {},
    "templates": {},
    "seed_library_dir": None,
    "prefer_model_specific_seeds": True,
    "save_successful_seeds": True,
    "successful_seed_score": None,
    "max_successful_seeds_per_model": 20,
    "parent_plan_memory_enabled": True,
    "parent_plan_memory_mode": "block",
    "parent_plan_regression_threshold": 8.0,
    "parent_plan_cooldown_generations": 2,
    "parent_plan_penalty": 15.0,
    "parent_plan_min_evaluated_children": 1,
    "coverage_novelty_enabled": True,
    "coverage_novelty_combo_failure_limit": 1,
    "coverage_novelty_route_failure_limit": 2,
    "coverage_novelty_cooldown_generations": 3,
    "coverage_novelty_distinct_routes_per_generation": True,
    "perturbation_trust_total_threshold": 60.0,
    "perturbation_trust_raw_threshold": 80.0,
    "perturbation_trust_min_similarity": 0.8,
    "perturbation_response_repair_min_similarity": 0.7,
    "perturbation_max_component_drop": 5.0,
    "perturbation_replay_control_enabled": False,
    "perturbation_scheduler": SchedulingPolicy().to_jsonable(),
    "perturbation_refinement": RefinementPolicy().to_jsonable(),
    "operator_outcome_memory_enabled": True,
    "operator_block_after_regressions": 1,
    "judge_parse_retries": 1,
    "paired_rejudge_enabled": False,
    "paired_rejudge_drop_threshold": 30.0,
    "fail_fast": False,
    "force_out_dir": False,
    "resume": False,
}

TOKEN_LIMIT_KEYS = {
    "target_max_tokens",
    "perturbator_max_tokens",
    "judge_max_tokens",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a generic MAP-Elites model evaluation harness through OpenRouter."
    )
    parser.add_argument("--config", type=Path, help="JSON run config.")
    parser.add_argument(
        "--list-objectives",
        action="store_true",
        help="List built-in objective ids and exit.",
    )
    parser.add_argument("--objective-id", dest="objective_id")
    parser.add_argument("--objective-text", dest="objective_text")
    parser.add_argument(
        "--target-model",
        dest="target_models",
        action="append",
        help="OpenRouter target model id. May be repeated or comma-separated.",
    )
    parser.add_argument("--perturbator-model")
    parser.add_argument(
        "--judge-model",
        dest="judge_models",
        action="append",
        help="OpenRouter judge model id. May be repeated or comma-separated, up to three.",
    )
    parser.add_argument("--seed-file", type=Path)
    parser.add_argument("--seed-library-dir", type=Path)
    parser.add_argument(
        "--no-model-specific-seeds",
        action="store_false",
        dest="prefer_model_specific_seeds",
        default=None,
    )
    parser.add_argument(
        "--no-save-successful-seeds",
        action="store_false",
        dest="save_successful_seeds",
        default=None,
    )
    parser.add_argument("--successful-seed-score", type=float)
    parser.add_argument("--max-successful-seeds-per-model", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--parents-per-iteration", type=int)
    parser.add_argument("--offspring-per-parent", type=int)
    parser.add_argument("--archive-per-cell", type=int)
    parser.add_argument("--max-evaluations", type=int)
    parser.add_argument("--stop-score", type=float)
    parser.add_argument("--parallelism", type=int)
    parser.add_argument("--judge-parallelism", type=int)
    parser.add_argument("--judge-disagreement-threshold", type=float)
    parser.add_argument("--target-temperature", type=float)
    parser.add_argument("--perturbator-temperature", type=float)
    parser.add_argument("--judge-temperature", type=float)
    parser.add_argument("--target-max-tokens", type=int)
    parser.add_argument("--perturbator-max-tokens", type=int)
    parser.add_argument("--judge-max-tokens", type=int)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        help="Set the same explicit output cap for target, perturbator, and judge calls.",
    )
    parser.add_argument("--target-system-prompt")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env")
    parser.add_argument("--http-referer")
    parser.add_argument("--app-title")
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--retries", type=int)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", default=None)
    parser.add_argument("--summary-response-chars", type=int)
    parser.add_argument(
        "--parent-plan-memory-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable parent-plan outcome memory across MAP-Elites generations.",
    )
    parser.add_argument(
        "--parent-plan-memory-mode",
        choices=("block", "penalize"),
        help="Whether regressed parent-plan tuples are blocked or priority-penalized.",
    )
    parser.add_argument("--parent-plan-regression-threshold", type=float)
    parser.add_argument("--parent-plan-cooldown-generations", type=int)
    parser.add_argument("--parent-plan-penalty", type=float)
    parser.add_argument("--parent-plan-min-evaluated-children", type=int)
    parser.add_argument(
        "--coverage-novelty-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--coverage-novelty-combo-failure-limit", type=int)
    parser.add_argument("--coverage-novelty-route-failure-limit", type=int)
    parser.add_argument("--coverage-novelty-cooldown-generations", type=int)
    parser.add_argument(
        "--coverage-novelty-distinct-routes-per-generation",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--perturbation-trust-total-threshold", type=float)
    parser.add_argument("--perturbation-trust-raw-threshold", type=float)
    parser.add_argument("--perturbation-trust-min-similarity", type=float)
    parser.add_argument(
        "--perturbation-response-repair-min-similarity",
        type=float,
    )
    parser.add_argument("--perturbation-max-component-drop", type=float)
    parser.add_argument(
        "--perturbation-replay-control-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--operator-outcome-memory-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--operator-block-after-regressions", type=int)
    parser.add_argument("--judge-parse-retries", type=int)
    parser.add_argument(
        "--paired-rejudge-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--paired-rejudge-drop-threshold", type=float)
    parser.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Abort the whole run after the first target-level failure.",
    )
    parser.add_argument(
        "--force-out-dir",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow writing into a non-empty output directory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=None,
        help="Skip targets already completed in an existing run_summary.json.",
    )
    return parser


def load_config_file(path: Path | None) -> dict[str, Any]:
    if not path:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise SystemExit(f"Config file must contain a JSON object: {path}")
    return data


def merge_config(file_config: dict[str, Any], cli_args: argparse.Namespace) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update(file_config)
    if "perturbation_architecture_version" in config:
        config.pop("perturbation_architecture_version", None)
        print(
            "Warning: perturbation_architecture_version is ignored; "
            "v2 mutation tasks are always used.",
            file=sys.stderr,
        )
    cli_token_limits: set[str] = set()
    cli_shared_token_limit = False
    for key, value in vars(cli_args).items():
        if key in {"config", "list_objectives"}:
            continue
        if value is None:
            continue
        if key == "max_output_tokens":
            cli_shared_token_limit = True
            for token_key in TOKEN_LIMIT_KEYS:
                config[token_key] = value
            cli_token_limits.update(TOKEN_LIMIT_KEYS)
        elif key in TOKEN_LIMIT_KEYS:
            config[key] = value
            cli_token_limits.add(key)
        elif key in {"target_models", "judge_models"}:
            config[key] = split_models(value)
        elif isinstance(value, Path):
            config[key] = str(value)
        else:
            config[key] = value
    if cli_shared_token_limit:
        for key in TOKEN_LIMIT_KEYS - cli_token_limits:
            config[key] = None
    config["cli_token_limits"] = sorted(cli_token_limits)
    if isinstance(config.get("target_models"), str):
        config["target_models"] = split_models([config["target_models"]])
    elif isinstance(config.get("target_models"), list):
        normalized: list[str] = []
        for value in config["target_models"]:
            normalized.extend(split_models([str(value)]))
        config["target_models"] = normalized
    if getattr(cli_args, "judge_models", None):
        judge_values: Any = config.get("judge_models")
    elif "judge_models" in file_config:
        judge_values = file_config.get("judge_models")
    elif "judge_model" in file_config:
        judge_values = file_config.get("judge_model")
    else:
        judge_values = DEFAULT_CONFIG.get("judge_model")
    if isinstance(judge_values, str):
        config["judge_models"] = split_models([judge_values])
    elif isinstance(judge_values, list):
        normalized_judges: list[str] = []
        for value in judge_values:
            normalized_judges.extend(split_models([str(value)]))
        config["judge_models"] = normalized_judges
    elif judge_values is None:
        config["judge_models"] = []
    else:
        config["judge_models"] = judge_values
    config["judge_model"] = (
        config["judge_models"][0]
        if isinstance(config["judge_models"], list) and len(config["judge_models"]) == 1
        else None
    )
    return config


def _path_from_config(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def validate_config(config: dict[str, Any], objective: Any) -> None:
    manifest = objective.MANIFEST
    config.setdefault("objective_text", getattr(objective, "DEFAULT_OBJECTIVE", manifest["title"]))
    default_seed_file = Path(manifest["default_seed_file"])
    if not config.get("seed_file"):
        config["seed_file"] = str(default_seed_file)
    if not config.get("seed_library_dir"):
        config["seed_library_dir"] = str(default_seed_file.parent)
    if "stop_score" not in config or config["stop_score"] is None:
        config["stop_score"] = manifest.get("default_stop_score", 0.0)
    if "successful_seed_score" not in config or config["successful_seed_score"] is None:
        config["successful_seed_score"] = manifest.get("successful_seed_score", 70.0)
    if not config.get("out_dir"):
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        config["out_dir"] = str(Path("runs") / stamp)

    int_fields = (
        "iterations",
        "parents_per_iteration",
        "offspring_per_parent",
        "archive_per_cell",
        "max_evaluations",
        "parallelism",
        "judge_parallelism",
        "retries",
        "summary_response_chars",
        "max_successful_seeds_per_model",
        "parent_plan_cooldown_generations",
        "parent_plan_min_evaluated_children",
        "coverage_novelty_combo_failure_limit",
        "coverage_novelty_route_failure_limit",
        "coverage_novelty_cooldown_generations",
        "operator_block_after_regressions",
        "judge_parse_retries",
    )
    for field in int_fields:
        config[field] = int(config[field])
    float_fields = (
        "stop_score",
        "target_temperature",
        "perturbator_temperature",
        "judge_temperature",
        "judge_disagreement_threshold",
        "timeout",
        "successful_seed_score",
        "parent_plan_regression_threshold",
        "parent_plan_penalty",
        "perturbation_trust_total_threshold",
        "perturbation_trust_raw_threshold",
        "perturbation_trust_min_similarity",
        "perturbation_response_repair_min_similarity",
        "perturbation_max_component_drop",
        "paired_rejudge_drop_threshold",
    )
    for field in float_fields:
        config[field] = float(config[field])
    for field in TOKEN_LIMIT_KEYS:
        if config.get(field) is not None:
            config[field] = int(config[field])
            if config[field] <= 0:
                raise SystemExit(f"{field} must be > 0 when provided")

    if config["dry_run"]:
        if not config.get("target_models"):
            config["target_models"] = ["dryrun/target"]
        if not config.get("perturbator_model"):
            config["perturbator_model"] = "dryrun/perturbator"
        if objective_uses_llm_judge(objective) and not config.get("judge_models"):
            config["judge_models"] = ["dryrun/judge"]
            config["judge_model"] = "dryrun/judge"
    else:
        if not config.get("target_models"):
            raise SystemExit("--target-model or target_models config is required unless dry_run is true")
        if not config.get("perturbator_model"):
            raise SystemExit("--perturbator-model or perturbator_model config is required")
        if objective_uses_llm_judge(objective) and not config.get("judge_models"):
            raise SystemExit(
                "--judge-model, judge_model, or judge_models config is required "
                "for llm_judge objectives"
            )
        if not os.environ.get(config["api_key_env"]):
            raise SystemExit(f"{config['api_key_env']} must be set unless dry_run is true")

    if config["iterations"] < 0:
        raise SystemExit("iterations must be >= 0")
    if config["parents_per_iteration"] <= 0:
        raise SystemExit("parents_per_iteration must be > 0")
    if config["offspring_per_parent"] <= 0:
        raise SystemExit("offspring_per_parent must be > 0")
    if config["archive_per_cell"] <= 0:
        raise SystemExit("archive_per_cell must be > 0")
    if config["parallelism"] <= 0:
        raise SystemExit("parallelism must be > 0")
    if config["judge_parallelism"] <= 0:
        raise SystemExit("judge_parallelism must be > 0")
    if config["judge_disagreement_threshold"] < 0:
        raise SystemExit("judge_disagreement_threshold must be >= 0")
    if config["max_evaluations"] < 0:
        raise SystemExit("max_evaluations must be >= 0")
    if config["stop_score"] < 0:
        raise SystemExit("stop_score must be >= 0")
    if config["parent_plan_cooldown_generations"] < 0:
        raise SystemExit("parent_plan_cooldown_generations must be >= 0")
    if config["parent_plan_min_evaluated_children"] <= 0:
        raise SystemExit("parent_plan_min_evaluated_children must be > 0")
    if config["parent_plan_regression_threshold"] < 0:
        raise SystemExit("parent_plan_regression_threshold must be >= 0")
    if config["parent_plan_penalty"] < 0:
        raise SystemExit("parent_plan_penalty must be >= 0")
    if config["coverage_novelty_combo_failure_limit"] <= 0:
        raise SystemExit("coverage_novelty_combo_failure_limit must be > 0")
    if config["coverage_novelty_route_failure_limit"] <= 0:
        raise SystemExit("coverage_novelty_route_failure_limit must be > 0")
    if config["coverage_novelty_cooldown_generations"] < 0:
        raise SystemExit("coverage_novelty_cooldown_generations must be >= 0")
    if not 0 <= config["perturbation_trust_min_similarity"] <= 1:
        raise SystemExit("perturbation_trust_min_similarity must be between 0 and 1")
    if not 0 <= config["perturbation_response_repair_min_similarity"] <= 1:
        raise SystemExit(
            "perturbation_response_repair_min_similarity must be between 0 and 1"
        )
    if config["perturbation_max_component_drop"] < 0:
        raise SystemExit("perturbation_max_component_drop must be >= 0")
    if config["operator_block_after_regressions"] <= 0:
        raise SystemExit("operator_block_after_regressions must be > 0")
    if config["judge_parse_retries"] < 0:
        raise SystemExit("judge_parse_retries must be >= 0")
    if config["paired_rejudge_drop_threshold"] < 0:
        raise SystemExit("paired_rejudge_drop_threshold must be >= 0")
    if config.get("parent_plan_memory_mode") not in {"block", "penalize"}:
        raise SystemExit("parent_plan_memory_mode must be 'block' or 'penalize'")
    if not isinstance(config.get("objective_config"), dict):
        raise SystemExit("objective_config must be a JSON object")
    if not isinstance(config.get("templates"), dict):
        raise SystemExit("templates must be a JSON object")
    try:
        scheduling_policy = SchedulingPolicy.from_config(
            config.get("perturbation_scheduler")
        )
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid perturbation_scheduler: {exc}") from exc
    config["perturbation_scheduler"] = scheduling_policy.to_jsonable()
    try:
        refinement_policy = RefinementPolicy.from_config(
            config.get("perturbation_refinement")
        )
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid perturbation_refinement: {exc}") from exc
    config["perturbation_refinement"] = refinement_policy.to_jsonable()
    judge_models = config.get("judge_models")
    if not isinstance(judge_models, list) or not all(
        isinstance(model, str) and model.strip() for model in judge_models
    ):
        raise SystemExit("judge_models must be a JSON list of model ids")
    if len(judge_models) > 3:
        raise SystemExit("judge_models supports at most 3 models")
    if len(set(judge_models)) != len(judge_models):
        raise SystemExit("judge_models must not contain duplicate model ids")
    overlap = sorted(set(judge_models) & set(config.get("target_models") or []))
    if overlap:
        print(
            "Warning: target model is also configured as a judge: "
            + ", ".join(overlap),
            file=sys.stderr,
        )
    if len(judge_models) > 1 and not callable(
        getattr(objective, "aggregate_judge_scores", None)
    ):
        raise SystemExit(
            f"Objective {objective.OBJECTIVE_ID!r} does not support multiple judges"
        )
    config["parent_plan_memory_enabled"] = bool(config.get("parent_plan_memory_enabled"))
    config["coverage_novelty_enabled"] = bool(
        config.get("coverage_novelty_enabled")
    )
    config["coverage_novelty_distinct_routes_per_generation"] = bool(
        config.get("coverage_novelty_distinct_routes_per_generation")
    )
    config["operator_outcome_memory_enabled"] = bool(
        config.get("operator_outcome_memory_enabled")
    )
    config["perturbation_replay_control_enabled"] = bool(
        config.get("perturbation_replay_control_enabled")
    )
    config["paired_rejudge_enabled"] = bool(config.get("paired_rejudge_enabled"))
    config["fail_fast"] = bool(config.get("fail_fast"))
    config["force_out_dir"] = bool(config.get("force_out_dir"))
    config["resume"] = bool(config.get("resume"))

def serializable_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config.items()
        if key != "target_system_prompt" or value
    }


def list_objectives() -> None:
    for objective_id in sorted(BUILTIN_OBJECTIVES):
        print(objective_id)


RUN_ARTIFACT_NAMES = {
    "events.jsonl",
    "search_stats.jsonl",
    "run_statistics.json",
    "run_summary.json",
    "status.json",
    "summary.md",
    "operator_outcomes.jsonl",
    "coverage_outcomes.jsonl",
}


def _has_run_artifacts(out_dir: Path) -> bool:
    return out_dir.exists() and any((out_dir / name).exists() for name in RUN_ARTIFACT_NAMES)


def _load_existing_summaries(out_dir: Path) -> list[dict[str, Any]]:
    summary_path = out_dir / "run_summary.json"
    if not summary_path.exists():
        return []
    with summary_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise SystemExit(f"Existing run_summary.json must contain a JSON list: {summary_path}")
    return [item for item in data if isinstance(item, dict)]


def _completed_targets(summaries: list[dict[str, Any]]) -> set[str]:
    completed: set[str] = set()
    for summary in summaries:
        metrics = summary.get("execution_metrics") or {}
        if metrics.get("completion_state", "completed") == "completed":
            completed.add(str(summary.get("target_model", "")))
    return completed


def _failed_target_summary(
    *,
    objective_id: str,
    target_model: str,
    seed_info: dict[str, Any],
    out_dir: Path,
    error: Exception,
) -> dict[str, Any]:
    error_text = f"{type(error).__name__}: {error}"
    return {
        "objective_id": objective_id,
        "target_model": target_model,
        "evaluations": 0,
        "total_target_calls": 0,
        "archive_cells": 0,
        "best_score": 0.0,
        "seed_info": seed_info,
        "saved_seed_info": None,
        "archive_path": str(out_dir / f"archive_{safe_slug(target_model)}.json"),
        "affordances_path": str(out_dir / f"affordances_{safe_slug(target_model)}.json"),
        "run_statistics_path": str(out_dir / "run_statistics.json"),
        "search_metrics": {},
        "search_top_parent_cells": [],
        "search_bottom_parent_cells": [],
        "execution_metrics": {
            "completion_state": "failed",
            "target_error_count": 0,
            "candidate_exception_count": 0,
            "judge_fallback_count": 0,
            "judge_call_count": 0,
            "judge_member_failure_count": 0,
            "judge_quorum_fallback_count": 0,
            "judge_degraded_quorum_count": 0,
            "judge_disagreement_count": 0,
            "judge_prompt_tokens": 0,
            "judge_completion_tokens": 0,
            "judge_total_tokens": 0,
            "judge_cost": 0.0,
            "perturbator_fallback_count": 0,
            "target_retry_count": 0,
            "judge_retry_count": 0,
            "perturbator_retry_count": 0,
            "seed_candidate_count": int(seed_info.get("seed_count", 0) or 0),
            "seed_candidates_truncated": 0,
        },
        "objective_metrics": {"execution_error": error_text},
        "top": [],
        "error": error_text,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.list_objectives:
        list_objectives()
        return 0

    file_config = load_config_file(args.config)
    config = merge_config(file_config, args)
    objective_id = config.get("objective_id")
    if not objective_id:
        raise SystemExit("--objective-id or objective_id config is required")

    objective = load_objective(str(objective_id))
    validate_config(config, objective)

    seed_file = _path_from_config(config["seed_file"])
    seed_library_dir = _path_from_config(config["seed_library_dir"])
    out_dir = _path_from_config(config["out_dir"])
    existing_summaries: list[dict[str, Any]] = []
    if _has_run_artifacts(out_dir):
        if config["resume"]:
            existing_summaries = _load_existing_summaries(out_dir)
            if not existing_summaries:
                raise SystemExit(
                    f"Cannot resume {out_dir}: no completed run_summary.json found"
                )
        elif not config["force_out_dir"]:
            raise SystemExit(
                f"Output directory already contains run artifacts: {out_dir}. "
                "Use --resume to skip completed targets or --force-out-dir to append explicitly."
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "config.json", serializable_config(config))

    if config["dry_run"]:
        client: OpenRouterClient | DryRunClient = DryRunClient()
    else:
        api_key = os.environ[config["api_key_env"]]
        client = OpenRouterClient(
            api_key=api_key,
            base_url=config["base_url"],
            http_referer=config.get("http_referer"),
            app_title=config.get("app_title"),
            timeout=config["timeout"],
            retries=config["retries"],
        )

    rng = random.Random(config["random_seed"])
    summaries: list[dict[str, Any]] = list(existing_summaries)
    completed_targets = _completed_targets(summaries) if config["resume"] else set()
    for target_model in config["target_models"]:
        if target_model in completed_targets:
            continue
        seeds, seed_info = load_seed_prompts_for_target(
            default_seed_file=seed_file,
            seed_library_dir=seed_library_dir,
            objective=objective,
            target_model=target_model,
            prefer_model_specific=config["prefer_model_specific_seeds"],
        )
        try:
            summary = run_for_target(
                client=client,
                objective=objective,
                target_model=target_model,
                perturbator_model=config["perturbator_model"],
                seeds=seeds,
                seed_info=seed_info,
                out_dir=out_dir,
                config=config,
                rng=rng,
            )
        except Exception as exc:
            write_status(
                out_dir,
                {
                    "phase": "failed_target",
                    "objective_id": objective.OBJECTIVE_ID,
                    "target_model": target_model,
                    "perturbator_model": config["perturbator_model"],
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            summary = _failed_target_summary(
                objective_id=objective.OBJECTIVE_ID,
                target_model=target_model,
                seed_info=seed_info,
                out_dir=out_dir,
                error=exc,
            )
            if config["fail_fast"]:
                summaries.append(summary)
                write_json(out_dir / "run_summary.json", summaries)
                write_summary(out_dir, summaries, objective, config)
                raise
        summaries.append(summary)
        write_json(out_dir / "run_summary.json", summaries)
        write_summary(out_dir, summaries, objective, config)

    write_json(out_dir / "run_summary.json", summaries)
    write_summary(out_dir, summaries, objective, config)
    print(f"Wrote run artifacts to {out_dir}")
    for summary in summaries:
        print(
            f"{summary['target_model']}: best_score={summary['best_score']} "
            f"evaluations={summary['evaluations']} cells={summary['archive_cells']}"
        )
    return 0
