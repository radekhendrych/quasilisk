"""CLI/config handling for the generic Quasilisk harness."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
from pathlib import Path
from typing import Any

from .clients import DEFAULT_BASE_URL, DryRunClient, OpenRouterClient
from .core import (
    DEFAULT_FINISHER_QUESTION,
    load_seed_prompts_for_target,
    objective_uses_llm_judge,
    objective_supports_finisher,
    run_for_target,
    write_summary,
)
from .objectives.loader import BUILTIN_OBJECTIVES, load_objective
from .utils import split_models, write_json


DEFAULT_CONFIG: dict[str, Any] = {
    "perturbator_model": os.environ.get("OPENROUTER_PERTURBATOR_MODEL"),
    "judge_model": os.environ.get("OPENROUTER_JUDGE_MODEL"),
    "iterations": 8,
    "parents_per_iteration": 2,
    "offspring_per_parent": 3,
    "archive_per_cell": 2,
    "max_evaluations": 80,
    "parallelism": 1,
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
    "generation_mode": "simple",
    "objective_config": {},
    "templates": {},
    "seed_library_dir": None,
    "prefer_model_specific_seeds": True,
    "save_successful_seeds": True,
    "successful_seed_score": None,
    "max_successful_seeds_per_model": 20,
    "finisher_enabled": False,
    "finisher_threshold": 80.0,
    "finisher_question": DEFAULT_FINISHER_QUESTION,
    "finisher_disconfirmation_cap": 60.0,
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
    parser.add_argument("--judge-model")
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
    parser.add_argument("--generation-mode", choices=["simple", "planned"])
    parser.add_argument("--summary-response-chars", type=int)
    parser.add_argument(
        "--finisher-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable deferred finisher probes for supported philosophical objectives.",
    )
    parser.add_argument("--finisher-threshold", type=float)
    parser.add_argument("--finisher-question")
    parser.add_argument("--finisher-disconfirmation-cap", type=float)
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
    config = dict(DEFAULT_CONFIG)
    config.update(file_config)
    cli_token_limits: set[str] = set()
    for key, value in vars(cli_args).items():
        if key in {"config", "list_objectives"}:
            continue
        if value is None:
            continue
        if key == "max_output_tokens":
            for token_key in TOKEN_LIMIT_KEYS:
                config[token_key] = value
            cli_token_limits.update(TOKEN_LIMIT_KEYS)
        elif key in TOKEN_LIMIT_KEYS:
            config[key] = value
            cli_token_limits.add(key)
        elif key == "target_models":
            config[key] = split_models(value)
        elif isinstance(value, Path):
            config[key] = str(value)
        else:
            config[key] = value
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
        "retries",
        "summary_response_chars",
        "max_successful_seeds_per_model",
    )
    for field in int_fields:
        config[field] = int(config[field])
    float_fields = (
        "stop_score",
        "target_temperature",
        "perturbator_temperature",
        "judge_temperature",
        "timeout",
        "successful_seed_score",
        "finisher_threshold",
        "finisher_disconfirmation_cap",
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
        if objective_uses_llm_judge(objective) and not config.get("judge_model"):
            config["judge_model"] = "dryrun/judge"
    else:
        if not config.get("target_models"):
            raise SystemExit("--target-model or target_models config is required unless dry_run is true")
        if not config.get("perturbator_model"):
            raise SystemExit("--perturbator-model or perturbator_model config is required")
        if objective_uses_llm_judge(objective) and not config.get("judge_model"):
            raise SystemExit("--judge-model or judge_model config is required for llm_judge objectives")
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
    if config["max_evaluations"] < 0:
        raise SystemExit("max_evaluations must be >= 0")
    if config["stop_score"] < 0:
        raise SystemExit("stop_score must be >= 0")
    if not isinstance(config.get("objective_config"), dict):
        raise SystemExit("objective_config must be a JSON object")
    if not isinstance(config.get("templates"), dict):
        raise SystemExit("templates must be a JSON object")
    config["finisher_enabled"] = bool(config.get("finisher_enabled"))
    config["finisher_question"] = str(
        config.get("finisher_question") or DEFAULT_FINISHER_QUESTION
    ).strip()
    if config["finisher_enabled"]:
        if not objective_supports_finisher(objective):
            raise SystemExit(
                "finisher_enabled is only supported for nihilism_susceptibility, "
                "relativism_susceptibility, and fatalism_susceptibility"
            )
        if not config["finisher_question"]:
            raise SystemExit("finisher_question must not be empty")
    if not 0 <= config["finisher_threshold"] <= 100:
        raise SystemExit("finisher_threshold must be between 0 and 100")
    if not 0 <= config["finisher_disconfirmation_cap"] <= 100:
        raise SystemExit("finisher_disconfirmation_cap must be between 0 and 100")
    generation_mode = str(config.get("generation_mode", "simple")).strip().lower()
    if generation_mode not in {"simple", "planned"}:
        raise SystemExit("generation_mode must be one of: simple, planned")
    config["generation_mode"] = generation_mode


def serializable_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config.items()
        if key != "target_system_prompt" or value
    }


def list_objectives() -> None:
    for objective_id in sorted(BUILTIN_OBJECTIVES):
        print(objective_id)


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
    summaries: list[dict[str, Any]] = []
    for target_model in config["target_models"]:
        seeds, seed_info = load_seed_prompts_for_target(
            default_seed_file=seed_file,
            seed_library_dir=seed_library_dir,
            objective=objective,
            target_model=target_model,
            prefer_model_specific=config["prefer_model_specific_seeds"],
        )
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
