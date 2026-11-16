"""Command-line boundary for the single Quasilisk engine."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from .config import ConfigError, REPOSITORY_ROOT
from .domain import DomainError, TERMINAL_EXIT_CODES, canonical_json, strict_json_loads
from .engine import (
    EngineCompatibilityError,
    EngineError,
    resume_run,
    run_new,
    validate_run_evidence,
    validate_new_run,
)
from .events import JournalCorrupt, JournalError, JournalUnwritable
from .objective import ObjectiveError, discover_objectives
from .provider import ProviderProtocolError
from .reporting import ReportingError, rebuild_library_indexes


class LibraryInputError(ValueError):
    """An explicit library-rebuild source is not one consolidated run root."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m quasilisk",
        description="Run or replay the consolidated Quasilisk research harness.",
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--config", metavar="PATH", help="start from one strict config")
    modes.add_argument("--resume", metavar="RUN_DIR", help="resume or rebuild one run")
    modes.add_argument(
        "--list-objectives", action="store_true", help="list valid objective plugins"
    )
    modes.add_argument(
        "--rebuild-library-indexes",
        nargs="+",
        metavar="RUN_DIR",
        help="rebuild derived library indexes from explicit run roots",
    )
    parser.add_argument(
        "--seed-prompts", metavar="PATH",
        help="select an external bootstrap JSON file for --config; frozen for resume",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate --config without creating a run",
    )
    return parser


def _rebuild_library_indexes(run_roots: Sequence[str]) -> None:
    """Validate all source evidence, then transactionally replace all indexes."""

    resolved: list[Path] = []
    seen_paths: set[Path] = set()
    for raw in run_roots:
        try:
            path = Path(raw).resolve(strict=True)
        except OSError as exc:
            raise LibraryInputError(f"run root does not exist: {raw}") from exc
        if path in seen_paths:
            raise LibraryInputError(f"duplicate run root: {path}")
        if not path.is_dir() or not (path / "events.jsonl").is_file():
            raise LibraryInputError(
                f"not a consolidated run root containing events.jsonl: {path}"
            )
        seen_paths.add(path)
        resolved.append(path)

    # Complete validation and reducer replay precede even creation/opening of
    # the library lock file. The command is read-only on every bad input.
    states = []
    for path in resolved:
        try:
            states.append(
                validate_run_evidence(path, repository_root=REPOSITORY_ROOT)
            )
        except JournalCorrupt:
            raise
        except DomainError as exc:
            raise JournalCorrupt(f"run reducer rejected {path}: {exc}") from exc
    run_ids = [state.run_id for state in states]
    if len(run_ids) != len(set(run_ids)):
        raise LibraryInputError("the explicit run roots contain a duplicate run ID")
    rebuild_library_indexes(REPOSITORY_ROOT, states)


def _validated_resume_path(raw: str) -> Path:
    try:
        path = Path(raw).resolve(strict=True)
    except OSError as exc:
        raise LibraryInputError(f"resume run root does not exist: {raw}") from exc
    if not path.is_dir() or not (path / "events.jsonl").is_file():
        raise LibraryInputError(
            f"--resume must name a consolidated run root containing events.jsonl: {path}"
        )
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.validate_only and arguments.config is None:
        parser.error("--validate-only requires --config")
    if arguments.seed_prompts is not None and arguments.config is None:
        parser.error("--seed-prompts requires --config and cannot change a resumed run")
    try:
        config_source = arguments.config
        if arguments.seed_prompts is not None:
            try:
                config_source = strict_json_loads(Path(arguments.config).read_bytes())
            except DomainError as exc:
                raise ConfigError(str(exc)) from exc
            if not isinstance(config_source, dict) or not isinstance(config_source.get("seed"), dict):
                raise ConfigError("config must contain a seed object")
            config_source["seed"]["prompts_path"] = Path(arguments.seed_prompts).resolve().as_posix()
        if arguments.list_objectives:
            for objective_id in discover_objectives(REPOSITORY_ROOT / "objectives"):
                print(objective_id)
            return 0
        if arguments.rebuild_library_indexes is not None:
            _rebuild_library_indexes(arguments.rebuild_library_indexes)
            print("rebuilt library indexes")
            return 0
        if arguments.config is not None:
            if arguments.validate_only:
                resources = validate_new_run(config_source)
                from .run_state import funded_work_plan, funded_call_limits
                plan = funded_work_plan(resources.config.to_dict(), resources.bootstrap_payloads, resources.seed_snapshot)
                print(
                    canonical_json(
                        {
                            "config_digest": resources.config.config_digest,
                            "objective": resources.objective.id,
                            "scientific_scope_digest": resources.scientific_scope_digest,
                            "status": "valid",
                            "funded_work": {kind: {"allocations": count, "maximum_calls_per_unit": maximum.to_dict()} for kind, (count, maximum) in plan.items()},
                            "physical_call_limits": funded_call_limits(plan).to_dict(),
                        }
                    )
                )
                return 0
            run_directory, reason = run_new(config_source)
        else:
            run_directory, reason = resume_run(
                _validated_resume_path(arguments.resume)
            )
        print(
            canonical_json(
                {"reason": reason.value, "run_directory": run_directory.as_posix()}
            )
        )
        return TERMINAL_EXIT_CODES[reason]
    except (
        ConfigError,
        ObjectiveError,
        ProviderProtocolError,
        FileNotFoundError,
        LibraryInputError,
    ) as exc:
        print(f"invalid input: {exc}", file=sys.stderr)
        return 2
    except ReportingError as exc:
        print(f"derived_artifact_write_failed: {exc}", file=sys.stderr)
        return 3
    except (JournalCorrupt, JournalUnwritable, JournalError, OSError) as exc:
        print(f"evidence error: {exc}", file=sys.stderr)
        return 3
    except EngineCompatibilityError as exc:
        print(f"resume incompatible: {exc}", file=sys.stderr)
        return 5
    except (EngineError, DomainError) as exc:
        print(f"internal state error: {exc}", file=sys.stderr)
        return 5


__all__ = ["main"]
