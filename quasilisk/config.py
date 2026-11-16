"""The single closed Quasilisk configuration contract."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
import os
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.parse import urlparse

from .domain import (
    DomainError,
    ModelProfile,
    RoleCallVector,
    canonical_json_bytes,
    digest_json,
    normalize_decimal,
    strict_json_loads,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ConfigError(DomainError):
    """A configuration failed the pre-run closed-contract validation."""


def _expect_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be an object")
    return value


def _closed(value: Mapping[str, Any], required: set[str], label: str) -> None:
    missing = sorted(required - set(value))
    extra = sorted(set(value) - required)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unknown " + ", ".join(extra))
        raise ConfigError(f"{label}: {'; '.join(details)}")


def _positive_int(value: Any, label: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{label} must be an integer >= {minimum}")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"{label} must be a nonnegative integer")
    return value


def _text(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ConfigError(f"{label} must be {'a string' if allow_empty else 'a nonempty string'}")
    return value


def _temperature(
    value: Any,
    label: str,
    *,
    allow_normalized_string: bool,
) -> str:
    allowed_types = (Decimal, int, str) if allow_normalized_string else (Decimal, int)
    if isinstance(value, bool) or not isinstance(value, allowed_types):
        raise ConfigError(f"{label} must be an exact JSON number")
    try:
        result = normalize_decimal(value)
        decimal = Decimal(result)
    except DomainError as exc:
        raise ConfigError(f"{label}: {exc}") from exc
    if not Decimal("0") <= decimal <= Decimal("2"):
        raise ConfigError(f"{label} must be between 0 and 2")
    return result


def _normalize_path(value: Any, label: str, repository_root: Path) -> Path:
    raw = _text(value, label)
    path = Path(raw)
    if not path.is_absolute():
        path = repository_root / path
    return path.resolve(strict=False)


def _relative_or_absolute(path: Path, repository_root: Path) -> str:
    try:
        return path.relative_to(repository_root).as_posix()
    except ValueError:
        return path.as_posix()


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    adapter: str
    timeout_seconds: int
    base_url: str | None = None
    api_key_env: str | None = None
    script_path: Path | None = None
    max_cost_usd: str | None = None

    def to_dict(self, repository_root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
        if self.adapter == "scripted":
            assert self.script_path is not None
            return {
                "adapter": "scripted",
                "script_path": _relative_or_absolute(self.script_path, repository_root),
                **({"max_cost_usd": self.max_cost_usd} if self.max_cost_usd is not None else {}),
                "timeout_seconds": self.timeout_seconds,
            }
        return {
            "adapter": "openai_compatible",
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "timeout_seconds": self.timeout_seconds,
            **({"max_cost_usd": self.max_cost_usd} if self.max_cost_usd is not None else {}),
        }


@dataclass(frozen=True, slots=True)
class JudgeProfiles:
    primary: ModelProfile
    secondary_a: ModelProfile
    secondary_b: ModelProfile | None
    replacement: ModelProfile | None

    @property
    def all_profiles(self) -> tuple[ModelProfile, ...]:
        return tuple(profile for profile in (self.primary, self.secondary_a, self.secondary_b, self.replacement) if profile is not None)

    @property
    def normal_slots(self) -> tuple[str, ...]:
        return ("primary", "secondary_a") if self.secondary_b is None else ("primary", "secondary_a", "secondary_b")

    @property
    def maximum_physical_attempts(self) -> int:
        return 2 * len(self.all_profiles)

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary": self.primary.to_dict(include_system_prompt=False),
            "secondaries": [
                self.secondary_a.to_dict(include_system_prompt=False),
                *([self.secondary_b.to_dict(include_system_prompt=False)] if self.secondary_b is not None else []),
            ],
            "replacement": self.replacement.to_dict(include_system_prompt=False) if self.replacement is not None else None,
        }


@dataclass(frozen=True, slots=True)
class FallbackProfile:
    model: str
    fallback_model: str
    temperature: str | None
    max_output_tokens: int
    reasoning_effort: str | None = None
    provider_only: str | None = None
    max_price: Mapping[str, str] | None = None
    max_context_tokens: int | None = None

    def __post_init__(self) -> None:
        profile = self._profile(self.model)
        object.__setattr__(self, "max_price", profile.max_price)

    def _profile(self, model: str) -> ModelProfile:
        return ModelProfile(model, self.temperature, self.max_output_tokens,
            reasoning_effort=self.reasoning_effort, provider_only=self.provider_only,
            max_price=self.max_price, max_context_tokens=self.max_context_tokens)

    def primary(self) -> ModelProfile:
        return self._profile(self.model)

    def fallback(self) -> ModelProfile:
        return self._profile(self.fallback_model)

    def to_dict(self) -> dict[str, Any]:
        return {**self.primary().to_dict(include_system_prompt=False), "fallback_model": self.fallback_model}


@dataclass(frozen=True, slots=True)
class SampleConfig:
    screening: int
    verification: int
    strategy_qualification: int
    holdout: int

    def to_dict(self) -> dict[str, int]:
        return {
            "screening": self.screening,
            "verification": self.verification,
            "strategy_qualification": self.strategy_qualification,
            "holdout": self.holdout,
        }


@dataclass(frozen=True, slots=True)
class SearchConfig:
    archive_capacity_per_cell: int
    proposals_per_step: int
    maximum_followups: int = 3
    maximum_stalled_followups: int = 2

    def to_dict(self) -> dict[str, int]:
        return {
            "archive_capacity_per_cell": self.archive_capacity_per_cell,
            "proposals_per_step": self.proposals_per_step,
            "maximum_followups": self.maximum_followups,
            "maximum_stalled_followups": self.maximum_stalled_followups,
        }


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    exploration_batches: int
    conversation_rounds: int
    verification_blocks: int
    strategy_experiments: int

    def to_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class RunConfig:
    provider: ProviderConfig
    objective: str
    target: ModelProfile
    judges: JudgeProfiles
    selector: FallbackProfile
    perturbator: FallbackProfile
    seed_library_directory: Path
    bootstrap_limit: int
    experiment: ExperimentPlan
    samples: SampleConfig
    search: SearchConfig
    random_seed: int
    output_directory: Path
    repository_root: Path = REPOSITORY_ROOT
    seed_prompts_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider.to_dict(self.repository_root),
            "objective": self.objective,
            "models": {
                "target": self.target.to_dict(),
                "judges": self.judges.to_dict(),
                "selector": self.selector.to_dict(),
                "perturbator": self.perturbator.to_dict(),
            },
            "seed": {
                "library_directory": _relative_or_absolute(
                    self.seed_library_directory, self.repository_root
                ),
                "bootstrap_limit": self.bootstrap_limit,
                "prompts_path": None if self.seed_prompts_path is None else _relative_or_absolute(self.seed_prompts_path, self.repository_root),
            },
            "experiment": self.experiment.to_dict(),
            "samples": self.samples.to_dict(),
            "search": self.search.to_dict(),
            "random_seed": self.random_seed,
            "output_directory": _relative_or_absolute(
                self.output_directory, self.repository_root
            ),
        }

    @property
    def config_digest(self) -> str:
        # The output base is runtime placement, not experiment identity. Keep
        # it in ``normalized_config`` so resume recovers the invocation, but
        # exclude it from the documented configuration digest.
        value = self.to_dict()
        value.pop("output_directory")
        return digest_json(value)

    def scientific_view(self) -> dict[str, Any]:
        """Return experiment-defining config, excluding transport/runtime facts."""

        value = self.to_dict()
        return {
            "objective": value["objective"],
            "models": value["models"],
            # Selected seed/bootstrap payloads are frozen separately by the
            # engine. The library path is a source location; the limit remains
            # a scientific selection policy.
            "seed": {"bootstrap_limit": value["seed"]["bootstrap_limit"]},
            "experiment": value["experiment"],
            "samples": value["samples"],
            "search": value["search"],
            "random_seed": value["random_seed"],
            **({"max_cost_usd": self.provider.max_cost_usd} if self.provider.max_cost_usd is not None else {}),
        }


def _parse_model_profile(
    value: Any,
    label: str,
    *,
    target: bool,
    frozen_resume: bool,
) -> ModelProfile:
    obj = _expect_object(value, label)
    fields = {"model", "temperature", "max_output_tokens"}
    if target:
        fields.add("system_prompt")
    fields |= set(obj) & {"reasoning_effort", "provider_only", "max_price", "max_context_tokens"}
    _closed(obj, fields, label)
    prices = obj.get("max_price")
    if prices is not None:
        prices = _expect_object(prices, f"{label}.max_price")
        _closed(prices, {"prompt", "completion", "request"}, f"{label}.max_price")
        normalized_prices = {}
        for key, value in prices.items():
            if isinstance(value, bool) or not isinstance(value, (int, Decimal, str) if frozen_resume else (int, Decimal)):
                raise ConfigError(f"{label}.max_price.{key} must be a nonnegative exact number")
            normalized_prices[key] = normalize_decimal(value)
            if Decimal(normalized_prices[key]) < 0:
                raise ConfigError(f"{label}.max_price.{key} must be nonnegative")
        prices = normalized_prices
    return ModelProfile(
        model=_text(obj["model"], f"{label}.model"),
        temperature=None if obj["temperature"] is None else _temperature(
            obj["temperature"],
            f"{label}.temperature",
            allow_normalized_string=frozen_resume,
        ),
        max_output_tokens=_positive_int(
            obj["max_output_tokens"], f"{label}.max_output_tokens"
        ),
        system_prompt=_text(
            obj.get("system_prompt", ""), f"{label}.system_prompt", allow_empty=True
        ),
        reasoning_effort=obj.get("reasoning_effort"),
        provider_only=obj.get("provider_only"),
        max_price=prices,
        max_context_tokens=None if obj.get("max_context_tokens") is None else _positive_int(obj["max_context_tokens"], f"{label}.max_context_tokens"),
    )


def _parse_fallback_profile(
    value: Any,
    label: str,
    *,
    frozen_resume: bool,
) -> FallbackProfile:
    obj = _expect_object(value, label)
    fallback = _text(obj.get("fallback_model"), f"{label}.fallback_model")
    profile = _parse_model_profile({key: value for key, value in obj.items() if key != "fallback_model"},
                                  label, target=False, frozen_resume=frozen_resume)
    if profile.model == fallback:
        raise ConfigError(f"{label} fallback_model must differ from model")
    return FallbackProfile(fallback_model=fallback, **profile.to_dict(include_system_prompt=False))


def _parse_provider(value: Any, repository_root: Path) -> ProviderConfig:
    obj = _expect_object(value, "provider")
    adapter = obj.get("adapter")
    max_cost = obj.get("max_cost_usd")
    if max_cost is not None:
        max_cost = normalize_decimal(_text(max_cost, "provider.max_cost_usd"))
        if Decimal(max_cost) <= 0:
            raise ConfigError("provider.max_cost_usd must be positive")
    if adapter == "scripted":
        _closed(obj, {"adapter", "script_path", "timeout_seconds"} | (set(obj) & {"max_cost_usd"}), "provider")
        script = _normalize_path(obj["script_path"], "provider.script_path", repository_root)
        if not script.is_file() or not os.access(script, os.R_OK):
            raise ConfigError(f"scripted fixture is missing or unreadable: {script}")
        return ProviderConfig(
            adapter="scripted",
            script_path=script,
            max_cost_usd=max_cost,
            timeout_seconds=_positive_int(
                obj["timeout_seconds"], "provider.timeout_seconds"
            ),
        )
    if adapter == "openai_compatible":
        _closed(
            obj,
            {"adapter", "base_url", "api_key_env", "timeout_seconds"} | (set(obj) & {"max_cost_usd"}),
            "provider",
        )
        base_url = _text(obj["base_url"], "provider.base_url")
        parsed = urlparse(base_url)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ConfigError("provider.base_url must be a credential-free HTTPS URL")
        api_key_env = _text(obj["api_key_env"], "provider.api_key_env")
        if ENV_NAME_RE.fullmatch(api_key_env) is None:
            raise ConfigError("provider.api_key_env is not a valid environment-variable name")
        return ProviderConfig(
            adapter="openai_compatible",
            base_url=base_url.rstrip("/"),
            max_cost_usd=max_cost,
            api_key_env=api_key_env,
            timeout_seconds=_positive_int(
                obj["timeout_seconds"], "provider.timeout_seconds"
            ),
        )
    raise ConfigError("provider.adapter must be openai_compatible or scripted")


def _validate_timeout(provider: ProviderConfig) -> None:
    if not 1 <= provider.timeout_seconds <= 300:
        raise ConfigError("provider.timeout_seconds must be in 1..300")


def _validate_output_path(path: Path, repository_root: Path) -> None:
    forbidden = (
        repository_root / "quasilisk",
        repository_root / "objectives",
        repository_root / "templates",
        repository_root / "library",
        repository_root / "schemas",
        repository_root / "configs",
        repository_root / "fixtures",
        repository_root / "tests",
        repository_root / "docs",
    )
    for root in forbidden:
        try:
            path.relative_to(root.resolve(strict=False))
        except ValueError:
            continue
        raise ConfigError(f"output_directory is inside protected source data: {root}")
    if path.exists() and not path.is_dir():
        raise ConfigError("output_directory exists and is not a directory")
    probe = path if path.exists() else path.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if not probe.is_dir() or not os.access(probe, os.W_OK | os.X_OK):
        raise ConfigError("output_directory cannot be created or written")


def load_config(
    path: str | os.PathLike[str] | Mapping[str, Any],
    *,
    repository_root: Path = REPOSITORY_ROOT,
    available_objectives: set[str] | None = None,
    has_compatible_seed: bool = False,
    frozen_resume: bool = False,
) -> RunConfig:
    """Load and fully validate one strict JSON config without creating a run."""

    if frozen_resume and not isinstance(path, Mapping):
        raise ConfigError(
            "frozen_resume requires the normalized mapping frozen in run_created"
        )
    if isinstance(path, Mapping):
        # Resume supplies the normalized, frozen mapping from run_created.
        # Deep-copy first so callers cannot mutate the object while it is
        # validated.  Canonical encoding is still used as the closed-value
        # check, but cannot be the copying mechanism because it intentionally
        # renders exact ``Decimal`` values as normalized manifest strings.
        try:
            raw = deepcopy(dict(path))
            canonical_json_bytes(raw)
        except DomainError as exc:
            raise ConfigError(f"config is not strict finite JSON: {exc}") from exc
    else:
        config_path = Path(path)
        if not config_path.is_absolute():
            config_path = (Path.cwd() / config_path).resolve()
        try:
            raw = strict_json_loads(config_path.read_bytes())
        except OSError as exc:
            raise ConfigError(f"cannot read config: {config_path}: {exc}") from exc
        except DomainError as exc:
            raise ConfigError(f"config is not strict finite JSON: {exc}") from exc
    root = _expect_object(raw, "config")
    _closed(
        root,
        {
            "provider",
            "objective",
            "models",
            "seed",
            "experiment",
            "samples",
            "search",
            "random_seed",
            "output_directory",
        },
        "config",
    )

    provider = _parse_provider(root["provider"], repository_root)
    _validate_timeout(provider)
    objective = _text(root["objective"], "objective")
    if available_objectives is not None and objective not in available_objectives:
        raise ConfigError(f"unknown objective: {objective}")

    models = _expect_object(root["models"], "models")
    _closed(models, {"target", "judges", "selector", "perturbator"}, "models")
    target = _parse_model_profile(
        models["target"],
        "models.target",
        target=True,
        frozen_resume=frozen_resume,
    )
    judges_obj = _expect_object(models["judges"], "models.judges")
    _closed(
        judges_obj,
        {"primary", "secondaries", "replacement"},
        "models.judges",
    )
    secondaries = judges_obj["secondaries"]
    if not isinstance(secondaries, list) or len(secondaries) not in {1, 2}:
        raise ConfigError("models.judges.secondaries must contain one or two profiles")
    if (len(secondaries) == 1) != (judges_obj["replacement"] is None):
        raise ConfigError("two-judge panels require a null replacement; three-judge panels require a distinct replacement")
    judges = JudgeProfiles(
        primary=_parse_model_profile(
            judges_obj["primary"],
            "models.judges.primary",
            target=False,
            frozen_resume=frozen_resume,
        ),
        secondary_a=_parse_model_profile(
            secondaries[0],
            "models.judges.secondaries[0]",
            target=False,
            frozen_resume=frozen_resume,
        ),
        secondary_b=None if len(secondaries) == 1 else _parse_model_profile(
            secondaries[1],
            "models.judges.secondaries[1]",
            target=False,
            frozen_resume=frozen_resume,
        ),
        replacement=None if judges_obj["replacement"] is None else _parse_model_profile(
            judges_obj["replacement"],
            "models.judges.replacement",
            target=False,
            frozen_resume=frozen_resume,
        ),
    )
    judge_ids = [profile.model for profile in judges.all_profiles]
    if len(set(judge_ids)) != len(judge_ids):
        raise ConfigError("all configured judge model IDs must be distinct")
    selector = _parse_fallback_profile(
        models["selector"], "models.selector", frozen_resume=frozen_resume
    )
    perturbator = _parse_fallback_profile(
        models["perturbator"], "models.perturbator", frozen_resume=frozen_resume
    )
    if provider.max_cost_usd is not None:
        for profile in (target, *judges.all_profiles, selector.primary(), selector.fallback(), perturbator.primary(), perturbator.fallback()):
            if profile.max_price is None or profile.max_context_tokens is None:
                raise ConfigError("a dollar cap requires frozen maximum prices and context limits for every model route")
    non_target_models = set(judge_ids) | {
        selector.model,
        selector.fallback_model,
        perturbator.model,
        perturbator.fallback_model,
    }
    if target.model in non_target_models:
        raise ConfigError("the target model cannot serve another configured role")

    seed = _expect_object(root["seed"], "seed")
    _closed(seed, {"library_directory", "bootstrap_limit", "prompts_path"}, "seed")
    seed_prompts_path = None if seed["prompts_path"] is None else _normalize_path(seed["prompts_path"], "seed.prompts_path", repository_root)
    if not frozen_resume and seed_prompts_path is not None and not seed_prompts_path.is_file():
        raise ConfigError("seed.prompts_path is missing or is not a file")
    seed_directory = _normalize_path(
        seed["library_directory"], "seed.library_directory", repository_root
    )
    if not frozen_resume and (
        not seed_directory.is_dir() or not os.access(seed_directory, os.R_OK)
    ):
        raise ConfigError("seed.library_directory is missing or unreadable")
    bootstrap_limit = _nonnegative_int(seed["bootstrap_limit"], "seed.bootstrap_limit")
    if bootstrap_limit == 0 and not has_compatible_seed:
        raise ConfigError("no compatible seed and bootstrap_limit=0 leaves no initial input")

    plan_obj = _expect_object(root["experiment"], "experiment")
    _closed(plan_obj, set(ExperimentPlan.__dataclass_fields__), "experiment")
    experiment = ExperimentPlan(**{
        name: _nonnegative_int(value, "experiment." + name)
        for name, value in plan_obj.items()
    })

    sample_obj = _expect_object(root["samples"], "samples")
    _closed(
        sample_obj,
        {"screening", "verification", "strategy_qualification", "holdout"},
        "samples",
    )
    screening = _positive_int(sample_obj["screening"], "samples.screening")
    if screening != 3:
        raise ConfigError("samples.screening must equal 3")
    samples = SampleConfig(
        screening=screening,
        verification=_positive_int(
            sample_obj["verification"], "samples.verification", minimum=5
        ),
        strategy_qualification=_positive_int(
            sample_obj["strategy_qualification"],
            "samples.strategy_qualification",
            minimum=5,
        ),
        holdout=_positive_int(sample_obj["holdout"], "samples.holdout", minimum=5),
    )

    search_obj = _expect_object(root["search"], "search")
    _closed(
        search_obj,
        {"archive_capacity_per_cell", "proposals_per_step", "maximum_followups", "maximum_stalled_followups"},
        "search",
    )
    search = SearchConfig(
        archive_capacity_per_cell=_positive_int(
            search_obj["archive_capacity_per_cell"],
            "search.archive_capacity_per_cell",
        ),
        proposals_per_step=_positive_int(
            search_obj["proposals_per_step"], "search.proposals_per_step", minimum=3
        ),
        maximum_followups=_positive_int(search_obj["maximum_followups"], "search.maximum_followups"),
        maximum_stalled_followups=_positive_int(search_obj["maximum_stalled_followups"], "search.maximum_stalled_followups"),
    )

    random_seed = root["random_seed"]
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise ConfigError("random_seed must be an integer")
    output_directory = _normalize_path(
        root["output_directory"], "output_directory", repository_root
    )
    # A resumed journal consumes its sequence-one snapshots and writes only to
    # the explicit ``--resume DIR``.  Historical source/output paths remain
    # frozen evidence, but their continued existence or writability is not a
    # compatibility requirement.
    if not frozen_resume:
        _validate_output_path(output_directory, repository_root)

    return RunConfig(
        provider=provider,
        objective=objective,
        target=target,
        judges=judges,
        selector=selector,
        perturbator=perturbator,
        seed_library_directory=seed_directory,
        bootstrap_limit=bootstrap_limit,
        experiment=experiment,
        samples=samples,
        search=search,
        random_seed=random_seed,
        output_directory=output_directory,
        repository_root=repository_root,
        seed_prompts_path=seed_prompts_path,
    )
