from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest

from quasilisk.config import ConfigError, load_config
from quasilisk.domain import DomainError, digest_json


class StrictConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "seed-library").mkdir()
        (self.root / "fixture.jsonl").write_bytes(b"")
        self.config = {
            "provider": {
                "adapter": "scripted",
                "script_path": "fixture.jsonl",
                "timeout_seconds": 1,
            },
            "objective": "synthetic_objective",
            "models": {
                "target": {
                    "model": "models/target",
                    "temperature": Decimal("0.20"),
                    "max_output_tokens": 64,
                    "system_prompt": "",
                },
                "judges": {
                    "primary": {
                        "model": "models/judge-primary",
                        "temperature": 0,
                        "max_output_tokens": 32,
                    },
                    "secondaries": [
                        {
                            "model": "models/judge-secondary-a",
                            "temperature": 0,
                            "max_output_tokens": 32,
                        },
                        {
                            "model": "models/judge-secondary-b",
                            "temperature": 0,
                            "max_output_tokens": 32,
                        },
                    ],
                    "replacement": {
                        "model": "models/judge-replacement",
                        "temperature": 0,
                        "max_output_tokens": 32,
                    },
                },
                "selector": {
                    "model": "models/selector",
                    "fallback_model": "models/selector-fallback",
                    "temperature": Decimal("0.2"),
                    "max_output_tokens": 64,
                },
                "perturbator": {
                    "model": "models/perturbator",
                    "fallback_model": "models/perturbator-fallback",
                    "temperature": Decimal("0.7"),
                    "max_output_tokens": 64,
                },
            },
            "seed": {"library_directory": "seed-library", "bootstrap_limit": 1, "prompts_path": None},
            "experiment": {"exploration_batches": 1, "conversation_rounds": 3, "verification_blocks": 1, "strategy_experiments": 0},
            "samples": {
                "screening": 3,
                "verification": 5,
                "strategy_qualification": 5,
                "holdout": 5,
            },
            "search": {"archive_capacity_per_cell": 1, "proposals_per_step": 3, "maximum_followups": 3, "maximum_stalled_followups": 2},
            "random_seed": 7,
            "output_directory": "runs",
        }

    def load(self, value: dict | Path | str, **kwargs: object):
        return load_config(
            value,
            repository_root=self.root,
            available_objectives={"synthetic_objective"},
            **kwargs,
        )

    def test_valid_config_is_normalized_without_creating_output(self) -> None:
        output_path = self.root / "runs"
        self.assertFalse(output_path.exists())
        loaded = self.load(self.config)
        self.assertEqual(loaded.target.temperature, "0.2")
        self.assertEqual(loaded.seed_library_directory, self.root / "seed-library")
        self.assertEqual(loaded.provider.script_path, self.root / "fixture.jsonl")
        self.assertEqual(loaded.output_directory, output_path)
        self.assertFalse(output_path.exists())
        self.assertEqual(loaded.to_dict()["models"]["target"]["temperature"], "0.2")

    def test_source_temperature_must_be_a_json_number(self) -> None:
        value = deepcopy(self.config)
        value["models"]["target"]["temperature"] = "0.2"
        with self.assertRaises(ConfigError):
            self.load(value)

        # Normalized decimal strings are an internal manifest representation
        # and remain accepted only at the explicit frozen-resume boundary.
        normalized = self.load(self.config).to_dict()
        resumed = self.load(
            normalized,
            has_compatible_seed=True,
            frozen_resume=True,
        )
        self.assertEqual(resumed.target.temperature, "0.2")

    def test_unknown_fields_are_rejected_at_top_level_and_nested_boundaries(self) -> None:
        variants = []
        top = deepcopy(self.config)
        top["feature_flag"] = True
        variants.append(top)
        nested = deepcopy(self.config)
        nested["models"]["target"]["unknown"] = "value"
        variants.append(nested)
        provider = deepcopy(self.config)
        provider["provider"]["base_url"] = "https://not-used.invalid"
        variants.append(provider)
        for value in variants:
            with self.subTest(keys=value.keys()), self.assertRaises(ConfigError):
                self.load(value)

    def test_duplicate_json_keys_and_nonfinite_values_are_rejected(self) -> None:
        serialized = json.dumps(
            self.config,
            default=lambda value: float(value) if isinstance(value, Decimal) else value,
        )
        needle = '"objective": "synthetic_objective"'
        duplicate = serialized.replace(needle, needle + ", " + needle, 1)
        path = self.root / "duplicate.json"
        path.write_text(duplicate, encoding="utf-8")
        with self.assertRaises(DomainError):
            self.load(path)

        nonfinite = deepcopy(self.config)
        nonfinite["models"]["target"]["temperature"] = float("nan")
        with self.assertRaises(DomainError):
            self.load(nonfinite)

    def test_model_role_identity_boundaries_are_strict(self) -> None:
        variants = []
        duplicate_judges = deepcopy(self.config)
        duplicate_judges["models"]["judges"]["replacement"]["model"] = (
            "models/judge-primary"
        )
        variants.append(duplicate_judges)
        reused_target = deepcopy(self.config)
        reused_target["models"]["selector"]["model"] = "models/target"
        variants.append(reused_target)
        same_fallback = deepcopy(self.config)
        same_fallback["models"]["perturbator"]["fallback_model"] = (
            "models/perturbator"
        )
        variants.append(same_fallback)
        wrong_secondary_cardinality = deepcopy(self.config)
        wrong_secondary_cardinality["models"]["judges"]["secondaries"].pop()
        variants.append(wrong_secondary_cardinality)
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ConfigError):
                self.load(value)

    def test_budget_sample_seed_and_search_boundaries_fail_before_output(self) -> None:
        variants = []
        target_budget = deepcopy(self.config)
        target_budget["experiment"]["conversation_rounds"] = -1
        variants.append(target_budget)
        judge_budget = deepcopy(self.config)
        judge_budget["experiment"]["verification_blocks"] = True
        variants.append(judge_budget)
        screening = deepcopy(self.config)
        screening["samples"]["screening"] = 2
        variants.append(screening)
        verification = deepcopy(self.config)
        verification["samples"]["verification"] = 4
        variants.append(verification)
        proposals = deepcopy(self.config)
        proposals["search"]["proposals_per_step"] = 2
        variants.append(proposals)
        no_seed = deepcopy(self.config)
        no_seed["seed"]["bootstrap_limit"] = 0
        variants.append(no_seed)
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ConfigError):
                self.load(value)
        self.assertFalse((self.root / "runs").exists())

    def test_zero_bootstrap_is_allowed_only_with_a_compatible_seed(self) -> None:
        value = deepcopy(self.config)
        value["seed"]["bootstrap_limit"] = 0
        loaded = self.load(value, has_compatible_seed=True)
        self.assertEqual(loaded.bootstrap_limit, 0)

    def test_frozen_resume_does_not_require_historical_library_or_output_paths(self) -> None:
        frozen = deepcopy(self.config)
        frozen["seed"]["library_directory"] = "removed-seed-library"
        frozen["output_directory"] = "removed-output-parent/runs"
        loaded = self.load(
            frozen,
            has_compatible_seed=True,
            frozen_resume=True,
        )
        self.assertEqual(
            loaded.seed_library_directory,
            self.root / "removed-seed-library",
        )
        self.assertEqual(
            loaded.output_directory,
            self.root / "removed-output-parent" / "runs",
        )

    def test_frozen_resume_accepts_only_the_manifest_mapping_boundary(self) -> None:
        config_path = self.root / "config.json"
        config_path.write_bytes(
            json.dumps(
                self.config,
                default=lambda value: (
                    float(value) if isinstance(value, Decimal) else value
                ),
            ).encode("utf-8")
        )
        with self.assertRaisesRegex(ConfigError, "normalized mapping"):
            self.load(
                config_path,
                has_compatible_seed=True,
                frozen_resume=True,
            )

    def test_unsafe_output_and_unknown_objective_are_rejected(self) -> None:
        unsafe = deepcopy(self.config)
        unsafe["output_directory"] = "objectives/generated-run"
        with self.assertRaises(ConfigError):
            self.load(unsafe)
        with self.assertRaises(ConfigError):
            load_config(
                self.config,
                repository_root=self.root,
                available_objectives={"different_objective"},
            )

    def test_network_provider_requires_closed_credential_free_https_config(self) -> None:
        valid = deepcopy(self.config)
        valid["provider"] = {
            "adapter": "openai_compatible",
            "base_url": "https://example.invalid/v1/",
            "api_key_env": "SYNTHETIC_API_KEY",
            "timeout_seconds": 300,
        }
        loaded = self.load(valid)
        self.assertEqual(loaded.provider.base_url, "https://example.invalid/v1")

        for bad_url in (
            "http://example.invalid/v1",
            "https://user:password@example.invalid/v1",
            "https://example.invalid/v1?token=secret",
        ):
            invalid = deepcopy(valid)
            invalid["provider"]["base_url"] = bad_url
            with self.subTest(url=bad_url), self.assertRaises(ConfigError):
                self.load(invalid)
        invalid_env = deepcopy(valid)
        invalid_env["provider"]["api_key_env"] = "NOT-AN-ENV-NAME"
        with self.assertRaises(ConfigError):
            self.load(invalid_env)
        invalid_timeout = deepcopy(valid)
        invalid_timeout["provider"]["timeout_seconds"] = 301
        with self.assertRaises(ConfigError):
            self.load(invalid_timeout)

    def test_scientific_projection_excludes_transport_and_output_location(self) -> None:
        first = self.load(self.config)
        moved = deepcopy(self.config)
        moved["provider"]["timeout_seconds"] = 2
        moved["output_directory"] = "other-runs"
        second = self.load(moved)
        self.assertEqual(
            digest_json(first.scientific_view()),
            digest_json(second.scientific_view()),
        )

    def test_config_digest_excludes_only_runtime_output_location(self) -> None:
        first = self.load(self.config)
        moved = deepcopy(self.config)
        moved["output_directory"] = "other-runs"
        second = self.load(moved)
        self.assertEqual(first.config_digest, second.config_digest)
        transported = deepcopy(self.config)
        transported["provider"]["timeout_seconds"] = 2
        third = self.load(transported)
        self.assertNotEqual(first.config_digest, third.config_digest)


if __name__ == "__main__":
    unittest.main()
