"""External seed selection stays strict, reproducible, and read-only on validation."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from integration_support import dryrun_config, write_config
from quasilisk.cli import main
from quasilisk.engine import load_resources
from quasilisk.objective import ObjectiveError


ROOT = Path(__file__).resolve().parent.parent


class ExternalSeedPromptTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "runs"
        self.config = dryrun_config("fatalism", self.output)
        self.payload = json.loads(
            (ROOT / "objectives/fatalism_susceptibility/bootstrap.json").read_text()
        )
        self.payload["probes"] = [self.payload["probes"][0]]
        self.seeds = self.root / "selected.json"
        self.config["seed"]["prompts_path"] = str(self.seeds)

    def write(self, payload):
        self.seeds.write_text(json.dumps(payload), encoding="utf-8")

    def test_contents_define_scope_while_file_placement_does_not(self):
        self.write(self.payload)
        original = load_resources(self.config)
        moved = self.root / "same-contents.json"
        moved.write_bytes(self.seeds.read_bytes())
        self.config["seed"]["prompts_path"] = str(moved)
        relocated = load_resources(self.config)
        self.assertEqual(original.scientific_scope_digest, relocated.scientific_scope_digest)
        changed = deepcopy(self.payload)
        changed["probes"][0]["prompt"] += " State the governing decision explicitly."
        moved.write_text(json.dumps(changed), encoding="utf-8")
        edited = load_resources(self.config)
        self.assertNotEqual(original.scientific_scope_digest, edited.scientific_scope_digest)
        self.assertEqual(edited.bootstrap_payloads[0]["prompt"], changed["probes"][0]["prompt"])
        self.assertFalse(self.output.exists())

    def test_wrong_objective_duplicate_prompt_and_unknown_fields_fail_closed(self):
        wrong = deepcopy(self.payload)
        wrong["objective_id"] = "nihilism_susceptibility"
        duplicate = deepcopy(self.payload)
        second = deepcopy(duplicate["probes"][0])
        second["probe_id"] += "_duplicate"
        duplicate["probes"].append(second)
        unknown = deepcopy(self.payload)
        unknown["override"] = True
        for payload in (wrong, duplicate, unknown):
            with self.subTest(payload=payload):
                self.write(payload)
                with self.assertRaises(ObjectiveError):
                    load_resources(self.config)
        self.assertFalse(self.output.exists())

    def test_cli_resolves_selected_file_against_cwd_and_validation_makes_no_call(self):
        self.write(self.payload)
        self.config["seed"]["prompts_path"] = None
        config_path = self.root / "config.json"
        write_config(config_path, self.config)
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            with patch("quasilisk.engine.create_provider") as provider:
                with redirect_stdout(StringIO()):
                    status = main([
                        "--config", str(config_path), "--seed-prompts", "selected.json",
                        "--validate-only",
                    ])
                self.assertEqual(status, 0)
                provider.return_value.complete.assert_not_called()
        finally:
            os.chdir(previous)
        self.assertFalse(self.output.exists())

    def test_cli_reports_malformed_selected_json_as_invalid_input(self):
        self.seeds.write_text('{"objective_id":', encoding="utf-8")
        config_path = self.root / "config.json"
        write_config(config_path, self.config)
        with redirect_stderr(StringIO()):
            self.assertEqual(main([
                "--config", str(config_path), "--seed-prompts", str(self.seeds),
                "--validate-only",
            ]), 2)
        config_path.write_text('{"seed":', encoding="utf-8")
        with redirect_stderr(StringIO()):
            self.assertEqual(main([
                "--config", str(config_path), "--seed-prompts", str(self.seeds),
                "--validate-only",
            ]), 2)

    def test_cli_cannot_change_seeds_during_resume(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
            main(["--resume", "unused", "--seed-prompts", "selected.json"])
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
