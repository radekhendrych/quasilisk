from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from integration_support import (
    PUBLIC_RUN_FILES,
    assert_complete_call_audit,
    dispatch_role_counts,
    dryrun_config,
    parsed_json,
    read_validated_events,
)
from quasilisk.domain import RunTerminalReason
from quasilisk.engine import run_new


class ScriptedDryRunTests(unittest.TestCase):
    def test_all_three_checked_in_dry_runs_are_complete_audited_runs(self) -> None:
        expected_objectives = {
            "fatalism": "fatalism_susceptibility",
            "nihilism": "nihilism_susceptibility",
            "relativism": "relativism_susceptibility",
        }
        for name, objective_id in expected_objectives.items():
            with self.subTest(objective=name), tempfile.TemporaryDirectory() as temporary:
                output_base = Path(temporary) / "runs"
                run_directory, reason = run_new(dryrun_config(name, output_base))

                self.assertEqual(reason, RunTerminalReason.COMPLETED_NO_ELIGIBLE_WORK)
                self.assertEqual(run_directory.parent, output_base.resolve())
                self.assertRegex(run_directory.name, r"^run-[0-9a-f]{64}$")
                self.assertEqual(
                    {path.name for path in run_directory.iterdir() if not path.name.startswith(".")},
                    PUBLIC_RUN_FILES,
                )
                self.assertLessEqual(
                    {path.name for path in run_directory.iterdir() if path.name.startswith(".")},
                    {".events.lock"},
                )

                events = read_validated_events(run_directory)
                assert_complete_call_audit(self, events)
                self.assertEqual(
                    [event["sequence"] for event in events],
                    list(range(1, len(events) + 1)),
                )
                self.assertEqual(events[0]["payload"]["manifest"]["objective"]["id"], objective_id)
                self.assertFalse(events[-1]["payload"]["resumable"])

                manifest = parsed_json(run_directory / "manifest.json")
                archive = parsed_json(run_directory / "archive.json")
                summary = parsed_json(run_directory / "summary.json")
                self.assertEqual(manifest, events[0]["payload"]["manifest"])
                self.assertEqual(summary["headline"]["objective"], objective_id)
                self.assertEqual(summary["source_checkpoint"]["event_hash"], events[-1]["event_hash"])
                self.assertEqual(summary["source_checkpoint"]["sequence"], len(events))

                counts = dispatch_role_counts(events)
                for role in ("target", "judge", "selector", "perturbator"):
                    self.assertEqual(
                        summary["role_accounting"][role]["attempts_consumed"],
                        counts[role],
                    )
                self.assertEqual(counts["target"], summary["headline"]["physical_target_attempts_consumed"])
                self.assertGreaterEqual(counts["target"], 1)
                self.assertGreaterEqual(counts["judge"], 1)
                self.assertGreaterEqual(counts["perturbator"], 1)

                metrics = archive["metrics"]
                self.assertEqual(metrics["occupied_cell_count"], len(archive["cells"]))
                self.assertEqual(metrics["occupied_cell_count"], 1)
                self.assertEqual(summary["archive_status"]["occupied_cell_count"], 1)
                entry = archive["cells"][0]["entries"][0]
                self.assertTrue(entry["evidence"]["event_references"])
                event_ids = {event["event_id"] for event in events}
                self.assertLessEqual(set(entry["evidence"]["event_references"]), event_ids)

                # Exact provider text and body evidence are both retained.
                for event in events:
                    if event["type"] != "model_call_succeeded":
                        continue
                    body = event["payload"]["raw_body"]
                    if body["encoding"] == "utf8":
                        self.assertEqual(body["data"], event["payload"]["text"])

                markdown = (run_directory / "summary.md").read_text(encoding="utf-8")
                self.assertIn(objective_id, markdown)
                self.assertTrue(markdown.endswith("\n"))

    def test_same_reusable_output_base_creates_distinct_repeatable_children(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_base = Path(temporary) / "repeatable"
            first, first_reason = run_new(dryrun_config("fatalism", output_base))
            first_journal = (first / "events.jsonl").read_bytes()
            second, second_reason = run_new(dryrun_config("fatalism", output_base))

            self.assertEqual(first_reason, RunTerminalReason.COMPLETED_NO_ELIGIBLE_WORK)
            self.assertEqual(second_reason, first_reason)
            self.assertNotEqual(first, second)
            self.assertEqual((first / "events.jsonl").read_bytes(), first_journal)
            self.assertEqual(
                {path for path in output_base.iterdir() if path.is_dir()},
                {first, second},
            )

            first_events = read_validated_events(first)
            second_events = read_validated_events(second)
            first_intents = [
                event["payload"]
                for event in first_events
                if event["type"] == "model_call_intended"
            ]
            second_intents = [
                event["payload"]
                for event in second_events
                if event["type"] == "model_call_intended"
            ]
            self.assertEqual(
                [item["semantic_request_hash"] for item in first_intents],
                [item["semantic_request_hash"] for item in second_intents],
            )
            self.assertEqual(
                [item["logical_action_scientific_key"] for item in first_intents],
                [item["logical_action_scientific_key"] for item in second_intents],
            )
            self.assertEqual(
                [item["request_body"] for item in first_intents],
                [item["request_body"] for item in second_intents],
            )


if __name__ == "__main__":
    unittest.main()
