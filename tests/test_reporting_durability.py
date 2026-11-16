from __future__ import annotations

import os
from pathlib import Path
import tempfile
from decimal import Decimal
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import quasilisk.reporting as reporting
from quasilisk.domain import digest_json, entity_id
from quasilisk.reporting import (
    DERIVED_FILENAMES,
    ReportBundle,
    ReportingError,
    library_index_payloads,
    write_derived_artifacts,
)
from quasilisk.run_state import RunState


class DerivedArtifactDurabilityTests(unittest.TestCase):
    BUNDLE = ReportBundle(
        manifest={"checkpoint": "new-manifest"},
        archive={"checkpoint": "new-archive"},
        summary={"checkpoint": "new-summary"},
        markdown="# new summary\n",
    )

    @staticmethod
    def _transients(directory: Path) -> list[str]:
        return sorted(
            path.name
            for path in directory.iterdir()
            if path.name not in DERIVED_FILENAMES
        )

    def test_replacement_failure_restores_exact_bytes_and_prior_absence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            prior = {
                "manifest.json": b"old manifest bytes\x00\n",
                "archive.json": None,
                "summary.json": b"old summary bytes\r\n",
                "summary.md": None,
            }
            for filename, content in prior.items():
                if content is not None:
                    (directory / filename).write_bytes(content)

            real_replace = os.replace
            new_replacements = 0

            def fail_third_new(source: object, target: object) -> None:
                nonlocal new_replacements
                if ".new." in Path(source).name:
                    new_replacements += 1
                    if new_replacements == 3:
                        raise OSError("injected third replacement failure")
                real_replace(source, target)

            with patch.object(
                reporting, "build_report_bundle", return_value=self.BUNDLE
            ), patch.object(reporting.os, "replace", side_effect=fail_third_new):
                with self.assertRaisesRegex(
                    ReportingError,
                    "^derived artifact replacement failed; prior artifacts restored: "
                    "injected third replacement failure$",
                ):
                    write_derived_artifacts(directory, object())  # type: ignore[arg-type]

            for filename, content in prior.items():
                target = directory / filename
                if content is None:
                    self.assertFalse(target.exists(), filename)
                else:
                    self.assertEqual(target.read_bytes(), content, filename)
            self.assertEqual(self._transients(directory), [])


    def test_commit_directory_fsync_failure_rolls_back_complete_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            prior = {
                filename: f"old {filename}\n".encode()
                for filename in DERIVED_FILENAMES
            }
            for filename, content in prior.items():
                (directory / filename).write_bytes(content)

            real_fsync_directory = reporting._fsync_directory
            calls = 0

            def fail_commit_fsync(path: Path) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected commit directory fsync failure")
                real_fsync_directory(path)

            with patch.object(
                reporting, "build_report_bundle", return_value=self.BUNDLE
            ), patch.object(
                reporting, "_fsync_directory", side_effect=fail_commit_fsync
            ):
                with self.assertRaisesRegex(
                    ReportingError,
                    "^derived artifact replacement failed; prior artifacts restored: "
                    "injected commit directory fsync failure$",
                ):
                    write_derived_artifacts(directory, object())  # type: ignore[arg-type]

            self.assertEqual(
                {
                    filename: (directory / filename).read_bytes()
                    for filename in DERIVED_FILENAMES
                },
                prior,
            )
            self.assertEqual(self._transients(directory), [])

    def test_cleanup_directory_fsync_failure_reconstructs_prior_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            prior = {
                filename: f"pre-transaction {filename}\n".encode()
                for filename in DERIVED_FILENAMES
            }
            for filename, content in prior.items():
                (directory / filename).write_bytes(content)

            real_fsync_directory = reporting._fsync_directory
            calls = 0

            def fail_cleanup_fsync(path: Path) -> None:
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise OSError("injected cleanup directory fsync failure")
                real_fsync_directory(path)

            with patch.object(
                reporting, "build_report_bundle", return_value=self.BUNDLE
            ), patch.object(
                reporting, "_fsync_directory", side_effect=fail_cleanup_fsync
            ):
                with self.assertRaisesRegex(
                    ReportingError,
                    "^derived artifact replacement failed; prior artifacts restored: "
                    "directory fsync: injected cleanup directory fsync failure$",
                ):
                    write_derived_artifacts(directory, object())  # type: ignore[arg-type]

            self.assertEqual(
                {
                    filename: (directory / filename).read_bytes()
                    for filename in DERIVED_FILENAMES
                },
                prior,
            )
            self.assertEqual(self._transients(directory), [])


class LibraryIndexProjectionTests(unittest.TestCase):
    def test_nonempty_indexes_are_derived_from_exact_source_projections(self) -> None:
        run_id = entity_id("run", ["reporting-library-source"])
        event_hash = digest_json(["source-event"])
        source_checkpoint = {
            "source_run_id": run_id,
            "sequence": 1,
            "event_hash": event_hash,
            "source_path": "/durable/runs/source",
        }
        planned = 3
        block_id = entity_id("sb", [run_id, "verification", planned])
        seed = {
            "seed_id": entity_id("sd", [run_id, "seed"]),
            "type": "entry_seed",
            "verification_block_id": block_id,
            "planned_count": planned,
            "resolved_count": planned,
            "high_count": 2,
            "observed_success_fraction": "0.6666666666666666666666666667",
            "source_checkpoint": source_checkpoint,
        }
        strategy = {
            "template_id": entity_id("st", [run_id, "strategy"]),
            "template_scientific_key": digest_json(["strategy"]),
            "compatibility_key": digest_json(["strategy-compatibility"]),
            "qualification_status": "qualified",
            "source_checkpoint": source_checkpoint,
        }
        seed_decision = SimpleNamespace(
            qualified=True,
            snapshot=SimpleNamespace(seed_id=seed["seed_id"]),
            verification=SimpleNamespace(
                effective_scores=(Decimal("70"), Decimal("95"), Decimal("80"))
            ),
        )
        state = SimpleNamespace(
            run_id=run_id,
            events=({"event_hash": event_hash},),
            manifest={
                "objective": {"id": "fatalism_susceptibility"},
                "normalized_config": {
                    "models": {
                        "target": {
                            "model": "target/model",
                            "temperature": "0",
                            "max_output_tokens": 32,
                            "system_prompt": "",
                        }
                    }
                },
            },
            seed_qualifications={seed["seed_id"]: seed_decision},
            current_seed_qualification_snapshots=(seed,),
            current_strategy_qualification_snapshots=(strategy,),
        )
        state.qualified_seed_verification_median = lambda seed_id: (
            RunState.qualified_seed_verification_median(state, seed_id)
        )

        payloads = library_index_payloads((state,))

        self.assertEqual(set(payloads), {"seeds", "strategies"})
        self.assertEqual(
            payloads["seeds"]["entries"][0]["qualification_summary"][
                "verification_median"
            ],
            "80",
        )
        for kind, subject in (
            ("seeds", seed),
            ("strategies", strategy),
        ):
            entry = payloads[kind]["entries"][0]
            self.assertEqual(entry["source_checkpoint"], source_checkpoint)
            self.assertEqual(entry["subject_payload"], subject)

if __name__ == "__main__":
    unittest.main()
