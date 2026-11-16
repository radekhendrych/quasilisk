from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from foundation_support import chain_events, run_created_payload
from integration_support import (
    PUBLIC_RUN_FILES,
    dryrun_config,
    parsed_json,
    write_config,
)
import quasilisk.cli as cli_module
import quasilisk.reporting as reporting_module
from quasilisk.cli import main
from quasilisk.domain import (
    RunTerminalReason,
    canonical_json_bytes,
    strict_json_loads,
)
from quasilisk.engine import load_resources, rebuild_reports, run_new
from quasilisk.events import read_events
from quasilisk.reporting import ReportingError, archive_projection
from quasilisk.run_state import reduce_events


class CliContractTests(unittest.TestCase):
    def test_malformed_strict_json_is_invalid_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "malformed.json"
            config_path.write_text('{"objective":"fatalism_susceptibility",', encoding="utf-8")
            errors = StringIO()
            with redirect_stderr(errors):
                status = main(
                    ["--config", config_path.as_posix(), "--validate-only"]
                )
        self.assertEqual(status, 2)
        self.assertIn("invalid input", errors.getvalue())

    def test_derived_write_failure_uses_the_stable_public_marker(self) -> None:
        errors = StringIO()
        with patch.object(
            cli_module,
            "run_new",
            side_effect=ReportingError("injected report replacement failure"),
        ), redirect_stderr(errors):
            status = main(["--config", "ignored.json"])
        self.assertEqual(status, 3)
        self.assertEqual(
            errors.getvalue(),
            "derived_artifact_write_failed: injected report replacement failure\n",
        )

    def test_objective_listing_and_validate_only_are_read_only(self) -> None:
        listed = StringIO()
        with redirect_stdout(listed):
            status = main(["--list-objectives"])
        self.assertEqual(status, 0)
        self.assertEqual(
            listed.getvalue().splitlines(),
            [
                "fatalism_susceptibility",
                "nihilism_susceptibility",
                "relativism_susceptibility",
            ],
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_base = root / "validated-but-not-created"
            config_path = root / "config.json"
            write_config(config_path, dryrun_config("fatalism", output_base))
            output = StringIO()
            with redirect_stdout(output):
                status = main(["--config", config_path.as_posix(), "--validate-only"])
            self.assertEqual(status, 0)
            result = strict_json_loads(output.getvalue())
            self.assertEqual(result["status"], "valid")
            self.assertEqual(result["objective"], "fatalism_susceptibility")
            self.assertFalse(output_base.exists())

    def test_cli_rejects_conflicting_modes(self) -> None:
        errors = StringIO()
        with redirect_stderr(errors), self.assertRaises(SystemExit) as raised:
            main(["--config", "one.json", "--resume", "run-anything"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("not allowed with argument", errors.getvalue())

    def test_resume_requires_a_consolidated_run_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            empty_directory = Path(temporary) / "not-a-run"
            empty_directory.mkdir()
            errors = StringIO()
            with redirect_stderr(errors):
                status = main(["--resume", empty_directory.as_posix()])
        self.assertEqual(status, 2)
        self.assertIn("invalid input", errors.getvalue())
        self.assertIn("events.jsonl", errors.getvalue())

    def test_nonresumable_cli_resume_rebuilds_views_without_appending(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_directory, reason = run_new(dryrun_config("fatalism", root / "runs"))
            self.assertEqual(reason, RunTerminalReason.COMPLETED_NO_ELIGIBLE_WORK)
            journal_before = (run_directory / "events.jsonl").read_bytes()
            reports_before = {
                name: (run_directory / name).read_bytes()
                for name in PUBLIC_RUN_FILES
                if name != "events.jsonl"
            }
            for name in reports_before:
                (run_directory / name).unlink()

            output = StringIO()
            with redirect_stdout(output):
                status = main(["--resume", run_directory.as_posix()])
            self.assertEqual(status, 0)
            result = strict_json_loads(output.getvalue())
            self.assertEqual(result["run_directory"], run_directory.as_posix())
            self.assertEqual(result["reason"], "completed_no_eligible_work")
            self.assertEqual((run_directory / "events.jsonl").read_bytes(), journal_before)
            self.assertEqual(
                {
                    name: (run_directory / name).read_bytes()
                    for name in reports_before
                },
                reports_before,
            )


class LibraryIndexRebuildTests(unittest.TestCase):
    INDEX_KINDS = ("seeds", "strategies")

    def _repository(self, root: Path, prior: bytes | None = None) -> Path:
        repository = root / "repository"
        for kind in self.INDEX_KINDS:
            directory = repository / "library" / kind
            directory.mkdir(parents=True, exist_ok=True)
            if prior is not None:
                (directory / "index.json").write_bytes(prior + kind.encode())
        return repository

    def _run_root(
        self,
        root: Path,
        label: str,
    ) -> Path:
        created = run_created_payload(label=label)
        events = chain_events(created, [])
        run_root = root / f"source-{label}"
        run_root.mkdir()
        run_root.joinpath("events.jsonl").write_bytes(
            b"".join(canonical_json_bytes(event) + b"\n" for event in events)
        )
        return run_root

    def _index_bytes(self, repository: Path) -> dict[str, bytes]:
        return {
            kind: (repository / "library" / kind / "index.json").read_bytes()
            for kind in self.INDEX_KINDS
        }

    @staticmethod
    def _validate_synthetic_run(path: Path, **_: object):
        """Keep transaction tests focused on prevalidated synthetic journals."""

        return reduce_events(read_events(path))

    def test_rebuild_is_read_only_until_every_source_journal_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = self._repository(root, prior=b"prior-")
            valid = self._run_root(root, "valid")
            corrupt = root / "source-corrupt"
            corrupt.mkdir()
            corrupt.joinpath("events.jsonl").write_bytes(b"not-json\n")
            before = self._index_bytes(repository)

            errors = StringIO()
            with patch.object(
                cli_module, "REPOSITORY_ROOT", repository
            ), patch.object(
                cli_module,
                "validate_run_evidence",
                side_effect=self._validate_synthetic_run,
            ), redirect_stderr(errors):
                status = main(
                    [
                        "--rebuild-library-indexes",
                        valid.as_posix(),
                        corrupt.as_posix(),
                    ]
                )

            self.assertEqual(status, 3)
            self.assertIn("evidence error", errors.getvalue())
            self.assertEqual(self._index_bytes(repository), before)
            self.assertFalse((repository / "library" / ".indexes.lock").exists())
            self.assertEqual(set(valid.iterdir()), {valid / "events.jsonl"})

    def test_bad_paths_and_duplicate_roots_exit_two_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = self._repository(root, prior=b"prior-")
            run_root = self._run_root(root, "duplicate")
            non_run = root / "not-a-run"
            non_run.mkdir()
            before = self._index_bytes(repository)

            for arguments in (
                [run_root.as_posix(), run_root.as_posix()],
                [non_run.as_posix()],
            ):
                with self.subTest(arguments=arguments):
                    errors = StringIO()
                    with patch.object(
                        cli_module, "REPOSITORY_ROOT", repository
                    ), patch.object(
                        cli_module,
                        "validate_run_evidence",
                        side_effect=self._validate_synthetic_run,
                    ), redirect_stderr(errors):
                        status = main(["--rebuild-library-indexes", *arguments])
                    self.assertEqual(status, 2)
                    self.assertIn("invalid input", errors.getvalue())
                    self.assertEqual(self._index_bytes(repository), before)
                    self.assertFalse(
                        (repository / "library" / ".indexes.lock").exists()
                    )

    def test_success_writes_two_empty_indexes_for_run_without_promotions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = self._repository(root, prior=b"prior-")
            run_root = self._run_root(root, "no-promotions")
            journal_before = run_root.joinpath("events.jsonl").read_bytes()
            output = StringIO()

            with patch.object(
                cli_module, "REPOSITORY_ROOT", repository
            ), patch.object(
                cli_module,
                "validate_run_evidence",
                side_effect=self._validate_synthetic_run,
            ), redirect_stdout(output):
                status = main(
                    ["--rebuild-library-indexes", run_root.as_posix()]
                )

            self.assertEqual(status, 0)
            self.assertEqual(output.getvalue(), "rebuilt library indexes\n")
            expected = canonical_json_bytes({"entries": []}) + b"\n"
            self.assertEqual(
                self._index_bytes(repository),
                {kind: expected for kind in self.INDEX_KINDS},
            )
            self.assertEqual(run_root.joinpath("events.jsonl").read_bytes(), journal_before)
            self.assertEqual(set(run_root.iterdir()), {run_root / "events.jsonl"})

    def test_replacement_failure_rolls_back_every_prior_index_byte(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = self._repository(root, prior=b"prior-")
            run_root = self._run_root(root, "rollback")
            before = self._index_bytes(repository)
            real_replace = os.replace
            new_replacements = 0

            def fail_second_new(source: object, target: object) -> None:
                nonlocal new_replacements
                if str(source).endswith(".new"):
                    new_replacements += 1
                    if new_replacements == 2:
                        raise OSError("injected second-index failure")
                real_replace(source, target)

            errors = StringIO()
            with patch.object(
                cli_module, "REPOSITORY_ROOT", repository
            ), patch.object(
                cli_module,
                "validate_run_evidence",
                side_effect=self._validate_synthetic_run,
            ), patch.object(
                reporting_module.os, "replace", side_effect=fail_second_new
            ), redirect_stderr(errors):
                status = main(
                    ["--rebuild-library-indexes", run_root.as_posix()]
                )

            self.assertEqual(status, 3)
            self.assertIn("prior indexes restored", errors.getvalue())
            self.assertEqual(self._index_bytes(repository), before)
            transient = [
                path
                for kind in self.INDEX_KINDS
                for path in (repository / "library" / kind).iterdir()
                if path.name != "index.json"
            ]
            self.assertEqual(transient, [])

    def test_interrupted_publish_is_recovered_before_the_next_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = self._repository(root, prior=b"prior-")
            targets = {
                kind: {"entries": []}
                for kind in self.INDEX_KINDS
            }
            real_replace = os.replace
            interrupted = False

            def stop_after_first_publish(source: object, target: object) -> None:
                nonlocal interrupted
                real_replace(source, target)
                if str(source).endswith(".new") and not interrupted:
                    interrupted = True
                    raise SystemExit("simulated process death")

            with patch.object(
                reporting_module.os,
                "replace",
                side_effect=stop_after_first_publish,
            ), self.assertRaises(SystemExit):
                reporting_module.replace_library_indexes(repository, targets)

            marker = repository / "library" / ".indexes.transaction.json"
            self.assertTrue(marker.is_file())
            self.assertNotEqual(len(set(self._index_bytes(repository).values())), 1)

            reporting_module.replace_library_indexes(repository, targets)

            expected = canonical_json_bytes({"entries": []}) + b"\n"
            self.assertEqual(
                self._index_bytes(repository),
                {kind: expected for kind in self.INDEX_KINDS},
            )
            self.assertFalse(marker.exists())
            transients = [
                path
                for kind in self.INDEX_KINDS
                for path in (repository / "library" / kind).iterdir()
                if path.name != "index.json"
            ]
            self.assertEqual(transients, [])

    def test_failed_rollback_retains_transaction_for_next_locked_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = self._repository(root, prior=b"prior-")
            targets = {
                kind: {"entries": []}
                for kind in self.INDEX_KINDS
            }
            real_replace = os.replace
            new_replacements = 0
            rollback_failed = False

            def fail_publish_then_first_rollback(
                source: object, target: object
            ) -> None:
                nonlocal new_replacements, rollback_failed
                source_name = str(source)
                if source_name.endswith(".new"):
                    new_replacements += 1
                    if new_replacements == 2:
                        raise OSError("injected publish failure")
                if source_name.endswith(".rollback") and not rollback_failed:
                    rollback_failed = True
                    raise OSError("injected rollback failure")
                real_replace(source, target)

            with patch.object(
                reporting_module.os,
                "replace",
                side_effect=fail_publish_then_first_rollback,
            ), self.assertRaisesRegex(
                OSError, "durable recovery transaction retained"
            ):
                reporting_module.replace_library_indexes(repository, targets)

            marker = repository / "library" / ".indexes.transaction.json"
            self.assertTrue(marker.is_file())

            reporting_module.replace_library_indexes(repository, targets)

            expected = canonical_json_bytes({"entries": []}) + b"\n"
            self.assertEqual(
                self._index_bytes(repository),
                {kind: expected for kind in self.INDEX_KINDS},
            )
            self.assertFalse(marker.exists())

    def test_corrupt_transaction_marker_cannot_target_unrelated_library_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            library_root = Path(temporary) / "library"
            library_root.mkdir()
            victim = library_root / "unrelated.json"
            victim.write_bytes(b"must remain")
            digest = "sha256:" + "0" * 64
            marker = library_root / ".indexes.transaction.json"
            marker.write_bytes(
                canonical_json_bytes(
                    {
                        "version": 1,
                        "records": [
                            {
                                "target": "unrelated.json",
                                "new": ".unrelated.json.000000000000000000000000.new",
                                "rollback": None,
                                "new_digest": digest,
                                "previous_digest": None,
                            }
                        ],
                    }
                )
                + b"\n"
            )

            with self.assertRaisesRegex(OSError, "invalid target"):
                reporting_module._recover_library_index_transaction(library_root)
            self.assertEqual(victim.read_bytes(), b"must remain")
            self.assertTrue(marker.exists())


class ReportingReplayTests(unittest.TestCase):
    def test_reports_are_reproducible_from_events_after_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = dryrun_config("nihilism", root / "runs")
            objective = load_resources(config).objective
            run_directory, reason = run_new(config)
            expected = {
                name: (run_directory / name).read_bytes()
                for name in PUBLIC_RUN_FILES
                if name != "events.jsonl"
            }
            evidence = (run_directory / "events.jsonl").read_bytes()
            for name in expected:
                (run_directory / name).write_bytes(b"stale derived artifact\n")

            rebuilt_reason = rebuild_reports(run_directory)
            self.assertEqual(rebuilt_reason, reason)
            self.assertEqual((run_directory / "events.jsonl").read_bytes(), evidence)
            self.assertEqual(
                {
                    name: (run_directory / name).read_bytes()
                    for name in expected
                },
                expected,
            )
            archive = parsed_json(run_directory / "archive.json")
            summary = parsed_json(run_directory / "summary.json")
            replayed = reduce_events(read_events(run_directory), objective)
            self.assertEqual(archive, archive_projection(replayed))
            self.assertEqual(
                summary["archive_status"]["occupied_cell_count"],
                archive["metrics"]["occupied_cell_count"],
            )


if __name__ == "__main__":
    unittest.main()
