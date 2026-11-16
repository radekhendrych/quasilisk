from __future__ import annotations

from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from quasilisk.domain import entity_id
from quasilisk.engine import EngineError, _write_terminal_derivatives
from quasilisk.reporting import refresh_library_indexes


class AutomaticLibraryRefreshTests(unittest.TestCase):
    @staticmethod
    def _state(*, terminal: bool) -> SimpleNamespace:
        return SimpleNamespace(
            run_id=entity_id("run", ["library-refresh", terminal]),
            events=(),
            manifest={},
            terminal_event={"type": "run_terminated"} if terminal else None,
            current_seed_qualification_snapshots=(),
            current_strategy_qualification_snapshots=(),
        )

    def test_terminal_engine_path_writes_reports_without_creating_empty_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            state = self._state(terminal=True)
            with patch("quasilisk.engine._write_reports") as write_reports:
                _write_terminal_derivatives(
                    repository / "runs" / "complete",
                    state,
                    repository_root=repository,
                )
            write_reports.assert_called_once()
            self.assertFalse((repository / "library").exists())

    def test_active_state_cannot_publish_terminal_derivatives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            state = self._state(terminal=False)
            with patch("quasilisk.engine._write_reports") as write_reports:
                with self.assertRaisesRegex(
                    EngineError, "durable run_terminated event"
                ):
                    _write_terminal_derivatives(
                        repository / "runs" / "active",
                        state,
                        repository_root=repository,
                    )
            write_reports.assert_not_called()

    def test_no_new_seed_or_strategy_returns_without_any_library_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            self.assertFalse(
                refresh_library_indexes(repository, self._state(terminal=True))
            )
            self.assertFalse((repository / "library").exists())

    def test_refresh_failure_is_a_warning_and_reports_still_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            state = self._state(terminal=True)
            errors = StringIO()
            with patch(
                "quasilisk.reporting.refresh_library_indexes",
                side_effect=OSError("injected cache failure"),
            ), patch(
                "quasilisk.engine._write_reports"
            ) as write_reports, redirect_stderr(errors):
                _write_terminal_derivatives(
                    repository / "runs" / "failure",
                    state,
                    repository_root=repository,
                )
            write_reports.assert_called_once()
            self.assertEqual(
                errors.getvalue(),
                "index_refresh_failed: injected cache failure\n",
            )


if __name__ == "__main__":
    unittest.main()
