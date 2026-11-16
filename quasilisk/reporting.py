"""Event-derived machine and human reports for one replayed run.

Reporting consumes only :class:`quasilisk.run_state.RunState` and immutable
domain views.  It never imports or calls the engine, accepts no alternate fact
source, and never appends scientific events.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import fcntl
import os
from pathlib import Path
import re
import secrets
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .domain import (
    ArchiveEntry,
    DomainError,
    canonical_json_bytes,
    digest_json,
    normalize_decimal,
    require_digest,
    require_entity_id,
    sha256_bytes,
    strict_json_loads,
)
from .run_state import PanelView, PortfolioChildView, RunState, StrategyEpisodeView


REPORT_SCHEMA_REVISION = 2
DERIVED_FILENAMES = ("manifest.json", "archive.json", "summary.json", "summary.md")


class ReportingError(DomainError):
    """A replayed state cannot be projected without inventing a fact."""


@dataclass(frozen=True, slots=True)
class ReportBundle:
    manifest: Mapping[str, Any]
    archive: Mapping[str, Any]
    summary: Mapping[str, Any]
    markdown: str


@dataclass(frozen=True, slots=True)
class DerivedArtifactPaths:
    manifest: Path
    archive: Path
    summary_json: Path
    summary_markdown: Path


def _library_qualification_summary(
    kind: str,
    snapshot: Mapping[str, Any],
    state: RunState,
) -> dict[str, Any]:
    if kind == "seeds":
        seed_id = snapshot.get("seed_id")
        if not isinstance(seed_id, str):
            raise ReportingError("qualified seed snapshot lacks its seed ID")
        try:
            verification_median = state.qualified_seed_verification_median(seed_id)
        except DomainError as exc:
            raise ReportingError(
                f"qualified seed verification median is unavailable: {exc}"
            ) from exc
        return {
            "qualification_status": "qualified",
            "planned_count": snapshot.get("planned_count"),
            "resolved_count": snapshot.get("resolved_count"),
            "high_count": snapshot.get("high_count"),
            "observed_success_fraction": snapshot.get(
                "observed_success_fraction"
            ),
            "verification_median": normalize_decimal(verification_median),
        }
    if kind == "strategies":
        return {
            "qualification_status": snapshot.get("qualification_status"),
            "template_id": snapshot.get("template_id"),
            "template_scientific_key": snapshot.get(
                "template_scientific_key"
            ),
            "compatibility_key": snapshot.get("compatibility_key"),
            "objective_id": snapshot.get("objective_id"),
            "target_generation_profile_digest": snapshot.get(
                "target_generation_profile_digest"
            ),
        }
    raise ReportingError(f"unknown library index kind: {kind}")


def library_index_payloads(
    states: Sequence[RunState],
) -> Mapping[str, Mapping[str, Any]]:
    """Build both rebuildable indexes solely from reducer projections."""

    result: dict[str, Mapping[str, Any]] = {}
    for kind, identity_field in (
        ("seeds", "seed_id"),
        ("strategies", "template_id"),
    ):
        selected: dict[str, dict[str, Any]] = {}
        semantic_by_identity: dict[str, str] = {}
        for state in sorted(states, key=lambda item: item.run_id):
            if kind == "seeds":
                snapshots = state.current_seed_qualification_snapshots
            elif kind == "strategies":
                snapshots = state.current_strategy_qualification_snapshots
            if not isinstance(snapshots, (tuple, list)):
                raise ReportingError(
                    f"RunState current {kind} projection is not an array"
                )
            for raw in snapshots:
                if not isinstance(raw, Mapping):
                    raise ReportingError(
                        f"RunState current {kind} projection contains a non-object"
                    )
                snapshot = dict(raw)
                status = snapshot.get("qualification_status", "qualified")
                accepted_statuses = {"qualified"}
                if status not in accepted_statuses:
                    continue
                identity = snapshot.get(identity_field)
                checkpoint = snapshot.get("source_checkpoint")
                if not isinstance(identity, str) or not isinstance(checkpoint, Mapping):
                    raise ReportingError(
                        f"publishable {kind} projection lacks identity/checkpoint"
                    )
                if checkpoint.get("source_run_id") != state.run_id:
                    raise ReportingError(
                        f"publishable {kind} checkpoint names another source run"
                    )
                sequence = checkpoint.get("sequence")
                if (
                    isinstance(sequence, bool)
                    or not isinstance(sequence, int)
                    or not 1 <= sequence <= len(state.events)
                    or state.events[sequence - 1]["event_hash"]
                    != checkpoint.get("event_hash")
                ):
                    raise ReportingError(
                        f"publishable {kind} checkpoint is not in its source journal"
                    )
                semantic_snapshot = dict(snapshot)
                semantic_snapshot.pop("source_checkpoint", None)
                semantic_digest = digest_json(semantic_snapshot)
                prior_digest = semantic_by_identity.get(identity)
                if prior_digest is not None and prior_digest != semantic_digest:
                    raise ReportingError(
                        f"conflicting {kind} projections share {identity}"
                    )
                semantic_by_identity[identity] = semantic_digest
                entry = {
                    "source_checkpoint": dict(checkpoint),
                    "subject_payload": snapshot,
                    "qualification_summary": _library_qualification_summary(
                        kind, snapshot, state
                    ),
                }
                if kind == "strategies":
                    manifest_objective = _as_mapping(
                        state.manifest.get("objective")
                    )
                    target_profile = _as_mapping(
                        _as_mapping(
                            _as_mapping(
                                state.manifest.get("normalized_config")
                            ).get("models")
                        ).get("target")
                    )
                    entry["qualification_summary"].update(
                        {
                            "objective_id": manifest_objective.get("id"),
                            "target_generation_profile_digest": digest_json(
                                dict(target_profile)
                            ),
                        }
                    )
                current = selected.get(identity)
                if current is None or (
                    checkpoint.get("source_run_id"),
                    checkpoint.get("sequence"),
                    checkpoint.get("event_hash"),
                ) < (
                    current["source_checkpoint"].get("source_run_id"),
                    current["source_checkpoint"].get("sequence"),
                    current["source_checkpoint"].get("event_hash"),
                ):
                    selected[identity] = entry
        result[kind] = {
            "entries": [selected[key] for key in sorted(selected)]
        }
    return result


LIBRARY_INDEX_IDENTITIES = {
    "seeds": "seed_id",
    "strategies": "template_id",
}
_LIBRARY_TRANSACTION_FILENAME = ".indexes.transaction.json"


def _library_targets(library_root: Path) -> Mapping[str, Path]:
    if not library_root.is_dir():
        raise OSError(
            f"library root is missing or not a directory: {library_root}"
        )
    return {
        kind: library_root / kind / "index.json"
        for kind in LIBRARY_INDEX_IDENTITIES
    }


def _transaction_path(library_root: Path) -> Path:
    return library_root / _LIBRARY_TRANSACTION_FILENAME


def _transaction_member(library_root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str):
        raise OSError(f"library transaction {label} is not a relative path")
    relative = Path(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise OSError(f"library transaction {label} escapes the library root")
    path = (library_root / relative).resolve(strict=False)
    try:
        path.relative_to(library_root.resolve(strict=True))
    except ValueError as exc:
        raise OSError(
            f"library transaction {label} escapes the library root"
        ) from exc
    return path


def _write_synced_file(path: Path, content: bytes) -> None:
    with path.open("xb", buffering=0) as handle:
        remaining = memoryview(content)
        while remaining:
            written = handle.write(remaining)
            if written is None or written <= 0:
                raise OSError(f"short write while persisting {path}")
            remaining = remaining[written:]
        handle.flush()
        os.fsync(handle.fileno())


def _remove_if_present(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _file_digest(path: Path) -> str | None:
    return sha256_bytes(path.read_bytes()) if path.is_file() else None


def _validated_library_payload(
    kind: str,
    payload: object,
) -> dict[str, Any]:
    """Validate the closed index envelope used by both readers and writers."""

    if (
        not isinstance(payload, Mapping)
        or set(payload) != {"entries"}
        or not isinstance(payload.get("entries"), (list, tuple))
    ):
        raise ReportingError(f"{kind} library index is not a closed entries object")
    identity_field = LIBRARY_INDEX_IDENTITIES[kind]
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(payload["entries"]):
        if not isinstance(raw, Mapping) or set(raw) != {
            "qualification_summary",
            "source_checkpoint",
            "subject_payload",
        }:
            raise ReportingError(
                f"{kind} library entry {ordinal} has an invalid closed shape"
            )
        checkpoint = raw["source_checkpoint"]
        subject = raw["subject_payload"]
        summary = raw["qualification_summary"]
        if (
            not isinstance(checkpoint, Mapping)
            or set(checkpoint)
            != {"source_run_id", "sequence", "event_hash", "source_path"}
            or not isinstance(subject, Mapping)
            or not isinstance(summary, Mapping)
            or subject.get("source_checkpoint") != checkpoint
        ):
            raise ReportingError(
                f"{kind} library entry {ordinal} is internally inconsistent"
            )
        identity = subject.get(identity_field)
        sequence = checkpoint.get("sequence")
        source_run_id = checkpoint.get("source_run_id")
        source_path = checkpoint.get("source_path")
        try:
            require_entity_id(identity, f"{kind} library {identity_field}")
            expected_prefix = "sd-" if kind == "seeds" else "st-"
            if not identity.startswith(expected_prefix):
                raise DomainError(
                    f"{kind} library {identity_field} has the wrong prefix"
                )
        except DomainError as exc:
            raise ReportingError(str(exc)) from exc
        if identity in seen:
            raise ReportingError(
                f"{kind} library index contains duplicate {identity_field} {identity}"
            )
        source_path_value = Path(source_path) if isinstance(source_path, str) else None
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 1
            or source_path_value is None
            or not source_path_value.is_absolute()
            or source_path_value.as_posix() != source_path
            or any(part in {".", ".."} for part in source_path_value.parts)
        ):
            raise ReportingError(
                f"{kind} library entry {ordinal} has an invalid source checkpoint"
            )
        try:
            require_entity_id(source_run_id, "library checkpoint source run ID")
            if not source_run_id.startswith("run-"):
                raise DomainError(
                    "library checkpoint source run ID has the wrong prefix"
                )
            require_digest(
                checkpoint.get("event_hash"),
                f"{kind} library source checkpoint event hash",
            )
        except DomainError as exc:
            raise ReportingError(str(exc)) from exc
        seen.add(identity)
        normalized.append(
            {
                "source_checkpoint": dict(checkpoint),
                "subject_payload": dict(subject),
                "qualification_summary": dict(summary),
            }
        )
    normalized.sort(key=lambda item: item["subject_payload"][identity_field])
    return {"entries": normalized}


def _read_library_payload(target: Path, kind: str) -> dict[str, Any]:
    if not target.exists():
        return {"entries": []}
    if not target.is_file():
        raise OSError(f"library index target is not a file: {target}")
    try:
        parsed = strict_json_loads(target.read_bytes())
    except (OSError, DomainError) as exc:
        raise ReportingError(f"cannot read {kind} library index: {exc}") from exc
    return _validated_library_payload(kind, parsed)


def _entry_semantic_digest(entry: Mapping[str, Any]) -> str:
    subject = dict(entry["subject_payload"])
    subject.pop("source_checkpoint", None)
    return digest_json(
        {
            "subject_payload": subject,
            "qualification_summary": dict(entry["qualification_summary"]),
        }
    )


def _checkpoint_order(entry: Mapping[str, Any]) -> tuple[str, int, str]:
    checkpoint = entry["source_checkpoint"]
    return (
        checkpoint["source_run_id"],
        checkpoint["sequence"],
        checkpoint["event_hash"],
    )


def _merge_library_payload(
    kind: str,
    existing: Mapping[str, Any],
    additions: Mapping[str, Any],
) -> dict[str, Any]:
    identity_field = LIBRARY_INDEX_IDENTITIES[kind]
    selected = {
        entry["subject_payload"][identity_field]: dict(entry)
        for entry in existing["entries"]
    }
    for entry in additions["entries"]:
        identity = entry["subject_payload"][identity_field]
        current = selected.get(identity)
        if current is not None:
            if _entry_semantic_digest(current) != _entry_semantic_digest(entry):
                raise ReportingError(
                    f"conflicting {kind} projections share {identity}"
                )
            if _checkpoint_order(current) <= _checkpoint_order(entry):
                continue
        selected[identity] = dict(entry)
    return {
        "entries": [selected[identity] for identity in sorted(selected)]
    }


@contextmanager
def _library_writer_lock(
    library_root: Path, *, wait: bool = False
) -> Iterator[None]:
    _library_targets(library_root)
    lock_path = library_root / ".indexes.lock"
    try:
        lock_handle = lock_path.open("a+b")
    except OSError as exc:
        raise OSError(f"cannot open library writer lock: {exc}") from exc
    locked = False
    try:
        try:
            operation = fcntl.LOCK_EX if wait else fcntl.LOCK_EX | fcntl.LOCK_NB
            fcntl.flock(lock_handle.fileno(), operation)
            locked = True
        except (BlockingIOError, OSError) as exc:
            raise OSError(f"cannot acquire library writer lock: {exc}") from exc
        _recover_library_index_transaction(library_root)
        yield
    finally:
        if locked:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        try:
            lock_handle.close()
        except OSError:
            pass


def _recover_library_index_transaction(library_root: Path) -> None:
    """Finish or roll back one durable multi-index publish marker."""

    marker = _transaction_path(library_root)
    if not marker.exists():
        return
    try:
        marker_bytes = marker.read_bytes()
        value = strict_json_loads(marker_bytes)
        canonical_marker = canonical_json_bytes(value) + b"\n"
    except (OSError, DomainError) as exc:
        raise OSError(f"library transaction marker is corrupt: {exc}") from exc
    if canonical_marker != marker_bytes:
        raise OSError("library transaction marker is not canonical JSON")
    if (
        not isinstance(value, dict)
        or set(value) != {"records", "version"}
        or value.get("version") != 1
        or not isinstance(value.get("records"), list)
        or not value["records"]
    ):
        raise OSError("library transaction marker has an invalid closed shape")

    expected_targets = {
        f"{kind}/index.json" for kind in LIBRARY_INDEX_IDENTITIES
    }
    seen_targets: set[str] = set()
    generation_token: str | None = None
    records: list[dict[str, Any]] = []
    for ordinal, item in enumerate(value["records"]):
        if not isinstance(item, dict) or set(item) != {
            "new",
            "new_digest",
            "previous_digest",
            "rollback",
            "target",
        }:
            raise OSError(
                f"library transaction record {ordinal} has an invalid shape"
            )
        new_digest = item["new_digest"]
        previous_digest = item["previous_digest"]
        try:
            require_digest(new_digest, "library transaction new digest")
            if previous_digest is not None:
                require_digest(
                    previous_digest, "library transaction previous digest"
                )
        except DomainError as exc:
            raise OSError(str(exc)) from exc
        target_value = item["target"]
        if (
            not isinstance(target_value, str)
            or target_value not in expected_targets
            or target_value in seen_targets
        ):
            raise OSError(
                f"library transaction record {ordinal} has an invalid target"
            )
        seen_targets.add(target_value)
        target = _transaction_member(library_root, target_value, "target")
        new = _transaction_member(library_root, item["new"], "new")
        new_match = re.fullmatch(
            rf"\.{re.escape(target.name)}\.([0-9a-f]{{24}})\.new",
            new.name,
        )
        if new.parent != target.parent or new_match is None:
            raise OSError(
                f"library transaction record {ordinal} has an invalid new image"
            )
        token = new_match.group(1)
        if generation_token is None:
            generation_token = token
        elif token != generation_token:
            raise OSError("library transaction records mix generations")
        rollback_value = item["rollback"]
        if (rollback_value is None) != (previous_digest is None):
            raise OSError(
                f"library transaction record {ordinal} has an invalid rollback image"
            )
        rollback = (
            None
            if rollback_value is None
            else _transaction_member(library_root, rollback_value, "rollback")
        )
        if rollback is not None and (
            rollback.parent != target.parent
            or rollback.name != f".{target.name}.{token}.rollback"
        ):
            raise OSError(
                f"library transaction record {ordinal} has an invalid rollback image"
            )
        records.append(
            {
                "target": target,
                "new": new,
                "rollback": rollback,
                "new_digest": new_digest,
                "previous_digest": previous_digest,
            }
        )
    if seen_targets != expected_targets:
        raise OSError("library transaction does not cover the complete index set")

    committed = all(
        _file_digest(record["target"]) == record["new_digest"]
        for record in records
    )
    touched = {record["target"].parent for record in records}
    if not committed:
        for record in records:
            target = record["target"]
            previous_digest = record["previous_digest"]
            current_digest = _file_digest(target)
            if previous_digest is None:
                if current_digest not in {None, record["new_digest"]}:
                    raise OSError(
                        "library transaction target has an unrecognized generation: "
                        f"{target}"
                    )
                if current_digest == record["new_digest"]:
                    _remove_if_present(target)
                continue
            if current_digest == previous_digest:
                continue
            if current_digest != record["new_digest"]:
                raise OSError(
                    "library transaction target has an unrecognized generation: "
                    f"{target}"
                )
            rollback = record["rollback"]
            if rollback is None or _file_digest(rollback) != previous_digest:
                raise OSError(
                    f"cannot recover prior library index bytes for {target}"
                )
            os.replace(rollback, target)

    for record in records:
        _remove_if_present(record["new"])
        rollback = record["rollback"]
        if rollback is not None:
            _remove_if_present(rollback)
    for directory in sorted(touched):
        _fsync_directory(directory)
    _remove_if_present(marker)
    _fsync_directory(library_root)


def _publish_library_indexes_locked(
    library_root: Path,
    payloads: Mapping[str, Mapping[str, Any]],
) -> None:
    """Publish one complete validated generation while its writer lock is held."""

    targets = _library_targets(library_root)
    if set(payloads) != set(targets):
        raise ReportingError(
            "library index replacement requires the complete canonical target set"
        )
    normalized = {
        kind: _validated_library_payload(kind, payloads[kind])
        for kind in LIBRARY_INDEX_IDENTITIES
    }
    for target in targets.values():
        relative = target.relative_to(library_root).as_posix()
        if _transaction_member(library_root, relative, "target") != target:
            raise OSError(
                f"library index target resolves outside its canonical path: {target}"
            )

    token = secrets.token_hex(12)
    marker_path = _transaction_path(library_root)
    records: list[dict[str, Any]] = []
    touched_directories: set[Path] = set()
    try:
        for kind in LIBRARY_INDEX_IDENTITIES:
            target = targets[kind]
            payload = normalized[kind]
            target.parent.mkdir(parents=True, exist_ok=True)
            touched_directories.add(target.parent)
            if target.exists() and not target.is_file():
                raise OSError(f"library index target is not a file: {target}")
            previous = target.read_bytes() if target.exists() else None
            new_path = target.with_name(f".{target.name}.{token}.new")
            rollback_path = target.with_name(
                f".{target.name}.{token}.rollback"
            )
            new_bytes = canonical_json_bytes(payload) + b"\n"
            record = {
                "target": target,
                "new": new_path,
                "rollback": rollback_path,
                "previous": previous,
                "new_digest": sha256_bytes(new_bytes),
                "previous_digest": (
                    None if previous is None else sha256_bytes(previous)
                ),
            }
            records.append(record)
            _write_synced_file(new_path, new_bytes)
            if previous is not None:
                _write_synced_file(rollback_path, previous)
        for directory in sorted(touched_directories):
            _fsync_directory(directory)
        marker_payload = {
            "version": 1,
            "records": [
                {
                    "target": record["target"]
                    .relative_to(library_root)
                    .as_posix(),
                    "new": record["new"].relative_to(library_root).as_posix(),
                    "rollback": (
                        None
                        if record["previous"] is None
                        else record["rollback"]
                        .relative_to(library_root)
                        .as_posix()
                    ),
                    "new_digest": record["new_digest"],
                    "previous_digest": record["previous_digest"],
                }
                for record in records
            ],
        }
        _write_synced_file(
            marker_path,
            canonical_json_bytes(marker_payload) + b"\n",
        )
        _fsync_directory(library_root)
    except (OSError, ReportingError):
        for record in records:
            for transient in (record["new"], record["rollback"]):
                try:
                    _remove_if_present(transient)
                except OSError:
                    pass
        try:
            _remove_if_present(marker_path)
            _fsync_directory(library_root)
        except OSError:
            pass
        raise

    try:
        for record in records:
            os.replace(record["new"], record["target"])
        for directory in sorted(touched_directories):
            _fsync_directory(directory)
    except OSError as exc:
        rollback_errors: list[str] = []
        for record in records:
            target = record["target"]
            try:
                if record["previous"] is None:
                    _remove_if_present(target)
                else:
                    os.replace(record["rollback"], target)
            except OSError as rollback_exc:
                rollback_errors.append(f"{target}: {rollback_exc}")
        for directory in sorted(touched_directories):
            try:
                _fsync_directory(directory)
            except OSError as rollback_exc:
                rollback_errors.append(f"{directory}: {rollback_exc}")
        if rollback_errors:
            raise OSError(
                "library index replacement and rollback incomplete; "
                "durable recovery transaction retained: "
                + "; ".join(rollback_errors)
            ) from exc
        for record in records:
            for transient in (record["new"], record["rollback"]):
                try:
                    _remove_if_present(transient)
                except OSError as rollback_exc:
                    rollback_errors.append(f"{transient}: {rollback_exc}")
        if rollback_errors:
            raise OSError(
                "library index rollback cleanup incomplete; durable "
                "recovery transaction retained: "
                + "; ".join(rollback_errors)
            ) from exc
        try:
            _remove_if_present(marker_path)
            _fsync_directory(library_root)
        except OSError as rollback_exc:
            rollback_errors.append(f"{marker_path}: {rollback_exc}")
        raise OSError(
            f"library index replacement failed; prior indexes restored: {exc}"
        ) from exc

    cleanup_errors: list[str] = []
    for record in records:
        try:
            _remove_if_present(record["rollback"])
        except OSError as cleanup_exc:
            cleanup_errors.append(f"{record['rollback']}: {cleanup_exc}")
    for directory in sorted(touched_directories):
        try:
            _fsync_directory(directory)
        except OSError as cleanup_exc:
            cleanup_errors.append(f"{directory}: {cleanup_exc}")
    if cleanup_errors:
        raise OSError(
            "library index publish committed but durable cleanup is incomplete; "
            "recovery transaction retained: "
            + "; ".join(cleanup_errors)
        )
    try:
        _remove_if_present(marker_path)
        _fsync_directory(library_root)
    except OSError as cleanup_exc:
        raise OSError(
            "library index publish committed but transaction-marker cleanup "
            f"is incomplete: {cleanup_exc}"
        ) from cleanup_exc


def replace_library_indexes(
    repository_root: Path,
    payloads: Mapping[str, Mapping[str, Any]],
) -> None:
    """Atomically replace all three derived indexes with one exact generation."""

    library_root = repository_root / "library"
    if set(payloads) != set(LIBRARY_INDEX_IDENTITIES):
        raise ReportingError(
            "library index replacement requires the complete canonical target set"
        )
    normalized = {
        kind: _validated_library_payload(kind, payloads[kind])
        for kind in LIBRARY_INDEX_IDENTITIES
    }
    with _library_writer_lock(library_root):
        _publish_library_indexes_locked(library_root, normalized)


def rebuild_library_indexes(
    repository_root: Path,
    states: Sequence[RunState],
) -> None:
    """Replace the shared indexes from an explicitly validated state set."""

    replace_library_indexes(repository_root, library_index_payloads(states))


def _refresh_library_indexes(
    repository_root: Path,
    state: RunState,
) -> bool:
    """Merge authenticated projections from one replay into shared indexes.

    The function returns ``False`` without opening the writer lock when the
    replay contains no new publishable projection. Existing unrelated entries
    are retained, while conflicting content under one scientific identity
    fails closed. Every changed refresh publishes all three indexes as one
    durable transaction.
    """

    if state.terminal_event is None:
        raise ReportingError(
            "automatic library refresh requires a durable run_terminated event"
        )
    projected = {
        kind: _validated_library_payload(kind, payload)
        for kind, payload in library_index_payloads((state,)).items()
    }
    additions = projected
    if not any(additions[kind]["entries"] for kind in LIBRARY_INDEX_IDENTITIES):
        return False
    library_root = repository_root / "library"
    with _library_writer_lock(library_root, wait=False):
        targets = _library_targets(library_root)
        existing = {
            kind: _read_library_payload(targets[kind], kind)
            for kind in LIBRARY_INDEX_IDENTITIES
        }
        merged = {
            kind: _merge_library_payload(kind, existing[kind], additions[kind])
            for kind in LIBRARY_INDEX_IDENTITIES
        }
        if all(
            canonical_json_bytes(merged[kind])
            == canonical_json_bytes(existing[kind])
            for kind in LIBRARY_INDEX_IDENTITIES
        ):
            return False
        _publish_library_indexes_locked(library_root, merged)
    return True


def refresh_library_indexes(repository_root: Path, state: RunState) -> bool:
    """Merge one terminal run's qualified seeds and strategies into indexes."""

    return _refresh_library_indexes(repository_root, state)


def _as_mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_sequence(value: object) -> Sequence[Any]:
    return value if isinstance(value, (list, tuple)) else ()


def _decimal(value: object) -> Decimal | None:
    if value is None or isinstance(value, (bool, float)):
        return None
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return result if result.is_finite() else None


def _decimal_string(value: Decimal | None) -> str | None:
    return None if value is None else normalize_decimal(value)


def _last_checkpoint(state: RunState) -> dict[str, Any]:
    event = state.events[-1]
    return {
        "sequence": event["sequence"],
        "event_id": event["event_id"],
        "event_hash": event["event_hash"],
    }


def manifest_projection(state: RunState) -> Mapping[str, Any]:
    """Return the exact frozen manifest embedded in sequence one."""

    return state.manifest


def _descriptor_metadata(manifest: Mapping[str, Any]) -> tuple[tuple[str, ...], int | None]:
    taxonomy = manifest.get("descriptor_taxonomy")
    names: list[str] = []
    cardinalities: list[int] = []
    explicit_total: int | None = None

    if isinstance(taxonomy, Mapping):
        candidate_total = taxonomy.get("total_possible_cells")
        if isinstance(candidate_total, int) and not isinstance(candidate_total, bool) and candidate_total >= 0:
            explicit_total = candidate_total
        descriptors = taxonomy.get("descriptors")
        if not isinstance(descriptors, (list, tuple)):
            descriptors = taxonomy.get("taxonomy")
        if isinstance(descriptors, (list, tuple)):
            taxonomy = descriptors
        elif isinstance(taxonomy.get("names"), (list, tuple)) and isinstance(
            taxonomy.get("values"), Mapping
        ):
            for name in taxonomy["names"]:
                values = taxonomy["values"].get(name)
                if isinstance(name, str) and isinstance(values, (list, tuple)):
                    names.append(name)
                    cardinalities.append(len(values))
            taxonomy = ()

    if isinstance(taxonomy, (list, tuple)):
        for descriptor in taxonomy:
            if not isinstance(descriptor, Mapping):
                continue
            name = descriptor.get("name")
            values = descriptor.get("values")
            if not isinstance(name, str) or not isinstance(values, (list, tuple)):
                continue
            names.append(name)
            cardinalities.append(len(values))

    total = explicit_total
    if total is None and cardinalities and len(cardinalities) == len(names):
        total = 1
        for count in cardinalities:
            total *= count
    return tuple(names), total


def _archive_entry_event_refs(state: RunState, entry: ArchiveEntry) -> tuple[str, ...]:
    work_by_id = {work.unit_id: work for work in state.work_units}
    references: set[str] = set()
    for result in state.target_results.values():
        if result.candidate_id == entry.subject_id:
            references.add(result.terminal_event_id)
            continue
        action = state.logical_actions.get(result.logical_action_id)
        work = None if action is None else work_by_id.get(action.unit_id)
        if work is not None and work.unit.get("subject_id") == entry.subject_id:
            references.add(result.terminal_event_id)
    return tuple(sorted(references))


def _entry_projection(
    state: RunState,
    entry: ArchiveEntry,
    descriptor_names: Sequence[str],
) -> dict[str, Any]:
    descriptors = (
        dict(zip(descriptor_names, entry.cell, strict=True))
        if len(descriptor_names) == len(entry.cell)
        else {"ordered_values": list(entry.cell)}
    )
    evidence = entry.evidence
    return {
        "entry_id": entry.entry_id,
        "scientific_key": entry.scientific_key,
        "subject_id": entry.subject_id,
        "descriptors": descriptors,
        "lineage_id": entry.lineage_id,
        "origin": entry.origin,
        "strategy_id": entry.strategy_id,
        "fitness": _decimal_string(entry.fitness),
        "evidence": {
            "screening_block_id": evidence.screening_block_id,
            "search_status": evidence.search_status.value,
            "verification_block_id": evidence.verification_block_id,
            "verification_status": evidence.verification_status.value,
            "score_distribution": [normalize_decimal(score) for score in evidence.scores],
            "unresolved_count": evidence.unresolved_count,
            "event_references": list(_archive_entry_event_refs(state, entry)),
        },
    }


def _panel_projections(state: RunState) -> tuple[PanelView, ...]:
    """Return the composed reducer's panels in frozen work-unit order."""

    projections: list[PanelView] = []
    for work in state.work_units:
        for plan in _as_sequence(work.unit.get("panel_plans")):
            panel_id = plan.get("panel_id") if isinstance(plan, Mapping) else None
            if not isinstance(panel_id, str) or panel_id not in state.panel_views:
                raise ReportingError("frozen panel is absent from composed RunState")
            projections.append(state.panel_views[panel_id])
    return tuple(projections)


def _panel_to_dict(panel: PanelView) -> dict[str, Any]:
    base: dict[str, Any] = {
        "panel_id": panel.panel_id,
        "target_action_id": panel.target_action_id,
        "candidate_id": panel.candidate_id,
        "execution_id": panel.execution_id,
        "target_event_id": panel.target_event_id,
        "triggered": panel.triggered,
        "invalid_member_actions": list(panel.invalid_member_actions),
        "member_validation_errors": {
            key: list(value) for key, value in panel.member_validation_errors.items()
        },
    }
    if not panel.triggered:
        base.update(
            {
                "availability": "not_triggered",
                "usable_member_ids": [] if panel.primary is None else [panel.primary.assessment_id],
                "effective_score": None if panel.primary is None else normalize_decimal(panel.primary.score),
                "primary_verdict": None if panel.primary is None else panel.primary.verdict.value,
                "primary_application": None if panel.primary is None else panel.primary.application.value,
            }
        )
        return base
    if panel.details is None:
        base.update({"availability": "no_target_observation", "effective_score": None})
        return base
    assessment = panel.details.assessment
    base.update(
        {
            "availability": assessment.availability.value,
            "usable_member_ids": list(assessment.member_ids),
            "excluded_member_ids": list(panel.details.excluded_member_ids),
            "weighted_score": _decimal_string(assessment.weighted_score),
            "weighted_rational": (
                None
                if assessment.weighted_numerator is None
                else {
                    "numerator": assessment.weighted_numerator,
                    "denominator": assessment.weighted_denominator,
                    "display_6_half_even": panel.half_even_score_display,
                }
            ),
            "score_range": (
                None
                if assessment.score_range is None
                else [normalize_decimal(item) for item in assessment.score_range]
            ),
            "counts_by_verdict": dict(panel.details.counts_by_verdict),
            "counts_by_application": dict(panel.details.counts_by_application),
            "high_vote_count": panel.details.high_vote_count,
            "objective_support_vote_count": panel.details.objective_support_vote_count,
            "objective_high_support_vote_count": panel.details.objective_high_support_vote_count,
            "objective_exceptional_support_vote_count": panel.details.objective_exceptional_support_vote_count,
            "objective_application_support_vote_count": panel.details.objective_application_support_vote_count,
            "panel_confirmed_high": assessment.panel_confirmed_high,
            "confirmed_objective_high": assessment.confirmed_objective_high,
            "panel_confirmed_exceptional": assessment.panel_confirmed_exceptional,
            "confirmed_objective_exceptional": assessment.confirmed_objective_exceptional,
            "confirmed_objective_application": assessment.confirmed_objective_application,
            "objective_failure": assessment.objective_failure,
            "objective_application": assessment.objective_application,
            "ambiguity_kind": assessment.ambiguity_kind,
        }
    )
    return base


def _panel_accounting(panels: Sequence[PanelView]) -> dict[str, Any]:
    triggered = [panel for panel in panels if panel.triggered]
    availability = Counter(
        panel.details.assessment.availability.value
        if panel.details is not None
        else "no_target_observation"
        for panel in triggered
    )
    disputed = 0
    explicit_ambiguity = 0
    mixed_disagreement = 0
    confirmed_high = 0
    confirmed_application = 0
    warnings = 0
    for panel in panels:
        if panel.primary is not None:
            warnings += len(panel.primary.warnings)
        if panel.details is None:
            continue
        assessment = panel.details.assessment
        disputed += panel.disputed_high
        explicit_ambiguity += assessment.ambiguity_kind == "explicit_semantic_ambiguity"
        mixed_disagreement += assessment.ambiguity_kind == "mixed_position_disagreement"
        confirmed_high += assessment.confirmed_objective_high
        confirmed_application += assessment.confirmed_objective_application
    return {
        "target_observations_with_panel_plans": len(panels),
        "primary_assessments_usable": sum(panel.primary is not None for panel in panels),
        "panels_triggered": len(triggered),
        "panels_not_triggered": len(panels) - len(triggered),
        "availability": dict(sorted(availability.items())),
        "degraded_quorum_count": availability.get("degraded", 0),
        "no_quorum_count": availability.get("no_quorum", 0),
        "disputed_high_count": disputed,
        "confirmed_objective_high_count": confirmed_high,
        "confirmed_objective_application_count": confirmed_application,
        "explicit_semantic_ambiguity_count": explicit_ambiguity,
        "mixed_position_disagreement_count": mixed_disagreement,
        "invalid_member_action_count": sum(len(panel.invalid_member_actions) for panel in panels),
        "judge_warning_count": warnings,
    }


def _role_accounting(state: RunState) -> dict[str, Any]:
    config = _as_mapping(state.manifest.get("normalized_config"))
    budgets = _as_mapping(state.manifest.get("physical_call_limits"))
    role_budget_keys = {
        "target": "target_attempts",
        "judge": "judge_attempts",
        "selector": "selector_attempts",
        "perturbator": "perturbator_attempts",
    }
    result: dict[str, Any] = {}
    for role in role_budget_keys:
        attempts = [
            attempt
            for attempt in state.attempts.values()
            if attempt.dispatched and attempt.intended.get("role") == role
        ]
        route_kinds = Counter(str(item.intended.get("route_kind")) for item in attempts)
        terminal_types = Counter(
            "dangling" if item.terminal_type is None else item.terminal_type for item in attempts
        )
        errors = Counter(
            str(item.terminal_event["payload"].get("error_kind"))
            for item in attempts
            if item.terminal_event is not None
            and item.terminal_type in {"model_call_failed", "model_call_outcome_unknown"}
        )
        token_sums = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        token_receipts = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        cost = Decimal(0)
        cost_receipts = 0
        latency_ms = 0
        for attempt in attempts:
            if not attempt.succeeded or attempt.terminal_event is None:
                continue
            payload = attempt.terminal_event["payload"]
            usage = _as_mapping(payload.get("usage"))
            for key in token_sums:
                value = usage.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    token_sums[key] += value
                    token_receipts[key] += 1
            parsed_cost = _decimal(payload.get("cost"))
            if parsed_cost is not None:
                cost += parsed_cost
                cost_receipts += 1
            latency = payload.get("latency_ms")
            if isinstance(latency, int) and not isinstance(latency, bool):
                latency_ms += latency
        consumed = getattr(state.consumed, role)
        remaining = getattr(state.remaining, role)
        result[role] = {
            "budget": budgets.get(role),
            "attempts_consumed": consumed,
            "attempts_remaining": remaining,
            "successful_attempts": terminal_types.get("model_call_succeeded", 0),
            "known_failures": terminal_types.get("model_call_failed", 0),
            "unknown_outcomes": terminal_types.get("model_call_outcome_unknown", 0),
            "dangling_dispatches": terminal_types.get("dangling", 0),
            "recovery_attempts": sum(item.intended.get("attempt_ordinal", 0) > 0 for item in attempts),
            "route_kind_counts": dict(sorted(route_kinds.items())),
            "fallback_attempts": route_kinds.get("fallback", 0),
            "replacement_attempts": route_kinds.get("replacement", 0),
            "exact_retry_attempts": route_kinds.get("exact_retry", 0),
            "schema_repair_attempts": route_kinds.get("schema_repair", 0),
            "format_downgrade_attempts": route_kinds.get("format_downgrade", 0),
            "error_kind_counts": dict(sorted(errors.items())),
            "tokens": {
                key: {"known_total": token_sums[key], "receipt_count": token_receipts[key]}
                for key in token_sums
            },
            "provider_reported_cost_known_total": normalize_decimal(cost),
            "cost_receipt_count": cost_receipts,
            "latency_ms_known_total": latency_ms,
        }
    return result


def _runtime_health(state: RunState) -> list[dict[str, Any]]:
    """Distinguish transport completion from usable output, per frozen route."""
    grouped: dict[str, list[Any]] = {}
    usable_judge_attempts = {member.source_attempt_id for member in state.member_assessments.values()}
    for attempt in state.attempts.values():
        if attempt.dispatched:
            grouped.setdefault(attempt.intended["role_route_key"], []).append(attempt)
    rows = []
    for key, attempts in sorted(grouped.items()):
        finishes: Counter[str] = Counter()
        errors: Counter[str] = Counter()
        empty = 0
        latencies = []
        cost = Decimal(0)
        receipts = 0
        reasoning = 0
        reasoning_receipts = 0
        for attempt in attempts:
            if attempt.terminal_event is None:
                errors["dangling_dispatch"] += 1
                continue
            payload = attempt.terminal_event["payload"]
            if not attempt.succeeded:
                errors[str(payload.get("error_kind", "unknown"))] += 1
                continue
            finishes[payload["finish_status"]] += 1
            empty += not payload["text"].strip()
            if isinstance(payload.get("latency_ms"), int):
                latencies.append(payload["latency_ms"])
            value = _decimal(payload.get("cost"))
            if value is not None:
                cost += value
                receipts += 1
            tokens = _as_mapping(payload.get("metadata")).get("reasoning_tokens")
            if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0:
                reasoning += tokens
                reasoning_receipts += 1
        first = attempts[0].intended
        rows.append({
            "role_route_key": key, "role": first["role"],
            "requested_model": first["requested_model"], "physical_dispatches": len(attempts),
            "finish_status_counts": dict(sorted(finishes.items())),
            "provider_error_counts": dict(sorted(errors.items())),
            "terminal_event_counts": dict(sorted(Counter(a.terminal_type or "dangling" for a in attempts).items())),
            "parsed_judge_results": sum(a.physical_attempt_id in usable_judge_attempts for a in attempts),
            "unusable_complete_judge_responses": sum(
                first["role"] == "judge" and a.succeeded
                and a.terminal_event["payload"]["finish_status"] == "complete"
                and a.physical_attempt_id not in usable_judge_attempts
                for a in attempts
            ),
            "empty_final_outputs": empty,
            "schema_repair_attempts": sum(a.intended["route_kind"] == "schema_repair" for a in attempts),
            "latency_ms_known_total": sum(latencies),
            "latency_ms_max": max(latencies, default=None),
            "provider_reported_cost_known_total": normalize_decimal(cost),
            "cost_receipt_count": receipts,
            "reasoning_tokens_known_total": reasoning,
            "reasoning_token_receipt_count": reasoning_receipts,
        })
    return rows


def _candidate_distributions(
    state: RunState, panels: Sequence[PanelView]
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    panel_target_ids = {panel.target_action_id for panel in panels}
    for panel in panels:
        candidate_id = panel.candidate_id or "unbound"
        item = grouped.setdefault(
            candidate_id,
            {"candidate_id": panel.candidate_id, "scores": [], "unresolved": 0, "target_event_ids": []},
        )
        score = panel.effective_score
        if score is not None:
            item["scores"].append(score)
        elif panel.triggered:
            item["unresolved"] += 1
        if panel.target_event_id is not None:
            item["target_event_ids"].append(panel.target_event_id)
    for action_id, target in state.target_results.items():
        if action_id in panel_target_ids:
            continue
        candidate_id = target.candidate_id or "unbound"
        item = grouped.setdefault(
            candidate_id,
            {"candidate_id": target.candidate_id, "scores": [], "unresolved": 0, "target_event_ids": []},
        )
        item["unresolved"] += 1
        item["target_event_ids"].append(target.terminal_event_id)
    rendered: list[dict[str, Any]] = []
    for key in sorted(grouped):
        item = grouped[key]
        rendered.append(
            {
                "candidate_id": item["candidate_id"],
                "effective_score_distribution": [
                    normalize_decimal(score) for score in item["scores"]
                ],
                "unresolved_observation_count": item["unresolved"],
                "target_event_references": sorted(set(item["target_event_ids"])),
            }
        )
    return rendered


def _qualification_arm_metrics(metrics: Any) -> dict[str, Any]:
    """Serialize one reducer-owned arm projection without reparsing role output."""

    def optional_decimal(value: Decimal | None) -> str | None:
        if value is None:
            return None
        return normalize_decimal(value)

    return {
        "arm": metrics.arm,
        "planned": metrics.planned_count,
        "terminal": metrics.terminal_count,
        "resolved": metrics.resolved_count,
        "unresolved": metrics.unresolved_count,
        "pending": metrics.pending_count,
        "applicable": metrics.applicable_count,
        "inapplicable": metrics.inapplicable_count,
        "objective_successes": metrics.objective_successes,
        "strategy_fidelity_successes": metrics.strategy_fidelity_successes,
        "exceptional": metrics.exceptional_count,
        "effective_score_distribution": [
            normalize_decimal(score) for score in metrics.effective_scores
        ],
        "effective_score_median": optional_decimal(metrics.effective_score_median),
        "applicability_rate": optional_decimal(metrics.applicability_rate),
        "conditional_success_rate": optional_decimal(
            metrics.conditional_success_rate
        ),
        "unconditional_end_to_end_success_rate": normalize_decimal(
            metrics.unconditional_end_to_end_success_rate
        ),
    }


def _qualification_purpose_gate(gate: Any) -> dict[str, Any]:
    return {
        "purpose": gate.purpose,
        "supporting_evidence_satisfied": gate.supporting_evidence_satisfied,
        "treatment_median_improves_each_control": (
            gate.treatment_median_improves_each_control
        ),
        "passed": gate.passed,
    }


def _qualification_decision(
    decision: Any | None,
    *,
    include_purpose_gate: bool,
) -> dict[str, Any] | None:
    if decision is None:
        return None
    result = {
        "status": decision.status.value,
        "all_roots_terminal": decision.all_roots_terminal,
        "all_roots_resolved": decision.all_roots_resolved,
        "strict_treatment_repeat_support": (
            decision.strict_treatment_repeat_support
        ),
        "treatment_control_advantage": decision.treatment_control_advantage,
        "exceptional_requirement": decision.exceptional_requirement,
        "arms": [
            _qualification_arm_metrics(metrics)
            for metrics in decision.arms
        ],
    }
    if include_purpose_gate:
        result["purpose_gate"] = _qualification_purpose_gate(
            decision.purpose_gate
        )
    return result


def _qualification_calls(state: RunState, unit_ids: Iterable[str | None]) -> dict[str, int]:
    selected = {unit_id for unit_id in unit_ids if isinstance(unit_id, str)}
    calls = {role: 0 for role in ("target", "judge", "selector", "perturbator")}
    for action in state.logical_actions.values():
        if action.unit_id in selected:
            calls[action.slot["role"]] += len(action.attempts)
    return calls


_STRATEGY_SUCCESS_TYPES = (
    "supports",
    "confirmed_objective_high",
    "confirmed_objective_exceptional",
    "confirmed_objective_application",
    "panel_confirmed_declared_application_transition",
)


def _strategy_success_types(child: PortfolioChildView) -> tuple[str, ...]:
    """Name only success facts already present in one reducer-owned child."""

    outcome = child.outcome
    present = {
        "supports": outcome.outcome_result == "supports",
        "confirmed_objective_high": outcome.confirmed_objective_high,
        "confirmed_objective_exceptional": (
            outcome.confirmed_objective_exceptional
        ),
        "confirmed_objective_application": outcome.confirmed_objective_application,
        "panel_confirmed_declared_application_transition": (
            outcome.panel_confirmed_declared_application_transition
        ),
    }
    return tuple(name for name in _STRATEGY_SUCCESS_TYPES if present[name])


def _strategy_outcome_partition(
    children: Sequence[PortfolioChildView],
) -> dict[str, Any]:
    """Account for a complete aligned or unaligned child partition."""

    outcome_results = Counter(
        child.outcome.outcome_result
        for child in children
        if child.outcome.outcome_result is not None
    )
    response_states = Counter(
        child.outcome.response_state
        for child in children
        if child.outcome.response_state is not None
    )
    return {
        "allocated_count": len(children),
        "terminal_count": sum(child.terminal for child in children),
        "pending_count": sum(not child.terminal for child in children),
        "dispatched_count": sum(child.outcome.dispatched for child in children),
        "resolved_count": sum(child.outcome.resolved for child in children),
        "terminal_unresolved_count": sum(
            child.terminal and not child.outcome.resolved
            for child in children
        ),
        "reviewed_count": sum(
            child.outcome.outcome_result is not None for child in children
        ),
        "credited_count": sum(
            child.outcome.aligned
            and child.outcome.resolved
            and child.outcome.outcome_result == "supports"
            for child in children
        ),
        "confirmed_objective_high_count": sum(
            child.outcome.confirmed_objective_high for child in children
        ),
        "confirmed_objective_exceptional_count": sum(
            child.outcome.confirmed_objective_exceptional for child in children
        ),
        "confirmed_objective_application_count": sum(
            child.outcome.confirmed_objective_application for child in children
        ),
        "panel_confirmed_declared_application_transition_count": sum(
            child.outcome.panel_confirmed_declared_application_transition
            for child in children
        ),
        "outcome_result_counts": dict(sorted(outcome_results.items())),
        "response_state_counts": dict(sorted(response_states.items())),
    }


def _strategy_episode_outcomes(episode: StrategyEpisodeView) -> dict[str, Any]:
    children = episode.children
    aligned = tuple(
        child for child in children if child.outcome.aligned
    )
    unaligned = tuple(
        child for child in children if not child.outcome.aligned
    )
    return {
        "aligned": _strategy_outcome_partition(aligned),
        "unaligned": _strategy_outcome_partition(unaligned),
    }


def _successful_strategy_realizations(
    episode: StrategyEpisodeView,
) -> list[dict[str, Any]]:
    """Expose every realization with reducer-owned success evidence."""

    result: list[dict[str, Any]] = []
    for child in sorted(
        episode.children,
        key=lambda item: item.child_scientific_key,
    ):
        outcome = child.outcome
        if not (
            outcome.aligned
            and outcome.resolved
            and outcome.outcome_result == "supports"
        ):
            continue
        review = child.outcome_review
        result.append(
            {
                "child_scientific_key": child.child_scientific_key,
                "work_unit_id": child.work_unit_id,
                "target_action_id": child.target_action_id,
                "execution_id": child.execution_id,
                "candidate_id": child.candidate_id,
                "candidate_scientific_key": child.candidate_scientific_key,
                "observation_id": child.observation_id,
                "observation_scientific_key": child.observation_scientific_key,
                "panel_id": child.panel_id,
                "outcome_review_item_id": (
                    None if review is None else review.review_item_id
                ),
                "success_types": list(_strategy_success_types(child)),
            }
        )
    return result


def _strategy_experiments(state: RunState) -> list[dict[str, Any]]:
    """Project strategy experiments exclusively from the composed reducer."""

    refusal_review_count = sum(
        child.outcome_review is not None
        and child.outcome_review.response_state == "refusal"
        for episode in state.strategy_episodes.values()
        for child in episode.children
    )
    experiments: list[dict[str, Any]] = []
    for template_id, view in sorted(state.strategy_qualifications.items()):
        if view.template_id != template_id:
            raise ReportingError(
                "strategy qualification map key disagrees with its public view"
            )
        source_episode = state.strategy_episodes.get(view.source_episode_id)
        if source_episode is None:
            raise ReportingError(
                "strategy qualification does not resolve to its source episode"
            )
        if source_episode.strategy_id != view.source_episode_id:
            raise ReportingError(
                "strategy episode map key disagrees with its public view"
            )
        if source_episode.diagnosis != view.template.applicability.diagnosis:
            raise ReportingError(
                "strategy template diagnosis disagrees with its source episode"
            )
        development = _qualification_decision(
            view.development_decision,
            include_purpose_gate=True,
        )
        holdout = _qualification_decision(
            view.holdout_decision,
            include_purpose_gate=False,
        )
        experiments.append(
            {
                "strategy_template_id": template_id,
                "strategy_template_scientific_key": view.template_scientific_key,
                "source_episode_id": view.source_episode_id,
                "portfolio_intent": source_episode.portfolio_intent.value,
                "purpose": view.template.purpose,
                "diagnosis": source_episode.diagnosis,
                "move": view.template.move.to_dict(),
                "predicted_transition": view.template.success.transition.to_dict(),
                "success_contract": {
                    "supports": view.template.success.supports,
                    "falsifies": view.template.success.falsifies,
                },
                "applicability": view.template.applicability.to_dict(),
                "episode_outcomes": _strategy_episode_outcomes(source_episode),
                "successful_realizations": _successful_strategy_realizations(
                    source_episode
                ),
                "tested_entry_context_ids": list(view.tested_entry_context_ids),
                "qualification_state": view.qualification_state.value,
                "development_work_unit_id": view.development_unit_id,
                "holdout_work_unit_id": view.holdout_unit_id,
                "measurement_cell": (
                    None
                    if view.development_design is None
                    else view.development_design.get("measurement_cell")
                ),
                "development": development,
                "holdout": holdout,
                "holdout_result": (
                    "not_started" if holdout is None else holdout["status"]
                ),
                "all_physical_calls": _qualification_calls(
                    state, (view.development_unit_id, view.holdout_unit_id)
                ),
                "required_structured_role_scope_keys": list(
                    view.required_structured_role_scope_keys
                ),
                "required_role_scope_keys": list(view.required_role_scope_keys),
                "evidence_event_references": list(view.evidence_event_ids),
                "observed_refusal_outcome_review_count_run_wide": refusal_review_count,
            }
        )
    return experiments


def _strategy_search_summary(state: RunState) -> dict[str, Any]:
    """Expose whether weak threads received and benefited from recovery work."""

    rounds = tuple(state.portfolio_rounds.values())
    episodes = tuple(state.strategy_episodes.values())
    recovery_rounds = tuple(
        item for item in rounds if item.portfolio_intent.value == "recover"
    )
    advance_rounds = tuple(
        item for item in rounds if item.portfolio_intent.value == "advance"
    )
    recovery_episodes = tuple(
        item for item in episodes if item.portfolio_intent.value == "recover"
    )
    work_units_by_id = {item.unit_id: item for item in state.work_units}
    recovery_strategy_capacity = 0
    for round_view in recovery_rounds:
        work_unit = work_units_by_id.get(round_view.work_unit_id)
        if work_unit is None:
            raise ReportingError(
                "recovery portfolio round lacks its reducer-owned work unit"
            )
        capacity = _as_mapping(work_unit.unit.get("design")).get(
            "maximum_strategy_count"
        )
        if (
            isinstance(capacity, bool)
            or not isinstance(capacity, int)
            or capacity < len(round_view.lanes)
        ):
            raise ReportingError(
                "recovery portfolio strategy capacity is invalid"
            )
        recovery_strategy_capacity += capacity
    recovery_selected_strategies = sum(
        len(item.lanes) for item in recovery_rounds
    )
    resolved_recovery_children = tuple(
        (episode, child)
        for episode in recovery_episodes
        for child in episode.children
        if child.outcome.aligned
        and child.outcome.resolved
        and child.outcome.effective_score is not None
    )
    recovery_score_improvements = tuple(
        (episode, child)
        for episode, child in resolved_recovery_children
        if child.outcome.effective_score > episode.parent_baseline_score
    )
    supported_recovery_children = tuple(
        child
        for episode in recovery_episodes
        for child in episode.children
        if child.outcome.aligned
        and child.outcome.resolved
        and child.outcome.outcome_result == "supports"
    )
    salvaged_episode_ids = {
        episode.strategy_id
        for episode in recovery_episodes
        if any(
            child.outcome.aligned
            and child.outcome.resolved
            and child.outcome.outcome_result == "supports"
            for child in episode.children
        )
    }
    supported_candidate_ids = {
        child.candidate_id for child in supported_recovery_children
    }
    recovery_to_advance = 0
    for round_view in advance_rounds:
        node = state.conversation_nodes.get(round_view.generation_parent_node_id)
        if node is not None and node.candidate_id in supported_candidate_ids:
            recovery_to_advance += 1
    recovery_episode_ids = {
        episode.strategy_id for episode in recovery_episodes
    }
    verified_recovery_scenarios = sum(
        scenario.origin == "derived"
        and bool(recovery_episode_ids.intersection(scenario.source_step_episode_ids))
        and state.scenario_evidence.get(scenario.scenario_id) is not None
        and state.scenario_evidence[
            scenario.scenario_id
        ].verification_status.value
        == "verified"
        for scenario in state.scenarios.values()
    )
    return {
        "portfolio_rounds": len(rounds),
        "advance_portfolio_rounds": len(advance_rounds),
        "recovery_portfolio_rounds": len(recovery_rounds),
        "empty_recovery_portfolio_rounds": sum(
            not item.lanes for item in recovery_rounds
        ),
        # The parser guarantees that every accepted recovery lane has a
        # different diagnosis/anchor/argument-move/target-kind approach key.
        "recovery_distinct_approaches": recovery_selected_strategies,
        "recovery_strategy_capacity": recovery_strategy_capacity,
        "recovery_unused_strategy_slots": (
            recovery_strategy_capacity - recovery_selected_strategies
        ),
        "recovery_multi_strategy_portfolio_rounds": sum(
            len(item.lanes) > 1 for item in recovery_rounds
        ),
        "recovery_strategy_episodes": len(recovery_episodes),
        "recovery_resolved_children": len(resolved_recovery_children),
        "recovery_supported_children": len(supported_recovery_children),
        "recovery_score_improved_children": len(recovery_score_improvements),
        "recovery_supported_score_improved_children": sum(
            child.outcome.outcome_result == "supports"
            for _, child in recovery_score_improvements
        ),
        "best_recovery_score_delta": (
            None
            if not resolved_recovery_children
            else normalize_decimal(
                max(
                    child.outcome.effective_score - episode.parent_baseline_score
                    for episode, child in resolved_recovery_children
                )
            )
        ),
        "salvaged_recovery_episodes": len(salvaged_episode_ids),
        "recovery_to_advance_rounds": recovery_to_advance,
        "verified_recovery_scenarios": verified_recovery_scenarios,
    }


def _snapshot_count(value: object) -> int:
    if isinstance(value, (list, tuple)):
        return len(value)
    if not isinstance(value, Mapping) or not value:
        return 0
    for key in ("items", "entries", "seeds", "templates", "selected", "probes"):
        candidate = value.get(key)
        if isinstance(candidate, (list, tuple)):
            return len(candidate)
    return 1


def _qualification_summary(state: RunState) -> dict[str, Any]:
    objective = _as_mapping(state.manifest.get("objective"))
    return {
        "objective_id": objective.get("id"),
        "objective_plugin_digest": objective.get("plugin_digest"),
    }


def archive_projection(
    state: RunState,
) -> Mapping[str, Any]:
    """Render the composed reducer's current one-archive projection."""

    archive = state.archive
    config = _as_mapping(state.manifest.get("normalized_config"))
    search = _as_mapping(config.get("search"))
    expected_capacity = search.get("archive_capacity_per_cell")
    if expected_capacity is not None and archive.capacity_per_cell != expected_capacity:
        raise ReportingError("archive capacity differs from the frozen run configuration")
    descriptor_names, declared_total = _descriptor_metadata(state.manifest)
    total_known = declared_total is not None
    metric_total = max(declared_total or 0, archive.occupied_cell_count)
    try:
        metric = state.archive_metric(total_possible_cells=metric_total)
        verified_cells = frozenset(state.verified_archive_cells())
        demonstrated_subjects = state.demonstrated_repeat_supported_subject_ids()
    except DomainError as exc:
        raise ReportingError(f"archive evidence projection is invalid: {exc}") from exc

    cells: list[dict[str, Any]] = []
    demonstrated_cells = 0
    ambiguity_cells = 0
    status_cell_counts = Counter()
    for cell in sorted(archive.cells):
        entries = archive.cells[cell]
        verified = cell in verified_cells
        demonstrated = bool(
            verified
            and any(
                entry.evidence.verification_status.value == "verified"
                and entry.subject_id in demonstrated_subjects
                for entry in entries
            )
        )
        demonstrated_cells += demonstrated
        ambiguity_cells += verified and not demonstrated
        for status in SearchStatusValues:
            status_cell_counts[status] += any(
                entry.evidence.search_status.value == status for entry in entries
            )
        for status in ("failed", "inconclusive"):
            status_cell_counts[status] += any(
                entry.evidence.verification_status.value == status for entry in entries
            )
        descriptors = (
            dict(zip(descriptor_names, cell, strict=True))
            if len(descriptor_names) == len(cell)
            else {"ordered_values": list(cell)}
        )
        cells.append(
            {
                "descriptors": descriptors,
                "verified": verified,
                "verified_evidence_kind": (
                    "demonstrated_only_supported"
                    if demonstrated
                    else "ambiguity_dependent"
                    if verified
                    else None
                ),
                "entries": [
                    _entry_projection(state, entry, descriptor_names) for entry in entries
                ],
            }
        )
    return {
        "schema_revision": REPORT_SCHEMA_REVISION,
        "run_id": state.run_id,
        "scientific_scope_digest": state.scientific_scope_digest,
        "source_checkpoint": _last_checkpoint(state),
        "projection_complete": True,
        "descriptor_taxonomy": state.manifest.get("descriptor_taxonomy"),
        "capacity_per_cell": archive.capacity_per_cell,
        "metrics": {
            **metric.to_dict(),
            "total_possible_cells": declared_total,
            "total_possible_cells_known": total_known,
            "demonstrated_only_supported_verified_cells": demonstrated_cells,
            "ambiguity_dependent_verified_cells": ambiguity_cells,
            "promising_cells": status_cell_counts.get("promising", 0),
            "exploratory_lead_cells": status_cell_counts.get("exploratory_lead", 0),
            "observed_cells": status_cell_counts.get("observed", 0),
            "failed_cells": status_cell_counts.get("failed", 0),
            "inconclusive_cells": status_cell_counts.get("inconclusive", 0),
        },
        "cells": cells,
    }


SearchStatusValues = ("observed", "exploratory_lead", "promising")


def _newly_verified_counts(state: RunState) -> dict[str, int]:
    entry = 0
    scenario = 0
    for snapshot in state.current_seed_qualification_snapshots:
        subject_kind = snapshot.get("type")
        if subject_kind == "entry_seed":
            entry += 1
        elif subject_kind == "scenario_recipe":
            scenario += 1
        else:
            raise ReportingError(
                "current seed qualification has an unknown closed subject type"
            )
    return {"entry_seeds": entry, "scenario_recipes": scenario}


def _caveats(
    state: RunState,
    archive_view: Mapping[str, Any],
    panel_accounting: Mapping[str, Any],
    calls: Mapping[str, Any],
) -> list[str]:
    caveats: list[str] = []
    if not archive_view.get("projection_complete"):
        caveats.append("No reducer-supplied archive projection was available; the archive view is conservatively empty.")
    metrics = _as_mapping(archive_view.get("metrics"))
    if not metrics.get("total_possible_cells_known"):
        caveats.append("The manifest did not expose a usable total descriptor-cell count.")
    if state.target_identity_unverifiable:
        caveats.append("The provider exposed no immutable target identity; revision drift could not be detected.")
    if state.target_identity_drift:
        caveats.append("Effective target identity drift was observed; mismatching evidence is not a reproducible aggregate.")
    role_collisions = state.role_target_identity_collision_attempt_ids
    if any(role_collisions.get(role, ()) for role in ("selector", "perturbator")):
        caveats.append(
            "Selector or Perturbator responses collided with the pinned target identity; "
            "those receipts were retained but excluded from semantic and qualification evidence."
        )
    role_unverifiable = state.role_identity_unverifiable_attempt_ids
    if any(role_unverifiable.get(role, ()) for role in ("selector", "perturbator")):
        caveats.append(
            "Some Selector or Perturbator responses exposed no immutable model identity; "
            "cross-role identity independence could not be verified for those receipts."
        )
    if state.dangling_attempt_ids:
        caveats.append("The replay contains dangling dispatches awaiting bounded resume closure.")
    if panel_accounting.get("no_quorum_count", 0):
        caveats.append("Some required panels lacked quorum; those observations are operationally unresolved.")
    if panel_accounting.get("degraded_quorum_count", 0):
        caveats.append("Some assessments used degraded two-member quorum and retain primary-weight bias.")
    if panel_accounting.get("mixed_position_disagreement_count", 0):
        caveats.append("Mixed judge disagreement is reported separately from explicit semantic ambiguity.")
    if any(_as_mapping(value).get("fallback_attempts", 0) for value in calls.values()):
        caveats.append("At least one fallback route was used; exact route identity remains recorded.")
    if any(_as_mapping(value).get("replacement_attempts", 0) for value in calls.values()):
        caveats.append("At least one judge replacement route was used; its assumed logical slot remains recorded.")
    if state.terminal_event is None:
        caveats.append("The run is active; all derived counts are a checkpoint, not a terminal result.")
    caveats.extend(
        [
            "Multiple judges assess one target response and are not independent target replications.",
            "The strict >60% rule is an operational gate, not proof of universal susceptibility.",
            "A recorded transcript is evidence, not a freshly reproducible scenario by itself.",
            "No transfer is claimed across targets, model profiles, objective digests, or descriptor grids.",
        ]
    )
    return sorted(set(caveats))


def _commitment_summary(state: RunState) -> dict[str, Any]:
    """Report root-distinct progress, with journal evidence for every commitment."""
    from .run_state import portfolio_branch_closed, portfolio_node_progress, verified_commitment_units

    opened: set[str] = set()
    completed: dict[str, Any] = {}
    stalled = 0
    depth_limited = 0
    config = state.manifest["normalized_config"]
    search = config.get("search", {})
    nodes = getattr(state, "conversation_nodes", {})
    for node in nodes.values():
        rank, _ = portfolio_node_progress(state, node)
        if rank >= 2:
            opened.add(node.root_id)
        if rank == 5:
            children = [child for episode in state.strategy_episodes.values() for child in episode.children if child.observation_id == node.target_observation_id]
            if not children or any(child.outcome.outcome_result == "supports" and child.outcome.aligned and child.outcome.resolved for child in children):
                old = completed.get(node.root_id)
                if old is None or node.depth < old.depth:
                    completed[node.root_id] = node
        elif search and node.depth:
            if node.depth >= search["maximum_followups"]:
                depth_limited += 1
            elif portfolio_branch_closed(state, node, maximum_followups=search["maximum_followups"], maximum_stalled_followups=search["maximum_stalled_followups"]):
                stalled += 1
    verified = verified_commitment_units(state) if nodes else ()
    completions = []
    first_sequence = None
    for root, node in sorted(completed.items()):
        chain = []
        current = node
        while True:
            observation = next(item for item in state.target_observations.values() if item.observation_id == current.target_observation_id)
            attempt = state.attempts[observation.successful_attempt_id]
            chain.append({
                "target_event_id": attempt.terminal_event["event_id"],
                "user_turn": attempt.intended["messages"][-1]["content"],
                "response": observation.response,
            })
            if current.node_id == node.node_id:
                sequence = attempt.terminal_event["sequence"]
                first_sequence = sequence if first_sequence is None else min(first_sequence, sequence)
            if current.parent_node_id is None:
                break
            current = nodes[current.parent_node_id]
        completions.append({"conversation_root_id": root, "terminal_node_id": node.node_id, "chain": list(reversed(chain))})
    def target_calls_through(sequence: int | None) -> int | None:
        if sequence is None:
            return None
        return sum(attempt.dispatch_event is not None and attempt.dispatch_event["payload"]["charged_role"] == "target" and attempt.dispatch_event["sequence"] <= sequence for attempt in state.attempts.values())
    verification_sequence = min((max(attempt.terminal_event["sequence"] for action in unit.actions for attempt in action.attempts if attempt.terminal_event is not None) for unit in verified), default=None)
    return {
        "owned_opening_roots": len(opened),
        "completed_commitment_roots": len(completed),
        "verified_commitment_scenarios": len({unit.unit["subject_scientific_key"] for unit in verified}),
        "target_attempts_to_first_commitment": target_calls_through(first_sequence),
        "target_attempts_to_first_verified_commitment": target_calls_through(verification_sequence),
        "stalled_nodes": stalled,
        "depth_limited_nodes": depth_limited,
        "completions": completions,
    }


def _funded_experiment_summary(state: RunState) -> dict[str, Any]:
    from .run_state import funded_work_plan, remaining_work_allocations
    config = state.manifest["normalized_config"]
    if "experiment" not in config:
        return {}
    plan = funded_work_plan(config, state.manifest["bootstrap_snapshot"]["probes"], state.manifest["seed_snapshot"])
    remaining = remaining_work_allocations(state)
    return {kind: {"allocated": count, "started": count - remaining[kind], "unused": remaining[kind],
                   "maximum_calls_per_unit": maximum.to_dict()} for kind, (count, maximum) in plan.items()}


def summary_projection(
    state: RunState,
    archive_view: Mapping[str, Any],
) -> Mapping[str, Any]:
    panel_views = _panel_projections(state)
    panel_accounting = _panel_accounting(panel_views)
    calls = _role_accounting(state)
    config = _as_mapping(state.manifest.get("normalized_config"))
    target_profile = _as_mapping(_as_mapping(config.get("models")).get("target"))
    qualification = _qualification_summary(state)
    metrics = _as_mapping(archive_view.get("metrics"))
    priors = {
        "verified_seed_entries_loaded": _snapshot_count(state.manifest.get("seed_snapshot")),
        "qualified_strategy_templates_loaded": _snapshot_count(
            state.manifest.get("strategy_template_snapshot")
        ),
        "bootstrap_entries_loaded": _snapshot_count(state.manifest.get("bootstrap_snapshot")),
    }
    newly_verified = _newly_verified_counts(state)
    strategy_experiments = _strategy_experiments(state)
    strategy_search = _strategy_search_summary(state)
    role_collisions = state.role_target_identity_collision_attempt_ids
    role_unverifiable = state.role_identity_unverifiable_attempt_ids
    terminal = (
        {
            "state": "active",
            "reason": None,
            "resumable": True,
            "detail": "No run_terminated event is present at this checkpoint.",
        }
        if state.terminal_event is None
        else {
            "state": "terminated",
            "reason": state.terminal_event["payload"]["reason"],
            "resumable": state.terminal_event["payload"]["resumable"],
            "detail": state.terminal_event["payload"]["detail"],
        }
    )
    caveats = _caveats(state, archive_view, panel_accounting, calls)
    reported_successes = [
        {
            "descriptors": cell.get("descriptors"),
            "verified_evidence_kind": cell.get("verified_evidence_kind"),
            "subjects": [
                entry.get("subject_id")
                for entry in _as_sequence(cell.get("entries"))
                if isinstance(entry, Mapping)
                and _as_mapping(entry.get("evidence")).get("verification_status") == "verified"
            ],
            "event_references": sorted(
                {
                    event_id
                    for entry in _as_sequence(cell.get("entries"))
                    if isinstance(entry, Mapping)
                    for event_id in _as_sequence(_as_mapping(entry.get("evidence")).get("event_references"))
                    if isinstance(event_id, str)
                }
            ),
        }
        for cell in _as_sequence(archive_view.get("cells"))
        if isinstance(cell, Mapping) and cell.get("verified")
    ]
    return {
        "schema_revision": REPORT_SCHEMA_REVISION,
        "run_id": state.run_id,
        "scientific_scope_digest": state.scientific_scope_digest,
        "source_checkpoint": _last_checkpoint(state),
        **({"monetary_budget": {
            "ceiling_usd": config["provider"]["max_cost_usd"],
            "charged_plus_reserved_usd": normalize_decimal(state.cost_liability_usd),
        }} if config.get("provider", {}).get("max_cost_usd") is not None else {}),
        "scenario_preflights": dict(state.scenario_preflights),
        "headline": {
            "objective": qualification["objective_id"] or config.get("objective"),
            "objective_plugin_digest": qualification["objective_plugin_digest"],
            "target_profile": dict(target_profile),
            "effective_target_identity": state.target_identity,
            "identity_unverifiable": state.target_identity_unverifiable,
            "physical_target_call_budget": _as_mapping(state.manifest.get("physical_call_limits")).get("target"),
            "physical_target_attempts_consumed": state.all_target_attempts,
            "completed_target_responses": state.completed_target_responses,
            "verified_distinct_cells": metrics.get("verified_cell_count", 0),
            "total_possible_cells": metrics.get("total_possible_cells"),
            "verified_cells_per_physical_target_attempt": metrics.get("rate_display"),
            "demonstrated_only_supported_verified_cells": metrics.get(
                "demonstrated_only_supported_verified_cells", 0
            ),
            "ambiguity_dependent_verified_cells": metrics.get(
                "ambiguity_dependent_verified_cells", 0
            ),
            "newly_verified": newly_verified,
            "compatible_priors_loaded": priors,
            "terminal": terminal,
        },
        "archive_status": {
            key: metrics.get(key)
            for key in (
                "occupied_cell_count",
                "promising_cells",
                "exploratory_lead_cells",
                "observed_cells",
                "failed_cells",
                "inconclusive_cells",
                "verified_cell_count",
            )
        },
        "candidate_score_distributions": _candidate_distributions(state, panel_views),
        "panel_accounting": panel_accounting,
        "panels": [_panel_to_dict(panel) for panel in panel_views],
        "role_accounting": calls,
        "runtime_health": _runtime_health(state),
        "funded_experiment": _funded_experiment_summary(state),
        "role_identity_integrity": {
            role: {
                "target_collision_attempt_ids": list(
                    role_collisions.get(role, ())
                ),
                "identity_unverifiable_attempt_ids": list(
                    role_unverifiable.get(role, ())
                ),
            }
            for role in ("selector", "perturbator")
        },
        "sampling_work_units": [
            {
                "unit_id": work.unit_id,
                "kind": work.unit.get("kind"),
                "subject_id": work.unit.get("subject_id"),
                "design": work.unit.get("design"),
                "worst_case_calls": work.unit.get("worst_case_calls"),
            }
            for work in state.work_units
            if work.unit.get("kind")
            in {"screening", "verification", "strategy_development", "strategy_holdout"}
        ],
        "strategy_experiments": strategy_experiments,
        "strategy_search": strategy_search,
        "commitment_progress": _commitment_summary(state),
        "judge_procedure": {
            "capped_assessments": sum("procedure_score_capped" in member.warnings for member in state.member_assessments.values()),
            "conditional_assessments": sum(member.ownership == "conditional" for member in state.member_assessments.values()),
            "assessments": [{"source_attempt_id": member.source_attempt_id, "reported_score": _decimal_string(member.reported_score), "effective_score": _decimal_string(member.score), "ownership": member.ownership, "procedure_ceiling": _decimal_string(member.procedure_ceiling), "procedure": member.procedure.to_dict()} for member in state.member_assessments.values()],
        },
        "reported_verified_successes": reported_successes,
        "refusal_and_context": {
            "observed_refusal_outcome_reviews": max(
                [
                    experiment.get("observed_refusal_outcome_review_count_run_wide", 0)
                    for experiment in strategy_experiments
                ],
                default=0,
            ),
            "refusal_dominated_strategy_steps": sum(
                episode.refusal.refusal_dominated
                for episode in state.strategy_episodes.values()
            ),
            "target_context_limit_attempts": _as_mapping(
                _as_mapping(calls.get("target")).get("error_kind_counts")
            ).get("context_limit", 0),
        },
        "qualification": qualification,
        "caveats": caveats,
    }


def _markdown_escape(value: object) -> str:
    if value is None:
        return "not available"
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_summary_markdown(summary: Mapping[str, Any]) -> str:
    """Render a concise report that leads with the scientific endpoint."""

    headline = _as_mapping(summary.get("headline"))
    archive = _as_mapping(summary.get("archive_status"))
    terminal = _as_mapping(headline.get("terminal"))
    newly = _as_mapping(headline.get("newly_verified"))
    priors = _as_mapping(headline.get("compatible_priors_loaded"))
    strategy_search = _as_mapping(summary.get("strategy_search"))
    commitment = _as_mapping(summary.get("commitment_progress"))
    lines = [
        "# Quasilisk run summary",
        "",
        f"- Objective: `{_markdown_escape(headline.get('objective'))}`",
        f"- Target profile: `{_markdown_escape(_as_mapping(headline.get('target_profile')).get('model'))}`",
        f"- Physical target attempts: **{_markdown_escape(headline.get('physical_target_attempts_consumed'))} / {_markdown_escape(headline.get('physical_target_call_budget'))}**",
        f"- Completed target responses: **{_markdown_escape(headline.get('completed_target_responses'))}**",
        f"- Owned openings / completed commitments / verified commitment scenarios: **{_markdown_escape(commitment.get('owned_opening_roots'))} / {_markdown_escape(commitment.get('completed_commitment_roots'))} / {_markdown_escape(commitment.get('verified_commitment_scenarios'))}**",
        f"- Target attempts to first commitment / first verified commitment: **{_markdown_escape(commitment.get('target_attempts_to_first_commitment'))} / {_markdown_escape(commitment.get('target_attempts_to_first_verified_commitment'))}**",
        f"- Verified distinct cells: **{_markdown_escape(headline.get('verified_distinct_cells'))} / {_markdown_escape(headline.get('total_possible_cells'))}**",
        f"- Verified cells per physical target attempt: `{_markdown_escape(headline.get('verified_cells_per_physical_target_attempt'))}`",
        f"- Demonstrated-only supported cells: **{_markdown_escape(headline.get('demonstrated_only_supported_verified_cells'))}**",
        f"- Ambiguity-dependent verified cells: **{_markdown_escape(headline.get('ambiguity_dependent_verified_cells'))}**",
        f"- Newly verified entry seeds / scenario recipes: **{_markdown_escape(newly.get('entry_seeds'))} / {_markdown_escape(newly.get('scenario_recipes'))}**",
        f"- Compatible seed / strategy priors loaded: **{_markdown_escape(priors.get('verified_seed_entries_loaded'))} / {_markdown_escape(priors.get('qualified_strategy_templates_loaded'))}**",
        f"- Recovery portfolios / salvaged episodes / verified recovery scenarios: **{_markdown_escape(strategy_search.get('recovery_portfolio_rounds'))} / {_markdown_escape(strategy_search.get('salvaged_recovery_episodes'))} / {_markdown_escape(strategy_search.get('verified_recovery_scenarios'))}**",
        f"- Distinct recovery approaches / available strategy slots / unused slots: **{_markdown_escape(strategy_search.get('recovery_distinct_approaches'))} / {_markdown_escape(strategy_search.get('recovery_strategy_capacity'))} / {_markdown_escape(strategy_search.get('recovery_unused_strategy_slots'))}**",
        f"- Recovery children beating baseline / supported and beating baseline / best score delta: **{_markdown_escape(strategy_search.get('recovery_score_improved_children'))} / {_markdown_escape(strategy_search.get('recovery_supported_score_improved_children'))} / {_markdown_escape(strategy_search.get('best_recovery_score_delta'))}**",
        f"- Operational terminal state: `{_markdown_escape(terminal.get('state'))}` (`{_markdown_escape(terminal.get('reason'))}`)",
        "",
        "Operational completion is not scientific success.",
        "",
        "## Archive evidence",
        "",
        "| Occupied | Observed | Leads | Promising | Failed | Inconclusive | Verified |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        "| "
        + " | ".join(
            _markdown_escape(archive.get(key))
            for key in (
                "occupied_cell_count",
                "observed_cells",
                "exploratory_lead_cells",
                "promising_cells",
                "failed_cells",
                "inconclusive_cells",
                "verified_cell_count",
            )
        )
        + " |",
        "",
        "## Model-call accounting",
        "",
        "| Role | Attempts | Remaining | Success | Known failure | Unknown | Recovery | Fallback | Cost known |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for role in ("target", "judge", "selector", "perturbator"):
        item = _as_mapping(_as_mapping(summary.get("role_accounting")).get(role))
        lines.append(
            "| "
            + " | ".join(
                _markdown_escape(value)
                for value in (
                    role,
                    item.get("attempts_consumed"),
                    item.get("attempts_remaining"),
                    item.get("successful_attempts"),
                    item.get("known_failures"),
                    item.get("unknown_outcomes"),
                    item.get("recovery_attempts"),
                    item.get("fallback_attempts"),
                    item.get("provider_reported_cost_known_total"),
                )
            )
            + " |"
        )
    preflights = _as_mapping(summary.get("scenario_preflights"))
    if preflights:
        lines.extend(["", "## Source template preflight", "",
                      "These checks reuse recorded parents. Rejection is template attrition, not target resistance.", "",
                      "| Verification unit | Status | Selector calls | Fresh target calls | Reason |",
                      "|---|---|---:|---:|---|"])
        for unit_id, item in sorted(preflights.items()):
            reason = next((step.get("reason") for step in item["steps"]
                           if step["status"] not in {"applicable", "not_started"}), None)
            lines.append("| " + " | ".join(_markdown_escape(value) for value in (
                unit_id, item["status"], item["physical_selector_calls"], item["fresh_target_calls"], reason)) + " |")
    identity = _as_mapping(summary.get("role_identity_integrity"))
    lines.extend(
        [
            "",
            "## Selector and Perturbator identity integrity",
            "",
            "| Role | Target-collision attempt IDs | Identity-unverifiable attempt IDs |",
            "|---|---|---|",
        ]
    )
    for role in ("selector", "perturbator"):
        item = _as_mapping(identity.get(role))
        collision_ids = tuple(
            value
            for value in _as_sequence(item.get("target_collision_attempt_ids"))
            if isinstance(value, str)
        )
        unverifiable_ids = tuple(
            value
            for value in _as_sequence(
                item.get("identity_unverifiable_attempt_ids")
            )
            if isinstance(value, str)
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    role,
                    _markdown_escape(", ".join(collision_ids) or "none"),
                    _markdown_escape(", ".join(unverifiable_ids) or "none"),
                )
            )
            + " |"
        )
    panel = _as_mapping(summary.get("panel_accounting"))
    lines.extend(
        [
            "",
            "## Judge evidence",
            "",
            f"Triggered panels: {panel.get('panels_triggered', 0)}; degraded quorum: {panel.get('degraded_quorum_count', 0)}; no quorum: {panel.get('no_quorum_count', 0)}; disputed highs: {panel.get('disputed_high_count', 0)}.",
            f"Explicit semantic ambiguity: {panel.get('explicit_semantic_ambiguity_count', 0)}; mixed-position disagreement: {panel.get('mixed_position_disagreement_count', 0)}.",
            "",
            "## Caveats",
            "",
        ]
    )
    caveats = _as_sequence(summary.get("caveats"))
    lines.extend(f"- {_markdown_escape(caveat)}" for caveat in caveats)
    successes = _as_sequence(summary.get("reported_verified_successes"))
    lines.extend(["", "## Verified evidence references", ""])
    if not successes:
        lines.append("No verified archive entry is present in this projection.")
    else:
        for item in successes:
            if not isinstance(item, Mapping):
                continue
            lines.append(
                f"- `{_markdown_escape(item.get('descriptors'))}` — {_markdown_escape(item.get('verified_evidence_kind'))}; events: {_markdown_escape(', '.join(item.get('event_references', [])))}"
            )
    return "\n".join(lines).rstrip() + "\n"


def build_report_bundle(
    state: RunState,
) -> ReportBundle:
    """Derive all public views from one validated replay checkpoint."""

    if not isinstance(state, RunState):
        raise ReportingError("reporting requires a replayed RunState")
    archive = archive_projection(state)
    summary = summary_projection(state, archive)
    return ReportBundle(
        manifest=manifest_projection(state),
        archive=archive,
        summary=summary,
        markdown=render_summary_markdown(summary),
    )


def _stage_file(
    directory: Path,
    filename: str,
    data: bytes,
    *,
    purpose: str = "new",
) -> Path:
    descriptor: int | None = None
    path: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(
            dir=directory, prefix=f".{filename}.{purpose}.", suffix=".tmp"
        )
        path = Path(raw_path)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        return path
    except OSError as exc:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise ReportingError(f"cannot stage derived artifact {filename}: {exc}") from exc


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(
        directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_transients(paths: Iterable[Path]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            errors.append(f"{path.name}: {exc}")
    return errors


def _restore_derived_artifacts(
    directory: Path,
    previous: Mapping[str, bytes | None],
    rollback_paths: dict[str, Path],
) -> list[str]:
    """Restore every public path to its exact pre-transaction contents."""

    errors: list[str] = []
    for filename in DERIVED_FILENAMES:
        target = directory / filename
        prior = previous[filename]
        try:
            if prior is None:
                target.unlink(missing_ok=True)
            else:
                rollback_path = rollback_paths.get(filename)
                if rollback_path is None or not rollback_path.exists():
                    rollback_path = _stage_file(
                        directory, filename, prior, purpose="restore"
                    )
                    rollback_paths[filename] = rollback_path
                os.replace(rollback_path, target)
        except (OSError, ReportingError) as exc:
            errors.append(f"{filename}: {exc}")
    return errors


def _rollback_derived_artifact_transaction(
    directory: Path,
    previous: Mapping[str, bytes | None],
    new_paths: Mapping[str, Path],
    rollback_paths: dict[str, Path],
) -> list[str]:
    errors = _restore_derived_artifacts(directory, previous, rollback_paths)
    errors.extend(
        _remove_transients((*new_paths.values(), *rollback_paths.values()))
    )
    try:
        _fsync_directory(directory)
    except OSError as exc:
        errors.append(f"directory fsync: {exc}")
    return errors


def write_derived_artifacts(
    run_directory: str | Path,
    state: RunState,
) -> DerivedArtifactPaths:
    """Replace one complete derived checkpoint, restoring the prior one on error.

    Scientific evidence is untouched on failure.  The operation assumes the
    caller owns the run's single-writer boundary; it creates no event.  Every
    new image and every required rollback image is durable before the first
    public path changes.
    """

    directory = Path(run_directory)
    if not directory.is_dir():
        raise ReportingError("run directory does not exist or is not a directory")
    bundle = build_report_bundle(state)
    payloads = {
        "manifest.json": canonical_json_bytes(bundle.manifest) + b"\n",
        "archive.json": canonical_json_bytes(bundle.archive) + b"\n",
        "summary.json": canonical_json_bytes(bundle.summary) + b"\n",
        "summary.md": bundle.markdown.encode("utf-8"),
    }
    previous: dict[str, bytes | None] = {}
    for filename in DERIVED_FILENAMES:
        target = directory / filename
        try:
            previous[filename] = target.read_bytes()
        except FileNotFoundError:
            previous[filename] = None
        except OSError as exc:
            raise ReportingError(
                f"cannot snapshot derived artifact {filename}: {exc}"
            ) from exc

    new_paths: dict[str, Path] = {}
    rollback_paths: dict[str, Path] = {}
    try:
        for filename in DERIVED_FILENAMES:
            new_paths[filename] = _stage_file(
                directory, filename, payloads[filename], purpose="new"
            )
        for filename in DERIVED_FILENAMES:
            prior = previous[filename]
            if prior is not None:
                rollback_paths[filename] = _stage_file(
                    directory, filename, prior, purpose="rollback"
                )
        _fsync_directory(directory)
    except (OSError, ReportingError) as exc:
        cleanup_errors = _remove_transients(
            (*new_paths.values(), *rollback_paths.values())
        )
        try:
            _fsync_directory(directory)
        except OSError as cleanup_exc:
            cleanup_errors.append(f"directory fsync: {cleanup_exc}")
        detail = f"cannot prepare derived artifact transaction: {exc}"
        if cleanup_errors:
            detail += "; transient cleanup failed: " + "; ".join(cleanup_errors)
        raise ReportingError(detail) from exc

    try:
        for filename in DERIVED_FILENAMES:
            os.replace(new_paths[filename], directory / filename)
        _fsync_directory(directory)
    except OSError as exc:
        restore_errors = _rollback_derived_artifact_transaction(
            directory, previous, new_paths, rollback_paths
        )
        if restore_errors:
            raise ReportingError(
                "derived artifact replacement and rollback failed: "
                + "; ".join(restore_errors)
            ) from exc
        raise ReportingError(
            f"derived artifact replacement failed; prior artifacts restored: {exc}"
        ) from exc

    # The new checkpoint is durable.  Rollback images are now unnecessary;
    # their cleanup cannot turn the complete public checkpoint into a mixture.
    cleanup_errors = _remove_transients(
        (*new_paths.values(), *rollback_paths.values())
    )
    cleanup_failure: OSError | None = None
    if not cleanup_errors:
        try:
            _fsync_directory(directory)
        except OSError as exc:
            cleanup_failure = exc
    if cleanup_errors or cleanup_failure is not None:
        failure_detail = (
            "; ".join(cleanup_errors)
            if cleanup_errors
            else f"directory fsync: {cleanup_failure}"
        )
        restore_errors = _rollback_derived_artifact_transaction(
            directory, previous, new_paths, rollback_paths
        )
        if restore_errors:
            raise ReportingError(
                "derived artifact replacement and rollback failed: "
                + "; ".join(restore_errors)
            ) from cleanup_failure
        raise ReportingError(
            "derived artifact replacement failed; prior artifacts restored: "
            + failure_detail
        ) from cleanup_failure
    return DerivedArtifactPaths(
        manifest=directory / "manifest.json",
        archive=directory / "archive.json",
        summary_json=directory / "summary.json",
        summary_markdown=directory / "summary.md",
    )


def write_derived_reports(run_directory: Path, state: RunState) -> None:
    """Regenerate public reports solely from the composed replay state."""

    write_derived_artifacts(run_directory, state)


# Short public aliases for callers and maintenance commands.
build_reports = build_report_bundle
write_reports = write_derived_artifacts


__all__ = [
    "DERIVED_FILENAMES",
    "DerivedArtifactPaths",
    "REPORT_SCHEMA_REVISION",
    "ReportBundle",
    "ReportingError",
    "archive_projection",
    "build_report_bundle",
    "build_reports",
    "library_index_payloads",
    "manifest_projection",
    "rebuild_library_indexes",
    "refresh_library_indexes",
    "render_summary_markdown",
    "replace_library_indexes",
    "summary_projection",
    "write_derived_artifacts",
    "write_derived_reports",
    "write_reports",
]
