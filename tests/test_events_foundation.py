from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from foundation_support import (
    FIXED_TIMESTAMP,
    chain_events,
    invocation_id,
    run_created_payload,
)
from quasilisk.domain import canonical_json_bytes
from quasilisk.events import (
    AuthoritativeSchemas,
    EventJournal,
    JournalCorrupt,
    ZERO_HASH,
    build_event,
    parse_journal_bytes,
    read_events,
    validate_events,
)


def terminal_payload(label: str = "terminal") -> dict:
    return {
        "invocation_id": invocation_id(label),
        "reason": "operator_interrupted",
        "last_completed_work_unit_id": None,
        "resumable": True,
        "detail": "test termination",
    }


def event_bytes(event: dict) -> bytes:
    return canonical_json_bytes(event) + b"\n"


def permissive_schemas() -> AuthoritativeSchemas:
    return AuthoritativeSchemas({"type": "object"}, {"type": "object"})


class EventEnvelopeTests(unittest.TestCase):
    def test_event_identity_excludes_timestamp_while_hash_commits_to_it(self) -> None:
        payload = run_created_payload(label="identity")
        first = build_event(
            run_id=payload["run_id"],
            sequence=1,
            event_type="run_created",
            payload=payload,
            previous_hash=ZERO_HASH,
            timestamp_utc="2026-08-18T12:00:00.000000Z",
        )
        later = build_event(
            run_id=payload["run_id"],
            sequence=1,
            event_type="run_created",
            payload=payload,
            previous_hash=ZERO_HASH,
            timestamp_utc="2026-08-18T12:00:01.000000Z",
        )
        self.assertEqual(first["event_id"], later["event_id"])
        self.assertNotEqual(first["event_hash"], later["event_hash"])

    def test_chain_validation_commits_sequence_previous_hash_payload_and_causes(self) -> None:
        created_payload = run_created_payload(label="chain")
        created = chain_events(created_payload, [])[0]
        terminated = build_event(
            run_id=created_payload["run_id"],
            sequence=2,
            event_type="run_terminated",
            payload=terminal_payload("chain-terminal"),
            previous_hash=created["event_hash"],
            causes=(created["event_id"],),
            timestamp_utc=FIXED_TIMESTAMP,
        )
        self.assertEqual(validate_events([created, terminated]), [created, terminated])

        for field, replacement in (
            ("sequence", 3),
            ("previous_hash", ZERO_HASH),
            ("event_hash", ZERO_HASH),
        ):
            tampered = deepcopy(terminated)
            tampered[field] = replacement
            with self.subTest(field=field), self.assertRaises(JournalCorrupt):
                validate_events([created, tampered])
        tampered_payload = deepcopy(terminated)
        tampered_payload["payload"]["detail"] = "changed after hashing"
        with self.assertRaises(JournalCorrupt):
            validate_events([created, tampered_payload])

    def test_envelopes_and_payloads_are_closed(self) -> None:
        payload = run_created_payload(label="closed")
        event = chain_events(payload, [])[0]
        extra_envelope = deepcopy(event)
        extra_envelope["unknown"] = None
        with self.assertRaises(JournalCorrupt):
            validate_events([extra_envelope])

        extra_payload = deepcopy(payload)
        extra_payload["unknown"] = None
        with self.assertRaises(JournalCorrupt):
            build_event(
                run_id=payload["run_id"],
                sequence=1,
                event_type="run_created",
                payload=extra_payload,
                previous_hash=ZERO_HASH,
                timestamp_utc=FIXED_TIMESTAMP,
            )

    def test_journal_bytes_must_be_canonical_complete_jsonl(self) -> None:
        event = chain_events(run_created_payload(label="bytes"), [])[0]
        canonical = event_bytes(event)
        self.assertEqual(parse_journal_bytes(canonical), [event])
        with self.assertRaises(JournalCorrupt):
            parse_journal_bytes(canonical.rstrip(b"\n"))
        noncanonical = json.dumps(event, sort_keys=False).encode() + b"\n"
        self.assertNotEqual(noncanonical, canonical)
        with self.assertRaises(JournalCorrupt):
            parse_journal_bytes(noncanonical)
        with self.assertRaisesRegex(JournalCorrupt, "canonical JSON"):
            parse_journal_bytes(canonical.rstrip(b"\n") + b"\r\n")

    def test_authoritative_schema_inputs_are_snapshotted_and_tampering_fails_closed(self) -> None:
        event_schema = {
            "type": "object",
            "$defs": {"value": {"type": "string"}},
        }
        work_unit_schema = {"type": "object"}
        schemas = AuthoritativeSchemas(event_schema, work_unit_schema)

        event_schema["type"] = "array"
        self.assertEqual(schemas.event["type"], "object")
        with self.assertRaises(TypeError):
            schemas.event["type"] = "array"  # type: ignore[index]

        with self.assertRaises(TypeError):
            schemas.event["$defs"]["value"]["type"] = "integer"
        with self.assertRaises(TypeError):
            schemas.event["$defs"]["other"] = {"type": "integer"}


class DurableJournalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run_directory = Path(self.temporary.name) / "run"

    def test_append_persists_exact_valid_bytes_before_exposing_event(self) -> None:
        created_payload = run_created_payload(label="append")
        with EventJournal(self.run_directory) as journal:
            created = journal.append(
                "run_created", created_payload, timestamp_utc=FIXED_TIMESTAMP
            )
            terminated = journal.append(
                "run_terminated",
                terminal_payload("append-terminal"),
                causes=(created["event_id"],),
                timestamp_utc=FIXED_TIMESTAMP,
            )
            self.assertEqual(journal.events[-1], terminated)
            self.assertEqual(journal.terminal_hash, terminated["event_hash"])
            self.assertFalse((self.run_directory / ".event.pending").exists())
            self.assertFalse((self.run_directory / ".event.pending.tmp").exists())

        expected = event_bytes(created) + event_bytes(terminated)
        self.assertEqual((self.run_directory / "events.jsonl").read_bytes(), expected)
        self.assertEqual(read_events(self.run_directory), [created, terminated])

    def test_checkpoint_prefix_read_does_not_consume_later_history(self) -> None:
        created_payload = run_created_payload(label="checkpoint-prefix")
        created, terminated = chain_events(
            created_payload,
            [("run_terminated", terminal_payload("checkpoint-terminal"))],
        )
        self.run_directory.mkdir()
        (self.run_directory / "events.jsonl").write_bytes(
            event_bytes(created) + event_bytes(terminated) + b"not-json\n"
        )

        self.assertEqual(
            read_events(self.run_directory, through_sequence=2),
            [created, terminated],
        )
        with self.assertRaises(JournalCorrupt):
            read_events(self.run_directory)
        for invalid in (True, 0, -1, "1"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    read_events(self.run_directory, through_sequence=invalid)

    def test_append_and_event_views_do_not_share_mutable_payload_state(self) -> None:
        payload = run_created_payload(label="event-snapshot")
        with EventJournal(self.run_directory) as journal:
            created = journal.append(
                "run_created", payload, timestamp_utc=FIXED_TIMESTAMP
            )
            authoritative_seed = journal.events[0]["payload"]["manifest"][
                "random_seed"
            ]
            payload["manifest"]["random_seed"] = authoritative_seed + 1
            created["payload"]["manifest"]["random_seed"] = authoritative_seed + 2
            exposed = journal.events
            exposed[0]["payload"]["manifest"]["random_seed"] = (
                authoritative_seed + 3
            )
            self.assertEqual(
                journal.events[0]["payload"]["manifest"]["random_seed"],
                authoritative_seed,
            )

        persisted = read_events(self.run_directory)
        self.assertEqual(
            persisted[0]["payload"]["manifest"]["random_seed"],
            authoritative_seed,
        )

    def prepare_pending_next(self, label: str) -> tuple[dict, dict]:
        self.run_directory.mkdir()
        created_payload = run_created_payload(label=label)
        created = chain_events(created_payload, [])[0]
        pending = build_event(
            run_id=created_payload["run_id"],
            sequence=2,
            event_type="run_terminated",
            payload=terminal_payload(f"{label}-terminal"),
            previous_hash=created["event_hash"],
            timestamp_utc=FIXED_TIMESTAMP,
        )
        return created, pending

    def test_pending_next_event_is_appended_and_recovery_is_audited(self) -> None:
        created, pending = self.prepare_pending_next("pending-next")
        (self.run_directory / "events.jsonl").write_bytes(event_bytes(created))
        (self.run_directory / ".event.pending").write_bytes(event_bytes(pending))

        with EventJournal(
            self.run_directory,
            expected_run_id=created["run_id"],
            recovery_invocation_id=invocation_id("pending-next-recovery"),
        ) as journal:
            self.assertEqual(
                (self.run_directory / "events.jsonl").read_bytes(),
                event_bytes(created),
            )
            journal.bind_schemas(permissive_schemas())
            recovered = list(journal.events)

        self.assertEqual([event["type"] for event in recovered], [
            "run_created",
            "run_terminated",
            "journal_tail_recovered",
        ])
        audit = recovered[-1]["payload"]
        self.assertEqual(audit["action"], "pending_appended")
        self.assertEqual(audit["pending_event_hash"], pending["event_hash"])
        self.assertEqual(audit["resulting_terminal_hash"], pending["event_hash"])
        self.assertFalse((self.run_directory / ".event.pending").exists())
        self.assertEqual(read_events(self.run_directory), recovered)

    def test_pending_equal_to_durable_tail_is_cleanup_only(self) -> None:
        self.run_directory.mkdir()
        created = chain_events(run_created_payload(label="pending-equals-tail"), [])[0]
        (self.run_directory / "events.jsonl").write_bytes(event_bytes(created))
        (self.run_directory / ".event.pending").write_bytes(event_bytes(created))

        with EventJournal(
            self.run_directory,
            expected_run_id=created["run_id"],
            recovery_invocation_id=invocation_id("pending-removed-recovery"),
        ) as journal:
            journal.bind_schemas(permissive_schemas())
            recovered = list(journal.events)

        self.assertEqual(
            [event["type"] for event in recovered],
            ["run_created"],
        )
        self.assertFalse((self.run_directory / ".event.pending").exists())

    def test_failed_enter_releases_the_advisory_lock(self) -> None:
        self.run_directory.mkdir()
        (self.run_directory / "events.jsonl").write_bytes(b"corrupt\n")
        journal = EventJournal(self.run_directory)
        with self.assertRaises(JournalCorrupt):
            journal.__enter__()
        self.assertIsNone(journal._lock_handle)

        created = chain_events(run_created_payload(label="lock-released"), [])[0]
        (self.run_directory / "events.jsonl").write_bytes(event_bytes(created))
        with EventJournal(
            self.run_directory,
            expected_run_id=created["run_id"],
        ) as reopened:
            self.assertEqual(len(reopened.events), 1)

    def test_matching_partial_tail_is_replaced_and_audited(self) -> None:
        created, pending = self.prepare_pending_next("partial")
        complete_pending = event_bytes(pending)
        fragment = complete_pending[: len(complete_pending) // 3]
        original = event_bytes(created) + fragment
        (self.run_directory / "events.jsonl").write_bytes(original)
        (self.run_directory / ".event.pending").write_bytes(complete_pending)

        with EventJournal(
            self.run_directory,
            expected_run_id=created["run_id"],
            recovery_invocation_id=invocation_id("partial-recovery"),
        ) as journal:
            self.assertEqual(
                (self.run_directory / "events.jsonl").read_bytes(), original
            )
            journal.bind_schemas(permissive_schemas())
            recovered = list(journal.events)

        self.assertEqual(recovered[-1]["payload"]["action"], "partial_tail_replaced")
        self.assertEqual(recovered[-1]["payload"]["original_journal_bytes"], len(original))
        self.assertEqual(
            [event["type"] for event in recovered],
            ["run_created", "run_terminated", "journal_tail_recovered"],
        )
        self.assertEqual(read_events(self.run_directory), recovered)

    def test_orphan_temp_is_removed_only_after_validating_clean_journal(self) -> None:
        self.run_directory.mkdir()
        created = chain_events(run_created_payload(label="orphan-temp"), [])[0]
        (self.run_directory / "events.jsonl").write_bytes(event_bytes(created))
        temp = self.run_directory / ".event.pending.tmp"
        temp.write_bytes(b"not authoritative")
        with EventJournal(
            self.run_directory, expected_run_id=created["run_id"]
        ) as journal:
            self.assertEqual(list(journal.events), [created])
        self.assertFalse(temp.exists())

    def test_mismatching_pending_never_rewrites_partial_evidence(self) -> None:
        created, pending_a = self.prepare_pending_next("mismatch")
        pending_b = build_event(
            run_id=created["run_id"],
            sequence=2,
            event_type="run_terminated",
            payload={**terminal_payload("mismatch-b"), "detail": "different event"},
            previous_hash=created["event_hash"],
            timestamp_utc=FIXED_TIMESTAMP,
        )
        pending_a_bytes = event_bytes(pending_a)
        pending_b_bytes = event_bytes(pending_b)
        divergence = next(
            index
            for index, (left, right) in enumerate(zip(pending_a_bytes, pending_b_bytes))
            if left != right
        )
        original_journal = event_bytes(created) + pending_a_bytes[: divergence + 1]
        (self.run_directory / "events.jsonl").write_bytes(original_journal)
        (self.run_directory / ".event.pending").write_bytes(pending_b_bytes)

        journal = EventJournal(
            self.run_directory,
            expected_run_id=created["run_id"],
            recovery_invocation_id=invocation_id("mismatch-recovery"),
        )
        try:
            with self.assertRaises(JournalCorrupt):
                journal.__enter__()
        finally:
            journal.__exit__(None, None, None)
        self.assertEqual(
            (self.run_directory / "events.jsonl").read_bytes(), original_journal
        )
        self.assertEqual(
            (self.run_directory / ".event.pending").read_bytes(), pending_b_bytes
        )

    def test_pending_event_is_not_appended_before_materialized_schema_validation(self) -> None:
        created, pending = self.prepare_pending_next("schema-before-recovery")
        journal_bytes = event_bytes(created)
        pending_bytes = event_bytes(pending)
        (self.run_directory / "events.jsonl").write_bytes(journal_bytes)
        (self.run_directory / ".event.pending").write_bytes(pending_bytes)
        rejecting = AuthoritativeSchemas(
            {
                "type": "object",
                "properties": {"type": {"const": "run_created"}},
                "required": ["type"],
                "additionalProperties": True,
            },
            {"type": "object"},
        )

        with EventJournal(
            self.run_directory,
            expected_run_id=created["run_id"],
            recovery_invocation_id=invocation_id("schema-before-recovery"),
        ) as journal:
            with self.assertRaisesRegex(
                JournalCorrupt, "authoritative schema"
            ):
                journal.bind_schemas(rejecting)

        self.assertEqual(
            (self.run_directory / "events.jsonl").read_bytes(), journal_bytes
        )
        self.assertEqual(
            (self.run_directory / ".event.pending").read_bytes(), pending_bytes
        )


if __name__ == "__main__":
    unittest.main()
