import dataclasses
import unittest
from typing import Any

from quasilisk_harness.perturbation.models import MutationTask, TaskPurpose
from quasilisk_harness.perturbation.scheduler import (
    SchedulingPolicy,
    schedule_candidates,
    schedule_mutation_tasks,
)


@dataclasses.dataclass(frozen=True)
class _Candidate:
    id: str
    parent_id: str | None
    metadata: dict[str, Any]


def _candidate(
    candidate_id: str,
    purpose: TaskPurpose,
    attempt_id: str,
) -> _Candidate:
    return _Candidate(
        id=candidate_id,
        parent_id=f"parent-{attempt_id}",
        metadata={
            "task_purpose": purpose.value,
            "perturbation_attempt_id": attempt_id,
        },
    )


def _task(task_id: str, purpose: TaskPurpose) -> MutationTask:
    return MutationTask.from_jsonable(
        {
            "task_id": task_id,
            "operator_id": task_id,
            "purpose": purpose.value,
        }
    )


class SchedulingPolicyTests(unittest.TestCase):
    def test_partial_config_uses_defaults_and_round_trips(self):
        policy = SchedulingPolicy.from_config(
            {
                "purpose_weights": {"optimization": 8},
                "max_controls_per_attempt": 2,
                "max_control_share": 0.4,
            }
        )
        self.assertEqual(policy.weight(TaskPurpose.OPTIMIZATION), 8)
        self.assertEqual(policy.weight(TaskPurpose.DISCOVERY), 1)
        self.assertEqual(policy.weight(TaskPurpose.DIAGNOSTIC), 1)
        self.assertEqual(policy.max_controls_per_attempt, 2)
        self.assertEqual(policy.max_control_share, 0.4)
        self.assertEqual(
            SchedulingPolicy.from_config(policy.to_jsonable()),
            policy,
        )

    def test_config_rejects_unknown_fields_and_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "unknown perturbation_scheduler"):
            SchedulingPolicy.from_config({"strategy": "fifo"})
        with self.assertRaisesRegex(ValueError, "unknown scheduling purpose"):
            SchedulingPolicy.from_config(
                {"purpose_weights": {"control": 1}}
            )
        with self.assertRaisesRegex(ValueError, "finite and > 0"):
            SchedulingPolicy.from_config(
                {"purpose_weights": {"optimization": 0}}
            )
        with self.assertRaisesRegex(TypeError, "must be an integer"):
            SchedulingPolicy.from_config({"max_controls_per_attempt": True})
        with self.assertRaisesRegex(ValueError, r"in \[0, 1\)"):
            SchedulingPolicy.from_config({"max_control_share": 1})


class CandidateSchedulerTests(unittest.TestCase):
    def test_optimization_is_not_starved_by_early_discovery_candidates(self):
        candidates = [
            _candidate("discovery-a-1", TaskPurpose.DISCOVERY, "a"),
            _candidate("discovery-a-2", TaskPurpose.DISCOVERY, "a"),
            _candidate("optimization-b", TaskPurpose.OPTIMIZATION, "b"),
            _candidate("optimization-c", TaskPurpose.OPTIMIZATION, "c"),
        ]
        schedule = schedule_candidates(candidates, budget=2)
        self.assertEqual(
            {candidate.id for candidate in schedule.scheduled},
            {"optimization-b", "optimization-c"},
        )
        self.assertEqual(
            schedule.audit["by_purpose"]["optimization"]["scheduled"],
            2,
        )
        self.assertTrue(schedule.audit["invariants"]["within_budget"])

    def test_attempts_receive_one_work_item_before_repeats(self):
        candidates = [
            _candidate("a-1", TaskPurpose.OPTIMIZATION, "a"),
            _candidate("a-2", TaskPurpose.OPTIMIZATION, "a"),
            _candidate("a-3", TaskPurpose.OPTIMIZATION, "a"),
            _candidate("b-1", TaskPurpose.OPTIMIZATION, "b"),
        ]
        schedule = schedule_candidates(candidates, budget=2)
        self.assertEqual(
            {candidate.id for candidate in schedule.scheduled},
            {"a-1", "b-1"},
        )

    def test_attempt_fairness_floor_spans_mixed_purposes(self):
        candidates = [
            _candidate("a-opt-1", TaskPurpose.OPTIMIZATION, "a"),
            _candidate("a-opt-2", TaskPurpose.OPTIMIZATION, "a"),
            _candidate("b-discovery", TaskPurpose.DISCOVERY, "b"),
            _candidate("c-diagnostic", TaskPurpose.DIAGNOSTIC, "c"),
        ]
        schedule = schedule_candidates(candidates, budget=3)
        self.assertEqual(
            {
                candidate.metadata["perturbation_attempt_id"]
                for candidate in schedule.scheduled
            },
            {"a", "b", "c"},
        )
        self.assertTrue(
            schedule.audit["invariants"]["attempt_fairness_floor_met"]
        )
        self.assertEqual(
            schedule.audit["allocation"]["scheduled_work_attempts"],
            3,
        )

    def test_controls_are_paired_only_after_each_attempt_has_work(self):
        candidates = [
            _candidate("control-a", TaskPurpose.CONTROL, "a"),
            _candidate("a-1", TaskPurpose.OPTIMIZATION, "a"),
            _candidate("a-2", TaskPurpose.OPTIMIZATION, "a"),
            _candidate("control-b", TaskPurpose.CONTROL, "b"),
            _candidate("b-1", TaskPurpose.OPTIMIZATION, "b"),
            _candidate("b-2", TaskPurpose.OPTIMIZATION, "b"),
        ]

        two_slots = schedule_candidates(candidates, budget=2)
        self.assertEqual(
            {candidate.id for candidate in two_slots.scheduled},
            {"a-1", "b-1"},
        )

        three_slots = schedule_candidates(candidates, budget=3)
        self.assertEqual(
            {candidate.id for candidate in three_slots.scheduled},
            {"control-a", "a-1", "b-1"},
        )
        self.assertTrue(
            three_slots.audit["invariants"]["controls_have_scheduled_work"]
        )
        self.assertTrue(
            three_slots.audit["invariants"]["control_share_limit_met"]
        )
        self.assertEqual(
            three_slots.audit["allocation"]["realized_control_share"],
            round(1 / 3, 6),
        )

        four_slots = schedule_candidates(candidates, budget=4)
        self.assertEqual(
            {candidate.id for candidate in four_slots.scheduled},
            {"control-a", "a-1", "a-2", "b-1"},
        )
        deferred_reasons = {
            item["candidate_id"]: item["reason"]
            for item in four_slots.audit["deferred_candidates"]
        }
        self.assertEqual(deferred_reasons["control-b"], "control_budget")

    def test_orphan_and_excess_controls_are_deferred_with_reasons(self):
        candidates = [
            _candidate("control-a-1", TaskPurpose.CONTROL, "a"),
            _candidate("control-a-2", TaskPurpose.CONTROL, "a"),
            _candidate("work-a", TaskPurpose.OPTIMIZATION, "a"),
            _candidate("control-orphan", TaskPurpose.CONTROL, "orphan"),
        ]
        schedule = schedule_candidates(candidates, budget=None)
        self.assertEqual(
            {candidate.id for candidate in schedule.scheduled},
            {"control-a-1", "work-a"},
        )
        reasons = {
            item["candidate_id"]: item["reason"]
            for item in schedule.audit["deferred_candidates"]
        }
        self.assertEqual(reasons["control-a-2"], "control_limit")
        self.assertEqual(reasons["control-orphan"], "orphan_control")

    def test_weighted_work_order_preserves_an_optimization_majority(self):
        candidates = [
            *(
                _candidate(f"optimization-{index}", TaskPurpose.OPTIMIZATION, "a")
                for index in range(6)
            ),
            *(
                _candidate(f"discovery-{index}", TaskPurpose.DISCOVERY, "a")
                for index in range(3)
            ),
            *(
                _candidate(f"diagnostic-{index}", TaskPurpose.DIAGNOSTIC, "a")
                for index in range(3)
            ),
        ]
        schedule = schedule_candidates(candidates, budget=6)
        self.assertGreaterEqual(
            schedule.audit["by_purpose"]["optimization"]["scheduled"],
            4,
        )

    def test_scheduler_is_work_conserving_when_only_discovery_exists(self):
        candidates = [
            _candidate(f"discovery-{index}", TaskPurpose.DISCOVERY, "a")
            for index in range(4)
        ]
        schedule = schedule_candidates(candidates, budget=3)
        self.assertEqual(len(schedule.scheduled), 3)
        self.assertTrue(
            all(
                candidate.metadata["task_purpose"] == "discovery"
                for candidate in schedule.scheduled
            )
        )

    def test_invalid_input_fails_closed(self):
        duplicate = _candidate("same", TaskPurpose.OPTIMIZATION, "a")
        with self.assertRaisesRegex(ValueError, "duplicate candidate id"):
            schedule_candidates([duplicate, duplicate], budget=1)
        malformed = dataclasses.replace(
            duplicate,
            id="malformed",
            metadata={"task_purpose": "optimisation"},
        )
        with self.assertRaisesRegex(ValueError, "unknown task purpose"):
            schedule_candidates([malformed], budget=1)
        with self.assertRaisesRegex(ValueError, "budget"):
            schedule_candidates([duplicate], budget=-1)
        ranked_out = dataclasses.replace(
            duplicate,
            id="ranked-out",
            metadata={
                **duplicate.metadata,
                "refinement": {"disposition": "ranked_out"},
            },
        )
        with self.assertRaisesRegex(ValueError, "ranked-out candidate"):
            schedule_candidates([ranked_out], budget=1)

    def test_nested_task_purpose_is_canonical_and_conflicts_fail_closed(self):
        nested_control = dataclasses.replace(
            _candidate("nested-control", TaskPurpose.OPTIMIZATION, "a"),
            metadata={
                "perturbation_attempt_id": "a",
                "mutation_task": {"purpose": "control"},
            },
        )
        work = _candidate("work", TaskPurpose.OPTIMIZATION, "a")
        schedule = schedule_candidates([nested_control, work], budget=2)
        self.assertEqual(schedule.scheduled[0].id, "nested-control")

        conflicting = dataclasses.replace(
            nested_control,
            id="conflicting",
            metadata={
                **nested_control.metadata,
                "task_purpose": "optimization",
            },
        )
        with self.assertRaisesRegex(ValueError, "conflicting task purposes"):
            schedule_candidates([conflicting, work], budget=2)

        enum_metadata = dataclasses.replace(
            work,
            id="enum-purpose",
            metadata={"task_purpose": TaskPurpose.DISCOVERY},
        )
        enum_schedule = schedule_candidates([enum_metadata], budget=1)
        self.assertEqual(
            enum_schedule.audit["by_purpose"]["discovery"]["scheduled"],
            1,
        )


class MutationTaskSchedulerTests(unittest.TestCase):
    def test_controls_do_not_consume_generation_quota(self):
        tasks = [
            _task("control", TaskPurpose.CONTROL),
            _task("optimization-1", TaskPurpose.OPTIMIZATION),
            _task("optimization-2", TaskPurpose.OPTIMIZATION),
            _task("optimization-3", TaskPurpose.OPTIMIZATION),
            _task("discovery", TaskPurpose.DISCOVERY),
        ]
        schedule = schedule_mutation_tasks(tasks, offspring_count=3)
        self.assertEqual(len(schedule.controls), 1)
        self.assertEqual(len(schedule.generation_tasks), 3)
        self.assertEqual(
            sum(
                task.purpose is TaskPurpose.OPTIMIZATION
                for task in schedule.generation_tasks
            ),
            2,
        )
        self.assertEqual(
            sum(
                task.purpose is TaskPurpose.DISCOVERY
                for task in schedule.generation_tasks
            ),
            1,
        )
        self.assertTrue(
            schedule.audit["invariants"]["controls_excluded_from_offspring_count"]
        )
        self.assertEqual(schedule.deferred[0].task_id, "optimization-3")
        self.assertEqual(
            schedule.audit["deferred_tasks"][0]["reason"],
            "offspring_quota",
        )

    def test_task_schedule_is_work_conserving_for_single_purpose(self):
        tasks = [
            _task(f"discovery-{index}", TaskPurpose.DISCOVERY)
            for index in range(4)
        ]
        schedule = schedule_mutation_tasks(tasks, offspring_count=3)
        self.assertEqual(len(schedule.generation_tasks), 3)
        self.assertTrue(
            all(
                task.purpose is TaskPurpose.DISCOVERY
                for task in schedule.generation_tasks
            )
        )

    def test_task_schedule_rejects_duplicate_ids_and_invalid_quota(self):
        task = _task("duplicate", TaskPurpose.OPTIMIZATION)
        with self.assertRaisesRegex(ValueError, "task ids must be unique"):
            schedule_mutation_tasks([task, task], offspring_count=1)
        with self.assertRaisesRegex(ValueError, "offspring_count"):
            schedule_mutation_tasks([task], offspring_count=-1)

    def test_task_schedule_does_not_select_orphan_or_disabled_controls(self):
        control = _task("control", TaskPurpose.CONTROL)
        no_work = schedule_mutation_tasks([control], offspring_count=3)
        self.assertEqual(no_work.controls, ())
        self.assertEqual(
            no_work.audit["deferred_tasks"][0]["reason"],
            "orphan_control",
        )

        disabled = schedule_mutation_tasks(
            [control, _task("work", TaskPurpose.OPTIMIZATION)],
            offspring_count=1,
            policy=SchedulingPolicy(max_control_share=0),
        )
        self.assertEqual(disabled.controls, ())
        self.assertEqual(
            disabled.audit["deferred_tasks"][0]["reason"],
            "control_disabled",
        )


if __name__ == "__main__":
    unittest.main()
