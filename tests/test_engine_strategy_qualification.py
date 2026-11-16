"""Canonical engine conformance for Section 12.16 qualification units."""

from __future__ import annotations

from foundation_support import strategy_wire_contract

from quasilisk.domain import JudgeProcedure

from copy import deepcopy
from decimal import Decimal
import inspect
import json
from pathlib import Path
import re
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from integration_support import dryrun_config
from quasilisk.domain import (
    ActionPurpose,
    PortfolioIntent,
    Role,
    RoleCallVector,
    SearchStatus,
    entity_id,
    scientific_key,
)
from quasilisk.engine import (
    CandidatePlan,
    RunContext,
    _guard_allows,
    _initialize_routes,
    _messages_for_action,
    _qualification_binding_result,
    _qualification_perturbator_request,
    _qualification_proposal_result,
    _strategy_development_units,
    _strategy_holdout_units,
    build_strategy_development,
    build_strategy_holdout,
    load_resources,
)
from quasilisk.perturbation import make_proposal_slots
from quasilisk.scheduler import validate_work_unit_maximum
from quasilisk.strategy import (
    StrategyValidationError,
    make_template_binding_item_id,
    parse_strategy_template,
)


class EngineStrategyQualificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        resources = load_resources(dryrun_config("fatalism", root / "runs"))
        self.context = RunContext(
            resources=resources,
            run_id=entity_id("run", ["qualification-engine"]),
            run_nonce="0" * 32,
            invocation_id=entity_id("iv", ["qualification-engine"]),
            run_directory=root / "run",
            candidates={},
            routes={},
        )
        _initialize_routes(self.context)
        self.template = self._template()
        self.parent_unit_id = entity_id("wu", ["qualification-parent"])

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _attach_promising_screen(state, candidate: CandidatePlan) -> None:
        block_id = entity_id("sb", ["qualification", candidate.scientific_key])
        state.candidate_evidence = {
            candidate.candidate_id: SimpleNamespace(
                subject_id=candidate.candidate_id,
                screening_block_id=block_id,
                search_status=SearchStatus.PROMISING,
            )
        }
        state.screening_blocks = {
            candidate.candidate_id: SimpleNamespace(
                candidate_id=candidate.candidate_id,
                candidate_scientific_key=candidate.scientific_key,
                block_id=block_id,
                block_terminated=True,
                status=SearchStatus.PROMISING,
                effective_scores=(Decimal("70"), Decimal("75"), Decimal("80")),
            )
        }

    @staticmethod
    def _attach_parent_evidence(state, parent_node) -> None:
        observation_id = parent_node.target_observation_id
        turn_key = scientific_key(["qualification-parent-turn", observation_id])
        response = "The target owns this supported opening."
        primary = SimpleNamespace(
            observation_id=observation_id,
            evidence=(SimpleNamespace(quote=response, polarity="supports"),),
         ownership="adopted", procedure=JudgeProcedure(premise="adopted", inference="endorsed", reservation="none"), justification="Exact test evidence.")
        state.target_observations = {
            "parent": SimpleNamespace(
                observation_id=observation_id,
                scientific_key=turn_key,
                response=response,
            )
        }
        state.panel_views = {
            "parent": SimpleNamespace(
                observation_id=observation_id,
                resolved=True,
                effective_score=Decimal("75"),
                details=None,
                primary=primary,
             assessment=None)
        }
        state.member_assessments = {"primary": primary}

    def _template(self, *, safe: bool = True):
        return parse_strategy_template(
            {
                "purpose": "repair",
                "applicability": {
                    "diagnosis": "conditional_acceptance",
                    "requires": "A stated condition remains unresolved.",
                    "excludes": "No stated condition remains.",
                },
                "move": {
                    "act": "repair_grounded_gap",
                    "target_kind": "condition",
                    "descriptors": {
                        "objective_route": "causal_fixity",
                        "argument_move": "answer_objection",
                        "application_scope": "reasoning_policy",
                    },
                    "blueprint": "Resolve the condition without changing the premise.",
                },
                "hypothesis": "Resolving the target's condition should permit progress.",
                "preserve": [
                    {"kind": "meaning", "statement": "Preserve the accepted meaning."}
                ],
                "vary": [
                    {
                        "dimension": "framing" if safe else "supporting_material",
                        "instruction": "Vary only this declared dimension.",
                    }
                ],
                "success": {
                    "transition": {"from": "conditional", "to": "resolved"},
                    "supports": "The target resolves its own stated condition.",
                    "falsifies": "The target retains the condition.",
                },
            },
            self.context.objective.descriptor_values,
            self.context.objective.diagnosis_categories,
        )

    @staticmethod
    def _actions_for_plan(unit, plan):
        by_id = {item["action_id"]: item for item in unit.logical_action_slots}
        return tuple(by_id[item] for item in plan["ordered_action_ids"])

    def test_development_freezes_exact_three_arm_graph_and_future_seal(self) -> None:
        unit = build_strategy_development(
            self.context, self.template, parent_unit_id=self.parent_unit_id
        )
        self.context.resources.authoritative_schemas.validate_work_unit(unit.to_dict())
        self.assertEqual(
            validate_work_unit_maximum(unit), RoleCallVector(50, 200, 120, 80)
        )
        self.assertEqual(len(unit.execution_slots), 15)
        self.assertEqual(len(unit.proposal_slots), 10)
        self.assertEqual(len(unit.panel_plans), 25)
        self.assertEqual(len(unit.logical_action_slots), 175)

        plans = unit.design["root_plans"]
        self.assertEqual(
            [item["arm"] for item in plans],
            ["treatment"] * 5
            + ["parent_baseline"] * 5
            + ["strategy_agnostic"] * 5,
        )
        expected = {
            "treatment": {
                (Role.TARGET.value, ActionPurpose.TARGET_SAMPLE.value): 2,
                (Role.JUDGE.value, ActionPurpose.JUDGE_MEMBER.value): 8,
                (Role.SELECTOR.value, ActionPurpose.TEMPLATE_BINDING.value): 1,
                (Role.SELECTOR.value, ActionPurpose.PROBE_REVIEW.value): 2,
                (Role.SELECTOR.value, ActionPurpose.OUTCOME_REVIEW.value): 1,
                (Role.PERTURBATOR.value, ActionPurpose.PERTURBATOR_PROPOSAL.value): 2,
            },
            "parent_baseline": {
                (Role.TARGET.value, ActionPurpose.TARGET_SAMPLE.value): 1,
                (Role.JUDGE.value, ActionPurpose.JUDGE_MEMBER.value): 4,
            },
            "strategy_agnostic": {
                (Role.TARGET.value, ActionPurpose.TARGET_SAMPLE.value): 2,
                (Role.JUDGE.value, ActionPurpose.JUDGE_MEMBER.value): 8,
                (Role.SELECTOR.value, ActionPurpose.PROBE_REVIEW.value): 2,
                (Role.PERTURBATOR.value, ActionPurpose.PERTURBATOR_PROPOSAL.value): 2,
            },
        }
        for plan in plans:
            counts: dict[tuple[str, str], int] = {}
            for action in self._actions_for_plan(unit, plan):
                key = (action["role"], action["purpose"])
                counts[key] = counts.get(key, 0) + 1
            self.assertEqual(counts, expected[plan["arm"]])
            self.assertTrue(
                all(
                    panel["trigger_policy"] == "always_full"
                    for panel in unit.panel_plans
                    if panel["target_action_id"] in plan["ordered_action_ids"]
                )
            )

        by_plan_id = {item["root_plan_id"]: item for item in plans}
        interleaved = [by_plan_id[item] for item in unit.design["interleaved_root_order"]]
        for offset in range(0, len(interleaved), 3):
            self.assertEqual(
                {item["arm"] for item in interleaved[offset : offset + 3]},
                {"treatment", "parent_baseline", "strategy_agnostic"},
            )
        sealed_text = json.dumps(unit.design["sealed_holdout_plan"], sort_keys=True)
        self.assertIsNone(re.search(r"(?:ex|rp|la|wu)-[0-9a-f]{64}", sealed_text))
        self.assertNotIn("development_outcome", sealed_text)

    def test_qualification_root_authentication_uses_frozen_judge_configuration(self) -> None:
        from dataclasses import replace
        from quasilisk.engine import _manifest
        from quasilisk.events import JournalCorrupt
        from quasilisk.run_state import LogicalActionView, WorkUnitView, _validate_qualification_root_graphs

        for judge_count in (2,4):
            with self.subTest(judges=judge_count):
                config=dryrun_config('fatalism',Path(self.temporary.name)/'runs')
                if judge_count==2:
                    config['models']['judges']['secondaries']=config['models']['judges']['secondaries'][:1]
                    config['models']['judges']['replacement']=None
                context=replace(self.context,resources=load_resources(config),routes={})
                _initialize_routes(context)
                manifest=_manifest(context)
                development=build_strategy_development(context,self.template,parent_unit_id=self.parent_unit_id)
                holdout=build_strategy_holdout(context,self.template,source_development_unit=development.to_dict())
                for unit in (development,holdout):
                    payload=unit.to_dict()
                    actions={slot['action_id']:LogicalActionView(slot['action_id'],unit.unit_id,slot,())
                             for slot in payload['logical_action_slots']}
                    view=WorkUnitView({},payload,tuple(actions.values()))
                    _validate_qualification_root_graphs(view,run_id=context.run_id,manifest=manifest,logical_actions=actions)
                    tampered=deepcopy(payload)
                    tampered['design']['root_plans'][0]['ordered_action_ids'].pop(1)
                    with self.assertRaises(JournalCorrupt):
                        _validate_qualification_root_graphs(WorkUnitView({},tampered,tuple(actions.values())),
                            run_id=context.run_id,manifest=manifest,logical_actions=actions)

    def test_holdout_materializes_only_the_seal_and_rejects_substitution(self) -> None:
        development = build_strategy_development(
            self.context, self.template, parent_unit_id=self.parent_unit_id
        )
        first = build_strategy_holdout(
            self.context,
            self.template,
            source_development_unit=development.to_dict(),
        )
        second = build_strategy_holdout(
            self.context,
            self.template,
            source_development_unit=development.to_dict(),
        )
        self.context.resources.authoritative_schemas.validate_work_unit(first.to_dict())
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.parent_unit_id, development.unit_id)
        self.assertEqual(
            validate_work_unit_maximum(first), RoleCallVector(40, 160, 120, 80)
        )
        self.assertEqual(len(first.execution_slots), 10)
        self.assertEqual(len(first.proposal_slots), 10)
        self.assertEqual(len(first.panel_plans), 20)
        self.assertEqual(len(first.logical_action_slots), 150)
        self.assertNotIn(
            "outcome", " ".join(inspect.signature(build_strategy_holdout).parameters)
        )

        tampered = deepcopy(development.to_dict())
        tampered["design"]["sealed_holdout_digest"] = scientific_key(["wrong"])
        with self.assertRaisesRegex(Exception, "seal digest"):
            build_strategy_holdout(
                self.context,
                self.template,
                source_development_unit=tampered,
            )
        with self.assertRaisesRegex(Exception, "StrategyTemplate identity"):
            build_strategy_holdout(
                self.context,
                self._template(safe=False),
                source_development_unit=development.to_dict(),
            )

    def test_development_identity_is_independent_of_run_local_parent_provenance(self) -> None:
        first = build_strategy_development(
            self.context, self.template, parent_unit_id=self.parent_unit_id
        )
        second = build_strategy_development(
            self.context,
            self.template,
            parent_unit_id=entity_id("wu", ["another-parent"]),
        )
        self.assertEqual(first.unit_scientific_key, second.unit_scientific_key)
        self.assertEqual(
            [item["action_scientific_key"] for item in first.logical_action_slots],
            [item["action_scientific_key"] for item in second.logical_action_slots],
        )
        self.assertNotEqual(first.parent_unit_id, second.parent_unit_id)
        with self.assertRaisesRegex(Exception, "control-safe"):
            build_strategy_development(
                self.context,
                self._template(safe=False),
                parent_unit_id=self.parent_unit_id,
            )

    def test_template_binding_id_uses_the_exact_five_part_material(self) -> None:
        unit_key = scientific_key(["binding", "unit"])
        execution_key = scientific_key(["binding", "execution"])
        template_key = scientific_key(["binding", "template"])
        self.assertEqual(
            make_template_binding_item_id(unit_key, execution_key, template_key, 0),
            entity_id(
                "bi",
                [unit_key, "template_binding", execution_key, template_key, 0],
            ),
        )
        with self.assertRaises(StrategyValidationError):
            make_template_binding_item_id(unit_key, execution_key, template_key, -1)

    def test_live_qualification_binding_uses_its_frozen_execution(self) -> None:
        unit_key = scientific_key(["live-binding", "unit"])
        execution_key = scientific_key(["live-binding", "execution"])
        template_key = scientific_key(["live-binding", "template"])
        turn_key = scientific_key(["live-binding", "turn"])
        payload = {
            "binding": "b0",
            "status": "applicable",
            "evidence": [{"evidence_id": "e0"}],
            "reason": "The fresh condition is unresolved.",
            "contract": {
                "purpose": "repair",
                "basis": [
                    {
                        "use": "continuation_anchor",
                        "evidence_id": "e0",
                        "reading": "The target states the unresolved condition.",
                    }
                ],
                "move": {
                    "act": "repair_grounded_gap",
                    "target": {
                        "kind": "condition",
                        "evidence_id": "e0",
                        "summary": "the unresolved condition",
                    },
                    "descriptors": dict(self.template.move.descriptors),
                    "blueprint": self.template.move.blueprint,
                },
                "hypothesis": self.template.hypothesis,
                "preserve": [
                    {
                        "kind": "meaning",
                        "statement": "Preserve the accepted meaning.",
                        "evidence_id": "e0",
                    }
                ],
                "vary": [item.to_dict() for item in self.template.vary],
                "success": self.template.success.to_dict(),
            },
        }
        payload["contract"] = strategy_wire_contract(payload["contract"])
        unit = SimpleNamespace(
            unit={
                "unit_scientific_key": unit_key,
                "design": {"strategy_template_scientific_key": template_key},
            }
        )
        execution = {"execution_scientific_key": execution_key}
        with (
            patch(
                "quasilisk.engine._qualification_root_context",
                return_value=({"arm": "treatment"}, execution, 0, 0, None, None),
            ),
            patch(
                "quasilisk.engine._qualification_template",
                return_value=self.template,
            ),
            patch(
                "quasilisk.engine._qualification_fresh_branch",
                return_value=([], turn_key, "The condition remains unresolved.", SimpleNamespace()),
            ),
            patch(
                "quasilisk.engine._schema_validated_payloads",
                return_value=(((None, None, payload),), ()),
            ),
        ):
            result = _qualification_binding_result(
                self.context, SimpleNamespace(), unit, SimpleNamespace()
            )
        self.assertTrue(result.usable)
        self.assertEqual(
            result.value.binding_item_id,
            make_template_binding_item_id(
                unit_key, execution_key, template_key, 0
            ),
        )

    def test_strategy_agnostic_request_keeps_the_closed_information_mask(self) -> None:
        unit_key = scientific_key(["masked-control-unit"])
        slot = make_proposal_slots(unit_key, 1)[0]
        unit = SimpleNamespace(
            unit={
                "kind": "strategy_holdout",
                "design": {
                    "sealed_holdout_plan": {
                        "comparison_dimensions": ["framing"],
                    }
                },
            }
        )
        root = {"arm": "strategy_agnostic"}
        with (
            patch(
                "quasilisk.engine._qualification_root_context",
                return_value=(root, {}, 0, 0, slot, "anchor"),
            ),
            patch(
                "quasilisk.engine._qualification_fresh_branch",
                return_value=(
                    [
                        {"role": "user", "content": "fresh prompt"},
                        {"role": "assistant", "content": "fresh response"},
                    ],
                    scientific_key(["fresh-turn"]),
                    "fresh response",
                    SimpleNamespace(),
                ),
            ),
            patch(
                "quasilisk.engine._qualification_correction_review",
                return_value=None,
            ),
        ):
            request = _qualification_perturbator_request(
                self.context, SimpleNamespace(), unit, SimpleNamespace()
            )
        self.assertEqual(
            set(request),
            {
                "mode",
                "proposal_slots",
                "fresh_branch_transcript",
                "objective_id",
                "scientific_question",
                "descriptor_vocabulary",
                "surface_intent",
                "comparison_dimensions",
                "instruction",
            },
        )
        self.assertEqual(
            request["proposal_slots"], [{"slot": "s0", "intent": "anchor"}]
        )
        serialized = json.dumps(request, sort_keys=True)
        for forbidden in (
            "diagnosis",
            "strategy_template",
            "bound_strategy",
            "hypothesis",
            "success_transition",
            "judge_result",
            "score",
            "archive",
            "holdout_outcome",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_qualification_treatment_dispatch_requires_alignment(self) -> None:
        unit_key = scientific_key(["qualification-dispatch-unit"])
        slot = make_proposal_slots(unit_key, 1)[0]
        review_item_id = entity_id("ri", ["qualification-review"])
        action = SimpleNamespace(
            slot={
                "guard": {
                    "kind": "if_dispatchable_probe",
                    "review_item_id": review_item_id,
                    "proposal_slot_scientific_key": slot.scientific_key,
                    "allowed": ["aligned"],
                }
            }
        )
        unit = SimpleNamespace(unit={"kind": "strategy_development"})

        def selected(status: str):
            return SimpleNamespace(
                review=SimpleNamespace(
                    status=status,
                    review_item_id=review_item_id,
                )
            )

        with (
            patch(
                "quasilisk.engine._qualification_root_context",
                return_value=({"arm": "treatment"}, {}, 0, 0, slot, "anchor"),
            ),
            patch(
                "quasilisk.engine._qualification_selected_realization",
                return_value=selected("invalid_contract"),
            ),
        ):
            self.assertFalse(
                _guard_allows(self.context, SimpleNamespace(), unit, action)
            )
        with (
            patch(
                "quasilisk.engine._qualification_root_context",
                return_value=({"arm": "treatment"}, {}, 0, 0, slot, "anchor"),
            ),
            patch(
                "quasilisk.engine._qualification_selected_realization",
                return_value=selected("aligned"),
            ),
        ):
            self.assertTrue(
                _guard_allows(self.context, SimpleNamespace(), unit, action)
            )

    def test_treatment_proposal_reaches_review_before_alignment_is_decided(self) -> None:
        unit_key = scientific_key(["qualification-pre-review-unit"])
        slot = make_proposal_slots(unit_key, 1)[0]
        initial = SimpleNamespace(
            action_id=entity_id("la", ["qualification-initial"]),
            slot={"purpose": ActionPurpose.PERTURBATOR_PROPOSAL.value},
        )
        corrective = SimpleNamespace(
            action_id=entity_id("la", ["qualification-corrective"]),
            slot={"purpose": ActionPurpose.PERTURBATOR_PROPOSAL.value},
        )
        measurement_cell = dict(self.template.move.descriptors)
        payload = {
            "items": [
                {
                    "slot": "s0",
                    "status": "proposed",
                    "text": "A structurally valid but potentially unaligned turn.",
                }
            ]
        }
        unit = SimpleNamespace(
            unit={
                "kind": "strategy_development",
                "unit_scientific_key": unit_key,
                "design": {
                    "comparison_dimensions": ["framing"],
                    "measurement_cell": measurement_cell,
                },
            }
        )
        with (
            patch(
                "quasilisk.engine._qualification_root_context",
                return_value=(
                    {"arm": "treatment"},
                    {},
                    0,
                    0,
                    slot,
                    "close",
                ),
            ),
            patch(
                "quasilisk.engine._qualification_root_actions",
                return_value=(initial, corrective),
            ),
            patch(
                "quasilisk.engine._schema_validated_payloads",
                return_value=(((None, None, payload),), ()),
            ),
        ):
            semantic = _qualification_proposal_result(
                self.context, SimpleNamespace(), unit, initial
            )
        self.assertTrue(semantic.usable)
        self.assertEqual(
            semantic.value.proposed_items[0].user_turn,
            "A structurally valid but potentially unaligned turn.",
        )

    def test_fresh_entry_request_replays_only_the_standalone_source_prompt(self) -> None:
        candidate = CandidatePlan(
            entity_id("pc", ["qualification-fresh-entry"]),
            scientific_key(["qualification-fresh-entry"]),
            "standalone source prompt",
            {
                "objective_route": "causal_fixity",
                "argument_move": "test_inference",
                "application_scope": "abstract",
            },
            "bootstrap",
            None,
        )
        self.context.candidates[candidate.scientific_key] = candidate
        parent_node_id = entity_id("cn", ["qualification-fresh-source"])
        parent_node_key = scientific_key(["qualification-fresh-source"])
        development = build_strategy_development(
            self.context, self.template, parent_unit_id=self.parent_unit_id
        )
        actions = tuple(
            SimpleNamespace(
                action_id=item["action_id"],
                slot=item,
                attempts=(),
            )
            for item in development.logical_action_slots
        )
        action_by_id = {item.action_id: item for item in actions}
        entry = action_by_id[development.design["root_plans"][0]["ordered_action_ids"][0]]
        episode = SimpleNamespace(
            strategy_id=entity_id("st", ["qualification-fresh-source"]),
            strategy_scientific_key=scientific_key(
                ["qualification-fresh-source"]
            ),
            generation_parent_node_id=parent_node_id,
            generation_parent_node_scientific_key=parent_node_key,
            conversation_root_id=entity_id("cr", ["qualification-fresh-source"]),
            parent_baseline_score=Decimal("75"),
            parent_search_status=SearchStatus.PROMISING,
            portfolio_intent=PortfolioIntent.ADVANCE,
            rounds=(
                SimpleNamespace(
                    work_unit_id=self.parent_unit_id,
                    lane=SimpleNamespace(
                        item=SimpleNamespace(template_candidate=self.template)
                    ),
                ),
            ),
        )
        state = SimpleNamespace(
            scientific_scope_digest=self.context.scope,
            strategy_episodes={episode.strategy_id: episode},
            conversation_nodes={
                parent_node_id: SimpleNamespace(
                    node_id=parent_node_id,
                    scientific_key=parent_node_key,
                    candidate_id=candidate.candidate_id,
                    root_id=episode.conversation_root_id,
                    parent_node_id=None,
                    depth=0,
                    target_observation_id=entity_id(
                        "to", ["qualification-fresh-source"]
                    ),
                )
            },
            logical_actions=action_by_id,
        )
        self._attach_promising_screen(state, candidate)
        self._attach_parent_evidence(state, state.conversation_nodes[parent_node_id])
        messages = _messages_for_action(
            self.context,
            state,
            SimpleNamespace(unit=development.to_dict(), actions=actions),
            entry,
        )
        expected = []
        if self.context.config.target.system_prompt:
            expected.append(
                ("system", self.context.config.target.system_prompt)
            )
        expected.append(("user", candidate.prompt))
        self.assertEqual([(item.role, item.content) for item in messages], expected)
        self.assertNotIn("recorded target output", json.dumps(expected))

        stateful = CandidatePlan(
            candidate.candidate_id,
            candidate.scientific_key,
            candidate.prompt,
            candidate.descriptors,
            "stateful_strategy",
            None,
            parent_node_id=entity_id("cn", ["prior"]),
            parent_node_scientific_key=scientific_key(["prior"]),
        )
        self.context.candidates[candidate.scientific_key] = stateful
        with self.assertRaisesRegex(Exception, "incomplete immutable lineage"):
            _messages_for_action(
                self.context,
                state,
                SimpleNamespace(unit=development.to_dict(), actions=actions),
                entry,
            )

    def test_holdout_scheduler_consumes_only_reducer_eligibility(self) -> None:
        candidate = CandidatePlan(
            entity_id("pc", ["qualification-holdout-entry"]),
            scientific_key(["qualification-holdout-entry"]),
            "holdout source prompt",
            {
                "objective_route": "causal_fixity",
                "argument_move": "test_inference",
                "application_scope": "abstract",
            },
            "bootstrap",
            None,
        )
        self.context.candidates[candidate.scientific_key] = candidate
        development = build_strategy_development(
            self.context, self.template, parent_unit_id=self.parent_unit_id
        )
        episode_id = entity_id("st", ["qualification-holdout-source"])
        parent_id = entity_id("cn", ["qualification-holdout-source"])
        parent_key = scientific_key(["qualification-holdout-source"])
        root_id = entity_id("cr", ["qualification-holdout-source"])
        episode = SimpleNamespace(
            strategy_id=episode_id,
            generation_parent_node_id=parent_id,
            generation_parent_node_scientific_key=parent_key,
            conversation_root_id=root_id,
            parent_baseline_score=Decimal("75"),
            parent_search_status=SearchStatus.PROMISING,
            portfolio_intent=PortfolioIntent.ADVANCE,
        )
        view = SimpleNamespace(
            template_id=development.subject_id,
            template=self.template,
            development_decision=SimpleNamespace(
                status=SimpleNamespace(value="eligible")
            ),
            development_unit_id=development.unit_id,
            holdout_unit_id=None,
            source_episode_id=episode_id,
            source_entry_candidate_id=candidate.candidate_id,
            source_entry_candidate_scientific_key=candidate.scientific_key,
        )
        state = SimpleNamespace(
            strategy_qualifications={development.subject_id: view},
            work_units=(
                SimpleNamespace(
                    unit_id=development.unit_id,
                    unit=development.to_dict(),
                ),
            ),
            strategy_episodes={episode_id: episode},
            conversation_nodes={
                parent_id: SimpleNamespace(
                    node_id=parent_id,
                    scientific_key=parent_key,
                    candidate_id=candidate.candidate_id,
                    root_id=root_id,
                    parent_node_id=None,
                    depth=0,
                    target_observation_id=entity_id(
                        "to", ["qualification-holdout-source"]
                    ),
                )
            },
        )
        self._attach_promising_screen(state, candidate)
        self._attach_parent_evidence(state, state.conversation_nodes[parent_id])
        scheduled = _strategy_holdout_units(self.context, state)
        self.assertEqual(len(scheduled), 1)
        self.assertEqual(scheduled[0][0].kind, "strategy_holdout")
        self.assertIs(scheduled[0][1], episode)
        self.assertIs(scheduled[0][2], candidate)

        view.development_decision.status.value = "rejected"
        self.assertEqual(_strategy_holdout_units(self.context, state), ())

    def test_only_preferred_exceptional_template_enters_development_once(self) -> None:
        candidate = CandidatePlan(
            entity_id("pc", ["qualification-entry"]),
            scientific_key(["qualification-entry"]),
            "fresh qualification prompt",
            {
                "objective_route": "causal_fixity",
                "argument_move": "test_inference",
                "application_scope": "abstract",
            },
            "bootstrap",
            None,
        )
        self.context.candidates[candidate.scientific_key] = candidate
        parent_id = entity_id("cn", ["qualification-source"])
        parent_key = scientific_key(["qualification-source"])
        root_id = entity_id("cr", ["qualification-source"])

        def episode(label: str, *, preferred: bool, exceptional: bool):
            strategy_key = scientific_key(["qualification-strategy", label])
            return SimpleNamespace(
                strategy_id=entity_id("st", [label]),
                strategy_scientific_key=strategy_key,
                preferred_in_run=preferred,
                generation_parent_node_id=parent_id,
                generation_parent_node_scientific_key=parent_key,
                conversation_root_id=root_id,
                parent_baseline_score=Decimal("75"),
                parent_search_status=SearchStatus.PROMISING,
                portfolio_intent=PortfolioIntent.ADVANCE,
                rounds=(
                    SimpleNamespace(
                        complete=True,
                        work_unit_id=self.parent_unit_id,
                        lane=SimpleNamespace(
                            item=SimpleNamespace(template_candidate=self.template)
                        ),
                    ),
                ),
                children=(
                    SimpleNamespace(
                        child_scientific_key=scientific_key(["child", label]),
                        alignment_status="aligned",
                        outcome=SimpleNamespace(
                            aligned=True,
                            resolved=True,
                            outcome_result="supports",
                            confirmed_objective_exceptional=exceptional,
                            confirmed_objective_application=False,
                            effective_score=Decimal(91),
                        ),
                    ),
                ),
            )

        eligible = episode("eligible", preferred=True, exceptional=True)
        duplicate = episode("duplicate", preferred=True, exceptional=True)
        ineligible = episode("ineligible", preferred=False, exceptional=True)
        state = SimpleNamespace(
            work_units=(),
            strategy_episodes={
                item.strategy_id: item for item in (eligible, duplicate, ineligible)
            },
            conversation_nodes={
                parent_id: SimpleNamespace(
                    node_id=parent_id,
                    scientific_key=parent_key,
                    candidate_id=candidate.candidate_id,
                    root_id=root_id,
                    parent_node_id=None,
                    depth=0,
                    target_observation_id=entity_id(
                        "to", ["qualification-source"]
                    ),
                )
            },
        )
        self._attach_promising_screen(state, candidate)
        self._attach_parent_evidence(state, state.conversation_nodes[parent_id])
        state.scenarios = {}
        state.scenario_evidence = {}
        self.assertEqual(_strategy_development_units(self.context, state), ())
        scenario_id = entity_id("sc", ["qualification-verified-scenario"])
        state.scenarios = {
            "verified": SimpleNamespace(
                origin="derived",
                scenario_id=scenario_id,
                source_step_episode_ids=(
                    eligible.strategy_id,
                    duplicate.strategy_id,
                ),
            )
        }
        state.scenario_evidence = {
            scenario_id: SimpleNamespace(
                verification_status=SimpleNamespace(value="verified")
            )
        }
        units = _strategy_development_units(self.context, state)
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0][0].kind, "strategy_development")


if __name__ == "__main__":
    unittest.main()
