"""Focused replay tests for reducer-owned stateful portfolio guards."""

from __future__ import annotations

from foundation_support import strategy_wire_contract

from foundation_support import selector_applicability_payload

from decimal import Decimal
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from integration_support import dryrun_config
from quasilisk.domain import (
    ActionPurpose,
    JudgeProcedure,
    Application,
    ChatMessage,
    ConversationNode,
    EvidenceExcerpt,
    EvidenceRecord,
    FinishStatus,
    MemberAssessment,
    PortfolioIntent,
    Role,
    RoleCallVector,
    SearchStatus,
    TargetObservation,
    Verdict,
    digest_json,
    entity_id,
)
from quasilisk.events import JournalCorrupt
from quasilisk.evidence import ExecutionEvidence
from quasilisk.perturbation import (
    ProbeReview,
    ProposalItem,
    SelectedRealization,
)
from quasilisk.engine import (
    CandidatePlan,
    RunContext,
    _initialize_routes,
    _manifest,
    build_portfolio_round,
    load_resources,
    _portfolio_current_assessment,
    _portfolio_strategy_evidence,
)
from quasilisk.run_state import (
    AttemptView,
    ConversationRootView,
    LogicalActionView,
    PanelView,
    WorkUnitView,
    _candidate_required_selector_scopes,
    _completion_semantic_value,
    _derive_panel_views,
    _derive_stateful_portfolios,
    _parent_strategy_evidence,
    _portfolio_stateful_dispatch_allows,
    strategy_evidence_polarities,
)


def _key(label: str) -> str:
    return digest_json(["stateful-reducer-test", label])


def _action(
    label: str,
    ordinal: int,
    *,
    purpose: str,
    role: str,
    owner: str,
    guard: dict | None = None,
    attempts: tuple[AttemptView, ...] = (),
) -> LogicalActionView:
    return LogicalActionView(
        entity_id("la", ["stateful-reducer-test", label]),
        entity_id("wu", ["stateful-reducer-test", "unit"]),
        {
            "ordinal": ordinal,
            "purpose": purpose,
            "role": role,
            "owner_position_scientific_key": owner,
            "guard": guard or {"kind": "always"},
        },
        attempts,
    )


class EffectivePrimaryProjectionTests(unittest.TestCase):
    def test_replacement_assuming_primary_is_the_effective_primary(self) -> None:
        panel_id = entity_id("pn", ["effective-primary"])
        target_action_id = entity_id("la", ["effective-primary", "target"])
        observation = TargetObservation(
            entity_id("to", [target_action_id]),
            _key("effective-primary-observation"),
            entity_id("pc", ["effective-primary"]),
            None,
            (ChatMessage("user", "test prompt"),),
            _key("target-profile"),
            "I accept the premise but stop at an abstract conclusion.",
            FinishStatus.COMPLETE,
            entity_id("pa", ["effective-primary", "target"]),
        )
        member_ids = {
            slot: entity_id("la", ["effective-primary", slot])
            for slot in ("primary", "secondary_a", "secondary_b", "replacement")
        }

        def action(slot: str) -> SimpleNamespace:
            lineage_slot = "primary" if slot == "replacement" else slot
            return SimpleNamespace(
                action_id=member_ids[slot],
                attempts=(
                    SimpleNamespace(
                        intended={
                            "lineage": {
                                "panel_id": panel_id,
                                "logical_member_slot": lineage_slot,
                            }
                        }
                    ),
                ),
            )

        actions = {slot: action(slot) for slot in member_ids}
        target_action = SimpleNamespace(action_id=target_action_id, attempts=())
        logical_actions = {
            target_action_id: target_action,
            **{item.action_id: item for item in actions.values()},
        }

        def assessment(slot: str, score: str) -> MemberAssessment:
            return MemberAssessment(
                entity_id("ma", ["effective-primary", slot]),
                observation.observation_id,
                slot,
                Decimal(score),
                Verdict.DEMONSTRATED,
                Application.ABSTRACT,
                (
                    EvidenceExcerpt(
                        observation.response,
                        "resists" if slot == "secondary_a" else "supports",
                    ),
                ),
                "The reply demonstrates the configured failure abstractly.",
                (),
                f"model/{slot}",
                _key(f"{slot}-route"),
                entity_id("pa", ["effective-primary", slot]),
             ownership="adopted")

        usable = {
            member_ids["secondary_a"]: assessment("secondary_a", "70"),
            member_ids["secondary_b"]: assessment("secondary_b", "75"),
            member_ids["replacement"]: assessment("primary", "80"),
        }

        def judge_result(*, action, **_kwargs):
            member = usable.get(action.action_id)
            return (member, () if member is not None else ("invalid primary",))

        plan = {
            "panel_id": panel_id,
            "target_action_id": target_action_id,
            "trigger_policy": "always_full",
            "normal_members": [
                {
                    "judge_action_id": member_ids[slot],
                    "logical_member_slot": slot,
                }
                for slot in ("primary", "secondary_a", "secondary_b")
            ],
            "replacement": {"judge_action_id": member_ids["replacement"]},
        }
        unit = SimpleNamespace(unit={"panel_plans": [plan]})
        target = SimpleNamespace(
            candidate_id=observation.candidate_id,
            execution_id=entity_id("ex", ["effective-primary"]),
            terminal_event_id=entity_id("ev", ["effective-primary"]),
            effective_model_id="model/target",
        )

        with (
            patch(
                "quasilisk.run_state._judge_member_for_action",
                side_effect=judge_result,
            ),
            patch(
                "quasilisk.run_state._completion_action_closed",
                return_value=True,
            ),
        ):
            _, _, panels = _derive_panel_views(
                run_id=entity_id("run", ["effective-primary"]),
                work_units=(unit,),
                logical_actions=logical_actions,
                target_results={target_action_id: target},
                target_observations={target_action_id: observation},
                response_schemas={},
            )

        panel = panels[panel_id]
        self.assertEqual(panel.primary, usable[member_ids["replacement"]])
        self.assertEqual(panel.primary.logical_member_slot, "primary")
        self.assertEqual(
            panel.primary.evidence,
            (EvidenceExcerpt(observation.response, "supports"),),
        )
        state = SimpleNamespace(
            target_observations={target_action_id: observation},
            panel_views=panels,
            member_assessments=usable,
        )
        current = _portfolio_current_assessment(
            state, [{"turn_ref": observation.scientific_key, "text": observation.response}])
        self.assertEqual(len(current["procedure_assessments"]), 3)
        self.assertEqual(
            {key: current[key] for key in ("verdict", "application", "ownership")},
            {
                "verdict": "demonstrated",
                "application": "abstract",
             "ownership": "adopted"},
        )
        with patch(
            "quasilisk.engine._portfolio_branch_turns",
            return_value=((None, None, observation),),
        ):
            evidence, polarities = _portfolio_strategy_evidence(
                SimpleNamespace(), state, SimpleNamespace()
            )
        self.assertEqual(
            evidence,
            [
                {
                    "evidence_id": "e0",
                    "text": observation.response,
                    "polarities": ["resists", "supports"],
                }
            ],
        )
        self.assertEqual(
            polarities,
            {
                (
                    observation.scientific_key,
                    observation.response,
                ): frozenset({"resists", "supports"})
            },
        )
        replayed_polarities = strategy_evidence_polarities(
            turn_texts={observation.scientific_key: observation.response},
            target_observations={target_action_id: observation},
            panel_views=panels,
            member_assessments=usable,
        )
        self.assertEqual(replayed_polarities, polarities)

        target.effective_model_id = "model/primary"
        with (
            patch(
                "quasilisk.run_state._judge_member_for_action",
                side_effect=judge_result,
            ),
            patch(
                "quasilisk.run_state._completion_action_closed",
                return_value=True,
            ),
        ):
            _, _, collided_panels = _derive_panel_views(
                run_id=entity_id("run", ["effective-primary", "collision"]),
                work_units=(unit,),
                logical_actions=logical_actions,
                target_results={target_action_id: target},
                target_observations={target_action_id: observation},
                response_schemas={},
            )
        self.assertIsNone(collided_panels[panel_id].primary)


class StatefulPortfolioGuardTests(unittest.TestCase):
    def test_linked_parent_uses_entry_status_and_immediate_node_baseline(self) -> None:
        entry_candidate_id = entity_id("pc", ["screen-entry"])
        child_candidate_id = entity_id("pc", ["screen-child"])
        entry_node_id = entity_id("cn", ["screen-entry"])
        child_node_id = entity_id("cn", ["screen-child"])
        root_id = entity_id("cr", ["screen-root"])
        trigger_id = entity_id("ex", ["screen-entry", "trigger"])
        new_ids = (
            entity_id("ex", ["screen-entry", 0]),
            entity_id("ex", ["screen-entry", 1]),
        )
        unit = SimpleNamespace(
            start_event={"sequence": 1},
            unit={
                "kind": "screening",
                "subject_id": entry_candidate_id,
                "design": {"triggering_execution_id": trigger_id},
                "execution_slots": [
                    {"execution_id": execution_id} for execution_id in new_ids
                ],
            },
        )
        evidence = {
            execution_id: ExecutionEvidence(
                execution_id,
                terminated=True,
                resolved=True,
                primary_score=score,
            )
            for execution_id, score in zip(
                (trigger_id, *new_ids),
                (Decimal("72"), Decimal("76"), Decimal("80")),
                strict=True,
            )
        }
        nodes = {
            entry_node_id: ConversationNode(
                entry_node_id,
                _key("screen-entry-node"),
                root_id,
                None,
                0,
                entry_candidate_id,
                entity_id("to", ["screen-entry"]),
            ),
            child_node_id: ConversationNode(
                child_node_id,
                _key("screen-child-node"),
                root_id,
                entry_node_id,
                1,
                child_candidate_id,
                entity_id("to", ["screen-child"]),
            ),
        }

        with patch(
            "quasilisk.run_state.evaluate_screening",
            return_value=SimpleNamespace(
                status=SearchStatus.PROMISING,
                effective_scores=(
                    Decimal("72"),
                    Decimal("76"),
                    Decimal("80"),
                ),
            ),
        ):
            baseline, status = _parent_strategy_evidence(
                parent_node_id=child_node_id,
                conversation_nodes=nodes,
                work_units=(unit,),
                execution_evidence=evidence,
                candidate_evidence={},
                panel_views={
                    "child": SimpleNamespace(
                        observation_id=nodes[child_node_id].target_observation_id,
                        resolved=True,
                        effective_score=Decimal("61"),
                    )
                },
                before_sequence=2,
            )

        self.assertEqual(baseline, Decimal("61"))
        self.assertIs(status, SearchStatus.PROMISING)

    def test_screened_exploratory_lead_keeps_its_immediate_node_baseline(self) -> None:
        candidate_id = entity_id("pc", ["screened-exploratory-lead"])
        node_id = entity_id("cn", ["screened-exploratory-lead"])
        trigger_id = entity_id("ex", ["screened-exploratory-lead", "trigger"])
        new_ids = (
            entity_id("ex", ["screened-exploratory-lead", 0]),
            entity_id("ex", ["screened-exploratory-lead", 1]),
        )
        unit = SimpleNamespace(
            start_event={"sequence": 1},
            unit={
                "kind": "screening",
                "subject_id": candidate_id,
                "design": {"triggering_execution_id": trigger_id},
                "execution_slots": [
                    {"execution_id": execution_id} for execution_id in new_ids
                ],
            },
        )
        evidence = {
            trigger_id: ExecutionEvidence(
                trigger_id,
                terminated=True,
                resolved=False,
                primary_score=Decimal("80"),
                panel_required=True,
            ),
            **{
                execution_id: ExecutionEvidence(
                    execution_id,
                    terminated=True,
                    resolved=True,
                    primary_score=Decimal("50"),
                )
                for execution_id in new_ids
            },
        }

        baseline, status = _parent_strategy_evidence(
            parent_node_id=node_id,
            conversation_nodes={
                node_id: ConversationNode(
                    node_id,
                    _key("screened-exploratory-lead-node"),
                    entity_id("cr", ["screened-exploratory-lead"]),
                    None,
                    0,
                    candidate_id,
                    entity_id("to", ["screened-exploratory-lead"]),
                )
            },
            work_units=(unit,),
            execution_evidence=evidence,
            candidate_evidence={},
            panel_views={
                "lead": SimpleNamespace(
                    observation_id=entity_id(
                        "to", ["screened-exploratory-lead"]
                    ),
                    resolved=True,
                    effective_score=Decimal("50"),
                )
            },
            before_sequence=2,
        )

        self.assertEqual(baseline, Decimal("50"))
        self.assertIs(status, SearchStatus.EXPLORATORY_LEAD)

    def test_completion_guard_materializes_local_proposal_and_review_aliases(self) -> None:
        unit_key = _key("alias-guard-unit")
        slot_id = entity_id("ps", ["alias-guard-slot"])
        slot_key = _key("alias-guard-slot")
        review_id = entity_id(
            "ri", ["review_item", unit_key, "probe", slot_key]
        )
        schema: dict = {}
        schema_digest = digest_json(schema)

        def attempt(label: str, value: dict) -> AttemptView:
            return AttemptView(
                entity_id("pa", ["alias-guard", label]),
                {"payload": {"response_schema_digest": schema_digest}},
                {"payload": {}},
                {
                    "type": "model_call_succeeded",
                    "payload": {
                        "finish_status": "complete",
                        "text": json.dumps(value),
                    },
                },
            )

        proposal = _action(
            "alias-proposal",
            0,
            purpose=ActionPurpose.PERTURBATOR_PROPOSAL.value,
            role=Role.PERTURBATOR.value,
            owner=unit_key,
            attempts=(
                attempt(
                    "proposal",
                    {
                        "items": [
                            {
                                "slot": "s0",
                                "status": "proposed",
                                "text": "A reviewed target turn.",
                            }
                        ]
                    },
                ),
            ),
        )
        review = _action(
            "alias-review",
            1,
            purpose=ActionPurpose.PROBE_REVIEW.value,
            role=Role.SELECTOR.value,
            owner=unit_key,
            attempts=(
                attempt(
                    "review",
                    {
                        "reviews": [
                            {
                                "review_alias": "r0",
                                "status": "valid",
                                "reviewed_descriptors": {"cell": "value"},
                                "reason": "The turn is admissible.",
                             "procedure": {"assumption": "none", "assumption_quote": None, "direction": "forward", "novelty": "new"}}
                        ]
                    },
                ),
            ),
        )
        review.slot["dependency_action_ids"] = [proposal.action_id]
        unit = WorkUnitView(
            {},
            {
                "unit_id": entity_id("wu", ["alias-guard-unit"]),
                "unit_scientific_key": unit_key,
                "kind": "map_exploration",
                "proposal_slots": [
                    {
                        "slot_id": slot_id,
                        "scientific_key": slot_key,
                        "global_ordinal": 0,
                    }
                ],
            },
            (proposal, review),
        )
        actions = {item.action_id: item for item in unit.actions}
        self.assertEqual(
            _completion_semantic_value(
                unit,
                review,
                actions,
                {},
                {schema_digest: schema},
                {},
            ),
            {review_id: "valid"},
        )

    def _graph(self):
        unit_key = _key("unit")
        lane_a, lane_b = _key("lane-a"), _key("lane-b")
        proposal_keys = tuple(_key(f"proposal-{index}") for index in range(6))
        proposal_slots = [
            {
                "slot_id": entity_id("ps", ["stateful-reducer-test", index]),
                "scientific_key": key,
                "global_ordinal": index,
            }
            for index, key in enumerate(proposal_keys)
        ]
        review_ids = tuple(
            entity_id("ri", ["stateful-reducer-test", index])
            for index in range(6)
        )
        selector = _action(
            "selector",
            0,
            purpose=ActionPurpose.SELECTOR_PORTFOLIO.value,
            role=Role.SELECTOR.value,
            owner=unit_key,
        )
        original_a = _action(
            "original-a",
            1,
            purpose=ActionPurpose.PERTURBATOR_PROPOSAL.value,
            role=Role.PERTURBATOR.value,
            owner=lane_a,
        )
        original_b = _action(
            "original-b",
            2,
            purpose=ActionPurpose.PERTURBATOR_PROPOSAL.value,
            role=Role.PERTURBATOR.value,
            owner=lane_b,
        )
        first_review = _action(
            "first-review",
            3,
            purpose=ActionPurpose.PROBE_REVIEW.value,
            role=Role.SELECTOR.value,
            owner=unit_key,
        )
        targets = tuple(
            _action(
                f"target-{index}",
                4 + index,
                purpose=ActionPurpose.TARGET_SAMPLE.value,
                role=Role.TARGET.value,
                owner=_key(f"execution-{index}"),
                guard={
                    "kind": "if_dispatchable_probe",
                    "review_item_id": review_ids[index],
                    "proposal_slot_scientific_key": proposal_keys[index],
                    "allowed": ["aligned"],
                },
            )
            for index in range(6)
        )
        unit = WorkUnitView(
            {},
            {
                "unit_id": entity_id("wu", ["stateful-reducer-test", "unit"]),
                "unit_scientific_key": unit_key,
                "proposal_slots": proposal_slots,
                "design": {"generation_parent_node_id": entity_id("cn", ["parent"])},
            },
            (
                selector,
                original_a,
                original_b,
                first_review,
                *targets,
            ),
        )
        return (
            unit,
            proposal_slots,
            review_ids,
            original_a,
            original_b,
            first_review,
            targets,
        )

    @staticmethod
    def _child(
        slot: dict,
        review_id: str,
        target: LogicalActionView,
        *,
        prompt: str,
        descriptors: dict[str, str],
    ) -> SimpleNamespace:
        proposal = ProposalItem(
            slot["slot_id"], "proposed", user_turn=prompt
        )
        review = ProbeReview(
            review_id,
            slot["slot_id"],
            "aligned",
            tuple(sorted(descriptors.items())),
            "aligned",
        )
        return SimpleNamespace(
            target_action_id=target.action_id,
            proposal_slot=SimpleNamespace(
                scientific_key=slot["scientific_key"]
            ),
            selected=SelectedRealization(proposal, review, "original"),
        )

    def test_same_parent_duplicate_has_one_nonce_free_slot_owner(self) -> None:
        (
            unit,
            slots,
            review_ids,
            original_a,
            _,
            first_review,
            targets,
        ) = self._graph()
        descriptors = {
            "objective_route": "route",
            "argument_move": "move",
        }
        del original_a, first_review
        round_view = SimpleNamespace(
            children=tuple(
                self._child(
                    slots[index],
                    review_ids[index],
                    targets[index],
                    prompt="same\r\nturn",
                    descriptors=descriptors,
                )
                for index in range(2)
            )
        )
        owner_index = min(
            range(2), key=lambda index: slots[index]["scientific_key"]
        )
        for index in range(2):
            self.assertEqual(
                _portfolio_stateful_dispatch_allows(
                    unit,
                    targets[index],
                    {item.action_id: item for item in unit.actions},
                    round_view,
                ),
                index == owner_index,
            )

        conflicting = SimpleNamespace(
            children=(
                round_view.children[0],
                self._child(
                    slots[1],
                    review_ids[1],
                    targets[1],
                    prompt="same\nturn",
                    descriptors={
                        "objective_route": "other-route",
                        "argument_move": "move",
                    },
                ),
            )
        )
        self.assertFalse(
            _portfolio_stateful_dispatch_allows(
                unit,
                targets[owner_index],
                {item.action_id: item for item in unit.actions},
                conflicting,
            )
        )

    def test_prior_intended_same_parent_turn_closes_new_dispatch(self) -> None:
        (
            unit,
            slots,
            review_ids,
            original_a,
            _,
            first_review,
            targets,
        ) = self._graph()
        descriptors = {
            "objective_route": "route",
            "argument_move": "move",
        }
        del original_a, first_review
        round_view = SimpleNamespace(
            children=(
                self._child(
                    slots[0],
                    review_ids[0],
                    targets[0],
                    prompt="same turn",
                    descriptors=descriptors,
                ),
            )
        )
        prior_attempt = AttemptView(
            entity_id("pa", ["prior"]),
            {
                "payload": {
                    "lineage": {
                        "parent_node_id": unit.unit["design"][
                            "generation_parent_node_id"
                        ]
                    },
                    "messages": [{"role": "user", "content": "same turn"}],
                }
            },
            None,
            None,
        )
        prior = _action(
            "prior-target",
            0,
            purpose=ActionPurpose.TARGET_SAMPLE.value,
            role=Role.TARGET.value,
            owner=_key("prior-execution"),
            attempts=(prior_attempt,),
        )
        actions = {item.action_id: item for item in (*unit.actions, prior)}
        self.assertFalse(
            _portfolio_stateful_dispatch_allows(
                unit, targets[0], actions, round_view
            )
        )

    def test_reversal_status_is_never_selected(self) -> None:
        (
            unit,
            slots,
            review_ids,
            _,
            _,
            _,
            targets,
        ) = self._graph()
        round_view = SimpleNamespace(
            children=(
                SimpleNamespace(
                    target_action_id=targets[0].action_id,
                    proposal_slot=SimpleNamespace(
                        scientific_key=slots[0]["scientific_key"]
                    ),
                    selected=None,
                ),
            )
        )
        self.assertFalse(
            _portfolio_stateful_dispatch_allows(
                unit,
                targets[0],
                {item.action_id: item for item in unit.actions},
                round_view,
            )
        )


class StatefulPortfolioProjectionTests(unittest.TestCase):
    def test_only_aligned_strategy_candidates_have_selector_scopes(self) -> None:
        with self.assertRaisesRegex(JournalCorrupt, "dispatchable review status"):
            _candidate_required_selector_scopes(
                {
                    "origin": "stateful_strategy",
                    "review_status": "invalid_contract",
                }
            )
        self.assertEqual(
            _candidate_required_selector_scopes(
                {
                    "origin": "stateful_strategy",
                    "review_status": "aligned",
                }
            ),
            ("probe_review", "outcome_review"),
        )

    def test_materialized_portfolio_projects_every_allocated_child_immutably(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = dryrun_config("fatalism", Path(temporary) / "runs")
            config["search"]["proposals_per_step"] = 7
            resources = load_resources(config)
            run_id = entity_id("run", ["stateful-projection"])
            context = RunContext(
                resources=resources,
                run_id=run_id,
                run_nonce="0" * 32,
                invocation_id=entity_id("iv", ["stateful-projection"]),
                run_directory=Path(temporary) / "run",
                candidates={},
                routes={},
            )
            _initialize_routes(context)
            parent = CandidatePlan(
                entity_id("pc", [run_id, "parent"]),
                _key("parent-candidate"),
                "parent prompt",
                {
                    "objective_route": "causal_fixity",
                    "argument_move": "test_inference",
                    "application_scope": "abstract",
                },
                "bootstrap",
                None,
            )
            root_id = entity_id("cr", ["stateful-projection"])
            root_key = _key("conversation-root")
            observation = TargetObservation(
                entity_id("to", ["stateful-projection", "parent"]),
                _key("parent-observation"),
                parent.candidate_id,
                None,
                (ChatMessage("user", parent.prompt),),
                context.config.target.digest,
                "The condition remains unresolved, although I accept the premise.",
                FinishStatus.COMPLETE,
                entity_id("pa", ["stateful-projection", "parent"]),
            )
            node = ConversationNode(
                entity_id("cn", ["stateful-projection", "parent"]),
                _key("parent-node"),
                root_id,
                None,
                0,
                parent.candidate_id,
                observation.observation_id,
            )
            frozen = build_portfolio_round(
                context,
                parent,
                parent_unit_id=entity_id("wu", ["stateful-projection", "parent"]),
                conversation_root_id=root_id,
                conversation_root_scientific_key=root_key,
                parent_node_id=node.node_id,
                parent_node_scientific_key=node.scientific_key,
                portfolio_intent=PortfolioIntent.ADVANCE,
            )
            unit_dict = frozen.to_dict()
            selector_slot = next(
                slot
                for slot in unit_dict["logical_action_slots"]
                if slot["purpose"] == ActionPurpose.SELECTOR_PORTFOLIO.value
            )
            contract = {
                "purpose": "repair",
                "basis": [
                    {
                        "use": "continuation_anchor",
                        "evidence_id": "e0",
                        "reading": "The target states the opening to preserve.",
                    },
                    {
                        "use": "judge_concern",
                        "evidence_id": "e0",
                        "reading": "The target states the exact gap to repair.",
                    },
                ],
                "move": {
                    "act": "repair_grounded_gap",
                    "target": {
                        "kind": "condition",
                        "evidence_id": "e0",
                        "summary": "the unresolved condition",
                    },
                    "descriptors": {
                        "objective_route": "causal_fixity",
                        "argument_move": "answer_objection",
                        "application_scope": "abstract",
                    },
                    "blueprint": "Ask the target to resolve only the stated condition.",
                },
                "hypothesis": "Resolving the condition may clarify the commitment.",
                "preserve": [
                    {
                        "kind": "premise",
                        "statement": "Preserve the accepted premise.",
                        "evidence_id": "e0",
                    }
                ],
                "vary": [
                    {
                        "dimension": "wording",
                        "instruction": "Vary only the wording.",
                    }
                ],
                "success": {
                    "transition": {"from": "condition", "to": "commitment"},
                    "supports": "The target resolves the condition.",
                    "falsifies": "The target retains the condition.",
                },
            }
            portfolio = {
                "items": [
                    {
                        "diagnosis": {
                            "category": "conditional_acceptance",
                            "summary": "Acceptance remains conditional.",
                            "evidence": [{"evidence_id": "e0"}],
                        },
                        "contract": strategy_wire_contract(contract),
                        "applicability": selector_applicability_payload(None),
                    }
                ]
            }
            schema = {"type": "object"}
            schema_digest = digest_json(schema)
            selector_attempt = AttemptView(
                entity_id("pa", ["stateful-projection", "selector"]),
                {
                    "payload": {
                        "response_schema_digest": schema_digest,
                        "role_route_key": selector_slot["routes"][0]["route_key"],
                    }
                },
                {"payload": {}},
                {
                    "type": "model_call_succeeded",
                    "payload": {
                        "finish_status": "complete",
                        "text": json.dumps(portfolio),
                    },
                },
            )
            actions = tuple(
                LogicalActionView(
                    slot["action_id"],
                    unit_dict["unit_id"],
                    slot,
                    (selector_attempt,)
                    if slot["action_id"] == selector_slot["action_id"]
                    else (),
                )
                for slot in unit_dict["logical_action_slots"]
            )
            unit = WorkUnitView({"sequence": 1}, unit_dict, actions)
            panels = {
                plan["panel_id"]: PanelView(
                    panel_id=plan["panel_id"],
                    target_action_id=plan["target_action_id"],
                    observation_id=None,
                    candidate_id=None,
                    execution_id=None,
                    target_event_id=None,
                    triggered=False,
                    primary=None,
                    details=None,
                    invalid_member_actions=(),
                    member_validation_errors={},
                    terminated=False,
                )
                for plan in unit_dict["panel_plans"]
            }
            panels["parent-panel"] = PanelView(
                panel_id=entity_id("pn", ["stateful-projection", "parent"]),
                target_action_id="parent-action",
                observation_id=observation.observation_id,
                candidate_id=parent.candidate_id,
                execution_id=None,
                target_event_id=None,
                triggered=False,
                primary=SimpleNamespace(
                    score=Decimal("75"),
                    ownership="adopted",
                    procedure=JudgeProcedure(premise="adopted", inference="rejected"),
                    verdict=Verdict.AMBIGUOUS,
                    application=Application.ABSTRACT,
                    evidence=(
                        EvidenceExcerpt(observation.response, "supports"),
                        EvidenceExcerpt(observation.response, "resists"),
                    )
                ),
                details=None,
                invalid_member_actions=(),
                member_validation_errors={},
                terminated=True,
            )
            rounds, episodes, ordering = _derive_stateful_portfolios(
                run_id=run_id,
                manifest=_manifest(context),
                objective=resources.objective,
                work_units=(unit,),
                logical_actions={item.action_id: item for item in actions},
                target_results={},
                target_observations={"parent-action": observation},
                panel_views=panels,
                member_assessments={},
                execution_evidence={},
                conversation_roots={
                    root_id: ConversationRootView(root_id, root_key, (), (node.node_id,))
                },
                conversation_nodes={node.node_id: node},
                candidate_evidence={
                    parent.candidate_id: EvidenceRecord(
                        parent.candidate_id,
                        search_status=SearchStatus.EXPLORATORY_LEAD,
                        scores=(Decimal("75"),),
                    )
                },
                response_schemas={schema_digest: schema},
                remaining=RoleCallVector(100, 100, 100, 100),
                terminal_event=None,
            )
            self.assertEqual(len(rounds), 1)
            round_view = next(iter(rounds.values()))
            self.assertEqual(len(round_view.lanes), 1)
            self.assertEqual(len(round_view.assignments), 3)
            self.assertEqual(len(round_view.children), 3)
            self.assertEqual(len(unit_dict["proposal_slots"]), 7)
            self.assertEqual(
                sum(
                    slot["role"] == Role.TARGET.value
                    for slot in unit_dict["logical_action_slots"]
                ),
                7,
            )
            self.assertTrue(all(child.allocated for child in round_view.children))
            self.assertTrue(all(not child.dispatched for child in round_view.children))
            self.assertEqual(len(episodes), 1)
            episode = next(iter(episodes.values()))
            self.assertEqual(episode.parent_baseline_score, Decimal("75"))
            self.assertIs(episode.portfolio_intent, PortfolioIntent.ADVANCE)
            self.assertFalse(episode.preferred_in_run)
            self.assertEqual(ordering, ())
            with self.assertRaises(TypeError):
                rounds["mutation"] = round_view  # type: ignore[index]

            # Intent does not reduce Kmax. A response over the physical Kmax
            # remains a terminal attempted result but creates no lane.
            over_cap = json.loads(json.dumps(portfolio))
            second_item = json.loads(json.dumps(portfolio["items"][0]))
            second_item["contract"]["hypothesis"] += " Distinct custom hypothesis."
            over_cap["items"].append(second_item)
            third_item = json.loads(json.dumps(portfolio["items"][0]))
            third_item["contract"]["hypothesis"] += " Third custom hypothesis."
            over_cap["items"].append(third_item)
            selector_routes = selector_slot["routes"]
            exhausted_attempts = tuple(
                AttemptView(
                    entity_id("pa", ["stateful-projection", "over-cap", index]),
                    {
                        "payload": {
                            "response_schema_digest": schema_digest,
                            "role_route_key": selector_routes[index // 2][
                                "route_key"
                            ],
                        }
                    },
                    {"payload": {}},
                    {
                        "type": "model_call_succeeded",
                        "payload": {
                            "finish_status": "complete",
                            "text": json.dumps(over_cap),
                        },
                    },
                )
                for index in range(4)
            )
            exhausted_actions = tuple(
                LogicalActionView(
                    slot["action_id"],
                    unit_dict["unit_id"],
                    slot,
                    exhausted_attempts
                    if slot["action_id"] == selector_slot["action_id"]
                    else (),
                )
                for slot in unit_dict["logical_action_slots"]
            )
            exhausted_unit = WorkUnitView(
                {"sequence": 4}, unit_dict, exhausted_actions
            )
            exhausted_rounds, exhausted_episodes, _ = (
                _derive_stateful_portfolios(
                    run_id=run_id,
                    manifest=_manifest(context),
                    objective=resources.objective,
                    work_units=(exhausted_unit,),
                    logical_actions={
                        item.action_id: item for item in exhausted_actions
                    },
                    target_results={},
                    target_observations={"parent-action": observation},
                    panel_views=panels,
                    member_assessments={},
                    execution_evidence={},
                    conversation_roots={
                        root_id: ConversationRootView(
                            root_id, root_key, (), (node.node_id,)
                        )
                    },
                    conversation_nodes={node.node_id: node},
                    candidate_evidence={
                        parent.candidate_id: EvidenceRecord(
                            parent.candidate_id,
                            search_status=SearchStatus.EXPLORATORY_LEAD,
                            scores=(Decimal("75"),),
                        )
                    },
                    response_schemas={schema_digest: schema},
                    remaining=RoleCallVector(100, 100, 100, 100),
                    terminal_event=None,
                )
            )
            exhausted_view = next(iter(exhausted_rounds.values()))
            self.assertEqual(exhausted_view.lanes, ())
            self.assertTrue(exhausted_view.complete)
            self.assertEqual(exhausted_episodes, {})


if __name__ == "__main__":
    unittest.main()
