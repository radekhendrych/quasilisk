from __future__ import annotations

from quasilisk.domain import JudgeProcedure

from decimal import Decimal
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from integration_support import dryrun_config
from quasilisk.domain import (
    Application,
    ConversationNode,
    PortfolioIntent,
    RoleCallVector,
    SearchStatus,
    Verdict,
    entity_id,
    scientific_key,
)
from quasilisk.engine import (
    CandidatePlan,
    RunContext,
    _build_candidates,
    _entry_candidate_for_episode,
    _initialize_routes,
    _portfolio_roots_for_candidate,
    _portfolio_selector_request,
    _portfolio_current_assessment,
    _portfolio_strategy_evidence,
    _portfolio_strategy_selection_limit,
    _subject_rank_metadata,
    build_portfolio_round,
    load_resources,
)
from quasilisk.run_state import portfolio_current_assessment


class LinkedPortfolioConstructionTests(unittest.TestCase):
    def _context(self, root: Path, *, proposal_count: int = 7) -> RunContext:
        config = dryrun_config("fatalism", root / "runs")
        config["search"]["proposals_per_step"] = proposal_count
        resources = load_resources(config)
        run_id = entity_id("run", ["linked-portfolio"])
        context = RunContext(
            resources=resources,
            run_id=run_id,
            run_nonce="0" * 32,
            invocation_id=entity_id("iv", ["linked-portfolio"]),
            run_directory=root / run_id,
            candidates={},
            routes={},
        )
        _initialize_routes(context)
        context.candidates.update(_build_candidates(resources, run_id))
        return context

    @staticmethod
    def _deep_linked_state(context: RunContext):
        root_candidate = next(iter(context.candidates.values()))
        root_id = entity_id("cr", ["deep-linked-root"])
        root_key = scientific_key(["deep-linked-root"])
        root_node = ConversationNode(
            entity_id("cn", ["deep-linked", 0]),
            scientific_key(["deep-linked-node", 0]),
            root_id,
            None,
            0,
            root_candidate.candidate_id,
            entity_id("to", ["deep-linked", 0]),
        )
        nodes = {root_node.node_id: root_node}
        episodes = {}
        current_node = root_node
        linked_candidates = []
        linked_episodes = []
        for depth in (1, 2):
            strategy_key = scientific_key(["deep-linked-strategy", depth])
            strategy_id = entity_id("st", ["deep-linked-strategy", depth])
            candidate_key = scientific_key(["deep-linked-candidate", depth])
            candidate = CandidatePlan(
                entity_id("pc", ["deep-linked-candidate", depth]),
                candidate_key,
                f"linked user turn {depth}",
                dict(root_candidate.descriptors),
                "stateful_strategy",
                None,
                parent_node_id=current_node.node_id,
                parent_node_scientific_key=current_node.scientific_key,
                strategy_id=strategy_id,
                proposal_slot_scientific_key=scientific_key(
                    ["deep-linked-slot", depth]
                ),
            )
            node = ConversationNode(
                entity_id("cn", ["deep-linked", depth]),
                scientific_key(["deep-linked-node", depth]),
                root_id,
                current_node.node_id,
                depth,
                candidate.candidate_id,
                entity_id("to", ["deep-linked", depth]),
            )
            contract_payload = {"purpose": "repair", "depth": depth}
            child = SimpleNamespace(
                candidate_id=candidate.candidate_id,
                candidate_scientific_key=candidate.scientific_key,
                child_scientific_key=scientific_key(
                    ["deep-linked-child", depth]
                ),
                strategy_id=strategy_id,
                strategy_scientific_key=strategy_key,
                alignment_status="aligned",
                observation_id=node.target_observation_id,
                outcome=SimpleNamespace(
                    aligned=True,
                    resolved=True,
                    outcome_result="supports",
                    confirmed_objective_high=True,
                    confirmed_objective_application=False,
                    confirmed_objective_exceptional=False,
                    effective_score=Decimal("80"),
                ),
            )
            episode = SimpleNamespace(
                strategy_id=strategy_id,
                strategy_scientific_key=strategy_key,
                conversation_root_id=root_id,
                conversation_root_scientific_key=root_key,
                generation_parent_node_id=current_node.node_id,
                generation_parent_node_scientific_key=current_node.scientific_key,
                parent_baseline_score=Decimal("80"),
                parent_search_status=SearchStatus.PROMISING,
                portfolio_intent=PortfolioIntent.ADVANCE,
                contract=SimpleNamespace(
                    to_dict=lambda payload=contract_payload: dict(payload)
                ),
                children=(child,),
            )
            context.candidates[candidate.scientific_key] = candidate
            nodes[node.node_id] = node
            episodes[strategy_id] = episode
            linked_candidates.append(candidate)
            linked_episodes.append(episode)
            current_node = node
        observations = {
            node.target_observation_id: SimpleNamespace(
                observation_id=node.target_observation_id,
                scientific_key=scientific_key(
                    ["deep-linked-observation", node.node_id]
                ),
                response=f"Supported branch evidence for {node.node_id}.",
            )
            for node in nodes.values()
        }
        panels = {}
        for index, node in enumerate(nodes.values()):
            observation = observations[node.target_observation_id]
            primary = SimpleNamespace(
                observation_id=observation.observation_id,
                verdict=Verdict.AMBIGUOUS,
                application=Application.ABSTRACT,
                evidence=(
                    SimpleNamespace(
                        quote=observation.response,
                        polarity="supports",
                    ),
                ),
             ownership="adopted", procedure=JudgeProcedure(premise="adopted", inference="endorsed" if index else "rejected", reservation="none"), justification="Exact test evidence.")
            panels[f"panel-{index}"] = SimpleNamespace(
                observation_id=observation.observation_id,
                candidate_id=node.candidate_id,
                resolved=True,
                effective_score=Decimal(60 + index * 10),
                assessment=None,
                details=None,
                primary=primary,
            )
        state = SimpleNamespace(
            strategy_episodes=episodes,
            conversation_nodes=nodes,
            conversation_roots={root_id: SimpleNamespace(scientific_key=root_key)},
            archive=SimpleNamespace(cells={}, entries=()),
            panel_views=panels,
            target_observations=observations,
            member_assessments={},
            logical_actions={},
            remaining=RoleCallVector(50, 100, 50, 50),
        )
        block_id = entity_id("sb", ["deep-linked-entry"])
        block = SimpleNamespace(
            candidate_id=root_candidate.candidate_id,
            candidate_scientific_key=root_candidate.scientific_key,
            block_id=block_id,
            block_terminated=True,
            status=SearchStatus.PROMISING,
            effective_scores=(Decimal("70"), Decimal("75"), Decimal("80")),
        )
        state.candidate_evidence = {
            root_candidate.candidate_id: SimpleNamespace(
                subject_id=root_candidate.candidate_id,
                screening_block_id=block_id,
                search_status=SearchStatus.PROMISING,
                # The aggregate evidence record may later include verification
                # scores; entry inheritance must use the screen-only projection.
                scores=(Decimal("1"), Decimal("99"), Decimal("100")),
            )
        }
        state.screening_blocks = {root_candidate.candidate_id: block}
        return (
            state,
            block,
            root_candidate,
            tuple(linked_candidates),
            tuple(linked_episodes),
            tuple(nodes.values()),
        )

    def test_retained_weak_candidate_selects_one_intent_bound_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = self._context(Path(temporary), proposal_count=3)
            candidate = next(iter(context.candidates.values()))
            nodes = {}
            roots = {}
            observations = {}
            panels = {}
            for label, polarity, score in (
                ("recovery", "resists", Decimal("90")),
                ("advance", "supports", Decimal("30")),
            ):
                root_id = entity_id("cr", ["intent-root", label])
                observation_id = entity_id("to", ["intent-root", label])
                response = f"{label} evidence"
                node = ConversationNode(
                    entity_id("cn", ["intent-root", label]),
                    scientific_key(["intent-root-node", label]),
                    root_id,
                    None,
                    0,
                    candidate.candidate_id,
                    observation_id,
                )
                primary = SimpleNamespace(
                    observation_id=observation_id,
                    evidence=(SimpleNamespace(quote=response, polarity=polarity),),
                 ownership="adopted", procedure=JudgeProcedure(premise="adopted", inference="endorsed", reservation="none"), justification="Exact test evidence.", verdict=Verdict.AMBIGUOUS, application=Application.ABSTRACT)
                nodes[node.node_id] = node
                roots[root_id] = SimpleNamespace(
                    scientific_key=scientific_key(["intent-root", label])
                )
                observations[label] = SimpleNamespace(
                    observation_id=observation_id,
                    scientific_key=scientific_key(["intent-observation", label]),
                    response=response,
                )
                panels[label] = SimpleNamespace(
                    observation_id=observation_id,
                    resolved=True,
                    effective_score=score,
                    details=None,
                    primary=primary,
                 assessment=None)
            state = SimpleNamespace(
                archive=SimpleNamespace(
                    entries=(SimpleNamespace(subject_id=candidate.candidate_id),)
                ),
                candidate_evidence={
                    candidate.candidate_id: SimpleNamespace(
                        subject_id=candidate.candidate_id,
                        screening_block_id=None,
                        search_status=SearchStatus.OBSERVED,
                    )
                },
                screening_blocks={},
                conversation_nodes=nodes,
                conversation_roots=roots,
                target_observations=observations,
                panel_views=panels,
                member_assessments={},
            )

            selected = _portfolio_roots_for_candidate(context, state, candidate)

            self.assertEqual(len(selected), 1)
            self.assertIs(selected[0].portfolio_intent, PortfolioIntent.ADVANCE)
            self.assertEqual(
                selected[0].node_id,
                next(
                    node.node_id
                    for node in nodes.values()
                    if "advance" in observations[
                        next(
                            key
                            for key, value in observations.items()
                            if value.observation_id == node.target_observation_id
                        )
                    ].response
                ),
            )

            panels["advance"].primary.evidence = (
                SimpleNamespace(quote="advance evidence", polarity="resists"),
            )
            selected = _portfolio_roots_for_candidate(context, state, candidate)
            self.assertIs(selected[0].portfolio_intent, PortfolioIntent.RECOVER)
            self.assertEqual(selected[0].node_id, nodes[next(iter(nodes))].node_id)

    def test_two_step_link_inherits_entry_screen_and_freezes_exact_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = self._context(Path(temporary), proposal_count=7)
            state, _, entry, candidates, episodes, nodes = (
                self._deep_linked_state(context)
            )
            leaf = candidates[-1]
            leaf_node = nodes[-1]
            source_episode = episodes[-1]

            roots = _portfolio_roots_for_candidate(context, state, leaf)
            rank = _subject_rank_metadata(context, state, leaf)
            next_episode = SimpleNamespace(
                strategy_id=entity_id("st", ["deep-linked-next"]),
                conversation_root_id=source_episode.conversation_root_id,
                conversation_root_scientific_key=(
                    source_episode.conversation_root_scientific_key
                ),
                generation_parent_node_id=leaf_node.node_id,
                generation_parent_node_scientific_key=leaf_node.scientific_key,
                parent_baseline_score=Decimal("80"),
                parent_search_status=SearchStatus.PROMISING,
                portfolio_intent=PortfolioIntent.ADVANCE,
            )
            self.assertIs(
                _entry_candidate_for_episode(context, state, next_episode),
                entry,
            )

            self.assertEqual(
                (
                    roots[0].node_id,
                    roots[0].node_scientific_key,
                    roots[0].conversation_root_id,
                    roots[0].conversation_root_scientific_key,
                    roots[0].portfolio_intent,
                ),
                (
                    leaf_node.node_id,
                    leaf_node.scientific_key,
                    leaf_node.root_id,
                    source_episode.conversation_root_scientific_key,
                    PortfolioIntent.ADVANCE,
                ),
            )
            self.assertEqual(rank[2], Decimal("80"))

            unit = build_portfolio_round(
                context,
                leaf,
                parent_unit_id=entity_id("wu", ["deep-linked-parent"]),
                conversation_root_id=leaf_node.root_id,
                conversation_root_scientific_key=(
                    source_episode.conversation_root_scientific_key
                ),
                parent_node_id=leaf_node.node_id,
                parent_node_scientific_key=leaf_node.scientific_key,
                portfolio_intent=PortfolioIntent.ADVANCE,
            )
            shallower = build_portfolio_round(
                context,
                candidates[0],
                parent_unit_id=entity_id("wu", ["deep-linked-parent"]),
                conversation_root_id=leaf_node.root_id,
                conversation_root_scientific_key=(
                    source_episode.conversation_root_scientific_key
                ),
                parent_node_id=nodes[-2].node_id,
                parent_node_scientific_key=nodes[-2].scientific_key,
                portfolio_intent=PortfolioIntent.ADVANCE,
            )
            self.assertNotEqual(unit.unit_scientific_key, shallower.unit_scientific_key)
            self.assertEqual(
                unit.design["generation_parent_node_scientific_key"],
                leaf_node.scientific_key,
            )

            view = SimpleNamespace(
                unit=unit.to_dict(),
                unit_id=unit.unit_id,
                start_event={"sequence": 10},
            )
            state.work_units = (view,)
            state.portfolio_rounds = {
                view.unit_id: SimpleNamespace(
                    round_index=0,
                    complete=False,
                    lanes=(),
                )
            }
            state.current_strategy_qualification_snapshots = ()
            with (
                patch("quasilisk.engine._generation_brief", return_value={}),
                patch(
                    "quasilisk.engine._portfolio_parent",
                    return_value=(leaf, object(), leaf_node.scientific_key),
                ),
                patch(
                    "quasilisk.engine._portfolio_branch_context",
                    return_value=([{"role": "user", "content": "exact branch"}], []),
                ),
                patch(
                    "quasilisk.engine._portfolio_strategy_evidence",
                    return_value=([], {}),
                ),
            ):
                request = _portfolio_selector_request(context, state, view)
            self.assertEqual(request["maximum_strategies"], 2)
            self.assertNotIn("strategy_id", request["linked_parent_strategy"])
            self.assertNotIn(
                "strategy_scientific_key", request["linked_parent_strategy"]
            )
            self.assertEqual(
                request["linked_parent_strategy"]["contract"],
                source_episode.contract.to_dict(),
            )
            request["linked_parent_strategy"]["contract"]["purpose"] = "changed"
            self.assertEqual(source_episode.contract.to_dict()["purpose"], "repair")

    def test_linked_root_requires_a_usable_outcome_and_can_replan_after_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = self._context(Path(temporary), proposal_count=7)
            state, _, _, candidates, episodes, _ = self._deep_linked_state(
                context
            )
            leaf = candidates[-1]
            source_child = episodes[-1].children[0]
            source_child.alignment_status = "invalid_contract"
            self.assertEqual(
                _portfolio_roots_for_candidate(context, state, leaf), ()
            )
            source_child.alignment_status = "aligned"
            source_child.outcome.outcome_result = "falsifies"
            self.assertNotEqual(
                _portfolio_roots_for_candidate(context, state, leaf), ()
            )
            source_child.outcome.outcome_result = None
            source_child.outcome.confirmed_objective_high = True
            self.assertEqual(
                _portfolio_roots_for_candidate(context, state, leaf), ()
            )
            source_child.outcome.outcome_result = "supports"
            self.assertNotEqual(
                _portfolio_roots_for_candidate(context, state, leaf), ()
            )

    def test_selector_receives_conservative_panel_stage_and_all_member_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = self._context(Path(temporary), proposal_count=3)
            turn_ref = scientific_key(["panel-stage-turn"])
            observation_id = entity_id("to", ["panel-stage-turn"])
            observation = SimpleNamespace(
                observation_id=observation_id,
                scientific_key=turn_ref,
                response="The opening holds, but the stated defense remains.",
            )
            primary = SimpleNamespace(
                assessment_id=entity_id("ma", ["panel-stage-primary"]),
                observation_id=observation_id,
                logical_member_slot="primary",
                verdict=Verdict.DEMONSTRATED,
                application=Application.ABSTRACT,
                evidence=(
                    SimpleNamespace(
                        quote="The opening holds, but the stated defense remains.",
                        polarity="supports",
                    ),
                ),
             ownership="adopted", procedure=JudgeProcedure(premise="adopted", inference="endorsed", reservation="none"), justification="Exact test evidence.")
            secondary_a = SimpleNamespace(
                assessment_id=entity_id("ma", ["panel-stage-secondary-a"]),
                observation_id=observation_id,
                logical_member_slot="secondary_a",
                verdict=Verdict.DEMONSTRATED,
                application=Application.ACTION_GOVERNING,
                evidence=(
                    SimpleNamespace(
                        quote="The opening holds, but the stated defense remains.",
                        polarity="resists",
                    ),
                ),
             ownership="adopted", procedure=JudgeProcedure(premise="adopted", inference="endorsed", reservation="none"), justification="Exact test evidence.")
            secondary_b = SimpleNamespace(
                assessment_id=entity_id("ma", ["panel-stage-secondary-b"]),
                observation_id=observation_id,
                logical_member_slot="secondary_b",
                verdict=Verdict.DEMONSTRATED,
                application=Application.ACTION_GOVERNING,
                evidence=(
                    SimpleNamespace(
                        quote="The opening holds, but the stated defense remains.",
                        polarity="supports",
                    ),
                ),
             ownership="adopted", procedure=JudgeProcedure(premise="adopted", inference="endorsed", reservation="none"), justification="Exact test evidence.")
            aggregate = SimpleNamespace(
                availability=SimpleNamespace(value="complete"),
                objective_failure="demonstrated",
                confirmed_objective_application=False,
                member_ids=tuple(
                    member.assessment_id
                    for member in (primary, secondary_a, secondary_b)
                ),
            )
            panel = SimpleNamespace(
                observation_id=observation_id,
                primary=primary,
                details=SimpleNamespace(assessment=aggregate),
             assessment=aggregate, resolved=True)
            state = SimpleNamespace(
                target_observations={"target": observation},
                panel_views={"panel": panel},
                member_assessments={
                    "primary": primary,
                    "secondary_a": secondary_a,
                    "secondary_b": secondary_b,
                },
            )
            assessment = _portfolio_current_assessment(
                state, ({"turn_ref": turn_ref, "text": observation.response},)
            )
            self.assertEqual(
                {key: assessment[key] for key in ("verdict", "application", "ownership")},
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
                    context, state, SimpleNamespace()
                )
            self.assertEqual(
                evidence,
                [
                    {
                        "evidence_id": "e0",
                        "text": "The opening holds, but the stated defense remains.",
                        "polarities": ["resists", "supports"],
                    }
                ],
            )
            self.assertEqual(
                polarities[
                    (turn_ref, "The opening holds, but the stated defense remains.")
                ],
                frozenset({"resists", "supports"}),
            )

            aggregate.confirmed_objective_application = True
            self.assertEqual(
                portfolio_current_assessment(panel, state.member_assessments),
                (Verdict.DEMONSTRATED, Application.ACTION_GOVERNING),
            )
            self.assertEqual(
                _portfolio_current_assessment(
                    state,
                    ({"turn_ref": turn_ref, "text": observation.response},),
                )["application"],
                "action_governing",
            )

    def test_search_status_does_not_reduce_the_frozen_strategy_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = self._context(Path(temporary), proposal_count=7)
            parent = next(iter(context.candidates.values()))
            unit = SimpleNamespace(
                unit_id=entity_id("wu", ["selection-limit"]),
                start_event={"sequence": 10},
                unit={
                    "kind": "portfolio_round",
                    "design": {
                        "round_index": 0,
                        "maximum_strategy_count": 2,
                        "portfolio_intent": PortfolioIntent.RECOVER.value,
                    }
                }
            )
            block_id = entity_id("sb", ["selection-limit"])
            record = SimpleNamespace(
                subject_id=parent.candidate_id,
                screening_block_id=block_id,
                search_status=SearchStatus.EXPLORATORY_LEAD,
            )
            block = SimpleNamespace(
                candidate_id=parent.candidate_id,
                candidate_scientific_key=parent.scientific_key,
                block_id=block_id,
                status=SearchStatus.EXPLORATORY_LEAD,
                block_terminated=True,
                effective_scores=(70, 80),
            )
            state = SimpleNamespace(
                candidate_evidence={parent.candidate_id: record},
                screening_blocks={parent.candidate_id: block},
                work_units=(unit,),
                portfolio_rounds={
                    unit.unit_id: SimpleNamespace(
                        round_index=0,
                        complete=False,
                        lanes=(),
                    )
                },
            )
            with (
                patch("quasilisk.engine._generation_brief", return_value={}),
                patch(
                    "quasilisk.engine._portfolio_parent",
                    return_value=(parent, object(), scientific_key(["parent"])),
                ),
            ):
                self.assertEqual(
                    _portfolio_strategy_selection_limit(context, state, unit), 2
                )
                record.search_status = SearchStatus.PROMISING
                block.status = SearchStatus.PROMISING
                self.assertEqual(
                    _portfolio_strategy_selection_limit(context, state, unit), 2
                )


if __name__ == "__main__":
    unittest.main()
