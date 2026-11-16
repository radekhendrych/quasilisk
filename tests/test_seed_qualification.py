"""Pure Section 13 seed/scenario qualification and snapshot tests."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import unittest

from quasilisk.domain import (
    StructuredRoleScope,
    DomainError,
    PanelAssessment,
    PanelAvailability,
    digest_json,
    entity_id,
    sha256_bytes,
)
from quasilisk.evidence import (
    EntrySeedSubject,
    ExecutionEvidence,
    NamedScopeEvidence,
    ObjectiveScopeEvidence,
    PerturbatorRoleScope,
    ProviderIdentity,
    ScenarioRecipe,
    ScenarioSeedSubject,
    SeedEvidenceEvent,
    SeedEvidenceSource,
    SeedQualificationContext,
    SeedVerificationBlock,
    SeedVerificationExecution,
    SeedVerificationRule,
    evaluate_seed_qualification,
)
from quasilisk.strategy import parse_strategy_template


DESCRIPTORS = (
    ("objective_route", "route"),
    ("argument_move", "answer_objection"),
    ("application_scope", "reasoning_policy"),
)
VOCABULARY = {
    "objective_route": ("route", "alternate"),
    "argument_move": ("answer_objection", "test_inference"),
    "application_scope": ("reasoning_policy", "action_policy"),
}


def _digest(label: str) -> str:
    return digest_json(["seed-qualification-test", label])


def _template():
    return parse_strategy_template(
        {
            "purpose": "repair",
            "applicability": {
                "diagnosis": "conditional_acceptance",
                "requires": "A stated condition remains.",
                "excludes": "The condition has already been resolved.",
            },
            "move": {
                "act": "repair_grounded_gap",
                "target_kind": "condition",
                "descriptors": dict(DESCRIPTORS),
                "blueprint": "Resolve only the stated condition.",
            },
            "hypothesis": "Resolving the condition should permit progress.",
            "preserve": [{"kind": "meaning", "statement": "Preserve meaning."}],
            "vary": [{"dimension": "wording", "instruction": "Vary wording."}],
            "success": {
                "transition": {"from": "conditional", "to": "resolved"},
                "supports": "The target resolves the condition.",
                "falsifies": "The target retains the condition.",
            },
        },
        VOCABULARY,
        ("conditional_acceptance", "other"),
    )


def _panel(index: int, *, high: bool) -> PanelAssessment:
    score = Decimal("80") if high else Decimal("40")
    return PanelAssessment(
        panel_id=entity_id("pn", ["seed", index]),
        observation_id=entity_id("to", ["seed", index]),
        availability=PanelAvailability.COMPLETE,
        member_ids=(
            entity_id("ma", ["seed", index, 0]),
            entity_id("ma", ["seed", index, 1]),
        ),
        weighted_numerator=int(score),
        weighted_denominator=1,
        weighted_score=score,
        score_range=(score, score),
        panel_confirmed_high=high,
        confirmed_objective_high=high,
        panel_confirmed_exceptional=False,
        confirmed_objective_exceptional=False,
        confirmed_objective_application=False,
        objective_failure="failure" if high else None,
        objective_application=None,
        ambiguity_kind=None,
    )


def _execution(
    index: int,
    *,
    high: bool,
    fresh: bool = True,
    end_to_end: bool = False,
    injected: bool = False,
    resolved: bool = True,
) -> SeedVerificationExecution:
    evidence = (
        ExecutionEvidence(
            execution_id=entity_id("ex", ["seed", index]),
            terminated=True,
            resolved=True,
            primary_score=Decimal("80") if high else Decimal("40"),
            panel_required=True,
            panel=_panel(index, high=high),
        )
        if resolved
        else ExecutionEvidence(
            execution_id=entity_id("ex", ["seed", index]),
            terminated=True,
            resolved=False,
            panel_required=True,
        )
    )
    return SeedVerificationExecution(evidence, fresh, end_to_end, injected)


def _block(
    *,
    high_count: int = 4,
    end_to_end: bool = False,
    injected_index: int | None = None,
    stale_index: int | None = None,
    unresolved_index: int | None = None,
) -> SeedVerificationBlock:
    return SeedVerificationBlock(
        entity_id("sb", ["seed", "verification"]),
        SeedVerificationRule(5),
        tuple(
            _execution(
                index,
                high=index < high_count,
                fresh=index != stale_index,
                end_to_end=end_to_end,
                injected=index == injected_index,
                resolved=index != unresolved_index,
            )
            for index in range(5)
        ),
        (
            ("judge_primary", _digest("judge-primary")),
            ("panel", _digest("panel")),
        ),
    )


def _component(scope: str, label: str | None = None) -> StructuredRoleScope:
    scope_label = label or scope
    return StructuredRoleScope(
        scope,
        _digest(f"{scope_label}-key"),
        "selector-model",
        "json_schema",
        (("schema", _digest(f"{scope_label}-schema")),),
    )


def _perturbator(label: str = "current") -> PerturbatorRoleScope:
    return PerturbatorRoleScope(
        _digest(f"{label}-route"),
        "perturbator-model",
        _digest(f"{label}-profile"),
        "json_schema",
        (
            ("strategy", _digest(f"{label}-strategy-prompt")),
            ("strategy_correction", _digest(f"{label}-correction-prompt")),
        ),
    )


def _source() -> SeedEvidenceSource:
    return SeedEvidenceSource(
        entity_id("run", ["seed", "source"]),
        "runs/source/events.jsonl",
        (
            SeedEvidenceEvent(
                entity_id("ev", ["seed", 1]),
                1,
                _digest("event-hash-1"),
                "2026-08-20T10:00:00.000001Z",
            ),
            SeedEvidenceEvent(
                entity_id("ev", ["seed", 9]),
                9,
                _digest("event-hash-9"),
                "2026-08-20T10:01:00.000009Z",
            ),
        ),
    )


def _context(
    components: tuple[StructuredRoleScope, ...],
    *,
    perturbator: PerturbatorRoleScope | None = None,
) -> SeedQualificationContext:
    return SeedQualificationContext(
        ObjectiveScopeEvidence(
            "test_objective",
            _digest("plugin"),
            _digest("objective-scope"),
        ),
        _digest("judge-protocol"),
        (
            NamedScopeEvidence("judge_primary", _digest("judge-primary")),
            NamedScopeEvidence("panel", _digest("panel")),
        ),
        components,
        perturbator,
    )


def _entry(*, generated: bool = True, identity: ProviderIdentity | None = None):
    return EntrySeedSubject(
        entity_id("sd", ["entry"]),
        "Exact entry prompt.",
        DESCRIPTORS,
        "test_objective",
        _digest("plugin"),
        _digest("target-profile"),
        _digest("descriptor-taxonomy"),
        identity or ProviderIdentity("provider/target-alias", "target-revision-1"),
        generated,
        _digest("probe_review-key") if generated else None,
    )


def _scenario(
    *,
    perturbator: PerturbatorRoleScope | None = None,
    probe_key: str | None = None,
    outcome_key: str | None = None,
    binding_key: str | None = None,
):
    frozen_perturbator = perturbator or _perturbator()
    return ScenarioSeedSubject(
        entity_id("sd", ["scenario"]),
        "Exact scenario entry prompt.",
        DESCRIPTORS,
        "test_objective",
        _digest("plugin"),
        _digest("target-profile"),
        _digest("descriptor-taxonomy"),
        ProviderIdentity("provider/target-alias", "target-revision-1"),
        ScenarioRecipe(
            sha256_bytes(b"Exact scenario entry prompt."),
            _digest("target-profile"),
            (_template(),),
            DESCRIPTORS,
            2,
            1,
            ("anchor",),
            (
                ("outcome_review", outcome_key or _digest("outcome_review-key")),
                ("probe_review", probe_key or _digest("probe_review-key")),
                (
                    "template_binding",
                    binding_key or _digest("template_binding-key"),
                ),
            ),
            frozen_perturbator,
        ),
    )


class EntrySeedQualificationTests(unittest.TestCase):
    def test_generated_entry_qualifies_and_materializes_the_closed_snapshot(self) -> None:
        probe = _component("probe_review")
        subject = _entry()
        block = _block()
        context = _context((probe,))
        source = _source()
        decision = evaluate_seed_qualification(subject, block, context, source)

        self.assertTrue(decision.qualified)
        self.assertEqual(decision.failed_gates, ())
        self.assertIsNotNone(decision.snapshot)
        assert decision.snapshot is not None
        snapshot = decision.snapshot.to_dict()
        self.assertEqual(snapshot["type"], "entry_seed")
        self.assertIsNone(snapshot["scenario_recipe"])
        self.assertEqual(snapshot["observed_success_fraction"], "0.8")
        self.assertEqual(snapshot["source_checkpoint"]["sequence"], 9)
        self.assertEqual(
            snapshot["source_checkpoint"]["event_hash"],
            source.last_required_event.event_hash,
        )
        self.assertEqual(snapshot["qualified_at"], source.last_required_event.timestamp_utc)
        self.assertEqual(
            set(snapshot),
            {
                "seed_id",
                "type",
                "entry_prompt",
                "scenario_recipe",
                "objective_id",
                "plugin_digest",
                "target_generation_profile_digest",
                "descriptors",
                "verification_block_id",
                "verification_rule",
                "planned_count",
                "resolved_count",
                "high_count",
                "observed_success_fraction",
                "judge_protocol_digest",
                "qualification_digests",
                "evidence_event_ids",
                "qualified_at",
                "compatibility_key",
                "identity_token",
                "identity_unverifiable",
                "source_checkpoint",
            },
        )
        expected_key = digest_json(
            [
                "verified_entry_seed_compatibility",
                subject.objective_id,
                subject.plugin_digest,
                subject.target_generation_profile_digest,
                subject.target_identity.token,
                subject.descriptor_taxonomy_digest,
                [list(item) for item in block.judge_panel_scope_keys],
                block.rule.to_dict(),
                subject.assigned_probe_review_scope_key,
            ]
        )
        self.assertEqual(decision.compatibility_key, expected_key)

    def test_every_promotion_gate_is_independent_and_fail_closed(self) -> None:
        probe = _component("probe_review")
        cases = (
            (
                "missing generated probe scope",
                _entry(),
                _block(),
                _context(()),
                "component_scope_set_mismatch",
            ),
            (
                "generated probe scope drift",
                _entry(),
                _block(),
                _context((_component("probe_review", "drifted"),)),
                "probe_review_scope_mismatch",
            ),
            (
                "verification judge scope drift",
                _entry(),
                replace(
                    _block(),
                    judge_panel_scope_keys=(
                        ("judge_primary", _digest("other-judge")),
                        ("panel", _digest("panel")),
                    ),
                ),
                _context((probe,)),
                "judge_panel_scope_mismatch",
            ),
            (
                "extra trusted probe scope",
                _entry(generated=False),
                _block(),
                _context((probe,)),
                "component_scope_set_mismatch",
            ),
            (
                "not fresh",
                _entry(),
                _block(stale_index=0),
                _context((probe,)),
                "verification_not_fresh",
            ),
            (
                "unresolved",
                _entry(),
                _block(unresolved_index=4),
                _context((probe,)),
                "verification_not_fully_resolved",
            ),
            (
                "three of five",
                _entry(),
                _block(high_count=3),
                _context((probe,)),
                "verification_strict_sixty_percent_gate_failed",
            ),
        )
        for label, subject, block, context, failure in cases:
            with self.subTest(label=label):
                decision = evaluate_seed_qualification(subject, block, context, _source())
                self.assertFalse(decision.qualified)
                self.assertIn(failure, decision.failed_gates)
                self.assertIsNone(decision.snapshot)

    def test_effective_and_requested_identity_tokens_never_alias(self) -> None:
        probe = _component("probe_review")
        effective = evaluate_seed_qualification(
            _entry(identity=ProviderIdentity("provider/alias", "immutable-revision")),
            _block(),
            _context((probe,)),
            _source(),
        )
        requested = evaluate_seed_qualification(
            _entry(identity=ProviderIdentity("provider/alias", None)),
            _block(),
            _context((probe,)),
            _source(),
        )
        assert effective.snapshot is not None and requested.snapshot is not None
        self.assertEqual(effective.snapshot.identity_token, "effective:immutable-revision")
        self.assertFalse(effective.snapshot.identity_unverifiable)
        self.assertEqual(requested.snapshot.identity_token, "requested:provider/alias")
        self.assertTrue(requested.snapshot.identity_unverifiable)
        self.assertNotEqual(effective.compatibility_key, requested.compatibility_key)


class ScenarioSeedQualificationTests(unittest.TestCase):
    def test_scenario_binds_recipe_role_scopes_and_materializes_templates(self) -> None:
        probe = _component("probe_review")
        outcome = _component("outcome_review")
        perturbator = _perturbator()
        subject = _scenario(
            perturbator=perturbator,
            probe_key=probe.scope_key,
            outcome_key=outcome.scope_key,
        )
        block = _block(end_to_end=True)
        context = _context((outcome, probe), perturbator=perturbator)
        decision = evaluate_seed_qualification(subject, block, context, _source())

        self.assertTrue(decision.qualified)
        assert decision.snapshot is not None
        snapshot = decision.snapshot.to_dict()
        self.assertEqual(snapshot["type"], "scenario_recipe")
        self.assertEqual(
            snapshot["scenario_recipe"]["ordered_templates"],
            [_template().to_dict()],
        )
        expected_key = digest_json(
            [
                "verified_scenario_seed_compatibility",
                subject.objective_id,
                subject.plugin_digest,
                subject.target_generation_profile_digest,
                subject.target_identity.token,
                subject.descriptor_taxonomy_digest,
                [list(item) for item in block.judge_panel_scope_keys],
                subject.recipe.content_hash,
                [item.content_hash for item in subject.recipe.templates],
                {
                    "designated_realization_policies": list(
                        subject.recipe.designated_realization_policies
                    ),
                    "maximum_target_turns": subject.recipe.maximum_target_turns,
                    "maximum_strategy_rounds": subject.recipe.maximum_strategy_rounds,
                },
                [list(item) for item in subject.recipe.selector_scope_keys],
                perturbator.key_material(),
                dict(subject.measurement_cell),
                block.rule.to_dict(),
            ]
        )
        self.assertEqual(decision.compatibility_key, expected_key)

    def test_scenario_key_covers_the_complete_entry_and_verification_scope(self) -> None:
        probe = _component("probe_review")
        outcome = _component("outcome_review")
        perturbator = _perturbator()
        subject = _scenario(
            perturbator=perturbator,
            probe_key=probe.scope_key,
            outcome_key=outcome.scope_key,
        )
        context = _context((outcome, probe), perturbator=perturbator)
        block = _block(end_to_end=True)
        baseline = evaluate_seed_qualification(
            subject, block, context, _source()
        ).compatibility_key

        subject_variants = (
            (
                replace(subject, plugin_digest=_digest("other-plugin")),
                replace(
                    context,
                    objective=replace(
                        context.objective,
                        plugin_digest=_digest("other-plugin"),
                    ),
                ),
            ),
            (
                replace(
                    subject,
                    descriptor_taxonomy_digest=_digest("other-taxonomy"),
                ),
                context,
            ),
            (
                replace(
                    subject,
                    target_identity=ProviderIdentity(
                        subject.target_identity.requested_alias,
                        "other-target-revision",
                    ),
                ),
                context,
            ),
        )
        for variant, variant_context in subject_variants:
            with self.subTest(subject=variant):
                self.assertNotEqual(
                    evaluate_seed_qualification(
                        variant, block, variant_context, _source()
                    ).compatibility_key,
                    baseline,
                )

        changed_judges = replace(
            block,
            judge_panel_scope_keys=(
                ("judge_primary", _digest("other-judge")),
                ("panel", _digest("panel")),
            ),
        )
        self.assertNotEqual(
            evaluate_seed_qualification(
                subject, changed_judges, context, _source()
            ).compatibility_key,
            baseline,
        )

    def test_scenario_rejects_scope_drift_non_e2e_and_target_message_injection(self) -> None:
        probe = _component("probe_review")
        outcome = _component("outcome_review")
        perturbator = _perturbator()
        subject = _scenario(
            perturbator=perturbator,
            probe_key=probe.scope_key,
            outcome_key=outcome.scope_key,
        )
        cases = (
            (
                "Perturbator drift",
                _block(end_to_end=True),
                _context((outcome, probe), perturbator=_perturbator("drifted")),
                "perturbator_scope_mismatch",
            ),
            (
                "Selector drift",
                _block(end_to_end=True),
                _context((outcome, _component("probe_review", "drifted")), perturbator=perturbator),
                "selector_scope_mismatch",
            ),
            (
                "not end to end",
                _block(end_to_end=False),
                _context((outcome, probe), perturbator=perturbator),
                "scenario_not_end_to_end",
            ),
            (
                "message injection",
                _block(end_to_end=True, injected_index=2),
                _context((outcome, probe), perturbator=perturbator),
                "scenario_target_message_injection",
            ),
        )
        for label, block, context, failure in cases:
            with self.subTest(label=label):
                decision = evaluate_seed_qualification(subject, block, context, _source())
                self.assertFalse(decision.qualified)
                self.assertIn(failure, decision.failed_gates)
                self.assertIsNone(decision.snapshot)


class SeedContractClosureTests(unittest.TestCase):
    def test_source_checkpoint_is_exactly_the_last_required_evidence_event(self) -> None:
        source = _source()
        self.assertEqual(
            source.checkpoint_dict(),
            {
                "source_run_id": source.source_run_id,
                "sequence": 9,
                "event_hash": source.required_events[-1].event_hash,
                "source_path": source.source_path,
            },
        )
        with self.assertRaisesRegex(DomainError, "increasing"):
            SeedEvidenceSource(
                source.source_run_id,
                source.source_path,
                tuple(reversed(source.required_events)),
            )
        with self.assertRaisesRegex(DomainError, "real UTC"):
            replace(source.required_events[0], timestamp_utc="2026-02-30T10:00:00.000001Z")

    def test_mismatched_objective_and_entry_only_scenario_scope_are_rejected(self) -> None:
        probe = _component("probe_review")
        wrong_objective = replace(
            _context((probe,)).objective,
            objective_id="other_objective",
        )
        with self.assertRaisesRegex(DomainError, "disagree"):
            evaluate_seed_qualification(
                _entry(),
                _block(),
                replace(_context((probe,)), objective=wrong_objective),
                _source(),
            )
        with self.assertRaisesRegex(DomainError, "scenario-only"):
            evaluate_seed_qualification(
                _entry(),
                _block(),
                _context((probe,), perturbator=_perturbator()),
                _source(),
            )


if __name__ == "__main__":
    unittest.main()
