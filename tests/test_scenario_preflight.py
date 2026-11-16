"""Source-template rejection is bounded and precedes independent target repeats."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import signal
import tempfile
import unittest
from unittest.mock import patch

from integration_support import dryrun_config, assert_complete_call_audit
from test_finisher_end_to_end import FinisherProvider, request_data, ROOT
from test_funded_adaptive_experiment import response
from quasilisk.engine import run_new, resume_run, validate_run_evidence
from quasilisk.events import read_events, JournalCorrupt
from quasilisk.run_state import _derive_scenario_preflights


class SourceBindingProvider(FinisherProvider):
    def __init__(self, behavior):
        super().__init__('nihilism_susceptibility')
        self.behavior = behavior
        self.bindings = []

    def complete(self, request):
        if request.purpose.value != 'template_binding':
            return super().complete(request)
        self.bindings.append(request)
        if self.behavior == 'interrupt_pass' and len(self.bindings) == 1:
            result = super().complete(request)
            signal.raise_signal(signal.SIGINT)
            return result
        self.requests.append(request)
        return response(request, {'binding': 'b0',
            'status': 'applicable' if self.behavior == 'invalid' else 'inapplicable',
            'evidence': [{'evidence_id': 'e0'}],
            'reason': 'The frozen exclusion incorrectly excludes the source parent.',
            'contract': None})


class ScenarioPreflightTests(unittest.TestCase):
    def exercise(self, behavior):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / 'repository'
            for name in ('quasilisk', 'objectives', 'templates', 'schemas', 'fixtures'):
                shutil.copytree(ROOT / name, repository / name, ignore=shutil.ignore_patterns('__pycache__'))
            (repository / 'library/seeds').mkdir(parents=True)
            (repository / 'library/strategies').mkdir(parents=True)
            config = dryrun_config('nihilism', root / 'runs')
            config['search']['maximum_followups'] = 1
            config['experiment'].update(exploration_batches=0, conversation_rounds=1, verification_blocks=1)
            provider = SourceBindingProvider(behavior)
            with patch('quasilisk.engine.create_provider', return_value=provider):
                run, reason = run_new(config, repository_root=repository, sleeper=lambda _: None)
                if behavior == 'interrupt_pass':
                    self.assertEqual(reason.value, 'operator_interrupted')
                    state = validate_run_evidence(run, repository_root=repository)
                    self.assertEqual(next(iter(state.scenario_preflights.values()))['fresh_target_calls'], 0)
                    _, reason = resume_run(run, repository_root=repository, sleeper=lambda _: None)
            self.assertIn(reason.value, {'completed_plan', 'completed_no_eligible_work'})
            state = validate_run_evidence(run, repository_root=repository)
            unit = next(u for u in state.work_units if u.unit['kind'] == 'verification')
            report = state.scenario_preflights[unit.unit_id]
            action_id = unit.unit['design']['source_preflight'][0]['action_id']
            source_calls = [r for r in provider.bindings if r.logical_action_id == action_id]
            self.assertEqual(len(source_calls), 4 if behavior == 'invalid' else 1)
            self.assertEqual(report['physical_selector_calls'], len(source_calls))
            self.assertEqual(report['status'], {'reject': 'rejected', 'invalid': 'unresolved', 'interrupt_pass': 'passed'}[behavior])
            self.assertEqual(report['fresh_target_calls'], 5 if behavior == 'interrupt_pass' else 0)
            self.assertIn(unit.unit_id, state.completed_work_unit_ids)
            self.assertFalse(any(d.qualified for d in state.seed_qualifications.values()))
            source_data = request_data(source_calls[0], 'UNTRUSTED_TEMPLATE_BINDING_REQUEST_JSON')
            scenario = state.scenarios[unit.unit['subject_scientific_key']]
            episode = state.strategy_episodes[scenario.source_step_episode_ids[0]]
            node = state.conversation_nodes[episode.generation_parent_node_id]
            observation = next(o for o in state.target_observations.values() if o.observation_id == node.target_observation_id)
            self.assertEqual(source_data['fresh_conversation'], [*(m.to_dict() for m in observation.messages),
                             {'role': 'assistant', 'content': observation.response}])
            self.assertEqual(source_data['strategy_template'], scenario.recipe.templates[0].to_dict())
            # Preflight source identity is authenticated against the episode,
            # even if a well-formed unit tries to cite a different parent.
            forged = unit.unit.copy()
            forged['design'] = {**unit.unit['design'], 'source_preflight': [
                {**unit.unit['design']['source_preflight'][0], 'parent_node_scientific_key': 'sha256:' + 'f' * 64}]}
            with self.assertRaisesRegex(JournalCorrupt, 'source parent chain'):
                _derive_scenario_preflights(run_id=state.run_id, manifest=state.manifest, objective=None,
                    work_units=[replace(unit, unit=forged)], scenarios=state.scenarios,
                    strategy_episodes=state.strategy_episodes, logical_actions=state.logical_actions,
                    target_results=state.target_results, target_observations=state.target_observations,
                    conversation_nodes=state.conversation_nodes, response_schemas={})
            assert_complete_call_audit(self, read_events(run))
            summary = json.loads((run / 'summary.json').read_text())
            self.assertEqual(summary['scenario_preflights'][unit.unit_id], report)
            before = (run / 'events.jsonl').read_bytes()
            resume_run(run, repository_root=repository, sleeper=lambda _: None)
            self.assertEqual((run / 'events.jsonl').read_bytes(), before)

    def test_overbroad_source_exclusion_is_template_rejection_without_target_calls(self):
        self.exercise('reject')

    def test_schema_valid_but_invalid_binding_exhausts_helpers_without_target_calls(self):
        self.exercise('invalid')

    def test_completed_source_check_is_not_dispatched_again_after_resume(self):
        self.exercise('interrupt_pass')


if __name__ == '__main__':
    unittest.main()
