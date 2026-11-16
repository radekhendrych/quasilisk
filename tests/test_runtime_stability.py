"""Fault-injection tests for bounded recovery, fresh-root gates and accounting."""
from dataclasses import replace
from decimal import Decimal
import json
import os
from pathlib import Path
import shutil
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from integration_support import dryrun_config, assert_complete_call_audit
from test_funded_adaptive_experiment import AdaptiveProvider, response, request_data, ROOT
from test_provider_foundation import make_request
from quasilisk.config import ProviderConfig
from quasilisk.domain import FinishStatus, canonical_json_bytes, digest_json
from quasilisk.engine import EngineError, run_new, resume_run, validate_run_evidence
from quasilisk.events import read_events
from quasilisk.provider import OpenAICompatibleProvider
from quasilisk.run_state import (
    ProviderCircuitOutcome, reduce_provider_circuit, portfolio_node_progress,
    portfolio_branch_closed, portfolio_transition,
)


class TruncatedJudgeProvider(AdaptiveProvider):
    recovered = False

    def complete(self, request):
        if request.purpose.value == 'target_sample':
            self.requests.append(request)
            self.target_calls += 1
            result = response(request, self.commitment + '\nFresh target ' + str(self.target_calls))
            return replace(result, cost='0.002', metadata={**result.metadata, 'reasoning_tokens': 7})
        if request.requested_model == 'scripted/judge-secondary-a' and not self.recovered:
            self.requests.append(request)
            return replace(response(request, ''), finish_status=FinishStatus.OUTPUT_LIMIT)
        return super().complete(request)


class FailedEntryProvider(AdaptiveProvider):
    saw_discovery_endpoint = False
    bad_entry_seen = False

    def complete(self, request):
        if request.purpose.value == 'target_sample':
            depth = sum(m.role == 'assistant' for m in request.messages)
            result = super().complete(request)
            if depth == 3:
                self.saw_discovery_endpoint = True
            if self.saw_discovery_endpoint and depth == 0 and not self.bad_entry_seen:
                self.bad_entry_seen = True
                return replace(result, text=result.text + '\nFault-injected fresh entry.')
            return result
        if request.purpose.value == 'judge_member':
            catalog = request_data(request, 'UNTRUSTED_CURRENT_TARGET_EVIDENCE_JSON')
            if any(item['text'] == 'Fault-injected fresh entry.' for item in catalog):
                self.requests.append(request)
                if request.requested_model == 'scripted/judge-replacement' and request.attempt_ordinal == 1:
                    signal.raise_signal(signal.SIGINT)
                return response(request, {'invalid': True})
        return super().complete(request)


class RuntimeStabilityTests(unittest.TestCase):
    def repository(self, root):
        repo = root / 'repository'
        for directory in ('quasilisk', 'objectives', 'templates', 'schemas', 'fixtures'):
            shutil.copytree(ROOT / directory, repo / directory, ignore=shutil.ignore_patterns('__pycache__'))
        (repo / 'library/seeds').mkdir(parents=True)
        (repo / 'library/strategies').mkdir(parents=True)
        return repo

    def test_truncated_judge_uses_no_same_cap_repair_and_profile_circuit_survives_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); repo = self.repository(root)
            config = dryrun_config('fatalism', root / 'runs')
            config['experiment'].update(conversation_rounds=3, exploration_batches=0)
            provider = TruncatedJudgeProvider()
            config['seed']['bootstrap_limit'] = 6
            with patch('quasilisk.engine.create_provider', return_value=provider):
                run, reason = run_new(config, repository_root=repo, sleeper=lambda _: None)
                self.assertEqual(reason.value, 'systemic_provider_failure')
                before = read_events(run)
                self.assertIn('judge_provider_route_failure_circuit:', before[-1]['payload']['detail'])
                failed = [r for r in provider.requests if r.requested_model == 'scripted/judge-secondary-a']
                self.assertEqual(len(failed), 3)
                self.assertTrue(all(r.attempt_ordinal == 0 for r in failed))
                state = validate_run_evidence(run, repository_root=repo)
                self.assertTrue(any(p.resolved and p.triggered for p in state.panel_views.values()))
                summary = json.loads((run / 'summary.json').read_text())
                health = next(x for x in summary['runtime_health'] if x['requested_model'] == 'scripted/judge-secondary-a')
                self.assertEqual(health['finish_status_counts'], {'output_limit': 3})
                self.assertEqual(health['schema_repair_attempts'], 0)
                target_health = next(x for x in summary['runtime_health'] if x['role'] == 'target')
                self.assertEqual(target_health['cost_receipt_count'], 3)
                self.assertEqual(target_health['provider_reported_cost_known_total'], '0.006')
                self.assertEqual(target_health['reasoning_tokens_known_total'], 21)
                provider.recovered = True
                _, reason = resume_run(run, repository_root=repo, sleeper=lambda _: None)
            self.assertIn(reason.value, {'completed_plan', 'completed_no_eligible_work'})
            validate_run_evidence(run, repository_root=repo)
            after = read_events(run)
            self.assertEqual(after[:len(before)], before)
            assert_complete_call_audit(self, after)

    def test_unresolved_entry_skips_every_continuation_and_retains_root_across_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); repo = self.repository(root)
            config = dryrun_config('fatalism', root / 'runs')
            config['experiment'].update(conversation_rounds=3, exploration_batches=0)
            provider = FailedEntryProvider()
            with patch('quasilisk.engine.create_provider', return_value=provider):
                run, reason = run_new(config, repository_root=repo, sleeper=lambda _: None)
                self.assertEqual(reason.value, 'operator_interrupted')
                _, reason = resume_run(run, repository_root=repo, sleeper=lambda _: None)
            self.assertIn(reason.value, {'completed_plan', 'completed_no_eligible_work'})
            state = validate_run_evidence(run, repository_root=repo)
            unit = next(u for u in state.work_units if u.unit['kind'] == 'verification')
            self.assertEqual(len(unit.unit['execution_slots']), 5)
            first = unit.unit['design']['root_plans'][0]
            first_actions = [state.logical_actions[k] for k in first['ordered_action_ids']]
            self.assertTrue(first_actions[0].attempts)
            self.assertFalse(any(a.attempts for a in first_actions[5:]))
            evidence = state.execution_evidence[first['execution_id']]
            self.assertTrue(evidence.terminated)
            self.assertFalse(evidence.resolved)
            self.assertFalse(evidence.semantic_nonsuccess)
            self.assertTrue(any(state.logical_actions[k].attempts for k in unit.unit['design']['root_plans'][1]['ordered_action_ids'][5:]))
            assert_complete_call_audit(self, read_events(run))

    def test_terminal_record_does_not_reenter_failed_engine_completion_parser(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); repo = self.repository(root)
            config = dryrun_config('fatalism', root / 'runs')
            with patch('quasilisk.engine._unit_complete', side_effect=EngineError('injected completion failure')):
                run, reason = run_new(config, repository_root=repo, sleeper=lambda _: None)
            self.assertEqual(reason.value, 'impossible_internal_state')
            self.assertEqual(read_events(run)[-1]['type'], 'run_terminated')
            validate_run_evidence(run, repository_root=repo)

    def test_route_circuit_is_not_reset_by_success_elsewhere(self):
        route = digest_json(['bad-profile'])
        outcomes = []
        for i in range(3):
            outcomes.append(ProviderCircuitOutcome('judge', i*2+1, digest_json([i]), 'failure', route))
            outcomes.append(ProviderCircuitOutcome('judge', i*2+2, digest_json(['panel', i]), 'success'))
        state = reduce_provider_circuit(outcomes)
        self.assertEqual(state.judge, 0)
        self.assertEqual(state.tripped_route_key, route)
        self.assertEqual(state.tripped_sequence, 5)
        self.assertIsNone(reduce_provider_circuit(outcomes, reset_after={'judge': 6}).tripped_role)

    def test_adopted_rule_resets_stall_even_while_reservation_requires_repair(self):
        panels = {}; nodes = {}; members = {}
        for depth, (inference, commitment) in enumerate([('rejected','absent'), ('rejected','absent'), ('endorsed','adopted')]):
            member = SimpleNamespace(ownership='adopted', procedure=SimpleNamespace(inference=inference, commitment=commitment, reservation='material'))
            panels[str(depth)] = SimpleNamespace(observation_id=str(depth), resolved=True, primary=member, details=None, assessment=None, effective_score=Decimal(25))
            nodes[str(depth)] = SimpleNamespace(target_observation_id=str(depth), node_id=str(depth), parent_node_id=None if depth == 0 else str(depth-1), depth=depth)
        state = SimpleNamespace(panel_views=panels, member_assessments=members, conversation_nodes=nodes)
        self.assertEqual(portfolio_transition(panels['2'], members), 'repair_inference')
        self.assertEqual(portfolio_node_progress(state,nodes['2'])[0], 4)
        self.assertFalse(portfolio_branch_closed(state,nodes['2'],maximum_followups=3,maximum_stalled_followups=2))

    def test_cost_and_reasoning_receipts_preserved_without_poisoning_completion(self):
        provider = OpenAICompatibleProvider(ProviderConfig('openai_compatible', 1, 'https://provider.example/v1', 'QUASILISK_TEST_KEY'))
        for value, expected in [(Decimal('0.00123'),'0.00123'), (0,'0'), (None,None), (-1,None), (True,None), ('bad',None)]:
            with self.subTest(value=value):
                raw = canonical_json_bytes({'model':'effective/model','choices':[{'message':{'content':None},'finish_reason':'length'}], 'usage':{'cost':value,'completion_tokens':6144,'completion_tokens_details':{'reasoning_tokens':6144}}})
                class Response:
                    status = 200
                    headers = {'Content-Type':'application/json'}
                    def __enter__(self): return self
                    def __exit__(self,*args): return False
                    def read(self): return raw
                with patch.dict(os.environ, {'QUASILISK_TEST_KEY':'secret'}), patch('quasilisk.provider.urlrequest.urlopen',return_value=Response()):
                    result = provider.complete(make_request())
                self.assertEqual(result.finish_status, FinishStatus.OUTPUT_LIMIT)
                self.assertEqual(result.cost, expected)
                self.assertEqual(result.metadata['reasoning_tokens'],6144)
                self.assertEqual(result.raw_body.data,raw.decode())
