"""Two distinct judges use the canonical planner, quorum and replay path."""
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from integration_support import dryrun_config, assert_complete_call_audit
from test_finisher_end_to_end import FinisherProvider, ROOT
from test_provider_foundation import make_request
from quasilisk.config import load_config, ConfigError
from quasilisk.domain import ModelProfile, DomainError
from quasilisk.engine import run_new, resume_run, validate_run_evidence
from quasilisk.events import read_events
from quasilisk.provider import render_request_body
from quasilisk.run_state import funded_work_plan, funded_call_limits
from dataclasses import replace


def pair(config):
    judges = config['models']['judges']
    judges['secondaries'] = judges['secondaries'][:1]
    judges['replacement'] = None
    judges['primary'].update(reasoning_effort='low', provider_only='morph/fp8',
        max_price={'prompt': Decimal('.022'), 'completion': Decimal('.4'), 'request': 0})
    judges['secondaries'][0].update(temperature=None, reasoning_effort='medium',
        max_price={'prompt': Decimal('.1'), 'completion': Decimal('.5'), 'request': 0})
    return config


class TwoJudgePanelTests(unittest.TestCase):
    def repository(self, root):
        destination = root/'repository'
        for name in ('quasilisk', 'objectives', 'templates', 'schemas', 'fixtures'):
            shutil.copytree(ROOT/name, destination/name, ignore=shutil.ignore_patterns('__pycache__'))
        (destination/'library/seeds').mkdir(parents=True)
        (destination/'library/strategies').mkdir(parents=True)
        return destination

    def test_closed_configuration_and_exact_funded_judge_maximum(self):
        original = dryrun_config('nihilism', Path('/tmp/two-judge-config-test'))
        config = pair(deepcopy(original))
        parsed = load_config(config)
        self.assertEqual(parsed.judges.normal_slots, ('primary', 'secondary_a'))
        self.assertEqual(parsed.judges.maximum_physical_attempts, 4)
        self.assertEqual(len(parsed.judges.all_profiles), 2)
        first = funded_call_limits(funded_work_plan(original, [{}], ()))
        second = funded_call_limits(funded_work_plan(config, [{}], ()))
        self.assertEqual(first.judge, second.judge * 2)
        self.assertEqual((first.target, first.selector, first.perturbator),
                         (second.target, second.selector, second.perturbator))
        with self.assertRaises(ConfigError):
            bad = deepcopy(config); bad['models']['judges']['replacement'] = deepcopy(original['models']['judges']['replacement']); load_config(bad)
        with self.assertRaises(ConfigError):
            bad = deepcopy(config); bad['models']['judges']['secondaries'][0]['model'] = bad['models']['judges']['primary']['model']; load_config(bad)

    def test_tested_profile_parameters_are_exact_request_bytes(self):
        profile = ModelProfile('judge', None, 16384, reasoning_effort='medium',
            provider_only='test/provider', max_price={'prompt':'0.1','completion':'0.5','request':'0'})
        request = replace(make_request(), generation_parameters=profile.generation_parameters)
        body = json.loads(render_request_body(request))
        self.assertNotIn('temperature', body)
        self.assertEqual(body['reasoning'], {'effort':'medium'})
        self.assertEqual(body['provider'], {'only':['test/provider'],'require_parameters':True,
            'allow_fallbacks':False,'sort':'price','max_price':{'prompt':.1,'completion':.5,'request':0}})
        with self.assertRaises(TypeError): request.generation_parameters['max_price']['prompt'] = '100'
        with self.assertRaises(DomainError): ModelProfile('judge', '0', 10, reasoning_effort='surprise')

    def test_two_judge_scenario_verification_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); repository=self.repository(root)
            config=pair(dryrun_config('nihilism', root/'runs'))
            config['search']['maximum_followups']=1
            config['experiment'].update(exploration_batches=1,conversation_rounds=1,verification_blocks=1)
            bootstrap=json.loads((ROOT/'objectives/nihilism_susceptibility/bootstrap.json').read_text())
            bootstrap['probes']=[next(p for p in bootstrap['probes'] if p['descriptors']['application_scope']=='concrete_action')]
            seeds=root/'seeds.json';seeds.write_text(json.dumps(bootstrap));config['seed']['prompts_path']=str(seeds)
            provider=FinisherProvider('nihilism_susceptibility',interrupt_after=5)
            with patch('quasilisk.engine.create_provider',return_value=provider):
                run,reason=run_new(config,repository_root=repository,sleeper=lambda _:None)
                self.assertEqual(reason.value,'operator_interrupted')
                _,reason=resume_run(run,repository_root=repository,sleeper=lambda _:None)
            self.assertIn(reason.value,{'completed_plan','completed_no_eligible_work'})
            state=validate_run_evidence(run,repository_root=repository)
            self.assertEqual(json.loads((run/'summary.json').read_text())['commitment_progress']['verified_commitment_scenarios'],1)
            for unit in state.work_units:
                for panel in unit.unit['panel_plans']:
                    self.assertEqual(len(panel['normal_members']),2)
                    self.assertIsNone(panel['replacement'])
            judges=[request for request in provider.requests if request.role.value=='judge']
            self.assertEqual({request.requested_model for request in judges},{'scripted/judge-primary','scripted/judge-secondary-a'})
            for request in judges:
                self.assertIn(request.generation_parameters['reasoning_effort'],{'low','medium'})
            assert_complete_call_audit(self,read_events(run))

    def test_two_judge_entry_verification_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); repository=self.repository(root)
            config=pair(dryrun_config('nihilism',root/'runs'))
            config['seed']['bootstrap_limit']=1
            config['experiment'].update(exploration_batches=0,conversation_rounds=0,verification_blocks=1)
            provider=FinisherProvider('nihilism_susceptibility',interrupt_after=4)
            provider.opening=provider.commitment
            with patch('quasilisk.engine.create_provider',return_value=provider):
                run,reason=run_new(config,repository_root=repository,sleeper=lambda _:None)
                self.assertEqual(reason.value,'operator_interrupted')
                _,reason=resume_run(run,repository_root=repository,sleeper=lambda _:None)
            self.assertIn(reason.value,{'completed_plan','completed_no_eligible_work'})
            state=validate_run_evidence(run,repository_root=repository)
            verification=[u for u in state.work_units if u.unit['kind']=='verification']
            self.assertEqual(len(verification),1)
            roots=verification[0].unit['design']['root_plans']
            self.assertEqual(len(roots),5)
            self.assertTrue(all(len(r['ordered_action_ids'])==3 for r in roots))
            target_requests=[request for request in provider.requests
                if request.logical_action_id in {r['terminal_endpoint_action_id'] for r in roots}]
            self.assertEqual(len(target_requests),5)
            self.assertTrue(all(not any(m.role=='assistant' for m in r.messages) for r in target_requests))
            summary=json.loads((run/'summary.json').read_text())
            self.assertEqual(summary['headline']['newly_verified']['entry_seeds'],1)
            assert_complete_call_audit(self,read_events(run))

    def test_missing_independent_judge_never_creates_quorum(self):
        class InvalidSecondary(FinisherProvider):
            def complete(self, request):
                value=super().complete(request)
                if request.requested_model=='scripted/judge-secondary-a':
                    return replace(value,text='{}')
                return value
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);repository=self.repository(root)
            config=pair(dryrun_config('nihilism',root/'runs'))
            config['experiment'].update(exploration_batches=0,conversation_rounds=0,verification_blocks=0)
            provider=InvalidSecondary('nihilism_susceptibility')
            with patch('quasilisk.engine.create_provider',return_value=provider):
                run,_=run_new(config,repository_root=repository,sleeper=lambda _:None)
            state=validate_run_evidence(run,repository_root=repository)
            self.assertEqual(json.loads((run/'summary.json').read_text())['commitment_progress']['verified_commitment_scenarios'],0)
            self.assertTrue(all(panel.assessment.availability.value == 'no_quorum'
                for panel in state.panel_views.values() if panel.assessment is not None))

    def capped_config(self, root, ceiling):
        config=pair(dryrun_config('nihilism',root/'runs'))
        config['provider']['max_cost_usd']=ceiling
        config['experiment'].update(exploration_batches=0,conversation_rounds=0,verification_blocks=0)
        profiles=[config['models']['target'],config['models']['judges']['primary'],
                  *config['models']['judges']['secondaries'],*config['models']['selector'].values(),*config['models']['perturbator'].values()]
        for profile in profiles:
            profile.update(max_context_tokens=1000000,
                max_price={'prompt':Decimal('.1'),'completion':Decimal('.5'),'request':0})
        return config

    def test_dollar_cap_blocks_before_intent_or_physical_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);repository=self.repository(root)
            config=self.capped_config(root,'0.01');provider=FinisherProvider('nihilism_susceptibility')
            with patch('quasilisk.engine.create_provider',return_value=provider):
                run,reason=run_new(config,repository_root=repository,sleeper=lambda _:None)
            self.assertEqual(reason.value,'cost_limit_reached')
            self.assertEqual(provider.requests,[])
            self.assertFalse(any(e['type']=='model_call_intended' for e in read_events(run)))
            self.assertEqual(validate_run_evidence(run,repository_root=repository).cost_liability_usd,0)

    def test_unknown_receipt_reservation_survives_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);repository=self.repository(root)
            config=self.capped_config(root,'0.15');provider=FinisherProvider('nihilism_susceptibility',interrupt_after=1)
            with patch('quasilisk.engine.create_provider',return_value=provider):
                run,reason=run_new(config,repository_root=repository,sleeper=lambda _:None)
                self.assertEqual(reason.value,'operator_interrupted')
                _,reason=resume_run(run,repository_root=repository,sleeper=lambda _:None)
            self.assertEqual(reason.value,'cost_limit_reached')
            self.assertEqual(len(provider.requests),1)
            self.assertGreater(validate_run_evidence(run,repository_root=repository).cost_liability_usd,Decimal('.1'))

    def test_exact_receipts_release_reservations(self):
        class Priced(FinisherProvider):
            def complete(self,request):return replace(super().complete(request),cost='0.001')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);repository=self.repository(root)
            config=self.capped_config(root,'0.15');provider=Priced('nihilism_susceptibility')
            with patch('quasilisk.engine.create_provider',return_value=provider):
                run,reason=run_new(config,repository_root=repository,sleeper=lambda _:None)
            self.assertEqual(reason.value,'completed_plan')
            state=validate_run_evidence(run,repository_root=repository)
            self.assertEqual(state.cost_liability_usd,Decimal('.001')*len(provider.requests))


if __name__=='__main__':unittest.main()
