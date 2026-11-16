"""Independent helper route admission, receipts, and frozen replay."""
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
import shutil
import signal
import tempfile
import unittest
from unittest.mock import patch

from integration_support import dryrun_config, assert_complete_call_audit
from test_finisher_end_to_end import FinisherProvider, ROOT
from test_two_judge_panel import pair
from quasilisk.config import ConfigError, load_config
from quasilisk.domain import ProviderErrorKind, ProviderFailure
from quasilisk.engine import run_new, resume_run, validate_run_evidence
from quasilisk.events import read_events
from quasilisk.run_state import _profile_for_binding, model_cost_reservation


class TimeoutPrimary(FinisherProvider):
    def __init__(self, interrupt=False):
        super().__init__('nihilism_susceptibility')
        self.interrupt = interrupt
        self.failed = 0

    def complete(self, request):
        if request.requested_model == 'scripted/selector-primary' and request.purpose.value == 'selector_portfolio':
            self.requests.append(request)
            self.failed += 1
            if self.interrupt and self.failed == 1:
                signal.raise_signal(signal.SIGINT)
            return ProviderFailure(ProviderErrorKind.TIMEOUT, 'outcome_unknown',
                                   'Injected timeout', None, None, None, None)
        return replace(super().complete(request), cost=(
            '0.001' if request.requested_model == 'scripted/selector-fallback' else '0'))


def config_for(root, ceiling='0.05'):
    config = pair(dryrun_config('nihilism', root / 'runs'))
    config['seed']['bootstrap_limit'] = 1
    config['experiment'].update(exploration_batches=0, conversation_rounds=1, verification_blocks=0)
    config['provider']['max_cost_usd'] = ceiling
    models = config['models']
    profiles = [models['target'], models['judges']['primary'], *models['judges']['secondaries'],
                *models['selector'].values(), *models['perturbator'].values()]
    for profile in profiles:
        profile.update(max_context_tokens=1000000, max_price={'prompt': 0, 'completion': 0, 'request': 0})
    models['selector']['primary'].update(max_context_tokens=100000, provider_only='test/cheap',
        max_output_tokens=1024, max_price={'prompt': Decimal('0.1'), 'completion': 0, 'request': 0})
    models['selector']['fallback'].update(max_context_tokens=1000000, provider_only='test/fallback',
        max_output_tokens=4096, max_price={'prompt': Decimal('0.1'), 'completion': 0, 'request': 0})
    return config


class HelperProfileTests(unittest.TestCase):
    def test_closed_profiles_and_frozen_roundtrip_preserve_independent_bounds(self):
        config = config_for(Path('/tmp/helper-profile-config'))
        loaded = load_config(config)
        frozen = loaded.to_dict()
        resumed = load_config(frozen, frozen_resume=True, has_compatible_seed=True)
        self.assertEqual(resumed.to_dict(), frozen)
        for slot, bound, tokens in [('selector_primary', '0.01', 1024), ('selector_fallback', '0.1', 4096)]:
            profile, model = _profile_for_binding({'model_profiles': frozen['models']}, 'selector', slot)
            self.assertEqual(model_cost_reservation(profile), Decimal(bound))
            self.assertEqual(profile['max_output_tokens'], tokens)
            self.assertEqual(model, loaded.selector.primary().model if slot.endswith('primary') else loaded.selector.fallback().model)
        for role in ('selector', 'perturbator'):
            for slot in ('primary', 'fallback'):
                bad = deepcopy(config)
                del bad['models'][role][slot]['max_context_tokens']
                with self.assertRaisesRegex(ConfigError, 'dollar cap'):
                    load_config(bad)
        old = deepcopy(config)
        old['models']['selector'] = {**old['models']['selector']['primary'], 'fallback_model': 'legacy/fallback'}
        with self.assertRaises(ConfigError):
            load_config(old)

    def exercise(self, ceiling):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / 'repository'
            for name in ('quasilisk', 'objectives', 'templates', 'schemas', 'fixtures'):
                shutil.copytree(ROOT / name, repository / name, ignore=shutil.ignore_patterns('__pycache__'))
            (repository / 'library/seeds').mkdir(parents=True)
            (repository / 'library/strategies').mkdir(parents=True)
            config = config_for(root, ceiling)
            provider = TimeoutPrimary(interrupt=True)
            with patch('quasilisk.engine.create_provider', return_value=provider):
                run, reason = run_new(config, repository_root=repository, sleeper=lambda _: None)
                self.assertEqual(reason.value, 'operator_interrupted')
                state = validate_run_evidence(run, repository_root=repository)
                self.assertEqual(state.cost_liability_usd, Decimal('0.01'))
                _, reason = resume_run(run, repository_root=repository, sleeper=lambda _: None)
            state = validate_run_evidence(run, repository_root=repository)
            events = read_events(run)
            fallback_calls = [r for r in provider.requests if r.requested_model == 'scripted/selector-fallback']
            self.assertEqual(provider.failed, 2)
            self.assertEqual(state.manifest['normalized_config']['models']['selector']['primary']['max_context_tokens'], 100000)
            if ceiling == '0.05':
                self.assertEqual(reason.value, 'cost_limit_reached')
                self.assertEqual(state.cost_liability_usd, Decimal('0.02'))
                self.assertEqual(fallback_calls, [])
                self.assertFalse(any(e['type'] == 'model_call_intended' and e['payload']['requested_model'] == 'scripted/selector-fallback' for e in events))
            else:
                self.assertIn(reason.value, {'completed_plan', 'completed_no_eligible_work'})
                self.assertGreater(len(fallback_calls), 0)
                self.assertEqual(state.cost_liability_usd, Decimal('0.02') + Decimal('0.001') * len(fallback_calls))
                self.assertEqual(fallback_calls[0].generation_parameters['max_output_tokens'], 4096)
                self.assertEqual(fallback_calls[0].generation_parameters['provider_only'], 'test/fallback')
            assert_complete_call_audit(self, events)
            before = (run / 'events.jsonl').read_bytes()
            resume_run(run, repository_root=repository, sleeper=lambda _: None)
            self.assertEqual((run / 'events.jsonl').read_bytes(), before)

    def test_expensive_fallback_is_denied_before_intent_after_two_unknowns(self):
        self.exercise('0.05')

    def test_fallback_receipts_settle_only_their_own_attempts(self):
        self.exercise('0.15')


if __name__ == '__main__':
    unittest.main()
