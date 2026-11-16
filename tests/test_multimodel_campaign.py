"""Aggregate budget admission and matched canonical experiment configuration."""
from decimal import Decimal
import importlib.util
import json
from pathlib import Path
import unittest
import tempfile
from unittest.mock import Mock, patch
from quasilisk.engine import validate_new_run

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('multimodel_campaign',ROOT/'experiments/multimodel_verification/campaign.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

class CampaignTests(unittest.TestCase):
    def test_accepted_cost_limited_cell_remains_partial_and_fully_charged(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            entry={'cell':'partial','output_directory':'partial',
                   'accept_partial_reason':'Cost-limited endpoint reviewed'}
            plan={'authorized_usd':'4','engine_pool_usd':'3.65','audit_reserved_usd':'0.35',
                  'unallocated_margin_usd':'0','entries':[entry]}
            (root/'plan.json').write_text(json.dumps(plan))
            (root/'partial'/'run-evidence').mkdir(parents=True)
            record={'cell':'partial','terminal_reason':'cost_limit_reached','cost_liability_usd':'0.34'}
            with patch.object(c,'HERE',root),patch.object(c,'ROOT',root),patch.object(c,'validated_record',return_value=record),patch.object(c,'run_cli') as dispatch:
                c.run()
                result=json.loads((root/'execution.json').read_text())
                self.assertEqual(result['cells'],[])
                self.assertEqual(result['partial_cells'],[record])
                self.assertEqual(result['engine_liability_usd'],'0.34')
                self.assertFalse(result['all_cells_completed'])
                dispatch.assert_not_called()
                del entry['accept_partial_reason']
                (root/'plan.json').write_text(json.dumps(plan))
                with self.assertRaisesRegex(RuntimeError,'Existing unfinished/failed cell'):
                    c.run()

    def test_budget_exclusion_cannot_hide_paid_evidence_or_claim_complete_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            entry={'cell':'excluded','output_directory':'unstarted','not_run_reason':'insufficient admission headroom'}
            plan={'authorized_usd':'4','engine_pool_usd':'3.35','audit_reserved_usd':'0.60',
                  'unallocated_margin_usd':'0.05','entries':[entry]}
            (root/'plan.json').write_text(json.dumps(plan))
            with patch.object(c,'HERE',root),patch.object(c,'ROOT',root),patch.object(c,'run_cli') as dispatch:
                c.run()
                result=json.loads((root/'execution.json').read_text())
                self.assertTrue(result['campaign_finished'])
                self.assertFalse(result['all_cells_completed'])
                self.assertEqual(result['not_run'][0]['cell'],'excluded')
                dispatch.assert_not_called()
                (root/'unstarted'/'run-evidence').mkdir(parents=True)
                with self.assertRaisesRegex(RuntimeError,'Cannot omit existing paid evidence'):
                    c.run()

    def test_launcher_interrupt_waits_for_canonical_engine_shutdown(self):
        child=Mock();child.wait.side_effect=[KeyboardInterrupt(),0]
        with patch.object(c.subprocess,'Popen',return_value=child):
            self.assertEqual(c.run_cli(Path('/tmp/example.json'),None),(0,True))
        child.send_signal.assert_called_once_with(c.signal.SIGINT)
        child.kill.assert_not_called()

    def test_receipts_and_unresolved_liabilities_both_reduce_new_run_ceiling(self):
        self.assertEqual(c.remaining_cap('3.35',['0.12','0.3022668']),Decimal('2.9277332'))
        with self.assertRaises(RuntimeError):c.remaining_cap('3.35',['3.35'])

    @unittest.skipUnless((c.HERE / 'plan.json').exists(), 'requires local private campaign evidence')
    def test_interrupted_route_stays_separate_and_fully_reserved(self):
        plan=json.loads((c.HERE/'plan.json').read_text())
        for entry in plan.get('abandoned_entries', []):
            liability=plan['route_amendment']['abandoned_liability_usd']
            self.assertTrue(entry['abandoned_checkpoint'].startswith('sha256:'))
            self.assertGreater(Decimal(liability),0)
            self.assertNotIn(entry['output_directory'],[e['output_directory'] for e in plan['entries']])
            self.assertLess(c.remaining_cap(plan['engine_pool_usd'],[liability]),Decimal(plan['engine_pool_usd']))

    @unittest.skipUnless((c.HERE / 'plan.json').exists(), 'requires local private campaign evidence')
    def test_nine_cells_use_matched_work_and_distinct_targets_and_judges(self):
        plan=json.loads((c.HERE/'plan.json').read_text())
        self.assertEqual(len(plan['entries']),9)
        configs=[json.loads((ROOT/e['config']).read_text()) for e in plan['entries']]
        self.assertEqual(len({x['objective'] for x in configs}),3)
        self.assertEqual(len({x['models']['target']['model'] for x in configs}),3)
        first=configs[0]
        for config,entry in zip(configs,plan['entries']):
            for key in ('experiment','search','seed','samples','random_seed'):
                self.assertEqual(config[key],first[key])
            self.assertEqual(config['models']['judges'],first['models']['judges'])
            target=config['models']['target']['model']
            self.assertNotIn(target,[config['models']['judges']['primary']['model'],config['models']['judges']['secondaries'][0]['model']])
            self.assertIsNone(config['models']['judges']['replacement'])
            validate_new_run(ROOT/entry['config'])
        self.assertLessEqual(sum(Decimal(plan[x]) for x in ('engine_pool_usd','audit_reserved_usd','unallocated_margin_usd')),Decimal(plan['authorized_usd']))

if __name__=='__main__':unittest.main()
