"""Offline checks of the study's spend ledger and strict result classification."""
import importlib.util
import json
from pathlib import Path
from decimal import Decimal
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('judge_benchmark', Path(__file__).with_name('benchmark.py'))
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


class BenchmarkTests(unittest.TestCase):
    def test_unknown_dispatch_keeps_reservation(self):
        rows=[{'type':'intent','call_id':'a','reservation_usd':'0.5'},
              {'type':'intent','call_id':'b','reservation_usd':'0.4'},
              {'type':'result','call_id':'a','cost_usd':'0.01'},
              {'type':'result','call_id':'b','cost_usd':None}]
        self.assertEqual(b.charged(rows),Decimal('.41'))

    def test_invalid_receipt_is_unknown(self):
        for value in (None, True, -1, 'NaN', 'Infinity', 'wrong'):
            self.assertIsNone(b.receipt(json.dumps({'usage':{'cost':value}})))
        self.assertEqual(b.receipt('{"usage":{"cost":0.001}}'),'0.001')

    def test_journal_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(b,'HERE',Path(directory)):
            rows=[];b.append(rows,'test',value=42)
            self.assertEqual(b.history(),rows)
            p=Path(directory)/'events.jsonl';p.write_text(p.read_text().replace('42','43'))
            with self.assertRaises(RuntimeError):b.history()

    def test_provider_error_and_truncation_fail_closed(self):
        case={'response_format':{'json_schema':{'schema':{'type':'object'}}},'target_reply':'Hello'}
        self.assertFalse(b.assess('{"error":{"message":"bad"}}',case)['usable'])
        r=b.assess('{"choices":[{"finish_reason":"length","message":{"content":"{}"}}]}',case)
        self.assertEqual(r['category'],'incomplete');self.assertFalse(r['usable'])

    def test_balanced_sampling_and_price_limits(self):
        corpus=[{'id':str(i),'objective':str(i%3),'focus_slot':'primary'} for i in range(9)]
        sample=b.schedule(corpus,600,1)
        self.assertEqual({sum(x['objective']==str(i) for x in sample) for i in range(3)},{200})
        profile={'model':'test','prompt_price':'.000001','completion_price':'.000002','context_length':100000,'max_tokens':1000,'temperature':0,'reasoning':None}
        body=b.request_body(profile,{'messages':[],'response_format':{}})
        self.assertEqual(b.reservation(profile),Decimal('.102'))
        self.assertEqual(body['provider']['max_price'],{'prompt':1,'completion':2,'request':0})
        self.assertFalse(body['provider']['allow_fallbacks'])

    def test_confirmation_confidence_threshold(self):
        self.assertGreater(b.lower_bound(600,600),.995)
        self.assertLess(b.lower_bound(599,600),.995)
        self.assertAlmostEqual(b.lower_bound(1,2),1-(.95**.5))

    def test_unaffordable_dispatch_is_never_written_or_sent(self):
        profile={'model':'test','prompt_price':'1','completion_price':'1','context_length':10,'max_tokens':10,'temperature':0,'reasoning':None}
        config={'source_sha256':{},'budget_usd':'5','dispatch_budget_usd':'4.5','profiles':[profile], 'screen_calls':1,'seed':1,'concurrency':1,'deadline_seconds':1}
        corpus=[{'id':'case','objective':'fatalism','focus_slot':'primary'}]
        with tempfile.TemporaryDirectory() as directory, patch.object(b,'HERE',Path(directory)), patch.dict(b.os.environ,{'OPENROUTER_API_KEY':'unused-test-placeholder'}), patch('builtins.print'):
            b.atomic_json(Path(directory)/'study.json',config)
            b.atomic_json(Path(directory)/'corpus.json',corpus)
            b.run('screen')
            rows=b.history()
            self.assertFalse(any(r['type']=='intent' for r in rows))
            self.assertTrue(any(r['type']=='budget_stopped' for r in rows))

    def test_offline_audit_refuses_active_writer(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(b,'HERE',Path(directory)):
            with (Path(directory)/'.lock').open('w') as writer:
                b.fcntl.flock(writer,b.fcntl.LOCK_EX|b.fcntl.LOCK_NB)
                with self.assertRaises(SystemExit),b.snapshot_lock():pass


if __name__=='__main__':unittest.main()
