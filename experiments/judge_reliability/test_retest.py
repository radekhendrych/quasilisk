"""Offline safeguards for the cumulative-budget contract retest."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import retest as r


class RetestTests(unittest.TestCase):
    def test_transformation_preserves_workload_and_binds_current_catalog(self):
        old = json.loads((r.BASE/'corpus.json').read_text())[0]
        messages, response_format = r.transform(old, old)
        for name in ('UNTRUSTED_OBJECTIVE_JSON', 'UNTRUSTED_CONVERSATION_JSON', 'GOVERNING_REVIEW_FOCUS_JSON'):
            self.assertEqual(r.block(messages[1]['content'], name),
                             r.block(old['messages'][1]['content'], name))
        schema = response_format['json_schema']['schema']
        self.assertEqual(schema['$defs']['EvidenceId']['enum'],
                         [x['evidence_id'] for x in r.judge_evidence_catalog(old['target_reply'])])
        self.assertNotIn('evidence', schema['properties'])

    def test_cumulative_budget_and_checkpoint_cannot_be_reset(self):
        rows = [{'type': 'intent', 'call_id': 'a', 'reservation_usd': '2.1', 'hash': 'checkpoint'}]
        config = {'baseline_checkpoint': 'checkpoint', 'baseline_liability_usd': '2.1',
                  'dispatch_budget_usd': '2.4', 'budget_usd': '2.9'}
        with tempfile.TemporaryDirectory() as directory, patch.object(r, 'DEST', Path(directory)):
            def save():
                (Path(directory)/'study.json').write_text(json.dumps(config))
            save(); self.assertEqual(r.check_baseline(rows), config)
            config['dispatch_budget_usd'] = '4.5'; save()
            with self.assertRaises(AssertionError): r.check_baseline(rows)
            config['dispatch_budget_usd'] = '2.4'; config['baseline_checkpoint'] = 'changed'; save()
            with self.assertRaises(AssertionError): r.check_baseline(rows)

    def test_baseline_writer_is_blocked_while_retest_holds_snapshot(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(r, 'BASE', Path(directory)):
            with r.baseline_snapshot():
                with (Path(directory)/'.lock').open('a') as writer:
                    with self.assertRaises(BlockingIOError):
                        r.b.fcntl.flock(writer, r.b.fcntl.LOCK_EX | r.b.fcntl.LOCK_NB)


if __name__ == '__main__': unittest.main()
