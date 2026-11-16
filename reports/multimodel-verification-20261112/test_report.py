"""Evidence fidelity checks for the static review, separate from engine tests."""
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / 'site/data'
CAMPAIGN = ROOT / 'experiments/multimodel_verification'


def read(path):
    return json.loads(path.read_text())


class ReportEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = read(DATA / 'report.json')

    def test_every_target_branch_and_production_assessment_is_preserved(self):
        seen = set()
        for run in self.data['runs']:
            audit = read(CAMPAIGN / (run['cell'] + '.audit.json'))
            for observation in audit['observations']:
                path = DATA / 'conversations' / (run['cell'] + '-target-' + str(observation['target_sequence']) + '.json')
                detail = read(path)
                self.assertEqual(detail['messages'], observation['target_messages'] + [{'role':'assistant', 'content':observation['target_reply']}])
                self.assertEqual(detail['score'], observation['effective_score'])
                self.assertEqual(detail['panel'], observation['panel'])
                self.assertEqual([{k:v for k,v in m.items() if k != 'call_id'} for m in detail['members']], observation['members'])
                self.assertNotIn(detail['id'], seen)
                seen.add(detail['id'])
        self.assertEqual(len(seen), 84)
        self.assertEqual(seen, {o['id'] for o in self.data['observations']})

    def test_all_physical_dispatches_have_exact_requests_and_no_authorization_headers(self):
        count = 0
        for run in self.data['runs']:
            journal = ROOT / run['run_directory'] / 'events.jsonl'
            self.assertEqual(hashlib.sha256(journal.read_bytes()).hexdigest(), run['source']['sha256'])
            events = [json.loads(line) for line in journal.read_text().splitlines()]
            intents = {e['payload']['physical_attempt_id']:e for e in events if e['type']=='model_call_intended'}
            terminals = {e['payload']['physical_attempt_id']:e for e in events if e['type'] in {'model_call_succeeded','model_call_failed','model_call_outcome_unknown'}}
            for event in events:
                if event['type'] != 'model_dispatch_started': continue
                count += 1
                intent = intents[event['payload']['physical_attempt_id']]
                detail = read(DATA / 'calls' / (run['cell'] + '-call-' + str(intent['sequence']) + '.json'))
                self.assertEqual(detail['request'], json.loads(intent['payload']['request_body']['data']))
                terminal = terminals.get(event['payload']['physical_attempt_id'])
                self.assertEqual(detail['result'], {k:v for k,v in terminal['payload'].items() if k!='raw_body'} if terminal else {})
                self.assertNotIn('headers', detail['request'])
        self.assertEqual(count, 325)

    def test_audit_requests_and_outputs_preserve_frozen_schedule_and_references(self):
        events = [json.loads(line) for line in (CAMPAIGN / 'events.jsonl').read_text().splitlines()]
        intents = {e['call_id']:e for e in events if e['type']=='intent'}
        results = [e for e in events if e['type']=='result']
        for result in results:
            detail = read(DATA / 'calls' / ('audit-' + result['call_id'] + '.json'))
            self.assertEqual(detail['request'], intents[result['call_id']]['request'])
            self.assertEqual(detail['result'], json.loads(result['raw_body']))
        self.assertEqual(len(results),48)
        self.assertEqual(self.data['semantic_results'], read(CAMPAIGN / 'semantic_results.json'))
        self.assertEqual(self.data['panel_results'], read(CAMPAIGN / 'panel_results.json'))

    def test_counts_costs_and_incomplete_states_reconcile(self):
        calls = self.data['calls']
        self.assertEqual(len(calls), 373)
        self.assertEqual(sum(c['status'] != 'complete' for c in calls),17)
        self.assertEqual(sum(c['status'] == 'unresolved' for c in calls),2)
        self.assertEqual(sum(Decimal(c['cost']) for c in calls if c['cost'] is not None), Decimal(self.data['budget']['total_receipts_usd']))
        self.assertLessEqual(Decimal(self.data['budget']['combined_peak_liability_usd']),Decimal(4))
        self.assertEqual([sum(r['status']==status for r in self.data['runs']) for status in ['complete','partial','abandoned']], [5,1,3])
        self.assertEqual(len(self.data['not_run']),3)
        self.assertEqual(sum(o['score'] is not None and Decimal(o['score'])>=90 for o in self.data['observations']),17)

    def test_every_strategy_link_and_branch_parent_resolves(self):
        observations = {o['id']:o for o in self.data['observations']}
        for o in observations.values():
            if o['parent']:
                self.assertIn(o['parent'], observations)
                self.assertEqual(observations[o['parent']]['cell'],o['cell'])
                self.assertLess(observations[o['parent']]['depth'],o['depth'])
        for p in self.data['portfolios']:
            detail = read(DATA / 'strategies' / (p['id'] + '.json'))
            self.assertIn(detail['parent'], observations)
            self.assertEqual(len(detail['children']),p['children'])
            for child in detail['children']:
                if child['case']:self.assertIn(child['case'],observations)


if __name__=='__main__':unittest.main()
