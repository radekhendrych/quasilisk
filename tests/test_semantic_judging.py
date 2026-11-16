"""Unified practical evidence, canonical request isolation, and Selector transitions."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import io

from quasilisk.judging import validate_judge_output, member_assessment_from_output, aggregate_panel_details
from quasilisk.run_state import PanelView, portfolio_transition

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('semantic_retest', ROOT / 'experiments/judge_semantic_retest/run.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def focused_cases():
    """Curated regression inputs work without private study evidence."""
    cases = []
    for mode in r.c.MODES:
        cases.extend(json.loads((ROOT / 'objectives' / (mode + '_susceptibility') /
                                 'semantic_regressions.json').read_text())['cases'])
    for case in cases:
        case['response_format'] = r.c.render(case, 'primary')[1]
    return cases


def oracle(case):
    expected = case['expected']
    procedure = {}
    for field, statuses in expected['procedure'].items():
        status = statuses[0]
        evidence = [] if status in {'absent', 'none'} else [{
            'evidence_id': 'e0' if field == 'premise' else 'e1',
            'polarity': 'resists' if status in {'rejected', 'material'} else 'supports',
        }]
        procedure[field] = {'status': status, 'evidence': evidence}
    return {'score': expected['score_range'][0], 'verdict': expected['verdict'][0],
            'application': expected['application'][0], 'procedure': procedure,
            'justification': 'Offline reference procedure; no claim of model semantic accuracy.'}


def panel_transition(case, raw):
    parsed = validate_judge_output(json.dumps(raw), case['target_reply'], schema=case['response_format']['json_schema']['schema'])
    assert parsed.usable, parsed.errors
    members = [member_assessment_from_output(parsed.output, assessment_id='ma-'+str(n)*64,
        observation_id='ob-'+'a'*64, logical_member_slot=slot, effective_model_id='judge/'+slot,
        role_route_key='route/'+slot, source_attempt_id='pa-'+str(n)*64)
        for n, slot in enumerate(('primary', 'secondary_a'), 1)]
    details = aggregate_panel_details('pn-'+'b'*64, 'ob-'+'a'*64, members)
    panel = PanelView('pn-'+'b'*64, 'action', 'ob-'+'a'*64, None, None, None,
                      True, members[0], details, (), {}, True)
    return portfolio_transition(panel, {m.assessment_id:m for m in members}), details.assessment


class SemanticJudgingTests(unittest.TestCase):
    @unittest.skipUnless((r.BASE / 'corpus.json').exists(), 'requires local private study evidence')
    def test_focused_requests_preserve_authentic_text_and_current_contract(self):
        cases = r.build()
        self.assertEqual(len(cases), 57)
        self.assertEqual(r.s.source_audit(cases), 30)
        self.assertEqual(len(r.s.schedule(cases, 114, 7)), 114)
        baseline = {c['id']:c for c in json.loads((r.BASE/'corpus.json').read_text())}
        for case in cases:
            if 'baseline_case_id' in case:
                old = baseline[case['baseline_case_id']]
                self.assertEqual(case['conversation'], old['conversation'])
                self.assertEqual(case['target_reply'], old['target_reply'])
            for slot in r.c.FOCI:
                self.assertEqual(case['messages_by_slot'][slot], r.c.render(case, slot)[0])
                self.assertNotIn('label_provenance', case['messages_by_slot'][slot][1]['content'])
            self.assertEqual(set(case['response_format']['json_schema']['schema']['properties']),
                             {'score','verdict','application','procedure','justification'})

    def test_reference_procedures_drive_the_correct_selector_transition(self):
        for case in focused_cases():
            if '/focused_' not in case['name']:
                continue
            with self.subTest(case=case['name']):
                raw = oracle(case)
                transition, panel = panel_transition(case, raw)
                kind = case['expected']['kind']
                expected = {'positive':'verify_commitment', 'abstract':'complete_commitment', 'ambiguous':'repair_inference'}
                if kind == 'negative':
                    want = 'resolve_ownership' if raw['procedure']['premise']['status']=='conditional' else 'repair_inference'
                else:
                    want = expected[kind]
                self.assertEqual(transition, want)
                self.assertEqual(panel.confirmed_objective_application, kind=='positive')

    def test_separate_counterevidence_still_blocks_an_abstract_opening(self):
        for case in focused_cases():
            if not case['name'].endswith('/focused_missing_vs_opposed/missing'):
                continue
            raw = oracle(case)
            self.assertEqual(panel_transition(case, raw)[0], 'complete_commitment')
            raw['procedure']['reservation'] = {'status':'material','evidence':[{'evidence_id':'e2','polarity':'resists'}]}
            transition, panel = panel_transition(case, raw)
            self.assertEqual(transition, 'repair_inference')
            self.assertLessEqual(panel.weighted_score, 50)
            self.assertFalse(panel.confirmed_objective_application)

    @unittest.skipUnless((r.BASE / 'study.json').exists(), 'requires local private study evidence')
    def test_focused_runner_complete_offline_rehearsal_and_resume(self):
        def dispatch(body, deadline):
            value = {'score':0,'verdict':'not_demonstrated','application':'none','justification':'Offline.',
              'procedure':{'premise':{'status':'absent','evidence':[]},'inference':{'status':'rejected','evidence':[{'evidence_id':'e0','polarity':'resists'}]},'commitment':{'status':'absent','evidence':[]},'reservation':{'status':'none','evidence':[]}}}
            return {'raw_body':json.dumps({'model':body['model'],'provider':'offline','usage':{'cost':0.0001},'choices':[{'finish_reason':'stop','message':{'content':json.dumps(value)}}]}),'http_status':200,'latency_seconds':0}
        with tempfile.TemporaryDirectory() as directory, patch.object(r,'HERE',Path(directory)), redirect_stdout(io.StringIO()):
            r.prepare()
            with r.configured(), patch.object(r.b,'ProcessPoolExecutor',ThreadPoolExecutor), patch.object(r.b,'dispatch',side_effect=dispatch) as calls, patch.dict(r.b.os.environ,{'OPENROUTER_API_KEY':'offline-placeholder'}):
                r.b.run('confirm')
                self.assertEqual(calls.call_count, 228)
                before=(Path(directory)/'events.jsonl').read_bytes()
                r.b.run('confirm')
                self.assertEqual(calls.call_count, 228)
                self.assertEqual(before,(Path(directory)/'events.jsonl').read_bytes())
                rows,cases,config=r.s.verify()
                r.s.analyze(rows,cases,config)
                result=json.loads((Path(directory)/'verification.json').read_text())
                self.assertTrue(result['complete_planned_workload'])
                self.assertEqual(result['physical_calls'],228)

if __name__ == '__main__':
    unittest.main()
