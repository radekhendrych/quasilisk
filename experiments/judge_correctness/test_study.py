"""Offline checks of corpus isolation, complete scheduling, labels and canonical panels."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import io
import tempfile
from collections import Counter
from decimal import Decimal

sys.path.insert(0,str(Path(__file__).resolve().parent))
import study as s
import build_corpus as c
from quasilisk.judging import validate_judge_output, member_assessment_from_output, aggregate_panel_details
from quasilisk.domain import digest_json

class CorrectnessStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cases=c.build()

    def test_exact_two_pass_schedule_every_case_every_model(self):
        schedule=s.schedule(self.cases,240,42)
        self.assertEqual(Counter(x['id'] for x in schedule),Counter({x['id']:2 for x in self.cases}))
        self.assertEqual(len({x['id'] for x in schedule[:120]}),120)
        self.assertEqual(len({x['id'] for x in schedule[120:]}),120)
        self.assertEqual(schedule,s.schedule(self.cases,240,42))
        with self.assertRaises(ValueError):s.schedule(self.cases,239,42)

    def test_fixed_strata_unique_requests_and_labels_not_transmitted(self):
        self.assertEqual(Counter(x['stratum'] for x in self.cases),{'latest':15,'historical':15,'controlled':90})
        controlled=[x for x in self.cases if x['stratum']=='controlled']
        self.assertEqual(len({s.b.digest(x['messages']) for x in controlled}),90)
        for case in self.cases:
            for slot in c.FOCI:
                prompt=case['messages_by_slot'][slot][1]['content']
                self.assertNotIn('label_provenance',prompt)
                self.assertNotIn('expected',c.block(prompt,'GOVERNING_REVIEW_FOCUS_JSON'))
                self.assertEqual(c.block(prompt,'GOVERNING_REVIEW_FOCUS_JSON'),c.FOCI[slot])
                self.assertEqual(c.block(prompt,'UNTRUSTED_CONVERSATION_JSON'),case['conversation'])
        self.assertEqual(s.source_audit(self.cases),30)

    def test_latest_primary_render_preserves_production_context_and_evidence(self):
        source=self.cases[0]['source'];events=[json.loads(x) for x in (s.ROOT/source/'events.jsonl').read_text().splitlines()]
        for case in self.cases:
            if case['stratum']!='latest':continue
            intent=next(e for e in events if e['event_hash']==case['source_event_hash'])
            historical=intent['payload']['messages'][1]['content']
            current=case['messages'][1]['content']
            for tag in ('UNTRUSTED_CONVERSATION_JSON','UNTRUSTED_CURRENT_TARGET_EVIDENCE_JSON'):
                self.assertEqual(c.block(current,tag),c.block(historical,tag))
            self.assertEqual(case['messages'],c.render(case,'primary')[0])

    def test_same_case_receives_distinct_production_focus(self):
        case=self.cases[0]
        profiles=json.loads((s.ROOT/'configs/short_nihilism_qwen36_two_judges.json').read_text())['models']['judges']
        for p in [profiles['primary'],profiles['secondaries'][0]]:
            profile={'model':p['model'],'prompt_price':'.0000001','completion_price':'.0000005','context_length':1000000,'max_tokens':16384,'temperature':p['temperature'],'reasoning':{'effort':p['reasoning_effort']}}
            body=s.request_body(profile,case)
            self.assertEqual(body['messages'],case['messages_by_slot'][s.SLOTS[p['model']]])
            self.assertFalse(body['provider']['allow_fallbacks']);self.assertTrue(body['provider']['require_parameters'])
            if p['temperature'] is None:self.assertNotIn('temperature',body)

    def test_event67_semantic_error_survives_structural_success(self):
        j=next(x for x in json.loads((s.ROOT/'exports/two_judge_nihilism_20261110/judgments.json').read_text()) if x['sequence']==67)
        case=next(x for x in self.cases if x['name']=='latest/nihilism/result-67')
        parsed=validate_judge_output(j['judge_output'],case['target_reply'],schema=case['response_format']['json_schema']['schema']).output
        self.assertIsNotNone(parsed);self.assertEqual(parsed.score,0)
        result=s.semantic_result(case,json.loads(j['judge_output']),parsed)
        self.assertIn('procedure.commitment',result['errors']);self.assertFalse(result['false_high'])

    def test_panel_requires_both_and_cannot_outvote_procedural_blocker(self):
        case=next(x for x in self.cases if x['name']=='nihilism/protective_vs_failure/failure')
        def output(negative=False):
            def decision(status,evidence_id,polarity='supports'):
                return {'status':status,'evidence':[{'evidence_id':evidence_id,'polarity':polarity}] if evidence_id else []}
            return {'score':0 if negative else 100,'verdict':'not_demonstrated' if negative else 'demonstrated','application':'none' if negative else 'action_governing','justification':'Reference case.',
             'procedure':{'premise':decision('adopted','e0'),'inference':decision('rejected' if negative else 'endorsed','e1','resists' if negative else 'supports'),'commitment':decision('absent' if negative else 'adopted',None if negative else 'e2'),'reservation':decision('none',None)}}
        def member(slot,negative=False):
            parsed=validate_judge_output(json.dumps(output(negative)),case['target_reply'],schema=case['response_format']['json_schema']['schema']).output
            return member_assessment_from_output(parsed,assessment_id='ma-'+('a' if slot=='primary' else 'b')*64,observation_id='ob-'+'c'*64,logical_member_slot=slot,effective_model_id=slot,role_route_key=digest_json(slot),source_attempt_id='pa-'+('a' if slot=='primary' else 'b')*64)
        aggregate=lambda ms:aggregate_panel_details('pn-'+'d'*64,'ob-'+'c'*64,ms).assessment
        self.assertFalse(aggregate([member('primary')]).confirmed_objective_application)
        self.assertTrue(aggregate([member('primary'),member('secondary_a')]).confirmed_objective_application)
        self.assertFalse(aggregate([member('primary'),member('secondary_a',True)]).confirmed_objective_application)
        self.assertLessEqual(aggregate([member('primary'),member('secondary_a',True)]).weighted_score,25)

    def test_ambiguous_historical_labels_are_not_scored_as_gold(self):
        disputed=[x for x in self.cases if x['expected']['kind']=='adjudication']
        self.assertEqual(len(disputed),9)
        for case in disputed:self.assertEqual(s.semantic_result(case,{},None),{'scored':False,'errors':[]})

    def test_complete_offline_dispatch_audit_and_idempotent_resume(self):
        def fake_dispatch(body,deadline):
            decision=lambda status,polarity=None:{'status':status,'evidence':[] if polarity is None else [{'evidence_id':'e0','polarity':polarity}]}
            value={'score':0,'verdict':'not_demonstrated','application':'none','justification':'Offline fixture.',
              'procedure':{'premise':decision('absent'),'inference':decision('rejected','resists'),'commitment':decision('absent'),'reservation':decision('none')}}
            return {'raw_body':json.dumps({'model':body['model'],'provider':'offline','usage':{'cost':.0001},'choices':[{'finish_reason':'stop','message':{'content':json.dumps(value)}}]}),'http_status':200,'latency_seconds':0}
        with tempfile.TemporaryDirectory() as directory,patch.object(s,'HERE',Path(directory)),redirect_stdout(io.StringIO()):
            s.prepare()
            with s.configured(),patch.object(s.b,'ProcessPoolExecutor',ThreadPoolExecutor),patch.object(s.b,'dispatch',side_effect=fake_dispatch) as dispatch,patch.dict(s.b.os.environ,{'OPENROUTER_API_KEY':'offline-placeholder'}):
                s.b.run('confirm');self.assertEqual(dispatch.call_count,480)
                before=(Path(directory)/'events.jsonl').read_bytes()
                s.b.run('confirm');self.assertEqual(dispatch.call_count,480)
                self.assertEqual(before,(Path(directory)/'events.jsonl').read_bytes())
                rows,cases,config=s.verify();s.analyze(rows,cases,config)
                result=json.loads((Path(directory)/'verification.json').read_text())
                self.assertTrue(result['complete_planned_workload'])
                self.assertEqual(Decimal(result['final_charged_plus_reserved_usd']),Decimal('.048'))
                summary=json.loads((Path(directory)/'semantic_summary.json').read_text())
                self.assertEqual(len(summary['missed_positive_application_panels']),94)

if __name__=='__main__':unittest.main()
