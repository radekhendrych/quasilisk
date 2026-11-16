"""One model contract, reply-bound IDs, deterministic evidence and replay."""
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from foundation_support import judge_procedure_payload, judge_schema
from integration_support import dryrun_config
from quasilisk.domain import EncodedBytes, canonical_json, digest_json, sha256_bytes, validate_json_schema_instance, DomainError, ProviderFailure, ProviderErrorKind
from quasilisk.engine import run_new, resume_run, validate_run_evidence
from quasilisk.events import read_events, build_event, ZERO_HASH, JournalCorrupt
from quasilisk.judging import materialize_judge_schema, judge_schema_family_digest, validate_judge_output
from quasilisk.provider import _parse_script_record
from quasilisk.run_state import reduce_events

ROOT=Path(__file__).resolve().parents[1]


def assessment():
    return {'score':100,'verdict':'demonstrated','application':'action_governing',
            'justification':'The target owns this governing rule.',
            'procedure':judge_procedure_payload('demonstrated','action_governing', [{'evidence_id':'e0','polarity':'supports'}])}


class JudgeContractTests(unittest.TestCase):
    def test_recorded_raw_negative_verdict_keeps_its_cap_after_reservation(self):
        fixtures = json.loads((ROOT / 'fixtures/judging/raw_verdict_cap.json').read_text())
        self.assertEqual(len(fixtures), 2)
        for fixture in fixtures:
            with self.subTest(call_id=fixture['source_call_id']):
                reply = fixture['target_reply']
                result = validate_judge_output(
                    fixture['raw_judge_output'], reply, schema=judge_schema(reply),
                )
                self.assertTrue(result.usable, result.errors)
                parsed = result.output
                self.assertEqual(parsed.reported_score, fixture['expected_reported_score'])
                self.assertEqual(parsed.score, fixture['expected_effective_score'])
                self.assertEqual(parsed.procedure_ceiling, 25)
                self.assertEqual(parsed.verdict.value, fixture['expected_effective_verdict'])
                self.assertEqual(parsed.application.value, fixture['expected_effective_application'])
                self.assertEqual(parsed.procedure.reservation, 'material')
                self.assertIn('procedure_score_capped', parsed.warnings)

    def test_raw_caps_and_procedure_caps_combine_without_raising_scores(self):
        # Contract table: each row's procedure ceiling is independent of raw fields.
        procedures = (
            (None, None, 100),
            ('commitment', 'absent', 89),
            ('reservation', 'material', 50),
            ('reservation', 'unclear', 50),
            ('premise', 'unclear', 50),
            ('inference', 'unclear', 50),
            ('inference', 'rejected', 25),
            ('premise', 'conditional', 25),
        )
        for field, status, procedure_cap in procedures:
            for verdict, verdict_cap in (('not_demonstrated', 25), ('ambiguous', 50), ('demonstrated', 100)):
                for application, application_cap in (('none', 100), ('abstract', 89), ('action_governing', 100)):
                    for reported in (100, 12.5):
                        with self.subTest(field=field, status=status, verdict=verdict, application=application, reported=reported):
                            value = assessment()
                            value.update(score=reported, verdict=verdict, application=application)
                            if field:
                                value['procedure'][field] = {
                                    'status': status,
                                    'evidence': [] if status == 'absent' else [{
                                        'evidence_id': 'e0',
                                        'polarity': 'resists' if status in {'material', 'rejected'} else 'supports',
                                    }],
                                }
                            raw = json.dumps(value)
                            result = validate_judge_output(raw, 'Grounded reply.', schema=judge_schema('Grounded reply.'))
                            self.assertTrue(result.usable, result.errors)
                            ceiling = min(procedure_cap, verdict_cap, application_cap)
                            self.assertEqual(result.output.procedure_ceiling, ceiling)
                            self.assertEqual(result.output.score, min(Decimal(str(reported)), ceiling))
                            self.assertEqual(result.output.reported_score, Decimal(str(reported)))
                            self.assertEqual('procedure_score_capped' in result.output.warnings, reported > ceiling)
                            self.assertEqual(json.dumps(value), raw)

    def test_decisions_own_evidence_without_a_second_list(self):
        value=assessment();value['procedure']['inference']['evidence'][0]['evidence_id']='e1'
        value['procedure']['commitment']['evidence'][0]['evidence_id']='e2'
        reply='Owned premise.\nOwned inference.\nOwned governing commitment.'
        result=validate_judge_output(canonical_json(value),reply,schema=judge_schema(reply))
        self.assertTrue(result.usable,result.errors)
        self.assertEqual([e.quote for e in result.output.evidence],reply.splitlines())
        self.assertEqual(result.output.procedure.commitment_evidence,('Owned governing commitment.',))
        value['evidence']=[]
        self.assertFalse(validate_judge_output(canonical_json(value),reply,schema=judge_schema(reply)).usable)

    def test_duplicate_references_add_no_evidence_weight(self):
        value=assessment();value['procedure']['commitment']['evidence']*=3
        result=validate_judge_output(canonical_json(value),'Owned rule.',schema=judge_schema('Owned rule.'))
        self.assertTrue(result.usable,result.errors)
        self.assertEqual(len(result.output.evidence),1)
        self.assertEqual(result.output.procedure.commitment_evidence,('Owned rule.',))

    def test_every_mechanical_failure_is_already_a_schema_failure(self):
        variants=[]
        wrong_id=assessment();wrong_id['procedure']['premise']['evidence'][0]['evidence_id']='e999';variants.append(wrong_id)
        wrong_polarity=assessment();wrong_polarity['procedure']['commitment']['evidence'][0]['polarity']='resists';variants.append(wrong_polarity)
        no_evidence=assessment();no_evidence['procedure']['inference']['evidence']=[];variants.append(no_evidence)
        absent_with_evidence=assessment();absent_with_evidence['procedure']['commitment']['status']='absent';variants.append(absent_with_evidence)
        wrong_reservation=assessment();wrong_reservation['procedure']['reservation']={'status':'material','evidence':[{'evidence_id':'e0','polarity':'supports'}]};variants.append(wrong_reservation)
        blank=assessment();blank['justification']=' ';variants.append(blank)
        for value in variants:
            with self.subTest(value=value):
                with self.assertRaises(DomainError):validate_json_schema_instance(value,judge_schema('reply'))
                self.assertFalse(validate_judge_output(canonical_json(value),'reply',schema=judge_schema('reply')).usable)

    def test_only_id_enum_changes_and_transport_family_is_stable(self):
        template=json.loads((ROOT/'schemas/judge_output.json').read_text());original=deepcopy(template)
        one=materialize_judge_schema(template,'one');two=materialize_judge_schema(template,'one\n\ntwo')
        self.assertEqual(template,original)
        self.assertEqual(two['$defs']['EvidenceId']['enum'],['e0','e1'])
        self.assertNotEqual(digest_json(one),digest_json(two))
        self.assertEqual(judge_schema_family_digest(one),digest_json(template))
        self.assertEqual(judge_schema_family_digest(two),digest_json(template))

    def test_summary_retains_all_decisions_and_conflicting_polarities(self):
        value=assessment();value['procedure']['reservation']={'status':'material','evidence':[{'evidence_id':'e0','polarity':'resists'}]}
        result=validate_judge_output(canonical_json(value),'Mixed statement.',schema=judge_schema('Mixed statement.'))
        self.assertTrue(result.usable,result.errors)
        self.assertEqual([e.polarity for e in result.output.evidence],['supports','resists'])
        self.assertEqual(result.output.score,50)


class CatalogProvider:
    def __init__(self):
        raw=(ROOT/'fixtures/dryrun/fatalism.jsonl').read_bytes()
        self.content_digest=sha256_bytes(raw)
        self.records={x['effective_model_id']:x for x in map(json.loads,raw.splitlines())}
        self.requests=[];self.judges=0

    def complete(self,request):
        self.requests.append(request);record=deepcopy(self.records[request.requested_model])
        if request.role.value=='target':record['text']+='\nA second exact evidence line.'
        if request.role.value=='judge':
            self.judges+=1
            value=json.loads(record['text'])
            value['procedure']['inference']['evidence'][0]['evidence_id']='e999' if self.judges==1 else 'e1'
            record['text']=canonical_json(value)
            if self.judges==1:signal.raise_signal(signal.SIGINT)
        record['raw_body']['data']=record['text']
        return _parse_script_record(record,1).result


class JudgeJournalBindingTests(unittest.TestCase):
    def test_json_object_fallback_is_shared_across_different_catalog_sizes(self):
        class Provider(CatalogProvider):
            targets=0
            def complete(self,request):
                self.requests.append(request)
                record=deepcopy(self.records[request.requested_model])
                if request.role.value=='target':
                    self.targets+=1
                    record['text']+='\nExtra line.'*self.targets
                if request.role.value=='judge':
                    self.judges+=1
                    if self.judges==1:
                        return ProviderFailure(ProviderErrorKind.UNSUPPORTED_RESPONSE_FORMAT,'known_no_output','response_format_not_supported',None,None,400,None)
                record['raw_body']['data']=record['text']
                return _parse_script_record(record,1).result
        provider=Provider()
        with tempfile.TemporaryDirectory() as directory:
            config=dryrun_config('fatalism',Path(directory));config['seed']['bootstrap_limit']=2
            with patch('quasilisk.engine.create_provider',return_value=provider):run,reason=run_new(config)
            self.assertEqual(reason.value,'completed_plan')
            validate_run_evidence(run)
            requests=[r for r in provider.requests if r.role.value=='judge']
            self.assertEqual([r.response_mode for r in requests],['json_schema','json_object','json_object'])
            self.assertEqual([len(r.response_schema['$defs']['EvidenceId']['enum']) for r in requests],[2,2,3])

    def test_interrupted_repair_resume_uses_the_frozen_exact_catalog(self):
        provider=CatalogProvider()
        with tempfile.TemporaryDirectory() as directory:
            with patch('quasilisk.engine.create_provider',return_value=provider):
                run,reason=run_new(dryrun_config('fatalism',Path(directory)))
                self.assertEqual(reason.value,'operator_interrupted')
                _,reason=resume_run(run)
            self.assertEqual(reason.value,'completed_no_eligible_work')
            validate_run_evidence(run)
            requests=[r for r in provider.requests if r.role.value=='judge']
            self.assertEqual(len(requests),2)
            self.assertEqual(requests[1].route_kind,'schema_repair')
            for r in requests:
                self.assertEqual(list(r.response_schema['$defs']['EvidenceId']['enum']),['e0','e1'])
                self.assertEqual(r.response_schema_digest,digest_json(r.response_schema))
            events=read_events(run)
            index=next(i for i,e in enumerate(events) if e['type']=='model_call_intended' and e['payload']['role']=='judge')
            prefix=deepcopy(events[:index+1]);payload=prefix[-1]['payload']
            body=json.loads(payload['request_body']['data']);schema=body['response_format']['json_schema']['schema']
            schema['$defs']['EvidenceId']['enum'].append('e999')
            payload['request_body']['data']=canonical_json(body);payload['response_schema_digest']=digest_json(schema)
            last=prefix[-1]
            prefix[-1]=build_event(run_id=last['run_id'],sequence=last['sequence'],event_type=last['type'],payload=payload,previous_hash=last['previous_hash'],causes=last['causes'],timestamp_utc=last['timestamp_utc'])
            with self.assertRaisesRegex(JournalCorrupt,'response schema is not frozen'):
                reduce_events(prefix)
