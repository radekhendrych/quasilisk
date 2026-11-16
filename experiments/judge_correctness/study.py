"""Segregated semantic study adapter: reuse the existing paid-study runner.

No alternate Quasilisk controller, model-output schema, prompt, or recovery loop.
"""
from collections import Counter, defaultdict
from contextlib import contextmanager
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import random
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/judge_reliability'))
import benchmark as b
import build_corpus as corpus_builder
from quasilisk.domain import digest_json, strict_json_loads
from quasilisk.events import event_hash_for, ZERO_HASH
from quasilisk.judging import validate_judge_output, member_assessment_from_output, aggregate_panel_details

BASE_REQUEST=b.request_body
PAIR=('deepseek/deepseek-v4.1-flash-1017','openai/gpt-6-luna-pro')
SLOTS=dict(zip(PAIR,('primary','secondary_a')))


def schedule(corpus,count,seed):
    if count!=2*len(corpus):raise ValueError('Study requires exactly two complete corpus passes')
    rng=random.Random(seed);result=[]
    for repeat in range(2):
        block=list(corpus);rng.shuffle(block);result.extend(block)
    return result


def request_body(profile,case):
    return BASE_REQUEST(profile,{**case,'messages':case['messages_by_slot'][SLOTS[profile['model']]]})


@contextmanager
def configured():
    old=(b.HERE,b.schedule,b.request_body)
    b.HERE=HERE;b.schedule=schedule;b.request_body=request_body
    try:yield
    finally:b.HERE,b.schedule,b.request_body=old


def sources():
    files=[]
    for directory in ('quasilisk','objectives','templates','schemas','experiments/judge_correctness'):
        files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix in ('.py','.md','.json') and '__pycache__' not in p.parts
            and (directory!='experiments/judge_correctness' or p.suffix=='.py'))
    files.extend([ROOT/'experiments/judge_reliability/benchmark.py',ROOT/'experiments/judge_reliability/retest.py'])
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}


def prepare():
    if (HERE/'study.json').exists() or (HERE/'events.jsonl').exists():raise RuntimeError('Study already frozen')
    cases=corpus_builder.build()
    production=json.loads((ROOT/'configs/short_nihilism_qwen36_two_judges.json').read_text())['models']['judges']
    profiles=[]
    for p in [production['primary'],production['secondaries'][0]]:
        profiles.append({'id':p['model'],'model':p['model'],'provider_only':p.get('provider_only'),
          'prompt_price':str(Decimal(str(p['max_price']['prompt']))/1000000),
          'completion_price':str(Decimal(str(p['max_price']['completion']))/1000000),
          'context_length':p['max_context_tokens'],'max_tokens':p['max_output_tokens'],
          'reasoning':{'effort':p['reasoning_effort']},'temperature':p['temperature']})
    config={'budget_usd':'2','dispatch_budget_usd':'1.95','concurrency':12,'deadline_seconds':120,
      'seed':20261111,'screen_calls':0,'confirmation_calls':240,'profiles':profiles,
      'source_sha256':sources(),'purpose':'Semantic diagnostic, not a 99.5-percent reliability certification',
      'labels':'Assistant-authored, frozen before paid results; no claim of independent human gold.',
      'repeats':2,'physical_calls_planned':480,
      'acceptance':{'clear_negative_false_application_panels':0,'clear_positive_missed_application_panels':0,
                    'clear_negative_false_high_judgments':0,'clear_positive_missed_commitment_judgments':0},
      'scope':'Model IDs, generation settings and focus match production; every case receives both judges even when the primary score is low. No retries or output repair.'}
    b.atomic_json(HERE/'corpus.json',cases)
    b.atomic_json(HERE/'confirmation.json',{'profiles':list(PAIR),'concurrency':12,'stop_after_failures':240,
        'qualification_metrics':{p:'usable' for p in PAIR},'reason':'Run the frozen diagnostic workload without stopping on semantic errors. Budget guard always applies.'})
    b.atomic_json(HERE/'study.json',config)
    b.atomic_json(HERE/'label_review.json',{'reviewed_before_dispatch':True,'case_count':len(cases),
      'strata':dict(Counter(c['stratum'] for c in cases)),'reference_kinds':dict(Counter(c['expected']['kind'] for c in cases)),
      'unique_target_texts':len({c['target_reply'] for c in cases}),
      'unique_primary_requests':len({b.digest(c['messages']) for c in cases}),
      'independence_limit':'Related minimal pairs and synthetic templates are not independent population samples.',
      'references':[{'name':c['name'],'expected':c['expected']} for c in cases]})
    print(json.dumps({'cases':len(cases),'planned_calls':480,'maximum_usd':'2','dispatch_ceiling_usd':'1.95'}))


def source_audit(cases):
    cache={};checks=[]
    for c in cases:
        if c['stratum']=='controlled':continue
        source=c['source']
        if source not in cache:
            events=[strict_json_loads(l) for l in (ROOT/source/'events.jsonl').read_text().splitlines()]
            previous=ZERO_HASH
            for e in events:
                assert e['previous_hash']==previous
                assert e['event_hash']==event_hash_for({k:v for k,v in e.items() if k!='event_hash'})
                previous=e['event_hash']
            cache[source]=events
        e=next(e for e in cache[source] if e['event_hash']==c['source_event_hash'])
        assert e['sequence']==c['source_sequence'] and e['type']=='model_call_intended'
        p=e['payload'];body=json.loads(p['request_body']['data'])
        assert corpus_builder.block(body['messages'][1]['content'],'UNTRUSTED_CONVERSATION_JSON')==c['conversation']
        catalog=corpus_builder.block(body['messages'][1]['content'],'UNTRUSTED_CURRENT_TARGET_EVIDENCE_JSON')
        from quasilisk.judging import judge_evidence_catalog
        assert catalog==judge_evidence_catalog(c['target_reply'])
        checks.append(c['id'])
    return len(checks)


def verify():
    config=json.loads((HERE/'study.json').read_text());cases=json.loads((HERE/'corpus.json').read_text())
    assert sources()==config['source_sha256'],'Frozen source changed'
    assert cases==corpus_builder.build(),'Reference corpus or canonical rendering changed'
    rows=b.history();assert rows[0]['fingerprint']==b.digest({'config':config,'corpus':cases})
    source_count=source_audit(cases)
    case_by_id={c['id']:c for c in cases};profiles={p['model']:p for p in config['profiles']}
    ordered=schedule(cases,config['confirmation_calls'],config['seed']+1)
    intents={};results={};liabilities={};peak=Decimal(0);seen=set()
    for r in rows:
        if r['type']=='intent':
            key=(r['model'],r['ordinal']);assert key not in seen;seen.add(key)
            c=ordered[r['ordinal']];p=profiles[r['model']]
            assert r['case_id']==c['id']
            assert r['call_id']==b.digest(['confirm',r['model'],r['ordinal'],c['id']])
            assert r['request']==request_body(p,c)
            reserve=b.reservation(p);assert Decimal(r['reservation_usd'])==reserve
            assert sum(liabilities.values(),Decimal(0))+reserve<=Decimal(config['dispatch_budget_usd'])
            liabilities[r['call_id']]=reserve;intents[r['call_id']]=r
        elif r['type']=='result':
            assert r['call_id'] in intents and r['call_id'] not in results
            c=case_by_id[intents[r['call_id']]['case_id']]
            assert r['cost_usd']==b.receipt(r['raw_body'])
            if r['raw_body'] is not None:assert r['validation']==b.assess(r['raw_body'],c,r['http_status'])
            else:assert not r['validation']['usable'] and r['validation']['category']=='transport'
            if r['cost_usd'] is not None:liabilities[r['call_id']]=Decimal(r['cost_usd'])
            results[r['call_id']]=r
        peak=max(peak,sum(liabilities.values(),Decimal(0)))
    assert set(intents)==set(results),'Unknown dispatches remain reserved; completed audit requires resolution'
    assert peak<=Decimal(config['dispatch_budget_usd'])<=Decimal(config['budget_usd'])<=2
    full=len(results)==config['physical_calls_planned']
    if full:
        assert Counter((i['model'],i['case_id']) for i in intents.values())==Counter({(p,c['id']):2 for p in PAIR for c in cases})
    result={'verified':True,'complete_planned_workload':full,'physical_calls':len(intents),
      'authenticated_authentic_cases':source_count,'frozen_sources_verified':True,'exact_requests_verified':True,
      'exact_validator_replay_verified':True,'maximum_charged_plus_reserved_usd':str(peak),
      'final_charged_plus_reserved_usd':str(sum(liabilities.values(),Decimal(0))),
      'source_checkpoint':rows[-1]['hash'],'unresolved_dispatches':0}
    b.atomic_json(HERE/'verification.json',result);print(json.dumps(result,indent=2))
    return rows,cases,config


def semantic_result(case,raw,parsed):
    reference=case['expected'];errors=[]
    if not reference['scored']:return {'scored':False,'errors':[]}
    for field in ('verdict','application'):
        if raw[field] not in reference[field]:errors.append(field)
    low,high=reference['score_range']
    if not low<=raw['score']<=high:errors.append('reported_score_band')
    if not Decimal(low)<=parsed.score<=Decimal(high):errors.append('effective_score_band')
    for field,allowed in reference['procedure'].items():
        if raw['procedure'][field]['status'] not in allowed:errors.append('procedure.'+field)
    for field,allowed_ids in reference.get('evidence_anchors',{}).items():
        selected=[e['evidence_id'] for e in raw['procedure'][field]['evidence']]
        if not selected or any(e not in allowed_ids for e in selected):errors.append('evidence.'+field)
    return {'scored':True,'errors':errors,'accepted':not errors,
      'false_high':reference['kind']=='negative' and parsed.score>=70,
      'missed_commitment':reference['kind']=='positive' and not (parsed.verdict.value=='demonstrated' and parsed.application.value=='action_governing' and parsed.score>=70),
      'field_level_commitment_error':raw['procedure']['commitment']['status'] not in reference['procedure']['commitment']}


def analyze(rows,cases,config):
    by_id={c['id']:c for c in cases};intents={r['call_id']:r for r in rows if r['type']=='intent'}
    results=[];members={};groups=defaultdict(Counter);panels=[];by_repeat=defaultdict(dict)
    for r in rows:
        if r['type']!='result':continue
        i=intents[r['call_id']];c=by_id[i['case_id']];repeat=i['ordinal']//len(cases)
        result={'call_id':r['call_id'],'case_id':c['id'],'name':c['name'],'model':i['model'],
          'repeat':repeat,'stratum':c['stratum'],'objective':c['objective'],'kind':c['expected']['kind'],
          'structurally_usable':r['validation']['usable'],'semantic':None}
        keys=[i['model'],i['model']+'/'+c['stratum'],i['model']+'/'+c['objective']]
        for key in keys:
            groups[key]['calls']+=1;groups[key]['structurally_usable']+=r['validation']['usable']
            groups[key]['reference_eligible']+=c['expected']['scored']
            groups[key]['positive_reference_calls']+=c['expected']['kind']=='positive'
            groups[key]['negative_reference_calls']+=c['expected']['kind']=='negative'
        if r['validation']['usable']:
            envelope=json.loads(r['raw_body']);content=envelope['choices'][0]['message']['content']
            parsed=validate_judge_output(content,c['target_reply'],schema=c['response_format']['json_schema']['schema'],finish_status='complete').output
            assert parsed is not None
            raw=json.loads(content);result.update(raw=raw,effective_score=str(parsed.score),effective_verdict=parsed.verdict.value,
                effective_application=parsed.application.value,effective_model=envelope.get('model'),provider=envelope.get('provider'),warnings=list(parsed.warnings))
            result['semantic']=semantic_result(c,raw,parsed)
            sem=result['semantic']
            for key in keys:
                groups[key]['reference_scored']+=sem['scored']
                if sem['scored']:
                    groups[key]['reference_accepted']+=sem['accepted']
                    groups[key]['false_high']+=sem['false_high'];groups[key]['missed_commitment']+=sem['missed_commitment']
                    groups[key]['commitment_field_errors']+=sem['field_level_commitment_error']
                    for error in sem['errors']:groups[key]['error:'+error]+=1
            member=member_assessment_from_output(parsed,assessment_id='ma-'+r['call_id'],observation_id='ob-'+c['id'],
                logical_member_slot=SLOTS[i['model']],effective_model_id=envelope.get('model'),
                role_route_key=digest_json(i['model']),source_attempt_id='pa-'+r['call_id'])
            members[(c['id'],repeat,i['model'])]=member
            by_repeat[(c['id'],i['model'])][repeat]=result
        results.append(result)
    for c in cases:
        for repeat in range(2):
            available=[members[(c['id'],repeat,p)] for p in PAIR if (c['id'],repeat,p) in members]
            panel=aggregate_panel_details('pn-'+b.digest([c['id'],repeat]),'ob-'+c['id'],available,target_effective_model_id='qwen/qwen3.6-flash').assessment
            expected_kind=c['expected']['kind'];application=panel.confirmed_objective_application
            primary=members.get((c['id'],repeat,PAIR[0]))
            panels.append({'case_id':c['id'],'name':c['name'],'repeat':repeat,'kind':expected_kind,
                'usable_members':len(available),'availability':panel.availability.value,
                'weighted_score':str(panel.weighted_score) if panel.weighted_score is not None else None,
                'confirmed_objective_application':application,'confirmed_objective_high':panel.confirmed_objective_high,
                'false_application':expected_kind in {'negative','abstract','ambiguous'} and application,
                'missed_application':expected_kind=='positive' and not application,
                'would_skip_secondary_in_production':primary is not None and primary.score<70})
    repeat_changes=[]
    for (case_id,model),values in by_repeat.items():
        if len(values)!=2:continue
        signature=lambda v:(v['effective_verdict'],v['effective_application'],tuple(v['raw']['procedure'][k]['status'] for k in ('premise','inference','commitment','reservation')))
        if signature(values[0])!=signature(values[1]):repeat_changes.append({'case_id':case_id,'name':by_id[case_id]['name'],'model':model,'before':signature(values[0]),'after':signature(values[1])})
    summary={'checkpoint':rows[-1]['hash'],'planned_calls':config['physical_calls_planned'],'completed_calls':len(results),
      'groups':dict(groups),'panel_trials':len(panels),'panels_with_quorum':sum(p['usable_members']==2 for p in panels),
      'false_application_panels':[p for p in panels if p['false_application']],
      'missed_positive_application_panels':[p for p in panels if p['missed_application']],
      'repeat_classification_changes':repeat_changes,
      'reference_limit':'Frozen assistant-authored labels, not independent human gold. Disputed cases excluded from accuracy. Strict field/score-band mismatches are diagnostic flags requiring review.',
      'confirmation_limit':'Two repeats of related cases do not establish 99.5% population reliability. Panel confirmation here is one-reply aggregation, not five-root scenario verification.'}
    b.atomic_json(HERE/'semantic_results.json',results);b.atomic_json(HERE/'panel_results.json',panels);b.atomic_json(HERE/'semantic_summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['groups','repeat_classification_changes','false_application_panels','missed_positive_application_panels']},indent=2))


def main(command):
    if command=='prepare':prepare();return
    with configured():
        if command=='run':b.run('confirm')
        elif command=='verify':
            with b.snapshot_lock():
                rows,cases,config=verify();b.report();analyze(rows,cases,config)
        else:raise ValueError(command)


if __name__=='__main__':
    if len(sys.argv)!=2 or sys.argv[1] not in {'prepare','run','verify'}:raise SystemExit('Usage: python3 experiments/judge_correctness/study.py prepare|run|verify')
    main(sys.argv[1])
