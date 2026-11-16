"""Offline replay of study evidence, validators, identity and every admission."""
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path
import benchmark as b
from quasilisk.events import event_hash_for, ZERO_HASH


def main(source_request_verifier=None):
    rows=b.history()
    corpus=json.loads((b.HERE/'corpus.json').read_text())
    config=json.loads((b.HERE/'study.json').read_text())
    assert rows[0]['fingerprint']==b.digest({'config':config,'corpus':corpus})
    cases={c['id']:c for c in corpus}
    source_checks=[]
    for directory in sorted({c['source'] for c in corpus}):
        path=b.ROOT/directory/'events.jsonl'
        if not path.exists():
            source_checks.append({'directory':directory,'available':False})
            continue
        events=[b.strict_json_loads(line) for line in path.read_text().splitlines()]
        previous=ZERO_HASH
        for event in events:
            assert event['previous_hash']==previous
            assert event['event_hash']==event_hash_for({k:v for k,v in event.items() if k!='event_hash'})
            previous=event['event_hash']
        by_hash={e['event_hash']:e for e in events}
        target_intents={e['payload']['physical_attempt_id']:e['payload'] for e in events if e['type']=='model_call_intended' and e['payload']['role']=='target'}
        target_results={e['payload']['physical_attempt_id']:e['payload'] for e in events if e['type']=='model_call_succeeded'}
        for case in corpus:
            if case['source']!=directory:continue
            event=by_hash[case['source_event_hash']];intent=event['payload']
            assert event['sequence']==case['source_sequence']
            body=json.loads(intent['request_body']['data'])
            if source_request_verifier is None:
                assert case['messages']==body['messages'] and case['response_format']==body['response_format']
            else:
                source_request_verifier(case, body)
            matches=[target_results[attempt]['text'] for attempt,target in target_intents.items() if attempt in target_results and target_results[attempt]['finish_status']=='complete' and all(target['lineage'][k]==intent['lineage'][k] for k in ('execution_id','candidate_id','parent_node_id'))]
            assert matches==[case['target_reply']]
        source_checks.append({'directory':directory,'available':True,'hash_chain_valid':True,
                              'requests_authenticated':True,'targets_exact':True,'checkpoint':previous})
    profiles={p.get('id',p['model']):p for p in config['profiles']}
    intents={};completed=set();prefix=[];peak=Decimal(0);counts=Counter();positions=set();stopped=set()
    phase_counts={'screen':config['screen_calls'],'confirm':config['confirmation_calls']}
    for row in rows:
        if row['type']=='extension_selected':
            profiles.update({p.get('id',p['model']):p for p in row['extension']['profiles']})
            phase_counts['routing_screen']=row['extension']['calls']
        elif row['type']=='confirmation_selected':
            profiles.update({p.get('id',p['model']):p for p in row['selection'].get('new_profiles',[])})
        elif row['type']=='intent':
            assert row['call_id'] not in intents
            profile_id=row.get('profile_id',row['model'])
            position=(row['stage'],profile_id,row['ordinal'])
            assert position not in positions and 0<=row['ordinal']<phase_counts[row['stage']]
            assert (row['stage'],profile_id) not in stopped
            positions.add(position)
            profile=profiles[row.get('profile_id',row['model'])];case=cases[row['case_id']]
            assert row['call_id']==b.digest([row['stage'],profile.get('id',profile['model']),row['ordinal'],case['id']])
            assert row['request']==b.request_body(profile,case)
            assert Decimal(row['reservation_usd'])==b.reservation(profile)
            assert b.charged(prefix)+b.reservation(profile)<=Decimal(config['dispatch_budget_usd'])
            intents[row['call_id']]=row
        elif row['type']=='result':
            assert row['call_id'] not in completed
            completed.add(row['call_id']);i=intents[row['call_id']]
            assert row['cost_usd']==b.receipt(row['raw_body'])
            if row['raw_body'] is not None:
                assert row['validation']==b.assess(row['raw_body'],cases[i['case_id']],row['http_status'])
            else:
                assert row['validation']['category']=='transport'
                assert not row['validation']['usable'] and not row['validation']['native_schema']
            counts[row['validation']['category']]+=1
        elif row['type']=='profile_stopped':
            stopped.add((row['stage'],row['profile_id']))
        prefix.append(row);peak=max(peak,b.charged(prefix))
    unresolved = set(intents) - completed
    operator_stop = next((row for row in rows if row['type']=='operator_stopped'), None)
    if unresolved:
        assert operator_stop is not None, 'Unresolved physical dispatches remain'
        assert rows[-1] == operator_stop, 'Evidence was appended after the operator stop'
        assert set(operator_stop['unresolved_call_ids']) == unresolved
        assert operator_stop['reason'] == 'user_requested_stop'
    assert peak<=Decimal(config['dispatch_budget_usd'])<=Decimal(config['budget_usd'])<=5
    result={'verified':True,'physical_calls':len(intents),'counts':dict(counts),
            'maximum_charged_plus_reserved_usd':str(peak),'final_charged_plus_reserved_usd':str(b.charged(rows)),
            'source_checkpoint':rows[-1]['hash'],'historical_sources':source_checks,
            'operator_stopped':operator_stop is not None,
            'unresolved_dispatches':len(unresolved),
            'checks':['hash chain','frozen study identity','physical dispatch uniqueness','exact request rendering','exact schema and evidence validation replay','receipt normalization','every pre-dispatch cost admission',
                      'operator-interrupted dispatches retain full reservations' if unresolved else 'no unresolved dispatches']}
    b.atomic_json(b.HERE/'verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    with b.snapshot_lock():main()
