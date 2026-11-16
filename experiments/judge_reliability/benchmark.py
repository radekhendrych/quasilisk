"""Standalone, maintainer-authorized provider study; never a Quasilisk run gate.

Uses recorded requests and canonical validators, with no generated prompts,
alternative engine, output repair, or production configuration modification.
All paid operations require study.json and append intent before dispatch.
"""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from collections import Counter, defaultdict
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from quasilisk.domain import DomainError, strict_json_loads, validate_json_schema_instance
from quasilisk.judging import validate_judge_output


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def prepare():
    paths = {'fatalism': json.loads((ROOT/'exports/funded_fatalism_20261110/comparison.json').read_text())['funded']['run_directory']}
    for mode in ('nihilism', 'relativism'):
        paths[mode] = json.loads((ROOT/'exports/funded_other_modes_20261110'/f'{mode}.evidence.json').read_text())['run_directory']
    corpus = []; seen = set()
    for mode, directory in paths.items():
        events = [json.loads(line) for line in (ROOT/directory/'events.jsonl').read_text().splitlines()]
        intents = {e['payload']['physical_attempt_id']: e for e in events if e['type'] == 'model_call_intended'}
        results = {e['payload']['physical_attempt_id']: e for e in events if e['type'] == 'model_call_succeeded'}
        targets = {}
        for attempt, e in intents.items():
            p = e['payload']
            if p['role'] == 'target' and attempt in results and results[attempt]['payload']['finish_status'] == 'complete':
                targets[tuple(p['lineage'][k] for k in ('execution_id', 'candidate_id', 'parent_node_id'))] = results[attempt]['payload']['text']
        for attempt, e in intents.items():
            p = e['payload']
            if p['role'] != 'judge' or p['attempt_ordinal'] != 0:
                continue
            target = targets.get(tuple(p['lineage'][k] for k in ('execution_id', 'candidate_id', 'parent_node_id')))
            if target is None:
                continue
            body = json.loads(p['request_body']['data'])
            key = digest({'messages': body['messages'], 'schema': body['response_format']})
            if key in seen:
                continue
            seen.add(key)
            slot = p['lineage']['logical_member_slot']
            corpus.append({'id': key, 'objective': mode, 'focus_slot': slot,
                           'source': directory, 'source_event_hash': e['event_hash'],
                           'source_sequence': e['sequence'], 'target_reply': target,
                           'messages': body['messages'], 'response_format': body['response_format']})
    if (HERE/'events.jsonl').exists():
        raise RuntimeError('Cannot replace corpus after paid evidence exists')
    atomic_json(HERE/'corpus.json', corpus)
    print(json.dumps({'cases': len(corpus), 'strata': dict(Counter(c['objective']+'/'+c['focus_slot'] for c in corpus)), 'largest_request_bytes': max(len(json.dumps(c).encode()) for c in corpus)}))


def assess(raw, case, http_status=200):
    errors = []; native = False; grounded = False; finish = None
    try:
        envelope = json.loads(raw)
        if http_status != 200 or envelope.get('error'):
            return {'usable': False, 'native_schema': False, 'category': 'provider_error', 'errors': [str(envelope.get('error', http_status))]}
        choice = envelope['choices'][0]; finish = choice.get('finish_reason')
        content = choice['message'].get('content')
        if finish != 'stop':
            errors.append('incomplete finish: '+str(finish))
        if not isinstance(content, str):
            errors.append('missing text content')
        else:
            try:
                validate_json_schema_instance(strict_json_loads(content), case['response_format']['json_schema']['schema'])
                native = True
            except DomainError as exc:
                errors.append(str(exc))
            parsed = validate_judge_output(content, case['target_reply'],
                                           schema=case['response_format']['json_schema']['schema'])
            grounded = parsed.output is not None
            errors.extend(parsed.errors)
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        errors.append(type(exc).__name__+': '+str(exc))
    usable = native and grounded and finish == 'stop'
    category = 'valid' if usable else 'incomplete' if finish != 'stop' else 'schema' if not native else 'evidence_links'
    return {'usable': usable, 'native_schema': native and finish == 'stop', 'category': category, 'errors': sorted(set(errors))}


def request_body(profile, case):
    body = {'model': profile['model'], 'messages': case['messages'],
            'response_format': case['response_format'], 'max_tokens': profile['max_tokens'],
            'provider': {'require_parameters': True, 'allow_fallbacks': False, 'sort': 'price',
                         'max_price': {'prompt': float(Decimal(profile['prompt_price'])*1000000),
                                       'completion': float(Decimal(profile['completion_price'])*1000000),
                                       'request': 0}}}
    if profile['temperature'] is not None:
        body['temperature'] = profile['temperature']
    if profile['reasoning'] is not None:
        body['reasoning'] = profile['reasoning']
    if profile.get('provider_only'):
        body['provider']['only'] = [profile['provider_only']]
    return body


def reservation(profile):
    # Full advertised context at explicit rate ceiling, plus maximum output.
    # This deliberately avoids assuming a local token estimator is an upper bound.
    return Decimal(profile['context_length'])*Decimal(profile['prompt_price']) + Decimal(profile['max_tokens'])*Decimal(profile['completion_price'])


def dispatch(body, deadline):
    def timeout(signum, frame):
        raise TimeoutError('aggregate request deadline exceeded')
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, deadline)
    started = time.monotonic()
    try:
        request = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',
            data=json.dumps(body, ensure_ascii=False).encode(), headers={
                'Authorization': 'Bearer '+os.environ['OPENROUTER_API_KEY'], 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=deadline) as response:
                raw = response.read().decode('utf-8'); status = response.status
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode('utf-8', errors='replace'); status = exc.code
        return {'raw_body': raw, 'http_status': status, 'latency_seconds': time.monotonic()-started}
    except Exception as exc:
        return {'raw_body': None, 'http_status': None, 'latency_seconds': time.monotonic()-started, 'transport_error': type(exc).__name__+': '+str(exc)}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def history():
    path = HERE/'events.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    previous = '0'*64
    for row in rows:
        payload = {k:v for k,v in row.items() if k != 'hash'}
        if row['previous_hash'] != previous or digest(payload) != row['hash']:
            raise RuntimeError('Study journal hash-chain mismatch')
        previous = row['hash']
    return rows


@contextmanager
def snapshot_lock():
    """An offline audit must not race a live writer or inspect partial work."""
    with (HERE/'.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Study is still running; audit after the writer finishes.')
        yield


def append(rows, kind, **payload):
    row = {'sequence': len(rows)+1, 'type': kind, 'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'previous_hash': rows[-1]['hash'] if rows else '0'*64, **payload}
    row['hash'] = digest(row)
    with (HERE/'events.jsonl').open('a') as stream:
        stream.write(json.dumps(row, ensure_ascii=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
    rows.append(row)


def charged(rows):
    intents = {r['call_id']: Decimal(r['reservation_usd']) for r in rows if r['type']=='intent'}
    for r in rows:
        if r['type']=='result' and r['cost_usd'] is not None:
            intents[r['call_id']] = Decimal(r['cost_usd'])
    return sum(intents.values(), Decimal(0))


def receipt(raw):
    try:
        value = json.loads(raw, parse_float=Decimal).get('usage', {}).get('cost')
        if value is None or isinstance(value, bool):
            return None
        cost = Decimal(str(value))
        return str(cost) if cost.is_finite() and cost >= 0 else None
    except (ValueError, TypeError, AttributeError, InvalidOperation):
        return None


def schedule(corpus, count, seed):
    groups = defaultdict(list)
    for case in corpus:
        groups[case['objective']+'/'+case['focus_slot']].append(case)
    rng = random.Random(seed)
    keys = sorted(groups)
    # Balanced strata, random cases without replacement within each stratum cycle.
    pools = {}
    result = []
    for i in range(count):
        key = keys[i % len(keys)]
        if not pools.get(key):
            pools[key] = list(groups[key]); rng.shuffle(pools[key])
        result.append(pools[key].pop())
    return result


def run(stage):
    config = json.loads((HERE/'study.json').read_text()); corpus = json.loads((HERE/'corpus.json').read_text())
    for source, expected in config['source_sha256'].items():
        if hashlib.sha256((ROOT/source).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Canonical validator or prompt source changed: '+source)
    if not os.environ.get('OPENROUTER_API_KEY'):
        raise RuntimeError('OPENROUTER_API_KEY is absent')
    if Decimal(config['dispatch_budget_usd']) > Decimal(config['budget_usd']) or Decimal(config['budget_usd']) > 5:
        raise RuntimeError('Study exceeds authorized ceiling')
    with (HERE/'.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = history(); fingerprint = digest({'config':config, 'corpus':corpus})
        if not rows:
            append(rows, 'study_created', fingerprint=fingerprint, config=config, corpus_digest=digest(corpus))
        elif rows[0]['fingerprint'] != fingerprint:
            raise RuntimeError('Frozen study changed')
        if any(r['type']=='intent' and not any(s['type']=='result' and s['call_id']==r['call_id'] for s in rows) for r in rows):
            raise RuntimeError('Unresolved dispatch: reserve retained; inspect before further work')
        profiles = config['profiles']
        if stage == 'routing_screen':
            extension = json.loads((HERE/'extension.json').read_text())
            existing = next((r for r in rows if r['type']=='extension_selected'), None)
            if existing and existing['extension'] != extension:
                raise RuntimeError('Frozen extension changed')
            if not existing: append(rows, 'extension_selected', extension=extension)
            profiles = extension['profiles']
        if stage == 'confirm':
            chosen = json.loads((HERE/'confirmation.json').read_text())
            extension = next((r['extension'] for r in rows if r['type']=='extension_selected'), {'profiles':[]})
            profiles = [p for p in profiles+extension['profiles']+chosen.get('new_profiles',[]) if p.get('id',p['model']) in chosen['profiles']]
            if len(profiles) != len(chosen['profiles']):
                raise RuntimeError('Unknown confirmation profile')
            if any(r['type']=='confirmation_selected' for r in rows):
                frozen = next(r for r in rows if r['type']=='confirmation_selected')['selection']
                if frozen != chosen: raise RuntimeError('Confirmation selection changed')
            else:
                append(rows, 'confirmation_selected', selection=chosen)
        count = config['confirmation_calls'] if stage == 'confirm' else config['screen_calls'] if stage == 'screen' else extension['calls']
        workers = chosen.get('concurrency', config['concurrency']) if stage == 'confirm' else config['concurrency']
        if not isinstance(workers,int) or not 1 <= workers <= 16:
            raise RuntimeError('Invalid worker bound')
        cases = schedule(corpus, count, config['seed']+(stage=='confirm'))
        tasks = [(profile, case, ordinal) for ordinal, case in enumerate(cases) for profile in profiles]
        done = {r['call_id'] for r in rows if r['type']=='intent'}
        intent_by_id = {r['call_id']:r for r in rows if r['type']=='intent'}
        failures = Counter()
        def count_failure(intent, result):
            if stage != 'confirm' or intent['stage'] != stage: return
            key = intent.get('profile_id',intent['model'])
            metric = chosen['qualification_metrics'][key]
            if not result['validation'][metric]: failures[key] += 1
        for row in rows:
            if row['type']=='result':count_failure(intent_by_id[row['call_id']],row)
        pending = {}; index = 0; stopped = False
        with ProcessPoolExecutor(max_workers=workers) as pool:
            while index < len(tasks) or pending:
                while index < len(tasks) and len(pending)<workers and not stopped:
                    profile, case, ordinal = tasks[index]; index += 1
                    profile_id = profile.get('id',profile['model'])
                    # Four failures cannot meet 99.5% observed accuracy in 600
                    # trials even if every remaining planned request passes.
                    if stage == 'confirm' and failures[profile_id] >= chosen['stop_after_failures']:
                        continue
                    call_id = digest([stage,profile_id,ordinal,case['id']])
                    if call_id in done: continue
                    reserve = reservation(profile)
                    if charged(rows)+reserve > Decimal(config['dispatch_budget_usd']):
                        if pending:
                            index -= 1; break
                        append(rows, 'budget_stopped', charged_usd=str(charged(rows)), next_reservation_usd=str(reserve)); stopped=True; break
                    body = request_body(profile, case)
                    append(rows, 'intent', call_id=call_id, stage=stage, ordinal=ordinal,
                           model=profile['model'], profile_id=profile_id, case_id=case['id'], reservation_usd=str(reserve), request=body)
                    intent_by_id[call_id] = rows[-1]
                    future = pool.submit(dispatch, body, config['deadline_seconds'])
                    pending[future] = (call_id, profile, case, reserve)
                if not pending: break
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    call_id, profile, case, reserve = pending.pop(future)
                    try: response = future.result()
                    except Exception as exc: response = {'raw_body':None, 'transport_error':type(exc).__name__, 'http_status':None}
                    cost = receipt(response['raw_body'])
                    result = assess(response['raw_body'], case, response['http_status']) if response['raw_body'] is not None else {'usable':False,'native_schema':False,'category':'transport','errors':[response['transport_error']]}
                    append(rows, 'result', call_id=call_id, cost_usd=cost, validation=result, **response)
                    prior_failures = failures[intent_by_id[call_id]['profile_id']]
                    count_failure(intent_by_id[call_id],rows[-1])
                    if stage=='confirm':
                        key=intent_by_id[call_id]['profile_id']
                        if prior_failures < chosen['stop_after_failures'] <= failures[key]:
                            append(rows,'profile_stopped',profile_id=key,stage=stage,reason='cannot_meet_observed_threshold_within_fixed_confirmation',failures=failures[key])
                    print(json.dumps({'stage':stage,'model':profile['model'],'category':result['category'],'cost':cost,'charged_and_reserved':str(charged(rows))}), flush=True)
                    if cost is not None and Decimal(cost)>reserve:
                        append(rows,'price_bound_violation',call_id=call_id); stopped=True
                if stopped and not pending: break
        report()


def report():
    rows=history(); intents={r['call_id']:r for r in rows if r['type']=='intent'}
    groups={}
    for r in rows:
        if r['type']!='result': continue
        i=intents[r['call_id']]; key=i['stage']+':'+i.get('profile_id',i['model'])
        g=groups.setdefault(key, {'calls':0,'valid':0,'native_schema':0,'categories':Counter(),'cost_usd':Decimal(0),'unpriced':0,'cases':set(),'providers':Counter(),'latencies':[]})
        g['calls']+=1; g['valid']+=r['validation']['usable'];g['native_schema']+=r['validation']['native_schema'];g['categories'][r['validation']['category']]+=1;g['cases'].add(i['case_id'])
        if r['cost_usd'] is None:g['unpriced']+=1
        else:g['cost_usd']+=Decimal(r['cost_usd'])
        if r.get('latency_seconds') is not None:g['latencies'].append(r['latency_seconds'])
        try:g['providers'][json.loads(r['raw_body']).get('provider','unknown')]+=1
        except (ValueError,TypeError):pass
    for g in groups.values():
        g['cost_usd']=str(g['cost_usd']);g['distinct_cases']=len(g.pop('cases'));g['observed_valid_rate']=g['valid']/g['calls']
        g['full_contract_one_sided_95_lower_bound']=lower_bound(g['valid'],g['calls'])
        g['native_schema_one_sided_95_lower_bound']=lower_bound(g['native_schema'],g['calls'])
        lat=g.pop('latencies');g['mean_latency_seconds']=sum(lat)/len(lat) if lat else None
    result={'charged_plus_unresolved_reservations_usd':str(charged(rows)), 'receipt_cost_usd':str(sum((Decimal(r['cost_usd']) for r in rows if r['type']=='result' and r['cost_usd'] is not None),Decimal(0))), 'source_checkpoint':rows[-1]['hash'] if rows else None,'profiles':groups}
    atomic_json(HERE/'results.json', result)
    print(json.dumps(result,indent=2))


def lower_bound(successes, n, alpha=.05):
    """Exact one-sided Clopper-Pearson bound, without a numerical dependency."""
    if not successes: return 0.0
    if successes == n: return alpha**(1/n)
    failures = n-successes
    low, high = 0.0, 1.0
    coefficients = [math.lgamma(n+1)-math.lgamma(j+1)-math.lgamma(n-j+1) for j in range(failures+1)]
    for _ in range(70):
        q = (low+high)/2
        terms = [coefficients[j]+j*math.log(q)+(n-j)*math.log1p(-q) for j in range(failures+1)]
        maximum = max(terms)
        cdf = math.exp(maximum)*sum(math.exp(t-maximum) for t in terms)
        if cdf>alpha:low=q
        else:high=q
    return 1-(low+high)/2


if __name__=='__main__':
    if len(sys.argv)!=2 or sys.argv[1] not in ('prepare','screen','routing_screen','confirm','report'):
        raise SystemExit('Usage: python3 experiments/judge_reliability/benchmark.py prepare|screen|routing_screen|confirm|report')
    command=sys.argv[1]
    if command=='prepare':prepare()
    elif command=='report':
        with snapshot_lock():report()
    else:run(command)
