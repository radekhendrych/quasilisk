"""Rebuild a read-only report from authenticated, frozen campaign evidence.

No model dispatches, scoring policy or reducers live here. Existing canonical
replay and campaign projections authenticate every exported scientific result.
"""
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CAMPAIGN = ROOT / 'experiments/multimodel_verification'
SITE = HERE / 'site'


def read(path):
    return json.loads(path.read_text())


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def authenticate(plan):
    """Replay with each run's exact source; never rewrite campaign artifacts."""
    for entry in plan['entries'] + plan['abandoned_entries']:
        if entry.get('not_run_reason'):
            assert not list((ROOT / entry['output_directory']).glob('run-*'))
            continue
        code = ('import sys,json; from pathlib import Path; '
                f'sys.path.insert(0,{str(CAMPAIGN)!r}); import audit; '
                'audit.atomic=lambda path,value:None; '
                'entry=json.loads(sys.argv[1]); result=audit.audit(entry); '
                f'expected=json.loads((Path({str(CAMPAIGN)!r})/(entry["cell"]+".audit.json")).read_text()); '
                'assert result==expected,"Frozen projection differs from canonical replay"')
        subprocess.run([sys.executable, '-c', code, json.dumps(entry)],
                       cwd=ROOT / entry.get('source_snapshot', '.'), check=True)
        print('Authenticated', entry['cell'], flush=True)
    sys.path.insert(0, str(CAMPAIGN))
    import rejudge
    captured = {}
    with rejudge.configured(), redirect_stdout(io.StringIO()), patch.object(
            rejudge.b, 'atomic_json', side_effect=lambda path, value: captured.update({path.name: value})):
        rows, cases, config = rejudge.s.verify()
        rejudge.s.analyze(rows, cases, config)
    for name, value in captured.items():
        assert json.loads(json.dumps(value)) == read(CAMPAIGN / name), name
    print('Authenticated independent judging audit', flush=True)


def percentile(values, p):
    """Nearest-rank percentile, explicitly named in the report."""
    import math
    return sorted(values)[max(0, math.ceil(len(values) * p) - 1)] if values else None


def metric_summary(calls):
    values = [c['latency_ms'] / 1000 for c in calls if c['latency_ms'] is not None]
    usage = [c['usage'] for c in calls if c['usage'] is not None]
    return {'calls': len(calls), 'statuses': dict(Counter(c['status'] for c in calls)),
            'receipts': str(sum((Decimal(c['cost']) for c in calls if c['cost'] is not None), Decimal(0))),
            'known_latency_count': len(values), 'p50_seconds': statistics.median(values) if values else None,
            'p95_seconds': percentile(values, .95), 'max_seconds': max(values) if values else None,
            'sum_seconds': sum(values), 'known_usage_count': len(usage),
            'input_tokens': sum(u.get('input_tokens', 0) or 0 for u in usage),
            'output_tokens': sum(u.get('output_tokens', 0) or 0 for u in usage),
            'max_input_tokens': max((u.get('input_tokens', 0) or 0 for u in usage), default=0)}


def engine_calls(cell, events):
    intentions = {e['payload']['physical_attempt_id']: e for e in events if e['type'] == 'model_call_intended'}
    terminals = {e['payload']['physical_attempt_id']: e for e in events if e['type'] in
                 {'model_call_succeeded', 'model_call_failed', 'model_call_outcome_unknown'}}
    result = []
    for event in events:
        if event['type'] != 'model_dispatch_started':
            continue
        identity = event['payload']['physical_attempt_id']
        intent = intentions[identity]
        p = intent['payload']
        terminal = terminals.get(identity)
        q = terminal['payload'] if terminal else {}
        status = ('complete' if q.get('finish_status') == 'complete' else q.get('finish_status')) if terminal and terminal['type'] == 'model_call_succeeded' else q.get('error_kind', 'unresolved')
        raw = q.get('raw_body')
        provider = None
        if raw and raw.get('data'):
            try:
                provider = json.loads(raw['data']).get('provider')
            except (ValueError, AttributeError):
                pass
        row = {'id': cell + '-call-' + str(intent['sequence']), 'cell': cell, 'cohort': 'engine',
               'role': p['role'], 'purpose': p['purpose'], 'model': p['requested_model'],
               'effective_model': q.get('effective_model_id'), 'provider': provider,
               'route': p['route_kind'], 'attempt_ordinal': p['attempt_ordinal'],
               'intent_sequence': intent['sequence'], 'dispatch_sequence': event['sequence'],
               'result_sequence': terminal['sequence'] if terminal else None,
               'event_hash': terminal['event_hash'] if terminal else event['event_hash'],
               'started_at': event['timestamp_utc'], 'ended_at': terminal['timestamp_utc'] if terminal else None,
               'status': status, 'cost': q.get('cost'), 'latency_ms': q.get('latency_ms'),
               'usage': q.get('usage'), 'reasoning_tokens': q.get('metadata', {}).get('reasoning_tokens'),
               'detail': q.get('detail'), 'logical_action_id': p['logical_action_id']}
        detail = {**row, 'request': json.loads(p['request_body']['data']),
                  'result': {k: v for k, v in q.items() if k != 'raw_body'},
                  'source': {'journal': cell, 'intent_hash': intent['event_hash'],
                             'dispatch_hash': event['event_hash'], 'result_hash': terminal['event_hash'] if terminal else None}}
        atomic(SITE / 'data/calls' / (row['id'] + '.json'), detail)
        result.append(row)
    return result


def build():
    plan = read(CAMPAIGN / 'plan.json')
    authenticate(plan)
    from quasilisk.judging import judge_evidence_catalog
    execution = read(CAMPAIGN / 'execution.json')
    summary = read(CAMPAIGN / 'campaign_summary.json')
    semantic = read(CAMPAIGN / 'semantic_results.json')
    panel_results = read(CAMPAIGN / 'panel_results.json')
    references = read(CAMPAIGN / 'reference_labels.json')
    adjudication = read(CAMPAIGN / 'adjudication.json')
    run_summaries = {r['cell']: r for r in summary['trials']}
    runs, observations, calls, portfolios, blocks = [], [], [], [], []
    sources = []
    for entry in plan['entries'] + plan['abandoned_entries']:
        if entry.get('not_run_reason'):
            continue
        cell = entry['cell']
        a = read(CAMPAIGN / (cell + '.audit.json'))
        journal = ROOT / a['run_directory'] / 'events.jsonl'
        events = [json.loads(line) for line in journal.read_text().splitlines()]
        sources.append({'cell': cell, 'journal': str(journal.relative_to(ROOT)), 'sha256': digest(journal),
                        'checkpoint': a['checkpoint'], 'validated_checkpoint': a['validated_checkpoint'],
                        'scope': 'dispatch prefix through 56; invalid undispatched tail preserved' if a['validated_dispatch_prefix'] else 'complete journal',
                        'event_count': len(events), 'source_snapshot': a['source_snapshot']})
        manifest = events[0]['payload']['manifest']
        cell_calls = engine_calls(cell, events)
        calls.extend(cell_calls)
        call_by_result = {c['result_sequence']: c for c in cell_calls if c['result_sequence']}
        by_obs = {o['observation_id']: f"{cell}/target-{o['target_sequence']}" for o in a['observations']}
        node_for = {n['target_observation_id']: n for n in a['nodes'].values()}
        for o in a['observations']:
            node = node_for.get(o['observation_id'])
            parent_node = a['nodes'].get(node['parent_node_id']) if node else None
            parent = by_obs.get(parent_node['target_observation_id']) if parent_node else None
            identity = by_obs[o['observation_id']]
            call = call_by_result[o['target_sequence']]
            members = []
            for m in o['members']:
                m = dict(m)
                m['call_id'] = next(c['id'] for c in cell_calls if c['result_sequence'] in
                                   [e['sequence'] for e in o['judge_results'] if e['payload']['physical_attempt_id'] == m['source_attempt_id']])
                members.append(m)
            related = []
            for pr in a['portfolio_rounds'].values():
                child = next((c for c in pr['children'] if c['observation_id'] == o['observation_id']), None)
                if child:
                    related.append({'round': pr['work_unit_id'], 'kind': 'produced_by', 'child': child})
                if node and pr['generation_parent_node_id'] == node['node_id']:
                    related.append({'round': pr['work_unit_id'], 'kind': 'selected_as_parent'})
            detail = {'id': identity, 'cell': cell, 'objective': entry['objective'], 'target': entry['target'],
                      'target_sequence': o['target_sequence'], 'observation_id': o['observation_id'],
                      'call_id': call['id'], 'parent': parent, 'root': node['root_id'] if node else None,
                      'depth': node['depth'] if node else 0,
                      'score': o['effective_score'], 'transition': o['transition'],
                      'messages': o['target_messages'] + [{'role': 'assistant', 'content': o['target_reply']}],
                      'evidence_catalog': judge_evidence_catalog(o['target_reply']),
                      'members': members, 'panel': o['panel'], 'strategy_links': related,
                      'audit': [r for r in semantic if r['name'] == identity],
                      'audit_panels': [r for r in panel_results if r['name'] == identity],
                      'reference': references.get(identity),
                      'adjudication': [r for r in adjudication['reviews'] if r['name'] == identity],
                      'source': sources[-1], 'event_hash': call['event_hash']}
            atomic(SITE / 'data/conversations' / (cell + '-target-' + str(o['target_sequence']) + '.json'), detail)
            observations.append({k: detail[k] for k in ('id', 'cell', 'objective', 'target', 'target_sequence',
                                  'score', 'transition', 'parent', 'root', 'depth', 'call_id')} |
                                {'turns': sum(m['role'] == 'assistant' for m in detail['messages']),
                                 'audited': bool(detail['audit']), 'reference_kind': (detail['reference'] or {}).get('kind'),
                                 'text': '\n'.join(m['content'] for m in detail['messages']),
                                 'preview': o['target_reply'][:220],
                                 'member_scores': [m['score'] for m in members]})
        for r in a['portfolio_rounds'].values():
            parent = by_obs.get(a['nodes'][r['generation_parent_node_id']]['target_observation_id'])
            pr = {'id': r['work_unit_id'], 'cell': cell, 'parent': parent, 'intent': r['portfolio_intent'],
                  'complete': r['complete'], 'lanes': r['lanes'], 'children': [dict(c, case=by_obs.get(c['observation_id'])) for c in r['children']],
                  'outcome_reviews': r['outcome_reviews']}
            atomic(SITE / 'data/strategies' / (r['work_unit_id'] + '.json'), pr)
            portfolios.append({'id': pr['id'], 'cell': cell, 'parent': parent, 'intent': pr['intent'],
                               'complete': pr['complete'], 'lanes': len(pr['lanes']), 'children': len(pr['children']),
                               'dispatched': sum(bool(c['dispatched_attempt_ids']) for c in pr['children']),
                               'supports': sum(c['outcome']['outcome_result'] == 'supports' for c in pr['children']),
                               'falsifies': sum(c['outcome']['outcome_result'] == 'falsifies' for c in pr['children']),
                               'cases': [c['case'] for c in pr['children'] if c['case']]})
        for b in a['verification_blocks']:
            blocks.append(dict(b, cell=cell))
        row = run_summaries[cell]
        status = ('abandoned' if row['abandoned'] else 'complete' if row['terminal_reason'] == 'completed_plan' else 'partial')
        config = manifest['normalized_config']
        runs.append({**row, 'status': status, 'run_directory': a['run_directory'], 'source_snapshot': a['source_snapshot'],
                     'source': sources[-1], 'observations': len(a['observations']),
                     'high90': sum(o['effective_score'] is not None and Decimal(o['effective_score']) >= 90 for o in a['observations']),
                     'high70': sum(o['effective_score'] is not None and Decimal(o['effective_score']) >= 70 for o in a['observations']),
                     'best_score': max((Decimal(o['effective_score']) for o in a['observations'] if o['effective_score'] is not None), default=Decimal(0)).to_eng_string(),
                     'metrics': metric_summary(cell_calls), 'role_metrics': {role: metric_summary([c for c in cell_calls if c['role'] == role]) for role in ('target', 'judge', 'selector', 'perturbator')},
                     'start': events[0]['timestamp_utc'], 'end': events[-1]['timestamp_utc'],
                     'conflicts': a['selector_contract_diagnostics']['conflicts'],
                     'config': config, 'failure_events': a['failures'], 'maximum_input_tokens': a['maximum_input_tokens']})
    audit_events = [json.loads(line) for line in (CAMPAIGN / 'events.jsonl').read_text().splitlines()]
    audit_intents = {e['call_id']: e for e in audit_events if e['type'] == 'intent'}
    cases = {c['id']: c for c in read(CAMPAIGN / 'corpus.json')}
    for e in audit_events:
        if e['type'] != 'result':
            continue
        i = audit_intents[e['call_id']]
        envelope = json.loads(e['raw_body']) if e['raw_body'] else {}
        u = envelope.get('usage')
        row = {'id': 'audit-' + e['call_id'], 'cell': 'independent_audit', 'cohort': 'audit', 'case': cases[i['case_id']]['name'],
               'role': 'judge', 'purpose': 'paired_rejudge', 'model': i['model'],
               'effective_model': envelope.get('model'), 'provider': envelope.get('provider'), 'route': 'primary', 'attempt_ordinal': 0,
               'intent_sequence': i['sequence'], 'result_sequence': e['sequence'], 'event_hash': e['hash'],
               'started_at': i['time_utc'], 'ended_at': e['time_utc'],
               'status': 'complete' if e['validation']['usable'] else e['validation']['category'],
               'cost': e['cost_usd'], 'latency_ms': e.get('latency_seconds', 0) * 1000,
               'usage': {'input_tokens': u.get('prompt_tokens'), 'output_tokens': u.get('completion_tokens'), 'total_tokens': u.get('total_tokens')} if u else None,
               'detail': None, 'logical_action_id': None}
        atomic(SITE / 'data/calls' / (row['id'] + '.json'), {**row, 'request': i['request'], 'result': envelope,
                                                         'source': {'journal': 'independent audit', 'intent_hash': i['hash'], 'result_hash': e['hash']}})
        calls.append(row)
    assert len(calls) == 373 and len(observations) == 84
    assert sum(Decimal(c['cost']) for c in calls if c['cost'] is not None) == Decimal(read(CAMPAIGN / 'combined_budget.json')['total_receipts_usd'])
    assert Counter(c['role'] for c in calls if c['cohort'] == 'engine') == summary['totals']['physical_calls']
    assert sum(o['score'] is not None and Decimal(o['score']) >= 90 for o in observations) == 17
    role_metrics = {role: metric_summary([c for c in calls if c['cohort'] == 'engine' and c['role'] == role]) for role in ('target', 'judge', 'selector', 'perturbator')}
    model_metrics = []
    for cohort, model, role in sorted({(c['cohort'], c['model'], c['role']) for c in calls}):
        model_metrics.append({'cohort': cohort, 'model': model, 'role': role,
                              **metric_summary([c for c in calls if (c['cohort'], c['model'], c['role']) == (cohort, model, role)])})
    exported = {'title': 'Quasilisk Observatory', 'date': '12 November 2026',
                'totals': summary['totals'], 'budget': read(CAMPAIGN / 'combined_budget.json'),
                'runs': runs, 'not_run': execution['not_run'], 'plan_entries': plan['entries'],
                'matched_work': plan['matched_work'], 'role_metrics': role_metrics, 'model_metrics': model_metrics,
                'observations': observations, 'calls': calls, 'portfolios': portfolios, 'verification_blocks': blocks,
                'semantic_summary': read(CAMPAIGN / 'semantic_summary.json'), 'semantic_results': semantic,
                'panel_results': panel_results, 'adjudication': adjudication, 'audit_results': read(CAMPAIGN / 'results.json'),
                'sources': sources, 'validation': read(CAMPAIGN / 'test_validation.json'),
                'audit_verification': read(CAMPAIGN / 'verification.json'),
                'audit_resume': read(CAMPAIGN / 'resume_validation.json'),
                'live_resume': read(CAMPAIGN / 'live_resume_validation.json'),
                'source_hashes': {str(p.relative_to(ROOT)): digest(p) for p in sorted(CAMPAIGN.glob('*')) if p.is_file() and p.suffix in {'.py', '.json', '.jsonl', '.txt', '.patch'}}}
    atomic(SITE / 'data/report.json', exported)
    download_names = ['report.txt', 'high_responses.txt', 'campaign_summary.json', 'execution.json', 'combined_budget.json',
                      'semantic_results.json', 'semantic_summary.json', 'adjudication.json', 'reference_labels.json',
                      'verification.json', 'plan.json', 'runtime_fix.patch', 'test_validation.json']
    for name in download_names:
        p = SITE / 'downloads' / name
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + '.tmp')
        shutil.copyfile(CAMPAIGN / name, tmp)
        tmp.replace(p)
    print(json.dumps({'conversations': len(observations), 'physical_calls': len(calls), 'trials': len(runs),
                      'data_sha256': digest(SITE / 'data/report.json')}))


if __name__ == '__main__':
    build()
