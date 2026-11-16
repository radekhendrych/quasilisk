"""Contract retest using the existing paid harness and canonical rendering.

The baseline ledger is locked and its entire conservative liability is reserved.
Historical evidence is never rewritten. No retries or output repair are used.
"""
from contextlib import contextmanager
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

import benchmark as b
import verify
from quasilisk.domain import canonical_json, strict_json_loads
from quasilisk.engine import _render_template
from quasilisk.judging import judge_evidence_catalog, materialize_judge_schema

BASE = Path(__file__).resolve().parent
DEST = BASE / 'retest'
PAIR = ('deepseek/deepseek-v4.1-flash-1017@morph/fp8-low',
        'openai/gpt-6-luna-pro@medium')


@contextmanager
def baseline_snapshot():
    original = b.HERE
    b.HERE = BASE
    try:
        with b.snapshot_lock():
            rows = b.history()
            yield rows
    finally:
        b.HERE = original


def block(text, name):
    start, end = '<' + name + '>\n', '\n</' + name + '>'
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError('Ambiguous historical prompt block: ' + name)
    return strict_json_loads(text.split(start)[1].split(end)[0])


def transform(case, body):
    """Reuse exact objective, conversation, focus and target, with current contract."""
    assert [m['role'] for m in body['messages']] == ['system', 'user']
    historical = body['messages'][1]['content']
    schema = materialize_judge_schema(
        strict_json_loads((b.ROOT/'schemas/judge_output.json').read_text()),
        case['target_reply'])
    values = {
        'OBJECTIVE_JSON': block(historical, 'UNTRUSTED_OBJECTIVE_JSON'),
        'CONVERSATION_JSON': block(historical, 'UNTRUSTED_CONVERSATION_JSON'),
        'JUDGE_FOCUS_JSON': block(historical, 'GOVERNING_REVIEW_FOCUS_JSON'),
        'CURRENT_TARGET_EVIDENCE_JSON': judge_evidence_catalog(case['target_reply']),
        'OUTPUT_SCHEMA_JSON': schema,
        'SCHEMA_REPAIR_CONTEXT_JSON': None,
    }
    messages = [
        {'role': 'system', 'content': (b.ROOT/'templates/judge_system.md').read_text()},
        {'role': 'user', 'content': _render_template(
            (b.ROOT/'templates/judge_user.md').read_text(),
            {key: canonical_json(value) for key, value in values.items()})},
    ]
    response_format = {'type': 'json_schema', 'json_schema': {
        'name': 'quasilisk_response', 'strict': True, 'schema': schema}}
    return messages, response_format


def verify_transformation(case, body):
    messages, response_format = transform(case, body)
    assert case['messages'] == messages and case['response_format'] == response_format
    assert case['source_case_id'] == b.digest({'messages': body['messages'], 'schema': body['response_format']})
    assert case['id'] == b.digest({'messages': messages, 'schema': response_format})


def prepare():
    with baseline_snapshot() as rows:
        if (DEST/'study.json').exists():
            raise RuntimeError('Retest already frozen; refusing replacement')
        baseline_config = json.loads((BASE/'study.json').read_text())
        extension = json.loads((BASE/'extension.json').read_text())
        confirmation = json.loads((BASE/'confirmation.json').read_text())
        profiles = {p.get('id', p['model']): p for p in
                    extension['profiles'] + confirmation['new_profiles']}
        corpus = []
        for old in json.loads((BASE/'corpus.json').read_text()):
            messages, response_format = transform(old, old)
            corpus.append({**old, 'source_case_id': old['id'],
                           'id': b.digest({'messages': messages, 'schema': response_format}),
                           'messages': messages, 'response_format': response_format})
        liability = b.charged(rows)
        sources = set(baseline_config['source_sha256']) | {
            'quasilisk/engine.py', 'experiments/judge_reliability/benchmark.py',
            'experiments/judge_reliability/retest.py', 'experiments/judge_reliability/verify.py'}
        config = {
            'budget_usd': str(Decimal('5') - liability),
            'dispatch_budget_usd': str(Decimal('4.5') - liability),
            'baseline_checkpoint': rows[-1]['hash'],
            'baseline_liability_usd': str(liability),
            'baseline_receipt_usd': str(sum((Decimal(r['cost_usd']) for r in rows
                if r['type'] == 'result' and r['cost_usd'] is not None), Decimal(0))),
            'concurrency': 12, 'deadline_seconds': baseline_config['deadline_seconds'],
            'seed': baseline_config['seed'], 'screen_calls': 0, 'confirmation_calls': 600,
            'profiles': [profiles[key] for key in PAIR],
            'source_sha256': {name: hashlib.sha256((b.ROOT/name).read_bytes()).hexdigest()
                              for name in sorted(sources)},
        }
        DEST.mkdir(exist_ok=True)
        b.atomic_json(DEST/'corpus.json', corpus)
        b.atomic_json(DEST/'confirmation.json', {
            'profiles': list(PAIR), 'concurrency': 12, 'stop_after_failures': 4,
            'qualification_metrics': {key: 'usable' for key in PAIR},
            'reason': 'Same 600-trial plan and futility rule; current canonical contract; no repairs.'})
        b.atomic_json(DEST/'study.json', config)
        print(json.dumps({'cases': len(corpus), 'prior_liability_usd': str(liability),
                          'retest_dispatch_allowance_usd': config['dispatch_budget_usd']}))


def check_baseline(rows):
    config = json.loads((DEST/'study.json').read_text())
    assert rows[-1]['hash'] == config['baseline_checkpoint'], 'Baseline changed'
    assert b.charged(rows) == Decimal(config['baseline_liability_usd'])
    assert b.charged(rows) + Decimal(config['dispatch_budget_usd']) <= Decimal('4.5')
    assert b.charged(rows) + Decimal(config['budget_usd']) <= Decimal('5')
    return config


def comparison_report(config, baseline_rows):
    b.report()
    new = json.loads((DEST/'results.json').read_text())
    old = json.loads((BASE/'results.json').read_text())
    rows = b.history()
    intents = {r['call_id']: r for r in rows if r['type'] == 'intent'}
    failures = [{'profile': intents[r['call_id']]['profile_id'],
                 'ordinal': intents[r['call_id']]['ordinal'],
                 'case_id': intents[r['call_id']]['case_id'],
                 'validation': r['validation']}
                for r in rows if r['type'] == 'result' and not r['validation']['usable']]
    data = {
        'baseline_checkpoint': baseline_rows[-1]['hash'],
        'source_checkpoint': rows[-1]['hash'],
        'combined_receipt_usd': str(Decimal(config['baseline_receipt_usd']) + Decimal(new['receipt_cost_usd'])),
        'combined_conservative_liability_usd': str(Decimal(config['baseline_liability_usd']) + Decimal(new['charged_plus_unresolved_reservations_usd'])),
        'profiles': {key: {'before': old['profiles']['confirm:'+key],
                           'after': new['profiles'].get('confirm:'+key)} for key in PAIR},
        'failures': failures,
        'operator_stop': next((row for row in rows if row['type']=='operator_stopped'), None),
        'interpretation': 'Structural usability only; semantic judgment accuracy is not measured. '
            'Confidence bounds assume independent calls; 137 historical cases are repeated. '
            'Model settings and routes are matched, but this is a temporal retest, not a randomized concurrent A/B.'}
    b.atomic_json(DEST/'comparison.json', data)
    lines = ['Preferred-pair canonical contract retest', '']
    if data['operator_stop']:
        lines.extend(['Stopped at the user\'s request and accepted as successful.',
            f"{len(data['operator_stop']['unresolved_call_ids'])} interrupted dispatches have unknown outcomes and retain full reservations.",
            'Rates below cover returned results only; the planned 600-call confidence target was not reached.', ''])
    for key, values in data['profiles'].items():
        before, after = values['before'], values['after']
        lines.extend([key, f"  Before: {before['valid']}/{before['calls']} fully usable.",
            f"  After: {after['valid']}/{after['calls']} fully usable; native schema {after['native_schema']}/{after['calls']}.",
            f"  One-sided 95% lower bound: {after['full_contract_one_sided_95_lower_bound']:.6%}.",
            f"  Retest receipts: ${after['cost_usd']}; categories: {after['categories']}", ''])
    lines.extend([f"Combined reported charges: ${data['combined_receipt_usd']}",
                  f"Combined charges plus retained unpriced reservations: ${data['combined_conservative_liability_usd']}",
                  data['interpretation'], '', 'See comparison.json for every failure; events.jsonl contains exact requests and results.'])
    (DEST/'report.txt.tmp').write_text('\n'.join(lines)+'\n')
    (DEST/'report.txt.tmp').replace(DEST/'report.txt')


def main(command):
    if command == 'prepare':
        prepare()
        return
    # Hold the baseline shared lock throughout: its live writer cannot spend.
    with baseline_snapshot() as baseline_rows:
        config = check_baseline(baseline_rows)
        b.HERE = DEST
        if command == 'run':
            b.run('confirm')
        else:
            with b.snapshot_lock():
                verify.main(source_request_verifier=verify_transformation)
                comparison_report(config, baseline_rows)


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ('prepare', 'run', 'verify'):
        raise SystemExit('Usage: python3 experiments/judge_reliability/retest.py prepare|run|verify')
    main(sys.argv[1])
