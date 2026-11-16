"""Paired audit adapter using the existing study runner and canonical contracts."""
from contextlib import contextmanager
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / 'experiments/judge_correctness'
sys.path.insert(0, str(BASE))
import study as s
c, b = s.corpus_builder, s.b
BASE_SOURCES = s.sources
BASE_AGGREGATE = s.aggregate_panel_details


def build():
    cases = json.loads((HERE / 'selected_cases.json').read_text())
    labels = json.loads((HERE / 'reference_labels.json').read_text())
    assert set(labels) == {x['name'] for x in cases}
    for case in cases:
        case['expected'] = labels[case['name']]
        case['label_provenance'] = 'Assistant reference review before independent audit dispatch; disputed cases excluded.'
        case['id'] = b.digest(case)
        case['messages_by_slot'] = {slot: c.render(case, slot)[0] for slot in c.FOCI}
        case['messages'] = case['messages_by_slot']['primary']
        case['response_format'] = c.render(case, 'primary')[1]
        case['focus_slot'] = 'paired'
    return cases


def sources():
    result = BASE_SOURCES()
    paths = list(HERE.glob('*.py')) + [HERE / name for name in (
        'selected_cases.json', 'reference_labels.json', 'plan.json', 'execution.json', 'campaign_summary.json')]
    for path in paths:
        result[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


@contextmanager
def configured():
    old = s.HERE, s.sources, c.build, s.aggregate_panel_details
    target_by_observation = {'ob-' + case['id']: case['target_model'] for case in build()}
    def aggregate(panel_id, observation_id, members, **kwargs):
        # The legacy study's single-target diagnostic default must not leak
        # into this multi-target audit. All aggregation remains canonical.
        kwargs['target_effective_model_id'] = target_by_observation[observation_id]
        return BASE_AGGREGATE(panel_id, observation_id, members, **kwargs)
    s.HERE, s.sources, c.build, s.aggregate_panel_details = HERE, sources, build, aggregate
    try:
        with s.configured():
            yield
    finally:
        s.HERE, s.sources, c.build, s.aggregate_panel_details = old


def prepare():
    if (HERE / 'study.json').exists() or (HERE / 'events.jsonl').exists():
        raise RuntimeError('Cannot replace a frozen audit')
    execution = json.loads((HERE / 'execution.json').read_text())
    assert execution['campaign_finished']
    assert {e['cell'] for e in execution['cells'] + execution['partial_cells']} | {e['cell'] for e in execution['not_run']} == {
        e['cell'] for e in json.loads((HERE / 'plan.json').read_text())['entries']}
    plan = json.loads((HERE / 'plan.json').read_text())
    liability = Decimal(execution['engine_liability_usd'])
    # Once engine work is complete its unused allocation can fund the audit.
    # All unknown/unpriced engine dispatches remain in the frozen liability.
    ceiling = min(Decimal('2'), Decimal(plan['authorized_usd'])
                  - Decimal(plan['unallocated_margin_usd']) - liability)
    assert ceiling >= Decimal(plan['audit_reserved_usd'])
    assert liability + ceiling + Decimal(plan['unallocated_margin_usd']) <= Decimal(plan['authorized_usd'])
    cases = build()
    assert 0 < len(cases) <= 6 * (len(plan['entries']) + len(plan.get('abandoned_entries', [])))
    s.source_audit(cases)
    config = deepcopy(json.loads((BASE / 'study.json').read_text()))
    config.update(budget_usd=str(ceiling), dispatch_budget_usd=str(ceiling),
        seed=20261113, confirmation_calls=2 * len(cases), physical_calls_planned=4 * len(cases),
        deadline_seconds=plan['audit_deadline_seconds'],
        source_sha256=sources(), engine_liability_usd=str(liability), combined_authorized_usd=plan['authorized_usd'],
        purpose='Independent multi-model semantic and schema verification; authentic fresh observations only')
    b.atomic_json(HERE / 'corpus.json', cases)
    b.atomic_json(HERE / 'study.json', config)
    b.atomic_json(HERE / 'confirmation.json', {'profiles': list(s.PAIR), 'concurrency': 3,
        'stop_after_failures': 2 * len(cases), 'qualification_metrics': {p: 'usable' for p in s.PAIR},
        'reason': 'Two complete independent passes without retries or repair; strict shared budget admission.'})
    print(json.dumps({'cases': len(cases), 'calls': 4 * len(cases), 'audit_ceiling_usd': str(ceiling)}))


def verify_combined():
    config = json.loads((HERE / 'study.json').read_text())
    execution = json.loads((HERE / 'execution.json').read_text())
    audit = json.loads((HERE / 'verification.json').read_text())
    results = json.loads((HERE / 'semantic_results.json').read_text())
    engine = json.loads((HERE / 'campaign_summary.json').read_text())['totals']
    assert config['engine_liability_usd'] == execution['engine_liability_usd'] == engine['liability_usd']
    usable = [r for r in results if r['structurally_usable']]
    mismatches = [r['call_id'] for r in usable if r['effective_model'] != r['model']]
    assert not mismatches, 'Returned judge identity differs from requested judge'
    final = Decimal(engine['liability_usd']) + Decimal(audit['final_charged_plus_reserved_usd'])
    peak = Decimal(engine['liability_usd']) + Decimal(audit['maximum_charged_plus_reserved_usd'])
    assert final <= peak <= Decimal(config['combined_authorized_usd'])
    receipts = sum((Decimal(r['cost_usd']) for r in b.history()
                    if r['type'] == 'result' and r['cost_usd'] is not None), Decimal(0))
    result = {'authorized_usd': config['combined_authorized_usd'],
        'engine_receipts_usd': engine['receipt_cost_usd'], 'audit_receipts_usd': str(receipts),
        'total_receipts_usd': str(Decimal(engine['receipt_cost_usd']) + receipts),
        'engine_liability_usd': engine['liability_usd'],
        'audit_liability_usd': audit['final_charged_plus_reserved_usd'],
        'combined_final_liability_usd': str(final), 'combined_peak_liability_usd': str(peak),
        'structurally_usable_identity_checks': len(usable), 'judge_identity_mismatches': mismatches,
        'complete_audit': audit['complete_planned_workload'],
        'completed_matrix_cells': len(execution['cells']), 'partial_cells': execution['partial_cells'],
        'not_run': execution['not_run']}
    b.atomic_json(HERE / 'combined_budget.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in {'prepare', 'run', 'verify'}:
        raise SystemExit('Usage: python3 experiments/multimodel_verification/rejudge.py prepare|run|verify')
    if sys.argv[1] == 'prepare':
        prepare()
    else:
        with configured():
            s.main(sys.argv[1])
            if sys.argv[1] == 'verify':
                verify_combined()
