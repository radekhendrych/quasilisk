"""Focused retest adapter; reuse the canonical renderer and existing study runner."""
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
c = s.corpus_builder
b = s.b
BASE_SOURCES = s.sources


def build():
    baseline = json.loads((BASE / 'corpus.json').read_text())
    cases = []
    for old in baseline:
        if old['stratum'] == 'controlled' and not old['name'].endswith('/abstract_vs_action/abstract'):
            continue
        case = deepcopy({k: v for k, v in old.items() if k not in {'id', 'messages', 'messages_by_slot', 'response_format', 'focus_slot'}})
        case['baseline_case_id'] = old['id']
        if case['expected']['kind'] == 'negative':
            case['expected']['procedure']['reservation'] = ['none', 'material']
            case['label_provenance'] += ' Retest permits material counterevidence on an explicitly resisting reply; original reference unchanged.'
        if case['name'] == 'historical/nihilism/intent-22':
            case['expected'] = c.expected('adjudication', rationale='Pre-retest review identified a contestable independently binding norm qualifier. Original positive label is preserved in the baseline; this retest excludes it from accuracy claims.')
        cases.append(case)
    for mode in c.MODES:
        cases.extend(json.loads((ROOT / 'objectives' / (mode + '_susceptibility') / 'semantic_regressions.json').read_text())['cases'])
    for case in cases:
        case['id'] = b.digest(case)
        case['messages_by_slot'] = {slot: c.render(case, slot)[0] for slot in c.FOCI}
        case['messages'] = case['messages_by_slot']['primary']
        case['response_format'] = c.render(case, 'primary')[1]
        case['focus_slot'] = 'paired'
    assert len(cases) == 57 and len({x['id'] for x in cases}) == 57
    return cases


def sources():
    result = BASE_SOURCES()
    for p in sorted(HERE.glob('*.py')):
        result[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    result[str(BASE.relative_to(ROOT) / 'corpus.json')] = hashlib.sha256((BASE / 'corpus.json').read_bytes()).hexdigest()
    contract = ROOT / 'docs/quasilisk_redesign_specification.md'
    result[str(contract.relative_to(ROOT))] = hashlib.sha256(contract.read_bytes()).hexdigest()
    return result


@contextmanager
def configured():
    old = s.HERE, s.sources, c.build
    s.HERE, s.sources, c.build = HERE, sources, build
    try:
        with s.configured():
            yield
    finally:
        s.HERE, s.sources, c.build = old


def prepare():
    if (HERE / 'study.json').exists() or (HERE / 'events.jsonl').exists():
        raise RuntimeError('Cannot replace a frozen retest')
    cases = build()
    config = deepcopy(json.loads((BASE / 'study.json').read_text()))
    prior = Decimal(json.loads((BASE / 'verification.json').read_text())['final_charged_plus_reserved_usd'])
    config.update(budget_usd=str(Decimal('2') - prior), dispatch_budget_usd='1.55',
                  seed=20261112, confirmation_calls=2 * len(cases), physical_calls_planned=4 * len(cases),
                  source_sha256=sources(), prior_study_cost_usd=str(prior), combined_authorized_usd='2',
                  purpose='Focused semantic correction retest: historical disputes, abstract concessions, action/omission and adoption contrasts')
    assert Decimal(config['dispatch_budget_usd']) + prior < Decimal('2')
    b.atomic_json(HERE / 'corpus.json', cases)
    b.atomic_json(HERE / 'study.json', config)
    b.atomic_json(HERE / 'confirmation.json', {'profiles': list(s.PAIR), 'concurrency': 12, 'stop_after_failures': 114,
        'qualification_metrics': {p: 'usable' for p in s.PAIR}, 'reason': 'Complete the focused two-pass workload within the remaining original $2 authorization; no retries or repairs.'})
    b.atomic_json(HERE / 'label_review.json', {'frozen_before_dispatch': True, 'cases': len(cases),
        'authentic': 30, 'reused_abstract': 3, 'new_contrasts': 24,
        'reference_changes': ['Negative references permit reservation=material for genuine opposition.', 'Historical nihilism intent-22 is excluded from accuracy as predeclared adjudication.'],
        'limits': 'Assistant-authored labels, related controls and repeats; no independent human-gold or population-accuracy claim.',
        'references': [{'name': x['name'], 'expected': x['expected']} for x in cases]})
    print(json.dumps({'cases': len(cases), 'calls': 4 * len(cases), 'dispatch_ceiling_usd': '1.55', 'combined_ceiling_usd': str(prior + Decimal('1.55'))}))


def main(command):
    if command == 'prepare':
        prepare()
    else:
        with configured():
            s.main(command)


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in {'prepare', 'run', 'verify'}:
        raise SystemExit('Usage: python3 experiments/judge_semantic_retest/run.py prepare|run|verify')
    main(sys.argv[1])
