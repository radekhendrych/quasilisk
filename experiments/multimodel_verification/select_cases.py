"""Predeclared, deterministic sample selection; no reference labels inferred."""
import json
from decimal import Decimal
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'experiments/judge_correctness'))
import build_corpus as c
from campaign import atomic


def select(observations):
    unique, seen = [], set()
    for row in observations:
        if not row['judge_intents'] or row['target_reply'] in seen:
            continue
        seen.add(row['target_reply'])
        unique.append(row)
    score = lambda row: Decimal(row['effective_score'] or '-1')
    def signature(row):
        return (row['transition'], tuple(sorted(set(tuple(member.get('procedure', {}).get(k)
                    for k in ('premise', 'inference', 'commitment', 'reservation'))
                    for member in row.get('members', [])))))
    def claimed_stages(row):
        return sum(member.get('procedure', {}).get(k) in {'adopted', 'endorsed'}
                   for member in row.get('members', [])
                   for k in ('premise', 'inference', 'commitment'))
    # Equal effective scores may conceal a wrongly adopted commitment under
    # another stage's cap. Prioritize that claim without treating it as truth.
    ranked = sorted(unique, key=lambda x: (-score(x), -claimed_stages(x), x['target_sequence']))
    chosen = ranked[:1]
    # Preserve at least one stateful context when available, instead of
    # spending both high-score slots on equivalent one-turn repetitions.
    stateful = lambda row: any(m['role'] == 'assistant' for m in row['target_messages'])
    if len(ranked) > 1:
        other_context = next((row for row in ranked[1:]
                              if stateful(row) != stateful(chosen[0])), None)
        chosen.append(other_context or ranked[1])
    def disagreement(row):
        members = row.get('members', [])
        fields = sum(len({m.get('procedure', {}).get(k) for m in members}) > 1
                     for k in ('premise', 'inference', 'commitment', 'reservation'))
        scores = [Decimal(m['score']) for m in members]
        spread = max(scores) - min(scores) if scores else Decimal(0)
        return (-fields, -spread, row['target_sequence'])
    disagreements = [row for row in unique if row not in chosen and disagreement(row)[0] < 0]
    if disagreements:
        chosen.append(min(disagreements, key=disagreement))
    abstract = [row for row in unique if row not in chosen and score(row) >= 70
                and any(m['application'] == 'abstract' for m in row.get('members', []))]
    if abstract and len(chosen) < 4:
        chosen.append(min(abstract, key=lambda row: (-score(row), row['target_sequence'])))
    signatures = {signature(row) for row in chosen}
    for row in sorted(unique, key=disagreement):
        if len(chosen) >= 4:
            break
        if row not in chosen and signature(row) not in signatures:
            chosen.append(row)
            signatures.add(signature(row))
    for row in sorted(unique, key=lambda x: (score(x), x['target_sequence'])):
        if len(chosen) >= 6:
            break
        if row not in chosen:
            chosen.append(row)
    return sorted(chosen, key=lambda x: x['target_sequence'])


def main():
    if (HERE / 'selected_cases.json').exists() or (HERE / 'study.json').exists():
        raise RuntimeError('Selection already frozen')
    plan = json.loads((HERE / 'plan.json').read_text())
    cases, selection = [], []
    for entry in plan['entries'] + plan.get('abandoned_entries', []):
        if entry.get('not_run_reason'):
            assert not list((ROOT / entry['output_directory']).glob('run-*'))
            continue
        audit = json.loads((HERE / (entry['cell'] + '.audit.json')).read_text())
        for row in select(audit['observations']):
            intent = next((e for e in row['judge_intents'] if e['payload']['requested_model'].startswith('deepseek/')), row['judge_intents'][0])
            body = json.loads(intent['payload']['request_body']['data'])
            name = entry['cell'] + '/target-' + str(row['target_sequence'])
            cases.append({'name': name, 'objective': entry['objective'], 'target_model': entry['target'],
                'stratum': 'fresh_authentic', 'source': audit['run_directory'],
                'source_event_hash': intent['event_hash'], 'source_sequence': intent['sequence'],
                'conversation': c.block(body['messages'][1]['content'], 'UNTRUSTED_CONVERSATION_JSON'),
                'target_reply': row['target_reply']})
            selection.append({'name': name, 'prior_score': row['effective_score'], 'transition': row['transition'],
                              'checkpoint': audit['checkpoint'], 'target_sequence': row['target_sequence']})
    atomic(HERE / 'candidate_cases.json', cases)
    atomic(HERE / 'candidate_selection.json', selection)
    names = plan.get('audit_case_names')
    if names is not None:
        assert len(names) == len(set(names))
        by_name = {case['name']: case for case in cases}
        assert set(names) <= set(by_name)
        cases = [by_name[name] for name in names]
        selection = [row for row in selection if row['name'] in set(names)]
    atomic(HERE / 'selected_cases.json', cases)
    atomic(HERE / 'selection.json', selection)
    print(json.dumps({'cases': len(cases), 'planned_matrix_cells': len(plan['entries'])}))


if __name__ == '__main__':
    main()
