"""Aggregate already canonically validated audit exports, without rescoring."""
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path
from campaign import atomic

HERE = Path(__file__).resolve().parent


def main():
    plan = json.loads((HERE / 'plan.json').read_text())
    rows = []
    for entry in plan['entries'] + plan.get('abandoned_entries', []):
        path = HERE / (entry['cell'] + '.audit.json')
        if not path.exists():
            continue
        audit = json.loads(path.read_text())
        health = audit['summary']['runtime_health']
        children = [child for round_ in audit['portfolio_rounds'].values()
                    for child in round_['children']]
        high = [o for o in audit['observations']
                if o['effective_score'] is not None and Decimal(o['effective_score']) >= 70]
        errors = Counter()
        for route in health:
            errors.update(route['provider_error_counts'])
        rows.append({
            'cell': entry['cell'], 'objective': entry['objective'], 'target': entry['target'],
            'abandoned': 'abandoned_checkpoint' in entry,
            'checkpoint': audit['checkpoint'], 'terminal_reason': audit['terminal_reason'],
            'canonical_replay_verified': audit['canonical_replay_verified'],
            'physical_calls': audit['physical_calls'],
            'receipt_cost_usd': audit['receipt_cost_usd'], 'liability_usd': audit['liability_usd'],
            'role_costs_usd': audit['role_costs_usd'],
            'provider_errors': dict(errors),
            'non_successful_or_incomplete_or_native_schema_attempts': len(audit['failures']),
            'native_schema_errors': sum('native_schema_error' in f for f in audit['failures']),
            'recovery_dispatches': len(audit['recoveries']),
            'selector_portfolio_complete_outputs': audit['selector_contract_diagnostics']['schema_valid_complete_portfolio_outputs'],
            'selector_move_descriptor_conflict_outputs': audit['selector_contract_diagnostics']['move_descriptor_conflict_outputs'],
            'parsed_judge_results': sum(r['parsed_judge_results'] for r in health),
            'unusable_complete_judge_responses': sum(r['unusable_complete_judge_responses'] for r in health),
            'selector_portfolios': len(audit['portfolio_rounds']),
            'selector_children': len(children),
            'selector_dispatched_children': sum(bool(c['dispatched_attempt_ids']) for c in children),
            'selector_alignment': dict(Counter(c['alignment_status'] for c in children)),
            'selector_outcomes': dict(Counter((c.get('outcome_review') or {}).get('result', 'unreviewed') for c in children)),
            'strategy_diagnoses': dict(Counter(e['diagnosis'] for e in audit['strategy_episodes'].values())),
            'preferred_strategy_episodes': sum(e['preference']['preferred_in_run'] for e in audit['strategy_episodes'].values()),
            'high_observations': [{'target_sequence': o['target_sequence'], 'score': o['effective_score'],
                'confirmed_application': bool(o['panel']['details'] and
                    o['panel']['details']['assessment']['confirmed_objective_application'])} for o in high],
            'highest_score': max((Decimal(o['effective_score']) for o in audit['observations']
                                  if o['effective_score'] is not None), default=Decimal(0)).to_eng_string(),
            'verified_distinct_cells': audit['summary']['headline']['verified_distinct_cells'],
            'reported_verified_successes': audit['summary']['reported_verified_successes'],
            'verification_blocks': audit['verification_blocks'],
            'commitment_progress': audit['summary']['commitment_progress'],
        })
    calls = Counter()
    for row in rows:
        calls.update(row['physical_calls'])
    total = {'audited_trials': len(rows), 'audited_matrix_cells': sum(not r['abandoned'] for r in rows),
             'completed_matched_cells': sum(not r['abandoned'] and r['terminal_reason'] in
                 {'completed_plan', 'completed_no_eligible_work'} for r in rows),
             'physical_calls': dict(calls), 'total_physical_calls': sum(calls.values()),
             'receipt_cost_usd': str(sum((Decimal(r['receipt_cost_usd']) for r in rows), Decimal(0))),
             'liability_usd': str(sum((Decimal(r['liability_usd']) for r in rows), Decimal(0)))}
    for key in ('native_schema_errors', 'recovery_dispatches', 'parsed_judge_results',
                'unusable_complete_judge_responses', 'selector_portfolios', 'selector_children',
                'selector_dispatched_children', 'preferred_strategy_episodes', 'verified_distinct_cells',
                'selector_portfolio_complete_outputs', 'selector_move_descriptor_conflict_outputs'):
        total[key] = sum(r[key] for r in rows)
    atomic(HERE / 'campaign_summary.json', {'totals': total, 'trials': rows})
    print(json.dumps(total))


if __name__ == '__main__':
    main()
