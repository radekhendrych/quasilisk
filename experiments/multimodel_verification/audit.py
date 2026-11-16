"""Read-only campaign evidence export through the canonical validator/reducer."""
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from decimal import Decimal
from enum import Enum
import json
from pathlib import Path
import sys
import statistics
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from campaign import atomic, existing
sys.path.insert(0, str(Path.cwd()))
from evidence import read_state
from quasilisk.reporting import build_report_bundle
from quasilisk.run_state import portfolio_transition
from quasilisk.domain import strict_json_loads, validate_json_schema_instance, DomainError
from quasilisk.strategy import _allowed_argument_moves
from quasilisk.evidence import evaluate_verification


def serial(value):
    if is_dataclass(value):
        return {f.name: serial(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    return value


def audit(entry):
    path, state, full_events = read_state(entry)
    events = state.events
    summary = serial(build_report_bundle(state).summary)
    cached_summary = path / 'summary.json'
    if cached_summary.exists():
        assert json.loads(cached_summary.read_text()) == summary, 'Cached report differs from validated journal'
    failures, recoveries, observations, move_conflicts = [], [], [], []
    portfolio_complete_outputs = 0
    roles, purposes, finishes = Counter(), Counter(), Counter()
    latencies, role_costs, max_inputs = defaultdict(list), defaultdict(Decimal), Counter()
    receipt_cost = Decimal(0)
    for attempt in state.attempts.values():
        p = attempt.intended
        if not attempt.dispatched:
            continue
        roles[p['role']] += 1
        purposes[p['purpose']] += 1
        record = {'intent_sequence': attempt.intent_event['sequence'], 'role': p['role'],
                  'purpose': p['purpose'], 'model': p['requested_model']}
        if p['attempt_ordinal']:
            errors = []
            for message in p['messages']:
                start = '<SCHEMA_REPAIR_CONTEXT_JSON>\n'
                end = '\n</SCHEMA_REPAIR_CONTEXT_JSON>'
                if start in message['content']:
                    repair = json.loads(message['content'].split(start, 1)[1].split(end, 1)[0])
                    if repair is not None:
                        errors = repair['structural_errors']
            recoveries.append({**record, 'route_kind': p['route_kind'],
                               'attempt_ordinal': p['attempt_ordinal'], 'recorded_errors': errors})
        if not attempt.succeeded:
            failures.append({**record, 'terminal': serial(attempt.terminal_event)})
            continue
        result = attempt.terminal_event['payload']
        if result['cost'] is not None:
            receipt_cost += Decimal(result['cost'])
            role_costs[p['role']] += Decimal(result['cost'])
        latencies[p['role']].append(result['latency_ms'])
        if result['usage'] is not None:
            max_inputs[p['role']] = max(max_inputs[p['role']], result['usage'].get('input_tokens') or 0)
        finishes[p['role'] + ':' + result['finish_status']] += 1
        if result['finish_status'] != 'complete':
            failures.append({**record, 'incomplete': result['finish_status']})
        elif p['response_mode'] == 'json_schema':
            schema = json.loads(p['request_body']['data'])['response_format']['json_schema']['schema']
            try:
                raw = strict_json_loads(result['text'])
                validate_json_schema_instance(raw, schema)
            except DomainError as exc:
                failures.append({**record, 'native_schema_error': str(exc)})
            else:
                if p['purpose'] == 'selector_portfolio':
                    portfolio_complete_outputs += 1
                    conflicts = []
                    for index, item in enumerate(raw['items']):
                        move = item['contract']['move']
                        allowed = _allowed_argument_moves(move['act'])
                        actual = move['descriptors']['argument_move']
                        if actual not in allowed:
                            conflicts.append({'item': index, 'act': move['act'],
                                'argument_move': actual, 'allowed_by_canonical_contract': sorted(allowed)})
                    if conflicts:
                        move_conflicts.append({**record, 'result_sequence': attempt.terminal_event['sequence'],
                                               'conflicts': conflicts})
    for panel in state.panel_views.values():
        target = state.target_results.get(panel.target_action_id)
        if target is None:
            continue
        attempt = state.attempts[target.physical_attempt_id]
        members = [m for m in state.member_assessments.values() if m.observation_id == panel.observation_id]
        judge_attempts = [a for a in state.attempts.values()
                          if a.physical_attempt_id in {m.source_attempt_id for m in members}]
        observations.append({'target_sequence': target.terminal_sequence,
            'target_action_id': panel.target_action_id, 'observation_id': panel.observation_id,
            'target_reply': target.text, 'target_messages': attempt.intended['messages'],
            'effective_score': serial(panel.effective_score),
            'transition': portfolio_transition(panel, state.member_assessments),
            'panel': serial(panel), 'members': serial(members),
            'judge_intents': [serial(a.intent_event) for a in judge_attempts],
            'judge_results': [serial(a.terminal_event) for a in judge_attempts]})
    verification_blocks = []
    for unit in state.work_units:
        if unit.unit['kind'] != 'verification':
            continue
        design = unit.unit['design']
        executions = [state.execution_evidence[root['execution_id']] for root in design['root_plans']]
        roots = []
        for execution in executions:
            panels = [p for p in state.panel_views.values() if p.execution_id == execution.execution_id]
            roots.append({'execution': serial(execution),
                'effective_score': str(execution.effective_score) if execution.effective_score is not None else None,
                'target_sequences': sorted(state.target_results[p.target_action_id].terminal_sequence
                    for p in panels if p.target_action_id in state.target_results),
                'binding_and_outcome_results': [
                    {'result_sequence': a.terminal_event['sequence'], 'purpose': a.intended['purpose'],
                     'model': a.intended['requested_model'], 'text': a.terminal_event['payload']['text']}
                    for a in state.attempts.values() if a.succeeded
                    and a.intended['lineage'].get('execution_id') == execution.execution_id
                    and a.intended['purpose'] in {'template_binding', 'outcome_review'}]})
        verification_blocks.append({'unit_id': unit.unit_id, 'subject_kind': design['subject_kind'],
            'decision': serial(evaluate_verification(executions, planned_count=design['planned_count'])),
            'roots': roots})
    result = {'cell': entry['cell'], 'target': entry['target'], 'objective': entry['objective'],
        'run_directory': str(path.relative_to(ROOT)), 'checkpoint': full_events[-1]['event_hash'],
        'validated_checkpoint': events[-1]['event_hash'],
        'canonical_replay_verified': not bool(entry.get('valid_prefix_sequence')),
        'validated_dispatch_prefix': entry.get('valid_prefix_sequence'),
        'source_snapshot': entry.get('source_snapshot', '.'),
        'terminal_reason': serial(state.terminal_reason),
        'physical_calls': dict(roles), 'calls_by_purpose': dict(purposes),
        'finish_statuses': dict(finishes), 'receipt_cost_usd': str(receipt_cost),
        'role_costs_usd': serial(role_costs), 'maximum_input_tokens': dict(max_inputs),
        'latency_ms': {role: {'median': statistics.median(values), 'maximum': max(values)}
                       for role, values in latencies.items()},
        'liability_usd': str(state.cost_liability_usd), 'failures': failures, 'recoveries': recoveries,
        'selector_contract_diagnostics': {'schema_valid_complete_portfolio_outputs': portfolio_complete_outputs,
            'move_descriptor_conflict_outputs': len(move_conflicts), 'conflicts': move_conflicts,
            'scope': 'Exact consistency check using the canonical allowed-move function, including exhausted last attempts; not a census of all possible contextual parser failures.'},
        'observations': sorted(observations, key=lambda x: x['target_sequence']),
        'portfolio_rounds': serial(state.portfolio_rounds), 'strategy_episodes': serial(state.strategy_episodes),
        'execution_evidence': serial(state.execution_evidence), 'scenario_evidence': serial(state.scenario_evidence),
        'seed_qualifications': serial(state.seed_qualifications),
        'verification_blocks': verification_blocks,
        'nodes': serial(state.conversation_nodes),
        'summary': summary, 'cached_report_verified': cached_summary.exists(),
        'identity': {key: serial(getattr(state, key)) for key in (
            'target_identity', 'target_identity_drift', 'target_identity_unverifiable',
            'target_identity_mismatch_attempt_ids', 'role_target_identity_collision_attempt_ids',
            'role_identity_unverifiable_attempt_ids')}}
    atomic(HERE / (entry['cell'] + '.audit.json'), result)
    return result


if __name__ == '__main__':
    plan = json.loads((HERE / 'plan.json').read_text())
    for entry in plan['entries'] + plan.get('abandoned_entries', []):
        if len(sys.argv) == 2 and entry['cell'] != sys.argv[1]:
            continue
        path = existing(entry)
        if path is None:
            continue
        last = json.loads((path / 'events.jsonl').read_text().splitlines()[-1])
        if last['type'] != 'run_terminated' and 'abandoned_checkpoint' not in entry:
            continue
        source = (ROOT / entry.get('source_snapshot', '.')).resolve()
        if source != Path.cwd():
            subprocess.run([sys.executable, str(Path(__file__).resolve()), entry['cell']], cwd=source, check=True)
            continue
        result = audit(entry)
        print(json.dumps({k: result[k] for k in ('cell', 'terminal_reason', 'physical_calls',
            'liability_usd')} | {'failures': len(result['failures']), 'recoveries': len(result['recoveries'])}), flush=True)
