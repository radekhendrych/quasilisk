"""Read-only evidence worker, executed from each run's exact source snapshot.

The known failed verification allocation has a valid dispatch prefix followed
by one undispatched work-unit record. Authenticate that prefix for accounting;
never trim the authoritative journal or represent its invalid tail as valid.
"""
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path.cwd()
sys.path.insert(0, str(SOURCE_ROOT))
from quasilisk.engine import validate_run_evidence
from quasilisk.events import read_events


def read_state(entry):
    paths = sorted((ROOT / entry['output_directory']).glob('run-*'))
    if len(paths) != 1:
        raise RuntimeError('Expected one original journal')
    path = paths[0]
    events = read_events(path)
    if entry.get('abandoned_checkpoint'):
        assert events[-1]['event_hash'] == entry['abandoned_checkpoint']
    cutoff = entry.get('valid_prefix_sequence')
    if cutoff is None:
        return path, validate_run_evidence(path, repository_root=SOURCE_ROOT), events
    assert 0 < cutoff < len(events)
    assert events[cutoff - 1]['sequence'] == cutoff
    # No omitted dispatch, receipt, unknown outcome or monetary event.
    assert len(events[cutoff:]) == 1
    assert events[cutoff]['type'] == 'work_unit_started'
    with tempfile.TemporaryDirectory(prefix='quasilisk-read-only-prefix-') as directory:
        prefix = Path(directory)
        lines = (path / 'events.jsonl').read_bytes().splitlines(keepends=True)
        (prefix / 'events.jsonl').write_bytes(b''.join(lines[:cutoff]))
        state = validate_run_evidence(prefix, repository_root=SOURCE_ROOT)
    assert not state.dangling_attempt_ids
    return path, state, events


def record(entry):
    path, state, events = read_state(entry)
    return {'cell': entry['cell'], 'objective': entry['objective'], 'target': entry['target'],
        'run_directory': str(path.relative_to(ROOT)),
        'terminal_reason': state.terminal_reason.value if state.terminal_reason else None,
        'cost_liability_usd': str(state.cost_liability_usd), 'checkpoint': events[-1]['event_hash'],
        'validated_checkpoint': state.events[-1]['event_hash'],
        'validation_scope': 'dispatch_prefix_only' if entry.get('valid_prefix_sequence') else 'complete_journal',
        'source_snapshot': entry.get('source_snapshot', '.')}


if __name__ == '__main__':
    print(json.dumps(record(json.loads(sys.argv[1]))))
