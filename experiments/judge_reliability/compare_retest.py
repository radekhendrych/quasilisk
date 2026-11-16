"""Offline matched-case comparison, including every original pair failure."""
import json
import retest as r


def main():
    with r.baseline_snapshot() as old:
        r.check_baseline(old)
        r.b.HERE = r.DEST
        with r.b.snapshot_lock():
            new = r.b.history()
            cases = {c['id']: c for c in json.loads((r.DEST/'corpus.json').read_text())}
            old_intents = {x['call_id']: x for x in old if x['type'] == 'intent'}
            new_intents = {x['call_id']: x for x in new if x['type'] == 'intent'}
            results = {(new_intents[x['call_id']]['profile_id'], new_intents[x['call_id']]['ordinal']): x
                       for x in new if x['type'] == 'result'}
            matched = []
            for row in old:
                if row['type'] != 'result':
                    continue
                intent = old_intents[row['call_id']]
                profile = intent.get('profile_id', intent['model'])
                if intent['stage'] != 'confirm' or profile not in r.PAIR:
                    continue
                current = results.get((profile, intent['ordinal']))
                if current is None:
                    continue
                current_intent = new_intents[current['call_id']]
                assert cases[current_intent['case_id']]['source_case_id'] == intent['case_id']
                matched.append({'profile': profile, 'ordinal': intent['ordinal'],
                    'source_case_id': intent['case_id'], 'old_call_id': row['call_id'],
                    'new_call_id': current['call_id'],
                    'before': row['validation'], 'after': current['validation']})
            pairs = sorted(set(ordinal for profile, ordinal in results if profile == r.PAIR[0]) &
                           set(ordinal for profile, ordinal in results if profile == r.PAIR[1]))
            valid_pairs = sum(all(results[(profile, ordinal)]['validation']['usable'] for profile in r.PAIR)
                              for ordinal in pairs)
            failed_before = [x for x in matched if not x['before']['usable']]
            r.b.atomic_json(r.DEST/'matched_cases.json', {
                'baseline_checkpoint': old[-1]['hash'], 'retest_checkpoint': new[-1]['hash'],
                'matched_calls': matched,
                'baseline_failed_calls_retested': len(failed_before),
                'baseline_failed_calls_now_usable': sum(x['after']['usable'] for x in failed_before),
                'paired_ordinals': len(pairs), 'fully_usable_pairs': valid_pairs,
                'pair_one_sided_95_lower_bound': r.b.lower_bound(valid_pairs, len(pairs)) if pairs else None,
                'scope': 'Format and harness usability, not semantic correctness. Confidence bound '
                    'assumes independent paired trials; repeated corpus and provider correlation limit generalization.'})
            print(json.dumps({'matched_calls': len(matched),
                'old_failures_retested': len(failed_before),
                'old_failures_now_usable': sum(x['after']['usable'] for x in failed_before),
                'fully_usable_pairs': valid_pairs, 'paired_ordinals': len(pairs)}))


if __name__ == '__main__': main()
