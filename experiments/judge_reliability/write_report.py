"""Render a human-readable study report from the verified journal."""
from collections import Counter
from decimal import Decimal
import json
import benchmark as b


def main():
    b.report()
    result=json.loads((b.HERE/'results.json').read_text())
    verification=json.loads((b.HERE/'verification.json').read_text())
    assert verification['source_checkpoint']==result['source_checkpoint']
    rows=b.history();intents={r['call_id']:r for r in rows if r['type']=='intent'}
    config=rows[0]['config'];profiles={p.get('id',p['model']):p for p in config['profiles']}
    for r in rows:
        if r['type']=='extension_selected':profiles.update({p.get('id',p['model']):p for p in r['extension']['profiles']})
        if r['type']=='confirmation_selected':profiles.update({p.get('id',p['model']):p for p in r['selection'].get('new_profiles',[])})
    contract_counts=Counter();contract_cases=[]
    for r in rows:
        if r['type']!='result':continue
        i=intents[r['call_id']]
        if i['stage']!='confirm' or i['model']=='inclusionai/ling-3.0-tiny' or r['validation']['usable']:continue
        if r['raw_body'] is None:continue
        envelope=json.loads(r['raw_body'])
        if not envelope.get('choices'):continue
        output=json.loads(envelope['choices'][0]['message']['content'])
        findings=[]
        for error in r['validation']['errors']:
            if not error.endswith(' requires matching polarized evidence'):continue
            name=error.split('.')[1].split()[0];status=output['procedure'][name]
            polarity='supports' if status in ('adopted','endorsed') else 'resists'
            present=any(e['polarity']==polarity for e in output['evidence'])
            category='unstated_exact_ID_overlap' if present else 'missing_explicit_supports' if polarity=='supports' else 'unstated_resists_presence'
            contract_counts[category]+=1;findings.append({'field':name,'category':category,'required_polarity':polarity})
        if findings:contract_cases.append({'call_id':r['call_id'],'profile_id':i['profile_id'],'case_id':i['case_id'],'evidence':output['evidence'],'procedure':output['procedure'],'errors':r['validation']['errors'],'findings':findings})
    b.atomic_json(b.HERE/'contract_diagnostics.json',{'findings':dict(contract_counts),'failed_outputs':len(contract_cases),'cases':contract_cases,'scope':'Mechanical cross-field failures in full-contract confirmation candidates; no semantic correctness assessment and no reclassification as usable.'})
    full=[];native=[]
    for key,g in result['profiles'].items():
        if not key.startswith('confirm:') or g['calls']<600:continue
        if g['full_contract_one_sided_95_lower_bound']>=.995:full.append(key)
        if g['native_schema_one_sided_95_lower_bound']>=.995:native.append(key)
    lines=['Judge model reliability study — final results','',
           'FULL CONTRACT: '+(', '.join(full) if full else 'No tested profile established at least 99.5% full-contract reliability at the predeclared one-sided 95% confidence threshold.'),
           'NATIVE JSON SCHEMA: '+(', '.join(native) if native else 'No profile established the threshold.'),'',
           'These are separate outcomes. JSON-Schema validity alone does not make an output usable by Quasilisk. The full check also requires exact target evidence IDs and consistent procedure/evidence/polarity relationships. No semantic correctness was scored.', '',
           'Budget and accounting',
           'Provider usage.cost receipts: $'+result['receipt_cost_usd'],
           'Receipts plus full reservations retained for unpriced calls: $'+result['charged_plus_unresolved_reservations_usd'],
           'Peak charged-plus-reserved admission ledger: $'+verification['maximum_charged_plus_reserved_usd'],
           'Dispatch ledger ceiling: $4.50; user authorization: $5.00. No automatic retries or repairs. Unknown charges were never assumed free.', '',
           'Measured profiles (valid counts include transport/provider failures in the denominator)',
           'Phase / profile | attempts | JSON Schema | full contract | receipts USD | unpriced']
    for key,g in result['profiles'].items():
        lines.append(f"{key} | {g['calls']} | {g['native_schema']}/{g['calls']} | {g['valid']}/{g['calls']} | {g['cost_usd']} | {g['unpriced']}")
    lines+=['','Confirmation detail']
    for key,g in result['profiles'].items():
        if not key.startswith('confirm:'):continue
        profile=profiles[key.removeprefix('confirm:')];uncached=[];reasons=Counter()
        for r in rows:
            if r['type']!='result':continue
            i=intents[r['call_id']]
            if i['stage']+':'+i.get('profile_id',i['model'])!=key:continue
            reasons.update(r['validation']['errors'])
            try:
                usage=json.loads(r['raw_body'])['usage']
                uncached.append(Decimal(usage['prompt_tokens'])*Decimal(profile['prompt_price'])+Decimal(usage['completion_tokens'])*Decimal(profile['completion_price']))
            except (KeyError,TypeError,ValueError):pass
        lines += ['',key,
                  f"Distinct exact requests: {g['distinct_cases']}; providers: {dict(g['providers'])}",
                  f"Schema pass rate: {100*g['native_schema']/g['calls']:.6f}%; one-sided 95% lower bound: {100*g['native_schema_one_sided_95_lower_bound']:.6f}%",
                  f"Full-contract pass rate: {100*g['valid']/g['calls']:.6f}%; one-sided 95% lower bound: {100*g['full_contract_one_sided_95_lower_bound']:.6f}%",
                  'Failure categories: '+str(g['categories']),
                  'Most common exact validation errors: '+str(reasons.most_common(6)),
                  'Actual receipts per 1,000 attempts: $'+str(Decimal(g['cost_usd'])/g['calls']*1000),
                  'Mean token-based uncached cost per 1,000 priced token records: $'+(str(sum(uncached)/len(uncached)*1000) if uncached else 'unknown'),
                  'Mean latency seconds: '+str(g['mean_latency_seconds'])]
    lines += ['', 'Interpretation and limits',
              'Contract gap: the prompt requires supports evidence for endorsed inference/adopted commitment, but does not fully state the exact-ID intersection rules enforced in judging.py. The JSON Schema does not encode these intersections. This is a contract alignment problem in addition to model errors; strict generation cannot guarantee a constraint absent from the supplied schema.',
              'In the full-contract confirmation candidates, '+str(len(contract_cases))+' failed outputs contain '+str(sum(contract_counts.values()))+' polarized-evidence error findings: '+str(dict(contract_counts))+'. Findings can overlap within one output and are not additional failed calls. A missing_explicit_supports finding violates an instruction already present in the prompt; the omitted overlap rule does not excuse it. Exact examples are in contract_diagnostics.json.',
              'A candidate was stopped after four failures of its chosen confirmation metric: even perfect remaining calls could not meet 99.5% observed accuracy in the fixed 600-call plan. Already outstanding calls still completed and count. Ling was explicitly tested for native JSON Schema; other confirmation candidates were tested for full usability. This does not excuse Ling evidence-link errors.',
              'Confidence bounds assume independent calls from the tested workload. The corpus contains 137 historical requests across three objectives and seven available objective/review-focus strata, repeated in confirmation. It is not 600 novel conversations; the result does not certify future models, providers, prompts or semantic judgment accuracy. Bounds on early-stopped exploratory cohorts are descriptive and are not qualification claims. Confidence levels are per profile, not a simultaneous family-wide guarantee.',
              'Candidate selection is exploratory and not exhaustive. Free/quota-limited models, batch APIs and rolling aliases were excluded. The cheapest tested successful profile is not a proof of the globally cheapest model.',
              'Provider capability matters: Qwen 3.7 Flash advertised response_format but its captured endpoints did not advertise structured_outputs; strict parameter routing rejected it. The lowest advertised GLM price belonged to a route without structured_outputs. A supported GLM endpoint completed JSON successfully but continued to violate evidence relationships. Native-schema success must not be substituted for canonical judge usability.',
              'No production prompts, schemas, validators, engine configuration or judge roster were changed. All study outputs remain raw; no prose scraping, repairs or semantic fallback made invalid responses count as valid.', '',
              'Verification',
              '434 repository unit tests passed; all three serial dry runs completed. The isolated benchmark tests check conservative charging, malformed receipts, tamper detection, truncated/provider-error rejection, sampling, confidence math and pre-dispatch budget refusal.',
              'verify.py replays the hash chain, every exact request, every canonical validation result, every receipt, uniqueness and every cost admission; verification.json records the final checkpoint.',
              'Reproduce: python3 experiments/judge_reliability/verify.py',
              'Rebuild this report: python3 experiments/judge_reliability/write_report.py']
    path=b.HERE/'report.txt';temp=path.with_suffix('.txt.tmp');temp.write_text('\n'.join(lines)+'\n');temp.replace(path)


if __name__=='__main__':
    with b.snapshot_lock():main()
