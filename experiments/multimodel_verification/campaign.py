"""Serial launch/audit of canonical CLI runs under one aggregate authorization.

This is an experiment launcher, not a Quasilisk controller. Every run, recovery,
work allocation, role dispatch and reducer remains owned by the canonical engine.
"""
from decimal import Decimal
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import signal

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))


def atomic(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(path)


def existing(entry):
    directory=ROOT/entry['output_directory']
    runs=sorted(directory.glob('run-*')) if directory.exists() else []
    if len(runs)>1:raise RuntimeError('Ambiguous campaign cell; review its journals before continuing: '+entry['cell'])
    return runs[0] if runs else None


def remaining_cap(pool,prior_liabilities):
    result=Decimal(pool)-sum((Decimal(x) for x in prior_liabilities),Decimal(0))
    if result<=0:raise RuntimeError('Campaign engine pool exhausted; no dispatch authorized')
    return result


def validated_record(entry):
    source=ROOT/entry.get('source_snapshot','.')
    result=subprocess.run([sys.executable,str(HERE/'evidence.py'),json.dumps(entry)],
                          cwd=source,check=True,capture_output=True,text=True)
    return json.loads(result.stdout)


def run_cli(config_path, log):
    child=subprocess.Popen([sys.executable,'-m','quasilisk','--config',str(config_path)],
                           cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    try:
        return child.wait(),False
    except KeyboardInterrupt:
        # Let the canonical engine journal its current receipt and stop. The
        # subprocess.run exception path would kill it while handling SIGINT.
        child.send_signal(signal.SIGINT)
        return child.wait(),True


def run():
    from quasilisk.engine import validate_new_run
    plan=json.loads((HERE/'plan.json').read_text());completed=[];partial=[];not_run=[]
    assert Decimal(plan['engine_pool_usd'])+Decimal(plan['audit_reserved_usd'])+Decimal(plan['unallocated_margin_usd'])<=Decimal(plan['authorized_usd'])
    with (HERE/'.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        abandoned=[]
        for entry in plan.get('abandoned_entries', []):
            path=existing(entry)
            if path is None:raise RuntimeError('Missing abandoned evidence: '+entry['cell'])
            row=validated_record(entry)
            if row['checkpoint']!=entry['abandoned_checkpoint']:
                raise RuntimeError('Abandoned journal changed; reassess campaign accounting')
            abandoned.append(row)
        for entry in plan['entries']:
            if entry.get('not_run_reason'):
                if existing(entry) is not None:
                    raise RuntimeError('Cannot omit existing paid evidence: '+entry['cell'])
                not_run.append({'cell':entry['cell'],'reason':entry['not_run_reason']})
                continue
            prior=[x['cost_liability_usd'] for x in abandoned+completed+partial]
            cap=remaining_cap(plan['engine_pool_usd'],prior)
            path=existing(entry)
            if path is not None:
                row=validated_record(entry)
                if row['terminal_reason'] in {'completed_plan','completed_no_eligible_work'}:
                    completed.append(row);continue
                if row['terminal_reason']=='cost_limit_reached' and entry.get('accept_partial_reason'):
                    partial.append(row);continue
                raise RuntimeError('Existing unfinished/failed cell requires evidence review; no duplicate launch: '+entry['cell'])
            if cap<=Decimal(entry['maximum_single_dispatch_usd']):
                raise RuntimeError('Remaining pool cannot safely admit every role for '+entry['cell'])
            config_path=ROOT/entry['config'];config=json.loads(config_path.read_text())
            config['provider']['max_cost_usd']=str(cap);atomic(config_path,config)
            resources=validate_new_run(config_path)
            admission={'cell':entry['cell'],'prior_validated_liabilities_usd':str(sum(map(Decimal,prior),Decimal(0))),
                'frozen_run_ceiling_usd':str(cap),'engine_pool_usd':plan['engine_pool_usd'],
                'config_digest':resources.config.config_digest,'prior_runs':abandoned+completed}
            atomic(HERE/(entry['cell']+'.admission.json'),admission)
            print(json.dumps({'starting':entry['cell'],'ceiling_usd':str(cap)}),flush=True)
            with (HERE/(entry['cell']+'.log')).open('a') as log:
                returncode,interrupted=run_cli(config_path,log)
            path=existing(entry)
            if path is None:raise RuntimeError('CLI did not create a journal: '+entry['cell'])
            row=validated_record(entry);row['exit_code']=returncode;row['launcher_interrupted']=interrupted
            completed.append(row)
            total=sum((Decimal(x['cost_liability_usd']) for x in abandoned+completed+partial),Decimal(0))
            assert total<=Decimal(plan['engine_pool_usd'])
            atomic(HERE/'execution.json',{'cells':completed,'abandoned_cells':abandoned,'engine_liability_usd':str(total),'audit_reserved_usd':plan['audit_reserved_usd']})
            print(json.dumps(row),flush=True)
            if interrupted or returncode!=0 or row['terminal_reason'] not in {'completed_plan','completed_no_eligible_work'}:
                raise RuntimeError('Cell stopped; inspect exact evidence before any next launch: '+entry['cell'])
        atomic(HERE/'execution.json',{'cells':completed,'partial_cells':partial,'abandoned_cells':abandoned,'not_run':not_run,'engine_liability_usd':str(sum((Decimal(x['cost_liability_usd']) for x in abandoned+completed+partial),Decimal(0))),'audit_reserved_usd':plan['audit_reserved_usd'],'all_cells_completed':not (not_run or partial),'campaign_finished':True})


if __name__=='__main__':
    if len(sys.argv)!=2 or sys.argv[1]!='run':raise SystemExit('Usage: python3 experiments/multimodel_verification/campaign.py run')
    # Background jobs can inherit SIGINT=SIG_IGN from an interactive shell.
    # Restore Python's handler so a targeted kill -INT waits for CLI shutdown.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    run()
