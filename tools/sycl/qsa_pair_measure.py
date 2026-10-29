#!/usr/bin/env python3
"""Frozen full dependency-path comparison after pair component qualification.

Repository LICENSE applies. Does not change production dispatch or select a
model. Keeps the original failed/refused outcomes and stops on safety errors.
"""
import ast
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
sys.dont_write_bytecode=True
import measurement_clock as clock
import qsa_pair_probe as q
P,B,ROOT,RESULT=q.P,q.B,q.ROOT,q.RESULT
FOLDER=ROOT/'performance'


def inputs(revision):
    pending=[Path(__file__).resolve()];seen=set();pins={}
    while pending:
        path=pending.pop()
        if path in seen:continue
        seen.add(path);name=str(path.relative_to(P));raw=q.git('show',revision+':'+name)
        q.require(raw==path.read_bytes(),'commit exact comparison controller: '+name)
        pins[name]=hashlib.sha256(raw).hexdigest()
        for node in ast.walk(ast.parse(raw)):
            mods=[n.name for n in node.names] if isinstance(node,ast.Import) else \
                [node.module] if isinstance(node,ast.ImportFrom) else []
            for mod in mods:
                local=path.parent/((mod or '').split('.')[0]+'.py')
                if local.is_file():pending.append(local)
    return pins


def admit(record,plan):
    native,rt=q.admitted_build(record)
    gate=record['qsa_pair_proposal']['qualification']['correctness']['outcome']
    q.require(gate['passed'] and gate['complete'] and gate['ownership_closed'] and
        not gate['active'],'complete component oracle gate required')
    q.require(plan['binary']==native['outcome']['binary'] and plan['runtime_files']==rt['runtime_files'] and
        inputs(plan['source_commit'])==plan['controller_inputs'],'comparison inputs changed')
    fixture=plan['fixture'];q.require(q.digest(fixture['path'])==fixture['sha256'] and
        Path(fixture['path']).stat().st_size==fixture['bytes'],'initialized selected-ID fixture changed')
    clock.check(plan['clock']);return native,rt


def prepare():
    record=q.validation.committed(RESULT);proposal=record['qsa_pair_proposal']
    q.require('performance' not in proposal and not FOLDER.exists(),'fresh prospective comparison only')
    native,rt=q.admitted_build(record)
    gate=proposal['qualification']['correctness']['outcome']
    q.require(gate['passed'] and gate['complete'] and gate['ownership_closed'] and not gate['active'],
        'correctness gate failed/open')
    FOLDER.mkdir(mode=0o700)
    revision=q.git('rev-parse','HEAD').decode().strip()
    provider=next(p for p in rt['runtime_files'] if Path(p).name.startswith('libstdc++.so.'))
    hostenv=dict(native['environment'])
    hostenv['TMPDIR']=str(FOLDER/'tmp');hostenv['XDG_CACHE_HOME']=str(FOLDER/'cache')
    for name in ['tmp','cache']:(FOLDER/name).mkdir()
    timing=clock.prepare(native['outcome']['binary'],native['compiler'],hostenv,FOLDER,
        {'path':provider,'sha256':rt['runtime_files'][provider]})
    original=record['qsa_query_union_cost']['fixture']
    fixture={k:original[k] for k in ['path','sha256','bytes']}
    cases=[]
    for i,first in enumerate([0,1,0]):
        folder=FOLDER/f'process-{i}';folder.mkdir()
        for name in ['tmp','cache']:(folder/name).mkdir()
        env=dict(rt['environment'],TMPDIR=str(folder/'tmp'),XDG_CACHE_HOME=str(folder/'cache'),
            ZE_ENABLE_ALT_DRIVERS=next(p for p in rt['runtime_files'] if 'libze_intel_gpu' in p),
            UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
            UR_LOG_TRACING='level:warning;flush:warning;output:stderr',
            UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(folder/'validation.log'))
        cases.append(dict(label=f'process-{i}',first_arm=first,
            command=[native['outcome']['binary']['path'],'bench',fixture['path'],str(first)],environment=env))
    plan=dict(stage='prepared',source_commit=revision,controller_inputs=inputs(revision),
        binary=native['outcome']['binary'],runtime_files=rt['runtime_files'],fixture=fixture,
        clock=timing,cases=cases,
        budgets=dict(wall=240,cpu=900,rss_cap=8<<30,text_cap=8<<20,total_cap=256<<20,file_cap=64<<20),
        criteria='Three independent processes,first arm0/1/0;six alternating paired rounds per process,four calls per batch,two untimed one-call warmups. Arm0is selected ordered-exp pack+attention;arm1is identical pack+GPUunion+pair attention. All36samples eligible only with wall and summed actual-event service>=100ms,exact expected events/order,all-output constant-V oracle after every batch,actual captured union IDs/masks/suffix oracle before/after timing,immutable inputs/guards,clock endpoint/rounding<=0.1percent at100ms,quiet pre-main parameter validation,closed owners and no fresh fault. Advance component only if every process has candidate/base median wall AND full GPU service<=0.99. Stop first safety/coverage/correctness failure;no model trial on failed performance gate.',
        hardware_scope='All timed operands resident;commonGPUKVpack and union preparation included,CPU submission/drain in wall. Actual event sums and pack/prep/attention separated;in-order event span includes gaps. No timed PCIe copy,physical VRAM traffic,saturation or full-model ceiling inferred.',
        numerical_scope='Synthetic Q/K and uniform finiteV provide exact full-output timing oracle;nonuniform BF16 tensor/online contract and diagnostics qualify separately. Actual layer0tail IDs,6491queries,cap2051,live39259;not all layers/chunks or full-model quality.',
        retention='Retain36individual timing samples,per-process medians/ratios,clock/provider/source/binary and safety outcomes. Retire successful native traces/logs/cache and closed candidate after exact source/recipe recovery. Keep selected6e40 engine+12fixtures untouched.')
    admit(record,plan);q.write(FOLDER/'plan.json',plan);proposal['performance']=plan;q.write(RESULT,record)
    print('Prepared full dependency-path comparison;commit before clock compilation/native runs.',flush=True)


def parse(stdout,first,clock_result):
    rows=[]
    for line in stdout.splitlines():
        if not line.startswith('PAIR_SAMPLE '):continue
        fields=dict(part.split('=',1) for part in line[12:].split())
        q.require(set(fields)=={'round','order','arm','repeat','wall','service','span','pack','prep','attention','passed'},'unexpected timing schema')
        row={k:int(fields[k]) for k in ['round','order','arm','repeat','passed']}
        row.update({k:float(fields[k]) for k in ['wall','service','span','pack','prep','attention']})
        q.require(row['passed']==1 and row['repeat']==4 and row['arm']==(first+row['round']+row['order'])%2,
            'timed dispatch/order differs')
        q.require(all(math.isfinite(row[k]) and row[k]>=0 for k in ['wall','service','span','pack','prep','attention']) and
            row['wall']>=.1 and row['service']>=.1 and row['span']>=row['service']-1e-8,
            'ineligible clock/batch')
        q.require(math.isclose(row['service'],row['pack']+row['prep']+row['attention'],rel_tol=2e-8,abs_tol=2e-9) and
            (row['arm']==1 or row['prep']==0),'service decomposition differs')
        fraction=(2*clock_result['max_ns']/1e9+.00005)/row['wall']
        q.require(fraction<=.001,'actual endpoint/rounding fraction exceeds0.1percent')
        rows.append({k:v for k,v in row.items() if k not in ['repeat','passed']})
    q.require([(r['round'],r['order']) for r in rows]==[(r,o) for r in range(6) for o in range(2)] and
        stdout.count('PAIR_DONE mode=bench passed=1')==1,'timing coverage incomplete')
    medians={str(arm):{k:statistics.median([r[k] for r in rows if r['arm']==arm])
        for k in ['wall','service','span','pack','prep','attention']} for arm in [0,1]}
    ratios={k:medians['1'][k]/medians['0'][k] for k in ['wall','service']}
    return dict(samples=rows,medians=medians,ratios=ratios)


def run():
    record=q.validation.committed(RESULT);plan=record['qsa_pair_proposal']['performance']
    q.require(plan['stage']=='prepared' and plan==json.loads((FOLDER/'plan.json').read_text()) and
        not (FOLDER/'owned.json').exists(),'fresh committed timing plan required')
    native,rt=admit(record,plan)
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    text=q.subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=re.findall(r'^-- cursor: (.+)$',text,re.M);q.require(len(cursor)==1,'kernel cursor missing')
    result=dict(passed=False,complete=False,active=False,ownership_closed=False,processes=[],advance_to_model=False)
    owner=None
    def persist():
        result['active']=bool(owner and owner.active is not None)
        q.write(FOLDER/'owned.json',dict(result,commands=owner.commands if owner else []))
    owner=q.supervision.Owner(FOLDER/'owned',persist)
    try:
        timing=plan['clock'];clock.check(timing)
        c,_,_=owner.run('clock-build',timing['command'],timing['environment'],P,wall=60,cpu=120,
            rss_cap=1<<30,text_cap=1<<20,total_cap=256<<20,file_cap=8<<20)
        q.require(q.supervision.closed(c) and c['exit_code']==0,'clock compiler failed')
        c,out,_=owner.run('clock',[str(FOLDER/'clock_probe')],timing['environment'],P,wall=30,cpu=30,
            rss_cap=1<<30,text_cap=1<<20,total_cap=256<<20,file_cap=8<<20)
        q.require(q.supervision.closed(c) and c['exit_code']==0,'clock control failed')
        result['clock']=clock.evidence(out.read_text(),timing['provider'])
        q.require((2*result['clock']['max_ns']/1e9+.00005)/.1<=.001,'clock cost fails100ms endpoint floor')
        for case in plan['cases']:
            observed={}
            def interaction(io,entry):
                q.require(io.line(30)=='PAIR_GATE before-runtime','unexpected timing entry');io.send(b'X')
                q.require(io.line(90)=='PAIR_GATE runtime-ready','timing runtime failed')
                observed['start']=q.live_native(io.proc.pid,case['environment'],rt,plan['binary']);io.send(b'X')
                while io.line(180)!='PAIR_GATE before-exit':pass
                observed['end']=q.live_native(io.proc.pid,case['environment'],rt,plan['binary'])
                io.send(b'X');io.proc.stdin.close();io.exit(25)
            c,out,err=owner.run(case['label'],case['command'],case['environment'],P,interaction=interaction,**plan['budgets'])
            q.require(q.supervision.closed(c) and c['exit_code']==0,'timing child failed/open')
            stderr=err.read_text();proof=q.validation.preload_evidence(stderr,c['pid'])
            q.require(proof['loader']==rt['preload']['loader'],'timing UR provider mismatch')
            body=stderr.split(f'strata UR validation: pid={c["pid"]} phase=before-main',1)[1]
            q.require(not re.search('UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed|PAIR_ERROR',body),'timing native error')
            warnings=FOLDER/case['label']/'validation.log'
            q.require(not warnings.exists() or warnings.stat().st_size==0,'timing validation warning')
            evidence=parse(out.read_text(),case['first_arm'],result['clock'])
            result['processes'].append(dict(label=case['label'],**evidence,validation=proof,observed=observed))
            check=dict(passed=True,ownership_closed=True);q.postchecks(check,boot,cursor[0]);q.require(check['passed'],'fresh fault/boot failure')
            admit(record,plan);persist();print('Closed timing '+case['label']+' '+json.dumps(evidence['ratios']),flush=True)
        ratios=[p['ratios'] for p in result['processes']]
        result.update(passed=True,complete=True,advance_to_model=all(r[k]<=.99 for r in ratios for k in ['wall','service']),
            geomean={k:math.exp(statistics.mean(math.log(r[k]) for r in ratios)) for k in ['wall','service']})
    except BaseException as error:result['error']=repr(error)
    finally:
        result['ownership_closed']=owner.active is None and all(q.supervision.closed(c) for c in owner.commands)
        result['active']=False;q.postchecks(result,boot,cursor[0]);persist()
        if not result['passed']:result['advance_to_model']=False
        plan['stage']='closed' if result['passed'] else 'failed';plan['outcome']=result;q.write(RESULT,record)
    print(json.dumps({k:v for k,v in result.items() if k!='processes'}),flush=True);return int(not result['passed'])


def main():
    q.require(len(sys.argv)==2 and sys.argv[1] in ['prepare','run'],'use prepare|run')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);q.guards.no_model_owners([]);q.guards.driver_guard()
        if sys.argv[1]=='prepare':prepare();return 0
        return run()


if __name__=='__main__':raise SystemExit(main())
