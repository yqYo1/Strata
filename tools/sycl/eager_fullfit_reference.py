#!/usr/bin/env python3
"""Current-period original14508 control after closed U1; Main authored.

Repository LICENSE. Two descriptive controls distinguish the historical speed
gap from the same-image U1effect. They do not replace its paired result or prove
an original/new build gain. No native build or changed experiment criteria.
"""
import copy
import fcntl
import json
from pathlib import Path
import socket
import statistics
import sys
from types import SimpleNamespace

sys.dont_write_bytecode=True
import eager_fullfit_bench as unit

P,B,RESULT=unit.P,unit.B,unit.RESULT
ROOT,KEY=B/'prefill-eager-original-control-v1','prefill_eager_original_control_v1'
THIS='tools/sycl/eager_fullfit_reference.py'
ORDER=[(32769,1),(65537,1)]
require,sha,write=unit.require,unit.sha,unit.write
compact=unit.compact


def save(row):
    write(ROOT/'record.json',row);data=json.loads(RESULT.read_text());data[KEY]=compact(row);write(RESULT,data)


def basis(expected=None):
    row=json.loads((unit.ROOT/'record.json').read_text());unit.current(row)
    require(row['stage']=='closed' and row['passed'] and row['complete'] and row['ownership_closed']
        and not row['active'] and len(row['processes'])==14,'closed original U1cohort required')
    # unit.current just called device.verify -> native -> build.check,which
    # fully verifies this exact original source/recipe/SDK/census/12ELF proof.
    # Reuse it within this call rather than hashing the SDK a second time.
    candidate=json.loads((unit.build.ROOT/'record.json').read_text())
    proof=candidate['foundation']
    require(expected is None or proof==expected,'original proof changed')
    require(sha(unit.build.original.ROOT/'record.json')==proof['original_record_sha256'],
        'verified original receipt changed')
    n=json.loads((unit.build.original.ROOT/'record.json').read_text())
    return row,n,proof


def current(row):
    require(unit.build.ur_validation.committed(RESULT,(KEY,))==compact(row),'commit preceding original-control state')
    require(unit.build.source.pin(P,list(row['sources']))['source_sha256']==row['sources'],'control/frontend changed')
    u,n,_=basis(row['original_proof'])
    require(row['binary']==n['binaries']['strata'] and row['u1_record_sha256']==sha(unit.ROOT/'record.json')
        and row['configuration']==u['configuration'] and row['model_assets']==u['model_assets']
        and row['boot']==u['boot'] and row['driver']==u['driver'],'frozen control origin changed')
    files=json.loads(Path(row['runtime_manifest']['path']).read_text())
    require(sha(row['runtime_manifest']['path'])==row['runtime_manifest']['sha256'] and
        all(sha(p)==h for p,h in files.items()),'actual runtime providers changed')
    contract,python=unit.core.frontend.python_manifest()
    require(contract==row['python_contract'] and python['selected']==row['python_selected'],'Python changed')
    unit.comparison.shared.models.model_pins(row['configuration'],row['model_assets'],full=False)
    return files,python


def native_context(row,plan):
    require((plan['read'],plan['heads'])==ORDER[plan['index']],'unplanned control')
    return {k:row[k] for k in ['binary','native_source_commit','native_validation_probe','boot']}


def decision(processes):
    u=json.loads((unit.ROOT/'record.json').read_text());points=[]
    require(len(processes)==2 and all(p['passed'] and p['complete'] and p['ownership_closed']
        and p['native_exit_code']==0 and p['fresh_faults']==[] for p in processes),'two normal closed controls required')
    for p in processes:
        old=next(x for x in u['decision']['points'] if x['read']==p['read'])
        points.append(dict(read=p['read'],original_current_PP=p['read']*1000/p['subsequent_median_ms'],
            original_current_TTFT_s=statistics.median(s['ttft_s'] for s in p['samples'][1:]),
            U1_jit_PP=statistics.median(x['warmed_PP'] for x in old['individual_processes']['jit']),
            U1_eager_PP=statistics.median(x['warmed_PP'] for x in old['individual_processes']['eager'])))
    return dict(points=points,descriptive_current_period_control_only=True,
        new_paired_build_gain_or_regression_qualified=False,inference_adopted=False,
        limits='One original process per length versus an earlier closed3pair same-image unit. Retain samples;no automatic attribution to code,hardware,clock or source path.')


def context():
    return SimpleNamespace(ROOT=ROOT,KEY=KEY,ORDER=ORDER,RUNS=unit.RUNS,current=current,
        environment=unit.environment,native_context=native_context,request_route=unit.request_route,
        check_sample=unit.speed.context().check_sample,decision=decision,compact=compact,save=save)


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()),'fresh bounded control only;no retry')
    u,n,proof=basis();ROOT.mkdir(mode=0o700)
    row={k:copy.deepcopy(u[k]) for k in ['configuration','environment_template','model_assets','template',
        'python_contract','python_selected','runtime_manifest','native_validation_probe','boot','driver','budgets','whole_bytes','request_hashes']}
    row['environment_template']={k:v.replace(str(unit.ROOT),str(ROOT)) for k,v in row['environment_template'].items()}
    row.update(stage='prepared',source_commit=unit.build.source.git(P,'rev-parse','HEAD').decode().strip(),
        sources=unit.core.pins(unit.build.source.git(P,'rev-parse','HEAD').decode().strip()),
        binary=n['binaries']['strata'],native_source_commit=unit.build.original.SOURCE,original_proof=proof,
        u1_record_sha256=sha(unit.ROOT/'record.json'),plans=[],processes=[],host_passed=False,
        passed=False,complete=False,active=False,ownership_closed=True,
        plan=dict(question='Does the unmodified retained fastest14508ELF reproduce its historical speed in the present period and exact U1frontend/configuration/workloads?',
            order='Two fresh original processes,32Kthen64K,three thinking-on exact24-output pinned bench.sh requests each;first separate from warmed2. Commit every closed process.',
            acceptance='Same exact runtime/Python/models/configuration/request hashes and all actual typed-QSA/largeGU/private-bank/MTP restore/zero-reuse/count/TTFT/provider/owner/fault gates as U1. Only ELF/native source switches to retained dcb8/14508. Fresh host probes. Reuse closed U1final full15asset hashes after stat identities;hash all15again after second process.',
            stopping='Stop first non-performance admission/route/owner/budget/fault failure;no retry,hash refresh or performance exclusion.',
            limits='Descriptive current-period control,not a new paired cross-build claim,extended Main regression investigation or route adoption. Preserve original U1mechanism decision unchanged.',
            retention='No new native generation. Retain compact samples and original/build identities for the immediate fastest-path choice;retire successful raw output after closure.'))
    for i,(read,heads) in enumerate(ORDER):
        root=ROOT/str(i);root.mkdir(mode=0o700)
        for name in ['tmp','cache']:(root/name).mkdir(mode=0o700)
        cfg=copy.deepcopy(row['configuration']);cfg['log']=str(root/'engine.stderr');write(root/'config.json',cfg)
        write(root/'plan.json',dict(binary=row['binary'],native_validation_probe=row['native_validation_probe']))
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        plan=dict(index=i,read=read,heads=heads,kernel='original14508',upload='jit',trace_progress=False,root=str(root),
            generated={name:sha(root/name) for name in ['config.json','plan.json']},
            command=['/bin/sh',str(P/'bench.sh'),'--config',str(root/'config.json'),'--port',str(port),
                '--prefill-exact-tokens',str(read),'--think','--runs',str(unit.RUNS),'--profile-request-gates',
                '--workload-id','native_gdn_20261018','--fail-fast-engine-error','--ready-timeout','600','--out',str(root/'bench')])
        plan['environment_sha256']=unit.comparison.fingerprint(unit.environment(row,plan));row['plans'].append(plan)
    save(row);print('Prepared two current-period original controls;commit before host/native.',flush=True)


if __name__=='__main__':
    require(sys.executable==unit.speed.model.PYTHON,'use qualified Python')
    require(sys.argv[1:] in [['prepare'],['host'],['0'],['1']],'prepare|host|0|1')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);unit.build.guards.no_model_owners([])
        if sys.argv[1]=='prepare':prepare()
        elif sys.argv[1]=='host':unit.comparison.host(context())
        else:unit.comparison.run(int(sys.argv[1]),context())
