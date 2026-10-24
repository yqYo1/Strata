#!/usr/bin/env python3
"""Owned host-only collector gate. No engine build, model, GPU or speed claim."""
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
import build_engine as build
import measurement_guards as guards
import measurement_owner as supervision

P,B=build.P,build.B
ROOT=B/'cpu-service-host-v1'
RESULT=P/'bench/results/2026-10-15-pr6-recovery-audit/cpu-service-current.json'
NATIVE=('tests/core/cpu_service_profile_test.cpp','include/strata/kernels/cpu/service_profile.hpp',
        'include/strata/kernels/cpu/native_expert.hpp')
INPUTS=(*NATIVE,'tools/sycl/cpu_service_profile.py','tools/sycl/build_engine.py',
        'tools/sycl/measurement_guards.py','tools/sycl/measurement_owner.py')
EXPECTED='cpu_service_profile counters,phase isolation,thread ownership,clocks,refusals=PASS\n'

def write(path,row): path.write_text(json.dumps(row,indent=2)+'\n')

def pins(revision):
    out={}
    for name,raw in build.blobs(P,revision,INPUTS):
        guards.require(raw==(P/name).read_bytes(),'commit exact collector input: '+name)
        out[name]=build.digest(P/name)
    return out

def current(row):
    guards.require(RESULT.read_bytes()==build.git(P,'show','HEAD:'+str(RESULT.relative_to(P))) and
        json.loads(RESULT.read_text())==json.loads((ROOT/'plan.json').read_text())==row,
        'commit exact collector plan')
    guards.require(pins(row['source_commit'])==row['input_sha256'],'collector source changed')
    for item in (row['compiler'],row['stdlib']):
        guards.require(build.digest(item['path'])==item['sha256'],'host provider changed')

def prepare():
    guards.require(not ROOT.exists() and not RESULT.exists(),'fresh collector gate only;no retry')
    revision=build.git(P,'rev-parse','HEAD').decode().strip();source=pins(revision)
    compiler='/usr/bin/g++';stdlib='/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33'
    providers={n:dict(path=p,sha256=build.digest(p))for n,p in [('compiler',compiler),('stdlib',stdlib)]}
    providers['compiler']['version']=subprocess.check_output([compiler,'--version'],text=True,timeout=10).splitlines()[0]
    ROOT.mkdir(mode=0o700);(ROOT/'tmp').mkdir()
    row=dict(stage='prepared',source_commit=revision,input_sha256=source,**providers,
        command=[compiler,'-std=c++20','-O2','-Wall','-Wextra','-Werror','-pedantic','-pthread',
            '-DSTRATA_CPU_SERVICE_PROFILE=1','-I',str(P/'include'),'-MMD','-MF',str(ROOT/'fixture.d'),
            str(P/NATIVE[0]),'-o',str(ROOT/'fixture')],
        environment=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8',TMPDIR=str(ROOT/'tmp')),
        budgets=dict(wall=90,cpu=120,rss_cap=2<<30,text_cap=2<<20,file_cap=8<<20,total_cap=16<<20),
        criteria='Owned compile/test exit0;actual project include closure exactly the three pinned native inputs;counter/row/NT conservation,phase exclusion,thread publication/identity,clock wall versus CPU,refusal/overflow and invalid-delta controls. No production kernels or GPU.',
        retention='Keep source,recipe,host outcome and limits;retire successful raw output and standalone fixture after committed replay. No extra native engine generation.',
        passed=False,complete=False,active=False,ownership_closed=True,GPU_executed=False,
        production_pool_checked=False,model_adoption_qualified=False)
    write(ROOT/'plan.json',row);write(RESULT,row)
    print('Prepared host-only collector gate;commit plan before compile/test.',flush=True)

def host(row):
    current(row);guards.require(row['stage']=='prepared' and not (ROOT/'owned').exists(),'single-use host gate')
    outcome=dict(row);owner=None
    def persist():
        write(ROOT/'record.json',dict(outcome,commands=owner.commands if owner else [],
            active=bool(owner and owner.active is not None)))
    try:
        owner=supervision.Owner(ROOT/'owned',persist)
        entry,_,_=owner.run('compile',row['command'],row['environment'],P,**row['budgets'])
        guards.require(supervision.closed(entry) and entry['exit_code']==0,'collector compile failed')
        dependencies=shlex.split((ROOT/'fixture.d').read_text().replace('\\\n',' ').partition(':')[2])
        found={str(Path(p).resolve().relative_to(P)) for p in dependencies}
        guards.require(found==set(NATIVE),'actual project include closure changed')
        outcome['compiled_project_inputs']=len(found)
        outcome['fixture']=dict(path=str(ROOT/'fixture'),sha256=build.digest(ROOT/'fixture'),bytes=(ROOT/'fixture').stat().st_size)
        entry,out,err=owner.run('check',[str(ROOT/'fixture')],row['environment'],ROOT,**row['budgets'])
        guards.require(supervision.closed(entry) and entry['exit_code']==0 and out.read_text()==EXPECTED,
            'collector controls failed')
        guards.require(err.read_text().startswith('strata CPU_SERVICE request=9 valid=0 ') and
            len(err.read_text().splitlines())==1,'expected negative-delta diagnostic differs')
        current(row);outcome.update(stage='host-checked',passed=True,complete=True)
    except BaseException as error:outcome.update(stage='failed',error=repr(error))
    finally:
        outcome.update(active=False,ownership_closed=bool(owner and owner.active is None and
            all(supervision.closed(c) for c in owner.commands)))
        persist();write(RESULT,outcome)
    print(json.dumps({k:outcome[k]for k in ('stage','passed','complete','ownership_closed')}),flush=True)
    return int(not outcome['passed'])

if __name__=='__main__':
    guards.require(len(sys.argv)==2 and sys.argv[1]in('prepare','host'),'prepare|host')
    with (B/'owned-v0141-measurement.lock').open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([])
        if sys.argv[1]=='prepare':prepare()
        else:sys.exit(host(json.loads(RESULT.read_text())))
