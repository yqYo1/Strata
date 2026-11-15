#!/usr/bin/env python3
"""Main-owned CPU-only byte/publication oracle for a prefill upload proposal.

Repository LICENSE. Commit source,prepare,commit recipe,then compile/run once.
No inference target, GPU enumeration, native rebuild, or performance claim.
"""
import fcntl
import json
from pathlib import Path
import shlex
import sys

sys.dont_write_bytecode=True
import coherent_original_reuse as original
import measurement_owner as ownership
import ur_validation

P,B=original.P,original.B
ROOT=B/'fullfit-upload-host-v2'
RESULT=P/'bench/results/2026-10-15-pr6-recovery-audit/cpu-profiler-current.json'
KEY='prefill_fullfit_upload_host_v2'
SOURCE='tests/prefill/fullfit_upload_plan_test.cpp'
HEADER='include/strata/prefill/fullfit_upload_plan.hpp'
SELF='tools/sycl/fullfit_upload_host.py'
PREFIXES=['tools/sycl',HEADER,SOURCE]
require,sha=ownership.require,ownership.sha
write=original.coherent.build.common.write
EXPECTED='prefill full-fit upload oracle: byte_cases=6 refusals=10 failure_publication=checked'
LIMITS=dict(wall=120,cpu=120,rss_cap=2<<30,text_cap=1<<20,file_cap=32<<20,total_cap=48<<20)


def save(row):
    write(ROOT/'record.json',row);data=json.loads(RESULT.read_text())
    data[KEY]={k:v for k,v in row.items() if k not in ['project','foundation','commands','actual_inputs']}
    data[KEY]['source_commit']=row['project']['revision']
    data[KEY]['source_sha256']=row['project']['aggregate_sha256']
    if 'actual_inputs' in row:data[KEY]['actual_inputs']={k:dict(files=len(v),sha256=original.fingerprint(v)) for k,v in row['actual_inputs'].items()}
    write(RESULT,data)


def verify(row,committed=True):
    if committed:require(json.loads(RESULT.read_text())[KEY]==ur_validation.committed(RESULT,(KEY,)),'commit exact host recipe/stage')
    require(original.coherent.build.common.source.pin(P,PREFIXES)['source_sha256']==row['project']['source_sha256'],'host inputs changed')
    native,_=original.verify_original(expected=row['foundation'])
    return native


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()),'fresh CPU oracle only')
    prior=ur_validation.committed(RESULT,('prefill_fullfit_upload_host_v1',))
    require(prior['ownership_closed'] and not prior['active'] and not prior['passed']
        and prior.get('qualification_hold') and not (B/'fullfit-upload-host-v1').exists(),
        'retain original launch-guard hold and retire ended CPUgeneration before V2')
    native,proof=original.verify_original();cc='/opt/intel/oneapi/compiler/2026.2/bin/icpx'
    flags=[cc,'--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13','-std=c++20','-O2','-Wall','-Wextra','-Werror','-fno-fast-math','-ffp-contract=off','-I'+str(P/'include')]
    row=dict(stage='prepared',project=original.coherent.build.common.source.pin(P,PREFIXES),foundation=proof,
        compiler=dict(path=cc,sha256=sha(cc)),external_inputs=native['external_inputs'],
        recipe=[('compile',[*flags,'-MD','-MF',str(ROOT/'actual.d'),'-c',str(P/SOURCE),'-o',str(ROOT/'oracle.o')]),
            ('link',[cc,'--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13',str(ROOT/'oracle.o'),'-Wl,-t','-o',str(ROOT/'oracle')]),
            ('oracle',[str(ROOT/'oracle')])],
        environment={k:v for k,v in native['recipe']['environment'].items() if k in ['PATH','HOME','LANG','LC_ALL','LD_LIBRARY_PATH']},
        expected=EXPECTED,passed=False,complete=False,active=False,ownership_closed=True,GPU_executed=False,
        plan=dict(unit='CPU-only immutable plan and accepted-enqueue publication for empty512-slot full-fit prefill banks.',
            resource_evidence='Whole-request unitrace_current_request_resources_v1:copy union10.309796s,kernel union28.658416s,about4ms overlap;service is not physical traffic/saturation. Actual32Ksource1/staging0has28full-fit layers among48. Ordinary Runtime queues all use ordinal0/index0. Test whether fixed copy transaction/submission work can be amortized while bytes stay identical;compatible hardware headroom remains unknown.',
            next_consumer='Main same-queue native transfer integration;retire CPUELF after original input recovery and retain source only through this bounded proposal.',
            oracle='Six independent byte-layout cases,ten refusals including failed third enqueue;unchanged payload/guards/dense destination ranks and no failed-copy publication.',
            future_comparison_units=['U1:JIT per-expert versus eager per-expert with identical bytes,math,placement and queue.','U2:eager per-expert versus eager contiguous-run copies with identical schedule,bytes and permanent-slot ownership.','Then original JIT versus the useful combination;retain U1even if slower. No constituent adoption from combined timing alone.'],
            gate='Exact committed source/SDK/compiler/actual header/link inputs,normal0/empty owned sessions and exact oracle marker. Not native events,lifetimes,performance,model quality,handoff,decode or physical262144qualification.'))
    row['environment'].update(TMPDIR=str(ROOT/'tmp'),PYTHONDONTWRITEBYTECODE='1')
    ROOT.mkdir(mode=0o700);(ROOT/'tmp').mkdir(mode=0o700)
    save(row);print('Prepared CPU-only upload-plan oracle;commit before compilation.',flush=True)


def execute():
    row=json.loads((ROOT/'record.json').read_text());native=verify(row)
    require(row['stage']=='prepared','one oracle attempt')
    permit=json.loads(Path(native['external_inputs']['path']).read_text())
    for pin in [native['project'],*native['dependencies']]:permit.update({str((Path(pin['root'])/p).resolve()):h for p,h in pin['source_sha256'].items()})
    for p in [SOURCE,HEADER]:permit[str(P/p)]=row['project']['source_sha256'][p]
    owner=ownership.Owner(ROOT/'owned',lambda:write(ROOT/'ownership.json',owner.commands));actual=dict(compiled={},links={})
    try:
        for label,args in row['recipe']:
            require(original.coherent.build.common.source.pin(P,PREFIXES)['source_sha256']==row['project']['source_sha256']
                and sha(row['compiler']['path'])==row['compiler']['sha256'], 'before-command source/compiler mismatch')
            if label=='oracle':
                require(args==[row['binary']['path']] and sha(args[0])==row['binary']['sha256']
                    and all(sha(p)==h for group in actual.values() for p,h in group.items()),
                    'before-launch executable or consumed input mismatch')
            entry,out,err=owner.run(label,args,row['environment'],P,**LIMITS)
            require(ownership.closed(entry) and entry['exit_code']==0,'host command failed: '+label)
            if label=='compile':
                for p in shlex.split((ROOT/'actual.d').read_text().replace('\\\n',' ').split(':',1)[1]):
                    path=str(Path(p).resolve());h=sha(path);require(permit.get(path)==h,'unfrozen header: '+path);actual['compiled'][path]=h
                permit[str(ROOT/'oracle.o')]=sha(ROOT/'oracle.o')
            elif label=='link':
                for line in out.read_text().splitlines():
                    p=Path(line.strip())
                    if p.is_absolute() and p.is_file():
                        path=str(p.resolve());h=sha(path);require(permit.get(path)==h,'unfrozen link: '+path);actual['links'][path]=h
                row['binary']=dict(path=str(ROOT/'oracle'),sha256=sha(ROOT/'oracle'))
            else:require(out.read_text().strip()==EXPECTED and not err.stat().st_size,'byte/publication oracle failed')
        row.update(stage='closed',passed=True,complete=True,actual_inputs=actual)
    except BaseException as error:row.update(stage='failed',passed=False,error=repr(error));raise
    finally:
        row.update(commands=owner.commands,active=owner.active is not None,ownership_closed=owner.active is None and all(ownership.closed(e) for e in owner.commands));save(row)
    print(EXPECTED,flush=True)


if __name__=='__main__':
    require(sys.argv[1:] in [['prepare'],['run']],'prepare|run')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if sys.argv[1]=='prepare':prepare()
        else:execute()
