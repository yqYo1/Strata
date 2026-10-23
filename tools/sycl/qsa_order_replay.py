#!/usr/bin/env python3
"""Main-owned consumer build and checked readback qualification; repository LICENSE.

The replay links the diagnostic generation's unchanged production archives.
Prepare/commit/build and prepare-readback/commit/readback are separate stages.
No model capture, query-order optimization or decode adoption occurs here.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys

sys.dont_write_bytecode=True
import build_locality_capture as diagnostic
import build_engine as build
import generation_engine as gen
import measurement_guards as guards
import measurement_owner as supervision
import ur_validation as validation
from measure_down_scatter import postchecks

P,B,A=gen.P,gen.B,gen.A
ROOT=diagnostic.ROOT/'replay'
RESULT=A/'qsa-order-replay-build-current.json'
READBACK=diagnostic.ROOT/'readback-v1'
READBACK_RESULT=A/'qsa-locality-readback-current.json'
FIXTURE='bench/sycl/qsa_order_replay.cpp'
SCOPE='qsa_bounded_tail_step_capture_audit_20261023'


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def diagnostic_admission():
    closed=gen.committed(diagnostic.RESULT)
    guards.require(closed['passed'] and closed['complete'] and closed['ownership_closed'] and
        not closed['active'] and closed['stage']=='closed','closed diagnostic build required')
    plan=json.loads((diagnostic.ROOT/'plan.json').read_text())
    guards.require(build.digest(diagnostic.ROOT/'plan.json')==closed['plan_sha256'],'diagnostic plan changed')
    diagnostic.current(plan)
    for item in closed['artifacts'].values():
        guards.require(build.digest(item['path'])==item['sha256'],'diagnostic artifact changed')
    return closed,plan


def runtime_lineage():
    # Independent Main-authored lineage check previously qualified at a509cdbbf.
    path=A/'qsa-selection-current.json';closed=gen.committed(path)
    guards.require(closed['passed'] and closed['complete'] and closed['ownership_closed'] and
        not closed['active'] and not closed['fresh_faults'] and closed['boot_unchanged'] and
        closed['native_validation']['normal_native_exit']==0,'closed model/runtime required')
    ref=closed['prepared_plan'];raw=gen.git('show',ref['commit']+':'+ref['path'])
    guards.require(hashlib.sha256(raw).hexdigest()==ref['sha256'],'runtime source plan changed')
    prior=json.loads(raw);pins=prior['diagnostic_qualification']['runtime_lineage']['runtime_pins']
    guards.require(len(pins)==31,'complete runtime inventory required')
    for name,pin in pins.items():guards.require(build.digest(name)==pin,'runtime changed: '+name)
    probe=validation.preload_qualification()
    guards.require(probe==prior['native_validation_probe'],'private validation probe changed')
    return dict(runtime_pins=pins,native_validation_probe=probe)


def inputs(revision):
    pins=gen.benchmark_pins(revision)
    raw=gen.git('show',revision+':'+FIXTURE)
    guards.require(raw==(P/FIXTURE).read_bytes(),'commit exact replay fixture first')
    return dict(pins,**{FIXTURE:hashlib.sha256(raw).hexdigest()})


def prepare():
    guards.require(not ROOT.exists() and not RESULT.exists(),'fresh replay build only')
    closed,plan=diagnostic_admission();revision=gen.git('rev-parse','HEAD').decode().strip()
    ROOT.mkdir(mode=0o700)
    archives=[diagnostic.ROOT/'build'/name for name in ('libstrata_kernels.a','libstrata_sycl_runtime.a')]
    argv=[plan['compiler']['path'],'-fsycl','-fsycl-id-queries-range=size_t','-DSTRATA_ENABLE_SYCL=1',
        '-std=c++20','-O3','-DNDEBUG','-Wall','-Wextra','-Werror','-fno-fast-math','-ffp-contract=off',
        '-I'+str(P/'include'),'-I'+str(P/'include/strata/sycl/compat'),str(P/FIXTURE),
        '-Wl,--start-group',*[str(p) for p in archives],'-Wl,--end-group','-pthread','-ldl',
        '-o',str(ROOT/'qsa_order_replay')]
    record=dict(stage='prepared',source_commit=revision,inputs=inputs(revision),
        diagnostic_build_sha256=build.digest(diagnostic.RESULT),
        production_source_commit=closed['source_commit'],archives={str(p):build.digest(p) for p in archives},
        command=argv,environment=plan['environment'],passed=False,complete=False,active=False,
        ownership_closed=True,GPU_executed=False,model_adoption_qualified=False,
        budgets=dict(wall=600,cpu=2400,rss_cap=8<<30,text_cap=16<<20,file_cap=128<<20,total_cap=192<<20),
        consumer='Actual117K final512-row ID/step replay: unchanged production ordered-exp wrapper; '
            'natural/reverse/CPU16-query overlap grouping,original query output ownership and bit preservation. '
            'Host order construction/upload/restore excluded;whole wrapper includes prefix pack and drain. '
            'Must separately qualify smoke and full replay with actual masks before any clean timing. '
            'Retire this same additional diagnostic generation and raw IDs after its decision.')
    write(ROOT/'plan.json',record);write(RESULT,record)


def build_replay():
    record=gen.committed(RESULT)
    guards.require(record==json.loads((ROOT/'plan.json').read_text()) and not (ROOT/'record.json').exists(),
        'fresh committed replay build plan')
    diagnostic_admission()
    guards.require(build.digest(diagnostic.RESULT)==record['diagnostic_build_sha256'] and
        inputs(record['source_commit'])==record['inputs'],'replay build inputs changed')
    for p,h in record['archives'].items():guards.require(build.digest(p)==h,'production archive changed')
    owner=None
    def persist():write(ROOT/'record.json',dict(record,active=bool(owner and owner.active),commands=owner.commands if owner else []))
    owner=supervision.Owner(ROOT/'owned',persist)
    try:
        entry,_,_=owner.run('build',record['command'],record['environment'],P,**record['budgets'])
        guards.require(supervision.closed(entry) and entry['exit_code']==0,'replay build failed')
        diagnostic_admission()
        guards.require(inputs(record['source_commit'])==record['inputs'],'replay inputs changed during build')
        p=ROOT/'qsa_order_replay';record.update(passed=True,complete=True,
            binary=dict(path=str(p),bytes=p.stat().st_size,sha256=build.digest(p)))
    except BaseException as error:record['error']=repr(error)
    finally:
        record.update(stage='closed',active=False,ownership_closed=owner.active is None and all(supervision.closed(c) for c in owner.commands))
        persist();write(RESULT,record)
    print(json.dumps({k:record.get(k) for k in ('passed','complete','ownership_closed','error')}),flush=True)
    return 0 if record['passed'] else 1


def readback_admission(record):
    closed,_=diagnostic_admission()
    guards.require(closed['artifacts']['readback']==record['binary'] and
        inputs(record['controller_source_commit'])==record['inputs'],'readback input changed')
    scope=gen.committed(A/'research-index.json')[SCOPE]
    guards.require(scope['small_gpu_launch_eligible'] and scope['review']==record['review'],
        'reviewed small launch required;unrelated research rows are not inputs')
    for k in ('report','native_review'):
        item=record['review'][k];guards.require(build.digest(item['path'])==item['sha256'],'source review changed')
    guards.require(runtime_lineage()==record['runtime_lineage'],'runtime lineage changed')


def prepare_readback():
    guards.require(not READBACK.exists() and not READBACK_RESULT.exists(),'fresh readback transaction only')
    closed,plan=diagnostic_admission();scope=gen.committed(A/'research-index.json')[SCOPE]
    guards.require(scope.get('small_gpu_launch_eligible') is True,'review small GPU readiness first')
    READBACK.mkdir(mode=0o700);jobs=[];lineage=runtime_lineage();probe=lineage['native_validation_probe']
    transaction=secrets.token_hex(16)
    for mode in ('small','full'):
        root=READBACK/mode;root.mkdir()
        for name in ('tmp','cache'):(root/name).mkdir()
        base=root/'capture'
        guards.require(not Path(str(base)+'.ids').exists() and not Path(str(base)+'.steps').exists(),'fresh paired destination')
        env=dict(plan['environment'],TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'),STRATA_TRACE='1',
            LD_PRELOAD=probe['library']['path'],STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'],
            UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION' if mode=='small' else 'UR_LAYER_PARAMETER_VALIDATION',
            UR_LOG_TRACING='level:info;flush:info;output:stderr',UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
            UR_LOG_LOADER='level:warning;flush:warning;output:stderr',UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
        jobs.append(dict(mode=mode,base=str(base),transaction_id=transaction+'-'+mode,
            command=[closed['artifacts']['readback']['path'],mode,str(base)],environment=env))
    revision=gen.git('rev-parse','HEAD').decode().strip()
    record=dict(stage='prepared',controller_source_commit=revision,inputs=inputs(revision),
        native_source_commit=closed['source_commit'],binary=closed['artifacts']['readback'],review=scope['review'],
        runtime_lineage=lineage,jobs=jobs,checks=[],active=False,ownership_closed=True,passed=False,complete=False,GPU_executed=False,
        model_capture_qualified=False,model_adoption_qualified=False,
        transaction_id=transaction,
        transaction='Main creates fresh mode0700 directories and refuses existing paired files before child launch. '
            'An exclusive durable transaction.claim precedes all children and prohibits resumption/retry. '
            'Each child checks exact headers/payload/EOF;no append/retry/reuse across transactions. '
            'Only checked native exit,parser proof,validation and source/runtime checks produce an exclusive completion.json '
            'with transaction/source/binary and both file hashes. Any partial pair fails and is ineligible for replay. '
            'Model capture requires separate committed request gates '
            'and implemented replay source/binary qualification;these synthetic ABI checks do not qualify it.',
        budgets=dict(wall=120,cpu=600,rss_cap=4<<30,text_cap=24<<20,file_cap=32<<20,total_cap=64<<20))
    readback_admission(record);write(READBACK/'plan.json',record);write(READBACK_RESULT,record)


def run_readback():
    record=gen.committed(READBACK_RESULT)
    guards.require(record==json.loads((READBACK/'plan.json').read_text()) and not (READBACK/'record.json').exists()
        and not (READBACK/'transaction.claim').exists(),'fresh committed readback plan')
    readback_admission(record);boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    journal=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=next(x.split(': ',1)[1] for x in journal.splitlines() if x.startswith('-- cursor: '));owner=None
    def persist():write(READBACK/'record.json',dict(record,active=bool(owner and owner.active),commands=owner.commands if owner else []))
    owner=supervision.Owner(READBACK/'owned',persist)
    try:
        with (READBACK/'transaction.claim').open('x') as claim:
            claim.write(record['transaction_id']+'\n');claim.flush();os.fsync(claim.fileno())
        persist()
        for job in record['jobs']:
            for suffix in ('.ids','.steps'):guards.require(not Path(job['base']+suffix).exists(),'capture transaction already used')
            entry,out,err=owner.run(job['mode'],job['command'],job['environment'],P,**record['budgets']);record['GPU_executed']=True
            guards.require(supervision.closed(entry) and entry['exit_code']==0,'readback child did not close normally')
            text=out.read_text();trace=err.read_text()
            rows,cap,take=(17,33,17) if job['mode']=='small' else (8192,2051,512)
            expected=f'locality_capture mode={job["mode"]} source_rows={rows} captured_rows={take} cap={cap} '
            guards.require(text.count(expected+'exact=PASS guards=PASS invalid_geometry=PASS inherited_drain=PASS')==1,
                'missing exact readback proof')
            activation=validation.preload_evidence(trace,entry['pid'])
            body=trace.split(f'strata UR validation: pid={entry["pid"]} phase=before-main',1)[1]
            guards.require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|Strata SYCL cleanup:',body),'runtime validation error')
            pair=dict(mode=job['mode'],transaction_id=job['transaction_id'],state='complete',
                native_source_commit=record['native_source_commit'],binary=record['binary'],exit_code=0,
                normal_owned_exit=supervision.closed(entry),activation=activation,
                files={suffix:dict(bytes=Path(job['base']+suffix).stat().st_size,sha256=build.digest(job['base']+suffix)) for suffix in ('.ids','.steps')})
            readback_admission(record);probe=dict(passed=True,ownership_closed=True);postchecks(probe,boot,cursor)
            guards.require(probe['passed'],'fresh fault or incomplete observation;stop before next child')
            with (Path(job['base']).parent/'completion.json').open('x') as completion:
                json.dump(pair,completion,indent=2);completion.write('\n');completion.flush();os.fsync(completion.fileno())
            record['checks'].append(pair);persist()
        record.update(passed=True,complete=True)
    except BaseException as error:record.update(error=repr(error),passed=False,complete=False)
    finally:
        record.update(stage='closed',active=False,ownership_closed=owner.active is None and all(supervision.closed(c) for c in owner.commands))
        postchecks(record,boot,cursor);persist();write(READBACK_RESULT,record)
    print(json.dumps({k:record.get(k) for k in ('passed','complete','ownership_closed','fresh_faults','error')}),flush=True)
    return 0 if record['passed'] else 1


if __name__=='__main__':
    operations={'prepare':prepare,'build':build_replay,'prepare-readback':prepare_readback,'readback':run_readback}
    guards.require(len(sys.argv)==2 and sys.argv[1] in operations,'use prepare|build|prepare-readback|readback')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        sys.exit(operations[sys.argv[1]]() or 0)
