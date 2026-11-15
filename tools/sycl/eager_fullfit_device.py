#!/usr/bin/env python3
"""Bounded U1 production upload checks; Main authored, repository LICENSE.

Both commands use the fresh coherent fixture. No model timing, decode or
physical-context qualification is inferred from these byte/lifetime checks.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode=True
import eager_fullfit_build as build
import qsa_cached_subgroup as diagnostics

P,B,RESULT=build.P,build.B,build.RESULT
ROOT,KEY=B/'prefill-eager-device-v1','prefill_eager_device_v1'
THIS='tools/sycl/eager_fullfit_device.py'
require,sha,write=build.require,build.sha,build.write
STAGES=['smoke','validate']
BUDGET=dict(wall=180,cpu=240,rss_cap=4<<30,text_cap=32<<20,file_cap=64<<20,total_cap=80<<20)


def compact(row):
    out={k:v for k,v in row.items() if k not in ['sources','commands']}
    out['sources']=dict(files=len(row['sources']),sha256=build.original.fingerprint(row['sources']))
    return out


def save(row):
    write(ROOT/'record.json',row);data=json.loads(RESULT.read_text());data[KEY]=compact(row);write(RESULT,data)


def native():
    row=json.loads((build.ROOT/'record.json').read_text());build.check(row)
    require(row['stage']=='built' and row['passed'] and row['complete'] and
        row['ownership_closed'] and not row['active'],'closed coherent candidate build required')
    for name in ['compiled','generated','links','objects','produced']:
        require(all(sha(p)==h for p,h in row['actual_inputs'][name].items()),'actual build input changed: '+name)
    require(all(sha(v['path'])==v['sha256'] and Path(v['path']).stat().st_size==v['bytes']
        for v in row['binaries'].values()),'candidate executable changed')
    return row


def verify(row):
    require(build.ur_validation.committed(RESULT,(KEY,))==compact(row),'commit preceding device stage')
    require(build.source.pin(P,list(row['sources']))['source_sha256']==row['sources'],'device controls changed')
    n=native()
    require(sha(build.ROOT/'record.json')==row['native_record_sha256'] and n['binaries']==row['binaries'],
        'exact native generation changed')
    require(build.guards.driver_guard()==row['driver'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip()==row['boot'],'platform changed')
    return n


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()),'fresh plan only;no retry')
    n=native();ROOT.mkdir(mode=0o700)
    sources=build.source.pin(P,['tools/sycl'])
    manifest=n['foundation']['runtime_manifest']
    require(sha(manifest['path'])==manifest['sha256'],'runtime manifest changed')
    row=dict(stage='prepared',source_commit=sources['revision'],sources=sources['source_sha256'],
        native_base_commit=n['native_base_commit'],native_overlay_commit=n['source_commit'],
        native_record_sha256=sha(build.ROOT/'record.json'),binaries=n['binaries'],
        environment=n['recipe']['environment'],runtime_manifest=manifest,
        native_validation_probe=n['native_validation_probe'],driver=build.guards.driver_guard(),
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),results=[],commands=[],
        passed=False,complete=False,active=False,ownership_closed=True,
        plan=dict(order=STAGES,first='Trace+parameter-validation+pre-main probe;two4expert cases,normal and cancellation.',
            second='Quiet parameter validation;0,1,4,127,512experts,strided source,full readback,unused bank/source guards,alias/extent/nonempty/partial refusals and checked cancellation.',
            providers='LD_DEBUG=libs records actual initialized ELF paths in these diagnostic-only processes;check all against original runtime84 and require actual SYCL,URv2,LevelZero GPU,IGC and pre-main libraries.',
            acceptance='Both normal0,exact case markers,quiet validation,closed owners,unchanged boot/driver and zero fresh faults. Stop first failure. No component performance gate.',
            next='Only after both pass,commit a new matched same-ELF jit/eager model-handoff and32K/64K speed plan. Keep decode/task/agent/fullphysical262144/RESTORE gates open.',
            retention='Keep useful case/closure/provider-conservation outcomes;retire successful API/loader traces after model admission consumes these gates. No new build generation.'),budget=BUDGET)
    save(row);print('Prepared two fresh eager-upload native checks;commit before execution.',flush=True)


def loaded_providers(errors,row):
    providers=json.loads(Path(row['runtime_manifest']['path']).read_text())
    probe=row['native_validation_probe'];providers[probe['library']['path']]=probe['library']['sha256']
    allowed={str(Path(p).resolve()):h for p,h in providers.items()}
    loaded={str(Path(p.strip()).resolve()) for p in re.findall(r'^\s*\d+:\s+calling init:\s*(/[^\n]+)$',errors,re.M)}
    require(loaded and all(p in allowed and sha(p)==allowed[p] for p in loaded),'unexpected initialized ELF/provider')
    required=[probe['library']['path'],probe['loader']['path'],row['environment']['UR_ADAPTERS_FORCE_LOAD']]
    require(all(str(Path(p).resolve()) in loaded for p in required),'required runtime/probe not actually initialized')
    require(any('libsycl.so' in p for p in loaded) and any('libze_intel_gpu.so' in p for p in loaded)
        and any('/libigc.so' in p for p in loaded),'actual SYCL/GPU/compiler provider absent')
    return dict(initialized_ELFs=len(loaded),sha256=build.original.fingerprint({p:allowed[p] for p in sorted(loaded)}),
        required_providers_verified=True,source='glibc loader constructor diagnostics;not timing or hardware utilization')


def run(stage):
    row=json.loads((ROOT/'record.json').read_text());verify(row);frozen=copy.deepcopy(row)
    require(row['stage'] in ['prepared','in_progress'] and len(row['results'])==STAGES.index(stage)
        and all(v['passed'] for v in row['results']),'one ordered stage;no retry')
    root=ROOT/stage;root.mkdir(mode=0o700)
    for name in ['tmp','cache']:(root/name).mkdir(mode=0o700)
    cursor=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).strip().split('-- cursor: ')[-1]
    require(cursor and '\n' not in cursor,'kernel cursor required')
    owner=None;result=dict(mode=stage,passed=False,complete=False,GPU_executed=False,host_probe_accepted=False)
    try:
        owner=build.ownership.Owner(root/'owned',lambda:write(root/'ownership.json',owner.commands))
        probe=row['native_validation_probe'];trace=stage=='smoke'
        env=dict(row['environment'],STRATA_SYCL_PREFILL_FULLFIT_UPLOAD='eager',STRATA_TRACE='1',
            TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'),LD_DEBUG='libs',
            LD_PRELOAD=probe['library']['path'],STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'],
            UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION' if trace else 'UR_LAYER_PARAMETER_VALIDATION',
            UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(root/'validation.log'))
        if not trace:env.pop('UR_LOG_TRACING',None)
        result['native_execution_attempted']=True
        entry,out,err=owner.run(stage,[row['binaries']['sycl_expert_bank_test']['path'],'--fullfit-eager-'+stage],env,P,**BUDGET)
        errors=err.read_text();proof=build.ur_validation.preload_evidence(errors,entry['pid'])
        diagnostics.diagnostics(errors,entry['pid'])
        require(build.ownership.closed(entry) and entry['exit_code']==0,'normal native closure required')
        require(proof['loader']==probe['loader']['path'],'actual validation loader changed')
        require(not trace or 'urEnqueue' in errors,'actual traced GPU submissions absent')
        require(not (root/'validation.log').exists() or (root/'validation.log').stat().st_size==0,'parameter validation warning')
        cases=[tuple(map(int,m)) for m in re.findall(r'^prefill_fullfit_eager PASS experts=(\d+) source_stride=(\d+) layers=(\d+) cancellations=(\d+) rejects=(\d+)$',out.read_text(),re.M)]
        expected=[(4,256,2,0,4),(4,512,2,2,4)]
        if not trace:expected += [(0,256,2,0,4),(1,256,3,3,4),(127,512,2,0,4),(512,256,2,0,4)]
        require(cases==expected and out.read_text().count('prefill_fullfit_eager_fixture PASS model=UNRUN physical262144_RESTORE=UNQUALIFIED')==1,'fixture case coverage differs')
        providers=loaded_providers(errors,row);verify(frozen)
        result.update(passed=True,complete=True,host_probe_accepted=True,GPU_executed=True,native_exit_code=0,
            cases=[dict(zip(['experts','source_stride','layers','cancellations','refusals'],case)) for case in cases],
            validation=proof,providers=providers)
    except BaseException as exc:result['error']=repr(exc)
    finally:
        if owner:build.ur_validation.finish(result,owner,cursor,row['boot'])
        result['passed']=bool(result['passed'] and result['complete'] and result.get('driver_after')==row['driver'])
        row['results'].append(result);row['commands']+=owner.commands if owner else []
        row['active']=bool(owner and owner.active);row['ownership_closed']=not row['active'] and all(build.ownership.closed(v) for v in row['commands'])
        row['complete']=len(row['results'])==len(STAGES) and all(v['passed'] for v in row['results'])
        row['passed']=row['complete'];row['stage']='failed' if not result['passed'] else 'closed' if row['complete'] else 'in_progress';save(row)
    require(result['passed'] and row['ownership_closed'],result.get('error','native safety check failed'))
    print('U1native '+stage+' closed0;model/speed/full context unqualified.',flush=True)


if __name__=='__main__':
    require(sys.argv[1:] in [['prepare'],['smoke'],['validate']],'prepare|smoke|validate')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);build.guards.no_model_owners([])
        if sys.argv[1]=='prepare':prepare()
        else:run(sys.argv[1])
