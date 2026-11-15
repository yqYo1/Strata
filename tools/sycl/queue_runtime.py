#!/usr/bin/env python3
"""Query actual production Runtime queue handles; Main-authored, repo LICENSE.

Commit source -> prepare -> commit recipe -> build -> commit ELF -> run.
Two fresh host-only TUs;no inference build,queue-policy change or submitted work.
"""
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

sys.dont_write_bytecode=True
import coherent_original_reuse as original
import engine_once
import measurement_guards as guards
import measurement_owner as owner_api
import ur_validation

P,B=original.P,original.B
ROOT=B/'queue-runtime-v2'
RESULT=P/'bench/results/2026-10-15-pr6-recovery-audit/cpu-profiler-current.json'
KEY='prefill_runtime_queues_v2'
SOURCE='bench/sycl/queue_runtime.cpp'
THIS='tools/sycl/queue_runtime.py'
PREFIXES=['include','src/sycl/runtime.cpp','tools/sycl',SOURCE,'AGENTS.md','docs/ARTIFACT_RETENTION.md']
LOADER='/usr/lib/x86_64-linux-gnu/libze_loader.so.1.32.0'
LIMITS=dict(wall=180,cpu=240,rss_cap=4<<30,text_cap=8<<20,file_cap=64<<20,total_cap=128<<20)
require,sha=guards.require,owner_api.sha
write=original.coherent.build.common.write
git=original.coherent.build.common.source.git


def save(row):
    write(ROOT/'record.json',row)
    data=json.loads(RESULT.read_text())
    data[KEY]={k:v for k,v in row.items() if k not in ['project','foundation','actual_inputs','commands']}
    data[KEY]['project']={k:v for k,v in row['project'].items() if k!='source_sha256'}
    if 'actual_inputs' in row:
        data[KEY]['actual_inputs']={k:dict(files=len(v),sha256=original.fingerprint(v)) for k,v in row['actual_inputs'].items()}
    data[KEY]['foundation_sha256']=original.fingerprint(row['foundation'])
    write(RESULT,data)


def check(row,committed=True):
    if committed:
        public=json.loads(RESULT.read_text())[KEY]
        require(public==ur_validation.committed(RESULT,(KEY,)), 'commit exact queue metadata stage')
    current=original.coherent.build.common.source.pin(P,PREFIXES)
    require(row['project']['source_sha256']==current['source_sha256'], 'source inputs changed')
    native,proof=original.verify_original(expected=row['foundation'])
    require(native['external_inputs']==row['external_inputs'],'original SDK manifest changed')
    require(guards.driver_guard()==row['driver'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==row['boot'],'boot/driver changed')
    if 'binary' in row:
        require(sha(row['binary']['path'])==row['binary']['sha256'],'exact executable changed')
        for items in row['actual_inputs'].values():
            require(all(sha(p)==h for p,h in items.items()),'actual compiler/link input changed')
    return native


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()),'no implicit metadata retry')
    old=ur_validation.committed(RESULT,('prefill_runtime_queues_v1',))
    require(old['stage']=='failed' and old['ownership_closed'] and not old['active']
        and old['error']=="RuntimeError('unfrozen link input: /usr/lib/x86_64-linux-gnu/libze_loader.so.1.32.0')",
        'preserve the closed linker-admission failure;no native queue inventory ran')
    native,proof=original.verify_original()
    database=json.loads((original.ROOT/'build/compile_commands.json').read_text())
    entry,=[v for v in database if v['file']==str(P/'src/sycl/runtime.cpp')]
    args=shlex.split(entry['command']);cc=args[0];flags=[];skip=False
    for a in args[1:]:
        if skip:skip=False;continue
        if a=='-o':skip=True;continue
        if a in ['-c',str(P/'src/sycl/runtime.cpp')]:continue
        flags.append(a)
    flags.append('-fsycl-host-only')
    require(not skip and '-fsycl' in flags and '-DSTRATA_CPU_SERVICE_PROFILE=1' in flags,'original runtime host compile contract required')
    recipe=[]
    for name,source in [('metadata',SOURCE),('runtime','src/sycl/runtime.cpp')]:
        recipe.append((name,[cc,*flags,'-MD','-MF',str(ROOT/(name+'.d')),'-c',str(P/source),'-o',str(ROOT/(name+'.o'))]))
    recipe.append(('link',[cc,*flags,str(ROOT/'metadata.o'),str(ROOT/'runtime.o'),LOADER,'-Wl,-t','-o',str(ROOT/'queue_runtime')]))
    seed=json.loads((B/'coherent-generations-prefill-v1/record.json').read_text())
    env={k:v for k,v in seed['environment_template'].items() if k in ['PATH','HOME','LANG','LC_ALL','LD_LIBRARY_PATH','ONEAPI_DEVICE_SELECTOR','UR_ADAPTERS_FORCE_LOAD','NEOReadDebugKeys','EnableDirectSubmission','UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD','STRATA_SYCL_PREFILL_SAME_QUEUE','ZE_ENABLE_ALT_DRIVERS','ZES_ENABLE_SYSMAN','SYCL_CACHE_PERSISTENT']}
    env.update(TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'),CUDA_MODULE_LOADING='EAGER',
        UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_TRACING='level:info;flush:info;output:file,'+str(ROOT/'ur.trace'),
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(ROOT/'validation.log'))
    row=dict(stage='prepared',project=original.coherent.build.common.source.pin(P,PREFIXES),
        foundation=proof,external_inputs=native['external_inputs'],native_validation_probe=native['native_validation_probe'],
        runtime_manifest=seed['runtime_manifest'],recipe=recipe,build_environment=native['recipe']['environment'],
        environment=env,driver=guards.driver_guard(),boot=seed['boot'],commands=[],
        passed=False,complete=False,active=False,ownership_closed=True,application_work_submitted=False,
        plan=dict(question='Actual Level Zero ordinals/indices returned by the unchanged production Runtime queue constructors',
            previous='V1compilation/link commands closed0;admission omitted the already frozen runtime manifest from the link allowlist. Keep V1failed;new V2plan includes that original unchanged manifest.',
            unit='Two fresh host-only TUs:metadata fixture and original-matching Runtime. No engine,math,kernel,placement,phase selector or safety-setting change.',
            qualification='Host-only metadata build has no usable inference image. It queries handles of Runtime compute,transfer,make_queue(true/false);zero URenqueue calls required.',
            gate='Exact committed source/original Runtime headers/SDK/recipe/executable/providers/private before-main validation;four in-order queue records,normal0,quiet validation,closed ownership,unchanged boot/driver,no fresh fault.',
            stop='Stop first compiler,identity,metadata API,provider,owner,validation,fault or finite budget failure;no retry or hash refresh.',
            retention='Temporary component beside two unchanged engine generations. Retire source/executable/logs after the event-lane audit receives compact evidence and original source/recipe recovery.',
            references=['https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_oneapi_backend_level_zero.md','https://oneapi-src.github.io/level-zero-spec/level-zero/1.16.24/core/api.html']))
    ROOT.mkdir(mode=0o700)
    for n in ['tmp','cache','owned']:(ROOT/n).mkdir(mode=0o700)
    save(row);print('Prepared actual Runtime queue inventory;commit before build.',flush=True)


def allowed(native,row):
    values=json.loads(Path(native['external_inputs']['path']).read_text())
    values.update(json.loads(Path(row['runtime_manifest']['path']).read_text()))
    for pin in [native['project'],*native['dependencies']]:
        values.update({str((Path(pin['root'])/n).resolve()):h for n,h in pin['source_sha256'].items()})
    values[str(P/SOURCE)]=sha(P/SOURCE)
    return values


def mappings(pid,row):
    providers=json.loads(Path(row['runtime_manifest']['path']).read_text())
    providers[str(Path(row['binary']['path']).resolve())]=row['binary']['sha256']
    for kind in ['loader','library']:
        x=row['native_validation_probe'][kind];providers[str(Path(x['path']).resolve())]=x['sha256']
    found={}
    for line in Path('/proc',str(pid),'maps').read_text().splitlines():
        f=line.split(maxsplit=5)
        if len(f)==6 and 'x' in f[1] and f[5].startswith('/'):
            p=str(Path(f[5]).resolve());require(p in providers and sha(p)==providers[p],'unfrozen executable map: '+p);found[p]=providers[p]
    return found


def execute(action):
    row=json.loads((ROOT/'record.json').read_text());native=check(row)
    require(row['stage']==('prepared' if action=='build' else 'built'),'fresh ordered action only')
    owner=owner_api.Owner(ROOT/'owned',lambda:write(ROOT/'ownership.json',owner.commands))
    try:
        if action=='build':
            permit=allowed(native,row);actual=dict(compiled={},links={},objects={})
            for name,argv in row['recipe']:
                entry,out,err=owner.run(name,argv,row['build_environment'],P,**LIMITS)
                require(owner_api.closed(entry) and entry['exit_code']==0,'compiler/link failed: '+name)
                if name!='link':
                    names=shlex.split((ROOT/(name+'.d')).read_text().replace('\\\n',' ').split(':',1)[1])
                    for n in names:
                        p=str(Path(n).resolve());h=sha(p);require(permit.get(p)==h,'compiler input differs from original qualified Runtime/SDK: '+p);actual['compiled'][p]=h
                    actual['objects'][str(ROOT/(name+'.o'))]=sha(ROOT/(name+'.o'))
                else:
                    permit.update(actual['objects'])
                    for line in out.read_text().splitlines():
                        p=Path(line.strip())
                        if p.is_absolute() and p.is_file():
                            h=sha(p);require(permit.get(str(p.resolve()))==h,'unfrozen link input: '+str(p));actual['links'][str(p.resolve())]=h
            require(len(actual['compiled'])>100 and len(actual['links'])>8,'complete actual dependency closure required')
            row.update(stage='built',actual_inputs=actual,binary=dict(path=str(ROOT/'queue_runtime'),sha256=sha(ROOT/'queue_runtime')),passed=True)
        else:
            row.update(stage='running',active=True,ownership_closed=False);save(row)
            records=[];gates=[];maps={}
            def interaction(channel,_entry):
                while True:
                    item=json.loads(channel.line(30))
                    if item['kind']=='gate':
                        name=item['name'];require(name==['before-runtime','runtime-ready','before-exit'][len(gates)],'metadata gate order')
                        maps[name]=mappings(channel.proc.pid,row);gates.append(name);channel.send(b'continue\n')
                    else:
                        records.append(item)
                        if item['kind']=='done':channel.exit(20);break
            env=engine_once.native_environment(row,row['environment'])
            entry,out,err=owner.run('inventory',[row['binary']['path']],env,P,interaction=interaction,**LIMITS)
            require(owner_api.closed(entry) and entry['exit_code']==0,'native inventory did not close normally')
            evidence=ur_validation.preload_evidence(err.read_text(),entry['pid'])
            validation=ROOT/'validation.log';require(not validation.exists() or validation.stat().st_size==0,'validation warning/error')
            trace=(ROOT/'ur.trace').read_text();require('urQueueCreate' in trace and not re.search(r'\burEnqueue\w+',trace),'metadata inventory unexpectedly enqueued work')
            queues=[v for v in records if v['kind']=='queue'];require([v['name'] for v in queues]==['compute','transfer','make_queue_profiled','make_queue_unprofiled'] and all(v['in_order'] for v in queues),'four actual in-order queue records required')
            check(row,False)
            row.update(stage='closed',passed=True,complete=True,active=False,ownership_closed=True,queues=queues,
                gates=gates,actual_providers=maps,validation=evidence,UR_enqueue_calls=0,native_exit_code=0,
                fresh_faults=False,boot_unchanged=True)
    except BaseException as exc:
        row.update(stage='failed',passed=False,complete=False,error=repr(exc));raise
    finally:
        row['commands']=owner.commands;row['active']=owner.active is not None
        row['ownership_closed']=owner.active is None and all(c.get('session_empty') and c.get('direct_child_reaped') for c in owner.commands)
        save(row)
    print(json.dumps({k:row.get(k) for k in ['stage','queues','ownership_closed','UR_enqueue_calls']}),flush=True)


if __name__=='__main__':
    require(sys.argv[1:] in [['prepare'],['build'],['run']],'prepare|build|run')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([])
        if sys.argv[1]=='prepare':prepare()
        else:execute(sys.argv[1])
