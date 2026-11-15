#!/usr/bin/env python3
"""Fresh coherent U1 engine from committed14508sources plus four owned files.

Main-authored; repository LICENSE. The explicit source composition excludes
unadopted KV engine changes. No engine/object reuse, GPU work or implicit retry.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.dont_write_bytecode=True
import coherent_original_reuse as original
import measurement_guards as guards
import measurement_owner as ownership
import ur_validation

P,B=original.P,original.B
ROOT=B/'prefill-eager-native-v1'
RESULT=P/'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
KEY='prefill_eager_native_v1'
OVERLAYS=['include/strata/prefill/fullfit_upload_plan.hpp',
    'include/strata/prefill/expert_bank_upload.hpp','src/prefill/moe_window.inc','tests/sycl/expert_bank.cpp']
PREFIXES=['tools/sycl',*OVERLAYS]
TARGETS=['strata','sycl_expert_bank_test']
LIMITS=dict(wall=2400,cpu=7200,rss_cap=12<<30,text_cap=32<<20,file_cap=512<<20,total_cap=4<<30)
require,sha=guards.require,ownership.sha
source=original.coherent.build.common.source
write=original.coherent.build.common.write


def compact(row):
    value={k:v for k,v in row.items() if k not in ['controller_sources','source_files','commands','actual_inputs','foundation']}
    for field in ['controller_sources','source_files']:
        value[field]=dict(files=len(row[field]),sha256=original.fingerprint(row[field]))
    value['foundation_sha256']=original.fingerprint(row['foundation'])
    if 'actual_inputs' in row:
        value['actual_inputs']={k:dict(files=len(v),sha256=original.fingerprint(v)) if isinstance(v,dict) else v
            for k,v in row['actual_inputs'].items()}
    return value


def save(row):
    write(ROOT/'record.json',row);data=json.loads(RESULT.read_text());data[KEY]=compact(row);write(RESULT,data)


def check(row,committed=True):
    if committed:require(ur_validation.committed(RESULT,(KEY,))==compact(row),'commit exact build stage')
    require(source.pin(P,list(row['controller_sources']))['source_sha256']==row['controller_sources'],'committed controller/overlay changed')
    native,_=original.verify_original(expected=row['foundation'])
    for pin in native['dependencies']:
        require(all(sha(Path(pin['root'])/p)==h for p,h in pin['source_sha256'].items()),'committed dependency changed')
    require(all(sha(ROOT/'source'/p)==h for p,h in row['source_files'].items()),'materialized committed native input changed')
    require(guards.driver_guard()==row['driver'],'driver changed')
    return native


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()),'fresh build only;no implicit retry')
    native,proof=original.verify_original();revision=source.git(P,'rev-parse','HEAD').decode().strip()
    files={};payload={}
    for name,raw in source.blobs(P,original.SOURCE,native['project']['source_sha256']):
        files[name]=hashlib.sha256(raw).hexdigest();payload[name]=raw
    for name in OVERLAYS:
        raw=source.git(P,'show',revision+':'+name);require(raw==(P/name).read_bytes(),'uncommitted overlay')
        files[name]=hashlib.sha256(raw).hexdigest();payload[name]=raw
    # The kernel/math/placement sources and class definitions come from the
    # coherent fastest reference,including its original Gemm header.
    for name in ['include/strata/prefill/gemm.hpp','src/prefill/prefill.cpp','src/core/verify.cpp']:
        require(files[name]==native['project']['source_sha256'][name],'unrelated inference change bundled')
    ROOT.mkdir(mode=0o700)
    for part in ['source','tmp','cache','owned']:(ROOT/part).mkdir(mode=0o700)
    for name,raw in payload.items():
        target=ROOT/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    configure=list(native['recipe']['configure']);configure[configure.index('-S')+1]=str(ROOT/'source')
    configure[configure.index('-B')+1]=str(ROOT/'build')
    env={k:v.replace(str(original.ROOT),str(ROOT)) for k,v in native['recipe']['environment'].items()}
    row=dict(stage='prepared',source_commit=revision,native_base_commit=original.SOURCE,
        native_source_identity='Committed base plus explicit overlays;not the whole overlay revision native tree',
        overlays={p:files[p] for p in OVERLAYS},controller_sources=source.pin(P,PREFIXES)['source_sha256'],
        source_files=files,foundation=proof,external_inputs=native['external_inputs'],
        recipe=dict(configure=configure,environment=env,targets=TARGETS),
        native_validation_probe=native['native_validation_probe'],driver=guards.driver_guard(),
        passed=False,complete=False,active=False,ownership_closed=True,GPU_executed=False,
        plan=dict(unit='U1only:all existing individual expert uploads move before compute on empty full-fit512banks,same bytes/queue/layout/math. U2coalescing is not selected by the native caller.',
            source='Every native source/header recovered from coherent14508except four committed owned files. Complete source map and compiler/link census bind the composed snapshot.',
            qualification='Build fresh engine and actual expert-bank fixture. Commit build;traced tiny eager fixture,then quiet full eager fixture. No component performance gate;model timing requires those safety gates and a separate prospective matched plan.',
            limits='CPUplanner qualification is reused only for its unchanged header/oracle. Native safety,model handoff,32K/64Kspeed,decode preservation,quality and physical262144/RESTORE are unqualified.',
            retention='Bounded third generation:original14508prefill reference plus namingAdecode/phase reference remain unchanged. This candidate is for the immediate U1safety/model comparison;retire after its decision before another candidate. No ended generation renewed.'),commands=[])
    check(row,False);save(row);print('Prepared committed coherent U1 source composition;commit before build.',flush=True)


def census(row,native,stdout):
    build=ROOT/'build';database=json.loads((build/'compile_commands.json').read_text())
    allowed={str((ROOT/'source'/p).resolve()):h for p,h in row['source_files'].items()}
    for pin in native['dependencies']:
        allowed.update({str((Path(pin['root'])/p).resolve()):h for p,h in pin['source_sha256'].items()})
    allowed.update(json.loads(Path(row['external_inputs']['path']).read_text()))
    runtime=row['foundation']['runtime_manifest']
    require(sha(runtime['path'])==runtime['sha256'],'original runtime manifest changed')
    allowed.update(json.loads(Path(runtime['path']).read_text()))
    allowed.update(native['configured_files'])
    compiled,generated,objects,classes={},{},{},{}
    selected=[]
    for item in database:
        args=shlex.split(item['command']);require('-o' in args,'compile object missing')
        obj=Path(args[args.index('-o')+1]);obj=obj if obj.is_absolute() else Path(item['directory'])/obj
        if not obj.exists():continue
        objects[str(obj.resolve())]=sha(obj);selected.append(item)
        text=subprocess.check_output(['/usr/bin/ninja','-t','deps',str(obj.relative_to(build))],cwd=build,text=True,timeout=30)
        require('VALID' in text.splitlines()[0],'actual compiler dependencies missing')
        for line in text.splitlines()[1:]:
            if not line.startswith('    '):continue
            p=Path(line.strip());p=p if p.is_absolute() else build/p;p=p.resolve();h=sha(p)
            if str(p) in allowed:
                require(allowed[str(p)]==h,'compiled input changed: '+str(p));compiled[str(p)]=h
            else:
                require(p.is_relative_to(build) and p.suffix in ['.h','.hpp'],'unfrozen compiler input: '+str(p));generated[str(p)]=h
            if p==ROOT/'source/include/strata/prefill/gemm.hpp':
                require('-DSTRATA_ENABLE_SYCL=1' in args,'inconsistent Gemm class condition')
                classes[str(obj.resolve())]=h
    expected_classes={'CMakeFiles/strata_prefill.dir/src/prefill/'+p+'.o'
        for p in ['prefill.cpp','sycl/gemm.cpp','sycl/gemm16.cpp','sycl/qsa_score_gemm.cpp']}
    require(len(selected)>100 and {str(Path(p).relative_to(build)) for p in classes}==expected_classes,
        'complete fresh engine and all four production Gemm includers required')
    produced={str(p.resolve()):sha(p) for p in build.rglob('*') if p.is_file()
        and (p.suffix=='.a' or '.so' in p.name or str(p.resolve()) in objects)}
    links={}
    for line in stdout.read_text().splitlines():
        value=line.strip()
        if not value.startswith('/') or len(value)>4096 or any(c.isspace() for c in value):continue
        p=Path(value)
        if not p.is_file():continue
        p=p.resolve();h=sha(p)
        if str(p) in produced:require(produced[str(p)]==h,'produced link input changed')
        elif str(p) in allowed:require(allowed[str(p)]==h,'external link input changed');links[str(p)]=h
        elif line.strip().startswith('/'):raise RuntimeError('unfrozen link input: '+str(p))
    require(len(links)>8,'actual external link closure missing')
    return dict(compiled=compiled,generated=generated,links=links,objects=objects,produced=produced,
        Gemm_includers=classes,translation_units=len({x['file'] for x in selected}),fresh_objects=len(selected),
        partial_archive_reuse=False,uncovered=0)


def execute(stage):
    row=json.loads((ROOT/'record.json').read_text());native=check(row)
    require(row['stage']==('prepared' if stage=='configure' else 'configured'),'one ordered stage only')
    owner=ownership.Owner(ROOT/'owned',lambda:write(ROOT/'ownership.json',owner.commands))
    args=row['recipe']['configure'] if stage=='configure' else ['/usr/bin/cmake','--build',str(ROOT/'build'),'--target',*TARGETS,'-j','4']
    try:
        entry,out,err=owner.run(stage,args,row['recipe']['environment'],P,**LIMITS)
        require(ownership.closed(entry) and entry['exit_code']==0,stage+'failed;inspect existing receipt')
        if stage=='configure':row['stage']='configured'
        else:
            row['actual_inputs']=census(row,native,out)
            row['binaries']={name:dict(path=str(ROOT/'build'/name),sha256=sha(ROOT/'build'/name),bytes=(ROOT/'build'/name).stat().st_size) for name in TARGETS}
            row.update(stage='built',passed=True,complete=True)
        check(row,False)
    except BaseException as exc:row.update(stage='failed',passed=False,error=repr(exc));raise
    finally:
        row['commands']+=owner.commands;row['active']=owner.active is not None
        row['ownership_closed']=owner.active is None and all(ownership.closed(v) for v in owner.commands);save(row)
    print('U1build stage '+row['stage']+';no GPU/model qualification.',flush=True)


if __name__=='__main__':
    require(sys.argv[1:] in [['prepare'],['configure'],['compile']],'prepare|configure|compile')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([])
        if sys.argv[1]=='prepare':prepare()
        else:execute(sys.argv[1])
