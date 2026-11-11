#!/usr/bin/env python3
"""Build a current-API compact-GU fixture from exact qualified native libraries.

Main-authored; repository LICENSE applies. No engine rebuild or GPU execution.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
sys.dont_write_bytecode = True
import check_moe_census_model as device
import measurement_owner as owner

P,B,RESULT=device.P,device.B,device.RESULT
ROOT,KEY=B/'gu-compact-component-v5','gu_compact_component_v5'
THIS,CPP='tools/sycl/build_gu_compact.py','bench/sycl/gu_compact.cpp'
WRAPPER,HEADER='src/prefill/sycl/gemm16.cpp','include/strata/prefill/gemm.hpp'
NATIVE=device.build.ROOT
require,sha=device.require,device.sha


def candidate_delta(before,after,observers):
    changes={p for p in before if p in after and before[p]!=after[p]}
    require(set(after)-set(before)==set(observers) and not set(before)-set(after)
        and changes=={WRAPPER,HEADER},'only declared wrapper/header and committed observers may differ')
    return changes


def source_controls():
    before={WRAPPER:'old-cpp',HEADER:'old-header','src/other.cpp':'fixed'}
    good=dict(before,**{WRAPPER:'new-cpp',HEADER:'new-header',THIS:'observer'})
    candidate_delta(before,good,{THIS})
    cases=[dict(good,**{'src/other.cpp':'changed'}),
        {p:h for p,h in good.items() if p!='src/other.cpp'},
        dict(good,**{'src/unplanned.cpp':'extra'}),
        {p:h for p,h in good.items() if p!=THIS},dict(good,**{HEADER:'old-header'})]
    for altered in cases:
        try:candidate_delta(before,altered,{THIS})
        except RuntimeError:continue
        raise RuntimeError('source admission accepted an undeclared native/controller mutation')
    binaries={name:dict(path=str(NATIVE/'build'/name),bytes=i+1,sha256=str(i))
        for i,name in enumerate(device.build.TARGETS)}
    inspect=lambda v:(v['bytes'],v['sha256'])
    current_outputs(binaries,inspect)
    for name in binaries:
        for entries,reader in (({k:v for k,v in binaries.items() if k!=name},inspect),
                (binaries,lambda v:(v['bytes'],'changed') if v is binaries[name] else inspect(v))):
            try:current_outputs(entries,reader)
            except RuntimeError:continue
            raise RuntimeError('current output admission accepted a missing/changed fixture')
    return dict(passed=True,positive=2,negative=len(cases)+2*len(binaries),GPU_executed=False)


def current_outputs(binaries,inspect=None):
    require(set(binaries)==set(device.build.TARGETS),'all nine actual baseline ELFs required')
    if inspect is None:inspect=lambda v:(Path(v['path']).stat().st_size,sha(v['path']))
    for name,v in binaries.items():
        require(Path(v['path'])==NATIVE/'build'/name and inspect(v)==(v['bytes'],v['sha256']),
            'current baseline engine/fixture changed')


def native(extra_observers=frozenset()):
    # Prospective component admission. The original97fda engine/SDK remain
    # exact; two declared source inputs are rebuilt into a private archive.
    old=json.loads((NATIVE/'record.json').read_text())
    validation=device.original.validation
    require(validation.committed(RESULT,(device.build.KEY,))==device.build.context().compact(old)
        and old['stage']=='built' and old['passed'] and old['complete']
        and old['ownership_closed'] and not old['active'],'closed exact baseline required')
    require(old['native_source_commit']==old['project']['revision']==old['actual_inputs']['source_commit']
        and old['actual_inputs']['translation_units']==132 and old['actual_inputs']['uncovered']==0
        and len(old['planned_compilation'])==133 and set(old['binaries'])==set(device.build.TARGETS),
        'exact original nine-target source/compiler census required')
    before=old['project']['source_sha256']
    source=device.original.common.source
    require(before=={p:hashlib.sha256(raw).hexdigest() for p,raw in
        source.blobs(P,old['project']['revision'],before)},'original baseline source unrecoverable')
    after=source.pin(P,device.original.common.PREFIXES)['source_sha256']
    observers={device.THIS,'tools/sycl/moe_host_census_model.py',THIS,CPP,
        'tools/sycl/check_gu_compact.py',*extra_observers}
    changes=candidate_delta(before,after,observers)
    for dep in old['dependencies']:
        require(dep['source_sha256']=={p:hashlib.sha256(raw).hexdigest() for p,raw in
            source.blobs(Path(dep['root']),dep['revision'],dep['source_sha256'])}
            and all(sha(Path(dep['root'])/p)==h for p,h in dep['source_sha256'].items()),
            'baseline dependency source changed')
    require(sha(old['recipe_seed']['path'])==old['recipe_seed']['sha256']
        and sha(old['external_inputs']['path'])==old['external_inputs']['sha256']
        and all(sha(p)==h for p,h in old['toolchain'].items())
        and all(sha(p)==h for p,h in json.loads(Path(old['external_inputs']['path']).read_text()).items()),
        'baseline compiler/SDK/recipe changed')
    require(all(sha(p)==h for p,h in (old['configured_files']|old['generated_inputs']).items())
        and sha(old['actual_inputs']['path'])==old['actual_inputs']['sha256'],'baseline generated/census input changed')
    actual=json.loads(Path(old['actual_inputs']['path']).read_text())
    for family in ('compiled','generated','links'):
        for p,h in actual[family].items():
            if Path(p) in (P/WRAPPER,P/HEADER):
                require(family=='compiled' and before[str(Path(p).relative_to(P))]==h,
                    'declared source must match original compiler evidence')
            else:require(sha(p)==h,'undeclared baseline actual input changed: '+p)
    current_outputs(old['binaries'])
    require(device.original.candidate.probe_qualification()==
        (old['native_validation_probe'],old['validation_qualification']),'qualified runtime probe changed')
    proof=dict(native_source_commit=old['native_source_commit'],original_source_files=len(before),
        observer_source_files=len(after),added_observers={p:after[p] for p in sorted(observers)},
        declared_recompiled_inputs={p:dict(original=before[p],candidate=after[p]) for p in sorted(changes)},
        source_admission_controls=source_controls(),
        baseline_compiler_and_nine_ELFs_exact=True,historical_ELFs_provenance_only=len(old['preserved_binaries']),
        native_engine_recompiled=False,candidate_wrapper_requires_recompile=True,GPU_result_inherited=False)
    return dict(old,observer_admission=proof)


def compact(row):
    return {k:v for k,v in row.items() if k not in ('execution','source_files')}


def save(row):
    (ROOT/'record.json').write_text(json.dumps(row,indent=2)+'\n')
    public=json.loads(RESULT.read_text());public[KEY]=compact(row)
    RESULT.write_text(json.dumps(public,indent=2)+'\n')


def current(row,extra_observers=frozenset()):
    require(device.original.validation.committed(RESULT,(KEY,))==compact(row),'commit exact component plan')
    n=native(extra_observers)
    require(n['native_source_commit']==row['native_source_commit']
        and n['binaries']['strata']==row['engine'] and n['observer_admission']==row['native_admission'],
        'qualified baseline or declared candidate input changed')
    for name,h in row['source_files'].items():
        raw=device.original.common.source.git(P,'show',row['source_commit']+':'+name)
        require(hashlib.sha256(raw).hexdigest()==h==sha(P/name),'component source changed')
    require(sha(row['compile_seed']['path'])==row['compile_seed']['sha256'],'compiler recipe changed')
    require(all(sha(p)==h for p,h in row['link_inputs'].items()),'same-build archive/image/runtime link input changed')
    require(sha(row['archive_tool']['path'])==row['archive_tool']['sha256'],'archive tool changed')
    if 'patched_archive' in row:
        require(sha(row['patched_archive']['path'])==row['patched_archive']['sha256'],'candidate archive changed')
    return n


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()),'fresh component only')
    n=native();model=device.original.validation.committed(RESULT,('moe_host_census_model_v1',))
    require(model['stage']=='closed' and model['passed'] and model['complete']
        and model['ownership_closed'] and not model['active']
        and model['native_build']['binaries']==n['binaries'],'closed exact model handoff required')
    build=NATIVE/'build';seed=build/'compile_commands.json';entries=json.loads(seed.read_text())
    entry,=[e for e in entries if 'sycl_verifier_retirement_test.dir' in e['command']]
    compile_argv=shlex.split(entry['command'])
    compile_argv=[x for x in compile_argv if not x.startswith('-DSTRATA_VERSION=')]
    compile_argv[compile_argv.index(entry['file'])]=str(P/CPP)
    compile_argv[compile_argv.index('-o')+1]=str(ROOT/'fixture.o')
    compile_argv+=['-fno-fast-math','-ffp-contract=off','-MD','-MF',str(ROOT/'fixture.d')]
    wrapper,=[e for e in entries if e['file']==str(P/WRAPPER)]
    wrapper_argv=shlex.split(wrapper['command'])
    wrapper_argv[wrapper_argv.index('-o')+1]=str(ROOT/'gemm16.cpp.o')
    wrapper_argv+=['-fno-fast-math','-ffp-contract=off','-MD','-MF',str(ROOT/'wrapper.d')]
    text=subprocess.check_output(['ninja','-C',str(build),'-t','commands',
        'strata'],text=True).splitlines()[-1]
    require(text.startswith(': && ') and text.endswith(' && :'),'known exact link rule required')
    link=shlex.split(text[5:-5])
    programs=[x for x in link if x.startswith('CMakeFiles/strata.dir/') and x.endswith('.o')]
    require(len(programs)==2 and all(link.count(x)==1 for x in programs),
        'two exact engine program objects required')
    at=link.index(programs[0]);link=[x for x in link if x not in programs]
    link.insert(at,str(ROOT/'fixture.o'));link[link.index('-o')+1]=str(ROOT/'fixture')
    inputs=[x for x in link if x.endswith('.a') or x in
        ('strata_vllm_xe2_image.o','strata_vllm_xe2_host.o') or (x.startswith('/') and x.endswith('.so'))]
    libraries={str((build/x).resolve()):sha((build/x).resolve()) for x in inputs}
    require(sum(x.endswith('.a') for x in inputs)==9 and
        all(x in inputs for x in ('strata_vllm_xe2_image.o','strata_vllm_xe2_host.o')),
        'nine exact engine archives and both vllm image/host objects required')
    require(link.count('libstrata_prefill.a')==1,'exact prefill archive replacement required')
    link[link.index('libstrata_prefill.a')]=str(ROOT/'libstrata_prefill.a')
    cache=(build/'CMakeCache.txt').read_text().splitlines()
    ar,=[s.split('=',1)[1] for s in cache if s.startswith('CMAKE_AR:FILEPATH=')]
    archive_argv=[ar,'r',str(ROOT/'libstrata_prefill.a'),str(ROOT/'gemm16.cpp.o')]
    revision=device.original.common.source.git(P,'rev-parse','HEAD').decode().strip()
    sources={x:sha(P/x) for x in (THIS,CPP,WRAPPER,HEADER)}
    require(all(hashlib.sha256(device.original.common.source.git(P,'show',revision+':'+x)).hexdigest()==h
        for x,h in sources.items()),'commit exact component source first')
    env=dict(n['recipe']['environment'],TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'))
    ROOT.mkdir(mode=0o700);(ROOT/'tmp').mkdir();(ROOT/'cache').mkdir()
    row=dict(stage='prepared',source_commit=revision,source_files=sources,
        native_source_commit=n['native_source_commit'],engine=n['binaries']['strata'],
        native_admission=n['observer_admission'],compile_seed=dict(path=str(seed),sha256=sha(seed)),
        link_inputs=libraries,archive_tool=dict(path=ar,sha256=sha(ar)),
        recipe=dict(compile=compile_argv,wrapper=wrapper_argv,archive=archive_argv,link=link,cwd=str(build),environment=env),
        plan='Compile fixture and the declared Gemm16 wrapper using original exact compiler entries. Privately copy the baseline prefill archive and replace exactly gemm16.cpp.o;verify every other member unchanged. Link all other original archives/images/runtime unchanged. Never overwrite/reconfigure the current engine. No old numerical/model result inherited.',
        prospective_checks='Actual production GU calls in both arms,with explicit per-job compact eligibility false/true and immutable startup auto configuration. Validate exact actual route outcomes every call. Smoke,44numerical cases including95/96/97,2333/2334and8192rows,then3processes x18shapes x4rotated pairs;150mscalibration/100mssample floors,max262144repeats. Select only changed small-row shapes with repeated gain. No model PP/TG/adoption follows.',
        retention='One current-API component fixture using the existing native generation;no new engine generation. Retire object/build output after useful compiled-input/safety/speed evidence is committed and the next production decision closes. Five existing engine/proof holds remain separate and are under source review.',
        previous_failure=dict(key='gu_compact_component_v1',source_commit='e3b4069d086c250b05748c20eec50e058c3adcf5',evidence_commit='b56f19571',stage='failed',GPU_executed=False,reason='Verifier-only link rule omitted prefill and vllm image/host objects. V2uses the actual engine link rule;no engine rebuild or mathematical change.'),
        previous_calibration=dict(key='gu_compact_device_v1',evidence_commit='03a4407eaa0968e74c2a2e897bb51f9ff9ec7348',stage='failed',native_exit_code=1,ownership_closed=True,fresh_faults=[],reason='M1 exhausted8192repeats before the frozen150mscalibration floor. No eligible speed sample. Increase only the finite repeat cap to262144;keep150mscalibration/100mseligibility and every numerical gate. Rebuild and requalify the fixture;do not relabel V1.'),
        production_comparison_basis=dict(key='gu_compact_device_v2',evidence_commit='d6bbdccf892a5293c38e5c8f2c6a7cb25513b03a',reason='Compact-only paired component gains do not compare against production auto,which selects wide atM>=128. V4measures the actual unchanged production baseline and more row boundaries. Candidate includes common finish but not future production binding;no engine or decode change.'),
        budget=dict(wall=300,cpu=600,rss_cap=8<<30,text_cap=32<<20,file_cap=128<<20,total_cap=512<<20),
        active=False,ownership_closed=True,passed=False,complete=False,GPU_executed=False,commands=[])
    save(row);print('Prepared current-library compact-GU fixture;commit before building.',flush=True)


def build():
    row=json.loads((ROOT/'record.json').read_text());current(row);frozen=copy.deepcopy(row)
    require(row['stage']=='prepared' and not row['active'] and row['ownership_closed'],'one prospective build')
    scope=None
    def persist():
        (ROOT/'execution.json').write_text(json.dumps(dict(active=bool(scope and scope.active),
            commands=scope.commands if scope else []),indent=2)+'\n')
    try:
        scope=owner.Owner(ROOT/'owned-build',persist)
        for label in ('compile','wrapper','archive','link'):
            if label=='archive':
                shutil.copyfile(NATIVE/'build/libstrata_prefill.a',ROOT/'libstrata_prefill.a')
            e,out,err=scope.run(label,row['recipe'][label],row['recipe']['environment'],
                Path(row['recipe']['cwd']),**row['budget'])
            require(owner.closed(e) and e['exit_code']==0,'component '+label+' failed')
            row['commands'].append(dict(label=label,argv=row['recipe'][label],exit_code=e['exit_code'],
                ownership_closed=True,stdout_sha256=sha(out),stderr_sha256=sha(err)))
        ar=row['archive_tool']['path']
        members=lambda archive:subprocess.check_output([ar,'t',str(archive)],text=True).splitlines()
        original=NATIVE/'build/libstrata_prefill.a';patched=ROOT/'libstrata_prefill.a'
        names=members(original)
        require(names==members(patched) and len(names)==len(set(names))
            and names.count('gemm16.cpp.o')==1,'archive member identities changed')
        hashes=lambda archive:{name:hashlib.sha256(subprocess.check_output([ar,'p',str(archive),name])).hexdigest() for name in names}
        before,after=hashes(original),hashes(patched)
        require({name for name in names if before[name]!=after[name]}=={'gemm16.cpp.o'}
            and after['gemm16.cpp.o']==sha(ROOT/'gemm16.cpp.o'),'only the declared wrapper object may change')
        row['patched_archive']=dict(path=str(patched),sha256=sha(patched),original_sha256=sha(original),
            changed_member='gemm16.cpp.o',member_sha256=after['gemm16.cpp.o'],unchanged_members=len(names)-1)
        headers={}
        for depfile in ('fixture.d','wrapper.d'):
            deps=(ROOT/depfile).read_text().replace('\\\n',' ').split(':',1)[1]
            headers.update({str((Path(row['recipe']['cwd'])/p).resolve()):sha((Path(row['recipe']['cwd'])/p).resolve())
                for p in shlex.split(deps)})
        (ROOT/'actual-headers.json').write_text(json.dumps(headers,indent=2)+'\n')
        row['actual_headers']=dict(path=str(ROOT/'actual-headers.json'),sha256=sha(ROOT/'actual-headers.json'),files=len(headers))
        row['binary']=dict(path=str(ROOT/'fixture'),sha256=sha(ROOT/'fixture'),bytes=(ROOT/'fixture').stat().st_size)
        current(frozen);row.update(stage='built',passed=True,complete=True)
    except BaseException as error:
        row.update(stage='failed',error=repr(error));raise
    finally:
        row['active']=bool(scope and scope.active)
        row['ownership_closed']=bool(scope and not scope.active and all(owner.closed(x) for x in scope.commands))
        save(row)
    print('Built compact-GU fixture only;GPU and speed unqualified.',flush=True)


if __name__=='__main__':
    require(len(sys.argv)==2 and sys.argv[1] in ('prepare','build'),'prepare|build')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);device.original.guards.no_model_owners([])
        prepare() if sys.argv[1]=='prepare' else build()
