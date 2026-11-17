#!/usr/bin/env python3
"""Fresh coherent long-prefill window proposal from pinned owned sources.

Main-authored from owned eager_fullfit_build.py at51ea99cae;repository LICENSE.
Preserve the independently owned native-job bound and positive U1 component.
Exclude unrelated KV/verify proposals. No object reuse,GPU work or implicit retry.
"""
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
ROOT=B/'prefill-window-native-v1'
RESULT=P/'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
KEY='prefill_window_native_v1'
U1_MEASURED_SOURCE='51ea99cae7ef4b1e41e419bd5f5a8a1dc17d0054'
U1_SOURCE='f13b3baf46af5f104d11178a094f561ec1cfc8f6'
SAFETY_SOURCE='9849e35803f011aaafb2753c1e0f31411e40308e'
OVERLAYS=[
    'include/strata/prefill/fullfit_upload_plan.hpp',
    'include/strata/prefill/expert_bank_upload.hpp',
    'include/strata/prefill/moe_accum_plan.hpp',
    'include/strata/prefill/moe_window_routes.hpp',
    'src/core/expert_source.cpp',
    'src/prefill/prefill.cpp', 'src/prefill/layer_major.inc', 'src/prefill/moe_window.inc',
    'src/prefill/sycl/native16.cpp', 'src/prefill/sycl/vllm_xe2/native16.cpp',
    'cmake/sycl_vllm_xe2_native16.cmake',
    'tests/sycl/expert_bank.cpp', 'tests/moe_accum_routes.cpp',
    'tests/prefill_moe_workspace.cpp', 'bench/sycl/moe_down_scatter.cpp']
PREFIXES=['tools/sycl',*OVERLAYS,'AGENTS.md','docs/ARTIFACT_RETENTION.md','docs/BENCHMARKS.md']
TARGETS=['strata','prefill_moe_routes_test','prefill_moe_workspace_test',
    'sycl_expert_bank_test','sycl_moe_down_scatter_bench','sycl_native16_paired_gemm_bench',
    'strata_full_context_diagnostic','full_context_diagnostic_host_test']
LIMITS=dict(wall=2400,cpu=7200,rss_cap=12<<30,text_cap=32<<20,file_cap=512<<20,total_cap=4<<30)
require,sha=guards.require,ownership.sha
source=original.coherent.build.common.source
write=original.coherent.build.common.write


def bank_gate():
    value=ur_validation.committed(RESULT,('prefill_bank_minimum_v2',))
    require(value['stage']=='closed' and not value['active'] and value['passed']
        and value['complete'] and value['ownership_closed'] and len(value['processes'])==9
        and value['native_source_commit']==original.SOURCE
        and value['decision']['eligible_for_separately_planned_window_question'] is True,
        'finish the prospective independent bank-only comparison before a new build')
    # A safe measured bank-only loss remains a loss. The original prospective
    # plan explicitly permits a separate window question;this is not adoption.
    return dict(key='prefill_bank_minimum_v2',source_commit=value['source_commit'],
        source_inputs=value['source_inputs'],binary=value['binary'],decision=value['decision'],
        scope='Closed bank safety/capacity/cost only;no window/math/model/decode adoption inherited.')


def decode_reference(recover=False):
    path=B/'phase-kv-naming-native-v1/record.json'
    value=json.loads(path.read_text());binary=value['binaries']['strata']
    require(value['stage']=='built' and value['passed'] and value['complete']
        and not value['active'] and value['ownership_closed']
        and value['native_source_commit']==value['project']['revision']==SAFETY_SOURCE
        and binary['sha256']=='6d2b4f7d6a79424c3da5ac6e35d7e1336016604e2f7bdb68b84031d2879903de'
        and sha(binary['path'])==binary['sha256']
        and Path(binary['path']).stat().st_size==binary['bytes'],
        'preserve exact independent decode reference before adding a generation')
    if recover:
        original.coherent.recover(value['project'])
        for pin in value['dependencies']:original.coherent.recover(pin)
    return dict(source=SAFETY_SOURCE,binary=binary,record_sha256=sha(path),
        project_inputs=original.fingerprint(value['project']),
        recipe_sha256=original.fingerprint(value['recipe']),
        scope='Artifact continuity and original Git/recipe preservation only;new prefill build inherits no decode/model qualification.')


def u1_continuity(payload):
    header='include/strata/prefill/fullfit_upload_plan.hpp'
    wrapper='include/strata/prefill/expert_bank_upload.hpp'
    fixture='tests/sycl/expert_bank.cpp'
    for name in (header,wrapper,fixture):
        require(payload[name]==source.git(P,'show',U1_SOURCE+':'+name),
            'review a changed retained individual-upload input:'+name)
    # The rejected U2cleanup removed only a branch that the measured U1caller
    # always disabled. Verify that reviewed transformation,not a hash refresh.
    prior=source.git(P,'show',U1_MEASURED_SOURCE+':'+header)
    argument=b', bool coalesce'
    branch=(b'            if (coalesce && source_stride==bank.stride)\n'
            b'                while (end<ids.size() && ids[end]==ids[end-1]+1) ++end;\n')
    require(prior.count(argument)==prior.count(branch)==1 and
        prior.replace(argument,b'').replace(branch,b'')==payload[header],
        'retained planner differs beyond the reviewed unreachable U2branch removal')
    prior=source.git(P,'show',U1_MEASURED_SOURCE+':'+wrapper)
    old=b'PrefillFullfitUploadPlan::make(experts,layout,source_stride,false)'
    new=b'PrefillFullfitUploadPlan::make(experts,layout,source_stride)'
    require(prior.count(old)==1 and prior.replace(old,new)==payload[wrapper]
        and payload[fixture]==source.git(P,'show',U1_MEASURED_SOURCE+':'+fixture),
        'U1selected caller or original fixture changed')
    return dict(measured_source=U1_MEASURED_SOURCE,retained_source=U1_SOURCE,
        reviewed_change='Remove the unused coalesce argument/branch after failed U2;U1always passed false. Caller and complete retained planner match that explicit transformation;original device fixture bytes unchanged.',
        compiled_or_model_qualification_inherited=False)


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
    require(bank_gate()==row['bank_gate'],'original independent bank decision changed')
    require(decode_reference()==row['retained_decode_reference'],'independent decode reference changed')
    return native


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()),'fresh build only;no implicit retry')
    gate=bank_gate();decode=decode_reference(recover=True)
    native,proof=original.verify_original();revision=source.git(P,'rev-parse','HEAD').decode().strip()
    files={};payload={}
    for name,raw in source.blobs(P,original.SOURCE,native['project']['source_sha256']):
        files[name]=hashlib.sha256(raw).hexdigest();payload[name]=raw
    for name in OVERLAYS:
        raw=source.git(P,'show',revision+':'+name);require(raw==(P/name).read_bytes(),'uncommitted overlay')
        files[name]=hashlib.sha256(raw).hexdigest();payload[name]=raw
    # Preserve the independently qualified job-capacity guard;it must not be
    # lost while excluding the separate KV/verify performance proposals.
    guard='src/core/expert_source.cpp'
    require(payload[guard]==source.git(P,'show',SAFETY_SOURCE+':'+guard),
        'review a changed independent native-job safety fix')
    continuity=u1_continuity(payload)
    for name in ['include/strata/prefill/gemm.hpp','src/core/verify.cpp',
                 'src/kernels/sycl/kv_stream.cpp','cmake/sycl_backend.cmake']:
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
        independent_baselines=dict(native_job_guard=SAFETY_SOURCE,U1=U1_SOURCE),bank_gate=gate,
        U1_selected_source_continuity=continuity,
        retained_decode_reference=decode,
        source_files=files,foundation=proof,external_inputs=native['external_inputs'],
        recipe=dict(configure=configure,environment=env,targets=TARGETS),
        native_validation_probe=native['native_validation_probe'],driver=guards.driver_guard(),
        passed=False,complete=False,active=False,ownership_closed=True,GPU_executed=False,
        plan=dict(unit='One inseparable long-prefill window caller/owner/layout/row-bound unit. Same math,chunk8192,same queue and immutable pre-exec flags. Retain independent U1 and native-job safety baseline. No new KV/verify/decode policy or U2coalescing.',
            source='Recover14508native sources plus the15explicit owned overlays at source_commit. Gemm class and original KV/verify/CMake backend remain14508;native-job guard is byte-identical to9849. Full map and actual compiler/link census bind the composition,not the whole live native tree.',
            qualification='Fresh engine,CPUroute/workspace,expert-bank and actual Down fixture. Matching residentGEMM fixture is retained for the named next arithmetic/resource calibration. Commit actual objects/ELFs;new committed numerical/device plan before any GPU test or model run. CPU49Klayout proof is reused only at its original scope.',
            comparison='Independent bank-only outcome remains intact. Then relocate at98K/bank128/equal window;extend at128K/bank1/fixed bank;compare combination against bank256/window10. Preserve old-path coherence control and all losses/infeasible cells. No source-bundle aggregate acceptance.',
            limits='Successful build proves no GPU allocation,safety,numerical result,PP,target,task/agent,thinking sustainedTG,decode preservation,physical262144/tails/RESTORE or ordinary teardown. Both prefill strategies remain available. Broader model routes require current decode implementation and repeated preservation gates.',
            retention='Bounded third generation:14508prefill reference and9849decode reference stay unchanged. This candidate serves the immediate long-window safety/layout/window/resource comparisons;select and retire it after that decision before another generation. CPUworkspace output is already retired;legacy provenance/configuration/cache holds remain concrete and separate.'),commands=[])
    check(row,False);save(row);print('Prepared committed coherent long-prefill composition;commit before build.',flush=True)


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
    class_headers={name:{} for name in ('include/strata/prefill/gemm.hpp',
        'include/strata/prefill/moe_accum_plan.hpp','include/strata/prefill/moe_window_routes.hpp')}
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
            for header,includers in class_headers.items():
                if p==ROOT/'source'/header:
                    require(h==row['source_files'][header],'inconsistent native class definition:'+header)
                    includers[str(obj.resolve())]=dict(header_sha256=h,source=item['file'])
    expected_classes={'CMakeFiles/strata_prefill.dir/src/prefill/'+p+'.o'
        for p in ['prefill.cpp','sycl/gemm.cpp','sycl/gemm16.cpp','sycl/qsa_score_gemm.cpp']}
    expected_classes.update({
        'CMakeFiles/sycl_moe_down_scatter_bench.dir/bench/sycl/moe_down_scatter.cpp.o',
        'CMakeFiles/sycl_native16_paired_gemm_bench.dir/bench/sycl/native16_gemm.cpp.o'})
    require(len(selected)>100 and {str(Path(p).relative_to(build)) for p in classes}==expected_classes,
        'complete fresh engine and all six production/fixture Gemm includers required')
    for header,names in {
        'include/strata/prefill/moe_accum_plan.hpp':
            {'src/prefill/prefill.cpp','tests/prefill_moe_workspace.cpp'},
        'include/strata/prefill/moe_window_routes.hpp':
            {'src/prefill/prefill.cpp','src/prefill/sycl/native16.cpp',
             'src/prefill/sycl/vllm_xe2/native16.cpp','tests/moe_accum_routes.cpp',
             'bench/sycl/moe_down_scatter.cpp'},
    }.items():
        actual={str(Path(v['source']).relative_to(ROOT/'source')) for v in class_headers[header].values()}
        require(names.issubset(actual),'missing rebuilt class producer/consumer:'+header)
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
        Gemm_includers=classes,class_definition_includers=class_headers,
        translation_units=len({x['file'] for x in selected}),fresh_objects=len(selected),
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
    print('Long-prefill build stage '+row['stage']+';no GPU/model qualification.',flush=True)


if __name__=='__main__':
    require(sys.argv[1:] in [['prepare'],['configure'],['compile']],'prepare|configure|compile')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([])
        if sys.argv[1]=='prepare':prepare()
        else:execute(sys.argv[1])
