#!/usr/bin/env python3
"""Build an isolated real-ID QSA working-set discriminator; no engine rebuild."""
import ast
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

sys.dont_write_bytecode=True
import qsa_workingset_sources as proposal
import runtime_probe as qualified_runtime
import generation_engine as gen
import measurement_guards as guards
import measurement_owner as supervision

P,B=gen.P,gen.B
ROOT=B/'qsa-working-set-v1'
RESULT=gen.A/'qsa-working-set-current.json'
FIXTURE='bench/sycl/qsa_workingset.cpp'
# Explicit new-build header delta; never relabel the original compiler manifest
# or original6e40 generation. Verify actual dependency files after compilation.
HEADER_DELTAS={
    '/usr/include/metee.h':('f063edcb00c88f3ddebc10f9b0487fe813cfb7ba71dc4b798729bed8902018a3',
        'd07e18ce9228c3fc570b186eab13bf8a3e6c04214992f0de9d53b855dc837252'),
    '/usr/include/opencl-c-base.h':('e1683c3db9cf9a95e9d02acbbfa74dcd29d4deaf4a856430f538d76985b04114',
        '673c4720f879c2f101504979edbd0216c9cb6a982ce2b62bd467b81fd53d8e33'),
    '/usr/include/opencl-c.h':('579b58f482ab0b3dfd396cab32ea3e0537bc5cf4c1067377b83fe417cbcc513b',
        'aa7e871a8f7d961aa4126a9fadc2196f822c21811946c90147dc7415fe677de9')}


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def inputs(revision):
    names=set((*proposal.NATIVE,FIXTURE,'tests/sycl/qsa_bf16_prefix.cpp',
        'tools/sycl/measurement_environment.json'))
    names.update(gen.git('ls-files','include').decode().splitlines())
    pending=[Path(__file__).resolve()];seen=set()
    while pending:
        path=pending.pop()
        if path in seen:continue
        seen.add(path);names.add(str(path.relative_to(P)))
        for node in ast.walk(ast.parse(path.read_text())):
            modules=[v.name for v in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) else []
            for module in modules:
                local=path.parent/((module or '').split('.')[0]+'.py')
                if local.is_file():pending.append(local)
    pins={}
    for name in sorted(names):
        raw=gen.git('show',revision+':'+name)
        guards.require(raw==(P/name).read_bytes(),'commit exact QSA working-set input: '+name)
        pins[name]=hashlib.sha256(raw).hexdigest()
    for name in proposal.NATIVE:
        if name == 'src/sycl/command_profile.cpp':
            guards.require(pins[name] == 'dcc7720b153ea2b5ecd3beacfd487ffd546ee67235e51c143e1075bab75b144f', 'reviewed inactive observer changed')
            continue
        guards.require(hashlib.sha256(gen.git('show',proposal.BASE+':'+name)).hexdigest()==pins[name],
            'qualified native source changed: '+name)
    return pins


def experiment():
    return dict(question='Does actual selected-QSA attention latency depend on rotating byte-identical prepacked KV copies?',
        mechanism='Original39260-token service:attention8.264100184s versus pack0.016599873s. Compiler-spill reduction alone repeatedly lost. Exact layer0 selected IDs now exist;physical traffic/saturation remain unknown.',
        arms='Prepacked selected arithmetic with1/2/4/8 rotating KV copies;original production pack+attention reference with1copy. Fixed original kernel body,DPAS/order/FP32 state/rounding/SLM/barriers/WG128/SG16/defaultGRF. Overlay only omits pack in a private diagnostic namespace,after original pack and checked drain initialized every replica.',
        geometry='Two separate shapes:all6491actual queries at positions32768..39258,width2051,live39259,identity page4;and real query0 repeated256times to fill512workgroups. Synthetic finite Q,K8,V8 andFP16scales are explicitly not model tensor contents. Repeated-query work is identical across all copy-count arms and is a controlled counterfactual,not a production replay.',
        qualification='Before timing:source overlay reversal,exact capture/model/native/runtime/SDK input checks;3small traced/validated smoke processes;3component processes cover real16query masks,8exact prepared replicas,alias/guard/input immutability and physical262144 final-address cases. Check actual selected and prepacked compiler resources before clean timing. No environment mutation or unsupported cache flush.',
        performance='3independent clean processes,6rotated rounds,5arms,2shapes=180individual samples. Order=(process+round+offset)%5;arm0..3rotate1/2/4/8,arm4originalpack. Equal2Rwarmup calls;64calls/batch repeated256,2calls/batch full6491. Every batch>=100ms and>=1000times128pair positive minimum timer floor. No adaptive retries. Record all ratios/spread. Directionally consistent R8/R1>=1.05 in all3processes supports sensitivity for that shape;otherwise effect small or inconclusive. This diagnostic gate never advances a model/inference trial.',
        provenance='Main-authored ownMIT6e40 overlay and fixture,using original project buffer/owner/build scaffolds. No XeStrata application source. Original engine13assets/decode1256/production dispatch stay unchanged;installed SDK paths do not imply application-source adoption.',
        memory='Each full39260-cell packed KV replica uses80404480bytes plus314080bad-row bytes. Eight replicas~646MB;Q/outputs/IDs/raw pools bring the real6491case below2GiB of device allocations. Every allocation is admitted by Runtime,guarded and below the device single-allocation limit. Physical262144 component uses only2replicas. Host RSS cap8GiB;no model capacity claim.',
        limits='Source-derived Mode0 identity map,not a captured device table. No real Q/K/V. Counter-free latency sensitivity is not measured physical traffic/cache hits/tier/bandwidth/saturation or complete critical-path attribution. Timing includes unchanged host admission and checked drain;preparation/readback/checks excluded. Full-wrapper arm includes prefix refresh;do not subtract unequal wrappers as pure pack time. First use outside steady-state samples. No sustained decode/quality/agent/full-context adoption.',
        retention='Keep selected engine+12fixtures and one imminent component generation. Current capture holds exactly one53MiB initialized fixture for this comparison. Retire component/generated/IR/logs and capture when its named consumer closes,after useful180samples/source/recipe/hashes/outcomes are committed. Separate144artifact/46root/14legacy provenance holds remain explicit,not another comparison allowance.')


def qualified_baseline():
    row = qualified_runtime.validation.committed(qualified_runtime.RESULT)
    guards.require(row['stage']=='device-checked' and row['passed'] and row['complete'] and
        row['ownership_closed'] and not row['active'] and not row['fresh_faults'], 'closed current runtime qualification required')
    full = qualified_runtime.current(row)
    return full, dict(original_build=row['original_build'],runtime_record_sha256=digest(qualified_runtime.RESULT),
        runtime_files=row['runtime_files'],preload=row['preload'])


def compiled_inputs(record):
    rows=json.loads(Path(record['compiler_inputs']['path']).read_text())
    allowed={Path(r['path']).resolve():r['sha256'] for r in rows}
    allowed.update({(P/name).resolve():sha for name,sha in record['input_sha256'].items()})
    allowed.update({Path(name).resolve():sha for name,sha in record['generated'].items()})
    actual={}
    for index in range(len(record['sources'])):
        text=(ROOT/f'input-{index}.d').read_text().replace('\\\n',' ')
        target,dependencies=text.split(':',1)
        guards.require(Path(target).resolve()==(ROOT/f'input-{index}.o').resolve(),'unexpected dependency target')
        for value in shlex.split(dependencies):
            path=Path(value).resolve();sha=digest(path)
            guards.require(allowed.get(path)==sha,'unfrozen actual compiler input: '+str(path))
            actual[str(path)]=sha
    guards.require(len(actual)>100,'incomplete compiler dependencies')
    return dict(files=len(actual),source_sha256=actual,
        changed_headers_consumed=sorted(set(actual)&set(HEADER_DELTAS)))


def admission(record,binary=False):
    guards.require(inputs(record['source_commit'])==record['input_sha256'] and record['plan']==experiment(),
        'QSA working-set source or prospective plan changed')
    full,proof=qualified_baseline()
    guards.require(full['binary']==record['retained_engine'] and proof==record['native_qualification'],
        'original engine/13artifact qualification changed')
    for name,body in proposal.generated(P).items():
        path=ROOT/name
        guards.require(path.read_bytes()==body and digest(path)==record['generated'][str(path)],'generated source changed')
    manifest=record['compiler_inputs']
    guards.require(digest(manifest['path'])==manifest['sha256'],'compiler manifest changed')
    rows=json.loads(Path(manifest['path']).read_text())
    guards.require(len(rows)==manifest['files'],'compiler input count changed')
    for row in rows:guards.require(digest(row['path'])==row['sha256'],'compiler/dependency changed: '+row['path'])
    fixture=record['selected_fixture']
    guards.require(digest(fixture['path'])==fixture['sha256'] and Path(fixture['path']).stat().st_size==fixture['bytes'],'source-bound selected fixture changed')
    if binary:
        for artifact in record['binaries'].values():
            guards.require(digest(artifact['path'])==artifact['sha256'],'QSA working-set fixture bytes changed')


def prepare():
    guards.require(not RESULT.exists() and not ROOT.exists(),'fresh single-use QSA working-set build only;inspect existing owner')
    revision=gen.git('rev-parse','HEAD').decode().strip();pins=inputs(revision)
    full,proof=qualified_baseline()
    prof=gen.committed(gen.A/'cpu-profiler-current.json')
    replay=prof['native_model_aggregate_v2_reanalysis']
    guards.require(replay['passed'] and replay['complete'] and replay['original_failure_preserved'], 'closed attribution replay required')
    origin=prof['unitrace_indexed_graph']['compiler_inputs']
    guards.require(digest(origin['path'])==origin['sha256'],'original SDK inventory changed')
    rows=json.loads(Path(origin['path']).read_text())
    guards.require(len(rows)==origin['files'],'original SDK input count differs')
    paths={Path(row['path']) for row in rows}
    compiler=full['compiler'];guards.require(digest(compiler['path'])==compiler['sha256'],'compiler changed')
    for name in ('crtbeginS.o','crtendS.o','libstdc++.so','libgcc_s.so'):
        path=Path(subprocess.check_output([compiler['path'],'-print-file-name='+name],text=True,timeout=15).strip()).resolve()
        guards.require(path.is_file(),'link input absent: '+name);paths.add(path)
    changed={}
    for row in rows:
        actual=digest(row['path'])
        if actual!=row['sha256']:
            guards.require(HEADER_DELTAS.get(row['path'])==(row['sha256'],actual),
                'additional compiler asset changed: '+row['path'])
            changed[row['path']]=dict(original_sha256=row['sha256'],new_build_sha256=actual)
    guards.require(set(changed)==set(HEADER_DELTAS),'declared header delta is not the actual delta')
    ROOT.mkdir(mode=0o700);(ROOT/'tmp').mkdir()
    for name,body in proposal.generated(P).items():(ROOT/name).write_bytes(body)
    manifest=ROOT/'compiler-inputs.json'
    write(manifest,[dict(path=str(path),sha256=digest(path)) for path in sorted(paths)])
    generated={str(ROOT/name):digest(ROOT/name) for name in proposal.generated(P)}
    install=Path(compiler['path']).parents[1]
    flags=['-std=c++20','-O3','-DNDEBUG','-g0','-fsycl','-fsycl-id-queries-range=size_t',
        '-fno-fast-math','-ffp-contract=off','-DSTRATA_ENABLE_SYCL=1',
        '-I'+str(P/'include'),'-I'+str(P/'include/strata/sycl/compat'),'-I'+str(ROOT)]
    sources=[P/name for name in proposal.NATIVE]+[ROOT/'prepacked.cpp',P/FIXTURE]
    objects=[ROOT/f'input-{i}.o' for i in range(len(sources))]
    commands=[[compiler['path'],*flags,'-MD','-MF',str(obj.with_suffix('.d')),
        '-c',str(src),'-o',str(obj)] for src,obj in zip(sources,objects)]
    commands.append([compiler['path'],*flags,*map(str,objects),'-pthread','-ldl','-o',str(ROOT/'fixture')])
    for name,path in (('original',P/proposal.PREFIX),('prepacked',ROOT/'prepacked.cpp')):
        commands.append([compiler['path'],*flags,'-fsycl-device-only','-S','-emit-llvm',str(path),'-o',str(ROOT/(name+'.ll'))])
    capture=qualified_runtime.validation.committed(gen.A/'prefill-resources-current.json',('event_reconciliation','selected_qsa_capture_v1'))
    guards.require(capture['passed'] and capture['complete'] and capture['ownership_closed'] and not capture['active'] and not capture['fresh_faults'],'closed actual selected-ID capture required')
    fixture=capture['analysis']['fixture']
    guards.require(digest(fixture['path'])==fixture['sha256'] and fixture['header']==[0,32768,6491,2051],'selected-ID identity differs')
    legacy=json.loads((gen.A/'native-gdn-engine-build.json').read_text())['legacy_provenance_exception']['additional_old_product_review']
    record=dict(stage='prepared',source_commit=revision,input_sha256=pins,plan=experiment(),sources=list(map(str,sources)),selected_fixture=fixture,
        declared_compiler_header_delta=changed,
        retained_engine=full['binary'],native_qualification=proof,generated=generated,compiler=compiler,
        compiler_inputs=dict(path=str(manifest),files=len(paths),sha256=digest(manifest)),commands=commands,
        build_environment=dict(PATH=str(install/'bin')+':/usr/bin:/bin',LANG='C',LC_ALL='C',TMPDIR=str(ROOT/'tmp'),
            LD_LIBRARY_PATH=str(install/'lib'),PYTHONDONTWRITEBYTECODE='1'),
        budgets=dict(total_wall_s=1800,per_command_wall_s=420,cpu_s=900,rss_bytes=12<<30,
            text_bytes=8<<20,file_bytes=512<<20,total_bytes=1<<30),
        legacy_provenance_holds=dict(result='native-gdn-engine-build.json',review_sha256=hashlib.sha256(json.dumps(legacy,sort_keys=True).encode()).hexdigest(),
            next_action='Retire the15source-recovered QSA fixtures after bounded consumer/hash checks;carry the remaining distinct source/recipe holds in the maintained native build result. Recheck actual paths before another generation.',full_final_input_proof=False),
        active=False,ownership_closed=True,passed=False,complete=False,GPU_executed=False,model_adoption_qualified=False)
    admission(record);write(ROOT/'plan.json',record);write(RESULT,record)
    print('Prepared isolated QSA working-set build;commit exact source/generated plan before build.',flush=True)


def build():
    record=gen.committed(RESULT)
    guards.require(record['stage']=='prepared' and record==json.loads((ROOT/'plan.json').read_text())
        and not (ROOT/'build-record.json').exists(),'inspect existing build;no retry')
    admission(record);owner=None
    def persist():
        record['active']=bool(owner and owner.active is not None)
        write(ROOT/'build-record.json',dict(record,owned_commands=owner.commands if owner else []))
    owner=supervision.Owner(ROOT/'build-owned',persist);deadline=time.monotonic()+record['budgets']['total_wall_s']
    record['stage']='building';persist()
    try:
        b=record['budgets']
        for index,command in enumerate(record['commands']):
            remaining=deadline-time.monotonic();guards.require(remaining>1,'whole build deadline')
            print(f'Compiling/linking isolated QSA working-set input {index+1}/{len(record["commands"])}',flush=True)
            entry,_,_=owner.run('build-'+str(index),command,record['build_environment'],P,
                wall=min(b['per_command_wall_s'],remaining),cpu=b['cpu_s'],rss_cap=b['rss_bytes'],
                text_cap=b['text_bytes'],file_cap=b['file_bytes'],total_cap=b['total_bytes'])
            guards.require(supervision.closed(entry) and entry['exit_code']==0,'QSA working-set compile/link failed')
        admission(record)
        record['actual_compiled_inputs']=compiled_inputs(record)
        record.update(stage='built',passed=True,complete=True,binaries={name:dict(path=str(ROOT/name),
            sha256=digest(ROOT/name),bytes=(ROOT/name).stat().st_size) for name in ('fixture',)})
        record['device_IR']={name:dict(path=str(ROOT/(name+'.ll')),sha256=digest(ROOT/(name+'.ll')),
            bytes=(ROOT/(name+'.ll')).stat().st_size) for name in ('original','prepacked')}
    except BaseException as error:record.update(stage='failed',error=repr(error))
    finally:
        record['ownership_closed']=owner.active is None and all(supervision.closed(c) for c in owner.commands)
        persist();write(RESULT,record)
    print(json.dumps({k:record.get(k) for k in ('stage','passed','complete','ownership_closed','error')}),flush=True)
    return 0 if record['passed'] else 1


if __name__=='__main__':
    guards.require(len(sys.argv)==2 and sys.argv[1] in ('prepare','build'),'use prepare|build')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        sys.exit(prepare() if sys.argv[1]=='prepare' else build())
