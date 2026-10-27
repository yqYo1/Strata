#!/usr/bin/env python3
"""Build isolated same-math QSA barrier controls; no engine rebuild."""
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
sys.path.insert(0,str(Path(__file__).resolve().parent))
import qsa_barrier_sources as proposal
import qsa_barrier_checks as oracle
import runtime_probe as qualified_runtime
import generation_engine as gen
import measurement_guards as guards
import measurement_owner as supervision

P,B=gen.P,gen.B
ROOT=B/'qsa-barrier-v1'
RESULT=gen.A/'qsa-barrier-current.json'
FIXTURE='bench/sycl/qsa_barrier.cpp'
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


def generated_sources():
    return {**proposal.generated(P), **oracle.generated(P)}


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
        guards.require(raw==(P/name).read_bytes(),'commit exact QSA barrier input: '+name)
        pins[name]=hashlib.sha256(raw).hexdigest()
    for name in proposal.NATIVE:
        if name == 'src/sycl/command_profile.cpp':
            guards.require(pins[name] == 'dcc7720b153ea2b5ecd3beacfd487ffd546ee67235e51c143e1075bab75b144f', 'reviewed inactive observer changed')
            continue
        guards.require(hashlib.sha256(gen.git('show',proposal.BASE+':'+name)).hexdigest()==pins[name],
            'qualified native source changed: '+name)
    guards.require(hashlib.sha256(gen.git('show',proposal.BASE+':'+oracle.ORACLE)).hexdigest()==pins[oracle.ORACLE],
        'independent original oracle changed')
    return pins


def experiment():
    return dict(question='Can merging the end-of-tile barrier with the next existing row-publication barrier reduce selected-QSA GPU latency without changing arithmetic?',
        hardware='Closed actual-mask calibration:6491query median wall124.1591ms/service123.9911ms. GPU-local kernel service dominates that fixture. The inter-call1to8copy screen failed;physical bandwidth,cache tier,compute or synchronization saturation remain unqualified.',
        arms='Unmodified original ordered-exp production wrapper;private unchanged control with direct-event retention;private candidate with identical arithmetic/layout/event retention and one loop-end barrier moved after the loop. All arms retain full prefix pack. Selected original6e40/12fixtures and1256decode placement unchanged.',
        source_contract='Source reversal and original6e40 native/oracle identities required. Candidate has4barriers per cell iteration plusone final fence instead of5per iteration. Row publication touches rows/private flags only;next existing barrier precedes K overwriting prior V. Preserve collective convergence and all FP32/BF16 operations. Independent Orca Luna contract review plus native Luna/max curation required before GPU execution.',
        qualification='Commit native/controller/oracle/generated inputs and recipe first. Three fresh traced smoke processes and three component processes independently select original/control/merged for negative admission while every supported invocation bit-compares all3arms. Independent host oracle,zero/invalid/tail widths,duplicates,aliased/missing pages,first/later K/V diagnostics,NaN,input/guard checks,2-byte alignment and physical262144 final-address component;not full-model fill. Actual3query smoke fixture separately checks strong direct-event clocks before resource/timing work.',
        resources='One small profiled actual3query fixture:conserve full table,require original/control/merged attention2calls each,SIMD16/WG128/18arguments,original-control identical descriptors. Require merged same rawSLM/matrix shape/default128GRF;record its spill/private descriptors separately because barrier placement can change liveness. Source dynamic local payload20036bytes fixed. Different merged spill alone is neither success nor failure;the experiment includes scheduling/liveness effects,not a pure barrier instruction-cost claim.',
        performance='3independent processes,6rotated rounds,3arms,2actual-mask shapes512/6491=108samples. Order=(process+round+offset)%3;2warmup calls per arm.32calls/batch for512,2for6491. Each batch>=100ms and>=1000times128-pair positive minimum timer floor. Original wall only;no invented production event. Control/merged strongly hold actual submission events through drain,read after wall end;positive ordered nonoverlapping service within wall+1ms. Each shape requires control/original wall within1% in all3processes and merged/control wall AND service<=0.99 in all3processes. No adaptive retry/threshold change. A failed gate does not advance to model.',
        model='Only after component numerical/resource/performance gates pass,commit a separate phase-specific production hook and matched official bench.sh comparison using>=32Kactual input,thinking enabled,3independent processes with first-use/subsequent samples separated. Preserve decode/config/placement and verify repeated decode/handoff. Goal1000prefill/70decode,agent/long strong-weak generation/quality/physical262144fill/RESTORE/ordinary teardown remain independent requirements.',
        budgets='One imminent component generation beside the retained known-good continuation engine. Oracle physical262144case keeps full source pools but bounded packed prefix;single allocations remain below hardware admission. Perjob wall300s except timing600s,RSS8GiB,rawtext16MiB,total192MiB. No GPU resets,service changes,driver changes or process-environment mutation.',
        provenance='Main-authored ownMIT6e40/2d scaffold and immutable original oracle reuse;no XeStrata application source. Pin compiler/runtime,include headers,helpers,workload,source and final executable separately.',
        retention='Keep useful individual samples,outcomes/source/recipe and one53.25MBactual-ID fixture. Retire closed component binaries,IR,success logs and one-use controllers after exact source recovery. Retire fixture at this named comparison closure or supersession. Separate legacy source/recipe holds remain unresolved;no global cleanup claim.')


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
        'QSA barrier source or prospective plan changed')
    full,proof=qualified_baseline()
    guards.require(full['binary']==record['retained_engine'] and proof==record['native_qualification'],
        'original engine/13artifact qualification changed')
    for name,body in generated_sources().items():
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
            guards.require(digest(artifact['path'])==artifact['sha256'],'QSA barrier fixture bytes changed')


def prepare():
    guards.require(not RESULT.exists() and not ROOT.exists(),'fresh barrier generation only;inspect any existing result/job before retry')
    previous=gen.committed(gen.A/'qsa-working-set-current.json')
    guards.require(previous['stage']=='closed-working-set-diagnostic' and previous['retirement']['status']=='complete' and
        all(not (B/name).exists() for name in ('qsa-working-set-v1','qsa-working-set-v2','qsa-working-set-v3')),
        'closed prior component and actual retirement required')
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
    for name,body in generated_sources().items():(ROOT/name).write_bytes(body)
    manifest=ROOT/'compiler-inputs.json'
    write(manifest,[dict(path=str(path),sha256=digest(path)) for path in sorted(paths)])
    generated={str(ROOT/name):digest(ROOT/name) for name in generated_sources()}
    install=Path(compiler['path']).parents[1]
    flags=['-std=c++20','-O3','-DNDEBUG','-g0','-fsycl','-fsycl-id-queries-range=size_t',
        '-fno-fast-math','-ffp-contract=off','-DSTRATA_ENABLE_SYCL=1','-DSTRATA_QSA_PACKED_GUARD=17',
        '-I'+str(P/'include'),'-I'+str(P/'include/strata/sycl/compat'),'-I'+str(ROOT)]
    sources=[P/name for name in proposal.NATIVE]+[ROOT/'barrier_control.cpp',ROOT/'barrier_merged.cpp',ROOT/'barrier-checks.cpp',P/FIXTURE]
    objects=[ROOT/f'input-{i}.o' for i in range(len(sources))]
    commands=[[compiler['path'],*flags,'-MD','-MF',str(obj.with_suffix('.d')),
        '-c',str(src),'-o',str(obj)] for src,obj in zip(sources,objects)]
    for name,tail in (('oracle',objects[7]),('fixture',objects[8])):
        commands.append([compiler['path'],*flags,*map(str,objects[:7]),str(tail),'-pthread','-ldl','-o',str(ROOT/name)])
    for name,path in (('original',P/proposal.PREFIX),('control',ROOT/'barrier_control.cpp'),('merged',ROOT/'barrier_merged.cpp')):
        commands.append([compiler['path'],*flags,'-fsycl-device-only','-S','-emit-llvm',str(path),'-o',str(ROOT/(name+'.ll'))])
    capture=qualified_runtime.validation.committed(gen.A/'prefill-resources-current.json',('event_reconciliation','selected_qsa_capture_v1'))
    guards.require(capture['passed'] and capture['complete'] and capture['ownership_closed'] and not capture['active'] and not capture['fresh_faults'],'closed actual selected-ID capture required')
    fixture=capture['analysis']['fixture']
    guards.require(digest(fixture['path'])==fixture['sha256'] and fixture['header']==[0,32768,6491,2051],'selected-ID identity differs')
    record=dict(stage='prepared',source_commit=revision,input_sha256=pins,plan=experiment(),sources=list(map(str,sources)),selected_fixture=fixture,
        previous_diagnostic=dict(result='qsa-working-set-current.json',closed_result_commit='03421d914',generation_retired=True),
        declared_compiler_header_delta=changed,
        retained_engine=full['binary'],native_qualification=proof,generated=generated,compiler=compiler,
        compiler_inputs=dict(path=str(manifest),files=len(paths),sha256=digest(manifest)),commands=commands,
        build_environment=dict(PATH=str(install/'bin')+':/usr/bin:/bin',LANG='C',LC_ALL='C',TMPDIR=str(ROOT/'tmp'),
            LD_LIBRARY_PATH=str(install/'lib'),PYTHONDONTWRITEBYTECODE='1'),
        budgets=dict(total_wall_s=1800,per_command_wall_s=420,cpu_s=900,rss_bytes=12<<30,
            text_bytes=8<<20,file_bytes=512<<20,total_bytes=1<<30),
        legacy_provenance_holds=dict(result='native-gdn-engine-build.json',full_final_input_proof=False,
            next_action='Carry separate missing original-source/recipe holds through their existing bounded review;closed QSA generations are gone. No new comparison uses unqualified legacy artifacts.'),
        active=False,ownership_closed=True,passed=False,complete=False,GPU_executed=False,model_adoption_qualified=False)
    admission(record);write(ROOT/'plan.json',record);write(RESULT,record)
    print('Prepared isolated QSA barrier build;commit exact source/generated plan before build.',flush=True)


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
            print(f'Compiling/linking isolated QSA barrier input {index+1}/{len(record["commands"])}',flush=True)
            entry,_,_=owner.run('build-'+str(index),command,record['build_environment'],P,
                wall=min(b['per_command_wall_s'],remaining),cpu=b['cpu_s'],rss_cap=b['rss_bytes'],
                text_cap=b['text_bytes'],file_cap=b['file_bytes'],total_cap=b['total_bytes'])
            guards.require(supervision.closed(entry) and entry['exit_code']==0,'QSA barrier compile/link failed')
        admission(record)
        record['actual_compiled_inputs']=compiled_inputs(record)
        record.update(stage='built',passed=True,complete=True,binaries={name:dict(path=str(ROOT/name),
            sha256=digest(ROOT/name),bytes=(ROOT/name).stat().st_size) for name in ('oracle','fixture')})
        record['device_IR']={name:dict(path=str(ROOT/(name+'.ll')),sha256=digest(ROOT/(name+'.ll')),
            bytes=(ROOT/(name+'.ll')).stat().st_size) for name in ('original','control','merged')}
    except BaseException as error:
        record.update(stage='failed',passed=False,complete=False,error=repr(error))
    finally:
        record['ownership_closed']=owner.active is None and all(supervision.closed(c) for c in owner.commands)
        if not record['ownership_closed']:
            record.update(stage='failed',passed=False,complete=False,error='build ownership not closed')
        persist();write(RESULT,record)
    print(json.dumps({k:record.get(k) for k in ('stage','passed','complete','ownership_closed','error')}),flush=True)
    return 0 if record['passed'] else 1


if __name__=='__main__':
    guards.require(len(sys.argv)==2 and sys.argv[1] in ('prepare','build'),'use prepare|build')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        sys.exit(prepare() if sys.argv[1]=='prepare' else build())
