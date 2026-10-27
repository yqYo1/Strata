#!/usr/bin/env python3
"""Build the independently authored subgroup-K component; never execute GPU work.

Own MIT build scaffold recovered fromcfabc1ee2, not XeStrata application code.
"""
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
import qsa_k_staging_sources as proposal
import runtime_probe as qualified_runtime
import generation_engine as gen
import measurement_guards as guards
import measurement_owner as supervision

P,B=gen.P,gen.B
ROOT=B/'qsa-k-staging-packed32-v1'
RESULT=gen.A/'qsa-k-staging-current.json'
FIXTURE='bench/sycl/qsa_k_staging.cpp'
ADMISSION='bench/sycl/qsa_k_staging_admission.hpp'
ADAPTER='bench/sycl/qsa_k_staging_oracle_adapter.cpp'
ORACLE='tests/sycl/qsa_bf16_prefix.cpp'
NATIVE=('src/sycl/runtime.cpp','src/sycl/compat.cpp','src/sycl/command_profile.cpp',
        'src/kernels/sycl/prompt_attention.cpp','src/kernels/sycl/qsa_bf16_prefix.cpp')
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
    return proposal.generated(P)


def source_contract():
    scope=qualified_runtime.validation.committed(gen.A/'research-index.json',
        ('qsa_packed_k_staging_source_contract_20261028',))
    guards.require(scope['status']=='reviewed_source_contract_no_device_qualification' and
        scope['worker_closed'] and scope['review']['status']=='completed' and
        scope['review']['model']=='gpt-6-luna' and scope['review']['reasoning_effort']=='max',
        'closed version-matched source audit and native curation required')
    for name,sha in scope['source_snapshot'].items():
        raw=gen.git('show',scope['source_commit']+':'+name)
        guards.require(hashlib.sha256(raw).hexdigest()==sha and digest(P/name)==sha,
            'source-reviewed packed K input changed: '+name)
    guards.require(digest(P/scope['prerequisites_evidence']['source']['path'])==
        scope['prerequisites_evidence']['source']['sha256'], 'source-reviewed K generator changed')
    contract=scope['qualified_source_contract']
    for path,sha in contract['headers'].items():
        guards.require(digest(path)==sha,'reviewed SDK header changed: '+path)
    guards.require(digest(contract['compiler']['path'])==contract['compiler']['sha256'],
        'source-reviewed compiler changed')
    return contract


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def inputs(revision):
    names=set((*NATIVE,FIXTURE,ADMISSION,ADAPTER,ORACLE,
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
        guards.require(raw==(P/name).read_bytes(),'commit exact QSA K-staging input: '+name)
        pins[name]=hashlib.sha256(raw).hexdigest()
    for name in NATIVE:
        if name == 'src/sycl/command_profile.cpp':
            guards.require(pins[name] == 'dcc7720b153ea2b5ecd3beacfd487ffd546ee67235e51c143e1075bab75b144f', 'reviewed inactive observer changed')
            continue
        guards.require(hashlib.sha256(gen.git('show',proposal.BASE+':'+name)).hexdigest()==pins[name],
            'qualified native source changed: '+name)
    return pins


def experiment():
    return dict(
        question='Does a packed-u32 transpose make four-cell-by16dim K staging useful after the eight-u16 variant lost11.8..14.8percent wall time? This controller only compiles;no GPU execution.',
        mechanism='Keep the rejected variant\'s mapping/global loads/typed local stores,but pack each pair of needed uint16 values into one uint32 and use uniform XOR4/XOR8 exchanges. Same8192BF16 values,K32/V256,online32cell order and matrix math. Source collective instances drop1024to256 per WG/32selected cells across all256Kdimensions. Packing/select cost and real lowering may erase the benefit;no bank/physical-traffic/occupancy assumption.',
        arms='OwnMIT6e40 original;private identical-control with direct attention event;private packed-u32 K staging. No production or decode source/placement change.',
        source_contract='Own cfabc1ee2 harness/build/property scaffold and own6e40prefix;own96aa/e1 staging predecessor,not XeStrata application code. Require36-mask symbolic plus216numeric packing cases and independent version-matched source/harness audit/native Luna-max curation before build. No device qualification from proof.',
        compile='Ten object compiles,two links,three device LLVM emissions. Link the actual-ID comparison and separately the existing independent numerical/guard fixture through a candidate-only adapter. Verify dependency files against frozen project/generated/compiler manifests;commit prepared commands and hashes before building.',
        admission='Separate committed property-only plan queries all three exact compiled kernels,WG128/SG16/18args,qualified loader and declared20036source local bytes before kernel submission. Traced smoke verifies actual local arguments and runtime.',
        numerical='Every K-staging output bit must match unmodified original32 and private32 on finite actual-ID inputs,repeats,guards,immutable inputs and packed buffers. Independent scalar spots remain. The separate existing oracle covers masked/tail/missing/nonfinite inputs,align2scratch,alias/no-op/admission and numerical bounds against prefix33. Three translated queries reach262144final slots but do not qualify full-context model fill.',
        performance='Only after admission/traced smoke/boundary gates,freeze three-process512/6491actual-ID comparison:six rotated rounds,64/4calls per batch,>=100ms and>=1000clock floor. Private-control/original wall0.99..1.01;candidate/control wall and attention-service<=0.99 in every process/shape. Failure prohibits a model trial;no adoption from component success.',
        budgets='One additional component beside selected6e40engine/12fixtures. Build1800s/per-command420s/RSS12GiB/text8MiB/totaloutput1GiB. Current qualified runtime safety and decode1256placement remain.',
        retention='Immediate exact-kernel admission,then traced correctness and actual-ID comparison. Retire successful raw logs/IR and rejected build after exact input/recipe recovery. The single53MB initialized fixture serves only this comparison;legacy provenance/numerical holds stay separate.')


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
        'QSA K-staging source or prospective plan changed')
    guards.require(source_contract()==record['source_contract'],'reviewed source contract changed')
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
            guards.require(digest(artifact['path'])==artifact['sha256'],'QSA K-staging fixture bytes changed')


def prepare():
    guards.require(not RESULT.exists() and not ROOT.exists(),'fresh K-staging generation only;inspect any existing result/job before retry')
    previous=gen.committed(gen.A/'fused-moe-service-current.json')['qsa_k_staging_v1_closed']
    guards.require(previous['retirement']['complete'] and not previous['advance_to_model']
        and not previous['model_adopted'], 'previous K-staging comparison must be closed and retired')
    contract=source_contract()
    revision=gen.git('rev-parse','HEAD').decode().strip();pins=inputs(revision)
    fixture=qualified_runtime.validation.committed(gen.A/'fused-moe-service-current.json',
        ('qsa_k_staging_preflight','fixture'))
    guards.require(digest(fixture['path'])==fixture['sha256'] and
        fixture['header']==[0,32768,6491,2051],'selected-ID fixture identity differs')
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
    sources=[P/name for name in NATIVE]+[ROOT/'kstaging_control.cpp',ROOT/'kstaging_candidate.cpp',
        P/FIXTURE,P/ADAPTER,P/ORACLE]
    objects=[ROOT/f'input-{i}.o' for i in range(len(sources))]
    commands=[]
    for src,obj in zip(sources,objects):
        selection=(['-Dqsa_bf16_prefix_batch=qsa_k_staging_test_batch',
            '-DSTRATA_QSA_COMPARE_PREFIX33=1'] if src==P/ORACLE else [])
        commands.append([compiler['path'],*flags,*selection,'-MD','-MF',str(obj.with_suffix('.d')),
            '-c',str(src),'-o',str(obj)])
    commands.append([compiler['path'],*flags,*map(str,objects[:8]),'-pthread','-ldl','-o',str(ROOT/'fixture')])
    commands.append([compiler['path'],*flags,*map(str,objects[:7]+objects[8:]),'-pthread','-ldl',
        '-o',str(ROOT/'numerical')])
    for name,path in (('original',P/proposal.PREFIX),('control',ROOT/'kstaging_control.cpp'),('candidate',ROOT/'kstaging_candidate.cpp')):
        commands.append([compiler['path'],*flags,'-fsycl-device-only','-S','-emit-llvm',str(path),'-o',str(ROOT/(name+'.ll'))])
    record=dict(stage='prepared',source_commit=revision,input_sha256=pins,source_contract=contract,plan=experiment(),sources=list(map(str,sources)),selected_fixture=fixture,
        previous_diagnostic=dict(result='qsa-slm-pitch-current.json',closed_result_commit='cfabc1ee2',generation_retired=True),
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
    print('Prepared isolated QSA K-staging build;commit exact source/generated plan before build.',flush=True)


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
            print(f'Compiling/linking isolated QSA K-staging input {index+1}/{len(record["commands"])}',flush=True)
            entry,_,_=owner.run('build-'+str(index),command,record['build_environment'],P,
                wall=min(b['per_command_wall_s'],remaining),cpu=b['cpu_s'],rss_cap=b['rss_bytes'],
                text_cap=b['text_bytes'],file_cap=b['file_bytes'],total_cap=b['total_bytes'])
            guards.require(supervision.closed(entry) and entry['exit_code']==0,'QSA K-staging compile/link failed')
        admission(record)
        record['actual_compiled_inputs']=compiled_inputs(record)
        record.update(stage='built',passed=True,complete=True,binaries={name:dict(path=str(ROOT/name),
            sha256=digest(ROOT/name),bytes=(ROOT/name).stat().st_size) for name in ('fixture','numerical')})
        record['device_IR']={name:dict(path=str(ROOT/(name+'.ll')),sha256=digest(ROOT/(name+'.ll')),
            bytes=(ROOT/(name+'.ll')).stat().st_size) for name in ('original','control','candidate')}
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
