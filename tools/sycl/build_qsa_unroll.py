#!/usr/bin/env python3
"""Build an isolated current-QSA loop-hint discriminator; no engine rebuild."""
import ast
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode=True
import qsa_unroll_sources as proposal
import resource_engine as resource
import generation_engine as gen
import measurement_guards as guards
import measurement_owner as supervision

P,B=gen.P,gen.B
ROOT=B/'qsa-qcache-unroll-v1'
RESULT=gen.A/'qsa-unroll-current.json'
FIXTURE='bench/sycl/qsa_unroll_numerical.cpp'


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def inputs(revision):
    names=set((*proposal.NATIVE,proposal.BENCH,FIXTURE,'tests/sycl/qsa_bf16_prefix.cpp',
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
        guards.require(raw==(P/name).read_bytes(),'commit exact static-unroll input: '+name)
        pins[name]=hashlib.sha256(raw).hexdigest()
    for name in proposal.NATIVE:
        guards.require(hashlib.sha256(gen.git('show',proposal.BASE+':'+name)).hexdigest()==pins[name],
            'qualified native source changed: '+name)
    return pins


def experiment():
    return dict(question='Does explicit full unrolling of the16 cached-Q fragment loads and16 dependent QK matrix steps improve the actual current ordered-exp implementation?',
        mechanism='Current QSA device service and GRF128/spill12480 metadata are measured; changing GRF mode alone slowed6.7%. Hardware saturation remains unknown because counter setup is refused. This controlled compiler-sensitivity test targets dynamic fragment indexing/loop scheduling without changing requested memory/PCIe bytes or math. No claimed hot-spill,occupancy or integer/compute saturation.',
        arms='Unmodified production ordered-exp; isolated namespace with identical body(control); same namespace form with only two clang unroll(full) hints(unroll). Dim256 means16 cached fragments. Default GRF policy,subgroup16,128 workitems,SLM shape,query/cache lifetime,ascending QK/softmax order,ordinary exp2,FP32 accumulation andBF16 boundaries stay fixed. Reversing namespace/hints recovers every original body byte.',
        provenance='Main-authored overlay reuses own MIT GRF discriminator source b4beb8d53 and check a2fbf491f; no XeStrata implementation. Intel oneAPI GPU optimization guide2024.2 and experimental SYCL matrix extension supply contract/design context only. No new production dispatch,engine build or decode-source/configuration/1256slot placement change.',
        qualification='Three traced/parameter-validated small numerical processes followed by original independent tensor oracle,exact ordered-output bits,guards,input immutability,alias/domain and physical262144 component cases in each arm. Normal ownership closure,no freshfault and actual compiled GRF/resource observations required. Inspect compiler IR for the two loop hints and remaining dynamic indexing when available; source hint alone does not prove changed codegen. Component tests do not qualify full-context model/session/RESTORE.',
        performance='Only after all component gates:3independent clean processes,6rotated rounds,3arms,3artificial shapes(32768/8192/2051 gather;117488/8192/2051 gather andclustered),all162individual latencies. Each batch>=100us and>=1000timesminimum positive128-pair clock floor. Unroll/default ANDunroll/control geomeans<=.95,no shape>1.05,both117488patterns improve in everyprocess. Preserve original outcomes;no rule change or model trial after failed gate.',
        limits='Timed complete wrapper includes admission,prefix preparation and checked drain. Synthetic locality is not production distribution; timing and allocation descriptors are not physical traffic or saturation. Even a passing component needs a separately committed long-input/agent/thinking-friendly-andweak-generation plan preserving exactdecodehandoff/configuration/placement and repeated speed before adoption.',
        retention='Retain original13engine/fixture assets unchanged plus one immediate two-executable component generation. Keep source,recipe,all clean samples,correctness/fault evidence and compact decision. Retire consumed build/API logs,generated copies,objects and auxiliary executables after consumer closes;legacy provenance holds remain explicit.')


def admission(record,binary=False):
    guards.require(inputs(record['source_commit'])==record['input_sha256'] and record['plan']==experiment(),
        'static-unroll source or prospective plan changed')
    full,proof,_=resource.qualification()
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
    if binary:
        for artifact in record['binaries'].values():
            guards.require(digest(artifact['path'])==artifact['sha256'],'static-unroll fixture bytes changed')


def prepare():
    guards.require(not RESULT.exists() and not ROOT.exists(),'fresh single-use static-unroll build only;inspect existing owner')
    revision=gen.git('rev-parse','HEAD').decode().strip();pins=inputs(revision)
    full,proof,_=resource.qualification()
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
    for row in rows:guards.require(digest(row['path'])==row['sha256'],'original SDK asset changed')
    ROOT.mkdir(mode=0o700);(ROOT/'tmp').mkdir()
    for name,body in proposal.generated(P).items():(ROOT/name).write_bytes(body)
    manifest=ROOT/'compiler-inputs.json'
    write(manifest,[dict(path=str(path),sha256=digest(path)) for path in sorted(paths)])
    generated={str(ROOT/name):digest(ROOT/name) for name in proposal.generated(P)}
    install=Path(compiler['path']).parents[1]
    flags=['-std=c++20','-O3','-DNDEBUG','-g0','-fsycl','-fsycl-id-queries-range=size_t',
        '-fno-fast-math','-ffp-contract=off','-DSTRATA_ENABLE_SYCL=1',
        '-I'+str(P/'include'),'-I'+str(P/'include/strata/sycl/compat'),'-I'+str(ROOT)]
    sources=[P/name for name in proposal.NATIVE]+[ROOT/'control.cpp',ROOT/'unroll.cpp']
    sources += [P/FIXTURE,ROOT/'timing.cpp']
    objects=[ROOT/f'input-{i}.o' for i in range(len(sources))]
    commands=[[compiler['path'],*flags,'-c',str(src),'-o',str(obj)] for src,obj in zip(sources,objects)]
    for index,name in ((7,'numerical'),(8,'timing')):
        commands.append([compiler['path'],*flags,*map(str,objects[:7]),str(objects[index]),
            '-pthread','-ldl','-o',str(ROOT/name)])
    for name in ('control','unroll'):
        commands.append([compiler['path'],*flags,'-fsycl-device-only','-S','-emit-llvm',
            str(ROOT/(name+'.cpp')),'-o',str(ROOT/(name+'.ll'))])
    legacy=json.loads((gen.A/'native-gdn-engine-build.json').read_text())['legacy_provenance_exception']['additional_old_product_review']
    record=dict(stage='prepared',source_commit=revision,input_sha256=pins,plan=experiment(),
        retained_engine=full['binary'],native_qualification=proof,generated=generated,compiler=compiler,
        compiler_inputs=dict(path=str(manifest),files=len(paths),sha256=digest(manifest)),commands=commands,
        build_environment=dict(PATH=str(install/'bin')+':/usr/bin:/bin',LANG='C',LC_ALL='C',TMPDIR=str(ROOT/'tmp'),
            LD_LIBRARY_PATH=str(install/'lib'),PYTHONDONTWRITEBYTECODE='1'),
        budgets=dict(total_wall_s=1800,per_command_wall_s=420,cpu_s=900,rss_bytes=12<<30,
            text_bytes=8<<20,file_bytes=512<<20,total_bytes=1<<30),
        legacy_provenance_holds=dict(result='native-gdn-engine-build.json',review_sha256=hashlib.sha256(json.dumps(legacy,sort_keys=True).encode()).hexdigest(),
            next_action=legacy['next_action'],full_final_input_proof=False),
        active=False,ownership_closed=True,passed=False,complete=False,GPU_executed=False,model_adoption_qualified=False)
    admission(record);write(ROOT/'plan.json',record);write(RESULT,record)
    print('Prepared isolated current-QSA static-unroll build;commit exact source/generated plan before build.',flush=True)


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
            print(f'Compiling/linking isolated static-unroll input {index+1}/{len(record["commands"])}',flush=True)
            entry,_,_=owner.run('build-'+str(index),command,record['build_environment'],P,
                wall=min(b['per_command_wall_s'],remaining),cpu=b['cpu_s'],rss_cap=b['rss_bytes'],
                text_cap=b['text_bytes'],file_cap=b['file_bytes'],total_cap=b['total_bytes'])
            guards.require(supervision.closed(entry) and entry['exit_code']==0,'static-unroll compile/link failed')
        admission(record)
        record.update(stage='built',passed=True,complete=True,binaries={name:dict(path=str(ROOT/name),
            sha256=digest(ROOT/name),bytes=(ROOT/name).stat().st_size) for name in ('numerical','timing')})
        record['device_IR']={name:dict(path=str(ROOT/(name+'.ll')),sha256=digest(ROOT/(name+'.ll')),
            bytes=(ROOT/(name+'.ll')).stat().st_size) for name in ('control','unroll')}
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
