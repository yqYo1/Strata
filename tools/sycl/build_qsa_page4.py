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
import qsa_page4_sources as proposal
import resource_engine as resource
import generation_engine as gen
import measurement_guards as guards
import measurement_owner as supervision

P,B=gen.P,gen.B
ROOT=B/'qsa-page4-address-v1'
RESULT=gen.A/'qsa-page4-current.json'
FIXTURE='bench/sycl/qsa_page4_numerical.cpp'


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
        guards.require(raw==(P/name).read_bytes(),'commit exact Page4-address input: '+name)
        pins[name]=hashlib.sha256(raw).hexdigest()
    for name in proposal.NATIVE:
        guards.require(hashlib.sha256(gen.git('show',proposal.BASE+':'+name)).hexdigest()==pins[name],
            'qualified native source changed: '+name)
    return pins


def experiment():
    return dict(question='Does compile-time Page4 address specialization reduce current long-prefill QSA latency without changing math or decode?',
        mechanism='Current source/LLVM has2signed+2unsignedi64 divisions across pack/attention modules. Hardwarecountercollection is refused;divisioncriticality/finalISA remain unknown. Test constant page division/remainder strength reduction,not a claim of ALU/VRAM/PCIe saturation. Requested bytes,logical mappings and selected-cell order are unchanged.',
        arms='Original production ordered-exp; unchanged isolated namespacecontrol; Page4clone replacing only shape.page_size occurrences with int64_t{4},guardedby calleractualpage4 andotherwiseoriginalgenericfallback. All index/count/pointer arithmetic stays64bit/checked,no32bit narrowing,no matrix/unroll/precision/state/queue/SLM/GRF changes. Reversible overlay restores every original body byte. Component-only one-thread host route counters add candidate launch overhead and are included in wrapper timing.',
        provenance='Main-authored MIT source/fixture/controller from own b072267fa discriminator and ae125c373d7f06a831e64b4e545011b828f63e0e77276ce0b80f587b18400bf3 reviewedsourceaudit;no external implementation copied andnoXeStrata code. Native6e40engine+12fixtures,decodecode/config/1256slots untouched;noenginebuild/adoption.',
        qualification='Compile2executables+deviceIRfirst. Require2kernels/module,controlintegerdivisionspresent,candidatezeroi64divisions;this is frontendnot finalISA. Smoke3arms*Page4andnonpower7*2cases;full3arms*pages1,4,7,16*11cases inclactualphysical262144lastslot/tail,INT_MIN/MAX,out-of-range/negative/duplicate/pageboundary IDs,missing/malformedpage andexistingrefusals. Independent tensororacle/exactbits/guards/immutability/ownedclosure/validatedfirstUR/noFault. CandidatePage4mustexecute specializedroute;otherpagesmustonlyusegenericfallback. Normalzeroqueryfallbackremains. Componentaddressingdoesnotqualifymodelfullcontext/session/RESTORE.',
        performance='After smoke/component/resources:Page4inall3arms.3independentcleanprocesses,6rotatedrounds,3arms,3artificialpatterns(32768/8192/2051gather;117488/8192/2051gatherandclustered),162samplesonce. Everyinvocationverifiescandidatefastincrement1/generic0 andcontrolsrouteunchanged outside timer;hostcounterincrementisinside candidatecall. Eachbatch>=100us and>=1000*positive128-pairclockfloor. AdvanceonlyifPage4/defaultandPage4/control latencygeomeans<=.95,no shape>1.05,andboth117488patternsbeatbothcontrolsineveryprocess. Failedgateforbidsmodeltrial;noafter-the-factrulechange.',
        limits='Wholewrapperincludesadmission,pack andcheckeddrain. Page4matchesproductiongranule,butartificiallocalityisnotproductiondistribution. Typedmathunchanged;finalISA,addresscriticalpath,physicaltraffic,saturation andcleanmodelgainunknown. Passingcomponentneedsnewlong-input/agent/thinkingfriendlyandweakgenerationplan preservingdecodehandoff/repeatedspeed beforeadoption.',
        retention='One known-goodnativegeneration remains. Auxiliarytwo-executablegeneration onlyforimmediatecomparison. Keepcompactprovenance,all162samples,correctness/fault/IR/resource decisions;retire consumedlogs,IR,generatedcopies,objects,binariesandunusedhelpersafterclosure. Carryolderprovenance/lifecycleholdsexplicitly.')


def admission(record,binary=False):
    guards.require(inputs(record['source_commit'])==record['input_sha256'] and record['plan']==experiment(),
        'Page4-address source or prospective plan changed')
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
            guards.require(digest(artifact['path'])==artifact['sha256'],'Page4-address fixture bytes changed')


def prepare():
    guards.require(not RESULT.exists() and not ROOT.exists(),'fresh single-use Page4-address build only;inspect existing owner')
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
    sources=[P/name for name in proposal.NATIVE]+[ROOT/'control.cpp',ROOT/'page4.cpp']
    sources += [P/FIXTURE,ROOT/'timing.cpp']
    objects=[ROOT/f'input-{i}.o' for i in range(len(sources))]
    commands=[[compiler['path'],*flags,'-c',str(src),'-o',str(obj)] for src,obj in zip(sources,objects)]
    for index,name in ((7,'numerical'),(8,'timing')):
        commands.append([compiler['path'],*flags,*map(str,objects[:7]),str(objects[index]),
            '-pthread','-ldl','-o',str(ROOT/name)])
    for name in ('control','page4'):
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
    print('Prepared isolated current-QSA Page4-address build;commit exact source/generated plan before build.',flush=True)


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
            print(f'Compiling/linking isolated Page4-address input {index+1}/{len(record["commands"])}',flush=True)
            entry,_,_=owner.run('build-'+str(index),command,record['build_environment'],P,
                wall=min(b['per_command_wall_s'],remaining),cpu=b['cpu_s'],rss_cap=b['rss_bytes'],
                text_cap=b['text_bytes'],file_cap=b['file_bytes'],total_cap=b['total_bytes'])
            guards.require(supervision.closed(entry) and entry['exit_code']==0,'Page4-address compile/link failed')
        admission(record)
        record.update(stage='built',passed=True,complete=True,binaries={name:dict(path=str(ROOT/name),
            sha256=digest(ROOT/name),bytes=(ROOT/name).stat().st_size) for name in ('numerical','timing')})
        record['device_IR']={name:dict(path=str(ROOT/(name+'.ll')),sha256=digest(ROOT/(name+'.ll')),
            bytes=(ROOT/(name+'.ll')).stat().st_size) for name in ('control','page4')}
        for name,item in record['device_IR'].items():
            text=Path(item['path']).read_text()
            item['kernels']=sum(line.startswith('define ') and 'spir_kernel' in line for line in text.splitlines())
            item['i64_divisions']=text.count('sdiv i64')+text.count('udiv i64')
            guards.require(item['kernels']==2 and (item['i64_divisions']==0 if name=='page4' else item['i64_divisions']==4),
                'planned constant-divisor frontend transformation not established')

    except BaseException as error:record.update(stage='failed',passed=False,complete=False,error=repr(error))
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
