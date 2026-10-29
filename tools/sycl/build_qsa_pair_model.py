#!/usr/bin/env python3
"""Own bounded model build:selected6e40 plus opt-in pair-prefill only.

The numerical fixture links the actual seven production object files from
this engine build. No XeStrata application source or benchmark code is used.
"""
import ast
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import select
import shlex
import subprocess
import sys
import time
sys.dont_write_bytecode=True
import build_fused_profile as closure
import qsa_pair_model_sources as model
import runtime_probe as runtime

P,B=runtime.P,runtime.B
ROOT=B/'qsa-pair-model-v1'
RESULT=P/'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
KEY='qsa_pair_model'
ORIGIN=B/'main-native-gdn-engine/record.json'
ORIGIN_SHA='a38f477670fafcd1a931be2e20d90d6d9718df75b61954b421f361af9f6881b2'
TARGETS=['strata','sycl_submission_test','sycl_graph_envelope_test']
FIXTURES=['bench/sycl/qsa_pair_probe.hpp','bench/sycl/qsa_pair_probe.cpp']
COMMON={**{n:'strata_sycl_runtime' for n in ['src/sycl/runtime.cpp','src/sycl/compat.cpp',
    'src/sycl/command_profile.cpp']},**{n:'strata_kernels' for n in [
    'src/kernels/sycl/prompt_attention.cpp','src/kernels/sycl/qsa_pair_union.cpp',
    'src/kernels/sycl/qsa_pair_attention.cpp','src/kernels/sycl/qsa_pair_prefix.cpp']}}
require,digest,write=runtime.require,runtime.digest,runtime.write


def origin():
    require(digest(ORIGIN)==ORIGIN_SHA,'original selected receipt changed')
    row=json.loads(ORIGIN.read_text())
    require(row['source_commit']==model.BASE and row['project']['count']==542,
            'unexpected selected source')
    return row


def sources(revision):
    pending=[Path(__file__).resolve()];seen=set();pins={}
    while pending:
        path=pending.pop()
        if path in seen:continue
        seen.add(path);name=str(path.relative_to(P));raw=model.git('show',revision+':'+name)
        require(path.read_bytes()==raw,'commit exact controller:'+name);pins[name]=model.sha(raw)
        for node in ast.walk(ast.parse(raw)):
            modules=[v.name for v in node.names] if isinstance(node,ast.Import) else \
                [node.module] if isinstance(node,ast.ImportFrom) else []
            for module in modules:
                local=path.parent/((module or '').split('.')[0]+'.py')
                if local.is_file():pending.append(local)
    for name in [*FIXTURES,'tools/sycl/measurement_environment.json']:
        raw=model.git('show',revision+':'+name)
        require((P/name).read_bytes()==raw,'commit exact fixture:'+name);pins[name]=model.sha(raw)
    return pins


def native_sources(full,revision):
    expected=full['project']['source_sha256'];names=[*expected,*model.CONFIGURATION]
    data={}
    for name,raw in runtime.build.blobs(P,model.BASE,names):
        require(name not in expected or model.sha(raw)==expected[name],'original source differs:'+name)
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'source path escape')
        data[name]=raw
    overlays,proof=model.overlays();data.update(overlays)
    for name in FIXTURES:data[name]=model.git('show',revision+':'+name)
    return data,proof


def compact(row):
    out={k:v for k,v in row.items() if k not in ['input_sha256','commands_executed']}
    out['controller_inputs']=dict(files=len(row['input_sha256']),aggregate_sha256=model.sha(
        json.dumps(row['input_sha256'],sort_keys=True,separators=(',',':')).encode()))
    return out


def admit(row):
    full=origin()
    require(sources(row['source_commit'])==row['input_sha256'],'controller/fixture inputs changed')
    require(closure.sdk_inputs()==row['sdk'],'compiler SDK changed')
    require(digest(full['compiler']['path'])==full['compiler']['sha256'],'compiler changed')
    for group in full['dependencies']:
        require(runtime.build.pin(Path(group['root']),['.'])==group,'selected dependency changed')
    rt=runtime.validation.committed(runtime.RESULT)
    require(digest(runtime.RESULT)==row['runtime_sha256'],'runtime qualification changed')
    require(runtime.current(rt)['binary']==row['selected_binary'],'selected fallback changed')
    external=row['external_manifest']
    require(digest(external['path'])==external['sha256'],'external input inventory changed')
    allowed=json.loads(Path(external['path']).read_text())
    require(len(allowed)==external['files'],'external input count changed')
    for path,sha in allowed.items():require(digest(path)==sha,'external tool/input changed:'+path)
    for v in json.loads(closure.lineage.SDK.read_text()):
        allowed[str(Path(v['path']).resolve())]=closure.SYSTEM_HEADER_UPDATES.get(v['path'],(None,v['sha256']))[1]
    data,proof=native_sources(full,row['source_commit'])
    require(proof==row['production_projection'],'production projection changed')
    require(model.sha(json.dumps({n:model.sha(raw) for n,raw in sorted(data.items())},
        sort_keys=True,separators=(',',':')).encode())==row['native_source_digest'],'isolated source mapping changed')
    for name,raw in data.items():
        path=ROOT/'source'/name
        require(path.is_file() and not path.is_symlink() and path.read_bytes()==raw,'isolated source changed:'+name)
        allowed[str(path.resolve())]=model.sha(raw)
    for group in full['dependencies']:
        allowed.update({str((Path(group['root'])/name).resolve()):sha for name,sha in group['source_sha256'].items()})
    selected=model.selected_fixture()
    require((ROOT/'selected.cpp').read_bytes()==selected and model.sha(selected)==row['selected_fixture_sha256'],
            'selected fixture source changed')
    allowed[str(ROOT/'selected.cpp')]=model.sha(selected)
    allowed.update(row['generated_expected'])
    return allowed


def prepare():
    require(not ROOT.exists(),'fresh generation only;inspect owner before resuming')
    record=runtime.validation.committed(RESULT)
    require(KEY not in record,'close prior model attempt first')
    component=record['qsa_pair_proposal']
    require(component['performance']['outcome']['advance_to_model'] and
        component['component_retirement_preflight']['retired'],'component gain and retirement required')
    review=runtime.validation.committed(P/'bench/results/2026-10-15-pr6-recovery-audit/research-index.json',
        ('qsa_pair_prefill_lease_contract_20261029','native_review'))
    require(review['accepted_for_model_build'] and review['source_revision']==model.git('rev-parse',model.INTEGRATION).decode().strip(),
            'independent production lease/source review required')
    full=origin();revision=model.git('rev-parse','HEAD').decode().strip();pins=sources(revision)
    rt=runtime.validation.committed(runtime.RESULT);runtime.current(rt)
    data,proof=native_sources(full,revision);sdk=closure.sdk_inputs()
    external=closure.external_inputs(rt)
    for root in [Path('/usr/share/cmake-3.28/Modules'),Path('/usr/share/cmake-3.28/Templates')]:
        external.update({str(p.resolve()):digest(p) for p in root.rglob('*') if p.is_file()})
    tool_paths=[str(Path('/usr/bin/'+n).resolve()) for n in ['cmake','ninja','cc','ar','ranlib','as','ld','git']]
    tool_paths+=[subprocess.check_output(['/usr/bin/cc','-print-prog-name=cc1'],text=True,timeout=10).strip(),full['compiler']['path']]
    external.update({str(Path(p).resolve()):digest(p) for p in tool_paths})
    old=json.loads(model.git('show','8849d71e5:'+str(RESULT.relative_to(P))))
    external.update(old['direct_K_loop_invariant']['native_build']['link_inputs'])
    require(all(digest(p)==sha for p,sha in external.items()),'original imported link input changed')
    ROOT.mkdir(mode=0o700)
    for name in ['source','tmp','cache']:(ROOT/name).mkdir()
    for name,raw in data.items():
        path=ROOT/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    selected=model.selected_fixture();(ROOT/'selected.cpp').write_bytes(selected)
    write(ROOT/'external-inputs.json',external)
    argv=list(full['configure_argv']);argv[argv.index('-S')+1]=str(ROOT/'source');argv[argv.index('-B')+1]=str(ROOT/'build')
    env=dict(full['environment'],TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'))
    flags=['-std=c++20','-O3','-DNDEBUG','-g0','-fsycl','-fsycl-id-queries-range=size_t',
        '-fno-fast-math','-ffp-contract=off','-DSTRATA_ENABLE_SYCL=1',
        '-I'+str(ROOT/'source/include'),'-I'+str(ROOT/'source/include/strata/sycl/compat'),
        '-I'+str(ROOT/'source/bench/sycl')]
    commands=[['configure',argv],['engine',['/usr/bin/cmake','--build',str(ROOT/'build'),'--target',*TARGETS,'-j','4']]]
    for name,path,extra in [('selected',ROOT/'selected.cpp',['-include',str(ROOT/'source/bench/sycl/qsa_pair_probe.hpp')]),
                            ('fixture',ROOT/'source/bench/sycl/qsa_pair_probe.cpp',[])]:
        commands.append([name,[full['compiler']['path'],*flags,*extra,'-MD','-MF',str(ROOT/(name+'.d')),
            '-c',str(path),'-o',str(ROOT/(name+'.o'))]])
    common=[str(ROOT/'build/CMakeFiles'/ (target+'.dir')/(name+'.o')) for name,target in COMMON.items()]
    commands.append(['fixture-link',[full['compiler']['path'],*flags,str(ROOT/'selected.o'),str(ROOT/'fixture.o'),
        *common,'-pthread','-ldl','-o',str(ROOT/'fixture')]])
    commands.append(['dependencies',['/usr/bin/ninja','-C',str(ROOT/'build'),'-t','deps']])
    row=dict(stage='prepared',source_commit=revision,input_sha256=pins,original_source=model.BASE,
        original_build=dict(path=str(ORIGIN),sha256=ORIGIN_SHA),production_projection=proof,
        native_source_digest=model.sha(json.dumps({n:model.sha(raw) for n,raw in sorted(data.items())},sort_keys=True,separators=(',',':')).encode()),
        native_source_count=len(data),selected_fixture_sha256=model.sha(selected),
        selected_binary=full['binary'],compiler=full['compiler'],sdk=sdk,
        external_manifest=dict(path=str(ROOT/'external-inputs.json'),sha256=digest(ROOT/'external-inputs.json'),files=len(external)),
        runtime_sha256=digest(runtime.RESULT),boot=rt['boot'],driver=rt['driver'],environment=env,
        source_contract=review,targets=TARGETS,commands=commands,common_production_objects=common,
        generated_expected={str(ROOT/'build/ggml/src/ggml-version.h'):model.sha(closure.HEADER.encode())},
        budgets=dict(wall=1800,cpu=7200,rss_cap=12<<30,text_cap=32<<20,file_cap=512<<20,total_cap=4<<30),
        total_wall_seconds=3600,active=False,ownership_closed=True,passed=False,GPU_executed=False,model_built=False,model_selected=False,
        criteria='CPU build only. Recover original542project and9311dependency inputs,project only declared pair-prefill edit spans and six new files. Nine original configuration-only tools and two fixture files. Compare all actual dependencies/imported link providers;zero unfrozen inputs. Fixture reuses seven actual production objects. No numerical/model/performance qualification from compilation.',
        retention='One imminent pair-prefill model/fixture candidate against original6e40+12fixtures. Retire candidate/normal logs/objects after its decision and exact source/recipe recovery. Legacy provenance holds remain separate.')
    admit(row);write(ROOT/'record.json',row);record[KEY]=dict(build=compact(row));write(RESULT,record)
    print('CPU build prepared;commit frozen plan before build.',flush=True)


def run():
    record=runtime.validation.committed(RESULT);row=json.loads((ROOT/'record.json').read_text())
    require(row['stage']=='prepared' and compact(row)==record[KEY]['build'] and not (ROOT/'owned-build').exists(),
            'fresh committed build plan required')
    allowed=admit(row);original=copy.deepcopy(row);owner=None;deadline=time.monotonic()+row['total_wall_seconds']
    def persist():
        row['active']=bool(owner and owner.active is not None)
        write(ROOT/'record.json',dict(row,commands_executed=owner.commands if owner else []))
    def monitor(io,entry):
        next_size=0
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll()
            if time.monotonic()>=next_size:
                require(runtime.supervision.artifact_bytes(ROOT)<=row['budgets']['total_cap'],'whole build size budget')
                next_size=time.monotonic()+1
            if select.select([io.proc.stdout],[],[],.02)[0]:io.read();io.pending.clear()
        io.exit(1)
    try:
        row['stage']='building';owner=runtime.supervision.Owner(ROOT/'owned-build',persist)
        for label,argv in row['commands']:
            remaining=deadline-time.monotonic();require(remaining>1,'whole build deadline')
            print('CPU build:'+label,flush=True)
            entry,out,err=owner.run(label,argv,row['environment'],ROOT,interaction=monitor,
                **dict(row['budgets'],wall=min(remaining,row['budgets']['wall'])))
            require(runtime.supervision.closed(entry) and entry['exit_code']==0,
                label+' failed:'+(out.read_text(errors='replace')+err.read_text(errors='replace'))[-5000:])
        admit(original)
        paths={Path(line.strip()) for line in out.read_text().splitlines() if line.startswith('    ') and line.strip()}
        paths={(p if p.is_absolute() else ROOT/'build'/p).resolve() for p in paths}
        for dep in ROOT.glob('*.d'):
            paths.update(Path(p).resolve() for p in shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1]))
        require(len(paths)>700 and {ROOT/'source'/n for n in COMMON}<=paths,'incomplete actual dependency closure')
        observed={}
        for path in sorted(paths):
            sha=digest(path);require(allowed.get(str(path))==sha,'unfrozen actual compiler input:'+str(path));observed[str(path)]=sha
        write(ROOT/'compiler-inputs.json',observed)
        ninja=(ROOT/'build/build.ninja').read_text();imported={}
        for name in TARGETS:
            stanza=ninja[ninja.index('build '+name+':'):].split('\n\n',1)[0]
            links=shlex.split(next(line.split(' = ',1)[1] for line in stanza.splitlines() if line.startswith('  LINK_LIBRARIES = ')))
            for path in (Path(p).resolve() for p in links if p.startswith('/')):
                require(allowed.get(str(path))==digest(path),'unfrozen imported provider:'+str(path));imported[str(path)]=digest(path)
        def asset(path):return dict(path=str(path),bytes=path.stat().st_size,sha256=digest(path))
        row.update(stage='built-not-executed',passed=True,model_built=True,
            binaries={n:asset(ROOT/'build'/n) for n in TARGETS},fixture=asset(ROOT/'fixture'),
            common_production_objects={str(p):asset(Path(p)) for p in row['common_production_objects']},
            imported_link_inputs=imported,actual_compiler_inputs=dict(files=len(observed),path=str(ROOT/'compiler-inputs.json'),
                sha256=digest(ROOT/'compiler-inputs.json'),uncovered_files=0),
            compile_database_sha256=digest(ROOT/'build/compile_commands.json'))
    except BaseException as error:row.update(stage='failed',passed=False,error=repr(error))
    finally:
        row['ownership_closed']=owner is None or owner.active is None and all(runtime.supervision.closed(c) for c in owner.commands)
        row['passed']=bool(row['passed'] and row['ownership_closed']);persist();record[KEY]['build']=compact(row);write(RESULT,record)
    print(json.dumps({k:row.get(k) for k in ['stage','passed','model_built','ownership_closed','error']}),flush=True)
    require(row['passed'],'model build did not qualify')


if __name__=='__main__':
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'prepare':prepare,'build':run}[sys.argv[1]]()
