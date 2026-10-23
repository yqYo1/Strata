#!/usr/bin/env python3
"""Main-owned bounded QSA production/producer calibration. Repository LICENSE.

Commit source,then prepare,commit the generated plan,then build. GPU execution
is deliberately a separate qualification step; no engine or decode is changed.
"""
import ast
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import generation_engine as gen
import measurement_guards as guards
import measurement_owner as owner_tools
import qsa_calibration_sources as source
import resource_engine as resource
import ur_validation as validation

P, B, A = gen.P, gen.B, gen.A
ROOT = B / 'qsa-resource-calibration-v1'
RESULT = A / 'qsa-resource-calibration-current.json'
NATIVE = ('src/sycl/runtime.cpp','src/sycl/compat.cpp','src/sycl/command_profile.cpp',
          'src/kernels/sycl/prompt_attention.cpp','src/kernels/sycl/qsa_bf16_prefix.cpp')
FIXTURE = 'bench/sycl/qsa_resource_calibration.cpp'
HEADER = 'bench/sycl/qsa_resource_calibration.hpp'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def inputs(revision):
    names = set((*NATIVE,FIXTURE,HEADER,'tools/sycl/measurement_environment.json'))
    names.update(gen.git('ls-files','include').decode().splitlines())
    pending, seen = [Path(__file__).resolve()], set()
    while pending:
        path = pending.pop()
        if path in seen: continue
        seen.add(path);names.add(str(path.relative_to(P)))
        for node in ast.walk(ast.parse(path.read_text())):
            modules = ([v.name for v in node.names] if isinstance(node,ast.Import)
                else [node.module] if isinstance(node,ast.ImportFrom) else [])
            for module in modules:
                local = path.parent / ((module or '').split('.')[0]+'.py')
                if local.is_file(): pending.append(local)
    pins = {}
    for name in sorted(names):
        raw = gen.git('show',revision+':'+name)
        guards.require(raw==(P/name).read_bytes(),'commit exact calibration input: '+name)
        pins[name] = hashlib.sha256(raw).hexdigest()
    for name in NATIVE:
        guards.require(gen.git('show',source.BASE+':'+name)==(P/name).read_bytes(),
                       'selected native source changed: '+name)
    return pins


def runtime_lineage():
    """New diagnostic scope;bind the actually passed model/runtime,not a mode alias."""
    path = A/'qsa-selection-current.json';closed = gen.committed(path)
    guards.require(closed['passed'] and closed['complete'] and closed['ownership_closed'] and
        not closed['active'] and not closed['fresh_faults'] and closed['boot_unchanged'] and
        closed['native_validation']['normal_native_exit']==0,'closed checked model/runtime required')
    ref = closed['prepared_plan'];raw = gen.git('show',ref['commit']+':'+ref['path'])
    guards.require(hashlib.sha256(raw).hexdigest()==ref['sha256'],'runtime source plan changed')
    prior = json.loads(raw);pins = prior['diagnostic_qualification']['runtime_lineage']['runtime_pins']
    guards.require(len(pins)==31,'complete runtime inventory required')
    for name,pin in pins.items(): guards.require(digest(name)==pin,'runtime bytes changed: '+name)
    probe = validation.preload_qualification()
    guards.require(probe==prior['native_validation_probe'],'private validation probe changed')
    return dict(model_reference=dict(path=str(path.relative_to(P)),sha256=digest(path)),
                runtime_pins=pins,native_validation_probe=probe)


def admission(record,binaries=False):
    guards.require(inputs(record['source_commit'])==record['inputs'],'calibration source changed')
    full,proof,_ = resource.qualification()
    guards.require(full['binary']==record['selected_engine'] and proof==record['selected_qualification'],
                   'selected fallback qualification changed')
    guards.require(record['runtime_lineage']==runtime_lineage(),'runtime qualification changed')
    generated = ROOT/'calibration.cpp'
    guards.require(generated.read_bytes()==source.generated(P) and
        digest(generated)==record['generated_sha256'],'generated calibration input changed')
    manifest = record['compiler_inputs']
    guards.require(digest(manifest['path'])==manifest['sha256'],'compiler inventory changed')
    rows = json.loads(Path(manifest['path']).read_text())
    guards.require(len(rows)==manifest['files'],'compiler inventory extent changed')
    for row in rows: guards.require(digest(row['path'])==row['sha256'],'compiler/SDK input changed: '+row['path'])
    guards.require(digest(record['compiler']['path'])==record['compiler']['sha256'],'compiler changed')
    if binaries:
        item = record['binary'];guards.require(digest(item['path'])==item['sha256'],'fixture changed')


def prepare():
    guards.require(not ROOT.exists() and not RESULT.exists(),'fresh single-use calibration only')
    previous = gen.committed(A/'qsa-selection-current.json')
    guards.require(previous['retirement']['complete'] and not (B/'qsa-selection-native-v1').exists(),
                   'retire finished diagnostic generation first')
    legacy = gen.committed(A/'native-gdn-engine-build.json')['legacy_provenance_exception']
    old = legacy['additional_actual_product_roots_review_20261023']
    guards.require(old['state']=='products_retired_source_qualification_remains_failed' and
        all(not (Path(r['root'])/'strata').exists() for r in old['roots']), 'close actual legacy product review first')
    revision = gen.git('rev-parse','HEAD').decode().strip();pins = inputs(revision)
    full,proof,_ = resource.qualification();runtime = runtime_lineage();compiler = full['compiler']
    sdk = gen.committed(A/'cpu-profiler-current.json')['unitrace_indexed_graph']['compiler_inputs']
    guards.require(digest(sdk['path'])==sdk['sha256'],'qualified SDK manifest changed')
    rows = json.loads(Path(sdk['path']).read_text());guards.require(len(rows)==sdk['files'],'SDK manifest extent')
    for row in rows: guards.require(digest(row['path'])==row['sha256'],'SDK input changed')
    paths = {Path(row['path']) for row in rows}
    for name in ('crtbeginS.o','crtendS.o','libstdc++.so','libgcc_s.so'):
        path = Path(subprocess.check_output([compiler['path'],'-print-file-name='+name],text=True,timeout=15).strip()).resolve()
        guards.require(path.is_file(),'link input absent');paths.add(path)
    ROOT.mkdir(mode=0o700);(ROOT/'tmp').mkdir();(ROOT/'cache').mkdir()
    (ROOT/'calibration.cpp').write_bytes(source.generated(P))
    manifest = ROOT/'compiler-inputs.json'
    write(manifest,[dict(path=str(p),sha256=digest(p)) for p in sorted(paths)])
    flags = ['-std=c++20','-O3','-DNDEBUG','-g0','-fsycl','-fsycl-id-queries-range=size_t',
        '-fno-fast-math','-ffp-contract=off','-DSTRATA_ENABLE_SYCL=1',
        '-I'+str(P/'include'),'-I'+str(P/'include/strata/sycl/compat'),'-I'+str(P/'bench/sycl')]
    sources = [P/n for n in NATIVE]+[ROOT/'calibration.cpp',P/FIXTURE]
    objects = [ROOT/f'input-{i}.o' for i in range(len(sources))]
    commands = [[compiler['path'],*flags,'-c',str(s),'-o',str(o)] for s,o in zip(sources,objects)]
    commands.append([compiler['path'],*flags,*map(str,objects),'-pthread','-ldl','-o',str(ROOT/'fixture')])
    for name,path in (('production',P/NATIVE[-1]),('calibration',ROOT/'calibration.cpp')):
        commands.append([compiler['path'],*flags,'-fsycl-device-only','-S','-emit-llvm',str(path),'-o',str(ROOT/(name+'.ll'))])
    install = Path(compiler['path']).parents[1]
    record = dict(stage='prepared',mode='QSA-production-row-gather-stage-calibration',source_commit=revision,
        inputs=pins,selected_engine=full['binary'],selected_qualification=proof,runtime_lineage=runtime,
        compiler=compiler,compiler_inputs=dict(path=str(manifest),sha256=digest(manifest),files=len(paths)),
        generated_sha256=digest(ROOT/'calibration.cpp'),commands=commands,
        environment=dict(PATH=str(install/'bin')+':/usr/bin:/bin',LANG='C',LC_ALL='C',
            LD_LIBRARY_PATH=str(install/'lib'),TMPDIR=str(ROOT/'tmp'),PYTHONDONTWRITEBYTECODE='1'),
        budgets=dict(wall=1800,per_command_wall=420,cpu=900,rss_cap=12<<30,text_cap=8<<20,file_cap=512<<20,total_cap=1<<30),
        protocol='P=actual production ordered-exp prefix wrapper;C=namespace/template control exact-bit check. '
            'L=original row-publication mapping plus BF16 vector K/V gathers and observable uint32 checksum,no KV payload SLM. '
            'S=original K/V SLM layouts and five explicit tile barriers,checksum consumers replacing QK/softmax/PV. '
            'All arms retain admission and once-per-call INT8-to-BF16 prefix preparation. '
            'S/L outputs must match full host checksums;V perturbation must be observed;all guards/input immutability. '
            'First traced97/3/33 small process,then validated512-query cap2051 synthetic32768/117488 gather and117488 clustered cases. '
            'Inspect codegen actual loads,SLM/barriers,MAD presence/absence and resource divergence before clean timing. '
            'If valid,three independent processes,five cyclically balanced rounds/arm/shape,128 calls/batch. '
            'Each batch>=100ms and>=1000 median clock-pair overhead;first failure stops,no adaptive enlargement/retry. '
            'P-L/P-S are not additive saved-time estimates;probe sinks and changed resources prevent exclusive resource attribution. '
            'No model trial/adoption from calibration;M/resident matrix is a separately planned follow-up if unresolved.',
        retention='One extra diagnostic fixture only through bounded producer calibration;retire at closure after source/recipe recovery. '
            'Keep useful individual timings/checks/codegen limits,not successful traces or intermediate binaries. '
            'Original6e40 engine/12fixtures and decode remain unchanged.',
        active=False,ownership_closed=True,passed=False,complete=False,GPU_executed=False,
        clean_model_speed_qualified=False,hardware_saturation_qualified=False,model_adoption_qualified=False)
    admission(record);write(ROOT/'build-plan.json',record);write(RESULT,record)
    print('Prepared calibration;commit exact generated plan before build.',flush=True)


def build():
    record = gen.committed(RESULT)
    guards.require(record==json.loads((ROOT/'build-plan.json').read_text()) and record['stage']=='prepared'
        and not (ROOT/'build-record.json').exists(),'fresh committed build plan required')
    admission(record);owner = None
    def persist():
        record['active'] = bool(owner and owner.active is not None)
        write(ROOT/'build-record.json',dict(record,owned_commands=owner.commands if owner else []))
    owner = owner_tools.Owner(ROOT/'owned',persist);deadline=time.monotonic()+record['budgets']['wall']
    try:
        for i,command in enumerate(record['commands']):
            budget=record['budgets'];remaining=deadline-time.monotonic();guards.require(remaining>1,'build deadline')
            print(f'Compiling/linking calibration {i+1}/{len(record["commands"])}',flush=True)
            entry,_,_=owner.run('build-'+str(i),command,record['environment'],P,
                wall=min(remaining,budget['per_command_wall']),cpu=budget['cpu'],rss_cap=budget['rss_cap'],
                text_cap=budget['text_cap'],file_cap=budget['file_cap'],total_cap=budget['total_cap'])
            guards.require(owner_tools.closed(entry) and entry['exit_code']==0,'calibration compile/link failed')
        admission(record);path=ROOT/'fixture'
        record.update(stage='built',passed=True,complete=True,binary=dict(path=str(path),bytes=path.stat().st_size,sha256=digest(path)))
    except BaseException as error: record.update(stage='failed',error=repr(error),passed=False,complete=False)
    finally:
        record['ownership_closed']=owner.active is None and all(owner_tools.closed(c) for c in owner.commands)
        persist();write(RESULT,record)
    print(json.dumps({k:record.get(k) for k in ('stage','passed','ownership_closed','error')}),flush=True)
    return 0 if record['passed'] else 1


if __name__=='__main__':
    guards.require(len(sys.argv)==2 and sys.argv[1] in ('prepare','build'),'use prepare|build')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        sys.exit(prepare() if sys.argv[1]=='prepare' else build())
