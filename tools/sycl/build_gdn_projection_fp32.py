#!/usr/bin/env python3
"""Compile the private GDN FP32 producer through the production AOT pipeline.

Main-authored; repository LICENSE. CPU-only compile/link, no model or device
execution. Preserve the selected engine and retire the closed reference before
creating another compiled generation.
"""
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

sys.dont_write_bytecode = True
import retained_prefill_candidate as retained
import gdn_projection_fp32_sources as source

base = retained.base
P, B, RESULT = base.P, base.B, base.RESULT
ROOT = B / 'official-gdn-fp32-projection-v1'
KEY = 'official_gdn_fp32_projection_build_v1'
require, digest, write = base.require, base.digest, base.write


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = base.compact(row); write(RESULT, data)


def prepare():
    data = base.committed(RESULT)
    require(KEY not in data and not ROOT.exists(), 'fresh compile-only candidate required')
    chosen = data['official_prefill_strategy_speed_v1']
    require(chosen['passed'] and chosen['complete'] and not chosen['active'] and
            chosen['ownership_closed'] and chosen['next_development']['native_source_commit'] ==
            retained.NATIVE, 'closed selected comparison required')
    prior, providers, probe = retained.verify()
    _, native = source.sources()
    revision = base.basis.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    controls = base.controls(revision, controller=Path(__file__).resolve())
    db = retained.ROOT / 'build/compile_commands.json'
    entry = [v for v in json.loads(db.read_text()) if
             v['file'].endswith('/src/prefill/sycl/vllm_xe2/native16.cpp')]
    require(len(entry) == 1, 'one production private GEMM translation unit required')
    argv = entry[0].get('arguments') or shlex.split(entry[0]['command'])
    argv = [v.replace(str(retained.ROOT / 'source'), str(ROOT / 'source')) for v in argv]
    argv[argv.index('-o') + 1] = str(ROOT / 'native16.o')
    argv += ['-MD', '-MF', str(ROOT / 'native16.d')]
    require(argv[0] == base.COMPILER and '-fsycl-targets=spir64_gen' in argv,
            'qualified compiler/AOT flags required')
    env = dict(prior['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    bundler = subprocess.check_output([base.COMPILER, '-print-prog-name=clang-offload-bundler'],
        text=True, env=env, timeout=15).strip()
    triple = subprocess.check_output([base.COMPILER, '-dumpmachine'], text=True, env=env, timeout=15).strip()
    require(Path(bundler).is_absolute() and Path(bundler).is_file(), 'SDK bundler required')
    link = [base.COMPILER, '-O3', '-fsycl', '-fsycl-targets=spir64_gen', '-fsycl-link',
        '-fno-sycl-instrument-device-code', '-Xspirv-translator',
        '-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate',
        '-Xsycl-target-backend=spir64_gen', '-device bmg-g21-a0 -internal_options -cl-intel-256-GRF-per-thread',
        str(ROOT / 'native16.o'), '-o', str(ROOT / 'image.o')]
    commands = [('compile', argv), ('aot-link', link), ('host-slice', [bundler, '-unbundle', '-type=o',
        '-targets=host-' + triple, '-input=' + str(ROOT / 'native16.o'), '-output=' + str(ROOT / 'host.o')]),
        ('device-image', [bundler, '-unbundle', '-type=o', '-targets=sycl-spir64_gen',
        '-input=' + str(ROOT / 'image.o'), '-output=' + str(ROOT / 'device.bin')])]
    ROOT.mkdir(mode=0o700)
    for part in ('source', 'tmp', 'cache'): (ROOT / part).mkdir()
    for name, raw in native.items():
        path = ROOT / 'source' / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    row = dict(stage='prepared', source_commit=revision, implementation_commit=source.IMPLEMENTATION,
        controller_inputs=controls, native_sources={n:hashlib.sha256(raw).hexdigest() for n,raw in native.items()},
        source_basis=dict(selected=retained.NATIVE, changed_paths=list(source.CHANGED),
            other_native_sources_unchanged=True, production_dispatch_changed=False),
        compiler=prior['compiler'], sdk=prior['sdk'], external_manifest=prior['external_manifest'],
        dependencies=prior['dependencies'], runtime_manifest=data['official_prefill_strategies_device_v2']['runtime_manifest'],
        native_validation_probe=probe, environment=env, commands=commands, commands_executed=[],
        selected_binaries=prior['binaries'], selected_record=dict(path=str(retained.ROOT / 'record-v2.json'),
            sha256=digest(retained.ROOT / 'record-v2.json')), reference_retirement=dict(complete=False,
            next='Recover032source/f640recipe,retire seven obsolete ELFs/build outputs;keep consumed manifests.'),
        original_compile_database=dict(path=str(db),sha256=digest(db)),
        boot=prior['boot'], driver=prior['driver'], budgets=dict(wall=1200,cpu=3600,rss_cap=12<<30,
            text_cap=16<<20,file_cap=512<<20,total_cap=4<<30), whole_wall_seconds=2400,
        active=False, ownership_closed=True, passed=False, GPU_executed=False, inference_adopted=False,
        plan=dict(scope='Compile the exact private production translation unit and its BMG G21 AOT image;no executable/GPU launch.',
            acceptance='All source/SDK/dependency/provider/fixture pins must match;owned commands exit0;actual compiler dependencies covered;actual device image retained for direct FP32-store inspection.',
            next='Inspect new GdnProjectionFp32Kernel in this exact image;then commit ownership/tail/numerical and bounded traced device fixture before execution.No PP/TG/262K qualification from compilation.'),
        reference_admission_corrections=[dict(source='18d5f437a',outcome='failed',reason='Full dependency maps compared with compact descriptors.'),
            dict(source='c272c2abd',outcome='failed',reason='Dependency aggregate requires original compact JSON encoding.'),
            dict(source='d2f2056d7',outcome='passed',GPU_executed=False)],
        retention='Selected eight-ELF engine plus this named component generation;retire closed reference before compile.Raw compile output retires after useful diagnostics/input evidence and codegen question close.')
    save(row); print('Prepared555-file private FP32 production-AOT compile plan;commit and reference retirement required.', flush=True)


def admit(row):
    require(base.committed(RESULT)[KEY] == base.compact(row), 'commit exact compile plan/result')
    require(base.controls(row['source_commit'],controller=Path(__file__).resolve()) ==
            row['controller_inputs'], 'frozen compile controls changed')
    prior, providers, _ = retained.verify()
    require(prior['binaries'] == row['selected_binaries'], 'selected fallback changed')
    _, native = source.sources()
    require({n:hashlib.sha256(raw).hexdigest() for n,raw in native.items()} == row['native_sources'],
            'candidate source reconstruction differs')
    allowed = base.sdk(); allowed.update(retained.runtime.inventory(row['external_manifest'])); allowed.update(providers)
    for name, raw in native.items():
        path=ROOT/'source'/name
        require(path.is_file() and not path.is_symlink() and path.read_bytes()==raw, 'frozen source changed: '+name)
        allowed[str(path.resolve())]=row['native_sources'][name]
    return allowed


def run():
    row=json.loads((ROOT/'record.json').read_text())
    require(row['stage']=='prepared' and not (ROOT/'owned-build').exists(), 'fresh committed compile only')
    require(row['reference_retirement']['complete'], 'retire closed reference generation before compile')
    allowed=admit(row); owner=None; deadline=time.monotonic()+row['whole_wall_seconds']
    def persist():
        write(ROOT/'record.json',dict(row,commands_executed=owner.commands if owner else [],active=bool(owner and owner.active)))
    def monitor(io, entry):
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll(); require(base.supervision.artifact_bytes(ROOT)<=row['budgets']['total_cap'],'whole compile size budget')
            if select.select([io.proc.stdout],[],[],.02)[0]: io.read(); io.pending.clear()
        io.exit(1)
    try:
        row['stage']='compiling'; owner=base.supervision.Owner(ROOT/'owned-build',persist)
        for label,argv in row['commands']:
            print('CPU-only GDN FP32:',label,flush=True)
            entry,out,err=owner.run(label,argv,row['environment'],ROOT,interaction=monitor,
                **dict(row['budgets'],wall=min(row['budgets']['wall'],deadline-time.monotonic())))
            require(base.supervision.closed(entry) and entry['exit_code']==0,
                label+' failed: '+(out.read_text(errors='replace')+err.read_text(errors='replace'))[-4000:])
        raw=(ROOT/'native16.d').read_text().replace('\\\n','')
        found=set()
        for line in raw.splitlines():
            if not line.strip(): continue
            target, colon, dependencies = line.partition(':')
            require(colon and target.strip(), 'invalid compiler dependency rule')
            found.update(str(Path(p).resolve()) for p in shlex.split(dependencies))
        require(str(ROOT/'source/src/prefill/sycl/vllm_xe2/native16.hpp') in found and all(
            '/MistVVK/' not in p and allowed.get(p)==digest(p) for p in found), 'uncovered/changed actual compiler input')
        write(ROOT/'compiler-inputs.json',{p:allowed[p] for p in sorted(found)})
        row.update(stage='compiled-not-executed',passed=True,compiled_inputs=dict(path=str(ROOT/'compiler-inputs.json'),
            sha256=digest(ROOT/'compiler-inputs.json'),files=len(found),uncovered=0),
            outputs={n:dict(path=str(ROOT/n),sha256=digest(ROOT/n),bytes=(ROOT/n).stat().st_size)
                     for n in ('native16.o','image.o','host.o','device.bin')})
    except BaseException as error:
        row.update(stage='failed',passed=False,error=repr(error))
    finally:
        row.update(active=False,ownership_closed=bool(owner and all(base.supervision.closed(v) for v in owner.commands)),
                   commands_executed=owner.commands if owner else [])
        require(owner is None or owner.active is None,'compile ownership unresolved'); save(row)
    print(json.dumps({k:row.get(k) for k in ('stage','passed','GPU_executed','error')}),flush=True)
    require(row['passed'],'private FP32 compile failed')


if __name__=='__main__':
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); base.guards.no_model_owners([])
        if sys.argv[1:]==['prepare']: prepare()
        elif sys.argv[1:]==['build']: run()
        else: raise RuntimeError('build_gdn_projection_fp32.py prepare|build')
