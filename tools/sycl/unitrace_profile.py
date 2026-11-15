#!/usr/bin/env python3
"""Prepare/build a pinned official Level Zero collector; never launch a GPU target."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

sys.dont_write_bytecode = True
import measurement_guards as guards
import measurement_owner as supervision

P = Path(__file__).resolve().parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT = B / 'unitrace-l0-4b6d742-v2'
PTI = Path('/home/yayoi/ghq/github.com/intel/pti-gpu')
REV = '4b6d74294cdc72766c00a9211fe2e4261ee94244'
L0 = Path('/home/yayoi/ghq/github.com/oneapi-src/level-zero')
L0_REV = 'd3b3efbdaa27a0aef9a9812cc8fa4260557c1e7c'
NEO = Path('/home/yayoi/ghq/github.com/intel/compute-runtime')
NEO_REV = '050536ff3b6f830199ec99cff8ffcac3d804c40a'
IGA_REV = '4ba3c2aaab4f189b143ef288bc3e75eb02ad6f82'
U = PTI / 'tools/unitrace'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/cpu-profiler-current.json'
INPUTS = ('tools/sycl/unitrace_profile.py', 'tools/sycl/measurement_guards.py',
          'tools/sycl/measurement_owner.py', 'tools/sycl/measurement_environment.json')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(rows):
    return dict(files=len(rows), sha256=sha(json.dumps(sorted(rows), separators=(',', ':')).encode()))


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], timeout=60)


def source_inputs(revision):
    rows = []
    for name in INPUTS:
        original = git(P, 'show', revision + ':' + name)
        guards.require(original == (P/name).read_bytes(), 'uncommitted/changed controller input: ' + name)
        rows.append((name, sha(original)))
    return dict(rows)


def official_source():
    guards.require(git(PTI, 'rev-parse', 'HEAD').decode().strip() == REV, 'official PTI revision changed')
    guards.require(not git(PTI, 'status', '--porcelain').strip(), 'official PTI must remain clean')
    rows = []
    for name in git(PTI, 'ls-tree', '-r', '--name-only', REV, 'tools/unitrace', 'utils').decode().splitlines():
        original = git(PTI, 'show', REV + ':' + name)
        guards.require(original == (PTI/name).read_bytes(), 'official source mismatch: ' + name)
        rows.append((name, sha(original)))
    return fingerprint(rows)


def materialize_headers():
    records = []

    def write(name, data, origin):
        dest = ROOT/'include'/name
        guards.require(not dest.exists(), 'no overwrite of dependency input')
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        records.append(dict(path=str(dest.relative_to(ROOT)), sha256=sha(data), origin=origin))

    for name in ('ze_api.h', 'zes_api.h', 'zet_api.h', 'zer_api.h',
                 'layers/zel_tracing_api.h', 'layers/zel_tracing_register_cb.h'):
        write('level_zero/'+name, git(L0, 'show', L0_REV+':include/'+name),
              dict(repo=str(L0), revision=L0_REV, path='include/'+name))
    for name in ('tracing_api.h', 'tracing_types.h'):
        path = 'opencl/source/tracing/'+name
        data = git(NEO, 'show', NEO_REV+':'+path)
        if name == 'tracing_api.h':
            data = data.replace(b'opencl/source/tracing/tracing_types.h', b'tracing_types.h')
        write('CL/'+name, data, dict(repo=str(NEO), revision=NEO_REV, path=path,
              transform='get_compute_runtime_headers.py local tracing_types include' if name=='tracing_api.h' else None))
    pairs = [('opencl/extensions/public/cl_ext_private.h', 'CL/cl_ext_private.h')]
    base = 'level_zero/include/level_zero/'
    pairs += [(base+name, 'level_zero/'+name) for name in ('ze_stypes.h', 'ze_intel_gpu.h')]
    pairs += [(name, name.removeprefix('level_zero/include/')) for name in
              git(NEO, 'ls-tree', '-r', '--name-only', NEO_REV, base+'driver_experimental').decode().splitlines()
              if Path(name).name.startswith('zex_') and name.endswith('.h')]
    for path, dest in pairs:
        write(dest, git(NEO, 'show', NEO_REV+':'+path), dict(repo=str(NEO), revision=NEO_REV, path=path))
    for name in ('iga.h', 'iga_types_ext.hpp', 'iga_types_swsb.hpp', 'iga_bxml_ops.hpp', 'iga_bxml_enums.hpp'):
        url = 'https://raw.githubusercontent.com/intel/intel-graphics-compiler/'+IGA_REV+'/visa/iga/IGALibrary/api/'+name
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read((4<<20)+1)
        guards.require(0 < len(data) <= 4<<20 and b'#' in data and b'<html' not in data.lower(), 'invalid official IGA header')
        write('iga/'+name, data, dict(url=url, revision=IGA_REV))
    return records


def commands():
    flags = ['/usr/bin/g++', '-std=c++17', '-O3', '-DNDEBUG', '-fvisibility=default', '-pthread',
             '-DPTI_LEVEL_ZERO=1', '-DCXX_STD_FILESYSTEM_EXPERIMENTAL=0', '-DBUILD_WITH_L0=1']
    flags += ['-DBUILD_WITH_'+name+'=0' for name in ('XPTI', 'ITT', 'OPENCL', 'OMP', 'MPI', 'PERFETTO')]
    for path in (ROOT/'include', ROOT/'generated', U, U/'src', U/'src/utils', U/'src/opencl', U/'src/levelzero', PTI):
        flags += ['-I', str(path)]
    logger = str(U/'src/utils/logger_factory.cc')
    build = {}
    for name, source in (('collector', 'tracer.cc'), ('launcher', 'unitrace.cc')):
        output = ROOT/('libunitrace_tool.so' if name=='collector' else 'unitrace')
        extra = ['-shared', '-fPIC'] if name=='collector' else []
        build[name] = flags + extra + [str(U/'src'/source), logger, '-ldl', '-lrt', '-o', str(output)]
    return build


def dependency_files(build, run):
    paths = set()
    for label, args in build.items():
        # Preprocessing only: obtain the complete system-header closure before native compilation.
        sources = [arg for arg in args if arg.endswith('.cc')]
        prefix = args[:args.index(sources[0])]
        prefix = [arg for arg in prefix if arg != '-shared']
        for n, source in enumerate(sources):
            depfile = ROOT/'generated'/f'{label}-{n}.d'
            run(f'deps-{label}-{n}', prefix+['-M', '-MT', 'target', '-MF', str(depfile), source], 120)
            body = depfile.read_text().replace('\\\n', ' ').split(':', 1)[1]
            guards.require('\\' not in body, 'unexpected escaped dependency path')
            paths.update(Path(name).resolve() for name in body.split())
    tool_names = ['/usr/bin/g++', '/usr/bin/as', '/usr/bin/ld', '/usr/bin/python3', '/usr/lib/x86_64-linux-gnu/libc.so.6']
    for name in ('cc1plus', 'collect2'):
        tool_names.append(subprocess.check_output(['/usr/bin/g++', '-print-prog-name='+name], text=True).strip())
    for name in ('crtbeginS.o', 'crtendS.o', 'libstdc++.so', 'libgcc_s.so'):
        tool_names.append(subprocess.check_output(['/usr/bin/g++', '-print-file-name='+name], text=True).strip())
    paths.update(Path(name).resolve() for name in tool_names)
    guards.require(all(path.is_file() for path in paths), 'missing compile dependency')
    return [dict(path=str(path), sha256=sha(path.read_bytes())) for path in sorted(paths)]


def verify(record):
    guards.require(source_inputs(record['source_commit']) == record['input_sha256'], 'controller source changed')
    guards.require(official_source() == record['official_source'], 'PTI source fingerprint changed')
    manifest = ROOT/'dependencies.json'
    guards.require(sha(manifest.read_bytes()) == record['dependencies']['sha256'], 'compile dependency manifest changed')
    dependencies = json.loads(manifest.read_text())
    guards.require(len(dependencies) == record['dependencies']['files'], 'dependency count changed')
    for row in dependencies + record['generated']:
        path = Path(row['path'])
        if not path.is_absolute(): path = ROOT/path
        guards.require(path.is_file() and sha(path.read_bytes()) == row['sha256'], 'changed build input: '+str(path))
    guards.require(commands() == record['commands'], 'compile recipe differs')


def main(action):
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        container = json.loads(RESULT.read_text())
        guards.require(container['complete'] and container['ownership_closed'] and not container['active'], 'capability probe not closed')
        record = container.get('unitrace_build')
        if action == 'prepare':
            guards.require(record and record['stage']=='failed' and record['ownership_closed'] and not record['active']
                and record['source_commit'].startswith('31e142594')
                and record['error']=="RuntimeError('missing compile dependency')" and not ROOT.exists(),
                'only the closed, diagnosed v1 compiler program-name failure permits this new v2 plan')
            revision = git(P, 'rev-parse', 'HEAD').decode().strip()
            previous = RESULT.read_bytes()
            guards.require(previous==git(P,'show',revision+':'+str(RESULT.relative_to(P))), 'commit original failed attempt first')
            inputs = source_inputs(revision)
            official = official_source()
            ROOT.mkdir(mode=0o700);(ROOT/'generated').mkdir();(ROOT/'tmp').mkdir()
            record = dict(schema='strata-unitrace-host-build-v1', stage='preparing', source_commit=revision,
                input_sha256=inputs, official_source=official, official_revision=REV, active=False,
                previous_attempt=dict(commit=revision, path=str(RESULT.relative_to(P)), sha256=sha(previous),
                    outcome='Closed v1 preprocessing passed; program lookup incorrectly used -print-file-name for cc1plus/collect2. No native target compilation or GPU execution. New v2 uses -print-prog-name; original failure unchanged.'),
                consumer='Bounded L0 collector/UR-validation coexistence and graph/copy/fill qualification, then32K+ production service/submission trace.',
                retention='One current profiler tool build, separate from model engines. Retire build logs and consumed generated inputs after qualification; keep source/recipe/hashes. No new model generation.',
                safety='HOST BUILD ONLY. No unitrace/collector execution, GPU target, privileges, sysctl, driver reset or service change.',
                gpu_qualified=False, model_qualified=False, passed=False,
                environment=dict(PATH='/usr/bin:/bin', LANG='C', LC_ALL='C', TMPDIR=str(ROOT/'tmp'), PYTHONDONTWRITEBYTECODE='1'),
                budgets=dict(wall_s_per_compile=900, cpu_s_per_compile=900, rss_bytes=8<<30,
                             text_bytes_per_command=16<<20, total_bytes=512<<20, file_bytes=128<<20))
        else:
            guards.require(action=='build' and record and record['stage']=='prepared', 'build requires prepared plan; no retry')
            guards.require(RESULT.read_bytes()==git(P,'show','HEAD:'+str(RESULT.relative_to(P))), 'commit generated build plan first')
            verify(record)
            record['stage']='building'
        owner = None

        def persist():
            record['active']=bool(owner and owner.active is not None)
            (ROOT/'record.json').write_text(json.dumps(dict(record, owned_commands=owner.commands if owner else []), indent=2)+'\n')

        owner = supervision.Owner(ROOT/('prepare-owned' if action=='prepare' else 'build-owned'), persist)
        persist()

        def run(label, argv, wall=120):
            entry, _, err = owner.run(label, argv, record['environment'], PTI, wall=wall, cpu=wall,
                rss_cap=8<<30, text_cap=16<<20, total_cap=512<<20, file_cap=128<<20)
            guards.require(supervision.closed(entry) and entry.get('exit_code')==0,
                           label+' failed; inspect existing owned handle/log '+str(err))

        try:
            if action=='prepare':
                record['headers']=materialize_headers()
                for script, output in (('gen_tracing_common_header.py','common_header.gen'),
                                       ('gen_tracing_callbacks.py','tracing.gen'), ('gen_l0_loader.py','l0_loader.gen')):
                    run(script, ['/usr/bin/python3', str(U/'scripts'/script), str(ROOT/'generated'),
                                 str(ROOT/'include/level_zero'), str(ROOT/'include/CL')])
                for name in ('unitrace_commit_hash.h', 'unitrace_tool_commit_hash.h'):
                    run(name, ['/usr/bin/python3', str(U/'scripts/get_commit_hash.py'), str(ROOT/'generated'), name])
                record['commands']=commands()
                dependencies = dependency_files(record['commands'],run)
                manifest = ROOT/'dependencies.json'
                manifest.write_text(json.dumps(dependencies,indent=2)+'\n')
                record['dependencies']=dict(path=str(manifest), files=len(dependencies), sha256=sha(manifest.read_bytes()),
                    scope='Complete preprocessor header closure plus compiler/linker/startup/runtime assets; retained externally for current build/qualification admission.')
                record['generated']=[dict(path=str(path.relative_to(ROOT)), sha256=sha(path.read_bytes()))
                    for folder in ('include','generated') for path in sorted((ROOT/folder).rglob('*')) if path.is_file()]
                record['stage']='prepared'
            else:
                for label, argv in record['commands'].items(): run(label, argv, 900)
                verify(record)
                record['artifacts']={name:dict(path=str(ROOT/name), sha256=sha((ROOT/name).read_bytes()), bytes=(ROOT/name).stat().st_size)
                                     for name in ('libunitrace_tool.so','unitrace')}
                record.update(stage='built', passed=True)
        except BaseException as error:
            record.update(stage='failed', error=repr(error), passed=False)
        finally:
            record['ownership_closed']=owner.active is None and all(supervision.closed(e) for e in owner.commands)
            record['passed']=bool(record['passed'] and record['ownership_closed'])
            persist();container['unitrace_build']=record
            RESULT.write_text(json.dumps(container,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps({k:record.get(k) for k in ('stage','passed','ownership_closed','gpu_qualified','model_qualified','error')}))
        return 0 if record['stage'] in ('prepared','built') and record['ownership_closed'] else 1


if __name__=='__main__':
    guards.require(len(sys.argv)==2 and sys.argv[1] in ('prepare','build'), 'use unitrace_profile.py prepare|build')
    raise SystemExit(main(sys.argv[1]))
