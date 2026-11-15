#!/usr/bin/env python3
"""Bounded component build for selected fused-prefill service attribution.

Main-authored, repository LICENSE. SDK checks reuse our fc4907e8d wrapper
builder;no third-party implementation or XeStrata source is imported.
The original selected6e40 engine and12fixtures are never overwritten.
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
import runtime_probe as r
import build_full_context as lineage
build, P, B = r.build, r.P, r.B
ROOT = B / 'prefill-fused-command-profile-v2'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/prefill-resources-current.json'
TARGETS = ['sycl_command_profile_test','sycl_moe_gu_swiglu_bench','sycl_moe_down_scatter_bench']
PREFIXES = ['CMakeLists.txt','cmake','include','src','tests','third_party','bench/sycl',
    'tools/sycl/build_fused_profile.py','tools/sycl/command_service.py',
    'tools/sycl/runtime_probe.py','tools/sycl/build_engine.py','tools/sycl/build_full_context.py',
    'tools/sycl/check_full_context_source.py','tools/sycl/measurement_owner.py',
    'tools/sycl/measurement_guards.py','tools/sycl/ur_validation.py','tools/sycl/measurement_environment.json']
HEADER = '#pragma once\n\n#define GGML_VERSION "0.24.0"\n#define GGML_COMMIT  "3cf03257f"\n'
require, digest, write = r.require, r.digest, r.write


SYSTEM_HEADER_UPDATES = {
    '/usr/include/metee.h': ('f063edcb00c88f3ddebc10f9b0487fe813cfb7ba71dc4b798729bed8902018a3',
                            'd07e18ce9228c3fc570b186eab13bf8a3e6c04214992f0de9d53b855dc837252'),
    '/usr/include/opencl-c-base.h': ('e1683c3db9cf9a95e9d02acbbfa74dcd29d4deaf4a856430f538d76985b04114',
                                   '673c4720f879c2f101504979edbd0216c9cb6a982ce2b62bd467b81fd53d8e33'),
    '/usr/include/opencl-c.h': ('579b58f482ab0b3dfd396cab32ea3e0537bc5cf4c1067377b83fe417cbcc513b',
                              'aa7e871a8f7d961aa4126a9fadc2196f822c21811946c90147dc7415fe677de9'),
}


def sdk_inputs():
    require(digest(lineage.SDK) == lineage.SDK_SHA, 'original SDK inventory changed')
    rows = json.loads(lineage.SDK.read_text())
    require(len(rows) == 8697, 'original SDK inventory count differs')
    for item in rows:
        path = item['path']
        before, expected = SYSTEM_HEADER_UPDATES.get(path, (item['sha256'], item['sha256']))
        require(item['sha256'] == before and digest(path) == expected, 'SDK/system input changed: ' + path)
    require(set(SYSTEM_HEADER_UPDATES) <= {v['path'] for v in rows}, 'unknown header update')
    packages = subprocess.check_output(['dpkg-query', '-W', '-f=${Package} ${Version}\n',
        'libmetee-dev', 'libigdfcl2'], text=True, timeout=10).splitlines()
    require(packages == ['libigdfcl2 2.41.15-1~24.04', 'libmetee-dev 6.2.5-1~24.04~ppa2'], 'updated header packages changed')
    return dict(path=str(lineage.SDK), sha256=lineage.SDK_SHA, original_files=8697, admitted_files=8697,
        prospective_system_header_updates={k: list(v) for k, v in SYSTEM_HEADER_UPDATES.items()},
        updated_header_packages=packages,
        condition='New v2 input snapshot only;all8694 other original hashes remain mandatory. Actual dependency closure records which updated headers were compiled. No qualification inherited for the new fixture.')


def external_inputs(rt):
    mkl = Path('/opt/intel/oneapi/mkl/2026.1')
    paths = {p.resolve() for d in ('include', 'lib/cmake/mkl') for p in (mkl / d).rglob('*') if p.is_file()}
    paths.update(Path(p) for p in rt['runtime_files'])
    # Imported MKL link names can resolve to their versioned runtime libraries.
    paths.update(p.resolve() for p in (mkl / 'lib').glob('libmkl*.so') if p.is_file())
    return {str(p): digest(p) for p in sorted(paths)}


def compact(row):
    out = {k: v for k, v in row.items() if k not in
           ('project', 'dependencies', 'external_inputs', 'commands')}
    out['source_groups'] = [{k: g[k] for k in ('root', 'revision', 'count', 'aggregate_sha256')}
                            for g in [row['project'], *row['dependencies']]]
    return out


def admit(row):
    require(build.pin(P, PREFIXES)['source_sha256'] == row['project']['source_sha256'], 'committed build inputs changed')
    for dep in row['dependencies']:
        require(build.pin(Path(dep['root']), ['.']) == dep, 'independent dependency changed')
    require(sdk_inputs() == row['sdk'], 'SDK inputs changed')
    for path, sha in row['external_inputs'].items():
        require(digest(path) == sha, 'external input changed: ' + path)
    for tool in row['tools'].values():
        require(digest(tool['path']) == tool['sha256'], 'build tool changed')
    rt = r.validation.committed(r.RESULT)
    require(digest(r.RESULT) == row['runtime_sha256'], 'runtime qualification changed')
    require(r.current(rt)['binary'] == row['selected_binary'], 'selected fallback changed')


def dependencies(row, text):
    expected = {}
    for group in [row['project'], *row['dependencies']]:
        expected.update({str((Path(group['root']) / n).resolve()): h
                         for n, h in group['source_sha256'].items()})
    expected.update({str(Path(v['path']).resolve()): SYSTEM_HEADER_UPDATES.get(v['path'], (None, v['sha256']))[1]
                     for v in json.loads(lineage.SDK.read_text())})
    expected.update(row['external_inputs'])
    paths = {Path(line.strip()) for line in text.splitlines() if line.startswith('    ') and line.strip()}
    paths = {(p if p.is_absolute() else ROOT / 'build' / p).resolve() for p in paths}
    required = {'src/sycl/command_profile.cpp', 'src/prefill/sycl/gemm16.cpp',
                'tests/sycl/command_profile.cpp', 'bench/sycl/moe_gu_swiglu.cpp', 'bench/sycl/moe_down_scatter.cpp'}
    require({P / n for n in required} <= paths and len(paths) >= 100, 'incomplete actual dependency closure')
    observed, generated = {}, {}
    for path in sorted(paths):
        sha = digest(path)
        if str(path) in expected:
            require(sha == expected[str(path)], 'compiled source differs: ' + str(path))
        else:
            require(str(path) in row['generated_expected'] and sha == row['generated_expected'][str(path)],
                    'unplanned generated/compiler input: ' + str(path))
            generated[str(path)] = sha
        observed[str(path)] = sha
    manifest = ROOT / 'compiler-inputs.json'
    write(manifest, observed)
    return dict(files=len(observed), manifest=dict(path=str(manifest), sha256=digest(manifest)),
                generated_sha256=generated, uncovered_files=0,
                used_updated_system_headers=sorted(set(observed) & set(SYSTEM_HEADER_UPDATES)))


def prepare():
    require(not ROOT.exists(), 'fresh component build only;inspect existing owner,do not retry')
    container = r.validation.committed(RESULT)
    scope = container['event_reconciliation']['fused_command_profile']
    require(scope['stage'] == 'sources-prepared' and scope['host_controls']['passed'] and
            scope['host_controls']['tests'] == 8 and not scope['GPU_executed'], 'closed committed CPU controls required')
    require(container['event_reconciliation']['oneMKL_wrapper_build_v2']['retirement']['status'] == 'retired',
            'retire obsolete wrapper generation first')
    rt = r.validation.committed(r.RESULT)
    full = r.current(rt)
    project = build.pin(P, PREFIXES)
    deps = [build.pin(Path(g['root']), ['.']) for g in full['dependencies']]
    require(all(a['source_sha256'] == b['source_sha256'] for a, b in zip(deps, full['dependencies'])),
            'selected independent dependencies differ')
    paths = {n: str(Path('/usr/bin/' + n).resolve()) for n in ('cmake', 'ninja', 'cc', 'ar', 'ranlib', 'as', 'ld')}
    paths['cc1'] = subprocess.check_output(['/usr/bin/cc','-print-prog-name=cc1'], text=True, timeout=10).strip()
    paths['sycl'] = full['compiler']['path']
    tools = {n: dict(path=p, sha256=digest(p)) for n, p in paths.items()}
    argv = list(full['configure_argv']); argv[argv.index('-B') + 1] = str(ROOT / 'build')
    require(argv[argv.index('-S') + 1] == str(P), 'unexpected source resolver')
    env = dict(full['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    row = dict(stage='prepared',source_commit=project['revision'],project=project,dependencies=deps,
        sdk=sdk_inputs(),external_inputs=external_inputs(rt),tools=tools,configure_argv=argv,environment=env,
        targets=TARGETS,selected_binary=full['binary'],runtime_sha256=digest(r.RESULT),boot=rt['boot'],driver=rt['driver'],
        generated_expected={str(ROOT / 'build/ggml/src/ggml-version.h'): hashlib.sha256(HEADER.encode()).hexdigest()},
        build_command=['/usr/bin/cmake','--build',str(ROOT / 'build'),'--target',*TARGETS,'-j','4'],
        dependency_command=['/usr/bin/ninja','-C',str(ROOT / 'build'),'-t','deps'],
        active=False,ownership_closed=True,passed=False,GPU_executed=False,engine_built=False,production_selected=False,
        budgets=dict(wall=1800,cpu=7200,rss_cap=12 << 30,text_cap=32 << 20,file_cap=256 << 20,total_cap=4 << 30),
        question='Build only sparse sampler and existing GU/Down fixtures with production-hook profiling modes. '
            'No arithmetic,queue,threshold or decode changes;no model/native engine execution.',
        criteria='Committed source,tools,SDK8697/MKL/dependencies and generated77-byte ggml header;owned configure/build/dependency scan exit0. '
            'Zero uncovered compiler inputs;all3 fixture hashes and imported link providers verified;selected13 artifacts preserved.',
        retention='One extra component generation only for imminent traced sparse/fused-hook qualification. '
            'Keep original6e40+12fixtures untouched;retire normal build logs/objects/JIT after extraction and source recovery.')
    ROOT.mkdir(mode=0o700)
    for name in ('tmp','cache'):
        (ROOT / name).mkdir()
    write(ROOT / 'record.json',row)
    scope['build'] = compact(row)
    write(RESULT,container)
    print('Prepared frozen fused-profile fixture build;commit recipe before build.',flush=True)


def run():
    container = r.validation.committed(RESULT)
    row = json.loads((ROOT / 'record.json').read_text())
    scope = container['event_reconciliation']['fused_command_profile']
    require(scope['build'] == compact(row) and row['stage'] == 'prepared' and not (ROOT / 'owned-build').exists(),
            'single-use committed build;no duplicate/retry')
    admit(row)
    original = copy.deepcopy(row)
    owner = None
    def persist():
        row['active'] = bool(owner and owner.active is not None)
        write(ROOT / 'record.json',dict(row,commands=owner.commands if owner else []))
    def monitor(io, entry):
        next_size = 0
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll()
            if time.monotonic() >= next_size:
                require(r.supervision.artifact_bytes(ROOT) <= row['budgets']['total_cap'], 'whole component build size budget')
                next_size = time.monotonic() + 1
            if select.select([io.proc.stdout],[],[],.02)[0]:
                io.read();io.pending.clear()
        io.exit(1)
        require(r.supervision.artifact_bytes(ROOT) <= row['budgets']['total_cap'], 'final component build size budget')
    try:
        row['stage'] = 'building'
        owner = r.supervision.Owner(ROOT / 'owned-build',persist);persist()
        for label, command in [('configure',row['configure_argv']),('compile',row['build_command']),
                               ('dependencies',row['dependency_command'])]:
            entry,out,err = owner.run(label,command,row['environment'],P,interaction=monitor,**row['budgets'])
            require(r.supervision.closed(entry) and entry['exit_code'] == 0,
                    label + ' failed: ' + (out.read_text(errors='replace') + err.read_text(errors='replace'))[-6000:])
            if label == 'configure':
                admit(original)
            if label == 'dependencies':
                row['compiled_inputs'] = dependencies(row,out.read_text())
        ninja = (ROOT / 'build/build.ninja').read_text()
        origins = {str(Path(v['path']).resolve()): SYSTEM_HEADER_UPDATES.get(v['path'],(None,v['sha256']))[1]
                   for v in json.loads(lineage.SDK.read_text())}
        origins.update(row['external_inputs'])
        fixtures = {}
        for name in TARGETS:
            binary = ROOT / 'build' / name
            fixtures[name] = dict(path=str(binary),sha256=digest(binary),bytes=binary.stat().st_size)
            stanza = ninja[ninja.index('build ' + name + ':'):].split('\n\n',1)[0]
            links = shlex.split(next(line.split(' = ',1)[1] for line in stanza.splitlines()
                                    if line.startswith('  LINK_LIBRARIES = ')))
            for path in (Path(p).resolve() for p in links if p.startswith('/')):
                require(str(path) in origins and digest(path) == origins[str(path)], 'unplanned imported link input: ' + str(path))
        require(not (ROOT / 'build/strata').exists(),'component build must not produce another engine')
        admit(original)
        row.update(stage='built-not-executed',passed=True,fixtures=fixtures)
    except BaseException as error:
        row.update(stage='failed',passed=False,error=repr(error))
    finally:
        row['ownership_closed'] = owner is None or owner.active is None and all(r.supervision.closed(c) for c in owner.commands)
        row['passed'] = bool(row['passed'] and row['ownership_closed'])
        persist();scope['build'] = compact(row);write(RESULT,container)
    print(json.dumps({k:row.get(k) for k in ('stage','passed','ownership_closed','fixtures','error')}),flush=True)
    return int(not row['passed'])


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'],['build']),'build_fused_profile.py prepare|build')
    with (B / 'owned-v0141-measurement.lock').open('r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        r.guards.no_model_owners([])
        sys.exit(prepare() if sys.argv[1] == 'prepare' else run())
