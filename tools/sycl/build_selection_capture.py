#!/usr/bin/env python3
"""Main-owned separate checked-dump build; never overwrites the selected engine.

Repository LICENSE applies. No GPU execution. prepare writes a frozen plan;
commit its compact reference before build. All diagnostic execution needs a
separate committed admission plan after independent source review.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
import build_engine as build
import generation_engine as gen
import measurement_guards as guards
import measurement_owner as supervision
import resource_engine as resource
import selection_capture_overlay as overlay

P, B, A = gen.P, gen.B, gen.A
ROOT = B / 'qsa-selection-native-v1'
RESULT = A / 'qsa-selection-build-current.json'
PYTHON = '/usr/bin/python3'


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def fingerprint(pins):
    return hashlib.sha256(json.dumps(pins, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def current(plan):
    for source in [plan['project'], *plan['dependencies']]:
        recovered = {name: hashlib.sha256(raw).hexdigest() for name, raw in
            build.blobs(Path(source['root']), source['revision'], source['source_sha256'])}
        guards.require(recovered == source['source_sha256'], 'build inputs not recoverable from Git')
        guards.require(all(build.digest(Path(source['root']) / n) == h for n, h in recovered.items()),
                       'build source changed')
    guards.require(build.digest(plan['compiler']['path']) == plan['compiler']['sha256'], 'compiler changed')
    guards.require(all(build.digest(n) == h for n, h in plan['sdk_headers'].items()), 'SDK headers changed')
    guards.require(build.digest(plan['overlay']['path']) == plan['overlay']['sha256'] and
        Path(plan['overlay']['path']).read_bytes() == overlay.generated(P), 'native source overlay changed')
    selected, proof, _ = resource.qualification()
    guards.require(proof == plan['selected_qualification'] and selected['binary'] == plan['selected_binary'],
                   'selected fallback qualification changed')


def prepare():
    guards.require(not ROOT.exists() and not RESULT.exists(), 'single-use fresh diagnostic build only')
    full, proof, _ = resource.qualification()
    project = build.pin(P, ['CMakeLists.txt', 'cmake', 'include', 'src', 'tests', 'third_party', 'bench/sycl',
        'tools/sycl/build_selection_capture.py', 'tools/sycl/selection_capture_overlay.py',
        'tools/sycl/build_engine.py', 'tools/sycl/measurement_owner.py', 'tools/sycl/measurement_guards.py',
        'tools/sycl/measurement_environment.json'])
    dependencies = [build.pin(Path(s['root']), ['.']) for s in full['dependencies']]
    compiler = full['compiler']
    header_roots = [Path(compiler['path']).parent.parent / 'include', Path('/opt/intel/oneapi/mkl/2026.1/include')]
    headers = sorted({p for root in header_roots for p in root.rglob('*') if p.is_file()})
    guards.require(0 < len(headers) < 30000, 'bounded installed SDK inventory')
    sdk = {str(p): build.digest(p) for p in headers}
    native = overlay.generated(P)
    ROOT.mkdir(mode=0o700)
    for name in ('cache', 'tmp'): (ROOT / name).mkdir()
    (ROOT / 'prefill.cpp').write_bytes(native)
    argv = list(full['configure_argv'])
    argv[argv.index('-B') + 1] = str(ROOT / 'build')
    argv += ['-DCMAKE_CXX_COMPILER_LAUNCHER=' + ';'.join(
        [PYTHON, str(P / 'tools/sycl/selection_capture_overlay.py'), 'compile', str(ROOT / 'plan.json')])]
    fixture = [compiler['path'], '-fsycl', '-fsycl-id-queries-range=size_t', '-DSTRATA_ENABLE_SYCL=1',
        '-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror', '-I' + str(P / 'include'),
        '-I' + str(P / 'include/strata/sycl/compat'), str(P / 'tests/sycl/selection_dump.cpp'),
        str(ROOT / 'build/libstrata_sycl_runtime.a'), '-o', str(ROOT / 'selection_dump_test')]
    env = dict(full['environment'], XDG_CACHE_HOME=str(ROOT / 'cache'), TMPDIR=str(ROOT / 'tmp'))
    plan = dict(root=str(ROOT), source_commit=project['revision'], project=project, dependencies=dependencies,
        compiler=compiler, sdk_headers=sdk, configure_argv=argv, fixture_command=fixture, environment=env,
        overlay=dict(path=str(ROOT / 'prefill.cpp'), sha256=hashlib.sha256(native).hexdigest(), bytes=len(native),
            generator_sha256=build.digest(P / 'tools/sycl/selection_capture_overlay.py')),
        selected_binary=full['binary'], selected_qualification=proof,
        native_changes='Only src/prefill/prefill.cpp is substituted: checked optional dump/helper. '
            'Production source files and selected engine/12fixtures/decode are unchanged. '
            'This diagnostic package is not adopted and does not itself prove unchanged decode performance.',
        commands=[('configure', argv), ('engine-and-runtime', ['/usr/bin/cmake', '--build', str(ROOT / 'build'),
            '--target', 'strata', 'sycl_compat_test', '--parallel', '2']), ('readback-fixture', fixture)])
    current(plan)
    write(ROOT / 'plan.json', plan)
    compact = dict(stage='prepared', source_commit=project['revision'], root=str(ROOT),
        plan_sha256=build.digest(ROOT / 'plan.json'), compiler=compiler, overlay=plan['overlay'],
        project_fingerprint=project['aggregate_sha256'],
        dependency_fingerprints=[s['aggregate_sha256'] for s in dependencies],
        sdk_headers=dict(count=len(sdk), fingerprint=fingerprint(sdk)),
        targets=['strata', 'sycl_compat_test', 'selection_dump_test'],
        complete=False, passed=False, active=False, ownership_closed=True, GPU_executed=False,
        model_adoption_qualified=False, clean_speed_qualified=False,
        retention='One additional diagnostic generation for bounded checked-readback qualification and '
            'one real-QSA-ID capture. Retire it after verified extraction/analysis; original6e40 remains selected. '
            'No raw ID retention without a named committed replay consumer.')
    write(RESULT, compact)
    print('Prepared separate checked-selection native build; commit its plan reference before build.', flush=True)


def compile_build():
    record = gen.committed(RESULT)
    guards.require(record['stage'] == 'prepared' and not (ROOT / 'record.json').exists(), 'fresh committed build only')
    guards.require(build.digest(ROOT / 'plan.json') == record['plan_sha256'], 'frozen build plan changed')
    plan = json.loads((ROOT / 'plan.json').read_text()); current(plan)
    record.update(stage='building', active=True, ownership_closed=False)
    owner = None
    def persist():
        write(ROOT / 'record.json', dict(record, commands=owner.commands if owner else []))
    owner = supervision.Owner(ROOT / 'owned', persist); persist()
    try:
        for label, argv in plan['commands']:
            entry, _, _ = owner.run(label, argv, plan['environment'], P, wall=1800, cpu=21600,
                rss_cap=32 << 30, text_cap=32 << 20, file_cap=512 << 20, total_cap=3 << 30)
            guards.require(supervision.closed(entry) and entry['exit_code'] == 0, 'diagnostic build failed: ' + label)
        current(plan)
        substituted = json.loads((ROOT / 'prefill-compile.json').read_text())
        guards.require(substituted['source_sha256'] == plan['overlay']['sha256'] and
            str(ROOT / 'prefill.cpp') in substituted['argv'], 'actual prefill substitution not recorded')
        record['artifacts'] = {n: dict(path=str(p), bytes=p.stat().st_size, sha256=build.digest(p)) for n, p in
            [('engine', ROOT / 'build/strata'), ('runtime', ROOT / 'build/sycl_compat_test'),
             ('readback', ROOT / 'selection_dump_test')]}
        record['compile_database_sha256'] = build.digest(ROOT / 'build/compile_commands.json')
        record['actual_prefill_command'] = substituted
        record.update(passed=True, complete=True)
    except BaseException as error:
        record['error'] = repr(error)
    finally:
        record.update(stage='closed', active=False,
            ownership_closed=owner.active is None and all(supervision.closed(c) for c in owner.commands))
        persist(); write(RESULT, record)
    print(json.dumps({k: record.get(k) for k in ('passed', 'complete', 'ownership_closed', 'error')}), flush=True)
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    guards.require(sys.argv[1:] in (['prepare'], ['build']), 'use prepare|build')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([]); guards.driver_guard()
        sys.exit(prepare() if sys.argv[1] == 'prepare' else compile_build())
