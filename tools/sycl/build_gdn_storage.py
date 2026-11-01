#!/usr/bin/env python3
"""Build an independently authored oneMKL output-storage fixture; MIT.

No production dispatch or GPU execution. Recover the selected generation and
derive its Runtime compiler flags; freeze exact inputs and commands before build.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import select
import shlex
import sys

sys.dont_write_bytecode = True
import retained_prefill_candidate as retained

base, runtime = retained.base, retained.runtime
P, B, RESULT = base.P, base.B, base.RESULT
ROOT = B / 'official-gdn-storage-v1'
KEY = 'official_gdn_storage_fixture_build_v1'
FIXTURE = 'bench/sycl/gdn_projection_storage.cpp'
CONTROLLER = Path(__file__).resolve()
require, digest, write = base.require, base.digest, base.write


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text())
    data[KEY] = base.compact(row)
    write(RESULT, data)


def allowed_inputs():
    original, providers, _ = retained.verify()
    allowed = base.sdk()
    allowed.update(runtime.inventory(original['external_manifest']))
    allowed.update(providers)
    actual = original['compiled_inputs']
    require(digest(actual['path']) == actual['sha256'], 'original input manifest changed')
    allowed.update(json.loads(Path(actual['path']).read_text()))
    for name, sha in original['native_sources'].items():
        path = (retained.ROOT / 'source' / name).resolve()
        require(digest(path) == sha, 'retained native input changed')
        allowed[str(path)] = sha
    allowed[str(P / FIXTURE)] = digest(P / FIXTURE)
    return original, allowed


def source_controls(row):
    require(base.controls(row['source_commit'], controller=CONTROLLER) == row['controller_inputs'],
            'committed build controls changed')
    for name, sha in row['native_sources'].items():
        raw = base.basis.build.git(P, 'show', row['source_commit'] + ':' + name)
        require(hashlib.sha256(raw).hexdigest() == sha == digest(P / name),
                'committed fixture/Runtime source changed')


def prepare():
    require(KEY not in base.committed(RESULT) and not ROOT.exists(), 'fresh storage fixture only')
    original, _ = allowed_inputs()
    revision = base.basis.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    pins = base.controls(revision, controller=CONTROLLER)
    names = (FIXTURE, 'src/sycl/runtime.cpp')
    native = {name: hashlib.sha256(base.basis.build.git(P, 'show', revision + ':' + name)).hexdigest()
              for name in names}
    require(all(digest(P / n) == sha for n, sha in native.items()), 'commit exact native inputs')
    runtime_source = retained.ROOT / 'source/src/sycl/runtime.cpp'
    require(digest(runtime_source) == native['src/sycl/runtime.cpp'], 'Runtime differs from selected generation')
    commands = json.loads((retained.ROOT / 'build/compile_commands.json').read_text())
    matches = [v for v in commands if v['file'] == str(runtime_source)]
    require(len(matches) == 1, 'one actual production Runtime recipe required')
    production = shlex.split(matches[0]['command'])
    require(production[0] == base.COMPILER, 'original compiler command changed')
    flags = production[1:production.index('-o')]
    build = ROOT / 'build'
    runtime_command = [base.COMPILER, *flags, '-MD', '-MF', str(build / 'runtime.d'),
        '-o', str(build / 'runtime.o'), '-c', str(runtime_source)]
    fixture_command = [base.COMPILER, *flags, '-isystem', '/opt/intel/oneapi/mkl/2026.1/include',
        '-DMKL_ILP64', '-fno-fast-math', '-ffp-contract=off', '-MD', '-MF', str(build / 'fixture.d'),
        '-o', str(build / 'fixture.o'), '-c', str(P / FIXTURE)]
    mkl = Path('/opt/intel/oneapi/mkl/2026.1/lib')
    libraries = [str(mkl / v) for v in ('libmkl_sycl_blas.so', 'libmkl_intel_ilp64.so',
                                      'libmkl_sequential.so', 'libmkl_core.so')]
    link = [base.COMPILER, '-O3', '-fsycl', '-fsycl-id-queries-range=size_t',
        '-fsycl-allow-device-image-dependencies', '-Wl,-t', str(build / 'fixture.o'),
        str(build / 'runtime.o'), '-Wl,-rpath=' + str(mkl), *libraries,
        '-lm', '-ldl', '-lpthread', '-o', str(build / 'gdn-storage')]
    ROOT.mkdir(mode=0o700)
    for part in ('build', 'tmp', 'cache'): (ROOT / part).mkdir()
    row = dict(stage='prepared', source_commit=revision, controller_inputs=pins, native_sources=native,
        selected_native_source=retained.NATIVE, selected_engine=original['binaries']['strata'],
        original_compile_database=dict(path=str(retained.ROOT / 'build/compile_commands.json'),
            sha256=digest(retained.ROOT / 'build/compile_commands.json')),
        original_Runtime_command=matches[0],
        commands=[['fixture-object', fixture_command], ['runtime-object', runtime_command], ['compile', link]],
        environment=dict(original['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache')),
        budgets=dict(wall=300, cpu=900, rss_cap=5 << 30, text_cap=16 << 20,
                     file_cap=256 << 20, total_cap=1 << 30),
        plan=dict(scope='CPU compile/link only;new resident oneMKL storage diagnostic,not model qualification.',
            arms=['oneMKL half/half/float', 'oneMKL half/half/half', 'oneMKL half/half/float plus explicit half conversion'],
            numerical='Finite normal half operands,certified absolute dot sum <65504;FP32-error-plus-half-rounding diagnostic,not proof of vendor accumulation type or model quality.',
            next='Commit a separately bounded canary/traced smoke before device validation;require its gates before clean3-process speed cohort.',
            speed='4096/8192 rows x6144/10240 columns,K2560;3fresh rotated-order processes,5x128-call batches per arm/shape;first calls separate;100ms floor and <=5%empty-control overhead.',
            retained='Selected eight-ELF engine is unchanged fallback. This one fixture is for the imminent GDN3 storage decision;retire it if rejected.'),
        active=False, ownership_closed=True, passed=False, GPU_executed=False, commands_executed=[])
    source_controls(row)
    save(row)
    print('Prepared CPU-only storage fixture;commit recipe before build.', flush=True)


def build():
    row = json.loads((ROOT / 'record.json').read_text())
    require(base.committed(RESULT)[KEY] == base.compact(row) and row['stage'] == 'prepared' and
            not (ROOT / 'owned-build').exists(), 'fresh committed build required')
    source_controls(row)
    _, allowed = allowed_inputs()
    asset = row['original_compile_database']
    require(digest(asset['path']) == asset['sha256'], 'original compiler recipe changed')
    owner = None
    def persist():
        write(ROOT / 'record.json', dict(row, commands_executed=owner.commands if owner else [],
            active=bool(owner and owner.active)))
    def monitor(io, entry):
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll()
            require(base.supervision.artifact_bytes(ROOT) <= row['budgets']['total_cap'], 'build byte budget')
            if select.select([io.proc.stdout], [], [], .02)[0]: io.read(); io.pending.clear()
        io.exit(1)
    try:
        row['stage'] = 'building'
        owner = base.supervision.Owner(ROOT / 'owned-build', persist)
        for label, argv in row['commands']:
            entry, out, err = owner.run(label, argv, row['environment'], ROOT,
                interaction=monitor, **row['budgets'])
            require(base.supervision.closed(entry) and entry['exit_code'] == 0,
                    label + ' failed: ' + (out.read_text(errors='replace') + err.read_text(errors='replace'))[-4000:])
        found = set()
        for name in ('fixture.d', 'runtime.d'):
            for line in (ROOT / 'build' / name).read_text().replace('\\\n', '').splitlines():
                if not line.strip(): continue
                left, colon, right = line.partition(':')
                require(colon and left.strip(), 'invalid compiler depfile')
                found.update(str((ROOT / p).resolve()) for p in shlex.split(right))
        require(all(allowed.get(p) == digest(p) for p in found), 'unfrozen actual compiler input')
        write(ROOT / 'compiler-inputs.json', {p: allowed[p] for p in sorted(found)})
        row['actual_compiler_inputs'] = dict(path=str(ROOT / 'compiler-inputs.json'),
            sha256=digest(ROOT / 'compiler-inputs.json'), files=len(found), uncovered=0)
        row['actual_link_inputs'] = base.traced_link_inputs(allowed, ROOT)
        binary = ROOT / 'build/gdn-storage'
        row.update(stage='built-not-executed', passed=True,
            binary=dict(path=str(binary), sha256=digest(binary), bytes=binary.stat().st_size))
        retained.verify()
    except BaseException as error:
        row.update(stage='failed', passed=False, error=repr(error))
    finally:
        row.update(active=bool(owner and owner.active), commands_executed=owner.commands if owner else [],
            ownership_closed=bool(owner and all(base.supervision.closed(v) for v in owner.commands)))
        require(owner is None or owner.active is None, 'build ownership unresolved')
        save(row)
    print(json.dumps({k: row.get(k) for k in ('stage', 'passed', 'error')}), flush=True)
    require(row['passed'], 'storage fixture build failed')


def verify():
    row = json.loads((ROOT / 'record.json').read_text())
    require(base.committed(RESULT)[KEY] == base.compact(row) and row['passed'] and
            row['stage'] == 'built-not-executed' and row['ownership_closed'] and not row['active'] and
            all(base.supervision.closed(v) for v in row['commands_executed']), 'closed committed build required')
    source_controls(row)
    _, allowed = allowed_inputs()
    asset = row['actual_compiler_inputs']
    require(digest(asset['path']) == asset['sha256'], 'compiler input manifest changed')
    actual = json.loads(Path(asset['path']).read_text())
    require(len(actual) == asset['files'] and all(allowed.get(p) == sha == digest(p)
            for p, sha in actual.items()), 'actual compiler inputs changed')
    require(row['actual_link_inputs']['uncovered_external_inputs'] == 0 and all(
            allowed.get(p) == sha == digest(p) for p, sha in
            row['actual_link_inputs']['frozen_inputs'].items()), 'actual linker inputs changed')
    asset = row['binary']
    require(digest(asset['path']) == asset['sha256'] and Path(asset['path']).stat().st_size == asset['bytes'],
            'actual fixture executable changed')
    return row


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'build', 'verify'), 'use prepare|build|verify')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.guards.no_model_owners([])
        if sys.argv[1] == 'prepare': prepare()
        elif sys.argv[1] == 'build': build()
        else:
            row = verify()
            print(json.dumps(dict(source_commit=row['source_commit'], binary=row['binary'], GPU_executed=False)))
