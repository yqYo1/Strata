#!/usr/bin/env python3
"""Build the pinned original-plus-observer engine after real-pool parity.

Repository LICENSE. Reuse the exact admitted source/configuration; do not build
the main worktree's unselected inference changes or run a GPU/model here.
"""
import copy
import fcntl
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import cpu_service_engine as pool

gen, native, runtime = pool.gen, pool.native, pool.runtime
P, B, ROOT, RESULT = pool.P, pool.B, pool.ROOT, pool.RESULT
RECORD = ROOT / 'native-record.json'
TARGETS = ('strata', 'sycl_submission_test', 'sycl_graph_envelope_test')
require, digest, write = pool.require, pool.digest, pool.write


def compact(row):
    return {k: v for k, v in row.items() if k not in ('commands', 'actual_compiled_inputs')}


def host_gate():
    row = json.loads((ROOT / 'record.json').read_text())
    pool.current(row)
    require(row['stage'] == 'pool-checked' and row['passed'] and row['complete'] and
        row['production_pool_checked'] and row['checksum_match'] and row['ownership_closed'] and
        not row['active'] and [case['observer'] for case in row['outcomes']] == [0, 1],
        'closed original-source real-pool parity required')
    require(digest(row['fixture']['path']) == row['fixture']['sha256'] and
        len(row['actual_compiled_inputs']) == row['dependency_closure']['files'] == 612 and
        native.fingerprint(row['actual_compiled_inputs']) == row['dependency_closure']['sha256'],
        'pool fixture/compiler closure changed')
    for path, sha in row['actual_compiled_inputs'].items():
        require(digest(path) == sha, 'pool compiled input changed: ' + path)
    return row


def prepare():
    container = gen.committed(RESULT)
    require('native_engine' not in container and not RECORD.exists() and
        not (ROOT / 'build/strata').exists(), 'fresh additional engine only; no overwrite/retry')
    host = host_gate()
    revision = gen.git('rev-parse', 'HEAD').decode().strip()
    pins = gen.benchmark_pins(revision)
    require('tools/sycl/cpu_service_native.py' in pins, 'commit exact native controller first')
    cache = (ROOT / 'build/CMakeCache.txt').read_text()
    for key, expected in (('STRATA_CPU_SERVICE_PROFILE', 'ON'), ('STRATA_ENABLE_SYCL', 'ON'),
            ('CMAKE_HOME_DIRECTORY', str(ROOT / 'source')), ('CMAKE_CXX_COMPILER', host['compiler']['path'])):
        value = re.findall(r'^' + key + r':[^=\n]+=([^\n]+)$', cache, re.M)
        require(value == [expected], 'admitted CMake setting differs: ' + key)
    protected = [ROOT / 'build' / name for name in ('CMakeCache.txt', 'compile_commands.json', 'build.ninja')]
    protected += [Path(host['fixture']['path']), ROOT / 'build/libstrata_kernels_cpu.a']
    row = dict(stage='prepared', source_commit=revision, base_native_source=host['base_native_source'],
        observer_source_reference=dict(source_commit=host['source_commit'],
            source_files=host['source_files'], source_fingerprint=host['source_fingerprint'],
            patches=host['patches'], changed_files=host['changed_native_files']),
        host_gate_reference=dict(path=str(ROOT / 'record.json'), sha256=digest(ROOT / 'record.json'),
            fixture=host['fixture'], compiler_closure=host['dependency_closure']),
        benchmark_files=len(pins), benchmark_fingerprint=native.fingerprint(pins),
        protected_inputs={str(path): digest(path) for path in protected},
        build_argv=['/usr/bin/cmake', '--build', str(ROOT / 'build'), '--target', *TARGETS, '-j', '4'],
        census_argv=['/usr/bin/ninja', '-C', str(ROOT / 'build'), '-t', 'deps'],
        environment=host['environment'], runtime_reference=host['runtime_reference'],
        budgets=dict(wall=1800, cpu=14400, rss_cap=16 << 30, text_cap=32 << 20,
            file_cap=64 << 20, total_cap=3 << 30),
        criteria='Compile exactly the already admitted original6e40 plus four scoped CPU-observer/test patches. Reuse existing CMake/CPU objects without changing settings;all three targets must link. Require complete actual compiler closure including generate.cpp/mtp.cpp,pinned source/SDK/MKL/generated header,unchanged CPU fixture/archive/configuration and normal owned exits. No engine/fixture invocation at this step.',
        limits='Successful linking is not native device/model qualification,observer overhead measurement,hardware attribution,throughput gain or inference adoption. Keep original selected6e40 and12same-build fixtures untouched.',
        retention='One extra generation shares the host-pool build for imminent bounded validated device/model checks,then observer overhead and same-request service/census attribution. Retire on rejection/closure after exact source/recipe recovery;normal build output is disposable after recorded closure.',
        passed=False, complete=False, active=False, ownership_closed=True,
        GPU_executed=False, model_executed=False, production_selected=False, assets={})
    write(RECORD, row)
    container['native_engine'] = compact(row)
    write(RESULT, container)
    print('Prepared three-target observer engine build; commit plan before compilation.', flush=True)


def current(row):
    require(gen.committed(RESULT)['native_engine'] == compact(row), 'commit exact native build plan')
    host = host_gate()
    require(digest(ROOT / 'record.json') == row['host_gate_reference']['sha256'] and
        host['environment'] == row['environment'] and host['runtime_reference'] == row['runtime_reference'],
        'original pool source/environment/reference changed')
    pins = gen.benchmark_pins(row['source_commit'])
    require(len(pins) == row['benchmark_files'] and native.fingerprint(pins) == row['benchmark_fingerprint'],
        'native build controller/helper changed')
    for path, sha in row['protected_inputs'].items():
        require(digest(path) == sha, 'admitted configuration/CPU asset changed: ' + path)
    return host


def build():
    container = gen.committed(RESULT)
    row = json.loads(RECORD.read_text())
    host = current(row)
    require(row['stage'] == 'prepared' and not (ROOT / 'native-owned').exists(), 'single-use native build')
    initial = copy.deepcopy(row)
    owner = None

    def persist():
        write(RECORD, dict(row, active=bool(owner and owner.active is not None),
            commands=owner.commands if owner else []))

    try:
        owner = native.supervision.Owner(ROOT / 'native-owned', persist)
        for label, command in (('build', row['build_argv']), ('compiler-census', row['census_argv'])):
            entry, out, err = owner.run(label, command, row['environment'], P, **row['budgets'])
            require(native.supervision.closed(entry) and entry['exit_code'] == 0,
                label + ' failed: ' + (out.read_text(errors='replace') + err.read_text(errors='replace'))[-4096:])
            if label == 'compiler-census':
                closure = copy.deepcopy(host)
                row['dependency_closure'] = pool.dependency_closure(closure, out.read_text())
                actual = closure['actual_compiled_inputs']
                require(all(str(ROOT / 'source' / name) in actual for name in
                    ('src/program/generate.cpp', 'src/core/mtp.cpp')), 'native engine compilation missing')
                require(not any(Path(path).is_relative_to(P) for path in actual), 'main worktree source leaked into build')
                row['actual_compiled_inputs'] = actual
            current(initial)
        row['assets'] = {name: dict(path=str(ROOT / 'build' / name),
            bytes=(ROOT / 'build' / name).stat().st_size, sha256=digest(ROOT / 'build' / name)) for name in TARGETS}
        row.update(passed=True, complete=True, stage='linked')
    except BaseException as error:
        row.update(passed=False, complete=False, stage='failed', error=repr(error))
    finally:
        row['ownership_closed'] = owner is None or owner.active is None and all(native.supervision.closed(e) for e in owner.commands)
        row['passed'] = row['complete'] = bool(row['passed'] and row['ownership_closed'])
        row['active'] = False
        row['command_outcomes'] = [dict(label=e['label'], exit_code=e.get('exit_code'),
            ownership_closed=native.supervision.closed(e)) for e in (owner.commands if owner else [])]
        persist()
        container['native_engine'] = compact(row)
        write(RESULT, container)
    print(json.dumps({key: row.get(key) for key in ('stage', 'passed', 'ownership_closed', 'error')}), flush=True)
    return int(not row['passed'])


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'], ['build']), 'prepare|build')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        native.guards.no_model_owners([])
        if sys.argv[1] == 'prepare':
            prepare()
        else:
            sys.exit(build())
