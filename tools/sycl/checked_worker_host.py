#!/usr/bin/env python3
"""Bounded CPU-only adaptive-worker lifetime regression; repository LICENSE.

Pin/preprocess, compile and execute at separate committed boundaries. This
fixture does not initialize SYCL, enqueue DMA or qualify any model route.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
import build_official_model as base

P, B, RESULT = base.P, base.B, base.RESULT
ROOT, KEY = B / 'checked-adaptive-worker-host-v1', 'checked_adaptive_worker_host_v1'
CPP = 'tests/sycl/checked_worker_host.cpp'
PROJECT = (CPP, 'include/strata/sycl/checked_worker.hpp', 'include/strata/sycl/checked_cleanup.hpp')
require, digest, write = base.require, base.digest, base.write
BUDGET = dict(wall=120, cpu=240, rss_cap=2 << 30, text_cap=1 << 20,
              file_cap=32 << 20, total_cap=64 << 20)
EXPECTED = ('checked_worker_host=PASS success_no_drain=1 failed_work_drains=4 '
            'main_unwind_joins=5 failed_drain_retains_owners=1')


def compact(row):
    result = {k: v for k, v in row.items() if k not in ('sources', 'commands_executed')}
    result['sources'] = dict(files=len(row['sources']), aggregate_sha256=hashlib.sha256(
        json.dumps(row['sources'], sort_keys=True).encode()).hexdigest())
    return result


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text())
    data[KEY] = compact(row)
    write(RESULT, data)


def sources(revision):
    pins = base.controls(revision, controller=Path(__file__).resolve())
    for name in PROJECT:
        raw = base.basis.build.git(P, 'show', revision + ':' + name)
        require((P / name).read_bytes() == raw, 'commit exact host test input: ' + name)
        pins[name] = hashlib.sha256(raw).hexdigest()
    return pins


def dependencies(path):
    raw = path.read_text().replace('\\\n', ' ')
    return {str(Path(value).resolve()) for value in shlex.split(raw.split(':', 1)[1])}


def compiler_inputs(row):
    require(sources(row['source_commit']) == row['sources'], 'host controls/source changed')
    allowed = base.sdk()
    inv = row['external_manifest']
    require(digest(inv['path']) == inv['sha256'], 'external compiler inventory changed')
    external = json.loads(Path(inv['path']).read_text())
    require(len(external) == inv['files'], 'external compiler inventory shape changed')
    allowed.update(external)
    allowed.update(row['host_link_inputs']['files'])
    for name in PROJECT:
        allowed[str((P / name).resolve())] = row['sources'][name]
    require(base.host_link_inventory.inventory(base.COMPILER, row['environment']) ==
            row['host_link_inputs'], 'actual compiler/linker tools changed')
    return allowed


def current(row):
    require(base.committed(RESULT)[KEY] == compact(row), 'commit exact host test recipe/results')
    allowed = compiler_inputs(row)
    inv = row['dependencies']
    require(digest(inv['path']) == inv['sha256'], 'host compiler-input map changed')
    deps = json.loads(Path(inv['path']).read_text())
    require(len(deps) == inv['files'] and all(allowed.get(p) == sha == digest(p)
            for p, sha in deps.items()), 'actual host compiler input changed')
    if 'binary' in row:
        require(row['actual_link_inputs']['uncovered_external_inputs'] == 0 and
                digest(row['binary']['path']) == row['binary']['sha256'] and
                all(allowed.get(p) == sha == digest(p)
                    for p, sha in row['actual_link_inputs']['frozen_inputs'].items()),
                'host ELF/link inputs changed')
    return allowed


def prepare():
    data = base.committed(RESULT)
    require(KEY not in data and not ROOT.exists(), 'one fresh host fixture')
    revision = base.basis.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    pins = sources(revision)
    native = data['cache_map_model_build_v1']
    require(native['passed'] and native['ownership_closed'] and not native['active'],
            'closed source/compiler basis required;no inference qualification inherited')
    ROOT.mkdir(mode=0o700)
    for part in ('tmp', 'cache'): (ROOT / part).mkdir()
    env = dict(PATH='/usr/bin:/bin', HOME=str(Path.home()), LANG='C.UTF-8', LC_ALL='C.UTF-8',
        LD_LIBRARY_PATH=native['environment']['LD_LIBRARY_PATH'], TMPDIR=str(ROOT / 'tmp'),
        XDG_CACHE_HOME=str(ROOT / 'cache'))
    flags = [base.COMPILER, '-std=c++20', '-O2', '-pthread', '-I' + str(P / 'include')]
    pre = [*flags, '-M', '-MF', str(ROOT / 'prepare.d'), str(P / CPP)]
    command = [*flags, '-MD', '-MF', str(ROOT / 'build.d'), str(P / CPP),
               '-Wl,-t', '-o', str(ROOT / 'fixture')]
    row = dict(stage='preparing', source_commit=revision, sources=pins,
        compiler=native['compiler'], sdk=native['sdk'], external_manifest=native['external_manifest'],
        host_link_inputs=base.host_link_inventory.inventory(base.COMPILER, env),
        environment=env, preprocess_command=pre, compile_command=command,
        budgets=BUDGET, GPU_executed=False, model_executed=False, passed=False,
        active=False, ownership_closed=True, commands_executed=[],
        contract='Five worker outcomes (success,false,std exception,nonstd exception,bad_alloc),main unwind joins before owners in all five,and child failed-drain fail-stop without owner unwind/atexit. No real DMA,cache-free,model,speed or262144 qualification.')
    owner = None
    try:
        allowed = compiler_inputs(row)
        owner = base.supervision.Owner(ROOT / 'prepare-owned')
        entry, _, _ = owner.run('headers', pre, env, P, **BUDGET)
        require(base.supervision.closed(entry) and entry['exit_code'] == 0, 'host preprocessing failed')
        deps = dependencies(ROOT / 'prepare.d')
        require(all(allowed.get(p) == digest(p) for p in deps), 'unplanned host compiler input')
        write(ROOT / 'compiler-inputs.json', {p: allowed[p] for p in sorted(deps)})
        row.update(stage='prepared', dependencies=dict(path=str(ROOT / 'compiler-inputs.json'),
            sha256=digest(ROOT / 'compiler-inputs.json'), files=len(deps)))
    except BaseException as error:
        row.update(stage='failed', error=repr(error))
    finally:
        row.update(active=bool(owner and owner.active), commands_executed=owner.commands if owner else [],
            ownership_closed=bool(owner and owner.active is None and
                all(base.supervision.closed(v) for v in owner.commands)))
        save(row)
    require(row['stage'] == 'prepared' and row['ownership_closed'], row.get('error', 'preparation failed'))
    print('Prepared CPU-only worker fixture;commit before compilation.', flush=True)


def execute(mode):
    row = json.loads((ROOT / 'record.json').read_text())
    allowed = current(row)
    require(row['stage'] == ('prepared' if mode == 'build' else 'built') and
            not (ROOT / ('owned-build' if mode == 'build' else 'owned-check')).exists(),
            'one fresh frozen host phase')
    owner = None
    try:
        owner = base.supervision.Owner(ROOT / ('owned-build' if mode == 'build' else 'owned-check'))
        command = row['compile_command'] if mode == 'build' else [row['binary']['path']]
        entry, out, err = owner.run('compile' if mode == 'build' else 'check', command,
            row['environment'], P, **BUDGET)
        require(base.supervision.closed(entry) and entry['exit_code'] == 0,
                'host phase failed: ' + err.read_text(errors='replace')[-3500:])
        current(row)
        if mode == 'build':
            require(dependencies(ROOT / 'build.d') == set(json.loads(
                (ROOT / 'compiler-inputs.json').read_text())), 'actual host compiler dependency set changed')
            dynamic = subprocess.check_output(['/usr/bin/readelf', '-d', str(ROOT / 'fixture')],
                text=True, timeout=10)
            require(not any(name in dynamic.lower() for name in
                ('libsycl', 'libze_', 'libopencl', 'libur_')), 'CPU fixture unexpectedly links a GPU runtime')
            row.update(stage='built', binary=dict(path=str(ROOT / 'fixture'),
                sha256=digest(ROOT / 'fixture'), bytes=(ROOT / 'fixture').stat().st_size),
                actual_link_inputs=base.traced_link_inputs(allowed, ROOT),
                no_GPU_runtime_needed=True)
        else:
            require(out.read_text().strip() == EXPECTED and
                err.read_text().count('Strata SYCL worker:') == 4 and
                err.read_text().count('Strata SYCL cleanup:') == 1,
                'host worker/failed-drain coverage differs')
            row.update(stage='closed', passed=True, coverage=EXPECTED,
                native_exit_code=0, expected_failed_drain_child_exit=1)
    except BaseException as error:
        row.update(stage='failed', passed=False, error=repr(error))
    finally:
        row.update(active=bool(owner and owner.active), commands_executed=owner.commands if owner else [],
            ownership_closed=bool(owner and owner.active is None and
                all(base.supervision.closed(v) for v in owner.commands)))
        require(owner is None or owner.active is None, 'host ownership unresolved')
        save(row)
    print(json.dumps({k: row.get(k) for k in ('stage', 'passed', 'ownership_closed', 'error')}), flush=True)
    require(row['stage'] == ('built' if mode == 'build' else 'closed') and row['ownership_closed'],
            'host phase did not complete')


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'build', 'check'), 'use prepare|build|check')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.guards.no_model_owners([])
        prepare() if sys.argv[1] == 'prepare' else execute(sys.argv[1])
