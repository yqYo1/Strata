#!/usr/bin/env python3
"""Owned CPU policy check for joint prefill heads; repository LICENSE.

Uses the pinned official compiler/input inventory. No GPU initialization,
engine build or inference qualification follows from these host policy cases.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import sys

sys.dont_write_bytecode = True
import build_official_model as model

P, B, RESULT = model.P, model.B, model.RESULT
ROOT, KEY = B / 'prefill-dense-head-scope-v1', 'joint_prefill_heads_host_v1'
TEST = 'tests/sycl/prefill_head_scope.cpp'
HEADER = 'include/strata/core/prefill_head_scope.hpp'
require, digest, write = model.require, model.digest, model.write


def pins(revision):
    values = model.controls(revision)
    for name in [str(Path(__file__).resolve().relative_to(P)), TEST, HEADER]:
        raw = model.source.build.git(P, 'show', revision + ':' + name)
        require(raw == (P / name).read_bytes(), 'commit exact host source: ' + name)
        values[name] = hashlib.sha256(raw).hexdigest()
    return values


def compact(row):
    value = {k: v for k, v in row.items() if k not in ('sources', 'environment', 'commands_executed')}
    value['source_inputs'] = dict(files=len(row['sources']), aggregate_sha256=hashlib.sha256(
        json.dumps(row['sources'], sort_keys=True).encode()).hexdigest())
    return value


def save(row):
    write(ROOT / 'record.json', row)
    result = json.loads(RESULT.read_text()); result[KEY] = compact(row); write(RESULT, result)


def inputs():
    native = json.loads((model.ROOT / 'record.json').read_text())
    require(native['passed'] and native['stage'] == 'built-not-executed' and native['ownership_closed'],
            'closed original official build required')
    return native, model.admit(native)


def prepare():
    require(not ROOT.exists() and KEY not in model.committed(RESULT), 'fresh host scope only')
    native, _ = inputs()
    revision = model.source.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    sources = pins(revision)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'): (ROOT / name).mkdir()
    env = dict(native['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'),
               ASAN_OPTIONS='detect_leaks=1:abort_on_error=1', UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
    command = [model.COMPILER, '-std=c++17', '-O1', '-g0', '-fno-omit-frame-pointer',
        '-fsanitize=address,undefined', '-fno-fast-math', '-ffp-contract=off', '-I' + str(P / 'include'),
        '-MD', '-MF', str(ROOT / 'policy.d'), '-save-temps=obj', '-v', '-Wl,-t',
        str(P / TEST), '-o', str(ROOT / 'prefill_head_scope')]
    row = dict(stage='prepared', source_commit=revision, sources=sources, compiler=native['compiler'],
        sdk=native['sdk'], official_build_source=native['source_commit'],
        environment=env, environment_overrides={k: v for k, v in env.items() if native['environment'].get(k) != v},
        environment_sha256=hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest(),
        commands=[['compile', command], ['policy', [str(ROOT / 'prefill_head_scope')]]],
        budgets=dict(wall=90, cpu=180, rss_cap=4 << 30, text_cap=1 << 20, file_cap=32 << 20, total_cap=64 << 20),
        criteria='Seven host-only cases:joint prepare refusal/throw,complete ordered joint begin/restore,'
            'standalone prompt success/false/throw and invalid state edges. Exact committed inputs,'
            'actual compiler/linker closure,no sanitizer output,normal owned exit. No device/graph/model qualification.',
        passed=False, complete=False, active=False, ownership_closed=True, GPU_executed=False)
    save(row)
    print('Prepared official host policy check;commit the plan before compile/run.', flush=True)


def run():
    row = json.loads((ROOT / 'record.json').read_text())
    require(model.committed(RESULT)[KEY] == compact(row) and row['stage'] == 'prepared'
            and not (ROOT / 'owned').exists(), 'fresh committed host plan required')
    require(pins(row['source_commit']) == row['sources'], 'host inputs changed')
    _, allowed = inputs(); allowed.update({str(P / n): sha for n, sha in row['sources'].items()})
    owner = None
    def persist():
        write(ROOT / 'execution.json', dict(active=bool(owner and owner.active), commands=owner.commands if owner else []))
    try:
        owner = model.supervision.Owner(ROOT / 'owned', persist)
        for label, argv in row['commands']:
            entry, out, err = owner.run(label, argv, row['environment'], ROOT, **row['budgets'])
            require(model.supervision.closed(entry) and entry['exit_code'] == 0,
                    label + ' failed: ' + (out.read_text() + err.read_text())[-4000:])
            if label == 'compile':
                words = shlex.split((ROOT / 'policy.d').read_text().replace('\\\n', ' ').split(':', 1)[1])
                dependencies = {str(Path(p).resolve()) for p in words}
                require(len(dependencies) > 40 and {str(P / TEST), str(P / HEADER)} <= dependencies
                        and all(allowed.get(p) == digest(p) for p in dependencies), 'unfrozen actual compiler input')
                links = {}
                for line in out.read_text().splitlines():
                    path = Path(line.strip())
                    if not path.is_absolute() or any(c.isspace() for c in str(path)): continue
                    path = path.resolve()
                    if path.is_relative_to(ROOT): continue  # owned compiler products
                    require(allowed.get(str(path)) == digest(path), 'unfrozen actual link input: ' + str(path))
                    links[str(path)] = digest(path)
                require(len(links) > 8, 'incomplete actual linker trace')
                row['compiled_inputs'] = dict(files=len(dependencies), uncovered=0, aggregate_sha256=hashlib.sha256(
                    json.dumps({p: allowed[p] for p in sorted(dependencies)}, sort_keys=True).encode()).hexdigest())
                row['link_inputs'] = dict(files=len(links), uncovered=0, aggregate_sha256=hashlib.sha256(
                    json.dumps(links, sort_keys=True).encode()).hexdigest())
                row['binary'] = dict(path=str(ROOT / 'prefill_head_scope'), sha256=digest(ROOT / 'prefill_head_scope'),
                                     bytes=(ROOT / 'prefill_head_scope').stat().st_size)
            else:
                result = json.loads(out.read_text())
                require(not err.read_bytes() and result['kind'] == 'prefill_head_scope' and result['cases'] == 7
                        and result['checks'] >= 100 and result['status'] == 'PASS', 'policy/sanitizer check failed')
                row['result'] = result
        inputs()
        require(pins(row['source_commit']) == row['sources'], 'host source changed during execution')
        row.update(passed=True, complete=True)
    except BaseException as error:
        row.update(error=repr(error), passed=False, complete=False)
    finally:
        row.update(stage='closed' if row['passed'] else 'failed', active=bool(owner and owner.active),
            ownership_closed=bool(owner and not owner.active and all(model.supervision.closed(e) for e in owner.commands)),
            commands_executed=owner.commands if owner else [])
        persist(); save(row)
    print(json.dumps({k: row.get(k) for k in ('stage', 'passed', 'complete', 'ownership_closed', 'result', 'error')}), flush=True)
    require(row['passed'] and row['ownership_closed'], 'host policy scope failed')


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'run'), 'use prepare|run')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        model.guards.no_model_owners([])
        {'prepare': prepare, 'run': run}[sys.argv[1]]()
