#!/usr/bin/env python3
"""Owned host-only clock-accounting contract; repository LICENSE.

Compile declared host fixtures;the default checks the ledger/wait bridge.
Private callers may supply another output contract. No inference engine,GPU
runtime,model attribution or performance qualification is created here.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
import measurement_guards as guards
import measurement_owner as owner

P = Path(__file__).resolve().parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT = B / 'decode-service-contract-v2'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
KEY = 'decode_service_host_contract_v2'
SOURCES = ('include/strata/sycl/decode_service_profile.hpp',
    'include/strata/sycl/cpu_wait_profile.hpp', 'bench/sycl/decode_wait_bridge.cpp',
    'tests/core/decode_service_profile_test.cpp', 'tools/sycl/decode_service_contract.py',
    'tools/sycl/measurement_guards.py', 'tools/sycl/measurement_owner.py',
    'AGENTS.md', 'docs/ARTIFACT_RETENTION.md')
COMPILER = Path('/usr/bin/x86_64-linux-gnu-g++-13')
require, sha = owner.require, owner.sha
FIXTURES = (
    ('ledger', 'tests/core/decode_service_profile_test.cpp', 7,
        'decode service profile: 7 host-only accounting cases passed\n'),
    ('bridge', 'bench/sycl/decode_wait_bridge.cpp', 5,
        'decode wait bridge: 5 host-only cases passed\n'))


def git(*args):
    return subprocess.check_output(['git', *args], cwd=P, timeout=20)


def compact(row):
    value = {k: v for k, v in row.items() if k != 'commands'}
    if 'dependencies' in value:
        value['dependencies'] = {name: dict(files=len(pins), aggregate_sha256=hashlib.sha256(
            json.dumps(pins, sort_keys=True).encode()).hexdigest())
            for name, pins in value['dependencies'].items()}
    return value


def save(row):
    raw = json.dumps(row, indent=2, ensure_ascii=False) + '\n'
    (ROOT / 'record.json').write_text(raw)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row)
    RESULT.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def prepare():
    require(RESULT.read_bytes() == git('show', 'HEAD:' + str(RESULT.relative_to(P))),
        'commit current results before preparing a new host contract')
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()), 'one fresh host attempt')
    revision = git('rev-parse', 'HEAD').decode().strip()
    pins = {}
    for name in SOURCES:
        require((P / name).read_bytes() == git('show', revision + ':' + name), 'commit source: ' + name)
        pins[name] = sha(P / name)
    env = dict(PATH='/usr/bin:/bin', LANG='C', LC_ALL='C',
        TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    version = subprocess.check_output([str(COMPILER), '-dumpfullversion'], env=env, timeout=10).decode().strip()
    cc1 = Path(subprocess.check_output([str(COMPILER), '-print-prog-name=cc1plus'],
        env=env, timeout=10).decode().strip()).resolve()
    packages = subprocess.check_output(['dpkg-query', '-W', '-f=${Package} ${Version}\n',
        'g++-13', 'libstdc++-13-dev', 'libstdc++6', 'libc6-dev'], env=env, timeout=10).decode().splitlines()
    row = dict(stage='prepared', source_commit=revision, sources=pins,
        compiler=dict(path=str(COMPILER), sha256=sha(COMPILER), version=version,
            cc1plus=dict(path=str(cc1), sha256=sha(cc1)), packages=packages), environment=env,
        fixtures=[dict(name=name, source=source, cases=cases, footer=footer,
            compile=[str(COMPILER), '-std=c++17', '-O2', '-g', '-Wall', '-Wextra', '-Werror',
                '-MD', '-MF', str(ROOT / (name + '.d')), '-I' + str(P / 'include'),
                str(P / source), '-o', str(ROOT / name)], command=[str(ROOT / name)])
            for name, source, cases, footer in FIXTURES],
        compile_budget=dict(wall=60, cpu=60, rss_cap=1 << 30, text_cap=64 << 10,
            file_cap=8 << 20, total_cap=16 << 20),
        test_budget=dict(wall=15, cpu=15, rss_cap=128 << 20, text_cap=16 << 10,
            file_cap=1 << 20, total_cap=16 << 20),
        profile_schema_version=2,
        plan='Compile and run the seven-case ledger,then compile and run the five-case actual wait bridge;'
            'four sequential owned children,stop first failure,no retry.'
            'Require normal0,quiet stderr,exact fixture footer,closed identities and unchanged inputs.'
            'No GPU,engine build,model timing or hardware-saturation claim. Compiler/system-package identities'
            ' are pinned before build;verify actual repository compiler dependencies against the committed pins'
            ' before each test,and hash all actual dependencies.',
        retention='Keep compact contract/dependency/source/compiler outcome for the diagnostic model host gate.'
            ' Retire both one-use executables and successful output after the result commit;'
            ' the diagnostic consumes source/recipe and results,not these executable bytes.',
        results=[], commands=[], active=False, ownership_closed=True, passed=False, complete=False,
        GPU_executed=False, model_executed=False)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'): (ROOT / name).mkdir()
    save(row)
    print('Prepared host-only decode service contract;commit before compile.', flush=True)


def unchanged(row):
    committed = json.loads(git('show', 'HEAD:' + str(RESULT.relative_to(P))))
    require(RESULT.read_bytes() == git('show', 'HEAD:' + str(RESULT.relative_to(P)))
        and committed[KEY] == compact(row), 'commit and freeze exact host contract')
    require(all(sha(P / n) == h and (P / n).read_bytes() == git('show', row['source_commit'] + ':' + n)
        for n, h in row['sources'].items()), 'host contract source drift')
    require(sha(COMPILER) == row['compiler']['sha256']
        and sha(row['compiler']['cc1plus']['path']) == row['compiler']['cc1plus']['sha256'],
        'compiler identity drift')
    packages = subprocess.check_output(['dpkg-query', '-W', '-f=${Package} ${Version}\n',
        'g++-13', 'libstdc++-13-dev', 'libstdc++6', 'libc6-dev'],
        env=row['environment'], timeout=10).decode().splitlines()
    require(packages == row['compiler']['packages'], 'compiler/header/runtime package drift')


def check_output(fixture, output, row):
    require(output.read_text() == fixture['footer'], 'complete host contract footer: ' + fixture['name'])


def run():
    row = json.loads((ROOT / 'record.json').read_text())
    require(row['stage'] == 'prepared' and not (ROOT / 'commands').exists(), 'fresh ordered host attempt only')
    unchanged(row)
    scope = None
    def persist():
        (ROOT / 'execution.json').write_text(json.dumps(dict(active=bool(scope and scope.active),
            commands=scope.commands if scope else []), indent=2) + '\n')
    try:
        scope = owner.Owner(ROOT / 'commands', persist=persist)
        row['dependencies'], row['binaries'] = {}, {}
        for fixture in row['fixtures']:
            name = fixture['name']
            for action, budget in [('compile', row['compile_budget']), ('test', row['test_budget'])]:
                command = fixture['compile'] if action == 'compile' else fixture['command']
                label = name + '-' + action
                entry, out, err = scope.run(label, command, row['environment'], ROOT, **budget)
                require(owner.closed(entry) and entry['exit_code'] == 0 and err.stat().st_size == 0,
                    label + ' failed: ' + err.read_text(errors='replace')[-1500:])
                if action == 'compile':
                    deps = (ROOT / (name + '.d')).read_text().replace('\\\n', ' ').split(':', 1)[1].split()
                    pins = {str(Path(n).resolve()): sha(n) for n in deps}
                    for path, value in pins.items():
                        actual = Path(path)
                        if actual.is_relative_to(P):
                            source = str(actual.relative_to(P))
                            require(source in row['sources'] and value == row['sources'][source],
                                'undeclared or changed repository compiler dependency: ' + source)
                    row['dependencies'][name] = pins
                    row['binaries'][name] = dict(path=str(ROOT / name), sha256=sha(ROOT / name))
                else:
                    check_output(fixture, out, row)
                    require(sha(ROOT / name) == row['binaries'][name]['sha256']
                        and all(sha(path) == value for path, value in row['dependencies'][name].items()),
                        'fixture or actual compiler dependencies changed during test: ' + name)
                row['results'].append(dict(name=label, passed=True, exit_code=0,
                    stdout_sha256=sha(out), stderr_sha256=sha(err)))
        frozen = json.loads((ROOT / 'record.json').read_text()); unchanged(frozen)
        row.update(stage='closed', passed=True, complete=True)
    except BaseException as error:
        row.update(stage='failed', error=repr(error)); raise
    finally:
        row.update(commands=scope.commands if scope else [], active=bool(scope and scope.active),
            ownership_closed=scope is None or (scope.active is None and all(owner.closed(e) for e in scope.commands)))
        save(row)
    print('Declared host fixtures passed;no model qualification.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'run'), 'use prepare|run')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        {'prepare': prepare, 'run': run}[sys.argv[1]]()
