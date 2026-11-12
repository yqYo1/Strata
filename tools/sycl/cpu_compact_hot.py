#!/usr/bin/env python3
"""Commit-checked CPU-only ownership fixture; no speed or numeric-oracle gate.

Main-authored, repository LICENSE. Preparation reads source/recipe inputs only;
build/check share the global measurement lock and use the maintained pidfd owner.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
import measurement_guards as guards
import measurement_owner as owner

P = Path(__file__).resolve().parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT = B / 'cpu-compact-hot-lifetime-v1'
PLAN = P / 'bench/sycl/cpu_compact_hot/plan.json'
PYTHON = '/home/yayoi/.local/share/strata-sycl/venv/bin/python3'
CXX = '/opt/intel/oneapi/compiler/2026.2/bin/icpx'
READELF = '/usr/bin/readelf'
PRLIMIT = '/usr/bin/prlimit'
SOURCES = (
    'bench/sycl/cpu_compact_hot/cache.hpp',
    'bench/sycl/cpu_compact_hot/compact_iq3xxs.hpp',
    'bench/sycl/cpu_compact_hot/lifetime.cpp',
    'include/strata/kernels/cpu/native_expert.hpp',
    'tools/sycl/cpu_compact_hot.py',
    'tools/sycl/measurement_guards.py',
    'tools/sycl/measurement_owner.py',
)
EXPECTED = ('compact_cache_lifetime=PASS phase_guard=PASS budget=PASS '
            'source_retirement=PASS numeric_oracle=UNRUN performance=UNRUN\n')


def need(value, reason):
    if not value:
        raise RuntimeError(reason)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=P, timeout=30)


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare():
    need(not PLAN.exists() and not ROOT.exists(), 'fresh plan/root required')
    revision = git('rev-parse', 'HEAD').decode().strip()
    pins = {}
    for name in SOURCES:
        data = git('show', revision + ':' + name)
        need(data == (P / name).read_bytes(), 'commit source before preparation: ' + name)
        pins[name] = hashlib.sha256(data).hexdigest()
    native_path = B / 'coherent-prefill-native-v1/record.json'
    native = json.loads(native_path.read_text())
    need(native['ownership_closed'] and not native['active'], 'closed coherent recipe required')
    dep, = [d for d in native['dependencies']
            if d['revision'] == '3cf03257f219afbe7334045ff7c6a06ac68c627d']
    compiler_inputs = native['actual_inputs']
    need(sha(compiler_inputs['path']) == compiler_inputs['sha256'], 'native dependency census changed')
    ggml = Path(dep['root'])
    headers = {str(ggml / name): dep['source_sha256'][name]
               for name in ('ggml/include/ggml.h', 'ggml/src/ggml-common.h')}
    need(all(sha(p) == h for p, h in headers.items()), 'GGML headers changed')
    need(sha(CXX) == native['toolchain'][CXX], 'qualified compiler changed')
    environment = dict(PATH='/opt/intel/oneapi/compiler/2026.2/bin:/usr/bin:/bin',
                       LD_LIBRARY_PATH='/opt/intel/oneapi/compiler/2026.2/lib:/usr/lib/x86_64-linux-gnu',
                       LANG='C.UTF-8', LC_ALL='C.UTF-8', TMPDIR=str(ROOT / 'tmp'))
    command = [PRLIMIT, '--as=8589934592', '--',
               CXX, '-std=c++20', '-O3', '-g0', '-march=znver3', '-fp-model=strict',
               '-pthread', '-MD', '-MF', str(ROOT / 'lifetime.d'),
               '-I' + str(P / 'include'), '-I' + str(ggml / 'ggml/include'),
               '-I' + str(ggml / 'ggml/src'), str(P / SOURCES[2]),
               '-o', str(ROOT / 'lifetime')]
    plan = dict(schema='cpu-compact-lifetime-v1', source_commit=revision, sources=pins,
                tools={CXX: sha(CXX), READELF: sha(READELF), PRLIMIT: sha(PRLIMIT), PYTHON: sha(PYTHON)},
                native_recipe=dict(path=str(native_path), sha256=sha(native_path)),
                compiler_inputs=dict(path=compiler_inputs['path'], sha256=compiler_inputs['sha256']),
                ggml_headers=headers, ggml_source_root=str(ggml),
                command=command, environment=environment,
                check_command=[PRLIMIT, '--as=8589934592', '--', str(ROOT / 'lifetime')],
                acceptance='Normal0,quiet stderr,closed pidfd owners,exact source/build/executable, '
                           'all actual dependency hashes covered,no GPU library,exact ownership marker. '
                           'This does not qualify independent arithmetic,RSS admission or performance.',
                budgets=dict(build_wall=180, check_wall=20, build_rss=2 << 30, check_rss=128 << 20,
                             output=64 << 10, total=64 << 20, cpu=240),
                retention='One current CPU auxiliary fixture for pending lifecycle/numeric/timing '
                          'candidate qualification;no engine generation overwritten. '
                          'Retire normal output at closure and binary when this candidate consumer ends.')
    ROOT.mkdir(mode=0o700)
    (ROOT / 'tmp').mkdir()
    write(PLAN, plan)
    print('Prepared source/recipe only;commit plan.json before build/check.', flush=True)


def verify(plan):
    need(git('show', 'HEAD:' + str(PLAN.relative_to(P))) == PLAN.read_bytes(), 'commit exact plan')
    for name, expected in plan['sources'].items():
        original = git('show', plan['source_commit'] + ':' + name)
        need(hashlib.sha256(original).hexdigest() == expected and sha(P / name) == expected,
             'source differs: ' + name)
    need(all(sha(p) == h for p, h in plan['tools'].items()), 'compiler/tool identity changed')
    need(all(sha(p) == h for p, h in plan['ggml_headers'].items()), 'GGML header changed')
    for field in ('native_recipe', 'compiler_inputs'):
        need(sha(plan[field]['path']) == plan[field]['sha256'], field + ' changed')
    census = json.loads(Path(plan['compiler_inputs']['path']).read_text())
    sdk = {p: h for p, h in (dict(census['compiled']) | dict(census['links'])).items()
           if p.startswith(('/usr/', '/opt/', plan['ggml_source_root'] + '/'))}
    need(sdk and all(sha(p) == h for p, h in sdk.items()), 'SDK/dependency input changed before launch')


def actual_dependencies(plan):
    data = (ROOT / 'lifetime.d').read_text().replace('\\\n', '')
    need(':' in data, 'dependency target missing')
    paths = [(Path(p) if Path(p).is_absolute() else P / p).resolve()
             for p in shlex.split(data.split(':', 1)[1])]
    need(0 < len(paths) <= 4096, 'dependency census size')
    census = json.loads(Path(plan['compiler_inputs']['path']).read_text())
    allowed = dict(census['compiled']) | dict(census['links']) | plan['ggml_headers']
    allowed.update({str(P / p): h for p, h in plan['sources'].items()})
    allowed = {str(Path(p).resolve()): h for p, h in allowed.items()}
    actual = {}
    for path in paths:
        name = str(path)
        need(name in allowed and sha(path) == allowed[name], 'uncovered/changed dependency: ' + name)
        actual[name] = allowed[name]
    return actual


def execute(action):
    attempt = ROOT / (action + '-attempt.json')
    with attempt.open('x') as stream:
        json.dump(dict(action=action, no_retry=True), stream)
    state = dict(stage=action, active=True, passed=False, GPU_executed=False,
                 numeric_oracle_qualified=False, performance_qualified=False, commands=[])
    record = ROOT / (action + '.json')
    def persist():
        write(record, state)
    persist()
    try:
        plan = json.loads(PLAN.read_text())
        verify(plan)
        guards.no_model_owners([])
        state.update(source_commit=plan['source_commit'], plan_sha256=sha(PLAN))
        scope = owner.Owner(ROOT / 'owned', persist)
        state['commands'] = scope.commands
        def run(label, argv, wall):
            entry, out, err = scope.run(action + '-' + label, argv, plan['environment'], P,
                wall=wall, text_cap=plan['budgets']['output'],
                rss_cap=plan['budgets']['build_rss' if action == 'build' else 'check_rss'],
                total_cap=plan['budgets']['total'], cpu=plan['budgets']['cpu'],
                file_cap=plan['budgets']['total'])
            need(owner.closed(entry) and entry['exit_code'] == 0, 'child failed/unclosed: ' + label)
            return out.read_text(), err.read_text()
        if action == 'build':
            _, compile_error = run('compile', plan['command'], plan['budgets']['build_wall'])
            need(not compile_error, 'compiler stderr is not quiet')
            dependencies = actual_dependencies(plan)
            output, error = run('libraries', [READELF, '--dynamic', str(ROOT / 'lifetime')], 10)
            libraries = re.findall(r'Shared library: \[([^\]]+)\]', output)
            need(libraries and not error and not any(
                bad in lib.lower() for lib in libraries
                for bad in ('sycl', 'level_zero', 'libze', 'ur_loader', 'opencl', 'cuda', 'amdhip')),
                'GPU library or invalid dynamic dependency output')
            state.update(binary=dict(path=str(ROOT / 'lifetime'), sha256=sha(ROOT / 'lifetime')),
                         dependencies=dependencies, libraries=libraries)
        else:
            build = json.loads((ROOT / 'build.json').read_text())
            need(build['passed'] and not build['active'] and build['ownership_closed']
                 and build['plan_sha256'] == sha(PLAN), 'qualified build required')
            need(sha(build['binary']['path']) == build['binary']['sha256'], 'executable changed')
            need(actual_dependencies(plan) == build['dependencies'], 'compiled dependency drift')
            output, error = run('lifetime', plan['check_command'], plan['budgets']['check_wall'])
            need(output == EXPECTED and not error, 'ownership marker/stderr differs')
            state.update(binary=build['binary'], ownership_fixture_passed=True)
        verify(plan)
        state.update(passed=True, complete=True, ownership_closed=True)
    except BaseException as error:
        state.update(error=repr(error), complete=False, ownership_closed=all(
            owner.closed(c) for c in state['commands']))
        raise
    finally:
        state['active'] = False
        persist()


if __name__ == '__main__':
    need(sys.executable == PYTHON, 'use qualified Python: ' + PYTHON)
    need(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'build', 'check'), 'prepare|build|check')
    if sys.argv[1] == 'prepare':
        prepare()
    else:
        with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            execute(sys.argv[1])
