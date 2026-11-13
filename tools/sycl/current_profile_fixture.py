#!/usr/bin/env python3
"""Bind a small profiling control to the current coherent runtime; repository LICENSE.

Only the fixture is compiled. Reuse the current runtime archive with its exact
source/objects/SDK proof; never rebuild or overwrite the retained engines.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
import cpu_mixed_cache_model as model
import measurement_guards as guards
import measurement_owner as ownership

P, B, RESULT = model.P, model.B, model.RESULT
ROOT = B / 'current-profile-fixture-v2'
KEY = 'current_profile_fixture_v2'
THIS = 'tools/sycl/current_profile_fixture.py'
OBSERVERS = {THIS, 'tools/sycl/cpu_mixed_cache_bench.py'}
CPP = 'tests/sycl/profile_compat_window.cpp'
TARGET = 'sycl_profile_compat_window'
require, sha = model.require, model.sha


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def compact(row):
    return {k: v for k, v in row.items() if k != 'commands'}


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text())
    data[KEY] = compact(row)
    write(RESULT, data)


def command(text):
    args = shlex.split(text)
    if args[:2] == [':', '&&'] and args[-2:] == ['&&', ':']:
        args = args[2:-2]
    require(args and not any(a in ('&&', '||', ';', '|', '>', '<') for a in args),
            'unsupported compound build command')
    return args


def native(expected=None):
    return model.native(expected, observers=OBSERVERS)


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()), 'fresh fixture plan only')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=P, text=True).strip()
    original = subprocess.check_output(['git', 'show', revision + ':' + THIS], cwd=P)
    require(hashlib.sha256(original).hexdigest() == sha(P / THIS), 'commit builder before preparation')
    n = native()
    cohort = json.loads((B / 'cpu-mixed-cache-speed-v1/record.json').read_text())
    require(cohort['stage'] == 'closed' and cohort['passed'] and cohort['complete']
            and cohort['ownership_closed'] and not cohort['active']
            and len(cohort['processes']) == 6
            and not any(v['reproducible_TG_gain'] for v in cohort['decision']['workloads'].values()),
            'close original comparison without advancing its failed cache gate')
    build = Path(n['binaries']['strata']['path']).parent
    database = json.loads((build / 'compile_commands.json').read_text())
    entries = [c for c in database if c['file'] == str(P / CPP)
               and TARGET + '.dir/' in c['output']]
    require(len(entries) == 1, 'one existing current-runtime fixture target required')
    entry, = entries
    compile_args = command(entry['command'])
    require(compile_args.count('-o') == 1 and '-c' in compile_args
            and '-DSTRATA_ENABLE_SYCL=1' in compile_args
            and '-DSTRATA_CPU_SERVICE_PROFILE=1' in compile_args,
            'current runtime class conditions required')
    original_object = compile_args[compile_args.index('-o') + 1]
    cache = (build / 'CMakeCache.txt').read_text()
    make, = re.findall(r'^CMAKE_MAKE_PROGRAM:FILEPATH=(.+)$', cache, re.M)
    generated = subprocess.check_output([make, '-t', 'commands', TARGET], cwd=build,
                                        text=True, timeout=30)
    links = []
    for line in generated.splitlines():
        tokens = shlex.split(line)
        if '-o' in tokens and tokens[tokens.index('-o') + 1] == TARGET:
            links.append(command(line))
    require(len(links) == 1, 'one generated fixture link command required')
    link_args, = links
    require(link_args.count(original_object) == 1, 'exact fixture object in link command')
    require(Path(link_args[0]).resolve() == Path(compile_args[0]).resolve(),
            'same pinned compiler driver for compile and link')
    archive = build / 'libstrata_sycl_runtime.a'
    require(archive.is_file() and str(archive.name) in link_args, 'current runtime archive required')
    runtime_objects = {}
    for c in database:
        if 'strata_sycl_runtime.dir/' in c['output']:
            obj = build / c['output']
            require(obj.is_file(), 'runtime object was not built')
            runtime_objects[str(obj)] = sha(obj)
    require(len(runtime_objects) == 3, 'three coherent runtime objects required')
    source = subprocess.check_output(['git', 'show', n['native_source_commit'] + ':' + CPP], cwd=P)
    require(hashlib.sha256(source).hexdigest() == sha(P / CPP), 'fixture source differs from native revision')
    compile_args[compile_args.index('-o') + 1] = str(ROOT / 'fixture.o')
    compile_args.extend(['-MD', '-MF', str(ROOT / 'fixture.d')])
    link_args[link_args.index(original_object)] = str(ROOT / 'fixture.o')
    link_args[link_args.index('-o') + 1] = str(ROOT / TARGET)
    environment = dict(n['recipe']['environment'], TMPDIR=str(ROOT / 'tmp'),
                       TMPPREFIX=str(ROOT / 'tmp/zsh'))
    row = dict(stage='prepared', source_commit=revision, builder_sha256=sha(P / THIS),
               native_source_commit=n['native_source_commit'], native_proof=n['model_observer_admission'],
               engines_and_fixtures=n['binaries'], build_root=str(build),
               compile_database_sha256=sha(build / 'compile_commands.json'),
               ninja={'path': make, 'sha256': sha(make)},
               generated_target_commands_sha256=hashlib.sha256(generated.encode()).hexdigest(),
               runtime_archive={'path': str(archive), 'sha256': sha(archive)},
               runtime_objects=runtime_objects, fixture_source={'path': CPP, 'sha256': sha(P / CPP)},
               recipe={'compile': compile_args, 'link': link_args, 'environment': environment},
               budget=dict(wall=600, cpu=2400, rss_cap=8 << 30, text_cap=16 << 20,
                           file_cap=128 << 20, total_cap=512 << 20),
               active=False, ownership_closed=True, passed=False, complete=False, commands=[],
               GPU_executed=False, model_executed=False,
               acceptance='Only one new fixture object/ELF;exact current runtime archive and all three objects plus13retained ELFs unchanged;dependency/source/SDK hashes admitted;normal0and closed owners. Build is not runtime/tool/model qualification.',
               consumer='Fresh bounded direct/active-from-exec collector controls,then current cache0GPU/CPU attribution;retire superseded fixture after original-input recovery.',
               review_point='After current collector control closure;no unused standalone generation retained.')
    ROOT.mkdir(mode=0o700)
    (ROOT / 'tmp').mkdir()
    save(row)
    print('Prepared one fixture object/ELF;commit plan before compilation.', flush=True)


def current(row):
    require(model.base.original.validation.committed(RESULT, (KEY,)) == compact(row),
            'commit exact fixture plan/result')
    require(sha(P / THIS) == row['builder_sha256'], 'builder changed')
    n = native(row['native_proof'])
    require(n['binaries'] == row['engines_and_fixtures'] and
            sha(row['runtime_archive']['path']) == row['runtime_archive']['sha256'] and
            all(sha(p) == h for p, h in row['runtime_objects'].items()), 'retained native inputs changed')
    build = Path(row['build_root'])
    require(sha(build / 'compile_commands.json') == row['compile_database_sha256'] and
            sha(row['ninja']['path']) == row['ninja']['sha256'] and
            sha(P / CPP) == row['fixture_source']['sha256'], 'fixture recipe input changed')
    text = subprocess.check_output([row['ninja']['path'], '-t', 'commands', TARGET],
                                   cwd=build, text=True, timeout=30)
    require(hashlib.sha256(text.encode()).hexdigest() == row['generated_target_commands_sha256'],
            'generated target recipe changed')
    return n


def build():
    row = json.loads((ROOT / 'record.json').read_text())
    n = current(row)
    require(row['stage'] == 'prepared' and not (ROOT / 'build-attempt.json').exists(), 'one build attempt')
    write(ROOT / 'build-attempt.json', {'source_commit': row['source_commit']})
    owner = ownership.Owner(ROOT / 'compile-owned')
    try:
        logs = {}
        for label in ('compile', 'link'):
            entry, output, _ = owner.run(label, row['recipe'][label], row['recipe']['environment'],
                                         Path(row['build_root']), **row['budget'])
            require(ownership.closed(entry) and entry['exit_code'] == 0, 'owned ' + label + ' failed')
            logs[label] = output
        deptext = (ROOT / 'fixture.d').read_text().replace('\\\n', ' ')
        dependencies = shlex.split(deptext.split(':', 1)[1])
        expected = {str(P / p): h for p, h in n['project']['source_sha256'].items()}
        expected[str(P / CPP)] = row['fixture_source']['sha256']
        expected.update(json.loads(Path(n['external_inputs']['path']).read_text()))
        expected.update(json.loads(Path(n['actual_inputs']['path']).read_text())['links'])
        for dep in n['dependencies']:
            expected.update({str(Path(dep['root']) / p): h for p, h in dep['source_sha256'].items()})
        normalized = {}
        for path, digest in expected.items():
            name = str(Path(path).resolve())
            require(name not in normalized or normalized[name] == digest, 'conflicting dependency alias')
            normalized[name] = digest
        expected = normalized
        actual = {}
        for p in dependencies:
            path = Path(p) if Path(p).is_absolute() else Path(row['build_root']) / p
            name = str(path.resolve())
            require(name in expected and sha(path) == expected[name], 'unadmitted fixture dependency: ' + name)
            actual[name] = expected[name]
        require(str((P / CPP).resolve()) in actual and len(actual) > 20, 'incomplete compiler dependency closure')
        link_inputs = {}
        outputs = {str((ROOT / 'fixture.o').resolve()): sha(ROOT / 'fixture.o'),
                   str(Path(row['runtime_archive']['path']).resolve()): row['runtime_archive']['sha256']}
        for line in Path(logs['link']).read_text().splitlines():
            path = Path(line.strip())
            path = path if path.is_absolute() else Path(row['build_root']) / path
            if not path.is_file():
                continue
            name = str(path.resolve())
            permitted = outputs | expected
            require(name in permitted and sha(path) == permitted[name], 'unadmitted fixture link input: ' + name)
            link_inputs[name] = permitted[name]
        require(len(link_inputs) > 8 and set(outputs) <= set(link_inputs), 'incomplete linker input closure')
        current(row)
        binary = ROOT / TARGET
        require(binary.read_bytes()[:4] == b'\x7fELF', 'fixture output is not ELF')
        row.update(stage='built_not_executed', passed=True, complete=True,
                   binary={'path': str(binary), 'sha256': sha(binary), 'bytes': binary.stat().st_size},
                   actual_compiler_inputs=dict(files=len(actual), sha256=model.base.original.signature(actual),
                       dependency_file={'path': str(ROOT / 'fixture.d'), 'sha256': sha(ROOT / 'fixture.d')}),
                   actual_link_inputs=dict(files=len(link_inputs), sha256=model.base.original.signature(link_inputs)),
                   retained_thirteen_ELFs_unchanged=True)
    except BaseException as error:
        row.update(stage='failed', passed=False, complete=False, error=repr(error))
        raise
    finally:
        row['commands'] = owner.commands
        row['active'] = owner.active is not None
        row['ownership_closed'] = owner.active is None and all(ownership.closed(e) for e in owner.commands)
        save(row)
    print('Built current-runtime fixture;GPU/tool/model qualification still pending.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'build'), 'use prepare|build')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        prepare() if sys.argv[1] == 'prepare' else build()
