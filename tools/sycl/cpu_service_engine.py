#!/usr/bin/env python3
"""Main-owned original6e40 plus CPU observer; production-pool host gate first.

Repository LICENSE. Reconstruct only the declared independent Git patches;
never import the current tree's unselected prefill/layout changes.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import generation_engine as gen
import mtp_resident_control as native
import runtime_probe as runtime

P, B = native.P, native.B
ROOT = B / 'cpu-service-original-v1'
RESULT = gen.A / 'cpu-service-current.json'
require, digest = native.guards.require, native.build.digest
PATCHES = (
    ('35f3683bc', ('CMakeLists.txt', 'include/strata/kernels/cpu/pool.hpp',
        'include/strata/kernels/cpu/service_profile.hpp', 'src/kernels/cpu/pool.cpp',
        'src/program/generate.cpp', 'tests/core/cpu_service_profile_test.cpp')),
    ('f26979a65', ('include/strata/kernels/cpu/service_profile.hpp',)),
    ('44b520e30', ('CMakeLists.txt', 'tests/core/cpu_service_pool_test.cpp')),
    ('589ff46aa', ('tests/core/cpu_service_pool_test.cpp',)),
    ('f935f0979', ('tests/core/cpu_service_profile_test.cpp',)))
CHANGED = sorted({name for _, names in PATCHES for name in names})


def write(path, row):
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + '\n')


def compact(row):
    return {k: v for k, v in row.items() if k not in
        ('generated_inputs', 'dependencies', 'sdk_inputs', 'external_inputs', 'commands', 'actual_compiled_inputs')}


def prepare():
    container = gen.committed(RESULT)
    require(container['passed'] and container['ownership_closed'] and not container['active'] and
        'production_pool' not in container and not ROOT.exists(), 'fresh production-pool gate only')
    retired = gen.committed(native.RESULT)['artifact_retirement']
    require(retired['complete'] and not Path(retired['root']).exists(), 'retire extra engine before a new generation')
    qualification = gen.committed(runtime.RESULT)
    original = runtime.current(qualification)
    model_gate = gen.committed(runtime.RESULT.with_name('runtime-model-current.json'))
    require(model_gate['passed'] and model_gate['ownership_closed'] and model_gate['binary'] == original['binary'],
        'updated-runtime original model execution gate required')
    revision = gen.git('rev-parse', 'HEAD').decode().strip()
    inputs = gen.benchmark_pins(revision)
    require('tools/sycl/cpu_service_engine.py' in inputs, 'commit this controller before preparation')
    project = native.project_sources(original)
    sdk_original = json.loads(native.prior.SDK.read_text())
    require(digest(native.prior.SDK) == native.prior.SDK_SHA and len(sdk_original) == 8697, 'original SDK manifest differs')
    sdk = {item['path']: digest(item['path']) for item in sdk_original}
    changes = {item['path']: dict(original=item['sha256'], current=sdk[item['path']])
        for item in sdk_original if item['sha256'] != sdk[item['path']]}
    require(len(changes) == 3 and {Path(p).name for p in changes} ==
        {'metee.h', 'opencl-c-base.h', 'opencl-c.h'}, 'undeclared SDK transition')
    ROOT.mkdir(mode=0o700)
    for name in ('source', 'tmp', 'cache'):
        (ROOT / name).mkdir()
    source = ROOT / 'source'
    for name, raw in native.build.blobs(P, native.BASE, project):
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    patches = []
    for revision_id, names in PATCHES:
        patch = gen.git('show', '--format=', '--binary', revision_id, '--', *names)
        require(patch, 'empty declared observer patch')
        for arguments in (['--check'], []):
            subprocess.run(['git', 'apply', *arguments, '-'], input=patch, cwd=source, check=True, timeout=15)
        patches.append(dict(commit=revision_id, paths=list(names), sha256=hashlib.sha256(patch).hexdigest()))
    generated = {str(p.relative_to(source)): digest(p) for p in source.rglob('*') if p.is_file()}
    require(sorted(name for name, sha in generated.items() if project.get(name) != sha) == CHANGED,
        'unexpected observer source difference')
    tools = {name: dict(path=path, sha256=digest(path)) for name, path in
        (('cmake', '/usr/bin/cmake'), ('ninja', '/usr/bin/ninja'))}
    args = list(original['configure_argv'])
    args[args.index('-S') + 1] = str(source)
    args[args.index('-B') + 1] = str(ROOT / 'build')
    args += ['-DCMAKE_MAKE_PROGRAM:FILEPATH=/usr/bin/ninja', '-DSTRATA_CPU_SERVICE_PROFILE=ON']
    prior = json.loads(gen.git('show', '26d484985:' + str(native.RESULT.relative_to(P))))
    # The closed private build's external MKL input set is still the original
    # build dependency set; its native changes and executable are not reused.
    external = json.loads(native.prior.ENGINE_RECORD.read_text())['external_inputs']
    require(prior['base_native_source'] == original['source_commit'], 'external-input provenance differs')
    for path, sha in external.items():
        require(digest(path) == sha, 'external MKL input changed: ' + path)
    row = dict(stage='prepared', source_commit=revision, base_native_source=native.BASE,
        source_files=len(generated), source_fingerprint=native.fingerprint(generated),
        patches=patches, changed_native_files=CHANGED, generated_inputs=generated,
        benchmark_files=len(inputs), benchmark_fingerprint=native.fingerprint(inputs),
        compiler=original['compiler'], tools=tools, dependencies=original['dependencies'],
        sdk_inputs=sdk, sdk_original=dict(path=str(native.prior.SDK), sha256=native.prior.SDK_SHA),
        sdk_fingerprint=native.fingerprint(sdk), sdk_files=len(sdk), declared_sdk_changes=changes,
        external_inputs=external, external_fingerprint=native.fingerprint(external),
        external_reference=dict(path=str(native.prior.ENGINE_RECORD), sha256=digest(native.prior.ENGINE_RECORD)),
        runtime_reference=dict(path=str(runtime.RESULT.relative_to(P)), sha256=digest(runtime.RESULT)),
        configure_argv=args, build_argv=['/usr/bin/cmake', '--build', str(ROOT / 'build'),
            '--target', 'cpu_service_pool_test', '-j', '4'],
        environment=dict(original['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache')),
        fixture_environment=dict(STRATA_IQ2_PAIR='1', STRATA_IQ_SINGLE_EXACT='0'),
        criteria='Build only the real CPU-pool fixture from selected6e40 plus declared observer/test patches;then separate preexec observer0/1 processes. Require both PASS,28 format/width cases,97 jobs per batch,counter conservation,canaries and identical output checksums. Check complete actual compiler dependencies against pinned source/SDK/MKL/generated header and all owned exits. No native engine or GPU/model invocation at this gate.',
        limits='Same-leaf parity,not an independent mathematical oracle. No full observer/model overhead,speedup,CPU resource attribution or decode nonregression claim. Prefill/decode adoption unchanged.',
        retention='Keep compact source/recipe,outcomes/checksum/closure. Keep this one extra build only for the imminent original-plus-observer engine/model attribution;retire on rejection/closure. Original validated6e40 and12fixtures remain intact;legacy holds separate.',
        budgets=dict(wall=900, cpu=3600, rss_cap=16 << 30, text_cap=32 << 20,
            file_cap=32 << 20, total_cap=2 << 30),
        passed=False, complete=False, active=False, ownership_closed=True, GPU_executed=False,
        native_engine_built=False, production_pool_checked=False, outcomes=[])
    write(ROOT / 'record.json', row)
    container['production_pool'] = compact(row)
    write(RESULT, container)
    print('Prepared original-plus-observer CPU pool gate; commit exact plan before build.', flush=True)


def current(row):
    require(gen.committed(RESULT)['production_pool'] == compact(row), 'commit exact CPU-pool plan')
    pins = gen.benchmark_pins(row['source_commit'])
    require(len(pins) == row['benchmark_files'] and native.fingerprint(pins) == row['benchmark_fingerprint'],
        'controller/benchmark source changed')
    qualification = gen.committed(runtime.RESULT)
    require(digest(runtime.RESULT) == row['runtime_reference']['sha256'], 'runtime qualification changed')
    original = runtime.current(qualification)
    require(row['base_native_source'] == original['source_commit'] and
        row['dependencies'] == original['dependencies'] and row['compiler'] == original['compiler'],
        'selected original dependency/compiler identity changed')
    require(native.fingerprint(row['sdk_inputs']) == row['sdk_fingerprint'] and
        native.fingerprint(row['external_inputs']) == row['external_fingerprint'] and
        native.fingerprint(row['generated_inputs']) == row['source_fingerprint'], 'private build input seal changed')
    require({str(p.relative_to(ROOT / 'source')) for p in (ROOT / 'source').rglob('*') if p.is_file()} ==
        set(row['generated_inputs']), 'generated source extent differs')
    for name, sha in row['generated_inputs'].items():
        require(digest(ROOT / 'source' / name) == sha, 'generated source changed: ' + name)
    for paths in (row['sdk_inputs'], row['external_inputs']):
        for path, sha in paths.items():
            require(digest(path) == sha, 'build input changed: ' + path)
    for dependency in row['dependencies']:
        require(native.build.pin(Path(dependency['root']), ['.']) == dependency, 'native dependency changed')
    for item in [row['compiler'], *row['tools'].values()]:
        require(digest(item['path']) == item['sha256'], 'build tool changed')
    require(digest(row['external_reference']['path']) == row['external_reference']['sha256'] and
        digest(row['sdk_original']['path']) == row['sdk_original']['sha256'], 'original dependency recipe changed')


def dependency_closure(row, text):
    expected = {str((ROOT / 'source' / name).resolve()): sha for name, sha in row['generated_inputs'].items()}
    for source in row['dependencies']:
        expected.update({str((Path(source['root']) / name).resolve()): sha for name, sha in source['source_sha256'].items()})
    expected.update(row['sdk_inputs'])
    expected.update(row['external_inputs'])
    dependency = row['dependencies'][0]
    get = lambda name: native.build.git(Path(dependency['root']), 'show', dependency['revision'] + ':' + name).decode()
    cmake = get('ggml/CMakeLists.txt')
    version = '.'.join(re.search(r'set\(GGML_VERSION_' + part + r' (\d+)\)', cmake)[1]
        for part in ('MAJOR', 'MINOR', 'PATCH'))
    header = get('ggml/src/ggml-version.h.in').replace('@GGML_VERSION@', version).replace(
        '@GGML_BUILD_COMMIT@', dependency['revision'][:9]).encode()
    expected[str(ROOT / 'build/ggml/src/ggml-version.h')] = hashlib.sha256(header).hexdigest()
    paths = {Path(line.strip()) for line in text.splitlines() if line.startswith('    ') and line.strip()}
    paths = {(path if path.is_absolute() else ROOT / 'build' / path).resolve() for path in paths}
    require(len(paths) > 100 and all((ROOT / 'source' / name).resolve() in paths for name in
        ('src/kernels/cpu/pool.cpp', 'include/strata/kernels/cpu/service_profile.hpp',
         'tests/core/cpu_service_pool_test.cpp')), 'incomplete CPU pool compiler census')
    uncovered = sorted(str(path) for path in paths if str(path) not in expected)
    require(not uncovered, 'uncovered actual compiled inputs: ' + repr(uncovered[:12]))
    for path in paths:
        require(digest(path) == expected[str(path)], 'actual compiler input changed: ' + str(path))
    row['actual_compiled_inputs'] = {str(path): expected[str(path)] for path in sorted(paths)}
    return dict(files=len(paths), uncovered_files=0, sha256=native.fingerprint(row['actual_compiled_inputs']))


def build():
    container = gen.committed(RESULT)
    row = json.loads((ROOT / 'record.json').read_text())
    current(row)
    require(row['stage'] == 'prepared' and not (ROOT / 'owned').exists(), 'fresh single-use CPU-pool build')
    initial = copy.deepcopy(row)
    owner = None

    def persist():
        write(ROOT / 'record.json', dict(row, active=bool(owner and owner.active is not None),
            commands=owner.commands if owner else []))

    try:
        owner = native.supervision.Owner(ROOT / 'owned', persist)
        for label, command in (('configure', row['configure_argv']), ('build-pool', row['build_argv']),
                ('compiler-census', ['/usr/bin/ninja', '-C', str(ROOT / 'build'), '-t', 'deps'])):
            entry, out, err = owner.run(label, command, row['environment'], P, **row['budgets'])
            require(native.supervision.closed(entry) and entry['exit_code'] == 0,
                label + ' failed: ' + (out.read_text(errors='replace') + err.read_text(errors='replace'))[-4096:])
            if label == 'compiler-census':
                row['dependency_closure'] = dependency_closure(row, out.read_text())
            current(initial)
        fixture = ROOT / 'build/cpu_service_pool_test'
        row['fixture'] = dict(path=str(fixture), bytes=fixture.stat().st_size, sha256=digest(fixture))
        checksums = []
        for setting in ('0', '1'):
            environment = dict(row['environment'], **row['fixture_environment'], STRATA_CPU_SERVICE=setting)
            entry, out, err = owner.run('observer' + setting, [str(fixture), setting], environment, P, **row['budgets'])
            require(native.supervision.closed(entry) and entry['exit_code'] == 0, 'real pool fixture failed: ' + err.read_text()[-4096:])
            match = re.fullmatch(r'cpu_service_pool PASS observer=' + setting +
                r' formats=7 nt=1\.\.4 cases=28 checksum_fnv1a64_le=([0-9a-f]{16})\n', out.read_text())
            require(match is not None, 'CPU pool completion marker differs')
            checksums.append(match[1])
            row['outcomes'].append(dict(observer=int(setting), passed=True, exit_code=0, ownership_closed=True,
                cases=28, formats=7, widths=[1, 2, 3, 4], checksum_fnv1a64_le=match[1]))
            current(initial)
        require(len(set(checksums)) == 1, 'observer changed real pool outputs')
        row.update(passed=True, complete=True, production_pool_checked=True, checksum_match=True, stage='pool-checked')
    except BaseException as error:
        row.update(passed=False, complete=False, stage='failed', error=repr(error))
    finally:
        row['ownership_closed'] = owner is None or owner.active is None and all(native.supervision.closed(e) for e in owner.commands)
        row['passed'] = row['complete'] = bool(row['passed'] and row['ownership_closed'])
        row['active'] = False
        persist()
        container['production_pool'] = compact(row)
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
