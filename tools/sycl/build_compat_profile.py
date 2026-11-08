#!/usr/bin/env python3
"""Build only the production-adapter profiling fixture; never load a GPU.

Main-authored; repository LICENSE. The selected engine stays unchanged. Its
original source is recovered from Git, without requiring a retired child checkout.
Run prepare/build only after the frozen thinking-generation comparison closes.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import sys

sys.dont_write_bytecode = True
import build_engine as source
import build_upstream_features as build
import check_cache_loan_cli as feature
import measurement_owner as ownership
import profile_split as profile

P, B, RESULT = source.P, source.B, feature.STATUS
ROOT, KEY = B / 'compat-profile-fixture-v1', 'compat_profile_fixture_build_v1'
TARGET = 'sycl_profile_compat_window'
CPP = 'tests/sycl/profile_compat_window.cpp'
UNITS = (CPP, 'src/sycl/runtime.cpp', 'src/sycl/compat.cpp', 'src/sycl/command_profile.cpp')
PREFIXES = ('CMakeLists.txt', 'cmake', 'include', 'src', 'tests/sycl', 'tools/sycl')
require, digest, write, signature = feature.require, source.digest, build.write, build.signature
BUDGET = dict(wall=1200, cpu=3600, rss_cap=8 << 30, text_cap=32 << 20,
              file_cap=128 << 20, total_cap=512 << 20)


def basis():
    generation = feature.validation.committed(build.STATUS,
        ('current_sync', 'dflash2_cache_loan_generation_v1'))
    require(generation['passed'] and generation['complete'] and generation['ownership_closed'] and
        len(generation['processes']) == 6 and all(p['passed'] and p['complete'] and
        p['ownership_closed'] and p['native_exit_code'] == 0 and p['fresh_faults'] == []
        for p in generation['processes']), 'close the frozen six-process comparison first')
    proof = json.loads(source.git(P, 'show', feature.EVIDENCE + ':' +
        str(build.STATUS.relative_to(P))))['current_sync'][feature.BUILD_KEY]
    path = Path(proof['record']['path'])
    require(digest(path) == proof['record']['sha256'], 'original native build receipt changed')
    native = json.loads(path.read_text())
    require(native['stage'] == 'built' and native['passed'] and native['ownership_closed'] and
        proof['passed'] and proof['ownership_closed'] and
        proof['source_commit'] == native['project']['revision'] and
        proof['binaries'] == native['binaries'] and
        generation['binary'] == native['binaries']['strata'], 'selected comparison engine differs')
    project = native['project']
    recovered = {n: hashlib.sha256(raw).hexdigest() for n, raw in
        source.blobs(P, project['revision'], project['source_sha256'])}
    require(recovered == project['source_sha256'], 'original native sources unavailable in Git')
    source.git(P, 'merge-base', '--is-ancestor', project['revision'], 'origin/feature/sycl')
    runtime = {n: sha for n, sha in recovered.items()
        if n.startswith(('src/sycl/', 'include/strata/sycl/'))}
    require(len(runtime) == 21 and all(digest(P / n) == sha for n, sha in runtime.items()),
        'profiling fixture must use the selected production runtime sources')
    engine = native['binaries']['strata']
    require(digest(engine['path']) == engine['sha256'] and
        Path(engine['path']).stat().st_size == engine['bytes'], 'selected engine ELF changed')
    return native, dict(record=proof['record'], source_commit=project['revision'], engine=engine,
        runtime_sources=runtime)


def compact(row):
    out = {k: v for k, v in row.items() if k not in ('project', 'dependencies', 'commands', 'recipe')}
    for key in ('project',):
        out[key] = {k: v for k, v in row[key].items() if k != 'source_sha256'}
    out['dependencies'] = [{k: v for k, v in d.items() if k != 'source_sha256'}
                           for d in row['dependencies']]
    out['recipe'] = {k: v for k, v in row['recipe'].items() if k != 'environment'}
    out['build_environment_sha256'] = signature(row['recipe']['environment'])
    out['selected_native'] = {k: v for k, v in row['selected_native'].items() if k != 'runtime_sources'}
    runtime = row['selected_native']['runtime_sources']
    out['selected_native']['runtime_sources'] = dict(files=len(runtime), sha256=signature(runtime))
    out['commands'] = [dict(label=c['label'], exit_code=c.get('exit_code'),
        ownership_closed=ownership.closed(c), error=c.get('error'), errors=c.get('errors', []))
        for c in row['commands']]
    return out


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row); write(RESULT, data)


def current(row, committed=True):
    if committed:
        require(feature.validation.committed(RESULT, (KEY,)) == compact(row),
                'commit the exact fixture build plan/result first')
    _, identity = basis()
    require(identity == row['selected_native'], 'selected native provenance changed')
    for pin in (row['project'], *row['dependencies']):
        require(len(pin['source_sha256']) == pin['count'] and
            signature(pin['source_sha256']) == pin['aggregate_sha256'], 'source manifest count/hash differs')
    require(all(n in row['project']['source_sha256'] for n in UNITS) and
        'tools/sycl/build_compat_profile.py' in row['project']['source_sha256'], 'missing fixture/controller source')
    build.check_inputs(row)
    require(profile.collector_assets(feature.validation.committed(RESULT)) == row['collector'],
            'current collector admission changed;do not refresh hashes')
    require(row['recipe']['targets'] == row['recipe']['build_targets'] == [TARGET] and
        '-DSTRATA_ENABLE_CUDA:BOOL=OFF' in row['recipe']['configure'] and
        '-DSTRATA_ENABLE_HIP:BOOL=OFF' in row['recipe']['configure'] and
        not row['recipe']['environment'].get('LD_PRELOAD'), 'fixture-only build envelope differs')


def prepare():
    require(not ROOT.exists() and KEY not in feature.validation.committed(RESULT),
            'fresh fixture build only;no retry or replacement of a failed result')
    native, identity = basis()
    assets = profile.collector_assets(feature.validation.committed(RESULT))
    project = source.pin(P, PREFIXES); project['prefixes'] = list(PREFIXES)
    recipe = copy.deepcopy(native['recipe'])
    recipe['configure'] = [v.replace(native['project']['root'], str(P)).replace(str(feature.NATIVE), str(ROOT))
                           for v in recipe['configure']]
    recipe['environment'] = {k: v.replace(str(feature.NATIVE), str(ROOT))
                             for k, v in recipe['environment'].items()}
    recipe['targets'] = recipe['build_targets'] = [TARGET]
    recipe.pop('host_modes', None)
    require(recipe['configure'][recipe['configure'].index('-S') + 1] == str(P) and
        recipe['configure'][recipe['configure'].index('-B') + 1] == str(ROOT / 'build'),
        'isolated source/output paths required')
    row = dict(stage='prepared', project=project, dependencies=copy.deepcopy(native['dependencies']),
        recipe=recipe, toolchain=native['toolchain'], external_inputs=native['external_inputs'],
        fallbacks=native['fallbacks'], selected_native=identity, collector=assets,
        budgets=BUDGET, commands=[], passed=False, complete=False, active=False, ownership_closed=True,
        GPU_qualified=False, model_qualified=False,
        scope='Build only four production-runtime/fixture translation units. No engine build,fixture execution,'
              'device enumeration,collector load,metric/Sysman request or model speed claim.',
        consumer='Next serialized direct,fully active and separately paused-construction production-adapter checks.'
                 ' Verify counts,flows,guards,clock bounds,normal exit,ownership and fresh faults before model profiling.',
        retention='One fixture build for this named current comparison;no extra model-engine generation.'
                  ' Keep original source/recipe and compact compiled-input census,then retire successful build output'
                  ' and the fixture after the collector boundary question closes. Preserve engine/phase guards.'
                  ' Existing legacy provenance holds are unchanged.')
    # No process or output directory exists until every prospective input gate passes.
    current(row, committed=False)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'): (ROOT / name).mkdir(mode=0o700)
    save(row)
    print('Prepared fixture-only production-runtime build;commit before configure/compile.', flush=True)


def census(row, dependency_output, link_output):
    allowed = {}
    for pin in (row['project'], *row['dependencies']):
        allowed.update({str((Path(pin['root']) / p).resolve()): sha
                        for p, sha in pin['source_sha256'].items()})
    allowed.update({str(Path(p).resolve()): sha for p, sha in
        json.loads(Path(row['external_inputs']['path']).read_text()).items()})
    allowed.update({str(Path(p).resolve()): sha for p, sha in row['toolchain'].items()})
    compiled, generated = {}, {}
    for line in dependency_output.read_text().splitlines():
        if not line.startswith('    ') or not line.strip(): continue
        path = Path(line.strip())
        path = (path if path.is_absolute() else ROOT / 'build' / path).resolve()
        sha = digest(path)
        if path.is_relative_to(ROOT / 'build'): generated[str(path)] = sha
        else:
            require(allowed.get(str(path)) == sha, 'unfrozen actual compiler input: ' + str(path))
            compiled[str(path)] = sha
    expected = {str((P / p).resolve()) for p in UNITS}
    observed = {p for p in compiled if Path(p).suffix in ('.cpp', '.c', '.cc', '.cu')}
    require(observed == expected and len(compiled) > 100, 'actual fixture translation-unit census differs')
    links, owned = {}, set()
    for line in link_output.read_text(errors='replace').splitlines():
        name = line.strip(); path = Path(name)
        if not name or any(c.isspace() for c in name) or not (
            path.is_absolute() or path.suffix in ('.o', '.a') or '.so' in path.name): continue
        path = (path if path.is_absolute() else ROOT / 'build' / path).resolve()
        if any(path.is_relative_to(ROOT / n) for n in ('build', 'tmp')):
            owned.add(str(path)); continue
        require(path.is_file() and allowed.get(str(path)) == digest(path),
                'unfrozen actual linker input: ' + str(path))
        links[str(path)] = digest(path)
    require(len(links) > 8, 'actual link census incomplete')
    database = json.loads((ROOT / 'build/compile_commands.json').read_text())
    commands = [c for c in database if c['file'] in expected]
    require(len(commands) == len(UNITS) and {c['file'] for c in commands} == expected,
            'one production compile command per fixture/runtime unit required')
    for c in commands:
        args = shlex.split(c['command'])
        require(all(x in args for x in ('-DSTRATA_ENABLE_SYCL=1', '-O3', '-fsycl',
            '-fsycl-id-queries-range=size_t')) and not any(x in args for x in ('-Ofast', '-ffast-math')),
            'production fixture compile flags differ')
    result = dict(compiled=compiled, generated=generated, links=links,
                  owned_link_products=sorted(owned), compile_commands=commands)
    path = ROOT / 'actual-inputs.json'; write(path, result)
    return dict(path=str(path), sha256=digest(path), compiler_files=len(compiled),
        owned_generated_files=len(generated), external_link_files=len(links), uncovered=0,
        translation_units=list(UNITS), source_commit=row['project']['revision'])


def compile_fixture():
    row = json.loads((ROOT / 'record.json').read_text()); current(row)
    require(row['stage'] == 'prepared' and not row['commands'], 'fresh committed build only')
    frozen = copy.deepcopy(row); owner = None
    def persist():
        write(ROOT / 'execution.json', dict(active=bool(owner and owner.active),
            commands=owner.commands if owner else []))
    try:
        owner = ownership.Owner(ROOT / 'owned', persist)
        commands = [('configure', row['recipe']['configure']),
            ('compile', ['/usr/bin/cmake', '--build', str(ROOT / 'build'), '--target', TARGET, '-j', '2']),
            ('dependencies', ['/usr/bin/ninja', '-C', str(ROOT / 'build'), '-t', 'deps'])]
        outputs = {}
        for label, command in commands:
            current(frozen)
            entry, out, _ = owner.run(label, command, row['recipe']['environment'], P, **BUDGET)
            require(ownership.closed(entry) and entry['exit_code'] == 0, 'fixture build command failed: ' + label)
            outputs[label] = out
        row['actual_inputs'] = census(row, outputs['dependencies'], outputs['compile'])
        binary = ROOT / 'build' / TARGET
        with binary.open('rb') as stream:
            require(stream.read(4) == b'\x7fELF', 'fixture ELF absent')
        row['binary'] = dict(path=str(binary), sha256=digest(binary), bytes=binary.stat().st_size,
                             source_commit=row['project']['revision'])
        current(frozen)
        row.update(stage='built', passed=True, complete=True)
    except BaseException as error:
        row.update(stage='failed', passed=False, complete=False, error=repr(error))
    finally:
        row['commands'] = owner.commands if owner else []
        row['active'] = bool(owner and owner.active)
        row['ownership_closed'] = not row['active'] and all(ownership.closed(c) for c in row['commands'])
        row['passed'] = bool(row['passed'] and row['ownership_closed'])
        save(row)
    require(row['passed'], row.get('error', 'fixture compilation failed'))
    print('Fixture-only build closed;GPU/collector execution remains unqualified.', flush=True)


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'], ['build']), 'build_compat_profile.py prepare|build')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        feature.guards.no_model_owners([])
        {'prepare': prepare, 'build': compile_fixture}[sys.argv[1]]()
