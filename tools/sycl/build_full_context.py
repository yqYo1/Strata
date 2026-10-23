#!/usr/bin/env python3
"""Build/check the unselected full-context diagnostic; Main only.

Repository LICENSE. This controller never executes or selects an engine.
Each generated command plan must be committed before its next owned operation.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import sys

sys.dont_write_bytecode = True
import build_engine as build
import check_full_context_source as source
import measurement_guards as guards
import measurement_owner as supervision

P, B = build.P, build.B
ROOT = B / 'full-context-native-v1'
RESULT = source.RESULT
KEY = 'native_diagnostic'
MACRO = '-DSTRATA_SYCL_FULL_CONTEXT_DIAGNOSTIC=1'
PREFIXES = ['CMakeLists.txt', 'cmake', 'include', 'src', 'tests', 'third_party',
            'bench/sycl', 'tools/sycl/build_full_context.py', *source.INPUTS]
SDK = B / 'unitrace-warmed-graph-window-v1/compiler-inputs.json'
SDK_SHA = '436cedc9dbd41810cd882e9040a129e6d8b7dbebb98806b00c4f2950b1dc1705'


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def summary(record):
    compact = {k: v for k, v in record.items()
               if k not in ('project', 'dependencies', 'environment', 'commands', 'external_inputs')}
    compact['sources'] = [{k: s[k] for k in ('root', 'revision', 'count', 'aggregate_sha256')}
                          for s in [record['project'], *record['dependencies']]]
    return compact


def sdk():
    guards.require(build.digest(SDK) == SDK_SHA, 'original SDK manifest changed')
    rows = json.loads(SDK.read_text())
    guards.require(len(rows) == 8697, 'SDK input count differs')
    for row in rows:
        guards.require(build.digest(row['path']) == row['sha256'], 'SDK input changed: ' + row['path'])
    return dict(path=str(SDK), sha256=SDK_SHA, files=len(rows))


def check_inputs(record):
    _, reference = source.baseline()
    guards.require(record['selected_original'] == reference, 'selected fallback changed')
    guards.require(build.pin(P, PREFIXES)['source_sha256'] == record['project']['source_sha256'],
                   'native/controller inputs changed')
    for dependency in record['dependencies']:
        actual = build.pin(Path(dependency['root']), ['.'])
        guards.require(actual == dependency, 'native dependency changed')
    guards.require(record['sdk'] == sdk(), 'SDK admission differs')
    for item in record['tools'].values():
        guards.require(build.digest(item['path']) == item['sha256'], 'build tool changed')


def admission(record, container):
    guards.require(RESULT.read_bytes() == build.git(P, 'show', 'HEAD:' + str(RESULT.relative_to(P))),
                   'commit the exact generated plan before its operation')
    guards.require(container[KEY] == summary(record), 'external and committed plans differ')
    check_inputs(record)


def prepare(container):
    guards.require(KEY not in container and not ROOT.exists(), 'single-use native diagnostic; inspect existing owner')
    guards.require(container['passed'] and container['cases'] == 134 and container['ownership_closed']
                   and not container['active'] and not container['GPU_executed'], 'closed host source gate required')
    full, reference = source.baseline()
    project = build.pin(P, PREFIXES)
    for name in ('src/program/generate.cpp', 'include/strata/program/full_context_diagnostic.hpp',
                 'tests/core/full_context_diagnostic_test.cpp'):
        guards.require(project['source_sha256'][name] == container['input_sha256'][name],
                       'host source gate does not cover these diagnostic bytes')
    dependencies = [build.pin(Path(d['root']), ['.']) for d in full['dependencies']]
    guards.require(all(a['source_sha256'] == b['source_sha256'] for a, b in
                       zip(dependencies, full['dependencies'])), 'original dependencies differ')
    compiler = full['compiler']
    guards.require(build.digest(compiler['path']) == compiler['sha256'], 'original compiler changed')
    sdk_pin = sdk()
    argv = list(full['configure_argv'])
    guards.require(argv[argv.index('-S') + 1] == str(P) and not any(
        a.startswith('-DCMAKE_CXX_FLAGS') for a in argv), 'unexpected original configure recipe')
    argv[argv.index('-B') + 1] = str(ROOT / 'build')
    argv.append('-DCMAKE_CXX_FLAGS:STRING=' + MACRO)
    env = dict(full['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    record = dict(stage='prepared_configure', source_commit=project['revision'], project=project,
        dependencies=dependencies, selected_original=reference, compiler=compiler,
        sdk=sdk_pin, tools={n: dict(path=p, sha256=build.digest(p)) for n, p in
            (('cmake', argv[0]), ('ninja', '/usr/bin/ninja'))},
        configure_argv=argv, environment=env, active=False, ownership_closed=True,
        passed=False, complete=False, GPU_executed=False, native_engine_built=False,
        production_selected=False, model_adoption_qualified=False, physical262144_qualified=False,
        budgets=dict(wall_s=300, cpu_s=600, rss_bytes=8 << 30, text_bytes=8 << 20,
                     file_bytes=64 << 20, total_bytes=256 << 20),
        criteria='Owned configure, then separately committed enabled/default generator syntax commands; '
            'both must exit zero with frozen source, dependencies, SDK and selected13 original assets. '
            'No engine execution, speed, decode-preservation or physical-context claim.',
        retention='Original qualified6e40 engine+12fixtures remain selected and untouched. '
            'This separate configure/syntax tree is solely for the imminent full-context diagnostic build; '
            'no new comparison engine is built here. Retire routine output after extracting results. '
            'Legacy unqualified products are not used as fallbacks or reference generations.',
        next_gate='Native link plan, actual allocation/lifetime admission, bounded device validation, '
            'full-fill/near-cap/RESTORE/fault-owner checks and independently repeated phase-preservation.')
    ROOT.mkdir(mode=0o700)
    (ROOT / 'tmp').mkdir()
    (ROOT / 'cache').mkdir()
    write(ROOT / 'record.json', record)
    container[KEY] = summary(record)
    write(RESULT, container)
    print('Prepared configure plan; commit before configure.', flush=True)


def syntax_commands(record):
    database = ROOT / 'build/compile_commands.json'
    entries = json.loads(database.read_text())
    matches = [e for e in entries if Path(e['file']).resolve() == P / 'src/program/generate.cpp']
    guards.require(len(matches) == 1, 'one actual generator compile entry required')
    entry = matches[0]
    argv = entry.get('arguments') or shlex.split(entry['command'])
    guards.require(argv[0] == record['compiler']['path'] and argv.count(MACRO) == 1 and
                   argv.count('-c') == 1 and argv.count('-o') == 1 and
                   not any(a.startswith('@') for a in argv), 'unexpected generator compile recipe')
    i = argv.index('-o')
    del argv[i:i + 2]
    argv.remove('-c')
    argv.append('-fsyntax-only')
    default = [a for a in argv if a != MACRO]
    return dict(database=dict(path=str(database), sha256=build.digest(database), entries=len(entries)),
                directory=entry['directory'], commands=[dict(label='syntax-enabled', argv=argv),
                                                       dict(label='syntax-default', argv=default)])


def run(container, operation):
    record = json.loads((ROOT / 'record.json').read_text())
    expected = 'prepared_configure' if operation == 'configure' else 'prepared_syntax'
    guards.require(record['stage'] == expected, 'no retry or operation without its prepared plan')
    admission(record, container)
    if operation == 'syntax':
        guards.require(syntax_commands(record) == record['syntax_plan'], 'generated syntax recipe changed')
    owner = None
    def persist():
        record['active'] = bool(owner and owner.active is not None)
        write(ROOT / 'record.json', dict(record, commands=owner.commands if owner else []))
    owner = supervision.Owner(ROOT / ('owned-' + operation), persist)
    record['stage'] = operation + '_running'
    persist()
    try:
        commands = [dict(label='configure', argv=record['configure_argv'])] if operation == 'configure' else record['syntax_plan']['commands']
        b = record['budgets']
        for command in commands:
            entry, out, err = owner.run(command['label'], command['argv'], record['environment'],
                P if operation == 'configure' else Path(record['syntax_plan']['directory']),
                wall=b['wall_s'], cpu=b['cpu_s'], rss_cap=b['rss_bytes'], text_cap=b['text_bytes'],
                file_cap=b['file_bytes'], total_cap=b['total_bytes'])
            record.setdefault('outcomes', []).append(dict(operation=command['label'],
                exit_code=entry.get('exit_code'), ownership_closed=supervision.closed(entry),
                stdout_sha256=build.digest(out), stderr_sha256=build.digest(err)))
            guards.require(supervision.closed(entry) and entry['exit_code'] == 0,
                           command['label'] + ' failed: ' + err.read_text(errors='replace')[-4096:])
        # Verify against the still-committed pre-operation plan before changing
        # its compact result. Never refresh hashes to admit edited inputs.
        original = json.loads((ROOT / 'admitted-plan.json').read_text())
        admission(original, container)
        if operation == 'configure':
            record['syntax_plan'] = syntax_commands(record)
            record['stage'] = 'prepared_syntax'
            record['configure_passed'] = True
        else:
            guards.require(syntax_commands(record) == record['syntax_plan'], 'syntax inputs changed during operation')
            record.update(stage='closed_syntax', passed=True, complete=True)
    except BaseException as error:
        record.update(stage=operation + '_failed', error=repr(error), passed=False, complete=False)
    finally:
        record['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
        record['passed'] = bool(record['passed'] and record['ownership_closed'])
        persist()
        container[KEY] = summary(record)
        write(RESULT, container)
    print(json.dumps({k: record.get(k) for k in ('stage', 'passed', 'complete', 'ownership_closed', 'error')}), flush=True)
    return 0 if record['ownership_closed'] and record['stage'] in ('prepared_syntax', 'closed_syntax') else 1


ENGINE_KEY = 'native_engine_build'
ENGINE_RECORD = ROOT / 'engine-record.json'


def configuration():
    return {name: build.digest(ROOT / 'build' / name) for name in
            ('compile_commands.json', 'build.ninja', 'CMakeFiles/rules.ninja', 'CMakeCache.txt')}


def mkl_inputs():
    root = Path('/opt/intel/oneapi/mkl/2026.1')
    paths = {p.resolve() for d in ('include', 'lib/cmake/mkl') for p in (root / d).rglob('*') if p.is_file()}
    ninja = (ROOT / 'build/build.ninja').read_text()
    stanza = ninja[ninja.index('build strata:'):].split('\n\n', 1)[0]
    line = re.search(r'^  LINK_LIBRARIES = (.+)$', stanza, re.M)
    guards.require(line is not None, 'missing actual executable link libraries')
    libraries = [Path(a) for a in shlex.split(line[1]) if a.startswith('/')]
    guards.require(len(libraries) == 4 and all(p.is_relative_to(root / 'lib') and p.is_file()
                                            for p in libraries), 'unexpected imported link assets')
    paths.update(p.resolve() for p in libraries)
    guards.require(len(paths) >= 230, 'MKL header/config/link closure unexpectedly small')
    return {str(p): build.digest(p) for p in sorted(paths)}


def prepare_link(container):
    guards.require(ENGINE_KEY not in container and not ENGINE_RECORD.exists() and
                   not (ROOT / 'build/strata').exists(), 'fresh diagnostic link only;inspect existing owner')
    old = json.loads((ROOT / 'record.json').read_text())
    guards.require(old['stage'] == 'closed_syntax' and old['passed'] and old['complete'] and
                   old['ownership_closed'] and not old['active'] and all(
                       container[KEY].get(k) == v for k, v in summary(old).items()), 'closed syntax proof differs')
    for name, data in build.blobs(P, old['project']['revision'], old['project']['source_sha256']):
        guards.require(hashlib.sha256(data).hexdigest() == old['project']['source_sha256'][name],
                       'configured source is unrecoverable')
    project = build.pin(P, PREFIXES)
    before, after = old['project']['source_sha256'], project['source_sha256']
    changes = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    guards.require(changes == ['tools/sycl/build_full_context.py'],
                   'configure reuse requires exactly the declared controller-only change')
    guards.require(syntax_commands(old) == old['syntax_plan'], 'configured generator command changed')
    record = dict(source_commit=project['revision'], project=project, dependencies=old['dependencies'],
        selected_original=old['selected_original'], compiler=old['compiler'], sdk=old['sdk'], tools=old['tools'],
        configure_argv=old['configure_argv'], environment=old['environment'], stage='prepared_link',
        configured_from=dict(source_commit=old['source_commit'], receipt_sha256=build.digest(ROOT / 'record.json'),
            source_changes=changes, scope='All native sources and configure inputs are identical;only this uncompiled controller changed. No GPU or decode qualification is reused.'),
        configuration_inputs=configuration(), external_inputs=mkl_inputs(),
        command=['/usr/bin/cmake', '--build', str(ROOT / 'build'), '--target', 'strata', '-j', '4'],
        active=False, ownership_closed=True, passed=False, complete=False, GPU_executed=False,
        native_engine_built=False, production_selected=False, model_adoption_qualified=False,
        physical262144_qualified=False,
        budgets=dict(wall_s=1800, cpu_s=7200, rss_bytes=12 << 30, text_bytes=64 << 20,
                     file_bytes=512 << 20, total_bytes=4 << 30),
        criteria='Owned compile/link must exit0;post-build actual Ninja dependency closure must be covered by frozen project/dependency/SDK/MKL inputs or generated inside this private build root. Preserve original13assets. No engine/GPU execution or adoption.',
        retention='One selected qualified6e40 engine+12fixtures untouched;one additional diagnostic engine solely for imminent262144 allocation/full-fill/near-cap/RESTORE qualification. Compile no duplicate same-build fixtures. Retire the extra after its consumer closes;legacy unqualified products are separately reviewed.',
        next_gate='Bounded traced/validated device smoke,then actual allocation/lifetime admission and full-fill/near-cap/RESTORE/fault-owner checks. Repeated matched decode preservation remains mandatory before adoption.')
    pins = record['external_inputs']
    record['external_inputs_summary'] = dict(files=len(pins), sha256=hashlib.sha256(
        json.dumps(pins, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        scope='Installed MKL2026.1 include tree,two CMake config files,and four actual imported executable link libraries.')
    check_inputs(record)
    write(ENGINE_RECORD, record)
    container[ENGINE_KEY] = summary(record)
    write(RESULT, container)
    print('Prepared isolated native link with explicit MKL pins;commit before build.', flush=True)


def link_admission(record, container):
    guards.require(RESULT.read_bytes() == build.git(P, 'show', 'HEAD:' + str(RESULT.relative_to(P))) and
                   container[ENGINE_KEY] == summary(record), 'commit exact native link plan first')
    check_inputs(record)
    guards.require(configuration() == record['configuration_inputs'], 'configured build inputs changed')
    guards.require(mkl_inputs() == record['external_inputs'], 'MKL external inputs changed')


def actual_dependency_closure(record, output):
    expected = {}
    for s in [record['project'], *record['dependencies']]:
        expected.update({str((Path(s['root']) / n).resolve()): h for n, h in s['source_sha256'].items()})
    expected.update({str(Path(v['path']).resolve()): v['sha256'] for v in json.loads(SDK.read_text())})
    expected.update(record['external_inputs'])
    paths = {Path(line.strip()) for line in output.read_text().splitlines() if line.startswith('    ') and line.strip()}
    paths = {(p if p.is_absolute() else ROOT / 'build' / p).resolve() for p in paths}
    guards.require((P / 'src/program/generate.cpp').resolve() in paths and len(paths) >= 100,
                   'actual compiler dependency census incomplete')
    uncovered, generated = [], {}
    for p in sorted(paths):
        if str(p) in expected:
            guards.require(build.digest(p) == expected[str(p)], 'actual compiled input changed: ' + str(p))
        elif p.is_relative_to(ROOT.resolve()):
            generated[str(p)] = build.digest(p)
        else:
            uncovered.append(str(p))
    record['actual_dependency_closure'] = dict(files=len(paths), generated_files=len(generated),
        generated_sha256=generated, uncovered_files=len(uncovered), uncovered=uncovered)
    guards.require(not uncovered, 'compiled external inputs absent from prebuild pins: ' + repr(uncovered[:12]))


def link(container):
    record = json.loads(ENGINE_RECORD.read_text())
    guards.require(record['stage'] == 'prepared_link', 'no duplicate/retry native build')
    link_admission(record, container)
    original = json.loads(ENGINE_RECORD.read_text())
    owner = None
    def persist():
        record['active'] = bool(owner and owner.active is not None)
        write(ENGINE_RECORD, dict(record, commands=owner.commands if owner else []))
    owner = supervision.Owner(ROOT / 'owned-engine', persist)
    record['stage'] = 'link_running'
    persist()
    try:
        b = record['budgets']
        entry, out, err = owner.run('compile-link', record['command'], record['environment'], P,
            wall=b['wall_s'], cpu=b['cpu_s'], rss_cap=b['rss_bytes'], text_cap=b['text_bytes'],
            file_cap=b['file_bytes'], total_cap=b['total_bytes'])
        guards.require(supervision.closed(entry) and entry['exit_code'] == 0,
                       'native compile/link failed: ' + (err.read_text(errors='replace') + out.read_text(errors='replace'))[-8192:])
        binary = ROOT / 'build/strata'
        record['binary'] = dict(path=str(binary), bytes=binary.stat().st_size, sha256=build.digest(binary))
        record['native_engine_built'] = True
        entry, deps, _ = owner.run('actual-dependencies', ['/usr/bin/ninja', '-C', str(ROOT / 'build'), '-t', 'deps'],
            record['environment'], ROOT, wall=30, cpu=30, rss_cap=1 << 30, text_cap=64 << 20,
            file_cap=b['file_bytes'], total_cap=b['total_bytes'])
        guards.require(supervision.closed(entry) and entry['exit_code'] == 0, 'actual dependency census failed')
        actual_dependency_closure(record, deps)
        link_admission(original, container)
        record.update(stage='closed_link', passed=True, complete=True)
    except BaseException as error:
        record.update(stage='link_failed', error=repr(error), passed=False, complete=False)
    finally:
        record['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
        record['passed'] = bool(record['passed'] and record['ownership_closed'])
        persist()
        container[ENGINE_KEY] = summary(record)
        write(RESULT, container)
    print(json.dumps({k: record.get(k) for k in ('stage', 'passed', 'complete', 'native_engine_built', 'ownership_closed', 'error')}), flush=True)
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    guards.require(sys.argv[1:] in (['prepare'], ['configure'], ['syntax'], ['prepare-link'], ['link']),
                   'prepare|configure|syntax|prepare-link|link')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        container = json.loads(RESULT.read_text())
        if sys.argv[1] == 'prepare-link':
            prepare_link(container)
        elif sys.argv[1] == 'link':
            sys.exit(link(container))
        elif sys.argv[1] == 'prepare':
            prepare(container)
        else:
            record = json.loads((ROOT / 'record.json').read_text())
            # A transient exact copy solely for post-operation admission. The
            # committed compact plan and frozen source remain authoritative.
            admission(record, container)
            write(ROOT / 'admitted-plan.json', record)
            sys.exit(run(container, sys.argv[1]))
