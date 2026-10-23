#!/usr/bin/env python3
"""Configure and syntax-check the unselected full-context diagnostic; Main only.

Repository LICENSE. This controller does not execute or select an engine.
Each generated command plan must be committed before its next owned operation.
"""
import fcntl
import hashlib
import json
from pathlib import Path
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
               if k not in ('project', 'dependencies', 'environment', 'commands')}
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


def admission(record, container):
    guards.require(RESULT.read_bytes() == build.git(P, 'show', 'HEAD:' + str(RESULT.relative_to(P))),
                   'commit the exact generated plan before its operation')
    guards.require(container[KEY] == summary(record), 'external and committed plans differ')
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


if __name__ == '__main__':
    guards.require(sys.argv[1:] in (['prepare'], ['configure'], ['syntax']), 'prepare|configure|syntax')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        container = json.loads(RESULT.read_text())
        if sys.argv[1] == 'prepare':
            prepare(container)
        else:
            record = json.loads((ROOT / 'record.json').read_text())
            # A transient exact copy solely for post-operation admission. The
            # committed compact plan and frozen source remain authoritative.
            admission(record, container)
            write(ROOT / 'admitted-plan.json', record)
            sys.exit(run(container, sys.argv[1]))
