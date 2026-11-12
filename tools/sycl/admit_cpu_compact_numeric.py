#!/usr/bin/env python3
"""Prospective controller-only admission of the unchanged committed V3 build.

Main-authored; repository LICENSE. Original V3 stays failed. No rebuild, source
replacement, hash refresh, fixture execution before admission, or speed claim.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import cpu_compact_numeric as producer

hot = producer.hot
P, B = producer.P, producer.B
ROOT = B / 'cpu-compact-hot-numeric-admission-v4'
PLAN = P / 'bench/sycl/cpu_compact_hot/admission-plan-v4.json'
THIS = 'tools/sycl/admit_cpu_compact_numeric.py'
need, sha, write = producer.need, producer.sha, producer.write
ABSENT = {'ggml/src/CMakeFiles/ggml.dir/ggml-backend-dl.cpp.o',
          'ggml/src/CMakeFiles/ggml.dir/ggml-backend-reg.cpp.o'}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def metadata(original):
    """Check all actually compiled objects, not unused CMake command exports."""
    build = producer.ROOT / 'build'
    census = json.loads(Path(original['compiler_inputs']['path']).read_text())
    known = dict(census['compiled']) | dict(census['links'])
    known.update({str(P / p): h for p, h in original['sources'].items()})
    known.update({str(Path(original['GGML']['root']) / p): h
                  for p, h in original['GGML']['source_sha256'].items()})
    known.update({str(build / p): h for p, h in original['generated_inputs'].items()})
    known = {str(Path(p).resolve()): h for p, h in known.items()}
    text = (producer.ROOT / 'owned/build-dependencies.stdout').read_text()
    objects, compiled = {}, {}
    current, expected, seen = None, 0, 0
    for line in text.splitlines() + ['']:
        if line.startswith('    '):
            need(current is not None, 'orphan dependency')
            path = Path(line.strip())
            if not path.is_absolute(): path = build / path
            name = str(path.resolve())
            need(name in known and sha(path) == known[name], 'actual compiler dependency drift: ' + name)
            compiled[name] = known[name]; seen += 1
        elif line:
            if current is not None: need(seen == expected, 'incomplete dependency block')
            match = re.fullmatch(r'(.+\.o): #deps (\d+), deps mtime (\d+) \(VALID\)', line)
            need(match is not None, 'invalid/stale object dependency record')
            current, expected, seen = match[1], int(match[2]), 0
            path = build / current
            need(current not in objects and path.is_file() and path.stat().st_mtime_ns == int(match[3]),
                 'missing/changed/duplicate compiled object')
            objects[current] = sha(path)
        else:
            if current is not None: need(seen == expected, 'truncated dependency block')
            current = None
    commands = json.loads((build / 'compile_commands.json').read_text())
    actual, absent, source_files = set(), set(), {}
    for command in commands:
        output = str(Path(command['output']))
        if output not in objects:
            need(output in ABSENT and not (build / output).exists(), 'unexpected missing compiled target')
            absent.add(output); continue
        need(output not in actual, 'duplicate compiled command')
        actual.add(output)
        path = str(Path(command['file']).resolve())
        need(path in known and sha(path) == known[path], 'actual source drift: ' + path)
        source_files[path] = known[path]
    need(len(commands) == 37 and len(objects) == 35 and actual == set(objects) and absent == ABSENT
         and len(compiled) == 625, 'actual command/object/dependency coverage differs')
    required = {str(P / 'bench/sycl/cpu_compact_hot/numeric.cpp'),
        str(P / 'src/kernels/cpu/native_expert.cpp'), str(P / 'src/kernels/cpu/iq_single_avx2.cpp')}
    need(required <= set(source_files), 'actual fixture/production dispatcher/pair not compiled')
    external = {str(Path(p).resolve()): h for p, h in
                json.loads(Path(original['external_inputs']['path']).read_text()).items()}
    links = {}
    for line in (producer.ROOT / 'owned/build-compile.stdout').read_text().splitlines():
        path = Path(line.strip())
        if not path.is_absolute() or not path.is_file() or path.is_relative_to(producer.ROOT): continue
        name = str(path.resolve())
        need(name in external and sha(path) == external[name], 'actual external link input drift')
        links[name] = external[name]
    need(len(links) == 22, 'actual link coverage differs')
    return dict(objects=objects, compiled=compiled, sources=source_files, links=links,
                absent_exports=sorted(absent), command_count=len(commands))


def prepare():
    need(not ROOT.exists() and not PLAN.exists(), 'fresh admission only')
    original = json.loads(producer.PLAN.read_text()); producer.verify(original)
    failed = json.loads(producer.RESULT.read_text())['compact_cache_numeric_v3']
    private = json.loads((producer.ROOT / 'build.json').read_text())
    need(not failed['passed'] and not failed['numeric_fixture_executed'] and failed['ownership_closed']
         and not private['passed'] and not private['active'] and private['ownership_closed']
         and [c['exit_code'] for c in private['commands']] == [0, 0, 0]
         and all(hot.owner.closed(c) for c in private['commands'])
         and not (producer.ROOT / 'check-attempt.json').exists(), 'closed unexecuted V3 required')
    need(original['source_commit'] == private['source_commit'] == failed['source_commit']
         and sha(producer.PLAN) == private['plan_sha256'] == failed['plan_sha256']
         and private['error'] == failed['failure'], 'original source/plan/failure binding differs')
    need(hot.git('show', 'HEAD:' + str(producer.RESULT.relative_to(P))) == producer.RESULT.read_bytes(),
         'commit original failed V3 before new admission')
    need(sha(failed['unqualified_binary']['path']) == failed['unqualified_binary']['sha256'], 'original ELF drift')
    actual = metadata(original)
    paths = [producer.ROOT / 'build.json', producer.PLAN, producer.ROOT / 'build/compile_commands.json',
        producer.ROOT / 'owned/build-compile.stdout', producer.ROOT / 'owned/build-compile.stderr',
        producer.ROOT / 'owned/build-dependencies.stdout', producer.ROOT / 'owned/build-dependencies.stderr',
        Path(failed['unqualified_binary']['path'])]
    paths += [producer.ROOT / 'build' / p for p in actual['objects']]
    paths += [producer.ROOT / 'build' / p for p in original['generated_inputs']]
    for name in ('libcompact_reference_cpu.a', 'ggml/src/libggml-cpu.a', 'ggml/src/libggml-base.a'):
        paths.append(producer.ROOT / 'build' / name)
    revision = hot.git('rev-parse', 'HEAD').decode().strip()
    raw = hot.git('show', revision + ':' + THIS); need(raw == (P / THIS).read_bytes(), 'commit controller')
    env = dict(original['native_environment']); env['TMPDIR'] = str(ROOT / 'tmp')
    tool_env = dict(original['environment']); tool_env['TMPDIR'] = str(ROOT / 'tmp')
    plan = dict(schema='compact-cache-numeric-admission-v4', admission_source_commit=revision,
        controller_sha256=hashlib.sha256(raw).hexdigest(), original_source_commit=original['source_commit'],
        original_plan_commit=failed['plan_commit'], original_plan_sha256=sha(producer.PLAN),
        original_failed_evidence=failed, binary=failed['unqualified_binary'],
        artifacts={str(p): sha(p) for p in paths}, actual_metadata_sha256=fingerprint(actual),
        actual_counts=dict(objects=35, compiler_inputs=625, links=22, unused_exports=2),
        formats=original['formats'], environment=tool_env, native_environment=env,
        environment_change='Only TMPDIR moves to the new owned admission root;all inference settings remain immutable.',
        acceptance='Original source/recipe/toolchain/ELF immutable;all35actual objects,625compiler inputs,'
            '22links and2absent unused exports checked. V3remains failed. New readelf admission then the'
            'unchanged two-format32expert/384serial/640threaded/392raw-call/181376NT1/181248paired'
            ' numeric criteria. No speed,peakRSS,model qualification or new native build.',
        stopping='Single-use actions;stop on any missing/changed evidence,owner,numeric,dispatch or budget'
            'failure. No hash refresh,source replacement,math/criteria change or retry.',
        budgets=original['budgets'], retention='One existing V3CPUcandidate for immediate V4admission/numerics'
            ' and next hot/cold timing,plus99KBownership fallback until successor qualification. No new build.')
    ROOT.mkdir(mode=0o700); (ROOT / 'tmp').mkdir(); write(PLAN, plan)
    print('Prepared controller-only admission of original V3output;commit before admit/check.', flush=True)


def verify(plan):
    need(hot.git('show', 'HEAD:' + str(PLAN.relative_to(P))) == PLAN.read_bytes(), 'commit exact admission plan')
    raw = hot.git('show', plan['admission_source_commit'] + ':' + THIS)
    need(hashlib.sha256(raw).hexdigest() == plan['controller_sha256'] == sha(P / THIS), 'admission controller drift')
    need(sha(producer.PLAN) == plan['original_plan_sha256']
         and all(sha(p) == h for p, h in plan['artifacts'].items()), 'original producer artifacts drift')
    original = json.loads(producer.PLAN.read_text()); producer.verify(original)
    need(json.loads(producer.RESULT.read_text())['compact_cache_numeric_v3'] == plan['original_failed_evidence'],
         'original V3failure changed')
    need(fingerprint(metadata(original)) == plan['actual_metadata_sha256'], 'actual source/build inputs drift')


def execute(action):
    with (ROOT / (action + '-attempt.json')).open('x') as stream: json.dump(dict(action=action, no_retry=True), stream)
    state = dict(stage=action, passed=False, complete=False, active=True, GPU_executed=False,
                 numerical_qualified=False, performance_qualified=False, commands=[])
    def persist(): write(ROOT / (action + '.json'), state)
    persist()
    try:
        plan = json.loads(PLAN.read_text()); verify(plan)
        scope = hot.owner.Owner(ROOT / 'owned', persist); state['commands'] = scope.commands
        state.update(admission_source_commit=plan['admission_source_commit'],
            native_source_commit=plan['original_source_commit'], plan_sha256=sha(PLAN), binary=plan['binary'])
        def run(label, command, wall, native=False):
            entry, out, error = scope.run(action + '-' + label,
                [hot.PRLIMIT, '--as=8589934592', '--', *command],
                plan['native_environment'] if native else plan['environment'], P,
                wall=wall, text_cap=plan['budgets']['output'], rss_cap=plan['budgets']['rss'],
                total_cap=plan['budgets']['total'], file_cap=plan['budgets']['total'], cpu=plan['budgets']['cpu'])
            need(hot.owner.closed(entry) and entry['exit_code'] == 0, 'failed/unclosed ' + label)
            return out.read_text(), error.read_text()
        if action == 'admit':
            text, error = run('libraries', [hot.READELF, '--dynamic', plan['binary']['path']], 10)
            libraries = re.findall(r'Shared library: \[([^\]]+)\]', text)
            need(libraries and not error and not any(bad in lib.lower() for lib in libraries
                for bad in ('sycl', 'level_zero', 'libze', 'ur_loader', 'opencl', 'cuda', 'amdhip')), 'GPU library')
            state.update(source_build_admitted=True, libraries=libraries, actual_counts=plan['actual_counts'])
        else:
            admitted = json.loads((ROOT / 'admit.json').read_text())
            need(admitted['passed'] and admitted['complete'] and admitted['ownership_closed']
                 and not admitted['active'] and admitted['plan_sha256'] == sha(PLAN)
                 and admitted['binary'] == plan['binary'], 'new exact admission required')
            results = []
            for row in plan['formats']:
                command = [plan['binary']['path'], row['shard'], str(row['down']),
                           *map(str, row['offsets']), str(row['blob_bytes']), '512']
                text, error = run('down' + str(row['down']), command, plan['budgets']['check_wall'], True)
                need(not error, 'numeric stderr')
                result = json.loads(text)
                expected = dict(component='compact_cache_numeric', passed=True, down=row['down'], experts=32,
                    serial_cases=384, threaded_rows=640, raw_calls=392, raw_nt1_rows=181376,
                    raw_pair_rows=181248, performance_qualified=False)
                need({k: result[k] for k in expected} == expected and -128 <= result['q8_min'] < 0
                     and 0 < result['q8_max'] <= 127, 'numeric/actual dispatch mismatch')
                results.append(result)
            state.update(results=results, numerical_qualified=True)
        verify(plan); state.update(passed=True, complete=True, ownership_closed=True)
    except BaseException as error:
        state.update(error=repr(error), ownership_closed=all(hot.owner.closed(c) for c in state['commands'])); raise
    finally:
        state['active'] = False; persist()


if __name__ == '__main__':
    need(sys.executable == hot.PYTHON, 'use qualified Python')
    need(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'admit', 'check'), 'prepare|admit|check')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); hot.guards.no_model_owners([])
        prepare() if sys.argv[1] == 'prepare' else execute(sys.argv[1])
