#!/usr/bin/env python3
"""Build the selected prefill restoration without overwriting its fallback.

Repository LICENSE. Independently authored from this fork's owned build and
runtime qualification helpers. A linked build is not model/phase qualification.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import generation_engine as gen
import mtp_resident_control as native
import runtime_probe as runtime

build, guards, supervision = native.build, native.guards, native.supervision
P, B = native.P, native.B
ROOT = B / 'selected-prefill-restoration-v2'
RESULT = gen.A / 'prefill-restore-current.json'
BASE = native.BASE
RESTORED = (
    'include/strata/prefill/expert_bank_plan.hpp',
    'include/strata/prefill/prefill.hpp',
    'include/strata/prefill/qsa_native16.hpp',
    'src/prefill/layer_major.inc', 'src/prefill/prefill.cpp',
    'src/prefill/sycl/qsa_native16.cpp', 'src/sycl/compat.cpp',
    'tests/expert_bank_plan.cpp', 'tests/sycl/prefill_scratch.cpp')
REMOVED = ('include/strata/sycl/compat_host_span.hpp',
           'tests/sycl/qsa_streamed_native16.cpp')
NATIVE_DELTA = {'CMakeLists.txt', 'cmake/sycl_backend.cmake',
    'include/strata/core/conversation_snapshot.hpp',
    'include/strata/kernels/cpu/pool.hpp', 'include/strata/kernels/cpu/service_profile.hpp',
    'include/strata/program/diagnostic_kv_ranges.hpp',
    'include/strata/program/full_context_diagnostic.hpp',
    'src/core/conversation_snapshot_test.cpp', 'src/core/conversation_state.cpp',
    'src/kernels/cpu/pool.cpp', 'src/program/generate.cpp'}
HOST_TARGETS = ('prefill_expert_bank_plan_test', 'full_context_diagnostic_host_test',
    'cache_tail_host_test', 'prefill_scratch_host_test',
    'residency_publication_host_test', 'cpu_work_census_host_test', 'iq2_phase_host_test')
PROJECT_PATHS = ['CMakeLists.txt', 'cmake', 'src', 'include', 'third_party', 'tests',
    'bench/sycl', 'bench/micro', 'tools/sycl']


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def compact(row):
    return {k: v for k, v in row.items() if k not in (
        'project', 'dependencies', 'sdk_inputs', 'external_inputs',
        'commands', 'actual_compiled_inputs')}


def restoration_source(revision):
    changed = set(gen.git('diff', '--name-only', BASE, revision, '--',
        'src', 'include', 'CMakeLists.txt', 'cmake').decode().splitlines())
    guards.require(changed == NATIVE_DELTA, 'undeclared native change outside restored prefill/diagnostics')
    for name in RESTORED:
        guards.require(gen.git('show', revision + ':' + name) == gen.git('show', BASE + ':' + name)
            == (P / name).read_bytes(), 'restore selected source first: ' + name)
    guards.require(all(not (P / name).exists() for name in REMOVED), 'remove unused streamed-QSA exports')
    generator = gen.git('show', revision + ':src/program/generate.cpp').decode()
    guards.require(not any(s in generator for s in
        ('separate_prefill_layout', 'PrefillLayout', 'prompt.uses_layer_major', 'prompt.layout()')),
        'unselected prefill dispatch remains')
    guards.require('conversation_snapshot_verify(*incoming' in generator and
        'diagnostic::kv_hash_spans' in generator, 'preserve independent state verification')
    cmake = gen.git('show', revision + ':cmake/sycl_backend.cmake').decode()
    guards.require('sycl_native16_qsa_streamed_test' not in cmake and
        'target_compile_definitions(strata_full_context_diagnostic PRIVATE STRATA_SYCL_FULL_CONTEXT_DIAGNOSTIC=1)' in cmake,
        'full-context macro must belong to the manual diagnostic target')


def prepare():
    guards.require(not ROOT.exists(), 'fresh single-use restoration build')
    previous = gen.committed(RESULT)
    retirement = previous['retirement']
    guards.require(previous['stage'] == 'failed' and not previous['passed']
        and not previous['active'] and previous['ownership_closed']
        and previous['failure_diagnosis']['uncovered_input'] == 'third_party/ggml/ggml-common.h'
        and previous['source_recovery']['actual_compiler_inputs_recovered_and_matched'] == 1416
        and retirement['complete'] and not Path(retirement['root']).exists(),
        'preserve failed census and retire its generation before the corrected build')
    prior_failed = dict(commit=gen.git('log', '-1', '--format=%H', '--',
        str(RESULT.relative_to(P))).decode().strip(), path=str(RESULT.relative_to(P)),
        sha256=build.digest(RESULT))
    closed = gen.committed(gen.A / 'cpu-service-current.json')
    cohort = closed['observer_overhead']
    retired = closed['observer_generation_retirement']
    guards.require(cohort['stage'] == 'closed' and cohort['complete'] and cohort['ownership_closed']
        and not cohort['active'] and len(cohort['processes']) == 9
        and retired['complete'] and not Path(retired['root']).exists(),
        'close cohort and retire the extra generation before another build')
    original, proof = native.selected.baseline()
    runtime_record = gen.committed(runtime.RESULT)
    qualified = runtime.current(runtime_record)
    guards.require(qualified['binary'] == original['binary'], 'selected fallback identity differs')
    revision = gen.git('rev-parse', 'HEAD').decode().strip()
    restoration_source(revision)
    pins = gen.benchmark_pins(revision)
    guards.require('tools/sycl/prefill_restore.py' in pins, 'commit build controller first')
    project = build.pin(P, PROJECT_PATHS)
    dependencies = original['dependencies']
    for dependency in dependencies:
        guards.require(build.pin(Path(dependency['root']), ['.']) == dependency, 'dependency changed')
    sdk_original = json.loads(native.prior.SDK.read_text())
    guards.require(build.digest(native.prior.SDK) == native.prior.SDK_SHA and len(sdk_original) == 8697,
        'original installed-header manifest changed')
    sdk = {v['path']: build.digest(v['path']) for v in sdk_original}
    sdk_changes = {v['path']: dict(original=v['sha256'], current=sdk[v['path']])
        for v in sdk_original if v['sha256'] != sdk[v['path']]}
    guards.require(len(sdk_changes) == 3 and {Path(p).name for p in sdk_changes} ==
        {'metee.h', 'opencl-c-base.h', 'opencl-c.h'}, 'unqualified installed-header transition')
    external = json.loads(native.prior.ENGINE_RECORD.read_text())['external_inputs']
    guards.require(all(build.digest(p) == sha for p, sha in external.items()), 'external MKL input changed')
    tools = {name: dict(path=path, sha256=build.digest(path)) for name, path in
        (('cmake', '/usr/bin/cmake'), ('ninja', '/usr/bin/ninja'))}
    args = list(original['configure_argv'])
    args[args.index('-S') + 1] = str(P)
    args[args.index('-B') + 1] = str(ROOT / 'build')
    args += ['-DCMAKE_MAKE_PROGRAM:FILEPATH=/usr/bin/ninja', '-DSTRATA_CPU_SERVICE_PROFILE=OFF']
    targets = list(dict.fromkeys(['strata', 'strata_full_context_diagnostic',
        *original['fixtures'], 'sycl_conversation_snapshot_test', *HOST_TARGETS]))
    dependency = dependencies[0]
    fetch = lambda name: build.git(Path(dependency['root']), 'show', dependency['revision'] + ':' + name).decode()
    version_cmake = fetch('ggml/CMakeLists.txt')
    version = '.'.join(re.search(r'set\(GGML_VERSION_' + part + r' (\d+)\)', version_cmake)[1]
        for part in ('MAJOR', 'MINOR', 'PATCH'))
    header = fetch('ggml/src/ggml-version.h.in').replace('@GGML_VERSION@', version).replace(
        '@GGML_BUILD_COMMIT@', dependency['revision'][:9]).encode()
    row = dict(stage='prepared', source_commit=revision, selected_source=BASE,
        prior_failed_build=prior_failed,
        fallback=proof, fallback_binary=original['binary'], project=project,
        project_files=project['count'], project_fingerprint=project['aggregate_sha256'],
        dependencies=dependencies, dependency_revisions=[v['revision'] for v in dependencies],
        benchmark_files=len(pins), benchmark_fingerprint=native.fingerprint(pins),
        sdk_inputs=sdk, sdk_fingerprint=native.fingerprint(sdk), sdk_files=len(sdk),
        sdk_reference=dict(path=str(native.prior.SDK), sha256=native.prior.SDK_SHA),
        declared_sdk_changes=sdk_changes, external_inputs=external,
        external_fingerprint=native.fingerprint(external),
        external_reference=dict(path=str(native.prior.ENGINE_RECORD), sha256=build.digest(native.prior.ENGINE_RECORD)),
        generated_headers={str(ROOT / 'build/ggml/src/ggml-version.h'): hashlib.sha256(header).hexdigest()},
        compiler=original['compiler'], tools=tools, configure_argv=args, targets=targets,
        build_argv=['/usr/bin/cmake', '--build', str(ROOT / 'build'), '--target', *targets, '-j', '4'],
        environment=dict(original['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache')),
        runtime_boot=runtime_record['boot'], driver=runtime_record['driver'],
        criteria='Exact restored native sources and committed complete project/benchmark/recipe pins;'
            'original dependency/toolchain identities with the declared three installed-header changes;'
            'complete actual compiler dependency census;all link/host commands close normally. '
            'Normal strata compiles CPU service out and has no full-context macro;the manual diagnostic has its private macro. '
            'Retain original planner/bounds/hash/loan/publication/census/phase host controls. No GPU/model work here.',
        retention='Selected original6e40 remains the validated fallback. This one extra source generation,including '
            'its manual diagnostic and same-build fixtures,is for imminent bounded device/admission,model-state '
            'and independent prefill/decode comparisons. Retire it on rejection or select only after the applicable '
            'model/phase gates;linking/host controls alone never select it.',
        pending='Traced bounded device/admission checks,model allocation/state handoff,matched repeated long prefill '
            'and decode,quality/agent/session/RESTORE/physical262144/failure-owner/ordinary runtime teardown.',
        budgets=dict(wall=1800, cpu=14400, rss_cap=16 << 30, text_cap=32 << 20,
            file_cap=512 << 20, total_cap=6 << 30),
        active=False, ownership_closed=True, passed=False, complete=False,
        GPU_executed=False, production_selected=False, model_adoption_qualified=False, assets={}, host=[])
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (ROOT / name).mkdir()
    write(ROOT / 'record.json', row)
    write(RESULT, compact(row))
    print('Prepared restoration build;commit exact plan before build.', flush=True)


def current(row):
    guards.require(gen.committed(RESULT) == compact(row), 'commit exact restoration plan')
    actual_project = build.pin(P, PROJECT_PATHS)
    # A later plan/result commit may move HEAD without changing any build input.
    # Compare its complete tracked extent and hashes, retaining the original
    # native source revision rather than relabelling it with that later HEAD.
    guards.require(row['project']['revision'] == row['source_commit'] and
        row['project']['count'] == row['project_files'] and
        row['project']['aggregate_sha256'] == row['project_fingerprint'] and
        native.fingerprint(row['project']['source_sha256']) == row['project_fingerprint'] and
        dict(actual_project, revision=row['source_commit']) == row['project'], 'project source changed')
    pins = gen.benchmark_pins(row['source_commit'])
    guards.require(len(pins) == row['benchmark_files'] and native.fingerprint(pins) ==
        row['benchmark_fingerprint'], 'benchmark/controller changed')
    restoration_source(row['source_commit'])
    for dependency in row['dependencies']:
        guards.require(build.pin(Path(dependency['root']), ['.']) == dependency, 'dependency changed')
    for values, fingerprint in ((row['sdk_inputs'], 'sdk_fingerprint'),
                               (row['external_inputs'], 'external_fingerprint')):
        guards.require(native.fingerprint(values) == row[fingerprint] and
            all(build.digest(p) == sha for p, sha in values.items()), 'installed build input changed')
    for item in [row['compiler'], *row['tools'].values(), row['sdk_reference'], row['external_reference']]:
        guards.require(build.digest(item['path']) == item['sha256'], 'build asset/reference changed')
    original, proof = native.selected.baseline()
    guards.require(proof == row['fallback'] and original['binary'] == row['fallback_binary'] and
        row['dependencies'] == original['dependencies'], 'fallback or dependency identity changed')
    if 'actual_compiled_inputs' in row:
        closure = row['compiler_closure']
        guards.require(len(row['actual_compiled_inputs']) == closure['files'] and closure['uncovered'] == 0 and
            native.fingerprint(row['actual_compiled_inputs']) == closure['sha256'] and
            all(build.digest(p) == sha for p, sha in row['actual_compiled_inputs'].items()),
            'retained compiler closure changed')
    runtime_record = gen.committed(runtime.RESULT)
    runtime.current(runtime_record)
    guards.require(runtime_record['boot'] == row['runtime_boot'] and runtime_record['driver'] == row['driver'], 'runtime changed')


def compiler_closure(row, text):
    expected = {str((P / n).resolve()): sha for n, sha in row['project']['source_sha256'].items()}
    for dependency in row['dependencies']:
        expected.update({str((Path(dependency['root']) / n).resolve()): sha
            for n, sha in dependency['source_sha256'].items()})
    expected.update(row['sdk_inputs']); expected.update(row['external_inputs']); expected.update(row['generated_headers'])
    paths = {Path(line.strip()) for line in text.splitlines() if line.startswith('    ') and line.strip()}
    paths = {(p if p.is_absolute() else ROOT / 'build' / p).resolve() for p in paths}
    guards.require(len(paths) > 1000 and all((P / n).resolve() in paths for n in
        ('src/program/generate.cpp', 'src/core/conversation_state.cpp',
         'tests/core/full_context_diagnostic_test.cpp', 'tests/expert_bank_plan.cpp')),
        'actual compiler census incomplete')
    guards.require(all(str(p) in expected for p in paths), 'uncovered actual compiler input')
    guards.require(all(build.digest(p) == expected[str(p)] for p in paths), 'actual compiler input changed')
    closure = {str(p): expected[str(p)] for p in sorted(paths)}
    row['actual_compiled_inputs'] = closure
    return dict(files=len(closure), sha256=native.fingerprint(closure), uncovered=0)


def compile_engine():
    row = json.loads((ROOT / 'record.json').read_text())
    current(row)
    guards.require(row['stage'] == 'prepared' and not (ROOT / 'owned').exists(), 'single-use build;never retry')
    initial, owner = copy.deepcopy(row), None

    def persist():
        write(ROOT / 'record.json', dict(row, active=bool(owner and owner.active is not None),
            commands=owner.commands if owner else []))

    try:
        owner = supervision.Owner(ROOT / 'owned', persist)
        for label, command in (('configure', row['configure_argv']), ('link', row['build_argv']),
                ('compiler-census', ['/usr/bin/ninja', '-C', str(ROOT / 'build'), '-t', 'deps'])):
            entry, out, err = owner.run(label, command, row['environment'], P, **row['budgets'])
            guards.require(supervision.closed(entry) and entry['exit_code'] == 0,
                label + ' failed: ' + (out.read_text(errors='replace') + err.read_text(errors='replace'))[-2048:])
            if label == 'compiler-census':
                row['compiler_closure'] = compiler_closure(row, out.read_text())
            current(initial)
        database = json.loads((ROOT / 'build/compile_commands.json').read_text())
        generators = [v for v in database if Path(v['file']).resolve() == (P / 'src/program/generate.cpp').resolve()]
        guards.require(len(generators) == 2 and all('STRATA_CPU_SERVICE_PROFILE' not in v['command'] for v in generators)
            and sum('STRATA_SYCL_FULL_CONTEXT_DIAGNOSTIC=1' in v['command'] for v in generators) == 1,
            'normal and diagnostic compile definitions differ from plan')
        row['assets'] = {name: dict(path=str(ROOT / 'build' / name), bytes=(ROOT / 'build' / name).stat().st_size,
            sha256=build.digest(ROOT / 'build' / name)) for name in row['targets']}
        for name in HOST_TARGETS:
            for extra in (('--pair=0',), ('--pair=1',)) if name == 'iq2_phase_host_test' else ((),):
                entry, out, err = owner.run('host-' + name + ''.join(extra), [row['assets'][name]['path'], *extra],
                    row['environment'], P, wall=60, cpu=60, rss_cap=1 << 30,
                    text_cap=1 << 20, file_cap=1 << 20, total_cap=4 << 20)
                guards.require(supervision.closed(entry) and entry['exit_code'] == 0, 'host control failed: ' + name)
                text = out.read_text()
                if name == 'full_context_diagnostic_host_test':
                    guards.require('cases=134 full_context_request_and_terminal_bounds=PASS' in text and
                        'byte_order_and_bounds=PASS full_extent_spans=1 GPU=UNRUN' in text, 'bounds/hash controls differ')
                row['host'].append(dict(target=name, args=list(extra), exit_code=0,
                    stdout_sha256=build.digest(out), stderr_sha256=build.digest(err)))
        current(initial)
        row.update(stage='build-host-closed', passed=True, complete=True)
    except BaseException as error:
        row.update(stage='failed', error=repr(error), passed=False)
    finally:
        row['active'] = owner is not None and owner.active is not None
        row['ownership_closed'] = not row['active'] and (owner is None or all(supervision.closed(v) for v in owner.commands))
        row['passed'] = bool(row['passed'] and row['ownership_closed'])
        persist(); write(RESULT, compact(row))
    print(json.dumps({k: row.get(k) for k in ('stage', 'passed', 'ownership_closed', 'error')}), flush=True)
    return int(not row['passed'])


if __name__ == '__main__':
    guards.require(sys.argv[1:] in (['prepare'], ['build']), 'prepare|build')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        if sys.argv[1:] == ['prepare']:
            prepare()
        else:
            sys.exit(compile_engine())
