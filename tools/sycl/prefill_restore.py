#!/usr/bin/env python3
"""Verify original restored native inputs for forthcoming model/phase checks.

Repository LICENSE. Independently authored from this fork's owned build and
runtime qualification helpers. Closed build launchers recover from Git; this
reader preserves native identity across separately committed Python controllers.
A linked build is not model/phase qualification.
"""
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import generation_engine as gen
import mtp_resident_control as native
import runtime_probe as runtime

build, guards = native.build, native.guards
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


def current(row, *, controller_revision=None):
    guards.require(gen.committed(RESULT) == compact(row), 'commit exact restoration plan')
    actual_project = build.pin(P, PROJECT_PATHS)
    # A later plan/result commit may move HEAD without changing any build input.
    # Compare its complete tracked extent and hashes, retaining the original
    # native source revision rather than relabelling it with that later HEAD.
    guards.require(row['project']['revision'] == row['source_commit'] and
        row['project']['count'] == row['project_files'] and
        row['project']['aggregate_sha256'] == row['project_fingerprint'] and
        native.fingerprint(row['project']['source_sha256']) == row['project_fingerprint'],
        'original project manifest changed')
    if controller_revision is None:
        guards.require(dict(actual_project, revision=row['source_commit']) == row['project'], 'project source changed')
    else:
        # A separately committed host/device controller may reuse linked native
        # bytes. Recover the complete original snapshot, then require every
        # non-Python project input and its tracked extent to remain identical.
        # Seal the NEW controller/benchmark snapshot independently; do not
        # relabel this build or inherit its failed host observation.
        recovered = {n: hashlib.sha256(raw).hexdigest() for n, raw in
            build.blobs(P, row['source_commit'], row['project']['source_sha256'])}
        guards.require(recovered == row['project']['source_sha256'], 'original project cannot recover')
        native_inputs = lambda values: {n: sha for n, sha in values.items() if not n.endswith('.py')}
        guards.require(native_inputs(actual_project['source_sha256']) == native_inputs(recovered),
            'controller reuse changed native project inputs')
        guards.require('actual_compiled_inputs' in row and len(row['assets']) == len(row['targets'])
            and gen.benchmark_pins(controller_revision) ==
                gen.benchmark_pins(gen.git('rev-parse', 'HEAD').decode().strip()),
            'pin complete new controller snapshot separately from native source')
    pins = gen.benchmark_pins(row['source_commit'], match_worktree=controller_revision is None)
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
    guards.require(len(row['assets']) == len(row['targets']) == 17, 'complete restored native package required')
    for asset in row['assets'].values():
        guards.require(Path(asset['path']).stat().st_size == asset['bytes'] and
            build.digest(asset['path']) == asset['sha256'], 'restored native artifact changed')
    if 'actual_compiled_inputs' in row:
        closure = row['compiler_closure']
        guards.require(len(row['actual_compiled_inputs']) == closure['files'] and closure['uncovered'] == 0 and
            native.fingerprint(row['actual_compiled_inputs']) == closure['sha256'] and
            all(build.digest(p) == sha for p, sha in row['actual_compiled_inputs'].items()),
            'retained compiler closure changed')
    runtime_record = gen.committed(runtime.RESULT)
    runtime.current(runtime_record)
    guards.require(runtime_record['boot'] == row['runtime_boot'] and runtime_record['driver'] == row['driver'], 'runtime changed')
