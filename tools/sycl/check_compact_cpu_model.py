#!/usr/bin/env python3
"""Fresh native checks for the compact-CPU full engine;repository LICENSE.

Main-authored from the existing owned coherent qualifier. No earlier model or
partial-build result is inherited. Native source remains the recorded build.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import build_compact_cpu_model as candidate
import check_coherent_prefill as inherited

build = candidate.coherent
P, B, RESULT = build.P, build.B, build.RESULT
ROOT, KEY = B / 'compact-cpu-device-v1', 'compact_cpu_device_v1'
THIS = 'tools/sycl/check_compact_cpu_model.py'
STAGES = inherited.STAGES
original = inherited.original
require, sha = build.require, build.sha

def native(expected=None, *, observers=None):
    observers = {THIS} if observers is None else set(observers)
    require(THIS in observers, 'declare this qualification observer')
    row = json.loads((build.ROOT/'record.json').read_text())
    require(original.validation.committed(RESULT, (build.KEY,)) == build.context().compact(row)
        and row['stage'] == 'built' and row['passed'] and row['complete']
        and row['ownership_closed'] and not row['active'] and not row['GPU_executed'],
        'committed closed coherent build required')
    before = row['project']['source_sha256']
    after = original.common.source.pin(P, original.common.PREFIXES)['source_sha256']
    recovered = {p: hashlib.sha256(raw).hexdigest() for p, raw in
        original.common.source.blobs(P, row['project']['revision'], before)}
    require(recovered == before and set(after)-set(before) == observers
        and not set(before)-set(after) and all(after[p] == h for p, h in before.items()),
        'only explicitly named committed observers may extend the native source snapshot')
    actual = row['actual_inputs']
    require(row['native_source_commit'] == row['project']['revision'] == actual['source_commit']
        and actual['translation_units'] == 135 and actual['uncovered'] == 0
        and len(row['planned_compilation']) == 138
        and row['duplicate_source_admission'] == build.DUPLICATES
        and set(row['binaries']) == set(build.TARGETS), 'exact twelve-target full-build census required')
    coherence = actual['class_definition_coherence']
    require(coherence['header_sha256'] == sha(build.HEADER)
        and coherence['includer_count'] >= 3 and coherence['same_SYCL_class_condition']
        and coherence['all_objects_from_fresh_build'] and not coherence['partial_archive_reuse']
        and build.prerequisite() == row['build_prerequisite'], 'fresh consistent Gemm definitions required')
    for dep in row['dependencies']:
        recovered = {p: hashlib.sha256(raw).hexdigest() for p, raw in
            original.common.source.blobs(Path(dep['root']), dep['revision'], dep['source_sha256'])}
        require(recovered == dep['source_sha256'] and
            all(sha(Path(dep['root'])/p) == h for p, h in recovered.items()), 'dependency source changed')
    require(sha(build.SEED) == row['recipe_seed']['sha256']
        and sha(row['external_inputs']['path']) == row['external_inputs']['sha256']
        and all(sha(p) == h for p, h in row['toolchain'].items())
        and all(sha(p) == h for p, h in json.loads(Path(row['external_inputs']['path']).read_text()).items()),
        'compiler,SDK or original recipe changed')
    seed = json.loads(build.SEED.read_text())
    recipe = copy.deepcopy(seed['recipe']); old = str(build.SEED.parent)
    recipe['configure'] = [str(build.ROOT)+v[len(old):] if v.startswith(old) else v
        for v in recipe['configure']]
    recipe['environment'] = {k: v.replace(old, str(build.ROOT)) for k, v in recipe['environment'].items()}
    recipe['targets'] = recipe['build_targets'] = build.TARGETS
    require(recipe == row['recipe'], 'coherent build must retain its exact original recipe')
    python_seed = json.loads(original.candidate.SEED.read_text())
    require(original.common.HOST_PYTHON == python_seed['host_python']
        and sha(original.common.HOST_PYTHON) == python_seed['host_python_sha256'], 'host interpreter changed')
    for item in (*row['preserved_binaries'], *row['binaries'].values()):
        require(Path(item['path']).stat().st_size == item['bytes'] and
            sha(item['path']) == item['sha256'], 'same-build fixture or phase reference changed')
    require(original.candidate.probe_qualification() ==
        (row['native_validation_probe'], row['validation_qualification']), 'runtime probe changed')
    require(all(sha(p) == h for p, h in (row['configured_files'] | row['generated_inputs']).items())
        and sha(actual['path']) == actual['sha256'], 'generated or compiler census input changed')
    census = json.loads(Path(actual['path']).read_text())
    require(all(sha(p) == h for family in ('compiled', 'generated', 'links')
        for p, h in census[family].items()), 'actual compiler/header/link input changed')
    proof = dict(native_source_commit=row['native_source_commit'], original_source_files=len(before),
        observer_source_files=len(after), observer_source_sha256=original.common.signature(after),
        added_observers={p: after[p] for p in sorted(observers)}, exact_twelve_ELFs_census_recipe=True,
        class_coherence=coherence, exact_seed_recipe_sha256=hashlib.sha256(
            json.dumps(recipe, sort_keys=True).encode()).hexdigest(),
        native_recompiled=False, GPU_result_inherited=False)
    require(expected is None or proof == expected, 'native observer admission changed')
    return dict(row, observer_admission=proof)

# Scope the maintained stage runner to this new build and admission. These are
# controller module globals,before any child runtime/provider initializes;
# native child environments remain immutable after exec.
inherited.ROOT, inherited.KEY, inherited.THIS = ROOT, KEY, THIS
inherited.build, inherited.native = build, native

if __name__ == '__main__':
    require(sys.executable == candidate.cpu.hot.PYTHON and len(sys.argv) == 2
        and sys.argv[1] in ('prepare', *STAGES), 'qualified Python;prepare|'+'|'.join(STAGES))
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original.guards.no_model_owners([])
        if sys.argv[1] == 'prepare':
            inherited.prepare()
        elif sys.argv[1].startswith('gu-'):
            inherited.gu_stage(sys.argv[1])
        elif sys.argv[1] in original.STAGES:
            inherited.context().context().execute(sys.argv[1])
        else:
            inherited.context().extra(sys.argv[1])
