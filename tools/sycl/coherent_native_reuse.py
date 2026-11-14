#!/usr/bin/env python3
"""Verify the retained coherent build for a new, committed observer plan.

Main-authored; repository LICENSE applies. This does not run an executable,
rebuild native code, refresh old input hashes or qualify another model route.
Historical fallback ELFs are not execution inputs of this prospective reader.
"""
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import check_qsa16_stream as build

P, B, RESULT = build.P, build.B, build.RESULT
ROOT, KEY = B / 'coherent-prefill-native-v1', 'coherent_prefill_native_v1'
THIS = 'tools/sycl/coherent_native_reuse.py'
SOURCE = '14508acbf1b70eb994fdf37f97fb6d0f5cff7d43'
PREFIXES = build.common.PREFIXES
ORIGINAL_OBSERVERS = {
    'tools/sycl/check_coherent_prefill.py',
    'tools/sycl/coherent_prefill_model.py',
    'tools/sycl/coherent_prefill_bench.py',
}
# The sole change to a file present at native compilation is an uncompiled
# builder fix: general TU count and multi-target fixture receipt recording.
BUILDER_DELTA = {
    'tools/sycl/check_qsa16_stream.py': (
        'eb0d4d00869b2d1838c767c59386d56caa2ef103790d000d8c32f8751ddc32c7',
        '2203d0b5a02afa73a497df674c2a84ffa60a740e69c959798e11deefb1c5ef88'),
}
require, sha = build.require, build.sha


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def recover(pin):
    root = Path(pin['root'])
    found = {p: hashlib.sha256(raw).hexdigest() for p, raw in
        build.common.source.blobs(root, pin['revision'], pin['source_sha256'])}
    require(found == pin['source_sha256'], 'original committed native inputs unrecoverable')
    return {str(root / p): h for p, h in found.items()}


def verify(*, observers=(), expected=None):
    """Call while Main holds the measurement lock; freeze the returned proof.

    New observers must be explicitly named, committed Python files, outside the
    actual compiler inputs. An existing frozen plan must supply expected.
    """
    additions = ORIGINAL_OBSERVERS | {THIS, *observers}
    require(all(p.startswith('tools/sycl/') and p.endswith('.py') for p in additions),
        'declare only uncompiled Python observers')
    row = json.loads((ROOT / 'record.json').read_text())
    require(build.validation.committed(RESULT, (KEY,)) == build.compact(row)
        and row['stage'] == 'built' and all(row[k] for k in
            ('passed', 'complete', 'ownership_closed')) and not row['active']
        and row['native_source_commit'] == row['project']['revision'] == SOURCE,
        'original coherent build/receipt must be committed and closed')
    current = build.common.source.pin(P, PREFIXES)
    untracked = build.common.source.git(P, 'ls-files', '--others',
        '--exclude-standard', '-z', '--', *PREFIXES).decode().split('\0')
    require(not any(untracked), 'uncommitted helper/fixture in the observer source scope')
    before, after = row['project']['source_sha256'], current['source_sha256']
    recovered = recover(row['project'])
    changed = {p: (before[p], after[p]) for p in before if p in after and before[p] != after[p]}
    require(set(after) - set(before) == additions and not set(before) - set(after)
        and changed == BUILDER_DELTA,
        'native snapshot changed outside the exact builder delta/new observers')
    # The three original observers are part of the already-closed model/speed
    # evidence. They cannot evolve just because they were additions to SOURCE.
    for p in ORIGINAL_OBSERVERS:
        require(sha(P / p) == hashlib.sha256(build.common.source.git(P, 'show',
            '6a6f1eb389a0093a69c55ba420cbcaaa889fbac8:' + p)).hexdigest(),
            'original coherent qualification observer changed')
    for pin in row['dependencies']:
        recovered.update(recover(pin))
        require(all(sha(Path(pin['root']) / p) == h for p, h in pin['source_sha256'].items()),
            'compiled dependency source changed')
    require(sha(row['recipe_seed']['path']) == row['recipe_seed']['sha256']
        and sha(row['external_inputs']['path']) == row['external_inputs']['sha256']
        and all(sha(p) == h for p, h in row['toolchain'].items()),
        'original recipe/compiler/SDK manifest changed')
    external = json.loads(Path(row['external_inputs']['path']).read_text())
    require(all(sha(p) == h for p, h in external.items()), 'original SDK/link inputs changed')
    require(all(sha(p) == h for p, h in
        (row['configured_files'] | row['generated_inputs']).items()), 'configured/generated inputs changed')
    census = row['actual_inputs']
    require(sha(census['path']) == census['sha256'] and census['uncovered'] == 0
        and census['source_commit'] == SOURCE and census['translation_units'] == 134
        and len(row['planned_compilation']) == 137, 'complete coherent build census required')
    actual = json.loads(Path(census['path']).read_text())
    allowed = recovered | external | row['generated_inputs']
    require(len(actual['compiled']) == 1470 and len(actual['links']) == 29
        and all(allowed.get(p) == h and sha(p) == h
            for family in ('compiled', 'generated', 'links') for p, h in actual[family].items())
        and not any(str(P / p) in actual['compiled'] for p in additions | set(changed)),
        'actual compiled/header/link inputs differ from their recovered native recipe')
    coherence = census['class_definition_coherence']
    require(coherence['header_sha256'] == sha(P / 'include/strata/prefill/gemm.hpp')
        and coherence['includer_count'] == 5 and coherence['same_SYCL_class_condition']
        and coherence['all_objects_from_fresh_build'] and not coherence['partial_archive_reuse'],
        'fresh consistent Gemm class definitions required')
    require(len(row['binaries']) == 12 and all(Path(v['path']).stat().st_size == v['bytes']
        and sha(v['path']) == v['sha256'] for v in row['binaries'].values()),
        'one of the twelve same-build executables changed')
    require(build.probe_qualification() ==
        (row['native_validation_probe'], row['validation_qualification']), 'qualified startup probe changed')
    evidence = {}
    for key in ('coherent_prefill_device_v1', 'coherent_prefill_model_v3', 'coherent_prefill_speed_v1'):
        value = build.validation.committed(RESULT, (key,))
        require(value['stage'] == 'closed' and all(value[k] for k in
            ('passed', 'complete', 'ownership_closed')) and not value['active']
            and value['native_source_commit'] == SOURCE,
            'fresh coherent device/model/speed closure required')
        evidence[key] = fingerprint(value)
    device = build.validation.committed(RESULT, ('coherent_prefill_device_v1',))
    require(device['binaries'] == row['binaries'] and len(device['results']) == 11
        and all(v['passed'] and v['ownership_closed'] and v['native_exit_code'] == 0
            for v in device['results'])
        and device['results'][0]['mode'] == 'host'
        and not device['results'][0]['GPU_executed']
        and all(v['GPU_executed'] and not v['fresh_faults'] and v['boot_unchanged']
            and v['driver_after'] == device['driver'] for v in device['results'][1:]),
        'same-build host/GPU native gates changed')
    proof = dict(original_native_source=SOURCE, original_record_sha256=sha(ROOT / 'record.json'),
        current_observer_source=current['revision'], current_source_sha256=current['aggregate_sha256'],
        original_source_dependency_files=len(recovered),
        exact_uncompiled_builder_delta={p: dict(original=h[0], observer=h[1])
            for p, h in changed.items()},
        declared_observers={p: after[p] for p in sorted(additions)},
        actual_compiler_inputs=1470, external_link_inputs=29, coherent_translation_units=134,
        actual_objects=137, same_build_executables=row['binaries'], class_coherence=coherence,
        original_recipe_sha256=fingerprint(row['recipe']), closed_evidence_sha256=evidence,
        preserved_old_fallback_ELFs_not_execution_inputs=True,
        native_recompiled=False, new_GPU_or_model_qualification=False)
    # Later result commits need not have the observer plan's original HEAD.
    # All actual source bytes remain frozen by the fingerprint and observer map.
    if expected is not None:
        for p in additions | set(changed):
            require(hashlib.sha256(build.common.source.git(P, 'show',
                expected['current_observer_source'] + ':' + p)).hexdigest() == after[p],
                'original committed observer plan source unrecoverable')
        proof['current_observer_source'] = expected['current_observer_source']
        require(proof == expected, 'frozen coherent observer/native proof changed')
    return row, proof
