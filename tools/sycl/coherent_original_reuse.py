#!/usr/bin/env python3
"""Recover the exact coherent14508 reference despite later worktree changes.

Main-authored from the owned coherent build checks; repository LICENSE.
This is a historical source/artifact reader, distinct from verify()'s current
source contract. It never builds, runs a device or qualifies a new model route.
The consuming comparison must pin this observer, current frontend and workloads,
match model/runtime/configuration identities and own every native process.
"""
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import coherent_native_reuse as coherent

P, B, RESULT = coherent.P, coherent.B, coherent.RESULT
ROOT, KEY, SOURCE = coherent.ROOT, coherent.KEY, coherent.SOURCE
EVIDENCE = 'cca8ac8d9426769c995e99e4af0aa7d5b624ff11'
require, sha, fingerprint = coherent.require, coherent.sha, coherent.fingerprint


def verify_original(*, expected=None):
    """Check original Git bytes, exact recipe/census and all twelve ELF files.

    Current project/dependency source is deliberately not substituted for the
    original build. SDK, generated files, libraries and executables must still
    match their original bytes. The old current-source reader stays unchanged.
    Call under Main's serialization lock and freeze the returned proof.
    """
    row = json.loads((ROOT/'record.json').read_text())
    build = coherent.build
    original = json.loads(build.common.source.git(P, 'show', EVIDENCE+':'+str(RESULT.relative_to(P))))[KEY]
    require(original == build.compact(row) and row['stage'] == 'built'
        and row['native_source_commit'] == row['project']['revision'] == SOURCE
        and all(row[k] for k in ('passed','complete','ownership_closed'))
        and not row['active'], 'exact committed coherent build receipt required')
    recovered = coherent.recover(row['project'])
    for pin in row['dependencies']:
        recovered.update(coherent.recover(pin))
    require(sha(row['recipe_seed']['path']) == row['recipe_seed']['sha256']
        and sha(row['external_inputs']['path']) == row['external_inputs']['sha256']
        and all(sha(p) == h for p,h in row['toolchain'].items()),
        'original recipe,compiler or SDK manifest changed')
    external = json.loads(Path(row['external_inputs']['path']).read_text())
    require(all(sha(p) == h for p,h in external.items()), 'original SDK/library bytes changed')
    for p,h in (row['configured_files'] | row['generated_inputs']).items():
        require((recovered[p] == h if p in recovered else sha(p) == h),
            'original configured/generated input changed')
    census = row['actual_inputs']
    require(sha(census['path']) == census['sha256'] and census['source_commit'] == SOURCE
        and census['uncovered'] == 0 and census['translation_units'] == 134
        and len(row['planned_compilation']) == 137, 'complete original compiler census required')
    actual = json.loads(Path(census['path']).read_text())
    allowed = recovered | external | row['generated_inputs']
    require(len(actual['compiled']) == 1470 and len(actual['links']) == 29
        and all(allowed.get(p) == h and (p in recovered or sha(p) == h)
            for family in ('compiled','generated','links') for p,h in actual[family].items()),
        'compiler/link inputs do not bind the recovered historical snapshot')
    coherence = census['class_definition_coherence']
    header = str(Path(row['project']['root'])/'include/strata/prefill/gemm.hpp')
    require(coherence['header_sha256'] == recovered[header]
        and coherence['includer_count'] == 5 and coherence['same_SYCL_class_condition']
        and coherence['all_objects_from_fresh_build'] and not coherence['partial_archive_reuse'],
        'fresh original Gemm definition closure required')
    require(len(row['binaries']) == 12 and all(Path(v['path']).stat().st_size == v['bytes']
        and sha(v['path']) == v['sha256'] for v in row['binaries'].values()),
        'one of the twelve original coherent executables changed')
    for kind in ('loader','library'):
        image = row['native_validation_probe'][kind]
        require(sha(image['path']) == image['sha256'], 'original validation provider changed')
    evidence = {}
    for key in ('coherent_prefill_device_v1','coherent_prefill_model_v3','coherent_prefill_speed_v1'):
        value = build.validation.committed(RESULT,(key,))
        require(value['stage'] == 'closed' and not value['active']
            and value['native_source_commit'] == SOURCE
            and all(value[k] for k in ('passed','complete','ownership_closed')),
            'closed original device/model/speed evidence required')
        evidence[key] = value
    device = evidence['coherent_prefill_device_v1']
    require(device['binaries'] == row['binaries'] and len(device['results']) == 11
        and all(v['passed'] and v['ownership_closed'] and v['native_exit_code'] == 0
            for v in device['results'])
        and device['results'][0]['mode'] == 'host' and not device['results'][0]['GPU_executed']
        and all(v['GPU_executed'] and not v['fresh_faults'] and v['boot_unchanged']
            and v['driver_after'] == device['driver'] for v in device['results'][1:]),
        'original same-build qualification stages changed')
    model = evidence['coherent_prefill_model_v3']
    speed = evidence['coherent_prefill_speed_v1']
    require(model['native_build']['binaries'] == row['binaries']
        and speed['binary'] == row['binaries']['strata'], 'original model/speed ELF identity differs')
    manifest = speed['runtime_manifest']
    require(sha(manifest['path']) == manifest['sha256'], 'original runtime manifest changed')
    providers = json.loads(Path(manifest['path']).read_text())
    require(all(sha(p) == h for p,h in providers.items()), 'original runtime provider changed')
    proof = dict(original_native_source=SOURCE,original_build_evidence=EVIDENCE,
        original_record_sha256=sha(ROOT/'record.json'),reader_sha256=sha(__file__),
        original_source_dependency_files=len(recovered),actual_compiler_census_sha256=census['sha256'],
        original_recipe_sha256=fingerprint(row['recipe']),actual_compiler_inputs=1470,
        external_link_inputs=29,coherent_translation_units=134,actual_objects=137,
        same_build_executables=row['binaries'],class_coherence=coherence,
        closed_evidence_sha256={k:fingerprint(v) for k,v in evidence.items()},
        runtime_manifest=manifest,historical_sources_recovered_from_git=True,
        current_worktree_not_substituted_for_native_source=True,native_recompiled=False,
        new_GPU_or_model_qualification=False,physical262144_or_adoption_qualified=False)
    require(expected is None or proof == expected, 'frozen original-native proof changed')
    return row,proof
