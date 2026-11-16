#!/usr/bin/env python3
"""Historical admission of the two fresh E2 images; no GPU work or rebuild.

Main-authored; repository LICENSE. Verify the exact completed component,
not a nearby engine ELF or the current worktree's uncompiled observer files.
"""
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import prefill_upload_overlap as component

P, B, RESULT, ROOT, KEY = component.P, component.B, component.RESULT, component.ROOT, component.KEY
EVIDENCE = '3ee710401f9cb8323757161dd53a17dce6689094'
NATIVE = 'd4e45762a56b169036a67eb9535cd3ce9b238a46'
ORIGINAL = '14508acbf1b70eb994fdf37f97fb6d0f5cff7d43'
BUILD_EVIDENCE = 'cca8ac8d9426769c995e99e4af0aa7d5b624ff11'
require, sha, signature = component.require, component.sha, component.fingerprint
git = component.git


def original_gate(public, row):
    """No I/O: reject altered/partial evidence before source or SDK admission."""
    require(signature(component.compact(row))==signature(public), 'exact typed original E2 receipt required')
    require(row['native_source_commit']==row['project']['revision']==NATIVE
        and row['stage']=='closed' and row['complete'] is True and row['passed'] is True
        and row['ownership_closed'] is True and row['active'] is False
        and row['model_trial_allowed'] is False and len(row['results'])==23,
        'closed original E2 source,process scope and model exclusion required')
    modes=['metadata','forward','reverse','permanent','cancel','cancel-throw','refuse']
    require([x['label'] for x in row['results'][:7]]==['runtime-'+m for m in modes]
        and all(x['passed'] is True and x['native_exit_code']==0 and x['ownership_closed'] is True
                for x in row['results']), 'all original Runtime/native cases required')
    require(row['analysis']['1024']['repeated_component_wall_gain'] is False
        and row['analysis']['4096']['repeated_component_wall_gain'] is True,
        'preserve the failed1024and passed4096screens,without broad adoption')
    require(sum('dep' in x for x in row['recipe'])==5
        and len(row['actual_inputs']['compiled'])==925
        and len(row['actual_inputs']['links'])==25
        and len(row['actual_inputs']['objects'])==7
        and len(row['commands'])==33
        and all(component.owner_api.closed(x) and x['exit_code']==0 for x in row['commands']),
        'five fresh TUs and complete925/25/7census,with33closed commands required')


def verify(*, expected=None):
    current = component.ur_validation.committed(RESULT,(KEY,))
    require(current['original_full_evidence_commit']==EVIDENCE, 'original E2 evidence pointer changed')
    row = json.loads((ROOT/'record.json').read_text())
    require(sha(ROOT/'record.json')==current['source_build_record']['sha256'], 'original E2 private record changed')
    public = json.loads(git(P,'show',EVIDENCE+':'+str(RESULT.relative_to(P))))[KEY]
    original_gate(public,row)
    # compact() is the original receipt encoder; a new observer cannot redefine it.
    recovered = component.original.coherent.recover(row['project'])
    require(sha(component.__file__)==recovered[str(P/'tools/sycl/prefill_upload_overlap.py')],
        'original E2 receipt encoder changed')
    build_path=component.original.RESULT.relative_to(P)
    base = json.loads(git(P,'show',BUILD_EVIDENCE+':'+str(build_path)))['coherent_prefill_native_v1']
    sdk = base['external_inputs']; require(sha(sdk['path'])==sdk['sha256'],'original SDK manifest changed')
    allowed = json.loads(Path(sdk['path']).read_text())
    # Actual dependency files are tied to their immutable repository revisions.
    original_record = json.loads((component.original.ROOT/'record.json').read_text())
    require(original_record['project']['revision']==ORIGINAL
        and signature(component.original.coherent.build.compact(original_record))==signature(base),
        'exact original source/dependency recipe record required')
    for dependency in original_record['dependencies']:
        if 'sycl-tla' in dependency['root']:
            allowed.update(component.original.coherent.recover(dependency))
    view = {str(path):sha(path) for path in component.VIEW.rglob('*') if path.is_file()}
    names = [str(Path(path).relative_to(component.VIEW)) for path in view
             if str(Path(path).relative_to(component.VIEW))!=component.HEADER]
    from_git = {str(component.VIEW/name):hashlib.sha256(raw).hexdigest() for name,raw in
        component.original.coherent.build.common.source.blobs(P,ORIGINAL,names)}
    from_git[str(component.VIEW/component.HEADER)] = recovered[str(P/component.HEADER)]
    require(view==from_git and len(view)==399 and signature(view)==row['source_view']['sha256'],
        'exact original Runtime/class/math view plus owned lane overlay required')
    allowed.update(recovered); allowed.update(from_git)
    runtime = row['runtime_manifest']
    require(sha(runtime['path'])==runtime['sha256'], 'original E2 runtime manifest changed')
    providers = json.loads(Path(runtime['path']).read_text()); allowed.update(providers)
    for image in [row['validation_provider'],*row['native_validation_probe'].values()]:
        require(sha(image['path'])==image['sha256'], 'E2 validation provider changed')
        allowed[str(Path(image['path']).resolve())]=image['sha256']
    require(all(sha(path)==h for path,h in providers.items())
        and all(sha(path)==h for path,h in row['recipe_tools'].items()), 'E2 runtime/tool bytes changed')
    actual=row['actual_inputs']
    require(all(allowed.get(path)==h and sha(path)==h for path,h in actual['compiled'].items()),
        'actual E2 compiler input differs from recovered Git/SDK sources')
    outputs={path for recipe in row['recipe'] for path in recipe['outputs']}
    require(set(actual['objects'])<=outputs and all(sha(path)==h for path,h in actual['objects'].items()),
        'fresh E2 object identity changed')
    allowed.update(actual['objects'])
    require(all(allowed.get(path)==h and sha(path)==h for path,h in actual['links'].items())
        and all(sha(image['path'])==image['sha256'] for image in row['binaries'].values()),
        'actual E2 link inputs or either executable changed')
    require(component.guards.driver_guard()==row['driver']
        and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==row['boot'], 'E2 boot/driver changed')
    proof=dict(native_source_commit=NATIVE,original_Runtime_math_source=ORIGINAL,evidence_commit=EVIDENCE,
        source_build_record_sha256=sha(ROOT/'record.json'),recipe_sha256=signature(row['recipe']),
        source_view_sha256=signature(view),project_Git_files=len(recovered),view_Git_files=399,
        actual_input_sha256=signature(actual),compiler_inputs=925,link_inputs=25,fresh_objects=7,
        binaries=row['binaries'],runtime_manifest=runtime,SDK_manifest=sdk,
        original_runtime_safety_cases=7,original_native_processes=23,
        original_1024_speed_gate_passed=False,original_4096_speed_gate_passed=True,
        old_engine_executables_or_archives_are_not_execution_inputs=True,
        new_GPU_qualification=False,new_model_or_physical262144_qualification=False)
    require(expected is None or signature(proof)==signature(expected),'frozen historical E2 proof changed')
    return row,proof
